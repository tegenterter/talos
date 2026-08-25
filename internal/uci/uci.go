// Package uci implements the Universal Chess Interface protocol loop:
// reading commands from stdin and writing responses to stdout.
package uci

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"talos/internal/board"
	"talos/internal/nnue"
	"talos/internal/search"
	"talos/internal/tablebase"
)

// Run reads UCI commands from stdin until "quit" or EOF.
func Run() {
	run(os.Stdin, os.Stdout)
}

func run(in io.Reader, rawOut io.Writer) {
	// info/bestmove lines are written both from this loop and from the
	// background goroutine "go" starts (see below), so writes need to be
	// serialized to avoid interleaving two lines' bytes together.
	out := &syncWriter{w: rawOut}

	b := board.StartingBoard()
	// Hashes of the positions preceding b in the game, rebuilt by each
	// "position" command (see parsePosition) and handed to the search so it
	// can recognize repeating a position the game has already seen.
	var gameHistory []uint64

	// Configurable via "setoption" (see the "setoption" case below); read
	// only by this loop's own goroutine when building the next "go"'s
	// search.Options, so no synchronization is needed for these two.
	threads := 1
	hashMB := search.DefaultHashMB
	// The transposition table lives across moves rather than being rebuilt
	// per "go" — see search.Table. Allocated on first use so an engine that
	// is only queried (e.g. "uci" then "quit", as GUIs do when enumerating
	// engines) never pays for it, reallocated when "Hash" changes, and
	// cleared on "ucinewgame". Owned solely by this goroutine; a search
	// reads it concurrently, but only between the "go" that starts it and
	// the wg.Wait() that joins it, and every mutation below happens outside
	// that window.
	var tt *search.Table
	// nil means no Syzygy tables are loaded, so "go" leaves
	// search.Options.Tablebase nil and search runs exactly as it did
	// before this option existed.
	var tb *tablebase.Tablebase

	// State for whatever search is currently running in the background,
	// guarded by mu since "stop"/"ponderhit"/"quit"/a new "go" can arrive
	// while a previous "go"'s goroutine is still working.
	var (
		mu           sync.Mutex
		cancel       context.CancelFunc
		wg           sync.WaitGroup
		ponderBudget time.Duration
	)

	scanner := bufio.NewScanner(in)
	// The 64KB default token size can't fit a pathologically long single
	// line (e.g. "position ... moves ..." replaying a very long game); grow
	// it well past any realistic UCI line instead of failing silently.
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		fields := strings.Fields(line)
		switch fields[0] {
		case "uci":
			fmt.Fprintln(out, "id name Talos")
			fmt.Fprintln(out, "id author Thibaut Vandenbussche")
			fmt.Fprintln(out, "option name Ponder type check default false")
			fmt.Fprintf(out, "option name Threads type spin default 1 min %d max %d\n", minThreads, maxThreads)
			fmt.Fprintf(out, "option name Hash type spin default %d min %d max %d\n", search.DefaultHashMB, minHashMB, maxHashMB)
			fmt.Fprintln(out, "option name SyzygyPath type string default <empty>")
			fmt.Fprintln(out, "uciok")
		case "isready":
			// Answered immediately even mid-search, per the UCI spec:
			// this loop only blocks briefly (see "go" below), so it's
			// always free to read and answer the next command.
			fmt.Fprintln(out, "readyok")
		case "setoption":
			name, value, ok := parseSetOption(fields[1:])
			if !ok {
				continue
			}
			switch name {
			case "Threads":
				if n, err := strconv.Atoi(value); err == nil {
					threads = clampInt(n, minThreads, maxThreads)
				}
			case "Hash":
				if n, err := strconv.Atoi(value); err == nil {
					if v := clampInt(n, minHashMB, maxHashMB); v != hashMB {
						hashMB = v
						// Drop the old table so the next "go" allocates at
						// the new size. Dropping rather than resizing in
						// place keeps this safe even if a search is still
						// running: it holds its own reference and finishes
						// against the old table.
						tt = nil
					}
				}
			case "SyzygyPath":
				tb = loadTablebase(out, value)
			}
			// Any other option name is ignored per the UCI spec (this
			// engine declares nothing else configurable).
		case "debug", "register":
			// Accepted per the UCI spec's requirement to ignore commands
			// for unsupported features, rather than erroring: this engine
			// has no debug logging, and no registration is required.
		case "ucinewgame":
			b = board.StartingBoard()
			gameHistory = nil
			// Entries from the previous game are still valid results, but
			// they'd occupy space that the new game's tree needs; drop the
			// table so the next "go" starts from a clean one.
			tt = nil
		case "position":
			if nb, nh, ok := parsePosition(out, fields[1:]); ok {
				b, gameHistory = nb, nh
			}
		case "eval":
			// Non-standard, like Stockfish's own "eval": a human-facing
			// breakdown of the static evaluation of the current position,
			// printed as plain text. No GUI sends this, so it can't disturb
			// the protocol; it's for inspecting what the network thinks.
			printEval(out, &b)
		case "bench":
			// Non-standard, like "eval" above and like Stockfish's own
			// "bench": a fixed-position, fixed-depth search whose total node
			// count is a single-number regression signal. See bench.go.
			RunBench(out)
		case "go":
			opts, infinite, ponder, pBudget := buildGoOptions(&b, fields[1:])

			// If a previous search is still running (the GUI is supposed
			// to send "stop" first, but may not), cancel it and wait for
			// its goroutine to finish before starting the new one — both
			// to avoid two goroutines racing to print "bestmove", and so
			// this wait can never hang: cancellation guarantees the old
			// search returns within roughly one node-check interval
			// (~2048 nodes — see internal/search's nodeCheckInterval).
			mu.Lock()
			prevCancel := cancel
			mu.Unlock()
			if prevCancel != nil {
				prevCancel()
			}
			wg.Wait()

			ctx, cancelFn := context.WithCancel(context.Background())
			mu.Lock()
			cancel = cancelFn
			ponderBudget = pBudget
			mu.Unlock()

			if tt == nil {
				tt = search.NewTable(hashMB)
			}

			opts.Context = ctx
			opts.Infinite = infinite || ponder
			opts.Threads = threads
			opts.HashMB = hashMB
			opts.Table = tt
			opts.GameHistory = gameHistory
			opts.Tablebase = tb

			boardCopy := b
			wg.Add(1)
			go func() {
				defer wg.Done()
				var lastInfo search.Info
				opts.OnInfo = func(i search.Info) {
					printInfo(out, i)
					// Only a report that carries a line updates what the
					// ponder move below is read from: currmove reports and
					// heartbeats have no PV, and letting one overwrite
					// this would drop the ponder move for no reason.
					if len(i.PV) > 0 {
						lastInfo = i
					}
				}

				move, ok := search.Search(boardCopy, opts)
				if !ok {
					fmt.Fprintln(out, "bestmove 0000")
					return
				}
				line := "bestmove " + move.String()
				if len(lastInfo.PV) > 1 {
					// A naive ponder suggestion: the second move of the
					// current best line (principal variation) the search
					// found, i.e. the reply it expects if the opponent
					// plays into that line.
					line += " ponder " + lastInfo.PV[1].String()
				}
				fmt.Fprintln(out, line)
			}()
		case "stop":
			mu.Lock()
			c := cancel
			mu.Unlock()
			if c != nil {
				c()
			}
		case "ponderhit":
			// The GUI is confirming its predicted opponent move (which we
			// were already analyzing via "go ... ponder") was played, so
			// the position we're searching is correct — start counting
			// down the real thinking-time budget from now, computed back
			// when "go ponder" arrived, instead of stopping immediately.
			mu.Lock()
			c := cancel
			budget := ponderBudget
			mu.Unlock()
			if c != nil {
				time.AfterFunc(budget, c)
			}
		case "quit":
			mu.Lock()
			c := cancel
			mu.Unlock()
			if c != nil {
				c()
			}
			wg.Wait()
			return
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "uci: reading input: %v\n", err)
	}
	// stdin closed without a "quit" (e.g. the GUI just closed the pipe) —
	// clean up exactly like "quit" would, so a goroutine from an in-flight
	// "go" isn't left running (and, if run() is reused, doesn't race a
	// later call).
	mu.Lock()
	c := cancel
	mu.Unlock()
	if c != nil {
		c()
	}
	wg.Wait()
}

// parsePosition handles "position startpos|fen <fen> [moves ...]".
//
// A move that can't be parsed, or that doesn't match any legal move in the
// position reached so far, fails the whole command rather than returning the
// partially-replayed board. Returning a truncated position would be silently
// game-losing: the engine would search a position several plies behind the
// GUI's — quite possibly with the opposite side to move — and answer with a
// move that is illegal in the real game, which an arbiter scores as a
// forfeit. Since the caller keeps the previous position when ok is false, a
// malformed command costs at most one stale search instead.
//
// The problem is reported as an "info string" (UCI gives "position" no error
// response, the same situation loadTablebase handles the same way) rather than
// swallowed, because the engine rejecting a move the GUI considers legal
// implies a move-generation bug — exactly the kind of thing that must not fail
// quietly.
//
// It also returns the Zobrist hashes of every position the replay passed
// through *before* the final one, which becomes search.Options.GameHistory so
// the search can detect repeating a position the game already visited. GUIs
// resend the full move list with every "position" command, so this is rebuilt
// per command and needs no state carried between them.
func parsePosition(out io.Writer, args []string) (board.Board, []uint64, bool) {
	if len(args) == 0 {
		return board.Board{}, nil, false
	}

	var b board.Board
	var rest []string

	switch args[0] {
	case "startpos":
		b = board.StartingBoard()
		rest = args[1:]
	case "fen":
		end := 1
		for end < len(args) && args[end] != "moves" {
			end++
		}
		parsed, err := board.ParseFEN(strings.Join(args[1:end], " "))
		if err != nil {
			return board.Board{}, nil, false
		}
		b = parsed
		rest = args[end:]
	default:
		return board.Board{}, nil, false
	}

	var history []uint64
	if len(rest) > 0 && rest[0] == "moves" {
		for i, moveStr := range rest[1:] {
			parsed, ok := board.ParseUCIMove(moveStr)
			if !ok {
				fmt.Fprintf(out, "info string position: ignoring command, cannot parse move %d (%q)\n", i+1, moveStr)
				return board.Board{}, nil, false
			}
			legal, ok := matchLegalMove(&b, parsed)
			if !ok {
				fmt.Fprintf(out, "info string position: ignoring command, move %d (%q) is not legal here\n", i+1, moveStr)
				return board.Board{}, nil, false
			}
			// Recorded before the move is made, so history holds the
			// positions the game passed through and not the one the search
			// will start from (which the search puts on its own path).
			history = append(history, b.Hash())
			b = board.MakeMove(b, legal)
		}
	}

	return b, history, true
}

// matchLegalMove finds the legal move in the current position matching m's
// From/To/Promotion, so it carries the special-move flag (castling, en
// passant, double push) that MakeMove needs but UCI notation doesn't encode.
func matchLegalMove(b *board.Board, m board.Move) (board.Move, bool) {
	for _, legal := range board.GenerateLegalMoves(b) {
		if legal.From == m.From && legal.To == m.To && legal.Promotion == m.Promotion {
			return legal, true
		}
	}
	return board.Move{}, false
}

// buildGoOptions turns "go" parameters into search.Options, plus whether
// infinite/ponder search was requested and, for ponder, the time budget to
// start counting down once "ponderhit" confirms the pondered move was
// played. "mate" and "searchmoves" have no case in the switch below, so
// they (and searchmoves' move-list arguments) are simply skipped over like
// any other unrecognized token: forced-mate search and move-restricted
// search aren't implemented. "depth" is acted on (see
// search.Options.MaxDepth).
func buildGoOptions(b *board.Board, args []string) (opts search.Options, infinite, ponder bool, ponderBudget time.Duration) {
	var wtime, btime, winc, binc, movestogo int
	haveWtime, haveBtime := false, false

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "movetime":
			if v, ok := nextInt(args, &i); ok {
				opts.MaxTime = time.Duration(v) * time.Millisecond
			}
		case "nodes":
			if v, ok := nextInt(args, &i); ok {
				opts.MaxIterations = v
			}
		case "depth":
			if v, ok := nextInt(args, &i); ok {
				opts.MaxDepth = v
			}
		case "wtime":
			if v, ok := nextInt(args, &i); ok {
				wtime, haveWtime = v, true
			}
		case "btime":
			if v, ok := nextInt(args, &i); ok {
				btime, haveBtime = v, true
			}
		case "winc":
			if v, ok := nextInt(args, &i); ok {
				winc = v
			}
		case "binc":
			if v, ok := nextInt(args, &i); ok {
				binc = v
			}
		case "movestogo":
			if v, ok := nextInt(args, &i); ok {
				movestogo = v
			}
		case "infinite":
			infinite = true
		case "ponder":
			ponder = true
		}
	}

	remaining, inc := wtime, winc
	if b.SideToMove == board.Black {
		remaining, inc = btime, binc
	}

	switch {
	case opts.MaxTime > 0 || opts.MaxIterations > 0 || infinite || ponder:
		// movetime/nodes/infinite/ponder fully determine the budget on
		// their own; a clock (wtime/btime) given alongside one of them is
		// ignored in favor of the more specific instruction.
	case haveWtime || haveBtime:
		opts.MaxTime = allocateTime(remaining, inc, movestogo)
	}

	if ponder {
		if haveWtime || haveBtime {
			ponderBudget = allocateTime(remaining, inc, movestogo)
		} else {
			ponderBudget = search.DefaultMaxTime
		}
	}

	return opts, infinite, ponder, ponderBudget
}

// nextInt reads the integer following args[*i], advancing *i past it.
func nextInt(args []string, i *int) (int, bool) {
	if *i+1 >= len(args) {
		return 0, false
	}
	*i++
	n, err := strconv.Atoi(args[*i])
	return n, err == nil
}

// Bounds for the "Threads" and "Hash" options, declared to the GUI via
// "uci" and enforced (by clamping) in "setoption". Hash's bounds mirror
// Stockfish's own declared range; Threads' max is just a generous static
// cap — goroutines are cheap, so oversubscribing beyond the machine's
// core count isn't unsafe, just not obviously useful for CPU-bound work.
const (
	minThreads = 1
	maxThreads = 512
	minHashMB  = 1
	maxHashMB  = 33554432
)

func clampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// loadTablebase implements "setoption name SyzygyPath value <path>":
// value is one or more directories of Syzygy WDL (.rtbw) files, joined by
// the OS path-list separator (':' on Unix, ';' on Windows) — matching
// Stockfish's own SyzygyPath convention — each scanned via
// tablebase.Tablebase.AddDirectory. An empty value (including
// "<empty>", the literal placeholder some GUIs send back verbatim for a
// string option they haven't set) clears any previously loaded tables.
// Problems loading a directory are reported as "info string" lines rather
// than failing the command outright — UCI's setoption has no error
// response — so a typo in one of several configured paths doesn't lose
// the tables the others do have.
func loadTablebase(out io.Writer, value string) *tablebase.Tablebase {
	if value == "" || value == "<empty>" {
		return nil
	}

	tb := tablebase.NewTablebase()
	total := 0
	for _, dir := range strings.Split(value, string(os.PathListSeparator)) {
		dir = strings.TrimSpace(dir)
		if dir == "" {
			continue
		}
		n, err := tb.AddDirectory(dir)
		if err != nil {
			fmt.Fprintf(out, "info string SyzygyPath: %v\n", err)
			continue
		}
		total += n
	}

	fmt.Fprintf(out, "info string SyzygyPath: loaded %d tablebase file(s)\n", total)
	if total == 0 {
		return nil
	}
	return tb
}

// parseSetOption parses "setoption name <name> [value <value>]", where
// <name> (and <value>) may contain spaces — both run to the next known
// keyword or the end of the command, per the UCI spec.
func parseSetOption(args []string) (name, value string, ok bool) {
	if len(args) < 2 || args[0] != "name" {
		return "", "", false
	}
	i := 1
	var nameParts []string
	for i < len(args) && args[i] != "value" {
		nameParts = append(nameParts, args[i])
		i++
	}
	if len(nameParts) == 0 {
		return "", "", false
	}
	name = strings.Join(nameParts, " ")
	if i < len(args) && args[i] == "value" {
		value = strings.Join(args[i+1:], " ")
	}
	return name, value, true
}

// Constants for allocateTime's simplified time management. Real engines
// like Stockfish weigh many more factors (position complexity, score
// stability across iterations, game phase); this is a deliberately basic
// stand-in: assume 30 more moves if the GUI didn't say otherwise, spend a
// fraction of the remaining clock plus the increment, and always keep a
// safety buffer so the engine can't flag (lose on time) from its own
// overhead.
const (
	assumedMovesToGo   = 30
	timeSafetyBufferMs = 50
	minMoveTimeMs      = 10
)

// allocateTime turns a clock (remaining time, increment, moves to the
// next time control — all in milliseconds, 0 movesToGo meaning "unknown,
// sudden death") into a budget for the current move.
func allocateTime(remainingMs, incMs, movesToGo int) time.Duration {
	if remainingMs <= 0 {
		return minMoveTimeMs * time.Millisecond
	}

	mtg := movesToGo
	if mtg <= 0 {
		mtg = assumedMovesToGo
	}

	budget := remainingMs/mtg + incMs
	if maxUsable := remainingMs - timeSafetyBufferMs; budget > maxUsable {
		budget = maxUsable
	}
	if budget < minMoveTimeMs {
		budget = minMoveTimeMs
	}
	return time.Duration(budget) * time.Millisecond
}

// printInfo writes one UCI "info" line, in the same style Stockfish and
// other engines use to report search progress: depth reached, the
// engine's score for the position (as a forced mate distance when the
// search has found one, in centipawns otherwise), how many nodes it's
// searched and how fast, and the line (principal variation) it currently
// favors.
// printInfo writes one UCI "info" line. Three shapes come out of
// internal/search's reporting (see its Info), and they are distinguished
// here by which fields are populated rather than by a tag:
//
//   - a currmove report (CurrMoveNumber set): which root move is being
//     searched right now, carrying no result of its own;
//   - a result (PV non-empty): the usual depth/score/pv line, tagged
//     lowerbound/upperbound when the score is only a bound;
//   - a heartbeat (neither): nodes/nps/time/hashfull while a long
//     iteration is still running, deliberately without a score, since the
//     search has not proved one for this depth yet.
//
// Field order follows Stockfish's, which is what GUIs are used to reading,
// though UCI itself imposes none.
func printInfo(out io.Writer, i search.Info) {
	if i.CurrMoveNumber > 0 {
		fmt.Fprintf(out, "info depth %d currmove %s currmovenumber %d\n",
			i.Depth, i.CurrMove.String(), i.CurrMoveNumber)
		return
	}

	var line strings.Builder
	fmt.Fprintf(&line, "info depth %d seldepth %d", i.Depth, i.SelDepth)

	if len(i.PV) > 0 {
		if i.Mate != 0 {
			fmt.Fprintf(&line, " score mate %d", i.Mate)
		} else {
			fmt.Fprintf(&line, " score cp %d", i.ScoreCP)
		}
		switch i.Bound {
		case search.BoundLower:
			line.WriteString(" lowerbound")
		case search.BoundUpper:
			line.WriteString(" upperbound")
		}
	}

	fmt.Fprintf(&line, " nodes %d nps %d", i.Nodes, i.Nps)
	if i.HashFull > 0 {
		fmt.Fprintf(&line, " hashfull %d", i.HashFull)
	}
	fmt.Fprintf(&line, " time %d", i.Time.Milliseconds())

	if len(i.PV) > 0 {
		pv := make([]string, len(i.PV))
		for idx, m := range i.PV {
			pv[idx] = m.String()
		}
		fmt.Fprintf(&line, " pv %s", strings.Join(pv, " "))
	}

	fmt.Fprintln(out, line.String())
}

// pieceLetter renders a piece in FEN's convention: uppercase for White,
// lowercase for Black.
func pieceLetter(c board.Color, pt board.PieceType) string {
	const letters = "pnbrqk"
	ch := letters[pt]
	if c == board.White {
		ch -= 'a' - 'A'
	}
	return string(ch)
}

// printEval writes a per-piece breakdown of the static evaluation, as
// produced by nnue.Explain: what each piece is worth to this position
// according to the network, ranked by how much it matters.
//
// Every score is from the side to move's perspective, matching
// nnue.Evaluate's convention, so a positive contribution always means "this
// piece helps whoever is to move" regardless of which side owns it.
func printEval(out io.Writer, b *board.Board) {
	e := nnue.Explain(b)

	mover := "White"
	if b.SideToMove == board.Black {
		mover = "Black"
	}

	fmt.Fprintf(out, "NNUE evaluation: %+d cp (%s to move)\n", e.TotalCP, mover)
	fmt.Fprintln(out, "Per-piece contribution, from the side to move's perspective:")
	fmt.Fprintln(out)
	for _, c := range e.Contributions {
		fmt.Fprintf(out, "  %s %-2s  %+5d\n", pieceLetter(c.Color, c.Piece), c.Square, c.DeltaCP)
	}
	fmt.Fprintln(out)
	// Kings are omitted above because HalfKP doesn't represent them as
	// features, and the residual is the part no single piece can be
	// credited with — see nnue.Network.Explain.
	fmt.Fprintf(out, "  baseline (kings only)  %+5d\n", e.BaselineCP)
	fmt.Fprintf(out, "  residual (interaction) %+5d\n", e.ResidualCP)
	fmt.Fprintf(out, "  total                  %+5d\n", e.TotalCP)
}

// syncWriter serializes writes from multiple goroutines (the main command
// loop and the background search goroutine "go" starts) onto one
// io.Writer, so their output lines can't interleave into garbled output.
type syncWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (s *syncWriter) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.w.Write(p)
}
