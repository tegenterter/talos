// Package datagen generates NNUE training data by self-play.
//
// Training a network needs millions of positions, each labelled with what
// the engine thinks of it and how the game it came from ended. That is what
// this produces: fixed-node self-play games from randomised openings,
// written in the plain text format nnue-pytorch reads, so an existing
// trainer works unmodified.
//
// The quality of the labels is the quality of the engine, which is why this
// arrives after the search work rather than before it — and why it is worth
// re-generating data after a strength gain rather than training forever on
// the first dump.
package datagen

import (
	"bufio"
	"fmt"
	"io"
	"math/rand"
	"sync"
	"time"

	"talos/internal/board"
	"talos/internal/eval"
	"talos/internal/search"
)

// Options configures a generation run.
type Options struct {
	// Games is how many complete games to play.
	Games int
	// Nodes is the search budget per move. Fixed nodes rather than fixed
	// time on purpose: it makes a run reproducible from Seed, and it makes
	// the labels independent of how loaded the machine was.
	Nodes int
	// Threads is how many games to play concurrently. Each game searches
	// single-threaded — parallelism across games rather than within one is
	// both simpler and more efficient, since a single-threaded search has no
	// coordination overhead and every core stays busy.
	Threads int
	// RandomPlies is how many opening moves to play at random before
	// recording anything, which is what stops every game being the same
	// game. The positions themselves are not recorded: they are noise, not
	// play.
	RandomPlies int
	// MaxPlies abandons a game that has gone on too long without a result.
	MaxPlies int
	// AdjudicateScore, if positive, ends a game once one side has been ahead
	// by at least this much for AdjudicatePlies consecutive plies. Most of a
	// won game is technique the trainer does not need to see, and playing it
	// out costs nodes that could be another game.
	AdjudicateScore int
	AdjudicatePlies int
	// Seed fixes the randomness, so a run can be reproduced exactly.
	Seed int64
	// Out receives the records. Progress goes to Log.
	Out io.Writer
	Log io.Writer
}

// DefaultOptions are reasonable settings for a first run.
func DefaultOptions() Options {
	return Options{
		Games:           1000,
		Nodes:           5000,
		Threads:         1,
		RandomPlies:     8,
		MaxPlies:        400,
		AdjudicateScore: 2000,
		AdjudicatePlies: 4,
		Seed:            1,
	}
}

// result is a game's outcome from White's point of view.
type result int

const (
	whiteWins result = 1
	draw      result = 0
	blackWins result = -1
)

// sample is one recorded position.
type sample struct {
	fen   string
	move  string
	score int // centipawns, from the side to move's point of view
	ply   int
	stm   board.Color
}

// Run plays Games self-play games and writes their positions to Out.
func Run(opts Options) error {
	if opts.Threads < 1 {
		opts.Threads = 1
	}
	if opts.Out == nil {
		return fmt.Errorf("datagen: no output writer")
	}

	w := bufio.NewWriterSize(opts.Out, 1<<20)
	defer w.Flush()

	var (
		mu       sync.Mutex // guards w and the counters
		written  int
		finished int
		start    = time.Now()
	)

	games := make(chan int)
	var wg sync.WaitGroup
	for i := 0; i < opts.Threads; i++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			// Each worker gets its own deterministic stream and its own
			// transposition table, reused across its games the way the UCI
			// loop reuses one across moves.
			rng := rand.New(rand.NewSource(opts.Seed*1_000_003 + int64(worker)))
			table := search.NewTable(16)

			for range games {
				samples, res := playGame(opts, rng, table)

				mu.Lock()
				for _, s := range samples {
					writeSample(w, s, res)
					written++
				}
				finished++
				if opts.Log != nil && finished%100 == 0 {
					elapsed := time.Since(start)
					fmt.Fprintf(opts.Log, "%d/%d games, %d positions, %.0f games/min\n",
						finished, opts.Games, written, float64(finished)/elapsed.Minutes())
				}
				mu.Unlock()
			}
		}(i)
	}

	for i := 0; i < opts.Games; i++ {
		games <- i
	}
	close(games)
	wg.Wait()

	if opts.Log != nil {
		fmt.Fprintf(opts.Log, "done: %d games, %d positions in %v\n",
			finished, written, time.Since(start).Round(time.Second))
	}
	return w.Flush()
}

// playGame plays one game and returns the positions worth training on.
func playGame(opts Options, rng *rand.Rand, table *search.Table) ([]sample, result) {
	b := board.StartingBoard()
	history := []uint64{}

	// A random opening. Its positions are deliberately not recorded — they
	// are there to make the games differ, not to be learned from.
	for i := 0; i < opts.RandomPlies; i++ {
		moves := board.GenerateLegalMoves(&b)
		if len(moves) == 0 {
			return nil, draw
		}
		history = append(history, b.Hash())
		b = board.MakeMove(b, moves[rng.Intn(len(moves))])
	}

	var samples []sample
	decisive, decisiveFor := 0, draw

	for ply := 0; ply < opts.MaxPlies; ply++ {
		if res, over := outcome(&b, history); over {
			return samples, res
		}

		var info search.Info
		move, ok := search.Search(b, search.Options{
			MaxIterations: opts.Nodes,
			Threads:       1,
			Table:         table,
			GameHistory:   history,
			InfoInterval:  time.Hour,
			OnInfo: func(i search.Info) {
				if len(i.PV) > 0 {
					info = i
				}
			},
		})
		if !ok {
			return samples, draw
		}

		if s, keep := recordable(&b, move, info, ply); keep {
			samples = append(samples, s)
		}

		// Adjudication: once one side is clearly winning and stays that way,
		// the rest of the game is technique the trainer does not need.
		if opts.AdjudicateScore > 0 {
			switch {
			case info.Mate > 0 || info.ScoreCP >= opts.AdjudicateScore:
				decisive, decisiveFor = decisive+1, winnerFrom(b.SideToMove, true)
			case info.Mate < 0 || info.ScoreCP <= -opts.AdjudicateScore:
				decisive, decisiveFor = decisive+1, winnerFrom(b.SideToMove, false)
			default:
				decisive, decisiveFor = 0, draw
			}
			if decisive >= opts.AdjudicatePlies {
				return samples, decisiveFor
			}
		}

		history = append(history, b.Hash())
		b = board.MakeMove(b, move)
	}
	return samples, draw
}

// winnerFrom turns "the side to move is winning" into a result from White's
// point of view.
func winnerFrom(stm board.Color, stmWinning bool) result {
	if (stm == board.White) == stmWinning {
		return whiteWins
	}
	return blackWins
}

// recordable decides whether a position belongs in the training set, and
// builds its record if so.
//
// Three exclusions, all standard, and all for the same reason: a static
// evaluator is being trained, so it should only be shown positions where a
// static evaluation is a meaningful thing to ask for.
//
//   - In check: the side to move is under threat and the position's value is
//     whatever the reply is worth, not what the pieces are worth.
//   - Best move is a capture: the position is mid-exchange, which is exactly
//     what the search's quiescence exists to resolve.
//   - Mate scores: not evaluations of a position at all.
func recordable(b *board.Board, move board.Move, info search.Info, ply int) (sample, bool) {
	if info.Mate != 0 {
		return sample{}, false
	}
	kingSq := b.Pieces[b.SideToMove][board.King].LSB()
	if board.IsSquareAttacked(b, kingSq, b.SideToMove.Opposite()) {
		return sample{}, false
	}
	if _, _, occupied := b.PieceAt(move.To); occupied || move.Flag == board.EnPassantCapture {
		return sample{}, false
	}
	return sample{
		fen:   b.FEN(),
		move:  move.String(),
		score: info.ScoreCP,
		ply:   ply,
		stm:   b.SideToMove,
	}, true
}

// outcome reports whether the game is over and how it ended.
func outcome(b *board.Board, history []uint64) (result, bool) {
	if len(board.GenerateLegalMoves(b)) == 0 {
		kingSq := b.Pieces[b.SideToMove][board.King].LSB()
		if board.IsSquareAttacked(b, kingSq, b.SideToMove.Opposite()) {
			// Checkmate: whoever is to move has lost.
			return winnerFrom(b.SideToMove, false), true
		}
		return draw, true // stalemate
	}
	if b.HalfmoveClock >= 100 || eval.InsufficientMaterial(b) {
		return draw, true
	}
	// Threefold repetition. The engine's own search treats a single
	// repetition as a draw; a game is only actually drawn on the third.
	hash, seen := b.Hash(), 0
	for _, h := range history {
		if h == hash {
			seen++
		}
	}
	if seen >= 2 {
		return draw, true
	}
	return draw, false
}

// writeSample emits one record in nnue-pytorch's plain text format. The
// result is written from the recorded position's side to move, which is the
// convention that format expects.
func writeSample(w *bufio.Writer, s sample, res result) {
	outcome := int(res)
	if s.stm == board.Black {
		outcome = -outcome
	}
	fmt.Fprintf(w, "fen %s\nmove %s\nscore %d\nply %d\nresult %d\ne\n",
		s.fen, s.move, s.score, s.ply, outcome)
}
