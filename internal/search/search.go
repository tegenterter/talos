// Package search implements move selection via fail-soft negamax
// alpha-beta search with iterative deepening, in the style of classical
// (pre-NNUE) Stockfish: a shared transposition table (tt.go), SEE +
// killer-move + history move ordering (ordering.go, see.go), quiescence
// search and check extensions (quiescence.go, negamax.go), null-move
// pruning, principal variation search, late move reductions (negamax.go),
// mate distance pruning, quiescence delta pruning (quiescence.go), and
// aspiration windows (aspirationSearch below).
//
// Known gaps, deliberate rather than overlooked, roughly in the order they
// are worth closing: LMR reduces by a flat 1-2 plies instead of scaling
// with log(depth)*log(moveIndex); history has no malus for quiet moves that
// were searched and failed to cut off, so it discriminates "ever cut off"
// more than "cuts off often"; null-move reduction is a flat constant with
// no verification search; check extensions are unconditional; and futility
// pruning, late move pruning, internal iterative reduction, singular
// extensions and continuation history are all absent. Raw speed is the
// larger lever than any of these — nnue.Evaluate is over half of all CPU
// because it rebuilds both accumulators per call — but that work belongs in
// internal/nnue rather than here.
//
// Options.Threads > 1 splits the search tree across goroutines, following
// the Young Brothers Wait Concept: a node searches its first move to
// completion — that move's score is what gives every sibling a useful
// alpha — and only then hands remaining siblings to idle helper threads.
// See negamax.go's move loop and splitPoint.
//
// This replaced Lazy SMP, which ran N goroutines over the identical
// iterative-deepening loop and kept whichever reached the greatest depth;
// nearly everything the extra threads did was redundant by construction.
// Splitting instead has each helper search a sibling nobody else will:
// node overhead at 8 threads fell from ~2.6x to ~1.2x (see
// parallel_test.go).
//
// The design leans on two properties of this codebase that make structured
// parallelism unusually cheap here, and that a mutable-board engine can't
// exploit. board.Board is immutable — board.MakeMove takes a board by
// value and returns a new one — so a position can be handed to another
// goroutine with no locking, no undo stack, and no way for one thread's
// search to corrupt another's. And nnue.Evaluate is a pure function with
// no accumulator state to keep in sync. Engines built on mutable boards
// and incrementally-updated accumulators are pushed toward Lazy SMP
// precisely because sharing positions across threads costs them so much.
//
// Concurrency here is therefore about handing out *threads*, not guarding
// data: a thread (its killers, history, and search path) is owned by
// exactly one goroutine at a time and transferred through a channel, which
// supplies the happens-before edge that makes every field on it safe to
// touch without synchronization. Only three things are genuinely shared —
// the transposition table (sharded, mutex-guarded), the node counter, and
// each split point's alpha/cutoff pair.
package search

import (
	"context"
	"sync/atomic"
	"time"

	"talos/internal/board"
	"talos/internal/tablebase"
)

// Options configures a Search call. Leaving every field zero runs
// single-threaded with DefaultMaxTime, DefaultHashMB, and no progress
// reporting.
type Options struct {
	// MaxTime bounds how long Search may run. Ignored if Infinite or
	// MaxIterations > 0.
	MaxTime time.Duration
	// MaxIterations, if positive, is a total node budget shared across
	// all threads instead of a time limit (e.g. for UCI's "go nodes").
	// Ignored if Infinite.
	MaxIterations int
	// Infinite, if true, ignores MaxTime and MaxIterations and searches
	// until Context is cancelled (UCI's "go infinite"/"go ponder"). A
	// Context that never cancels combined with Infinite runs forever.
	Infinite bool
	// Context, if set, is checked periodically regardless of mode, so
	// callers can interrupt a search early (UCI's "stop") in addition to
	// whatever MaxTime/MaxIterations/Infinite already governs. A nil
	// Context behaves like context.Background() (never cancels on its own).
	Context context.Context
	// OnInfo, if set, is called with each completed iterative-deepening
	// depth, with the final statistics before Search returns, and — once a
	// search has run long enough that an unfinished iteration would
	// otherwise look like a hang — with progress from inside that
	// iteration, mirroring how UCI engines stream "info" lines. See
	// reporter (report.go) for what gets reported when.
	OnInfo func(Info)
	// InfoInterval is how often the periodic progress heartbeat may
	// report while an iteration is still running; completed depths are
	// never throttled. Zero means DefaultInfoInterval.
	InfoInterval time.Duration
	// Threads is how many goroutines the search may use: the calling
	// goroutine drives iterative deepening, and the other Threads-1 are
	// helpers it can hand subtrees to (see the package doc). Zero or
	// negative means 1, which searches entirely sequentially.
	//
	// Note that results are only reproducible at Threads == 1. Above that,
	// whether a given move is searched inline or by a helper — and what
	// alpha that helper starts from — depends on scheduling, so node
	// counts and the choice among equally-scoring moves vary run to run.
	Threads int
	// HashMB approximately sizes the shared transposition table in
	// megabytes (see tt.go). Zero or negative means DefaultHashMB.
	// Ignored when Table is set, since that table is already allocated.
	HashMB int
	// Table, if set, is a transposition table reused across Search calls
	// (see Table's doc comment for why that matters). Nil means allocate a
	// throwaway table sized by HashMB for this search alone, which is the
	// right default for one-off searches and tests.
	Table *Table
	// MaxDepth, if positive, bounds iterative deepening to that many plies
	// (UCI's "go depth"). It composes with the other limits: the search
	// stops at whichever bound is hit first. If MaxDepth is the *only*
	// limit given (no MaxTime/MaxIterations/Infinite), the search runs
	// with no time cap until that depth completes — like Infinite, but
	// bounded by depth instead of by Context cancellation.
	MaxDepth int
	// GameHistory holds the Zobrist hashes (board.Board.Hash()) of the
	// positions that occurred in the actual game before root, in order,
	// most recent last. It excludes root itself, which the search records
	// on its own path. Supplying it lets the search see that a line repeats
	// a position the game has already visited; without it, only repetitions
	// occurring wholly inside the search are detected, and the engine will
	// cheerfully repeat its way into a draw it never saw coming.
	//
	// Callers may pass the whole game; Search trims it to the span that can
	// actually repeat. The slice is not modified, and is only read during
	// the search.
	GameHistory []uint64
	// Tablebase, if set, is probed for WDL (win/draw/loss) knowledge at
	// every non-root node reached during search — see negamax.go. Nil
	// means no tablebase probing, regardless of what's loaded into it
	// (there's nothing to disable otherwise; just don't set this field).
	Tablebase *tablebase.Tablebase
}

// DefaultMaxTime is used when Options specifies neither a time nor an
// iteration limit.
const DefaultMaxTime = 1 * time.Second

// DefaultInfoInterval is used when Options.OnInfo is set but
// Options.InfoInterval is zero. A second, rather than something snappier,
// because this now paces only the heartbeat that fills silence during a
// long iteration (see reporter) — results are reported the moment they
// exist — and a line every 200ms of a multi-minute iteration is a wall of
// near-identical text to scroll past rather than a sign of life.
const DefaultInfoInterval = 1 * time.Second

// Soft-limit tuning. An iterative-deepening iteration that gets aborted is
// discarded wholesale, so every millisecond spent past the last *completed*
// depth returns the same move it already had. This constant decides when not
// to begin another one.
//
// iterationGrowthFactor estimates the next iteration's cost as a multiple of
// the one just finished, which is what makes the decision adaptive: a fixed
// "stop after X% of the budget" rule can't tell a position whose iterations
// are still cheap from one whose iterations have started doubling, and in
// practice it lets a search begin a depth it has no chance of finishing just
// because the clock hadn't quite crossed the threshold.
//
// Measured on this engine (startpos, kiwipete and the Italian, depths 1-11,
// single-threaded), successive iterations cost 2.0-3.3x the previous one,
// and the ratio runs highest exactly where it matters most — the deep,
// expensive iterations near the end of a search. Startpos depth 10 cost 4.9x
// depth 9. The value below sits at the upper end of that range rather than
// the median on purpose, because the two failure directions are not
// symmetric: predicting too low starts a doomed iteration and throws away
// everything spent on it (up to half the move's budget), while predicting too
// high only forfeits the occasional ply that would just barely have landed.
//
// There is deliberately NO additional "stop once X% of the budget is gone"
// rule alongside this prediction, though that is the more obvious way to
// write a soft limit. It was tried and cost about 230 Elo in self-play at a
// real time control: the time budget from internal/uci is already the
// engine's *intended* spend for the move, so refusing to search past half of
// it simply halves the thinking time, and the surplus is not handed back fast
// enough by the next move's allocation to make up for it. Skipping an
// iteration that is merely unlikely to finish is a real cost; skipping one
// that provably cannot finish is free, and only the latter is done here.
const iterationGrowthFactor = 2.5

// repeatableHistory trims a game's position hashes to the suffix that can
// still repeat: only the moves since the last irreversible one (a capture or
// pawn move, which is exactly what resets the halfmove clock). Anything
// before that boundary is unreachable by definition, so scanning it at every
// node would be wasted work. halfmoveClock is the root's, so it counts the
// plies back to that boundary.
func repeatableHistory(history []uint64, halfmoveClock int) []uint64 {
	if halfmoveClock <= 0 || len(history) == 0 {
		return nil
	}
	if halfmoveClock < len(history) {
		return history[len(history)-halfmoveClock:]
	}
	return history
}

// Search picks a move for root via iterative-deepening alpha-beta,
// returning the deepest completed iteration's best move (across however
// many threads ran — see Options.Threads). ok is false if root has no
// legal moves.
func Search(root board.Board, opts Options) (board.Move, bool) {
	legal := board.GenerateLegalMoves(&root)
	if len(legal) == 0 {
		return board.Move{}, false
	}
	// Unlike MCTS, a single legal move isn't worth short-circuiting: it
	// costs nothing extra to let iterative deepening run normally (the
	// root just has one child), and doing so matters for correctness —
	// the forced move might walk into a mate a few plies later that only
	// searching deeper would reveal.

	ctx := opts.Context
	if ctx == nil {
		ctx = context.Background()
	}
	threads := opts.Threads
	if threads <= 0 {
		threads = 1
	}
	// A caller-supplied table is reused as-is (and kept warm between moves);
	// otherwise this search gets a throwaway one. Allocating only in the
	// nil case matters: at Hash 1024 an unnecessary allocation here would
	// cost a gigabyte per search.
	tt := opts.Table.table(opts.HashMB)

	maxDepth := opts.MaxDepth
	if maxDepth <= 0 || maxDepth > maxPly {
		maxDepth = maxPly
	}
	// depthOnly mirrors Infinite's time handling (no deadline, only
	// Context can abort) when depth is the sole limit given: a bare "go
	// depth N" must run however long it takes to complete depth N, not
	// get cut off by an unrelated default time budget.
	depthOnly := opts.MaxDepth > 0 && opts.MaxTime <= 0 && opts.MaxIterations <= 0 && !opts.Infinite

	maxTime := opts.MaxTime
	if maxTime <= 0 && !depthOnly {
		maxTime = DefaultMaxTime
	}
	infoInterval := opts.InfoInterval
	if infoInterval <= 0 {
		infoInterval = DefaultInfoInterval
	}

	start := time.Now()
	var nodes atomic.Int64

	// timeLimited is "this search is bounded by a clock", the only mode in
	// which the soft limit below applies: Infinite and a bare "go depth N"
	// have no deadline at all, and a node-limited search is bounded by work
	// rather than time.
	timeLimited := !opts.Infinite && !depthOnly && opts.MaxIterations <= 0

	// Everything the threads share, built once. Each thread then carries
	// only its own move-ordering heuristics and search path (see thread).
	sc := &searchCtx{
		tt:            tt,
		nodes:         &nodes,
		ctx:           ctx,
		infinite:      opts.Infinite || depthOnly,
		nodeLimited:   !opts.Infinite && !depthOnly && opts.MaxIterations > 0,
		nodeLimit:     int64(opts.MaxIterations),
		deadline:      start.Add(maxTime),
		tablebase:     opts.Tablebase,
		gameHistory:   repeatableHistory(opts.GameHistory, root.HalfmoveClock),
		splitMinDepth: splitMinDepth,
	}
	sc.rep = newReporter(opts.OnInfo, infoInterval, start, tt, &nodes)

	// The helper pool. Threads are allocated once, up front, and handed
	// around through a channel for the rest of the search — a thread is
	// owned by exactly one goroutine at a time, so its killers, history and
	// search path need no synchronization and stay warm across every
	// subtree it works on. That is the whole reason the pool is threads
	// rather than goroutines: forking fresh per-goroutine state at each
	// split would throw those heuristics away thousands of times a second.
	//
	// Capacity is threads-1 because the calling goroutine is itself one of
	// the workers; it drives iterative deepening and hands subtrees out.
	var helpers []*thread
	if threads > 1 {
		sc.sem = make(chan *thread, threads-1)
		helpers = make([]*thread, 0, threads-1)
		for i := 0; i < threads-1; i++ {
			h := &thread{s: sc}
			helpers = append(helpers, h)
			sc.sem <- h
		}
	}

	// One iterative-deepening driver, running on the calling goroutine.
	//
	// This used to be `threads` goroutines each running this identical loop
	// (Lazy SMP), with Search keeping whichever reached the greatest
	// completed depth. That shape is incompatible with splitting the tree:
	// a split needs one authoritative deepening loop that *hands subtrees
	// out* to helpers, not N loops racing to redo each other's work. The
	// helpers and the split points arrive next; until then this is simply a
	// single-threaded search, and opts.Threads has no effect (see its doc).
	type result struct {
		score, depth, selDepth int
		pv                     []board.Move
	}

	t := &thread{s: sc, driver: true}

	var best result
	var prevScore int
	haveScore := false
	// How long the most recently completed iteration took, used to predict
	// the next one's cost for the soft limit below.
	var lastIteration time.Duration

	for depth := 1; depth <= maxDepth; depth++ {
		// Soft limit: don't *start* an iteration there isn't realistically
		// time to finish, since an aborted one is discarded and its time
		// buys nothing. Stopping here returns the same move sooner and banks
		// the remainder for later moves in the game; the hard deadline still
		// governs aborting an iteration already under way.
		if timeLimited && depth > 1 {
			elapsed := time.Since(start)
			predicted := time.Duration(float64(lastIteration) * iterationGrowthFactor)
			if elapsed+predicted > maxTime {
				break
			}
		}

		iterationStart := time.Now()
		sc.rep.beginIteration(depth)

		// Every thread's history is decayed, not just the driver's. Helpers
		// accumulate history for the whole search, so decaying only this
		// thread let theirs grow without bound — and decayHistory's halving
		// is precisely what keeps a history score below the killer and
		// good-capture ordering bands, so an undecayed helper would
		// eventually order stale quiets above both. Safe to touch helper
		// state here for the same reason mergeSelDepth is safe below: every
		// split waits for the helpers it spawned before its node returns, so
		// the previous iteration's root having returned means none are still
		// running.
		t.decayHistory()
		for _, h := range helpers {
			h.decayHistory()
		}
		var pv []board.Move
		score := t.aspirationSearch(&root, depth, prevScore, haveScore, &pv)
		if t.aborted {
			break // discard this incomplete depth; keep the previous one
		}
		best = result{score: score, depth: depth, selDepth: t.selDepth, pv: pv}
		prevScore, haveScore = score, true
		lastIteration = time.Since(iterationStart)

		sc.rep.complete(best.depth, best.selDepth, best.score, best.pv)
	}

	if len(best.pv) == 0 {
		// The search aborted before completing even depth 1 (e.g. a
		// pathological "movetime 0"); fall back to any legal move rather
		// than returning nothing.
		best.pv = []board.Move{legal[0]}
	}
	// Publish every thread's deepest ply into the search-wide figure the
	// final report reads. The helpers are all idle by now: each split waits
	// for the helpers it spawned before its node returns, so the root
	// returning means none are left running.
	t.mergeSelDepth()
	for _, h := range helpers {
		h.mergeSelDepth()
	}

	// selDepth is reported across every thread, not just the winning one:
	// it means "the deepest ply this search reached anywhere", which is
	// what UCI's seldepth denotes regardless of thread count.
	sc.rep.complete(best.depth, int(sc.selDepth.Load()), best.score, best.pv)

	return best.pv[0], true
}

// aspirationMinDepth is the shallowest iteration aspiration windows apply
// to: a score from a shallow iteration is too unstable (few plies of
// lookahead) to window profitably, so early depths always get the full
// -infinity..infinity window instead.
const aspirationMinDepth = 4

// aspirationWindowCP is the initial half-width (in centipawns) of the
// window searched around the previous iteration's score. Doubled on each
// fail-low/fail-high before re-searching the same depth. A var, not a
// const, so aspiration_test.go can force a fail on the first try by
// shrinking it drastically.
var aspirationWindowCP = 25

// aspirationEnabled exists only so aspiration_test.go can measure
// aspiration windows' actual effect (compare node counts, and confirm
// fail-low/fail-high re-search recovers the exact correct score, with it
// on vs. off) — there's no UCI option or other production path that ever
// sets it false.
var aspirationEnabled = true

// aspirationSearch runs one iterative-deepening iteration at depth, using a
// narrow window around prevScore (the previous iteration's completed
// score) when that's likely to pay off, re-searching the same depth with a
// widened window on fail-low/fail-high. haveScore is false for the very
// first iteration, which has no previous score to window around.
//
// A narrow window lets negamax's alpha-beta cutoffs trigger sooner — most
// of the time the true score doesn't move much between adjacent
// iterations, so a tight window around the last one is usually right on
// the first try. When it isn't, the failed search still wasn't wasted: it
// only ever narrows down which side (fail-low or fail-high) the widened
// re-search needs to explore further.
func (t *thread) aspirationSearch(root *board.Board, depth, prevScore int, haveScore bool, pv *[]board.Move) int {
	if !aspirationEnabled || !haveScore || depth < aspirationMinDepth ||
		prevScore <= -mateThreshold || prevScore >= mateThreshold {
		// Mate scores are excluded too: narrowing around one is either
		// meaningless (any window around a near-mateValue score is either
		// far too wide or immediately fails) or an easy source of subtle
		// window-arithmetic bugs near mateValue, for no real benefit.
		return t.negamax(root, depth, 0, -infinity, infinity, pv, 0)
	}

	window := aspirationWindowCP
	alpha, beta := prevScore-window, prevScore+window
	for {
		*pv = nil
		score := t.negamax(root, depth, 0, alpha, beta, pv, 0)
		if t.aborted {
			return score
		}
		if score <= alpha {
			// A fail-low is the one root result the move loop can't report
			// itself: no move raised alpha, so nothing in there fired. It
			// is also the report a watching human most wants — the score
			// is dropping, and the re-search below may take a while.
			t.s.rep.rootUpdate(t.selDepth, score, BoundUpper, *pv)
			window *= 2
			alpha = prevScore - window
			if alpha < -infinity {
				alpha = -infinity
			}
			continue
		}
		if score >= beta {
			window *= 2
			beta = prevScore + window
			if beta > infinity {
				beta = infinity
			}
			continue
		}
		return score
	}
}
