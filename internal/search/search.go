// Package search implements move selection via fail-soft negamax
// alpha-beta search with iterative deepening, in the style of classical
// (pre-NNUE) Stockfish: a shared transposition table (tt.go), SEE +
// killer-move + history move ordering (ordering.go, see.go), quiescence
// search and check extensions (quiescence.go, negamax.go), null-move
// pruning, principal variation search, and late move reductions
// (negamax.go). It intentionally leaves out the deeper end of Stockfish's
// own technique list — aspiration windows, futility pruning, singular
// extensions — to keep this a manageable, well-understood implementation.
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
	// OnInfo, if set, is called periodically (every InfoInterval) during
	// the search and once more with the final statistics before Search
	// returns, mirroring how UCI engines stream "info" lines.
	OnInfo func(Info)
	// InfoInterval throttles OnInfo. Zero means DefaultInfoInterval.
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
	HashMB int
	// MaxDepth, if positive, bounds iterative deepening to that many plies
	// (UCI's "go depth"). It composes with the other limits: the search
	// stops at whichever bound is hit first. If MaxDepth is the *only*
	// limit given (no MaxTime/MaxIterations/Infinite), the search runs
	// with no time cap until that depth completes — like Infinite, but
	// bounded by depth instead of by Context cancellation.
	MaxDepth int
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
// Options.InfoInterval is zero.
const DefaultInfoInterval = 200 * time.Millisecond

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
	tt := newTranspositionTable(opts.HashMB)

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
		splitMinDepth: splitMinDepth,
	}

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

	t := &thread{s: sc}

	lastInfo := start
	var best result

	for depth := 1; depth <= maxDepth; depth++ {
		t.decayHistory()
		var pv []board.Move
		score := t.negamax(&root, depth, 0, -infinity, infinity, &pv, 0)
		if t.aborted {
			break // discard this incomplete depth; keep the previous one
		}
		best = result{score: score, depth: depth, selDepth: t.selDepth, pv: pv}

		if opts.OnInfo != nil && time.Since(lastInfo) >= infoInterval {
			opts.OnInfo(buildInfo(best.depth, best.selDepth, best.score, int(nodes.Load()), time.Since(start), best.pv))
			lastInfo = time.Now()
		}
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

	if opts.OnInfo != nil {
		// selDepth is reported across every thread, not just the winning
		// one: it means "the deepest ply this search reached anywhere",
		// which is what UCI's seldepth denotes regardless of thread count.
		opts.OnInfo(buildInfo(best.depth, int(sc.selDepth.Load()), best.score, int(nodes.Load()), time.Since(start), best.pv))
	}

	return best.pv[0], true
}
