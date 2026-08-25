package search

import (
	"context"
	"runtime"
	"testing"
	"time"

	"talos/internal/board"
)

// Tests for the parallel search: the tree is split across helper
// goroutines (YBWC — see negamax.go's move loop), rather than running N
// copies of the whole search as Lazy SMP did.
//
// Results at Threads > 1 are deliberately NOT deterministic: which helper
// finishes first, and therefore how much alpha had improved when each one
// started, depends on scheduling. So these tests assert properties that
// must hold regardless of timing — never exact node counts. Exact output is
// pinned separately, at Threads: 1, by golden_test.go.

const parallelFEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

// TestParallelSolvesTacticsIdentically is the correctness bar, applied to
// positions that have exactly one right answer. On those, threading must
// not change the move: if it does, results are being raced or dropped.
//
// It deliberately does NOT assert move equality on quiet positions. A
// parallel search is legitimately nondeterministic here: whether a given
// move gets searched inline or by a helper depends on whether a helper
// happened to be free, and a helper reads alpha when it starts, so the
// window each move is searched with varies run to run. Among near-equal
// moves that genuinely changes which one comes out on top — the rook
// endgame in goldenPositions has two moves scoring identically, and either
// may win. Demanding determinism there would be demanding the search give
// up the parallelism it was rewritten for.
func TestParallelSolvesTacticsIdentically(t *testing.T) {
	positions := []string{
		mateInOneFEN,
		"4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1", // free queen capture
	}
	for _, fen := range positions {
		b := mustFEN(t, fen)
		want, ok := Search(b, Options{MaxDepth: 6, Threads: 1})
		if !ok {
			t.Fatalf("%s: no legal move", fen)
		}
		for _, threads := range []int{2, 4, 8} {
			for rep := 0; rep < 3; rep++ {
				got, ok := Search(b, Options{MaxDepth: 6, Threads: threads})
				if !ok {
					t.Fatalf("%s: no legal move at %d threads", fen, threads)
				}
				if got != want {
					t.Errorf("%s: %d threads (rep %d) picked %v, single-threaded picked %v",
						fen, threads, rep, got, want)
				}
			}
		}
	}
}

// TestParallelDoesNotDegradeScores is the weaker but broader companion to
// the tactical check: across every golden position, threading may change
// which of several near-equal moves is chosen, but it must not come back
// with a materially worse assessment. A parallel search that dropped
// results, folded stale scores wrongly, or lost a cutoff would show up as
// scores drifting downward.
//
// In practice threading often scores *better* here, which is not a
// paradox: helpers start from whatever alpha had been published when they
// began, so they frequently search a wider window than the sequential loop
// would have used by that point, prune less, and return a sharper number.
func TestParallelDoesNotDegradeScores(t *testing.T) {
	// Generous, because fixed-depth scores legitimately move when the
	// window does. The point is to catch collapse, not jitter.
	const tolerance = 100

	for _, p := range goldenPositions {
		b := mustFEN(t, p.fen)

		var seq Info
		Search(b, Options{MaxDepth: 7, Threads: 1, OnInfo: func(i Info) { seq = i }})

		for _, threads := range []int{2, 8} {
			var par Info
			Search(b, Options{MaxDepth: 7, Threads: threads, OnInfo: func(i Info) { par = i }})

			if seq.Mate != 0 || par.Mate != 0 {
				if seq.Mate != par.Mate {
					t.Errorf("%s: %d threads reported mate %d, single-threaded reported mate %d",
						p.name, threads, par.Mate, seq.Mate)
				}
				continue
			}
			if par.ScoreCP < seq.ScoreCP-tolerance {
				t.Errorf("%s: %d threads scored %d, materially worse than single-threaded %d",
					p.name, threads, par.ScoreCP, seq.ScoreCP)
			}
		}
	}
}

// TestParallelNodeOverhead measures the duplicated work splitting costs,
// which is the number the whole rewrite exists to improve.
//
// Lazy SMP ran N identical searches and kept the deepest, so nearly
// everything every extra thread did was redundant. Splitting the tree
// instead means a helper searches a sibling nobody else will. The
// historical figures at 8 threads, depth 6, on this position:
//
//	~2.6x  Lazy SMP with randomized move-ordering tie-breaks
//	~6.4x  Lazy SMP after those were removed (a hobbled intermediate)
//	~1.2x  splitting the tree (measured at depth 8)
//
// 2.6x was always the real bar — 6.4x was a state that existed only
// between two commits. The assertion below is loose because the exact
// figure moves with scheduling; the logged number is the interesting part.
func TestParallelNodeOverhead(t *testing.T) {
	if testing.Short() {
		t.Skip("overhead measurement is slow")
	}
	b := mustFEN(t, parallelFEN)

	nodesAt := func(threads int) int {
		var last Info
		Search(b, Options{MaxDepth: 8, Threads: threads, OnInfo: func(i Info) { last = i }})
		return last.Nodes
	}

	one := nodesAt(1)
	eight := nodesAt(8)
	ratio := float64(eight) / float64(one)
	t.Logf("node overhead at depth 8: 1 thread = %d, 8 threads = %d (%.2fx)", one, eight, ratio)

	if ratio > 2.6 {
		t.Errorf("8 threads searched %.2fx the nodes of 1 thread; Lazy SMP's ~2.6x was the bar to beat", ratio)
	}
}

// TestParallelSpeedsUpSearch checks the point of all this actually
// materializes in wall-clock time, not just in node accounting — a search
// can have excellent node overhead and still be slow if the helpers spend
// their time idle.
func TestParallelSpeedsUpSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("timing measurement is slow")
	}
	if runtime.GOMAXPROCS(0) < 4 {
		t.Skipf("needs >= 4 usable cores, have %d", runtime.GOMAXPROCS(0))
	}
	b := mustFEN(t, parallelFEN)

	timeAt := func(threads int) time.Duration {
		start := time.Now()
		Search(b, Options{MaxDepth: 8, Threads: threads})
		return time.Since(start)
	}

	one := timeAt(1)
	four := timeAt(4)
	speedup := float64(one) / float64(four)
	t.Logf("depth 8: 1 thread = %v, 4 threads = %v (%.2fx speedup)", one, four, speedup)

	// Deliberately undemanding: this runs on shared CI-style machines where
	// timing is noisy, and the purpose is to catch parallelism that does
	// nothing (or actively hurts), not to hold a performance target.
	if speedup < 1.3 {
		t.Errorf("4 threads gave only %.2fx speedup over 1; splitting isn't paying for itself", speedup)
	}
}

// TestParallelDoesNotLeakGoroutines checks the invariant every split
// depends on: a node waits for the helpers it spawned before returning.
// Helpers read the parent's board and write into its result slice, so one
// outliving its node is both a leak and a use-after-return.
func TestParallelDoesNotLeakGoroutines(t *testing.T) {
	b := mustFEN(t, parallelFEN)

	// Warm up first so pool allocation and lazily-started runtime
	// goroutines don't count against the measurement.
	Search(b, Options{MaxDepth: 6, Threads: 8})
	runtime.Gosched()
	before := runtime.NumGoroutine()

	for i := 0; i < 3; i++ {
		Search(b, Options{MaxDepth: 7, Threads: 8})
	}
	runtime.Gosched()

	if after := runtime.NumGoroutine(); after > before {
		t.Errorf("goroutines grew from %d to %d across searches; helpers are outliving their split points", before, after)
	}
}

// TestParallelDoesNotLeakGoroutinesOnAbort is the same invariant on the
// path most likely to break it: a search cancelled mid-flight, where nodes
// unwind through their abort returns rather than completing normally.
func TestParallelDoesNotLeakGoroutinesOnAbort(t *testing.T) {
	b := mustFEN(t, parallelFEN)

	Search(b, Options{MaxDepth: 6, Threads: 8})
	runtime.Gosched()
	before := runtime.NumGoroutine()

	for i := 0; i < 3; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
		move, ok := Search(b, Options{Infinite: true, Threads: 8, Context: ctx})
		cancel()
		if !ok {
			t.Fatal("cancelled search returned no move")
		}
		// Even an aborted search must hand back something legal to play.
		if !moveIn(move, board.GenerateLegalMoves(&b)) {
			t.Fatalf("cancelled search returned illegal move %v", move)
		}
	}
	runtime.Gosched()

	if after := runtime.NumGoroutine(); after > before {
		t.Errorf("goroutines grew from %d to %d across cancelled searches", before, after)
	}
}

// TestParallelWithMoreThreadsThanCores is the deadlock guard. Helper
// acquisition is a non-blocking receive that falls back to searching
// inline, so a node can never wait on a thread that can't be freed — but
// that is exactly the kind of property worth testing rather than trusting,
// since the failure mode is a silent hang.
func TestParallelWithMoreThreadsThanCores(t *testing.T) {
	b := mustFEN(t, parallelFEN)
	for _, threads := range []int{16, 64, 200} {
		move, ok := Search(b, Options{MaxDepth: 6, Threads: threads})
		if !ok {
			t.Fatalf("threads=%d: no legal move", threads)
		}
		if !moveIn(move, board.GenerateLegalMoves(&b)) {
			t.Fatalf("threads=%d: returned illegal move %v", threads, move)
		}
	}
}
