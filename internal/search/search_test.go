package search

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"talos/internal/board"
)

func mustFEN(t *testing.T, fen string) board.Board {
	t.Helper()
	b, err := board.ParseFEN(fen)
	if err != nil {
		t.Fatalf("ParseFEN(%q): %v", fen, err)
	}
	return b
}

func moveIn(move board.Move, moves []board.Move) bool {
	for _, m := range moves {
		if m == move {
			return true
		}
	}
	return false
}

func TestSearchReturnsALegalMove(t *testing.T) {
	b := board.StartingBoard()
	move, ok := Search(b, Options{MaxTime: 200 * time.Millisecond})
	if !ok {
		t.Fatal("Search reported no legal moves at startpos")
	}
	if legal := board.GenerateLegalMoves(&b); !moveIn(move, legal) {
		t.Errorf("Search returned %v, which is not among the legal moves %v", move, legal)
	}
}

func TestSearchDetectsNoLegalMoves(t *testing.T) {
	// Fool's mate: White to move, checkmated.
	b := mustFEN(t, "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
	if _, ok := Search(b, Options{MaxTime: 50 * time.Millisecond}); ok {
		t.Error("Search reported a move for a position with no legal moves")
	}
}

func TestSearchFindsAFreeQueenCapture(t *testing.T) {
	// White knight on c3 can capture a completely undefended queen on d5
	// for free; no other white move comes close in value.
	b := mustFEN(t, "4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1")
	move, ok := Search(b, Options{MaxTime: 200 * time.Millisecond})
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	from, _ := board.ParseSquare("c3")
	to, _ := board.ParseSquare("d5")
	if move.From != from || move.To != to {
		t.Errorf("Search picked %v, want the free queen capture c3d5", move)
	}
}

// mateInOneFEN: White to move, black king g8 walled in by its own pawns
// f7/g7/h7, white rook a1 free to swing to a8 for a back-rank mate. The
// only mating move is Ra8# (verified programmatically: every other legal
// move leaves the opponent with replies, only a1a8 leaves zero).
const mateInOneFEN = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"

func TestSearchFindsMateInOne(t *testing.T) {
	b := mustFEN(t, mateInOneFEN)
	move, ok := Search(b, Options{MaxTime: 300 * time.Millisecond})
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	from, _ := board.ParseSquare("a1")
	to, _ := board.ParseSquare("a8")
	if move.From != from || move.To != to {
		t.Errorf("Search picked %v, want the mate-in-1 Ra8#", move)
	}
}

func TestSearchReportsMateScore(t *testing.T) {
	b := mustFEN(t, mateInOneFEN)
	var last Info
	_, ok := Search(b, Options{MaxTime: 300 * time.Millisecond, OnInfo: func(i Info) { last = i }})
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	if last.Mate != 1 {
		t.Errorf("final Info.Mate = %d, want 1 (mate in 1 for the side to move)", last.Mate)
	}
}

func TestSearchReportsGettingMated(t *testing.T) {
	// Black king a8 has exactly one legal move (Kb8, boxed in otherwise
	// by the white king on b6); from there White has a forced Rh8#. So
	// Black, to move here, is getting mated in 1 (verified
	// programmatically: a8b8 is forced, and after it h1h8 is the only
	// move leaving White's opponent with zero replies while in check).
	b := mustFEN(t, "k7/8/1K6/8/8/8/8/7R b - - 0 1")
	var last Info
	_, ok := Search(b, Options{MaxTime: 300 * time.Millisecond, OnInfo: func(i Info) { last = i }})
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	if last.Mate != -1 {
		t.Errorf("final Info.Mate = %d, want -1 (side to move gets mated in 1)", last.Mate)
	}
}

func TestSearchPlaysAForcedSingleLegalMove(t *testing.T) {
	// Search deliberately doesn't short-circuit a single legal move the
	// way the earlier MCTS implementation did (see search.go) — a forced
	// move can still walk into a mate a few plies later that only
	// actually searching would reveal — so this just confirms the normal
	// search still ends up returning it.
	b := mustFEN(t, "8/8/8/8/8/1k6/8/K7 w - - 0 1")
	move, ok := Search(b, Options{MaxTime: 50 * time.Millisecond})
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	to, _ := board.ParseSquare("b1")
	if move.To != to {
		t.Errorf("Search picked %v, want the only legal move (a1b1)", move)
	}
}

func TestSearchRespectsMaxTime(t *testing.T) {
	b := board.StartingBoard()
	start := time.Now()
	if _, ok := Search(b, Options{MaxTime: 50 * time.Millisecond}); !ok {
		t.Fatal("Search reported no legal moves at startpos")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("Search took %v, want roughly its 50ms budget", elapsed)
	}
}

func TestSearchRespectsNodeBudget(t *testing.T) {
	// The node limit is only checked every nodeCheckInterval+1 nodes (see
	// negamax.go), so some overshoot up to that much is expected and
	// correct, not a bug — this bound allows for it with margin.
	const target = 5000
	const tolerance = 2 * (nodeCheckInterval + 1)

	b := board.StartingBoard()
	var last Info
	_, ok := Search(b, Options{MaxIterations: target, OnInfo: func(i Info) { last = i }})
	if !ok {
		t.Fatal("Search reported no legal moves at startpos")
	}
	if last.Nodes > target+tolerance {
		t.Errorf("final Info.Nodes = %d, want <= %d (target %d plus checking granularity)", last.Nodes, target+tolerance, target)
	}
}

func TestSearchReportsInfo(t *testing.T) {
	b := board.StartingBoard()
	var infos []Info
	move, ok := Search(b, Options{
		MaxTime:      150 * time.Millisecond,
		OnInfo:       func(i Info) { infos = append(infos, i) },
		InfoInterval: time.Nanosecond,
	})
	if !ok {
		t.Fatal("Search reported no legal moves at startpos")
	}
	if len(infos) == 0 {
		t.Fatal("OnInfo was never called")
	}

	last := infos[len(infos)-1]
	if last.Depth < 1 {
		t.Errorf("final Info.Depth = %d, want >= 1", last.Depth)
	}
	if len(last.PV) == 0 {
		t.Fatal("final Info.PV is empty")
	}
	if last.PV[0] != move {
		t.Errorf("final Info.PV[0] = %v, want the returned move %v", last.PV[0], move)
	}
	if last.SelDepth < last.Depth {
		t.Errorf("final Info.SelDepth = %d, want >= Info.Depth = %d", last.SelDepth, last.Depth)
	}
}

func TestSearchInfiniteRunsUntilContextCancelled(t *testing.T) {
	b := board.StartingBoard()
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(50*time.Millisecond, cancel)

	start := time.Now()
	move, ok := Search(b, Options{Infinite: true, Context: ctx})
	elapsed := time.Since(start)

	if !ok {
		t.Fatal("Search reported no legal moves at startpos")
	}
	if elapsed < 50*time.Millisecond {
		t.Errorf("Search returned after %v, before its 50ms cancellation", elapsed)
	}
	if elapsed > 2*time.Second {
		t.Errorf("Search took %v after cancellation, want prompt return", elapsed)
	}
	if legal := board.GenerateLegalMoves(&b); !moveIn(move, legal) {
		t.Errorf("Search returned %v, which is not among the legal moves %v", move, legal)
	}
}

func TestSearchContextCancellationStopsTimedSearch(t *testing.T) {
	b := board.StartingBoard()
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(30*time.Millisecond, cancel)

	start := time.Now()
	if _, ok := Search(b, Options{MaxTime: 10 * time.Second, Context: ctx}); !ok {
		t.Fatal("Search reported no legal moves at startpos")
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("Search took %v, want prompt return after Context cancellation", elapsed)
	}
}

func TestSearchSingleLegalMoveStillReportsInfo(t *testing.T) {
	b := mustFEN(t, "8/8/8/8/8/1k6/8/K7 w - - 0 1")
	var infos []Info
	move, ok := Search(b, Options{MaxTime: 50 * time.Millisecond, OnInfo: func(i Info) { infos = append(infos, i) }})
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	if len(infos) == 0 {
		t.Fatal("OnInfo was never called")
	}
	last := infos[len(infos)-1]
	if len(last.PV) == 0 || last.PV[0] != move {
		t.Errorf("final Info.PV = %v, want to start with %v", last.PV, move)
	}
}

func TestSearchHandlesFiftyMoveRuleDraw(t *testing.T) {
	// One ply from the fifty-move rule: whatever White plays here (other
	// than a capture or pawn move) immediately hits it. Score should
	// reflect a draw, not a false material evaluation.
	b := mustFEN(t, "8/8/8/4k3/8/4K3/8/7R w - - 99 60")
	var last Info
	_, ok := Search(b, Options{MaxTime: 150 * time.Millisecond, OnInfo: func(i Info) { last = i }})
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	if last.Mate != 0 {
		t.Errorf("Info.Mate = %d, want 0 (no forced mate before the fifty-move draw)", last.Mate)
	}
}

func TestSearchMultiThreadedReturnsALegalMove(t *testing.T) {
	b := board.StartingBoard()
	move, ok := Search(b, Options{MaxTime: 200 * time.Millisecond, Threads: 4})
	if !ok {
		t.Fatal("Search reported no legal moves at startpos")
	}
	if legal := board.GenerateLegalMoves(&b); !moveIn(move, legal) {
		t.Errorf("Search returned %v, which is not among the legal moves %v", move, legal)
	}
}

func TestSearchMultiThreadedFindsAFreeQueenCapture(t *testing.T) {
	b := mustFEN(t, "4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1")
	move, ok := Search(b, Options{MaxTime: 200 * time.Millisecond, Threads: 4})
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	from, _ := board.ParseSquare("c3")
	to, _ := board.ParseSquare("d5")
	if move.From != from || move.To != to {
		t.Errorf("Search picked %v, want the free queen capture c3d5", move)
	}
}

// NOTE: this test is currently vacuous — Options.Threads has no effect
// while the search is single-threaded (see TestThreadsHaveNoEffectYet), so
// it passes trivially. It is kept because the invariant it checks becomes
// real again the moment the search splits work across goroutines, and it is
// far easier to keep a guard passing than to remember to write one.
func TestSearchMultiThreadedNodesAreAggregateNotPerThread(t *testing.T) {
	const target = 5000
	const threads = 4
	// Each thread independently checks the shared counter only every
	// nodeCheckInterval+1 of its own nodes, so in the worst-case
	// interleaving all of them can overshoot by that much before any
	// observes the others' progress — bounded by thread count, not
	// unbounded.
	const tolerance = threads * (nodeCheckInterval + 1)

	b := board.StartingBoard()
	var last Info
	_, ok := Search(b, Options{
		MaxIterations: target,
		Threads:       threads,
		OnInfo:        func(i Info) { last = i },
	})
	if !ok {
		t.Fatal("Search reported no legal moves at startpos")
	}
	if last.Nodes > target+tolerance {
		t.Errorf("final Info.Nodes = %d, want <= %d (shared budget plus checking granularity)", last.Nodes, target+tolerance)
	}
	// The budget is shared across threads (one atomic counter), not
	// per-thread — confirm it's nowhere near what 4x the work would look
	// like, which is the actual property this test protects.
	if last.Nodes >= target*threads {
		t.Errorf("final Info.Nodes = %d, suspiciously close to target*Threads (%d) — looks like a per-thread budget, not shared", last.Nodes, target*threads)
	}
}

// NOTE: this test is currently vacuous — Options.Threads has no effect
// while the search is single-threaded (see TestThreadsHaveNoEffectYet), so
// it passes trivially. It is kept because the invariant it checks becomes
// real again the moment the search splits work across goroutines, and it is
// far easier to keep a guard passing than to remember to write one.
func TestSearchMultiThreadedOnlyOneThreadReportsInfo(t *testing.T) {
	b := board.StartingBoard()
	var calls int32
	_, ok := Search(b, Options{
		MaxTime:      150 * time.Millisecond,
		Threads:      4,
		InfoInterval: time.Nanosecond,
		OnInfo:       func(Info) { atomic.AddInt32(&calls, 1) },
	})
	if !ok {
		t.Fatal("Search reported no legal moves at startpos")
	}
	// A loose upper bound: periodic reporting should come from roughly
	// one thread's depth progression, not scale up with thread count.
	if calls > 200 {
		t.Errorf("OnInfo called %d times across 4 threads, want roughly one thread's worth", calls)
	}
}
