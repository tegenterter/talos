package search

import (
	"testing"
	"time"

	"talos/internal/board"
)

// TestLMRReductionExemptions checks every condition that must force
// lmrReduction to 0 — each one tested in isolation (every other input
// left at values that would otherwise qualify for a reduction), so a
// regression in any single exemption shows up as its own failure rather
// than being masked by the others.
func TestLMRReductionExemptions(t *testing.T) {
	// A baseline where every input qualifies for a reduction, so any test
	// case below only needs to flip the one condition it's checking.
	const (
		qualifyingDepth = 6
		qualifyingIndex = 6
	)
	if got := lmrReduction(qualifyingDepth, qualifyingIndex, false, false, false, false, false, true, false); got == 0 {
		t.Fatal("sanity check failed: the qualifying baseline itself returned 0")
	}

	tests := []struct {
		name                                              string
		depth, index                                      int
		inCheck, givesCheck, capture, promotion, isKiller bool
	}{
		{"depth below lmrMinDepth", lmrMinDepth - 1, qualifyingIndex, false, false, false, false, false},
		{"index below lmrMinMoveIndex", qualifyingDepth, lmrMinMoveIndex - 1, false, false, false, false, false},
		{"side to move is in check", qualifyingDepth, qualifyingIndex, true, false, false, false, false},
		{"move gives check", qualifyingDepth, qualifyingIndex, false, true, false, false, false},
		{"move is a capture", qualifyingDepth, qualifyingIndex, false, false, true, false, false},
		{"move is a promotion", qualifyingDepth, qualifyingIndex, false, false, false, true, false},
		{"move is a killer", qualifyingDepth, qualifyingIndex, false, false, false, false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := lmrReduction(tt.depth, tt.index, tt.inCheck, tt.givesCheck, tt.capture, tt.promotion, tt.isKiller, true, false); got != 0 {
				t.Errorf("lmrReduction(depth=%d, index=%d, inCheck=%v, givesCheck=%v, capture=%v, promotion=%v, isKiller=%v) = %d, want 0",
					tt.depth, tt.index, tt.inCheck, tt.givesCheck, tt.capture, tt.promotion, tt.isKiller, got)
			}
		})
	}
}

// TestLMRReductionMagnitude pins down the actual reduction sizes for
// qualifying (non-exempt) moves at various depth/index combinations.
//
// The reduction is a log table now (lmrTable), not the flat 1-2 plies these
// cases used to pin, so what is checked here is the table's *shape*: a
// reduction that grows with both depth and move index, and that grows fast
// enough at depth to be worth having a table for at all.
func TestLMRReductionMagnitude(t *testing.T) {
	red := func(depth, index int) int {
		return lmrReduction(depth, index, false, false, false, false, false, true, false)
	}

	// Deeper and later must never reduce less than shallower and earlier.
	for depth := lmrMinDepth; depth <= 30; depth++ {
		for index := lmrMinMoveIndex; index <= 30; index++ {
			if r, deeper := red(depth, index), red(depth+1, index); deeper < r {
				t.Errorf("reduction fell with depth: (%d,%d)=%d but (%d,%d)=%d", depth, index, r, depth+1, index, deeper)
			}
			if r, later := red(depth, index), red(depth, index+1); later < r {
				t.Errorf("reduction fell with move index: (%d,%d)=%d but (%d,%d)=%d", depth, index, r, depth, index+1, later)
			}
		}
	}

	// And it must actually scale: the old flat table capped at 2 plies, so a
	// late move in a deep search was reduced no more than one in a shallow
	// search. That is the whole thing this replaced.
	if shallow, deep := red(lmrMinDepth, lmrMinMoveIndex), red(24, 24); deep <= shallow+1 {
		t.Errorf("reduction at depth 24 / index 24 is %d against %d at the threshold; the table is not scaling", deep, shallow)
	}
}

// TestLMRIsGentlerOnPVNodesAndHarsherWhenNotImproving pins the two
// adjustments: a principal-variation node's score is what everything else is
// measured against, so it is searched closer to full width, while a position
// going the wrong way gets its late quiet moves cut harder.
func TestLMRIsGentlerOnPVNodesAndHarsherWhenNotImproving(t *testing.T) {
	const depth, index = 12, 12
	base := lmrReduction(depth, index, false, false, false, false, false, true, false)
	pv := lmrReduction(depth, index, false, false, false, false, false, true, true)
	notImproving := lmrReduction(depth, index, false, false, false, false, false, false, false)

	if pv >= base {
		t.Errorf("PV node reduced by %d, non-PV by %d; want the PV node reduced less", pv, base)
	}
	if notImproving <= base {
		t.Errorf("not-improving reduced by %d, improving by %d; want not-improving reduced more", notImproving, base)
	}
}

// TestLMRReductionNeverNegativeOrExcessive is a broad sweep guarding the two
// properties that must hold whatever the table says: a reduction is never
// negative (that would *increase* depth, which is not what "reduction"
// means), and it never reduces a move into quiescence — the point of a
// reduced search is a cheaper verdict from the same kind of search, not a
// different one.
func TestLMRReductionNeverNegativeOrExcessive(t *testing.T) {
	for depth := 0; depth <= 40; depth++ {
		for index := 0; index <= 40; index++ {
			for _, improving := range []bool{true, false} {
				for _, pv := range []bool{true, false} {
					got := lmrReduction(depth, index, false, false, false, false, false, improving, pv)
					if got < 0 {
						t.Fatalf("lmrReduction(depth=%d, index=%d) = %d, want >= 0", depth, index, got)
					}
					if got > 0 && got >= depth {
						t.Fatalf("lmrReduction(depth=%d, index=%d) = %d, which would search the move at depth <= 0 and drop it into quiescence", depth, index, got)
					}
				}
			}
		}
	}
}

// TestLMRReducesNodeCount verifies LMR actually does something, not just
// that it's harmless: the same position, to the same fixed depth, should
// take measurably fewer nodes with reductions enabled than with them
// switched off via lmrEnabled (see negamax.go — a testing-only hook with
// no production path that ever sets it false). A fixed fair-ish middlegame
// position and Threads:1 (avoiding cross-thread move-ordering randomness)
// keep this about as deterministic as a real alpha-beta search gets;
// there's still some run-to-run variance from orderMoves' shuffle-based
// tie-breaking, but LMR's effect is large enough (a majority reduction in
// typical positions) that it should never be close to that noise floor.
func TestLMRReducesNodeCount(t *testing.T) {
	b := mustFEN(t, "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
	const fixedDepth = 7

	run := func(enabled bool) int {
		lmrEnabled = enabled
		defer func() { lmrEnabled = true }()

		var lastInfo Info
		_, ok := Search(b, Options{
			MaxDepth: fixedDepth,
			Threads:  1,
			OnInfo:   func(i Info) { lastInfo = i },
		})
		if !ok {
			t.Fatal("Search reported no legal moves")
		}
		return lastInfo.Nodes
	}

	withLMR := run(true)
	withoutLMR := run(false)

	t.Logf("nodes at depth %d: with LMR = %d, without LMR = %d", fixedDepth, withLMR, withoutLMR)
	if withLMR >= withoutLMR {
		t.Errorf("nodes with LMR (%d) was not less than without (%d) — LMR doesn't seem to be reducing anything", withLMR, withoutLMR)
	}
}

// TestLMRDoesNotChangeKnownBestMoves re-runs a couple of this package's
// existing hand-verified tactical positions at a fixed depth deep enough
// for LMR to actually engage (>= lmrMinDepth, with room for several late
// moves per node), and checks the best move found is identical whether
// LMR is enabled or not. This is a stronger check than "doesn't crash":
// it confirms the reduce-then-verify-on-fail-high safety net is actually
// doing its job — a reduction is only ever supposed to change *how much
// work* finding the best move takes, never *which move* is found.
func TestLMRDoesNotChangeKnownBestMoves(t *testing.T) {
	positions := []string{
		mateInOneFEN,
		"4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1", // free queen capture, see TestSearchFindsAFreeQueenCapture
	}

	for _, fen := range positions {
		t.Run(fen, func(t *testing.T) {
			b := mustFEN(t, fen)

			find := func(enabled bool) board.Move {
				lmrEnabled = enabled
				defer func() { lmrEnabled = true }()
				move, ok := Search(b, Options{MaxDepth: 6, Threads: 1})
				if !ok {
					t.Fatal("Search reported no legal moves")
				}
				return move
			}

			withLMR := find(true)
			withoutLMR := find(false)
			if withLMR != withoutLMR {
				t.Errorf("best move with LMR = %v, without LMR = %v, want equal", withLMR, withoutLMR)
			}
		})
	}
}

// TestLMRRespectsTimeControl is a light guard against LMR interacting
// badly with cancellation (e.g. a reduced search's extra re-search call
// forgetting to check t.aborted): search under a tight time budget must
// still return promptly with a legal move, the same guarantee
// TestSearchRespectsMaxTime already makes for the search as a whole.
func TestLMRRespectsTimeControl(t *testing.T) {
	b := board.StartingBoard()
	start := time.Now()
	_, ok := Search(b, Options{MaxTime: 100 * time.Millisecond})
	elapsed := time.Since(start)
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	if elapsed > time.Second {
		t.Errorf("Search took %s for a 100ms budget, want well under 1s", elapsed)
	}
}
