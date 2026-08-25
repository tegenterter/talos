package search

import (
	"testing"

	"talos/internal/board"
)

// TestPVSDoesNotChangeKnownBestMoves is the correctness guard on Principal
// Variation Search: narrowing the window and re-searching is only supposed
// to change how much work a result costs, never which move comes back on a
// position with one clearly correct answer. Mirrors
// TestLMRDoesNotChangeKnownBestMoves.
//
// Note this deliberately uses positions with an unambiguous best move. PVS
// legitimately *can* pick a different move among near-equal ones at a fixed
// depth — it prunes on bounds rather than exact scores, so a move it proved
// merely "not better than alpha" never gets an exact score to tie-break on.
func TestPVSDoesNotChangeKnownBestMoves(t *testing.T) {
	positions := []string{
		mateInOneFEN,
		"4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1", // free queen capture
	}

	for _, fen := range positions {
		t.Run(fen, func(t *testing.T) {
			b := mustFEN(t, fen)

			find := func(enabled bool) board.Move {
				pvsEnabled = enabled
				defer func() { pvsEnabled = true }()
				move, ok := Search(b, Options{MaxDepth: 6, Threads: 1})
				if !ok {
					t.Fatal("Search reported no legal moves")
				}
				return move
			}

			if with, without := find(true), find(false); with != without {
				t.Errorf("best move with PVS = %v, without PVS = %v, want equal", with, without)
			}
		})
	}
}

// TestPVSNodeEffectScalesWithDepth records what PVS actually costs and
// saves here, because the honest answer is "it depends on depth" and that
// is worth pinning down rather than assuming.
//
// PVS searches every move after the first on a null window, which can only
// prove "> alpha" or "<= alpha". Proving "<= alpha" is cheap and discards
// the move; proving "> alpha" means paying for a full re-search on top of
// the scout that was supposed to save work. So PVS wins exactly when the
// first move is usually best — i.e. when move ordering is good — and loses
// when it isn't.
//
// Measured on this engine, totals across the golden positions:
//
//	depth 6:  +7.8%  (worse)
//	depth 7:  +2.8%  (worse)
//	depth 8:  -0.7%  (about even)
//	depth 9:  -8.1%  (better)
//
// The crossover sits near depth 8. That matters for reading the golden
// baselines, which are recorded at depth 6 and therefore capture PVS at its
// *worst*: they got bigger when PVS landed, and that is expected, not a
// regression at the depths real games are played at. Two caveats worth
// keeping in mind: the per-position spread is wide (one endgame is +90% at
// depth 9, where many near-equal moves make scout searches fail high
// repeatedly), and the gain is smaller than textbook PVS because LMR
// already claims much of the same ground — both exist to spend less on
// moves ordered late.
//
// PVS is kept despite the shallow-depth cost because tree-splitting depends
// on it structurally: parallel siblings searched on a null window barely
// care that their alpha is stale, whereas full-width siblings would each
// pay for a full re-search.
func TestPVSNodeEffectScalesWithDepth(t *testing.T) {
	if testing.Short() {
		t.Skip("depth sweep is slow")
	}

	nodesAt := func(fen string, depth int, enabled bool) int {
		pvsEnabled = enabled
		defer func() { pvsEnabled = true }()
		b := mustFEN(t, fen)
		var last Info
		Search(b, Options{MaxDepth: depth, Threads: 1, OnInfo: func(i Info) { last = i }})
		return last.Nodes
	}

	totals := map[int][2]int{} // depth -> {off, on}
	for _, depth := range []int{6, 8} {
		var on, off int
		for _, p := range goldenPositions {
			off += nodesAt(p.fen, depth, false)
			on += nodesAt(p.fen, depth, true)
		}
		totals[depth] = [2]int{off, on}
		t.Logf("depth %d: PVS off = %d nodes, on = %d nodes (%+.1f%%)",
			depth, off, on, 100*float64(on-off)/float64(off))
	}

	// Assert only the direction that matters and is stable: by depth 8 PVS
	// must not be meaningfully more expensive. Deliberately loose — exact
	// counts belong in the golden baselines, not here.
	off, on := totals[8][0], totals[8][1]
	if float64(on) > 1.05*float64(off) {
		t.Errorf("at depth 8 PVS cost %d nodes vs %d without (%+.1f%%); expected roughly break-even or better",
			on, off, 100*float64(on-off)/float64(off))
	}
}
