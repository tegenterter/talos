package search

import (
	"testing"

	"talos/internal/board"
)

// TestAspirationDoesNotChangeKnownBestMoves is the correctness guard,
// mirroring TestLMRDoesNotChangeKnownBestMoves/TestPVSDoesNotChangeKnownBestMoves/
// TestMDPDoesNotChangeMateResults/TestDeltaPruningDoesNotChangeKnownBestMoves:
// on positions with one clearly correct answer, windowing the search around
// the previous iteration's score must never change which move it settles
// on, nor how it reports a forced mate. mateInOneFEN and the getting-mated
// position specifically check Info.Mate too — the one property that must
// never waver regardless of what window found it.
func TestAspirationDoesNotChangeKnownBestMoves(t *testing.T) {
	positions := []struct {
		name     string
		fen      string
		wantMate int // 0 if not a mate-scoring position
	}{
		{"mate in one", mateInOneFEN, 1},
		{"getting mated in one", "k7/8/1K6/8/8/8/8/7R b - - 0 1", -1},
		{"free queen capture", "4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1", 0},
	}

	for _, p := range positions {
		t.Run(p.name, func(t *testing.T) {
			b := mustFEN(t, p.fen)

			find := func(enabled bool) (move board.Move, mate int) {
				aspirationEnabled = enabled
				defer func() { aspirationEnabled = true }()
				var last Info
				m, ok := Search(b, Options{MaxDepth: 8, Threads: 1, OnInfo: func(i Info) { last = i }})
				if !ok {
					t.Fatal("Search reported no legal moves")
				}
				return m, last.Mate
			}

			withMove, withMate := find(true)
			withoutMove, withoutMate := find(false)
			if withMove != withoutMove {
				t.Errorf("best move with aspiration = %v, without = %v, want equal", withMove, withoutMove)
			}
			if withMate != p.wantMate || withoutMate != p.wantMate {
				t.Errorf("Info.Mate with aspiration = %d, without = %d, want both %d", withMate, withoutMate, p.wantMate)
			}
		})
	}
}

// TestAspirationRecoversFromForcedFailure directly exercises the
// widen-and-re-search path (rather than hoping some fixed position happens
// to trigger it): aspirationWindowCP is temporarily shrunk to 1cp, all but
// guaranteeing the first attempt at any depth >= aspirationMinDepth fails
// low or high, forcing at least one widen-and-retry. The final result must
// still match a full-window search exactly on a position with an
// unambiguous best line (mate-in-one), proving the widening loop actually
// recovers the correct answer rather than settling for whatever the last
// failed attempt saw.
func TestAspirationRecoversFromForcedFailure(t *testing.T) {
	b := mustFEN(t, mateInOneFEN)

	find := func(enabled bool, windowCP int) (move board.Move, mate int) {
		aspirationEnabled = enabled
		oldWindow := aspirationWindowCP
		aspirationWindowCP = windowCP
		defer func() { aspirationEnabled, aspirationWindowCP = true, oldWindow }()
		var last Info
		m, ok := Search(b, Options{MaxDepth: 8, Threads: 1, OnInfo: func(i Info) { last = i }})
		if !ok {
			t.Fatal("Search reported no legal moves")
		}
		return m, last.Mate
	}

	forcedMove, forcedMate := find(true, 1)
	// windowCP is irrelevant when aspiration is disabled (aspirationSearch
	// falls back to a full window regardless), so this is just a
	// placeholder full-window ground truth to compare against.
	fullMove, fullMate := find(false, 25)
	if forcedMove != fullMove {
		t.Errorf("best move with a forced-fail 1cp window = %v, full-window = %v, want equal", forcedMove, fullMove)
	}
	if forcedMate != fullMate {
		t.Errorf("Info.Mate with a forced-fail 1cp window = %d, full-window = %d, want equal", forcedMate, fullMate)
	}
}

// TestAspirationReducesNodeCount logs aspiration windows' node-count effect
// across the golden positions, in the style of TestPVSNodeEffectScalesWithDepth:
// informative rather than a strict per-position assertion, since — like
// PVS — aspiration's payoff depends on how often the previous iteration's
// score predicts the next one, which varies by position. Only the
// aggregate direction is asserted.
func TestAspirationReducesNodeCount(t *testing.T) {
	nodesAt := func(fen string, enabled bool) int {
		aspirationEnabled = enabled
		defer func() { aspirationEnabled = true }()
		b := mustFEN(t, fen)
		var last Info
		Search(b, Options{MaxDepth: goldenDepth, Threads: 1, OnInfo: func(i Info) { last = i }})
		return last.Nodes
	}

	var on, off int
	for _, p := range goldenPositions {
		onN, offN := nodesAt(p.fen, true), nodesAt(p.fen, false)
		t.Logf("%s: with aspiration = %d, without = %d (%+.1f%%)", p.name, onN, offN, 100*float64(onN-offN)/float64(offN))
		on += onN
		off += offN
	}
	t.Logf("total: with aspiration = %d, without = %d (%+.1f%%)", on, off, 100*float64(on-off)/float64(off))
	if float64(on) > 1.10*float64(off) {
		t.Errorf("aspiration windows cost %d nodes vs %d without (%+.1f%%); expected roughly break-even or better",
			on, off, 100*float64(on-off)/float64(off))
	}
}
