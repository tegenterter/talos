package search

import (
	"strings"
	"testing"

	"talos/internal/board"
)

// TestDeltaPruningReducesNodeCount confirms quiescence delta pruning is
// actually doing something, not just harmlessly present. kiwipete is used
// because it's the golden position where delta pruning's node effect is
// largest (94942 -> 68749 nodes at goldenDepth with aspiration windows and
// mate distance pruning also active, a ~28% drop) — a dense tactical
// middlegame with many candidate captures for delta pruning to screen. Its
// material-phase-scaled margin (quiescence.go's deltaPruningMarginFor)
// lands close to deltaPruningMarginMin here, since a dense middlegame is
// exactly the regime that constant is tuned for.
// Mirrors TestLMRReducesNodeCount/TestMDPReducesNodeCount.
func TestDeltaPruningReducesNodeCount(t *testing.T) {
	b := mustFEN(t, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1") // kiwipete
	const fixedDepth = 6

	run := func(enabled bool) int {
		deltaPruningEnabled = enabled
		defer func() { deltaPruningEnabled = true }()

		var last Info
		_, ok := Search(b, Options{MaxDepth: fixedDepth, Threads: 1, OnInfo: func(i Info) { last = i }})
		if !ok {
			t.Fatal("Search reported no legal moves")
		}
		return last.Nodes
	}

	withPruning := run(true)
	withoutPruning := run(false)

	t.Logf("nodes at depth %d: with delta pruning = %d, without = %d", fixedDepth, withPruning, withoutPruning)
	if withPruning >= withoutPruning {
		t.Errorf("nodes with delta pruning (%d) was not less than without (%d) — delta pruning doesn't seem to be reducing anything", withPruning, withoutPruning)
	}
}

// TestDeltaPruningDoesNotChangeKnownBestMoves is the correctness guard,
// mirroring TestLMRDoesNotChangeKnownBestMoves/TestPVSDoesNotChangeKnownBestMoves/
// TestMDPDoesNotChangeMateResults: on positions with one clearly correct
// answer, pruning "not good enough" captures out of quiescence must never
// change which move the search settles on.
func TestDeltaPruningDoesNotChangeKnownBestMoves(t *testing.T) {
	positions := []string{
		mateInOneFEN,
		"4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1", // free queen capture
	}

	for _, fen := range positions {
		t.Run(fen, func(t *testing.T) {
			b := mustFEN(t, fen)

			find := func(enabled bool) board.Move {
				deltaPruningEnabled = enabled
				defer func() { deltaPruningEnabled = true }()
				move, ok := Search(b, Options{MaxDepth: 6, Threads: 1})
				if !ok {
					t.Fatal("Search reported no legal moves")
				}
				return move
			}

			if with, without := find(true), find(false); with != without {
				t.Errorf("best move with delta pruning = %v, without = %v, want equal", with, without)
			}
		})
	}
}

// TestDeltaPruningNearMarginStillResolvesCorrectly checks that pruning
// doesn't just leave the *chosen move* alone (the weaker guarantee
// TestDeltaPruningDoesNotChangeKnownBestMoves already covers) but leaves the
// search's actual conclusion — its exact score and full principal
// variation — unchanged on positions where pruning is known to be heavily
// engaged.
//
// A hand-built position calibrated to sit exactly at the margin boundary
// isn't really constructible by hand: the prune decision depends on
// nnue.Evaluate's stand-pat score (an opaque trained network) and on the
// alpha this particular node happens to inherit from the rest of the search
// tree, neither of which can be predicted analytically for an arbitrary
// FEN. So instead this uses empirical evidence of real engagement: kiwipete
// and italian are two golden positions (see golden_test.go) where pruning
// measurably reduces node count (kiwipete: ~28%, italian: ~5%, see
// TestDeltaPruningReducesNodeCount) while their recorded score and PV stay
// byte-identical to the unpruned baseline — i.e. real positions, not
// contrived ones, where pruning discards some of quiescence's candidate
// captures without changing the answer. This test pins that same guarantee
// (identical score and PV, not just root move) so a future change that
// makes pruning more aggressive can't silently start trading that
// correctness away for a bigger node-count win. It's also the reason
// deltaPruningMarginFor is phase-scaled rather than a single small flat
// value (see its doc comment): this exact assertion is what a too-small
// flat margin broke on a third golden position (pawn-endgame) during
// development.
func TestDeltaPruningNearMarginStillResolvesCorrectly(t *testing.T) {
	positions := []struct {
		name string
		fen  string
	}{
		{"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"},
		{"italian", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1"},
	}

	for _, p := range positions {
		t.Run(p.name, func(t *testing.T) {
			b := mustFEN(t, p.fen)

			run := func(enabled bool) (scoreCP int, pv string) {
				deltaPruningEnabled = enabled
				defer func() { deltaPruningEnabled = true }()
				var last Info
				_, ok := Search(b, Options{MaxDepth: goldenDepth, Threads: 1, OnInfo: func(i Info) { last = i }})
				if !ok {
					t.Fatal("Search reported no legal moves")
				}
				moves := make([]string, len(last.PV))
				for i, m := range last.PV {
					moves[i] = m.String()
				}
				return last.ScoreCP, strings.Join(moves, " ")
			}

			withScore, withPV := run(true)
			withoutScore, withoutPV := run(false)
			if withScore != withoutScore || withPV != withoutPV {
				t.Errorf("%s: pruning changed the search's conclusion:\n  with delta pruning:    score=%d pv=%q\n  without delta pruning: score=%d pv=%q",
					p.name, withScore, withPV, withoutScore, withoutPV)
			}
		})
	}
}
