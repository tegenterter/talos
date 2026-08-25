package search

import (
	"strings"
	"testing"

	"talos/internal/board"
)

// Golden-baseline tests: exact-output regression anchors for the search.
//
// This package has no strength-measurement infrastructure (no self-play, no
// SPRT, no Elo estimate), which means a refactor that quietly makes the
// engine weaker cannot be detected by playing games. What CAN be detected is
// a refactor that changes the search's output at all when it was supposed to
// be behavior-preserving. That's what these tests are for: a pure
// restructuring (extracting shared state, renaming, re-plumbing threads)
// must leave Nodes, Depth, ScoreCP and the PV byte-for-byte identical, and
// any change here is either a bug or a deliberate behavior change that needs
// the goldens re-recorded on purpose.
//
// They depend on the search being deterministic (it contains no randomness
// at all — see ordering.go's orderMoves) and on Threads: 1, so no scheduling
// nondeterminism enters.
// A change that legitimately alters search behavior (e.g. adding PVS) is
// expected to fail these; re-record them in the same commit, and say so.

// goldenPositions spans an opening, a dense tactical middlegame, a quiet
// positional middlegame, a pawn endgame, and a near-bare endgame — enough
// shape variety that a subtle change touches at least one of them.
var goldenPositions = []struct {
	name string
	fen  string
}{
	{"startpos", board.StartFEN},
	{"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"},
	{"italian", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1"},
	{"pawn-endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"},
	{"rook-endgame", "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"},
}

const goldenDepth = 6

// runGolden searches one position under fully-pinned, deterministic
// settings and returns the observable result.
func runGolden(t *testing.T, fen string) (nodes, depth, scoreCP int, pv string) {
	t.Helper()
	b := mustFEN(t, fen)

	var last Info
	move, ok := Search(b, Options{
		MaxDepth: goldenDepth,
		Threads:  1,
		OnInfo:   func(i Info) { last = i },
	})
	if !ok {
		t.Fatalf("Search(%q) found no legal move", fen)
	}
	if len(last.PV) == 0 || last.PV[0] != move {
		t.Fatalf("Search(%q) returned %v but final info PV starts %v", fen, move, last.PV)
	}

	moves := make([]string, len(last.PV))
	for i, m := range last.PV {
		moves[i] = m.String()
	}
	return last.Nodes, last.Depth, last.ScoreCP, strings.Join(moves, " ")
}

// TestSearchIsDeterministic is the precondition for every other golden test:
// without it, the recorded baselines below would be noise.
func TestSearchIsDeterministic(t *testing.T) {
	for _, p := range goldenPositions {
		n1, d1, s1, pv1 := runGolden(t, p.fen)
		n2, d2, s2, pv2 := runGolden(t, p.fen)
		if n1 != n2 || d1 != d2 || s1 != s2 || pv1 != pv2 {
			t.Errorf("%s: search is not deterministic across runs:\n  run 1: nodes=%d depth=%d score=%d pv=%q\n  run 2: nodes=%d depth=%d score=%d pv=%q",
				p.name, n1, d1, s1, pv1, n2, d2, s2, pv2)
		}
	}
}

// TestSearchMatchesGoldenBaselines pins the search's exact output. See the
// file comment for what to do when it fails.
func TestSearchMatchesGoldenBaselines(t *testing.T) {
	// Recorded at goldenDepth, Threads: 1. Last re-recorded when promotions
	// got their own move-ordering band and history gained a hard clamp
	// (ordering.go): only kiwipete moved, and only its node count and PV
	// tail — its score and root move are unchanged. Kiwipete has a black
	// pawn on h3 that can promote within depth 6, so it is the one golden
	// position the promotion band actually reorders. The PV came back one
	// move shorter because PV reconstruction truncates at a transposition
	// -table cutoff (negamax.go returns a single-move PV there), so its
	// length tracks where TT hits land rather than the true line length;
	// reordering moved one such hit. Before that, re-recorded when quiescence
	// delta pruning's margin became material-phase-scaled (deltaPruningMarginMin/
	// Max in quiescence.go) instead of a flat 800: node counts dropped in
	// the three dense-middlegame positions (startpos/kiwipete/italian, all
	// scores unchanged) since they now get a margin close to
	// deltaPruningMarginMin=200 instead of the old flat 800, recovering
	// most of delta pruning's efficiency benefit where it's safe.
	// pawn-endgame's score moved from 660 to 704 — its low material phase
	// keeps its effective margin close to deltaPruningMarginMax=800
	// regardless of Min (verified: identical at every Min from 100-250 in
	// a calibration sweep), and 704 is the same value flat margin=700
	// produced during the original investigation, i.e. delta pruning's own
	// ordinary effect, not the aspiration-interaction corruption that
	// motivated raising the margin in the first place (which produced
	// scores like 1269/1464/1306 — see deltaPruningMarginMin/Max's doc
	// comment). rook-endgame was unaffected. Before that, re-recorded when
	// aspiration windows landed (search.go): they change node counts/PV
	// tails at this depth (aspiration only activates at depth >=
	// aspirationMinDepth), and required the delta-pruning margin to be
	// raised from 200 to a flat 800 as an interim fix — see quiescence.go's
	// doc comment for the full story now that it's phase-scaled instead.
	// Before that, re-recorded when quiescence delta pruning landed: it changed the
	// deeper quiescence continuation (and, on pawn-endgame, the reported
	// score/PV tail) on positions with prunable near-margin captures,
	// though the actual root move chosen stayed the same everywhere.
	// Before that, re-recorded when mate distance pruning landed, which
	// only lowered node counts (no score/PV changed) since none of these
	// positions are near a mate score at this depth. Before that,
	// re-recorded when PVS landed, which raised counts at this depth on
	// purpose — see TestPVSNodeEffectScalesWithDepth for why depth 6
	// flatters PVS least.
	golden := map[string]struct {
		nodes, depth, scoreCP int
		pv                    string
	}{
		"startpos":     {20110, 6, 45, "e2e4 c7c5 g1f3 b8c6 d2d4"},
		"kiwipete":     {68455, 6, -196, "e2a6 b4c3 d2c3 e6d5 e4d5"},
		"italian":      {29522, 6, -58, "g8f6 d2d4 e5d4 e1g1 f8c5 e4e5"},
		"pawn-endgame": {9200, 6, 704, "b4f4 h4g3 f4c4 h5c5"},
		"rook-endgame": {21500, 6, 1606, "e1e2 e8d7 a1a3 d7d6 a3g3"},
	}

	for _, p := range goldenPositions {
		want, ok := golden[p.name]
		if !ok {
			t.Fatalf("no golden recorded for %s", p.name)
		}
		nodes, depth, scoreCP, pv := runGolden(t, p.fen)
		if nodes != want.nodes || depth != want.depth || scoreCP != want.scoreCP || pv != want.pv {
			t.Errorf("%s changed:\n  got:  nodes=%d depth=%d score=%d pv=%q\n  want: nodes=%d depth=%d score=%d pv=%q",
				p.name, nodes, depth, scoreCP, pv, want.nodes, want.depth, want.scoreCP, want.pv)
		}
	}
}
