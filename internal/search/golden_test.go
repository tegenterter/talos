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
	// A forced mate, which the other five deliberately are not. (The FEN is
	// mdp_test.go's, named for the mate it guarantees at worst; the search
	// finds a faster one, which is why the baseline records mate 2.) Without
	// it the baselines cannot see mate distance pruning at all — a hole
	// TestGoldenDepthExercisesEveryHeuristic found the moment it was
	// written, and the same class of hole that hid singular extensions.
	{"mate-in-three", mateInThreeFEN},
}

// goldenDepth has to be deep enough that every heuristic in the search
// actually fires at it, or the baselines cannot see changes to the ones that
// do not. That is not hypothetical: at depth 6 singular extensions were
// invisible here — they start at singularMinDepth, which is 7 — so the
// regression net had a hole exactly where the newest code lived.
// TestGoldenDepthExercisesEveryHeuristic keeps it honest.
const goldenDepth = 8

// runGolden searches one position under fully-pinned, deterministic
// settings and returns the observable result.
func runGolden(t *testing.T, fen string) (nodes, depth, scoreCP, mate int, pv string) {
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
	// Mate is reported alongside ScoreCP, not folded into it: a forced mate
	// carries its distance in Info.Mate and leaves ScoreCP at zero, so a
	// baseline pinning only ScoreCP would not notice a mate turning into a
	// quiet zero.
	return last.Nodes, last.Depth, last.ScoreCP, last.Mate, strings.Join(moves, " ")
}

// TestSearchIsDeterministic is the precondition for every other golden test:
// without it, the recorded baselines below would be noise.
func TestSearchIsDeterministic(t *testing.T) {
	for _, p := range goldenPositions {
		n1, d1, s1, m1, pv1 := runGolden(t, p.fen)
		n2, d2, s2, m2, pv2 := runGolden(t, p.fen)
		if n1 != n2 || d1 != d2 || s1 != s2 || m1 != m2 || pv1 != pv2 {
			t.Errorf("%s: search is not deterministic across runs:\n  run 1: nodes=%d depth=%d score=%d pv=%q\n  run 2: nodes=%d depth=%d score=%d pv=%q",
				p.name, n1, d1, s1, pv1, n2, d2, s2, pv2)
		}
	}
}

// TestSearchMatchesGoldenBaselines pins the search's exact output. See the
// file comment for what to do when it fails.
func TestSearchMatchesGoldenBaselines(t *testing.T) {
	// Recorded at goldenDepth, Threads: 1. Last re-recorded when goldenDepth
	// went from 6 to 8, because at 6 these baselines could not see singular
	// extensions at all — they start at depth 7, so the regression net had a
	// hole exactly where the newest code lived. See
	// TestGoldenDepthExercisesEveryHeuristic, which now fails if any
	// heuristic becomes invisible here again.
	//
	// Before that, re-recorded when capture
	// history, internal iterative reduction and transposition-table aging
	// landed (ordering.go, negamax.go, tt.go). Scores and root moves are
	// unchanged everywhere; bench node count fell 19%. Kiwipete's count rose
	// here while bench's fell, which is what a reordering does — it is not a
	// uniform win position by position.
	//
	// Also re-recorded *without* the transposition-table bucketing that was
	// tried in the same sitting and reverted: four-entry buckets with
	// age-weighted replacement cost 18% more bench nodes than the existing
	// single-entry, depth-preferred policy, and every bucket size tried
	// (2, 3, 4, 8) was worse. Aging survived on its own, where it now means
	// "a deeper entry from an *earlier* search no longer outranks a fresh
	// one for its slot" — the gap CLAUDE.md had recorded.
	//
	// Before that, re-recorded when continuation
	// history landed (conthistory.go) and both history tables moved onto one
	// scale with a shared bonus and gravity update (ordering.go). Scores are
	// unchanged everywhere; startpos needs 32% fewer nodes for the same
	// answer, which is what better move ordering looks like.
	//
	// Before that, re-recorded when late move
	// reductions became a log table and the history heuristic gained its
	// malus, and — in the same change — when reverse futility, late move and
	// futility pruning were **removed again** after being measured. That
	// negative result is worth keeping:
	//
	//   - The three of them cut bench nodes 87% and bought two full plies of
	//     depth at a fixed time control.
	//   - They scored 0.504 over 354 self-play games: nothing.
	//   - Asked how often each build agrees with an independent 2s reference
	//     search over 40 real game positions, the pruning builds landed at
	//     15-23 of 40 against 26 with no shallow pruning at all.
	//
	// So the depth was bought by going blind, in almost exactly the
	// proportion that cancelled it. The likeliest reason is specific to this
	// engine rather than to the heuristics: pruning "late" moves is only
	// safe when late means unpromising, and this move ordering has no
	// continuation or capture history yet. Ordering first, then pruning.
	//
	// Before that, re-recorded when quiescence
	// began probing and storing in the transposition table, and entries
	// started carrying a static evaluation and the clock it applies at
	// (tt.go, quiescence.go). Every position's node count fell — 4-8% on the
	// four with material on the board — while every score stayed identical,
	// which is the shape a pure "reuse work already done" change should
	// have. Three PVs came back shorter for the reason this file's older
	// note already records: PV reconstruction truncates at a transposition
	// -table cutoff, so PV length tracks where TT hits land rather than the
	// true line length, and there are now more hits.
	//
	// Before that, re-recorded when the
	// fifty-move clock started damping the static evaluation and lopsided
	// endgames started using internal/eval instead of the network (see
	// evaluate.go). Both were expected to move these numbers, and both
	// moved them in the direction they were meant to:
	//
	//   - The four positions with material still on the board kept their
	//     scores exactly and shifted node counts by a few percent. Damping
	//     scales every leaf, and a leaf's clock is small this close to the
	//     root, so the effect is a small reordering rather than a
	//     re-evaluation.
	//   - rook-endgame (KR vs K) changed completely, which is the point:
	//     the network scored a bare rook ending at +1606, nearly three
	//     rooks' worth of confidence in a position worth one, and its PV
	//     shuffled the king. The classical evaluator scores it +630 —
	//     material plus a mating net — and the new PV starts a1a7, cutting
	//     the defending king off on the back rank, which is how the ending
	//     is actually won. Node count more than halved because a gradient
	//     the search can follow prunes far better than a plateau.
	//
	// Before that, re-recorded when promotions
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
		nodes, depth, scoreCP, mate int
		pv                          string
	}{
		"startpos":      {95239, 8, 49, 0, "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6"},
		"kiwipete":      {331436, 8, -156, 0, "e2a6 b4c3 d2c3"},
		"italian":       {128617, 8, -36, 0, "g8f6 d2d3 f8c5 e1g1 d7d6 c2c3 c5b6 b2b4"},
		"pawn-endgame":  {97972, 8, 280, 0, "b4f4 h4g3 f4f3 g3g2 a5a6 h5c5 f3f8 g2g3 f8f7 g3g4 f7g7 g4h4"},
		"rook-endgame":  {28551, 8, 633, 0, "a1a7 e8d8 e1e2 d8e8 e2d3 e8d8 d3e4 d8e8"},
		"mate-in-three": {21244, 8, 0, 2, "d1d7 g7g6 b2g7"},
	}

	for _, p := range goldenPositions {
		want, ok := golden[p.name]
		if !ok {
			t.Fatalf("no golden recorded for %s", p.name)
		}
		nodes, depth, scoreCP, mate, pv := runGolden(t, p.fen)
		if nodes != want.nodes || depth != want.depth || scoreCP != want.scoreCP || mate != want.mate || pv != want.pv {
			t.Errorf("%s changed:\n  got:  nodes=%d depth=%d score=%d mate=%d pv=%q\n  want: nodes=%d depth=%d score=%d mate=%d pv=%q",
				p.name, nodes, depth, scoreCP, mate, pv, want.nodes, want.depth, want.scoreCP, want.mate, want.pv)
		}
	}
}

// TestGoldenDepthExercisesEveryHeuristic checks that the baselines above can
// actually see each part of the search: flipping any one heuristic off must
// change what they record.
//
// This exists because of a hole it would have caught. Singular extensions
// start at depth 7 and the golden positions were pinned at depth 6, so they
// were completely blind to the newest and most intricate code in the search
// — a refactor could have broken it outright and every test would still have
// passed. A regression anchor that cannot see a feature is not protecting
// it, and the only way to know is to check.
//
// One position is enough per heuristic, and the first one that reacts ends
// the search for that heuristic, so this stays cheap despite running at
// goldenDepth.
func TestGoldenDepthExercisesEveryHeuristic(t *testing.T) {
	if testing.Short() {
		t.Skip("runs several searches at goldenDepth")
	}

	heuristics := []struct {
		name string
		flag *bool
	}{
		{"late move reductions", &lmrEnabled},
		{"principal variation search", &pvsEnabled},
		{"mate distance pruning", &mdpEnabled},
		{"quiescence delta pruning", &deltaPruningEnabled},
		{"aspiration windows", &aspirationEnabled},
		{"internal iterative reduction", &iirEnabled},
		{"singular extensions", &singularEnabled},
	}

	baseline := make(map[string][4]interface{}, len(goldenPositions))
	for _, p := range goldenPositions {
		nodes, _, score, mate, pv := runGolden(t, p.fen)
		baseline[p.name] = [4]interface{}{nodes, score, mate, pv}
	}

	for _, h := range heuristics {
		t.Run(h.name, func(t *testing.T) {
			*h.flag = false
			defer func() { *h.flag = true }()

			for _, p := range goldenPositions {
				nodes, _, score, mate, pv := runGolden(t, p.fen)
				if [4]interface{}{nodes, score, mate, pv} != baseline[p.name] {
					return // this position reacts; the baselines can see it
				}
			}
			t.Errorf("turning off %s changed none of the golden positions at depth %d: "+
				"the baselines are blind to it, and a regression in it would go unnoticed",
				h.name, goldenDepth)
		})
	}
}
