package nnue

import (
	"testing"

	"talos/internal/board"
)

// explainPositions covers an opening, a dense tactical middlegame, an
// endgame, and a position with a hanging piece — enough variety that a
// mistake in the ablation arithmetic shows up somewhere.
var explainPositions = []string{
	board.StartFEN,
	"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // Kiwipete
	"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
	"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
	"4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
}

// TestExplainTotalMatchesEvaluate pins the explainer to the evaluator: an
// explanation that reports a different score than Evaluate would be
// describing a network nobody is actually playing with.
func TestExplainTotalMatchesEvaluate(t *testing.T) {
	for _, fen := range explainPositions {
		b := mustFEN(t, fen)
		if got, want := Explain(&b).TotalCP, Evaluate(&b); got != want {
			t.Errorf("Explain(%q).TotalCP = %d, Evaluate = %d, want equal", fen, got, want)
		}
	}
}

// TestExplainCoversEveryNonKingPiece checks the piece enumeration doesn't
// silently skip anyone, and that kings are excluded (they aren't features —
// see Network.Explain).
func TestExplainCoversEveryNonKingPiece(t *testing.T) {
	for _, fen := range explainPositions {
		b := mustFEN(t, fen)
		want := b.OccupiedBB().Count() - 2 // both kings
		got := len(Explain(&b).Contributions)
		if got != want {
			t.Errorf("Explain(%q) returned %d contributions, want %d (all non-king pieces)", fen, got, want)
		}
		for _, c := range Explain(&b).Contributions {
			if c.Piece == board.King {
				t.Errorf("Explain(%q) attributed a contribution to a king on %s", fen, c.Square)
			}
			if color, pt, ok := b.PieceAt(c.Square); !ok || color != c.Color || pt != c.Piece {
				t.Errorf("Explain(%q) reported %v %v on %s, which isn't what's there", fen, c.Color, c.Piece, c.Square)
			}
		}
	}
}

// TestExplainAblationMatchesRebuiltBoard is the core correctness check: the
// fast path subtracts a piece's two weight rows from the accumulators, which
// must produce exactly what removing the piece from the board and calling
// Evaluate from scratch produces. This is what proves the incremental
// subtraction (and both feature indices behind it) is right.
func TestExplainAblationMatchesRebuiltBoard(t *testing.T) {
	for _, fen := range explainPositions {
		b := mustFEN(t, fen)
		total := Evaluate(&b)

		for _, c := range Explain(&b).Contributions {
			stripped := b
			stripped.Pieces[c.Color][c.Piece] &^= board.Bitboard(1) << c.Square
			// En passant is irrelevant to the feature set but keeping it
			// pointing at a now-vacated square would be nonsense; the
			// accumulator ignores it either way.
			want := total - Evaluate(&stripped)
			if c.DeltaCP != want {
				t.Errorf("Explain(%q): %v %v on %s has DeltaCP %d, but rebuilding the board without it gives %d",
					fen, c.Color, c.Piece, c.Square, c.DeltaCP, want)
			}
		}
	}
}

// TestExplainDecompositionAddsUp verifies the documented identity
// TotalCP = BaselineCP + sum(Contributions) + ResidualCP. This is arithmetic
// rather than a claim about the network, but it's what makes ResidualCP
// interpretable instead of a fudge factor.
func TestExplainDecompositionAddsUp(t *testing.T) {
	for _, fen := range explainPositions {
		b := mustFEN(t, fen)
		e := Explain(&b)
		sum := 0
		for _, c := range e.Contributions {
			sum += c.DeltaCP
		}
		if got := e.BaselineCP + sum + e.ResidualCP; got != e.TotalCP {
			t.Errorf("Explain(%q): baseline %d + contributions %d + residual %d = %d, want TotalCP %d",
				fen, e.BaselineCP, sum, e.ResidualCP, got, e.TotalCP)
		}
	}
}

// TestExplainSortedByMagnitude checks the ordering contract callers rely on
// to show "the pieces that matter most" first.
func TestExplainSortedByMagnitude(t *testing.T) {
	b := mustFEN(t, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
	cs := Explain(&b).Contributions
	for i := 1; i < len(cs); i++ {
		if abs(cs[i-1].DeltaCP) < abs(cs[i].DeltaCP) {
			t.Fatalf("contributions not sorted by |DeltaCP|: index %d (%d) before %d (%d)",
				i-1, cs[i-1].DeltaCP, i, cs[i].DeltaCP)
		}
	}
}

// TestExplainHangingQueenDominates is the sanity check that the numbers mean
// something in chess terms: with White a whole queen up and nothing else
// unbalanced, that queen should be the single largest contributor.
func TestExplainHangingQueenDominates(t *testing.T) {
	// Black is missing its queen; White to move (same position as
	// TestMissingQueenIsHeavilyPenalized).
	b := mustFEN(t, "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
	cs := Explain(&b).Contributions
	if len(cs) == 0 {
		t.Fatal("no contributions")
	}
	top := cs[0]
	if top.Piece != board.Queen || top.Color != board.White {
		t.Errorf("largest contributor is %v %v on %s, want the White queen", top.Color, top.Piece, top.Square)
	}
	if top.DeltaCP <= 0 {
		t.Errorf("White's extra queen has DeltaCP %d, want positive for the side to move", top.DeltaCP)
	}
}

// TestExplainPointSymmetry extends TestPointSymmetry's invariant to the
// explainer: under a 180-degree rotation with colors and side to move
// swapped, every piece's contribution must land on its rotated counterpart
// unchanged. This catches a feature-index mistake that happens to cancel out
// in the total but not per-piece.
func TestExplainPointSymmetry(t *testing.T) {
	for _, fen := range explainPositions {
		b := mustFEN(t, fen)
		mirrored := pointMirror(b)

		type key struct {
			color  board.Color
			piece  board.PieceType
			square board.Square
		}
		got := map[key]int{}
		for _, c := range Explain(&mirrored).Contributions {
			got[key{c.Color, c.Piece, c.Square}] = c.DeltaCP
		}

		for _, c := range Explain(&b).Contributions {
			k := key{c.Color.Opposite(), c.Piece, c.Square ^ 63}
			mirroredDelta, ok := got[k]
			if !ok {
				t.Errorf("Explain(%q): %v %v on %s has no counterpart in the mirrored position",
					fen, c.Color, c.Piece, c.Square)
				continue
			}
			if mirroredDelta != c.DeltaCP {
				t.Errorf("Explain(%q): %v %v on %s contributes %d, mirrored counterpart contributes %d, want equal",
					fen, c.Color, c.Piece, c.Square, c.DeltaCP, mirroredDelta)
			}
		}
	}
}
