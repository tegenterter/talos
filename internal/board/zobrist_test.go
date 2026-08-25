package board

import "testing"

func playMove(t *testing.T, b Board, uci string) Board {
	t.Helper()
	m, ok := ParseUCIMove(uci)
	if !ok {
		t.Fatalf("bad move %q", uci)
	}
	for _, legal := range GenerateLegalMoves(&b) {
		if legal.From == m.From && legal.To == m.To && legal.Promotion == m.Promotion {
			return MakeMove(b, legal)
		}
	}
	t.Fatalf("move %q not legal in position", uci)
	return Board{}
}

func TestHashDetectsTranspositionsIgnoringMoveClocks(t *testing.T) {
	// e2e4 and g1f3 are White's two moves here and are independent
	// (commute); a black move must separate them (can't play two White
	// moves in a row), and the final move in both sequences is the same
	// non-pawn move so neither ends with an active en passant target that
	// the other lacks (which g8f6...e2e4-last would, since a pawn double
	// push sets one and a knight move doesn't).
	seq1 := StartingBoard()
	for _, mv := range []string{"e2e4", "g8f6", "g1f3", "b8c6"} {
		seq1 = playMove(t, seq1, mv)
	}
	seq2 := StartingBoard()
	for _, mv := range []string{"g1f3", "g8f6", "e2e4", "b8c6"} {
		seq2 = playMove(t, seq2, mv)
	}

	if seq1.HalfmoveClock == seq2.HalfmoveClock {
		t.Fatal("test setup: expected halfmove clocks to differ between move orders")
	}
	if seq1.Hash() != seq2.Hash() {
		t.Errorf("Hash() differs for transposed (identical) positions: %d vs %d", seq1.Hash(), seq2.Hash())
	}
}

func TestHashDiffersForDifferentPositions(t *testing.T) {
	a := StartingBoard()
	b := playMove(t, StartingBoard(), "e2e4")
	if a.Hash() == b.Hash() {
		t.Error("Hash() collided for genuinely different positions")
	}
}

func TestHashDiffersBySideToMoveAndCastlingRights(t *testing.T) {
	white, err := ParseFEN("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	black, err := ParseFEN("4k3/8/8/8/8/8/8/4K3 b - - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	if white.Hash() == black.Hash() {
		t.Error("Hash() ignored side to move")
	}

	withRights, err := ParseFEN("r3k3/8/8/8/8/8/8/4K2R w KQ - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	withoutRights, err := ParseFEN("r3k3/8/8/8/8/8/8/4K2R w - - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	if withRights.Hash() == withoutRights.Hash() {
		t.Error("Hash() ignored castling rights")
	}
}
