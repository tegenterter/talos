package board

import "testing"

// perft counts the number of leaf positions reachable in exactly depth
// plies, and is the standard way to validate a move generator: any bug in
// pseudo-legal generation, check detection, castling, en passant, or
// promotion handling almost always desyncs the count from known-correct
// reference values.
func perft(b Board, depth int) uint64 {
	if depth == 0 {
		return 1
	}
	moves := GenerateLegalMoves(&b)
	if depth == 1 {
		return uint64(len(moves))
	}
	var nodes uint64
	for _, m := range moves {
		nodes += perft(MakeMove(b, m), depth-1)
	}
	return nodes
}

func TestPerftStartPos(t *testing.T) {
	want := []uint64{20, 400, 8902, 197281}
	b := StartingBoard()
	for i, w := range want {
		depth := i + 1
		if got := perft(b, depth); got != w {
			t.Errorf("perft(startpos, %d) = %d, want %d", depth, got, w)
		}
	}
}

// Kiwipete: a well-known perft test position exercising castling (both
// sides, both directions), en passant, and pinned pieces.
func TestPerftKiwipete(t *testing.T) {
	b, err := ParseFEN("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
	if err != nil {
		t.Fatalf("ParseFEN: %v", err)
	}
	want := []uint64{48, 2039, 97862}
	for i, w := range want {
		depth := i + 1
		if got := perft(b, depth); got != w {
			t.Errorf("perft(kiwipete, %d) = %d, want %d", depth, got, w)
		}
	}
}

func TestPawnPromotionGeneratesAllPieces(t *testing.T) {
	b, err := ParseFEN("8/P7/8/8/8/8/8/k6K w - - 0 1")
	if err != nil {
		t.Fatalf("ParseFEN: %v", err)
	}
	moves := GenerateLegalMoves(&b)
	got := map[PieceType]bool{}
	for _, m := range moves {
		if m.From.String() == "a7" && m.To.String() == "a8" {
			got[m.Promotion] = true
		}
	}
	for _, p := range [4]PieceType{Queen, Rook, Bishop, Knight} {
		if !got[p] {
			t.Errorf("missing promotion to piece type %d", p)
		}
	}
}
