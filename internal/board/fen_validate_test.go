package board

import "testing"

// TestParseFENRejectsIllegalPositions covers the positions that used to
// reach the evaluator and crash it. A board with no king sends
// nnue.Evaluate indexing past the end of its feature table; a board where
// the waiting side is in check lets the side to move capture that king
// (GenerateLegalMoves only checks the *mover's* king), producing a
// kingless board one ply later.
func TestParseFENRejectsIllegalPositions(t *testing.T) {
	tests := []struct {
		name string
		fen  string
	}{
		{"no black king", "8/8/8/8/8/8/8/R3K3 w - - 0 1"},
		{"no white king", "4k3/8/8/8/8/8/8/r7 w - - 0 1"},
		{"two white kings", "4k3/8/8/8/8/8/8/K3K3 w - - 0 1"},
		{"waiting side in check", "8/8/8/3k4/1b6/2r5/8/K6B w - - 0 1"},
		{"waiting side in check by rook", "4k3/8/8/8/8/8/8/r3K3 b - - 0 1"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ParseFEN(tc.fen); err == nil {
				t.Errorf("ParseFEN(%q) succeeded, want an error", tc.fen)
			}
		})
	}
}

// TestParseFENAcceptsLegalPositions guards the other direction: the
// validation must not reject ordinary positions, including one where the
// side *to* move is in check, which is perfectly legal.
func TestParseFENAcceptsLegalPositions(t *testing.T) {
	for _, fen := range []string{
		StartFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"8/k7/2B5/b7/8/6r1/4K3/8 b - - 1 128",
		"4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
		"R3k3/8/4K3/8/8/8/8/8 b - - 0 1", // black to move, in check: legal
	} {
		if _, err := ParseFEN(fen); err != nil {
			t.Errorf("ParseFEN(%q) = %v, want success", fen, err)
		}
	}
}
