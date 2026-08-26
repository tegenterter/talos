package board

import (
	"math/rand"
	"testing"
)

// TestFENRoundTrip is the property that matters for a FEN writer: whatever
// ParseFEN accepts, FEN must render back to the identical string, or data
// written for something outside this program (training data, a bug report)
// describes a different position than the one the engine had.
func TestFENRoundTrip(t *testing.T) {
	for _, fen := range []string{
		StartFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
		"rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
		"8/k7/2B5/b7/8/6r1/4K3/8 b - - 17 128",
		"4k3/8/8/8/8/8/8/4K2R w K - 0 1",
	} {
		b, err := ParseFEN(fen)
		if err != nil {
			t.Fatalf("ParseFEN(%q): %v", fen, err)
		}
		if got := b.FEN(); got != fen {
			t.Errorf("round trip:\n got  %q\n want %q", got, fen)
		}
	}
}

// TestFENRoundTripOverPlayedGames covers what fixed cases cannot: positions
// the engine actually reaches, including every castling-right and en passant
// state a real game passes through.
func TestFENRoundTripOverPlayedGames(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for game := 0; game < 30; game++ {
		b := StartingBoard()
		for ply := 0; ply < 120; ply++ {
			moves := GenerateLegalMoves(&b)
			if len(moves) == 0 {
				break
			}
			b = MakeMove(b, moves[rng.Intn(len(moves))])

			fen := b.FEN()
			reparsed, err := ParseFEN(fen)
			if err != nil {
				t.Fatalf("game %d ply %d: FEN %q does not parse: %v", game, ply, fen, err)
			}
			if got := reparsed.FEN(); got != fen {
				t.Fatalf("game %d ply %d: round trip changed %q into %q", game, ply, fen, got)
			}
			if reparsed != b {
				t.Fatalf("game %d ply %d: round trip through %q produced a different board", game, ply, fen)
			}
		}
	}
}
