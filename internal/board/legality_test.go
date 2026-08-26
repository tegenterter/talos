package board

import (
	"math/rand"
	"sort"
	"testing"
)

// TestPerftSuite runs the standard perft positions, which exist precisely
// because each one breaks a plausible-looking move generator in a different
// way. Position 3 is the one that matters most for legality.go: it is the
// classic en passant discovered check, where capturing en passant takes two
// pawns off the same rank at once and exposes the king to a rook along it.
//
// The existing startpos/Kiwipete tests (board_test.go) stop at depths 4 and
// 3; a legality filter that decides moves from pins and checkers rather than
// by playing them deserves more than that.
func TestPerftSuite(t *testing.T) {
	tests := []struct {
		name  string
		fen   string
		depth int
		want  uint64
	}{
		{"startpos", StartFEN, 5, 4865609},
		{"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
		{"position 3 (en passant discovered check)", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},
		{"position 4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},
		{"position 5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487},
		{"position 6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b, err := ParseFEN(tc.fen)
			if err != nil {
				t.Fatalf("ParseFEN: %v", err)
			}
			if got := perft(b, tc.depth); got != tc.want {
				t.Errorf("perft(%q, %d) = %d, want %d", tc.fen, tc.depth, got, tc.want)
			}
		})
	}
}

// TestLegalMovesMatchMakeMoveReference is the differential test: the fast
// filter must agree with the obviously-correct one that plays every move,
// move for move, on positions reached by actually playing games. Perft
// proves the counts add up; this proves the two implementations agree on
// *which* moves, which is what a caller consumes.
func TestLegalMovesMatchMakeMoveReference(t *testing.T) {
	rng := rand.New(rand.NewSource(20260826))

	const games, maxPlies = 60, 140
	positions, inCheck := 0, 0

	for g := 0; g < games; g++ {
		b := StartingBoard()
		for ply := 0; ply < maxPlies; ply++ {
			fast := GenerateLegalMoves(&b)
			slow := generateLegalMovesByMakeMove(&b)
			positions++
			if IsSquareAttacked(&b, b.Pieces[b.SideToMove][King].LSB(), b.SideToMove.Opposite()) {
				inCheck++
			}
			if !sameMoves(fast, slow) {
				t.Fatalf("game %d ply %d: legal move sets differ\n  fast: %v\n  slow: %v", g, ply, sorted(fast), sorted(slow))
			}
			if len(fast) == 0 {
				break
			}
			b = MakeMove(b, fast[rng.Intn(len(fast))])
		}
	}

	if inCheck == 0 {
		t.Error("no position in check occurred; the check-evasion path went untested")
	}
	t.Logf("compared %d positions, %d of them in check", positions, inCheck)
}

func sorted(moves []Move) []string {
	out := make([]string, len(moves))
	for i, m := range moves {
		out[i] = m.String()
	}
	sort.Strings(out)
	return out
}

func sameMoves(a, b []Move) bool {
	if len(a) != len(b) {
		return false
	}
	sa, sb := sorted(a), sorted(b)
	for i := range sa {
		if sa[i] != sb[i] {
			return false
		}
	}
	return true
}
