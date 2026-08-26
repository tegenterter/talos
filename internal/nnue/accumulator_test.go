package nnue

import (
	"math/rand"
	"testing"

	"talos/internal/board"
)

// TestIncrementalMatchesRebuilt is the load-bearing test for the whole
// incremental path: after any legal move, updating the accumulator must
// produce exactly what rebuilding it from the new position produces. It is
// the same shape of check as explain_test.go's
// TestExplainAblationMatchesRebuiltBoard, and for the same reason — a fast
// path that quietly disagrees with the slow one is worse than no fast path,
// because every score the search reports would be subtly wrong.
//
// Random legal games are played rather than fixed positions so the rarer
// move kinds (castling, en passant, promotion, king captures) actually
// occur; the seed is fixed so a failure is reproducible.
func TestIncrementalMatchesRebuilt(t *testing.T) {
	n := DefaultNetwork
	rng := rand.New(rand.NewSource(20260826))

	const games, maxPlies = 40, 120
	kinds := map[board.MoveFlag]int{}
	promotions, captures := 0, 0

	for g := 0; g < games; g++ {
		b := board.StartingBoard()
		var acc Accumulator
		n.Refresh(&acc, &b)

		for ply := 0; ply < maxPlies; ply++ {
			moves := board.GenerateLegalMoves(&b)
			if len(moves) == 0 {
				break
			}
			m := moves[rng.Intn(len(moves))]

			kinds[m.Flag]++
			if m.Promotion != board.NoPiece {
				promotions++
			}
			if _, _, ok := b.PieceAt(m.To); ok {
				captures++
			}

			after := board.MakeMove(b, m)

			var next Accumulator
			n.Update(&next, &acc, &b, m, &after)

			var want Accumulator
			n.Refresh(&want, &after)
			if next != want {
				t.Fatalf("game %d ply %d: accumulator after %v diverged from a rebuild\n  position before: %v\n", g, ply, m, b)
			}
			// The scores must agree too, which is what the search actually
			// consumes — a divergence too small to change the accumulator
			// bit pattern would still be caught above, but this pins the
			// property the caller cares about.
			if got, w := n.EvaluateAcc(&next, after.SideToMove), n.Evaluate(&after); got != w {
				t.Fatalf("game %d ply %d: EvaluateAcc = %d, Evaluate = %d after %v", g, ply, got, w, m)
			}

			acc, b = next, after
		}
	}

	// A test that never exercised castling or en passant would pass while
	// those paths were broken, so fail loudly if the random games did not
	// reach them.
	for _, f := range []board.MoveFlag{board.EnPassantCapture, board.CastleKingside, board.CastleQueenside} {
		if kinds[f] == 0 {
			t.Errorf("no move with flag %v occurred; this test proves nothing about that path", f)
		}
	}
	if promotions == 0 {
		t.Error("no promotion occurred; this test proves nothing about the promotion path")
	}
	t.Logf("covered %d captures, %d promotions, flags %v", captures, promotions, kinds)
}

// TestRefreshMatchesEvaluate checks the two entry points agree on a plain
// position, independent of any incremental update.
func TestRefreshMatchesEvaluate(t *testing.T) {
	for _, fen := range []string{
		board.StartFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"8/k7/2B5/b7/8/6r1/4K3/8 b - - 1 128",
	} {
		b, err := board.ParseFEN(fen)
		if err != nil {
			t.Fatal(err)
		}
		var acc Accumulator
		DefaultNetwork.Refresh(&acc, &b)
		if got, want := DefaultNetwork.EvaluateAcc(&acc, b.SideToMove), DefaultNetwork.Evaluate(&b); got != want {
			t.Errorf("%s: EvaluateAcc = %d, Evaluate = %d", fen, got, want)
		}
	}
}
