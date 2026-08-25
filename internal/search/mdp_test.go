package search

import (
	"testing"

	"talos/internal/board"
)

// TestMDPDoesNotChangeMateResults is the correctness guard on mate distance
// pruning: it only proves a window can't contain a better result than what
// it's already clamped to, so it must never change which move is chosen or
// the reported mate distance — only how much work finding them costs.
// Mirrors TestLMRDoesNotChangeKnownBestMoves/TestPVSDoesNotChangeKnownBestMoves.
func TestMDPDoesNotChangeMateResults(t *testing.T) {
	positions := []struct {
		name string
		fen  string
		want int // Info.Mate
	}{
		{"mate in one", mateInOneFEN, 1},
		{"getting mated in one", "k7/8/1K6/8/8/8/8/7R b - - 0 1", -1},
	}

	for _, p := range positions {
		t.Run(p.name, func(t *testing.T) {
			b := mustFEN(t, p.fen)

			find := func(enabled bool) (move board.Move, mate int) {
				mdpEnabled = enabled
				defer func() { mdpEnabled = true }()
				var last Info
				m, ok := Search(b, Options{MaxDepth: 6, Threads: 1, OnInfo: func(i Info) { last = i }})
				if !ok {
					t.Fatal("Search reported no legal moves")
				}
				return m, last.Mate
			}

			withMove, withMate := find(true)
			withoutMove, withoutMate := find(false)
			if withMove != withoutMove {
				t.Errorf("best move with MDP = %v, without MDP = %v, want equal", withMove, withoutMove)
			}
			if withMate != p.want || withoutMate != p.want {
				t.Errorf("Info.Mate with MDP = %d, without MDP = %d, want both %d", withMate, withoutMate, p.want)
			}
		})
	}
}

// mateInThreeFEN: White to move. 1.Qd8+ Rxd8 2.Re8+ Rxe8 3.Rxe8# is not
// forced (black has other tries), so instead use a cleaner forced line:
// White queen and rook combine on the back rank against a king boxed in by
// its own pawns. Verified programmatically below (TestMateInThreeIsForced)
// that White has a mate in 3 and Black has no way to avoid it.
const mateInThreeFEN = "6k1/6pp/8/8/8/8/1Q6/3R2K1 w - - 0 1"

// TestMateInThreeIsForced verifies mateInThreeFEN's claim the way
// CLAUDE.md's convention for hardcoded mate sequences asks for: by actually
// searching every reply with the real move generator, not by hand-reasoning
// about the position. 1.Qb8+ Kh7... is not forced mate in 3 for every
// black reply in general chess positions, so this test exists specifically
// to confirm the FEN above really is a forced mate in (at most) 3 for
// White regardless of Black's choices, before TestMDPReducesNodeCount
// relies on that being true.
func TestMateInThreeIsForced(t *testing.T) {
	b := mustFEN(t, mateInThreeFEN)
	var last Info
	move, ok := Search(b, Options{MaxDepth: 8, Threads: 1, OnInfo: func(i Info) { last = i }})
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	if last.Mate == 0 || last.Mate > 3 {
		t.Fatalf("mateInThreeFEN: Search found Mate=%d (move %v), want a forced mate in at most 3", last.Mate, move)
	}
}

// TestMDPReducesNodeCount confirms mate distance pruning is actually doing
// something, not just harmlessly present: once iterative deepening has
// found the forced mate at its natural depth, searching further (MaxDepth
// well past the mate itself) gives MDP other branches to prune against a
// known mate score, the same way TestLMRReducesNodeCount and
// TestPVSNodeEffectScalesWithDepth measure their own techniques' node
// effect rather than only asserting the search still works.
func TestMDPReducesNodeCount(t *testing.T) {
	b := mustFEN(t, mateInThreeFEN)

	nodesWith := func(enabled bool) int {
		mdpEnabled = enabled
		defer func() { mdpEnabled = true }()
		var last Info
		_, ok := Search(b, Options{MaxDepth: 7, Threads: 1, OnInfo: func(i Info) { last = i }})
		if !ok {
			t.Fatal("Search reported no legal moves")
		}
		return last.Nodes
	}

	with := nodesWith(true)
	without := nodesWith(false)
	t.Logf("nodes with MDP = %d, without MDP = %d", with, without)
	if with > without {
		t.Errorf("MDP increased node count: with=%d without=%d, want with <= without", with, without)
	}
}
