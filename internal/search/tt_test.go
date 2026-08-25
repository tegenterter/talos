package search

import (
	"testing"

	"talos/internal/board"
)

func TestTTStoreProbeRoundTrip(t *testing.T) {
	tt := newTranspositionTable(16)
	tt.store(42, boardMoveFixture(), 137, 5, ttExact, 3)

	e, ok := tt.probe(42, 3)
	if !ok {
		t.Fatal("probe(42) = not found, want found")
	}
	if e.score != 137 || e.depth != 5 || e.flag != ttExact || e.move != boardMoveFixture() {
		t.Errorf("probe(42) = %+v, want score=137 depth=5 flag=ttExact move=%v", e, boardMoveFixture())
	}
}

func TestTTProbeMissReturnsFalse(t *testing.T) {
	tt := newTranspositionTable(16)
	if _, ok := tt.probe(999, 0); ok {
		t.Error("probe on an empty table returned ok=true")
	}
}

func TestTTReplacementPrefersDeeperResults(t *testing.T) {
	tt := newTranspositionTable(16)
	tt.store(7, boardMoveFixture(), 100, 5, ttExact, 0)

	// A shallower result for the same position must not overwrite it.
	tt.store(7, boardMoveFixture(), 200, 3, ttExact, 0)
	e, _ := tt.probe(7, 0)
	if e.score != 100 || e.depth != 5 {
		t.Errorf("shallower store overwrote a deeper entry: got score=%d depth=%d, want score=100 depth=5", e.score, e.depth)
	}

	// A deeper result must replace it.
	tt.store(7, boardMoveFixture(), 300, 8, ttExact, 0)
	e, _ = tt.probe(7, 0)
	if e.score != 300 || e.depth != 8 {
		t.Errorf("deeper store did not replace the entry: got score=%d depth=%d, want score=300 depth=8", e.score, e.depth)
	}
}

func TestTTSizingFromHashMB(t *testing.T) {
	small := newTranspositionTable(1)
	big := newTranspositionTable(64)
	if totalTTEntries(big) <= totalTTEntries(small) {
		t.Errorf("newTranspositionTable(64) has %d total entries, want more than newTranspositionTable(1)'s %d", totalTTEntries(big), totalTTEntries(small))
	}

	zero := newTranspositionTable(0)
	if totalTTEntries(zero) != totalTTEntries(newTranspositionTable(DefaultHashMB)) {
		t.Error("newTranspositionTable(0) should size as DefaultHashMB")
	}
}

func totalTTEntries(tt *transpositionTable) int {
	total := 0
	for i := range tt.shards {
		total += len(tt.shards[i].entries)
	}
	return total
}

func TestMateScoreAdjustment(t *testing.T) {
	// A mate 5 plies from the original search root, encountered while
	// storing at ply 2 (so 3 plies further from this specific position):
	// normalized-for-storage should read as "mate in 3 from here"
	// (mateValue-3), then re-expanding at a *different* ply (7, as if
	// this same position were reached by another search rooted
	// elsewhere) should read as "mate in 10 from that root".
	stored := adjustMateScoreToTT(mateValue-5, 2)
	if want := mateValue - 3; stored != want {
		t.Errorf("adjustMateScoreToTT(mateValue-5, 2) = %d, want %d", stored, want)
	}
	retrieved := adjustMateScoreFromTT(stored, 7)
	if want := mateValue - 10; retrieved != want {
		t.Errorf("adjustMateScoreFromTT(%d, 7) = %d, want %d", stored, retrieved, want)
	}

	// Same shape, for the side about to get mated (negative scores).
	stored = adjustMateScoreToTT(-(mateValue - 5), 2)
	if want := -(mateValue - 3); stored != want {
		t.Errorf("adjustMateScoreToTT(-(mateValue-5), 2) = %d, want %d", stored, want)
	}
	retrieved = adjustMateScoreFromTT(stored, 7)
	if want := -(mateValue - 10); retrieved != want {
		t.Errorf("adjustMateScoreFromTT(%d, 7) = %d, want %d", stored, retrieved, want)
	}
}

func TestMateScoreAdjustmentRoundTripsAtSamePly(t *testing.T) {
	for _, score := range []int{mateValue, mateValue - 1, mateValue - 50, -(mateValue), -(mateValue - 20)} {
		for _, ply := range []int{0, 1, 10, 50} {
			got := adjustMateScoreFromTT(adjustMateScoreToTT(score, ply), ply)
			if got != score {
				t.Errorf("round trip at ply %d: adjustMateScoreToTT/FromTT(%d) = %d, want %d", ply, score, got, score)
			}
		}
	}
}

func TestMateScoreAdjustmentLeavesOrdinaryScoresAlone(t *testing.T) {
	for _, score := range []int{0, 137, -450, mateThreshold - 1, -(mateThreshold - 1)} {
		if got := adjustMateScoreToTT(score, 5); got != score {
			t.Errorf("adjustMateScoreToTT(%d, 5) = %d, want unchanged %d", score, got, score)
		}
		if got := adjustMateScoreFromTT(score, 5); got != score {
			t.Errorf("adjustMateScoreFromTT(%d, 5) = %d, want unchanged %d", score, got, score)
		}
	}
}

// boardMoveFixture is a fixed, arbitrary move used only to check that
// ttEntry round-trips a move value correctly; its actual legality in any
// position is irrelevant to these tests.
func boardMoveFixture() board.Move {
	return board.Move{From: 12, To: 28, Promotion: board.NoPiece, Flag: board.DoublePawnPush}
}
