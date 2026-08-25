package search

import (
	"testing"
	"time"

	"talos/internal/board"
)

// TestEveryCompletedDepthIsReported pins the fix for a real defect: the
// end-of-iteration report used to be throttled by InfoInterval, so any
// depth that finished within one interval of the previous one was dropped
// and never re-reported. From the opening that silently swallowed depths
// 1-6 — a GUI's first sight of the engine was "depth 7" — and a completed
// depth is exactly the report that must never be dropped.
func TestEveryCompletedDepthIsReported(t *testing.T) {
	b := board.StartingBoard()

	var depths []int
	Search(b, Options{
		MaxDepth: 6,
		Threads:  1,
		// Long enough that every throttled report would be suppressed.
		InfoInterval: time.Hour,
		OnInfo: func(i Info) {
			if len(i.PV) > 0 {
				depths = append(depths, i.Depth)
			}
		},
	})

	seen := map[int]bool{}
	for _, d := range depths {
		seen[d] = true
	}
	for d := 1; d <= 6; d++ {
		if !seen[d] {
			t.Errorf("depth %d was never reported; got depths %v", d, depths)
		}
	}
}

// TestReportsProgressWithinAnIteration covers what iterating-in-silence
// costs: past roughly depth 12 a single iteration runs for minutes, and an
// engine that only reports between iterations looks hung. progressReportAfter
// is shrunk to zero here so a short search stands in for that long one.
func TestReportsProgressWithinAnIteration(t *testing.T) {
	defer func(v time.Duration) { progressReportAfter = v }(progressReportAfter)
	progressReportAfter = 0

	b := board.StartingBoard()

	var currMoves, heartbeats, results int
	Search(b, Options{
		MaxDepth:     8,
		Threads:      1,
		InfoInterval: time.Millisecond,
		OnInfo: func(i Info) {
			switch {
			case i.CurrMoveNumber > 0:
				currMoves++
				if len(i.PV) > 0 {
					t.Errorf("currmove report carries a PV: %+v", i)
				}
				if i.CurrMove == (board.Move{}) {
					t.Errorf("currmove report has no move: %+v", i)
				}
			case len(i.PV) > 0:
				results++
			default:
				heartbeats++
			}
			if i.Depth < 1 {
				t.Errorf("report with depth %d: %+v", i.Depth, i)
			}
		},
	})

	if currMoves == 0 {
		t.Error("no currmove reports during the search")
	}
	if heartbeats == 0 {
		t.Error("no heartbeat reports during the search")
	}
	if results == 0 {
		t.Error("no result reports during the search")
	}
}

// TestRootUpdatesReportRealLines checks the property that makes an
// in-progress report worth printing at all: every line reported mid-search
// must start with a legal root move, so a GUI can display it (and a human
// can trust it) without waiting for the iteration to finish.
func TestRootUpdatesReportRealLines(t *testing.T) {
	defer func(v time.Duration) { progressReportAfter = v }(progressReportAfter)
	progressReportAfter = 0

	b := mustFEN(t, "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
	legal := map[board.Move]bool{}
	for _, m := range board.GenerateLegalMoves(&b) {
		legal[m] = true
	}

	var checked int
	Search(b, Options{
		MaxDepth:     7,
		Threads:      1,
		InfoInterval: time.Millisecond,
		OnInfo: func(i Info) {
			if len(i.PV) == 0 {
				return
			}
			checked++
			if !legal[i.PV[0]] {
				t.Errorf("reported PV starts with %v, not a legal move at root", i.PV[0])
			}
		},
	})
	if checked == 0 {
		t.Fatal("no PV was ever reported")
	}
}

// TestBoundedScoresAreReportedAsBounds covers the honesty requirement on
// in-progress reports: a score an unfinished iteration hasn't proved must
// be labelled as the bound it is. The aspiration window is shrunk to a
// centipawn so the root fails high and low constantly (the same trick
// aspiration_test.go uses), which is what produces those bounds.
func TestBoundedScoresAreReportedAsBounds(t *testing.T) {
	defer func(after time.Duration, window int) {
		progressReportAfter, aspirationWindowCP = after, window
	}(progressReportAfter, aspirationWindowCP)
	progressReportAfter, aspirationWindowCP = 0, 1

	b := mustFEN(t, "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")

	var bounded, exactCompleted int
	Search(b, Options{
		MaxDepth:     8,
		Threads:      1,
		InfoInterval: time.Millisecond,
		OnInfo: func(i Info) {
			if len(i.PV) == 0 {
				return
			}
			if i.Bound != BoundExact {
				bounded++
			} else {
				exactCompleted++
			}
		},
	})

	if bounded == 0 {
		t.Error("no report was labelled as a bound, though every aspiration window here fails")
	}
	if exactCompleted == 0 {
		t.Error("no report carried an exact score; completed depths must")
	}
}

// TestHashfullTracksOccupancy checks the sampled reading is actually a
// reading and not a constant: an untouched table is empty, and one a real
// search has filled is not.
func TestHashfullTracksOccupancy(t *testing.T) {
	tt := newTranspositionTable(1)
	if got := tt.hashfull(); got != 0 {
		t.Errorf("hashfull() on a fresh table = %d, want 0", got)
	}

	b := board.StartingBoard()
	Search(b, Options{MaxDepth: 7, Threads: 1, Table: &Table{tt: tt}})

	got := tt.hashfull()
	if got <= 0 || got > 1000 {
		t.Errorf("hashfull() after a search = %d, want a permille figure in (0, 1000]", got)
	}
}
