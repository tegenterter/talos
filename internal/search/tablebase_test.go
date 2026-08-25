package search

import (
	"testing"
	"time"

	"talos/internal/board"
	"talos/internal/tablebase"
)

// testTablebase loads internal/tablebase's own small real Syzygy test
// fixtures (see that package's testdata for what's covered and how those
// files were validated against python-chess).
func testTablebase(t *testing.T) *tablebase.Tablebase {
	t.Helper()
	tb := tablebase.NewTablebase()
	if _, err := tb.AddDirectory("../tablebase/testdata"); err != nil {
		t.Fatalf("AddDirectory: %v", err)
	}
	return tb
}

func TestSearchReportsTablebaseWin(t *testing.T) {
	// A elementary KPvK win: White king e1, pawn e2, lone Black king e8.
	b := mustFEN(t, "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")

	var lastInfo Info
	move, ok := Search(b, Options{
		MaxTime:   200 * time.Millisecond,
		Tablebase: testTablebase(t),
		OnInfo:    func(i Info) { lastInfo = i },
	})
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	if legal := board.GenerateLegalMoves(&b); !moveIn(move, legal) {
		t.Errorf("Search returned %v, which is not among the legal moves %v", move, legal)
	}
	if lastInfo.Mate != 0 {
		t.Errorf("Mate = %d, want 0 (a WDL-only win isn't a known mate distance)", lastInfo.Mate)
	}
	if lastInfo.ScoreCP <= 0 {
		t.Errorf("ScoreCP = %d, want a clearly positive (won) score", lastInfo.ScoreCP)
	}
}

func TestSearchReportsTablebaseLoss(t *testing.T) {
	// White's pawn is one step from queening and Black's lone king is far
	// too far away to catch it — a clear loss for Black to move (verified
	// independently against python-chess's syzygy probe on the same test
	// table).
	b := mustFEN(t, "8/8/4P3/8/8/8/1k6/4K3 b - - 0 1")

	var lastInfo Info
	_, ok := Search(b, Options{
		MaxTime:   200 * time.Millisecond,
		Tablebase: testTablebase(t),
		OnInfo:    func(i Info) { lastInfo = i },
	})
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	if lastInfo.ScoreCP >= 0 {
		t.Errorf("ScoreCP = %d, want a clearly negative (lost) score", lastInfo.ScoreCP)
	}
}

func TestSearchReportsTablebaseDraw(t *testing.T) {
	// Kings equidistant from a lone pawn with White to move: the king
	// reaches the pawn just in time, a known draw (verified independently
	// against python-chess's syzygy probe on the same test table).
	b := mustFEN(t, "4k3/8/4p3/8/8/8/8/4K3 w - - 0 1")

	var lastInfo Info
	_, ok := Search(b, Options{
		MaxTime:   200 * time.Millisecond,
		Tablebase: testTablebase(t),
		OnInfo:    func(i Info) { lastInfo = i },
	})
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	if lastInfo.Mate != 0 || lastInfo.ScoreCP != 0 {
		t.Errorf("got Mate=%d ScoreCP=%d, want an exact 0 (draw)", lastInfo.Mate, lastInfo.ScoreCP)
	}
}

func TestSearchWithoutTablebaseIgnoresIt(t *testing.T) {
	// Options.Tablebase left nil (the zero value): search must behave
	// exactly as it did before this package existed, not panic or treat
	// every position as unprobeable-but-required.
	b := mustFEN(t, "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
	if _, ok := Search(b, Options{MaxTime: 100 * time.Millisecond}); !ok {
		t.Fatal("Search reported no legal moves")
	}
}
