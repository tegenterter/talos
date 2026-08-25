package tablebase

import (
	"encoding/json"
	"os"
	"testing"

	"talos/internal/board"
)

// TestProbeMatchesPythonChess is the strongest correctness check this
// package has: testdata/wdl_ground_truth.json holds WDL values computed
// by python-chess's chess.syzygy — a real, independently-written, widely
// used Syzygy implementation — probing testdata/*.rtbw (small real
// Syzygy tables covering pawnless enc_type 0 (KBvK, KNvK, KRvK, KQvK,
// KRvKN), enc_type 2/"K2" (KRRvK), and pawn tables with one or two pawns
// on one or both sides (KPvK, KPvKP, KPPvK)) at 154 random legal
// positions. If this package's port of the format disagrees with that
// ground truth on any of them, something in the index/decompression math
// is wrong.
func TestProbeMatchesPythonChess(t *testing.T) {
	data, err := os.ReadFile("testdata/wdl_ground_truth.json")
	if err != nil {
		t.Fatalf("reading ground truth: %v", err)
	}
	var cases []struct {
		Material string `json:"material"`
		FEN      string `json:"fen"`
		WDL      int    `json:"wdl"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("parsing ground truth: %v", err)
	}
	if len(cases) == 0 {
		t.Fatal("no ground-truth cases loaded")
	}

	tb := NewTablebase()
	if _, err := tb.AddDirectory("testdata"); err != nil {
		t.Fatalf("AddDirectory: %v", err)
	}

	mismatches := 0
	for _, c := range cases {
		b, err := board.ParseFEN(c.FEN)
		if err != nil {
			t.Errorf("%s: ParseFEN(%q): %v", c.Material, c.FEN, err)
			continue
		}
		got, ok := tb.Probe(&b)
		if !ok {
			t.Errorf("%s: Probe(%q) = not ok, want %d", c.Material, c.FEN, c.WDL)
			mismatches++
			continue
		}
		if got != c.WDL {
			t.Errorf("%s: Probe(%q) = %d, want %d", c.Material, c.FEN, got, c.WDL)
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Fatalf("%d/%d positions disagreed with python-chess ground truth", mismatches, len(cases))
	}
}
