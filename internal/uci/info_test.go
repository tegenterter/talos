package uci

import (
	"strings"
	"testing"
	"time"

	"talos/internal/board"
	"talos/internal/search"
)

func mustMove(t *testing.T, s string) board.Move {
	t.Helper()
	m, ok := board.ParseUCIMove(s)
	if !ok {
		t.Fatalf("ParseUCIMove(%q) failed", s)
	}
	return m
}

// TestPrintInfoShapes pins the three kinds of "info" line the search now
// produces. The distinctions matter to a GUI: a heartbeat that claimed a
// score would have it plotting numbers the search never proved, and a
// bound printed as an exact score is the same lie in a subtler form.
func TestPrintInfoShapes(t *testing.T) {
	e2e4 := mustMove(t, "e2e4")
	e7e5 := mustMove(t, "e7e5")

	tests := []struct {
		name     string
		info     search.Info
		want     []string
		wantNot  []string
		wantLine string
	}{
		{
			name: "completed depth",
			info: search.Info{
				Depth: 12, SelDepth: 22, ScoreCP: 55, Nodes: 3152617, Nps: 162324,
				HashFull: 431, Time: 19421 * time.Millisecond, PV: []board.Move{e2e4, e7e5},
			},
			wantLine: "info depth 12 seldepth 22 score cp 55 nodes 3152617 nps 162324 hashfull 431 time 19421 pv e2e4 e7e5",
		},
		{
			name: "mate score",
			info: search.Info{Depth: 5, SelDepth: 7, Mate: 3, Nodes: 100, PV: []board.Move{e2e4}},
			want: []string{"score mate 3"},
		},
		{
			name: "fail high is a lower bound",
			info: search.Info{Depth: 12, ScoreCP: 300, Bound: search.BoundLower, PV: []board.Move{e2e4}},
			want: []string{"score cp 300 lowerbound"},
		},
		{
			name: "fail low is an upper bound",
			info: search.Info{Depth: 12, ScoreCP: -300, Bound: search.BoundUpper, PV: []board.Move{e2e4}},
			want: []string{"score cp -300 upperbound"},
		},
		{
			name:     "currmove",
			info:     search.Info{Depth: 14, CurrMove: e2e4, CurrMoveNumber: 3, Nodes: 5},
			wantLine: "info depth 14 currmove e2e4 currmovenumber 3",
		},
		{
			name:    "heartbeat carries no score",
			info:    search.Info{Depth: 14, SelDepth: 26, Nodes: 900, Nps: 150000, Time: 6 * time.Second},
			want:    []string{"info depth 14 seldepth 26 nodes 900 nps 150000 time 6000"},
			wantNot: []string{"score", "pv"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var out strings.Builder
			printInfo(&out, tc.info)
			got := strings.TrimSuffix(out.String(), "\n")

			if tc.wantLine != "" && got != tc.wantLine {
				t.Errorf("printInfo:\n got %q\nwant %q", got, tc.wantLine)
			}
			for _, w := range tc.want {
				if !strings.Contains(got, w) {
					t.Errorf("printInfo output %q missing %q", got, w)
				}
			}
			for _, w := range tc.wantNot {
				if strings.Contains(got, w) {
					t.Errorf("printInfo output %q contains %q, which it must not", got, w)
				}
			}
		})
	}
}

// TestGoInfiniteKeepsReporting is the end-to-end version of the bug this
// reporting exists for: a "go infinite" that has entered an iteration too
// deep to finish must keep saying something. Before, the engine went
// silent for as long as the iteration ran — minutes, past depth 12 — and
// looked hung to both a GUI and a human.
func TestGoInfiniteKeepsReporting(t *testing.T) {
	h := startEngine(t)
	h.send("position startpos")
	h.send("go infinite")

	if !h.waitFor("info depth 1 ", 2*time.Second) {
		t.Fatalf("no depth-1 report; output:\n%s", h.out.String())
	}
	h.send("stop")
	if !h.waitFor("bestmove", 2*time.Second) {
		t.Fatalf("no bestmove after stop; output:\n%s", h.out.String())
	}
	h.quit()
}
