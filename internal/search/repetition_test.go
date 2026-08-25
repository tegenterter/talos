package search

import (
	"testing"
	"time"

	"talos/internal/board"
)

// TestGameHistoryRepetitionIsSeen checks that a line repeating a position the
// *game* already visited scores as a draw, not just one repeating inside the
// search. Before Options.GameHistory existed, the search could only see
// repetitions it created itself, so an engine could walk into a threefold draw
// it had no way to notice — throwing away a won position, or missing a saving
// one.
//
// The position is a bare-king-and-queen-vs-king race where White is winning
// easily; if the search believes returning to a previously seen position is a
// draw, its evaluation of a line that does so collapses to 0.
func TestGameHistoryRepetitionIsSeen(t *testing.T) {
	// White queen and king against a lone king: overwhelmingly winning. The
	// halfmove clock is set high so repeatableHistory keeps every entry the
	// test supplies below (it trims history to the clock, since nothing
	// before the last irreversible move can recur) while staying clear of
	// 100, where the fifty-move rule would score the position 0 by itself
	// and the test would pass for the wrong reason.
	b := mustFEN(t, "7k/8/8/8/8/8/6Q1/6K1 w - - 90 1")

	scoreWith := func(history []uint64) int {
		var last Info
		_, ok := Search(b, Options{
			MaxDepth:    4,
			Threads:     1,
			GameHistory: history,
			OnInfo:      func(i Info) { last = i },
		})
		if !ok {
			t.Fatal("Search reported no legal moves")
		}
		return last.ScoreCP
	}

	plain := scoreWith(nil)
	if plain <= 0 {
		t.Fatalf("sanity check failed: K+Q vs K should be winning for White, got %d", plain)
	}

	// Now claim every position the search can reach in one move has already
	// occurred in the game. Every root move then leads to a position the
	// search must treat as a repetition, so the best it can claim is a draw.
	var reachable []uint64
	for _, m := range board.GenerateLegalMoves(&b) {
		child := board.MakeMove(b, m)
		reachable = append(reachable, child.Hash())
	}
	// Guard the test's own premise: if the clock were shorter than this
	// list, trimming would drop some entries and leave the search a
	// non-repeating move, making the assertion below meaningless.
	if len(reachable) > b.HalfmoveClock {
		t.Fatalf("test setup: %d reachable positions exceeds the halfmove clock %d, so history would be trimmed",
			len(reachable), b.HalfmoveClock)
	}
	if got := scoreWith(reachable); got != 0 {
		t.Errorf("with every reachable position already in the game history, score = %d, want 0 "+
			"(every move repeats, so the position is a draw)", got)
	}
}

// TestGameHistoryIsTrimmedToTheRepeatableSpan checks repeatableHistory keeps
// only what can still recur. A capture or pawn move is irreversible, so no
// position before it can ever appear again — the halfmove clock counts exactly
// the plies back to that boundary, and scanning past it at every node would be
// wasted work on a hot path.
func TestGameHistoryIsTrimmedToTheRepeatableSpan(t *testing.T) {
	history := []uint64{1, 2, 3, 4, 5, 6, 7, 8}

	tests := []struct {
		name          string
		halfmoveClock int
		want          []uint64
	}{
		{"clock zeroed by an irreversible move keeps nothing", 0, nil},
		{"clock shorter than history keeps the newest entries", 3, []uint64{6, 7, 8}},
		{"clock longer than history keeps all of it", 99, history},
		{"clock exactly the history length keeps all of it", 8, history},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := repeatableHistory(history, tt.halfmoveClock)
			if len(got) != len(tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("got %v, want %v", got, tt.want)
				}
			}
		})
	}
}

// TestSoftTimeLimitStopsEarly checks the search declines to begin an
// iteration it cannot finish. Because an aborted iteration is discarded
// entirely, time spent past the last completed depth buys nothing at all —
// the same move comes back either way — so the search should return well
// before its hard deadline rather than burning the remainder.
func TestSoftTimeLimitStopsEarly(t *testing.T) {
	b := board.StartingBoard()

	const budget = 2 * time.Second
	start := time.Now()
	var last Info
	_, ok := Search(b, Options{
		MaxTime: budget,
		Threads: 1,
		OnInfo:  func(i Info) { last = i },
	})
	elapsed := time.Since(start)
	if !ok {
		t.Fatal("Search reported no legal moves")
	}

	t.Logf("returned depth %d after %v of a %v budget", last.Depth, elapsed.Round(time.Millisecond), budget)
	if elapsed >= budget {
		t.Errorf("search used its whole %v budget (%v); the soft limit should have stopped it "+
			"before starting an iteration it could not finish", budget, elapsed)
	}
	// Guard the other direction too: stopping early is only worth anything
	// if the search still got real work done first.
	if last.Depth < 5 {
		t.Errorf("search only reached depth %d in %v; the soft limit looks far too aggressive", last.Depth, elapsed)
	}
}

// TestSoftTimeLimitDoesNotApplyToFixedDepth guards the exemptions: a bare "go
// depth N" has no clock at all and must run to completion however long it
// takes, which is what makes fixed-depth results (golden_test.go, bench)
// reproducible.
func TestSoftTimeLimitDoesNotApplyToFixedDepth(t *testing.T) {
	b := board.StartingBoard()
	var last Info
	_, ok := Search(b, Options{MaxDepth: 8, Threads: 1, OnInfo: func(i Info) { last = i }})
	if !ok {
		t.Fatal("Search reported no legal moves")
	}
	if last.Depth != 8 {
		t.Errorf("fixed-depth search completed depth %d, want 8 — a time limit must not truncate it", last.Depth)
	}
}
