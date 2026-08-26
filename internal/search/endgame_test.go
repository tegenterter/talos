package search

import (
	"fmt"
	"testing"

	"talos/internal/board"
)

// TestDampScalesWithTheFiftyMoveClock pins the curve itself. The bound at
// half matters as much as the slope: an evaluation that decays to nothing
// makes material worthless near the wall, and an engine that values a
// bishop at 33 centipawns gives it away (see damp's doc comment).
func TestDampScalesWithTheFiftyMoveClock(t *testing.T) {
	tests := []struct {
		clock, want int
	}{
		{0, 1000},
		{50, 750},
		{100, 500},
		{200, 500}, // clamped: negamax scores a draw before this, but never trust the caller
		{-5, 1000},
	}
	for _, tc := range tests {
		if got := damp(1000, tc.clock); got != tc.want {
			t.Errorf("damp(1000, %d) = %d, want %d", tc.clock, got, tc.want)
		}
	}
	if got := damp(-1000, 100); got != -500 {
		t.Errorf("damp(-1000, 100) = %d, want -500 (damping is symmetric)", got)
	}
}

// TestShufflingCostsScore is the fix for the drawn game in one assertion:
// the same winning position must be worth less with the fifty-move clock
// run up than with it fresh. Before, the score was flat (+863 at clock 0,
// 40 and 80 alike) until the draw entered the search horizon and it fell
// off a cliff to 0 — so nothing in the search preferred making progress.
func TestShufflingCostsScore(t *testing.T) {
	const fen = "8/k7/2B5/b7/8/6r1/4K3/8 b - - %d 128" // the drawn game, move 128

	score := func(clock int) int {
		b := mustFEN(t, fmtFEN(fen, clock))
		var last Info
		if _, ok := Search(b, Options{MaxDepth: 8, Threads: 1, OnInfo: func(i Info) { last = i }}); !ok {
			t.Fatal("Search found no legal move")
		}
		return last.ScoreCP
	}

	fresh, stale := score(0), score(80)
	if stale >= fresh {
		t.Errorf("score at clock 80 = %+d, at clock 0 = %+d; want the stale one strictly lower", stale, fresh)
	}
	if fresh <= 0 {
		t.Fatalf("score at clock 0 = %+d, want the rook-up side winning", fresh)
	}
}

// TestConvertsBasicMates is the regression guard for the game that
// prompted all of this: a rook-and-bishop-up endgame that the engine
// shuffled into a fifty-move draw. Each position below is played out
// against itself; the win has to arrive before the fifty-move rule does.
//
// These are deliberately end-to-end (play the whole ending, don't inspect
// a score) because that is the only thing that failed: every intermediate
// number looked fine, and the engine still drew.
func TestConvertsBasicMates(t *testing.T) {
	if testing.Short() {
		t.Skip("plays out whole endings")
	}

	tests := []struct {
		name     string
		fen      string
		depth    int
		maxPlies int
	}{
		{"KQ vs K", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 8, 40},
		{"KR vs K", "4k3/8/8/8/8/8/8/R3K3 w - - 0 1", 8, 60},
		{"KRB vs KB, from the drawn game", "8/k7/2B5/b7/8/6r1/4K3/8 b - - 1 128", 8, 80},
		// Bishop and knight is the hard one: it is the only basic mate
		// whose technique depends on *which* corner the defender is driven
		// to, and it needs real depth to steer. Given both, the engine
		// mates in about 30 moves, comfortably inside the fifty-move rule
		// it used to lose this ending to.
		{"KBN vs K", "4k3/8/8/8/8/8/8/2B1KN2 w - - 0 1", 12, 90},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := mustFEN(t, tc.fen)
			table := NewTable(16)
			// The played game's position hashes, exactly as internal/uci
			// supplies them. Without this the search can only see
			// repetitions it creates inside one line, so it will happily
			// repeat a position it has already stood in twice — which on a
			// first attempt at this test made a won ending look unwinnable
			// when the engine was fine and the harness was not.
			var history []uint64

			for ply := 0; ply < tc.maxPlies; ply++ {
				if mated, ok := gameOver(&b); ok {
					if !mated {
						t.Fatalf("ply %d: game ended in a draw, not mate: %s", ply, fen(&b))
					}
					return // mate delivered
				}
				if b.HalfmoveClock >= 100 {
					t.Fatalf("ply %d: fifty-move rule reached without mate: %s", ply, fen(&b))
				}

				move, ok := Search(b, Options{
					MaxDepth:    tc.depth,
					Threads:     1,
					Table:       table,
					GameHistory: history,
				})
				if !ok {
					t.Fatalf("ply %d: Search found no legal move in %s", ply, fen(&b))
				}
				history = append(history, b.Hash())
				b = board.MakeMove(b, move)
			}
			t.Errorf("no mate within %d plies; reached %s", tc.maxPlies, fen(&b))
		})
	}
}

// gameOver reports whether b has no legal moves, and if so whether that is
// checkmate (as opposed to stalemate).
func gameOver(b *board.Board) (mated, over bool) {
	if len(board.GenerateLegalMoves(b)) > 0 {
		return false, false
	}
	kingSq := b.Pieces[b.SideToMove][board.King].LSB()
	return board.IsSquareAttacked(b, kingSq, b.SideToMove.Opposite()), true
}

// fen renders just enough of a position to identify it in a failure
// message; the package has no FEN writer of its own.
func fen(b *board.Board) string {
	var sb []byte
	for rank := 7; rank >= 0; rank-- {
		empty := 0
		for file := 0; file < 8; file++ {
			c, pt, ok := b.PieceAt(board.Square(rank*8 + file))
			if !ok {
				empty++
				continue
			}
			if empty > 0 {
				sb = append(sb, byte('0'+empty))
				empty = 0
			}
			ch := "pnbrqk"[pt]
			if c == board.White {
				ch -= 'a' - 'A'
			}
			sb = append(sb, ch)
		}
		if empty > 0 {
			sb = append(sb, byte('0'+empty))
		}
		if rank > 0 {
			sb = append(sb, '/')
		}
	}
	side := " w"
	if b.SideToMove == board.Black {
		side = " b"
	}
	return string(sb) + side
}

// fmtFEN substitutes a halfmove clock into a FEN template.
func fmtFEN(template string, clock int) string {
	return fmt.Sprintf(template, clock)
}
