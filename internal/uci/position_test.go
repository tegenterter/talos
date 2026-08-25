package uci

import (
	"strings"
	"testing"
	"time"

	"talos/internal/board"
)

// TestPositionRejectsMalformedMoveList is the regression guard on what was a
// game-losing bug: parsePosition used to stop at the first bad move and return
// the partially-replayed board as if the command had succeeded. The engine
// then searched a position several plies behind the GUI's — with, in the case
// below, the wrong side to move — and replied with a move that is illegal in
// the real game, which an arbiter scores as a forfeit.
//
// The whole command must be rejected instead, leaving the previously set
// position untouched.
func TestPositionRejectsMalformedMoveList(t *testing.T) {
	tests := []struct {
		name string
		cmd  string
	}{
		{"unparsable move", "position startpos moves e2e4 e7e5 XXXX g1f3"},
		{"illegal move", "position startpos moves e2e4 e7e5 e1e8"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := startEngine(t)
			// Establish a known-good position first, so the test can tell
			// "kept the previous position" from "reset to something else".
			h.send("position startpos moves d2d4")
			h.send(tt.cmd)
			if !h.waitFor("info string position:", time.Second) {
				t.Fatalf("malformed position command was not reported; output:\n%s", h.out.String())
			}

			// The retained position is startpos+d2d4, so it is Black to
			// move and a black reply must come back — not a white move,
			// which is what the truncated-replay bug produced.
			h.send("go depth 4")
			if !h.waitFor("bestmove", 5*time.Second) {
				t.Fatalf("no bestmove; output:\n%s", h.out.String())
			}
			h.quit()

			move := bestmoveFrom(t, h.out.String())
			b := board.StartingBoard()
			b = playUCI(t, b, "d2d4")
			if !isLegalIn(&b, move) {
				t.Errorf("engine replied %q, which is not legal in the position it should have kept "+
					"(startpos + d2d4); the malformed command corrupted its position", move)
			}
		})
	}
}

// TestPositionAcceptsWellFormedMoveList is the other half of the guard above:
// rejecting malformed input must not mean rejecting valid input.
func TestPositionAcceptsWellFormedMoveList(t *testing.T) {
	h := startEngine(t)
	h.send("position startpos moves e2e4 e7e5 g1f3 b8c6")
	h.send("go depth 4")
	if !h.waitFor("bestmove", 5*time.Second) {
		t.Fatalf("no bestmove; output:\n%s", h.out.String())
	}
	h.quit()
	if strings.Contains(h.out.String(), "info string position:") {
		t.Errorf("a valid position command was reported as malformed; output:\n%s", h.out.String())
	}

	move := bestmoveFrom(t, h.out.String())
	b := board.StartingBoard()
	for _, m := range []string{"e2e4", "e7e5", "g1f3", "b8c6"} {
		b = playUCI(t, b, m)
	}
	if !isLegalIn(&b, move) {
		t.Errorf("engine replied %q, not legal in the position it was given", move)
	}
}

// bestmoveFrom extracts the move from the last "bestmove" line in out.
func bestmoveFrom(t *testing.T, out string) string {
	t.Helper()
	found := ""
	for _, line := range strings.Split(out, "\n") {
		if fields := strings.Fields(line); len(fields) >= 2 && fields[0] == "bestmove" {
			found = fields[1]
		}
	}
	if found == "" {
		t.Fatalf("no bestmove line in output:\n%s", out)
	}
	return found
}

func playUCI(t *testing.T, b board.Board, moveStr string) board.Board {
	t.Helper()
	parsed, ok := board.ParseUCIMove(moveStr)
	if !ok {
		t.Fatalf("test move %q does not parse", moveStr)
	}
	legal, ok := matchLegalMove(&b, parsed)
	if !ok {
		t.Fatalf("test move %q is not legal", moveStr)
	}
	return board.MakeMove(b, legal)
}

func isLegalIn(b *board.Board, moveStr string) bool {
	parsed, ok := board.ParseUCIMove(moveStr)
	if !ok {
		return false
	}
	_, ok = matchLegalMove(b, parsed)
	return ok
}
