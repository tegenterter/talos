package eval

import (
	"testing"

	"talos/internal/board"
)

func mustFEN(t *testing.T, fen string) board.Board {
	t.Helper()
	b, err := board.ParseFEN(fen)
	if err != nil {
		t.Fatalf("ParseFEN(%q): %v", fen, err)
	}
	return b
}

// TestLopsidedSelectsTheRightRegime pins which positions this evaluator
// claims. Claiming too many would replace a network that is better than
// this at ordinary chess; claiming too few leaves the won endgames it
// exists for to a network that measurably cannot judge them.
func TestLopsidedSelectsTheRightRegime(t *testing.T) {
	tests := []struct {
		name string
		fen  string
		want bool
	}{
		{"KR vs K", "4k3/8/8/8/8/8/8/R3K3 w - - 0 1", true},
		{"KQ vs K", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", true},
		{"KBN vs K", "4k3/8/8/8/8/8/8/2B1KN2 w - - 0 1", true},
		{"KRB vs KB (the drawn game)", "8/k7/2B5/b7/8/6r1/4K3/8 b - - 1 128", true},
		{"KQ vs KR", "4k3/8/8/8/8/8/8/3QK2r w - - 0 1", true},
		{"KR vs KR", "4k2r/8/8/8/8/8/8/R3K3 w - - 0 1", false},
		{"KR vs KB", "4k3/8/8/8/6b1/8/8/R3K3 w - - 0 1", false},
		{"starting position", board.StartFEN, false},
		{"a rook up with pawns still on", "4k3/5ppp/8/8/8/8/5PPP/R3K3 w - - 0 1", false},
		{"a queen up with pawns still on", "4k3/5ppp/8/8/8/8/5PPP/3QK3 w - - 0 1", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := mustFEN(t, tc.fen)
			if got := Lopsided(&b); got != tc.want {
				t.Errorf("Lopsided = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestInsufficientMaterialIsADraw checks the evaluator does not report a
// win it cannot deliver: a lone minor piece, or two knights, cannot force
// mate however the material counts.
func TestInsufficientMaterialIsADraw(t *testing.T) {
	for _, fen := range []string{
		"4k3/8/8/8/8/8/8/4K3 w - - 0 1",
		"4k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
		"4k3/8/8/8/8/8/8/3NK3 w - - 0 1",
		"4k3/8/8/8/8/8/8/1N1NK3 w - - 0 1",
		"4k3/8/8/8/8/6b1/8/2B1K3 w - - 0 1",
	} {
		b := mustFEN(t, fen)
		if got := Evaluate(&b); got != 0 {
			t.Errorf("Evaluate(%q) = %+d, want 0 (neither side can force mate)", fen, got)
		}
	}
}

// TestMatingNetPrefersProgress is the property the whole package exists
// for: among positions material scores identically, the evaluator must
// prefer the one closer to mate. Without this the search has no reason to
// choose one shuffle over another, which is precisely how a won endgame
// becomes a fifty-move draw.
func TestMatingNetPrefersProgress(t *testing.T) {
	tests := []struct {
		name          string
		better, worse string
		reason        string
	}{
		{
			name:   "defending king driven to the edge",
			better: "7k/8/6K1/8/8/8/8/R7 w - - 0 1",
			worse:  "8/8/8/3k4/8/4K3/8/R7 w - - 0 1",
			reason: "a cornered king is closer to mate than a centralized one",
		},
		{
			name:   "attacking king brought up",
			better: "7k/8/6K1/8/8/8/8/R7 w - - 0 1",
			worse:  "7k/8/8/8/8/8/8/R5K1 w - - 0 1",
			reason: "the attacking king has to join in to mate",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			better, worse := mustFEN(t, tc.better), mustFEN(t, tc.worse)
			bs, ws := Evaluate(&better), Evaluate(&worse)
			if bs <= ws {
				t.Errorf("Evaluate(better) = %+d, Evaluate(worse) = %+d; want better > worse (%s)", bs, ws, tc.reason)
			}
		})
	}
}

// TestBishopKnightDrivesToTheRightCorner is the one mating net whose
// direction matters: KBN vs K mates only in a corner the bishop covers, so
// driving the defender to the wrong corner is not progress at all — it is
// the classic way to shuffle out a won game.
func TestBishopKnightDrivesToTheRightCorner(t *testing.T) {
	// A dark-squared bishop (c1) mates on a1 or h8.
	right := mustFEN(t, "8/8/8/8/8/2N5/3K4/k1B5 w - - 0 1")
	wrong := mustFEN(t, "k7/8/8/8/8/2N5/3K4/2B5 w - - 0 1")
	if r, w := Evaluate(&right), Evaluate(&wrong); r <= w {
		t.Errorf("king in the bishop's corner scores %+d, in the wrong corner %+d; want the former higher", r, w)
	}
}

// TestEvaluateIsSideRelative checks the sign convention this shares with
// nnue.Evaluate: a score is always from the side to move's point of view.
func TestEvaluateIsSideRelative(t *testing.T) {
	white := mustFEN(t, "4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
	black := mustFEN(t, "4k3/8/8/8/8/8/8/R3K3 b - - 0 1")
	w, b := Evaluate(&white), Evaluate(&black)
	if w <= 0 {
		t.Errorf("Evaluate with the rook's owner to move = %+d, want positive", w)
	}
	if w != -b {
		t.Errorf("Evaluate = %+d with White to move but %+d with Black to move; want exact negation", w, b)
	}
}
