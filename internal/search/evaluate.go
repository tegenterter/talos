package search

import (
	"talos/internal/board"
	"talos/internal/eval"
	"talos/internal/nnue"
)

// staticEval is the search's one entry point to a static evaluation. It
// picks the evaluator the position is actually suited to, then damps the
// result by the fifty-move clock. Every leaf in negamax and quiescence goes
// through here; mate and tablebase scores deliberately do not, since
// neither is an evaluation of a position's features.
func staticEval(b *board.Board) int {
	if eval.InsufficientMaterial(b) {
		// Neither side can force mate, so the position is drawn no matter
		// what either evaluator makes of the pieces on it. Checked ahead of
		// everything else because it is the one judgement that is not an
		// approximation.
		return 0
	}

	v := nnue.Evaluate(b)
	if eval.Lopsided(b) {
		// A won-on-technique endgame, where the network is measurably
		// unreliable and has no gradient toward the win — see internal/eval.
		v = eval.Evaluate(b)
	}
	return damp(v, b.HalfmoveClock)
}

// damp scales an evaluation down as the fifty-move clock runs up: the score
// stands at clock 0 and is halved by clock 100, so a position's worth fades
// as the draw claim approaches.
//
// The slope stops at half rather than running to zero, and that bound is
// load-bearing. Damping all the way to nothing was tried first (Stockfish's
// older classical-eval line, v * (100 - clock) / 100) and it makes material
// itself worthless near the wall: at clock 90 keeping a bishop is worth 33
// centipawns, so the engine shrugs and lets it go. Measured, not theorized —
// it threw the bishop away in a KBN vs K conversion and drew a mate it was
// otherwise walking toward. Modern Stockfish uses the bounded form for the
// same reason.
//
// This is the fix for a game this engine drew from a winning position, and
// the reason it matters is that the fifty-move rule was previously a cliff
// rather than a slope. negamax returns 0 once the clock reaches 100, and
// nothing else in the engine knew the clock existed — so a rook-up endgame
// scored +863 at clock 0, +863 at 40, +863 at 80, and 0 at 96, the instant
// the draw came within the search horizon. Shuffling cost nothing until it
// cost everything, and by then no move could help. With the score decaying,
// a line that resets the clock (a capture, a pawn move) is worth more than
// one that does not, so making progress becomes something the search
// prefers on its own rather than something it has to be told.
//
// The clock is deliberately absent from board.Board.Hash(), so two searches
// of the same position at different clocks share transposition table
// entries and can read back a score damped by the wrong amount. Stockfish
// has the same hole for the same reason (keying the table on the clock
// would splinter it, and the error is bounded and small next to what the
// damping buys).
func damp(v, halfmoveClock int) int {
	if halfmoveClock <= 0 {
		return v
	}
	if halfmoveClock > 100 {
		halfmoveClock = 100
	}
	return v * (200 - halfmoveClock) / 200
}
