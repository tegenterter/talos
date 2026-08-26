package search

import "talos/internal/board"

// Continuation history: how good a move has been *given what was played
// immediately before it*.
//
// The plain history table (ordering.go) records that a move from square X to
// square Y has caused cutoffs, with no notion of context — the same entry is
// consulted whether the opponent just attacked something, traded, or shuffled
// a rook. Continuation history splits that by the previous move: "when a
// knight just landed on f3, how good is this bishop going to c5?" That is
// the pairing a plan is made of, and it is the standard answer to why
// ordering plateaus once killers and plain history are in.
//
// It exists here for a specific, measured reason. Reverse futility, late
// move and futility pruning were implemented, measured, and removed after
// scoring 0.504 over 354 games despite buying two plies of depth: they cut
// quiet moves for being *ordered late*, and this engine's ordering could not
// carry that weight (see golden_test.go's note). Pruning by move index is a
// bet on the ordering being right. This is the table that makes it a better
// bet.
//
// Indexed by (piece, destination) of both moves rather than from/to squares:
// what matters is which piece arrived where, not where it came from, and it
// keeps the table to a size a thread can own outright.
type continuationHistory [pieceIndexCount][64][pieceIndexCount][64]int16

// pieceIndexCount is colour × piece type — six kinds for each side.
const pieceIndexCount = 12

// pieceIndex packs a colour and piece type into one index.
func pieceIndex(c board.Color, pt board.PieceType) int {
	return int(c)*6 + int(pt)
}

// playedMove is what a ply contributes to its children's continuation
// lookups: which piece arrived, and where. ok is false at the root and after
// a null move, where there is no previous move to condition on.
type playedMove struct {
	piece int
	to    board.Square
	ok    bool
}

// contHistScore is the continuation bonus for playing m at ply, summed over
// the plies it conditions on (one and two moves back). Two is where the
// returns flatten for a table this size; Stockfish reaches further back
// because its tables are correspondingly bigger.
func (t *thread) contHistScore(b *board.Board, m board.Move, ply int) int {
	_, pt, ok := b.PieceAt(m.From)
	if !ok {
		return 0
	}
	idx := pieceIndex(b.SideToMove, pt)

	score := 0
	for _, back := range [2]int{1, 2} {
		if ply < back {
			continue
		}
		prev := t.playedStack[ply-back]
		if !prev.ok {
			continue
		}
		score += int(t.contHist[prev.piece][prev.to][idx][m.To])
	}
	return score
}

// updateContHistory credits (bonus > 0) or debits (bonus < 0) m in every
// continuation table that conditions on this ply.
func (t *thread) updateContHistory(b *board.Board, m board.Move, ply, bonus int) {
	_, pt, ok := b.PieceAt(m.From)
	if !ok {
		return
	}
	idx := pieceIndex(b.SideToMove, pt)

	for _, back := range [2]int{1, 2} {
		if ply < back {
			continue
		}
		prev := t.playedStack[ply-back]
		if !prev.ok {
			continue
		}
		// The same scale and the same gravity update as the plain table
		// (ordering.go's applyHistory), which is what makes the two
		// summable in moveOrderScore. Sweeping this table between
		// iterations the way the plain one used to be swept is also not an
		// option: at 589,824 entries per thread it cost 11% of nps on its
		// own, for a decay gravity does continuously and better.
		e := &t.contHist[prev.piece][prev.to][idx][m.To]
		v := int(*e)
		applyHistory(&v, bonus)
		*e = int16(v)
	}
}

// recordPlayed notes the move being made at ply, so the plies below it can
// condition on it.
func (t *thread) recordPlayed(b *board.Board, m board.Move, ply int) {
	if ply >= len(t.playedStack) {
		return
	}
	_, pt, ok := b.PieceAt(m.From)
	if !ok {
		t.playedStack[ply] = playedMove{}
		return
	}
	t.playedStack[ply] = playedMove{piece: pieceIndex(b.SideToMove, pt), to: m.To, ok: true}
}

// clearPlayed marks a ply as having no move to condition on — the null move,
// which reaches a position no move produced.
func (t *thread) clearPlayed(ply int) {
	if ply < len(t.playedStack) {
		t.playedStack[ply] = playedMove{}
	}
}
