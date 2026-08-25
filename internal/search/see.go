package search

import "talos/internal/board"

// see (Static Exchange Evaluation) returns the net material result, in
// centipawns from the mover's perspective, of playing m and then letting
// both sides repeatedly recapture on m.To — each side always using its
// least valuable remaining attacker — for as long as doing so is actually
// profitable for whoever's turn it is to decide. b is the position before
// m is played.
//
// This is "static" (no real search): the greedy least-valuable-attacker
// order is provably optimal for a single-square exchange (using a cheaper
// piece first never leaves a side worse off, since it can still choose to
// stop later), so this reduces to a simple backward induction over one
// fixed sequence rather than a real minimax search. see_test.go's
// TestSeeMatchesBruteForce cross-checks that reduction against an actual
// minimax over every legal recapture, restricted to moves landing on the
// target square, using this codebase's real move generator — the
// strongest correctness check available for this kind of algorithm.
//
// A pawn reaching the back rank partway through the *simulated* recapture
// sequence (not just as m itself, already handled via m.Promotion) is
// valued as promoting to a queen too — both for what that capture nets
// (the promotion bonus, same as m's own) and for what it then risks (a
// queen, not a pawn, sits on the square afterward). Skipping this looked
// like a reasonable "not a full search" simplification at first — it's
// what materially simpler SEE implementations do — but see_test.go's
// TestSeeMatchesBruteForce promptly found a real position (an advanced
// pawn contesting a back-rank square alongside a rook) where ignoring it
// was wrong by over 800cp, which is large enough, and this scenario
// unusual enough, to be worth the modest extra bookkeeping below rather
// than documenting it away.
//
// Two remaining deliberate simplifications, standard for a "not a full
// search" SEE — real engines' SEE implementations, Stockfish's included,
// accept the same ones, since this needs to stay cheap enough to call on
// every capture during move ordering and quiescence:
//   - A king illegally "recapturing" into a square still attacked *is*
//     handled (see the loop below), since it's cheap to check and a real
//     (if uncommon) source of wrong answers otherwise.
//   - Pins are not: leastValuableAttacker doesn't know a piece is pinned
//     against its own king and so can't actually make the move — that
//     would need a legality check (does removing this specific piece
//     expose its own king?) for every candidate attacker, not just the
//     bitboard membership check attackersTo already does cheaply. A pinned
//     "defender" can therefore make see() overestimate how well a square
//     is defended (see_test.go's TestSeeMatchesBruteForce documents a
//     concrete case it found this way, and excludes positions containing
//     any pin from its fuzzing rather than either chasing full legality
//     awareness here or accepting a weaker cross-check everywhere else).
func see(b *board.Board, m board.Move) int {
	to := m.To

	captured := 0
	if m.Flag == board.EnPassantCapture {
		captured = pieceOrderValue[board.Pawn]
	} else if _, pt, ok := b.PieceAt(to); ok {
		captured = pieceOrderValue[pt]
	}

	_, mover, _ := b.PieceAt(m.From)
	moverValue := pieceOrderValue[mover]
	if m.Promotion != board.NoPiece {
		captured += pieceOrderValue[m.Promotion] - pieceOrderValue[board.Pawn]
		moverValue = pieceOrderValue[m.Promotion]
	}

	occ := b.OccupiedBB() &^ sqBit(m.From)
	if m.Flag == board.EnPassantCapture {
		// The captured pawn sits behind `to`, not on it: same file as
		// `to`, same rank as `from`.
		capSq := board.Square((int(m.From)/8)*8 + int(to)%8)
		occ &^= sqBit(capSq)
	}

	side := b.SideToMove.Opposite()
	gains := []int{captured}
	occupant := moverValue

	for {
		attackers := attackersTo(b, to, occ) & b.ColorBB(side) & occ
		if attackers == 0 {
			break
		}
		sq, pt, value := leastValuableAttacker(b, attackers, side, to)
		if pt == board.NoPiece {
			break
		}
		if pt == board.King {
			// A king can't recapture into a square still covered by the
			// opponent — that would leave it in check, which is illegal.
			remaining := attackersTo(b, to, occ&^sqBit(sq)) & b.ColorBB(side.Opposite()) &^ sqBit(sq)
			if remaining != 0 {
				break
			}
		}

		bonus := 0
		if pt == board.Pawn && promotes(side, to) {
			bonus = pieceOrderValue[board.Queen] - pieceOrderValue[board.Pawn]
		}
		gains = append(gains, occupant+bonus)
		occupant = value
		occ &^= sqBit(sq)
		side = side.Opposite()
	}

	// Backward induction over the simulated recapture sequence: at every
	// step from the *last* back to the *second* (index 1), whoever's turn
	// it is may always decline to recapture (net 0) rather than go
	// through with a capture that nets them less than that — hence the
	// max(0, ...) clamp. Index 0 is different: it's m itself, already
	// played (that's the whole premise of "SEE of this move"), so it is
	// not optional and must not be clamped — see gets to return negative
	// values, which is the entire point of calling it on a losing capture.
	net := 0
	for i := len(gains) - 1; i >= 1; i-- {
		net = gains[i] - net
		if net < 0 {
			net = 0
		}
	}
	return gains[0] - net
}

// attackersTo returns every square (either color) holding a piece that
// attacks sq, given occupancy occ — occ, not the live board, determines
// which sliding-piece attacks are blocked, so shrinking it as pieces are
// removed from a simulated exchange (see above) automatically reveals
// x-ray attackers behind them with no separate bookkeeping needed.
func attackersTo(b *board.Board, sq board.Square, occ board.Bitboard) board.Bitboard {
	knights := b.Pieces[board.White][board.Knight] | b.Pieces[board.Black][board.Knight]
	kings := b.Pieces[board.White][board.King] | b.Pieces[board.Black][board.King]
	bishopsQueens := b.Pieces[board.White][board.Bishop] | b.Pieces[board.Black][board.Bishop] |
		b.Pieces[board.White][board.Queen] | b.Pieces[board.Black][board.Queen]
	rooksQueens := b.Pieces[board.White][board.Rook] | b.Pieces[board.Black][board.Rook] |
		b.Pieces[board.White][board.Queen] | b.Pieces[board.Black][board.Queen]

	var attackers board.Bitboard
	attackers |= board.KnightAttacks(sq) & knights
	attackers |= board.KingAttacks(sq) & kings
	attackers |= board.PawnAttacks(board.Black, sq) & b.Pieces[board.White][board.Pawn]
	attackers |= board.PawnAttacks(board.White, sq) & b.Pieces[board.Black][board.Pawn]
	attackers |= board.BishopAttacks(sq, occ) & bishopsQueens
	attackers |= board.RookAttacks(sq, occ) & rooksQueens
	return attackers & occ
}

// leastValuableAttacker picks the cheapest piece among attackers (a
// bitboard of candidate squares, all belonging to side), the standard SEE
// ordering: a side should always use its least valuable piece first in an
// exchange, since it can still choose to stop the sequence later if that
// turns out badly.
//
// The comparison deliberately still ranks a pawn as cheap (its nominal
// pieceOrderValue) even when moving to `to` would promote it, rather than
// what it's really put at risk there (a queen's worth) — that would seem
// like the more careful choice, but it isn't actually more correct: a
// promoting capture also *wins* its promotion bonus immediately, on this
// exact turn, which a plain "what does this piece risk" comparison has no
// way to weigh against the risk. (Once promotion is possible, no fixed
// per-piece ordering is provably optimal the way least-valuable-first is
// without it — this codebase's real engines-do-this-too tradeoff is the
// same nominal-value ordering Stockfish's own SEE uses.) value, what the
// caller should book this piece as being worth if the opponent recaptures
// it, does account for promotion, since that part has no such ambiguity:
// whatever gets selected, if it promotes, a queen and not a pawn is what
// actually ends up on the square. pt is board.NoPiece if attackers was
// empty.
func leastValuableAttacker(b *board.Board, attackers board.Bitboard, side board.Color, to board.Square) (board.Square, board.PieceType, int) {
	bestSq := board.NoSquare
	bestType := board.NoPiece
	bestValue := 1 << 30

	for bb := attackers; bb != 0; bb &= bb - 1 {
		sq := bb.LSB()
		_, pt, ok := b.PieceAt(sq)
		if !ok {
			continue
		}
		if v := pieceOrderValue[pt]; v < bestValue {
			bestValue = v
			bestSq = sq
			bestType = pt
		}
	}
	if bestSq == board.NoSquare {
		return 0, board.NoPiece, 0
	}
	value := pieceOrderValue[bestType]
	if bestType == board.Pawn && promotes(side, to) {
		value = pieceOrderValue[board.Queen]
	}
	return bestSq, bestType, value
}

// promotes reports whether a side's pawn moving to sq would promote.
func promotes(side board.Color, sq board.Square) bool {
	if side == board.White {
		return sq >= board.Square(56) // rank 8
	}
	return sq < board.Square(8) // rank 1
}

func sqBit(sq board.Square) board.Bitboard { return board.Bitboard(1) << uint(sq) }
