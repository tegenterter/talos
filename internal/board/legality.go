package board

// Legality filtering without playing the move.
//
// The obvious way to decide whether a pseudo-legal move is legal is to play
// it on a copy of the board and ask whether the mover's king is attacked —
// which is exactly what this package did, and it was 16% of the engine's
// total CPU: a board copy and a full attack scan for every candidate move at
// every node. Almost all of that work is redundant, because only three kinds
// of move can possibly expose your own king: a king move, a move by a pinned
// piece, and an en passant capture. Everything else is legal unless you are
// already in check, and when you are, the reply must capture or block the
// one piece giving it.
//
// So the state needed is computed once per node — who is checking the king,
// and which of our pieces are pinned to it — and each move is then a couple
// of bitboard tests. See isLegal below.

// betweenBB[a][b] holds the squares strictly between a and b when the two
// are on a common rank, file or diagonal, and is empty otherwise. lineBB[a]
// [b] holds the whole line through them, endpoints included. Both are the
// standard tables every move generator needs for pins and check evasions,
// and both are small enough (32KB each) to just precompute at startup.
var (
	betweenBB [64][64]Bitboard
	lineBB    [64][64]Bitboard
)

func init() {
	dirs := [8][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
	for from := 0; from < 64; from++ {
		for _, d := range dirs {
			forward := ray(Square(from), d)
			backward := ray(Square(from), [2]int{-d[0], -d[1]})
			line := forward | backward | sqBit(Square(from))

			var between Bitboard
			f, r := from%8+d[0], from/8+d[1]
			for f >= 0 && f < 8 && r >= 0 && r < 8 {
				to := Square(r*8 + f)
				betweenBB[from][to] = between
				lineBB[from][to] = line
				between |= sqBit(to)
				f, r = f+d[0], r+d[1]
			}
		}
	}
}

// ray returns every square from sq in direction d, excluding sq itself and
// ignoring occupancy.
func ray(sq Square, d [2]int) Bitboard {
	var bb Bitboard
	f, r := int(sq)%8+d[0], int(sq)/8+d[1]
	for f >= 0 && f < 8 && r >= 0 && r < 8 {
		bb |= sqBit(Square(r*8 + f))
		f, r = f+d[0], r+d[1]
	}
	return bb
}

// attackersTo returns byColor's pieces that attack sq, given occ as the
// occupancy sliders see. Taking occ as a parameter rather than reading the
// board's own is what lets a king move be tested with the king removed —
// otherwise a checking slider looks blocked by the very king it is chasing,
// and stepping one square further along its ray would look safe.
func attackersTo(b *Board, sq Square, byColor Color, occ Bitboard) Bitboard {
	return pawnAttacks[byColor.Opposite()][sq]&b.Pieces[byColor][Pawn] |
		knightAttacks[sq]&b.Pieces[byColor][Knight] |
		kingAttacks[sq]&b.Pieces[byColor][King] |
		bishopAttacks(sq, occ)&(b.Pieces[byColor][Bishop]|b.Pieces[byColor][Queen]) |
		rookAttacks(sq, occ)&(b.Pieces[byColor][Rook]|b.Pieces[byColor][Queen])
}

// pinnedTo returns us's pieces that are pinned against their own king: the
// single piece standing between the king and an enemy slider aimed at it.
// Such a piece may still move, but only along the pinning line.
func pinnedTo(b *Board, kingSq Square, us Color, occ Bitboard) Bitboard {
	them := us.Opposite()
	// Sliders that would be attacking the king if nothing were in the way —
	// attacks computed against an empty board, i.e. the raw rays.
	snipers := bishopAttacks(kingSq, 0)&(b.Pieces[them][Bishop]|b.Pieces[them][Queen]) |
		rookAttacks(kingSq, 0)&(b.Pieces[them][Rook]|b.Pieces[them][Queen])

	var pinned Bitboard
	ours := b.ColorBB(us)
	for snipers != 0 {
		sq := snipers.PopLSB()
		blockers := betweenBB[kingSq][sq] & occ
		// Exactly one piece in the way, and it is ours: that is a pin. Two
		// or more and nothing is pinned; one enemy piece and the pin is on
		// them, which is not this function's business.
		if blockers.Count() == 1 && blockers&ours != 0 {
			pinned |= blockers
		}
	}
	return pinned
}

// isLegal reports whether a pseudo-legal move is legal, given the node-wide
// facts gathered once by GenerateLegalMoves.
func isLegal(b *Board, m Move, us Color, kingSq Square, occ, checkers, pinned Bitboard) bool {
	them := us.Opposite()

	// En passant is the only move that takes two pieces off one rank at
	// once (the capturing pawn leaves, the captured pawn vanishes beside
	// it), so it can uncover a rook or queen along that rank in a way none
	// of the tests below can see. It is rare enough that playing it out and
	// asking is cheaper than a special case that has to be right.
	if m.Flag == EnPassantCapture {
		after := MakeMove(*b, m)
		return !IsSquareAttacked(&after, after.Pieces[us][King].LSB(), them)
	}

	if m.From == kingSq {
		// Castling already had every square it touches checked at
		// generation time (see generateCastlingMoves).
		if m.Flag == CastleKingside || m.Flag == CastleQueenside {
			return true
		}
		// The king may not step onto an attacked square. Its own square is
		// cleared from the occupancy first: a slider checking it right now
		// must not appear blocked when the king slides further down that
		// same ray.
		return attackersTo(b, m.To, them, occ&^sqBit(m.From)) == 0
	}

	// A pinned piece may move only along the line it is pinned on — which
	// includes capturing the pinner.
	if pinned&sqBit(m.From) != 0 && lineBB[kingSq][m.From]&sqBit(m.To) == 0 {
		return false
	}

	switch {
	case checkers == 0:
		return true
	case checkers.Count() > 1:
		// Nothing but a king move escapes a double check: blocking or
		// capturing deals with one checker and leaves the other.
		return false
	default:
		// One checker: capture it, or step into the line between it and the
		// king. (A knight or pawn check has no line, so betweenBB is empty
		// and only the capture is offered — which is correct.)
		checker := checkers.LSB()
		return m.To == checker || betweenBB[kingSq][checker]&sqBit(m.To) != 0
	}
}
