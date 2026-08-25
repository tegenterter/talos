package board

// castleRightsClear maps a square to the castling rights that are revoked
// when a king or rook leaves it, or when a rook is captured on it.
var castleRightsClear = map[Square]uint8{
	0:  WhiteQueenside,
	4:  WhiteKingside | WhiteQueenside,
	7:  WhiteKingside,
	56: BlackQueenside,
	60: BlackKingside | BlackQueenside,
	63: BlackKingside,
}

// MakeMove returns the board resulting from playing m on b. b is passed by
// value, so the caller's board is left untouched.
func MakeMove(b Board, m Move) Board {
	us := b.SideToMove
	them := us.Opposite()

	_, pt, ok := b.PieceAt(m.From)
	if !ok {
		// Every caller is expected to source m from GenerateLegalMoves, so
		// m.From is always occupied by the mover in practice; this exists
		// to fail loudly (rather than an opaque out-of-bounds panic
		// indexing Pieces[us][NoPiece] below) if that precondition is ever
		// violated by a future caller.
		panic("board: MakeMove called with no piece on m.From (" + m.From.String() + ")")
	}
	fromMask := sqBit(m.From)
	toMask := sqBit(m.To)
	isCapture := m.Flag == EnPassantCapture || b.ColorBB(them)&toMask != 0

	b.Pieces[us][pt] &^= fromMask

	if m.Flag == EnPassantCapture {
		// The captured pawn sits behind the destination square, not on it.
		capSq := m.To - 8
		if us == Black {
			capSq = m.To + 8
		}
		b.Pieces[them][Pawn] &^= sqBit(capSq)
	} else {
		for capPt := Pawn; capPt <= King; capPt++ {
			b.Pieces[them][capPt] &^= toMask
		}
	}

	placed := pt
	if m.Promotion != NoPiece {
		placed = m.Promotion
	}
	b.Pieces[us][placed] |= toMask

	switch m.Flag {
	case CastleKingside:
		rookFrom, rookTo := Square(7), Square(5)
		if us == Black {
			rookFrom, rookTo = 63, 61
		}
		b.Pieces[us][Rook] &^= sqBit(rookFrom)
		b.Pieces[us][Rook] |= sqBit(rookTo)
	case CastleQueenside:
		rookFrom, rookTo := Square(0), Square(3)
		if us == Black {
			rookFrom, rookTo = 56, 59
		}
		b.Pieces[us][Rook] &^= sqBit(rookFrom)
		b.Pieces[us][Rook] |= sqBit(rookTo)
	}

	b.CastlingRights &^= castleRightsClear[m.From]
	b.CastlingRights &^= castleRightsClear[m.To]

	if m.Flag == DoublePawnPush {
		if us == White {
			b.EnPassant = m.From + 8
		} else {
			b.EnPassant = m.From - 8
		}
	} else {
		b.EnPassant = NoSquare
	}

	if pt == Pawn || isCapture {
		b.HalfmoveClock = 0
	} else {
		b.HalfmoveClock++
	}
	if us == Black {
		b.FullmoveNumber++
	}
	b.SideToMove = them

	return b
}
