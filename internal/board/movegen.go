package board

// GeneratePseudoLegalMoves generates all moves for the side to move that
// follow piece movement rules, without checking whether the move leaves
// the mover's own king in check.
func GeneratePseudoLegalMoves(b *Board) []Move {
	us := b.SideToMove
	them := us.Opposite()
	own := b.ColorBB(us)
	enemy := b.ColorBB(them)
	occ := own | enemy

	moves := make([]Move, 0, 32)

	generatePawnMoves(b, us, occ, enemy, &moves)

	knights := b.Pieces[us][Knight]
	for knights != 0 {
		from := knights.PopLSB()
		addTargets(&moves, from, knightAttacks[from]&^own)
	}

	kings := b.Pieces[us][King]
	for kings != 0 {
		from := kings.PopLSB()
		addTargets(&moves, from, kingAttacks[from]&^own)
	}

	bishops := b.Pieces[us][Bishop]
	for bishops != 0 {
		from := bishops.PopLSB()
		addTargets(&moves, from, bishopAttacks(from, occ)&^own)
	}

	rooks := b.Pieces[us][Rook]
	for rooks != 0 {
		from := rooks.PopLSB()
		addTargets(&moves, from, rookAttacks(from, occ)&^own)
	}

	queens := b.Pieces[us][Queen]
	for queens != 0 {
		from := queens.PopLSB()
		addTargets(&moves, from, (bishopAttacks(from, occ)|rookAttacks(from, occ))&^own)
	}

	generateCastlingMoves(b, us, occ, &moves)

	return moves
}

// GenerateLegalMoves filters pseudo-legal moves down to those that don't
// leave the mover's own king in check.
func GenerateLegalMoves(b *Board) []Move {
	us := b.SideToMove
	them := us.Opposite()
	pseudo := GeneratePseudoLegalMoves(b)
	legal := make([]Move, 0, len(pseudo))
	for _, m := range pseudo {
		after := MakeMove(*b, m)
		kingSq := after.Pieces[us][King].LSB()
		if !IsSquareAttacked(&after, kingSq, them) {
			legal = append(legal, m)
		}
	}
	return legal
}

func addTargets(moves *[]Move, from Square, targets Bitboard) {
	for targets != 0 {
		to := targets.PopLSB()
		*moves = append(*moves, Move{From: from, To: to, Promotion: NoPiece, Flag: Quiet})
	}
}

func generatePawnMoves(b *Board, us Color, occ, enemy Bitboard, moves *[]Move) {
	pawns := b.Pieces[us][Pawn]

	var forward Square
	var startRank, promoRank int
	if us == White {
		forward, startRank, promoRank = 8, 1, 7
	} else {
		forward, startRank, promoRank = -8, 6, 0
	}

	for pawns != 0 {
		from := pawns.PopLSB()
		rank := int(from) / 8

		to := from + forward
		if occ&sqBit(to) == 0 {
			addPawnMove(moves, from, to, promoRank, Quiet)

			if rank == startRank {
				to2 := to + forward
				if occ&sqBit(to2) == 0 {
					*moves = append(*moves, Move{From: from, To: to2, Promotion: NoPiece, Flag: DoublePawnPush})
				}
			}
		}

		targets := pawnAttacks[us][from]
		captures := targets & enemy
		for captures != 0 {
			capTo := captures.PopLSB()
			addPawnMove(moves, from, capTo, promoRank, Quiet)
		}
		if b.EnPassant != NoSquare && targets&sqBit(b.EnPassant) != 0 {
			*moves = append(*moves, Move{From: from, To: b.EnPassant, Promotion: NoPiece, Flag: EnPassantCapture})
		}
	}
}

func addPawnMove(moves *[]Move, from, to Square, promoRank int, flag MoveFlag) {
	if int(to)/8 == promoRank {
		for _, p := range [4]PieceType{Queen, Rook, Bishop, Knight} {
			*moves = append(*moves, Move{From: from, To: to, Promotion: p, Flag: flag})
		}
	} else {
		*moves = append(*moves, Move{From: from, To: to, Promotion: NoPiece, Flag: flag})
	}
}

// generateCastlingMoves adds castling moves, checking that castling rights
// are held, the squares between king and rook are empty, and the king
// does not start, pass through, or land on an attacked square.
func generateCastlingMoves(b *Board, us Color, occ Bitboard, moves *[]Move) {
	them := us.Opposite()
	if us == White {
		if b.CastlingRights&WhiteKingside != 0 && occ&(sqBit(5)|sqBit(6)) == 0 &&
			!IsSquareAttacked(b, 4, them) && !IsSquareAttacked(b, 5, them) && !IsSquareAttacked(b, 6, them) {
			*moves = append(*moves, Move{From: 4, To: 6, Promotion: NoPiece, Flag: CastleKingside})
		}
		if b.CastlingRights&WhiteQueenside != 0 && occ&(sqBit(1)|sqBit(2)|sqBit(3)) == 0 &&
			!IsSquareAttacked(b, 4, them) && !IsSquareAttacked(b, 3, them) && !IsSquareAttacked(b, 2, them) {
			*moves = append(*moves, Move{From: 4, To: 2, Promotion: NoPiece, Flag: CastleQueenside})
		}
	} else {
		if b.CastlingRights&BlackKingside != 0 && occ&(sqBit(61)|sqBit(62)) == 0 &&
			!IsSquareAttacked(b, 60, them) && !IsSquareAttacked(b, 61, them) && !IsSquareAttacked(b, 62, them) {
			*moves = append(*moves, Move{From: 60, To: 62, Promotion: NoPiece, Flag: CastleKingside})
		}
		if b.CastlingRights&BlackQueenside != 0 && occ&(sqBit(57)|sqBit(58)|sqBit(59)) == 0 &&
			!IsSquareAttacked(b, 60, them) && !IsSquareAttacked(b, 59, them) && !IsSquareAttacked(b, 58, them) {
			*moves = append(*moves, Move{From: 60, To: 58, Promotion: NoPiece, Flag: CastleQueenside})
		}
	}
}
