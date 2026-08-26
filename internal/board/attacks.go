package board

// Precomputed attack tables for non-sliding pieces, indexed by square.
var (
	knightAttacks [64]Bitboard
	kingAttacks   [64]Bitboard
	pawnAttacks   [2][64]Bitboard // pawnAttacks[color][sq] = squares that color's pawn on sq attacks
)

func init() {
	knightOffsets := [][2]int{{1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {1, -2}, {2, -1}, {-1, -2}, {-2, -1}}
	kingOffsets := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}

	for sq := 0; sq < 64; sq++ {
		file, rank := sq%8, sq/8
		knightAttacks[sq] = offsetsToBitboard(file, rank, knightOffsets)
		kingAttacks[sq] = offsetsToBitboard(file, rank, kingOffsets)

		// White pawns attack diagonally toward higher ranks, black toward lower.
		if rank < 7 {
			if file > 0 {
				pawnAttacks[White][sq] |= sqBit(Square(sq + 7))
			}
			if file < 7 {
				pawnAttacks[White][sq] |= sqBit(Square(sq + 9))
			}
		}
		if rank > 0 {
			if file > 0 {
				pawnAttacks[Black][sq] |= sqBit(Square(sq - 9))
			}
			if file < 7 {
				pawnAttacks[Black][sq] |= sqBit(Square(sq - 7))
			}
		}
	}
}

func offsetsToBitboard(file, rank int, offsets [][2]int) Bitboard {
	var bb Bitboard
	for _, o := range offsets {
		f, r := file+o[1], rank+o[0]
		if f >= 0 && f < 8 && r >= 0 && r < 8 {
			bb |= sqBit(Square(r*8 + f))
		}
	}
	return bb
}

// Ray directions as {deltaRank, deltaFile}.
var (
	rookDirs   = [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	bishopDirs = [][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
)

// slidingAttacks casts a ray from sq in each direction, stopping at and
// including the first occupied square (a potential capture).
func slidingAttacks(sq Square, occ Bitboard, dirs [][2]int) Bitboard {
	var attacks Bitboard
	file0, rank0 := int(sq)%8, int(sq)/8
	for _, d := range dirs {
		file, rank := file0+d[1], rank0+d[0]
		for file >= 0 && file < 8 && rank >= 0 && rank < 8 {
			s := Square(rank*8 + file)
			attacks |= sqBit(s)
			if occ&sqBit(s) != 0 {
				break
			}
			file += d[1]
			rank += d[0]
		}
	}
	return attacks
}

// bishopAttacks and rookAttacks are magic-bitboard lookups (magic.go), not
// ray walks. slidingAttacks above is what builds those tables and remains
// the definition of correctness they are tested against.
func bishopAttacks(sq Square, occ Bitboard) Bitboard { return bishopAttacksMagic(sq, occ) }
func rookAttacks(sq Square, occ Bitboard) Bitboard   { return rookAttacksMagic(sq, occ) }

// KnightAttacks returns the squares a knight on sq attacks.
func KnightAttacks(sq Square) Bitboard { return knightAttacks[sq] }

// KingAttacks returns the squares a king on sq attacks.
func KingAttacks(sq Square) Bitboard { return kingAttacks[sq] }

// BishopAttacks returns the squares a bishop on sq attacks given board occupancy occ.
func BishopAttacks(sq Square, occ Bitboard) Bitboard { return bishopAttacks(sq, occ) }

// RookAttacks returns the squares a rook on sq attacks given board occupancy occ.
func RookAttacks(sq Square, occ Bitboard) Bitboard { return rookAttacks(sq, occ) }

// PawnAttacks returns the squares a color's pawn on sq attacks.
func PawnAttacks(c Color, sq Square) Bitboard { return pawnAttacks[c][sq] }

// IsSquareAttacked reports whether any piece of byColor attacks sq.
func IsSquareAttacked(b *Board, sq Square, byColor Color) bool {
	if pawnAttacks[byColor.Opposite()][sq]&b.Pieces[byColor][Pawn] != 0 {
		return true
	}
	if knightAttacks[sq]&b.Pieces[byColor][Knight] != 0 {
		return true
	}
	if kingAttacks[sq]&b.Pieces[byColor][King] != 0 {
		return true
	}

	occ := b.OccupiedBB()
	bishopsQueens := b.Pieces[byColor][Bishop] | b.Pieces[byColor][Queen]
	if bishopAttacks(sq, occ)&bishopsQueens != 0 {
		return true
	}
	rooksQueens := b.Pieces[byColor][Rook] | b.Pieces[byColor][Queen]
	if rookAttacks(sq, occ)&rooksQueens != 0 {
		return true
	}
	return false
}
