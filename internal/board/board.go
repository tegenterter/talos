// Package board implements a bitboard-based chess board representation,
// FEN parsing, and legal move generation.
package board

import "math/bits"

// Color identifies a side.
type Color int

const (
	White Color = iota
	Black
)

// Opposite returns the other color.
func (c Color) Opposite() Color { return White + Black - c }

// PieceType identifies a kind of piece, independent of color.
type PieceType int

const (
	Pawn PieceType = iota
	Knight
	Bishop
	Rook
	Queen
	King
	NoPiece
)

// Square is a board square index: 0 = a1, 7 = h1, 56 = a8, 63 = h8
// (rank-major, i.e. sq = rank*8 + file).
type Square int

// NoSquare marks the absence of a square, e.g. when there is no en passant target.
const NoSquare Square = -1

func (sq Square) String() string {
	file := byte('a' + sq%8)
	rank := byte('1' + sq/8)
	return string([]byte{file, rank})
}

// ParseSquare parses algebraic square notation such as "e4".
func ParseSquare(s string) (Square, bool) {
	if len(s) != 2 {
		return 0, false
	}
	file := s[0] - 'a'
	rank := s[1] - '1'
	if file > 7 || rank > 7 {
		return 0, false
	}
	return Square(int(rank)*8 + int(file)), true
}

// Bitboard is a 64-bit set of squares, one bit per square.
type Bitboard uint64

func sqBit(sq Square) Bitboard { return Bitboard(1) << sq }

// LSB returns the lowest-indexed set square in bb.
func (bb Bitboard) LSB() Square { return Square(bits.TrailingZeros64(uint64(bb))) }

// PopLSB clears and returns the lowest-indexed set square in bb.
func (bb *Bitboard) PopLSB() Square {
	sq := bb.LSB()
	*bb &= *bb - 1
	return sq
}

// Count returns the number of set squares in bb.
func (bb Bitboard) Count() int { return bits.OnesCount64(uint64(bb)) }

// Castling rights bits, matching FEN's KQkq order.
const (
	WhiteKingside uint8 = 1 << iota
	WhiteQueenside
	BlackKingside
	BlackQueenside
)

// Board is a complete chess position.
type Board struct {
	// Pieces[color][pieceType] is the bitboard of that piece's squares.
	Pieces         [2][6]Bitboard
	SideToMove     Color
	CastlingRights uint8
	EnPassant      Square
	HalfmoveClock  int
	FullmoveNumber int
}

// ColorBB returns the bitboard of all squares occupied by c.
func (b *Board) ColorBB(c Color) Bitboard {
	var bb Bitboard
	for pt := Pawn; pt <= King; pt++ {
		bb |= b.Pieces[c][pt]
	}
	return bb
}

// OccupiedBB returns the bitboard of all occupied squares.
func (b *Board) OccupiedBB() Bitboard { return b.ColorBB(White) | b.ColorBB(Black) }

// PieceAt returns the color and type of the piece on sq, if any.
func (b *Board) PieceAt(sq Square) (Color, PieceType, bool) {
	mask := sqBit(sq)
	for c := White; c <= Black; c++ {
		for pt := Pawn; pt <= King; pt++ {
			if b.Pieces[c][pt]&mask != 0 {
				return c, pt, true
			}
		}
	}
	return White, NoPiece, false
}
