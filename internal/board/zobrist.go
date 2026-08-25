package board

import "math/rand"

// Zobrist keys, one random 64-bit value per (color, piece type, square),
// per castling right, per en passant file, and one for side-to-move. A
// position's hash is the XOR of the keys for everything true about it;
// XOR's self-cancelling property is what makes updating (or, here,
// recomputing) a hash from these keys correct and order-independent.
var (
	zobristPieces        [2][6][64]uint64
	zobristCastling      [4]uint64 // WhiteKingside, WhiteQueenside, BlackKingside, BlackQueenside
	zobristEnPassantFile [8]uint64
	zobristBlackToMove   uint64
)

// zobristSeed fixes the key set so a given position hashes to the same
// value in every process, not just within one run. The keys only need to be
// well-distributed, never secret or varying, and real engines commonly ship
// a hardcoded table (Polyglot's is part of its format). Drawing them from
// math/rand's *global* source instead would re-randomize them on every
// start — Go auto-seeds that source — which makes transposition-table slot
// assignments, collision patterns, and therefore search node counts differ
// run to run, so no search result can be reproduced or regression-tested
// across processes.
const zobristSeed int64 = 0x1D39247E33776D41

func init() {
	r := rand.New(rand.NewSource(zobristSeed))
	for c := 0; c < 2; c++ {
		for pt := 0; pt < 6; pt++ {
			for sq := 0; sq < 64; sq++ {
				zobristPieces[c][pt][sq] = r.Uint64()
			}
		}
	}
	for i := range zobristCastling {
		zobristCastling[i] = r.Uint64()
	}
	for i := range zobristEnPassantFile {
		zobristEnPassantFile[i] = r.Uint64()
	}
	zobristBlackToMove = r.Uint64()
}

// Hash returns a Zobrist hash of the position. It deliberately excludes
// HalfmoveClock and FullmoveNumber: those affect draw-rule bookkeeping,
// not what moves are legal or how the position should be evaluated, so
// two positions that are otherwise identical hash identically regardless
// of the move order used to reach them (a "transposition") — which is
// exactly the property a hash-based cache needs to be useful.
func (b *Board) Hash() uint64 {
	var h uint64
	for c := White; c <= Black; c++ {
		for pt := Pawn; pt <= King; pt++ {
			bb := b.Pieces[c][pt]
			for bb != 0 {
				sq := bb.PopLSB()
				h ^= zobristPieces[c][pt][sq]
			}
		}
	}
	if b.CastlingRights&WhiteKingside != 0 {
		h ^= zobristCastling[0]
	}
	if b.CastlingRights&WhiteQueenside != 0 {
		h ^= zobristCastling[1]
	}
	if b.CastlingRights&BlackKingside != 0 {
		h ^= zobristCastling[2]
	}
	if b.CastlingRights&BlackQueenside != 0 {
		h ^= zobristCastling[3]
	}
	if b.EnPassant != NoSquare && enPassantCapturable(b) {
		h ^= zobristEnPassantFile[b.EnPassant%8]
	}
	if b.SideToMove == Black {
		h ^= zobristBlackToMove
	}
	return h
}

// enPassantCapturable reports whether a pawn of the side to move actually
// sits beside the square the just-doubled-pushed pawn landed on, i.e.
// whether an en passant capture is at least pseudo-legal here. Without this
// check, two positions differing only in a "phantom" en passant square (one
// no pawn can actually use — reachable via a hand-written FEN, since
// MakeMove only ever sets EnPassant right after a real double push, which
// always leaves a capturing pawn adjacent by construction) would hash
// differently even though they're otherwise identical — the standard
// Zobrist convention (and Stockfish's own) avoids that by only folding the
// en passant file into the hash when a capture is actually available.
func enPassantCapturable(b *Board) bool {
	landing := b.EnPassant + 8
	if b.SideToMove == White {
		landing = b.EnPassant - 8
	}
	pawns := b.Pieces[b.SideToMove][Pawn]
	file := landing % 8
	if file > 0 && pawns&sqBit(landing-1) != 0 {
		return true
	}
	if file < 7 && pawns&sqBit(landing+1) != 0 {
		return true
	}
	return false
}
