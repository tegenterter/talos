package nnue

import "talos/internal/board"

// HalfKP features describe every non-king piece relative to one side's king,
// from that side's own point of view ("perspective"). Each perspective gets
// its own accumulator, computed over the feature set below; the two are
// concatenated later so the network can compare "my position" against
// "their position" (see forward.go).

// psOwn/psOpponent give each non-king piece type's base feature-index slot,
// depending on whether the piece belongs to the perspective doing the
// accounting ("own") or the other side ("opponent") — not on the piece's
// absolute board color. This matches Stockfish's HalfKP PieceToIndex table:
// a perspective's own pawn and the opposing pawn occupy distinct slots, so
// the network can learn asymmetric, side-relative patterns (e.g. "my passed
// pawn" vs "their passed pawn") from a single shared weight table.
//
// Slot values follow the classic HalfKP enum: PS_W_PAWN=1, PS_B_PAWN=65,
// PS_W_KNIGHT=129, PS_B_KNIGHT=193, ... in Pawn/Knight/Bishop/Rook/Queen
// order, each 64 apart (one 64-square block per piece slot) with the king
// itself excluded (it selects which feature block to use, via ksq, rather
// than being a feature).
var psOwn = [5]int{1, 129, 257, 385, 513}
var psOpponent = [5]int{65, 193, 321, 449, 577}

const psEnd = 641 // slots per king square block: 10 piece slots * 64 squares + 1

// orient reflects a square for Black's perspective (the board is viewed
// upside-down and mirrored, i.e. a1<->h8, so that both perspectives'
// features are expressed in the same "my back rank is at the bottom" frame
// a network trained once can share weights across). White's perspective is
// unchanged.
func orient(perspective board.Color, sq board.Square) board.Square {
	if perspective == board.White {
		return sq
	}
	return sq ^ 63
}

// featureIndex computes the HalfKP feature index for one non-king piece
// (color pieceColor, type pt, on square sq) as seen from perspective, given
// that perspective's own (already-oriented) king square.
func featureIndex(perspective board.Color, orientedKingSq board.Square, pieceColor board.Color, pt board.PieceType, sq board.Square) int {
	slot := psOpponent[pt]
	if pieceColor == perspective {
		slot = psOwn[pt]
	}
	return int(orient(perspective, sq)) + slot + psEnd*int(orientedKingSq)
}

// accumulate returns the perspective's raw (unclamped) accumulator: the
// feature transformer's biases plus the weight row of every active feature
// (every non-king piece on the board, from perspective's point of view).
//
// Explain (explain.go) needs an accumulator it can keep and modify, so this
// allocates one; the evaluation path calls accumulateInto instead, with a
// destination the caller owns, since allocating two of these per evaluation
// put several million allocations a second through the garbage collector.
func (n *Network) accumulate(b *board.Board, perspective board.Color) []int16 {
	acc := make([]int16, halfDimensions)
	n.accumulateInto(acc, b, perspective)
	return acc
}

// accumulateInto is accumulate writing into a caller-owned destination,
// which must be halfDimensions long.
func (n *Network) accumulateInto(acc []int16, b *board.Board, perspective board.Color) {
	copy(acc, n.ftBiases)

	orientedKingSq := orient(perspective, b.Pieces[perspective][board.King].LSB())

	for _, pieceColor := range [2]board.Color{board.White, board.Black} {
		for pt := board.Pawn; pt <= board.Queen; pt++ {
			bb := b.Pieces[pieceColor][pt]
			for bb != 0 {
				sq := bb.LSB()
				bb &= bb - 1

				idx := featureIndex(perspective, orientedKingSq, pieceColor, pt, sq)
				row := n.ftWeights[idx*halfDimensions : idx*halfDimensions+halfDimensions]
				for i, w := range row {
					acc[i] += w
				}
			}
		}
	}
}
