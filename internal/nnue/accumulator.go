package nnue

import "talos/internal/board"

// Accumulator holds both perspectives' feature-transformer accumulators for
// one position — the sum of the weight rows of every active HalfKP feature,
// which is the expensive half of an evaluation.
//
// This is the "efficiently updatable" the E in NNUE refers to, and the one
// place this package trades its usual simplicity for speed. It earns it: a
// profile of the search put 30% of all CPU in rebuilding these from scratch
// at every leaf, and Update below reaches the same numbers by touching the
// handful of rows a single move actually changes.
//
// An Accumulator belongs to whoever holds it — internal/search keeps one per
// ply on each thread, exactly as it keeps its search path — and is never
// shared between goroutines.
type Accumulator struct {
	// acc is indexed by the perspective's colour, not by side to move: acc
	// [White] is always "the position as White's king sees it".
	acc [2][halfDimensions]int16
}

// Refresh recomputes both perspectives from scratch. Needed once at the root
// of a search, and again whenever a king moves (see Update).
func (n *Network) Refresh(a *Accumulator, b *board.Board) {
	n.accumulateInto(a.acc[board.White][:], b, board.White)
	n.accumulateInto(a.acc[board.Black][:], b, board.Black)
}

// Evaluate scores an already-accumulated position from stm's point of view,
// in the same centipawn convention as Network.Evaluate.
func (n *Network) EvaluateAcc(a *Accumulator, stm board.Color) int {
	return n.forwardPass(a.acc[stm][:], a.acc[stm.Opposite()][:])
}

// Update fills dst with the accumulators for the position reached by playing
// m from before, given src for before itself. after is the board m produces
// (the caller has it already; recomputing it here would undo the point).
//
// Only the features that actually changed are touched, with one exception
// that HalfKP forces: every feature is indexed against its own perspective's
// king square, so when a king moves, *that* perspective's entire accumulator
// re-indexes and has to be rebuilt. The other perspective is untouched by a
// king move, because kings are not features in HalfKP — a king selects which
// block of 641 slots the other pieces occupy rather than occupying one
// itself. Castling counts as a king move for the same reason, and also moves
// a rook, which the opponent's perspective does have to account for.
func (n *Network) Update(dst, src *Accumulator, before *board.Board, m board.Move, after *board.Board) {
	us := before.SideToMove
	_, moved, ok := before.PieceAt(m.From)
	if !ok {
		// Same precondition as board.MakeMove: m comes from move
		// generation, so this cannot happen unless a caller invents moves.
		panic("nnue: Update called with no piece on m.From (" + m.From.String() + ")")
	}

	for _, perspective := range [2]board.Color{board.White, board.Black} {
		if moved == board.King && perspective == us {
			// The mover's own king moved, so every feature index for this
			// perspective changed at once. Nothing to update incrementally.
			n.accumulateInto(dst.acc[perspective][:], after, perspective)
			continue
		}

		acc := &dst.acc[perspective]
		*acc = src.acc[perspective]
		kingSq := orient(perspective, after.Pieces[perspective][board.King].LSB())

		// The moving piece leaves From. A promoting pawn arrives as the
		// promoted piece instead of as a pawn.
		if moved != board.King {
			n.sub(acc, perspective, kingSq, us, moved, m.From)
			arrived := moved
			if m.Promotion != board.NoPiece {
				arrived = m.Promotion
			}
			n.add(acc, perspective, kingSq, us, arrived, m.To)
		}

		// A captured piece leaves the board. En passant takes the pawn from
		// behind the destination square rather than from it.
		switch {
		case m.Flag == board.EnPassantCapture:
			capSq := m.To - 8
			if us == board.Black {
				capSq = m.To + 8
			}
			n.sub(acc, perspective, kingSq, us.Opposite(), board.Pawn, capSq)
		default:
			if c, captured, ok := before.PieceAt(m.To); ok && c != us {
				n.sub(acc, perspective, kingSq, c, captured, m.To)
			}
		}

		// Castling moves a rook as well as the king. Only the opponent's
		// perspective reaches this — the mover's was rebuilt above — but the
		// rook is a feature there and has to move with it.
		switch m.Flag {
		case board.CastleKingside:
			n.moveRook(acc, perspective, kingSq, us, m.To+1, m.To-1)
		case board.CastleQueenside:
			n.moveRook(acc, perspective, kingSq, us, m.To-2, m.To+1)
		}
	}
}

// add folds one piece's weight row into the accumulator; sub removes it.
// Splitting them out keeps Update readable, and both inline.
func (n *Network) add(acc *[halfDimensions]int16, perspective board.Color, kingSq board.Square, c board.Color, pt board.PieceType, sq board.Square) {
	idx := featureIndex(perspective, kingSq, c, pt, sq)
	row := n.ftWeights[idx*halfDimensions : idx*halfDimensions+halfDimensions]
	for i, w := range row {
		acc[i] += w
	}
}

func (n *Network) sub(acc *[halfDimensions]int16, perspective board.Color, kingSq board.Square, c board.Color, pt board.PieceType, sq board.Square) {
	idx := featureIndex(perspective, kingSq, c, pt, sq)
	row := n.ftWeights[idx*halfDimensions : idx*halfDimensions+halfDimensions]
	for i, w := range row {
		acc[i] -= w
	}
}

func (n *Network) moveRook(acc *[halfDimensions]int16, perspective board.Color, kingSq board.Square, c board.Color, from, to board.Square) {
	n.sub(acc, perspective, kingSq, c, board.Rook, from)
	n.add(acc, perspective, kingSq, c, board.Rook, to)
}
