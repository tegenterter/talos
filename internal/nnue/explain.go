package nnue

import (
	"sort"

	"talos/internal/board"
)

// Contribution is one piece's share of a position's evaluation: what the
// score would lose if that piece vanished from the board, leaving everything
// else untouched.
//
// Because HalfKP indexes every piece relative to a king square (see
// featureIndex), DeltaCP is inherently king-relative: it answers "what is
// this piece worth *given where the kings are*", not "what is this piece
// worth in the abstract". That is a property of the feature set, not an
// approximation introduced here.
type Contribution struct {
	Color  board.Color
	Piece  board.PieceType
	Square board.Square
	// DeltaCP is Evaluate(b) - Evaluate(b without this piece), in
	// centipawns from the side to move's perspective. Positive means the
	// piece's presence helps the side to move.
	DeltaCP int
}

// Explanation is a per-piece breakdown of one position's evaluation.
//
// The decomposition is TotalCP = BaselineCP + sum(Contributions) + ResidualCP.
// See Network.Explain for why ResidualCP is necessarily nonzero.
type Explanation struct {
	// TotalCP is exactly Evaluate(b) — the score being explained.
	TotalCP int
	// BaselineCP is the network's output for a board holding only the two
	// kings (the feature transformer's biases alone, with no active
	// features). It is the zero point every contribution is measured
	// against.
	BaselineCP int
	// Contributions holds one entry per non-king piece, sorted by
	// descending |DeltaCP| so the pieces driving the evaluation come first.
	Contributions []Contribution
	// ResidualCP is the part of the evaluation that no single piece can be
	// credited with: TotalCP - BaselineCP - sum(Contributions). A large
	// residual is meaningful rather than an error — it means the pieces are
	// working together in ways a per-piece number cannot express.
	ResidualCP int
}

// Explain breaks b's evaluation down per piece, by leave-one-out ablation:
// for each non-king piece it recomputes the score with that piece's features
// removed and reports the difference.
//
// Two properties of this method are worth understanding before trusting its
// output:
//
// Kings are not attributable. In HalfKP the king is not a feature — it
// selects which 641-slot block every other feature is indexed into (see
// featureIndex). Removing a king would not remove a feature, it would
// invalidate every feature simultaneously, so there is no coherent "what is
// the king worth" question to ask of this network. Explanation covers
// non-king pieces only.
//
// Contributions do not sum to the total. clampFT and the two hidden layers'
// ClippedReLUs are nonlinear, so features are additive in accumulator space
// but never in centipawns. Whatever the per-piece deltas fail to account for
// is reported honestly as Explanation.ResidualCP rather than being silently
// distributed across the pieces.
//
// One further caveat: an ablated position is a counterfactual and may not be
// a legal chess position (removing a piece can leave a king in check, or
// strand a pawn on the first rank). That is fine here — the network is being
// asked about a set of features, not about a playable game state.
//
// Cost is roughly one accumulator build plus one forward pass per piece.
// Because accumulate sums weight rows, a piece is ablated by *subtracting*
// its two rows rather than rebuilding from scratch, which keeps the whole
// explanation O(pieces * halfDimensions) instead of O(pieces^2 *
// halfDimensions). Incremental accumulator updates are the "E" in NNUE,
// which this package deliberately forgoes for search (see the package doc);
// this is the one place the technique earns its keep, for explanation rather
// than for speed.
func (n *Network) Explain(b *board.Board) Explanation {
	stm := b.SideToMove
	opp := stm.Opposite()

	accSTM := n.accumulate(b, stm)
	accOpp := n.accumulate(b, opp)

	// Each perspective indexes features against its own king, so a single
	// piece has two different feature indices and two different weight rows
	// to remove.
	orientedKingSTM := orient(stm, b.Pieces[stm][board.King].LSB())
	orientedKingOpp := orient(opp, b.Pieces[opp][board.King].LSB())

	total := n.forwardPass(accSTM, accOpp)

	// Scratch accumulators reused across pieces: copy, subtract one piece's
	// rows, evaluate, restore. Avoids reallocating 2*256 int16 per piece.
	ablatedSTM := make([]int16, halfDimensions)
	ablatedOpp := make([]int16, halfDimensions)

	var contributions []Contribution
	sum := 0

	for _, pieceColor := range [2]board.Color{board.White, board.Black} {
		for pt := board.Pawn; pt <= board.Queen; pt++ {
			bb := b.Pieces[pieceColor][pt]
			for bb != 0 {
				sq := bb.PopLSB()

				copy(ablatedSTM, accSTM)
				copy(ablatedOpp, accOpp)
				subtractFeature(ablatedSTM, n, featureIndex(stm, orientedKingSTM, pieceColor, pt, sq))
				subtractFeature(ablatedOpp, n, featureIndex(opp, orientedKingOpp, pieceColor, pt, sq))

				delta := total - n.forwardPass(ablatedSTM, ablatedOpp)
				sum += delta
				contributions = append(contributions, Contribution{
					Color:   pieceColor,
					Piece:   pt,
					Square:  sq,
					DeltaCP: delta,
				})
			}
		}
	}

	sort.SliceStable(contributions, func(i, j int) bool {
		return abs(contributions[i].DeltaCP) > abs(contributions[j].DeltaCP)
	})

	// The zero point: no active features at all, i.e. the bare feature
	// transformer biases pushed through the output stack.
	baseline := n.forwardPass(n.ftBiases, n.ftBiases)

	return Explanation{
		TotalCP:       total,
		BaselineCP:    baseline,
		Contributions: contributions,
		ResidualCP:    total - baseline - sum,
	}
}

// subtractFeature removes one active feature's weight row from acc, undoing
// the addition accumulate performed for it.
func subtractFeature(acc []int16, n *Network, idx int) {
	row := n.ftWeights[idx*halfDimensions : idx*halfDimensions+halfDimensions]
	for i, w := range row {
		acc[i] -= w
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
