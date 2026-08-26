// Package eval implements a small classical evaluator, used only for the
// positions internal/nnue's network is demonstrably unreliable on: a side
// overwhelmingly ahead against a pawnless opponent, i.e. an endgame that is
// won on technique rather than judgement.
//
// This is not a revival of the general-purpose classical evaluator this
// engine used before NNUE, and it must not grow into one — it has no
// piece-square tables, no mobility, no king safety, and it would play
// ordinary chess badly. What it has is the one thing the network lacks
// exactly where it matters: a gradient toward the win. Material alone is
// flat across every arrangement of a won endgame, so the score below adds
// mating-net terms (drive the defending king to the edge, or to a corner
// the bishop covers in KBN vs K, and walk the attacking king toward it) —
// the same terms every classical engine has carried since the 1980s, and
// the reason those engines convert KBN vs K while this one could not.
//
// Why the network needs replacing here rather than correcting: measured
// over the 355 positions of one real game, the network's evaluation of the
// same position with only the side to move flipped — which should be its
// near-exact negation — disagrees by an average of 83cp with 32 pieces on
// the board, 36cp with 8, and 674cp with 5. Balanced sparse endgames are
// fine (KR vs KR: 93cp, KP vs KP: 30cp); lopsided ones are not (KR vs K:
// 715cp, KQ vs K: 1099cp, KRB vs KB: 678cp). The regime, not the sparsity,
// is what breaks it: HalfKP was trained on positions from real games, which
// essentially never contain a bare lost king. Stockfish 12, which shipped
// this very network, had the same problem and solved it the same way, by
// falling back to its classical evaluator when material was lopsided
// (NNUEThreshold1 in its Eval::evaluate).
package eval

import "talos/internal/board"

// Piece values in centipawns, matching internal/search's own ordering
// values so a capture is worth the same to move ordering as it is here.
var pieceValue = [6]int{100, 320, 330, 500, 900, 0}

// lopsidedMargin is how far ahead the stronger side must be, in
// centipawns, before this evaluator takes over from the network. 400 is
// the queen-for-rook gap, so it covers every endgame won by a clear
// exchange or more (KQ vs KR, KR vs K, KRB vs KB, KBN vs K) while leaving
// balanced or nearly-balanced endings — where the network is measurably
// fine — to the network.
const lopsidedMargin = 400

// Lopsided reports whether b is in the regime this package exists for: one
// side ahead by lopsidedMargin or more, against an opponent with no pawns.
//
// The pawnless requirement is what keeps this narrow. A pawn is a promotion
// (and so a comeback) away from changing the material balance completely,
// which is exactly the kind of dynamic judgement this evaluator does not
// have and the network does; and a position where the losing side still has
// pawns is a position the network has seen thousands of in training.
// Lopsided is called at every leaf of the search, so it counts each side's
// material exactly once and answers from that.
func Lopsided(b *board.Board) bool {
	white, black := material(b, board.White), material(b, board.Black)

	weak, diff := board.Black, white-black
	if black > white {
		weak, diff = board.White, black-white
	}
	if b.Pieces[weak][board.Pawn] != 0 {
		return false
	}
	return diff >= lopsidedMargin
}

// Evaluate scores b in centipawns from the side to move's point of view,
// the same convention nnue.Evaluate uses.
//
// Note the two evaluators are not calibrated against one another: the
// network reports roughly +870 for the rook-and-bishop-against-bishop
// endgame this package was written for, where this returns closer to +590.
// A single move can therefore cross the boundary — capturing the defender's
// last pawn switches evaluators — and appear to lose a couple of hundred
// centipawns for free. Stockfish 12 lived with the same seam for the same
// reason, and the alternative (rescaling one evaluator to match the other)
// would be a fudge factor fitted to whichever positions were measured last.
func Evaluate(b *board.Board) int {
	score := evaluateWhite(b)
	if b.SideToMove == board.Black {
		return -score
	}
	return score
}

// evaluateWhite scores b from White's point of view.
func evaluateWhite(b *board.Board) int {
	if insufficientMaterial(b) {
		// Neither side can force mate, so no amount of material counting
		// describes this position: it is a draw whoever is "ahead".
		return 0
	}

	white, black := material(b, board.White), material(b, board.Black)
	score := white - black

	strong, diff := board.White, score
	if score < 0 {
		strong, diff = board.Black, -score
	}
	if diff < lopsidedMargin {
		// No mating net without something to mate with. Guarding on the
		// same margin Lopsided uses keeps this function honest if it is
		// ever called outside that regime — otherwise a dead-equal
		// position would hand White a bonus for nothing more than being
		// the first colour checked.
		return score
	}

	bonus := matingNet(b, strong)
	if strong == board.White {
		return score + bonus
	}
	return score - bonus
}

// material totals c's pieces in centipawns.
func material(b *board.Board, c board.Color) int {
	total := 0
	for pt := board.Pawn; pt <= board.Queen; pt++ {
		total += b.Pieces[c][pt].Count() * pieceValue[pt]
	}
	return total
}

// Mating-net weights, in centipawns per unit of the distance each term
// measures. They are large enough to steer a search that would otherwise
// see every arrangement as identical, and small enough that their total
// (at most 260cp outside the bishop-and-knight mate) never buys a piece:
// giving one up drops the position either out of Lopsided entirely or,
// against a bare king, into insufficientMaterial's zero.
//
// cornerWeight is the outlier at twelve times the others, and it can afford
// to be: it only ever applies to bishop-and-knight against a bare king,
// where material cannot change except by hanging a piece, and hanging one
// there scores an immediate zero (KB vs K and KN vs K are both drawn by
// insufficient material) rather than the mere 320cp it looks like. At the
// modest weight the other terms use, the attacker drove the defending king
// into whichever corner was nearest, parked it in the wrong one, and
// shuffled until the fifty-move rule ended it — measured, not imagined.
const (
	kingProximityWeight = 10  // per square the attacking king closes in
	edgeWeight          = 20  // per step the defending king is pushed outward
	cornerWeight        = 120 // per step toward a corner the bishop covers
)

// matingNet returns strong's positional bonus for making progress toward
// mate, always non-negative. It is what material alone cannot express:
// every arrangement of a won endgame counts the same material, so without
// these terms the search sees a flat plateau and picks arbitrarily among
// moves — which is exactly how a rook-up endgame becomes a fifty-move draw.
func matingNet(b *board.Board, strong board.Color) int {
	weak := strong.Opposite()
	strongKing := b.Pieces[strong][board.King].LSB()
	weakKing := b.Pieces[weak][board.King].LSB()

	// Walking the attacking king toward the defending one is half of every
	// mating technique there is; without it the attacker shuffles its rook
	// while its king stands on the far side of the board.
	bonus := kingProximityWeight * (14 - manhattan(strongKing, weakKing))

	if bishopKnightMate(b, strong) {
		// KBN vs K mates only in a corner the bishop covers, so the
		// defender has to be driven to a *specific* pair of corners.
		// Driving it to the nearest one instead is the classic way to
		// shuffle for fifty moves and draw a won game.
		return bonus + cornerWeight*bishopCornerDistance(weakKing, b.Pieces[strong][board.Bishop].LSB())
	}
	return bonus + edgeWeight*edgeProximity(weakKing)
}

// manhattan is the taxicab distance between two squares, the metric a king
// walk actually costs when it has to go around rather than through.
func manhattan(a, b board.Square) int {
	return abs(int(a%8)-int(b%8)) + abs(int(a/8)-int(b/8))
}

// edgeProximity scores how far sq is from the middle of the board: 0 on
// the four central squares, 3 in the middle of an edge, 6 in a corner. A
// lone king is mated on an edge and most easily in a corner, so pushing it
// outward is progress in every basic mate.
func edgeProximity(sq board.Square) int {
	file, rank := int(sq%8), int(sq/8)
	return (3 - min(file, 7-file)) + (3 - min(rank, 7-rank))
}

// bishopCornerDistance measures how far the defending king has been driven
// toward a corner the bishop can actually mate on, from 0 to 7.
//
// The trick — Stockfish's — is that the two dark corners (a1, h8) are
// exactly the two squares furthest from the a8-h1 anti-diagonal, so
// |7 - rank - file| is already the distance-to-the-right-corner measure,
// with no table and no special cases. A light-squared bishop mates on a8
// and h1 instead, which is the same measure after mirroring the king's
// file.
func bishopCornerDistance(weakKing, bishop board.Square) int {
	sq := weakKing
	if !darkSquared(bishop) {
		sq = flipFile(sq)
	}
	return abs(7 - int(sq/8) - int(sq%8))
}

// flipFile mirrors a square left-to-right (a1 <-> h1).
func flipFile(sq board.Square) board.Square { return sq ^ 7 }

// darkSquared reports whether sq is a dark square, using the a1-h8
// diagonal's colouring (a1 is dark).
func darkSquared(sq board.Square) bool {
	return (int(sq/8)+int(sq%8))%2 == 0
}

// bishopKnightMate reports whether strong holds exactly a bishop and a
// knight against a bare king, the one basic mate whose technique depends on
// which corner the defender is driven into.
func bishopKnightMate(b *board.Board, strong board.Color) bool {
	weak := strong.Opposite()
	if material(b, weak) != 0 {
		return false
	}
	return b.Pieces[strong][board.Bishop].Count() == 1 &&
		b.Pieces[strong][board.Knight].Count() == 1 &&
		b.Pieces[strong][board.Pawn] == 0 &&
		b.Pieces[strong][board.Rook] == 0 &&
		b.Pieces[strong][board.Queen] == 0
}

// InsufficientMaterial reports whether neither side can force mate, which
// makes the position a draw whatever the material count says.
//
// Exported separately from Evaluate because it has to apply everywhere, not
// just in this package's own regime: Lopsided deliberately ignores
// positions less than 400cp apart, which is every insufficient-material
// ending there is (a lone bishop is 330), so without this the network would
// go on scoring king-and-bishop against king as a comfortable plus. It did,
// and the search duly hung a bishop to reach one.
func InsufficientMaterial(b *board.Board) bool { return insufficientMaterial(b) }

// insufficientMaterial reports whether neither side can force mate, which
// makes the position a draw no matter what the material count says. A lone
// minor piece cannot mate at all; two knights cannot force mate (they can
// only mate against a defender who cooperates), and scoring either as a win
// is how an engine trades into a drawn ending believing it is winning.
func insufficientMaterial(b *board.Board) bool {
	for _, c := range [2]board.Color{board.White, board.Black} {
		if b.Pieces[c][board.Pawn] != 0 || b.Pieces[c][board.Rook] != 0 || b.Pieces[c][board.Queen] != 0 {
			return false
		}
	}
	for _, c := range [2]board.Color{board.White, board.Black} {
		bishops := b.Pieces[c][board.Bishop].Count()
		knights := b.Pieces[c][board.Knight].Count()
		switch {
		case bishops >= 2:
			return false // two bishops mate
		case bishops >= 1 && knights >= 1:
			return false // bishop and knight mate
		case knights >= 3:
			return false // three knights mate
		}
	}
	return true
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
