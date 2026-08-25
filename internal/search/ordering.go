package search

import (
	"sort"

	"talos/internal/board"
)

// pieceOrderValue gives simple, fixed piece values used only for move
// ordering and pruning decisions — see.go's exchange evaluator and the
// null-move zugzwang guard below — not real evaluation, which is
// internal/nnue's job; ordering just needs a rough "how valuable"
// ranking, not a precise one.
var pieceOrderValue = [6]int{100, 320, 330, 500, 900, 20000}

// isCapture reports whether m captures a piece. m is assumed legal, so a
// non-en-passant move landing on an occupied square must be landing on an
// enemy piece (move generation never allows landing on one's own piece).
func isCapture(b *board.Board, m board.Move) bool {
	if m.Flag == board.EnPassantCapture {
		return true
	}
	_, _, occupied := b.PieceAt(m.To)
	return occupied
}

// capturedValue returns the value of the piece m captures, using the same
// rough ordering-only values (pieceOrderValue) the rest of this file relies
// on elsewhere — not a claim of positional accuracy, just "how much
// material is on the table here." m is assumed to be a capture (see
// isCapture).
func capturedValue(b *board.Board, m board.Move) int {
	if m.Flag == board.EnPassantCapture {
		return pieceOrderValue[board.Pawn]
	}
	_, pt, _ := b.PieceAt(m.To)
	return pieceOrderValue[pt]
}

// hasNonPawnMaterial reports whether c has any piece besides pawns and
// king — used to skip null-move pruning in likely-zugzwang positions
// (bare king-and-pawn endgames), null-move's best-known failure mode.
func hasNonPawnMaterial(b *board.Board, c board.Color) bool {
	return b.Pieces[c][board.Knight] != 0 || b.Pieces[c][board.Bishop] != 0 ||
		b.Pieces[c][board.Rook] != 0 || b.Pieces[c][board.Queen] != 0
}

// totalNonPawnMaterial sums pieceOrderValue over every knight/bishop/rook/
// queen on the board, both colors — a quantitative companion to
// hasNonPawnMaterial's boolean check, used to scale quiescence's delta
// pruning margin by how much of the game's material is still on the board
// (see quiescence.go's deltaPruningMarginFor). hasNonPawnMaterial itself
// can't serve that purpose: it's a presence check, so it can't distinguish
// a dense middlegame from a bare-bones single-rook-per-side endgame, and
// the latter is exactly where a flat pruning margin was found to be unsafe.
func totalNonPawnMaterial(b *board.Board) int {
	total := 0
	for c := board.White; c <= board.Black; c++ {
		for pt := board.Knight; pt <= board.Queen; pt++ {
			total += b.Pieces[c][pt].Count() * pieceOrderValue[pt]
		}
	}
	return total
}

// startingNonPawnMaterial is totalNonPawnMaterial's value at the game's
// start (2 knights + 2 bishops + 2 rooks + 1 queen per side), used to
// normalize it into a 0..1 "how much of the opening's material is left"
// phase.
const startingNonPawnMaterial = 2 * (2*320 + 2*330 + 2*500 + 900) // 6400

// orderMoves returns moves sorted so the search examines its most
// promising candidates first — critical for alpha-beta, since a good move
// early lets subsequent siblings be pruned rather than fully searched. The
// input slice is not modified. The sort is stable, so moves sharing a
// score keep the move generator's natural order, which makes the ordering
// a pure function of the position with no randomness anywhere in it.
//
// This used to shuffle before sorting, so that under Lazy SMP each thread
// broke ties differently and explored a different line instead of
// duplicating its siblings. That diversity was bought twice over: it
// degraded every thread's ordering quality (the whole point of ordering is
// that position in the list is meaningful — randomizing it throws that
// away), and it made the search irreproducible, so no result could be
// regression-tested. Perturbing tie-breaks is a workaround for threads
// that share no work structure; it is given up here deliberately, to be
// replaced by splitting the search tree across threads directly.
func (t *thread) orderMoves(b *board.Board, moves []board.Move, ttMove board.Move, haveTTMove bool, ply int) []board.Move {
	scores := make([]int, len(moves))
	for i, m := range moves {
		scores[i] = t.moveOrderScore(b, m, ttMove, haveTTMove, ply)
	}

	idx := make([]int, len(moves))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool { return scores[idx[i]] > scores[idx[j]] })

	ordered := make([]board.Move, len(moves))
	for i, j := range idx {
		ordered[i] = moves[j]
	}
	return ordered
}

// Move-ordering score bands, highest priority first. Bands are spaced far
// enough apart that within-band scoring (SEE values, history counts) can
// never cross into the next band. Captures split into two far-apart
// bands by their see() value: a "good" (SEE >= 0) capture ranks above
// even killers, since it's usually at least as promising as a quiet move
// that merely worked well in a sibling node once before — but a "bad"
// (SEE < 0) capture ranks below every quiet move instead of alongside
// them, since a losing trade is usually a worse try than an ordinary
// quiet move, even an unproven one. It still gets searched, eventually,
// same as any other legal move — this only affects the order.
const (
	orderScoreTTMove      = 10_000_000
	orderScoreGoodCapture = 1_000_000 // + see(b, m), which is >= 0 here
	orderScoreKiller1     = 900_000
	orderScoreKiller2     = 890_000
	orderScoreBadCapture  = -1_000_000 // + see(b, m), which is < 0 here
)

func (t *thread) moveOrderScore(b *board.Board, m board.Move, ttMove board.Move, haveTTMove bool, ply int) int {
	if haveTTMove && m == ttMove {
		return orderScoreTTMove
	}
	if isCapture(b, m) {
		s := see(b, m)
		if s >= 0 {
			return orderScoreGoodCapture + s
		}
		return orderScoreBadCapture + s
	}
	if ply < maxPly {
		if t.killers[ply][0] == m {
			return orderScoreKiller1
		}
		if t.killers[ply][1] == m {
			return orderScoreKiller2
		}
	}
	return t.history[b.SideToMove][m.From][m.To]
}

// recordKiller notes that m caused a beta cutoff at ply, so sibling
// nodes at the same ply try it early too — cutoffs tend to recur at the
// same ply across different branches (e.g. "the response that refutes
// most tries here is putting the rook on the open file").
func (t *thread) recordKiller(ply int, m board.Move) {
	if ply >= maxPly || t.killers[ply][0] == m {
		return
	}
	t.killers[ply][1] = t.killers[ply][0]
	t.killers[ply][0] = m
}

// recordHistory strengthens m's ordering score after it caused a cutoff,
// weighted by depth² (a cutoff found deeper in the search — i.e. after
// surviving more scrutiny — says more about the move's general quality
// than one found near a leaf).
func (t *thread) recordHistory(color board.Color, m board.Move, depth int) {
	t.history[color][m.From][m.To] += depth * depth
}

// decayHistory halves every history score. t.history otherwise only grows
// (recordHistory never subtracts, and one thread's history table is reused
// across every iterative-deepening depth of a single Search call — see
// search.go), so a long-running search (e.g. "go infinite") could
// eventually push a frequently-cutting-off entry past the killer-move and
// good-capture score bands above, which assume history stays comfortably
// below them. Called once per completed depth to keep that from happening
// while still letting recent cutoffs dominate stale ones.
func (t *thread) decayHistory() {
	for c := range t.history {
		for from := range t.history[c] {
			for to := range t.history[c][from] {
				t.history[c][from][to] /= 2
			}
		}
	}
}
