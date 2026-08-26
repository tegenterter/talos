package search

import (
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
	scores := t.scoreBuf[:len(moves)]
	for i, m := range moves {
		scores[i] = t.moveOrderScore(b, m, ttMove, haveTTMove, ply)
	}

	// Sorted in place, into the caller's own buffer, with a hand-written
	// insertion sort rather than sort.SliceStable: this ran at every node
	// and allocated three slices to do it (the scores, an index
	// permutation, and the reordered result), while a stable sort's output
	// is unique — so the same key produces the same permutation either way,
	// and the golden baselines prove it. Move lists are short (rarely past
	// 40) and already in memory, which is the regime insertion sort wins.
	//
	// Stability is not a nicety here: ties keep the move generator's natural
	// order, which is what makes the search reproducible (see the package
	// doc on determinism). The strict `<` below is what preserves it.
	for i := 1; i < len(moves); i++ {
		m, sc := moves[i], scores[i]
		j := i - 1
		for j >= 0 && scores[j] < sc {
			moves[j+1], scores[j+1] = moves[j], scores[j]
			j--
		}
		moves[j+1], scores[j+1] = m, sc
	}
	return moves
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
// A non-capturing promotion gets its own band rather than falling through to
// the history score, where it would be ordered among ordinary quiet moves
// despite typically being the strongest move on the board. A queen promotion
// wins roughly a queen outright, so it belongs above even a good capture;
// underpromotions sit just above quiets instead — almost always inferior to
// taking a queen, but a knight promotion occasionally forks or mates, so they
// must stay ahead of the quiet pack rather than being buried. (A promotion
// that is also a capture is scored by the capture bands below, whose see()
// already accounts for the promotion — see see.go.)
const (
	orderScoreTTMove      = 10_000_000
	orderScoreQueenPromo  = 2_000_000
	orderScoreGoodCapture = 1_000_000 // + see(b, m), which is >= 0 here
	orderScoreKiller1     = 900_000
	orderScoreKiller2     = 890_000
	orderScoreMinorPromo  = 880_000
	orderScoreBadCapture  = -1_000_000 // + see(b, m), which is < 0 here
)

func (t *thread) moveOrderScore(b *board.Board, m board.Move, ttMove board.Move, haveTTMove bool, ply int) int {
	if haveTTMove && m == ttMove {
		return orderScoreTTMove
	}
	if isCapture(b, m) {
		// SEE decides which band a capture lands in — winning or losing —
		// and capture history orders within it. SEE answers "does this
		// exchange win material", which is the first question and a static
		// one; capture history answers "has taking this piece with that
		// piece, on that square, actually been working", which SEE cannot
		// see and which separates captures it scores identically.
		s := see(b, m)
		if s >= 0 {
			return orderScoreGoodCapture + s + t.captureHistoryScore(b, m)/captureHistoryDivisor
		}
		return orderScoreBadCapture + s
	}
	if m.Promotion == board.Queen {
		return orderScoreQueenPromo
	}
	if m.Promotion != board.NoPiece {
		return orderScoreMinorPromo + pieceOrderValue[m.Promotion]
	}
	if ply < maxPly {
		if t.killers[ply][0] == m {
			return orderScoreKiller1
		}
		if t.killers[ply][1] == m {
			return orderScoreKiller2
		}
	}
	// A quiet move's score is its plain history plus what the continuation
	// tables make of it in this specific context (conthistory.go).
	//
	// The sum is allowed to go negative — a move the continuation tables
	// have learned is bad *here* should sort below one they know nothing
	// about, and clamping at zero throws that half of the signal away. It
	// is bounded on both sides only to keep the bands in this function
	// meaningful: a quiet move may not climb into the promotion band above,
	// nor sink into the losing-capture band below.
	score := t.history[b.SideToMove][m.From][m.To] + t.contHistScore(b, m, ply)
	if score > maxHistory {
		score = maxHistory
	}
	if score < minQuietScore {
		score = minQuietScore
	}
	return score
}

// minQuietScore floors a quiet move's ordering score well clear of
// orderScoreBadCapture, so even a thoroughly discredited quiet move is still
// tried before a capture that simply loses material.
const minQuietScore = orderScoreBadCapture / 2

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

// Both history tables — this one and the continuation tables in
// conthistory.go — share one scale, one bonus formula and one update rule.
// They did not at first, and that is precisely why continuation history did
// nothing when it landed: measured after a depth-8 search, plain history
// entries reached the hundreds while continuation entries sat at a median of
// -1 and a 90th percentile of 8, so summing them drowned the second in the
// first. A continuation table is far sparser by construction (589,824
// context-specific slots against 8,192), so each entry is updated far less
// often and needs the same bonus to reach a comparable magnitude.
//
// maxHistory also keeps the ordering bands in moveOrderScore meaningful: no
// history score can climb into the band above quiet moves, whatever the
// search does, so ordering cannot silently invert.
const maxHistory = 16384

// historyBonus is the credit a quiet move earns for causing a cutoff, and
// the debit one takes for having been searched and failed to. It grows with
// depth — a cutoff proven by a deeper search says more about the move than
// one found near a leaf — and is bounded so a single deep node cannot
// saturate an entry on its own.
func historyBonus(depth int) int {
	bonus := 16*depth*depth + 32*depth + 16
	if bonus > maxHistoryBonus {
		bonus = maxHistoryBonus
	}
	return bonus
}

const maxHistoryBonus = 1200

// applyHistory is the shared update: history gravity, pulling each update
// toward zero in proportion to how far the entry has already travelled. An
// entry approaches the bound asymptotically instead of hitting it, and old
// information fades continuously as new arrives — decay done per update
// rather than by sweeping the table between iterations.
func applyHistory(entry *int, bonus int) {
	v := *entry
	*entry = v + bonus - v*abs(bonus)/maxHistory
}

// recordHistory strengthens m's ordering score after it caused a cutoff.
func (t *thread) recordHistory(color board.Color, m board.Move, depth int) {
	applyHistory(&t.history[color][m.From][m.To], historyBonus(depth))
}

// penalizeHistory is recordHistory's opposite, applied to the quiet moves
// that were searched at a node and failed to cut off while a later one
// succeeded. Without it the table records only "this move cut off
// somewhere", conflating a move that cuts off almost every time with one
// that cut off once and has been tried fruitlessly ever since.
func (t *thread) penalizeHistory(color board.Color, m board.Move, depth int) {
	applyHistory(&t.history[color][m.From][m.To], -historyBonus(depth))
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

// abs is the small helper the gravity update needs.
func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// Capture history: how well capturing a given piece with a given piece, on
// a given square, has actually worked.
//
// It sits underneath SEE rather than replacing it. SEE is static and exact
// about material; this is empirical and about everything else — a capture
// that wins a pawn but walks into a bind scores the same as one that wins a
// pawn cleanly, until the search has been burnt by it a few times.
//
// Indexed by (moving piece, destination, captured piece type), which is the
// smallest key that distinguishes the cases that matter.
type captureHistoryTable [pieceIndexCount][64][6]int

// captureHistoryDivisor scales the table into the space between two SEE
// values, so capture history reorders captures SEE rates equally without
// ever reordering ones it does not.
const captureHistoryDivisor = 16

func (t *thread) captureHistoryScore(b *board.Board, m board.Move) int {
	idx, captured, ok := captureKey(b, m)
	if !ok {
		return 0
	}
	return t.captureHist[idx][m.To][captured]
}

func (t *thread) updateCaptureHistory(b *board.Board, m board.Move, bonus int) {
	idx, captured, ok := captureKey(b, m)
	if !ok {
		return
	}
	applyHistory(&t.captureHist[idx][m.To][captured], bonus)
}

// captureKey builds the table index for a capture, reporting false for a
// move that captures nothing (an en passant capture takes a pawn, but from a
// square other than the destination, so it is keyed as a pawn capture).
func captureKey(b *board.Board, m board.Move) (idx int, captured int, ok bool) {
	_, pt, found := b.PieceAt(m.From)
	if !found {
		return 0, 0, false
	}
	if m.Flag == board.EnPassantCapture {
		return pieceIndex(b.SideToMove, pt), int(board.Pawn), true
	}
	_, capturedPt, occupied := b.PieceAt(m.To)
	if !occupied {
		return 0, 0, false
	}
	return pieceIndex(b.SideToMove, pt), int(capturedPt), true
}
