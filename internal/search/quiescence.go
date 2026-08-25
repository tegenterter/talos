package search

import (
	"sort"

	"talos/internal/board"
	"talos/internal/nnue"
)

// maxQuiescencePly caps how much further quiescence can recurse beyond
// where it was entered, as a safety valve against pathological chains
// (e.g. repeated forcing checks) rather than a claim that quiescence
// should ever realistically need to go this deep.
const maxQuiescencePly = 16

// quiescence resolves tactical noise at the end of the main search: a
// plain static evaluation at an arbitrary cutoff depth is prone to the
// "horizon effect" (e.g. stopping right after a queen trade but before
// the recapture, evaluating a position that's about to lose a queen as
// fine) — so instead of returning nnue.Evaluate directly, this keeps
// playing out captures (and, if in check, every legal reply, since
// nothing is "quiet" while in check) until the position settles down.
// ply is the overall ply from the search root (for mate scoring); qply
// counts plies within this quiescence excursion (for maxQuiescencePly).
func (t *thread) quiescence(b *board.Board, ply, qply, alpha, beta int) int {
	if t.aborted {
		return 0
	}
	if n := t.s.nodes.Add(1); n&nodeCheckInterval == 0 {
		t.checkStop()
		if t.aborted {
			return 0
		}
	}
	if ply > t.selDepth {
		t.selDepth = ply
	}

	// Quiescence is only ever reached at ply > 0 (see negamax's depth<=0
	// dispatch), so — unlike negamax's own tablebase check — this needs
	// no ply guard. It matters here specifically: quiescence exists to
	// play out captures, which is exactly what keeps simplifying material
	// into tablebase-covered territory, so skipping this check here would
	// leave most tablebase hits undetected until the capture sequence
	// happened to end on a "quiet" move instead.
	if t.s.tablebase != nil {
		if wdl, ok := t.s.tablebase.Probe(b); ok {
			return tbScore(wdl, ply)
		}
	}

	kingSq := b.Pieces[b.SideToMove][board.King].LSB()
	inCheck := board.IsSquareAttacked(b, kingSq, b.SideToMove.Opposite())

	if qply > maxQuiescencePly {
		return nnue.Evaluate(b)
	}

	standPat := 0
	if !inCheck {
		standPat = nnue.Evaluate(b)
		if standPat >= beta {
			return standPat
		}
		if standPat > alpha {
			alpha = standPat
		}
	}

	var moves []board.Move
	if inCheck {
		moves = board.GenerateLegalMoves(b)
		if len(moves) == 0 {
			return -(mateValue - ply)
		}
	} else {
		all := board.GenerateLegalMoves(b)
		moves = make([]board.Move, 0, len(all))
		for _, m := range all {
			// A losing capture (see < 0) is dropped here rather than
			// searched: quiescence exists to resolve tactics, and a
			// capture that simply loses material practically never does
			// that, so paying to search it out isn't worth it. This is
			// the standard, if theoretically imperfect (see.go documents
			// where see() itself can be wrong), quiescence SEE pruning
			// every strong engine uses.
			if isCapture(b, m) && see(b, m) >= 0 {
				moves = append(moves, m)
			}
		}
		if len(moves) == 0 {
			return standPat
		}
	}
	sort.SliceStable(moves, func(i, j int) bool { return see(b, moves[i]) > see(b, moves[j]) })

	best := standPat
	if inCheck {
		best = -infinity // "do nothing" isn't an option while in check, so there's no stand-pat floor
	}
	for _, m := range moves {
		child := board.MakeMove(*b, m)
		score := -t.quiescence(&child, ply+1, qply+1, -beta, -alpha)
		if t.aborted {
			return 0
		}
		if score > best {
			best = score
		}
		if score > alpha {
			alpha = score
		}
		if alpha >= beta {
			break
		}
	}
	return best
}
