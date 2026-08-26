package search

import (
	"talos/internal/board"
)

// maxQuiescencePly caps how much further quiescence can recurse beyond
// where it was entered, as a safety valve against pathological chains
// (e.g. repeated forcing checks) rather than a claim that quiescence
// should ever realistically need to go this deep.
const maxQuiescencePly = 16

// deltaPruningMarginMin/Max bound the cp headroom given to a capture's raw
// material value, to account for promotion/positional upside the value
// alone misses, before delta pruning gives up on it — see
// deltaPruningMarginFor and the pruning check below. Vars, not consts, so
// tests can override them.
//
// The margin is phase-scaled rather than flat because of a real, measured
// interaction with aspiration windows (search.go): delta pruning's
// decision depends on alpha, which aspiration windows can hand down
// pre-elevated (close to the previous iteration's score) well before this
// node's own move exploration has "earned" that alpha the way a full
// -infinity..infinity search would. On most positions that just changes
// node counts, but on at least one tactically sharp, materially sparse
// pawn endgame in this package's own golden-position set
// (golden_test.go's "pawn-endgame"), a flat 200cp margin measurably
// discarded a capture that mattered, moving the final reported score by
// several hundred centipawns — verified empirically (not derived
// analytically) by sweeping margin values against that position with
// aspiration windows enabled: 200-700 produced inconsistent,
// sometimes-wrong scores, while 800 and everything tested above it
// matched the no-delta-pruning baseline exactly. A single flat margin
// therefore had to be either too loose to be safe everywhere, or (at 800)
// safe but conservative enough to prune almost nothing in an ordinary
// middlegame — where the overwhelming majority of a real game's nodes are
// actually spent.
//
// deltaPruningMarginMin is close to a standard engine's delta-pruning
// margin, used when most material is still on the board; Max is the flat
// value already verified safe on the sparsest, highest-risk endgames.
// deltaPruningMarginFor interpolates between them by remaining material,
// so ordinary middlegame play gets a standard margin's efficiency while
// the specific regime that was demonstrated to be risky keeps the
// conservative one.
var deltaPruningMarginMin = 200
var deltaPruningMarginMax = 800

// deltaPruningMarginFor scales the delta-pruning margin by remaining
// non-pawn material on b: deltaPruningMarginMin in a dense middlegame,
// rising linearly toward deltaPruningMarginMax as material drains toward
// a bare king-and-pawns endgame. See deltaPruningMarginMin/Max's doc
// comment for why a flat margin isn't safe across that whole range.
func deltaPruningMarginFor(b *board.Board) int {
	phase := float64(totalNonPawnMaterial(b)) / float64(startingNonPawnMaterial)
	if phase > 1 {
		phase = 1
	}
	return deltaPruningMarginMin + int((1-phase)*float64(deltaPruningMarginMax-deltaPruningMarginMin))
}

// deltaPruningEnabled exists only so delta_test.go can measure delta
// pruning's actual effect (compare node counts, and confirm a
// near-the-margin tactic still resolves correctly, with it on vs. off) —
// there's no UCI option or other production path that ever sets it false.
var deltaPruningEnabled = true

// quiescence resolves tactical noise at the end of the main search: a
// plain static evaluation at an arbitrary cutoff depth is prone to the
// "horizon effect" (e.g. stopping right after a queen trade but before
// the recapture, evaluating a position that's about to lose a queen as
// fine) — so instead of returning a static evaluation directly, this keeps
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

	// Quiescence probes the same table the main search fills, and it is the
	// single cheapest thing available here: this is where two thirds of all
	// nodes are spent, every one of them evaluating, and an entry left by an
	// earlier (often deeper) visit answers both questions at once — whether
	// this node can be cut off outright, and what the position evaluates to
	// if it cannot.
	hash := b.Hash()
	alphaOrig := alpha
	var ttMove board.Move
	haveTTMove := false
	cachedEval := noEval
	if entry, ok := t.s.tt.probe(hash, ply); ok {
		ttMove = entry.move
		cachedEval = entry.eval
		haveTTMove = entry.move != (board.Move{})
		// The eval above is undamped and used as-is; the score is rescaled
		// to this node's clock (see ttEntry.scoreAt). Any
		// stored depth is fine here: quiescence stores at depth 0, and
		// anything the main search stored was searched deeper still, so its
		// bound is at least as trustworthy as one computed at this node.
		score := entry.scoreAt(b.HalfmoveClock)
		switch entry.flag {
		case ttExact:
			return score
		case ttLowerBound:
			if score >= beta {
				return score
			}
		case ttUpperBound:
			if score <= alpha {
				return score
			}
		}
	}

	kingSq := b.Pieces[b.SideToMove][board.King].LSB()
	inCheck := board.IsSquareAttacked(b, kingSq, b.SideToMove.Opposite())

	// The raw (undamped) evaluation, computed once and reused: it is the
	// stand-pat score below and the value stored back into the table.
	raw := cachedEval
	if raw == noEval {
		raw = t.rawEval(b, ply)
	}

	if qply > maxQuiescencePly {
		return damp(raw, b.HalfmoveClock)
	}

	standPat := 0
	if !inCheck {
		standPat = damp(raw, b.HalfmoveClock)
		if standPat >= beta {
			t.s.tt.store(hash, board.Move{}, standPat, 0, ttLowerBound, ply, raw, b.HalfmoveClock)
			return standPat
		}
		if standPat > alpha {
			alpha = standPat
		}
	}

	var moves []board.Move
	if inCheck {
		moves = board.GenerateLegalMovesInto(b, t.moveBuf[ply][:])
		if len(moves) == 0 {
			return -(mateValue - ply)
		}
	} else {
		all := board.GenerateLegalMovesInto(b, t.moveBuf[ply][:])
		// Compacted in place into the same buffer: the kept moves are
		// always a prefix of the ones already examined, so writing over
		// them is safe and saves a second list per node.
		moves = all[:0]
		// Computed once per node, not per candidate move — see
		// deltaPruningMarginFor's doc comment for what it scales by.
		deltaMargin := deltaPruningMarginFor(b)
		for _, m := range all {
			// A losing capture (see < 0) is dropped here rather than
			// searched: quiescence exists to resolve tactics, and a
			// capture that simply loses material practically never does
			// that, so paying to search it out isn't worth it. This is
			// the standard, if theoretically imperfect (see.go documents
			// where see() itself can be wrong), quiescence SEE pruning
			// every strong engine uses.
			if !isCapture(b, m) || see(b, m) < 0 {
				continue
			}
			// Delta pruning: even winning the captured piece outright, on
			// top of the current static evaluation, can't reach alpha —
			// so no positional gain this capture might also carry (an
			// unstoppable follow-up threat, a passed pawn's promotion
			// path opening up) is worth searching to find out. deltaMargin
			// is the (material-phase-scaled) allowance for that upside;
			// the SEE check above already screens out captures that lose
			// material outright, so this only screens out captures that,
			// even though they don't lose material, aren't big enough to
			// matter at this node.
			if deltaPruningEnabled && standPat+capturedValue(b, m)+deltaMargin < alpha {
				continue
			}
			moves = append(moves, m)
		}
		if len(moves) == 0 {
			return standPat
		}
	}
	// Ordered by SEE, computed once per move rather than once per
	// comparison: the previous sort called see() inside the comparator, so
	// each move's exchange was resolved O(log n) times per node. The TT move
	// sorts above everything, as it does in the main search.
	//
	// Stable insertion sort for the same reason orderMoves uses one (see
	// ordering.go): equal scores keep the generator's order, which is what
	// keeps the search reproducible.
	scores := t.scoreBuf[:len(moves)]
	for i, m := range moves {
		scores[i] = see(b, m)
		if haveTTMove && m == ttMove {
			scores[i] = orderScoreTTMove
		}
	}
	for i := 1; i < len(moves); i++ {
		m, sc := moves[i], scores[i]
		j := i - 1
		for j >= 0 && scores[j] < sc {
			moves[j+1], scores[j+1] = moves[j], scores[j]
			j--
		}
		moves[j+1], scores[j+1] = m, sc
	}

	best := standPat
	if inCheck {
		best = -infinity // "do nothing" isn't an option while in check, so there's no stand-pat floor
	}
	var bestMove board.Move
	for _, m := range moves {
		child := board.MakeMove(*b, m)
		t.s.net.Update(&t.acc[ply+1], &t.acc[ply], b, m, &child)
		score := -t.quiescence(&child, ply+1, qply+1, -beta, -alpha)
		if t.aborted {
			return 0
		}
		if score > best {
			best = score
			bestMove = m
		}
		if score > alpha {
			alpha = score
		}
		if alpha >= beta {
			bestMove = m
			break
		}
	}

	if !t.aborted {
		flag := ttExact
		switch {
		case best <= alphaOrig:
			flag = ttUpperBound
		case best >= beta:
			flag = ttLowerBound
		}
		t.s.tt.store(hash, bestMove, best, 0, flag, ply, raw, b.HalfmoveClock)
	}
	return best
}
