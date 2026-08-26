package search

import (
	"context"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"talos/internal/board"
	"talos/internal/nnue"
	"talos/internal/tablebase"
)

// searchCtx holds everything the threads of one Search call share: the
// transposition table, the node counter, the limits governing when to stop,
// and the stop flag itself. Every field is either fixed for the duration of
// the search or accessed atomically, so any number of goroutines may use it
// concurrently.
type searchCtx struct {
	// net is the network this search evaluates with — see Options.Network.
	net         *nnue.Network
	tt          *transpositionTable
	nodes       *atomic.Int64
	ctx         context.Context
	infinite    bool
	nodeLimited bool
	nodeLimit   int64
	deadline    time.Time
	tablebase   *tablebase.Tablebase
	// gameHistory holds the Zobrist hashes of positions that occurred in the
	// actual game before this search's root, most recent last, trimmed to
	// the span since the last irreversible move (see Options.GameHistory).
	// Written once before any thread starts and only read afterwards, so
	// every thread can share it with no synchronization — unlike
	// thread.pathHashes, which each thread mutates as it walks the tree.
	gameHistory []uint64

	// stopped is the search-wide "everyone stop now" flag. Every limit that
	// can end a search (context cancellation, the node budget, the
	// deadline) is search-wide, so once any thread observes one there's no
	// reason to make the others rediscover it independently.
	stopped atomic.Bool
	// selDepth is the deepest ply reached by any thread, folded in from
	// each thread's own count as it finishes (see thread.mergeSelDepth).
	selDepth atomic.Int64

	// sem is the pool of idle helper threads, or nil when the search is
	// single-threaded. Handing a whole *thread through a channel — rather
	// than sharing one between goroutines — is what keeps every field on
	// thread unsynchronized: the channel transfers ownership, and supplies
	// the happens-before edge that makes the transfer safe.
	sem chan *thread
	// splitMinDepth is the shallowest depth at which work may be split.
	splitMinDepth int

	// rep reports search progress via Options.OnInfo, or is nil when there
	// is nothing to report to. Only the driver thread touches it — see
	// reporter's doc comment for why that restriction is the design rather
	// than an oversight.
	rep *reporter
}

// thread holds the state belonging to exactly one searching goroutine.
// Nothing here is synchronized and nothing here needs to be: a thread is
// owned by a single goroutine at a time. Keeping killers, history and the
// search path thread-local is also what lets the move-ordering heuristics
// stay warm without contending on a shared table.
type thread struct {
	s *searchCtx

	// aborted mirrors s.stopped so the per-node check stays a plain bool
	// test rather than an atomic load. It is refreshed only inside
	// checkStop, which runs once every nodeCheckInterval nodes.
	aborted bool
	// selDepth is this thread's deepest ply, merged into s.selDepth when
	// the thread finishes.
	selDepth int

	// driver marks the one thread running the iterative-deepening loop,
	// which is also the only thread that ever searches a root node. It is
	// the only thread allowed to report progress (see reporter).
	driver bool

	// sp is the innermost split point this thread is working under, or nil
	// if it isn't working on a split-off subtree.
	sp *splitPoint
	// cut mirrors "some split point above me was cut off, so this work is
	// unwanted", the same way aborted mirrors s.stopped — refreshed only at
	// the periodic check so the hot path stays a plain bool test. Distinct
	// from aborted because it means "throw this result away", not "the
	// whole search is over".
	cut bool

	killers [maxPly][2]board.Move
	history [2][64][64]int

	// acc[ply] is the NNUE accumulator for the position this thread is
	// searching at that ply, maintained incrementally as moves are made
	// rather than rebuilt at every leaf (see internal/nnue's Accumulator,
	// and staticEval in evaluate.go). Like pathHashes it is a fixed array
	// owned by one goroutine: a split copies the one entry the helper needs
	// at handoff, so no two goroutines ever touch the same element.
	//
	// The invariant every call site below maintains: when negamax or
	// quiescence is entered with a board at ply p, acc[p] describes that
	// board. Break it anywhere and the search silently evaluates the wrong
	// position — which is why golden_test.go's exact node counts are the
	// real test of this code.
	acc [accStackSize]nnue.Accumulator

	// moveBuf[ply] is this thread's move-list storage for that ply, reused
	// for every node it searches there. Move generation was two thirds of
	// everything the engine allocated — a list per node, grown by append —
	// and none of it outlives the node that made it. Owned per thread and
	// indexed by ply for exactly the reason pathHashes and acc are: two
	// concurrently-searching siblings must never write the same element.
	moveBuf [accStackSize][maxMovesPerPosition]board.Move
	// scoreBuf is orderMoves' scratch for one node's move scores. Unlike
	// moveBuf it needs no per-ply copy: the scores are dead the moment the
	// sort finishes, and orderMoves does not recurse.
	scoreBuf [maxMovesPerPosition]int
	// evalStack[ply] is the static evaluation of the position searched at
	// that ply, or noEval where none was computed (in check, or too deep for
	// any heuristic to want one). It exists for the improving flag below,
	// which compares a node against the same side's position two plies
	// earlier — the cheapest available answer to "is this going well?".
	evalStack [accStackSize]int
	// contHist is the continuation history (conthistory.go), and
	// playedStack[ply] is the move made at that ply, which is what the plies
	// below it condition on. Both are per-thread and owned by one goroutine,
	// like the rest of the ordering state; contHist is a megabyte, which is
	// why it is int16 rather than int.
	contHist    continuationHistory
	playedStack [accStackSize]playedMove
	// excluded[ply] is the move a singular verification search at that ply
	// is trying to do without. Zero when no such search is running.
	excluded [accStackSize]board.Move
	// captureHist is the capture-ordering history (ordering.go), and
	// captureBuf[ply] the captures searched at that ply, for its malus.
	captureHist captureHistoryTable
	captureBuf  [accStackSize][32]board.Move
	// quietBuf[ply] records the quiet moves searched at that ply, for the
	// history malus applied when a later one cuts off. Bounded well below
	// the move list's length on purpose: the moves that matter are the ones
	// tried early, and capping it keeps the per-thread footprint small.
	quietBuf [accStackSize][32]board.Move

	// pathHashes holds the hash of every position on the line currently
	// being searched, for in-search repetition detection. It is a fixed
	// per-thread array indexed by an explicit length rather than a slice
	// threaded through the recursion: `append`ing to a caller's slice can
	// reuse its backing array, which is harmless while the search is
	// sequential but would have two concurrently-searching siblings writing
	// the same element once the tree is split across goroutines. An owned
	// array cannot alias, needs no allocation per node, and a split only has
	// to copy the live prefix once.
	pathHashes [maxPly]uint64
}

// maxMovesPerPosition bounds a single position's move list. The most legal
// moves any legal chess position is known to offer is 218; pseudo-legal
// generation can offer a few more before the legality filter runs, so this
// is rounded up with room to spare rather than sized to the record.
const maxMovesPerPosition = 256

// accStackSize bounds the accumulator stack: negamax stops recursing at
// maxPly, and a quiescence excursion entered there can still run
// maxQuiescencePly deeper, so that sum is the deepest ply any board is ever
// evaluated at.
const accStackSize = maxPly + maxQuiescencePly + 2

// nodeCheckInterval throttles how often negamax/quiescence actually check
// the clock/context/node-limit (via a bitmask on the shared node
// counter), since time.Now() and ctx.Err() aren't free and every node
// would otherwise pay that cost.
const nodeCheckInterval = 2047 // must be 2^n - 1

// checkStop updates t.aborted if the search should stop now. Once true,
// it stays true for the rest of this iterative-deepening depth (checked
// at the top of negamax/quiescence) so an abort propagates back to the
// top of the tree promptly instead of continuing to do wasted work.
func (t *thread) checkStop() {
	if t.aborted {
		return
	}
	if t.s.stopped.Load() {
		t.aborted = true
		return
	}
	if t.s.ctx.Err() != nil {
		t.stop()
		return
	}
	if t.s.infinite {
		return
	}
	if t.s.nodeLimited {
		if t.s.nodes.Load() >= t.s.nodeLimit {
			t.stop()
		}
		return
	}
	if time.Now().After(t.s.deadline) {
		t.stop()
	}
}

// stop ends the whole search, not just this thread: every condition
// checkStop tests is search-wide, so the first thread to notice one
// publishes it for the rest.
func (t *thread) stop() {
	t.aborted = true
	t.s.stopped.Store(true)
}

// mergeSelDepth folds this thread's deepest ply into the search-wide
// maximum. CAS-looped because several threads can finish at once.
func (t *thread) mergeSelDepth() {
	for {
		cur := t.s.selDepth.Load()
		if int64(t.selDepth) <= cur {
			return
		}
		if t.s.selDepth.CompareAndSwap(cur, int64(t.selDepth)) {
			return
		}
	}
}

// negamax returns the score of b, from the perspective of b's own side to
// move, searched to depth (plies remaining) at ply (plies from this
// search's root). pathLen is how many entries of t.pathHashes describe the
// line leading to b, for in-search repetition detection. *pv is set to the
// best line found from this node onward (empty/unset if the search aborts
// before finishing this node).
func (t *thread) negamax(b *board.Board, depth, ply int, alpha, beta int, pv *[]board.Move, pathLen int) int {
	if t.aborted || t.cut {
		return 0
	}
	if n := t.s.nodes.Add(1); n&nodeCheckInterval == 0 {
		t.checkStop()
		if t.aborted {
			return 0
		}
		// Walking the split-point chain costs a few pointer hops, so it
		// rides along with the periodic check rather than running per node.
		// A cut-off subtree therefore keeps working for up to
		// nodeCheckInterval nodes — the same bounded overrun the stop
		// checks already accept, for the same reason.
		if t.sp.dead() {
			t.cut = true
			return 0
		}
		// Progress reporting rides the same sampled interval, for the same
		// reason the checks above do: it needs a time.Now() it would
		// otherwise have to pay for per node. Only negamax does this, not
		// quiescence — a quiescence excursion always returns here within a
		// few plies, so the driver passes through this point constantly.
		if t.driver {
			t.s.rep.heartbeat(t.selDepth)
		}
	}
	if ply > t.selDepth {
		t.selDepth = ply
	}
	if ply >= maxPly {
		return t.staticEval(b, ply)
	}

	// Mate distance pruning: a mate can never be reported faster than ply
	// plies from here, and this side can never be found to be losing more
	// slowly than getting mated at ply. Once alpha/beta are clamped into
	// that range, a return here is provably correct without a TT probe,
	// move generation, or searching a single child — the window itself
	// already proves no result inside it could change. Only ever tightens
	// (never fires) at the root, since alpha/beta start at ±infinity there
	// (search.go); it starts pruning once some branch elsewhere in the
	// tree has already proven a mate score.
	if mdpEnabled {
		if alpha < -mateValue+ply {
			alpha = -mateValue + ply
		}
		if beta > mateValue-ply {
			beta = mateValue - ply
		}
		if alpha >= beta {
			return alpha
		}
	}

	hash := b.Hash()

	// A singular verification search re-enters this function at the same
	// ply with one move excluded (see singularExtension). Such a node must
	// not read or write the transposition table for this position — the
	// entry it would read is the very move it is trying to disprove.
	excluded := t.excluded[ply]
	haveExcluded := excluded != board.Move{}

	// Draw detection only applies to positions reached *during* this
	// search (ply > 0): the root itself should still be searched
	// normally even if, say, the fifty-move counter already technically
	// allows a draw claim — claiming a draw is the GUI/arbiter's call,
	// not a reason for the engine to refuse to produce a move.
	if ply > 0 {
		// Note the fifty-move rule is deliberately NOT checked here, but
		// after the checkmate test below: delivering mate ends the game
		// immediately, so a mating move on the hundredth halfmove is a win,
		// not a draw. Repetition needs no such care — a checkmate position
		// can't recur within a line, since the game would have ended the
		// first time it was reached.
		for _, h := range t.pathHashes[:pathLen] {
			if h == hash {
				return 0
			}
		}
		// Positions from the game so far count as repetitions too, not just
		// those reached inside this search. Without this the engine happily
		// repeats a position it has already been in twice — throwing away a
		// won game by walking into a threefold draw it cannot see, and
		// equally failing to steer into one when worse.
		for _, h := range t.s.gameHistory {
			if h == hash {
				return 0
			}
		}
		// Same ply>0 restriction, and for the same reason as the draw
		// checks above: the root always needs a real searched move, not
		// just a score. A hit isn't stored in the transposition table —
		// unlike a searched score, it isn't tied to a particular depth,
		// and Tablebase.Probe is cheap enough (a handful of in-memory
		// capture lookups plus one table decode) that not caching it is a
		// reasonable simplicity/speed tradeoff, consistent with the rest
		// of this codebase's bias.
		if t.s.tablebase != nil {
			if wdl, ok := t.s.tablebase.Probe(b); ok {
				return tbScore(wdl, ply)
			}
		}
	}

	alphaOrig := alpha
	var ttMove board.Move
	haveTTMove := false
	cachedEval := noEval
	var ttScore, ttDepth int
	var ttFlag ttFlag
	if entry, ok := t.s.tt.probe(hash, ply); ok && !haveExcluded {
		ttMove, haveTTMove = entry.move, true
		cachedEval = entry.eval
		ttScore, ttDepth, ttFlag = entry.scoreAt(b.HalfmoveClock), entry.depth, entry.flag
		// Same ply>0 restriction, and for the same reason, as the
		// draw/tablebase checks above: the root always needs a move it
		// actually searched this iteration, with a real reconstructed
		// principal variation — not a bare best move and a cached score.
		// Without this, aspiration windows re-searching the same depth
		// after a fail-low/fail-high (search.go's aspirationSearch) could
		// have the root's own probe hit an entry its own first attempt at
		// this depth just stored, and return here with *pv truncated to
		// one move.
		if ply > 0 && entry.depth >= depth {
			// Adjusted to this node's fifty-move clock, which the hash does
			// not include — see ttEntry.scoreAt.
			score := entry.scoreAt(b.HalfmoveClock)
			switch entry.flag {
			case ttExact:
				setTTPV(pv, entry.move)
				return score
			case ttLowerBound:
				if score > alpha {
					alpha = score
				}
			case ttUpperBound:
				if score < beta {
					beta = score
				}
			}
			if alpha >= beta {
				setTTPV(pv, entry.move)
				return score
			}
		}
	}

	legalMoves := board.GenerateLegalMovesInto(b, t.moveBuf[ply][:])
	kingSq := b.Pieces[b.SideToMove][board.King].LSB()
	inCheck := board.IsSquareAttacked(b, kingSq, b.SideToMove.Opposite())

	if len(legalMoves) == 0 {
		if inCheck {
			return -(mateValue - ply)
		}
		return 0
	}

	// Fifty-move draw, checked here rather than alongside the other draw
	// tests above so that checkmate — which ends the game outright — takes
	// precedence over it. Same ply > 0 restriction as those tests.
	if ply > 0 && b.HalfmoveClock >= 100 {
		return 0
	}

	if depth <= 0 {
		return t.quiescence(b, ply, 0, alpha, beta)
	}

	// The static evaluation, computed once for every heuristic below that
	// wants it, and only at the depths where one does — an evaluation at
	// every node would cost more than the pruning saves. In check it is
	// meaningless (the side to move is under threat, not standing still) and
	// none of these heuristics apply.
	staticScore, haveStatic := 0, false
	if !inCheck && depth <= evalNeededMaxDepth {
		if cachedEval == noEval {
			cachedEval = t.rawEval(b, ply)
		}
		staticScore, haveStatic = damp(cachedEval, b.HalfmoveClock), true
	}
	t.evalStack[ply] = noEval
	if haveStatic {
		t.evalStack[ply] = staticScore
	}

	// improving asks whether the side to move is better off than it was two
	// plies ago — its own previous turn. It is the cheapest signal a search
	// has for "is this line going my way", and it is what lets the pruning
	// below distinguish a position that is quietly getting better (where a
	// late quiet move is unlikely to be the point) from one that is going
	// wrong (where it might be the only thing that saves it).
	//
	// Unknown counts as improving: in check there is no meaningful static
	// score, and treating a missing comparison as "going badly" would apply
	// the harsher pruning to exactly the tactical positions that least
	// deserve it.
	improving := true
	if ply >= 2 && haveStatic && t.evalStack[ply-2] != noEval {
		improving = staticScore >= t.evalStack[ply-2]
	}

	// Internal iterative reduction: a node this deep with no transposition
	// -table move has never been searched before, so its move ordering rests
	// on history and SEE alone — and the first move is unlikely to be best.
	// Searching it a ply shallower costs little (the next iteration will
	// revisit it *with* a table move, having spent less getting there) and
	// is worth more than it sounds, because the alternative is a full-depth
	// search with the ordering that made this expensive in the first place.
	//
	// It is a reduction, not a pruning: nothing is discarded, and the
	// shallower search still fills the table for the visit that follows.
	if iirEnabled && !haveTTMove && depth >= iirMinDepth {
		depth--
	}

	// Null-move pruning: if letting the opponent move twice in a row
	// (i.e. we do nothing) still can't stop them failing to reach beta at
	// a cheaper, reduced-depth search, our actual move can only be even
	// better, so this node is pruned without examining any real move.
	// Skipped in check (a null move can't be made from check), near the
	// root/at low depth (too unreliable to be worth it), and with only
	// pawns and a king left (the classic null-move zugzwang failure mode,
	// where "the right move is to do nothing" can genuinely be true).
	const nullMoveMinDepth = 3
	const nullMoveReduction = 2
	if !inCheck && depth >= nullMoveMinDepth && ply > 0 && hasNonPawnMaterial(b, b.SideToMove) {
		nullBoard := *b
		nullBoard.SideToMove = b.SideToMove.Opposite()
		nullBoard.EnPassant = board.NoSquare

		// A null move moves no piece, so both perspectives' features are
		// exactly the parent's; only whose turn it is changes, and
		// EvaluateAcc takes that as an argument.
		t.acc[ply+1] = t.acc[ply]
		// ...and there is no move for the plies below to condition on.
		t.clearPlayed(ply)

		var discardedPV []board.Move
		// pathLen is passed through unchanged: a null move doesn't reach a
		// new position by a real move, so this node's own hash is
		// deliberately not part of the line its reduced search sees.
		score := -t.negamax(&nullBoard, depth-1-nullMoveReduction, ply+1, -beta, -beta+1, &discardedPV, pathLen)
		if t.aborted {
			return 0
		}
		if score >= beta {
			return score
		}
	}

	// Singular extension test. Run *before* the move list is generated and
	// ordered, because the verification search re-enters this ply and would
	// otherwise overwrite the very buffers this node is about to iterate.
	ttMoveExtension := 0
	if singularEnabled && !haveExcluded && haveTTMove && ply > 0 &&
		depth >= singularMinDepth && ttDepth >= depth-singularDepthMargin &&
		ttFlag != ttUpperBound && ttScore > -mateThreshold && ttScore < mateThreshold {
		ttMoveExtension = t.singularExtension(b, ttMove, ttScore, depth, ply, pathLen)
	}

	ordered := t.orderMoves(b, legalMoves, ttMove, haveTTMove, ply)
	// Record this position on the line every child search will see. pathLen
	// is bounded by ply, which the maxPly guard above already enforces.
	t.pathHashes[pathLen] = hash
	childPathLen := pathLen + 1

	bestScore := -infinity
	var bestMove board.Move

	// Split state for this node, created only if the node is even a
	// candidate for handing work to helpers (see canSplitHere). Nodes that
	// aren't pay nothing for any of this.
	var sp *splitPoint
	var prevSP *splitPoint
	var wg sync.WaitGroup
	var siblings []siblingResult

	canSplit := t.canSplitHere(depth, inCheck, len(ordered))
	if canSplit {
		prevSP = t.sp
		sp = &splitPoint{parent: prevSP}
		sp.alpha.Store(int64(alpha))
		t.sp = sp
		siblings = make([]siblingResult, len(ordered))
		// Every exit from this node — normal, cutoff, or abort — must first
		// tell any helper still working under this split point to stop, and
		// then wait for it. Nothing may outlive the node it was spawned
		// from: helpers read b, and they write into siblings.
		defer func() {
			sp.cutoff.Store(true)
			wg.Wait()
			t.sp = prevSP
		}()
	}

	cutoff := false
	quietCount, captureCount := 0, 0
	for i, move := range ordered {
		if haveExcluded && move == excluded {
			continue
		}

		// "Young brothers wait": never split before the eldest sibling has
		// been searched. Its score is what sets alpha for everyone else, so
		// splitting before it lands would send helpers off with a window so
		// wide it defeats the point.
		if canSplit && i > 0 && !cutoff {
			if helper, ok := t.acquireHelper(); ok {
				// Everything the helper needs from this thread is copied to
				// it now, while this goroutine still owns it exclusively.
				copy(helper.pathHashes[:pathLen+1], t.pathHashes[:pathLen+1])
				// The helper makes its move from this node's board, so it
				// needs this node's accumulator to update from.
				helper.acc[ply] = t.acc[ply]
				helper.sp = sp
				helper.cut = false
				helper.aborted = t.s.stopped.Load()

				wg.Add(1)
				go func(i int, move board.Move) {
					defer wg.Done()
					defer t.releaseHelper(helper)

					// Read alpha as late as possible: the parent keeps
					// raising it while helpers run, and a fresher bound
					// means a narrower, cheaper search. A stale one is
					// never *wrong*, only slower — which is exactly why
					// this is safe to do without locking.
					a := int(sp.alpha.Load())

					var childPV []board.Move
					score := helper.searchChild(b, move, i, depth, ply, a, beta, inCheck, improving, extensionFor(move, ttMove, haveTTMove, ttMoveExtension), childPathLen, &childPV)
					siblings[i] = siblingResult{
						searched: true,
						usable:   !helper.aborted && !helper.cut,
						score:    score,
						move:     move,
						pv:       childPV,
					}
				}(i, move)
				continue
			}
			// No helper free: fall through and search it here. Acquisition
			// never blocks, which is what makes deadlock impossible.
		}

		if ply == 0 {
			// Reported only for root moves searched on this goroutine: a
			// move handed to a helper above is being searched alongside
			// several others, so "the move being searched now" isn't a
			// single answer there.
			t.s.rep.currMove(move, i+1, t.selDepth)
		}

		if isCapture(b, move) {
			if captureCount < len(t.captureBuf[ply]) {
				t.captureBuf[ply][captureCount] = move
				captureCount++
			}
		} else if quietCount < len(t.quietBuf[ply]) {
			t.quietBuf[ply][quietCount] = move
			quietCount++
		}

		var childPV []board.Move
		score := t.searchChild(b, move, i, depth, ply, alpha, beta, inCheck, improving, extensionFor(move, ttMove, haveTTMove, ttMoveExtension), childPathLen, &childPV)
		if t.aborted || t.cut {
			return 0
		}

		if score > bestScore {
			bestScore = score
			bestMove = move
			if score > alpha {
				// Only a move that raises alpha was searched on a full
				// window, so only then is childPV a real line rather than
				// the by-product of a bound-proving scout search.
				*pv = append([]board.Move{move}, childPV...)
				if ply == 0 {
					// A new best line at the root is news now, not when
					// the iteration eventually finishes — which on a deep
					// iteration can be minutes away.
					t.s.rep.rootUpdate(t.selDepth, score, rootBound(score, beta), *pv)
				}
			} else if len(*pv) == 0 {
				// Everything tried so far failed low, so there's no line
				// worth reporting — but callers still need a legal move,
				// so keep the best one found.
				*pv = []board.Move{move}
			}
		}
		if score > alpha {
			alpha = score
			if sp != nil {
				// Publish the improved bound so helpers spawned from here
				// on start with a narrower window.
				sp.alpha.Store(int64(alpha))
			}
		}
		if alpha >= beta {
			bonus := historyBonus(depth)
			if isCapture(b, move) {
				t.updateCaptureHistory(b, move, bonus)
			} else {
				t.recordKiller(ply, move)
				t.recordHistory(b.SideToMove, move, depth)
				t.updateContHistory(b, move, ply, bonus)
				// Every quiet searched before this one was tried and failed
				// to cut off; say so, or the tables record only that a move
				// once worked and never that it since stopped working.
				t.penalizeSearchedQuiets(b, ply, quietCount, move, depth)
			}
			// Captures that were tried and did not cut off are penalized
			// whatever the cutting move was: the point is that taking there
			// did not work, not what did.
			t.penalizeSearchedCaptures(b, ply, captureCount, move, depth)
			cutoff = true
			if sp != nil {
				// Stop helpers immediately rather than at the deferred
				// cleanup: their work is provably wasted from here.
				sp.cutoff.Store(true)
			}
			break
		}
	}

	if canSplit {
		// Collect whatever the helpers produced. Waiting here (rather than
		// leaving it to the deferred cleanup) is what makes their results
		// usable at all.
		wg.Wait()
		if t.aborted || t.cut {
			return 0
		}
		// Folded in move order, so ties resolve exactly as the sequential
		// loop would have resolved them — the earlier-ordered move wins.
		for i := range siblings {
			r := siblings[i]
			if cutoff {
				break
			}
			if !r.searched || !r.usable {
				// Either no helper took this move (it was searched inline
				// above, or never reached), or the helper's search was
				// abandoned partway and its score means nothing.
				continue
			}
			if r.score > bestScore {
				bestScore = r.score
				bestMove = r.move
				if r.score > alpha {
					*pv = append([]board.Move{r.move}, r.pv...)
					if ply == 0 {
						t.s.rep.rootUpdate(t.selDepth, r.score, rootBound(r.score, beta), *pv)
					}
				} else if len(*pv) == 0 {
					*pv = []board.Move{r.move}
				}
			}
			if r.score > alpha {
				alpha = r.score
				sp.alpha.Store(int64(alpha))
			}
			if alpha >= beta {
				if !isCapture(b, r.move) {
					t.recordKiller(ply, r.move)
					t.recordHistory(b.SideToMove, r.move, depth)
				}
				cutoff = true
				sp.cutoff.Store(true)
				break
			}
		}
	}

	flag := ttExact
	switch {
	case bestScore <= alphaOrig:
		flag = ttUpperBound
	case bestScore >= beta:
		flag = ttLowerBound
	}
	if !haveExcluded {
		// noEval: a node searched to depth never needed a static evaluation,
		// so there is none to cache here. store keeps whatever evaluation
		// the entry already carried.
		t.s.tt.store(hash, bestMove, bestScore, depth, flag, ply, noEval, b.HalfmoveClock)
	}

	return bestScore
}

// setTTPV reports a transposition-table cutoff's single-move line, unless
// the entry carries no move — quiescence stores a stand-pat cutoff with none,
// since no move produced it, and reporting a zero Move would put "a1a1" in
// the principal variation and in the ponder hint derived from it.
func setTTPV(pv *[]board.Move, m board.Move) {
	if m == (board.Move{}) {
		*pv = nil
		return
	}
	*pv = []board.Move{m}
}

// evalNeededMaxDepth is the deepest node that computes a static evaluation
// for the pruning heuristics below. Past it none of them apply, and an
// evaluation at every node would cost more than they save.
// evalNeededMaxDepth is the deepest node that computes a static evaluation.
// The only consumer left is the improving flag (which late move reductions
// read), and reductions stop mattering below lmrMinDepth, so there is no
// point evaluating deeper than the reductions reach.
const evalNeededMaxDepth = 8

// penalizeSearchedCaptures applies the capture-history malus to every
// capture tried at this node before the cutoff.
func (t *thread) penalizeSearchedCaptures(b *board.Board, ply, captureCount int, cutMove board.Move, depth int) {
	for _, m := range t.captureBuf[ply][:captureCount] {
		if m != cutMove {
			t.updateCaptureHistory(b, m, -historyBonus(depth))
		}
	}
}

// penalizeSearchedQuiets applies the history malus — in both the plain and
// the continuation tables — to every quiet move tried at this node before
// the one that cut off.
func (t *thread) penalizeSearchedQuiets(b *board.Board, ply, quietCount int, cutMove board.Move, depth int) {
	for _, m := range t.quietBuf[ply][:quietCount] {
		if m != cutMove {
			t.penalizeHistory(b.SideToMove, m, depth)
			t.updateContHistory(b, m, ply, -historyBonus(depth))
		}
	}
}

// iirMinDepth is the shallowest depth internal iterative reduction applies
// to. Shallow nodes are cheap enough that ordering them badly costs little,
// and reducing them risks dropping straight into quiescence.
const iirMinDepth = 4

// iirEnabled exists only so a test can measure what internal iterative
// reduction does, like lmrEnabled; no production path sets it false.
var iirEnabled = true

// extensionFor gives the transposition-table move whatever the singular test
// granted it, and every other move nothing.
func extensionFor(move, ttMove board.Move, haveTTMove bool, ttMoveExtension int) int {
	if haveTTMove && move == ttMove && ttMoveExtension > 0 {
		return ttMoveExtension
	}
	return 0
}

// singularExtension asks whether the transposition-table move is the *only*
// move that holds this position together, and returns a ply of extension if
// it is.
//
// The test: search every other move at reduced depth against a window just
// below what the table says this move is worth. If they all fail low, the
// table move is not merely best, it is singular — nothing else comes close —
// and the line running through it deserves to be searched deeper, because
// that is where the game is actually decided.
//
// It is the most expensive heuristic in the search (a whole extra search per
// qualifying node), which is why it is confined to deep nodes with a
// trustworthy table entry: shallow depth, a table entry that was only an
// upper bound, or a mate score all make the question meaningless or the
// answer unreliable.
func (t *thread) singularExtension(b *board.Board, ttMove board.Move, ttScore, depth, ply, pathLen int) int {
	target := ttScore - singularMarginPerPly*depth

	t.excluded[ply] = ttMove
	var discarded []board.Move
	score := t.negamax(b, (depth-1)/2, ply, target-1, target, &discarded, pathLen)
	t.excluded[ply] = board.Move{}

	if t.aborted || t.cut {
		return 0
	}
	if score < target {
		return 1
	}
	return 0
}

// Singular extension tuning. Conventional starting values.
const (
	singularMinDepth     = 7
	singularDepthMargin  = 3
	singularMarginPerPly = 2
)

// singularEnabled exists only so a test can measure what singular extensions
// do, like lmrEnabled; no production path sets it false.
var singularEnabled = true

// splitMinDepth is the shallowest remaining depth at which a node will
// hand siblings to helper goroutines. Splitting has real fixed costs — a
// goroutine, a pathHashes copy, a split point, and the loss of the
// sequential loop's ever-improving alpha — so it only pays on subtrees big
// enough to dwarf them. Shallow nodes are far more numerous, so splitting
// them would spend most of the engine's time on coordination.
const splitMinDepth = 5

// splitPoint is the coordination state shared between a node and the
// helpers searching its siblings. It is deliberately tiny: two atomics and
// a parent link, no mutex, because it sits in the hottest loop in the
// program and every field is either write-once-ish or a bound that is safe
// to read stale.
type splitPoint struct {
	parent *splitPoint
	// alpha is the node's best-so-far bound, republished as it improves so
	// helpers can start with a narrower window. Reading it stale costs
	// search efficiency, never correctness.
	alpha atomic.Int64
	// cutoff means the work under this split point is no longer wanted,
	// either because the node found a beta cutoff or because it is done.
	cutoff atomic.Bool
}

// dead reports whether this split point or any ancestor has been cut off.
// Safe on a nil receiver, which is the common case: most threads aren't
// working under a split at all.
func (sp *splitPoint) dead() bool {
	for ; sp != nil; sp = sp.parent {
		if sp.cutoff.Load() {
			return true
		}
	}
	return false
}

// siblingResult is one helper's answer for one move, handed back to the
// node that spawned it.
type siblingResult struct {
	searched bool // a helper actually took this move
	usable   bool // ...and ran to completion, so score means something
	score    int
	move     board.Move
	pv       []board.Move
}

// canSplitHere reports whether this node may hand siblings to helpers.
// inCheck nodes are excluded because every reply matters there, so the
// move ordering that makes "search the eldest first, then parallelize the
// rest" sensible doesn't apply. Nodes with fewer than three moves are
// excluded because after the eldest there's at most one sibling left,
// which isn't worth a goroutine.
func (t *thread) canSplitHere(depth int, inCheck bool, moves int) bool {
	return t.s.sem != nil && depth >= t.s.splitMinDepth && !inCheck && moves >= 3
}

// acquireHelper takes an idle thread from the pool if one is free, and
// never waits for one.
//
// The non-blocking receive is load-bearing, not an optimization: a node
// that blocked waiting for a helper could be holding the very thread a
// running helper needs to finish and hand back. Falling through to search
// the move inline makes that deadlock structurally impossible rather than
// merely unlikely, which matters because the failure mode is a hung engine
// with no diagnostic.
func (t *thread) acquireHelper() (*thread, bool) {
	select {
	case h := <-t.s.sem:
		return h, true
	default:
		return nil, false
	}
}

// releaseHelper returns a thread to the pool. The send cannot block: the
// channel's capacity equals the number of helper threads that exist, so
// there is always room for one that was taken out of it.
func (t *thread) releaseHelper(h *thread) { t.s.sem <- h }

// searchChild runs one move's full PVS + LMR ladder and returns its score
// from this node's point of view. It exists so the sequential path and the
// helper path search a move in provably identical ways — the two differ
// only in which thread runs this, and which alpha they were handed.
func (t *thread) searchChild(b *board.Board, move board.Move, i, depth, ply, alpha, beta int, inCheck, improving bool, extraExtension, childPathLen int, childPV *[]board.Move) int {
	child := board.MakeMove(*b, move)
	t.s.net.Update(&t.acc[ply+1], &t.acc[ply], b, move, &child)
	t.recordPlayed(b, move, ply)

	ext := extraExtension
	childKingSq := child.Pieces[child.SideToMove][board.King].LSB()
	givesCheck := board.IsSquareAttacked(&child, childKingSq, child.SideToMove.Opposite())
	if givesCheck && ext == 0 {
		ext = 1 // check extension: don't let a checking move hit the horizon at depth 0
	}
	newDepth := depth - 1 + ext

	reduction := 0
	if lmrEnabled {
		isKiller := ply < maxPly && (move == t.killers[ply][0] || move == t.killers[ply][1])
		reduction = lmrReduction(depth, i, inCheck, givesCheck, isCapture(b, move), move.Promotion != board.NoPiece, isKiller, improving, beta != alpha+1)
	}

	// Principal Variation Search, with LMR folded into the same ladder:
	// each rung is a cheaper search whose only job is to justify paying
	// for the next one.
	if i == 0 || !pvsEnabled {
		// The first move is the principal-variation candidate. It gets
		// the full window, because its exact score is the yardstick
		// every later move is measured against — guess it wrong and the
		// cheap searches below are all comparing to the wrong number.
		// (With PVS off this is also the path every later move takes:
		// full window, reduced depth, re-searched if it beats alpha.)
		score := -t.negamax(&child, newDepth-reduction, ply+1, -beta, -alpha, childPV, childPathLen)
		if t.aborted || t.cut {
			return 0
		}
		if reduction > 0 && score > alpha {
			score = -t.negamax(&child, newDepth, ply+1, -beta, -alpha, childPV, childPathLen)
		}
		return score
	}

	// Later moves are assumed not to beat the PV, and that assumption is
	// tested as cheaply as it can be: a null window (beta = alpha+1, i.e.
	// "is this better than alpha at all?") at whatever reduced depth LMR
	// allows. A null window can never return an exact score, only "> alpha"
	// or "<= alpha" — but "<= alpha" is the answer for most moves, it's all
	// that's needed to discard them, and it's far cheaper to obtain than a
	// full-width score nobody will use.
	score := -t.negamax(&child, newDepth-reduction, ply+1, -alpha-1, -alpha, childPV, childPathLen)
	if t.aborted || t.cut {
		return 0
	}
	if reduction > 0 && score > alpha {
		// The reduced search says this move might matter after all. Re-run
		// at the depth it actually earned — still on the cheap window — to
		// rule out the reduction being the only reason it looked good (see
		// lmrReduction's doc comment).
		score = -t.negamax(&child, newDepth, ply+1, -alpha-1, -alpha, childPV, childPathLen)
		if t.aborted || t.cut {
			return 0
		}
	}
	if score > alpha && score < beta {
		// It genuinely beats alpha and isn't already a cutoff, so this move
		// is a new PV candidate and its exact value now matters; only a
		// full-width search can produce that (and a PV worth reporting).
		// This can't fire when beta == alpha+1 — inside a null-window
		// search there's no gap between the bounds to land in — so scout
		// nodes never pay for it.
		score = -t.negamax(&child, newDepth, ply+1, -beta, -alpha, childPV, childPathLen)
	}
	return score
}

// pvsEnabled exists only so a test can measure what Principal Variation
// Search actually costs or saves on this engine (compare node counts with
// it on vs. off at the same position and depth) — there's no UCI option or
// other production path that ever sets it false. Same role as lmrEnabled
// below.
var pvsEnabled = true

// lmrEnabled exists only so lmr_test.go can measure LMR's actual effect
// (compare node counts with it on vs. off on the same position/depth) —
// there's no UCI option or other production path that ever sets it false.
var lmrEnabled = true

// mdpEnabled exists only so mdp_test.go can measure mate distance
// pruning's actual effect (compare node counts and confirm identical mate
// results with it on vs. off) — there's no UCI option or other production
// path that ever sets it false.
var mdpEnabled = true

// LMR (Late Move Reductions) tuning constants.
const (
	lmrMinDepth     = 3 // don't reduce at shallow depth, where a wrong reduction costs relatively more
	lmrMinMoveIndex = 3 // the first few (already well-ordered) moves at a node are always searched in full
)

// lmrReduction returns how many plies to shave off a late move's search
// depth before deciding whether it's worth searching properly. index is
// the move's 0-based position in orderMoves' output.
//
// The premise: orderMoves has already put the moves most likely to matter
// first (TT move, good captures, killers, history-favored quiets), so a
// move this far down the list is unlikely to turn out best, and a
// shallower search is normally enough to confirm that cheaply — the
// caller re-searches at full depth only if the reduced search comes back
// surprisingly good (beats alpha), so a reduction can make the search
// faster but never actually miss a move that matters; at worst it costs
// one extra (reduced) search before finding out a move needed the full
// depth after all.
//
// Never reduced: captures and promotions (their own SEE/MVV-LVA ordering
// already ranks them on merit, not just list position, so "late" doesn't
// mean "unpromising" the way it does for a quiet move), killer moves
// (proven to cause a cutoff at a sibling node of this exact ply — a
// stronger, more specific signal than "ordered late" should override),
// and anything to do with check — giving it (already gets a check
// extension instead; reducing an extended move would fight its own
// extension) or already being in it (every reply matters when in check,
// there's no "probably not the best move" to lean on).
func lmrReduction(depth, index int, inCheck, givesCheck, capture, promotion, isKiller, improving, pvNode bool) int {
	if depth < lmrMinDepth || index < lmrMinMoveIndex || inCheck || givesCheck || capture || promotion || isKiller {
		return 0
	}

	r := int(lmrTable[minInt(depth, maxPly-1)][minInt(index, maxMovesPerPosition-1)])
	if !improving {
		// The position is going the wrong way, so a move this far down the
		// list is even less likely to be what turns it around.
		r++
	}
	if pvNode {
		// A principal-variation node's score is the one everything else is
		// measured against, so it is searched closer to full width.
		r--
	}
	if r < 0 {
		r = 0
	}
	// Never reduce into quiescence: the point of a reduced search is a
	// cheaper verdict from the same kind of search, not a different one.
	if r > depth-1 {
		r = depth - 1
	}
	return r
}

// lmrTable[depth][index] is the base reduction, growing with the logarithm
// of both — the standard shape, and a real change from the flat 1-2 plies
// this used before. Flat reductions treat the 4th move at depth 4 like the
// 30th at depth 20; the logarithms say the second deserves far more
// scepticism than the first, which is what lets a deep search cut its late
// moves hard without gutting the shallow ones.
var lmrTable [maxPly][maxMovesPerPosition]int8

func init() {
	for d := 1; d < maxPly; d++ {
		for i := 1; i < maxMovesPerPosition; i++ {
			lmrTable[d][i] = int8(lmrBase + math.Log(float64(d))*math.Log(float64(i))/lmrDivisor)
		}
	}
}

// lmrBase and lmrDivisor shape the table above. Conventional starting
// values; tuning them means SPRT, not intuition.
const (
	lmrBase    = 0.75
	lmrDivisor = 2.25
)

// minInt keeps the table lookups in range without pulling in generics for
// two call sites.
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// tbWinScore is the score magnitude a tablebase-confirmed win at ply 0
// would report — comfortably below mateThreshold (so it's never
// misreported as a mate score by internal/uci) and comfortably above any
// realistic internal/nnue evaluation (so alpha-beta always treats a
// confirmed win as better than any merely-good-looking position).
// Subtracting ply, the same way mateValue does, gives a (weak, since this
// package only has WDL, not DTZ — see internal/tablebase's package doc)
// preference for reaching a confirmed-won simplification sooner rather
// than later, rather than shuffling indifferently within one.
const tbWinScore = 20000

// tbScore converts a tablebase.Tablebase.Probe result (in {-2,-1,0,1,2})
// into a search score from the perspective of the side to move at ply.
// Cursed wins and blessed losses — forced mates that are themselves
// beyond the fifty-move rule's reach even from a freshly reset counter —
// score as an exact draw (0): unlike a genuine win/loss, that outcome
// doesn't depend on the actual current halfmove clock, so it's not an
// approximation.
func tbScore(wdl, ply int) int {
	switch wdl {
	case 2:
		return tbWinScore - ply
	case -2:
		return -(tbWinScore - ply)
	default:
		return 0
	}
}
