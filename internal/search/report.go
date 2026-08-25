package search

import (
	"sync/atomic"
	"time"

	"talos/internal/board"
)

// Bound qualifies a reported score. Alpha-beta only proves an exact score
// when the search wasn't cut off, and a search that is still running has
// usually proved nothing more than a bound — so a progress report has to
// say which it is, or a GUI will draw a graph of numbers the engine never
// claimed. Rendered as UCI's "lowerbound"/"upperbound" by internal/uci.
type Bound uint8

const (
	// BoundExact means the search proved this score.
	BoundExact Bound = iota
	// BoundLower means the true score is at least this (the search failed
	// high: it found a move good enough to cut off, and stopped looking
	// for how much better it might be).
	BoundLower
	// BoundUpper means the true score is at most this (the search failed
	// low: nothing reached alpha, so the position is worse than the window
	// assumed, by an amount this search didn't measure).
	BoundUpper
)

// progressReportAfter is how long a search must have been running before
// it reports anything from *inside* an iteration.
//
// Reporting only between iterations is fine right up until an iteration
// takes longer than a person is willing to stare at an unchanging screen:
// past roughly depth 12 from the opening, each iteration costs 2-3x the
// last, so a "go infinite" goes silent for minutes at a time and looks
// hung when it is working perfectly. Below this threshold the same lines
// would be pure noise — depths 1-10 complete and report on their own in
// well under a second — so short searches (every blitz move, every test,
// every bench position) behave exactly as they did before. Stockfish
// draws the line in the same place, for the same reason.
//
// A var, not a const, so report_test.go can shrink it and observe the
// in-iteration reports without a test that has to run for three seconds.
var progressReportAfter = 3 * time.Second

// reporter turns search progress into Options.OnInfo calls: the completed
// iterations (which every search reports), plus — once a search has run
// past progressReportAfter — new principal variations as the root finds
// them, the root move currently being searched, and a periodic heartbeat
// so nps/nodes/hashfull keep moving even while nothing improves.
//
// Nothing here is synchronized, and nothing needs to be: every method is
// called from the goroutine driving iterative deepening (see thread.driver
// and Search), which is also the only goroutine that ever searches a root
// node. Helper threads deliberately stay silent — progress reporting is a
// property of the search as a whole, so N threads reporting it would be
// N-1 copies of the same news, arriving in an order that depends on
// scheduling, and Options.OnInfo callbacks would need locking of their own.
//
// A nil *reporter is a working no-op reporter, which is what a Search with
// no OnInfo gets, so no call site needs a nil check.
type reporter struct {
	onInfo   func(Info)
	interval time.Duration
	start    time.Time
	tt       *transpositionTable
	nodes    *atomic.Int64

	// depth is the iteration currently being searched, which is what an
	// in-progress report is about — unlike the completed depth Search
	// reports at the end of one.
	depth int
	// lastEmit is when anything was last reported, so the throttled
	// reports below stay quiet while other lines are already flowing.
	lastEmit time.Time
}

// newReporter returns nil when there's nothing to report to, so a search
// without OnInfo pays nothing for any of this.
func newReporter(onInfo func(Info), interval time.Duration, start time.Time, tt *transpositionTable, nodes *atomic.Int64) *reporter {
	if onInfo == nil {
		return nil
	}
	return &reporter{onInfo: onInfo, interval: interval, start: start, tt: tt, nodes: nodes, lastEmit: start}
}

// beginIteration records which depth subsequent in-progress reports are about.
func (r *reporter) beginIteration(depth int) {
	if r == nil {
		return
	}
	r.depth = depth
}

// complete reports a finished iterative-deepening iteration (or the final
// state of the search). Unlike everything else here it is never throttled:
// a completed depth is the one thing a GUI genuinely must not miss, and
// the old throttle silently dropped every iteration that finished within
// InfoInterval of the previous one — which is why a search from the
// opening used to appear to start at depth 7.
func (r *reporter) complete(depth, selDepth, score int, pv []board.Move) {
	if r == nil {
		return
	}
	now := time.Now()
	r.emit(buildInfo(depth, selDepth, score, int(r.nodes.Load()), now.Sub(r.start), pv), now)
}

// rootUpdate reports a new best line at the root, the moment the root
// finds one, rather than at the end of the iteration that found it. bound
// says what the score actually proves: a line that raised alpha inside an
// aspiration window is exact, one that cut off is a lower bound, and
// aspirationSearch reports its fail-lows here as upper bounds.
func (r *reporter) rootUpdate(selDepth, score int, bound Bound, pv []board.Move) {
	if r == nil || len(pv) == 0 {
		return
	}
	now := time.Now()
	if now.Sub(r.start) < progressReportAfter {
		return
	}
	// The caller keeps searching with its own pv slice; copy so a later
	// append can't rewrite a line already handed out.
	line := append([]board.Move(nil), pv...)
	info := buildInfo(r.depth, selDepth, score, int(r.nodes.Load()), now.Sub(r.start), line)
	info.Bound = bound
	r.emit(info, now)
}

// currMove reports which root move is being searched right now. It is the
// only report that says anything about work in progress rather than
// results, and on a long iteration it is the difference between "the
// engine is stuck" and "the engine is on move 14 of 31".
func (r *reporter) currMove(move board.Move, number, selDepth int) {
	if r == nil {
		return
	}
	now := time.Now()
	if now.Sub(r.start) < progressReportAfter {
		return
	}
	// Deliberately not throttled: this is already self-limiting at one line
	// per root move, and throttling it against the heartbeat below is what
	// an earlier version did — which starved out the more informative of
	// the two, since a root move on a deep iteration takes far longer than
	// the heartbeat interval.
	elapsed := now.Sub(r.start)
	nodes := int(r.nodes.Load())
	r.send(Info{
		Depth:          r.depth,
		SelDepth:       selDepth,
		Nodes:          nodes,
		Nps:            nps(nodes, elapsed),
		Time:           elapsed,
		CurrMove:       move,
		CurrMoveNumber: number,
	}, now)
}

// heartbeat reports nodes/nps/hashfull while an iteration grinds on with
// nothing new to say. It carries no score, because at this point the
// search has not proved one for this depth — reporting the previous
// depth's score here would be inventing news.
//
// Throttled against everything else reported, not just against itself: it
// exists to fill silence, so a currmove line or a new PV is reason enough
// for it to stay quiet.
func (r *reporter) heartbeat(selDepth int) {
	if r == nil {
		return
	}
	now := time.Now()
	if now.Sub(r.start) < progressReportAfter || now.Sub(r.lastEmit) < r.interval {
		return
	}
	elapsed := now.Sub(r.start)
	nodes := int(r.nodes.Load())
	r.emit(Info{
		Depth:    r.depth,
		SelDepth: selDepth,
		Nodes:    nodes,
		Nps:      nps(nodes, elapsed),
		Time:     elapsed,
	}, now)
}

// emit fills in the table occupancy every full report carries and hands
// the result to the callback.
func (r *reporter) emit(info Info, now time.Time) {
	info.HashFull = r.tt.hashfull()
	r.send(info, now)
}

// send is emit without the hashfull sampling, for the currmove line, whose
// whole point is to be cheap enough to print between root moves.
func (r *reporter) send(info Info, now time.Time) {
	r.lastEmit = now
	r.onInfo(info)
}

// rootBound classifies a root score that just raised alpha: anything at or
// above beta ended the search of this node early, so it is a lower bound
// rather than a proved score.
func rootBound(score, beta int) Bound {
	if score >= beta {
		return BoundLower
	}
	return BoundExact
}

// hashfullSamples is how many table slots hashfull inspects. Sampling
// rather than counting keeps the cost independent of the "Hash" setting
// (a full sweep of a 1 GB table would be a search-length pause of its
// own), and 1000 is both plenty for a permille figure and the sample size
// Stockfish uses for the same reading.
const hashfullSamples = 1000

// hashfull reports how full the transposition table is, in permille, as
// UCI's "hashfull" wants it. Samples are spread across shards so the
// figure doesn't depend on one shard's luck.
func (t *transpositionTable) hashfull() int {
	used := 0
	for i := 0; i < hashfullSamples; i++ {
		shard := &t.shards[i&int(t.shardMask)]
		slot := (i / len(t.shards)) % len(shard.entries)
		shard.mu.Lock()
		occupied := shard.entries[slot].hash != 0
		shard.mu.Unlock()
		if occupied {
			used++
		}
	}
	return used * 1000 / hashfullSamples
}
