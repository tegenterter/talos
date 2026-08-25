package search

import (
	"time"

	"talos/internal/board"
)

// Info is a snapshot of search progress, mirroring the fields UCI
// engines report via "info" lines. Depth/SelDepth now mean exactly what
// Stockfish means by them: Depth is the deepest iterative-deepening pass
// that finished, SelDepth the deepest ply actually reached by any of its
// branches (via quiescence or check extensions).
type Info struct {
	Depth    int
	SelDepth int
	// ScoreCP is the centipawn evaluation, meaningful only when Mate == 0.
	ScoreCP int
	// Mate is nonzero for a forced mate: positive N means the side to
	// move mates in N moves, negative N means it gets mated in N.
	Mate  int
	Nodes int
	Nps   int
	Time  time.Duration
	PV    []board.Move
	// Bound qualifies ScoreCP/Mate: BoundExact for a score the search
	// proved, and BoundLower/BoundUpper for one it only bounded (see
	// Bound). Only ever non-exact on a report made from inside an
	// unfinished iteration, since a completed one always has a real score.
	Bound Bound
	// HashFull is transposition table occupancy in permille, as UCI's
	// "hashfull" reports it. Zero on the currmove report below, which
	// deliberately skips sampling it.
	HashFull int
	// CurrMove and CurrMoveNumber, when CurrMoveNumber is nonzero, mean
	// this report is about the root move being searched right now rather
	// than about a result: the Info carries no score and no PV, and
	// CurrMoveNumber is the move's 1-based position in the root move list.
	CurrMove       board.Move
	CurrMoveNumber int
}

// nps is nodes per second over elapsed, or 0 when no time has passed yet.
func nps(nodes int, elapsed time.Duration) int {
	if elapsed <= 0 {
		return 0
	}
	return int(float64(nodes) / elapsed.Seconds())
}

// buildInfo turns a raw negamax score into an Info snapshot, splitting it
// into ScoreCP or Mate depending on whether it's a mate score (see
// consts.go's mateThreshold) and converting a mate score's ply count into
// the move count UCI's "score mate" expects.
func buildInfo(depth, selDepth, score, nodes int, elapsed time.Duration, pv []board.Move) Info {
	info := Info{Depth: depth, SelDepth: selDepth, Nodes: nodes, Nps: nps(nodes, elapsed), Time: elapsed, PV: pv}

	switch {
	case score >= mateThreshold:
		plies := mateValue - score
		info.Mate = (plies + 1) / 2
	case score <= -mateThreshold:
		plies := mateValue + score
		info.Mate = -((plies + 1) / 2)
	default:
		info.ScoreCP = score
	}

	return info
}
