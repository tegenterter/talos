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
}

// buildInfo turns a raw negamax score into an Info snapshot, splitting it
// into ScoreCP or Mate depending on whether it's a mate score (see
// consts.go's mateThreshold) and converting a mate score's ply count into
// the move count UCI's "score mate" expects.
func buildInfo(depth, selDepth, score, nodes int, elapsed time.Duration, pv []board.Move) Info {
	nps := 0
	if elapsed > 0 {
		nps = int(float64(nodes) / elapsed.Seconds())
	}

	info := Info{Depth: depth, SelDepth: selDepth, Nodes: nodes, Nps: nps, Time: elapsed, PV: pv}

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
