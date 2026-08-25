package uci

import (
	"fmt"
	"io"
	"time"

	"talos/internal/board"
	"talos/internal/search"
)

// Bench is a non-standard command (Stockfish has one too, and OpenBench-style
// tooling depends on it) that searches a fixed set of positions to a fixed
// depth and reports the total node count and speed.
//
// It exists as a *single-number* regression instrument, complementing
// internal/search's golden_test.go: the goldens pin exact per-position output
// and say precisely what changed, while the bench total is one number that can
// be compared across commits, across machines (for nps), and from a shell
// script or CI without any Go test plumbing. A change to search behaviour
// moves the node count; a change to raw speed moves nps while leaving the node
// count alone. That separation is exactly what makes it useful when optimizing.
//
// Determinism is the whole point, so the parameters are fixed rather than
// configurable: single-threaded (results are only reproducible at Threads == 1
// — see search.Options.Threads), a fresh transposition table per position so
// one position can't seed the next, and no time limit (depth is the sole
// bound, so a slow machine gets the same node count as a fast one).
// benchDepth is a var rather than a const only so bench_test.go can lower it:
// a full-depth bench takes ~10s, which is too slow to run on every `go test`.
// No production path ever changes it.
var benchDepth = 8

const benchHashMB = 16

// benchPositions spans opening, tactical and positional middlegames, and
// several endgames. The first five duplicate internal/search/golden_test.go's
// set on purpose — they are the positions whose behaviour is already pinned
// elsewhere, so a bench movement can be traced back to a named golden — but
// they live in a _test.go file there and can't be imported, so they are
// restated here. The rest broaden the mix so the total isn't dominated by any
// single position's quirks.
var benchPositions = []string{
	// Shared with golden_test.go.
	board.StartFEN,
	"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // kiwipete
	"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",    // italian
	"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                            // pawn endgame
	"4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",                                       // rook endgame
	// Additional breadth.
	"r1bq1rk1/pp2bppp/2n2n2/2pp4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 10",      // quiet middlegame
	"2rq1rk1/pp1bppbp/3p1np1/8/2BNP3/2N1BP2/PPPQ2PP/2KR3R w - - 0 12",         // opposite-side castling
	"8/8/4kpp1/3p1b2/p6P/2B5/6P1/6K1 w - - 0 1",                               // bishop endgame
	"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1", // symmetrical, many pieces
	"6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",                                       // back-rank mate available
}

// RunBench searches benchPositions and writes a summary to out. It is
// exported so main.go can offer it as a command-line argument ("talos bench"),
// the convention engine-testing tooling expects, in addition to the "bench"
// UCI command.
func RunBench(out io.Writer) {
	totalNodes := 0
	start := time.Now()

	for i, fen := range benchPositions {
		b, err := board.ParseFEN(fen)
		if err != nil {
			// benchPositions is a fixed literal, so this can only mean the
			// table above was edited badly — report it rather than silently
			// benchmarking fewer positions than it claims to.
			fmt.Fprintf(out, "info string bench: skipping invalid FEN %q: %v\n", fen, err)
			continue
		}

		var last search.Info
		search.Search(b, search.Options{
			MaxDepth: benchDepth,
			Threads:  1,
			HashMB:   benchHashMB,
			OnInfo:   func(i search.Info) { last = i },
		})
		totalNodes += last.Nodes

		fmt.Fprintf(out, "position %2d/%d  depth %2d  nodes %9d  %s\n",
			i+1, len(benchPositions), last.Depth, last.Nodes, fen)
	}

	elapsed := time.Since(start)
	nps := 0
	if elapsed > 0 {
		nps = int(float64(totalNodes) / elapsed.Seconds())
	}

	// "Nodes searched" on its own line is the conventional shape engine
	// tooling greps for.
	fmt.Fprintln(out)
	fmt.Fprintf(out, "Time  : %d ms\n", elapsed.Milliseconds())
	fmt.Fprintf(out, "Nodes searched: %d\n", totalNodes)
	fmt.Fprintf(out, "NPS   : %d\n", nps)
}
