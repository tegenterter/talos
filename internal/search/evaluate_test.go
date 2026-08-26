package search

import (
	"testing"

	"talos/internal/board"
	"talos/internal/eval"
	"talos/internal/nnue"
)

// TestStaticEvalPicksTheRightEvaluator covers the dispatch itself, which
// damp's own test does not: which evaluator answers, and the one case where
// neither does.
func TestStaticEvalPicksTheRightEvaluator(t *testing.T) {
	tests := []struct {
		name string
		fen  string
		want func(b *board.Board) int
		why  string
	}{
		{
			name: "ordinary position uses the network",
			fen:  board.StartFEN,
			want: func(b *board.Board) int { return nnue.Evaluate(b) },
			why:  "the network is better than internal/eval at everything except won endgames",
		},
		{
			name: "lopsided endgame uses the classical evaluator",
			fen:  "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
			want: func(b *board.Board) int { return eval.Evaluate(b) },
			why:  "the network scores this bare rook ending at nearly three rooks",
		},
		{
			name: "insufficient material is a draw whatever either evaluator says",
			fen:  "4k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
			want: func(*board.Board) int { return 0 },
			why:  "a lone bishop cannot mate, so no material count describes the position",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := mustFEN(t, tc.fen)
			if got, want := evalAt(&b), tc.want(&b); got != want {
				t.Errorf("staticEval = %+d, want %+d (%s)", got, want, tc.why)
			}
		})
	}
}

// TestStaticEvalDampsWhicheverEvaluatorAnswered checks the clock applies on
// both paths — an endgame that decays only on the network's side of the
// dispatch would have exactly the blind spot this all exists to remove.
func TestStaticEvalDampsWhicheverEvaluatorAnswered(t *testing.T) {
	for _, fen := range []string{
		"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - %d 4",
		"4k3/8/8/8/8/8/8/R3K3 w - - %d 1",
	} {
		fresh := mustFEN(t, fmtFEN(fen, 0))
		stale := mustFEN(t, fmtFEN(fen, 90))
		f, s := evalAt(&fresh), evalAt(&stale)
		if abs(s) >= abs(f) {
			t.Errorf("%q: |staticEval| at clock 90 = %d, at clock 0 = %d; want the stale one smaller", fen, abs(s), abs(f))
		}
	}
}

// evalAt is staticEval on a standalone position: the search normally keeps
// the accumulator for each ply up to date as it makes moves, so a caller
// starting from a bare position has to seed ply 0 itself.
func evalAt(b *board.Board) int {
	t := standaloneThread(b)
	return t.staticEval(b, 0)
}

// standaloneThread is a thread outside any Search call, with its ply-0
// accumulator seeded for b — what a test needs to evaluate a bare position,
// since the search normally maintains both as it makes moves.
func standaloneThread(b *board.Board) *thread {
	t := &thread{s: &searchCtx{net: nnue.DefaultNetwork}}
	t.s.net.Refresh(&t.acc[0], b)
	return t
}

// benchPositions for the evaluation path: a dense middlegame (what almost
// every node in a real search looks like), and the won endgame that made
// internal/eval necessary.
var evalBenchFENs = map[string]string{
	"middlegame": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
	"endgame":    "8/k7/2B5/b7/8/6r1/4K3/8 b - - 1 128",
}

// BenchmarkNNUEEvaluate and BenchmarkStaticEval exist to answer one
// question with a number instead of a guess: what does the dispatch in
// staticEval cost on top of the network call it wraps? The bench command's
// nps fell about 2% when this landed, and the search's node counts changed
// at the same time, so nps alone cannot separate "the evaluator path got
// slower" from "the tree got shaped differently".
func BenchmarkNNUEEvaluate(b *testing.B) {
	for name, fen := range evalBenchFENs {
		bd, err := board.ParseFEN(fen)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink = nnue.Evaluate(&bd)
			}
		})
	}
}

func BenchmarkStaticEval(b *testing.B) {
	for name, fen := range evalBenchFENs {
		bd, err := board.ParseFEN(fen)
		if err != nil {
			b.Fatal(err)
		}
		th := standaloneThread(&bd)
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink = th.staticEval(&bd, 0)
			}
		})
	}
}

// BenchmarkStaticEvalDispatch times only what staticEval adds around the
// network call — the two regime checks and the damping — so the paired
// benchmarks above can be read for what they are. If this is a handful of
// nanoseconds against the network's several thousand, any gap those two
// show is code layout and measurement noise, not work.
func BenchmarkStaticEvalDispatch(b *testing.B) {
	for name, fen := range evalBenchFENs {
		bd, err := board.ParseFEN(fen)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if eval.InsufficientMaterial(&bd) {
					sink = 0
					continue
				}
				if eval.Lopsided(&bd) {
					sink = damp(eval.Evaluate(&bd), bd.HalfmoveClock)
					continue
				}
				sink = damp(1234, bd.HalfmoveClock)
			}
		})
	}
}

// sink keeps the compiler from optimizing the benchmarked calls away.
var sink int
