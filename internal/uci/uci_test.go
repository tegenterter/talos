package uci

import (
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"talos/internal/board"
)

// syncBuffer is a thread-safe byte buffer, since run() writes from both
// its own goroutine and the background search goroutine "go" starts.
type syncBuffer struct {
	mu  sync.Mutex
	buf strings.Builder
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.String()
}

// harness runs the UCI loop against a pipe so a test can feed it commands
// and observe output while it's still running.
type harness struct {
	t    *testing.T
	pw   *io.PipeWriter
	out  *syncBuffer
	done chan struct{}
}

func startEngine(t *testing.T) *harness {
	t.Helper()
	pr, pw := io.Pipe()
	h := &harness{t: t, pw: pw, out: &syncBuffer{}, done: make(chan struct{})}
	go func() {
		run(pr, h.out)
		close(h.done)
	}()
	return h
}

func (h *harness) send(cmd string) {
	if _, err := h.pw.Write([]byte(cmd + "\n")); err != nil {
		h.t.Fatalf("write %q: %v", cmd, err)
	}
}

// waitFor polls the engine's output until substr appears or timeout elapses.
func (h *harness) waitFor(substr string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if strings.Contains(h.out.String(), substr) {
			return true
		}
		time.Sleep(2 * time.Millisecond)
	}
	return false
}

// quit sends "quit" and waits for run() to return, so all of its output
// (including anything a background search goroutine was about to write)
// is guaranteed flushed before the test inspects h.out.
func (h *harness) quit() {
	h.send("quit")
	h.pw.Close()
	select {
	case <-h.done:
	case <-time.After(2 * time.Second):
		h.t.Fatal("run did not return after quit")
	}
}

func TestGoMovetimeProducesBestmove(t *testing.T) {
	h := startEngine(t)
	h.send("position startpos")
	h.send("go movetime 100")
	if !h.waitFor("bestmove", time.Second) {
		t.Fatalf("no bestmove within 1s; output:\n%s", h.out.String())
	}
	h.quit()
}

func TestGoInfiniteStopsOnStop(t *testing.T) {
	h := startEngine(t)
	h.send("position startpos")
	h.send("go infinite")

	// The search should still be running; give it a moment, then confirm
	// it hasn't produced a bestmove on its own (nothing but "stop" or
	// "quit" should ever end an infinite search).
	time.Sleep(100 * time.Millisecond)
	if strings.Contains(h.out.String(), "bestmove") {
		t.Fatalf("go infinite produced bestmove before stop; output:\n%s", h.out.String())
	}

	h.send("stop")
	if !h.waitFor("bestmove", time.Second) {
		t.Fatalf("no bestmove within 1s of stop; output:\n%s", h.out.String())
	}
	h.quit()
}

func TestPonderhitStartsTimeBudget(t *testing.T) {
	h := startEngine(t)
	h.send("position startpos")
	// Small wtime/btime so allocateTime's budget is short once ponderhit
	// starts the clock, keeping the test fast.
	h.send("go ponder wtime 300 btime 300 winc 0 binc 0")

	time.Sleep(50 * time.Millisecond)
	if strings.Contains(h.out.String(), "bestmove") {
		t.Fatalf("pondering produced bestmove before ponderhit; output:\n%s", h.out.String())
	}

	h.send("ponderhit")
	if !h.waitFor("bestmove", 2*time.Second) {
		t.Fatalf("no bestmove within 2s of ponderhit; output:\n%s", h.out.String())
	}
	h.quit()
}

func TestStopWithoutActiveSearchIsNoop(t *testing.T) {
	h := startEngine(t)
	h.send("stop")
	h.send("isready")
	if !h.waitFor("readyok", time.Second) {
		t.Fatalf("no readyok within 1s; output:\n%s", h.out.String())
	}
	h.quit()
}

func TestOverlappingGoDoesNotDeadlock(t *testing.T) {
	h := startEngine(t)
	h.send("position startpos")
	h.send("go infinite")
	// No "stop" before this second "go": a misbehaving GUI, but the
	// engine must cancel the first search itself rather than hang.
	h.send("go movetime 100")

	if !h.waitFor("bestmove", 2*time.Second) {
		t.Fatalf("no bestmove within 2s of overlapping go; output:\n%s", h.out.String())
	}
	h.quit()
}

func TestUciDeclaresThreadsAndHashOptions(t *testing.T) {
	h := startEngine(t)
	h.send("uci")
	if !h.waitFor("uciok", time.Second) {
		t.Fatalf("no uciok within 1s; output:\n%s", h.out.String())
	}
	out := h.out.String()
	if !strings.Contains(out, "option name Threads type spin default 1 min 1 max 512") {
		t.Errorf("missing Threads option declaration; output:\n%s", out)
	}
	if !strings.Contains(out, "option name Hash type spin default 16 min 1 max 33554432") {
		t.Errorf("missing Hash option declaration; output:\n%s", out)
	}
	h.quit()
}

func TestSetOptionThreadsAndHashDoNotBreakGo(t *testing.T) {
	h := startEngine(t)
	h.send("setoption name Threads value 4")
	h.send("setoption name Hash value 8")
	h.send("position startpos")
	h.send("go movetime 200")
	if !h.waitFor("bestmove", 2*time.Second) {
		t.Fatalf("no bestmove within 2s after configuring Threads/Hash; output:\n%s", h.out.String())
	}
	h.quit()
}

func TestSetOptionIgnoresUnknownOptionName(t *testing.T) {
	h := startEngine(t)
	h.send("setoption name SomeUnknownOption value 42")
	h.send("isready")
	if !h.waitFor("readyok", time.Second) {
		t.Fatalf("no readyok after an unknown setoption; output:\n%s", h.out.String())
	}
	h.quit()
}

func TestParseSetOption(t *testing.T) {
	name, value, ok := parseSetOption([]string{"name", "Threads", "value", "4"})
	if !ok || name != "Threads" || value != "4" {
		t.Errorf("parseSetOption(Threads) = (%q, %q, %v), want (Threads, 4, true)", name, value, ok)
	}

	name, value, ok = parseSetOption([]string{"name", "Some", "Multi", "Word", "Option", "value", "a", "b"})
	if !ok || name != "Some Multi Word Option" || value != "a b" {
		t.Errorf("parseSetOption(multi-word) = (%q, %q, %v), want (\"Some Multi Word Option\", \"a b\", true)", name, value, ok)
	}

	// "value" is optional (e.g. a button-type option).
	name, value, ok = parseSetOption([]string{"name", "Ponder"})
	if !ok || name != "Ponder" || value != "" {
		t.Errorf("parseSetOption(no value) = (%q, %q, %v), want (Ponder, \"\", true)", name, value, ok)
	}

	if _, _, ok = parseSetOption([]string{}); ok {
		t.Error("parseSetOption(empty) = ok, want false")
	}
	if _, _, ok = parseSetOption([]string{"value", "4"}); ok {
		t.Error("parseSetOption without leading \"name\" = ok, want false")
	}
}

func TestClampInt(t *testing.T) {
	cases := []struct{ v, min, max, want int }{
		{5, 1, 10, 5},
		{-5, 1, 10, 1},
		{50, 1, 10, 10},
		{1, 1, 10, 1},
		{10, 1, 10, 10},
	}
	for _, c := range cases {
		if got := clampInt(c.v, c.min, c.max); got != c.want {
			t.Errorf("clampInt(%d, %d, %d) = %d, want %d", c.v, c.min, c.max, got, c.want)
		}
	}
}

func TestBuildGoOptionsMovetime(t *testing.T) {
	b := board.StartingBoard()
	opts, infinite, ponder, _ := buildGoOptions(&b, []string{"movetime", "500"})
	if opts.MaxTime != 500*time.Millisecond {
		t.Errorf("MaxTime = %v, want 500ms", opts.MaxTime)
	}
	if infinite || ponder {
		t.Errorf("infinite=%v ponder=%v, want both false", infinite, ponder)
	}
}

func TestBuildGoOptionsNodes(t *testing.T) {
	b := board.StartingBoard()
	opts, _, _, _ := buildGoOptions(&b, []string{"nodes", "1234"})
	if opts.MaxIterations != 1234 {
		t.Errorf("MaxIterations = %d, want 1234", opts.MaxIterations)
	}
}

func TestBuildGoOptionsInfinite(t *testing.T) {
	b := board.StartingBoard()
	_, infinite, ponder, _ := buildGoOptions(&b, []string{"infinite"})
	if !infinite || ponder {
		t.Errorf("infinite=%v ponder=%v, want infinite=true ponder=false", infinite, ponder)
	}
}

func TestBuildGoOptionsUsesClockForSideToMove(t *testing.T) {
	white := board.StartingBoard() // White to move
	opts, _, _, _ := buildGoOptions(&white, []string{"wtime", "60000", "btime", "1000", "movestogo", "20"})
	if _, want := allocateTime(60000, 0, 20); opts.MaxTime != want {
		t.Errorf("White MaxTime = %v, want %v (from wtime)", opts.MaxTime, want)
	}

	black, err := board.ParseFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
	if err != nil {
		t.Fatalf("ParseFEN: %v", err)
	}
	opts, _, _, _ = buildGoOptions(&black, []string{"wtime", "60000", "btime", "1000", "movestogo", "20"})
	if _, want := allocateTime(1000, 0, 20); opts.MaxTime != want {
		t.Errorf("Black MaxTime = %v, want %v (from btime)", opts.MaxTime, want)
	}
}

func TestBuildGoOptionsPonderBudget(t *testing.T) {
	b := board.StartingBoard()
	_, _, ponder, budget := buildGoOptions(&b, []string{"ponder", "wtime", "10000", "btime", "10000", "movestogo", "10"})
	if !ponder {
		t.Fatal("ponder = false, want true")
	}
	if want, _ := allocateTime(10000, 0, 10); budget != want {
		t.Errorf("ponderBudget = %v, want %v", budget, want)
	}

	_, _, ponder, budget = buildGoOptions(&b, []string{"ponder"})
	if !ponder {
		t.Fatal("ponder = false, want true")
	}
	if budget <= 0 {
		t.Errorf("ponderBudget = %v with no clock given, want a positive fallback", budget)
	}
}

func TestAllocateTime(t *testing.T) {
	cases := []struct {
		name                string
		remaining, inc, mtg int
		want                time.Duration
	}{
		// 60000/30 (assumed movestogo) + 0
		{"plenty of time, assumed movestogo", 60000, 0, 0, 2000 * time.Millisecond},
		// 10000/10 + 0
		{"respects explicit movestogo", 10000, 0, 10, 1000 * time.Millisecond},
		// 100/1 + 0 = 100, capped to remaining(100) - safety buffer(50)
		{"never exceeds remaining minus buffer", 100, 0, 1, 50 * time.Millisecond},
		// 100000/30 + 500 = 3333 + 500, well under the safety cap
		{"increment adds to the budget", 100000, 500, 30, 3833 * time.Millisecond},
		// remaining <= 0 short-circuits to the minimum move time
		{"zero remaining still returns a positive budget", 0, 0, 0, minMoveTimeMs * time.Millisecond},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, hard := allocateTime(c.remaining, c.inc, c.mtg)
			if got != c.want {
				t.Errorf("allocateTime(%d, %d, %d) target = %v, want %v", c.remaining, c.inc, c.mtg, got, c.want)
			}
			// The hard cap is what a move may spend when the search asks
			// for it: never below the target, and never into the clock's
			// safety buffer.
			if hard < got {
				t.Errorf("allocateTime(%d, %d, %d) hard cap %v is below the target %v", c.remaining, c.inc, c.mtg, hard, got)
			}
			if usable := time.Duration(c.remaining-timeSafetyBufferMs) * time.Millisecond; c.remaining > 0 && hard > usable {
				t.Errorf("allocateTime(%d, %d, %d) hard cap %v exceeds the usable clock %v", c.remaining, c.inc, c.mtg, hard, usable)
			}
		})
	}
}

// TestEvalCommandReportsPerPieceBreakdown checks the non-standard "eval"
// command surfaces nnue.Explain's breakdown for the current position,
// including the two lines (baseline, residual) that make the decomposition
// honest rather than a set of numbers that mysteriously don't add up.
func TestEvalCommandReportsPerPieceBreakdown(t *testing.T) {
	h := startEngine(t)
	// Black is missing its queen, so White's queen on d1 should be the
	// standout contributor and the score should favor White decisively.
	h.send("position fen rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
	h.send("eval")
	if !h.waitFor("total", time.Second) {
		t.Fatalf("eval produced no total line within 1s; output:\n%s", h.out.String())
	}
	h.quit()

	got := h.out.String()
	for _, want := range []string{
		"NNUE evaluation:",
		"White to move",
		"baseline (kings only)",
		"residual (interaction)",
		"Q d1",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("eval output missing %q; got:\n%s", want, got)
		}
	}
}

// TestEvalCommandDoesNotDisturbTheProtocol makes sure the non-standard
// command leaves the loop in a working state — a GUI that never sends
// "eval" must be unaffected, and one that does must still get its search.
func TestEvalCommandDoesNotDisturbTheProtocol(t *testing.T) {
	h := startEngine(t)
	h.send("position startpos")
	h.send("eval")
	h.send("isready")
	if !h.waitFor("readyok", time.Second) {
		t.Fatalf("no readyok after eval; output:\n%s", h.out.String())
	}
	h.send("go movetime 100")
	if !h.waitFor("bestmove", 2*time.Second) {
		t.Fatalf("no bestmove after eval; output:\n%s", h.out.String())
	}
	h.quit()
}
