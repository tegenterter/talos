package uci

import (
	"bytes"
	"strconv"
	"strings"
	"testing"
	"time"
)

// nodesSearchedFrom pulls the total off the conventional "Nodes searched: N"
// line, which is the shape engine-testing tooling greps for.
func nodesSearchedFrom(t *testing.T, out string) int {
	t.Helper()
	for _, line := range strings.Split(out, "\n") {
		rest, found := strings.CutPrefix(line, "Nodes searched:")
		if !found {
			continue
		}
		n, err := strconv.Atoi(strings.TrimSpace(rest))
		if err != nil {
			t.Fatalf("unparsable node total %q: %v", line, err)
		}
		return n
	}
	t.Fatalf("no %q line in bench output:\n%s", "Nodes searched:", out)
	return 0
}

// TestBenchIsDeterministic is the property that makes bench usable as a
// regression signal at all: the same build must produce the same node total
// every time, so that a total which *does* change identifies a real search
// behaviour change rather than noise. It is why bench pins Threads to 1 and
// uses a fresh table per position.
func TestBenchIsDeterministic(t *testing.T) {
	// A full-depth bench takes ~10s; depth is lowered here purely for test
	// runtime. Determinism doesn't depend on the depth.
	restore := benchDepth
	benchDepth = 4
	defer func() { benchDepth = restore }()

	var first, second bytes.Buffer
	RunBench(&first)
	RunBench(&second)

	a, b := nodesSearchedFrom(t, first.String()), nodesSearchedFrom(t, second.String())
	if a != b {
		t.Errorf("bench node totals differ between runs: %d then %d", a, b)
	}
	if a == 0 {
		t.Error("bench searched 0 nodes")
	}
	// Compare the per-position lines too, so a change that shifts work
	// between positions while coincidentally preserving the total is still
	// caught. Timing lines are excluded because elapsed time and nps are
	// measurements of the machine, not of the search, and legitimately
	// differ run to run — that separation is the point of reporting them
	// apart from the node counts.
	if a, b := deterministicLines(first.String()), deterministicLines(second.String()); a != b {
		t.Errorf("bench per-position output differs between runs:\n%s\n---\n%s", a, b)
	}
}

// deterministicLines drops the lines whose content depends on how fast the
// machine ran, leaving only what the search itself determines.
func deterministicLines(out string) string {
	var kept []string
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "Time") || strings.HasPrefix(line, "NPS") {
			continue
		}
		kept = append(kept, line)
	}
	return strings.Join(kept, "\n")
}

// TestBenchCommandRunsOverUCI checks the command is actually wired into the
// protocol loop, not just callable from Go.
func TestBenchCommandRunsOverUCI(t *testing.T) {
	restore := benchDepth
	benchDepth = 2
	defer func() { benchDepth = restore }()

	h := startEngine(t)
	h.send("bench")
	if !h.waitFor("Nodes searched:", 30*time.Second) {
		t.Fatalf("bench command produced no node total; output:\n%s", h.out.String())
	}
	h.quit()
}
