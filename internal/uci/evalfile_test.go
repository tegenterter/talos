package uci

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"testing"
	"time"

	"talos/internal/nnue"
)

// shiftedNetwork writes a copy of the embedded network with its output-layer
// bias moved by shiftCP centipawns, and returns the path.
//
// A network that differs from the embedded one in a way that is *obvious in
// the output* is what makes the tests below able to tell which network the
// engine actually used — the thing that was silently wrong before. The output
// bias is the cleanest such knob: it shifts every position by a known amount,
// so the assertion is arithmetic rather than "the number changed".
func shiftedNetwork(t *testing.T, shiftCP int) string {
	t.Helper()

	data, err := os.ReadFile(filepath.Join("..", "nnue", "nn-97f742aaefcd.nnue"))
	if err != nil {
		t.Fatalf("reading embedded network: %v", err)
	}

	// The output-layer bias is the last int32 before the final layer's 32
	// int8 weights — see internal/nnue's Load for the section order.
	off := len(data) - 32 - 4
	bias := int32(binary.LittleEndian.Uint32(data[off:]))

	// outputScale converts the network's internal unit to centipawns, so a
	// centipawn shift is that many units.
	shifted := make([]byte, len(data))
	copy(shifted, data)
	binary.LittleEndian.PutUint32(shifted[off:], uint32(bias+int32(shiftCP*16)))

	path := filepath.Join(t.TempDir(), "shifted.nnue")
	if err := os.WriteFile(path, shifted, 0o600); err != nil {
		t.Fatalf("writing shifted network: %v", err)
	}
	// Cheap sanity check that the file is still a network at all, so a
	// failure below points at the engine rather than at this helper.
	if _, err := nnue.LoadFile(path); err != nil {
		t.Fatalf("shifted network does not parse: %v", err)
	}
	return path
}

var evalCP = regexp.MustCompile(`NNUE evaluation: ([+-]\d+) cp`)

func lastEvalCP(t *testing.T, out string) int {
	t.Helper()
	m := evalCP.FindAllStringSubmatch(out, -1)
	if len(m) == 0 {
		t.Fatalf("no NNUE evaluation line in output:\n%s", out)
	}
	v, err := strconv.Atoi(m[len(m)-1][1])
	if err != nil {
		t.Fatalf("unparsable evaluation %q: %v", m[len(m)-1][1], err)
	}
	return v
}

// TestEvalCommandUsesTheLoadedNetwork is a regression test for a real bug:
// "eval" called nnue.Explain, which reads the *embedded* network directly, so
// it kept reporting the built-in network's opinion after "setoption name
// EvalFile" had loaded a different one. For a command whose entire purpose is
// inspecting what the network thinks, that is exactly backwards — and it
// would have quietly invalidated the parity harness that compares a freshly
// trained network against the trainer that produced it.
func TestEvalCommandUsesTheLoadedNetwork(t *testing.T) {
	const shift = 300
	path := shiftedNetwork(t, shift)

	// A fresh engine per measurement rather than two "eval"s down one: the
	// harness matches on everything printed so far, so a second "eval" would
	// be satisfied by the first one's output and silently compare a number
	// against itself.
	evalOf := func(setup func(*harness)) int {
		h := startEngine(t)
		setup(h)
		h.send("position startpos")
		h.send("eval")
		if !h.waitFor("NNUE evaluation", time.Second) {
			t.Fatalf("no eval output; got:\n%s", h.out.String())
		}
		out := h.out.String()
		h.quit()
		return lastEvalCP(t, out)
	}

	embedded := evalOf(func(*harness) {})
	loaded := evalOf(func(h *harness) {
		h.send("setoption name EvalFile value " + path)
		if !h.waitFor("EvalFile: loaded", time.Second) {
			t.Fatalf("network was not loaded; got:\n%s", h.out.String())
		}
	})

	if got := loaded - embedded; got != shift {
		t.Errorf("eval moved by %+d cp after loading a network shifted by %+d cp\n"+
			"  embedded: %+d\n  loaded:   %+d\n"+
			"a difference of zero means \"eval\" is still reading the embedded network",
			got, shift, embedded, loaded)
	}
}

// TestSearchUsesTheLoadedNetwork covers the other half: the search itself must
// play with the loaded network, not merely report it. A network that scores
// every position 300cp better for the side to move changes what the engine
// says about the position it searches.
func TestSearchUsesTheLoadedNetwork(t *testing.T) {
	const shift = 300
	path := shiftedNetwork(t, shift)

	scoreOf := func(setup func(*harness)) int {
		h := startEngine(t)
		setup(h)
		h.send("position startpos")
		h.send("go depth 4")
		if !h.waitFor("bestmove", 5*time.Second) {
			t.Fatalf("no bestmove; got:\n%s", h.out.String())
		}
		out := h.out.String()
		h.quit()

		m := regexp.MustCompile(`score cp (-?\d+)`).FindAllStringSubmatch(out, -1)
		if len(m) == 0 {
			t.Fatalf("no score in search output:\n%s", out)
		}
		v, _ := strconv.Atoi(m[len(m)-1][1])
		return v
	}

	embedded := scoreOf(func(*harness) {})
	loaded := scoreOf(func(h *harness) {
		h.send("setoption name EvalFile value " + path)
		if !h.waitFor("EvalFile: loaded", time.Second) {
			t.Fatalf("network was not loaded; got:\n%s", h.out.String())
		}
	})

	// Not an exact arithmetic check: a uniform evaluation shift does not move
	// a *searched* score by exactly that much, since the shift applies at
	// every leaf and alternating plies negate it. What must hold is that the
	// search noticed at all, and in the right direction.
	if loaded <= embedded {
		t.Errorf("search scored %+d with a network shifted %+d cp in the side to move's favour, "+
			"against %+d with the embedded one; the loaded network is not reaching the search",
			loaded, shift, embedded)
	}
}
