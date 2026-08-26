package search

import (
	"testing"
	"time"
)

// TestSoftBudgetScaling pins the two signals the soft budget reacts to, and
// the bounds on both. Time management is the one part of this search where
// being clever has repeatedly backfired (see softBudget's doc comment), so
// what matters here is less the exact numbers than that they stay bounded
// and point the right way.
func TestSoftBudgetScaling(t *testing.T) {
	const soft, hard = 1000 * time.Millisecond, 3000 * time.Millisecond

	base := softBudget(soft, hard, 0, 0)
	if base != soft {
		t.Errorf("with nothing to react to, budget = %v, want the target %v", base, soft)
	}

	// A settled root move gives time back.
	stable := softBudget(soft, hard, 5, 0)
	if stable >= base {
		t.Errorf("budget with a stable root move = %v, want less than %v", stable, base)
	}
	if stable < soft/2 {
		t.Errorf("budget with a stable root move = %v, want no less than half the target", stable)
	}
	// The discount stops accumulating.
	if longer := softBudget(soft, hard, 50, 0); longer != stable {
		t.Errorf("budget after 50 stable iterations = %v, want the same as after 5 (%v)", longer, stable)
	}

	// A falling score buys more.
	falling := softBudget(soft, hard, 0, fallingScoreThreshold)
	if falling <= base {
		t.Errorf("budget with a falling score = %v, want more than %v", falling, base)
	}
	// ...but not without limit, and never past the hard deadline.
	if huge := softBudget(soft, hard, 0, 100000); huge > hard {
		t.Errorf("budget with a collapsing score = %v, want no more than the hard deadline %v", huge, hard)
	}
	if huge := softBudget(soft, hard, 0, 100000); huge > 2*soft {
		t.Errorf("budget with a collapsing score = %v, want the extension bounded", huge)
	}
}

// TestSoftBudgetWithoutSoftTime covers the "go movetime" contract: a caller
// that gave only a hard deadline is instructing the engine to search for
// that long, not handing it a budget to manage, so nothing scales it.
func TestSoftBudgetWithoutSoftTime(t *testing.T) {
	const hard = 2 * time.Second
	for _, stable := range []int{0, 3, 10} {
		for _, drop := range []int{0, 500} {
			if got := softBudget(0, hard, stable, drop); got != hard {
				t.Errorf("softBudget(0, %v, %d, %d) = %v, want the hard deadline unchanged", hard, stable, drop, got)
			}
		}
	}
	// A soft budget at or above the hard deadline is not a budget either.
	if got := softBudget(hard, hard, 5, 0); got != hard {
		t.Errorf("softBudget(hard, hard, ...) = %v, want %v", got, hard)
	}
}

// TestSearchSpendsLessOnASettledPosition is the end-to-end half: given a
// soft budget and a position whose best move never wavers, the search
// should hand time back rather than spend the lot.
func TestSearchSpendsLessOnASettledPosition(t *testing.T) {
	if testing.Short() {
		t.Skip("timing measurement")
	}
	b := mustFEN(t, "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1") // a settled, simple ending

	start := time.Now()
	if _, ok := Search(b, Options{SoftTime: 500 * time.Millisecond, MaxTime: 1500 * time.Millisecond, Threads: 1}); !ok {
		t.Fatal("Search reported no legal moves")
	}
	elapsed := time.Since(start)

	if elapsed >= 500*time.Millisecond {
		t.Errorf("search took %v of a 500ms target on a settled position; the stability discount is not being applied", elapsed)
	}
}
