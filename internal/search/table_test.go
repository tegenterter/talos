package search

import (
	"testing"

	"talos/internal/board"
)

// TestTableIsReusedAcrossSearches checks that a caller-supplied Table really
// carries knowledge from one search into the next, rather than each Search
// starting from an empty table. Successive moves in a game search heavily
// overlapping trees, so a table rebuilt per move throws away precisely the
// results the next move would reuse — and re-allocates the whole thing, which
// at a large Hash setting is a gigabyte per move.
//
// A repeat search of the same position at the same depth is the clean
// observable: with a warm table the second run must visit strictly fewer
// nodes, since every position it reaches is already stored.
func TestTableIsReusedAcrossSearches(t *testing.T) {
	b := board.StartingBoard()
	tbl := NewTable(16)

	run := func(opts Options) int {
		opts.MaxDepth = 7
		opts.Threads = 1
		var last Info
		opts.OnInfo = func(i Info) { last = i }
		if _, ok := Search(b, opts); !ok {
			t.Fatal("Search reported no legal moves")
		}
		return last.Nodes
	}

	first := run(Options{Table: tbl})
	second := run(Options{Table: tbl})
	t.Logf("shared table: first search %d nodes, second %d", first, second)
	if second >= first {
		t.Errorf("second search with a warm table took %d nodes vs %d for the first; "+
			"the table does not appear to be carrying over", second, first)
	}

	// Without a shared table each search allocates its own, so two identical
	// runs must cost identically — this is what makes fixed-depth results
	// reproducible, and it confirms the reduction above came from reuse
	// rather than from anything else varying between runs.
	a := run(Options{})
	c := run(Options{})
	if a != c {
		t.Errorf("two table-less searches of the same position took %d and %d nodes, want equal", a, c)
	}
}

// TestNewTableSizesFromHashMB checks the exported constructor honours the
// "Hash" option, including its zero-means-default behaviour, since internal/uci
// relies on both when it allocates and reallocates the shared table.
func TestNewTableSizesFromHashMB(t *testing.T) {
	small := NewTable(1)
	big := NewTable(64)
	if totalTTEntries(big.tt) <= totalTTEntries(small.tt) {
		t.Errorf("NewTable(64) has %d entries, want more than NewTable(1)'s %d",
			totalTTEntries(big.tt), totalTTEntries(small.tt))
	}
	if got, want := totalTTEntries(NewTable(0).tt), totalTTEntries(NewTable(DefaultHashMB).tt); got != want {
		t.Errorf("NewTable(0) has %d entries, want DefaultHashMB's %d", got, want)
	}
}
