package datagen

import (
	"bytes"
	"strconv"
	"strings"
	"testing"

	"talos/internal/board"
)

// record is one parsed training sample.
type record struct {
	fen    string
	move   string
	score  int
	ply    int
	result int
}

func parseRecords(t *testing.T, data string) []record {
	t.Helper()
	var out []record
	var cur record
	for _, line := range strings.Split(strings.TrimSpace(data), "\n") {
		field, value, _ := strings.Cut(line, " ")
		switch field {
		case "fen":
			cur = record{fen: value}
		case "move":
			cur.move = value
		case "score":
			cur.score, _ = strconv.Atoi(value)
		case "ply":
			cur.ply, _ = strconv.Atoi(value)
		case "result":
			cur.result, _ = strconv.Atoi(value)
		case "e":
			out = append(out, cur)
		default:
			t.Fatalf("unexpected field %q in output", field)
		}
	}
	return out
}

func generate(t *testing.T, opts Options) []record {
	t.Helper()
	var buf bytes.Buffer
	opts.Out = &buf
	if err := Run(opts); err != nil {
		t.Fatalf("Run: %v", err)
	}
	return parseRecords(t, buf.String())
}

func testOptions() Options {
	o := DefaultOptions()
	o.Games, o.Nodes, o.MaxPlies = 6, 1500, 120
	return o
}

// TestRecordsAreWellFormed checks the output is data a trainer can actually
// read: every FEN parses, every move is legal in the position it is attached
// to, and every result is one of the three legal values.
func TestRecordsAreWellFormed(t *testing.T) {
	records := generate(t, testOptions())
	if len(records) == 0 {
		t.Fatal("no positions generated")
	}

	for i, r := range records {
		b, err := board.ParseFEN(r.fen)
		if err != nil {
			t.Fatalf("record %d: FEN %q does not parse: %v", i, r.fen, err)
		}
		mv, ok := board.ParseUCIMove(r.move)
		if !ok {
			t.Fatalf("record %d: move %q does not parse", i, r.move)
		}
		// Compared by from/to/promotion rather than as whole structs: the
		// generator marks a double pawn push, an en passant capture and a
		// castle with a Flag that UCI notation does not carry, so a parsed
		// move is never struct-equal to the generated one. internal/uci's
		// matchLegalMove compares the same three fields for the same reason.
		legal := false
		for _, m := range board.GenerateLegalMoves(&b) {
			if m.From == mv.From && m.To == mv.To && m.Promotion == mv.Promotion {
				legal = true
				break
			}
		}
		if !legal {
			t.Fatalf("record %d: move %s is not legal in %s", i, r.move, r.fen)
		}
		if r.result != -1 && r.result != 0 && r.result != 1 {
			t.Fatalf("record %d: result %d, want -1, 0 or 1", i, r.result)
		}
	}
	t.Logf("%d positions from %d games", len(records), testOptions().Games)
}

// TestFiltersExcludeUntrainablePositions covers the exclusions that make
// this a *static evaluator's* training set: a position in check, or one
// whose best move is a capture, has a value the search resolves rather than
// one the evaluator should be asked to guess.
func TestFiltersExcludeUntrainablePositions(t *testing.T) {
	for i, r := range generate(t, testOptions()) {
		b, err := board.ParseFEN(r.fen)
		if err != nil {
			t.Fatalf("record %d: %v", i, err)
		}

		kingSq := b.Pieces[b.SideToMove][board.King].LSB()
		if board.IsSquareAttacked(&b, kingSq, b.SideToMove.Opposite()) {
			t.Errorf("record %d is in check: %s", i, r.fen)
		}

		mv, _ := board.ParseUCIMove(r.move)
		if _, _, occupied := b.PieceAt(mv.To); occupied {
			t.Errorf("record %d's best move %s is a capture: %s", i, r.move, r.fen)
		}
	}
}

// TestRunIsReproducible pins the property that makes a data run debuggable
// and a bug report actionable: the same seed produces the same data.
func TestRunIsReproducible(t *testing.T) {
	opts := testOptions()
	first, second := generate(t, opts), generate(t, opts)

	if len(first) != len(second) {
		t.Fatalf("same seed produced %d positions and then %d", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("record %d differs between runs with the same seed:\n %+v\n %+v", i, first[i], second[i])
		}
	}

	opts.Seed++
	if other := generate(t, opts); len(other) == len(first) && len(first) > 0 && other[0] == first[0] {
		t.Error("a different seed produced the same first position; the seed is not being used")
	}
}

// TestResultsAgreeWithinAGame checks the labels are consistent: a game has
// one outcome, and each position records it from its own side to move — so
// consecutive positions, which alternate colours, must disagree in sign
// unless the game was drawn.
func TestResultsAgreeWithinAGame(t *testing.T) {
	opts := testOptions()
	opts.Games = 1
	records := generate(t, opts)
	if len(records) < 2 {
		t.Skip("game too short to compare")
	}

	for i := 1; i < len(records); i++ {
		prev, cur := records[i-1], records[i]
		if cur.ply != prev.ply+1 {
			continue // a filtered position sits between them; colours unknown
		}
		if cur.result != -prev.result {
			t.Errorf("plies %d and %d record results %d and %d; consecutive positions alternate colour, so they must be opposites",
				prev.ply, cur.ply, prev.result, cur.result)
		}
	}
}
