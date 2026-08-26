package datagen

import (
	"bytes"
	"strings"
	"testing"

	"talos/internal/board"
	"talos/internal/nnue"
)

// TestPackedFeaturesMatchTheEngine is the reason this format exists: what a
// trainer reads has to be exactly what the engine evaluates with. Anything
// else fails silently — the network trains, loads, plays, and is merely bad.
func TestPackedFeaturesMatchTheEngine(t *testing.T) {
	fens := []string{
		board.StartFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
		"8/k7/2B5/b7/8/6r1/4K3/8 b - - 1 128",
	}

	var in strings.Builder
	for i, fen := range fens {
		in.WriteString("fen " + fen + "\nmove e2e4\nscore " + itoa(i*37-40) + "\nply 1\nresult " + itoa(i%3-1) + "\ne\n")
	}

	var out bytes.Buffer
	n, err := Pack(strings.NewReader(in.String()), &out)
	if err != nil {
		t.Fatalf("Pack: %v", err)
	}
	if n != len(fens) {
		t.Fatalf("packed %d records, want %d", n, len(fens))
	}
	if out.Len() != n*RecordSize {
		t.Fatalf("output is %d bytes, want %d records of %d", out.Len(), n, RecordSize)
	}

	data := out.Bytes()
	for i, fen := range fens {
		rec := DecodeRecord(data[i*RecordSize : (i+1)*RecordSize])
		b, err := board.ParseFEN(fen)
		if err != nil {
			t.Fatal(err)
		}

		if got, want := int(rec.Score), i*37-40; got != want {
			t.Errorf("%s: score %d, want %d", fen, got, want)
		}
		if got, want := int(rec.Result), i%3-1; got != want {
			t.Errorf("%s: result %d, want %d", fen, got, want)
		}

		stm := b.SideToMove
		checkPerspective(t, fen, "stm", rec.STM, &b, stm)
		checkPerspective(t, fen, "opp", rec.Opp, &b, stm.Opposite())
	}
}

// checkPerspective compares one perspective's packed slots against the
// engine's own feature list, including that the unused slots are padding.
func checkPerspective(t *testing.T, fen, which string, slots [FeatureSlots]uint16, b *board.Board, perspective board.Color) {
	t.Helper()

	want := nnue.ActiveFeatures(nil, b, perspective)
	for i, f := range want {
		if slots[i] != f {
			t.Errorf("%s (%s): slot %d holds feature %d, engine says %d", fen, which, i, slots[i], f)
		}
	}
	for i := len(want); i < FeatureSlots; i++ {
		if slots[i] != nnue.FeaturePadding {
			t.Errorf("%s (%s): slot %d past the %d real features holds %d, want padding",
				fen, which, i, len(want), slots[i])
		}
	}
}

// TestPackRoundTripsARecord pins the on-disk layout itself, independent of any
// position: what encodeRecord writes, DecodeRecord must read back — and a
// trainer reading the same bytes with numpy depends on that layout being
// exactly what it looks like.
func TestPackRoundTripsARecord(t *testing.T) {
	want := Record{Score: -1234, Result: 1}
	for i := range want.STM {
		want.STM[i] = uint16(i * 3)
		want.Opp[i] = uint16(40000 + i)
	}

	buf := make([]byte, RecordSize)
	encodeRecord(buf, want)
	if got := DecodeRecord(buf); got != want {
		t.Errorf("round trip changed the record:\n got  %+v\n want %+v", got, want)
	}

	// The trailing pad byte must be zero: it is there to keep records
	// 4-byte aligned, and a trainer may reasonably assume it is not data.
	if buf[RecordSize-1] != 0 {
		t.Errorf("pad byte is %d, want 0", buf[RecordSize-1])
	}
}

// TestPackNegativeScoresSurvive guards the one field where a signed value
// travels through an unsigned encoding.
func TestPackNegativeScoresSurvive(t *testing.T) {
	for _, score := range []int{-30000, -1, 0, 1, 30000} {
		in := "fen " + board.StartFEN + "\nmove e2e4\nscore " + itoa(score) + "\nply 0\nresult -1\ne\n"
		var out bytes.Buffer
		if _, err := Pack(strings.NewReader(in), &out); err != nil {
			t.Fatalf("Pack(score=%d): %v", score, err)
		}
		if got := int(DecodeRecord(out.Bytes()).Score); got != score {
			t.Errorf("score %d packed as %d", score, got)
		}
		if got := int(DecodeRecord(out.Bytes()).Result); got != -1 {
			t.Errorf("result -1 packed as %d", got)
		}
	}
}

// TestPackRejectsMalformedInput checks corruption is loud. A packer that
// silently skips what it cannot read produces a dataset quietly missing
// positions, which is far worse than a failed run.
func TestPackRejectsMalformedInput(t *testing.T) {
	tests := []struct {
		name string
		in   string
	}{
		{"unknown field", "fen " + board.StartFEN + "\nnonsense 1\ne\n"},
		{"unparsable score", "fen " + board.StartFEN + "\nscore abc\ne\n"},
		{"unparsable result", "fen " + board.StartFEN + "\nscore 0\nresult xyz\ne\n"},
		{"result out of range", "fen " + board.StartFEN + "\nscore 0\nresult 7\ne\n"},
		{"invalid position", "fen 8/8/8/8/8/8/8/8 w - - 0 1\nscore 0\nresult 0\ne\n"},
		{"record with no position", "score 0\nresult 0\ne\n"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			if _, err := Pack(strings.NewReader(tc.in), &out); err == nil {
				t.Error("Pack accepted malformed input, want an error")
			}
		})
	}
}

// TestPackConsumesGeneratedData closes the loop end to end: whatever Run
// writes, Pack must read, with one record per position.
func TestPackConsumesGeneratedData(t *testing.T) {
	opts := DefaultOptions()
	opts.Games, opts.Nodes, opts.MaxPlies = 3, 1200, 80
	var plain bytes.Buffer
	opts.Out = &plain
	if err := Run(opts); err != nil {
		t.Fatalf("Run: %v", err)
	}

	positions := strings.Count(plain.String(), "\ne\n")
	if positions == 0 {
		t.Fatal("generation produced no positions")
	}

	var packed bytes.Buffer
	n, err := Pack(strings.NewReader(plain.String()), &packed)
	if err != nil {
		t.Fatalf("Pack: %v", err)
	}
	if n != positions {
		t.Errorf("packed %d records from %d generated positions", n, positions)
	}
	if packed.Len() != n*RecordSize {
		t.Errorf("packed %d bytes, want %d", packed.Len(), n*RecordSize)
	}
}

func itoa(v int) string {
	if v < 0 {
		return "-" + itoa(-v)
	}
	if v < 10 {
		return string(rune('0' + v))
	}
	return itoa(v/10) + string(rune('0'+v%10))
}
