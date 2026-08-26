package datagen

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"strconv"
	"strings"

	"talos/internal/board"
	"talos/internal/nnue"
)

// Packing turns the archival text format Run writes into fixed-size binary
// records a trainer can memory-map.
//
// The text format is deliberately kept as the thing generation produces: it is
// readable, diffable, and survives a change to the feature set. This is the
// derived form — regenerate it whenever the features change, and never treat
// it as the source of truth.
//
// The records carry **HalfKP feature indices computed by the engine itself**
// (nnue.ActiveFeatures), not the position. That is the point of the format: a
// trainer that derives features independently can disagree with the engine in
// ways nothing detects until the finished network simply plays badly. Here
// there is only one implementation, so there is nothing to disagree with.

const (
	// FeatureSlots is how many feature indices each perspective gets in a
	// record. Padded to a fixed width so records are addressable by index;
	// nnue.FeaturePadding (zero) fills the unused slots, which doubles as a
	// trainer's embedding padding index.
	FeatureSlots = 32

	// RecordSize is the on-disk size of one packed position.
	RecordSize = 2*FeatureSlots*2 + 2 + 1 + 1 // stm + opp features, score, result, pad

	// scoreClamp bounds the stored evaluation. Records come from Run, which
	// already drops mate scores, so this only guards against a pathological
	// evaluation overflowing the int16 it is stored in.
	scoreClamp = 30000
)

// Record is one packed training position, in the layout written to disk:
// side-to-move features, opponent features, score, result.
type Record struct {
	STM    [FeatureSlots]uint16
	Opp    [FeatureSlots]uint16
	Score  int16 // centipawns, from the side to move
	Result int8  // -1, 0 or 1, from the side to move
}

// Pack reads the text format Run produces and writes packed records,
// returning how many it wrote.
func Pack(in io.Reader, out io.Writer) (int, error) {
	r := bufio.NewScanner(in)
	// Positions are one short record each, but a pathological line should
	// fail loudly rather than silently truncate.
	r.Buffer(make([]byte, 0, 64*1024), 1<<20)

	w := bufio.NewWriterSize(out, 1<<20)
	buf := make([]byte, RecordSize)

	var (
		fen    string
		score  int
		result int
		count  int
		line   int
	)

	for r.Scan() {
		line++
		field, value, _ := strings.Cut(r.Text(), " ")
		switch field {
		case "fen":
			fen = value
		case "score":
			v, err := strconv.Atoi(value)
			if err != nil {
				return count, fmt.Errorf("datagen: line %d: unparsable score %q", line, value)
			}
			score = v
		case "result":
			v, err := strconv.Atoi(value)
			if err != nil {
				return count, fmt.Errorf("datagen: line %d: unparsable result %q", line, value)
			}
			result = v
		case "move", "ply":
			// Recorded for humans and for future use; the network is trained
			// on the evaluation and the outcome, not on the move played.
		case "e":
			rec, err := packOne(fen, score, result)
			if err != nil {
				return count, fmt.Errorf("datagen: line %d: %w", line, err)
			}
			encodeRecord(buf, rec)
			if _, err := w.Write(buf); err != nil {
				return count, err
			}
			count++
			fen, score, result = "", 0, 0
		default:
			return count, fmt.Errorf("datagen: line %d: unexpected field %q", line, field)
		}
	}
	if err := r.Err(); err != nil {
		return count, err
	}
	return count, w.Flush()
}

// packOne builds the record for a single position.
func packOne(fen string, score, result int) (Record, error) {
	if fen == "" {
		return Record{}, fmt.Errorf("record has no position")
	}
	b, err := board.ParseFEN(fen)
	if err != nil {
		return Record{}, fmt.Errorf("parsing %q: %w", fen, err)
	}
	if result < -1 || result > 1 {
		return Record{}, fmt.Errorf("result %d outside -1..1", result)
	}

	stm := b.SideToMove
	rec := Record{
		Score:  int16(clamp(score, -scoreClamp, scoreClamp)),
		Result: int8(result),
	}
	if err := fill(&rec.STM, &b, stm); err != nil {
		return Record{}, err
	}
	if err := fill(&rec.Opp, &b, stm.Opposite()); err != nil {
		return Record{}, err
	}
	return rec, nil
}

// fill writes one perspective's features into a fixed-width slot array,
// leaving the remainder as padding.
func fill(dst *[FeatureSlots]uint16, b *board.Board, perspective board.Color) error {
	features := nnue.ActiveFeatures(dst[:0], b, perspective)
	if len(features) > FeatureSlots {
		return fmt.Errorf("%d active features exceeds the %d slots per record", len(features), FeatureSlots)
	}
	for i := len(features); i < FeatureSlots; i++ {
		dst[i] = nnue.FeaturePadding
	}
	return nil
}

// encodeRecord writes a record into buf, which must be RecordSize long.
// Little-endian throughout, matching the .nnue format the engine already
// reads and what numpy defaults to on the platforms this runs on.
func encodeRecord(buf []byte, rec Record) {
	off := 0
	for _, f := range rec.STM {
		binary.LittleEndian.PutUint16(buf[off:], f)
		off += 2
	}
	for _, f := range rec.Opp {
		binary.LittleEndian.PutUint16(buf[off:], f)
		off += 2
	}
	binary.LittleEndian.PutUint16(buf[off:], uint16(rec.Score))
	off += 2
	buf[off] = byte(rec.Result)
	buf[off+1] = 0 // padding, keeping records 4-byte aligned
}

// DecodeRecord reads a record back. It exists for tests and for anything
// inspecting a dump; the trainer reads the same layout with numpy.
func DecodeRecord(buf []byte) Record {
	var rec Record
	off := 0
	for i := range rec.STM {
		rec.STM[i] = binary.LittleEndian.Uint16(buf[off:])
		off += 2
	}
	for i := range rec.Opp {
		rec.Opp[i] = binary.LittleEndian.Uint16(buf[off:])
		off += 2
	}
	rec.Score = int16(binary.LittleEndian.Uint16(buf[off:]))
	rec.Result = int8(buf[off+2])
	return rec
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
