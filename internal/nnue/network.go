// Package nnue implements NNUE (Efficiently Updatable Neural Network)
// position evaluation with a real trained network, in the style
// Stockfish pioneered in 2020.
//
// It targets the original "HalfKP 256x2-32-32-1" network architecture —
// Stockfish 12's default net (nn-97f742aaefcd.nnue, CC0-licensed, from
// https://github.com/official-stockfish/networks) — rather than the
// current, much larger and more complex HalfKAv2_hm/multi-net architecture
// modern Stockfish ships. HalfKP is fully documented and its binary format
// is stable, which makes an independent, correct reimplementation
// tractable; HalfKAv2_hm is a moving target tied to specific Stockfish
// source revisions (threat features, layer stacks, dual big/small nets)
// and was judged too high-risk to reimplement with confidence. HalfKP is
// far weaker than current Stockfish's evaluator, but still a genuine NNUE
// network and a large step up from hand-tuned PSTs.
//
// Like the rest of this codebase, this package favors simplicity over raw
// speed: rather than maintaining the accumulator incrementally across
// make/unmake (the technique the "E" in NNUE refers to, and what real
// engines rely on for performance), Evaluate recomputes both perspectives'
// accumulators from scratch on every call by summing the ~30 or so active
// features' weight rows. That's cheap enough in absolute terms (a few
// thousand int16 additions) to not need incremental updates, and it keeps
// this package a pure function of a board position with no accumulator
// state to keep in sync — consistent with board.Board's own copy-based
// (no make/unmake) move generation.
package nnue

import (
	"encoding/binary"
	"fmt"
)

// Network dimensions for the HalfKP 256x2-32-32-1 architecture.
const (
	halfDimensions  = 256   // accumulator size per perspective
	inputFeatures   = 41024 // HalfKP feature count: 64 king squares * 641 piece-square slots
	ftScale         = 127   // feature-transformer output clamp / "1.0" point
	hiddenDim       = 32
	weightScaleBits = 6  // right-shift applied after each hidden affine layer
	outputScale     = 16 // final divisor converting the network's output unit to centipawns
)

const (
	nnueVersion = 0x7AF32F16
)

// layer holds one quantized affine (fully-connected) layer: out = biases + weights @ in.
// weights is stored row-major: weights[row*in + col].
type layer struct {
	in, out int
	biases  []int32
	weights []int8
	// weightsT is weights transposed to column-major (weightsT[c*out+r] ==
	// weights[r*in+c]), built once at load time for the layers evaluated
	// sparsely — see forwardSparseInto. Nil for the layers that aren't.
	weightsT []int8
}

// transpose fills weightsT, which is what lets a column (one input's weights
// across every output) be read contiguously.
func (l *layer) transpose() {
	l.weightsT = make([]int8, len(l.weights))
	for r := 0; r < l.out; r++ {
		for c := 0; c < l.in; c++ {
			l.weightsT[c*l.out+r] = l.weights[r*l.in+c]
		}
	}
}

// forwardInto computes this layer's affine transform into a caller-owned
// destination, which must be l.out long. The evaluation path calls this
// with stack arrays; allocating the output here instead put three
// allocations per evaluation through the garbage collector, and this is the
// hottest function in the engine (a third of all CPU) so it is also written
// for the compiler rather than for looks:
//
//   - `row = row[:len(in)]` gives the bounds-check eliminator a length it
//     can prove, removing a check per multiply-add;
//   - four independent accumulators let the CPU overlap four multiply-adds
//     instead of serializing on one dependency chain.
//
// Both are portable Go — this package deliberately has no assembly, so the
// only speed available here is what the compiler can be persuaded to emit.
// forwardSparseInto is forwardInto for the first layer, whose input is the
// feature transformer's ClippedReLU output and is mostly zeros: a zero input
// contributes nothing to any output, so its whole column can be skipped
// rather than multiplied by 32 weights. It takes the raw accumulator and the
// column offset of the perspective it holds (0 for the side to move, 256 for
// the opponent), applying the activation itself — see the loop below.
//
// out must already hold the layer's biases; the caller adds both
// perspectives into the same accumulation.
//
// Measured over random positions, 81% of the values reaching this layer are
// zero — the feature transformer's ClippedReLU floors at zero and most
// features simply do not fire — so this does roughly a fifth of the dense
// version's arithmetic. That is worth restructuring the weights for, and it
// is the kind of win available in portable Go where SIMD is not: the dense
// loop is already about as good as the compiler will make it, but the
// cheapest multiply-add is the one not performed.
func (l *layer) forwardSparseInto(acc []int16, colOffset int, out []int32) {
	out = out[:l.out]

	for i, v := range acc {
		// clampFT(v) is zero for every v <= 0, so the accumulator can be
		// read directly: no clamped copy of it has to be materialized, and
		// the zero test that skips the column doubles as the activation's
		// lower bound. Fusing the two removed a 512-element buffer and a
		// clamp pass that was 8% of CPU on its own once the multiplies
		// below got cheap enough for it to matter.
		if v <= 0 {
			continue
		}
		x := int32(v)
		if x > ftScale {
			x = ftScale
		}
		col := l.weightsT[(colOffset+i)*l.out:]
		col = col[:len(out)]
		for r, w := range col {
			out[r] += int32(w) * x
		}
	}
}

func (l *layer) forwardInto(in []int8, out []int32) {
	in = in[:l.in]
	for r := 0; r < l.out; r++ {
		row := l.weights[r*l.in:]
		row = row[:len(in)]

		var s0, s1, s2, s3 int32
		c := 0
		for ; c+4 <= len(in); c += 4 {
			s0 += int32(row[c]) * int32(in[c])
			s1 += int32(row[c+1]) * int32(in[c+1])
			s2 += int32(row[c+2]) * int32(in[c+2])
			s3 += int32(row[c+3]) * int32(in[c+3])
		}
		for ; c < len(in); c++ {
			s0 += int32(row[c]) * int32(in[c])
		}
		out[r] = l.biases[r] + s0 + s1 + s2 + s3
	}
}

// Network is a loaded, ready-to-evaluate HalfKP 256x2-32-32-1 NNUE network.
type Network struct {
	ftBiases  []int16 // [halfDimensions]
	ftWeights []int16 // [inputFeatures][halfDimensions], row-major by feature index

	l1, l2, out layer // 512->32, 32->32, 32->1
}

// Load parses a Stockfish HalfKP 256x2-32-32-1 .nnue file.
func Load(data []byte) (*Network, error) {
	r := &reader{data: data}

	version := r.u32()
	_ = r.u32() // combined file hash, not independently verified here
	archLen := int(r.u32())
	r.skip(archLen) // human-readable architecture description string
	if r.err != nil {
		return nil, r.err
	}
	if version != nnueVersion {
		return nil, fmt.Errorf("nnue: unsupported network version %#x (want %#x, HalfKP 256x2-32-32-1)", version, uint32(nnueVersion))
	}

	_ = r.u32() // feature-transformer section hash
	n := &Network{}
	n.ftBiases = r.i16s(halfDimensions)
	n.ftWeights = r.i16s(inputFeatures * halfDimensions)

	_ = r.u32() // network (layer stack) section hash
	n.l1 = r.affineLayer(2*halfDimensions, hiddenDim)
	// The first layer is 512x32 of the forward pass's 17,440 multiply-adds,
	// i.e. essentially all of it, and its input is the feature transformer's
	// ClippedReLU output — which is measurably ~81% zeros on real positions.
	// Transposing lets forwardSparseInto skip those columns outright.
	n.l1.transpose()
	n.l2 = r.affineLayer(hiddenDim, hiddenDim)
	n.out = r.affineLayer(hiddenDim, 1)

	if r.err != nil {
		return nil, r.err
	}
	if r.pos != len(data) {
		return nil, fmt.Errorf("nnue: %d trailing bytes after parsing network", len(data)-r.pos)
	}
	return n, nil
}

// reader is a small sequential little-endian byte-stream decoder that
// records the first error it hits and turns every subsequent read into a
// no-op, so Load can check r.err once at the end instead of after every field.
type reader struct {
	data []byte
	pos  int
	err  error
}

func (r *reader) need(n int) []byte {
	if r.err != nil {
		return nil
	}
	if r.pos+n > len(r.data) {
		r.err = fmt.Errorf("nnue: unexpected end of file at offset %d, need %d more bytes", r.pos, n)
		return nil
	}
	b := r.data[r.pos : r.pos+n]
	r.pos += n
	return b
}

func (r *reader) skip(n int) { r.need(n) }

func (r *reader) u32() uint32 {
	b := r.need(4)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(b)
}

func (r *reader) i16s(count int) []int16 {
	b := r.need(2 * count)
	if b == nil {
		return nil
	}
	out := make([]int16, count)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return out
}

func (r *reader) i32s(count int) []int32 {
	b := r.need(4 * count)
	if b == nil {
		return nil
	}
	out := make([]int32, count)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func (r *reader) i8s(count int) []int8 {
	b := r.need(count)
	if b == nil {
		return nil
	}
	out := make([]int8, count)
	for i := range out {
		out[i] = int8(b[i])
	}
	return out
}

// affineLayer reads a quantized affine layer as Stockfish's NNUE format
// stores it: `out` int32 biases, then `out*in` int8 weights, row-major.
// Neither dimension used in this architecture (512, 32, 1) needs SIMD
// alignment padding, since all of them are already multiples of 32.
func (r *reader) affineLayer(in, out int) layer {
	return layer{
		in:      in,
		out:     out,
		biases:  r.i32s(out),
		weights: r.i8s(out * in),
	}
}
