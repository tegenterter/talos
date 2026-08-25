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
}

func (l *layer) forward(in []int8) []int32 {
	out := make([]int32, l.out)
	for r := 0; r < l.out; r++ {
		sum := l.biases[r]
		row := l.weights[r*l.in : r*l.in+l.in]
		for c, x := range in {
			sum += int32(row[c]) * int32(x)
		}
		out[r] = sum
	}
	return out
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
