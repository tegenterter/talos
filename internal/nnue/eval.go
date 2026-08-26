package nnue

import "talos/internal/board"

// clampFT quantizes one feature-transformer accumulator value into the
// [0,127] int8 range the first affine layer expects. This doubles as the
// feature transformer's activation function (a ClippedReLU): unlike the
// hidden layers below, the feature-transformer weights are already scaled
// so its raw accumulator sits in output units directly, with no shift needed.
func clampFT(v int16) int8 {
	switch {
	case v < 0:
		return 0
	case v > ftScale:
		return ftScale
	default:
		return int8(v)
	}
}

// clampHidden is the ClippedReLU applied after each hidden affine layer:
// undo that layer's weight quantization scale (a right-shift, since weights
// were scaled by 2^weightScaleBits) and clamp to the same [0,127] int8 range.
func clampHidden(v int32) int8 {
	v >>= weightScaleBits
	switch {
	case v < 0:
		return 0
	case v > 127:
		return 127
	default:
		return int8(v)
	}
}

// Evaluate returns a static evaluation of b in centipawns, from the
// perspective of the side to move: positive means the side to move is
// better off, negative means worse. This mirrors the signature and
// convention of the classical evaluator (internal/eval) it replaces, so
// internal/search's call sites didn't need to change beyond the import.
func (n *Network) Evaluate(b *board.Board) int {
	stm := b.SideToMove
	// Stack arrays, not slices from make: this runs at every leaf of the
	// search, and heap-allocating the two accumulators here was millions of
	// allocations a second.
	var accSTM, accOpp [halfDimensions]int16
	n.accumulateInto(accSTM[:], b, stm)
	n.accumulateInto(accOpp[:], b, stm.Opposite())
	return n.forwardPass(accSTM[:], accOpp[:])
}

// forwardPass runs the network's affine/ClippedReLU stack over two already-
// computed perspective accumulators and returns the centipawn score. It is
// split out from Evaluate so Explain (explain.go) can reuse the exact same
// output stack on hand-modified accumulators — the two must never drift
// apart, or an explanation would describe a network that isn't the one
// actually evaluating positions.
func (n *Network) forwardPass(accSTM, accOpp []int16) int {
	// The two perspectives are concatenated STM-first: the network was
	// trained to always see "my features, then their features" in that
	// order, so the same weights can be reused regardless of which color
	// is actually moving.
	// Every buffer below is a stack array for the same reason as Evaluate's
	// accumulators: this is the hot path, and none of them outlive the call.
	//
	// The two perspectives are fed to the first layer separately at their
	// respective column offsets rather than concatenated into one clamped
	// input buffer, because that layer reads accumulators directly (see
	// forwardSparseInto). The network still sees exactly the same 512-wide
	// input, STM-first.
	var l1out, l2out [hiddenDim]int32
	var h1, h2 [hiddenDim]int8

	copy(l1out[:], n.l1.biases)
	n.l1.forwardSparseInto(accSTM, 0, l1out[:])
	n.l1.forwardSparseInto(accOpp, halfDimensions, l1out[:])
	for i, v := range l1out {
		h1[i] = clampHidden(v)
	}

	n.l2.forwardInto(h1[:], l2out[:])
	for i, v := range l2out {
		h2[i] = clampHidden(v)
	}

	var outv [1]int32
	n.out.forwardInto(h2[:], outv[:])
	return int(outv[0]) / outputScale
}
