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
	return n.forwardPass(n.accumulate(b, stm), n.accumulate(b, stm.Opposite()))
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
	input := make([]int8, 2*halfDimensions)
	for i, v := range accSTM {
		input[i] = clampFT(v)
	}
	for i, v := range accOpp {
		input[halfDimensions+i] = clampFT(v)
	}

	h1 := make([]int8, hiddenDim)
	for i, v := range n.l1.forward(input) {
		h1[i] = clampHidden(v)
	}

	h2 := make([]int8, hiddenDim)
	for i, v := range n.l2.forward(h1) {
		h2[i] = clampHidden(v)
	}

	return int(n.out.forward(h2)[0]) / outputScale
}
