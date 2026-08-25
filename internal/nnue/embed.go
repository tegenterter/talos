package nnue

import (
	_ "embed"
	"fmt"

	"talos/internal/board"
)

// defaultNetworkBytes embeds Stockfish 12's default net directly into the
// binary, matching how Stockfish itself ships a built-in default network
// (Stockfish's build embeds one via incbin) — so `go build`,
// `go test`, and `go run` all produce a working NNUE evaluator standalone,
// with nothing to download or configure separately. It costs about 21MB of
// binary size and repo weight; there is no external EvalFile-style option
// to swap it for a different network file.
//
//go:embed nn-97f742aaefcd.nnue
var defaultNetworkBytes []byte

// DefaultNetwork is loaded once from the embedded network file. A parse
// failure here means the embedded file is corrupt or not the expected
// HalfKP 256x2-32-32-1 format, which can only be a bug in this package
// (the file is fixed at build time), so it panics at package init rather
// than forcing every caller to handle an error that isn't actionable at
// runtime.
var DefaultNetwork = func() *Network {
	n, err := Load(defaultNetworkBytes)
	if err != nil {
		panic(fmt.Sprintf("nnue: failed to load embedded default network: %v", err))
	}
	return n
}()

// Evaluate scores b using DefaultNetwork. See Network.Evaluate for the
// scoring convention.
func Evaluate(b *board.Board) int {
	return DefaultNetwork.Evaluate(b)
}

// Explain breaks down b's evaluation per piece using DefaultNetwork. See
// Network.Explain for what the breakdown means and what it cannot cover.
func Explain(b *board.Board) Explanation {
	return DefaultNetwork.Explain(b)
}
