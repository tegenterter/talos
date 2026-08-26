package nnue

import (
	_ "embed"
	"fmt"
	"os"

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

// Refresh, Update and EvaluateAcc are DefaultNetwork's incremental
// accumulator API (see accumulator.go), which is what internal/search's hot
// path uses; Evaluate above stays the from-scratch entry point for callers
// with no accumulator to maintain.
func Refresh(a *Accumulator, b *board.Board) { DefaultNetwork.Refresh(a, b) }

func Update(dst, src *Accumulator, before *board.Board, m board.Move, after *board.Board) {
	DefaultNetwork.Update(dst, src, before, m, after)
}

func EvaluateAcc(a *Accumulator, stm board.Color) int { return DefaultNetwork.EvaluateAcc(a, stm) }

// LoadFile reads a network from disk, for the UCI "EvalFile" option — the
// engine ships with the embedded network above and plays with it unless a
// caller points it at another. It is what makes training a network usable:
// a candidate has to be playable without rebuilding the engine, or every
// iteration of the training loop costs a compile.
func LoadFile(path string) (*Network, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("nnue: reading %s: %w", path, err)
	}
	n, err := Load(data)
	if err != nil {
		return nil, fmt.Errorf("nnue: parsing %s: %w", path, err)
	}
	return n, nil
}

// Explain breaks down b's evaluation per piece using DefaultNetwork. See
// Network.Explain for what the breakdown means and what it cannot cover.
func Explain(b *board.Board) Explanation {
	return DefaultNetwork.Explain(b)
}
