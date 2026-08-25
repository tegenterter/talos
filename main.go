package main

import (
	"os"

	"talos/internal/uci"
)

func main() {
	// "talos bench" runs the fixed benchmark and exits, without going through
	// the UCI loop at all. Engine-testing tooling (and CI, and shell scripts
	// comparing two builds) expects to invoke a benchmark as a plain command
	// rather than by piping protocol text into stdin, so this mirrors the
	// convention Stockfish established. The "bench" UCI command does the same
	// thing for an interactive session.
	if len(os.Args) > 1 && os.Args[1] == "bench" {
		uci.RunBench(os.Stdout)
		return
	}
	uci.Run()
}
