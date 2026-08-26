package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"talos/internal/datagen"
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
	// "talos datagen" generates NNUE training data by self-play. Like bench
	// it is a plain command rather than a UCI one: it runs for hours and
	// writes a file, which is not a conversation a GUI should be having.
	if len(os.Args) > 1 && os.Args[1] == "datagen" {
		if err := runDatagen(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, "datagen:", err)
			os.Exit(1)
		}
		return
	}
	// "talos pack" converts generated training data into the fixed-size
	// binary records a trainer memory-maps. Separate from datagen so the
	// text dump stays the archival artifact and packing can be redone
	// whenever the feature set changes.
	if len(os.Args) > 1 && os.Args[1] == "pack" {
		if err := runPack(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, "pack:", err)
			os.Exit(1)
		}
		return
	}
	uci.Run()
}

// runDatagen parses the flags for "talos datagen" and runs it.
func runDatagen(args []string) error {
	opts := datagen.DefaultOptions()

	fs := flag.NewFlagSet("datagen", flag.ContinueOnError)
	out := fs.String("out", "", "file to write training data to (default: stdout)")
	fs.IntVar(&opts.Games, "games", opts.Games, "number of self-play games")
	fs.IntVar(&opts.Nodes, "nodes", opts.Nodes, "search nodes per move")
	fs.IntVar(&opts.Threads, "threads", opts.Threads, "games to play concurrently")
	fs.IntVar(&opts.RandomPlies, "random-plies", opts.RandomPlies, "random opening moves before recording")
	fs.IntVar(&opts.MaxPlies, "max-plies", opts.MaxPlies, "abandon a game after this many plies")
	fs.IntVar(&opts.AdjudicateScore, "adjudicate-score", opts.AdjudicateScore, "centipawn margin for adjudicating a won game (0 disables)")
	fs.Int64Var(&opts.Seed, "seed", opts.Seed, "random seed; a run is reproducible from it")
	if err := fs.Parse(args); err != nil {
		return err
	}

	opts.Out = os.Stdout
	if *out != "" {
		f, err := os.Create(*out)
		if err != nil {
			return err
		}
		defer f.Close()
		opts.Out = f
	}
	// Progress goes to stderr so it never contaminates the data when the
	// output is a pipe.
	opts.Log = os.Stderr

	return datagen.Run(opts)
}

// runPack parses the flags for "talos pack" and runs it.
func runPack(args []string) error {
	fs := flag.NewFlagSet("pack", flag.ContinueOnError)
	in := fs.String("in", "", "training data to read (default: stdin)")
	out := fs.String("out", "", "packed records to write (default: stdout)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	src := io.Reader(os.Stdin)
	if *in != "" {
		f, err := os.Open(*in)
		if err != nil {
			return err
		}
		defer f.Close()
		src = f
	}

	dst := io.Writer(os.Stdout)
	if *out != "" {
		f, err := os.Create(*out)
		if err != nil {
			return err
		}
		defer f.Close()
		dst = f
	}

	n, err := datagen.Pack(src, dst)
	if err != nil {
		return err
	}
	// Progress to stderr so it never contaminates the data when output is a
	// pipe, matching datagen.
	fmt.Fprintf(os.Stderr, "packed %d positions (%d bytes each)\n", n, datagen.RecordSize)
	return nil
}
