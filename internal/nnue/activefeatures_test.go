package nnue

import (
	"math/rand"
	"testing"

	"talos/internal/board"
)

// TestActiveFeaturesMatchAccumulator is the load-bearing check for training
// data: summing the feature-transformer rows of ActiveFeatures must reproduce
// the accumulator the engine actually evaluates with, exactly.
//
// If these ever drift apart, a trained network would be learning a feature set
// the engine does not use — the failure mode where the pipeline runs
// end-to-end and the only symptom is that the network plays badly.
func TestActiveFeaturesMatchAccumulator(t *testing.T) {
	n := DefaultNetwork
	rng := rand.New(rand.NewSource(1234))

	for game := 0; game < 20; game++ {
		b := board.StartingBoard()
		for ply := 0; ply < 80; ply++ {
			moves := board.GenerateLegalMoves(&b)
			if len(moves) == 0 {
				break
			}
			b = board.MakeMove(b, moves[rng.Intn(len(moves))])

			for _, perspective := range [2]board.Color{board.White, board.Black} {
				// What the engine evaluates with.
				want := make([]int16, halfDimensions)
				n.accumulateInto(want, &b, perspective)

				// What training data would carry.
				var got [halfDimensions]int16
				copy(got[:], n.ftBiases)
				features := ActiveFeatures(nil, &b, perspective)
				for _, f := range features {
					row := n.ftWeights[int(f)*halfDimensions : (int(f)+1)*halfDimensions]
					for i, w := range row {
						got[i] += w
					}
				}

				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("game %d ply %d perspective %v: accumulator differs at %d (%d vs %d) over %d features",
							game, ply, perspective, i, got[i], want[i], len(features))
					}
				}
			}
		}
	}
}

// TestActiveFeaturesFitTheirBounds pins the two properties the packed record
// format depends on: a feature index fits in a uint16, and zero is never a
// real feature so it can serve as padding.
func TestActiveFeaturesFitTheirBounds(t *testing.T) {
	if inputFeatures-1 > 65535 {
		t.Fatalf("largest feature index %d does not fit a uint16", inputFeatures-1)
	}

	rng := rand.New(rand.NewSource(99))
	maxSeen := 0
	for game := 0; game < 20; game++ {
		b := board.StartingBoard()
		for ply := 0; ply < 80; ply++ {
			moves := board.GenerateLegalMoves(&b)
			if len(moves) == 0 {
				break
			}
			b = board.MakeMove(b, moves[rng.Intn(len(moves))])

			for _, perspective := range [2]board.Color{board.White, board.Black} {
				features := ActiveFeatures(nil, &b, perspective)
				if len(features) > MaxActiveFeatures {
					t.Fatalf("%d active features, above the stated maximum of %d", len(features), MaxActiveFeatures)
				}
				if len(features) > maxSeen {
					maxSeen = len(features)
				}
				for _, f := range features {
					if f == FeaturePadding {
						t.Fatalf("feature index %d collides with the padding value", f)
					}
					if int(f) >= inputFeatures {
						t.Fatalf("feature index %d is outside the table (%d entries)", f, inputFeatures)
					}
				}
			}
		}
	}
	t.Logf("most active features seen in a position: %d (bound %d)", maxSeen, MaxActiveFeatures)
}
