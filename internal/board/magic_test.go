package board

import (
	"math/rand"
	"testing"
	"time"
)

// TestMagicAttacksMatchRayCasting is the correctness check for magic.go: the
// table lookups must agree with slidingAttacks — the plain, obviously-right
// ray walk they were built from — for every square, over occupancies far
// beyond the ones perft happens to reach.
func TestMagicAttacksMatchRayCasting(t *testing.T) {
	rng := rand.New(rand.NewSource(99))

	for sq := Square(0); sq < 64; sq++ {
		for trial := 0; trial < 2000; trial++ {
			// Random occupancies, biased sparse (AND of two words) so both
			// crowded and near-empty boards are covered.
			occ := Bitboard(rng.Uint64() & rng.Uint64())

			if got, want := rookAttacksMagic(sq, occ), slidingAttacks(sq, occ, rookDirs); got != want {
				t.Fatalf("rook on %v with occ %#016x: magic %#016x, ray cast %#016x", sq, uint64(occ), uint64(got), uint64(want))
			}
			if got, want := bishopAttacksMagic(sq, occ), slidingAttacks(sq, occ, bishopDirs); got != want {
				t.Fatalf("bishop on %v with occ %#016x: magic %#016x, ray cast %#016x", sq, uint64(occ), uint64(got), uint64(want))
			}
		}
	}
}

// TestMagicMasksExcludeEdges pins the property that makes the index widths
// work: a piece on the far end of a ray blocks nothing beyond it, so it is
// not a relevant occupancy square, and leaving it out is what keeps a rook
// to 12 bits and a bishop to 9.
func TestMagicMasksExcludeEdges(t *testing.T) {
	for sq := Square(0); sq < 64; sq++ {
		if n := rookMask[sq].Count(); n > rookIndexBits {
			t.Errorf("rook mask on %v has %d relevant squares, more than the %d-bit index allows", sq, n, rookIndexBits)
		}
		if n := bishopMask[sq].Count(); n > bishopIndexBits {
			t.Errorf("bishop mask on %v has %d relevant squares, more than the %d-bit index allows", sq, n, bishopIndexBits)
		}
	}
}

// TestMagicGenerationIsFast guards the startup cost of searching for magics
// at init rather than hardcoding them: it happens in every process and every
// test binary, so it has to stay cheap enough not to notice.
func TestMagicGenerationIsFast(t *testing.T) {
	rng := rand.New(rand.NewSource(magicSeed))
	var table [1 << rookIndexBits]Bitboard

	start := time.Now()
	for sq := Square(0); sq < 64; sq++ {
		findMagic(rng, sq, relevantMask(sq, rookDirs), rookDirs, rookIndexBits, table[:])
		findMagic(rng, sq, relevantMask(sq, bishopDirs), bishopDirs, bishopIndexBits, table[:])
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("magic generation took %v; hardcode the magics if it ever gets this slow", elapsed)
	} else {
		t.Logf("magic generation took %v", elapsed)
	}
}
