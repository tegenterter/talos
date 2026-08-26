package board

import "math/rand"

// Magic bitboards for the sliding pieces.
//
// A bishop or rook's attacks depend only on the occupied squares along its
// own rays, which for any square is at most 9 or 12 relevant bits. A "magic"
// is a multiplier that hashes each of those occupancy patterns to a distinct
// index in a per-square table, turning a ray walk into one multiply, one
// shift and one load. This replaced slidingAttacks in the hot path, where
// ray-walking was ~8% of the engine's total CPU after the evaluator stopped
// dominating it.
//
// slidingAttacks itself is kept, and is still the definition of correctness:
// these tables are *built* from it at startup, and magic_test.go checks the
// two agree on random occupancies for every square. Nothing here is a second
// implementation of how a rook moves.
//
// The magics are searched for at init with a fixed-seed PRNG rather than
// pasted in as constants, for the same reason zobrist.go seeds its keys that
// way: a fixed seed makes startup reproducible across processes, while a
// table of unexplained 64-bit constants would be untestable and unfixable.

const (
	rookIndexBits   = 12 // a rook sees at most 12 relevant occupancy squares
	bishopIndexBits = 9  // a bishop at most 9
)

var (
	rookMask, bishopMask   [64]Bitboard
	rookMagic, bishopMagic [64]uint64
	rookTable              [64][1 << rookIndexBits]Bitboard
	bishopTable            [64][1 << bishopIndexBits]Bitboard
)

// magicSeed fixes the magic search, so every process finds the same magics
// and any bug is reproducible. See zobrist.go's zobristSeed for the same
// reasoning applied to hash keys.
const magicSeed = 0x5EED_1234_ABCD_0001

func init() {
	rng := rand.New(rand.NewSource(magicSeed))
	for sq := Square(0); sq < 64; sq++ {
		rookMask[sq] = relevantMask(sq, rookDirs)
		bishopMask[sq] = relevantMask(sq, bishopDirs)

		rookMagic[sq] = findMagic(rng, sq, rookMask[sq], rookDirs, rookIndexBits, rookTable[sq][:])
		bishopMagic[sq] = findMagic(rng, sq, bishopMask[sq], bishopDirs, bishopIndexBits, bishopTable[sq][:])
	}
}

// relevantMask is the set of squares whose occupancy can change what a
// slider on sq attacks: its rays, minus the far edge of each, since a piece
// on the last square of a ray blocks nothing beyond it.
func relevantMask(sq Square, dirs [][2]int) Bitboard {
	var mask Bitboard
	file0, rank0 := int(sq)%8, int(sq)/8
	for _, d := range dirs {
		file, rank := file0+d[1], rank0+d[0]
		for {
			next, nextRank := file+d[1], rank+d[0]
			if next < 0 || next > 7 || nextRank < 0 || nextRank > 7 {
				break // `file,rank` is on the edge: stop before including it
			}
			mask |= sqBit(Square(rank*8 + file))
			file, rank = next, nextRank
		}
	}
	return mask
}

// occupancySubsets enumerates every subset of mask, the standard
// carry-rippler trick: subtracting 1 from a subset and re-masking walks the
// subsets in order, ending back at zero.
func occupancySubsets(mask Bitboard) []Bitboard {
	var subsets []Bitboard
	sub := Bitboard(0)
	for {
		subsets = append(subsets, sub)
		sub = (sub - mask) & mask
		if sub == 0 {
			break
		}
	}
	return subsets
}

// findMagic searches for a multiplier that maps every occupancy subset of
// mask to a table slot holding the right attacks, and fills table with the
// result. Two subsets may share a slot as long as they have the same attacks
// — a "constructive collision", which is why this succeeds at these index
// widths rather than needing perfect hashing.
func findMagic(rng *rand.Rand, sq Square, mask Bitboard, dirs [][2]int, indexBits int, table []Bitboard) uint64 {
	subsets := occupancySubsets(mask)
	attacks := make([]Bitboard, len(subsets))
	for i, occ := range subsets {
		attacks[i] = slidingAttacks(sq, occ, dirs)
	}

	shift := 64 - indexBits
	// Marks which slots this attempt has filled, so a stale entry from a
	// failed attempt is never mistaken for a valid one.
	used := make([]int, len(table))
	for attempt := 1; ; attempt++ {
		// A magic works by gathering the masked bits into the top of the
		// word, so it wants few set bits: ANDing three random words is the
		// standard way to bias toward that, and makes the search converge in
		// a handful of attempts instead of millions.
		magic := rng.Uint64() & rng.Uint64() & rng.Uint64()

		ok := true
		for i, occ := range subsets {
			idx := (uint64(occ) * magic) >> shift
			switch {
			case used[idx] != attempt:
				used[idx] = attempt
				table[idx] = attacks[i]
			case table[idx] != attacks[i]:
				ok = false
			}
			if !ok {
				break
			}
		}
		if ok {
			return magic
		}
	}
}

// bishopAttacksMagic and rookAttacksMagic are the lookups the hot path uses.
func bishopAttacksMagic(sq Square, occ Bitboard) Bitboard {
	idx := (uint64(occ&bishopMask[sq]) * bishopMagic[sq]) >> (64 - bishopIndexBits)
	return bishopTable[sq][idx]
}

func rookAttacksMagic(sq Square, occ Bitboard) Bitboard {
	idx := (uint64(occ&rookMask[sq]) * rookMagic[sq]) >> (64 - rookIndexBits)
	return rookTable[sq][idx]
}
