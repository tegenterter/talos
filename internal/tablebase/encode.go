package tablebase

// offdiag reports how far above (positive) or below (negative) sq is from
// the a1-h8 diagonal.
func offdiag(sq int) int { return (sq >> 3) - (sq & 7) }

// flipdiag reflects sq across the a1-h8 diagonal by swapping its rank and
// file bit-fields.
func flipdiag(sq int) int { return ((sq >> 3) | (sq << 3)) & 63 }

// test45 reports whether sq is one of the specific 6 squares (a5-c5,
// a6-b6, a7) encode_piece's enc_type==3 case uses to decide a tie-break
// swap for the "KKvK"-shaped (two same-side identical pieces) encoding.
func test45(sq int) bool {
	switch sq {
	case 32, 33, 34, 40, 41, 48: // a5, b5, c5, a6, b6, a7
		return true
	default:
		return false
	}
}

// subfactor computes a running factor count python-chess's syzygy.py
// derives independently from binom (not simply binom with swapped
// arguments — they disagree at k=0, where this returns n rather than 1,
// which matters since calc_factors_piece/pawn call it exactly as written
// here), used to size how many index-space slots a group of k
// interchangeable pieces spread among n remaining squares needs.
func subfactor(k, n int) int {
	f := n
	l := 1
	for i := 1; i < k; i++ {
		f *= n - i
		l *= i + 1
	}
	return f / l
}

func (t *table) setNormPiece(norm []int, pieces []int) {
	switch t.encType {
	case 0:
		norm[0] = 3
	case 2:
		norm[0] = 2
	default:
		norm[0] = t.encType - 1
	}

	i := norm[0]
	for i < t.numPieces {
		j := i
		for j < t.numPieces && pieces[j] == pieces[i] {
			norm[i]++
			j++
		}
		i += norm[i]
	}
}

// pivFac is PIVFAC for connected_kings=false, the only case standard
// chess (this package's only target) needs.
var pivFac = [3]int{31332, 28056, 462}

func (t *table) calcFactorsPiece(factor *[tbPieces]int, order int, norm []int) int {
	n := 64 - norm[0]

	f := 1
	i := norm[0]
	k := 0
	for i < t.numPieces || k == order {
		if k == order {
			factor[0] = f
			if t.encType < 4 {
				f *= pivFac[t.encType]
			} else {
				f *= mFactor[t.encType-2]
			}
		} else {
			factor[i] = f
			f *= subfactor(norm[i], n)
			n -= norm[i]
			i += norm[i]
		}
		k++
	}
	return f
}

func (t *table) calcFactorsPawn(factor *[tbPieces]int, order, order2 int, norm []int, f int) int {
	i := norm[0]
	if order2 < 0x0f {
		i += norm[i]
	}
	n := 64 - i

	fac := 1
	k := 0
	for i < t.numPieces || k == order || k == order2 {
		switch k {
		case order:
			factor[0] = fac
			fac *= pFactor[norm[0]-1][f]
		case order2:
			factor[norm[0]] = fac
			fac *= subfactor(norm[norm[0]], 48-norm[0])
		default:
			factor[i] = fac
			fac *= subfactor(norm[i], n)
			n -= norm[i]
			i += norm[i]
		}
		k++
	}
	return fac
}

func (t *table) setNormPawn(norm []int, pieces []int) {
	norm[0] = t.pawns[0]
	if t.pawns[1] != 0 {
		norm[t.pawns[0]] = t.pawns[1]
	}

	i := t.pawns[0] + t.pawns[1]
	for i < t.numPieces {
		j := i
		for j < t.numPieces && pieces[j] == pieces[i] {
			norm[i]++
			j++
		}
		i += norm[i]
	}
}

// pawnFile determines which of the (up to 4) pawn-file sub-tables pos
// belongs in, first canonicalizing pos[0] to be the "most extreme" (by
// FLAP) of the stronger side's pawns — swapping it into pos[0] in place,
// exactly as python-chess's pawn_file does, since later encoding steps
// depend on that square being there.
func (t *table) pawnFile(pos []int) int {
	for i := 1; i < t.pawns[0]; i++ {
		if flap[pos[0]] > flap[pos[i]] {
			pos[0], pos[i] = pos[i], pos[0]
		}
	}
	return fileToFile[pos[0]&0x07]
}

// encodePiece computes the table index for a pawnless material
// configuration's piece placement pos (kings, then any doubled pieces
// grouped by norm), consuming symmetry (square mirroring/diagonal
// flipping, per enc_type) to canonicalize the position before combining
// each norm-group's own combinatorial sub-index via factor. pos is
// mutated in place (mirrored/reordered) — the caller must pass a fresh
// scratch copy, not a piece list it needs afterward.
func (t *table) encodePiece(norm []int, pos []int, factor *[tbPieces]int) int {
	n := t.numPieces
	var idx int

	if t.encType < 3 {
		if pos[0]&0x04 != 0 {
			for i := 0; i < n; i++ {
				pos[i] ^= 0x07
			}
		}
		if pos[0]&0x20 != 0 {
			for i := 0; i < n; i++ {
				pos[i] ^= 0x38
			}
		}

		i := 0
		for i = 0; i < n; i++ {
			if offdiag(pos[i]) != 0 {
				break
			}
		}
		limit := 2
		if t.encType == 0 {
			limit = 3
		}
		if i < limit && offdiag(pos[i]) > 0 {
			for i := 0; i < n; i++ {
				pos[i] = flipdiag(pos[i])
			}
		}
	}

	var i int
	switch {
	case t.encType == 0: // 111
		bi := b2i(pos[1] > pos[0])
		bj := b2i(pos[2] > pos[0]) + b2i(pos[2] > pos[1])

		switch {
		case offdiag(pos[0]) != 0:
			idx = triangle[pos[0]]*63*62 + (pos[1]-bi)*62 + (pos[2] - bj)
		case offdiag(pos[1]) != 0:
			idx = 6*63*62 + diag[pos[0]]*28*62 + lower[pos[1]]*62 + pos[2] - bj
		case offdiag(pos[2]) != 0:
			idx = 6*63*62 + 4*28*62 + diag[pos[0]]*7*28 + (diag[pos[1]]-bi)*28 + lower[pos[2]]
		default:
			idx = 6*63*62 + 4*28*62 + 4*7*28 + diag[pos[0]]*7*6 + (diag[pos[1]]-bi)*6 + (diag[pos[2]] - bj)
		}
		i = 3

	case t.encType == 2: // K2
		idx = kkIdx[triangle[pos[0]]][pos[1]]
		i = 2

	case t.encType == 3: // 2, e.g. KKvK
		if triangle[pos[0]] > triangle[pos[1]] {
			pos[0], pos[1] = pos[1], pos[0]
		}
		if pos[0]&0x04 != 0 {
			for j := 0; j < n; j++ {
				pos[j] ^= 0x07
			}
		}
		if pos[0]&0x20 != 0 {
			for j := 0; j < n; j++ {
				pos[j] ^= 0x38
			}
		}
		if offdiag(pos[0]) > 0 || (offdiag(pos[0]) == 0 && offdiag(pos[1]) > 0) {
			for j := 0; j < n; j++ {
				pos[j] = flipdiag(pos[j])
			}
		}
		if test45(pos[1]) && triangle[pos[0]] == triangle[pos[1]] {
			pos[0], pos[1] = pos[1], pos[0]
			for j := 0; j < n; j++ {
				pos[j] = flipdiag(pos[j] ^ 0x38)
			}
		}
		idx = ppIdx[triangle[pos[0]]][pos[1]]
		i = 2

	default: // 3 and higher, e.g. KKKvK and KKKKvK — unreachable for standard chess
		for j := 1; j < norm[0]; j++ {
			if triangle[pos[0]] > triangle[pos[j]] {
				pos[0], pos[j] = pos[j], pos[0]
			}
		}
		if pos[0]&0x04 != 0 {
			for j := 0; j < n; j++ {
				pos[j] ^= 0x07
			}
		}
		if pos[0]&0x20 != 0 {
			for j := 0; j < n; j++ {
				pos[j] ^= 0x38
			}
		}
		if offdiag(pos[0]) > 0 {
			for j := 0; j < n; j++ {
				pos[j] = flipdiag(pos[j])
			}
		}
		for a := 1; a < norm[0]; a++ {
			for b := a + 1; b < norm[0]; b++ {
				if mtwist[pos[a]] > mtwist[pos[b]] {
					pos[a], pos[b] = pos[b], pos[a]
				}
			}
		}

		idx = multIdx[norm[0]-1][triangle[pos[0]]]
		i = 1
		for i < norm[0] {
			idx += binom(mtwist[pos[i]], i)
			i++
		}
	}

	idx *= factor[0]

	for i < n {
		tt := norm[i]

		for j := i; j < i+tt; j++ {
			for k := j + 1; k < i+tt; k++ {
				if pos[j] > pos[k] {
					pos[j], pos[k] = pos[k], pos[j]
				}
			}
		}

		s := 0
		for m := i; m < i+tt; m++ {
			p := pos[m]
			j := 0
			for l := 0; l < i; l++ {
				j += b2i(p > pos[l])
			}
			s += binom(p-j, m-i+1)
		}

		idx += s * factor[i]
		i += tt
	}

	return idx
}

// encodePawn is encodePiece's counterpart for material with pawns:
// pos[0..pawns[0]-1] are the stronger side's pawns (already canonicalized
// by pawnFile), pos[pawns[0]..pawns[0]+pawns[1]-1] the weaker side's, and
// anything after that any remaining (non-pawn) pieces grouped by norm.
func (t *table) encodePawn(norm []int, pos []int, factor *[tbPieces]int) int {
	n := t.numPieces

	if pos[0]&0x04 != 0 {
		for i := 0; i < n; i++ {
			pos[i] ^= 0x07
		}
	}

	for i := 1; i < t.pawns[0]; i++ {
		for j := i + 1; j < t.pawns[0]; j++ {
			if ptwist[pos[i]] < ptwist[pos[j]] {
				pos[i], pos[j] = pos[j], pos[i]
			}
		}
	}

	tt := t.pawns[0] - 1
	idx := pawnIdx[tt][flap[pos[0]]]
	for i := tt; i > 0; i-- {
		idx += binom(ptwist[pos[i]], tt-i+1)
	}
	idx *= factor[0]

	i := t.pawns[0]
	tt2 := i + t.pawns[1]
	if tt2 > i {
		for j := i; j < tt2; j++ {
			for k := j + 1; k < tt2; k++ {
				if pos[j] > pos[k] {
					pos[j], pos[k] = pos[k], pos[j]
				}
			}
		}
		s := 0
		for m := i; m < tt2; m++ {
			p := pos[m]
			j := 0
			for k := 0; k < i; k++ {
				j += b2i(p > pos[k])
			}
			s += binom(p-j-8, m-i+1)
		}
		idx += s * factor[i]
		i = tt2
	}

	for i < n {
		tt = norm[i]
		for j := i; j < i+tt; j++ {
			for k := j + 1; k < i+tt; k++ {
				if pos[j] > pos[k] {
					pos[j], pos[k] = pos[k], pos[j]
				}
			}
		}

		s := 0
		for m := i; m < i+tt; m++ {
			p := pos[m]
			j := 0
			for k := 0; k < i; k++ {
				j += b2i(p > pos[k])
			}
			s += binom(p-j, m-i+1)
		}

		idx += s * factor[i]
		i += tt
	}

	return idx
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
