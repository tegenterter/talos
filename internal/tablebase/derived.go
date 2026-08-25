package tablebase

// pawnIdx and pFactor are derived from ptwist/invFlap via the same
// incremental-binomial-sum construction python-chess's syzygy.py computes
// at module load time (its module-level `for i in range(5): ...` loop
// building PAWNIDX/PFACTOR). j deliberately is NOT reset between the four
// "file group" segments below (files a-b, c-d, e-f, g-h) — it threads
// through all 24 values within one outer i iteration, exactly mirroring
// the Python loop's shared j variable, since these are running sums over
// the whole 0..23 range split only for where PFACTOR captures a subtotal.
var pawnIdx [5][24]int
var pFactor [5][4]int

// multIdx and mFactor are the equivalent running-sum tables for the
// "three or more of one piece type" (e.g. KNNNvK) encoding.
var multIdx [5][10]int
var mFactor [5]int

func init() {
	for i := 0; i < 5; i++ {
		j := 0
		s := 0
		for j < 6 {
			pawnIdx[i][j] = s
			s += pawnIdxStep(i, j)
			j++
		}
		pFactor[i][0] = s

		s = 0
		for j < 12 {
			pawnIdx[i][j] = s
			s += pawnIdxStep(i, j)
			j++
		}
		pFactor[i][1] = s

		s = 0
		for j < 18 {
			pawnIdx[i][j] = s
			s += pawnIdxStep(i, j)
			j++
		}
		pFactor[i][2] = s

		s = 0
		for j < 24 {
			pawnIdx[i][j] = s
			s += pawnIdxStep(i, j)
			j++
		}
		pFactor[i][3] = s
	}

	for i := 0; i < 5; i++ {
		s := 0
		for j := 0; j < 10; j++ {
			multIdx[i][j] = s
			if i == 0 {
				s++
			} else {
				s += binom(mtwist[invTriangle[j]], i)
			}
		}
		mFactor[i] = s
	}
}

func pawnIdxStep(i, j int) int {
	if i == 0 {
		return 1
	}
	return binom(ptwist[invFlap[j]], i)
}
