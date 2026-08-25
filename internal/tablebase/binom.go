package tablebase

// binom returns the binomial coefficient C(x, y), or 0 for any (x, y) that
// wouldn't be a valid combination count (y < 0, x < 0, or x < y) — mirroring
// python-chess's syzygy.py binom(), which computes this via exact-integer
// factorials and treats math.factorial's ValueError on a negative argument
// as "0". The incremental form used here (multiply by (x-y+i), divide by i,
// one step at a time) is the standard way to compute this exactly in
// fixed-width integers without needing x!/y!'s enormous intermediate
// values: the product of any y consecutive integers is always divisible by
// y!, and dividing after each multiplication keeps every intermediate
// result an exact integer.
func binom(x, y int) int {
	if x < 0 || y < 0 || x < y {
		return 0
	}
	result := 1
	for i := 1; i <= y; i++ {
		result = result * (x - y + i) / i
	}
	return result
}
