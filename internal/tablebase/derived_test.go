package tablebase

import "testing"

// TestDerivedTablesMatchPythonChess cross-checks pFactor/pawnIdx/mFactor/
// multIdx against actual values computed by python-chess's syzygy.py
// (ground truth, obtained by running that module directly — see
// internal/tablebase's package doc comment), to catch any porting mistake
// in the incremental binom-sum construction these tables are built from
// (derived.go's init()).
func TestDerivedTablesMatchPythonChess(t *testing.T) {
	wantPFactor := [3][4]int{
		{6, 6, 6, 6},
		{252, 180, 108, 36},
		{5201, 2645, 953, 125},
	}
	for i, want := range wantPFactor {
		if pFactor[i] != want {
			t.Errorf("pFactor[%d] = %v, want %v", i, pFactor[i], want)
		}
	}

	wantPawnIdx1 := [10]int{0, 47, 92, 135, 176, 215, 0, 35, 68, 99}
	for j, want := range wantPawnIdx1 {
		if pawnIdx[1][j] != want {
			t.Errorf("pawnIdx[1][%d] = %d, want %d", j, pawnIdx[1][j], want)
		}
	}

	wantMFactor := [5]int{10, 294, 6162, 98222, 1239386}
	if mFactor != wantMFactor {
		t.Errorf("mFactor = %v, want %v", mFactor, wantMFactor)
	}

	wantMultIdx2 := [10]int{0, 1953, 3438, 4519, 5260, 5725, 5978, 6083, 6138, 6159}
	if multIdx[2] != wantMultIdx2 {
		t.Errorf("multIdx[2] = %v, want %v", multIdx[2], wantMultIdx2)
	}
}
