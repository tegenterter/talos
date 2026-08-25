package search

import (
	"math/rand"
	"testing"

	"talos/internal/board"
)

// TestSeeSimpleCases is a small, readable, named corpus documenting what
// see() is meant to handle — one position per interesting code path. Each
// case's expected value comes from bruteForceSEE (see below), not from
// hand arithmetic: for anything beyond the most trivial one-capture
// exchange, working out the "obviously correct" answer by hand is
// surprisingly easy to get wrong (an earlier draft of this test did,
// twice — once by forgetting a king could legally recapture for free,
// once by picking a position that didn't actually exercise the x-ray
// case its name claimed to test). The brute-force oracle doesn't have
// that problem: it's real minimax over this codebase's actual move
// generator, not reasoning about what the rules allow.
func TestSeeSimpleCases(t *testing.T) {
	tests := []struct {
		name string
		fen  string
		uci  string
	}{
		{
			name: "undefended pawn capture wins a pawn outright",
			fen:  "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
			uci:  "e4d5",
		},
		{
			name: "pawn takes pawn defended by another pawn: an even trade",
			fen:  "4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1",
			uci:  "e4d5",
		},
		{
			name: "rook takes pawn defended by a pawn: loses the exchange",
			fen:  "4k3/8/2p5/3p4/8/8/8/3RK3 w - - 0 1",
			uci:  "d1d5",
		},
		{
			name: "rook takes rook defended by a rook approaching from a different line: an even trade",
			fen:  "r3r3/7k/8/8/8/8/8/R6K w - - 0 1",
			uci:  "a1a8",
		},
		{
			name: "x-ray: a second rook behind the first keeps defending after it's captured",
			fen:  "3r3k/8/8/3n4/8/8/3R4/3R3K w - - 0 1",
			uci:  "d2d5",
		},
		{
			name: "en passant captures the pawn behind the destination square",
			fen:  "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
			uci:  "e5d6",
		},
		{
			name: "promoting capture values the promoted piece — including the adjacent king recapturing it for free afterward",
			fen:  "3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1",
			uci:  "e7d8q",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := mustFEN(t, tt.fen)
			m, ok := parseAndMatch(t, &b, tt.uci)
			if !ok {
				t.Fatalf("%s is not legal in %q", tt.uci, tt.fen)
			}
			want := bruteForceSEE(b, m)
			if got := see(&b, m); got != want {
				t.Errorf("see(%q, %s) = %d, want %d (brute force)", tt.fen, tt.uci, got, want)
			}
		})
	}
}

func parseAndMatch(t *testing.T, b *board.Board, uci string) (board.Move, bool) {
	t.Helper()
	parsed, ok := board.ParseUCIMove(uci)
	if !ok {
		return board.Move{}, false
	}
	for _, m := range board.GenerateLegalMoves(b) {
		if m.From == parsed.From && m.To == parsed.To && m.Promotion == parsed.Promotion {
			return m, true
		}
	}
	return board.Move{}, false
}

// bruteForceSEE is an independently-derived reference implementation of
// the same quantity see() computes, used only by TestSeeMatchesBruteForce:
// a real minimax over every legal reply that recaptures on target,
// implemented with this codebase's actual move generator/make-move (so it
// automatically gets king-legality, promotion, and en passant right by
// construction, rather than by the same reasoning see() itself relies
// on). Any disagreement between the two on a legal position is a real bug
// in one of them.
//
// m itself is always played — it's the move being evaluated, not a choice
// among alternatives — and only the *subsequent* recapture decisions
// (bruteForceRecapture) are allowed to "decline". Getting that distinction
// backward here (letting the top-level call itself decline) was a real
// bug in an earlier draft: it silently floored every losing capture's
// brute-force answer at 0, masking the very cases most worth testing.
func bruteForceSEE(b board.Board, m board.Move) int {
	target := m.To
	gain := captureGain(&b, m)
	child := board.MakeMove(b, m)
	return gain - bruteForceRecapture(child, target, 0)
}

// bruteForceRecapture is the recursive minimax the side to move in b
// faces at target: play whichever legal move landing on target nets the
// most (accounting for the opponent's own best reply after), or decline
// and net 0 — declining is genuinely optional here, unlike m itself in
// bruteForceSEE above.
func bruteForceRecapture(b board.Board, target board.Square, depth int) int {
	if depth > 12 {
		return 0 // safety valve against pathological branching; not expected to trigger
	}
	best := 0
	for _, m := range board.GenerateLegalMoves(&b) {
		if m.To != target {
			continue
		}
		gain := captureGain(&b, m)
		child := board.MakeMove(b, m)
		if result := gain - bruteForceRecapture(child, target, depth+1); result > best {
			best = result
		}
	}
	return best
}

// captureGain is the material m itself wins on the spot, in b (before m
// is played) — shared by bruteForceSEE and bruteForceRecapture so the en
// passant/promotion handling can't drift between them.
func captureGain(b *board.Board, m board.Move) int {
	gain := 0
	if m.Flag == board.EnPassantCapture {
		gain = pieceOrderValue[board.Pawn]
	} else if _, pt, ok := b.PieceAt(m.To); ok {
		gain = pieceOrderValue[pt]
	}
	if m.Promotion != board.NoPiece {
		gain += pieceOrderValue[m.Promotion] - pieceOrderValue[board.Pawn]
	}
	return gain
}

func TestSeeMatchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	pieceTypes := []board.PieceType{board.Pawn, board.Knight, board.Bishop, board.Rook, board.Queen}

	checked := 0
	for trial := 0; trial < 300 && checked < 2000; trial++ {
		b, ok := randomPosition(rng, pieceTypes)
		if !ok {
			continue
		}
		for _, m := range board.GenerateLegalMoves(&b) {
			want := bruteForceSEE(b, m)
			got := see(&b, m)
			checked++
			if got != want {
				t.Fatalf("trial %d, side to move %v: see(%s) = %d, want %d (brute force); rerun with rand.NewSource(1) and this trial number to reproduce",
					trial, b.SideToMove, m.String(), got, want)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no positions were checked")
	}
	t.Logf("cross-checked %d (position, move) pairs against brute force", checked)
}

// randomPosition builds a random, legal (side not to move isn't in check),
// reasonably cluttered position: two kings apart from each other, plus a
// handful of random pieces of random types/colors placed near a common
// "hot square" so exchanges are actually likely, plus a few more placed
// anywhere for variety.
func randomPosition(rng *rand.Rand, pieceTypes []board.PieceType) (board.Board, bool) {
	var b board.Board
	occupied := map[board.Square]bool{}

	place := func(sq board.Square, c board.Color, pt board.PieceType) bool {
		if sq < 0 || sq > 63 || occupied[sq] {
			return false
		}
		if pt == board.Pawn && (sq/8 == 0 || sq/8 == 7) {
			return false
		}
		occupied[sq] = true
		b.Pieces[c][pt] |= board.Bitboard(1) << uint(sq)
		return true
	}

	wk := board.Square(rng.Intn(64))
	if !place(wk, board.White, board.King) {
		return b, false
	}
	var bk board.Square
	for tries := 0; tries < 20; tries++ {
		bk = board.Square(rng.Intn(64))
		if bk == wk {
			continue
		}
		df, dr := abs(int(bk)%8-int(wk)%8), abs(int(bk)/8-int(wk)/8)
		if df <= 1 && dr <= 1 {
			continue // kings can't be adjacent
		}
		if place(bk, board.Black, board.King) {
			break
		}
	}
	if b.Pieces[board.Black][board.King] == 0 {
		return b, false
	}

	hot := board.Square(rng.Intn(64))
	nearHot := func() board.Square {
		df, dr := rng.Intn(5)-2, rng.Intn(5)-2
		f, r := int(hot)%8+df, int(hot)/8+dr
		if f < 0 || f > 7 || r < 0 || r > 7 {
			return board.Square(rng.Intn(64))
		}
		return board.Square(r*8 + f)
	}

	n := 4 + rng.Intn(6)
	for i := 0; i < n; i++ {
		c := board.Color(rng.Intn(2))
		pt := pieceTypes[rng.Intn(len(pieceTypes))]
		sq := nearHot()
		if !place(sq, c, pt) {
			continue
		}
	}
	extra := rng.Intn(4)
	for i := 0; i < extra; i++ {
		c := board.Color(rng.Intn(2))
		pt := pieceTypes[rng.Intn(len(pieceTypes))]
		place(board.Square(rng.Intn(64)), c, pt)
	}

	b.SideToMove = board.Color(rng.Intn(2))
	b.EnPassant = board.NoSquare

	notToMove := b.SideToMove.Opposite()
	kingSq := b.Pieces[notToMove][board.King].LSB()
	if board.IsSquareAttacked(&b, kingSq, b.SideToMove) {
		return b, false // illegal: the side not to move is in check
	}
	if hasPin(&b) {
		// see() doesn't know about pins (see its doc comment) and can
		// overestimate a defender's availability when one is present —
		// an accepted limitation, not something this fuzz test should
		// report as a mismatch, so positions containing one are excluded
		// here rather than the comparison being weakened everywhere else.
		return b, false
	}
	return b, true
}

// hasPin reports whether any piece on the board is absolutely pinned —
// removing it would newly expose its own king to a slider attack that
// wasn't there before. It's deliberately conservative (a pin anywhere on
// the board excludes the position, not just one demonstrably relevant to
// the move under test), which is simple and safe for filtering random
// positions, if not the check see() itself would need to actually handle
// pins (which also has to know whether the specific target square lies
// back on the pin line, since a pinned piece can still move along it).
func hasPin(b *board.Board) bool {
	occ := b.OccupiedBB()
	for _, c := range [2]board.Color{board.White, board.Black} {
		king := b.Pieces[c][board.King].LSB()
		before := attackersTo(b, king, occ) & b.ColorBB(c.Opposite())
		for bb := b.ColorBB(c) &^ b.Pieces[c][board.King]; bb != 0; bb &= bb - 1 {
			sq := bb.LSB()
			after := attackersTo(b, king, occ&^sqBit(sq)) & b.ColorBB(c.Opposite())
			if after&^before != 0 {
				return true
			}
		}
	}
	return false
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
