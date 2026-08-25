package nnue

import (
	"testing"

	"talos/internal/board"
)

func mustFEN(t *testing.T, fen string) board.Board {
	t.Helper()
	b, err := board.ParseFEN(fen)
	if err != nil {
		t.Fatalf("ParseFEN(%q): %v", fen, err)
	}
	return b
}

func TestMissingQueenIsHeavilyPenalized(t *testing.T) {
	// Black is missing its queen; White to move.
	b := mustFEN(t, "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
	if got := Evaluate(&b); got < 500 {
		t.Errorf("Evaluate(missing black queen) = %d, want > 500", got)
	}
}

// pointMirror rotates the board 180 degrees (sq -> sq^63, flipping both
// rank and file) and swaps piece colors and the side to move. Unlike the
// vertical-only mirror (sq^56) that's the right invariant for a
// piece-square-table evaluator, HalfKP's own orient() function (see
// features.go) flips Black's perspective by sq^63, not sq^56 — a
// deliberately point-symmetric convention inherited from Stockfish's
// original HalfKP feature set, since chess has no exploitable left-right
// symmetry (castling sides and queen/king files break it) the way it has
// an exact top-bottom-plus-color symmetry. So point-mirroring, not
// vertical-mirroring, is the transform this network's Evaluate is
// provably invariant under.
func pointMirror(b board.Board) board.Board {
	var m board.Board
	for sq := board.Square(0); sq < 64; sq++ {
		color, pt, ok := b.PieceAt(sq)
		if !ok {
			continue
		}
		msq := sq ^ 63
		m.Pieces[color.Opposite()][pt] |= board.Bitboard(1) << msq
	}

	m.SideToMove = b.SideToMove.Opposite()

	if b.CastlingRights&board.WhiteKingside != 0 {
		m.CastlingRights |= board.BlackQueenside
	}
	if b.CastlingRights&board.WhiteQueenside != 0 {
		m.CastlingRights |= board.BlackKingside
	}
	if b.CastlingRights&board.BlackKingside != 0 {
		m.CastlingRights |= board.WhiteQueenside
	}
	if b.CastlingRights&board.BlackQueenside != 0 {
		m.CastlingRights |= board.WhiteKingside
	}

	if b.EnPassant == board.NoSquare {
		m.EnPassant = board.NoSquare
	} else {
		m.EnPassant = b.EnPassant ^ 63
	}

	return m
}

// TestPointSymmetry checks that Evaluate is invariant under rotating the
// board 180 degrees, swapping colors, and flipping the side to move — the
// symmetry HalfKP's feature indexing (features.go's orient) is built to
// respect by construction, since both perspectives end up looking at an
// identically-oriented set of feature indices. A mistake in the feature
// index formula (e.g. an unmirrored king square, or own/opponent slots
// swapped) breaks this exactly, which is what makes it a strong
// regression check for this package despite not depending on the trained
// weights being "correct" in any chess sense.
func TestPointSymmetry(t *testing.T) {
	positions := []string{
		board.StartFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // Kiwipete
		"8/P7/8/8/8/8/8/k6K w - - 0 1",
		"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
	}
	for _, fen := range positions {
		b := mustFEN(t, fen)
		mirrored := pointMirror(b)
		got, want := Evaluate(&b), Evaluate(&mirrored)
		if got != want {
			t.Errorf("Evaluate(%q) = %d, Evaluate(180-degree mirror) = %d, want equal", fen, got, want)
		}
	}
}
