package tablebase

import (
	"sort"
	"strings"

	"talos/internal/board"
)

// Raw piece-type codes as Syzygy table files encode them in their header
// bytes: pawn=1 through king=6, matching python-chess's chess.PAWN..
// chess.KING constants (which this package's file-parsing code is ported
// from) — unrelated to board.PieceType's own 0-indexed Pawn..King, since
// this numbering is a property of the file format, not this codebase.
const (
	rawPawn   = 1
	rawKnight = 2
	rawBishop = 3
	rawRook   = 4
	rawQueen  = 5
	rawKing   = 6
	rawColor  = 8 // bit set on a raw piece code for a black piece
)

// calcKey returns b's material signature in Syzygy tablename form, e.g.
// "KQvKR" for white K+Q vs black K+R — always listing White's pieces
// first (unlike a table's own normalized/canonical key, which may need
// White and Black swapped; see (*table).probe's cmirror/bside handling).
func calcKey(b *board.Board) string {
	return materialString(b, board.White) + "v" + materialString(b, board.Black)
}

func materialString(b *board.Board, c board.Color) string {
	var sb strings.Builder
	for _, pt := range [6]board.PieceType{board.King, board.Queen, board.Rook, board.Bishop, board.Knight, board.Pawn} {
		n := b.Pieces[c][pt].Count()
		for i := 0; i < n; i++ {
			sb.WriteByte(pieceChars[pt])
		}
	}
	return sb.String()
}

// recalcKey rebuilds a tablename from a table's own raw piece-code list
// (as parsed from its header — see setupPiecesPiece), rather than from
// the filename. Some endgames are stored under a different key than
// their filename would suggest: http://talkchess.com/forum/viewtopic.php?p=695509#695509
func recalcKey(pieces []int, mirror bool) string {
	own, opp := 0, rawColor
	if mirror {
		own, opp = rawColor, 0
	}
	return rawMaterialString(pieces, own) + "v" + rawMaterialString(pieces, opp)
}

func rawMaterialString(pieces []int, colorBit int) string {
	var sb strings.Builder
	for _, pt := range [6]int{rawKing, rawQueen, rawRook, rawBishop, rawKnight, rawPawn} {
		want := pt ^ colorBit
		for _, p := range pieces {
			if p == want {
				sb.WriteByte(pieceCharForRaw(pt))
			}
		}
	}
	return sb.String()
}

func pieceCharForRaw(rawType int) byte {
	switch rawType {
	case rawPawn:
		return 'P'
	case rawKnight:
		return 'N'
	case rawBishop:
		return 'B'
	case rawRook:
		return 'R'
	case rawQueen:
		return 'Q'
	default:
		return 'K'
	}
}

// normalizeTablename canonicalizes a "WhitePiecesvBlackPieces"-shaped
// name: each half's letters sorted into K,Q,R,B,N,P priority order, and
// the two halves swapped if needed so the same material always maps to
// the same key regardless of which side the filename happened to list
// first. mirror=true asks for the *other* canonical form (the one this
// same table also serves when Black's and White's roles are swapped —
// see (*table).probe).
func normalizeTablename(name string, mirror bool) string {
	w, b, _ := strings.Cut(name, "v")
	w = sortByPCHR(w)
	b = sortByPCHR(b)

	less := tupleLess(len(w), pchrIndices(b), len(b), pchrIndices(w))
	if mirror != less {
		return b + "v" + w
	}
	return w + "v" + b
}

func sortByPCHR(s string) string {
	r := []byte(s)
	sort.Slice(r, func(i, j int) bool { return pchrIndex(r[i]) < pchrIndex(r[j]) })
	return string(r)
}

func pchrIndex(c byte) int {
	for i, p := range pchrOrder {
		if p == c {
			return i
		}
	}
	return len(pchrOrder)
}

func pchrIndices(s string) []int {
	out := make([]int, len(s))
	for i := 0; i < len(s); i++ {
		out[i] = pchrIndex(s[i])
	}
	return out
}

// tupleLess compares (len1, list1) < (len2, list2) the way Python compares
// tuples: lexicographically, first element first.
func tupleLess(len1 int, list1 []int, len2 int, list2 []int) bool {
	if len1 != len2 {
		return len1 < len2
	}
	for i := 0; i < len(list1) && i < len(list2); i++ {
		if list1[i] != list2[i] {
			return list1[i] < list2[i]
		}
	}
	return len(list1) < len(list2)
}
