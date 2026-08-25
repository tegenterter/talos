package tablebase

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"talos/internal/board"
)

// wdlMagic is the fixed 4-byte header standard-chess Syzygy WDL files
// start with (python-chess's chess.Board.tbw_magic).
var wdlMagic = [4]byte{0x71, 0xe8, 0x23, 0x5d}

// loadTable reads and fully parses one Syzygy WDL (.rtbw) file.
func loadTable(path string) (*table, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < 4 || [4]byte(data[:4]) != wdlMagic {
		return nil, fmt.Errorf("tablebase: %s: not a Syzygy WDL file (bad magic header)", path)
	}

	base := filepath.Base(path)
	tablename := strings.TrimSuffix(base, filepath.Ext(base))

	registryKey := normalizeTablename(tablename, false)
	registryMirroredKey := normalizeTablename(tablename, true)
	t := &table{
		data:                data,
		key:                 registryKey,
		mirroredKey:         registryMirroredKey,
		registryKey:         registryKey,
		registryMirroredKey: registryMirroredKey,
		numPieces:           len(tablename) - 1,
		hasPawns:            strings.Contains(tablename, "P"),
	}
	t.symmetric = t.key == t.mirroredKey

	// blackPart/whitePart: these names are swapped relative to what they
	// actually hold (blackPart is the tablename's first, "white pieces"
	// half), matching python-chess's Table.__init__ exactly — preserved
	// as-is rather than "fixed", since only self-consistent usage below
	// (mirroring the Python) matters for correctness, not the names.
	blackPart, whitePart, ok := strings.Cut(tablename, "v")
	if !ok {
		return nil, fmt.Errorf("tablebase: %s: malformed tablename %q", path, tablename)
	}

	if t.hasPawns {
		t.pawns[0] = strings.Count(whitePart, "P")
		t.pawns[1] = strings.Count(blackPart, "P")
		if t.pawns[1] > 0 && (t.pawns[0] == 0 || t.pawns[1] < t.pawns[0]) {
			t.pawns[0], t.pawns[1] = t.pawns[1], t.pawns[0]
		}
	} else {
		j := 0
		for _, p := range pchrOrder {
			if strings.Count(blackPart, string(p)) == 1 {
				j++
			}
			if strings.Count(whitePart, string(p)) == 1 {
				j++
			}
		}
		switch {
		case j >= 3:
			t.encType = 0
		case j == 2:
			t.encType = 2
		default:
			t.encType = 1 // unreachable for standard chess (one_king=true)
		}
	}

	if err := t.initWDL(); err != nil {
		return nil, fmt.Errorf("tablebase: %s: %w", path, err)
	}
	return t, nil
}

// initWDL parses the file's layout header: for a pawnless table, one (or
// two, if the file is "split" by side to move) block-compressed section
// covering every position; for a table with pawns, the same but
// separately for each of up to 4 "pawn file" groups (see pawnFile).
func (t *table) initWDL() error {
	split := t.data[4]&0x01 != 0
	files := 1
	if t.data[4]&0x02 != 0 {
		files = 4
	}

	dataPtr := 5

	if !t.hasPawns {
		t.setupPiecesPiece(dataPtr)
		dataPtr += t.numPieces + 1
		dataPtr += dataPtr & 0x01

		var next int
		t.precomp[0], next = t.setupPairs(dataPtr, t.tbSize[0], 0)
		dataPtr = next
		if split {
			t.precomp[1], next = t.setupPairs(dataPtr, t.tbSize[1], 3)
			dataPtr = next
		}

		t.precomp[0].indexTable = dataPtr
		dataPtr += t.size[0]
		if split {
			t.precomp[1].indexTable = dataPtr
			dataPtr += t.size[3]
		}

		t.precomp[0].sizeTable = dataPtr
		dataPtr += t.size[1]
		if split {
			t.precomp[1].sizeTable = dataPtr
			dataPtr += t.size[4]
		}

		dataPtr = (dataPtr + 0x3f) &^ 0x3f
		t.precomp[0].data = dataPtr
		dataPtr += t.size[2]
		if split {
			dataPtr = (dataPtr + 0x3f) &^ 0x3f
			t.precomp[1].data = dataPtr
		}

		t.key = recalcKey(t.pieces[0], false)
		t.mirroredKey = recalcKey(t.pieces[0], true)
	} else {
		s := 1
		if t.pawns[1] > 0 {
			s = 2
		}
		for f := 0; f < 4; f++ {
			t.setupPiecesPawn(dataPtr, 2*f, f)
			dataPtr += t.numPieces + s
		}
		dataPtr += dataPtr & 0x01

		for f := 0; f < files; f++ {
			var next int
			t.files[f].precomp[0], next = t.setupPairs(dataPtr, t.tbSize[2*f], 6*f)
			dataPtr = next
			if split {
				t.files[f].precomp[1], next = t.setupPairs(dataPtr, t.tbSize[2*f+1], 6*f+3)
				dataPtr = next
			}
		}

		for f := 0; f < files; f++ {
			t.files[f].precomp[0].indexTable = dataPtr
			dataPtr += t.size[6*f]
			if split {
				t.files[f].precomp[1].indexTable = dataPtr
				dataPtr += t.size[6*f+3]
			}
		}

		for f := 0; f < files; f++ {
			t.files[f].precomp[0].sizeTable = dataPtr
			dataPtr += t.size[6*f+1]
			if split {
				t.files[f].precomp[1].sizeTable = dataPtr
				dataPtr += t.size[6*f+4]
			}
		}

		for f := 0; f < files; f++ {
			dataPtr = (dataPtr + 0x3f) &^ 0x3f
			t.files[f].precomp[0].data = dataPtr
			dataPtr += t.size[6*f+2]
			if split {
				dataPtr = (dataPtr + 0x3f) &^ 0x3f
				t.files[f].precomp[1].data = dataPtr
				dataPtr += t.size[6*f+5]
			}
		}
	}

	return nil
}

func (t *table) setupPiecesPiece(pData int) {
	t.pieces[0] = make([]int, t.numPieces)
	for i := 0; i < t.numPieces; i++ {
		t.pieces[0][i] = int(t.data[pData+i+1] & 0x0f)
	}
	order := int(t.data[pData] & 0x0f)
	t.norm[0] = make([]int, t.numPieces)
	t.setNormPiece(t.norm[0], t.pieces[0])
	t.tbSize[0] = t.calcFactorsPiece(&t.factor[0], order, t.norm[0])

	t.pieces[1] = make([]int, t.numPieces)
	for i := 0; i < t.numPieces; i++ {
		t.pieces[1][i] = int(t.data[pData+i+1] >> 4)
	}
	order = int(t.data[pData] >> 4)
	t.norm[1] = make([]int, t.numPieces)
	t.setNormPiece(t.norm[1], t.pieces[1])
	t.tbSize[1] = t.calcFactorsPiece(&t.factor[1], order, t.norm[1])
}

func (t *table) setupPiecesPawn(pData, pTbSize, f int) {
	j := 1
	if t.pawns[1] > 0 {
		j = 2
	}

	order := int(t.data[pData] & 0x0f)
	order2 := 0x0f
	if t.pawns[1] > 0 {
		order2 = int(t.data[pData+1] & 0x0f)
	}
	t.files[f].pieces[0] = make([]int, t.numPieces)
	for i := 0; i < t.numPieces; i++ {
		t.files[f].pieces[0][i] = int(t.data[pData+i+j] & 0x0f)
	}
	t.files[f].norm[0] = make([]int, t.numPieces)
	t.setNormPawn(t.files[f].norm[0], t.files[f].pieces[0])
	t.tbSize[pTbSize] = t.calcFactorsPawn(&t.files[f].factor[0], order, order2, t.files[f].norm[0], f)

	order = int(t.data[pData] >> 4)
	order2 = 0x0f
	if t.pawns[1] > 0 {
		order2 = int(t.data[pData+1] >> 4)
	}
	t.files[f].pieces[1] = make([]int, t.numPieces)
	for i := 0; i < t.numPieces; i++ {
		t.files[f].pieces[1][i] = int(t.data[pData+i+j] >> 4)
	}
	t.files[f].norm[1] = make([]int, t.numPieces)
	t.setNormPawn(t.files[f].norm[1], t.files[f].pieces[1])
	t.tbSize[pTbSize+1] = t.calcFactorsPawn(&t.files[f].factor[1], order, order2, t.files[f].norm[1], f)
}

// probe returns this table's raw WDL value (0..4, i.e. not yet rebased to
// the -2..2 range — see (*Tablebase).probeWDLTable) for b, which must
// already have exactly this table's material signature (up to color and
// mirroring).
func (t *table) probe(b *board.Board) int {
	key := calcKey(b)

	var cmirror, mirror int
	var bside int
	switch {
	case !t.symmetric && key != t.key:
		cmirror, mirror = 8, 0x38
		bside = b2i(b.SideToMove == board.White)
	case !t.symmetric:
		cmirror, mirror = 0, 0
		bside = b2i(b.SideToMove != board.White)
	default:
		if b.SideToMove == board.White {
			cmirror, mirror = 0, 0
		} else {
			cmirror, mirror = 8, 0x38
		}
		bside = 0
	}

	p := make([]int, tbPieces)

	if !t.hasPawns {
		i := 0
		for i < t.numPieces {
			raw := t.pieces[bside][i] ^ cmirror
			pt, color := rawPieceType(raw)
			bb := b.Pieces[color][pt]
			for bb != 0 {
				sq := bb.LSB()
				bb &= bb - 1
				p[i] = int(sq)
				i++
			}
		}

		idx := t.encodePiece(t.norm[bside], p, &t.factor[bside])
		return t.decompressPairs(t.precomp[bside], idx)
	}

	i := 0
	k := t.files[0].pieces[0][0] ^ cmirror
	pt, color := rawPieceType(k)
	bb := b.Pieces[color][pt]
	for bb != 0 {
		sq := bb.LSB()
		bb &= bb - 1
		p[i] = int(sq) ^ mirror
		i++
	}

	f := t.pawnFile(p)
	pc := t.files[f].pieces[bside]
	for i < t.numPieces {
		raw := pc[i] ^ cmirror
		pt, color := rawPieceType(raw)
		bb := b.Pieces[color][pt]
		for bb != 0 {
			sq := bb.LSB()
			bb &= bb - 1
			p[i] = int(sq) ^ mirror
			i++
		}
	}

	idx := t.encodePawn(t.files[f].norm[bside], p, &t.files[f].factor[bside])
	return t.decompressPairs(t.files[f].precomp[bside], idx)
}

// rawPieceType splits a raw (nibble) piece code into a board.PieceType
// and board.Color.
func rawPieceType(raw int) (board.PieceType, board.Color) {
	color := board.White
	if raw&rawColor != 0 {
		color = board.Black
	}
	var pt board.PieceType
	switch raw & 0x07 {
	case rawPawn:
		pt = board.Pawn
	case rawKnight:
		pt = board.Knight
	case rawBishop:
		pt = board.Bishop
	case rawRook:
		pt = board.Rook
	case rawQueen:
		pt = board.Queen
	default:
		pt = board.King
	}
	return pt, color
}
