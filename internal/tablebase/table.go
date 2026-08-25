// Package tablebase implements Syzygy WDL (win/draw/loss) endgame
// tablebase probing, ported from python-chess's chess/syzygy.py (a
// from-scratch, well-tested reimplementation of Ronald de Man's Syzygy
// format) — the standard, well-documented reference this port was
// verified line-by-line against, since the on-disk format itself (a
// custom block-compressed, Huffman-like symbol encoding over a
// combinatorial board-position index) has no official specification.
//
// Scope: WDL only, not DTZ (distance-to-zero). WDL alone gives exact
// game-theoretic knowledge of any loaded endgame — the engine will never
// misjudge a tablebase win, draw, or loss — but, unlike DTZ, doesn't
// guarantee finding the fastest possible conversion once very close to
// the fifty-move-rule limit. DTZ shares WDL's same complex decoder but
// adds its own value-mapping/zeroing-move subtleties on top, which was
// judged not worth the extra implementation and verification risk for
// what it buys (see the "tablebase" branch's task description). Also
// scoped out, since standard chess doesn't need them: DTZ, suicide/atomic/
// other variant support (connected_kings, captures_compulsory, multi-king
// encodings), and the mmap-based lazy file loading python-chess uses —
// this package just reads a whole table file into memory once, matching
// this codebase's established "simple over fast" bias (see internal/eval
// and internal/nnue's package docs) and keeping this port's already
// substantial complexity from growing further.
package tablebase

import "encoding/binary"

// tbPieces is the maximum number of pieces (kings included) a Syzygy
// table can describe.
const tbPieces = 7

// pairsData mirrors python-chess's PairsData: everything decompress_pairs
// needs to decode one value out of one block-compressed table. All the
// "*int" fields are byte offsets into the owning table's data slice, not
// real pointers — matching how the Python original treats mmap'd file
// offsets as plain integers.
type pairsData struct {
	indexTable int
	sizeTable  int
	data       int
	offset     int
	symLen     []int
	symPat     int
	blockSize  int
	idxBits    int
	minLen     int
	base       []uint64
}

// pawnFileData holds one of the (up to 4) "pawn file" sub-tables a table
// with pawns splits into, one per pairsData/factor/pieces/norm needed for
// each side (index 0/1 — see table.probe's bside).
type pawnFileData struct {
	precomp [2]*pairsData
	factor  [2][tbPieces]int
	pieces  [2][]int
	norm    [2][]int
}

// table is one parsed Syzygy WDL file (python-chess's Table + WdlTable
// combined, since this package only ever builds WDL tables).
type table struct {
	data []byte

	// key/mirroredKey start out equal to registryKey/registryMirroredKey
	// (both derived from the filename), but initWDL overwrites them for a
	// pawnless table whose header says its true material key differs from
	// what its filename implies (see initWDL's doc comment) — from then
	// on they're used only inside probe's symmetric/cmirror decision.
	// registryKey/registryMirroredKey never change after loadTable sets
	// them, and are what Tablebase looks tables up by: they must always
	// equal what calcKey(board) computes for a real board of this
	// material, which is exactly what a table's filename promises,
	// regardless of what its header internally claims about itself.
	key, mirroredKey                 string
	registryKey, registryMirroredKey string
	symmetric                        bool
	numPieces                        int // total pieces (both sides, kings included)
	hasPawns                         bool
	pawns                            [2]int // [0] = stronger side's pawn count (see loadTable)
	encType                          int    // only meaningful when !hasPawns

	tbSize [8]int
	size   [24]int

	// Populated when !hasPawns.
	precomp [2]*pairsData
	pieces  [2][]int
	factor  [2][tbPieces]int
	norm    [2][]int

	// Populated when hasPawns: one sub-table per pawn "file group" (see
	// pawnFile), only files[0] used when the layout doesn't split by file.
	files [4]pawnFileData
}

func (t *table) readUint64BE(off int) uint64 { return binary.BigEndian.Uint64(t.data[off:]) }
func (t *table) readUint32(off int) uint32   { return binary.LittleEndian.Uint32(t.data[off:]) }
func (t *table) readUint32BE(off int) uint32 { return binary.BigEndian.Uint32(t.data[off:]) }
func (t *table) readUint16(off int) uint16   { return binary.LittleEndian.Uint16(t.data[off:]) }

// setupPairs parses one block-compression header (a PairsData) starting
// at dataPtr, and returns the offset immediately following it (python's
// self._next, folded into the return here since Go can return both).
// size_idx selects which 3-slot window of t.size this table's derived
// section sizes are recorded into, for the caller to lay out the index
// table / block-size table / compressed data that follow every header in
// the file (see initWDL).
func (t *table) setupPairs(dataPtr, tbSize, sizeIdx int) (*pairsData, int) {
	d := &pairsData{}

	if t.data[dataPtr]&0x80 != 0 {
		// A "constant" table: every position has the same value, so
		// there's nothing to decompress — see decompress_pairs's
		// `if not d.idxbits: return d.min_len` fast path.
		d.idxBits = 0
		d.minLen = int(t.data[dataPtr+1])
		t.size[sizeIdx+0] = 0
		t.size[sizeIdx+1] = 0
		t.size[sizeIdx+2] = 0
		return d, dataPtr + 2
	}

	d.blockSize = int(t.data[dataPtr+1])
	d.idxBits = int(t.data[dataPtr+2])

	realNumBlocks := int(t.readUint32(dataPtr + 4))
	numBlocks := realNumBlocks + int(t.data[dataPtr+3])
	maxLen := int(t.data[dataPtr+8])
	minLen := int(t.data[dataPtr+9])
	h := maxLen - minLen + 1
	numSyms := int(t.readUint16(dataPtr + 10 + 2*h))

	d.offset = dataPtr + 10
	d.symLen = make([]int, h*8+numSyms)
	d.symPat = dataPtr + 12 + 2*h
	d.minLen = minLen

	next := dataPtr + 12 + 2*h + 3*numSyms + (numSyms & 1)

	numIndices := (tbSize + (1 << d.idxBits) - 1) >> d.idxBits
	t.size[sizeIdx+0] = 6 * numIndices
	t.size[sizeIdx+1] = 2 * numBlocks
	t.size[sizeIdx+2] = (1 << d.blockSize) * realNumBlocks

	tmp := make([]bool, numSyms)
	for i := 0; i < numSyms; i++ {
		if !tmp[i] {
			t.calcSymLen(d, i, tmp)
		}
	}

	d.base = make([]uint64, h)
	d.base[h-1] = 0
	for i := h - 2; i >= 0; i-- {
		d.base[i] = (d.base[i+1] + uint64(t.readUint16(d.offset+i*2)) - uint64(t.readUint16(d.offset+i*2+2))) / 2
	}
	for i := 0; i < h; i++ {
		d.base[i] <<= uint(64 - (minLen + i))
	}

	d.offset -= 2 * d.minLen

	return d, next
}

// calcSymLen recursively fills in d.symLen[s]: how many "leaf" values
// symbol s expands to, by walking its Huffman-style binary decomposition
// (each non-leaf symbol is a pair of two other symbols, s1/s2, packed
// into 3 bytes at sympat+3*s). tmp tracks which symbols are already
// resolved so shared sub-symbols aren't recomputed.
func (t *table) calcSymLen(d *pairsData, s int, tmp []bool) {
	w := d.symPat + 3*s
	s2 := (int(t.data[w+2]) << 4) | (int(t.data[w+1]) >> 4)
	if s2 == 0x0fff {
		d.symLen[s] = 0
	} else {
		s1 := ((int(t.data[w+1]) & 0xf) << 8) | int(t.data[w])
		if !tmp[s1] {
			t.calcSymLen(d, s1, tmp)
		}
		if !tmp[s2] {
			t.calcSymLen(d, s2, tmp)
		}
		d.symLen[s] = d.symLen[s1] + d.symLen[s2] + 1
	}
	tmp[s] = true
}

// decompressPairs decodes the value stored at table index idx within one
// block-compressed section d: it locates idx's block via d's sparse index
// table, walks that block's Huffman-style bitstream (code/base/offset)
// counting past shorter symbols until it lands on the one containing
// idx's position within the block (litidx), then descends that symbol's
// binary tree to the single-value leaf.
func (t *table) decompressPairs(d *pairsData, idx int) int {
	if d.idxBits == 0 {
		return d.minLen
	}

	mainIdx := idx >> d.idxBits
	litIdx := (idx & ((1 << d.idxBits) - 1)) - (1 << (d.idxBits - 1))
	block := int(t.readUint32(d.indexTable + 6*mainIdx))

	idxOffset := int(t.readUint16(d.indexTable + 6*mainIdx + 4))
	litIdx += idxOffset

	if litIdx < 0 {
		for litIdx < 0 {
			block--
			litIdx += int(t.readUint16(d.sizeTable+2*block)) + 1
		}
	} else {
		for litIdx > int(t.readUint16(d.sizeTable+2*block)) {
			litIdx -= int(t.readUint16(d.sizeTable+2*block)) + 1
			block++
		}
	}

	ptr := d.data + (block << d.blockSize)

	m := d.minLen
	baseIdx := -m
	symLenIdx := 0

	code := t.readUint64BE(ptr)
	ptr += 2 * 4
	bitCnt := 0

	var sym int
	for {
		l := m
		for code < d.base[baseIdx+l] {
			l++
		}
		sym = int(t.readUint16(d.offset + l*2))
		sym += int((code - d.base[baseIdx+l]) >> uint(64-l))
		if litIdx < d.symLen[symLenIdx+sym]+1 {
			break
		}
		litIdx -= d.symLen[symLenIdx+sym] + 1
		code <<= uint(l)
		bitCnt += l
		if bitCnt >= 32 {
			bitCnt -= 32
			code |= uint64(t.readUint32BE(ptr)) << uint(bitCnt)
			ptr += 4
		}
	}

	sympat := d.symPat
	for d.symLen[symLenIdx+sym] != 0 {
		w := sympat + 3*sym
		s1 := ((int(t.data[w+1]) & 0xf) << 8) | int(t.data[w])
		if litIdx < d.symLen[symLenIdx+s1]+1 {
			sym = s1
		} else {
			litIdx -= d.symLen[symLenIdx+s1] + 1
			sym = (int(t.data[w+2]) << 4) | (int(t.data[w+1]) >> 4)
		}
	}

	w := sympat + 3*sym
	return int(t.data[w])
}
