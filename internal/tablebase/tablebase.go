package tablebase

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"talos/internal/board"
)

// tablenameRegexp matches a Syzygy tablename's shape ("KQvKR"), without
// checking it's in canonical (sorted, correctly-ordered) form — see
// isTablename.
var tablenameRegexp = regexp.MustCompile(`^[KQRBNP]+v[KQRBNP]+$`)

// isTablename reports whether name (a file's basename with its extension
// already stripped) is a valid, canonically-normalized Syzygy tablename
// for standard chess (exactly one king per side, at most tbPieces total).
func isTablename(name string) bool {
	if len(name) > tbPieces+1 {
		return false
	}
	if !tablenameRegexp.MatchString(name) {
		return false
	}
	if normalizeTablename(name, false) != name {
		return false
	}
	return name != "KvK" && strings.HasPrefix(name, "K") && strings.Contains(name, "vK")
}

// Tablebase holds every Syzygy WDL table loaded via AddDirectory/AddFile,
// ready to probe with Probe.
type Tablebase struct {
	wdl map[string]*table
}

// NewTablebase returns an empty Tablebase; load tables into it with
// AddDirectory or AddFile before probing.
func NewTablebase() *Tablebase {
	return &Tablebase{wdl: map[string]*table{}}
}

// AddDirectory loads every recognizable Syzygy WDL (.rtbw) file directly
// inside dir (not recursively), skipping anything else in the directory
// (including .rtbz DTZ files, which this package doesn't use — see the
// package doc comment). It returns how many files were loaded.
func (tb *Tablebase) AddDirectory(dir string) (int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, fmt.Errorf("tablebase: reading %s: %w", dir, err)
	}
	n := 0
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ok, err := tb.AddFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return n, err
		}
		if ok {
			n++
		}
	}
	return n, nil
}

// AddFile loads path as a single Syzygy WDL table, if its name looks like
// one (e.g. "KQvKR.rtbw"); otherwise it's silently skipped (ok=false),
// which lets AddDirectory be pointed at a directory containing other
// files (DTZ tables, README, ...) without complaint.
func (tb *Tablebase) AddFile(path string) (bool, error) {
	base := filepath.Base(path)
	if filepath.Ext(base) != ".rtbw" {
		return false, nil
	}
	tablename := strings.TrimSuffix(base, ".rtbw")
	if !isTablename(tablename) {
		return false, nil
	}

	t, err := loadTable(path)
	if err != nil {
		return false, err
	}
	tb.wdl[t.registryKey] = t
	tb.wdl[t.registryMirroredKey] = t
	return true, nil
}

// wdlTable looks up the table for b's exact material signature. ok is
// false if no such table has been loaded (not necessarily an error — the
// caller may simply not have tables that deep loaded).
func (tb *Tablebase) wdlTable(b *board.Board) (*table, bool) {
	t, ok := tb.wdl[calcKey(b)]
	return t, ok
}

// probeWDLTable is one non-recursive WDL lookup: b must already reflect
// whatever move sequence got it there, and probeWDLTable itself doesn't
// consider any further moves — see probeAB for the capture-aware search
// this is a building block of. Returns ok=false if no table is loaded for
// b's material.
func (tb *Tablebase) probeWDLTable(b *board.Board) (int, bool) {
	if b.OccupiedBB() == b.Pieces[board.White][board.King]|b.Pieces[board.Black][board.King] {
		return 0, true // King vs King is always a draw.
	}
	t, ok := tb.wdlTable(b)
	if !ok {
		return 0, false
	}
	return t.probe(b) - 2, true
}

// probeAB is a shallow, capture-only search: it recurses through every
// legal non-en-passant capture (a capture always transitions to a
// strictly smaller, already-tabulated material signature, so this always
// bottoms out) before falling back to a direct WDL table lookup at a
// "quiet" (capture-free) position, returning the best (from the side to
// move's perspective) of the two. This is necessary even though a WDL
// table's own values already reflect optimal play within its own material
// signature, because captures move to a *different* table entirely, and
// only the search here (not any single table) can compare "capture now"
// against "the position's own tabulated value."
//
// alpha/beta prune the same way a normal alpha-beta search does, using
// WDL's narrow value range (an initial call of (-2, 2) — a null window
// only shrinks during recursion — is always enough to see the true value).
func (tb *Tablebase) probeAB(b *board.Board, alpha, beta int) (int, bool) {
	if popcount(b) > tbPieces+1 {
		return 0, false
	}

	for _, m := range nonEPCaptures(b) {
		child := board.MakeMove(*b, m)
		v, ok := tb.probeAB(&child, -beta, -alpha)
		if !ok {
			return 0, false
		}
		v = -v
		if v > alpha {
			if v >= beta {
				return v, true
			}
			alpha = v
		}
	}

	v, ok := tb.probeWDLTable(b)
	if !ok {
		return 0, false
	}
	if alpha >= v {
		return alpha, true
	}
	return v, true
}

// Probe returns b's WDL classification from the perspective of the side
// to move: 2 (win), 1 (cursed win — a win ignoring the fifty-move rule,
// but a draw under it), 0 (draw), -1 (blessed loss), or -2 (loss). ok is
// false if the required table(s) aren't loaded, or if b has more pieces
// than any loaded table (plus one, to allow resolving a single capture)
// can cover.
func (tb *Tablebase) Probe(b *board.Board) (int, bool) {
	v, ok := tb.probeAB(b, -2, 2)
	if !ok {
		return 0, false
	}

	if b.EnPassant == board.NoSquare {
		return v, true
	}

	legal := board.GenerateLegalMoves(b)
	v1 := -3
	epFound := false
	allEP := true
	for _, m := range legal {
		if m.Flag != board.EnPassantCapture {
			allEP = false
			continue
		}
		epFound = true
		child := board.MakeMove(*b, m)
		v0, ok := tb.probeAB(&child, -2, 2)
		if !ok {
			return 0, false
		}
		v0 = -v0
		if v0 > v1 {
			v1 = v0
		}
	}

	if epFound {
		switch {
		case v1 >= v:
			v = v1
		case v == 0 && allEP:
			// Every legal move is an en passant capture, so however bad
			// v1 (the best of them) is, it's forced.
			v = v1
		}
	}

	return v, true
}

// GetWDL is Probe, but returning a default value instead of ok=false —
// convenient for callers (like internal/search) that want to just fall
// back to normal evaluation/search when a position isn't covered.
func (tb *Tablebase) GetWDL(b *board.Board, def int) int {
	if v, ok := tb.Probe(b); ok {
		return v
	}
	return def
}

func popcount(b *board.Board) int {
	return b.OccupiedBB().Count()
}

// nonEPCaptures returns every legal move whose target square is occupied
// by the side not to move — i.e. every legal capture except en passant
// (whose target square, behind the captured pawn, is empty). This mirrors
// python-chess's `board.generate_legal_moves(to_mask=occupied_co[not turn])`.
func nonEPCaptures(b *board.Board) []board.Move {
	opponent := b.ColorBB(b.SideToMove.Opposite())
	legal := board.GenerateLegalMoves(b)
	captures := make([]board.Move, 0, len(legal))
	for _, m := range legal {
		if m.Flag != board.EnPassantCapture && opponent&sqBit(m.To) != 0 {
			captures = append(captures, m)
		}
	}
	return captures
}

func sqBit(sq board.Square) board.Bitboard { return board.Bitboard(1) << uint(sq) }
