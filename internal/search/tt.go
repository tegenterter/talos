package search

import (
	"sync"

	"talos/internal/board"
)

// DefaultHashMB matches Stockfish's own UCI "Hash" option default.
const DefaultHashMB = 16

// ttFlag records what kind of bound a stored score is, since alpha-beta
// only proves an exact score when the search wasn't cut off:
//   - ttExact: the true score of the position.
//   - ttLowerBound: the position is at least this good (search failed
//     high / caused a beta cutoff — the real score could be higher).
//   - ttUpperBound: the position is at most this good (search failed low
//     / never reached alpha — the real score could be lower).
type ttFlag uint8

const (
	ttExact ttFlag = iota
	ttLowerBound
	ttUpperBound
)

type ttEntry struct {
	hash  uint64
	move  board.Move
	score int
	depth int
	flag  ttFlag
}

// ttShards splits the table into independently-locked shards so many
// search threads probing/storing concurrently don't all contend on one
// mutex. Chosen as a power of two so shard selection is a cheap mask.
const ttShards = 256

type ttShard struct {
	mu      sync.Mutex
	entries []ttEntry
}

// transpositionTable is a fixed-size, always-present table of search
// results keyed by Zobrist hash (board.Board.Hash()) — reused across the
// whole search, and across every goroutine working on it, so that reaching
// the same position by a different move order lets one thread's earlier
// work shortcut another's. Entries are never explicitly deleted; a hash
// collision on the fixed-size table (a different position landing on the
// same slot) just gets overwritten per the replacement policy in store.
type transpositionTable struct {
	shardMask uint64
	shards    []ttShard
}

// approxTTEntryBytes is a rough, documented estimate of one ttEntry's
// real footprint (its fields plus Go struct padding/alignment), used
// only to translate a "Hash" megabyte setting into a table size — exact
// accounting would require reflect/unsafe for marginal benefit here.
const approxTTEntryBytes = 64

const minTTEntriesPerShard = 64

// newTranspositionTable sizes the table from a "Hash" setting in
// megabytes, split evenly across ttShards.
func newTranspositionTable(hashMB int) *transpositionTable {
	if hashMB <= 0 {
		hashMB = DefaultHashMB
	}
	perShard := (hashMB * 1024 * 1024) / approxTTEntryBytes / ttShards
	if perShard < minTTEntriesPerShard {
		perShard = minTTEntriesPerShard
	}

	shards := make([]ttShard, ttShards)
	for i := range shards {
		shards[i].entries = make([]ttEntry, perShard)
	}
	return &transpositionTable{shardMask: ttShards - 1, shards: shards}
}

// Table is a transposition table a caller can keep alive across separate
// Search calls, passed in via Options.Table.
//
// This matters for play strength, not just allocation cost. Successive moves
// in a game overwhelmingly search overlapping trees — the position after the
// opponent replies is one this search already examined — so a table rebuilt
// per move throws away exactly the results the next move would benefit from
// most, and every search restarts cold. (It also re-allocates the whole table
// each time, which at Hash 1024 means a gigabyte per move.)
//
// Reuse is safe without an aging/generation scheme because store's replacement
// policy only defers to an existing entry when the *hash matches*: an entry
// belonging to a different position is always evicted, so stale results can't
// accumulate and permanently clog the table. Aging would still help
// (it would let a shallow current-search entry displace a deep stale one on
// the same key) and is a reasonable later refinement, not a prerequisite.
// A Table is emptied by discarding it and allocating a new one (what
// internal/uci does for "ucinewgame" and for a changed "Hash"), rather than by
// a Clear method. That is deliberately the only way: a search running in the
// background holds its own reference to the underlying table, so replacing the
// caller's pointer is race-free, whereas zeroing entries in place would
// corrupt an in-flight search that the GUI never sent "stop" for.
type Table struct {
	tt *transpositionTable
}

// NewTable allocates a reusable table sized per the "Hash" UCI option, in
// megabytes. Zero or negative means DefaultHashMB.
func NewTable(hashMB int) *Table {
	return &Table{tt: newTranspositionTable(hashMB)}
}

// table returns the underlying table to search with, allocating a throwaway
// one sized by hashMB when t is nil. Defined on the pointer receiver so a nil
// Options.Table needs no special case at the call site.
func (t *Table) table(hashMB int) *transpositionTable {
	if t == nil {
		return newTranspositionTable(hashMB)
	}
	return t.tt
}

func (t *transpositionTable) shardFor(hash uint64) *ttShard {
	return &t.shards[hash&t.shardMask]
}

func (t *transpositionTable) slotFor(shard *ttShard, hash uint64) int {
	return int((hash >> 32) % uint64(len(shard.entries)))
}

// probe returns the entry stored for hash, if any, with any mate score it
// carries re-expressed relative to ply (see adjustMateScoreFromTT) — a
// hash collision on the fixed-size table (a different position occupying
// the same slot) is indistinguishable from "not found" and reported as such.
func (t *transpositionTable) probe(hash uint64, ply int) (ttEntry, bool) {
	shard := t.shardFor(hash)
	shard.mu.Lock()
	e := shard.entries[t.slotFor(shard, hash)]
	shard.mu.Unlock()

	if e.hash != hash {
		return ttEntry{}, false
	}
	e.score = adjustMateScoreFromTT(e.score, ply)
	return e, true
}

// store records an entry for hash, keeping whatever is already there
// instead if it came from a strictly deeper search (a simple
// "depth-preferred" replacement policy: deeper results are more
// trustworthy and more expensive to recompute; an equal-depth result still
// overwrites, so the table's occupant for a given depth stays fresh).
func (t *transpositionTable) store(hash uint64, move board.Move, score, depth int, flag ttFlag, ply int) {
	shard := t.shardFor(hash)
	shard.mu.Lock()
	defer shard.mu.Unlock()

	e := &shard.entries[t.slotFor(shard, hash)]
	if e.hash == hash && e.depth > depth {
		return
	}
	*e = ttEntry{
		hash:  hash,
		move:  move,
		score: adjustMateScoreToTT(score, ply),
		depth: depth,
		flag:  flag,
	}
}

// adjustMateScoreToTT converts a mate score from "plies to mate counted
// from the current search's root" (what negamax works with, so that a
// faster mate always scores higher than a slower one) into "plies to
// mate counted from this position" before storing it. Without this, a
// transposition reached at a different ply than where the entry was
// created would inherit the wrong mate distance — or even the wrong side
// winning, once shifted past the mate/non-mate threshold.
func adjustMateScoreToTT(score, ply int) int {
	switch {
	case score >= mateThreshold:
		return score + ply
	case score <= -mateThreshold:
		return score - ply
	default:
		return score
	}
}

// adjustMateScoreFromTT is adjustMateScoreToTT's inverse, applied when a
// stored entry is reused at (possibly) a different ply than it was
// stored at.
func adjustMateScoreFromTT(score, ply int) int {
	switch {
	case score >= mateThreshold:
		return score - ply
	case score <= -mateThreshold:
		return score + ply
	default:
		return score
	}
}
