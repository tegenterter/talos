package search

// infinity is a sentinel far beyond any realistic evaluation, used as the
// initial alpha-beta window bound.
const infinity = 1 << 30

// mateValue is the score magnitude for "checkmate delivered right now".
// A mate found ply plies deep scores ±(mateValue - ply), so a faster mate
// (smaller ply) always scores higher in magnitude than a slower one —
// negamax will therefore prefer the quickest forced mate it can find.
const mateValue = 32000

// maxPly bounds search depth (iterative deepening won't go beyond it) and
// sizes the killer-move table; also used to derive mateThreshold.
const maxPly = 128

// mateThreshold is the score magnitude above which a score is a mate
// score rather than a normal evaluation: mateValue minus the deepest
// mate distance maxPly can represent, so no ordinary centipawn
// evaluation (however lopsided) is ever misread as a mate score.
const mateThreshold = mateValue - maxPly
