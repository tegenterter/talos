package board

import (
	"fmt"
	"strconv"
	"strings"
)

const StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

func StartingBoard() Board {
	b, err := ParseFEN(StartFEN)
	if err != nil {
		panic("board: StartFEN must be valid: " + err.Error())
	}
	return b
}

var fenPieces = map[byte]struct {
	color Color
	piece PieceType
}{
	'P': {White, Pawn}, 'N': {White, Knight}, 'B': {White, Bishop},
	'R': {White, Rook}, 'Q': {White, Queen}, 'K': {White, King},
	'p': {Black, Pawn}, 'n': {Black, Knight}, 'b': {Black, Bishop},
	'r': {Black, Rook}, 'q': {Black, Queen}, 'k': {Black, King},
}

func ParseFEN(fen string) (Board, error) {
	fields := strings.Fields(fen)
	if len(fields) < 4 {
		return Board{}, fmt.Errorf("invalid FEN %q: expected at least 4 fields", fen)
	}

	var b Board

	ranks := strings.Split(fields[0], "/")
	if len(ranks) != 8 {
		return Board{}, fmt.Errorf("invalid FEN %q: expected 8 ranks", fen)
	}
	for i, rankStr := range ranks {
		rank := 7 - i
		file := 0
		for _, ch := range []byte(rankStr) {
			if ch >= '1' && ch <= '8' {
				file += int(ch - '0')
				continue
			}
			pc, ok := fenPieces[ch]
			if !ok {
				return Board{}, fmt.Errorf("invalid FEN %q: unknown piece %q", fen, ch)
			}
			if file > 7 {
				return Board{}, fmt.Errorf("invalid FEN %q: rank %q overflows", fen, rankStr)
			}
			b.Pieces[pc.color][pc.piece] |= sqBit(Square(rank*8 + file))
			file++
		}
	}

	switch fields[1] {
	case "w":
		b.SideToMove = White
	case "b":
		b.SideToMove = Black
	default:
		return Board{}, fmt.Errorf("invalid FEN %q: bad side to move %q", fen, fields[1])
	}

	if fields[2] != "-" {
		for _, ch := range []byte(fields[2]) {
			switch ch {
			case 'K':
				b.CastlingRights |= WhiteKingside
			case 'Q':
				b.CastlingRights |= WhiteQueenside
			case 'k':
				b.CastlingRights |= BlackKingside
			case 'q':
				b.CastlingRights |= BlackQueenside
			default:
				return Board{}, fmt.Errorf("invalid FEN %q: bad castling rights %q", fen, fields[2])
			}
		}
	}

	if fields[3] == "-" {
		b.EnPassant = NoSquare
	} else {
		sq, ok := ParseSquare(fields[3])
		if !ok {
			return Board{}, fmt.Errorf("invalid FEN %q: bad en passant square %q", fen, fields[3])
		}
		b.EnPassant = sq
	}

	b.HalfmoveClock = 0
	b.FullmoveNumber = 1
	if len(fields) >= 5 {
		if n, err := strconv.Atoi(fields[4]); err == nil {
			b.HalfmoveClock = n
		}
	}
	if len(fields) >= 6 {
		if n, err := strconv.Atoi(fields[5]); err == nil {
			b.FullmoveNumber = n
		}
	}

	if err := validate(&b, fen); err != nil {
		return Board{}, err
	}

	return b, nil
}

// validate rejects the two illegal positions that a FEN can express and
// that the rest of this package has no defence against, because every
// position it produces itself is legal by construction.
//
// Neither check is about being strict for its own sake. A board missing a
// king crashes anything that looks one up (nnue.Evaluate indexes its
// feature table by king square, and an empty king bitboard yields square
// 64, past the end of the table). A board where the side *not* to move is
// already in check is worse than it sounds: the side to move can simply
// capture that king, and GenerateLegalMoves — which asks only whether the
// mover's own king is left attacked — will happily call that legal, so the
// resulting board has no king on it either. Both arrive the same way, from
// a GUI or a script sending "position fen" with a hand-written position,
// and a crash there costs a game.
func validate(b *Board, fen string) error {
	for _, c := range [2]Color{White, Black} {
		if n := b.Pieces[c][King].Count(); n != 1 {
			name := "white"
			if c == Black {
				name = "black"
			}
			return fmt.Errorf("invalid FEN %q: %s has %d kings, want exactly 1", fen, name, n)
		}
	}

	waiting := b.SideToMove.Opposite()
	if IsSquareAttacked(b, b.Pieces[waiting][King].LSB(), b.SideToMove) {
		return fmt.Errorf("invalid FEN %q: side not to move is in check", fen)
	}

	return nil
}

// FEN renders b in Forsyth-Edwards notation, the inverse of ParseFEN.
//
// Written for callers that hand a position to something outside this program
// — training data (internal/datagen), a test failure message, a debugging
// session — rather than for the engine itself, which passes boards around
// directly.
func (b *Board) FEN() string {
	var sb strings.Builder

	for rank := 7; rank >= 0; rank-- {
		empty := 0
		for file := 0; file < 8; file++ {
			c, pt, ok := b.PieceAt(Square(rank*8 + file))
			if !ok {
				empty++
				continue
			}
			if empty > 0 {
				sb.WriteByte(byte('0' + empty))
				empty = 0
			}
			sb.WriteByte(fenPieceByte(c, pt))
		}
		if empty > 0 {
			sb.WriteByte(byte('0' + empty))
		}
		if rank > 0 {
			sb.WriteByte('/')
		}
	}

	if b.SideToMove == White {
		sb.WriteString(" w ")
	} else {
		sb.WriteString(" b ")
	}

	rights := ""
	for _, r := range []struct {
		bit  uint8
		char byte
	}{
		{WhiteKingside, 'K'}, {WhiteQueenside, 'Q'},
		{BlackKingside, 'k'}, {BlackQueenside, 'q'},
	} {
		if b.CastlingRights&r.bit != 0 {
			rights += string(r.char)
		}
	}
	if rights == "" {
		rights = "-"
	}
	sb.WriteString(rights)

	if b.EnPassant == NoSquare {
		sb.WriteString(" -")
	} else {
		sb.WriteString(" " + b.EnPassant.String())
	}

	sb.WriteString(" " + strconv.Itoa(b.HalfmoveClock))
	sb.WriteString(" " + strconv.Itoa(b.FullmoveNumber))
	return sb.String()
}

// fenPieceByte is the FEN letter for a piece: uppercase for White.
func fenPieceByte(c Color, pt PieceType) byte {
	ch := "pnbrqk"[pt]
	if c == White {
		ch -= 'a' - 'A'
	}
	return ch
}
