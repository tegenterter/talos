package board

// MoveFlag marks moves that need special handling beyond "piece goes from
// From to To".
type MoveFlag int

const (
	// Quiet also covers ordinary captures, despite the name: whether a
	// move captures is derived from board occupancy (MakeMove, isCapture)
	// rather than from this flag, so only the moves genuinely needing
	// dedicated handling — castling, en passant, and a pawn's double push
	// — get their own flag here.
	Quiet MoveFlag = iota
	DoublePawnPush
	EnPassantCapture
	CastleKingside
	CastleQueenside
)

// Move is a single chess move in a specific position. Flag disambiguates
// moves that MakeMove needs to treat specially (en passant, castling, a
// double pawn push that opens an en passant target).
type Move struct {
	From      Square
	To        Square
	Promotion PieceType // NoPiece unless this move promotes a pawn
	Flag      MoveFlag
}

func (m Move) String() string {
	s := m.From.String() + m.To.String()
	switch m.Promotion {
	case Queen:
		s += "q"
	case Rook:
		s += "r"
	case Bishop:
		s += "b"
	case Knight:
		s += "n"
	}
	return s
}

// ParseUCIMove parses UCI long algebraic notation (e.g. "e2e4", "e7e8q").
// The returned move's Flag is always Quiet, since determining special
// flags (castling, en passant, double push) requires knowing the position;
// callers should match the result against a generated legal move to get
// the correct flag before calling MakeMove.
func ParseUCIMove(s string) (Move, bool) {
	if len(s) != 4 && len(s) != 5 {
		return Move{}, false
	}
	from, ok := ParseSquare(s[0:2])
	if !ok {
		return Move{}, false
	}
	to, ok := ParseSquare(s[2:4])
	if !ok {
		return Move{}, false
	}
	promo := NoPiece
	if len(s) == 5 {
		switch s[4] {
		case 'q':
			promo = Queen
		case 'r':
			promo = Rook
		case 'b':
			promo = Bishop
		case 'n':
			promo = Knight
		default:
			return Move{}, false
		}
	}
	return Move{From: from, To: to, Promotion: promo}, true
}
