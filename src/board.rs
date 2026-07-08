use std::ops::Not;

use crate::board::Side::{Black, White};

pub type Bitboard = u64;
const DEFAULT_WHITE: Bitboard = 65535;
const DEFAULT_BLACK: Bitboard = 18446462598732840960;
const DEFAULT_WHITE_ARR: [Bitboard; 6] = [16, 8, 129, 36, 66, 65280];
const DEFAULT_BLACK_ARR: [Bitboard; 6] = [
    1152921504606846976,
    576460752303423488,
    9295429630892703744,
    2594073385365405696,
    4755801206503243776,
    71776119061217280,
];
const EMPTY_BITBOARD: Bitboard = 0;

pub fn debug_bitboard(bb: Bitboard) {
    let mut out: [u8; 64] = [0; 64];

    for sq in 0..64 {
        if bb & (1 << sq) != 0 {
            out[sq] = 1;
        }
    }

    out.reverse();

    for row in 0..8 {
        let slice = &out[row * 8..(row + 1) * 8];
        for bit in slice.iter().rev() {
            print!("{}", bit);
        }
        println!();
    }
}

pub struct Castling {
    pub white_king: bool,
    pub white_queen: bool,
    pub black_king: bool,
    pub black_queen: bool,
}

impl Castling {
    pub const WHITE_KING_MASK: Bitboard = 0x60;
    pub const WHITE_QUEEN_MASK: Bitboard = 0xe;
    pub const BLACK_KING_MASK: Bitboard = 0x6000000000000000;
    pub const BLACK_QUEEN_MASK: Bitboard = 0xe00000000000000;

    pub fn new() -> Castling {
        Castling {
            white_king: true,
            white_queen: true,
            black_king: true,
            black_queen: true,
        }
    }

    pub fn get_side(&self, side: Side) -> [bool; 2] {
        match side {
            White => [self.white_king, self.white_queen],
            Black => [self.black_king, self.black_queen],
        }
    }

    pub fn get_side_mask(side: Side) -> [Bitboard; 2] {
        match side {
            White => [Castling::WHITE_KING_MASK, Castling::WHITE_QUEEN_MASK],
            Black => [Castling::BLACK_KING_MASK, Castling::BLACK_QUEEN_MASK],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    White = 0,
    Black = 1,
}

impl Not for Side {
    type Output = Side;

    fn not(self) -> Side {
        match self {
            Side::White => Side::Black,
            Side::Black => Side::White,
        }
    }
}

#[derive(Debug)]
pub enum Piece {
    King = 0,
    Queen = 1,
    Rook = 2,
    Bishop = 3,
    Knight = 4,
    Pawn = 5,
}

impl Side {
    pub fn opp(s: Side) -> Side {
        match s {
            Side::White => Side::Black,
            Side::Black => Side::White,
        }
    }
}

pub struct Board {
    pub side: Side,
    pub castling: Castling,
    pub en_passant: Bitboard,
    pub white: [Bitboard; 6],
    pub black: [Bitboard; 6],
    pub white_bb: Bitboard,
    pub black_bb: Bitboard,
    pub half_move: usize,
}

#[derive(Debug)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub piece: Piece,
    pub promotion: Option<Piece>,
}

impl Move {
    pub const EMPTY: Move = Move {
        from: 0,
        to: 0,
        piece: Piece::King,
        promotion: None,
    };

    pub fn new(from: u8, to: u8, piece: Piece, promotion: Option<Piece>) -> Move {
        Move {
            from,
            to,
            piece,
            promotion,
        }
    }
}

pub struct MoveList {
    pub moves: [Move; 256],
    end: usize,
}

impl MoveList {
    pub fn new() -> MoveList {
        MoveList {
            moves: [Move::EMPTY; 256],
            end: 0,
        }
    }

    pub fn push(&mut self, m: Move) {
        if self.end == 256 {
            panic!("movelist at capacity, should not occur!");
        }
        self.moves[self.end] = m;
        self.end += 1;
    }

    pub fn get(&self, idx: usize) -> Option<&Move> {
        if idx >= self.end {
            return None;
        }
        Some(&self.moves[idx])
    }

    pub fn end(&self) -> usize {
        self.end
    }
}

impl Board {
    pub fn new() -> Board {
        Board {
            side: Side::White,
            castling: Castling::new(),
            en_passant: EMPTY_BITBOARD,
            white: DEFAULT_WHITE_ARR,
            black: DEFAULT_BLACK_ARR,
            white_bb: DEFAULT_WHITE,
            black_bb: DEFAULT_BLACK,
            half_move: 0,
        }
    }

    pub fn side_pieces(&self) -> [Bitboard; 6] {
        match self.side {
            White => self.white,
            Black => self.black,
        }
    }

    pub fn side_bb(&self) -> Bitboard {
        match self.side {
            Side::White => self.white_bb,
            Side::Black => self.black_bb,
        }
    }

    pub fn opp_bb(&self) -> Bitboard {
        match self.side {
            Side::White => self.black_bb,
            Side::Black => self.white_bb,
        }
    }
}
