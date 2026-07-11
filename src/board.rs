use crate::board::GameState::{BlackCheckmate, FiftyMove, Normal, Stalemate, WhiteCheckmate};
use crate::board::Side::{Black, White};
use crate::magic_table::COL_MASKS;
use crate::magics::{BISHOP_MAGICS, ROOK_MAGICS};
use Piece::*;
use core::fmt;
use std::mem;
use std::ops::Not;

pub type Bitboard = u64;

const fn compute_pawn_attacks() -> [[Bitboard; 64]; 2] {
    let mut out: [[Bitboard; 64]; 2] = [[0; 64]; 2];
    let mut sq = 0;
    while sq < 64 {
        let sq_bb: Bitboard = 1 << sq;

        let file = sq % 8;
        let rank = sq / 8;

        let mut white_bb: Bitboard = 0;
        if rank != 7 {
            if file != 0 {
                white_bb |= sq_bb << 7;
            }

            if file != 7 {
                white_bb |= sq_bb << 9;
            }
        }
        out[White as usize][sq] = white_bb;

        let mut black_bb: Bitboard = 0;

        if rank != 0 {
            if file != 0 {
                black_bb |= sq_bb >> 9;
            }

            if file != 7 {
                black_bb |= sq_bb >> 7;
            }
        }
        out[Black as usize][sq] = black_bb;
        sq += 1;
    }
    out
}

const fn compute_knight_moves() -> [Bitboard; 64] {
    let mut out: [Bitboard; 64] = [0; 64];
    let mut sq = 0;

    const NOT_A_FILE: Bitboard = 0xfefefefefefefefe;
    const NOT_H_FILE: Bitboard = 0x7f7f7f7f7f7f7f7f;
    const NOT_AB_FILE: Bitboard = 0xfcfcfcfcfcfcfcfc;
    const NOT_GH_FILE: Bitboard = 0x3f3f3f3f3f3f3f3f;

    while sq < 64 {
        let bb: Bitboard = 1 << sq;

        out[sq] |= (bb & NOT_A_FILE) << 15;
        out[sq] |= (bb & NOT_H_FILE) << 17;
        out[sq] |= (bb & NOT_AB_FILE) << 6;
        out[sq] |= (bb & NOT_GH_FILE) << 10;
        out[sq] |= (bb & NOT_H_FILE) >> 15;
        out[sq] |= (bb & NOT_A_FILE) >> 17;
        out[sq] |= (bb & NOT_GH_FILE) >> 6;
        out[sq] |= (bb & NOT_AB_FILE) >> 10;

        sq += 1;
    }

    out
}

const fn compute_king_moves() -> [Bitboard; 64] {
    let mut out: [Bitboard; 64] = [0; 64];
    let mut sq = 0;

    const NOT_A_FILE: Bitboard = 0xfefefefefefefefe;
    const NOT_H_FILE: Bitboard = 0x7f7f7f7f7f7f7f7f;
    const NOT_1_RANK: Bitboard = 0xffffffffffffff00;
    const NOT_8_RANK: Bitboard = 0x00ffffffffffffff;

    while sq < 64 {
        let bb: Bitboard = 1 << sq;
        out[sq] |= bb << 8;
        out[sq] |= bb >> 8;
        out[sq] |= (bb & NOT_H_FILE) << 1;
        out[sq] |= (bb & NOT_A_FILE) >> 1;

        out[sq] |= (bb & (NOT_H_FILE & NOT_8_RANK)) << 9;
        out[sq] |= (bb & (NOT_A_FILE & NOT_8_RANK)) << 7;
        out[sq] |= (bb & (NOT_H_FILE & NOT_1_RANK)) >> 7;
        out[sq] |= (bb & (NOT_A_FILE & NOT_1_RANK)) >> 9;

        sq += 1;
    }

    out
}

pub const PAWN_ATTACKS_LUT: [[Bitboard; 64]; 2] = compute_pawn_attacks();
pub const KNIGHT_MOVES: [Bitboard; 64] = compute_knight_moves();
pub const KING_MOVES: [Bitboard; 64] = compute_king_moves();
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

// (from, to)
pub fn rook_castle_masks(sq: usize) -> (Bitboard, Bitboard) {
    match sq {
        2 => (1, 8),
        6 => (128, 32),
        58 => (0x100000000000000, 0x800000000000000),
        62 => (0x8000000000000000, 0x2000000000000000),
        _ => (0, 0),
    }
}

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct Castling(u8); // WhiteKing, WhiteQueen, BlackKing, BlackQueen

impl Castling {
    pub const WHITE_KING_MASK: Bitboard = 0x60;
    pub const WHITE_QUEEN_MASK: Bitboard = 0xe;
    pub const BLACK_KING_MASK: Bitboard = 0x6000000000000000;
    pub const BLACK_QUEEN_MASK: Bitboard = 0xe00000000000000;

    pub fn new() -> Castling {
        Castling(0b00001111)
    }

    pub fn get_side(&self, side: Side) -> [bool; 2] {
        let c = self.0;
        match side {
            White => [c & 0b1 != 0, c & 0b10 != 0],
            Black => [c & 0b100 != 0, c & 0b1000 != 0],
        }
    }

    pub fn get_side_mask(side: Side) -> [Bitboard; 2] {
        match side {
            White => [Castling::WHITE_KING_MASK, Castling::WHITE_QUEEN_MASK],
            Black => [Castling::BLACK_KING_MASK, Castling::BLACK_QUEEN_MASK],
        }
    }

    pub fn set_side_zero(&mut self, side: Side) {
        self.0 &= !match side {
            White => 0b11,
            Black => 0b1100,
        };
    }

    pub fn set_zero_from_rook(&mut self, from: u8) {
        self.0 &= !match from {
            0 => 0b10,
            7 => 0b1,
            56 => 0b1000,
            63 => 0b100,
            _ => 0b0,
        };
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

#[derive(Debug, Copy, Clone)]
pub enum Piece {
    King = 0,
    Queen = 1,
    Rook = 2,
    Bishop = 3,
    Knight = 4,
    Pawn = 5,
    Empty = 6,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Board {
    pub side: Side,
    pub castling: Castling,
    pub en_passant: Bitboard,
    pub white: [Bitboard; 6],
    pub black: [Bitboard; 6],
    pub white_bb: Bitboard,
    pub black_bb: Bitboard,
    pub half_move: usize,
    pub full_move: usize,
    pub state: GameState,
}

#[derive(Debug)]
pub struct UndoInfo {
    pub prev_castle: Castling,
    pub prev_en_passant: Bitboard,
    pub prev_half_move: usize,
}

impl UndoInfo {
    pub fn new(
        prev_castle: Castling,
        prev_en_passant: Bitboard,
        prev_half_move: usize,
    ) -> UndoInfo {
        UndoInfo {
            prev_castle,
            prev_en_passant,
            prev_half_move,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub piece: Piece,
    pub promotion: Option<Piece>,
    pub is_castle: bool,
    pub is_en_passant: bool,
    pub capture: Option<Piece>,
}

impl Move {
    pub const EMPTY: Move = Move {
        from: 0,
        to: 0,
        piece: Piece::King,
        promotion: None,
        is_castle: false,
        is_en_passant: true,
        capture: None,
    };

    pub fn new(from: u8, to: u8, piece: Piece, promotion: Option<Piece>) -> Move {
        Move {
            from,
            to,
            piece,
            promotion,
            is_castle: false,
            is_en_passant: false,
            capture: None,
        }
    }

    pub fn castle(mut self) -> Move {
        self.is_castle = true;
        self
    }

    pub fn en_passant(mut self, en_passant: bool) -> Move {
        self.is_en_passant = en_passant;
        self
    }

    pub fn set_capture(&mut self, piece: Piece) {
        self.capture = Some(piece);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GameState {
    Normal,
    WhiteCheckmate,
    BlackCheckmate,
    Stalemate,
    Insufficient,
    Threefold,
    FiftyMove,
}

pub struct MoveList {
    moves: [Move; 256],
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

    pub fn pop(&mut self) -> Move {
        self.end -= 1;
        mem::replace(&mut self.moves[self.end], Move::EMPTY)
    }

    pub fn get(&mut self, idx: usize) -> Option<&mut Move> {
        if idx >= self.end {
            return None;
        }
        Some(&mut self.moves[idx])
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
            full_move: 0,
            state: Normal,
        }
    }

    pub fn perft_debug(&mut self, depth: usize) {
        let moves = self.get_all_moves();
        if moves.is_none() {
            println!("Total: 0");
            return;
        }

        let mut moves = moves.unwrap();
        let result = self.perft(depth);

        println!("Total: {}", result);

        for i in 0..moves.end() {
            let m = moves.get(i).unwrap();
            let undo = self.make_move(m);
            let curr = self.perft(depth - 1);
            self.unmake_move(m, &undo);
            let promotion = match m.promotion {
                None => "",
                Some(p) => match p {
                    Queen => "q",
                    Rook => "r",
                    Bishop => "b",
                    Knight => "n",
                    _ => "",
                },
            };
            println!(
                "{}{}{}: {}",
                Board::sq_to_notation(m.from as usize),
                Board::sq_to_notation(m.to as usize),
                promotion,
                curr
            );
        }
    }

    pub fn perft(&mut self, depth: usize) -> u64 {
        if depth == 0 {
            return 1;
        }

        let moves = self.get_all_moves();

        if moves.is_none() {
            return 0;
        }

        let mut moves = moves.unwrap();

        if depth == 1 {
            return moves.end() as u64;
        }

        let mut nodes = 0;
        for i in 0..moves.end() {
            let m = moves.get(i).unwrap();
            let undo = self.make_move(m);
            let curr = self.perft(depth - 1);
            nodes += curr;
            self.unmake_move(m, &undo);
        }

        nodes
    }

    fn sq_to_notation(sq: usize) -> String {
        let row = sq / 8;
        let col = sq % 8;
        let col = match col {
            0 => 'a',
            1 => 'b',
            2 => 'c',
            3 => 'd',
            4 => 'e',
            5 => 'f',
            6 => 'g',
            7 => 'h',
            _ => panic!("invalid column"),
        };

        format!("{}{}", col, row + 1)
    }

    fn notation_to_sq(not: &str) -> usize {
        // assumes not is two chars
        let row = not.chars().nth(1).unwrap().to_digit(10).unwrap() - 1;
        let col = match not.chars().nth(0).unwrap().to_ascii_lowercase() {
            'a' => 0,
            'b' => 1,
            'c' => 2,
            'd' => 3,
            'e' => 4,
            'f' => 5,
            'g' => 6,
            'h' => 7,
            _ => panic!("invalid notation"),
        };

        (row * 8 + col) as usize
    }

    fn char_to_piece(piece: char) -> (Side, Piece) {
        let side = if piece.is_uppercase() { White } else { Black };
        let piece_enum = match piece.to_ascii_lowercase() {
            'p' => Pawn,
            'r' => Rook,
            'b' => Bishop,
            'n' => Knight,
            'q' => Queen,
            'k' => King,
            _ => panic!("invalid piece type"),
        };
        (side, piece_enum)
    }

    pub fn from_fen(fen: &str) -> Board {
        let mut white: [Bitboard; 6] = [0; 6];
        let mut black: [Bitboard; 6] = [0; 6];
        let mut castling = Castling(0);
        let mut en_passant = 0;
        let mut side = White;

        let fields: Vec<&str> = fen.split_ascii_whitespace().collect();
        let mut sq = 63;

        for line in fields[0].split('/') {
            for c in line.chars().rev() {
                if c == '/' {
                    continue;
                }

                if c.is_ascii_digit() {
                    sq -= c.to_digit(10).unwrap();
                    continue;
                }

                let (s, p) = Board::char_to_piece(c);
                let bb = &mut (match s {
                    White => &mut white,
                    Black => &mut black,
                });
                bb[p as usize] |= 1 << sq;
                sq -= 1;
            }
        }

        if fields[1] == "b" {
            side = Black;
        }

        if fields[2] != "-" {
            for c in fields[2].chars() {
                castling.0 |= match c {
                    'K' => 0b1,
                    'Q' => 0b10,
                    'k' => 0b100,
                    'q' => 0b1000,
                    _ => 0,
                };
            }
        }

        if fields[3] != "-" {
            en_passant |= 1 << Board::notation_to_sq(fields[3].trim());
        }

        let white_bb = white.iter().copied().reduce(|acc, e| acc | e).unwrap();
        let black_bb = black.iter().copied().reduce(|acc, e| acc | e).unwrap();

        Board {
            side,
            castling,
            en_passant,
            white,
            black,
            white_bb,
            black_bb,
            half_move: fields.get(4).unwrap_or(&"0").parse().unwrap_or_default(),
            full_move: fields.get(5).unwrap_or(&"0").parse().unwrap_or_default(),
            state: Normal,
        }
    }

    pub fn is_legal(&mut self, m: &mut Move) -> bool {
        if m.is_castle {
            let sq = self.side_pieces()[King as usize].trailing_zeros();
            if self.is_attacked(sq as usize) {
                return false;
            }
            let masks = Castling::get_side_mask(self.side);
            let mut mask = if m.to % 8 == 6 {
                masks[0]
            } else {
                masks[1] & !COL_MASKS[1]
            };
            while mask != 0 {
                let sq = mask.trailing_zeros();
                if self.is_attacked(sq as usize) {
                    return false;
                }
                mask &= mask - 1;
            }
        }

        let undo = self.make_move(m);
        self.side = !self.side;
        let sq = self.side_pieces()[King as usize].trailing_zeros();
        let illegal = self.is_attacked(sq as usize);
        self.side = !self.side;
        self.unmake_move(m, &undo);
        !illegal
    }

    pub fn get_all_moves(&mut self) -> Option<MoveList> {
        match self.state {
            Normal => {}
            _ => return None,
        };
        let moves = self.get_legal_moves();

        if moves.end() == 0 {
            if self.is_attacked(self.side_pieces()[King as usize].trailing_zeros() as usize) {
                self.state = match self.side {
                    White => WhiteCheckmate,
                    Black => BlackCheckmate,
                }
            } else {
                self.state = Stalemate;
            }
            return None;
        }

        Some(moves)
    }
    fn get_legal_moves(&mut self) -> MoveList {
        let mut list = self.generate_pseudo_moves();
        let mut legal_list = MoveList::new();

        for i in 0..list.end() {
            let m = list.get(i).unwrap();
            if self.is_legal(m) {
                legal_list.push(std::mem::replace(m, Move::EMPTY));
            }
        }

        legal_list
    }

    pub fn move_to_sqs(m: &str) -> (usize, usize) {
        (
            Board::notation_to_sq(&m[0..2]),
            Board::notation_to_sq(&m[2..]),
        )
    }

    pub fn unmake_move(&mut self, m: &Move, undo: &UndoInfo) {
        self.en_passant = undo.prev_en_passant;
        self.castling = undo.prev_castle;
        self.half_move = undo.prev_half_move;
        self.side = !self.side;
        self.state = Normal;

        if let Black = self.side {
            self.full_move -= 1;
        }

        let bitboard = &mut self.side_pieces_mut()[m.piece as usize];
        let from = 1 << m.from;
        let mut to = 1 << m.to;

        *bitboard &= !to;
        *bitboard |= from;
        let bitboard = self.side_bb_mut();
        *bitboard &= !to;
        *bitboard |= from;

        if let Some(promotion) = m.promotion {
            let bitboard = &mut self.side_pieces_mut()[promotion as usize];
            *bitboard &= !to;
        }

        if let Some(capture) = m.capture {
            if m.is_en_passant {
                match self.side {
                    White => to >>= 8,
                    Black => to <<= 8,
                }
            }
            *self.opp_bb_mut() |= to;
            self.opp_pieces_mut()[capture as usize] |= to;
        }

        if m.is_castle {
            let (from_rook, to_rook) = rook_castle_masks(m.to as usize);

            let bitboard = &mut self.side_pieces_mut()[Rook as usize];
            *bitboard &= !to_rook;
            *bitboard |= from_rook;
            let bitboard = self.side_bb_mut();
            *bitboard &= !to_rook;
            *bitboard |= from_rook;
        }
    }

    pub fn make_move(&mut self, m: &mut Move) -> UndoInfo {
        // function assumes that the move is legal
        let undo = UndoInfo::new(self.castling, self.en_passant, self.half_move);
        self.half_move += 1;
        self.en_passant = 0;

        let bitboard = &mut self.side_pieces_mut()[m.piece as usize];
        let from = 1 << m.from;
        let mut to = 1 << m.to;

        *bitboard &= !from;
        *bitboard |= to;
        let bitboard = self.side_bb_mut();
        *bitboard &= !from;
        *bitboard |= to;

        if let Some(promotion) = m.promotion {
            let bitboard = &mut self.side_pieces_mut()[m.piece as usize];
            *bitboard &= !to;
            let bitboard = &mut self.side_pieces_mut()[promotion as usize];
            *bitboard |= to;
        }

        if let King = m.piece {
            self.castling.set_side_zero(self.side);
        } else if let Rook = m.piece {
            self.castling.set_zero_from_rook(m.from);
        } else if let Pawn = m.piece {
            self.half_move = 0;
            if m.to.abs_diff(m.from) == 16 {
                self.en_passant |= 1 << ((m.to + m.from) / 2);
            }
        }

        if m.is_en_passant {
            match self.side {
                White => to >>= 8,
                Black => to <<= 8,
            };
        }
        if to & self.opp_bb() != 0 {
            for (i, bb) in self.opp_pieces_mut().iter_mut().enumerate() {
                if to & *bb == 0 {
                    continue;
                }
                m.set_capture(piece_from_index(i));
                *bb &= !to;
                self.half_move = 0;
                break;
            }
            *self.opp_bb_mut() &= !to;
            self.castling.set_zero_from_rook(m.to);
        }

        if m.is_castle {
            let (from_rook, to_rook) = rook_castle_masks(m.to as usize);

            let bitboard = &mut self.side_pieces_mut()[Rook as usize];
            *bitboard &= !from_rook;
            *bitboard |= to_rook;
            let bitboard = self.side_bb_mut();
            *bitboard &= !from_rook;
            *bitboard |= to_rook;

            self.castling.set_side_zero(self.side);
        }

        self.side = !self.side;
        if let White = self.side {
            self.full_move += 1;
        }

        if self.half_move >= 50 {
            self.state = FiftyMove;
        }

        undo
    }

    pub fn is_attacked(&self, sq: usize) -> bool {
        let enemy = self.opp_pieces();
        let occ = self.white_bb | self.black_bb;

        // King
        if KING_MOVES[sq] & enemy[King as usize] != 0 {
            return true;
        }

        // Knight
        if KNIGHT_MOVES[sq] & enemy[Knight as usize] != 0 {
            return true;
        }

        // Pawn
        if PAWN_ATTACKS_LUT[self.side as usize][sq] & enemy[Pawn as usize] != 0 {
            return true;
        }

        // Sliders
        if ROOK_MAGICS[sq].get(occ) & (enemy[Rook as usize] | enemy[Queen as usize]) != 0 {
            return true;
        }

        if BISHOP_MAGICS[sq].get(occ) & (enemy[Bishop as usize] | enemy[Queen as usize]) != 0 {
            return true;
        }
        false
    }

    pub fn side_pieces(&self) -> [Bitboard; 6] {
        match self.side {
            White => self.white,
            Black => self.black,
        }
    }

    pub fn opp_pieces(&self) -> [Bitboard; 6] {
        match self.side {
            Black => self.white,
            White => self.black,
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

    pub fn side_pieces_mut(&mut self) -> &mut [Bitboard; 6] {
        match self.side {
            White => &mut self.white,
            Black => &mut self.black,
        }
    }

    pub fn opp_pieces_mut(&mut self) -> &mut [Bitboard; 6] {
        match self.side {
            Black => &mut self.white,
            White => &mut self.black,
        }
    }
    pub fn side_bb_mut(&mut self) -> &mut Bitboard {
        match self.side {
            Side::White => &mut self.white_bb,
            Side::Black => &mut self.black_bb,
        }
    }

    pub fn opp_bb_mut(&mut self) -> &mut Bitboard {
        match self.side {
            Side::White => &mut self.black_bb,
            Side::Black => &mut self.white_bb,
        }
    }

    pub fn get_side(&self, side: Side) -> &[Bitboard; 6] {
        match side {
            White => &self.white,
            Black => &self.black,
        }
    }

    pub fn generate_pseudo_moves(&self) -> MoveList {
        let mut list = MoveList::new();
        self.generate_moves_pawn(&mut list);
        self.generate_moves_knight(&mut list);
        self.get_bishop_moves(&mut list);
        self.get_rook_moves(&mut list);
        self.get_queen_moves(&mut list);
        self.generate_moves_king(&mut list);
        list
    }

    pub fn generate_moves_knight(&self, list: &mut MoveList) {
        let mut knights = self.side_pieces()[Knight as usize];
        while knights != 0 {
            let from = knights.trailing_zeros() as usize;
            let mut attacks = KNIGHT_MOVES[from] & !self.side_bb();
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                list.push(Move::new(from as u8, to as u8, Knight, None));
                attacks &= attacks - 1;
            }
            knights &= knights - 1;
        }
    }

    pub fn generate_moves_king(&self, list: &mut MoveList) {
        let king = self.side_pieces()[King as usize];
        if king != 0 {
            let from = king.trailing_zeros() as usize;
            let mut attacks = KING_MOVES[from] & !self.side_bb();
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                list.push(Move::new(from as u8, to as u8, King, None));
                attacks &= attacks - 1;
            }

            let castling = self.castling.get_side(self.side);
            let masks = Castling::get_side_mask(self.side);
            if castling[King as usize]
                && masks[King as usize] & (self.white_bb | self.black_bb) == 0
            {
                list.push(Move::new(from as u8, from as u8 + 2, King, None).castle());
            }
            if castling[Queen as usize]
                && masks[Queen as usize] & (self.white_bb | self.black_bb) == 0
            {
                list.push(Move::new(from as u8, from as u8 - 2, King, None).castle());
            }
        }
    }

    pub fn generate_moves_pawn(&self, list: &mut MoveList) {
        let mut pawns = self.side_pieces()[Pawn as usize];
        while pawns != 0 {
            let from = pawns.trailing_zeros() as usize;
            let mut attacks = self.pawn_move_at_sq(from);
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                let promotion = (self.side == White && to >= 56) || (self.side == Black && to <= 7);
                let en_passant = (1 << to) & self.en_passant != 0;
                if promotion {
                    list.push(
                        Move::new(from as u8, to as u8, Pawn, Some(Queen)).en_passant(en_passant),
                    );
                    list.push(
                        Move::new(from as u8, to as u8, Pawn, Some(Rook)).en_passant(en_passant),
                    );
                    list.push(
                        Move::new(from as u8, to as u8, Pawn, Some(Bishop)).en_passant(en_passant),
                    );
                    list.push(
                        Move::new(from as u8, to as u8, Pawn, Some(Knight)).en_passant(en_passant),
                    );
                } else {
                    list.push(Move::new(from as u8, to as u8, Pawn, None).en_passant(en_passant));
                }
                attacks &= attacks - 1;
            }
            pawns &= pawns - 1;
        }
    }

    pub fn pawn_move_at_sq(&self, sq: usize) -> Bitboard {
        let bb: Bitboard = 1 << sq;
        let forward = match self.side {
            White => {
                bb << 8
                    | (if sq / 8 == 1 && (bb << 8 & (self.white_bb | self.black_bb) == 0) {
                        bb << 16
                    } else {
                        0
                    })
            }
            Black => {
                bb >> 8
                    | (if sq / 8 == 6 && (bb >> 8 & (self.white_bb | self.black_bb) == 0) {
                        bb >> 16
                    } else {
                        0
                    })
            }
        } as Bitboard;

        (forward & !(self.white_bb | self.black_bb))
            | (PAWN_ATTACKS_LUT[self.side as usize][sq] & (self.opp_bb() | self.en_passant))
    }

    pub fn get_rook_moves(&self, list: &mut MoveList) {
        let mut rooks = self.side_pieces()[Rook as usize];
        while rooks != 0 {
            let from = rooks.trailing_zeros() as usize;
            let mut attacks =
                ROOK_MAGICS[from].get(self.white_bb | self.black_bb) & !self.side_bb();
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                list.push(Move::new(from as u8, to as u8, Rook, None));
                attacks &= attacks - 1;
            }
            rooks &= rooks - 1;
        }
    }

    pub fn get_bishop_moves(&self, list: &mut MoveList) {
        let mut bishop = self.side_pieces()[Bishop as usize];
        while bishop != 0 {
            let from = bishop.trailing_zeros() as usize;
            let mut attacks =
                BISHOP_MAGICS[from].get(self.white_bb | self.black_bb) & !self.side_bb();
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                list.push(Move::new(from as u8, to as u8, Bishop, None));
                attacks &= attacks - 1;
            }
            bishop &= bishop - 1;
        }
    }

    pub fn get_queen_moves(&self, list: &mut MoveList) {
        let mut queen = self.side_pieces()[Queen as usize];
        while queen != 0 {
            let from = queen.trailing_zeros() as usize;
            let mut attacks = (BISHOP_MAGICS[from].get(self.white_bb | self.black_bb)
                & !self.side_bb())
                | (ROOK_MAGICS[from].get(self.white_bb | self.black_bb) & !self.side_bb());
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                list.push(Move::new(from as u8, to as u8, Queen, None));
                attacks &= attacks - 1;
            }
            queen &= queen - 1;
        }
    }
}

struct PieceDisplay(Piece, Side);
const DEFAULT_PIECE_DISPLAY: PieceDisplay = PieceDisplay(Empty, White);

impl fmt::Display for PieceDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = match (&self.1, &self.0) {
            (White, Pawn) => '♟',
            (White, Knight) => '♞',
            (White, Bishop) => '♝',
            (White, Rook) => '♜',
            (White, Queen) => '♛',
            (White, King) => '♚',
            (Black, Pawn) => '♟',
            (Black, Knight) => '♞',
            (Black, Bishop) => '♝',
            (Black, Rook) => '♜',
            (Black, Queen) => '♛',
            (Black, King) => '♚',
            _ => ' ',
        };
        let fg = match self.1 {
            White => "\x1b[38;2;255;255;255m",
            Black => "\x1b[38;2;20;20;20m",
        };
        write!(f, "{fg} {symbol} \x1b[0m")
    }
}

pub const fn piece_from_index(idx: usize) -> Piece {
    match idx {
        0 => King,
        1 => Queen,
        2 => Rook,
        3 => Bishop,
        4 => Knight,
        5 => Pawn,
        _ => panic!("should not occur (piece_from_index)"),
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let light = "\x1b[48;2;235;209;166m"; // light square bg
        let dark = "\x1b[48;2;165;117;80m"; // dark square bg
        let reset = "\x1b[0m";

        let mut pieces = [DEFAULT_PIECE_DISPLAY; 64];
        let pbb = self.white_bb | self.black_bb;

        for sq in 0..64 {
            let bb = 1 << sq;
            if bb & pbb == 0 {
                continue;
            }

            if bb & self.white_bb != 0 {
                for (i, piece_bb) in self.white.iter().enumerate() {
                    if bb & piece_bb != 0 {
                        pieces[sq] = PieceDisplay(piece_from_index(i), White);
                        break;
                    }
                }
            } else {
                for (i, piece_bb) in self.black.iter().enumerate() {
                    if bb & piece_bb != 0 {
                        pieces[sq] = PieceDisplay(piece_from_index(i), Black);
                        break;
                    }
                }
            }
        }

        for row in (0..8).rev() {
            for col in 0..8 {
                let colour = if (row + col) % 2 == 0 { light } else { dark };
                write!(f, "{}{}{}", colour, pieces[row * 8 + col], reset)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}
