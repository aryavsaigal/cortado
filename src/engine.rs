use crate::{
    board::{
        Bitboard, Board, Castling, Move, MoveList,
        Piece::{self, Bishop, King, Knight, Pawn, Queen, Rook},
        Side::{Black, White},
    },
    magic_table::generate_rook_magic_table,
};

use crate::magic_table::Magic;
use crate::magics::{BISHOP_MAGICS, ROOK_MAGICS};

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
pub struct Engine {}

impl Engine {
    pub fn new() -> Engine {
        Engine {}
    }

    pub fn generate_pseudo_moves(board: &Board) -> MoveList {
        let mut list = MoveList::new();
        Engine::generate_moves_king(board, &mut list);
        Engine::generate_moves_knight(board, &mut list);
        Engine::generate_moves_pawn(board, &mut list);
        Engine::get_queen_moves(board, &mut list);
        Engine::get_rook_moves(board, &mut list);
        Engine::get_bishop_moves(board, &mut list);
        list
    }

    pub fn generate_moves_knight(board: &Board, list: &mut MoveList) {
        let mut knights = board.side_pieces()[Knight as usize];
        while knights != 0 {
            let from = knights.trailing_zeros() as usize;
            let mut attacks = KNIGHT_MOVES[from] & !board.side_bb();
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                list.push(Move::new(from as u8, to as u8, Knight, None));
                attacks &= attacks - 1;
            }
            knights &= knights - 1;
        }
    }

    pub fn no_check(board: &Board, sq: usize) -> bool {
        true
    }

    pub fn generate_moves_king(board: &Board, list: &mut MoveList) {
        let king = board.side_pieces()[King as usize];
        if king != 0 {
            let from = king.trailing_zeros() as usize;
            let mut attacks = KING_MOVES[from] & !board.side_bb();
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                list.push(Move::new(from as u8, to as u8, King, None));
                attacks &= attacks - 1;
            }

            let castling = board.castling.get_side(board.side);
            let masks = Castling::get_side_mask(board.side);
            if castling[King as usize]
                && masks[King as usize] & (board.white_bb | board.black_bb) == 0
            {
                list.push(Move::new(from as u8, from as u8 + 2, King, None));
            }
            if castling[Queen as usize]
                && masks[Queen as usize] & (board.white_bb | board.black_bb) == 0
            {
                list.push(Move::new(from as u8, from as u8 - 3, King, None));
            }
        }
    }

    pub fn generate_moves_pawn(board: &Board, list: &mut MoveList) {
        let mut pawns = board.side_pieces()[Pawn as usize];
        while pawns != 0 {
            let from = pawns.trailing_zeros() as usize;
            let mut attacks = Engine::pawn_move_at_sq(board, from);
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                let promotion =
                    (board.side == White && to >= 56) || (board.side == Black && to <= 7);
                if promotion {
                    list.push(Move::new(from as u8, to as u8, Pawn, Some(Queen)));
                    list.push(Move::new(from as u8, to as u8, Pawn, Some(Rook)));
                    list.push(Move::new(from as u8, to as u8, Pawn, Some(Bishop)));
                    list.push(Move::new(from as u8, to as u8, Pawn, Some(Knight)));
                } else {
                    list.push(Move::new(from as u8, to as u8, Pawn, None));
                }
                attacks &= attacks - 1;
            }
            pawns &= pawns - 1;
        }
    }

    pub fn pawn_move_at_sq(board: &Board, sq: usize) -> Bitboard {
        let bb: Bitboard = 1 << sq;
        let forward = match board.side {
            White => {
                bb << 8
                    | (if sq / 8 == 1 && (bb << 8 & (board.white_bb | board.black_bb) == 0) {
                        bb << 16
                    } else {
                        0
                    })
            }
            Black => {
                bb >> 8
                    | (if sq / 8 == 6 && (bb >> 8 & (board.white_bb | board.black_bb) == 0) {
                        bb >> 16
                    } else {
                        0
                    })
            }
        } as Bitboard;

        (forward & !(board.white_bb | board.black_bb))
            | (PAWN_ATTACKS_LUT[board.side as usize][sq] & (board.opp_bb() | board.en_passant))
    }

    pub fn get_rook_moves(board: &Board, list: &mut MoveList) {
        let mut rooks = board.side_pieces()[Rook as usize];
        while rooks != 0 {
            let from = rooks.trailing_zeros() as usize;
            let mut attacks =
                ROOK_MAGICS[from].get(board.white_bb | board.black_bb) & !board.side_bb();
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                list.push(Move::new(from as u8, to as u8, Rook, None));
                attacks &= attacks - 1;
            }
            rooks &= rooks - 1;
        }
    }

    pub fn get_bishop_moves(board: &Board, list: &mut MoveList) {
        let mut bishop = board.side_pieces()[Bishop as usize];
        while bishop != 0 {
            let from = bishop.trailing_zeros() as usize;
            let mut attacks =
                BISHOP_MAGICS[from].get(board.white_bb | board.black_bb) & !board.side_bb();
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                list.push(Move::new(from as u8, to as u8, Bishop, None));
                attacks &= attacks - 1;
            }
            bishop &= bishop - 1;
        }
    }

    pub fn get_queen_moves(board: &Board, list: &mut MoveList) {
        let mut queen = board.side_pieces()[Queen as usize];
        while queen != 0 {
            let from = queen.trailing_zeros() as usize;
            let mut attacks = (BISHOP_MAGICS[from].get(board.white_bb | board.black_bb)
                & !board.side_bb())
                | (ROOK_MAGICS[from].get(board.white_bb | board.black_bb) & !board.side_bb());
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                list.push(Move::new(from as u8, to as u8, Queen, None));
                attacks &= attacks - 1;
            }
            queen &= queen - 1;
        }
    }
}
