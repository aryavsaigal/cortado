use crate::board::{
    Bitboard, Board, Castling, Move, MoveList,
    Piece::{self, Bishop, King, Knight, Pawn, Queen, Rook},
    Side::{Black, White},
};

pub struct Engine {}

impl Engine {
    pub fn new() -> Engine {
        Engine {}
    }
}
