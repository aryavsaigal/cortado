use crate::engine::Engine;

pub(crate) mod board;
pub(crate) mod engine;
pub(crate) mod magic_table;
pub(crate) mod magics;

fn main() {
    let mut board = board::Board::new();
    let moves = Engine::generate_pseudo_moves(&board);
    for i in 0..moves.end() {
        println!("{:?}", moves.get(i).unwrap());
    }
    println!("{}", moves.end());
}
