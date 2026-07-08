use std::time::Duration;

use crate::{engine::Engine, magic_table::Rand};

pub(crate) mod board;
pub(crate) mod engine;
pub(crate) mod magic_table;
pub(crate) mod magics;

fn main() {
    let mut board = board::Board::new();
    let mut rand = Rand::new();
    loop {
        print!("\x1b[2J\x1b[H");
        println!("{}", board);
        let list = board.generate_pseudo_moves();
        let idx = rand.rand() as usize % list.end();
        board.make_move(list.get(idx).unwrap());
        std::thread::sleep(Duration::from_millis(500));
    }
}
