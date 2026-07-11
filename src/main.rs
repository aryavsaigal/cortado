use std::time::{Duration, Instant};

use crate::{engine::Engine, magic_table::Rand};

pub(crate) mod board;
pub(crate) mod engine;
pub(crate) mod magic_table;
pub(crate) mod magics;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let start = Instant::now();
    let engine = Engine::new();
    engine.save_games(
        args[1].parse().unwrap(),
        args[2].parse().unwrap(),
        args[3].parse().unwrap(),
        args[4].parse().unwrap(),
        args[5].clone(),
    );
    let duration = start.elapsed();
    println!("Debug: time elpased {:?}", duration);
}
