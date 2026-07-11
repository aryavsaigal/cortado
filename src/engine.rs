use std::{
    fs::File,
    io::{BufWriter, Write},
};

use crate::{
    board::{
        Bitboard, Board, Castling,
        GameState::{
            self, BlackCheckmate, FiftyMove, Insufficient, Normal, Stalemate, Threefold,
            WhiteCheckmate,
        },
        Move, MoveList,
        Piece::{self, Bishop, Empty, King, Knight, Pawn, Queen, Rook},
        Side::{self, Black, White},
        piece_from_index,
    },
    magic_table::{COL_MASKS, Rand},
};
const NEG_INF: isize = -1_000_000_000;
const POS_INF: isize = 1_000_000_000;

struct Matrix<const M: usize, const N: usize> {
    data: [[f32; N]; M],
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    pub fn zeroes() -> Self {
        Matrix { data: [[0.; N]; M] }
    }

    pub fn from_array(data: [[f32; N]; M]) -> Self {
        Matrix { data }
    }

    pub fn from_bytes(bytes: &[u8], offset: &mut usize) -> Self {
        let size = M * N * 4;
        let slice = &bytes[*offset..*offset + size];
        let mut data = [[0.; N]; M];

        for i in 0..M {
            for j in 0..N {
                let idx = (i * N + j) * 4;
                data[i][j] = f32::from_le_bytes(slice[idx..idx + 4].try_into().unwrap());
            }
        }
        *offset += size;
        Matrix { data }
    }

    pub fn scale(&mut self, factor: f32) {
        for i in 0..M {
            for j in 0..N {
                self.data[i][j] *= factor;
            }
        }
    }
    pub fn matmul<const K: usize>(&self, other: &Matrix<N, K>) -> Matrix<M, K> {
        let mut out = Matrix::<M, K>::zeroes();
        for i in 0..M {
            for c in 0..N {
                let a = self.data[i][c];
                for j in 0..K {
                    out.data[i][j] += a * other.data[c][j];
                }
            }
        }
        out
    }

    pub fn transpose(&self) -> Matrix<N, M> {
        let mut out = Matrix::<N, M>::zeroes();
        for i in 0..N {
            for j in 0..M {
                out.data[i][j] = self.data[j][i]
            }
        }
        out
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut out = Self::zeroes();
        for i in 0..M {
            for j in 0..N {
                out.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        out
    }

    pub fn relu(&self) -> Self {
        let mut out = Self::zeroes();
        for i in 0..M {
            for j in 0..N {
                out.data[i][j] = self.data[i][j].max(0.0);
            }
        }
        out
    }

    pub fn softmax_by_row(&self) -> Self {
        let mut out = Self::zeroes();
        for i in 0..M {
            let max = self.data[i].iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0.0;
            for j in 0..N {
                out.data[i][j] = (self.data[i][j] - max).exp();
                sum += out.data[i][j];
            }
            for j in 0..N {
                out.data[i][j] /= sum;
            }
        }
        out
    }

    pub fn mean(&self) -> Matrix<1, N> {
        let mut out = Matrix::<1, N>::zeroes();
        for i in 0..M {
            for j in 0..N {
                out.data[0][j] += self.data[i][j];
            }
        }
        for j in 0..N {
            out.data[0][j] /= M as f32;
        }
        out
    }
}

const N_DIM: usize = 32;

struct Linear<const IN: usize, const OUT: usize> {
    weights: Matrix<IN, OUT>,
    bias: Matrix<1, OUT>,
}

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    pub fn forward<const N: usize>(&self, x: &Matrix<N, IN>) -> Matrix<N, OUT> {
        let mut out = x.matmul(&self.weights);
        for i in 0..N {
            for j in 0..OUT {
                out.data[i][j] += self.bias.data[0][j];
            }
        }
        out
    }
}

struct Head {
    query: Matrix<N_DIM, N_DIM>,
    key: Matrix<N_DIM, N_DIM>,
    value: Matrix<N_DIM, N_DIM>,
    mlp_1: Linear<N_DIM, { N_DIM * 4 }>,
    mlp_2: Linear<{ N_DIM * 4 }, N_DIM>,
}

impl Head {
    pub fn forward(&self, x: Matrix<64, N_DIM>) -> Matrix<64, N_DIM> {
        let query = x.matmul(&self.query);
        let key = x.matmul(&self.key);
        let value = x.matmul(&self.value);

        let mut attn = query.matmul(&key.transpose());
        attn.scale(1.0 / (N_DIM as f32).sqrt());
        let attn = attn.softmax_by_row();
        let out = attn.matmul(&value).add(&x);
        let mlp = self.mlp_1.forward(&out);
        let mlp = self.mlp_2.forward(&mlp.relu());
        out.add(&mlp)
    }
}

pub struct Model {
    positional_embd: Matrix<64, N_DIM>,
    stm_embd: Matrix<2, N_DIM>,
    piece_embd: Matrix<13, N_DIM>,
    compress_1: Linear<N_DIM, { N_DIM / 2 }>,
    compress_2: Linear<{ N_DIM / 2 }, 1>,
    head: Head,
}

impl Model {
    pub fn load_model(path: &str) -> Self {
        let bytes = std::fs::read(path).unwrap();
        let mut offset = 0;
        Model {
            positional_embd: Matrix::from_bytes(&bytes, &mut offset),
            stm_embd: Matrix::from_bytes(&bytes, &mut offset),
            head: Head {
                query: Matrix::from_bytes(&bytes, &mut offset),
                key: Matrix::from_bytes(&bytes, &mut offset),
                value: Matrix::from_bytes(&bytes, &mut offset),
                mlp_1: Linear {
                    weights: Matrix::from_bytes(&bytes, &mut offset),
                    bias: Matrix::from_bytes(&bytes, &mut offset),
                },
                mlp_2: Linear {
                    weights: Matrix::from_bytes(&bytes, &mut offset),
                    bias: Matrix::from_bytes(&bytes, &mut offset),
                },
            },
            piece_embd: Matrix::from_bytes(&bytes, &mut offset),
            compress_1: Linear {
                weights: Matrix::from_bytes(&bytes, &mut offset),
                bias: Matrix::from_bytes(&bytes, &mut offset),
            },

            compress_2: Linear {
                weights: Matrix::from_bytes(&bytes, &mut offset),
                bias: Matrix::from_bytes(&bytes, &mut offset),
            },
        }
    }

    pub fn forward(&self, board: [u8; 64], stm: usize) -> f32 {
        let mut piece_stm_matrix = Matrix::<64, N_DIM>::zeroes();
        let stm_arr = self.stm_embd.data[stm];
        for sq in 0..64 {
            let row = board[sq] as usize;
            let piece = self.piece_embd.data[row];
            for col in 0..N_DIM {
                piece_stm_matrix.data[sq][col] = stm_arr[col] + piece[col];
            }
        }

        let out = self.positional_embd.add(&piece_stm_matrix);
        let out = self.head.forward(out);

        let out = self.compress_1.forward(&out.mean());
        self.compress_2.forward(&out.relu()).data[0][0]
    }
}

#[repr(C)]
pub struct GameSample {
    pub board: [u8; 64],
    pub stm: Side,
}

impl GameSample {
    pub fn new(board: [u8; 64], stm: Side) -> GameSample {
        GameSample { board, stm }
    }
}

pub struct Game {
    games: Vec<GameSample>,
    result: i8,
}

impl Game {
    pub fn new(games: Vec<GameSample>, result: i8) -> Game {
        Game { games, result }
    }
}

pub struct Engine {
    pub model: Model,
}

impl Engine {
    pub fn new() -> Engine {
        Engine {
            model: Model::load_model("./weights.bin"),
        }
    }

    pub fn piece_value(p: Piece) -> isize {
        match p {
            King => 0,
            Queen => 9,
            Rook => 5,
            Bishop | Knight => 3,
            Pawn => 1,
            Empty => 0,
        }
    }

    pub fn mvv_lva(m: &Move) -> isize {
        match m.capture {
            Some(v) => 10 * Engine::piece_value(v) - Engine::piece_value(m.piece),
            None => 0,
        }
    }

    pub fn outcome_to_score(state: GameState) -> i8 {
        match state {
            Normal => panic!("invalid outcome"),
            FiftyMove | Insufficient | Threefold | Stalemate => 0,
            WhiteCheckmate => -1,
            BlackCheckmate => 1,
        }
    }

    pub fn board_to_game(board: &Board) -> [u8; 64] {
        let mut out = [0; 64];
        for s in [Side::White, Side::Black] {
            for (i, mut bb) in board.get_side(s).iter().copied().enumerate() {
                while bb != 0 {
                    let sq = bb.trailing_zeros() as usize;
                    out[sq] = 1 + (s as u8 * 6) + i as u8;
                    bb &= bb - 1;
                }
            }
        }
        out
    }

    fn generate_game(&self, random_moves: usize, depth: usize, rand: &mut Rand) -> Game {
        let mut games = Vec::new();
        let mut board = Board::new();

        for _ in 0..random_moves {
            let moves = board.get_all_moves();
            if moves.is_none() {
                return Game::new(games, Engine::outcome_to_score(board.state));
            }
            let mut moves = moves.unwrap();
            let rand_idx = rand.rand() % moves.end() as u64;
            board.make_move(moves.get(rand_idx as usize).unwrap());
            games.push(GameSample::new(Engine::board_to_game(&board), board.side));
        }

        loop {
            let m = self.search(&mut board, depth);
            if m.is_none() {
                break;
            }
            board.make_move(&mut m.unwrap());
            games.push(GameSample::new(Engine::board_to_game(&board), board.side));
        }
        Game::new(games, Engine::outcome_to_score(board.state))
    }

    pub fn save_games(
        &self,
        count: usize,
        seed: u64,
        depth: usize,
        random_moves: usize,
        path: String,
    ) {
        let mut rand = Rand::new();
        rand.set_seed(seed);

        let file = File::create(path).unwrap();
        let mut writer = BufWriter::new(file);
        writer.write_all(&(count as u32).to_le_bytes()).unwrap();
        for _ in 0..count {
            let game = self.generate_game(random_moves, depth, &mut rand);
            writer
                .write_all(&(game.games.len() as u32).to_le_bytes())
                .unwrap();
            writer.write_all(&(game.result).to_le_bytes()).unwrap();
            for sample in &game.games {
                writer.write_all(&sample.board).unwrap();
                writer.write_all(&[sample.stm as u8]).unwrap();
            }
        }
    }

    pub fn negamax(&self, board: &mut Board, depth: usize, mut alpha: isize, beta: isize) -> isize {
        if depth == 0 {
            return self.score(board) * Engine::multiply(board.side);
        }

        let mut pseudo = board.generate_pseudo_moves();
        let mut found_legal = false;
        let mut value = NEG_INF;
        let mut order: [usize; 256] = std::array::from_fn(|i| i);
        order[..pseudo.end()]
            .sort_by_key(|&i| std::cmp::Reverse(Engine::mvv_lva(pseudo.get(i).unwrap())));
        'outer: for &i in &order[..pseudo.end()] {
            let m = pseudo.get(i).unwrap();
            if m.is_castle {
                let sq = board.side_pieces()[King as usize].trailing_zeros();
                if board.is_attacked(sq as usize) {
                    continue 'outer;
                }
                let masks = Castling::get_side_mask(board.side);
                let mut mask = if m.to % 8 == 6 {
                    masks[0]
                } else {
                    masks[1] & !COL_MASKS[1]
                };
                while mask != 0 {
                    let sq = mask.trailing_zeros();
                    if board.is_attacked(sq as usize) {
                        continue 'outer;
                    }
                    mask &= mask - 1;
                }
            }
            let undo = board.make_move(m);
            board.side = !board.side;
            let sq = board.side_pieces()[King as usize].trailing_zeros();
            let illegal = board.is_attacked(sq as usize);
            board.side = !board.side;
            if illegal {
                board.unmake_move(m, &undo);
                continue;
            }
            found_legal = true;
            value = value.max(-self.negamax(board, depth - 1, -beta, -alpha));
            board.unmake_move(m, &undo);
            alpha = alpha.max(value);
            if alpha >= beta {
                break;
            }
        }

        if !found_legal {
            if board.is_attacked(board.side_pieces()[King as usize].trailing_zeros() as usize) {
                board.state = match board.side {
                    White => WhiteCheckmate,
                    Black => BlackCheckmate,
                }
            } else {
                board.state = Stalemate;
            }
            return self.score(board) * Engine::multiply(board.side);
        }

        value
    }

    pub fn search(&self, board: &mut Board, depth: usize) -> Option<Move> {
        if depth == 0 {
            return None;
        }

        let moves = board.get_all_moves();
        moves.as_ref()?;
        let mut moves = moves.unwrap();

        let mut order: [usize; 256] = std::array::from_fn(|i| i);
        order[..moves.end()]
            .sort_by_key(|&i| std::cmp::Reverse(Engine::mvv_lva(moves.get(i).unwrap())));

        let mut best_move = None;
        let mut value = NEG_INF;

        let mut alpha = NEG_INF;
        let beta = POS_INF;
        for &i in &order[..moves.end()] {
            let m = moves.get(i).unwrap();
            let undo = board.make_move(m);
            let curr_val = -self.negamax(board, depth - 1, -beta, -alpha);
            board.unmake_move(m, &undo);
            if curr_val >= value {
                value = curr_val;
                best_move = Some(std::mem::replace(m, Move::EMPTY));
            }

            if curr_val > alpha {
                alpha = curr_val;
            }
        }

        best_move
    }

    pub fn multiply(side: Side) -> isize {
        match side {
            Side::White => 1,
            Side::Black => -1,
        }
    }

    pub fn score(&self, board: &Board) -> isize {
        const SCALE_FACTOR: f32 = 1000.;
        match board.state {
            Normal => {}
            FiftyMove | Insufficient | Threefold | Stalemate => return 0,
            WhiteCheckmate => return NEG_INF,
            BlackCheckmate => return POS_INF,
        };

        let b = Engine::board_to_game(board);
        (self.model.forward(b, board.side as usize) * SCALE_FACTOR) as isize
    }
}
