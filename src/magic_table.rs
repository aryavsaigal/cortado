pub type Bitboard = u64;

const SEED: u64 = 0x01010101010101;
const IMPOSSIBLE: Bitboard = 0xffffffffffffffff;

pub struct Rand {
    x: u64,
}

impl Rand {
    pub const fn new() -> Rand {
        Rand { x: SEED }
    }

    pub const fn rand(&mut self) -> u64 {
        let mut x = self.x;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 5;

        self.x = x;
        self.x
    }

    pub const fn set_seed(&mut self, seed: u64) {
        self.x = seed;
    }

    pub const fn sparse_rand(&mut self) -> u64 {
        self.rand() & self.rand() & self.rand()
    }
}

pub struct Magic<const N: usize> {
    pub mask: Bitboard,
    pub shift: u8,
    pub number: u64,
    pub entries: [Bitboard; N],
}

impl<const N: usize> Magic<N> {
    const EMPTY: Magic<N> = Magic {
        mask: 0,
        shift: 0,
        number: 0,
        entries: [IMPOSSIBLE; N],
    };
    pub const fn new(mask: Bitboard, shift: u8, number: u64, entries: [Bitboard; N]) -> Magic<N> {
        Magic {
            mask,
            shift,
            number,
            entries,
        }
    }

    pub const fn get(&self, mut occ: Bitboard) -> Bitboard {
        occ &= self.mask;
        self.entries[(occ.wrapping_mul(self.number) >> self.shift) as usize]
    }

    pub const fn get_index(&self, occ: Bitboard) -> usize {
        (occ.wrapping_mul(self.number) >> self.shift) as usize
    }
}

const fn row_masks() -> [Bitboard; 8] {
    let mut out: [Bitboard; 8] = [0; 8];
    let mut row = 0;
    while row < 8 {
        let mut bb = 0;
        let mut col = 0;
        while col < 8 {
            bb |= 1 << (row * 8 + col);
            col += 1;
        }
        out[row] = bb;
        row += 1;
    }
    out
}

const fn col_masks() -> [Bitboard; 8] {
    let mut out: [Bitboard; 8] = [0; 8];
    let mut col = 0;
    while col < 8 {
        let mut bb = 0;
        let mut row = 0;
        while row < 8 {
            bb |= 1 << (row * 8 + col);
            row += 1;
        }
        out[col] = bb;
        col += 1;
    }
    out
}

const fn rook_masks() -> [Bitboard; 64] {
    let mut out: [Bitboard; 64] = [0; 64];
    let mut sq = 0;
    while sq < 64 {
        // up
        let mut bb = 1 << sq;
        loop {
            bb <<= 8;

            if bb & ROW_MASKS[7] != 0 || bb == 0 {
                break;
            }
            out[sq] |= bb;
        }
        // down
        let mut bb = 1 << sq;
        loop {
            bb >>= 8;

            if bb & ROW_MASKS[0] != 0 || bb == 0 {
                break;
            }
            out[sq] |= bb;
        }
        // right
        let mut bb = (1 << sq) << 1;
        while bb & ROW_MASKS[sq / 8] != 0 {
            if bb & COL_MASKS[7] != 0 {
                break;
            }
            out[sq] |= bb;
            bb <<= 1;
        }
        // left
        let mut bb = (1 << sq) >> 1;
        while bb & ROW_MASKS[sq / 8] != 0 {
            if bb & COL_MASKS[0] != 0 {
                break;
            }
            out[sq] |= bb;
            bb >>= 1;
        }
        sq += 1;
    }
    out
}

const fn rook_slider_attack(occ: Bitboard, sq: usize) -> Bitboard {
    let mut out: Bitboard = 0;

    // up
    let mut bb = 1 << sq;
    loop {
        bb <<= 8;
        out |= bb;

        if bb & occ != 0 || bb == 0 {
            break;
        }
    }
    // down
    let mut bb = 1 << sq;
    loop {
        bb >>= 8;
        out |= bb;

        if bb & occ != 0 || bb == 0 {
            break;
        }
    }
    // right
    let mut bb = (1 << sq) << 1;
    while bb & ROW_MASKS[sq / 8] != 0 {
        out |= bb;
        if bb & occ != 0 {
            break;
        }
        bb <<= 1;
    }
    // left
    let mut bb = (1 << sq) >> 1;
    while bb & ROW_MASKS[sq / 8] != 0 {
        out |= bb;
        if bb & occ != 0 {
            break;
        }
        bb >>= 1;
    }

    out
}

pub const fn generate_rook_magic_table() -> [Magic<4096>; 64] {
    let mut rand = Rand::new();
    let mut out: [Magic<4096>; 64] = [Magic::EMPTY; 64];
    let mut sq = 0;

    while sq < 64 {
        let magic = &mut out[sq];
        magic.shift = (64 - ROOK_MASKS[sq].count_ones()) as u8;
        magic.mask = ROOK_MASKS[sq];
        'outer: loop {
            magic.entries = [IMPOSSIBLE; 4096];
            magic.number = rand.sparse_rand();
            let mut subset: Bitboard = 0;
            loop {
                let attack: Bitboard = rook_slider_attack(subset, sq);
                let idx = magic.get_index(subset);

                if magic.entries[idx] == IMPOSSIBLE {
                    magic.entries[idx] = attack;
                } else if magic.entries[idx] != attack {
                    continue 'outer;
                }

                subset = subset.wrapping_sub(magic.mask) & magic.mask;
                if subset == 0 {
                    break 'outer;
                }
            }
        }
        sq += 1;
    }

    out
}

pub const fn bishop_masks() -> [Bitboard; 64] {
    let mut out: [Bitboard; 64] = [0; 64];
    let mut sq = 0;
    while sq < 64 {
        // north-east
        let mut bb = 1 << sq;
        while bb & (ROW_MASKS[7] | COL_MASKS[7]) == 0 {
            bb <<= 9;
            if bb & (ROW_MASKS[7] | COL_MASKS[7]) != 0 || bb == 0 {
                break;
            }

            out[sq] |= bb;
        }
        // north-west
        let mut bb = 1 << sq;
        while bb & (ROW_MASKS[7] | COL_MASKS[0]) == 0 {
            bb <<= 7;
            if bb & (ROW_MASKS[7] | COL_MASKS[0]) != 0 || bb == 0 {
                break;
            }

            out[sq] |= bb;
        }

        // south-west
        let mut bb = 1 << sq;
        while bb & (ROW_MASKS[0] | COL_MASKS[0]) == 0 {
            bb >>= 9;
            if bb & (ROW_MASKS[0] | COL_MASKS[0]) != 0 || bb == 0 {
                break;
            }

            out[sq] |= bb;
        }
        // south-east
        let mut bb = 1 << sq;
        while bb & (ROW_MASKS[0] | COL_MASKS[7]) == 0 {
            bb >>= 7;
            if bb & (ROW_MASKS[0] | COL_MASKS[7]) != 0 || bb == 0 {
                break;
            }

            out[sq] |= bb;
        }
        sq += 1;
    }

    out
}

pub const fn generate_bishop_sliding(occ: Bitboard, sq: usize) -> Bitboard {
    let mut out: Bitboard = 0;
    // north-east
    let mut bb = 1 << sq;
    while bb & (ROW_MASKS[7] | COL_MASKS[7]) == 0 {
        bb <<= 9;
        out |= bb;
        if bb & occ != 0 || bb == 0 {
            break;
        }
    }
    // north-west
    let mut bb = 1 << sq;
    while bb & (ROW_MASKS[7] | COL_MASKS[0]) == 0 {
        bb <<= 7;
        out |= bb;
        if bb & occ != 0 || bb == 0 {
            break;
        }
    }

    // south-west
    let mut bb = 1 << sq;
    while bb & (ROW_MASKS[0] | COL_MASKS[0]) == 0 {
        bb >>= 9;
        out |= bb;
        if bb & occ != 0 || bb == 0 {
            break;
        }
    }
    // south-east
    let mut bb = 1 << sq;
    while bb & (ROW_MASKS[0] | COL_MASKS[7]) == 0 {
        bb >>= 7;
        out |= bb;
        if bb & occ != 0 || bb == 0 {
            break;
        }
    }

    out
}

pub const fn generate_bishop_magic_table() -> [Magic<512>; 64] {
    let mut rand = Rand::new();
    let mut out: [Magic<512>; 64] = [Magic::EMPTY; 64];
    let mut sq = 0;

    while sq < 64 {
        let magic = &mut out[sq];
        magic.shift = (64 - BISHOP_MASKS[sq].count_ones()) as u8;
        magic.mask = BISHOP_MASKS[sq];
        'outer: loop {
            magic.entries = [IMPOSSIBLE; 512];
            magic.number = rand.sparse_rand();
            let mut subset: Bitboard = 0;
            loop {
                let attack: Bitboard = generate_bishop_sliding(subset, sq);
                let idx = magic.get_index(subset);

                if magic.entries[idx] == IMPOSSIBLE {
                    magic.entries[idx] = attack;
                } else if magic.entries[idx] != attack {
                    continue 'outer;
                }

                subset = subset.wrapping_sub(magic.mask) & magic.mask;
                if subset == 0 {
                    break 'outer;
                }
            }
        }
        sq += 1;
    }

    out
}
pub const ROW_MASKS: [Bitboard; 8] = row_masks();
pub const COL_MASKS: [Bitboard; 8] = col_masks();
pub const BISHOP_MASKS: [Bitboard; 64] = bishop_masks();
pub const ROOK_MASKS: [Bitboard; 64] = rook_masks();
