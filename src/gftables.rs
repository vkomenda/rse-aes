//! GF(2^8) exp/log/mul tables using the polynomial modulus 0x11b.

use super::table_aes::{EXP_TABLE, LOG_TABLE, MUL_TABLE};

pub(crate) const FIELD_SIZE: usize = 256;
pub(crate) const EXP_TABLE_SIZE: usize = FIELD_SIZE * 2 - 2;
pub(crate) const POLYNOMIAL_MODULUS: u8 = 0x1b;
pub(crate) const GENERATING_ELEMENT: u8 = 3;

type LogTable = [u8; FIELD_SIZE];
type ExpTable = [u8; EXP_TABLE_SIZE];
type MulTable = [[u8; FIELD_SIZE]; FIELD_SIZE];

/// Multiply `a` by `b` in the AES Galois field.
fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut r = 0u8;
    while b != 0 {
        if (b & 1) != 0 {
            r ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= POLYNOMIAL_MODULUS;
        }
        b >>= 1;
    }
    r
}

/// Compile-time table generator.
pub fn generate() -> (ExpTable, LogTable, MulTable) {
    let mut exp_table = [0u8; EXP_TABLE_SIZE];
    let mut log_table = [0u8; FIELD_SIZE];
    let mut mul_table = [[0u8; FIELD_SIZE]; FIELD_SIZE];

    let mut x: u8 = 1;
    // build exp[0..254], log for non-zero
    for (i, e) in exp_table.iter_mut().take(FIELD_SIZE - 1).enumerate() {
        *e = x;
        log_table[x as usize] = i as u8;
        x = gf_mul(x, GENERATING_ELEMENT);
    }

    // copy for overflow-friendly indexing
    for i in 0..FIELD_SIZE - 1 {
        exp_table[FIELD_SIZE - 1 + i] = exp_table[i];
    }

    for (i, a) in log_table.iter().enumerate() {
        for (j, b) in log_table.iter().enumerate() {
            mul_table[i][j] = exp_table[*a as usize + *b as usize];
        }
    }

    (exp_table, log_table, mul_table)
}

fn mul_composite(log_table: &LogTable, exp_table: &ExpTable, a: u8, b: u8) -> u8 {
    let la = log_table[a as usize] as usize;
    let lb = log_table[b as usize] as usize;
    exp_table[la + lb]
}

#[inline]
pub fn mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        0
    } else {
        MUL_TABLE[a as usize][b as usize]
    }
}

#[inline]
pub fn inv(a: u8) -> u8 {
    debug_assert!(a != 0);
    let la = LOG_TABLE[a as usize] as usize;
    EXP_TABLE[FIELD_SIZE - 1 - la]
}

#[inline]
pub fn pow(a: u8, n: usize) -> u8 {
    if n == 0 {
        1
    } else if a == 0 {
        0
    } else {
        let la = LOG_TABLE[a as usize] as usize;
        let idx = (la * n) % (FIELD_SIZE - 1);
        EXP_TABLE[idx]
    }
}

#[inline]
pub fn exp(a: usize) -> u8 {
    EXP_TABLE[a]
}
