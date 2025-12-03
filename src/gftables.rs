/// GF(2^8) exp/log tables using polynomial 0x11b.
/// Ensure the polynomial and generator match what you used to build the Vandermonde matrix.
use super::table_aes::{EXP_TABLE, LOG_TABLE};

pub(crate) const FIELD_SIZE: usize = 256;
pub(crate) const EXP_TABLE_SIZE: usize = FIELD_SIZE * 2 - 2;
pub(crate) const POLYNOMIAL_MODULUS: u8 = 0x1b;
pub(crate) const GENERATING_ELEMENT: u8 = 3;

type LogTable = [u8; FIELD_SIZE];
type ExpTable = [u8; EXP_TABLE_SIZE];

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

pub fn generate() -> (ExpTable, LogTable) {
    let mut exp = [0u8; EXP_TABLE_SIZE];
    let mut log = [0u8; FIELD_SIZE];

    let mut x: u8 = 1;
    // build exp[0..254], log for non-zero
    for (i, e) in exp.iter_mut().take(FIELD_SIZE - 1).enumerate() {
        *e = x;
        log[x as usize] = i as u8;
        x = gf_mul(x, GENERATING_ELEMENT);
    }

    // copy for overflow-friendly indexing
    for i in 0..FIELD_SIZE - 1 {
        exp[FIELD_SIZE - 1 + i] = exp[i];
    }
    (exp, log)
}

#[inline]
pub fn mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        0
    } else {
        // FIXME: use MUL_TABLE
        let la = LOG_TABLE[a as usize] as usize;
        let lb = LOG_TABLE[b as usize] as usize;
        EXP_TABLE[la + lb]
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
