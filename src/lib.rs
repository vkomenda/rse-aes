mod matrix;

pub mod error;

use core::arch::x86_64::{self, __m512i};
use error::Error;
use lru::LruCache;
use matrix::Matrix;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::sync::Arc;

pub struct ReedSolomon {
    data_shard_count: usize,
    parity_shard_count: usize,
    total_shard_count: usize,
    encode_coeffs: Matrix,
    data_decode_coeffs_cache: Mutex<LruCache<Vec<usize>, Arc<Matrix>>>,
}

impl ReedSolomon {
    pub fn encode<T, U>(&self, mut shards: T) -> Result<(), Error>
    where
        T: AsRef<[U]> + AsMut<[U]>,
        U: AsRef<[u8]> + AsMut<[u8]>,
    {
        // TODO: checks

        let slices: &mut [U] = shards.as_mut();

        let (input, output) = slices.split_at_mut(self.data_shard_count);

        self.encode_sep(&*input, output)
    }

    pub fn encode_sep<T, U>(&self, input: &[T], output: &mut [U]) -> Result<(), Error>
    where
        T: AsRef<[u8]>,
        U: AsMut<[u8]>,
    {
        // TODO: checks or no checks?
        // do this at construction time: assert!(self.data_shard)_count > 0);
        let shard_len = input.len();
        // TODO: into Error
        assert_eq!(shard_len, output.len());

        let num_chunks = shard_len / 64;
        let tail_len = shard_len % 64;
        let tail_offset = num_chunks * 64;

        let encode_coeffs_parity_rows = self.get_encode_coeffs_parity_rows();

        for (coeff_row, p) in output.iter_mut().enumerate() {
            let p = p.as_mut();

            // Initialise the parity row `p`. Do that outside the loop below to avoid a conditional jump.
            let coeff0 = encode_coeffs_parity_rows[coeff_row][0];
            let d0 = input[0].as_ref();
            unsafe {
                mul_slice(coeff0, d0, p, num_chunks, tail_len, tail_offset);
            }

            // Sum up further codes on the same row `p`.
            for (coeff_col, d) in input.iter().enumerate().skip(1) {
                let coeff = encode_coeffs_parity_rows[coeff_row][coeff_col];
                let d = d.as_ref();
                unsafe {
                    mul_slice_add(coeff, d, p, num_chunks, tail_len, tail_offset);
                }
            }
        }

        Ok(())
    }

    fn get_encode_coeffs_parity_rows(&self) -> SmallVec<[&[u8]; 32]> {
        (self.data_shard_count..self.total_shard_count)
            .map(|i| self.encode_coeffs.get_row(i))
            .collect()
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn mul_slice(
    coeff: u8,
    input: &[u8],
    output: &mut [u8],
    num_chunks: usize,
    tail_len: usize,
    tail_offset: usize,
) {
    let vcoeff = x86_64::_mm512_set1_epi8(coeff as i8);
    for chunk in 0..num_chunks {
        let offset = chunk * 64;

        // load 64 bytes of data shard once for all parity rows to improve temporal locality
        let src = unsafe { x86_64::_mm512_loadu_si512(input.as_ptr().add(offset) as *const _) };

        // multiply GF(2^8) using GFNI affine table
        let prod = x86_64::_mm512_gf2p8mul_epi8(src, vcoeff);

        // store back
        unsafe { x86_64::_mm512_storeu_si512(output.as_mut_ptr().add(offset) as *mut _, prod) };
    }

    if tail_len > 0 {
        unsafe {
            mul_masked(
                input.as_ptr().add(tail_offset) as *const _,
                output.as_mut_ptr().add(tail_offset) as *mut _,
                vcoeff,
                tail_len,
            );
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn mul_slice_add(
    coeff: u8,
    input: &[u8],
    output: &mut [u8],
    num_chunks: usize,
    tail_len: usize,
    tail_offset: usize,
) {
    let vcoeff = x86_64::_mm512_set1_epi8(coeff as i8);
    for chunk in 0..num_chunks {
        let offset = chunk * 64;

        // load 64 bytes of data shard once for all parity rows to improve temporal locality
        let src = unsafe { x86_64::_mm512_loadu_si512(input.as_ptr().add(offset) as *const _) };

        // load current parity
        let dst = unsafe { x86_64::_mm512_loadu_si512(output.as_ptr().add(offset) as *const _) };

        // multiply GF(2^8) using GFNI affine table
        let prod = x86_64::_mm512_gf2p8mul_epi8(src, vcoeff);

        // accumulate into parity
        let sum = x86_64::_mm512_xor_si512(dst, prod);

        // store back
        unsafe { x86_64::_mm512_storeu_si512(output.as_mut_ptr().add(offset) as *mut _, sum) };
    }

    if tail_len > 0 {
        unsafe {
            mul_masked_add(
                input.as_ptr().add(tail_offset) as *const _,
                output.as_mut_ptr().add(tail_offset) as *mut _,
                vcoeff,
                tail_len,
            );
        }
    }
}

/// Initial multiplication discarding the contents of `dst`.
#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn mul_masked(src: *const i8, dst: *mut i8, coeff: __m512i, len: usize) {
    let mask = ((1u64 << len) - 1) as x86_64::__mmask64;
    let vsrc = unsafe { x86_64::_mm512_maskz_loadu_epi8(mask, src) };

    let vprod = x86_64::_mm512_gf2p8mul_epi8(vsrc, coeff);

    unsafe { x86_64::_mm512_mask_storeu_epi8(dst, mask, vprod) };
}

/// Follow-up multiplication building up on the contents of `dst`.
#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn mul_masked_add(src: *const i8, dst: *mut i8, coeff: __m512i, len: usize) {
    let mask = ((1u64 << len) - 1) as x86_64::__mmask64;
    let vsrc = unsafe { x86_64::_mm512_maskz_loadu_epi8(mask, src) };
    let vdst = unsafe { x86_64::_mm512_maskz_loadu_epi8(mask, dst) };

    let vprod = x86_64::_mm512_gf2p8mul_epi8(vsrc, coeff);
    let vsum = x86_64::_mm512_xor_si512(vdst, vprod);

    unsafe { x86_64::_mm512_mask_storeu_epi8(dst, mask, vsum) };
}
