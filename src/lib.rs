mod gftables;
mod matrix;
mod table_aes;

pub mod error;

use core::arch::x86_64::{self, __m512i};
use core::iter::{self, FromIterator};
use error::Error;
use lru::LruCache;
use matrix::Matrix;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::sync::Arc;

const DATA_DECODE_MATRIX_CACHE_CAPACITY: usize = 254;

/// Something which might hold a shard.
///
/// This trait is used in reconstruction, where some of the shards
/// may be unknown.
pub trait ReconstructShard {
    /// The size of the shard data; `None` if empty.
    fn len(&self) -> Option<usize>;

    /// Get a mutable reference to the shard data, returning `None` if uninitialized.
    fn get(&mut self) -> Option<&mut [u8]>;

    /// Get a mutable reference to the shard data, initializing it to the
    /// given length if it was `None`. Returns an error if initialization fails.
    fn get_or_initialize(&mut self, len: usize) -> Result<&mut [u8], Result<&mut [u8], Error>>;
}

impl<T: AsRef<[u8]> + AsMut<[u8]> + FromIterator<u8>> ReconstructShard for Option<T> {
    fn len(&self) -> Option<usize> {
        self.as_ref().map(|x| x.as_ref().len())
    }

    fn get(&mut self) -> Option<&mut [u8]> {
        self.as_mut().map(|x| x.as_mut())
    }

    fn get_or_initialize(&mut self, len: usize) -> Result<&mut [u8], Result<&mut [u8], Error>> {
        let is_some = self.is_some();
        let x = self
            .get_or_insert_with(|| iter::repeat_n(0, len).collect())
            .as_mut();

        if is_some { Ok(x) } else { Err(Ok(x)) }
    }
}

impl<T: AsRef<[u8]> + AsMut<[u8]>> ReconstructShard for (T, bool) {
    fn len(&self) -> Option<usize> {
        if !self.1 {
            None
        } else {
            Some(self.0.as_ref().len())
        }
    }

    fn get(&mut self) -> Option<&mut [u8]> {
        if !self.1 { None } else { Some(self.0.as_mut()) }
    }

    fn get_or_initialize(&mut self, len: usize) -> Result<&mut [u8], Result<&mut [u8], Error>> {
        let x = self.0.as_mut();
        if x.len() == len {
            if self.1 { Ok(x) } else { Err(Ok(x)) }
        } else {
            Err(Err(Error::IncorrectShardSize))
        }
    }
}

pub struct ReedSolomon {
    data_shard_count: usize,
    parity_shard_count: usize,
    total_shard_count: usize,
    encode_coeffs: Matrix,
    data_decode_coeffs_cache: Mutex<LruCache<Vec<usize>, Arc<Matrix>>>,
}

impl ReedSolomon {
    pub fn new(data_shards: usize, parity_shards: usize) -> Result<ReedSolomon, Error> {
        if data_shards == 0 {
            return Err(Error::TooFewDataShards);
        }
        if parity_shards == 0 {
            return Err(Error::TooFewParityShards);
        }
        if data_shards + parity_shards > F::ORDER {
            return Err(Error::TooManyShards);
        }

        let total_shards = data_shards + parity_shards;

        let encode_coeffs = Self::generate_encode_coeffs(data_shards, total_shards);

        Ok(ReedSolomon {
            data_shard_count: data_shards,
            parity_shard_count: parity_shards,
            total_shard_count: total_shards,
            encode_coeffs,
            data_decode_coeffs_cache: Mutex::new(LruCache::new(
                DATA_DECODE_MATRIX_CACHE_CAPACITY
                    .try_into()
                    .expect("non-0 constant; qed"),
            )),
        })
    }

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

    fn get_data_decode_coeffs(
        &self,
        valid_indices: &[usize],
        invalid_indices: &[usize],
    ) -> Arc<Matrix> {
        {
            let mut cache = self.data_decode_coeffs_cache.lock();
            if let Some(entry) = cache.get(invalid_indices) {
                return entry.clone();
            }
        }

        let mut inverted_decode_coeffs = Matrix::zero(self.data_shard_count, self.data_shard_count);
        for (inverted_decode_coeffs_row, &valid_index) in valid_indices.iter().enumerate() {
            for c in 0..self.data_shard_count {
                inverted_decode_coeffs.set(
                    inverted_decode_coeffs_row,
                    c,
                    self.encode_coeffs.get(valid_index, c),
                );
            }
        }
        let data_decode_coeffs = Arc::new(inverted_decode_coeffs.inv().unwrap());
        {
            let data_decode_coeffs = data_decode_coeffs.clone();
            let mut cache = self.data_decode_coeffs_cache.lock();
            cache.put(Vec::from(invalid_indices), data_decode_coeffs);
        }
        data_decode_coeffs
    }

    pub fn reconstruct<T: ReconstructShard>(&self, shards: &mut [T]) -> Result<(), Error> {
        // TODO: checks

        Ok(())
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
