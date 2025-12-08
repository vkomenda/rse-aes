mod gftables;
mod matrix;
mod table_aes;

#[cfg(test)]
mod tests;

pub mod error;

use core::arch::x86_64::{self, __m512i};
use core::iter::{self, FromIterator};
use error::Error;
use lru::LruCache;
use matrix::Matrix;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::sync::Arc;

const DECODE_MATRIX_CACHE_CAPACITY: usize = 254;
const DATA_OR_PARITY_SHARD_MAX_COUNT: usize = 32;

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

#[derive(Debug)]
pub struct ReedSolomon {
    data_shard_count: usize,
    parity_shard_count: usize,
    total_shard_count: usize,
    encode_coeffs: Matrix,
    decode_coeffs_cache: Mutex<LruCache<Vec<usize>, Arc<Matrix>>>,
}

impl ReedSolomon {
    pub fn new(data_shards: usize, parity_shards: usize) -> Result<ReedSolomon, Error> {
        if data_shards == 0 {
            return Err(Error::TooFewDataShards);
        }
        if parity_shards == 0 {
            return Err(Error::TooFewParityShards);
        }
        if data_shards + parity_shards > gftables::FIELD_SIZE {
            return Err(Error::TooManyShards);
        }

        let total_shards = data_shards + parity_shards;
        let encode_coeffs = Matrix::encode_coeffs(total_shards, data_shards);

        Ok(ReedSolomon {
            data_shard_count: data_shards,
            parity_shard_count: parity_shards,
            total_shard_count: total_shards,
            encode_coeffs,
            decode_coeffs_cache: Mutex::new(LruCache::new(
                DECODE_MATRIX_CACHE_CAPACITY
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
        // FIXME: checks

        let slices: &mut [U] = shards.as_mut();

        let (input, output) = slices.split_at_mut(self.data_shard_count);

        self.encode_sep(&*input, output)
    }

    pub fn encode_sep<T, U>(&self, input: &[T], output: &mut [U]) -> Result<(), Error>
    where
        T: AsRef<[u8]>,
        U: AsMut<[u8]>,
    {
        // FIXME: checks

        let encode_coeffs_parity_rows = self.get_encode_coeffs_parity_rows();
        self.apply_coeffs(&encode_coeffs_parity_rows, input, output);

        Ok(())
    }

    fn apply_coeffs<T, U>(&self, coeffs: &[&[u8]], input: &[T], output: &mut [U])
    where
        T: AsRef<[u8]>,
        U: AsMut<[u8]>,
    {
        let shard_len = input[0].as_ref().len();
        let num_chunks = shard_len / 64;
        let tail_len = shard_len % 64;
        let tail_offset = num_chunks * 64;

        for (coeff_row, out) in output.iter_mut().enumerate() {
            let out = out.as_mut();

            // Initialise the parity row `p`. Do that outside the loop below to avoid a conditional jump.
            let coeff0 = coeffs[coeff_row][0];
            let inp0 = input[0].as_ref();
            unsafe {
                mul_slice(coeff0, inp0, out, num_chunks, tail_len, tail_offset);
            }

            // Sum up further codes on the same row `p`.
            for (coeff_col, inp) in input.iter().enumerate().skip(1) {
                let coeff = coeffs[coeff_row][coeff_col];
                let inp = inp.as_ref();
                unsafe {
                    mul_slice_add(coeff, inp, out, num_chunks, tail_len, tail_offset);
                }
            }
        }
    }

    fn get_encode_coeffs_parity_rows(&self) -> SmallVec<[&[u8]; DATA_OR_PARITY_SHARD_MAX_COUNT]> {
        (self.data_shard_count..self.total_shard_count)
            .map(|i| self.encode_coeffs.row(i))
            .collect()
    }

    // //  A variant of the above.
    // fn parity_encode_coeffs(&self) -> SubmatrixRows {
    //     self.encode_coeffs
    //         .submatrix_rows(self.data_shard_count..self.total_shard_count)
    // }

    fn get_decode_coeffs(&self, valid_indices: &[usize]) -> Arc<Matrix> {
        // Exactly data_shard_count shards are required to recover the remaining shards.
        debug_assert!(valid_indices.len() == self.data_shard_count);

        {
            let mut cache = self.decode_coeffs_cache.lock();
            if let Some(entry) = cache.get(valid_indices) {
                return entry.clone();
            }
        }

        let mut encode_coeffs = Matrix::zero(self.data_shard_count, self.data_shard_count);
        for (encode_coeffs_row, &valid_index) in valid_indices.iter().enumerate() {
            for c in 0..self.data_shard_count {
                encode_coeffs.set(encode_coeffs_row, c, self.encode_coeffs.get(valid_index, c));
            }
        }
        let decode_coeffs = Arc::new(encode_coeffs.inv().unwrap());
        {
            let decode_coeffs = decode_coeffs.clone();
            let mut cache = self.decode_coeffs_cache.lock();
            cache.put(Vec::from(valid_indices), decode_coeffs);
        }
        decode_coeffs
    }

    pub fn reconstruct<T: ReconstructShard>(&self, shards: &mut [T]) -> Result<(), Error> {
        let check = CheckReconstructShards::new(shards)?;

        if check.non_empty_shard_count == self.total_shard_count {
            return Ok(());
        }

        if check.non_empty_shard_count < self.data_shard_count {
            return Err(Error::TooFewShardsPresent);
        }

        let shard_len = check.shard_len.expect("there are > 0 shards; qed");

        // FIXME: invalid_indices are not bound by data_shard_count, so the SmallVec should be larger.
        let mut valid_indices: SmallVec<[usize; 2 * DATA_OR_PARITY_SHARD_MAX_COUNT]> =
            SmallVec::with_capacity(self.total_shard_count);
        let mut invalid_indices: SmallVec<[usize; DATA_OR_PARITY_SHARD_MAX_COUNT]> =
            SmallVec::with_capacity(self.total_shard_count);
        let mut valid_shards: SmallVec<[&[u8]; 2 * DATA_OR_PARITY_SHARD_MAX_COUNT]> =
            SmallVec::with_capacity(self.total_shard_count);
        let mut missing_shards: SmallVec<[&mut [u8]; DATA_OR_PARITY_SHARD_MAX_COUNT]> =
            SmallVec::with_capacity(self.total_shard_count);

        for (shard_idx, shard) in shards.iter_mut().enumerate() {
            match shard.get_or_initialize(shard_len) {
                Ok(shard) => {
                    // Only `data_shard_count` shards are needed to recover remaining shards.
                    if valid_shards.len() < self.data_shard_count {
                        valid_shards.push(shard);
                        valid_indices.push(shard_idx);
                    }
                }
                Err(Err(_)) => {
                    // FIXME
                    // invalid_indices.push(shard_idx);
                }
                Err(Ok(shard)) => {
                    missing_shards.push(shard);
                    invalid_indices.push(shard_idx);
                }
            }
        }

        println!("{valid_indices:?}");
        println!("{invalid_indices:?}");

        let decode_coeffs = self.get_decode_coeffs(&valid_indices);

        // Decode coefficient matrix to recover the missing shards, both data and parity
        let missing_decode_coeffs: SmallVec<[&[u8]; DATA_OR_PARITY_SHARD_MAX_COUNT]> =
            invalid_indices
                .iter()
                .map(|i| decode_coeffs.row(*i))
                .collect();

        self.apply_coeffs(&missing_decode_coeffs, &valid_shards, &mut missing_shards);

        Ok(())
    }

    pub fn verify<T: std::fmt::Debug + AsRef<[u8]>>(&self, shards: &[T]) -> Result<bool, Error> {
        // FIXME: checks

        let shard_len = shards[0].as_ref().len();
        let data_shards = &shards[0..self.data_shard_count];
        let parity_shards_to_verify = &shards[self.data_shard_count..self.total_shard_count];
        let encode_coeffs_parity_rows = self.get_encode_coeffs_parity_rows();
        let mut new_parity_shards: SmallVec<[Vec<u8>; DATA_OR_PARITY_SHARD_MAX_COUNT]> =
            iter::repeat_n(vec![0; shard_len], self.parity_shard_count).collect();

        // assert_eq!(parity_shards_to_verify.len(), self.parity_shard_count);
        // assert_eq!(new_parity_shards.len(), self.parity_shard_count);
        // assert!(
        //     parity_shards_to_verify
        //         .iter()
        //         .zip(new_parity_shards.clone())
        //         .all(|(v, n)| v.as_ref().len() == n.len() && n.len() == shard_len)
        // );

        self.apply_coeffs(
            &encode_coeffs_parity_rows,
            data_shards,
            &mut new_parity_shards,
        );

        let parity_shards_match = new_parity_shards
            .iter_mut()
            .zip(parity_shards_to_verify)
            .all(|(new_shard, old_shard)| *new_shard == old_shard.as_ref());

        Ok(parity_shards_match)
    }
}

#[derive(Debug, Copy, Clone)]
struct CheckReconstructShards {
    non_empty_shard_count: usize,
    shard_len: Option<usize>,
}

impl CheckReconstructShards {
    fn new<T: ReconstructShard>(shards: &[T]) -> Result<CheckReconstructShards, Error> {
        let mut shard_len = None;
        let mut non_empty_shard_count = 0;
        for shard in shards {
            if let Some(len) = shard.len() {
                if len == 0 {
                    return Err(Error::EmptyShard);
                }
                non_empty_shard_count += 1;
                if let Some(prev_len) = shard_len
                    && len != prev_len
                {
                    return Err(Error::IncorrectShardSize);
                }

                shard_len = Some(len);
            }
        }

        Ok(Self {
            non_empty_shard_count,
            shard_len,
        })
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
