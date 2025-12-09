use std::mem::MaybeUninit;

use super::gftables;
use smallvec::{Array, SmallVec};

/// This value should have a corresponding implementation of the `smallvec::Array` trait.
const DATA_ARRAY_SIZE: usize = 1024;

const ROWS_ARRAY_MAX_SIZE: usize = 64;

// const MAX_SUBMATRIX_ROWS: usize = 32;

// pub type SubmatrixRows = SmallVec<[&[u8]; MAX_SUBMATRIX_ROWS]>;

#[derive(Debug, Clone)]
pub struct Matrix {
    row_count: usize,
    col_count: usize,
    data: SmallVec<[u8; DATA_ARRAY_SIZE]>,
}

pub type RowRef<'a> = &'a [u8];
pub type RowRefArr<'a> = SmallVec<[RowRef<'a>; ROWS_ARRAY_MAX_SIZE]>;

pub type RowMut<'a> = &'a mut [u8];
pub type RowMutArr<'a> = SmallVec<[RowMut<'a>; ROWS_ARRAY_MAX_SIZE]>;

pub struct SubmatrixMut<'a> {
    row_count: usize,
    col_count: usize,
    rows: RowMutArr<'a>,
}

impl<'a> SubmatrixMut<'a> {
    pub fn new(row_count: usize, col_count: usize, rows: RowMutArr<'a>) -> Self {
        debug_assert_eq!(rows.len(), row_count);
        debug_assert!(rows.iter().all(|row| row.len() == col_count));

        Self {
            row_count,
            col_count,
            rows,
        }
    }

    pub fn make_identity(&mut self) {
        for (i, row) in self.rows.iter_mut().enumerate() {
            for (j, a) in row.iter_mut().enumerate() {
                *a = (i == j) as u8;
            }
        }
    }

    pub fn make_vandermonde(&mut self) {
        for i in 0..self.row_count {
            let row_gen = gftables::pow(gftables::GENERATING_ELEMENT, i + 1);
            for j in 0..self.col_count {
                let a = gftables::pow(row_gen, j);
                self.rows[i][j] = a;
            }
        }
    }
}

impl Matrix {
    pub unsafe fn new_uninitialized(row_count: usize, col_count: usize) -> Self {
        let total = row_count * col_count;

        let mut tmp: SmallVec<[MaybeUninit<u8>; DATA_ARRAY_SIZE]> = SmallVec::with_capacity(total);

        unsafe {
            tmp.set_len(total); // now it contains uninitialized bytes
        }

        // Transmute SmallVec<MaybeUninit<u8>> → SmallVec<u8>
        let data: SmallVec<[u8; DATA_ARRAY_SIZE]> = unsafe { std::mem::transmute(tmp) };

        Matrix {
            row_count,
            col_count,
            data,
        }
    }

    pub fn zero(row_count: usize, col_count: usize) -> Self {
        Self {
            row_count,
            col_count,
            data: SmallVec::from_vec(vec![0u8; row_count * col_count]),
        }
    }

    pub fn from_rows<'a>(rows: RowRefArr<'a>) -> Self {
        let row_count = rows.len();
        let col_count = rows[0].len();
        debug_assert!(rows.iter().all(|row| row.len() == col_count));

        let mut data: SmallVec<[u8; DATA_ARRAY_SIZE]> =
            SmallVec::with_capacity(row_count * col_count);

        for row in rows.iter() {
            data.extend_from_slice(row);
        }

        Self {
            row_count,
            col_count,
            data,
        }
    }

    /// Access a coefficient (row j, col i)
    pub fn get(&self, row: usize, col: usize) -> u8 {
        // assert!(row < self.row_count);
        // assert!(col < self.col_count);
        self.data[row * self.col_count + col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: u8) {
        // assert!(row < self.row_count);
        // assert!(col < self.col_count);
        self.data[row * self.col_count + col] = val;
    }

    pub fn row(&self, row: usize) -> &[u8] {
        let start = row * self.col_count;
        let end = start + self.col_count;

        &self.data[start..end]
    }

    pub fn rows<'a>(&'a self) -> RowRefArr<'a> {
        self.data.chunks(self.col_count).collect()
    }

    pub fn submatrix<R, C>(&self, row_range: R, col_range: C) -> Self
    where
        R: std::ops::RangeBounds<usize>,
        C: std::ops::RangeBounds<usize>,
    {
        use std::ops::Bound::*;

        let row_start = match row_range.start_bound() {
            Included(&x) => x,
            Excluded(&x) => x + 1,
            Unbounded => 0,
        };
        let row_end = match row_range.end_bound() {
            Included(&x) => x + 1,
            Excluded(&x) => x,
            Unbounded => self.row_count,
        };

        let col_start = match col_range.start_bound() {
            Included(&x) => x,
            Excluded(&x) => x + 1,
            Unbounded => 0,
        };
        let col_end = match col_range.end_bound() {
            Included(&x) => x + 1,
            Excluded(&x) => x,
            Unbounded => self.col_count,
        };

        let rows = row_end - row_start;
        let cols = col_end - col_start;

        let mut m = unsafe { Matrix::new_uninitialized(rows, cols) };
        for (r, i) in (row_start..row_end).zip(0..) {
            for (c, j) in (col_start..col_end).zip(0..) {
                m.set(i, j, self.get(r, c));
            }
        }

        m
    }

    pub fn submatrix_mut<'a, R, C>(&'a mut self, row_range: R, col_range: C) -> SubmatrixMut<'a>
    where
        R: std::ops::RangeBounds<usize>,
        C: std::ops::RangeBounds<usize>,
    {
        use std::ops::Bound::*;

        let row_start = match row_range.start_bound() {
            Included(&x) => x,
            Excluded(&x) => x + 1,
            Unbounded => 0,
        };
        let row_end = match row_range.end_bound() {
            Included(&x) => x + 1,
            Excluded(&x) => x,
            Unbounded => self.row_count,
        };

        let col_start = match col_range.start_bound() {
            Included(&x) => x,
            Excluded(&x) => x + 1,
            Unbounded => 0,
        };
        let col_end = match col_range.end_bound() {
            Included(&x) => x + 1,
            Excluded(&x) => x,
            Unbounded => self.col_count,
        };

        let row_count = row_end - row_start;
        let col_count = col_end - col_start;
        let base_ptr = self.data.as_mut_ptr();

        let rows: RowMutArr = (row_start..row_end)
            .map(|i| unsafe {
                let row_ptr = base_ptr.add(i * self.col_count + col_start);
                std::slice::from_raw_parts_mut(row_ptr, col_count)
            })
            .collect();

        SubmatrixMut::new(row_count, col_count, rows)
    }

    pub fn rows_mut<'a>(&'a mut self) -> RowMutArr<'a> {
        self.data.chunks_mut(self.col_count).collect()
    }

    // pub fn submatrix_rows<R>(&self, row_range: R) -> SubmatrixRows
    // where
    //     R: std::ops::RangeBounds<usize>,
    // {
    //     use std::ops::Bound::*;

    //     let row_start = match row_range.start_bound() {
    //         Included(&x) => x,
    //         Excluded(&x) => x + 1,
    //         Unbounded => 0,
    //     };
    //     let row_end = match row_range.end_bound() {
    //         Included(&x) => x + 1,
    //         Excluded(&x) => x,
    //         Unbounded => self.row_count,
    //     };

    //     debug_assert!(row_end <= self.row_count);

    //     let mut rows = SmallVec::with_capacity(row_end - row_start);

    //     for i in row_start..row_end {
    //         let begin = i * self.col_count;
    //         let end = begin + self.col_count;
    //         rows.push(&self.data[begin..end]);
    //     }

    //     rows
    // }

    /// Build a Vandermonde matrix over GF(2^8) AES.
    pub fn vandermonde(row_count: usize, col_count: usize) -> Self {
        let data = (0..row_count)
            .flat_map(|i| {
                let row_gen = gftables::pow(gftables::GENERATING_ELEMENT, i + 1);
                (0..col_count)
                    .map(|j| gftables::pow(row_gen, j))
                    .collect::<Vec<_>>()
            })
            .collect();

        Self {
            data,
            row_count,
            col_count,
        }
    }

    pub fn encode_coeffs(total_shards: usize, data_shards: usize) -> Self {
        let mut mat = unsafe { Self::new_uninitialized(total_shards, data_shards) };
        {
            let mut top_square = mat.submatrix_mut(0..data_shards, 0..data_shards);
            top_square.make_identity();
        }
        {
            let mut bottom_mat = mat.submatrix_mut(data_shards..total_shards, 0..data_shards);
            bottom_mat.make_vandermonde();
        }
        mat
    }

    /// Multiply two matrices over GF(2^8) using the tables.
    /// Self: (m x k), other: (k x n) → returns (m x n)
    pub fn mul(&self, other: &Matrix) -> Matrix {
        debug_assert_eq!(
            self.col_count, other.row_count,
            "Incompatible matrix shapes"
        );

        let mut out = unsafe { Matrix::new_uninitialized(self.row_count, other.col_count) };

        for i in 0..self.row_count {
            for j in 0..other.col_count {
                let mut acc: u8 = 0;
                for l in 0..self.col_count {
                    let a = self.get(i, l);
                    let b = other.get(l, j);
                    acc ^= gftables::mul(a, b); // addition in GF(2^8) = XOR
                }
                out.set(i, j, acc);
            }
        }

        out
    }

    // TODO: break down into augment and gaussian_elim
    /// Invert a square matrix over GF(2^8) using the tables.
    /// Returns None if the matrix is singular.
    pub fn inv(&self) -> Option<Matrix> {
        debug_assert_eq!(
            self.row_count, self.col_count,
            "Matrix must be square for inversion"
        );
        let n = self.row_count;

        // Start with augmented matrix [self | I]
        let mut aug = Matrix::zero(n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, n + i, 1); // identity
        }

        let mut rows = aug.rows_mut();

        // Gaussian elimination
        for i in 0..n {
            // Find pivot
            let mut pivot_row = None;
            for r in i..n {
                if rows[r][i] != 0 {
                    pivot_row = Some(r);
                    break;
                }
            }
            let pivot_row = pivot_row?;

            // Swap to top
            if pivot_row != i {
                rows.swap(i, pivot_row);
            }

            // Scale pivot to 1
            let inv_pivot = gftables::inv(rows[i][i]);
            for a in rows[i].iter_mut() {
                *a = gftables::mul(*a, inv_pivot);
            }

            // Eliminate other rows
            for r in 0..n {
                if r != i && rows[r][i] != 0 {
                    let factor = rows[r][i];
                    for j in 0..2 * n {
                        rows[r][j] ^= gftables::mul(factor, rows[i][j]);
                    }
                }
            }
        }

        // Extract right half as inverse
        let mut inv = unsafe { Matrix::new_uninitialized(n, n) };
        for i in 0..n {
            for j in 0..n {
                inv.set(i, j, rows[i][n + j]);
            }
        }

        Some(inv)
    }
}
