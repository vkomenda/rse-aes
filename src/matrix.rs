use super::gftables;
use smallvec::SmallVec;

/// This value should have a corresponding implementation of the `smallvec::Array` trait.
const DATA_ARRAY_SIZE: usize = 1024;

#[derive(Debug, Clone)]
pub(crate) struct Matrix {
    row_count: usize,
    col_count: usize,
    data: SmallVec<[u8; DATA_ARRAY_SIZE]>,
}

impl Matrix {
    pub(crate) fn zero(row_count: usize, col_count: usize) -> Self {
        Self {
            row_count,
            col_count,
            data: SmallVec::from_vec(vec![0u8; row_count * col_count]),
        }
    }

    /// Access a coefficient (row j, col i)
    pub(crate) fn get(&self, row: usize, col: usize) -> u8 {
        // assert!(row < self.row_count);
        // assert!(col < self.col_count);
        self.data[row * self.col_count + col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: u8) {
        // assert!(row < self.row_count);
        // assert!(col < self.col_count);
        self.data[row * self.col_count + col] = val;
    }

    pub(crate) fn get_row(&self, row: usize) -> &[u8] {
        let start = row * self.col_count;
        let end = start + self.col_count;

        &self.data[start..end]
    }

    /// Build a Vandermonde matrix over GF(2^8) AES.
    pub fn vandermonde_with_tables(row_count: usize, col_count: usize) -> Self {
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

    /// Multiply two matrices over GF(2^8) using the tables.
    /// Self: (m x k), other: (k x n) → returns (m x n)
    pub fn mul(&self, other: &Matrix) -> Matrix {
        debug_assert_eq!(
            self.row_count, other.col_count,
            "Incompatible matrix shapes"
        );

        let mut out = Matrix::zero(self.row_count, other.col_count);

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
        // TODO: convert to SmallVec
        let mut aug = vec![vec![0u8; 2 * n]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = self.get(i, j);
            }
            aug[i][n + i] = 1; // identity
        }

        // Gaussian elimination
        for i in 0..n {
            // Find pivot
            let mut pivot_row = None;
            for r in i..n {
                if aug[r][i] != 0 {
                    pivot_row = Some(r);
                    break;
                }
            }
            let pivot_row = pivot_row?;

            // Swap to top
            if pivot_row != i {
                aug.swap(i, pivot_row);
            }

            // Scale pivot to 1
            let inv_pivot = gftables::inv(aug[i][i]);
            for j in 0..2 * n {
                aug[i][j] = gftables::mul(aug[i][j], inv_pivot);
            }

            // Eliminate other rows
            for r in 0..n {
                if r != i && aug[r][i] != 0 {
                    let factor = aug[r][i];
                    for j in 0..2 * n {
                        aug[r][j] ^= gftables::mul(factor, aug[i][j]);
                    }
                }
            }
        }

        // Extract right half as inverse
        let mut inv = Matrix::zero(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.set(i, j, aug[i][n + j]);
            }
        }

        Some(inv)
    }
}
