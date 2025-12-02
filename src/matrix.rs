use smallvec::SmallVec;

/// This value should have a corresponding implementation of the `smallvec::Array` trait.
const DATA_ARRAY_SIZE: usize = 1024;

pub(crate) struct Matrix {
    row_count: usize,
    col_count: usize,
    data: SmallVec<[u8; DATA_ARRAY_SIZE]>,
}

impl Matrix {
    /// Access a coefficient (row j, col i)
    pub(crate) fn get(&self, row: usize, col: usize) -> u8 {
        // assert!(row < self.row_count);
        // assert!(col < self.col_count);
        self.data[row * self.col_count + col]
    }

    pub(crate) fn get_row(&self, row: usize) -> &[u8] {
        let start = row * self.col_count;
        let end = start + self.col_count;

        &self.data[start..end]
    }
}
