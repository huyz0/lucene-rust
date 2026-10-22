//! `PointsStackTracker`: the current BKD stack, tracked solely from the
//! calls to `IntersectVisitor.compare`.

/// One BKD stack frame: a cell's packed bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cell {
    pub min_packed_value: Vec<u8>,
    pub max_packed_value: Vec<u8>,
}

/// `PointsStackTracker`.
#[derive(Debug, Clone)]
pub struct PointsStackTracker {
    num_dims: usize,
    bytes_per_dim: usize,
    /// The stack, root first.
    pub stack: Vec<Cell>,
}

impl PointsStackTracker {
    /// `new PointsStackTracker(numDims, bytesPerDim)`.
    pub fn new(num_dims: usize, bytes_per_dim: usize) -> Self {
        Self {
            num_dims,
            bytes_per_dim,
            stack: Vec::new(),
        }
    }

    /// `Cell.contains(other)`: whether `outer` holds `inner` in every
    /// dimension (bytes compared unsigned).
    pub fn contains(&self, outer: &Cell, inner: &Cell) -> bool {
        (0..self.num_dims).all(|dim| {
            let r = dim * self.bytes_per_dim..(dim + 1) * self.bytes_per_dim;
            inner.min_packed_value[r.clone()] >= outer.min_packed_value[r.clone()]
                && inner.max_packed_value[r.clone()] <= outer.max_packed_value[r]
        })
    }

    /// `onCompare(min, max)`: pops every frame that does not contain the
    /// cell, then pushes it.
    pub fn on_compare(&mut self, min_packed_value: &[u8], max_packed_value: &[u8]) {
        let cell = Cell {
            min_packed_value: min_packed_value.to_vec(),
            max_packed_value: max_packed_value.to_vec(),
        };
        while self
            .stack
            .last()
            .is_some_and(|top| !self.contains(top, &cell))
        {
            self.stack.pop();
        }
        self.stack.push(cell);
    }
}
