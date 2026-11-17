//! The visited set of a graph search, cleared by bumping an epoch instead of
//! zeroing one mark per vertex.

use std::cell::Cell;

#[derive(Default)]
pub(crate) struct Visited {
    marks: Vec<u32>,
    epoch: u32,
}

impl Visited {
    pub(crate) fn new(n: usize) -> Self {
        let mut v = Visited::default();
        v.begin(n);
        v
    }

    /// Starts a new search over `n` vertices with nothing visited.
    pub(crate) fn begin(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
    }

    pub(crate) fn clear(&mut self) {
        let n = self.marks.len();
        self.begin(n);
    }

    #[inline]
    pub(crate) fn set(&mut self, i: usize) {
        self.marks[i] = self.epoch;
    }

    #[inline]
    pub(crate) fn seen(&self, i: usize) -> bool {
        self.marks[i] == self.epoch
    }
}

thread_local! {
    /// One set per searching thread, reused by every query on it.
    pub(crate) static VISITED: Cell<Visited> = Cell::new(Visited::default());
}

#[cfg(test)]
mod tests;
