//! Where a graph index keeps its vectors: raw floats or `SQ8` codes. The
//! graph code asks a store for distances and nothing else.

use crate::distance::distance;
use crate::format::{Flat, Metric, SqFlat};

pub(crate) trait VectorStore: Sync {
    /// Vectors held.
    fn len(&self) -> usize;
    /// The distance of the index.
    fn metric(&self) -> Metric;
    /// Distance from the float vector `q` to stored vector `i`.
    fn dis_query(&self, q: &[f32], i: usize) -> f32;
    /// Distance between stored vectors `a` and `b`.
    fn dis_pair(&self, a: usize, b: usize) -> f32;
    /// Starts loading stored vector `i`.
    fn prefetch(&self, i: usize);
}

impl VectorStore for Flat {
    fn len(&self) -> usize {
        Flat::len(self)
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    #[inline]
    fn dis_query(&self, q: &[f32], i: usize) -> f32 {
        distance(self.metric, q, self.vector(i))
    }

    #[inline]
    fn dis_pair(&self, a: usize, b: usize) -> f32 {
        distance(self.metric, self.vector(a), self.vector(b))
    }

    #[inline]
    fn prefetch(&self, i: usize) {
        Flat::prefetch(self, i);
    }
}

thread_local! {
    static DECODED: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl SqFlat {
    fn code(&self, i: usize) -> &[u8] {
        let d = self.sq.d();
        &self.codes[i * d..(i + 1) * d]
    }
}

impl VectorStore for SqFlat {
    fn len(&self) -> usize {
        SqFlat::len(self)
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    #[inline]
    fn dis_query(&self, q: &[f32], i: usize) -> f32 {
        self.sq.distance(self.metric, q, self.code(i))
    }

    fn dis_pair(&self, a: usize, b: usize) -> f32 {
        // Decode one side, then it is a query like any other
        // (`SQDistanceComputer::symmetric_dis`).
        DECODED.with(|buf| {
            let mut buf = buf.borrow_mut();
            buf.resize(self.sq.d(), 0.0);
            self.sq.decode(self.code(a), &mut buf);
            self.sq.distance(self.metric, &buf, self.code(b))
        })
    }

    #[inline]
    fn prefetch(&self, i: usize) {
        let d = self.sq.d();
        let mut at = i * d;
        while at < (i + 1) * d {
            lucene_util::simd::prefetch(&self.codes, at);
            at += 64;
        }
    }
}
