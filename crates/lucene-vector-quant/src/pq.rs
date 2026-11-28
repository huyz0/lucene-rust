//! Product quantization (Jégou et al., "Product quantization for nearest
//! neighbor search", 2011), the encoder Faiss's `IndexIVFPQ` uses: the
//! dimensions split into `m` contiguous sub-vectors, each replaced by the
//! index of its nearest entry among 256 trained ones, so a code is `m` bytes.
//!
//! The quantizer encodes the **residual** of a vector against its partition's
//! centroid (`uses_centroid`), so the codebooks are trained on residuals: the
//! sample [`QuantizerFactory::train`] takes is a sample of `vector - centroid`
//! and the caller computes it. Scoring is asymmetric distance computation
//! (ADC): for a query and a partition one lookup table of `m x 256` entries is
//! built, and a code scores as the sum of `m` entries.
//!
//! - `Euclidean`: the table holds `|(q - c)_s - book_s[j]|^2` and the score is
//!   `1 / (1 + sum)`.
//! - `DotProduct`: the table holds `q_s . book_s[j]`, the partition adds
//!   `q . c`, and the score is `max((1 + dot) / 2, 0)`. `Cosine` is the same
//!   on unit-normalized vectors; `MaximumInnerProduct` scales the raw dot
//!   with `scale_max_inner_product_score`.
//!
//! State: `m` (`u32`, LE) then the codebooks, `m x 256 x (dimension / m)`
//! `f32` (LE), sub-quantizer major. The estimates have no error bound, so a
//! scorer returns `false` from `error_bounds`.

mod kmeans;
#[cfg(test)]
mod tests;

use rayon::prelude::*;

use crate::{
    check_sample, prepare_vector, Error, Quantizer, QuantizerFactory, QueryScorer, Result,
    Similarity,
};

/// The registry name of [`PqFactory`].
pub const NAME: &str = "pq";

/// Entries per sub-quantizer: 8-bit codes.
const KSUB: usize = 256;
/// Faiss's `max_points_per_centroid`: a sub-quantizer trains on at most this
/// many points an entry.
const POINTS_PER_CENTROID: usize = 256;
const SEED: u64 = 0x5051_5f54_5241_494e;

/// Builds [`PqQuantizer`]s of `m` sub-quantizers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PqFactory {
    m: usize,
}

impl PqFactory {
    /// A factory training `m` sub-quantizers; `m` must divide the dimension
    /// at [`QuantizerFactory::train`]. [`QuantizerFactory::read`] takes `m`
    /// from the stored state and ignores this one.
    pub fn new(m: usize) -> Self {
        PqFactory { m }
    }

    /// Sub-quantizers.
    pub fn m(&self) -> usize {
        self.m
    }
}

fn check_split(dimension: usize, m: usize) -> std::result::Result<(), String> {
    if m == 0 {
        return Err("m must be positive".into());
    }
    if !dimension.is_multiple_of(m) {
        return Err(format!("m ({m}) must divide the dimension ({dimension})"));
    }
    Ok(())
}

impl QuantizerFactory for PqFactory {
    fn name(&self) -> &str {
        NAME
    }

    fn train(
        &self,
        dimension: usize,
        similarity: Similarity,
        sample: &[f32],
    ) -> Result<Box<dyn Quantizer>> {
        check_sample(dimension, sample)?;
        check_split(dimension, self.m).map_err(Error::IllegalArgument)?;
        if sample.iter().any(|v| !v.is_finite()) {
            return Err(Error::IllegalArgument(
                "training vectors must be finite".into(),
            ));
        }
        let n = sample.len() / dimension;
        if n < KSUB {
            return Err(Error::IllegalArgument(format!(
                "Number of training points ({n}) should be at least as large as number of clusters ({KSUB})"
            )));
        }
        let dsub = dimension / self.m;
        let want = n.min(KSUB * POINTS_PER_CENTROID);
        let books: Vec<Vec<f32>> = (0..self.m)
            .into_par_iter()
            .map(|s| {
                // One subspace's sub-vectors, subsampled to `want` of the `n`.
                let mut order: Vec<usize> = (0..n).collect();
                let mut state = SEED ^ (s as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                for i in 0..want {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let j = i + ((state >> 33) as usize) % (n - i);
                    order.swap(i, j);
                }
                let mut sub = Vec::with_capacity(want * dsub);
                for &i in &order[..want] {
                    let at = i * dimension + s * dsub;
                    sub.extend_from_slice(&sample[at..at + dsub]);
                }
                kmeans::train(&sub, dsub, KSUB, SEED.wrapping_add(s as u64))
            })
            .collect();
        Ok(Box::new(PqQuantizer {
            dimension,
            similarity,
            m: self.m,
            books: books.concat(),
        }))
    }

    fn read(
        &self,
        dimension: usize,
        similarity: Similarity,
        state: &[u8],
    ) -> Result<Box<dyn Quantizer>> {
        check_sample(dimension, &[])?;
        let corrupt = |why: String| Error::CorruptState(format!("pq state: {why}"));
        if state.len() < 4 {
            return Err(corrupt(format!("{} bytes", state.len())));
        }
        let m = u32::from_le_bytes([state[0], state[1], state[2], state[3]]) as usize;
        check_split(dimension, m).map_err(corrupt)?;
        let floats = m * KSUB * (dimension / m);
        if state.len() != 4 + floats * 4 {
            return Err(corrupt(format!(
                "{} bytes for m {m}, dimension {dimension}",
                state.len()
            )));
        }
        let books: Vec<f32> = state[4..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if books.iter().any(|v| !v.is_finite()) {
            return Err(corrupt("a codebook entry is not finite".into()));
        }
        Ok(Box::new(PqQuantizer {
            dimension,
            similarity,
            m,
            books,
        }))
    }
}

/// A field's trained product quantizer.
#[derive(Debug, Clone)]
pub struct PqQuantizer {
    dimension: usize,
    similarity: Similarity,
    m: usize,
    /// `m x KSUB x dsub`, sub-quantizer major.
    books: Vec<f32>,
}

impl PqQuantizer {
    fn dsub(&self) -> usize {
        self.dimension / self.m
    }

    /// Sub-quantizer `s`'s entries: `KSUB` rows of `dsub` floats.
    fn book(&self, s: usize) -> &[f32] {
        let len = KSUB * self.dsub();
        &self.books[s * len..(s + 1) * len]
    }

    fn is_inner_product(&self) -> bool {
        self.similarity != Similarity::Euclidean
    }
}

impl Quantizer for PqQuantizer {
    fn name(&self) -> &str {
        NAME
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn similarity(&self) -> Similarity {
        self.similarity
    }

    fn code_len(&self) -> usize {
        self.m
    }

    fn uses_centroid(&self) -> bool {
        true
    }

    fn encode(&self, vector: &[f32], centroid: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dimension, "vector dimension");
        assert_eq!(centroid.len(), self.dimension, "centroid dimension");
        assert_eq!(out.len(), self.m, "code length");
        let mut residual = vec![0f32; self.dimension];
        prepare_vector(self.similarity, vector, &mut residual);
        for (r, c) in residual.iter_mut().zip(centroid) {
            *r -= c;
        }
        let dsub = self.dsub();
        for (s, code) in out.iter_mut().enumerate() {
            let sub = &residual[s * dsub..(s + 1) * dsub];
            let mut best = 0;
            let mut best_d = f32::INFINITY;
            for (j, entry) in self.book(s).chunks_exact(dsub).enumerate() {
                let d: f32 = entry.iter().zip(sub).map(|(e, v)| (e - v) * (e - v)).sum();
                if d < best_d {
                    best = j;
                    best_d = d;
                }
            }
            // `best` is below KSUB = 256.
            *code = best as u8;
        }
    }

    fn query_scorer(&self, query: &[f32]) -> Box<dyn QueryScorer + '_> {
        assert_eq!(query.len(), self.dimension, "query dimension");
        let mut q = vec![0f32; self.dimension];
        prepare_vector(self.similarity, query, &mut q);
        Box::new(PqScorer {
            quantizer: self,
            query: q,
            table: vec![0.0; self.m * KSUB],
            base: 0.0,
        })
    }

    fn state(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.books.len() * 4);
        out.extend_from_slice(&(self.m as u32).to_le_bytes());
        for v in &self.books {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// A query's lookup tables against one partition's centroid.
#[derive(Debug)]
struct PqScorer<'q> {
    quantizer: &'q PqQuantizer,
    /// The query, unit-normalized for `Cosine`.
    query: Vec<f32>,
    /// `m x KSUB` partial sums for the targeted centroid.
    table: Vec<f32>,
    /// `q . c` for the inner-product similarities, else zero.
    base: f32,
}

impl QueryScorer for PqScorer<'_> {
    fn set_centroid(&mut self, centroid: &[f32]) {
        let q = self.quantizer;
        assert_eq!(centroid.len(), q.dimension, "centroid dimension");
        let dsub = q.dsub();
        if q.is_inner_product() {
            self.base = self.query.iter().zip(centroid).map(|(a, b)| a * b).sum();
        }
        for s in 0..q.m {
            let qs = &self.query[s * dsub..(s + 1) * dsub];
            let cs = &centroid[s * dsub..(s + 1) * dsub];
            let row = &mut self.table[s * KSUB..(s + 1) * KSUB];
            for (slot, entry) in row.iter_mut().zip(q.book(s).chunks_exact(dsub)) {
                *slot = if q.is_inner_product() {
                    qs.iter().zip(entry).map(|(a, b)| a * b).sum()
                } else {
                    qs.iter()
                        .zip(cs)
                        .zip(entry)
                        .map(|((a, c), e)| (a - c - e) * (a - c - e))
                        .sum()
                };
            }
        }
    }

    fn score(&self, code: &[u8]) -> f32 {
        let sum: f32 = code
            .iter()
            .enumerate()
            .map(|(s, &j)| self.table[s * KSUB + usize::from(j)])
            .sum();
        match self.quantizer.similarity {
            Similarity::Euclidean => 1.0 / (1.0 + sum),
            Similarity::DotProduct | Similarity::Cosine => ((1.0 + self.base + sum) / 2.0).max(0.0),
            Similarity::MaximumInnerProduct => {
                lucene_util::quantization::scale_max_inner_product_score(self.base + sum)
            }
        }
    }
}
