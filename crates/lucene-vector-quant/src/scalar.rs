//! Global scalar quantization in the style of Lucene's
//! `Lucene99ScalarQuantizedVectorsFormat`: one `[lower, upper]` quantile
//! interval for the whole field, learned from a sample, every dimension
//! mapped linearly onto `0..2^bits`.
//!
//! A code is the quantized bytes (one per dimension at 7 bits; two
//! dimensions per byte at 4 bits, as Lucene's `compress` option packs them)
//! followed by the vector's corrective offset (`f32`, LE). Scores are
//! `ScalarQuantizedVectorSimilarity`'s. The quantizer ignores centroids: the
//! interval is the field's, whatever partition a vector is in.
//!
//! Training follows `Lucene99ScalarQuantizedVectorsWriter`: 7 bits use the
//! confidence interval `max(0.9, 1 - 1/(dim + 1))`
//! (`calculateDefaultConfidenceInterval`); 4 bits search for the interval
//! that best preserves nearest-neighbour order
//! (`ScalarQuantizer.fromVectorsAutoInterval`).

use lucene_util::quantization::{self, ScalarQuantizer as Interval};
use lucene_util::vector_util;

use crate::{
    check_sample, f32_at, prepare_vector, Error, Quantizer, QueryScorer, Result, Similarity,
};

/// The registry name of [`ScalarFactory`].
pub const NAME: &str = "sq";

/// Builds [`ScalarQuantizer`]s of one bit width (7 or 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarFactory {
    bits: u8,
}

impl ScalarFactory {
    /// 7-bit quantization: what Lucene and OpenSearch call `int8`.
    pub fn int7() -> Self {
        ScalarFactory { bits: 7 }
    }

    /// 4-bit quantization, two dimensions per byte.
    pub fn int4() -> Self {
        ScalarFactory { bits: 4 }
    }

    /// The bit width new quantizers use.
    pub fn bits(&self) -> u8 {
        self.bits
    }
}

/// The similarity the integer scores use: `Cosine` vectors are normalized
/// first and scored as dot products.
fn integer_similarity(similarity: Similarity) -> Similarity {
    if similarity == Similarity::Cosine {
        Similarity::DotProduct
    } else {
        similarity
    }
}

impl crate::QuantizerFactory for ScalarFactory {
    fn name(&self) -> &str {
        NAME
    }

    /// The dimension rounded up to even at 4 bits, whose codes pack two
    /// dimensions per byte.
    fn code_dimension(&self, dimension: usize) -> usize {
        if self.bits <= 4 {
            dimension.checked_next_multiple_of(2).unwrap_or(dimension)
        } else {
            dimension
        }
    }

    fn train(
        &self,
        dimension: usize,
        similarity: Similarity,
        sample: &[f32],
    ) -> Result<Box<dyn Quantizer>> {
        check_sample(dimension, sample)?;
        let mut prepared = vec![0f32; sample.len()];
        for (v, out) in sample
            .chunks_exact(dimension)
            .zip(prepared.chunks_exact_mut(dimension))
        {
            prepare_vector(similarity, v, out);
        }
        let vectors: Vec<&[f32]> = prepared.chunks_exact(dimension).collect();
        let interval = if self.bits <= 4 {
            Interval::from_vectors_auto_interval(
                &vectors,
                integer_similarity(similarity),
                vectors.len(),
                self.bits,
            )
        } else {
            let dim = dimension as f32;
            let confidence = (1.0 - 1.0 / (dim + 1.0)).max(0.9);
            Interval::from_vectors(&vectors, confidence, vectors.len(), self.bits)
        }
        .map_err(|e| Error::IllegalArgument(e.to_string()))?;
        Ok(Box::new(ScalarQuantizer::new(
            dimension, similarity, interval,
        )))
    }

    fn read(
        &self,
        dimension: usize,
        similarity: Similarity,
        state: &[u8],
    ) -> Result<Box<dyn Quantizer>> {
        check_sample(dimension, &[])?;
        let [bits, rest @ ..] = state else {
            return Err(Error::CorruptState("empty sq state".into()));
        };
        if rest.len() != 8 || !matches!(bits, 4 | 7) {
            return Err(Error::CorruptState(format!(
                "sq state: {} bytes, {bits} bits",
                state.len()
            )));
        }
        let (lower, upper) = (f32_at(rest, 0), f32_at(rest, 4));
        // NaN and infinite bounds fail in `Interval::new`.
        if lower > upper {
            return Err(Error::CorruptState(format!(
                "sq interval [{lower}, {upper}] is inverted"
            )));
        }
        let interval =
            Interval::new(lower, upper, *bits).map_err(|e| Error::CorruptState(e.to_string()))?;
        Ok(Box::new(ScalarQuantizer::new(
            dimension, similarity, interval,
        )))
    }
}

/// A field's global scalar quantizer.
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    dimension: usize,
    similarity: Similarity,
    interval: Interval,
    scorer: quantization::ScalarQuantizedVectorSimilarity,
    /// Bytes of quantized values in a code (before the offset).
    value_bytes: usize,
}

impl ScalarQuantizer {
    /// A quantizer over `interval` (whose bit width is 7 or 4).
    pub fn new(dimension: usize, similarity: Similarity, interval: Interval) -> Self {
        let value_bytes = if interval.bits() <= 4 {
            dimension.div_ceil(2)
        } else {
            dimension
        };
        ScalarQuantizer {
            dimension,
            similarity,
            scorer: quantization::ScalarQuantizedVectorSimilarity::from_vector_similarity(
                integer_similarity(similarity),
                interval.constant_multiplier(),
                interval.bits(),
            ),
            interval,
            value_bytes,
        }
    }

    /// The learned interval.
    pub fn interval(&self) -> &Interval {
        &self.interval
    }

    fn packed(&self) -> bool {
        self.interval.bits() <= 4
    }

    /// `vector` quantized into `out` (one value per dimension, padded with
    /// zeros to an even length when packed); returns the corrective offset.
    fn quantize(&self, vector: &[f32], out: &mut [u8]) -> f32 {
        let mut work = vec![0f32; self.dimension];
        prepare_vector(self.similarity, vector, &mut work);
        self.interval.quantize(
            &work,
            &mut out[..self.dimension],
            integer_similarity(self.similarity),
        )
    }

    /// Unpacked values per code: the dimension, rounded up to even when
    /// packed so both nibble halves are whole.
    fn unpacked_len(&self) -> usize {
        if self.packed() {
            // ARITH: `div_ceil(2) * 2` is at most `dimension + 1`.
            #[allow(clippy::arithmetic_side_effects)]
            let n = self.dimension.div_ceil(2) * 2;
            n
        } else {
            self.dimension
        }
    }
}

impl Quantizer for ScalarQuantizer {
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
        // ARITH: at most a byte per dimension, plus four.
        #[allow(clippy::arithmetic_side_effects)]
        let len = self.value_bytes + 4;
        len
    }

    fn uses_centroid(&self) -> bool {
        false
    }

    fn encode(&self, vector: &[f32], _centroid: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dimension, "vector dimension");
        assert_eq!(out.len(), self.code_len(), "code length");
        let mut values = vec![0u8; self.unpacked_len()];
        let offset = self.quantize(vector, &mut values);
        let (codes, tail) = out.split_at_mut(self.value_bytes);
        if self.packed() {
            quantization::pack_nibbles(&values, codes);
        } else {
            codes.copy_from_slice(&values);
        }
        tail.copy_from_slice(&offset.to_le_bytes());
    }

    fn query_scorer(&self, query: &[f32]) -> Box<dyn QueryScorer + '_> {
        assert_eq!(query.len(), self.dimension, "query dimension");
        let mut values = vec![0u8; self.unpacked_len()];
        let offset = self.quantize(query, &mut values);
        Box::new(ScalarQueryScorer {
            q: self,
            values,
            offset,
        })
    }

    fn state(&self) -> Vec<u8> {
        let mut out = vec![self.interval.bits()];
        out.extend_from_slice(&self.interval.lower_quantile().to_le_bytes());
        out.extend_from_slice(&self.interval.upper_quantile().to_le_bytes());
        out
    }
}

/// A query quantized once with the field's interval.
#[derive(Debug)]
struct ScalarQueryScorer<'q> {
    q: &'q ScalarQuantizer,
    values: Vec<u8>,
    offset: f32,
}

impl QueryScorer for ScalarQueryScorer<'_> {
    fn set_centroid(&mut self, _centroid: &[f32]) {}

    fn score(&self, code: &[u8]) -> f32 {
        let (stored, tail) = code.split_at(self.q.value_bytes);
        let vector_offset = f32_at(tail, 0);
        if !self.q.packed() {
            return self
                .q
                .scorer
                .score(&self.values, self.offset, stored, vector_offset);
        }
        // `ScalarQuantizedVectorSimilarity` with the stored side packed: the
        // same formulas over the single-packed kernels.
        use quantization::ScalarQuantizedVectorSimilarity as S;
        match self.q.scorer {
            S::Euclidean { const_multiplier } => {
                let d = vector_util::int4_square_distance_single_packed(&self.values, stored);
                1.0 / (1.0 + d as f32 * const_multiplier)
            }
            S::DotProduct {
                const_multiplier, ..
            } => {
                let dot = vector_util::int4_dot_product_single_packed(&self.values, stored);
                let adjusted = dot as f32 * const_multiplier + self.offset + vector_offset;
                ((1.0 + adjusted) / 2.0).max(0.0)
            }
            S::MaximumInnerProduct {
                const_multiplier, ..
            } => {
                let dot = vector_util::int4_dot_product_single_packed(&self.values, stored);
                quantization::scale_max_inner_product_score(
                    dot as f32 * const_multiplier + self.offset + vector_offset,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::QuantizerFactory;

    fn sample(n: usize, dim: usize) -> Vec<f32> {
        let mut s = 12345u64;
        (0..n * dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn estimates_track_exact_scores() {
        let dim = 48;
        let data = sample(300, dim);
        let queries = sample(4, dim);
        for factory in [ScalarFactory::int7(), ScalarFactory::int4()] {
            for sim in [
                Similarity::Euclidean,
                Similarity::DotProduct,
                Similarity::Cosine,
                Similarity::MaximumInnerProduct,
            ] {
                let q = factory.train(dim, sim, &data).unwrap();
                assert!(!q.uses_centroid());
                assert_eq!(q.similarity(), sim);
                assert_eq!(q.dimension(), dim);
                let mut codes = vec![0u8; q.code_len() * 300];
                for (v, out) in data
                    .chunks_exact(dim)
                    .zip(codes.chunks_exact_mut(q.code_len()))
                {
                    q.encode(v, &[], out);
                }
                let mut err = 0f64;
                let mut n = 0;
                for query in queries.chunks_exact(dim) {
                    let mut s = q.query_scorer(query);
                    s.set_centroid(&[]);
                    let mut est = vec![0f32; 300];
                    s.score_block(&codes, &mut est);
                    for (v, e) in data.chunks_exact(dim).zip(&est) {
                        let exact = if sim == Similarity::DotProduct {
                            // Raw vectors: the integer score assumes unit
                            // length, so compare against the cosine.
                            Similarity::Cosine.score(query, v)
                        } else {
                            sim.score(query, v)
                        };
                        err += ((exact - e) / exact.max(1e-3)).abs() as f64;
                        n += 1;
                    }
                }
                let mean = err / n as f64;
                let bound = if factory.bits() == 4 { 0.35 } else { 0.1 };
                if sim != Similarity::DotProduct {
                    assert!(mean < bound, "{} {sim:?}: {mean}", factory.bits());
                }
            }
        }
    }

    #[test]
    fn state_round_trips_and_rejects_garbage() {
        let f = ScalarFactory::int4();
        assert_eq!(f.bits(), 4);
        let q = f.train(8, Similarity::Euclidean, &sample(50, 8)).unwrap();
        assert_eq!(q.code_len(), 4 + 4);
        let back = f.read(8, Similarity::Euclidean, &q.state()).unwrap();
        assert_eq!(back.state(), q.state());
        let q7 = ScalarFactory::int7()
            .train(8, Similarity::Euclidean, &sample(50, 8))
            .unwrap();
        assert_eq!(q7.code_len(), 8 + 4);
        for bad in [
            vec![],
            vec![7u8; 3],
            {
                let mut s = q7.state();
                s[0] = 5;
                s
            },
            {
                let mut s = vec![7u8];
                s.extend_from_slice(&f32::NAN.to_le_bytes());
                s.extend_from_slice(&1f32.to_le_bytes());
                s
            },
            {
                let mut s = vec![7u8];
                s.extend_from_slice(&2f32.to_le_bytes());
                s.extend_from_slice(&1f32.to_le_bytes());
                s
            },
        ] {
            assert!(matches!(
                f.read(8, Similarity::Euclidean, &bad),
                Err(Error::CorruptState(_))
            ));
        }
        assert!(f.read(0, Similarity::Euclidean, &q.state()).is_err());
        assert!(matches!(
            f.train(2, Similarity::Euclidean, &[f32::NAN, 1.0, 2.0, 3.0]),
            Err(Error::IllegalArgument(_))
        ));
    }

    #[test]
    fn odd_dimension_packs() {
        let q = ScalarFactory::int4()
            .train(5, Similarity::Euclidean, &sample(40, 5))
            .unwrap();
        assert_eq!(q.code_len(), 3 + 4);
        let v = &sample(1, 5);
        let mut code = vec![0u8; q.code_len()];
        q.encode(v, &[], &mut code);
        let s = q.query_scorer(v);
        assert!(s.score(&code) > 0.5);
        let sq = ScalarQuantizer::new(
            5,
            Similarity::Euclidean,
            Interval::new(-0.5, 0.5, 4).unwrap(),
        );
        assert_eq!(sq.interval().bits(), 4);
    }
}
