#![forbid(unsafe_code)]
//! Vector quantizers behind one object-safe trait, for indexes that store a
//! compressed code per vector and score a query against many codes at once.
//!
//! An index (the IVF index in `lucene-ivf`, for one) stores what a
//! [`Quantizer`] makes and never looks inside it:
//!
//! - [`Quantizer::encode`] turns a vector into a fixed-length code of
//!   [`Quantizer::code_len`] bytes. A quantizer may encode relative to a
//!   centroid (the partition the vector belongs to), which is what makes
//!   per-partition codes more accurate than one global quantization.
//! - [`Quantizer::query_scorer`] prepares a query once;
//!   [`QueryScorer::set_centroid`] re-targets it at the next partition and
//!   [`QueryScorer::score_block`] scores a partition's codes in one call, so
//!   the dynamic dispatch is paid once per partition, not once per vector.
//!   A quantizer that works in another space than the vectors' (a rotation)
//!   maps each centroid once with [`Quantizer::prepare_centroid`]; the index
//!   keeps the result and passes it to [`Quantizer::encode_prepared`] and
//!   [`QueryScorer::set_prepared_centroid`].
//! - [`QueryScorer::error_bounds`] gives each estimate a margin the exact
//!   score stays under (with high probability; see each quantizer), so an
//!   index can stop reranking once no remaining estimate can matter.
//! - [`Quantizer::state`] is whatever the quantizer learned at training
//!   time (empty for the per-vector quantizers). The index writes it beside
//!   the quantizer's [`Quantizer::name`]; reading the index looks the name up
//!   in a [`QuantizerRegistry`] and rebuilds the quantizer from the state.
//!
//! Scores are estimates of the similarity's score (higher is closer), on
//! the same scale as [`Similarity::score`] on the original vectors, so an
//! index can rerank the best estimates with exact scores and compare the two.
//!
//! Three families are built in, and a rotation that wraps any of them:
//!
//! | name | quantizer | codes |
//! |---|---|---|
//! | `osq` | [`osq::OsqQuantizer`]: Lucene's optimized scalar quantization (`Lucene104ScalarQuantizedVectorsFormat`), per vector, centred on the partition centroid. [`osq::OsqFactory::bbq`] is the 1-bit document / 4-bit query encoding of Lucene's BBQ. | 1, 2, 4, 7 or 8 bits per dimension plus 16 bytes of corrective terms |
//! | `sq` | [`scalar::ScalarQuantizer`]: Lucene99-style global min/max (quantile) scalar quantization, int7 or packed int4 | 1 or 1/2 byte per dimension plus a 4-byte offset |
//! | `erq` | [`rabitq::ExRabitqQuantizer`]: Extended RaBitQ (Gao et al., SIGMOD 2025), per vector, centred on the partition centroid, after a random rotation | 2 to 6 bits per dimension padded to a multiple of 32, or to a power of two of at least 8 up to 32 (one byte at 7 and 8) plus 12 bytes of corrective terms |
//! | `rot-osq`, `rot-sq` | [`rotation::RotatedQuantizer`]: the inner quantizer on vectors zero-padded to its code dimension ([`QuantizerFactory::code_dimension`], a multiple of 64 for OSQ at 4 bits and under) and rotated there by a seeded randomized Hadamard transform (RaBitQ's rotation), built by [`rotation::RotatedFactory`] around any factory as `rot-<inner name>` | the inner quantizer's, no longer than without the rotation |
//!
//! A new quantizer (product quantization, say) implements
//! [`QuantizerFactory`], [`Quantizer`] and [`QueryScorer`] and registers its
//! factory; nothing in the index changes.

pub mod osq;
pub mod rabitq;
pub mod rotation;
pub mod scalar;

use std::sync::Arc;

pub use lucene_util::quantization::VectorSimilarityFunction as Similarity;

/// Errors building or reading a quantizer.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// No factory of this name is registered.
    #[error("unknown quantizer [{0}]")]
    UnknownQuantizer(String),
    /// The stored state does not decode.
    #[error("corrupt quantizer state: {0}")]
    CorruptState(String),
    /// Bad training input or parameters.
    #[error("{0}")]
    IllegalArgument(String),
}

/// This crate's result.
pub type Result<T> = std::result::Result<T, Error>;

/// A trained quantizer for one vector field: it encodes vectors into
/// fixed-length codes and prepares queries to score them.
pub trait Quantizer: Send + Sync + std::fmt::Debug {
    /// The registry name of the factory that reads this quantizer back.
    fn name(&self) -> &str;

    /// The vector dimension.
    fn dimension(&self) -> usize;

    /// The similarity scores estimate.
    fn similarity(&self) -> Similarity;

    /// Bytes of one code.
    fn code_len(&self) -> usize;

    /// Whether codes depend on the centroid passed to [`Quantizer::encode`]
    /// and [`QueryScorer::set_centroid`]. When `false`, the centroid is
    /// ignored and a query needs no per-partition preparation.
    fn uses_centroid(&self) -> bool;

    /// Encodes `vector` (of [`Quantizer::dimension`] floats, as indexed: a
    /// `Cosine` vector need not be normalized) relative to `centroid` into
    /// `out`, which must be exactly [`Quantizer::code_len`] bytes.
    ///
    /// # Panics
    ///
    /// On a wrong-length `vector`, `centroid` or `out`.
    fn encode(&self, vector: &[f32], centroid: &[f32], out: &mut [u8]);

    /// A scorer for `query` (of [`Quantizer::dimension`] floats). Before the
    /// first score, and before scoring each partition when
    /// [`Quantizer::uses_centroid`], call [`QueryScorer::set_centroid`].
    ///
    /// # Panics
    ///
    /// On a wrong-length `query`.
    fn query_scorer(&self, query: &[f32]) -> Box<dyn QueryScorer + '_>;

    /// What [`QuantizerFactory::read`] needs to rebuild this quantizer.
    fn state(&self) -> Vec<u8>;

    /// `centroid` (of [`Quantizer::dimension`] floats) mapped into the space
    /// this quantizer encodes and scores in, when that is not the original
    /// space: `None` (the default) means the centroid is used as it is. An
    /// index computes this once per partition and passes the result, or the
    /// centroid itself on `None`, to [`Quantizer::encode_prepared`] and
    /// [`QueryScorer::set_prepared_centroid`], so the mapping is not paid
    /// per vector or per probe. A quantizer returns `None` for every
    /// centroid or `Some` of one length for every centroid.
    ///
    /// # Panics
    ///
    /// On a wrong-length `centroid`.
    fn prepare_centroid(&self, centroid: &[f32]) -> Option<Vec<f32>> {
        let _ = centroid;
        None
    }

    /// [`Quantizer::encode`] against a centroid already passed through
    /// [`Quantizer::prepare_centroid`] (or the centroid itself when that
    /// returned `None`). By default the same as `encode`.
    ///
    /// # Panics
    ///
    /// On a wrong-length `vector`, `prepared` or `out`.
    fn encode_prepared(&self, vector: &[f32], prepared: &[f32], out: &mut [u8]) {
        self.encode(vector, prepared, out);
    }

    /// Bytes that `n` codes take laid out as a block for
    /// [`QueryScorer::score_block`]: by default the codes end to end.
    fn block_len(&self, n: usize) -> usize {
        self.code_len().saturating_mul(n)
    }

    /// Appends `codes` (whole codes end to end, as [`Quantizer::encode`]
    /// wrote them) to `out` in the layout [`QueryScorer::score_block`] reads,
    /// [`Quantizer::block_len`] bytes for their count. By default the codes
    /// are copied as they are; a quantizer can interleave them instead, so a
    /// block scores many codes per instruction.
    ///
    /// # Panics
    ///
    /// When `codes` is not a whole number of codes.
    fn write_block(&self, codes: &[u8], out: &mut Vec<u8>) {
        assert_eq!(
            codes.len().checked_rem(self.code_len()).unwrap_or(0),
            0,
            "codes must be whole codes"
        );
        out.extend_from_slice(codes);
    }
}

/// A query prepared against a [`Quantizer`]'s codes.
pub trait QueryScorer {
    /// Targets the codes encoded relative to `centroid`.
    fn set_centroid(&mut self, centroid: &[f32]);

    /// [`QueryScorer::set_centroid`] with a centroid already passed through
    /// [`Quantizer::prepare_centroid`] (or the centroid itself when that
    /// returned `None`). By default the same as `set_centroid`.
    fn set_prepared_centroid(&mut self, prepared: &[f32]) {
        self.set_centroid(prepared);
    }

    /// The estimated score of one code.
    fn score(&self, code: &[u8]) -> f32;

    /// The estimated scores of a block of `scores.len()` codes, laid out as
    /// [`Quantizer::write_block`] wrote them (by default end to end), into
    /// `scores`.
    ///
    /// # Panics
    ///
    /// When `block` is not [`Quantizer::block_len`] bytes for `scores.len()`
    /// codes.
    fn score_block(&self, codes: &[u8], scores: &mut [f32]) {
        let code_len = codes.len().checked_div(scores.len()).unwrap_or(0);
        assert_eq!(
            Some(codes.len()),
            code_len.checked_mul(scores.len()),
            "codes must hold one code per score"
        );
        if code_len == 0 {
            scores.fill(self.score(&[]));
            return;
        }
        for (s, code) in scores.iter_mut().zip(codes.chunks_exact(code_len)) {
            *s = self.score(code);
        }
    }

    /// Error margins of a block's estimates: for each code of `codes` (laid
    /// out as for [`QueryScorer::score_block`]), whose estimate is the same
    /// position of `scores`, writes into `bounds` a margin `b >= 0` such that
    /// the code's exact score is at most `score + b`, in score space. Returns
    /// `false` (the default) when the scorer cannot bound its error, leaving
    /// `bounds` unspecified.
    ///
    /// The margins of the built-in quantizers are confidence bounds, not
    /// guarantees (see each module): a margin is [`BOUND_CONFIDENCE`]
    /// standard deviations of the estimate's error, or a worst case that
    /// assumes no value fell outside its quantization interval.
    ///
    /// # Panics
    ///
    /// When `bounds` and `scores` differ in length, or `codes` is not a
    /// block of that many codes.
    fn error_bounds(&self, codes: &[u8], scores: &[f32], bounds: &mut [f32]) -> bool {
        let _ = (codes, scores, bounds);
        false
    }
}

/// The standard deviations of estimate error a probabilistic error bound
/// ([`QueryScorer::error_bounds`]) spans: RaBitQ's `epsilon_0` (Gao & Long,
/// SIGMOD 2024), which its authors found to keep nearly every true neighbour
/// above the bound.
pub const BOUND_CONFIDENCE: f32 = 1.9;

/// The margin to add to `score`, an estimate under `similarity`, when the
/// similarity's raw quantity may be off by `raw_error` in the favourable
/// direction: the squared distance `raw_error` smaller for `Euclidean`, the
/// inner product `raw_error` larger otherwise. The raw estimate is recovered
/// from the score, so a clamped score (zero for `DotProduct` and `Cosine`)
/// is taken at the clamp. Never negative; infinite for a NaN `raw_error`,
/// and for an infinite one except under `Euclidean`, whose score is at
/// most 1.
pub fn score_margin(similarity: Similarity, score: f32, raw_error: f32) -> f32 {
    if raw_error.is_nan() {
        return f32::INFINITY;
    }
    let upper = match similarity {
        Similarity::Euclidean => {
            let d2 = if score > 0.0 {
                1.0 / score - 1.0
            } else {
                f32::INFINITY
            };
            1.0 / (1.0 + (d2.max(0.0) - raw_error).max(0.0))
        }
        Similarity::DotProduct | Similarity::Cosine => {
            ((1.0 + (2.0 * score - 1.0) + raw_error) / 2.0).max(0.0)
        }
        Similarity::MaximumInnerProduct => {
            let raw = if score < 1.0 {
                1.0 - 1.0 / score
            } else {
                score - 1.0
            };
            lucene_util::quantization::scale_max_inner_product_score(raw + raw_error)
        }
    };
    let margin = upper - score;
    if margin.is_nan() {
        f32::INFINITY
    } else {
        margin.max(0.0)
    }
}

/// Builds a [`Quantizer`], either by training it on a sample of a field's
/// vectors or by reading back the state an index stored.
pub trait QuantizerFactory: Send + Sync + std::fmt::Debug {
    /// The name an index records, and [`QuantizerRegistry::get`] finds this
    /// factory by.
    fn name(&self) -> &str;

    /// The dimension the codes of `dimension`-dimensional vectors are
    /// padded to: zero coordinates up to it cost no code bytes, so a
    /// rotation in front of the quantizer can spread a vector's energy into
    /// them. By default `dimension`.
    fn code_dimension(&self, dimension: usize) -> usize {
        dimension
    }

    /// Trains a quantizer on `sample`: vectors of `dimension` floats laid end
    /// to end, as indexed.
    ///
    /// # Errors
    ///
    /// A zero dimension, a sample that is not a whole number of vectors, or
    /// training input the quantizer cannot use (non-finite values).
    fn train(
        &self,
        dimension: usize,
        similarity: Similarity,
        sample: &[f32],
    ) -> Result<Box<dyn Quantizer>>;

    /// Rebuilds the quantizer whose [`Quantizer::state`] was `state`.
    ///
    /// # Errors
    ///
    /// [`Error::CorruptState`] when `state` does not decode.
    fn read(
        &self,
        dimension: usize,
        similarity: Similarity,
        state: &[u8],
    ) -> Result<Box<dyn Quantizer>>;
}

/// Quantizer factories by name, for reading an index back.
#[derive(Debug, Clone, Default)]
pub struct QuantizerRegistry {
    factories: Vec<Arc<dyn QuantizerFactory>>,
}

impl QuantizerRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry holding the built-in quantizers (`osq`, `sq`, `erq`) and the
    /// rotated forms of the first two
    /// (`rot-osq`, `rot-sq`). A factory reads every encoding
    /// or bit width of its family back, so one entry per family suffices.
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        let osq: Arc<dyn QuantizerFactory> = Arc::new(osq::OsqFactory::bbq());
        let sq: Arc<dyn QuantizerFactory> = Arc::new(scalar::ScalarFactory::int7());
        r.register(Arc::clone(&osq));
        r.register(Arc::clone(&sq));
        r.register(Arc::new(rabitq::ExRabitqFactory::new(4)));
        r.register(Arc::new(rotation::RotatedFactory::new(osq)));
        r.register(Arc::new(rotation::RotatedFactory::new(sq)));
        r
    }

    /// Adds `factory`, replacing any factory of the same name.
    pub fn register(&mut self, factory: Arc<dyn QuantizerFactory>) {
        self.factories.retain(|f| f.name() != factory.name());
        self.factories.push(factory);
    }

    /// The factory registered as `name`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownQuantizer`] when none is.
    pub fn get(&self, name: &str) -> Result<&Arc<dyn QuantizerFactory>> {
        self.factories
            .iter()
            .find(|f| f.name() == name)
            .ok_or_else(|| Error::UnknownQuantizer(name.to_string()))
    }

    /// The registered names, in registration order.
    pub fn names(&self) -> Vec<&str> {
        self.factories.iter().map(|f| f.name()).collect()
    }
}

/// Checks the common training arguments: a non-zero dimension and a sample
/// of whole vectors.
pub(crate) fn check_sample(dimension: usize, sample: &[f32]) -> Result<()> {
    if dimension == 0 {
        return Err(Error::IllegalArgument("dimension must be positive".into()));
    }
    if !sample.len().is_multiple_of(dimension) {
        return Err(Error::IllegalArgument(format!(
            "sample of {} floats is not a whole number of {dimension}-dimensional vectors",
            sample.len()
        )));
    }
    Ok(())
}

/// `vector`, unit-normalized when `similarity` is `Cosine` (a zero vector is
/// left as is), into `out`.
pub(crate) fn prepare_vector(similarity: Similarity, vector: &[f32], out: &mut [f32]) {
    out.copy_from_slice(vector);
    if similarity == Similarity::Cosine {
        // A zero vector has no direction; it stays zero and scores as such.
        let _ = lucene_util::vector_util::l2normalize(out, false);
    }
}

/// Reads the little-endian `f32` at `bytes[at..at + 4]`.
#[inline]
pub(crate) fn f32_at(bytes: &[u8], at: usize) -> f32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[at..][..4]);
    f32::from_le_bytes(b)
}

/// Reads the little-endian `i32` at `bytes[at..at + 4]`.
#[inline]
pub(crate) fn i32_at(bytes: &[u8], at: usize) -> i32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[at..][..4]);
    i32::from_le_bytes(b)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    #[test]
    fn registry_finds_replaces_and_rejects() {
        let mut r = QuantizerRegistry::with_builtins();
        assert_eq!(r.names(), vec!["osq", "sq", "erq", "rot-osq", "rot-sq"]);
        assert!(r.get("osq").is_ok());
        assert!(r.get("rot-sq").is_ok());
        assert_eq!(
            r.get("rabitq").unwrap_err(),
            Error::UnknownQuantizer("rabitq".into())
        );
        r.register(Arc::new(osq::OsqFactory::bbq()));
        assert_eq!(r.names(), vec!["sq", "erq", "rot-osq", "rot-sq", "osq"]);
        assert!(QuantizerRegistry::new().names().is_empty());
    }

    #[test]
    fn centroids_are_prepared_as_they_are_by_default() {
        let q = osq::OsqFactory::bbq()
            .train(4, Similarity::Euclidean, &[])
            .unwrap();
        let (v, c) = ([1.0, -2.0, 0.5, 3.0], [0.25, 0.0, -1.0, 1.0]);
        assert_eq!(q.prepare_centroid(&c), None);
        let (mut a, mut b) = (vec![0u8; q.code_len()], vec![0u8; q.code_len()]);
        q.encode(&v, &c, &mut a);
        q.encode_prepared(&v, &c, &mut b);
        assert_eq!(a, b);
        let mut s = q.query_scorer(&v);
        s.set_centroid(&c);
        let plain = s.score(&a);
        s.set_centroid(&[0.0; 4]);
        s.set_prepared_centroid(&c);
        assert_eq!(s.score(&a).to_bits(), plain.to_bits());
    }

    #[test]
    fn sample_checks() {
        assert!(check_sample(0, &[]).is_err());
        assert!(check_sample(3, &[1.0; 4]).is_err());
        assert!(check_sample(2, &[1.0; 4]).is_ok());
    }

    #[test]
    fn prepare_normalizes_cosine_only() {
        let mut out = [0f32; 2];
        prepare_vector(Similarity::Cosine, &[3.0, 4.0], &mut out);
        assert!((out[0] - 0.6).abs() < 1e-6 && (out[1] - 0.8).abs() < 1e-6);
        prepare_vector(Similarity::DotProduct, &[3.0, 4.0], &mut out);
        assert_eq!(out, [3.0, 4.0]);
        prepare_vector(Similarity::Cosine, &[0.0, 0.0], &mut out);
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn score_margins() {
        use Similarity::*;
        // Euclidean: score 0.5 is d^2 = 1; 0.5 closer is score 1 / 1.5.
        assert!((score_margin(Euclidean, 0.5, 0.5) - (1.0 / 1.5 - 0.5)).abs() < 1e-6);
        assert_eq!(score_margin(Euclidean, 0.5, f32::INFINITY), 0.5);
        assert_eq!(score_margin(Euclidean, 0.0, 1.0), 0.0);
        for sim in [DotProduct, Cosine] {
            assert!((score_margin(sim, 0.75, 0.2) - 0.1).abs() < 1e-6);
            // A clamped score is taken at the clamp, raw -1.
            assert!((score_margin(sim, 0.0, 0.5) - 0.25).abs() < 1e-6);
            assert_eq!(score_margin(sim, 0.5, f32::INFINITY), f32::INFINITY);
        }
        // MIP: raw 1 scores 2, raw 1.5 scores 2.5; raw -1 scores 0.5.
        assert!((score_margin(MaximumInnerProduct, 2.0, 0.5) - 0.5).abs() < 1e-6);
        assert!((score_margin(MaximumInnerProduct, 0.5, 2.0) - 1.5).abs() < 1e-6);
        assert_eq!(score_margin(DotProduct, 0.5, f32::NAN), f32::INFINITY);
        assert_eq!(score_margin(DotProduct, 0.5, -1.0), 0.0);
        assert_eq!(score_margin(DotProduct, f32::NAN, 0.1), f32::INFINITY);
    }

    fn gaussian_vectors(n: usize, dim: usize, sim: Similarity, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let mut out = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let mut v: Vec<f32> = (0..dim)
                .map(|d| {
                    let g = next() + next() + next() + next();
                    0.3 + g * if d % 5 == 0 { 3.0 } else { 1.0 }
                })
                .collect();
            if matches!(sim, Similarity::DotProduct | Similarity::Cosine) {
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm);
            }
            out.extend(v);
        }
        out
    }

    /// The share of codes whose exact score exceeds estimate plus margin,
    /// and the mean margin, of `factory` over clustered data; `None` when
    /// the scorer has no bounds.
    fn bound_misses(
        factory: &dyn QuantizerFactory,
        sim: Similarity,
        dim: usize,
    ) -> Option<(f64, f64)> {
        let n = 600;
        let data = gaussian_vectors(n, dim, sim, 5);
        let queries = gaussian_vectors(12, dim, sim, 6);
        let q = factory.train(dim, sim, &data).unwrap();
        let mut centroid = vec![0f32; dim];
        for v in data.chunks_exact(dim) {
            for (c, x) in centroid.iter_mut().zip(v) {
                *c += x / n as f32;
            }
        }
        let prepared = q.prepare_centroid(&centroid);
        let target = prepared.as_deref().unwrap_or(&centroid);
        let mut codes = vec![0u8; q.code_len() * n];
        for (v, out) in data
            .chunks_exact(dim)
            .zip(codes.chunks_exact_mut(q.code_len()))
        {
            q.encode_prepared(v, target, out);
        }
        let mut block = Vec::new();
        q.write_block(&codes, &mut block);
        let (mut misses, mut total, mut margin) = (0usize, 0usize, 0f64);
        for query in queries.chunks_exact(dim) {
            let mut s = q.query_scorer(query);
            s.set_prepared_centroid(target);
            let mut est = vec![0f32; n];
            s.score_block(&block, &mut est);
            let mut bounds = vec![0f32; n];
            if !s.error_bounds(&block, &est, &mut bounds) {
                return None;
            }
            for ((v, &e), &b) in data.chunks_exact(dim).zip(&est).zip(&bounds) {
                assert!(b >= 0.0 && b.is_finite(), "{} {sim:?}: bound {b}", q.name());
                let exact = sim.score(query, v);
                misses += usize::from(exact > e + b + 1e-6);
                total += 1;
                margin += f64::from(b);
            }
        }
        Some((misses as f64 / total as f64, margin / total as f64))
    }

    #[test]
    fn error_bounds_cover_the_exact_scores() {
        use lucene_util::quantization::ScalarEncoding as E;
        let factories: Vec<(Box<dyn QuantizerFactory>, f64)> = vec![
            (Box::new(osq::OsqFactory::bbq()), 0.03),
            (Box::new(osq::OsqFactory::new(E::DibitQueryNibble)), 0.03),
            (Box::new(osq::OsqFactory::new(E::PackedNibble)), 0.03),
            (Box::new(osq::OsqFactory::new(E::SevenBit)), 0.03),
            (Box::new(osq::OsqFactory::new(E::UnsignedByte)), 0.03),
            (Box::new(scalar::ScalarFactory::int7()), 0.01),
            (Box::new(scalar::ScalarFactory::int4()), 0.01),
            (Box::new(rabitq::ExRabitqFactory::new(2)), 0.03),
            (Box::new(rabitq::ExRabitqFactory::new(4)), 0.03),
            (Box::new(rabitq::ExRabitqFactory::new(7)), 0.03),
            (
                Box::new(rotation::RotatedFactory::new(Arc::new(
                    osq::OsqFactory::bbq(),
                ))),
                0.03,
            ),
        ];
        for sim in [
            Similarity::Euclidean,
            Similarity::DotProduct,
            Similarity::Cosine,
            Similarity::MaximumInnerProduct,
        ] {
            for (f, allowed) in &factories {
                for dim in [37, 128] {
                    let got = bound_misses(f.as_ref(), sim, dim);
                    match got {
                        Some((share, _)) => assert!(
                            share <= *allowed,
                            "{} {sim:?} {dim}: {share} of exact scores above the bound",
                            f.name()
                        ),
                        None => assert!(
                            f.name() == rabitq::NAME && sim == Similarity::MaximumInnerProduct
                        ),
                    }
                }
            }
        }
    }

    #[derive(Debug)]
    struct Constant;
    impl QueryScorer for Constant {
        fn set_centroid(&mut self, _: &[f32]) {}
        fn score(&self, code: &[u8]) -> f32 {
            code.len() as f32
        }
    }

    #[test]
    fn default_score_block() {
        let mut scores = [0f32; 3];
        Constant.score_block(&[0; 6], &mut scores);
        assert_eq!(scores, [2.0; 3]);
        Constant.score_block(&[], &mut scores);
        assert_eq!(scores, [0.0; 3]);
        let mut none: [f32; 0] = [];
        Constant.score_block(&[], &mut none);
    }

    #[test]
    fn bounds_are_unknown_by_default() {
        assert!(!Constant.error_bounds(&[0; 4], &[1.0; 2], &mut [0.0; 2]));
    }

    #[test]
    #[should_panic(expected = "one code per score")]
    fn score_block_rejects_ragged_codes() {
        Constant.score_block(&[0; 5], &mut [0f32; 2]);
    }
}
