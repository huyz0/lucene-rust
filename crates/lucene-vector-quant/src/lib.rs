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
//! - [`Quantizer::state`] is whatever the quantizer learned at training
//!   time (empty for the per-vector quantizers). The index writes it beside
//!   the quantizer's [`Quantizer::name`]; reading the index looks the name up
//!   in a [`QuantizerRegistry`] and rebuilds the quantizer from the state.
//!
//! Scores are estimates of the similarity's score (higher is closer), on
//! the same scale as [`Similarity::score`] on the original vectors, so an
//! index can rerank the best estimates with exact scores and compare the two.
//!
//! Two families are built in, and a rotation that wraps either:
//!
//! | name | quantizer | codes |
//! |---|---|---|
//! | `osq` | [`osq::OsqQuantizer`]: Lucene's optimized scalar quantization (`Lucene104ScalarQuantizedVectorsFormat`), per vector, centred on the partition centroid. [`osq::OsqFactory::bbq`] is the 1-bit document / 4-bit query encoding of Lucene's BBQ. | 1, 2, 4, 7 or 8 bits per dimension plus 16 bytes of corrective terms |
//! | `sq` | [`scalar::ScalarQuantizer`]: Lucene99-style global min/max (quantile) scalar quantization, int7 or packed int4 | 1 or 1/2 byte per dimension plus a 4-byte offset |
//! | `rot-osq`, `rot-sq` | [`rotation::RotatedQuantizer`]: the inner quantizer on vectors zero-padded to the next power of two and rotated by a seeded randomized Hadamard transform (RaBitQ's rotation), built by [`rotation::RotatedFactory`] around any factory as `rot-<inner name>` | the inner quantizer's at the padded dimension |
//!
//! A new quantizer (Extended RaBitQ, product quantization) implements
//! [`QuantizerFactory`], [`Quantizer`] and [`QueryScorer`] and registers its
//! factory; nothing in the index changes.

pub mod osq;
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
}

/// Builds a [`Quantizer`], either by training it on a sample of a field's
/// vectors or by reading back the state an index stored.
pub trait QuantizerFactory: Send + Sync + std::fmt::Debug {
    /// The name an index records, and [`QuantizerRegistry::get`] finds this
    /// factory by.
    fn name(&self) -> &str;

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

    /// A registry holding the built-in quantizers (`osq`, `sq`) and their
    /// rotated forms (`rot-osq`, `rot-sq`). A factory reads every encoding
    /// or bit width of its family back, so one entry per family suffices.
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        let osq: Arc<dyn QuantizerFactory> = Arc::new(osq::OsqFactory::bbq());
        let sq: Arc<dyn QuantizerFactory> = Arc::new(scalar::ScalarFactory::int7());
        r.register(Arc::clone(&osq));
        r.register(Arc::clone(&sq));
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
        assert_eq!(r.names(), vec!["osq", "sq", "rot-osq", "rot-sq"]);
        assert!(r.get("osq").is_ok());
        assert!(r.get("rot-sq").is_ok());
        assert_eq!(
            r.get("rabitq").unwrap_err(),
            Error::UnknownQuantizer("rabitq".into())
        );
        r.register(Arc::new(osq::OsqFactory::bbq()));
        assert_eq!(r.names(), vec!["sq", "rot-osq", "rot-sq", "osq"]);
        assert!(QuantizerRegistry::new().names().is_empty());
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
    #[should_panic(expected = "one code per score")]
    fn score_block_rejects_ragged_codes() {
        Constant.score_block(&[0; 5], &mut [0f32; 2]);
    }
}
