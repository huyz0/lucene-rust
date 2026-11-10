//! `MaxSim` of a query multi-vector against stored values.

use lucene_util::max_sim::max_sim_blocks;

use crate::encoding::blocks_of;
use crate::{illegal, Result};

/// A query's token vectors, scored against stored multi-vectors as the sum,
/// over the query tokens, of each one's best dot product with a document
/// token. Unit-length tokens (ColBERT's) make that the sum of best cosines.
#[derive(Debug, Clone, PartialEq)]
pub struct MaxSimQuery {
    dim: usize,
    tokens: Vec<f32>,
}

impl MaxSimQuery {
    /// A query of `tokens`, each the same non-zero length.
    ///
    /// # Errors
    /// [`crate::Error::IllegalArgument`] for no tokens, an empty token, or
    /// tokens of different lengths (Lucene's `LateInteractionFloatValuesSource`
    /// messages).
    pub fn new(tokens: &[Vec<f32>]) -> Result<Self> {
        let first = tokens
            .first()
            .ok_or_else(|| illegal("queryVector must not be null or empty"))?;
        if first.is_empty() {
            return Err(illegal(
                "composing token vectors in provided query vector should not be null or empty",
            ));
        }
        if tokens.iter().any(|t| t.len() != first.len()) {
            return Err(illegal(
                "all composing token vectors in provided query vector should have the same length",
            ));
        }
        Ok(Self {
            dim: first.len(),
            tokens: tokens.concat(),
        })
    }

    /// The token dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The number of query tokens.
    pub fn len(&self) -> usize {
        // ARITH: `dim` is a non-empty token's length.
        #[allow(clippy::arithmetic_side_effects)]
        let n = self.tokens.len() / self.dim;
        n
    }

    /// Always false: a query has at least one token.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The `MaxSim` score of the stored value `value`. A value with no tokens
    /// scores `Float.MIN_VALUE` whatever its dimension, as Lucene's
    /// `SUM_MAX_SIM` does. `scratch` is reused between calls.
    ///
    /// # Errors
    /// [`crate::Error::IllegalArgument`] when `value` is not a multi-vector
    /// or its tokens have another dimension than the query's.
    pub fn score(&self, value: &[u8], scratch: &mut Vec<f32>) -> Result<f32> {
        let (blocks, dim, n) = blocks_of(value, scratch)?;
        if n == 0 {
            return Ok(f32::from_bits(1));
        }
        if dim != self.dim {
            return Err(illegal(format!(
                "Provided multi-vectors are incompatible. Their composing token vectors should \
                 have the same dimension, got {} != {dim}",
                self.dim
            )));
        }
        Ok(max_sim_blocks(&self.tokens, dim, blocks, n))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::encoding::{encode, TokenEncoding};

    #[test]
    fn scores_match_the_definition_for_every_encoding() {
        let q = vec![vec![1.0, 0.0], vec![0.6, 0.8]];
        let doc = [0.0, 1.0, 1.0, 0.0, 0.8, 0.6];
        let query = MaxSimQuery::new(&q).unwrap();
        assert_eq!((query.dim(), query.len()), (2, 2));
        let mut scratch = Vec::new();
        for enc in [
            TokenEncoding::Float32,
            TokenEncoding::Float16,
            TokenEncoding::Int8,
        ] {
            let v = encode(&doc, 2, enc).unwrap();
            let s = query.score(&v, &mut scratch).unwrap();
            // best of q0 = 1.0 (token 1), best of q1 = 0.96 (token 2)
            assert!((s - 1.96).abs() < 0.02, "{enc:?} {s}");
        }
        let other = encode(&[1.0, 2.0, 3.0], 3, TokenEncoding::Float32).unwrap();
        assert!(query.score(&other, &mut scratch).is_err());
    }

    #[test]
    fn bad_queries_are_refused() {
        assert!(MaxSimQuery::new(&[]).is_err());
        assert!(MaxSimQuery::new(&[vec![]]).is_err());
        assert!(MaxSimQuery::new(&[vec![1.0], vec![1.0, 2.0]]).is_err());
    }

    #[test]
    fn a_value_without_tokens_scores_float_min_value() {
        let query = MaxSimQuery::new(&[vec![1.0, 0.0]]).unwrap();
        assert!(!query.is_empty());
        let mut scratch = Vec::new();
        // `encode` refuses empty values, but a stored one may hold no tokens
        // (of any dimension, as Lucene checks emptiness first)
        let lucene: Vec<u8> = 3i32.to_le_bytes().to_vec();
        let half: Vec<u8> = [-1i32, 2, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
        for v in [lucene, half] {
            let s = query.score(&v, &mut scratch).unwrap();
            assert_eq!(s.to_bits(), 1, "Float.MIN_VALUE");
        }
    }
}
