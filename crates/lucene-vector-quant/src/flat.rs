//! The quantizer of an index that stores no codes: a cluster's vectors are
//! scored exactly from the full-precision file, as Faiss's `IndexIVFFlat`
//! does. Codes are zero bytes long, so a cluster's posting block is its slots
//! alone, and the index scans a probed cluster's vectors instead of codes
//! ([`NAME`]).

use crate::{check_sample, Error, Quantizer, QuantizerFactory, QueryScorer, Result, Similarity};

/// The registry name of [`FlatFactory`].
pub const NAME: &str = "flat";

/// Builds [`FlatQuantizer`]s.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlatFactory;

impl QuantizerFactory for FlatFactory {
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
        Ok(Box::new(FlatQuantizer {
            dimension,
            similarity,
        }))
    }

    fn read(
        &self,
        dimension: usize,
        similarity: Similarity,
        state: &[u8],
    ) -> Result<Box<dyn Quantizer>> {
        check_sample(dimension, &[])?;
        if !state.is_empty() {
            return Err(Error::CorruptState(format!(
                "flat state is {} bytes, not empty",
                state.len()
            )));
        }
        Ok(Box::new(FlatQuantizer {
            dimension,
            similarity,
        }))
    }
}

/// A quantizer with no codes.
#[derive(Debug, Clone, Copy)]
pub struct FlatQuantizer {
    dimension: usize,
    similarity: Similarity,
}

impl Quantizer for FlatQuantizer {
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
        0
    }

    fn uses_centroid(&self) -> bool {
        false
    }

    fn encode(&self, vector: &[f32], _centroid: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dimension, "vector dimension");
        assert!(out.is_empty(), "code length");
    }

    fn query_scorer(&self, query: &[f32]) -> Box<dyn QueryScorer + '_> {
        assert_eq!(query.len(), self.dimension, "query dimension");
        Box::new(FlatScorer)
    }

    fn state(&self) -> Vec<u8> {
        Vec::new()
    }
}

/// There is nothing to score a code with: the index scores vectors.
#[derive(Debug)]
struct FlatScorer;

impl QueryScorer for FlatScorer {
    fn set_centroid(&mut self, _centroid: &[f32]) {}

    fn score(&self, _code: &[u8]) -> f32 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_no_codes_and_no_state() {
        let q = FlatFactory
            .train(3, Similarity::Euclidean, &[1.0; 6])
            .unwrap();
        assert_eq!((q.name(), q.dimension(), q.code_len()), ("flat", 3, 0));
        assert!(!q.uses_centroid() && q.state().is_empty());
        assert_eq!(q.similarity(), Similarity::Euclidean);
        q.encode(&[0.0; 3], &[], &mut []);
        let mut s = q.query_scorer(&[0.0; 3]);
        s.set_centroid(&[]);
        assert_eq!(s.score(&[]), 0.0);
        assert_eq!(q.block_len(5), 0);
        let back = FlatFactory.read(3, Similarity::Euclidean, &[]).unwrap();
        assert_eq!(back.code_len(), 0);
    }

    #[test]
    fn refuses_bad_input() {
        assert!(FlatFactory.train(0, Similarity::Euclidean, &[]).is_err());
        assert!(FlatFactory
            .train(2, Similarity::Euclidean, &[1.0; 3])
            .is_err());
        assert!(matches!(
            FlatFactory.read(3, Similarity::Euclidean, &[1]),
            Err(Error::CorruptState(_))
        ));
        assert!(FlatFactory.read(0, Similarity::Euclidean, &[]).is_err());
    }
}
