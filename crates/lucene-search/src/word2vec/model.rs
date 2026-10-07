//! `Word2VecModel` (and `TermAndVector.normalizeVector`, `TermAndBoost`):
//! the terms of a word2vec model and their L2-normalised vectors, by ordinal
//! and by term.

use std::collections::HashMap;

use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_codecs::hnsw::{UpdateableVectorScorer, VectorScorer};
use lucene_util::vector_util::l2normalize;

use crate::{Error, Result};

/// `org.apache.lucene.analysis.synonym.word2vec.TermAndBoost`: a synonym and
/// its similarity to the term it was found for.
#[derive(Debug, Clone, PartialEq)]
pub struct TermAndBoost {
    /// The synonym's bytes.
    pub term: Vec<u8>,
    /// Its `DOT_PRODUCT` similarity, `(1 + dot) / 2`.
    pub boost: f32,
}

/// `org.apache.lucene.analysis.synonym.word2vec.Word2VecModel`.
///
/// Java's term ordinals come from a `BytesRefHash`, which numbers only
/// *new* terms, while vectors are stored in load order: after a duplicate
/// term the two numberings drift apart and `vectorValue(term)` answers with
/// another entry's vector. Kept, as Lucene's behaviour.
#[derive(Debug, Clone)]
pub struct Word2VecModel {
    dictionary_size: usize,
    vector_dimension: usize,
    /// `termsAndVectors`, in load order.
    entries: Vec<(Vec<u8>, Vec<f32>)>,
    /// `word2Vec`: each distinct term's `BytesRefHash` id.
    ids: HashMap<Vec<u8>, usize>,
}

impl Word2VecModel {
    /// `new Word2VecModel(dictionarySize, vectorDimension)`.
    pub fn new(dictionary_size: usize, vector_dimension: usize) -> Self {
        Word2VecModel {
            dictionary_size,
            vector_dimension,
            entries: Vec::new(),
            ids: HashMap::new(),
        }
    }

    /// `addTermAndVector(new TermAndVector(term, vector))`: stores the
    /// L2-normalised vector (`VectorUtil.l2normalize`).
    ///
    /// # Errors
    /// `IllegalArgument` for a zero vector, and past `dictionarySize` terms
    /// (Java's `ArrayIndexOutOfBoundsException`).
    pub fn add_term_and_vector(&mut self, term: Vec<u8>, mut vector: Vec<f32>) -> Result<()> {
        if self.entries.len() >= self.dictionary_size {
            return Err(Error::IllegalArgument(format!(
                "Word2Vec model declares {} terms and has more",
                self.dictionary_size
            )));
        }
        l2normalize(&mut vector, true).map_err(|e| Error::IllegalArgument(e.to_string()))?;
        let next = self.ids.len();
        self.ids.entry(term.clone()).or_insert(next);
        self.entries.push((term, vector));
        Ok(())
    }

    /// `vectorValue(int targetOrd)`.
    pub fn vector(&self, ord: usize) -> &[f32] {
        &self.entries[ord].1
    }

    /// `vectorValue(BytesRef term)`: `None` for a term not in the model.
    pub fn vector_of(&self, term: &[u8]) -> Option<&[f32]> {
        let id = *self.ids.get(term)?;
        self.entries.get(id).map(|e| e.1.as_slice())
    }

    /// `termValue(int targetOrd)`.
    pub fn term(&self, ord: usize) -> &[u8] {
        &self.entries[ord].0
    }

    /// `dimension()`.
    pub fn dimension(&self) -> usize {
        self.vector_dimension
    }

    /// `size()`: the declared dictionary size.
    pub fn size(&self) -> usize {
        self.dictionary_size
    }

    /// Terms loaded so far (`loadedCount`).
    pub fn loaded(&self) -> usize {
        self.entries.len()
    }
}

/// `DefaultFlatVectorScorer`'s `DOT_PRODUCT` scorer over a model, in both
/// roles: against an ordinal (graph construction) or an outside query.
pub(super) struct ModelScorer<'a> {
    pub(super) model: &'a Word2VecModel,
    pub(super) query: Vec<f32>,
}

impl VectorScorer for ModelScorer<'_> {
    fn score(&mut self, node: i32) -> lucene_codecs::vectors::Result<f32> {
        let v = self.model.vector(node as usize);
        Ok(VectorSimilarityFunction::DotProduct.score(&self.query, v))
    }

    fn max_ord(&self) -> i32 {
        self.model.loaded() as i32
    }
}

impl UpdateableVectorScorer for ModelScorer<'_> {
    fn set_scoring_ordinal(&mut self, ord: i32) -> lucene_codecs::vectors::Result<()> {
        self.query.clear();
        self.query
            .extend_from_slice(self.model.vector(ord as usize));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_terms_shift_lookups_as_in_java() {
        let mut m = Word2VecModel::new(3, 2);
        m.add_term_and_vector(b"a".to_vec(), vec![3.0, 4.0])
            .unwrap();
        m.add_term_and_vector(b"a".to_vec(), vec![1.0, 0.0])
            .unwrap();
        m.add_term_and_vector(b"c".to_vec(), vec![0.0, 2.0])
            .unwrap();
        assert_eq!(m.vector(0), [0.6, 0.8]);
        // `c` is BytesRefHash id 1, which is the second `a`'s slot.
        assert_eq!(m.vector_of(b"c"), Some(&[1.0, 0.0][..]));
        assert_eq!(m.vector_of(b"x"), None);
        assert_eq!(
            (m.term(2), m.dimension(), m.size(), m.loaded()),
            (&b"c"[..], 2, 3, 3)
        );
        assert!(m
            .add_term_and_vector(b"d".to_vec(), vec![1.0, 1.0])
            .is_err());
        let mut z = Word2VecModel::new(1, 2);
        assert!(z
            .add_term_and_vector(b"z".to_vec(), vec![0.0, 0.0])
            .is_err());
    }
}
