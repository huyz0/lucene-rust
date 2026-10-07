//! Port of `org.apache.lucene.analysis.synonym.word2vec` (analysis-common):
//! synonyms by vector similarity. A [`Word2VecModel`] (read by
//! [`read_dl4j_model`]) is indexed in an HNSW graph
//! ([`Word2VecSynonymProvider`]), and [`Word2VecSynonymFilter`] stacks each
//! token's nearest terms on it.
//!
//! Here rather than in `lucene-analysis` because the provider is
//! `HnswGraphBuilder` and `HnswGraphSearcher` over the `DOT_PRODUCT`
//! similarity, which live in `lucene-codecs` ([`lucene_codecs::hnsw`]) --
//! the same graph Java builds, arc for arc, from the same seed.

mod dl4j;
mod factory;
mod model;

use std::collections::VecDeque;
use std::sync::Arc;

use lucene_analysis::attributes::State;
use lucene_analysis::synonym::TYPE_SYNONYM;
use lucene_analysis::{AnalysisError, AttributeSource, TokenStream, Tokenizer};
use lucene_codecs::hnsw::{
    HnswGraphBuilder, HnswGraphSearcher, KnnCollector, OnHeapHnswGraph, DEFAULT_BEAM_WIDTH,
    DEFAULT_MAX_CONN, DEFAULT_RAND_SEED,
};

pub use dl4j::read_dl4j_model;
pub use factory::{
    get_synonym_provider, register_factories, Word2VecSynonymFilterFactory,
    DEFAULT_MAX_SYNONYMS_PER_TERM, DEFAULT_MIN_ACCEPTED_SIMILARITY,
};
use model::ModelScorer;
pub use model::{TermAndBoost, Word2VecModel};

use crate::{Error, Result};

/// `org.apache.lucene.analysis.synonym.word2vec.Word2VecSynonymProvider`.
#[derive(Debug)]
pub struct Word2VecSynonymProvider {
    model: Word2VecModel,
    graph: OnHeapHnswGraph,
}

impl Word2VecSynonymProvider {
    /// `new Word2VecSynonymProvider(model)`: the model's HNSW graph
    /// (`DEFAULT_MAX_CONN`, `DEFAULT_BEAM_WIDTH`, `HnswGraphBuilder.randSeed`).
    ///
    /// # Errors
    /// `IllegalArgument` when the model holds fewer terms than it declares
    /// (Java fails building on the missing vectors).
    pub fn new(model: Word2VecModel) -> Result<Self> {
        if model.loaded() != model.size() {
            return Err(Error::IllegalArgument(format!(
                "Word2Vec model declares {} terms but has {}",
                model.size(),
                model.loaded()
            )));
        }
        let scorer = ModelScorer {
            model: &model,
            query: Vec::new(),
        };
        let builder = HnswGraphBuilder::new(
            scorer,
            DEFAULT_MAX_CONN,
            DEFAULT_BEAM_WIDTH,
            DEFAULT_RAND_SEED,
        )?;
        let graph = builder.build(model.size() as i32)?;
        Ok(Word2VecSynonymProvider { model, graph })
    }

    /// The model.
    pub fn model(&self) -> &Word2VecModel {
        &self.model
    }

    /// `getSynonyms(term, maxSynonymsPerTerm, minAcceptedSimilarity)`: the
    /// `maxSynonymsPerTerm + 1` nearest terms (the term itself among them),
    /// most similar first, without the term and below the threshold.
    ///
    /// # Errors
    /// A failed graph search.
    pub fn synonyms(
        &self,
        term: &[u8],
        max_synonyms_per_term: i32,
        min_accepted_similarity: f32,
    ) -> Result<Vec<TermAndBoost>> {
        let Some(query) = self.model.vector_of(term) else {
            return Ok(Vec::new());
        };
        let mut scorer = ModelScorer {
            model: &self.model,
            query: query.to_vec(),
        };
        // `max + 1`, in Java's int arithmetic; a non-positive `k` finds nothing.
        let k = usize::try_from(max_synonyms_per_term.wrapping_add(1)).unwrap_or(0);
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut collector = KnnCollector::new(k, i32::MAX as u64);
        let mut searcher = HnswGraphSearcher::new(k, self.model.size() as i32);
        searcher.search(&mut collector, &mut scorer, &self.graph, None)?;
        Ok(collector
            .top_docs()
            .into_iter()
            .filter_map(|(ord, similarity)| {
                let synonym = self.model.term(ord as usize);
                (synonym != term && similarity >= min_accepted_similarity).then(|| TermAndBoost {
                    term: synonym.to_vec(),
                    boost: similarity,
                })
            })
            .collect())
    }
}

/// `org.apache.lucene.analysis.synonym.word2vec.Word2VecSynonymFilter`:
/// after each token, its synonyms at the same position, typed
/// `SYNONYM`.
pub struct Word2VecSynonymFilter<I> {
    input: I,
    provider: Arc<Word2VecSynonymProvider>,
    max_synonyms_per_term: i32,
    min_accepted_similarity: f32,
    synonym_buffer: VecDeque<TermAndBoost>,
    last_state: Option<State>,
}

impl<I: TokenStream> Word2VecSynonymFilter<I> {
    /// `new Word2VecSynonymFilter(input, provider, maxSynonymsPerTerm,
    /// minAcceptedSimilarity)`.
    pub fn new(
        input: I,
        provider: Arc<Word2VecSynonymProvider>,
        max_synonyms_per_term: i32,
        min_accepted_similarity: f32,
    ) -> Self {
        Word2VecSynonymFilter {
            input,
            provider,
            max_synonyms_per_term,
            min_accepted_similarity,
            synonym_buffer: VecDeque::new(),
            last_state: None,
        }
    }
}

impl<I: TokenStream> TokenStream for Word2VecSynonymFilter<I> {
    fn attributes(&self) -> &AttributeSource {
        self.input.attributes()
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        self.input.attributes_mut()
    }

    // Java: Word2VecSynonymFilter.incrementToken
    fn increment_token(&mut self) -> std::result::Result<bool, AnalysisError> {
        if let Some(synonym) = self.synonym_buffer.pop_front() {
            let a = self.input.attributes_mut();
            a.clear_attributes();
            if let Some(state) = &self.last_state {
                a.restore_state(state);
            }
            a.set_term(&String::from_utf8_lossy(&synonym.term));
            a.set_token_type(TYPE_SYNONYM);
            a.set_position_length(1)?;
            a.set_position_increment(0)?;
            return Ok(true);
        }
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes();
        let synonyms = self
            .provider
            .synonyms(
                a.term().as_bytes(),
                self.max_synonyms_per_term,
                self.min_accepted_similarity,
            )
            .map_err(|e| AnalysisError::Io(e.to_string()))?;
        if !synonyms.is_empty() {
            self.last_state = Some(a.capture_state());
            self.synonym_buffer.extend(synonyms);
        }
        Ok(true)
    }

    fn reset(&mut self) -> std::result::Result<(), AnalysisError> {
        self.input.reset()?;
        self.synonym_buffer.clear();
        Ok(())
    }

    fn end(&mut self) -> std::result::Result<(), AnalysisError> {
        self.input.end()
    }

    fn close(&mut self) -> std::result::Result<(), AnalysisError> {
        self.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        self.input.as_tokenizer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(terms: &[(&str, [f32; 2])]) -> Word2VecModel {
        let mut m = Word2VecModel::new(terms.len(), 2);
        for (t, v) in terms {
            m.add_term_and_vector(t.as_bytes().to_vec(), v.to_vec())
                .unwrap();
        }
        m
    }

    #[test]
    fn provider_edges() {
        let mut short = Word2VecModel::new(2, 2);
        short
            .add_term_and_vector(b"a".to_vec(), vec![1.0, 0.0])
            .unwrap();
        let e = Word2VecSynonymProvider::new(short).unwrap_err().to_string();
        assert!(e.contains("declares 2 terms but has 1"), "{e}");
        let p = Word2VecSynonymProvider::new(model(&[])).unwrap();
        assert!(p.synonyms(b"a", 3, 0.0).unwrap().is_empty());
        let p =
            Word2VecSynonymProvider::new(model(&[("a", [1.0, 0.0]), ("b", [1.0, 0.1])])).unwrap();
        assert_eq!(p.model().size(), 2);
        assert!(p.synonyms(b"a", -1, 0.0).unwrap().is_empty());
        assert!(p.synonyms(b"a", i32::MIN, 0.0).unwrap().is_empty());
        let s = p.synonyms(b"a", 1, 0.0).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].term, b"b");
        assert!(p.synonyms(b"a", 1, 1.0).unwrap().is_empty());
    }
}
