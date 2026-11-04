//! `org.apache.lucene.classification.utils.NearestFuzzyQuery`: each token of
//! the input expanded to its fuzzy variants (one edit, a two-character
//! prefix), the variants scored by similarity and idf, the best 300 kept.

use std::sync::Arc;

use lucene_analysis::Analyzer;

use super::TrainingIndex;
use lucene_search::fuzzy_like_this::{FuzzyLikeThisQuery, FuzzyReader};
use lucene_search::query::BooleanQuery;
use lucene_search::Result;

/// `(int) MIN_SIMILARITY`: the edits a variant may be away.
const MAX_EDITS: f32 = 1.0;
/// `PREFIX_LENGTH`.
const PREFIX_LENGTH: i32 = 2;
/// `MAX_NUM_TERMS`.
const MAX_NUM_TERMS: i32 = 300;

/// `NearestFuzzyQuery`: the fuzzy variants of the analysed input of each
/// added field, rewritten to a `BooleanQuery` of their term queries -- the
/// rewrite of a [`FuzzyLikeThisQuery`] of 300 terms, one edit and a
/// two-character prefix, which Java's class copies.
pub struct NearestFuzzyQuery {
    analyzer: Arc<Analyzer>,
    /// `fieldVals`: (field, query string).
    field_vals: Vec<(String, String)>,
}

impl NearestFuzzyQuery {
    /// `new NearestFuzzyQuery(analyzer)`.
    pub fn new(analyzer: Arc<Analyzer>) -> Self {
        Self {
            analyzer,
            field_vals: Vec::new(),
        }
    }

    /// `addTerms(queryString, fieldName)`.
    pub fn add_terms(&mut self, query_string: &str, field: &str) {
        self.field_vals
            .push((field.to_string(), query_string.to_string()));
    }

    /// `rewrite(searcher)`: the top variants of every token, grouped by the
    /// token they vary -- a lone variant a boosted `TermQuery`, several a
    /// `BooleanQuery` of them -- each term scored as if its `docFreq` were 1.
    /// The groups come in `HashMap<Term, _>` order, which is the order the
    /// scores of a document are summed in.
    ///
    /// # Errors
    /// What reading the index or analysing the input reports.
    pub fn rewrite(&self, index: &TrainingIndex<'_, '_>) -> Result<BooleanQuery> {
        let mut flt = FuzzyLikeThisQuery::new(MAX_NUM_TERMS, Arc::clone(&self.analyzer));
        for (field, text) in &self.field_vals {
            flt.add_terms(text, Some(field), MAX_EDITS, PREFIX_LENGTH)?;
        }
        flt.rewrite_over(&FuzzyReader {
            segments: index.segments,
            num_docs: index.reader.num_docs(),
            hash_seed: index.hash_seed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_one_field_value_per_call() {
        let mut q = NearestFuzzyQuery::new(Arc::new(Analyzer::standard(None)));
        q.add_terms("a b", "f");
        q.add_terms("c", "g");
        assert_eq!(q.field_vals.len(), 2);
    }
}
