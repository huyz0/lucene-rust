//! `org.apache.lucene.classification.utils.NearestFuzzyQuery`: each token of
//! the input expanded to its fuzzy variants (one edit, a two-character
//! prefix), the variants scored by similarity and idf, the best 300 kept.

use std::sync::Arc;

use lucene_analysis::Analyzer;

use super::TrainingIndex;
use crate::fuzzy_like_this::{FuzzyLikeThisQuery, FuzzyReader};
use crate::query::BooleanQuery;
use crate::Result;

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
    fn priority_queue_pops_least_first_and_overflows_like_java() {
        use crate::java_priority_queue::JavaPriorityQueue;
        let mut q = JavaPriorityQueue::new(3, |a: &i32, b: &i32| a < b);
        assert!(q.top().is_none());
        assert!(q.pop().is_none());
        for v in [5, 1, 4, 2, 3] {
            q.insert_with_overflow(v);
        }
        // 1 was dropped when 2 arrived, then 2 when 3 did.
        assert_eq!(q.size(), 3);
        assert_eq!(q.top(), Some(&3));
        let popped: Vec<i32> = std::iter::from_fn(|| q.pop()).collect();
        assert_eq!(popped, vec![3, 4, 5]);
        // the slots a pop frees are reused
        q.insert_with_overflow(9);
        q.insert_with_overflow(8);
        assert_eq!(
            std::iter::from_fn(|| q.pop()).collect::<Vec<_>>(),
            vec![8, 9]
        );
        // An element less than the top is not taken by a full queue.
        let mut q = JavaPriorityQueue::new(1, |a: &i32, b: &i32| a < b);
        q.insert_with_overflow(5);
        q.insert_with_overflow(4);
        assert_eq!(q.pop(), Some(5));
        // A zero-sized queue holds nothing.
        let mut q = JavaPriorityQueue::new(0, |a: &i32, b: &i32| a < b);
        q.insert_with_overflow(1);
        assert_eq!(q.size(), 0);
    }

    #[test]
    fn adds_one_field_value_per_call() {
        let mut q = NearestFuzzyQuery::new(Arc::new(Analyzer::standard(None)));
        q.add_terms("a b", "f");
        q.add_terms("c", "g");
        assert_eq!(q.field_vals.len(), 2);
    }
}
