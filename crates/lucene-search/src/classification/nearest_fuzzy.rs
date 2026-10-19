//! `org.apache.lucene.classification.utils.NearestFuzzyQuery`: each token of
//! the input expanded to its fuzzy variants (one edit, a two-character
//! prefix), the variants scored by similarity and idf, the best 300 kept.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_codecs::fuzzy::FuzzyMatch;
use lucene_util::strict_math::log;

use super::{analyze, TrainingIndex};
use crate::java_priority_queue::JavaPriorityQueue;
use crate::mlt::{java_hash_order_of, java_string_hash};
use crate::query::{BooleanQuery, BoostQuery, Clause, TermQuery};
use crate::Result;

/// `MAX_VARIANTS_PER_TERM`.
const MAX_VARIANTS_PER_TERM: usize = 50;
/// `(int) MIN_SIMILARITY`: the edits a variant may be away.
const MAX_EDITS: u8 = 1;
/// `PREFIX_LENGTH`.
const PREFIX_LENGTH: usize = 2;
/// `MAX_NUM_TERMS`.
const MAX_NUM_TERMS: usize = 300;

/// `ScoreTerm`: a variant, its score (the similarity, then the similarity
/// squared times idf) and the token it varies.
#[derive(Debug, Clone)]
struct ScoreTerm {
    field: String,
    term: Vec<u8>,
    score: f32,
    source: Vec<u8>,
}

/// `ScoreTermQueue.lessThan`: lower score first; of equal scores the later
/// term (`Term.compareTo`: field, then bytes).
fn score_term_less(a: &ScoreTerm, b: &ScoreTerm) -> bool {
    if a.score == b.score {
        let by_field = super::java_string_cmp(&a.field, &b.field);
        by_field.then_with(|| a.term.cmp(&b.term)) == std::cmp::Ordering::Greater
    } else {
        a.score < b.score
    }
}

/// A `ScoreTerm` of one token's variants, borrowing the matched term and the
/// token: most candidates lose their place in the queue, and an owned copy
/// is made only for the ones that keep it.
struct Candidate<'a> {
    field: &'a str,
    term: &'a [u8],
    score: f32,
    source: &'a [u8],
}

/// `score_term_less` over borrowed candidates.
fn candidate_less(a: &Candidate<'_>, b: &Candidate<'_>) -> bool {
    if a.score == b.score {
        let by_field = super::java_string_cmp(a.field, b.field);
        by_field.then_with(|| a.term.cmp(b.term)) == std::cmp::Ordering::Greater
    } else {
        a.score < b.score
    }
}

/// `Term.hashCode()`: `31 * (31 + field.hashCode()) + bytes.hashCode()`.
fn term_hash(index: &TrainingIndex<'_, '_>, field: &str, bytes: &[u8]) -> i32 {
    31i32
        .wrapping_add(java_string_hash(field))
        .wrapping_mul(31)
        .wrapping_add(index.bytes_hash(bytes))
}

/// `NearestFuzzyQuery`: the fuzzy variants of the analysed input of each
/// added field, rewritten to a `BooleanQuery` of their term queries.
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
        let mut q: JavaPriorityQueue<ScoreTerm, fn(&ScoreTerm, &ScoreTerm) -> bool> =
            JavaPriorityQueue::new(MAX_NUM_TERMS, score_term_less);
        for (field, text) in &self.field_vals {
            self.add_variants(index, field, text, &mut q)?;
        }
        // Step 1: the variants by the token they vary, in pop order.
        let mut sources: Vec<Vec<u8>> = Vec::new();
        let mut groups: Vec<Vec<ScoreTerm>> = Vec::new();
        let mut source_fields: Vec<String> = Vec::new();
        while let Some(st) = q.pop() {
            match sources
                .iter()
                .zip(&source_fields)
                .position(|(s, f)| *s == st.source && *f == st.field)
            {
                Some(i) => groups[i].push(st),
                None => {
                    sources.push(st.source.clone());
                    source_fields.push(st.field.clone());
                    groups.push(vec![st]);
                }
            }
        }
        // Step 2: one clause per token, in `HashMap` iteration order.
        let hashes: Vec<i32> = sources
            .iter()
            .zip(&source_fields)
            .map(|(s, f)| term_hash(index, f, s))
            .collect();
        let mut bq = BooleanQuery::default();
        for i in java_hash_order_of(&hashes) {
            let variants = &groups[i];
            let clause = |st: &ScoreTerm| {
                Clause::Boost(Box::new(BoostQuery::new(
                    TermQuery::new(st.field.as_str(), st.term.as_slice()).with_doc_freq(1),
                    st.score,
                )))
            };
            if let [only] = variants.as_slice() {
                bq.should.push(clause(only));
            } else {
                bq.should.push(Clause::Boolean(Box::new(BooleanQuery {
                    should: variants.iter().map(clause).collect(),
                    ..Default::default()
                })));
            }
        }
        Ok(bq)
    }

    /// `addTerms(reader, fieldVals, queue)`.
    fn add_variants(
        &self,
        index: &TrainingIndex<'_, '_>,
        field: &str,
        text: &str,
        q: &mut JavaPriorityQueue<ScoreTerm, fn(&ScoreTerm, &ScoreTerm) -> bool>,
    ) -> Result<()> {
        let leaves: Vec<_> = index
            .segments
            .iter()
            .filter_map(|s| s.fields.field(field))
            .collect();
        if leaves.is_empty() {
            return Ok(());
        }
        let corpus_num_docs = index.reader.num_docs();
        let mut processed = HashSet::new();
        for token in analyze(&self.analyzer, field, text)? {
            if !processed.insert(token.clone()) {
                continue;
            }
            let pattern = FuzzyMatch::new(token.as_bytes(), MAX_EDITS, PREFIX_LENGTH, true);
            // `FuzzyTermsEnum` over `MultiTerms`: the union of every leaf's
            // matches in term order, each with its summed `docFreq`.
            let mut matches: BTreeMap<Vec<u8>, (usize, i32)> = BTreeMap::new();
            for leaf in &leaves {
                let mut walk = leaf.fuzzy_intersect(&pattern);
                while let Some(next) = walk.next() {
                    let (term, stats) = next?;
                    let ed = walk.last_edits();
                    let e = matches.entry(term).or_insert((ed, 0));
                    e.1 = e.1.wrapping_add(stats.doc_freq);
                }
            }
            let mut variants: JavaPriorityQueue<
                Candidate<'_>,
                fn(&Candidate<'_>, &Candidate<'_>) -> bool,
            > = JavaPriorityQueue::new(MAX_VARIANTS_PER_TERM, candidate_less);
            let mut min_score = 0f32;
            let mut df = index.reader.doc_freq(field, token.as_bytes())? as i32;
            let mut num_variants = 0i32;
            let mut total_variant_doc_freqs = 0i32;
            // `FuzzyTermsEnum`'s `MaxNonCompetitiveBoostAttribute` loop: a
            // changed bottom seen when a term is returned is acted on at the
            // next `next()`, dropping the edit budget while no variant at it
            // could still compete (`bottomTerm` is never set, so `termAfter`).
            let term_length = pattern.term_length() as f32;
            let mut max_edits = MAX_EDITS;
            let mut seen_bottom = f32::NEG_INFINITY;
            let mut published = f32::NEG_INFINITY;
            let mut queued = false;
            for (term, &(ed, term_df)) in &matches {
                if queued {
                    while max_edits > 0 {
                        let max_boost = 1.0f32 - f32::from(max_edits) / term_length;
                        if seen_bottom < max_boost {
                            break;
                        }
                        max_edits -= 1;
                    }
                    queued = false;
                }
                if ed > usize::from(max_edits) {
                    continue;
                }
                if published != seen_bottom {
                    seen_bottom = published;
                    queued = true;
                }
                num_variants += 1;
                total_variant_doc_freqs = total_variant_doc_freqs.wrapping_add(term_df);
                let score = pattern.boost_from_edits(term, ed);
                if variants.size() < MAX_VARIANTS_PER_TERM || score > min_score {
                    variants.insert_with_overflow(Candidate {
                        field,
                        term,
                        score,
                        source: token.as_bytes(),
                    });
                    min_score = variants.top().map_or(min_score, |t| t.score);
                }
                published = if variants.size() >= MAX_VARIANTS_PER_TERM {
                    min_score
                } else {
                    f32::NEG_INFINITY
                };
            }
            if num_variants > 0 {
                let avg_df = total_variant_doc_freqs / num_variants;
                if df == 0 {
                    df = avg_df;
                }
                let idf = idf(df, corpus_num_docs);
                while let Some(st) = variants.pop() {
                    q.insert_with_overflow(ScoreTerm {
                        field: st.field.to_string(),
                        term: st.term.to_vec(),
                        score: (st.score * st.score) * idf,
                        source: st.source.to_vec(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// `idf(docFreq, docCount)`: `(float) (log((docCount + 1) / (double)
/// (docFreq + 1)) + 1.0)`.
fn idf(doc_freq: i32, doc_count: i32) -> f32 {
    (log(f64::from(doc_count.wrapping_add(1)) / f64::from(doc_freq.wrapping_add(1))) + 1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_queue_pops_least_first_and_overflows_like_java() {
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
    fn score_terms_order_by_score_then_later_term_first() {
        let st = |field: &str, term: &str, score| ScoreTerm {
            field: field.into(),
            term: term.as_bytes().to_vec(),
            score,
            source: Vec::new(),
        };
        assert!(score_term_less(&st("f", "a", 0.5), &st("f", "a", 0.6)));
        assert!(!score_term_less(&st("f", "a", 0.6), &st("f", "a", 0.5)));
        assert!(score_term_less(&st("f", "b", 0.5), &st("f", "a", 0.5)));
        assert!(!score_term_less(&st("f", "a", 0.5), &st("f", "b", 0.5)));
        assert!(score_term_less(&st("g", "a", 0.5), &st("f", "z", 0.5)));
    }

    #[test]
    fn idf_is_lucenes_classic_formula() {
        assert_eq!(idf(0, 0), 1.0);
        assert_eq!(idf(1, 9), (log(10.0 / 2.0) + 1.0) as f32);
    }
}
