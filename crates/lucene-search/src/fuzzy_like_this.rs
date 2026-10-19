//! `org.apache.lucene.sandbox.queries.FuzzyLikeThisQuery`: each token of the
//! analysed input of every added field expanded to its fuzzy variants, the
//! variants scored by their similarity and the source token's idf, the best
//! `maxNumTerms` of them kept and rewritten to a `BooleanQuery`.
//!
//! The engine is shared with the classification module's
//! `NearestFuzzyQuery`, which is this query with fixed settings (one edit, a
//! two-character prefix, 300 terms).
//!
//! **The hash seed.** The rewrite groups the variants by source term in a
//! `HashMap<Term, _>`, and the groups become the boolean's clauses in that
//! map's iteration order -- the order a document's clause scores are summed
//! in. A `Term`'s hash includes `BytesRef.hashCode()`, which is seeded with
//! `StringHelper.GOOD_FAST_HASH_SEED`: `tests.seed`'s `hashCode()` for a JVM
//! run with one, else the clock. [`FuzzyLikeThisQuery::hash_seed`] is that
//! seed (default `0`); the clause order, and so the last bits of a score,
//! follow Java only under the seed the JVM used.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use lucene_analysis::attributes::java_float_to_string;
use lucene_analysis::Analyzer;
use lucene_codecs::fuzzy::{FuzzyMatch, MAXIMUM_SUPPORTED_DISTANCE};
use lucene_util::strict_math::log;

use crate::index_searcher::IndexSearcher;
use crate::java_priority_queue::JavaPriorityQueue;
use crate::mlt::{java_hash_order_of, java_string_hash};
use crate::multi_segment::OpenSegment;
use crate::query::{BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, TermQuery};
use crate::{Error, Result};

/// `MAX_VARIANTS_PER_TERM`.
const MAX_VARIANTS_PER_TERM: usize = 50;

/// `ArrayUtil.MAX_ARRAY_LENGTH`: `Integer.MAX_VALUE` less a 16-byte array
/// header, the bound `PriorityQueue`'s constructor checks `maxSize` against.
const MAX_ARRAY_LENGTH: i32 = i32::MAX - 16;

/// `FieldVals`: one `addTerms` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldVals {
    /// `queryString`.
    pub query_string: String,
    /// `fieldName`, `null` when the XML builder found none to inherit.
    pub field_name: Option<String>,
    /// `maxEdits`: `(int) minSimilarity`, 0 to 2.
    pub max_edits: u8,
    /// `prefixLength`, checked (`>= 0`) only when the rewrite builds the
    /// fuzzy automaton, as Java's `FuzzyAutomatonBuilder` does.
    pub prefix_length: i32,
}

/// `FuzzyLikeThisQuery`: rewritten by the searcher before it is searched.
#[derive(Clone)]
pub struct FuzzyLikeThisQuery {
    /// `fieldVals`, in the order they were added.
    pub field_vals: Vec<FieldVals>,
    pub analyzer: Arc<Analyzer>,
    /// `maxNumTerms`: the clauses the rewrite keeps across every field.
    pub max_num_terms: i32,
    /// `ignoreTF`: each variant a `ConstantScoreQuery` rather than a
    /// `TermQuery` scoring as if its `docFreq` were 1.
    pub ignore_tf: bool,
    /// `StringHelper.GOOD_FAST_HASH_SEED` (see the module doc).
    pub hash_seed: i32,
}

impl std::fmt::Debug for FuzzyLikeThisQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuzzyLikeThisQuery")
            .field("field_vals", &self.field_vals)
            .field("max_num_terms", &self.max_num_terms)
            .field("ignore_tf", &self.ignore_tf)
            .field("hash_seed", &self.hash_seed)
            .finish_non_exhaustive()
    }
}

impl PartialEq for FuzzyLikeThisQuery {
    /// `equals`: the analyzer by identity, the field values, `ignoreTF` and
    /// `maxNumTerms` (the hash seed is the JVM's, not the query's).
    fn eq(&self, o: &Self) -> bool {
        Arc::ptr_eq(&self.analyzer, &o.analyzer)
            && self.field_vals == o.field_vals
            && self.ignore_tf == o.ignore_tf
            && self.max_num_terms == o.max_num_terms
    }
}

impl std::fmt::Display for FuzzyLikeThisQuery {
    /// `toString(field)` returns `null`, which prints as `null`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("null")
    }
}

impl From<FuzzyLikeThisQuery> for Clause {
    fn from(q: FuzzyLikeThisQuery) -> Self {
        Clause::Extended(Box::new(
            crate::extended_query::ExtendedQuery::FuzzyLikeThis(q),
        ))
    }
}

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
    term_less(&a.field, &a.term, a.score, &b.field, &b.term, b.score)
}

fn term_less(af: &str, at: &[u8], ascore: f32, bf: &str, bt: &[u8], bscore: f32) -> bool {
    if ascore == bscore {
        let by_field = crate::classification::java_string_cmp(af, bf);
        by_field.then_with(|| at.cmp(bt)) == std::cmp::Ordering::Greater
    } else {
        ascore < bscore
    }
}

/// A `ScoreTerm` of one token's variants, borrowing the matched term: most
/// candidates lose their place in the queue, and an owned copy is made only
/// for the ones that keep it.
struct Candidate<'a> {
    field: &'a str,
    term: &'a [u8],
    score: f32,
}

fn candidate_less(a: &Candidate<'_>, b: &Candidate<'_>) -> bool {
    term_less(a.field, a.term, a.score, b.field, b.term, b.score)
}

type TermQueue = JavaPriorityQueue<ScoreTerm, fn(&ScoreTerm, &ScoreTerm) -> bool>;

/// The reader a rewrite reads: its segments, `numDocs()`, and the hash seed
/// `Term.hashCode()` is computed under.
pub(crate) struct FuzzyReader<'s, 'a> {
    pub(crate) segments: &'s [OpenSegment<'a>],
    pub(crate) num_docs: i32,
    pub(crate) hash_seed: i32,
}

impl FuzzyReader<'_, '_> {
    /// `IndexReader.docFreq(term)`: deleted documents included.
    fn doc_freq(&self, field: &str, term: &[u8]) -> Result<i32> {
        let mut df = 0i32;
        for seg in self.segments {
            if let Some(ft) = seg.fields.field(field) {
                if let Some(stats) = ft.try_seek_exact(term)? {
                    df = df.wrapping_add(stats.doc_freq);
                }
            }
        }
        Ok(df)
    }

    /// `Term.hashCode()`: `31 * (31 + field.hashCode()) + bytes.hashCode()`.
    fn term_hash(&self, field: &str, bytes: &[u8]) -> i32 {
        31i32
            .wrapping_add(java_string_hash(field))
            .wrapping_mul(31)
            .wrapping_add(lucene_util::string_helper::murmurhash3_x86_32(
                bytes,
                self.hash_seed,
            ))
    }
}

/// `IndexReader.numDocs()` of the searcher's segments: each segment's
/// `maxDoc` less its deletions.
fn num_docs(segments: &[OpenSegment<'_>]) -> i32 {
    segments
        .iter()
        .map(|s| {
            let max_doc = s.max_doc.unwrap_or(0);
            match s.live_docs {
                Some(live) => i32::try_from(live.cardinality()).unwrap_or(i32::MAX),
                None => max_doc,
            }
        })
        .fold(0i32, i32::wrapping_add)
}

impl FuzzyLikeThisQuery {
    /// `new FuzzyLikeThisQuery(maxNumTerms, analyzer)`.
    pub fn new(max_num_terms: i32, analyzer: Arc<Analyzer>) -> Self {
        Self {
            field_vals: Vec::new(),
            analyzer,
            max_num_terms,
            ignore_tf: false,
            hash_seed: 0,
        }
    }

    /// `addTerms(queryString, fieldName, minSimilarity, prefixLength)`.
    ///
    /// # Errors
    /// `IllegalArgumentException` for a `minSimilarity` that is not a whole
    /// number of edits from 0 to 2.
    pub fn add_terms(
        &mut self,
        query_string: &str,
        field_name: Option<&str>,
        min_similarity: f32,
        prefix_length: i32,
    ) -> Result<()> {
        // `(int) minSimilarity`: NaN casts to 0, and the comparison fails
        let max_edits = min_similarity as i32;
        if max_edits as f32 != min_similarity
            || max_edits < 0
            || max_edits > i32::from(MAXIMUM_SUPPORTED_DISTANCE)
        {
            return Err(Error::IllegalArgument(format!(
                "minSimilarity must integer value between 0 and {MAXIMUM_SUPPORTED_DISTANCE}, \
                 inclusive; got {}",
                java_float_to_string(min_similarity)
            )));
        }
        self.field_vals.push(FieldVals {
            query_string: query_string.to_string(),
            field_name: field_name.map(str::to_string),
            max_edits: u8::try_from(max_edits).unwrap_or(MAXIMUM_SUPPORTED_DISTANCE),
            prefix_length,
        });
        Ok(())
    }

    /// `rewrite(searcher)`.
    ///
    /// # Errors
    /// `IllegalArgumentException` for a `maxNumTerms` the priority queue
    /// refuses or a negative prefix length; what reading the index or
    /// analysing the input reports.
    pub fn rewrite(&self, searcher: &IndexSearcher<'_, '_>) -> Result<BooleanQuery> {
        let reader = FuzzyReader {
            segments: searcher.segments(),
            num_docs: num_docs(searcher.segments()),
            hash_seed: self.hash_seed,
        };
        self.rewrite_over(&reader)
    }

    pub(crate) fn rewrite_over(&self, reader: &FuzzyReader<'_, '_>) -> Result<BooleanQuery> {
        if self.max_num_terms < 0 || self.max_num_terms >= MAX_ARRAY_LENGTH {
            return Err(Error::IllegalArgument(format!(
                "maxSize must be >= 0 and < {MAX_ARRAY_LENGTH}; got: {}",
                self.max_num_terms
            )));
        }
        let max_num_terms = usize::try_from(self.max_num_terms).unwrap_or(0);
        let mut q: TermQueue = JavaPriorityQueue::new(max_num_terms, score_term_less);
        for f in &self.field_vals {
            self.add_variants(reader, f, &mut q)?;
        }
        // Step 1: the variants by the token they vary, in pop order.
        let mut sources: Vec<(String, Vec<u8>)> = Vec::new();
        let mut groups: Vec<Vec<ScoreTerm>> = Vec::new();
        while let Some(st) = q.pop() {
            match sources
                .iter()
                .position(|(f, s)| *s == st.source && *f == st.field)
            {
                Some(i) => groups[i].push(st),
                None => {
                    sources.push((st.field.clone(), st.source.clone()));
                    groups.push(vec![st]);
                }
            }
        }
        // Step 2: one clause per token, in `HashMap` iteration order.
        let hashes: Vec<i32> = sources
            .iter()
            .map(|(f, s)| reader.term_hash(f, s))
            .collect();
        let mut bq = BooleanQuery::default();
        for i in java_hash_order_of(&hashes) {
            let variants = &groups[i];
            if let [only] = variants.as_slice() {
                bq.should.push(self.variant_clause(only));
            } else {
                bq.should.push(Clause::Boolean(Box::new(BooleanQuery {
                    should: variants.iter().map(|st| self.variant_clause(st)).collect(),
                    ..Default::default()
                })));
            }
        }
        Ok(bq)
    }

    /// `new BoostQuery(newTermQuery(reader, term), score)`.
    fn variant_clause(&self, st: &ScoreTerm) -> Clause {
        let tq = TermQuery::new(st.field.as_str(), st.term.as_slice());
        let inner = if self.ignore_tf {
            Clause::ConstantScore(Box::new(ConstantScoreQuery::new(tq, 1.0)))
        } else {
            // an artificial `TermStates` whose `docFreq` (and
            // `totalTermFreq`) is 1
            Clause::Term(tq.with_doc_freq(1))
        };
        Clause::Boost(Box::new(BoostQuery::new(inner, st.score)))
    }

    /// `addTerms(reader, fieldVals, queue)`.
    fn add_variants(
        &self,
        reader: &FuzzyReader<'_, '_>,
        f: &FieldVals,
        q: &mut TermQueue,
    ) -> Result<()> {
        // `MultiTerms.getTerms(reader, null)` finds no field
        let Some(field) = f.field_name.as_deref() else {
            return Ok(());
        };
        let leaves: Vec<_> = reader
            .segments
            .iter()
            .filter_map(|s| s.fields.field(field))
            .collect();
        if leaves.is_empty() {
            return Ok(());
        }
        let mut processed = HashSet::new();
        for token in crate::classification::analyze(&self.analyzer, field, &f.query_string)? {
            if !processed.insert(token.clone()) {
                continue;
            }
            // `FuzzyAutomatonBuilder`'s check, made when the enum is built
            let prefix_length = usize::try_from(f.prefix_length)
                .map_err(|_| Error::IllegalArgument("prefixLength cannot be less than 0".into()))?;
            let pattern = FuzzyMatch::new(token.as_bytes(), f.max_edits, prefix_length, true);
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
            let mut df = reader.doc_freq(field, token.as_bytes())?;
            let mut num_variants = 0i32;
            let mut total_variant_doc_freqs = 0i32;
            // `FuzzyTermsEnum`'s `MaxNonCompetitiveBoostAttribute` loop: a
            // changed bottom seen when a term is returned is acted on at the
            // next `next()`, dropping the edit budget while no variant at it
            // could still compete (`bottomTerm` is never set, so `termAfter`).
            // The constructor runs it once with the initial bottom.
            let term_length = pattern.term_length() as f32;
            let mut max_edits = f.max_edits;
            let mut seen_bottom = f32::NEG_INFINITY;
            let mut published = f32::NEG_INFINITY;
            let mut queued = true;
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
                num_variants = num_variants.wrapping_add(1);
                total_variant_doc_freqs = total_variant_doc_freqs.wrapping_add(term_df);
                let score = pattern.boost_from_edits(term, ed);
                if variants.size() < MAX_VARIANTS_PER_TERM || score > min_score {
                    variants.insert_with_overflow(Candidate { field, term, score });
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
                let idf = idf(df, reader.num_docs);
                while let Some(st) = variants.pop() {
                    q.insert_with_overflow(ScoreTerm {
                        field: st.field.to_string(),
                        term: st.term.to_vec(),
                        score: (st.score * st.score) * idf,
                        source: token.as_bytes().to_vec(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// `ClassicSimilarity.idf(docFreq, docCount)`: `(float) (log((docCount + 1)
/// / (double) (docFreq + 1)) + 1.0)`.
fn idf(doc_freq: i32, doc_count: i32) -> f32 {
    (log((f64::from(doc_count) + 1.0) / (f64::from(doc_freq) + 1.0)) + 1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyzer() -> Arc<Analyzer> {
        Arc::new(Analyzer::standard(None))
    }

    #[test]
    fn min_similarity_must_be_a_whole_number_of_edits() {
        let mut q = FuzzyLikeThisQuery::new(50, analyzer());
        q.add_terms("a", Some("f"), 2.0, 1).unwrap();
        q.add_terms("a", None, 0.0, -1).unwrap();
        for bad in [1.5f32, 3.0, -1.0, f32::NAN] {
            let e = q.add_terms("a", Some("f"), bad, 1).unwrap_err();
            assert_eq!(
                e.to_string(),
                Error::IllegalArgument(format!(
                    "minSimilarity must integer value between 0 and 2, inclusive; got {}",
                    java_float_to_string(bad)
                ))
                .to_string()
            );
        }
        assert_eq!(q.field_vals.len(), 2);
        assert_eq!(q.field_vals[0].max_edits, 2);
        assert_eq!(q.field_vals[1].field_name, None);
    }

    #[test]
    fn prints_null_and_compares_by_settings() {
        let a = analyzer();
        let mut q = FuzzyLikeThisQuery::new(50, Arc::clone(&a));
        assert_eq!(q.to_string(), "null");
        let mut other = q.clone();
        other.hash_seed = 7;
        assert_eq!(q, other);
        other.ignore_tf = true;
        assert_ne!(q, other);
        q.add_terms("x", Some("f"), 1.0, 0).unwrap();
        assert!(format!("{q:?}").contains("FuzzyLikeThisQuery"));
        assert_ne!(q, FuzzyLikeThisQuery::new(50, analyzer()));
        let c: Clause = q.into();
        assert!(matches!(c, Clause::Extended(_)));
    }

    #[test]
    fn a_queue_size_java_refuses_fails_the_rewrite() {
        for bad in [-1, MAX_ARRAY_LENGTH] {
            let q = FuzzyLikeThisQuery::new(bad, analyzer());
            let reader = FuzzyReader {
                segments: &[],
                num_docs: 0,
                hash_seed: 0,
            };
            let e = q.rewrite_over(&reader).unwrap_err();
            assert_eq!(
                e.to_string(),
                Error::IllegalArgument(format!(
                    "maxSize must be >= 0 and < 2147483631; got: {bad}"
                ))
                .to_string()
            );
        }
        // an empty reader rewrites to an empty boolean
        let mut q = FuzzyLikeThisQuery::new(0, analyzer());
        q.add_terms("x", Some("f"), 1.0, 0).unwrap();
        let reader = FuzzyReader {
            segments: &[],
            num_docs: 0,
            hash_seed: 0,
        };
        assert_eq!(q.rewrite_over(&reader).unwrap(), BooleanQuery::default());
        assert_eq!(num_docs(&[]), 0);
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
        // `long` arithmetic: no overflow at the top of the range
        assert!(idf(i32::MAX, i32::MAX).is_finite());
    }
}
