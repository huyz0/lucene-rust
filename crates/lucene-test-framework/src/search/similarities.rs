//! `org.apache.lucene.tests.search.similarities`: [`AssertingSimilarity`],
//! [`RandomSimilarity`] and `BaseSimilarityTestCase` ([`new_corpus`],
//! [`new_term`], [`random_scoring`], [`bulk_score`]).
//!
//! `SimScorer.explain` is not ported for similarities other than BM25
//! (`lucene_search::similarities`), so `doTestScoring`'s checks that a
//! score equals its explanation, and its `verifyExplanation` passes, are
//! not made; their `rarely()` draws are, so the random stream stays Java's.
//! `BaseSimilarityTestCase.beforeClass`'s one-document index is not built:
//! nothing in the base reads it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lucene_search::similarities::{
    AfterEffect, AxiomaticSimilarity, AxiomaticVariant, BasicModel, Bm25Similarity,
    BooleanSimilarity, ClassicSimilarity, CollectionModel, CollectionStatistics, DfiSimilarity,
    DfrSimilarity, Distribution, FieldInvertState, IbSimilarity, Independence, Lambda,
    LmDirichletSimilarity, LmJelinekMercerSimilarity, NormSimilarity, Normalization, SimScorer,
    Similarity, TermStatistics,
};
use lucene_util::java_random::JavaRandom;
use lucene_util::small_float;

use crate::util::{lucene_test_case, test_util};

// ---------------------------------------------------------------------------
// AssertingSimilarity
// ---------------------------------------------------------------------------

/// `AssertingSimilarity`: checks the norms its delegate computes and the
/// scores it gives.
#[derive(Debug, Clone)]
pub struct AssertingSimilarity {
    delegate: Arc<dyn Similarity>,
}

impl AssertingSimilarity {
    /// `new AssertingSimilarity(delegate)`.
    pub fn new(delegate: Arc<dyn Similarity>) -> Self {
        Self { delegate }
    }
}

impl NormSimilarity for AssertingSimilarity {
    fn compute_norm(&self, field: &str, state: &FieldInvertState) -> i64 {
        assert!(state.length > 0, "length {}", state.length);
        assert!(state.position >= 0, "position {}", state.position);
        assert!(state.offset >= 0, "offset {}", state.offset);
        assert!(state.max_term_frequency >= 0);
        assert!(state.max_term_frequency <= state.length);
        assert!(state.num_overlap >= 0);
        assert!(state.num_overlap < state.length);
        assert!(state.unique_term_count > 0);
        assert!(state.unique_term_count <= state.length);
        let norm = self.delegate.compute_norm(field, state);
        assert_ne!(norm, 0, "a norm of 0");
        norm
    }

    fn discount_overlaps(&self) -> bool {
        self.delegate.discount_overlaps()
    }
}

impl Similarity for AssertingSimilarity {
    fn scorer(
        &self,
        field: &str,
        boost: f32,
        collection: &CollectionStatistics,
        terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        assert!(boost >= 0.0, "boost {boost}");
        assert!(!terms.is_empty(), "no term statistics");
        let delegate = self.delegate.scorer(field, boost, collection, terms);
        Arc::new(AssertingSimScorer { delegate })
    }

    fn shared(&self) -> Option<Arc<dyn Similarity>> {
        Some(Arc::new(self.clone()))
    }

    fn simple_name(&self) -> &str {
        "AssertingSimilarity"
    }
}

/// `AssertingSimilarity.AssertingSimScorer`.
struct AssertingSimScorer {
    delegate: Arc<dyn SimScorer>,
}

impl SimScorer for AssertingSimScorer {
    fn score(&self, freq: f32, norm: i64) -> f32 {
        // freq in bounds
        assert!(freq.is_finite(), "freq {freq}");
        assert!(freq > 0.0, "freq {freq}");
        // result in bounds
        let score = self.delegate.score(freq, norm);
        assert!(score.is_finite(), "score {score}");
        assert!(
            score <= self.delegate.score(freq, 1),
            "score {score} above norm 1's"
        );
        assert!(score >= 0.0, "score {score}");
        score
    }

    fn score_bulk(&self, freqs: &[f32], norms: &[i64], scores: &mut Vec<f32>) {
        for (&f, &n) in freqs.iter().zip(norms) {
            assert!(f > 0.0, "freq {f}");
            assert_ne!(n, 0, "norm 0");
        }
        self.delegate.score_bulk(freqs, norms, scores);
        for &s in scores.iter() {
            assert!(s.is_finite(), "score {s}");
            assert!(s >= 0.0, "score {s}");
        }
    }
}

// ---------------------------------------------------------------------------
// RandomSimilarity
// ---------------------------------------------------------------------------

/// `RandomSimilarity.allSims`: every similarity it rotates through, in
/// Java's order.
#[must_use]
pub fn all_sims() -> Vec<Arc<dyn Similarity>> {
    let axiomatic = |variant| -> Arc<dyn Similarity> {
        Arc::new(AxiomaticSimilarity::new(variant, true, 0.25, 1, 0.35).expect("defaults"))
    };
    let normalizations = [
        Normalization::H1_DEFAULT,
        Normalization::H2_DEFAULT,
        Normalization::H3_DEFAULT,
        Normalization::Z_DEFAULT,
    ];
    let mut sims: Vec<Arc<dyn Similarity>> = vec![
        Arc::new(ClassicSimilarity::default()),
        Arc::new(Bm25Similarity::default()),
        axiomatic(AxiomaticVariant::F1Exp),
        axiomatic(AxiomaticVariant::F1Log),
        axiomatic(AxiomaticVariant::F2Exp),
        axiomatic(AxiomaticVariant::F2Log),
        Arc::new(BooleanSimilarity),
    ];
    for basic in [
        BasicModel::G,
        BasicModel::IF,
        BasicModel::In,
        BasicModel::Ine,
    ] {
        for after in [AfterEffect::B, AfterEffect::L] {
            for norm in normalizations {
                sims.push(Arc::new(
                    DfrSimilarity::new(basic, after, norm).expect("valid"),
                ));
            }
        }
    }
    for dist in [Distribution::LL, Distribution::SPL] {
        for lambda in [Lambda::DF, Lambda::TTF] {
            for norm in normalizations {
                sims.push(Arc::new(
                    IbSimilarity::new(dist, lambda, norm).expect("valid"),
                ));
            }
        }
    }
    sims.push(Arc::new(LmDirichletSimilarity::default()));
    for lambda in [0.1, 0.7] {
        sims.push(Arc::new(
            LmJelinekMercerSimilarity::new(CollectionModel::Default, true, lambda).expect("valid"),
        ));
    }
    for independence in [
        Independence::Standardized,
        Independence::Saturated,
        Independence::ChiSquared,
    ] {
        sims.push(Arc::new(DfiSimilarity::new(independence)));
    }
    sims
}

/// `Collections.shuffle(list, random)`.
pub fn java_shuffle<T>(list: &mut [T], random: &mut JavaRandom) {
    for i in (2..=list.len()).rev() {
        let bound = i32::try_from(i).expect("a list shorter than 2^31");
        let j = usize::try_from(random.next_int_bounded(bound)).expect("non-negative");
        list.swap(i - 1, j);
    }
}

/// A similarity of [`all_sims`] with its index there.
type KnownSim = (usize, Arc<dyn Similarity>);

/// `RandomSimilarity`: each field gets one of [`all_sims`], shuffled by
/// the random it was built with and picked by the field's hash.
#[derive(Debug)]
pub struct RandomSimilarity {
    /// [`all_sims`], shuffled, each with its index in [`all_sims`].
    known_sims: Vec<KnownSim>,
    previous_mappings: Mutex<HashMap<String, KnownSim>>,
    per_field_seed: i32,
    should_query_norm: bool,
}

impl RandomSimilarity {
    /// `new RandomSimilarity(random)`.
    pub fn new(random: &mut JavaRandom) -> Self {
        let per_field_seed = random.next_int();
        let should_query_norm = random.next_boolean();
        let mut known_sims: Vec<_> = all_sims().into_iter().enumerate().collect();
        java_shuffle(&mut known_sims, random);
        Self {
            known_sims,
            previous_mappings: Mutex::new(HashMap::new()),
            per_field_seed,
            should_query_norm,
        }
    }

    /// `get(field)`: the similarity `field` scores with, the same every call.
    pub fn get(&self, field: &str) -> Arc<dyn Similarity> {
        self.pick(field).1
    }

    /// The index in [`all_sims`] of the similarity `field` scores with.
    pub fn all_sims_index(&self, field: &str) -> usize {
        self.pick(field).0
    }

    fn pick(&self, field: &str) -> (usize, Arc<dyn Similarity>) {
        let mut map = self
            .previous_mappings
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let picked = map.entry(field.to_string()).or_insert_with(|| {
            let len = i32::try_from(self.known_sims.len()).expect("61 similarities");
            // `Math.max(0, Math.abs(seed ^ hash))`: `abs(Integer.MIN_VALUE)`
            // stays negative, and `max` makes it 0
            let at = (self.per_field_seed ^ lucene_test_case::java_string_hash(field))
                .wrapping_abs()
                .max(0)
                % len;
            let (i, sim) = &self.known_sims[usize::try_from(at).expect("non-negative")];
            (*i, Arc::clone(sim))
        });
        (picked.0, Arc::clone(&picked.1))
    }

    /// `shouldQueryNorm`, which only `toString` reads.
    #[must_use]
    pub fn should_query_norm(&self) -> bool {
        self.should_query_norm
    }
}

impl NormSimilarity for RandomSimilarity {
    fn compute_norm(&self, field: &str, state: &FieldInvertState) -> i64 {
        self.get(field).compute_norm(field, state)
    }
}

impl Similarity for RandomSimilarity {
    fn scorer(
        &self,
        field: &str,
        boost: f32,
        collection: &CollectionStatistics,
        terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        self.get(field).scorer(field, boost, collection, terms)
    }

    fn simple_name(&self) -> &str {
        "RandomSimilarity"
    }
}

// ---------------------------------------------------------------------------
// BaseSimilarityTestCase
// ---------------------------------------------------------------------------

/// `MAXDOC_FORTESTING`.
pub const MAXDOC_FORTESTING: i64 = 1 << 48;
/// `MAXTOKENS_FORTESTING`.
pub const MAXTOKENS_FORTESTING: i64 = 1 << 49;

/// `Float.MIN_VALUE`: the smallest positive (subnormal) `float`.
pub const FLOAT_MIN_VALUE: f32 = f32::from_bits(1);

/// `newCorpus(random, norm)`: a random corpus at least possible for a
/// single document of norm `norm`.
///
/// # Panics
/// When the drawn statistics are inconsistent (they never are).
pub fn new_corpus(random: &mut JavaRandom, norm: i32) -> CollectionStatistics {
    let lower_bound: i64 = if norm == 0 {
        1
    } else {
        i64::from(small_float::byte4_to_int(
            u8::try_from(norm).expect("a norm byte"),
        ))
    };
    let max_doc = match random.next_int_bounded(6) {
        0 => 1,
        1 => 2,
        2 => test_util::next_long(random, 3, 16),
        3 => test_util::next_long(random, 16, 100_000),
        4 => test_util::next_long(random, 100_000, MAXDOC_FORTESTING),
        _ => MAXDOC_FORTESTING,
    };
    let doc_count = match random.next_int_bounded(3) {
        0 => 1,
        1 => test_util::next_long(random, 1, max_doc),
        _ => max_doc,
    };
    let upper_bound = doc_count
        .checked_mul(i64::from(i32::MAX))
        .map_or(MAXTOKENS_FORTESTING, |m| m.min(MAXTOKENS_FORTESTING));
    let sum_doc_freq = match random.next_int_bounded(3) {
        0 => doc_count,
        1 => upper_bound + 1 - lower_bound,
        _ => test_util::next_long(random, doc_count, upper_bound + 1 - lower_bound),
    };
    let sum_total_term_freq = match random.next_int_bounded(4) {
        0 => sum_doc_freq,
        1 => sum_doc_freq - 1 + lower_bound,
        2 => upper_bound,
        _ => {
            assert!(sum_doc_freq - 1 + lower_bound <= upper_bound);
            test_util::next_long(random, sum_doc_freq - 1 + lower_bound, upper_bound)
        }
    };
    CollectionStatistics::new(max_doc, doc_count, sum_total_term_freq, sum_doc_freq)
        .expect("a consistent corpus")
}

/// `newTerm(random, corpus)`: a random term within the corpus's bounds.
///
/// # Panics
/// When the drawn statistics are inconsistent (they never are).
pub fn new_term(random: &mut JavaRandom, corpus: &CollectionStatistics) -> TermStatistics {
    let doc_freq = match random.next_int_bounded(3) {
        0 => 1,
        1 => corpus.doc_count,
        _ => test_util::next_long(random, 1, corpus.doc_count),
    };
    let upper_bound = doc_freq
        .checked_mul(i64::from(i32::MAX))
        .map_or(corpus.sum_total_term_freq, |m| {
            m.min(corpus.sum_total_term_freq)
        });
    let total_term_freq = if corpus.sum_total_term_freq == corpus.sum_doc_freq {
        doc_freq
    } else {
        match random.next_int_bounded(3) {
            0 => doc_freq,
            1 => upper_bound,
            _ => test_util::next_long(random, doc_freq, upper_bound),
        }
    };
    TermStatistics::new(doc_freq, total_term_freq).expect("a consistent term")
}

/// One `doTestScoring` call: its inputs, and the scores it compared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoringCase {
    /// The corpus.
    pub corpus: CollectionStatistics,
    /// The term.
    pub term: TermStatistics,
    /// The query boost.
    pub boost: f32,
    /// The term frequency.
    pub freq: f32,
    /// The norm (`0`: omitted).
    pub norm: i32,
    /// `score(Float.MAX_VALUE, 1)`.
    pub max_score: f32,
    /// `score(freq, norm)`.
    pub score: f32,
    /// The smaller frequency compared.
    pub prev_freq: f32,
    /// `score(prevFreq, norm)`.
    pub prev_score: f32,
    /// `score(freq, norm - 1)`, when `norm > 1`.
    pub prev_norm_score: Option<f32>,
    /// The rarer term's score, when the term is not the rarest possible.
    pub prev_term_score: Option<f32>,
}

/// The test run's knobs `rarely()` reads.
#[derive(Debug, Clone, Copy)]
pub struct Knobs {
    /// `TEST_NIGHTLY`.
    pub nightly: bool,
    /// `RANDOM_MULTIPLIER`.
    pub multiplier: i32,
}

impl Default for Knobs {
    fn default() -> Self {
        Self {
            nightly: false,
            multiplier: lucene_test_case::default_multiplier(false),
        }
    }
}

/// `testRandomScoring`: `atLeast(1)` random similarities, each scored over
/// 3 x 255 norms x 10 random terms and corpora by [`do_test_scoring`];
/// `on_case` sees each case.
pub fn random_scoring(
    random: &mut JavaRandom,
    knobs: Knobs,
    get_similarity: impl Fn(&mut JavaRandom) -> Arc<dyn Similarity>,
    mut on_case: impl FnMut(&ScoringCase),
) {
    let iterations = lucene_test_case::at_least(random, knobs.multiplier, 1);
    for _ in 0..iterations {
        let similarity = get_similarity(random);
        for _ in 0..3 {
            for k in 1..256 {
                let corpus = new_corpus(random, k);
                for _ in 0..10 {
                    let term = new_term(random, &corpus);
                    let freq = random_freq(random, &term);
                    let boost = match random.next_int_bounded(5) {
                        0 => 0.0,
                        1 => FLOAT_MIN_VALUE,
                        2 => 1.0,
                        3 => i32::MAX as f32,
                        _ => random.next_float() * i32::MAX as f32,
                    };
                    let case = do_test_scoring(
                        random,
                        knobs,
                        similarity.as_ref(),
                        corpus,
                        term,
                        boost,
                        freq,
                        k,
                    );
                    on_case(&case);
                }
            }
        }
    }
}

/// `testRandomScoring`'s frequency: whole or fractional, within what the
/// term's statistics allow.
fn random_freq(random: &mut JavaRandom, term: &TermStatistics) -> f32 {
    if term.total_term_freq == term.doc_freq {
        return 1.0;
    }
    if term.doc_freq == 1 {
        return i32::try_from(term.total_term_freq).expect("toIntExact") as f32;
    }
    let upper_bound =
        i32::try_from((term.total_term_freq - term.doc_freq + 1).min(i64::from(i32::MAX)))
            .expect("at most Integer.MAX_VALUE");
    if random.next_boolean() {
        match random.next_int_bounded(3) {
            0 => 1.0,
            1 => upper_bound as f32,
            _ => test_util::next_int(random, 1, upper_bound) as f32,
        }
    } else {
        let mut candidate = match random.next_int_bounded(2) {
            0 => FLOAT_MIN_VALUE,
            _ => upper_bound as f32 * random.next_float(),
        };
        if candidate <= FLOAT_MIN_VALUE {
            candidate = FLOAT_MIN_VALUE.next_up();
        }
        candidate
    }
}

/// `doTestScoring`: the score is finite, non-negative (but under
/// `IndriDirichletSimilarity`) and at most the max score; it does not
/// decrease as the frequency grows, nor as the document shortens, nor as
/// the term gets rarer.
///
/// # Panics
/// When a check fails.
#[allow(clippy::too_many_arguments)]
pub fn do_test_scoring(
    random: &mut JavaRandom,
    knobs: Knobs,
    similarity: &dyn Similarity,
    corpus: CollectionStatistics,
    term: TermStatistics,
    boost: f32,
    freq: f32,
    norm: i32,
) -> ScoringCase {
    let ctx = || {
        format!(
            "{} {corpus:?} {term:?} norm={norm} freq={freq} boost={boost}",
            similarity.simple_name()
        )
    };
    let non_negative = similarity.simple_name() != "IndriDirichletSimilarity";
    let rarely = |r: &mut JavaRandom| lucene_test_case::rarely(r, knobs.nightly, knobs.multiplier);
    let scorer = similarity.scorer("field", boost, &corpus, &[term]);
    let n = i64::from(norm);
    let max_score = scorer.score(f32::MAX, 1);
    assert!(!max_score.is_nan(), "maxScore is NaN: {}", ctx());
    let score = scorer.score(freq, n);
    assert!(score.is_finite(), "infinite/NaN score: {score}: {}", ctx());
    if non_negative {
        assert!(score >= 0.0, "negative score: {score}: {}", ctx());
    }
    assert!(
        score <= max_score,
        "greater than maxScore: {score}>{max_score}: {}",
        ctx()
    );
    // `verifyExplanation`, rarely: no explanation to verify
    let _ = rarely(random);

    let prev_freq =
        if random.next_boolean() && freq == freq.trunc() && freq > 1.0 && term.doc_freq > 1 {
            freq - 1.0
        } else {
            freq.next_down()
        };
    let prev_score = scorer.score(prev_freq, n);
    assert!(prev_score.is_finite(), "{}", ctx());
    if non_negative {
        assert!(prev_score >= 0.0, "{}", ctx());
    }
    let _ = rarely(random);
    assert!(
        prev_score <= score,
        "score({prev_freq})={prev_score} > score({freq})={score}: {}",
        ctx()
    );

    let mut prev_norm_score = None;
    if norm > 1 {
        let s = scorer.score(freq, n - 1);
        assert!(s.is_finite(), "{}", ctx());
        if non_negative {
            assert!(s >= 0.0, "{}", ctx());
        }
        let _ = rarely(random);
        assert!(
            s >= score,
            "score({freq},{})={s} < score({freq},{norm})={score}: {}",
            norm - 1,
            ctx()
        );
        prev_norm_score = Some(s);
    }

    let mut prev_term_score = None;
    // `freq < totalTermFreq` compares in float, as Java's promotion does
    if term.doc_freq > 1 && freq < term.total_term_freq as f32 {
        // Java builds the rarer term's statistics, then scores `term` again
        let prev_term_scorer = similarity.scorer("field", boost, &corpus, &[term]);
        let s = prev_term_scorer.score(freq, n);
        assert!(s.is_finite(), "{}", ctx());
        if non_negative {
            assert!(s >= 0.0, "{}", ctx());
        }
        let _ = rarely(random);
        assert!(s >= score, "prevTerm score {s} < {score}: {}", ctx());
        prev_term_score = Some(s);
    }
    ScoringCase {
        corpus,
        term,
        boost,
        freq,
        norm,
        max_score,
        score,
        prev_freq,
        prev_score,
        prev_norm_score,
        prev_term_score,
    }
}

/// `random.nextFloat(bound)` (`RandomSupport.boundedNextFloat`).
pub fn next_float_bounded(random: &mut JavaRandom, bound: f32) -> f32 {
    let r = random.next_float() * bound;
    if r >= bound {
        bound.next_down()
    } else {
        r
    }
}

/// `testBulkScore`: the bulk scorer gives each `(freq, norm)` exactly the
/// score the scorer does, over `atLeast(3)` batches of up to 200. Returns
/// every batch's scores.
///
/// # Panics
/// When a bulk score differs.
pub fn bulk_score(
    random: &mut JavaRandom,
    knobs: Knobs,
    get_similarity: impl Fn(&mut JavaRandom) -> Arc<dyn Similarity>,
) -> Vec<Vec<f32>> {
    let similarity = get_similarity(random);
    let corpus = new_corpus(random, 1);
    let term = new_term(random, &corpus);
    let boost = next_float_bounded(random, 5.0);
    let scorer = similarity.scorer("field", boost, &corpus, &[term]);
    let freq_upper_bound =
        i32::try_from((term.total_term_freq - term.doc_freq + 1).min(i64::from(i32::MAX)))
            .expect("at most Integer.MAX_VALUE");
    let iters = lucene_test_case::at_least(random, knobs.multiplier, 3);
    let mut out = Vec::new();
    let mut scores = Vec::new();
    for _ in 0..iters {
        let size = test_util::next_int(random, 0, 200);
        let mut freqs = Vec::new();
        let mut norms = Vec::new();
        for _ in 0..size {
            freqs.push(test_util::next_int(random, 1, freq_upper_bound) as f32);
            norms.push(test_util::next_long(random, 1, 255));
        }
        let expected: Vec<f32> = freqs
            .iter()
            .zip(&norms)
            .map(|(&f, &n)| scorer.score(f, n))
            .collect();
        scorer.score_bulk(&freqs, &norms, &mut scores);
        assert_eq!(
            expected.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            scores.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            "{}",
            similarity.simple_name()
        );
        out.push(scores.clone());
    }
    out
}
