//! Lucene's `BaseSimilarityTestCase` subclasses (`TestBM25Similarity` ...
//! `TestDistributionSPL`): `testRandomScoring` and `testBulkScore` for
//! each, plus `AssertingSimilarity` and `RandomSimilarity` seen to work
//! and to fire.

mod similarity_cases;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use lucene_search::similarities::{
    Bm25Similarity, CollectionStatistics, FieldInvertState, NormSimilarity, SimScorer, Similarity,
    TermStatistics,
};
use lucene_test_framework::search::similarities::{
    all_sims, bulk_score, do_test_scoring, java_shuffle, new_corpus, new_term, next_float_bounded,
    random_scoring, AssertingSimilarity, Knobs, RandomSimilarity, FLOAT_MIN_VALUE,
    MAXDOC_FORTESTING,
};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_util::java_random::JavaRandom;

fn knobs(tc: &LuceneTestCase) -> Knobs {
    Knobs {
        nightly: tc.nightly(),
        multiplier: tc.multiplier(),
    }
}

/// `testRandomScoring` and `testBulkScore` for one subclass.
fn run(name: &str) {
    let get = similarity_cases::get(name);
    let mut tc = LuceneTestCase::new(&format!("{name}::random_scoring"));
    let k = knobs(&tc);
    let mut n = 0;
    random_scoring(tc.random(), k, get, |_| n += 1);
    assert!(n >= 7650, "{n} cases");
    let mut tc = LuceneTestCase::new(&format!("{name}::bulk_score"));
    let k = knobs(&tc);
    assert!(bulk_score(tc.random(), k, get).len() >= 3);
}

macro_rules! similarity_tests {
    ($($test:ident: $name:literal;)*) => {
        $(
            #[test]
            fn $test() {
                run($name);
            }
        )*
    };
}

similarity_tests! {
    bm25: "BM25";
    boolean: "Boolean";
    classic: "Classic";
    independence_chi_squared: "IndependenceChiSquared";
    independence_saturated: "IndependenceSaturated";
    independence_standardized: "IndependenceStandardized";
    indri_dirichlet: "IndriDirichlet";
    lm_dirichlet: "LMDirichlet";
    lm_jelinek_mercer: "LMJelinekMercer";
    raw_tf: "RawTF";
    axiomatic_f1_exp: "AxiomaticF1EXP";
    axiomatic_f1_log: "AxiomaticF1LOG";
    axiomatic_f2_exp: "AxiomaticF2EXP";
    axiomatic_f2_log: "AxiomaticF2LOG";
    axiomatic_f3_exp: "AxiomaticF3EXP";
    axiomatic_f3_log: "AxiomaticF3LOG";
    basic_model_g: "BasicModelG";
    basic_model_if: "BasicModelIF";
    basic_model_in: "BasicModelIn";
    basic_model_ine: "BasicModelIne";
    distribution_ll: "DistributionLL";
    distribution_spl: "DistributionSPL";
}

fn panics(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("the check should fail");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_default()
}

fn corpus() -> CollectionStatistics {
    CollectionStatistics::new(100, 100, 1000, 500).expect("valid")
}

fn term() -> TermStatistics {
    TermStatistics::new(10, 40).expect("valid")
}

fn state(length: i32) -> FieldInvertState {
    FieldInvertState {
        position: length,
        length,
        offset: length * 3,
        max_term_frequency: 1,
        unique_term_count: length,
        ..FieldInvertState::default()
    }
}

/// A similarity that scores what it is told to, and norms everything 0.
#[derive(Debug)]
struct Broken(f32);

struct BrokenScorer(f32);

impl SimScorer for BrokenScorer {
    fn score(&self, _freq: f32, norm: i64) -> f32 {
        if norm == 1 {
            1.0
        } else {
            self.0
        }
    }
}

impl NormSimilarity for Broken {
    fn compute_norm(&self, _field: &str, _state: &FieldInvertState) -> i64 {
        0
    }
}

impl Similarity for Broken {
    fn scorer(
        &self,
        _field: &str,
        _boost: f32,
        _collection: &CollectionStatistics,
        _terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        Arc::new(BrokenScorer(self.0))
    }
}

/// Over a well-behaved similarity the asserting wrapper is invisible:
/// norms, scores and bulk scores are the delegate's.
#[test]
fn asserting_similarity_passes_bm25_through() {
    let bm25: Arc<dyn Similarity> = Arc::new(Bm25Similarity::default());
    let sim = AssertingSimilarity::new(Arc::clone(&bm25));
    assert_eq!(sim.simple_name(), "AssertingSimilarity");
    assert!(sim.discount_overlaps());
    assert!(sim.shared().is_some());
    let st = state(7);
    assert_eq!(sim.compute_norm("f", &st), bm25.compute_norm("f", &st));
    let (a, b) = (
        sim.scorer("f", 2.0, &corpus(), &[term()]),
        bm25.scorer("f", 2.0, &corpus(), &[term()]),
    );
    for norm in 1..256 {
        assert_eq!(a.score(3.0, norm).to_bits(), b.score(3.0, norm).to_bits());
    }
    let (freqs, norms) = (vec![1.0, 2.0, 5.0], vec![1, 40, 255]);
    let (mut sa, mut sb) = (Vec::new(), Vec::new());
    a.score_bulk(&freqs, &norms, &mut sa);
    b.score_bulk(&freqs, &norms, &mut sb);
    assert_eq!(sa, sb);
    // and the whole random-scoring sweep passes through it
    let mut r = JavaRandom::new(7);
    let knobs = Knobs {
        nightly: false,
        multiplier: 1,
    };
    let mut cases = 0;
    random_scoring(
        &mut r,
        knobs,
        |_| {
            Arc::new(AssertingSimilarity::new(
                Arc::new(Bm25Similarity::default()),
            ))
        },
        |_| cases += 1,
    );
    assert!(cases >= 7650);
}

/// Each of `AssertingSimilarity`'s checks fires.
#[test]
fn asserting_similarity_fires() {
    let ok: Arc<dyn Similarity> = Arc::new(Bm25Similarity::default());
    let sim = AssertingSimilarity::new(Arc::clone(&ok));
    // computeNorm's state checks
    let bad_states: [fn(&mut FieldInvertState); 9] = [
        |s| s.length = 0,
        |s| s.position = -1,
        |s| s.offset = -1,
        |s| s.max_term_frequency = -1,
        |s| s.max_term_frequency = 8,
        |s| s.num_overlap = -1,
        |s| s.num_overlap = 7,
        |s| s.unique_term_count = 0,
        |s| s.unique_term_count = 8,
    ];
    for bad in bad_states {
        let mut st = state(7);
        bad(&mut st);
        panics(|| {
            sim.compute_norm("f", &st);
        });
    }
    // a norm of 0
    let broken = AssertingSimilarity::new(Arc::new(Broken(0.5)));
    assert!(panics(|| {
        broken.compute_norm("f", &state(3));
    })
    .contains("a norm of 0"));
    // scorer's arguments
    assert!(panics(|| {
        sim.scorer("f", -1.0, &corpus(), &[term()]);
    })
    .contains("boost"));
    assert!(panics(|| {
        sim.scorer("f", 1.0, &corpus(), &[]);
    })
    .contains("no term statistics"));
    // score's bounds
    let scorer = sim.scorer("f", 1.0, &corpus(), &[term()]);
    assert!(panics(|| {
        scorer.score(f32::INFINITY, 1);
    })
    .contains("freq"));
    assert!(panics(|| {
        scorer.score(0.0, 1);
    })
    .contains("freq"));
    for (bad, msg) in [
        (f32::NAN, "score NaN"),
        (2.0, "above norm 1's"),
        (-0.5, "score -0.5"),
    ] {
        let s =
            AssertingSimilarity::new(Arc::new(Broken(bad))).scorer("f", 1.0, &corpus(), &[term()]);
        let got = panics(|| {
            s.score(1.0, 5);
        });
        assert!(got.contains(msg), "{got}");
    }
    // the bulk scorer's
    let mut out = Vec::new();
    assert!(panics(|| scorer.score_bulk(&[0.0], &[1], &mut out)).contains("freq"));
    assert!(panics(|| scorer.score_bulk(&[1.0], &[0], &mut out)).contains("norm 0"));
    let neg =
        AssertingSimilarity::new(Arc::new(Broken(-1.0))).scorer("f", 1.0, &corpus(), &[term()]);
    assert!(panics(|| neg.score_bulk(&[1.0], &[5], &mut out)).contains("score -1"));
    let nan =
        AssertingSimilarity::new(Arc::new(Broken(f32::NAN))).scorer("f", 1.0, &corpus(), &[term()]);
    assert!(panics(|| nan.score_bulk(&[1.0], &[5], &mut out)).contains("score NaN"));
}

/// `RandomSimilarity` sticks to one similarity per field, from the 61 of
/// `allSims`, and delegates norms and scores to it.
#[test]
fn random_similarity_is_sticky_and_delegates() {
    assert_eq!(all_sims().len(), 61);
    let sim = RandomSimilarity::new(&mut JavaRandom::new(3));
    let _ = sim.should_query_norm();
    assert_eq!(sim.simple_name(), "RandomSimilarity");
    let mut seen = std::collections::BTreeSet::new();
    for i in 0..200 {
        let field = format!("f{i}");
        let at = sim.all_sims_index(&field);
        assert_eq!(at, sim.all_sims_index(&field));
        seen.insert(at);
        let picked = &all_sims()[at];
        let st = state(5);
        assert_eq!(
            sim.compute_norm(&field, &st),
            picked.compute_norm(&field, &st)
        );
        let (a, b) = (
            sim.scorer(&field, 1.0, &corpus(), &[term()]),
            picked.scorer(&field, 1.0, &corpus(), &[term()]),
        );
        assert_eq!(a.score(2.0, 9).to_bits(), b.score(2.0, 9).to_bits());
        assert_eq!(sim.get(&field).simple_name(), picked.simple_name());
    }
    assert!(seen.len() > 30, "{} similarities picked", seen.len());
}

/// `Collections.shuffle(list, rnd)` permutes (Java's exact order is
/// checked against the twin's `shuffle` line in
/// `test_framework_search_fixtures.rs`).
#[test]
fn java_shuffle_permutes() {
    let mut v: Vec<i32> = (0..10).collect();
    java_shuffle(&mut v, &mut JavaRandom::new(42));
    let mut sorted = v.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    assert_ne!(v, sorted);
    let mut one = [1];
    java_shuffle(&mut one, &mut JavaRandom::new(1));
    assert_eq!(one, [1]);
}

/// `newCorpus`/`newTerm` stay within what Lucene's statistics allow, at
/// every norm.
#[test]
fn corpus_and_term_draws_are_consistent() {
    let mut r = JavaRandom::new(11);
    for _ in 0..20 {
        for norm in 0..256 {
            let c = new_corpus(&mut r, norm);
            assert!(c.max_doc <= MAXDOC_FORTESTING);
            let t = new_term(&mut r, &c);
            assert!(t.doc_freq <= c.doc_count);
            assert!(t.total_term_freq <= c.sum_total_term_freq);
        }
    }
}

/// `random.nextFloat(bound)` stays below `bound`.
#[test]
fn next_float_bounded_is_below_bound() {
    let mut r = JavaRandom::new(5);
    for _ in 0..1000 {
        let f = next_float_bounded(&mut r, 5.0);
        assert!((0.0..5.0).contains(&f));
    }
    // `nextFloat() * bound` can round up to `bound`, which maps below it
    assert!(next_float_bounded(&mut JavaRandom::new(0), FLOAT_MIN_VALUE) < FLOAT_MIN_VALUE);
}

/// `doTestScoring` fails on a similarity that breaks each of its rules.
#[test]
fn do_test_scoring_fires() {
    let knobs = Knobs::default();
    let c = corpus();
    let t = term();
    let check = |sim: &dyn Similarity, freq: f32, norm: i32| {
        panics(|| {
            do_test_scoring(&mut JavaRandom::new(1), knobs, sim, c, t, 1.0, freq, norm);
        })
    };
    assert!(check(&Broken(f32::NAN), 2.0, 5).contains("infinite/NaN"));
    assert!(check(&Broken(-1.0), 2.0, 5).contains("negative score"));
    assert!(check(&Broken(3.0), 2.0, 5).contains("greater than maxScore"));

    /// Scores grow as the norm grows (longer documents score higher).
    #[derive(Debug)]
    struct ByNorm;
    struct ByNormScorer;
    impl SimScorer for ByNormScorer {
        fn score(&self, freq: f32, norm: i64) -> f32 {
            if freq == f32::MAX {
                1e9
            } else {
                norm as f32
            }
        }
    }
    impl NormSimilarity for ByNorm {}
    impl Similarity for ByNorm {
        fn scorer(
            &self,
            _: &str,
            _: f32,
            _: &CollectionStatistics,
            _: &[TermStatistics],
        ) -> Arc<dyn SimScorer> {
            Arc::new(ByNormScorer)
        }
    }
    assert!(check(&ByNorm, 2.0, 5).contains("score(2,4)"));

    /// Scores fall as the frequency grows.
    #[derive(Debug)]
    struct ByFreq;
    struct ByFreqScorer;
    impl SimScorer for ByFreqScorer {
        fn score(&self, freq: f32, _: i64) -> f32 {
            if freq == f32::MAX {
                1e9
            } else {
                1e6 / freq
            }
        }
    }
    impl NormSimilarity for ByFreq {}
    impl Similarity for ByFreq {
        fn scorer(
            &self,
            _: &str,
            _: f32,
            _: &CollectionStatistics,
            _: &[TermStatistics],
        ) -> Arc<dyn SimScorer> {
            Arc::new(ByFreqScorer)
        }
    }
    assert!(check(&ByFreq, 2.0, 1).contains("score(1"));
}
