//! Lucene's `TestSimpleSearchEquivalence`: queries that must hit a subset
//! of (or the same set as, or score the same as) another, over
//! `SearchEquivalenceTestBase`'s random corpus.

use lucene_search::extended_query::SynonymQuery;
use lucene_search::query::{
    BooleanQuery, BoostQuery, Clause, DisjunctionMaxQuery, MultiPhraseQuery, PhraseQuery, TermQuery,
};
use lucene_search::test_support::TestQuery;
use lucene_test_framework::search::search_equivalence::{random_term, SearchEquivalence, FIELD};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_util::java_random::JavaRandom;

fn setup(name: &str) -> (LuceneTestCase, SearchEquivalence) {
    let mut tc = LuceneTestCase::new(name);
    let (nightly, multiplier) = (tc.nightly(), tc.multiplier());
    let s = SearchEquivalence::new(tc.random(), nightly, multiplier);
    (tc, s)
}

fn tq(t: &[u8]) -> Clause {
    Clause::Term(TermQuery::new(FIELD, t.to_vec()))
}

fn should(clauses: Vec<Clause>) -> Clause {
    BooleanQuery {
        should: clauses,
        ..BooleanQuery::default()
    }
    .into()
}

fn must(clauses: Vec<Clause>) -> Clause {
    BooleanQuery {
        must: clauses,
        ..BooleanQuery::default()
    }
    .into()
}

fn exact(t1: &[u8], t2: &[u8]) -> PhraseQuery {
    PhraseQuery::new(FIELD, [t1.to_vec(), t2.to_vec()])
}

fn with_holes(t1: &[u8], t2: &[u8]) -> PhraseQuery {
    PhraseQuery::with_positions(FIELD, [(t1.to_vec(), 0), (t2.to_vec(), 2)]).unwrap()
}

fn two_terms(r: &mut JavaRandom) -> (Vec<u8>, Vec<u8>) {
    let t1 = random_term(r);
    let t2 = random_term(r);
    (t1, t2)
}

#[test]
fn term_versus_boolean_or() {
    let (mut tc, s) = setup("term_versus_boolean_or");
    let (t1, t2) = two_terms(tc.random());
    s.assert_subset_of(tc.random(), &tq(&t1), &should(vec![tq(&t1), tq(&t2)]));
}

#[test]
fn term_versus_boolean_req_opt() {
    let (mut tc, s) = setup("term_versus_boolean_req_opt");
    let (t1, t2) = two_terms(tc.random());
    let q2 = BooleanQuery {
        must: vec![tq(&t1)],
        should: vec![tq(&t2)],
        ..BooleanQuery::default()
    };
    s.assert_subset_of(tc.random(), &tq(&t1), &q2.into());
}

#[test]
fn boolean_req_excl_versus_term() {
    let (mut tc, s) = setup("boolean_req_excl_versus_term");
    let (t1, t2) = two_terms(tc.random());
    let q1 = BooleanQuery {
        must: vec![tq(&t1)],
        must_not: vec![tq(&t2)],
        ..BooleanQuery::default()
    };
    s.assert_subset_of(tc.random(), &q1.into(), &tq(&t1));
}

#[test]
fn boolean_and_versus_boolean_or() {
    let (mut tc, s) = setup("boolean_and_versus_boolean_or");
    let (t1, t2) = two_terms(tc.random());
    // Java's test builds both sides as the same disjunction
    let q = should(vec![tq(&t1), tq(&t2)]);
    s.assert_subset_of(tc.random(), &q, &q.clone());
}

#[test]
fn disjunction_sum_versus_disjunction_max() {
    let (mut tc, s) = setup("disjunction_sum_versus_disjunction_max");
    let (t1, t2) = two_terms(tc.random());
    let q2 = DisjunctionMaxQuery {
        disjuncts: vec![tq(&t1), tq(&t2)],
        tie_breaker: 0.5,
    };
    s.assert_same_set(tc.random(), &should(vec![tq(&t1), tq(&t2)]), &q2.into());
}

#[test]
fn exact_phrase_versus_boolean_and() {
    let (mut tc, s) = setup("exact_phrase_versus_boolean_and");
    let (t1, t2) = two_terms(tc.random());
    s.assert_subset_of(
        tc.random(),
        &exact(&t1, &t2).into(),
        &must(vec![tq(&t1), tq(&t2)]),
    );
}

#[test]
fn exact_phrase_versus_boolean_and_with_holes() {
    let (mut tc, s) = setup("exact_phrase_versus_boolean_and_with_holes");
    let (t1, t2) = two_terms(tc.random());
    s.assert_subset_of(
        tc.random(),
        &with_holes(&t1, &t2).into(),
        &must(vec![tq(&t1), tq(&t2)]),
    );
}

#[test]
fn phrase_versus_sloppy_phrase() {
    let (mut tc, s) = setup("phrase_versus_sloppy_phrase");
    let (t1, t2) = two_terms(tc.random());
    let q2 = exact(&t1, &t2).with_slop(1);
    s.assert_subset_of(tc.random(), &exact(&t1, &t2).into(), &q2.into());
}

#[test]
fn phrase_versus_sloppy_phrase_with_holes() {
    let (mut tc, s) = setup("phrase_versus_sloppy_phrase_with_holes");
    let (t1, t2) = two_terms(tc.random());
    let q2 = with_holes(&t1, &t2).with_slop(2);
    s.assert_subset_of(tc.random(), &with_holes(&t1, &t2).into(), &q2.into());
}

#[test]
fn exact_phrase_versus_multi_phrase() {
    let (mut tc, s) = setup("exact_phrase_versus_multi_phrase");
    let (t1, t2) = two_terms(tc.random());
    let t3 = random_term(tc.random());
    let q2 = MultiPhraseQuery::new(FIELD, [vec![t1.clone()], vec![t2.clone(), t3]]);
    s.assert_subset_of(tc.random(), &exact(&t1, &t2).into(), &q2.into());
}

#[test]
fn exact_phrase_versus_multi_phrase_with_holes() {
    let (mut tc, s) = setup("exact_phrase_versus_multi_phrase_with_holes");
    let (t1, t2) = two_terms(tc.random());
    let t3 = random_term(tc.random());
    let q2 =
        MultiPhraseQuery::with_positions(FIELD, [(vec![t1.clone()], 0), (vec![t2.clone(), t3], 2)])
            .unwrap();
    s.assert_subset_of(tc.random(), &with_holes(&t1, &t2).into(), &q2.into());
}

#[test]
fn sloppy_phrase_versus_boolean_and() {
    let (mut tc, s) = setup("sloppy_phrase_versus_boolean_and");
    let t1 = random_term(tc.random());
    // SloppyPhrase handles repeats, so the terms must differ
    let mut t2 = random_term(tc.random());
    while t2 == t1 {
        t2 = random_term(tc.random());
    }
    let q1 = exact(&t1, &t2).with_slop(i32::MAX.unsigned_abs());
    s.assert_same_set(tc.random(), &q1.into(), &must(vec![tq(&t1), tq(&t2)]));
}

#[test]
fn phrase_relative_positions() {
    let (mut tc, s) = setup("phrase_relative_positions");
    let (t1, t2) = two_terms(tc.random());
    let q2 =
        PhraseQuery::with_positions(FIELD, [(t1.clone(), 10000), (t2.clone(), 10001)]).unwrap();
    s.assert_same_scores(tc.random(), &exact(&t1, &t2).into(), &q2.into());
}

#[test]
fn sloppy_phrase_relative_positions() {
    let (mut tc, s) = setup("sloppy_phrase_relative_positions");
    let (t1, t2) = two_terms(tc.random());
    let q1 = exact(&t1, &t2).with_slop(2);
    let q2 = PhraseQuery::with_positions(FIELD, [(t1, 10000), (t2, 10001)])
        .unwrap()
        .with_slop(2);
    s.assert_same_scores(tc.random(), &q1.into(), &q2.into());
}

/// Java wraps the inner boost in an `AssertingQuery` so `BoostQuery` cannot
/// merge the two boosts; here an opaque [`TestQuery`] does.
#[test]
fn boost_query_simplification() {
    let (mut tc, s) = setup("boost_query_simplification");
    let b1 = tc.random().next_float() * 10.0;
    let b2 = tc.random().next_float() * 10.0;
    let term = random_term(tc.random());
    let inner = || Clause::from(BoostQuery::new(tq(&term), b2));
    let q1 = BoostQuery::new(inner(), b1);
    let opaque = TestQuery::random_approximation(inner(), JavaRandom::new(tc.random().next_long()));
    let q2 = BoostQuery::new(opaque, b1);
    s.assert_same_scores(tc.random(), &q1.into(), &q2.into());
}

#[test]
fn boolean_boost_propagation() {
    let (mut tc, s) = setup("boolean_boost_propagation");
    let boost1 = tc.random().next_float();
    let term = random_term(tc.random());
    let tq_boosted = Clause::from(BoostQuery::new(tq(&term), boost1));
    let boost2 = tc.random().next_float();
    let q1 = BoostQuery::new(tq_boosted.clone(), boost2);
    let q2 = BooleanQuery {
        must: vec![tq_boosted.clone()],
        filter: vec![tq_boosted],
        ..BooleanQuery::default()
    };
    let q2 = BoostQuery::new(q2, boost2);
    s.assert_same_scores(tc.random(), &q1.into(), &q2.into());
}

#[test]
fn boolean_or_vs_synonym() {
    let (mut tc, s) = setup("boolean_or_vs_synonym");
    let (t1, t2) = two_terms(tc.random());
    let q1 = SynonymQuery::new(FIELD, [(t1.clone(), 1.0), (t2.clone(), 1.0)]).unwrap();
    s.assert_same_set(tc.random(), &q1.into(), &should(vec![tq(&t1), tq(&t2)]));
}

/// The corpus itself: `atLeast(100)` documents, a twentieth of them drawn
/// for deletion, and the stop letter never indexed.
#[test]
fn corpus_is_java_shaped() {
    let (_tc, s) = setup("corpus_is_java_shaped");
    assert!(s.num_docs() >= 100);
    let stop = s.stopword().to_string();
    s.with_searcher(|searcher| {
        assert_eq!(searcher.max_doc(), s.num_docs());
        let q = BooleanQuery {
            must: vec![tq(stop.as_bytes())],
            ..BooleanQuery::default()
        };
        assert_eq!(
            searcher.count(&q).unwrap(),
            0,
            "the stop letter {stop} is indexed"
        );
    });
}
