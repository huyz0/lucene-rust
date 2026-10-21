//! Lucene's `TestComplexExplanations`: zero boosts and nested clauses
//! under `ClassicSimilarity`, each through `qtest` or `bqtest`, and the
//! explanation of a boolean query's failing optional clauses.

mod explanation_dsl;

use explanation_dsl::*;
use lucene_search::query::Clause;
use lucene_search::similarities::ClassicSimilarity;
use lucene_test_framework::search::base_explanation::{top_level, BaseExplanation};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

/// `setUp`: the searcher scores with `ClassicSimilarity`.
fn base(tc: &mut LuceneTestCase) -> BaseExplanation {
    BaseExplanation::new(tc.random()).with_similarity(Box::new(ClassicSimilarity::default()))
}

fn qtest(name: &str, q: Clause, docs: &[i32]) {
    let mut tc = LuceneTestCase::new(name);
    base(&mut tc).qtest(tc.random(), q, docs);
}

fn bqtest(name: &str, q: Clause, docs: &[i32]) {
    let mut tc = LuceneTestCase::new(name);
    base(&mut tc).bqtest(tc.random(), q, docs);
}

#[test]
fn t3() {
    bqtest("t3", boost(t("w1"), 0.0), &[0, 1, 2, 3]);
}

#[test]
fn ma3() {
    bqtest("ma3", boost(match_all(), 0.0), &[0, 1, 2, 3]);
}

#[test]
fn fq5() {
    let filtered = bq(vec![
        (M, boost(t("xx"), 0.0)),
        (F, match_these_items(&[1, 3])),
    ]);
    bqtest("fq5", filtered, &[3]);
}

#[test]
fn csq4() {
    bqtest("csq4", boost(csq(match_these_items(&[3])), 0.0), &[3]);
}

#[test]
fn dmq10() {
    let q = dmq(
        vec![
            bq(vec![(S, t("yy")), (S, boost(t("w5"), 100.0))]),
            boost(t("xx"), 0.0),
        ],
        0.5,
    );
    bqtest("dmq10", boost(q, 0.0), &[0, 2, 3]);
}

#[test]
fn mpq7() {
    bqtest("mpq7", boost(mpq(1, &[&["w1"], &["w2"]]), 0.0), &[0, 1, 2]);
}

#[test]
fn bq12() {
    qtest(
        "bq12",
        bq(vec![(S, t("w1")), (S, boost(t("w2"), 0.0))]),
        &[0, 1, 2, 3],
    );
}

#[test]
fn bq13() {
    qtest(
        "bq13",
        bq(vec![(S, t("w1")), (N, boost(t("w5"), 0.0))]),
        &[1, 2, 3],
    );
}

#[test]
fn bq18() {
    qtest(
        "bq18",
        bq(vec![(M, boost(t("w1"), 0.0)), (S, t("w2"))]),
        &[0, 1, 2, 3],
    );
}

#[test]
fn bq21() {
    bqtest(
        "bq21",
        boost(bq(vec![(M, t("w1")), (S, t("w2"))]), 0.0),
        &[0, 1, 2, 3],
    );
}

#[test]
fn bq22() {
    let q = bq(vec![(M, boost(t("w1"), 0.0)), (S, t("w2"))]);
    bqtest("bq22", boost(q, 0.0), &[0, 1, 2, 3]);
}

#[test]
fn explain_failing_optional_clauses() {
    let mut tc = LuceneTestCase::new("explain_failing_optional_clauses");
    let base = base(&mut tc);
    let explain =
        |q: Clause| base.with_searcher(|s| s.explain(&top_level(q), 0).unwrap().to_string());

    let msm = explain(bq_msm(2, vec![(S, t("w1")), (S, t("xx")), (S, t("zz"))]));
    assert!(
        msm.contains("Failure to match minimum number of optional clauses: 2, matched: 1"),
        "{msm}"
    );
    assert!(
        msm.contains("no match on optional clause (field:xx)"),
        "{msm}"
    );
    assert!(
        msm.contains("no match on optional clause (field:zz)"),
        "{msm}"
    );
    assert!(
        !msm.contains("no match on optional clause (field:w1)"),
        "{msm}"
    );

    let disj = explain(bq(vec![(S, t("xx")), (S, t("zz"))]));
    assert!(
        disj.contains("no match on optional clause (field:xx)"),
        "{disj}"
    );
    assert!(
        disj.contains("no match on optional clause (field:zz)"),
        "{disj}"
    );

    let must_fail = explain(bq(vec![(M, t("missing")), (S, t("xx"))]));
    assert!(
        !must_fail.contains("no match on optional clause"),
        "{must_fail}"
    );
}
