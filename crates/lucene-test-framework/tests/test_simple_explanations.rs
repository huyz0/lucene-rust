//! Lucene's `TestSimpleExplanations`: each query through
//! `BaseExplanationTestCase.qtest` -- every `QueryUtils` check, the
//! explanations and the matches -- hitting exactly the documents named.
//!
//! The span-free part of the Java suite, test for test. Each test draws
//! from its own `LuceneTestCase` stream (`TESTS_SEED` reproduces it).

mod explanation_dsl;

use explanation_dsl::*;
use lucene_search::explain::Explanation;

qtests! {
    t1: t("w1") => [0, 1, 2, 3];
    t2: boost(t("w1"), 100.0) => [0, 1, 2, 3];
    ma1: match_all() => [0, 1, 2, 3];
    ma2: boost(match_all(), 1000.0) => [0, 1, 2, 3];
    p1: phrase(0, &["w1", "w2"]) => [0];
    p2: phrase(0, &["w1", "w3"]) => [1, 3];
    p3: phrase(1, &["w1", "w2"]) => [0, 1, 2];
    p4: phrase(1, &["w2", "w3"]) => [0, 1, 2, 3];
    p5: phrase(1, &["w3", "w2"]) => [1, 3];
    p6: phrase(2, &["w3", "w2"]) => [0, 1, 3];
    p7: phrase(3, &["w3", "w2"]) => [0, 1, 2, 3];
    csq1: csq(match_these_items(&[0, 1, 2, 3])) => [0, 1, 2, 3];
    csq2: csq(match_these_items(&[1, 3])) => [1, 3];
    csq3: boost(csq(match_these_items(&[0, 2])), 1000.0) => [0, 2];
    dmq1: dmq(vec![t("w1"), t("w5")], 0.0) => [0, 1, 2, 3];
    dmq2: dmq(vec![t("w1"), t("w5")], 0.5) => [0, 1, 2, 3];
    dmq3: dmq(vec![t("QQ"), t("w5")], 0.5) => [0];
    dmq4: dmq(vec![t("QQ"), t("xx")], 0.5) => [2, 3];
    dmq5: dmq(vec![bq(vec![(S, t("yy")), (N, t("QQ"))]), t("xx")], 0.5) => [2, 3];
    dmq6: dmq(vec![bq(vec![(N, t("yy")), (S, t("w3"))]), t("xx")], 0.5) => [0, 1, 2, 3];
    dmq7: dmq(vec![bq(vec![(N, t("yy")), (S, t("w3"))]), t("w2")], 0.5) => [0, 1, 2, 3];
    dmq8: dmq(
        vec![bq(vec![(S, t("yy")), (S, boost(t("w5"), 100.0))]), boost(t("xx"), 100_000.0)],
        0.5
    ) => [0, 2, 3];
    dmq9: dmq(
        vec![bq(vec![(S, t("yy")), (S, boost(t("w5"), 100.0))]), boost(t("xx"), 0.0)],
        0.5
    ) => [0, 2, 3];
    mpq1: mpq(0, &[&["w1"], &["w2", "w3", "xx"]]) => [0, 1, 2, 3];
    mpq2: mpq(0, &[&["w1"], &["w2", "w3"]]) => [0, 1, 3];
    mpq3: mpq(0, &[&["w1", "xx"], &["w2", "w3"]]) => [0, 1, 2, 3];
    mpq4: mpq(0, &[&["w1"], &["w2"]]) => [0];
    mpq5: mpq(1, &[&["w1"], &["w2"]]) => [0, 1, 2];
    mpq6: mpq(1, &[&["w1", "w3"], &["w2"]]) => [0, 1, 2, 3];
    bq1: bq(vec![(M, t("w1")), (M, t("w2"))]) => [0, 1, 2, 3];
    bq2: bq(vec![(M, t("yy")), (M, t("w3"))]) => [2, 3];
    bq3: bq(vec![(S, t("yy")), (M, t("w3"))]) => [0, 1, 2, 3];
    bq4: bq(vec![(S, t("w1")), (S, bq(vec![(N, t("xx")), (S, t("w2"))]))]) => [0, 1, 2, 3];
    bq5: bq(vec![(S, t("w1")), (S, bq(vec![(M, t("qq")), (S, t("w2"))]))]) => [0, 1, 2, 3];
    bq6: bq(vec![(S, t("w1")), (N, bq(vec![(N, t("qq")), (S, t("w5"))]))]) => [1, 2, 3];
    bq7: nested(M) => [0];
    bq8: nested(S) => [0, 1, 2, 3];
    bq9: nested_excl(S) => [0, 1, 2, 3];
    bq10: nested_excl(M) => [1];
    bq11: bq(vec![(S, t("w1")), (S, boost(t("w1"), 1000.0))]) => [0, 1, 2, 3];
    bq14: bq(vec![(S, t("QQQQQ")), (S, t("w1"))]) => [0, 1, 2, 3];
    bq15: bq(vec![(N, t("QQQQQ")), (S, t("w1"))]) => [0, 1, 2, 3];
    bq16: bq(vec![(S, t("QQQQQ")), (S, bq(vec![(S, t("w1")), (N, t("xx"))]))]) => [0, 1];
    bq17: bq(vec![(S, t("w2")), (S, bq(vec![(S, t("w1")), (N, t("xx"))]))]) => [0, 1, 2, 3];
    bq19: bq(vec![(N, t("yy")), (S, t("w3"))]) => [0, 1];
    bq20: bq_msm(
        2,
        vec![(S, t("QQQQQ")), (S, t("yy")), (S, t("zz")), (S, t("w5")), (S, t("w4"))]
    ) => [0, 3];
    bq21: bq(vec![(S, t("yy")), (S, t("zz"))]) => [1, 2, 3];
    bq23: bq(vec![(F, t("w1")), (F, t("w2"))]) => [0, 1, 2, 3];
    bq24: bq(vec![(F, t("w1")), (S, t("w2"))]) => [0, 1, 2, 3];
    bq25: bq(vec![(F, t("w1")), (M, t("w2"))]) => [0, 1, 2, 3];
    bq26: bq(vec![(F, t("w1")), (N, t("xx"))]) => [0, 1];
    multi_field_bq1: bq(vec![(M, t("w1")), (M, a("w2"))]) => [0, 1, 2, 3];
    multi_field_bq2: bq(vec![(M, t("yy")), (M, a("w3"))]) => [2, 3];
    multi_field_bq3: bq(vec![(S, t("yy")), (M, a("w3"))]) => [0, 1, 2, 3];
    multi_field_bq4: bq(vec![(S, t("w1")), (S, bq(vec![(N, t("xx")), (S, a("w2"))]))]) => [0, 1, 2, 3];
    multi_field_bq5: bq(vec![(S, t("w1")), (S, bq(vec![(M, a("qq")), (S, a("w2"))]))]) => [0, 1, 2, 3];
    multi_field_bq6: bq(vec![(S, t("w1")), (N, bq(vec![(N, a("qq")), (S, a("w5"))]))]) => [1, 2, 3];
    multi_field_bq7: bq(vec![
        (M, t("w1")),
        (M, bq(vec![
            (S, a("qq")),
            (S, bq(vec![(S, a("xx")), (N, a("w2"))])),
            (S, bq(vec![(M, a("w3")), (M, a("w4"))])),
        ])),
    ]) => [0];
    multi_field_bq8: bq(vec![
        (M, a("w1")),
        (S, bq(vec![
            (S, t("qq")),
            (S, bq(vec![(S, a("xx")), (N, t("w2"))])),
            (S, bq(vec![(M, a("w3")), (M, t("w4"))])),
        ])),
    ]) => [0, 1, 2, 3];
    multi_field_bq9: bq(vec![
        (M, t("w1")),
        (S, bq(vec![
            (S, a("qq")),
            (S, bq(vec![(N, t("xx")), (S, t("w2"))])),
            (N, bq(vec![(M, a("w3")), (M, t("w4"))])),
        ])),
    ]) => [0, 1, 2, 3];
    multi_field_bq10: bq(vec![
        (M, t("w1")),
        (M, bq(vec![
            (S, a("qq")),
            (S, bq(vec![(N, t("xx")), (S, a("w2"))])),
            (N, bq(vec![(M, a("w3")), (M, t("w4"))])),
        ])),
    ]) => [1];
    multi_field_bq_of_pq1: bq(vec![(S, phrase(0, &["w1", "w2"])), (S, alt_phrase(0, &["w1", "w2"]))]) => [0];
    multi_field_bq_of_pq2: bq(vec![(S, phrase(0, &["w1", "w3"])), (S, alt_phrase(0, &["w1", "w3"]))]) => [1, 3];
    multi_field_bq_of_pq3: bq(vec![(S, phrase(1, &["w1", "w2"])), (S, alt_phrase(1, &["w1", "w2"]))]) => [0, 1, 2];
    multi_field_bq_of_pq4: bq(vec![(S, phrase(1, &["w2", "w3"])), (S, alt_phrase(1, &["w2", "w3"]))]) => [0, 1, 2, 3];
    multi_field_bq_of_pq5: bq(vec![(S, phrase(1, &["w3", "w2"])), (S, alt_phrase(1, &["w3", "w2"]))]) => [1, 3];
    multi_field_bq_of_pq6: bq(vec![(S, phrase(2, &["w3", "w2"])), (S, alt_phrase(2, &["w3", "w2"]))]) => [0, 1, 3];
    multi_field_bq_of_pq7: bq(vec![(S, phrase(3, &["w3", "w2"])), (S, alt_phrase(1, &["w3", "w2"]))]) => [0, 1, 2, 3];
    synonym_query: synonym(&["w1", "w2"]) => [0, 1, 2, 3];
}

/// `testBQ7`/`testBQ8`'s shape: `FIELD:w1` required beside a disjunction
/// (itself `inner`) of `qq`, `xx` without `w2`, and `w3` with `w4`.
fn nested(inner: O) -> lucene_search::query::Clause {
    bq(vec![
        (M, t("w1")),
        (
            inner,
            bq(vec![
                (S, t("qq")),
                (S, bq(vec![(S, t("xx")), (N, t("w2"))])),
                (S, bq(vec![(M, t("w3")), (M, t("w4"))])),
            ]),
        ),
    ])
}

/// `testBQ9`/`testBQ10`'s shape: `w2` without `xx`, and the `w3`/`w4` pair
/// prohibited.
fn nested_excl(inner: O) -> lucene_search::query::Clause {
    bq(vec![
        (M, t("w1")),
        (
            inner,
            bq(vec![
                (S, t("qq")),
                (S, bq(vec![(N, t("xx")), (S, t("w2"))])),
                (N, bq(vec![(M, t("w3")), (M, t("w4"))])),
            ]),
        ),
    ])
}

/// `testEquality`: explanations are equal by value, description, match and
/// details.
#[test]
fn equality() {
    let e1 = Explanation::match_(1.0, "an explanation");
    let e2 = Explanation::match_(1.0, "an explanation")
        .with_details(vec![Explanation::match_(1.0, "a subexplanation")]);
    let e25 = Explanation::match_(1.0, "an explanation").with_details(vec![Explanation::match_(
        1.0,
        "a subexplanation",
    )
    .with_details(vec![Explanation::match_(1.0, "a subsubexplanation")])]);
    let e3 = Explanation::match_(1.0, "an explanation");
    let e4 = Explanation::match_(2.0, "an explanation");
    let e5 = Explanation::no_match("an explanation");
    let e6 = Explanation::no_match("an explanation")
        .with_details(vec![Explanation::match_(1.0, "a subexplanation")]);
    let e7 = Explanation::no_match("an explanation");
    let e8 = Explanation::match_(1.0, "another explanation");
    assert_eq!(e1, e3);
    assert_ne!(e1, e2);
    assert_ne!(e2, e25);
    assert_ne!(e1, e4);
    assert_ne!(e1, e5);
    assert_eq!(e5, e7);
    assert_ne!(e5, e6);
    assert_ne!(e1, e8);
}
