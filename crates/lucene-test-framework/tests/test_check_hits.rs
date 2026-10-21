//! `CheckHits`, `QueryUtils` and `AssertingIndexSearcher` over a small
//! index: sound queries pass every check, and each check fails on the
//! defect it targets.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::Analyzer;
use lucene_index::document::{Document, Store, StringField, TextField};
use lucene_search::collector::{ScoreMode, ScoringCollector, TotalHitsRelation};
use lucene_search::explain::Explanation;
use lucene_search::query::{BooleanQuery, Clause, DisjunctionMaxQuery, PhraseQuery, TermQuery};
use lucene_search::test_support::TestQuery;
use lucene_search::top_docs::{ShardScoreDoc, TopDocs};
use lucene_test_framework::search::asserting::{AssertingCollector, AssertingIndexSearcher};
use lucene_test_framework::search::check_hits;
use lucene_test_framework::search::query_utils;
use lucene_test_framework::search::test_index::TestIndex;
use lucene_util::java_random::JavaRandom;

const DOCS: [&str; 8] = [
    "w1 w2 w3 w4",
    "w1 w3 w2 w3",
    "w1 xx w2 yy w3",
    "w1 w3 xx w2 yy w3",
    "w2 w4",
    "xx yy zz",
    "w4 w4 w4 w1",
    "zz",
];

/// `DOCS` in two segments, with document 7 deleted.
fn index() -> TestIndex {
    let doc = |i: usize| {
        let mut d = Document::new();
        d.add(StringField::new("id", i.to_string(), Store::No));
        d.add(TextField::new("f", DOCS[i], Store::No));
        d
    };
    TestIndex::build(
        Arc::new(Analyzer::new(WhitespaceAnalyzer::default())),
        &[(0..5).map(doc).collect(), (5..8).map(doc).collect()],
        &[("id", "7")],
        &["f"],
    )
}

fn t(term: &str) -> Clause {
    Clause::Term(TermQuery::new("f", term))
}

fn should(clauses: Vec<Clause>) -> BooleanQuery {
    BooleanQuery {
        should: clauses,
        ..BooleanQuery::default()
    }
}

fn queries() -> Vec<(BooleanQuery, Vec<i32>)> {
    vec![
        (should(vec![t("w1")]), vec![0, 1, 2, 3, 6]),
        (should(vec![t("w4"), t("zz")]), vec![0, 4, 5, 6]),
        (
            BooleanQuery {
                must: vec![t("w1")],
                should: vec![t("w4")],
                must_not: vec![t("xx")],
                ..BooleanQuery::default()
            },
            vec![0, 1, 6],
        ),
        (
            should(vec![Clause::Phrase(PhraseQuery::new(
                "f",
                vec![b"w1".to_vec(), b"w2".to_vec()],
            ))]),
            vec![0],
        ),
        (
            should(vec![Clause::DisjunctionMax(Box::new(
                DisjunctionMaxQuery {
                    disjuncts: vec![t("w2"), t("yy")],
                    tie_breaker: 0.5,
                },
            ))]),
            vec![0, 1, 2, 3, 4, 5],
        ),
        (
            should(vec![TestQuery::random_approximation(
                t("w3"),
                JavaRandom::new(9),
            )]),
            vec![0, 1, 2, 3],
        ),
        (should(vec![t("nope")]), vec![]),
    ]
}

#[test]
fn sound_queries_pass_every_check() {
    let index = index();
    index.with_searcher(|s| {
        let mut random = JavaRandom::new(7);
        for (q, results) in queries() {
            check_hits::check_hits(&mut random, &q, s, &results).unwrap();
            check_hits::check_hit_collector(&mut random, &q, s, &results).unwrap();
            check_hits::check_no_match_explanations(&q, s, &results).unwrap();
            check_hits::check_explanations(&q, s, false).unwrap();
            check_hits::check_top_scores(&mut random, &q, s).unwrap();
            query_utils::check_explanations(&q, s).unwrap();

            let asserting = AssertingIndexSearcher::new(&mut random, s);
            assert!(std::ptr::eq(asserting.searcher(), s));
            let top = asserting.search(&q, 10).unwrap();
            let plain = s.search(&q, 10).unwrap();
            check_hits::check_equal(&q, &top.score_docs, &plain.score_docs);
            check_hits::check_hits_query(&q, &top.score_docs, &plain.score_docs, &{
                plain.score_docs.iter().map(|h| h.doc).collect::<Vec<_>>()
            });
            assert_eq!(asserting.count(&q).unwrap(), results.len() as u64);
            if let Some(&doc) = results.first() {
                assert!(asserting.explain(&q, doc).unwrap().matched);
            }
        }
    });
}

/// The panic message of `f`, which must panic.
fn panics(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a failed check");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|m| (*m).to_string()))
        .unwrap_or_default()
}

#[test]
fn hit_checks_fail_on_the_wrong_hits() {
    let index = index();
    index.with_searcher(|s| {
        let mut random = JavaRandom::new(7);
        let q = should(vec![t("w1")]);
        let m = panics(|| check_hits::check_hits(&mut random, &q, s, &[0, 1]).unwrap());
        assert!(m.contains("assertion `left == right` failed"), "{m}");
        let m = panics(|| check_hits::check_hit_collector(&mut random, &q, s, &[0]).unwrap());
        assert!(m.contains("Simple:"), "{m}");
        let m = panics(|| check_hits::check_no_match_explanations(&q, s, &[0, 1, 2]).unwrap());
        assert!(m.contains("doesn't indicate non-match"), "{m}");
    });
}

fn hit(doc: i32, score: f32) -> ShardScoreDoc {
    ShardScoreDoc::new(doc, score)
}

#[test]
fn hit_list_comparisons() {
    let q = should(vec![t("w1")]);
    let a = [hit(1, 1.0), hit(2, 0.5)];
    check_hits::check_doc_ids("ok", &[1, 2], &a);
    let m = panics(|| check_hits::check_doc_ids("x", &[1], &a));
    assert!(m.contains("x nr of hits"), "{m}");
    let m = panics(|| check_hits::check_doc_ids("x", &[1, 3], &a));
    assert!(m.contains("x doc nrs for hit 1"), "{m}");
    let m = panics(|| check_hits::check_equal(&q, &a, &a[..1]));
    assert!(m.contains("Unequal lengths: hits1=2,hits2=1"), "{m}");
    let m = panics(|| check_hits::check_equal(&q, &a, &[hit(1, 1.0), hit(3, 0.5)]));
    assert!(m.contains("Hit 1 docnumbers don't match"), "{m}");
    let m = panics(|| check_hits::check_equal(&q, &a, &[hit(1, 1.0), hit(2, 0.6)]));
    assert!(m.contains("unequal"), "{m}");

    let s = check_hits::hits2str(&a, &a[..1], 0, 0);
    assert_eq!(
        s,
        "Hits length1=2\tlength2=1\nhit=0: doc1=1 shardIndex=-1,\t doc1=1 shardIndex=-1\nhit=1: doc2=0.5 shardIndex=-1,\t\n"
    );
    let s = check_hits::hits2str(&a[..1], &a, 1, 2);
    assert_eq!(
        s,
        "Hits length1=1\tlength2=2\nhit=1:               ,\t doc2=0.5 shardIndex=-1\n"
    );
    let docs = TopDocs {
        total_hits: lucene_search::collector::TotalHits {
            value: 3,
            relation: TotalHitsRelation::GreaterThanOrEqualTo,
        },
        score_docs: a.to_vec(),
    };
    assert_eq!(
        check_hits::topdocs_string(&docs, 1, 0),
        "TopDocs totalHits=3+ hits top=2\n\t1) doc=2\tscore=0.5\n"
    );
    let docs = TopDocs {
        total_hits: lucene_search::collector::TotalHits {
            value: 2,
            relation: TotalHitsRelation::EqualTo,
        },
        ..docs
    };
    assert_eq!(
        check_hits::topdocs_string(&docs, 0, 1),
        "TopDocs totalHits=2 hits top=2\n\t0) doc=1\tscore=1\n"
    );
}

fn m(value: f32, description: &str, details: Vec<Explanation>) -> Explanation {
    Explanation::match_(value, description).with_details(details)
}

#[test]
fn explanations_must_account_for_the_score() {
    let leaf = |v: f32| m(v, "leaf", vec![]);
    let ok = |score: f32, e: &Explanation| check_hits::verify_explanation("q", 0, score, true, e);
    ok(3.0, &m(3.0, "sum of:", vec![leaf(1.0), leaf(2.0)]));
    ok(6.0, &m(6.0, "product of:", vec![leaf(2.0), leaf(3.0)]));
    ok(
        3.0,
        &m(3.0, "max of:", vec![leaf(2.0), leaf(3.0), leaf(-0.0)]),
    );
    ok(
        3.5,
        &m(
            3.5,
            "max plus 0.5 times others of:",
            vec![leaf(3.0), leaf(1.0)],
        ),
    );
    ok(
        7.0,
        &m(
            7.0,
            "x, computed as a + b from:",
            vec![leaf(1.0), leaf(2.0)],
        ),
    );
    ok(7.0, &m(7.0, "weight computed from:", vec![leaf(1.0)]));
    ok(2.0, &m(2.0, "one detail", vec![leaf(2.0)]));
    ok(2.0, &m(2.0, "tf, with freq of:", vec![leaf(9.0)]));
    ok(
        -1.0,
        &m(-1.0, "max plus 0.1 times others of:", vec![leaf(5.0)]),
    );
    // a non-match's details are not combined
    ok(
        0.0,
        &Explanation::no_match("no").with_details(vec![leaf(1.0), leaf(5.0)]),
    );
    check_hits::verify_explanation(
        "q",
        0,
        9.0,
        false,
        &m(9.0, "anything", vec![leaf(1.0), leaf(2.0)]),
    );

    let fails = |score: f32, e: Explanation| panics(move || ok(score, &e));
    let msg = fails(2.0, leaf(3.0));
    assert!(
        msg.contains("score(doc=0)=2 != explanationScore=3"),
        "{msg}"
    );
    let msg = fails(4.0, m(4.0, "sum of:", vec![leaf(1.0), leaf(2.0)]));
    assert!(msg.contains("combined==3 != value=4"), "{msg}");
    let msg = fails(4.0, m(4.0, "list of:", vec![leaf(1.0), leaf(3.0)]));
    assert!(msg.contains("multi valued explanation"), "{msg}");
    let msg = fails(
        4.0,
        m(
            4.0,
            "max plus x times others of:",
            vec![leaf(1.0), leaf(3.0)],
        ),
    );
    assert!(msg.contains("multi valued explanation"), "{msg}");
    let msg = fails(
        4.0,
        m(
            4.0,
            "max plus 0.5 then others of:",
            vec![leaf(1.0), leaf(3.0)],
        ),
    );
    assert!(msg.contains("multi valued explanation"), "{msg}");
    let msg = fails(4.0, m(4.0, "max plus 0.5", vec![leaf(1.0), leaf(3.0)]));
    assert!(msg.contains("multi valued explanation"), "{msg}");
    let msg = fails(2.0, m(2.0, "score based on 3 child docs in range", vec![]));
    assert!(msg.contains("Child doc explanations are missing"), "{msg}");
    // NaN equals NaN (`Double.compare`), so only a NaN beside a number fails
    ok(
        f32::NAN,
        &m(f32::NAN, "max of:", vec![leaf(f32::NAN), leaf(1.0)]),
    );
    let msg = fails(2.0, m(2.0, "max of:", vec![leaf(f32::NAN), leaf(1.0)]));
    assert!(msg.contains("combined==NaN"), "{msg}");
}

/// A collector keeping its hits.
#[derive(Default)]
struct Hits(Vec<i32>);

impl ScoringCollector for Hits {
    fn collect(&mut self, doc: i32, _score: f32) {
        self.0.push(doc);
    }
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }
}

#[test]
fn the_asserting_collector_forwards_in_order_hits() {
    let mut hits = Hits::default();
    let mut c = AssertingCollector::wrap(&mut hits);
    assert_eq!(c.score_mode(), ScoreMode::CompleteNoScores);
    let _ = (
        c.min_competitive_score(),
        c.pruning_threshold(),
        c.constant_score_hits_needed(),
        c.count_losing_hits(0, 0.0),
        c.add_hits(0),
    );
    c.collect(1, 0.0);
    c.collect_range(3, 5, 1.0);
    c.collect(5, 2.0);
    assert_eq!(hits.0, [1, 3, 4, 5]);
}

/// A misuse of the asserting collector, by the message it panics with.
type Misuse = (&'static str, fn(&mut AssertingCollector<'_, Hits>));

#[test]
fn the_asserting_collector_fires() {
    let cases: [Misuse; 5] = [
        ("collection is not in order: current doc=1 while 2", |c| {
            c.collect(2, 0.0);
            c.collect(1, 0.0);
        }),
        ("score=-1 for doc 1", |c| c.collect(1, -1.0)),
        ("empty range [3, 3)", |c| c.collect_range(3, 3, 0.0)),
        ("collection is not in order: current doc=3 while 4", |c| {
            c.collect_range(3, 5, 0.0);
            c.collect_range(3, 6, 0.0);
        }),
        ("score=NaN for docs [3, 5)", |c| {
            c.collect_range(3, 5, f32::NAN)
        }),
    ];
    for (expected, f) in cases {
        let msg = panics(|| {
            let mut hits = Hits::default();
            f(&mut AssertingCollector::wrap(&mut hits));
        });
        assert!(msg.contains(expected), "{expected}: {msg}");
    }
}

/// A collector that, through an asserting searcher, sees the documents
/// AssertingCollector lets through.
#[test]
fn the_asserting_searcher_runs_collectors_under_the_wrappers() {
    let index = index();
    index.with_searcher(|s| {
        let mut random = JavaRandom::new(3);
        let asserting = AssertingIndexSearcher::new(&mut random, s);
        let mut hits = Hits::default();
        asserting
            .search_collector(&should(vec![t("w4")]), &mut hits)
            .unwrap();
        assert_eq!(hits.0, [0, 4, 6]);
        assert!(!lucene_search::test_support::asserting_active());
    });
}
