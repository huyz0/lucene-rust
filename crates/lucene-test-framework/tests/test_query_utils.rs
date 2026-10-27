//! `TestQueryUtils`: `checkSkipTo` and `checkFirstSkipTo` catch each of
//! the broken scorers (`TestQuery::broken`), and pass a sound query and
//! its random approximation.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::Analyzer;
use lucene_index::document::{Document, Store, TextField};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{
    BooleanQuery, Clause, ConstantScoreQuery, MatchAllDocsQuery, TermQuery,
};
use lucene_search::test_support::{BrokenKind, TestQuery};
use lucene_test_framework::search::query_utils;
use lucene_test_framework::search::test_index::TestIndex;
use lucene_util::java_random::JavaRandom;

/// `setUp`: five documents of `f:v`, in one segment or one per document.
fn index(segments: usize) -> TestIndex {
    let doc = || {
        let mut d = Document::new();
        d.add(TextField::new("f", "v", Store::No));
        d
    };
    let per = 5 / segments;
    let segs: Vec<Vec<Document>> = (0..segments)
        .map(|_| (0..per).map(|_| doc()).collect())
        .collect();
    TestIndex::build(
        Arc::new(Analyzer::new(WhitespaceAnalyzer::default())),
        &segs,
        &[],
        &["f"],
    )
}

fn query(clause: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![clause],
        ..BooleanQuery::default()
    }
}

fn term() -> Clause {
    Clause::Term(TermQuery::new("f", "v"))
}

type Check = fn(&BooleanQuery, &IndexSearcher<'_, '_>) -> lucene_search::Result<()>;

/// The panic message of `check` over `q`, which must panic.
fn expect_panic(index: &TestIndex, q: &BooleanQuery, check: Check) -> String {
    index.with_searcher(|s| {
        let err = catch_unwind(AssertUnwindSafe(|| check(q, s)))
            .expect_err("the check passed a broken query");
        err.downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|m| (*m).to_string()))
            .unwrap_or_default()
    })
}

const CHECKS: [(&str, Check); 2] = [
    ("checkSkipTo", query_utils::check_skip_to),
    ("checkFirstSkipTo", query_utils::check_first_skip_to),
];

#[test]
fn checks_detect_broken_advance() {
    let index = index(1);
    let q = query(TestQuery::broken(term(), BrokenKind::Advance));
    for (name, check) in CHECKS {
        let m = expect_panic(&index, &q, check);
        assert!(m.contains("no more docs"), "{name}: {m}");
    }
}

#[test]
fn checks_detect_wrong_doc_from_advance() {
    let index = index(1);
    let q = query(TestQuery::broken(term(), BrokenKind::SkipTooFar));
    let m = expect_panic(&index, &q, query_utils::check_skip_to);
    assert!(m.contains("scorerDoc="), "{m}");
    let m = expect_panic(&index, &q, query_utils::check_first_skip_to);
    assert!(m.contains("got to"), "{m}");
}

#[test]
fn checks_detect_unstable_score() {
    let index = index(1);
    let q = query(TestQuery::broken(term(), BrokenKind::UnstableScore));
    let m = expect_panic(&index, &q, query_utils::check_skip_to);
    assert!(m.contains("scorerScore2="), "{m}");
    let m = expect_panic(&index, &q, query_utils::check_first_skip_to);
    assert!(m.contains("unstable advance"), "{m}");
}

#[test]
fn checks_detect_score_mismatch() {
    let index = index(1);
    let q = query(TestQuery::broken(term(), BrokenKind::ScoreMismatch));
    let m = expect_panic(&index, &q, query_utils::check_skip_to);
    assert!(m.contains("score="), "{m}");
    let m = expect_panic(&index, &q, query_utils::check_first_skip_to);
    assert!(m.contains("but advance("), "{m}");
}

#[test]
fn checks_detect_early_termination() {
    let index = index(1);
    let q = query(TestQuery::broken(term(), BrokenKind::EarlyTermination));
    for (name, check) in CHECKS {
        let m = expect_panic(&index, &q, check);
        assert!(m.contains("last doc was 0"), "{name}: {m}");
    }
}

#[test]
fn sound_queries_pass_every_check() {
    let mut random = JavaRandom::new(42);
    for segments in [1, 5] {
        let index = index(segments);
        index.with_searcher(|s| {
            for clause in [
                term(),
                TestQuery::random_approximation(term(), JavaRandom::new(random.next_long())),
            ] {
                query_utils::check_with_searcher(&mut random, &query(clause), s).unwrap();
            }
        });
    }
}

/// Queries whose bulk scorers hand matches over in batches (`checkCount`'s
/// `DocIdStream` paths) or find nothing at all.
#[test]
fn batching_and_empty_queries_pass_every_check() {
    let mut random = JavaRandom::new(7);
    let doc = |text: &str| {
        let mut d = Document::new();
        d.add(TextField::new("f", text, Store::No));
        d
    };
    let segs: Vec<Vec<Document>> = vec![
        (0..2100)
            .map(|i| doc(if i % 3 == 0 { "v w" } else { "v" }))
            .collect(),
        (0..10).map(|_| doc("w")).collect(),
    ];
    let index = TestIndex::build(
        Arc::new(Analyzer::new(WhitespaceAnalyzer::default())),
        &segs,
        &[],
        &["f"],
    );
    index.with_searcher(|s| {
        let w = || Clause::Term(TermQuery::new("f", "w"));
        for clause in [
            Clause::MatchAllDocs(MatchAllDocsQuery::new(2110)),
            Clause::ConstantScore(Box::new(ConstantScoreQuery::new(term(), 2.0))),
            Clause::Term(TermQuery::new("f", "absent")),
            Clause::Boolean(Box::new(BooleanQuery {
                should: vec![term(), w()],
                ..BooleanQuery::default()
            })),
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![w()],
                filter: vec![term()],
                ..BooleanQuery::default()
            })),
        ] {
            query_utils::check_with_searcher(&mut random, &query(clause), s).unwrap();
        }
    });
}

#[test]
fn test_queries_compare_by_their_wrapped_query() {
    let a = TestQuery::random_approximation(term(), JavaRandom::new(1));
    let b = TestQuery::random_approximation(term(), JavaRandom::new(2));
    assert_eq!(a, b);
    assert_ne!(a, TestQuery::broken(term(), BrokenKind::Advance));
    assert_ne!(
        TestQuery::broken(term(), BrokenKind::Advance),
        TestQuery::broken(term(), BrokenKind::SkipTooFar)
    );
    let Clause::Extended(e) = &a else {
        panic!("not an extended clause")
    };
    assert_eq!(e.name(), "RandomApproximationQuery");
    assert_eq!(e.children(), vec![&term()]);
    let Clause::Extended(e) = TestQuery::broken(term(), BrokenKind::Advance) else {
        panic!("not an extended clause")
    };
    assert_eq!(e.name(), "BrokenQuery");
    assert!(format!("{e:?}").contains("Broken(Advance)"));
    assert!(format!("{a:?}").contains("RandomApproximation"));
}

/// `wrapUnderlyingReader`: the searcher's segments first, then the empty,
/// fully deleted readers of each edge; the same hits, and every check
/// still passing over it.
#[test]
fn wrapped_reader_pads_with_deleted_leaves() {
    let index = index(5);
    let q = query(term());
    index.with_searcher(|s| {
        let hits = s.search(&q, 10).expect("search").score_docs;
        for (edge, padding) in [(-1, 16), (0, 12), (1, 0)] {
            query_utils::with_wrapped_underlying_reader(s, edge, |w| {
                let extra = query_utils::wrap_padding(edge).len();
                assert_eq!(w.segments().len(), s.segments().len() + extra);
                assert_eq!(w.max_doc(), s.max_doc() + padding);
                let wrapped = w.search(&q, 10)?.score_docs;
                assert_eq!(
                    wrapped.iter().map(|h| (h.doc, h.score)).collect::<Vec<_>>(),
                    hits.iter().map(|h| (h.doc, h.score)).collect::<Vec<_>>()
                );
                assert_eq!(w.count(&q)?, 5);
                Ok(())
            })
            .expect("wrapped searcher");
        }
        query_utils::check_with_searcher(&mut JavaRandom::new(7), &q, s).expect("checks pass");
    });
}
