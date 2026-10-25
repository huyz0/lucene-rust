//! Actual Lucene BlockScoreQueryWrapper traces, including sentinel-aligned blocks.
use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::Analyzer;
use lucene_index::document::{Document, Store, StringField};
use lucene_search::collector::ScoreMode;
use lucene_search::query::{BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, TermQuery};
use lucene_search::test_support::{with_leaf, NO_MORE_DOCS};
use lucene_test_framework::search::block_score_query_wrapper::wrap;
use lucene_test_framework::search::test_index::TestIndex;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

const DOCS: [i32; 8] = [0, 2, 3, 6, 8, 9, 11, 14];
const SCORES: [f32; 8] = [0.25, 2.0, 1.0, 5.0, 0.5, 3.0, 7.0, 4.0];
const LENGTHS: [i32; 7] = [1, 2, 3, 4, 7, 32, i32::MAX];
const TARGETS: [i32; 13] = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 15, i32::MAX];

fn inner(empty: bool) -> Clause {
    if empty {
        return TermQuery::new("id", "missing").into();
    }
    BooleanQuery::new()
        .with_should(DOCS.into_iter().zip(SCORES).map(|(doc, score)| {
            let term: Clause = TermQuery::new("id", doc.to_string()).into();
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::ConstantScore(Box::new(ConstantScoreQuery::new(term, 1.0))),
                score,
            )))
        }))
        .into()
}
fn must(query: Clause) -> BooleanQuery {
    BooleanQuery::new().with_must([query])
}
fn index() -> TestIndex {
    let segments = [16, 6].map(|size| {
        (0..size)
            .map(|id| {
                let mut doc = Document::new();
                doc.add(StringField::new("id", id.to_string(), Store::No));
                doc
            })
            .collect()
    });
    TestIndex::build(
        Arc::new(Analyzer::new(WhitespaceAnalyzer::default())),
        &segments,
        &[],
        &[],
    )
}

#[test]
fn every_java_scorer_trace_matches() {
    let mut out = Vec::<String>::new();
    macro_rules! line { ($($v:expr),* $(,)?) => { out.push(vec![$(format!("{}", $v)),*].join("\t")); }; }
    index().with_searcher(|searcher| {
        for empty in [false, true] {
            for length in LENGTHS {
                let query = wrap(inner(empty), length);
                line!(
                    "meta",
                    empty,
                    length,
                    lucene_search::test_support::query_string(&query)
                        == lucene_search::test_support::query_string(&inner(empty)),
                    query == wrap(inner(empty), length),
                    query == wrap(inner(empty), if length == 1 { 2 } else { 1 })
                );
                line!(
                    "visit",
                    empty,
                    length,
                    lucene_search::query_visitor::extract_terms(&query).len()
                );
                for leaf in 0..2 {
                    with_leaf(searcher, &must(query.clone()), leaf, |l| {
                        let scorer = l.scorer(ScoreMode::Complete)?;
                        line!(
                            "leaf",
                            empty,
                            length,
                            leaf,
                            scorer.is_none(),
                            lucene_search::segment_cacheable::is_cacheable(&query, None)
                        );
                        let Some(mut scorer) = scorer else {
                            return Ok(());
                        };
                        line!(
                            "initial",
                            empty,
                            length,
                            leaf,
                            scorer.doc_id(),
                            format!("{:08x}", scorer.score()?.to_bits()),
                            scorer.cost(),
                            !scorer.two_phase()
                        );
                        loop {
                            let doc = scorer.next_doc()?;
                            line!(
                                "next",
                                empty,
                                length,
                                leaf,
                                doc,
                                format!("{:08x}", scorer.score()?.to_bits())
                            );
                            if doc == NO_MORE_DOCS {
                                break;
                            }
                        }
                        for target in TARGETS {
                            let mut scorer = l.scorer(ScoreMode::Complete)?.unwrap();
                            let doc = scorer.advance(target)?;
                            line!(
                                "advance",
                                empty,
                                length,
                                leaf,
                                target,
                                doc,
                                format!("{:08x}", scorer.score()?.to_bits())
                            );
                            let mut scorer = l.scorer(ScoreMode::Complete)?.unwrap();
                            let end = scorer.advance_shallow(target)?;
                            for up_to in [target, end, i32::MAX] {
                                line!(
                                    "shallow",
                                    empty,
                                    length,
                                    leaf,
                                    target,
                                    end,
                                    up_to,
                                    format!("{:08x}", scorer.max_score(up_to)?.to_bits()),
                                    scorer.doc_id()
                                );
                            }
                            let mut scorer = l.scorer(ScoreMode::Complete)?.unwrap();
                            let first = scorer.next_doc()?;
                            let end = scorer.advance_shallow(first.max(target))?;
                            line!(
                                "mixed",
                                empty,
                                length,
                                leaf,
                                target,
                                end,
                                format!("{:08x}", scorer.max_score(i32::MAX)?.to_bits()),
                                scorer.doc_id()
                            );
                        }
                        Ok(())
                    })
                    .unwrap();
                }
            }
        }
        for length in [0, -1, i32::MIN] {
            let query = must(wrap(inner(false), length));
            let failed = catch_unwind(AssertUnwindSafe(|| {
                with_leaf(searcher, &query, 0, |l| {
                    l.scorer(ScoreMode::Complete)?.unwrap().advance_shallow(0)
                })
                .unwrap()
            }))
            .is_err();
            let error = if !failed {
                "none"
            } else if length == 0 {
                "ArithmeticException"
            } else {
                "ArrayIndexOutOfBoundsException"
            };
            line!("invalid", length, error);
            for (label, mode) in [
                ("COMPLETE_NO_SCORES", ScoreMode::CompleteNoScores),
                ("TOP_DOCS", ScoreMode::TopDocs),
            ] {
                with_leaf(searcher, &query, 0, |l| {
                    let mut scorer = l.scorer(mode)?.unwrap();
                    let mut count = 0;
                    while scorer.exact_next()? != NO_MORE_DOCS {
                        count += 1;
                    }
                    line!("noscores", length, label, count);
                    Ok(())
                })
                .unwrap();
            }
        }
        let rewrite_inner: Clause = Clause::Boost(Box::new(BoostQuery::new(inner(false), 1.0)));
        line!(
            "rewrite",
            wrap(rewrite_inner.clone(), 3).rewrite() == wrap(rewrite_inner.rewrite(), 3)
        );
    });
    let actual = out.join("\n") + "\n";
    let expected =
        include_str!("../../../fixtures/testframework/data/block_score_query_wrapper.tsv");
    assert_eq!(actual, expected);
}

#[test]
fn rewrites_visits_explanations_and_score_modes_use_the_child() {
    use lucene_search::query::MatchAllDocsQuery;
    index().with_searcher(|searcher| {
        for length in [1, 2, 3, 17] {
            let original = must(inner(false));
            let wrapped = must(wrap(inner(false), length));
            let a = searcher.search(&original, 32).unwrap();
            let b = searcher.search(&wrapped, 32).unwrap();
            assert_eq!(a, b);
            for doc in [0, 1, 2, 3, 15, 16, 21] {
                assert_eq!(
                    searcher.explain(&original, doc).unwrap(),
                    searcher.explain(&wrapped, doc).unwrap()
                );
            }
        }
        let all = must(wrap(Clause::MatchAllDocs(MatchAllDocsQuery::new(0)), 3));
        for doc in [0, 15, 16, 21] {
            let explanation = searcher.explain(&all, doc).unwrap();
            assert!(explanation.matched, "{explanation:?}");
        }
    });
}

#[test]
fn no_score_weight_retains_the_childs_specialized_bulk_scorer() {
    index().with_searcher(|searcher| {
        let term: Clause = TermQuery::new("id", "2").into();
        let original = must(term.clone());
        let wrapped = must(wrap(term.clone(), 0));
        with_leaf(searcher, &original, 0, |l| {
            assert!(!l
                .bulk_scorer(ScoreMode::CompleteNoScores)?
                .unwrap()
                .uses_default_scorer());
            Ok(())
        })
        .unwrap();
        with_leaf(searcher, &wrapped, 0, |l| {
            assert!(!l
                .bulk_scorer(ScoreMode::CompleteNoScores)?
                .unwrap()
                .uses_default_scorer());
            assert!(l
                .bulk_scorer(ScoreMode::Complete)?
                .unwrap()
                .uses_default_scorer());
            Ok(())
        })
        .unwrap();
        with_leaf(searcher, &wrapped, 0, |l| {
            assert!(!l
                .bulk_scorer(ScoreMode::TopDocs)?
                .unwrap()
                .uses_default_scorer());
            let mut scorer = l.scorer(ScoreMode::TopDocs)?.unwrap();
            assert_eq!(scorer.next_doc()?, 2);
            assert_eq!(scorer.next_doc()?, NO_MORE_DOCS);
            Ok(())
        })
        .unwrap();
        let boosted = must(Clause::Boost(Box::new(BoostQuery::new(wrap(term, 0), 2.0))));
        with_leaf(searcher, &boosted, 0, |l| {
            assert!(!l
                .bulk_scorer(ScoreMode::CompleteNoScores)?
                .unwrap()
                .uses_default_scorer());
            Ok(())
        })
        .unwrap();
    });
}
