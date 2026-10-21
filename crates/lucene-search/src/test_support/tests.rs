//! The seam over a small two-segment index: leaf scorers and bulk scorers,
//! and the test queries' scorers against their wrapped query's.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::Analyzer;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, Store, StringField, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::ByteBuffersDirectory;

use super::*;
use crate::directory_reader::DirectoryReader;
use crate::index_searcher::SegmentNorms;
use crate::query::TermQuery;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// Segment 0: `a b`, `b` (deleted), `a a c`, `c`, `a c`; segment 1: `a`,
/// `b c`.
fn index() -> ByteBuffersDirectory {
    let dir = ByteBuffersDirectory::new();
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_analyzer(Some(Arc::new(Analyzer::new(WhitespaceAnalyzer::default()))));
    for (i, segment) in [&["a b", "b", "a a c", "c", "a c"][..], &["a", "b c"][..]]
        .iter()
        .enumerate()
    {
        for (j, text) in segment.iter().enumerate() {
            let mut d = Document::new();
            d.add(StringField::new("id", format!("{i}{j}"), Store::No));
            d.add(TextField::new("f", *text, Store::No));
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
    }
    w.delete_documents_by_term(&[Term::new("id", "01")])
        .unwrap();
    w.commit().unwrap();
    w.close().unwrap();
    dir
}

fn with_searcher<R>(f: impl FnOnce(&IndexSearcher<'_, '_>) -> R) -> R {
    let dir = index();
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["f".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    f(&searcher)
}

fn term(t: &str) -> Clause {
    Clause::Term(TermQuery::new("f", t))
}

fn must(clause: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![clause],
        ..BooleanQuery::default()
    }
}

/// The exact iterator's `(doc, score)`s of `query` in `leaf`.
fn exact(searcher: &IndexSearcher<'_, '_>, query: &BooleanQuery, leaf: usize) -> Vec<(i32, f32)> {
    with_leaf(searcher, query, leaf, |l| {
        let mut out = Vec::new();
        let Some(mut s) = l.scorer(ScoreMode::Complete)? else {
            return Ok(out);
        };
        while s.exact_next()? != NO_MORE_DOCS {
            out.push((s.doc_id(), s.score()?));
        }
        Ok(out)
    })
    .unwrap()
}

/// Every hit, as `(doc, score)`.
#[derive(Default)]
struct All(Vec<(i32, f32)>);

impl ScoringCollector for All {
    fn collect(&mut self, doc: i32, score: f32) {
        self.0.push((doc, score));
    }
}

#[test]
fn leaves_expose_their_scorers() {
    with_searcher(|s| {
        assert!(matches!(
            with_leaf(s, &must(term("a")), 2, |_| Ok(())),
            Err(Error::SliceOutOfRange {
                segment: 2,
                segments: 2
            })
        ));
        let q = must(term("a"));
        with_leaf(s, &q, 0, |l| {
            assert_eq!(l.doc_base(), 0);
            assert_eq!(l.max_doc(), Some(5));
            assert!(!l.live_docs().unwrap().get_doc(1));
            let mut sc = l.scorer(ScoreMode::TopScores)?.unwrap();
            assert_eq!(sc.cost(), 3);
            assert!(!sc.two_phase());
            assert_eq!(sc.match_cost(), 0.0);
            assert_eq!(sc.doc_id(), -1);
            assert!(sc.advance_shallow(0)? >= 0);
            let max = sc.max_score(4)?;
            assert_eq!(sc.next_doc()?, 0);
            assert!(sc.matches()?);
            assert!(sc.score()? <= max);
            assert!(sc.doc_id_run_end() > 0);
            sc.set_min_competitive_score(0.0)?;
            assert_eq!(sc.advance(1)?, 2);
            assert_eq!(sc.exact_advance(3)?, 4);
            assert_eq!(sc.exact_next()?, NO_MORE_DOCS);
            assert!(l.scorer(ScoreMode::CompleteNoScores)?.is_some());
            assert!(l.scorer(ScoreMode::TopDocs)?.is_some());
            Ok(())
        })
        .unwrap();
        with_leaf(s, &must(term("zzz")), 1, |l| {
            assert_eq!(l.doc_base(), 5);
            assert!(l.scorer(ScoreMode::Complete)?.is_none());
            assert!(l.bulk_scorer(ScoreMode::Complete)?.is_none());
            Ok(())
        })
        .unwrap();
    });
}

#[test]
fn bulk_scorers_collect_what_scorers_find_with_or_without_the_wrappers() {
    with_searcher(|s| {
        for (t, seed) in [("a", None), ("b", None), ("a", Some(1)), ("b", Some(2))] {
            let q = must(term(t));
            let expected: Vec<(i32, f32)> = exact(s, &q, 0)
                .into_iter()
                .filter(|&(d, _)| d != 1)
                .collect();
            let _guard = seed.map(asserting);
            assert_eq!(asserting_active(), seed.is_some());
            let hits = with_leaf(s, &q, 0, |l| {
                let mut b = l.bulk_scorer(ScoreMode::Complete)?.unwrap();
                let mut all = All::default();
                let next = b.score(&mut all, l.live_docs(), 0, 2)?;
                assert!(next >= 2);
                assert_eq!(
                    b.score(&mut all, l.live_docs(), next, NO_MORE_DOCS)?,
                    NO_MORE_DOCS
                );
                Ok(all.0)
            })
            .unwrap();
            assert_eq!(hits, expected);
        }
        assert!(!asserting_active());
    });
}

#[test]
fn a_random_approximation_finds_its_query_s_hits() {
    with_searcher(|s| {
        for t in ["a", "b", "c"] {
            let plain = must(term(t));
            for seed in 0..10 {
                let random = must(TestQuery::random_approximation(
                    term(t),
                    JavaRandom::new(seed),
                ));
                for leaf in 0..2 {
                    assert_eq!(exact(s, &random, leaf), exact(s, &plain, leaf));
                }
                let _guard = asserting(seed);
                let top = s.search(&random, 10).unwrap();
                let expected = s.search(&plain, 10).unwrap();
                assert_eq!(top.score_docs, expected.score_docs);
                assert_eq!(s.count(&random).unwrap(), s.count(&plain).unwrap());
                // A `FilterWeight`: the wrapped query's explanation and matches.
                let doc = expected.score_docs[0].doc;
                assert_eq!(
                    s.explain(&random, doc).unwrap().value,
                    s.explain(&plain, doc).unwrap().value
                );
                assert!(crate::matches::matches(s, &random.must[0], doc)
                    .unwrap()
                    .is_some());
            }
        }
    });
}

#[test]
fn a_random_approximation_s_two_phase_view() {
    with_searcher(|s| {
        // Nested, so the inner approximation is two-phase too.
        let inner = TestQuery::random_approximation(term("c"), JavaRandom::new(3));
        let q = must(TestQuery::random_approximation(inner, JavaRandom::new(4)));
        with_leaf(s, &q, 0, |l| {
            let mut sc = l.scorer(ScoreMode::TopScores)?.unwrap();
            assert!(sc.two_phase());
            assert!((0.0..200.0).contains(&sc.match_cost()));
            assert_eq!(sc.cost(), 3);
            assert!(sc.advance_shallow(0)? >= 0);
            assert!(sc.max_score(NO_MORE_DOCS)? > 0.0);
            sc.set_min_competitive_score(0.0)?;
            let mut confirmed = Vec::new();
            loop {
                let doc = sc.next_doc()?;
                if doc == NO_MORE_DOCS {
                    break;
                }
                assert!(sc.doc_id_run_end() > doc);
                if sc.matches()? {
                    confirmed.push(doc);
                    assert!(sc.advance_shallow(doc)? >= doc);
                }
            }
            assert_eq!(confirmed, [2, 3, 4]);
            Ok(())
        })
        .unwrap();
        // Over a scorer that is not two-phase, on a match: the wrapped
        // scorer's run end.
        let q = must(TestQuery::random_approximation(
            term("b"),
            JavaRandom::new(5),
        ));
        with_leaf(s, &q, 1, |l| {
            let mut sc = l.scorer(ScoreMode::Complete)?.unwrap();
            assert_eq!(sc.exact_advance(1)?, 1);
            assert_eq!(sc.doc_id_run_end(), 2);
            // Shallow-advancing behind the wrapped scorer, which is ahead
            // of a false positive.
            Ok(())
        })
        .unwrap();
    });
}

#[test]
fn a_random_approximation_rejects_misplaced_matches_calls() {
    with_searcher(|s| {
        let q = must(TestQuery::random_approximation(
            term("a"),
            JavaRandom::new(1),
        ));
        with_leaf(s, &q, 0, |l| {
            let mut sc = l.scorer(ScoreMode::Complete)?.unwrap();
            let err = catch_unwind(AssertUnwindSafe(|| sc.matches())).unwrap_err();
            assert_eq!(
                err.downcast_ref::<String>().unwrap(),
                "matches() should not be called on doc ID -1"
            );
            let mut sc = l.scorer(ScoreMode::Complete)?.unwrap();
            let doc = sc.next_doc()?;
            sc.matches()?;
            let err = catch_unwind(AssertUnwindSafe(|| sc.matches())).unwrap_err();
            assert_eq!(
                err.downcast_ref::<String>().unwrap(),
                &format!("matches() has been called twice on doc ID {doc}")
            );
            Ok(())
        })
        .unwrap();
    });
}

#[test]
fn a_random_approximation_shallow_advances_to_the_wrapped_scorer() {
    with_searcher(|s| {
        // Seeds until one approximation lands on a false positive before a
        // match: `advance_shallow` then moves to the match.
        let mut seen = false;
        for seed in 0..50 {
            let q = must(TestQuery::random_approximation(
                term("c"),
                JavaRandom::new(seed),
            ));
            with_leaf(s, &q, 0, |l| {
                let mut sc = l.scorer(ScoreMode::TopScores)?.unwrap();
                let doc = sc.advance(1)?;
                if !sc.matches()? {
                    seen = true;
                    assert!(sc.advance_shallow(doc)? > doc);
                    assert_eq!(sc.doc_id_run_end(), doc + 1);
                }
                Ok(())
            })
            .unwrap();
        }
        assert!(seen);
    });
}

/// `(doc, score)`s of the broken query `kind` over `a` in segment 0,
/// moved by `ops` (`None` is `nextDoc`, `Some(t)` `advance(t)`), each
/// document scored twice.
fn broken(kind: BrokenKind, ops: &[Option<i32>]) -> Vec<(i32, f32, f32)> {
    with_searcher(|s| {
        let q = must(TestQuery::broken(term("a"), kind));
        with_leaf(s, &q, 0, |l| {
            let mut sc = l.scorer(ScoreMode::Complete)?.unwrap();
            assert_eq!(sc.cost(), 3);
            let mut out = Vec::new();
            for op in ops {
                let doc = match op {
                    None => sc.next_doc()?,
                    Some(t) => sc.advance(*t)?,
                };
                if doc == NO_MORE_DOCS {
                    out.push((doc, 0.0, 0.0));
                } else {
                    out.push((doc, sc.score()?, sc.score()?));
                }
            }
            Ok(out)
        })
        .unwrap()
    })
}

#[test]
fn broken_scorers_break_as_named() {
    let ops = [None, Some(1), None];
    let sound = broken(BrokenKind::EarlyTermination, &[Some(0), Some(1), None]);
    let base = sound[0].1;
    assert_eq!(sound[1].0, 2);
    assert_eq!(sound[2].0, NO_MORE_DOCS);
    // `advance` gives up.
    let b = broken(BrokenKind::Advance, &ops);
    assert_eq!((b[0].0, b[1].0), (0, NO_MORE_DOCS));
    // `advance` lands one match too far.
    let b = broken(BrokenKind::SkipTooFar, &[None, Some(1)]);
    assert_eq!((b[0].0, b[1].0), (0, 4));
    let b = broken(BrokenKind::SkipTooFar, &[Some(0), Some(5)]);
    assert_eq!((b[0].0, b[1].0), (2, NO_MORE_DOCS));
    // Every second score of a document is 1 more.
    let b = broken(BrokenKind::UnstableScore, &ops);
    assert_eq!((b[0].1, b[0].2), (base, base + 1.0));
    // A document reached by `advance` scores 100 more.
    let b = broken(BrokenKind::ScoreMismatch, &ops);
    assert_eq!(b[0].1, base);
    assert_eq!(b[1].1, sound[1].1 + 100.0);
    // `nextDoc` stops after its first call.
    let b = broken(BrokenKind::EarlyTermination, &[None, None]);
    assert_eq!((b[0].0, b[1].0), (0, NO_MORE_DOCS));
    // The maximum scores are raised as the scores are.
    with_searcher(|s| {
        let mut maxes = Vec::new();
        for kind in [
            BrokenKind::Advance,
            BrokenKind::UnstableScore,
            BrokenKind::ScoreMismatch,
        ] {
            let q = must(TestQuery::broken(term("a"), kind));
            maxes.push(
                with_leaf(s, &q, 0, |l| {
                    let mut sc = l.scorer(ScoreMode::TopScores)?.unwrap();
                    sc.advance_shallow(0)?;
                    sc.max_score(NO_MORE_DOCS)
                })
                .unwrap(),
            );
        }
        assert_eq!(maxes[1], maxes[0] + 1.0);
        assert_eq!(maxes[2], maxes[0] + 100.0);
    });
}

#[test]
fn test_queries_compare_name_and_print_themselves() {
    let a = TestQuery::random_approximation(term("a"), JavaRandom::new(1));
    assert_eq!(
        a,
        TestQuery::random_approximation(term("a"), JavaRandom::new(2))
    );
    assert_ne!(
        a,
        TestQuery::random_approximation(term("b"), JavaRandom::new(1))
    );
    assert_ne!(a, TestQuery::broken(term("a"), BrokenKind::Advance));
    let Clause::Extended(e) = &a else {
        panic!("{a:?}")
    };
    let crate::extended_query::ExtendedQuery::Test(t) = e.as_ref() else {
        panic!("{e:?}")
    };
    assert_eq!(t.query(), &term("a"));
    assert_eq!(t.name(), "RandomApproximationQuery");
    assert!(format!("{t:?}").contains("RandomApproximation"));
    let b = TestQuery::broken(term("a"), BrokenKind::SkipTooFar);
    assert!(format!("{b:?}").contains("Broken(SkipTooFar)"));
    let Clause::Extended(e) = &b else {
        panic!("{b:?}")
    };
    assert_eq!(e.name(), "BrokenQuery");
    assert_eq!(e.children(), vec![&term("a")]);
    assert!(!crate::segment_cacheable::is_cacheable(&b, None));
}
