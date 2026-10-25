//! Actual Java ScorerIndexSearcher and DisablingBulkScorerQuery over identical
//! three-leaf corpora, including deletions, two-phase phrases and similarities.
mod scorer_index_searcher_cases;
use lucene_search::collector::{ScoreDoc, ScoreMode, ScoringCollector};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::similarities::BooleanSimilarity;
use lucene_search::test_support::{self, scorer_only};
use lucene_test_framework::search::disabling_bulk_scorer_query;
use lucene_test_framework::search::scorer_index_searcher::ScorerIndexSearcher;
use scorer_index_searcher_cases::*;

#[derive(Debug)]
struct AlwaysTimeout;
#[derive(Debug)]
struct NeverTimeout;
impl lucene_search::reader::exitable::QueryTimeout for NeverTimeout {
    fn should_exit(&self) -> bool {
        false
    }
}
impl lucene_search::reader::exitable::QueryTimeout for AlwaysTimeout {
    fn should_exit(&self) -> bool {
        true
    }
}

struct Threshold(Hits, f32);
impl ScoringCollector for Threshold {
    fn collect(&mut self, doc: i32, score: f32) {
        self.0.collect(doc, score)
    }
    fn score_mode(&self) -> ScoreMode {
        self.0.mode
    }
    fn min_competitive_score(&self) -> Option<f32> {
        // The collector reports an exclusive bound; DefaultBulkScorer adds
        // one ULP before the scorer setter. Java's fixture calls that setter
        // directly, so translate its inclusive minimum back to this bound.
        Some(self.1.next_down())
    }
}

const DATA: &str = include_str!("../../../fixtures/testframework/data/scorer_index_searcher.tsv");

#[test]
fn indexed_searches_match_actual_java_wrappers() {
    let index = index(1);
    index.with_searcher(|base| {
        assert_eq!(base.segments().len(), 3);
        assert_eq!(base.max_doc(), 12);
        let norms: Vec<_> = (0..3).map(|i| base.norms(i)).collect();
        let boolean = BooleanSimilarity;
        let live = base.count(&top_level(clauses()[8].clone())).unwrap();
        let mut rows = vec![format!(
            "reader\t{}\t{}\t{live}",
            base.segments().len(),
            base.max_doc()
        )];
        for mode in ["scorer_searcher", "disabling_query"] {
            for sim in ["bm25", "boolean"] {
                let mut forced = ScorerIndexSearcher::new(base.segments(), &norms).unwrap();
                forced.set_slices(vec![vec![0], vec![1], vec![2]]).unwrap();
                let mut plain = IndexSearcher::new(base.segments(), &norms).unwrap();
                plain.set_slices(vec![vec![0], vec![1], vec![2]]).unwrap();
                if sim == "boolean" {
                    forced.set_similarity(&boolean);
                    plain.set_similarity(&boolean);
                }
                for (q, clause) in clauses().into_iter().enumerate() {
                    let query = top_level(if mode == "disabling_query" {
                        disabling_bulk_scorer_query::wrap(clause)
                    } else {
                        clause
                    });
                    // Java's DisablingBulkScorerQuery fails on a null supplier;
                    // Rust's safe optional scorer returns no hits instead.
                    if mode == "disabling_query" && q == 9 {
                        assert_eq!(plain.count(&query).unwrap(), 0);
                        assert!(DATA.lines().any(|row| row
                            == format!(
                                "error\t{mode}\t{sim}\t{q}\tNullPointerException:null-supplier"
                            )));
                        continue;
                    }
                    for (score_mode, label) in [
                        (ScoreMode::Complete, "COMPLETE"),
                        (ScoreMode::CompleteNoScores, "COMPLETE_NO_SCORES"),
                        (ScoreMode::TopDocs, "TOP_DOCS"),
                        (ScoreMode::TopDocsWithScores, "TOP_DOCS_WITH_SCORES"),
                    ] {
                        let mut hits = Hits::new(score_mode);
                        if mode == "scorer_searcher" {
                            forced.search_collector(&query, &mut hits).unwrap()
                        } else {
                            plain.search_collector(&query, &mut hits).unwrap()
                        }
                        rows.push(format!(
                            "hits\t{mode}\t{sim}\t{q}\t{label}\t{}",
                            hits_text(&hits.hits)
                        ));
                    }
                    let top = if mode == "scorer_searcher" {
                        forced.search(&query, 3).unwrap()
                    } else {
                        plain.search(&query, 3).unwrap()
                    };
                    rows.push(format!("top\t{mode}\t{sim}\t{q}\t{}", top_text(&top)));
                    let after = if let Some(hit) = top.score_docs.first() {
                        let after = ScoreDoc {
                            doc_id: hit.doc,
                            score: hit.score,
                        };
                        if mode == "scorer_searcher" {
                            forced.search_after(after, &query, 3).unwrap()
                        } else {
                            plain.search_after(after, &query, 3).unwrap()
                        }
                    } else {
                        top.clone()
                    };
                    rows.push(format!("after\t{mode}\t{sim}\t{q}\t{}", top_text(&after)));
                    let count = if mode == "scorer_searcher" {
                        forced.count(&query).unwrap()
                    } else {
                        plain.count(&query).unwrap()
                    };
                    rows.push(format!("count\t{mode}\t{sim}\t{q}\t{count}"));
                    let hits = if mode == "scorer_searcher" {
                        forced.search_manager(&query, &Manager).unwrap()
                    } else {
                        plain.search_manager(&query, &Manager).unwrap()
                    };
                    rows.push(format!("manager\t{mode}\t{sim}\t{q}\t{}", hits_text(&hits)));
                    if mode == "scorer_searcher" {
                        let mut hits = Hits::new(ScoreMode::Complete);
                        forced
                            .search_slice_collector(&query, &[2, 0], &mut hits)
                            .unwrap();
                        rows.push(format!(
                            "slice\t{mode}\t{sim}\t{q}\t{}",
                            hits_text(&hits.hits)
                        ));
                    }
                }
                // The Java trap verifies three scorer.get calls and zero
                // specialized-bulk/top-level-supplier callbacks. Rust dispatch
                // explicitly constructs Bulk::Scorer with top_level=false.
                let query = top_level(if mode == "disabling_query" {
                    disabling_bulk_scorer_query::wrap(term("a"))
                } else {
                    term("a")
                });
                let mut hits = Hits::new(ScoreMode::Complete);
                if mode == "scorer_searcher" {
                    forced.search_collector(&query, &mut hits).unwrap()
                } else {
                    test_support::with_leaf(&plain, &query, 0, |leaf| {
                        assert!(leaf
                            .bulk_scorer(ScoreMode::Complete)?
                            .unwrap()
                            .uses_default_scorer());
                        Ok(())
                    })
                    .unwrap();
                    plain.search_collector(&query, &mut hits).unwrap()
                }
                assert!(DATA
                    .lines()
                    .any(|row| row
                        == format!("trap\t{mode}\t{sim}\t3\t0\t0\t{}", hits_text(&hits.hits))));
                for (name, clause, count) in threshold_clauses() {
                    let query = top_level(if mode == "disabling_query" {
                        disabling_bulk_scorer_query::wrap(clause)
                    } else {
                        clause
                    });
                    let mut threshold = Threshold(Hits::new(ScoreMode::TopScores), f32::INFINITY);
                    if mode == "scorer_searcher" {
                        forced.search_collector(&query, &mut threshold).unwrap()
                    } else {
                        plain.search_collector(&query, &mut threshold).unwrap()
                    }
                    assert_eq!(threshold.0.hits.len(), count, "{mode}/{sim}/{name}");
                    rows.push(format!(
                        "threshold\t{mode}\t{sim}\t{name}\t{}",
                        hits_text(&threshold.0.hits)
                    ));
                }
                // Constant-score scorers prune by mode alone, whereas real
                // terms above also require top-level supplier notification.
                for (name, clause) in constant_threshold_clauses() {
                    let query = top_level(if mode == "disabling_query" { disabling_bulk_scorer_query::wrap(clause) } else { clause });
                    for (score_mode, label) in [
                        (ScoreMode::Complete, "COMPLETE"),
                        (ScoreMode::CompleteNoScores, "COMPLETE_NO_SCORES"),
                        (ScoreMode::TopScores, "TOP_SCORES"),
                        (ScoreMode::TopDocs, "TOP_DOCS"),
                        (ScoreMode::TopDocsWithScores, "TOP_DOCS_WITH_SCORES"),
                    ] {
                        for (threshold_label, minimum) in [("one", 1.0), ("two", 2.0), ("infinity", f32::INFINITY)] {
                            let mut threshold = Threshold(Hits::new(score_mode), minimum);
                            if mode == "scorer_searcher" { forced.search_collector(&query, &mut threshold).unwrap(); } else { plain.search_collector(&query, &mut threshold).unwrap(); }
                            rows.push(format!("constant_threshold\t{mode}\t{sim}\t{name}\t{label}\t{threshold_label}\t{}", hits_text(&threshold.0.hits)));
                        }
                    }
                }
                // Observe the ordinary scorer API directly: a specialized bulk
                // scorer may separately filter against a collector threshold.
                let ordinary_query = top_level(term("a"));
                let mut ordinary_threshold = Vec::new();
                for segment in 0..base.segments().len() {
                    test_support::with_leaf(&plain, &ordinary_query, segment, |leaf| {
                        let Some(mut scorer) = leaf.scorer(ScoreMode::TopDocsWithScores)? else {
                            return Ok(());
                        };
                        scorer.set_min_competitive_score(f32::INFINITY)?;
                        loop {
                            let doc = scorer.next_doc()?;
                            if doc == i32::MAX {
                                break;
                            }
                            let seg = &base.segments()[segment];
                            if seg.live_docs.is_none_or(|live| live.get(doc as usize)) {
                                ordinary_threshold
                                    .push((seg.doc_base + doc, scorer.score()?.to_bits()));
                                scorer.set_min_competitive_score(f32::INFINITY)?;
                            }
                        }
                        Ok(())
                    })
                    .unwrap();
                }
                assert_eq!(ordinary_threshold.len(), 6);
                rows.push(format!(
                    "ordinary_threshold\t{mode}\t{sim}\tTOP_DOCS_WITH_SCORES\t{}",
                    hits_text(&ordinary_threshold)
                ));
                let timeout = std::sync::Arc::new(AlwaysTimeout);
                let (count, timed) = if mode == "scorer_searcher" {
                    forced.set_timeout(Some(timeout));
                    (forced.count(&query).unwrap(), forced.timed_out())
                } else {
                    plain.set_timeout(Some(timeout));
                    (plain.count(&query).unwrap(), plain.timed_out())
                };
                rows.push(format!("timeout\t{mode}\t{sim}\t{count}\t{timed}"));
            }
        }
        let observed: Vec<_> = DATA
            .lines()
            .filter(|row| !row.starts_with("trap\t") && !row.starts_with("error\t"))
            .collect();
        assert_eq!(rows.len(), observed.len());
        for (index, (actual, expected)) in rows.iter().zip(observed).enumerate() {
            assert_eq!(actual, expected, "Java fixture row {index}");
        }
    });
}

#[test]
fn explicit_mode_preserves_configuration_and_timeout_state() {
    let index = index(1);
    index.with_searcher(|base| {
        let norms: Vec<_> = (0..3).map(|i| base.norms(i)).collect();
        let ordinary = IndexSearcher::new(base.segments(), &norms).unwrap();
        assert!(!scorer_only::enabled(&ordinary));
        let configured = scorer_only::enable(ordinary);
        assert!(scorer_only::enabled(&configured));
        let mut searcher = ScorerIndexSearcher::from_searcher(configured);
        let query = top_level(term("a"));
        searcher.set_timeout(Some(std::sync::Arc::new(NeverTimeout)));
        assert_eq!(searcher.count(&query).unwrap(), 6);
        assert!(!searcher.timed_out());
        searcher.set_timeout(Some(std::sync::Arc::new(AlwaysTimeout)));
        assert!(searcher.search(&query, 3).unwrap().score_docs.is_empty());
        assert!(searcher.timed_out());
        searcher.set_timeout(None);
        assert_eq!(searcher.count(&query).unwrap(), 6);
        assert!(searcher.timed_out());
        assert!(searcher.explain(&query, 0).unwrap().matched);
        assert!(!searcher.explain(&query, 1).unwrap().matched);
    });
}
