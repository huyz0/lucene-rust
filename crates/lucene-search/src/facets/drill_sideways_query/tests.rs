#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::test_index::{basic_config, basic_index};
use crate::facets::{FacetsCollector, FacetsCollectorManager};
use crate::query::{MatchAllDocsQuery, TermQuery};

fn term(field: &str, value: &str) -> Clause {
    Clause::Term(TermQuery::new(field, value))
}

fn drill(dim: &str, path: &str) -> Clause {
    Clause::Boolean(Box::new(BooleanQuery {
        should: vec![term("$facets", &format!("{dim}\u{1f}{path}"))],
        ..Default::default()
    }))
}

fn total(fc: &FacetsCollector<'_>) -> i32 {
    fc.matching_docs().iter().map(|m| m.total_hits).sum()
}

fn totals(per_segment: &[Vec<FacetsCollector<'_>>]) -> Vec<i32> {
    let dims = per_segment.first().map_or(0, Vec::len);
    (0..dims)
        .map(|d| per_segment.iter().map(|s| total(&s[d])).sum())
        .collect()
}

#[test]
fn the_score_mode_picks_the_scorers_mode() {
    assert_eq!(mode_of(ScoreMode::CompleteNoScores), Mode::NoScores);
    assert_eq!(mode_of(ScoreMode::Complete), Mode::Complete);
    assert_eq!(mode_of(ScoreMode::TopScores), Mode::TopScores);
}

#[test]
fn hits_and_near_misses_per_segment() {
    let index = basic_index();
    index.with(|searcher, _, _| {
        let dsq = DrillSidewaysQuery::new(
            Clause::MatchAllDocs(MatchAllDocsQuery::new(0)),
            vec![drill("Author", "Lisa"), drill("Publish Date", "2010")],
            false,
        );
        let fcm = FacetsCollectorManager::new(false);
        let sideways = [fcm, fcm];
        let (main, per_segment) = dsq.search(searcher, &fcm, &sideways).unwrap();
        assert_eq!(total(&main), 1);
        // One list of collectors per segment, one collector per dim.
        assert_eq!(per_segment.len(), 3);
        assert!(per_segment.iter().all(|s| s.len() == 2));
        // Author without itself: the 2010 documents; Publish Date without
        // itself: Lisa's.
        assert_eq!(totals(&per_segment), vec![2, 2]);
    });
}

#[test]
fn a_segment_two_dims_miss_is_skipped() {
    let index = basic_index();
    index.with(|searcher, _, _| {
        // Neither dim has a term anywhere: no scorer for either.
        let dsq = DrillSidewaysQuery::new(
            Clause::MatchAllDocs(MatchAllDocsQuery::new(0)),
            vec![drill("Author", "Nobody"), drill("Publish Date", "3000")],
            false,
        );
        let fcm = FacetsCollectorManager::new(false);
        let (main, per_segment) = dsq.search(searcher, &fcm, &[fcm, fcm]).unwrap();
        assert_eq!(total(&main), 0);
        assert_eq!(totals(&per_segment), vec![0, 0]);
        // One missing dim: its near misses are every base document.
        let dsq = DrillSidewaysQuery::new(
            Clause::MatchAllDocs(MatchAllDocsQuery::new(0)),
            vec![drill("Author", "Nobody"), drill("Publish Date", "2010")],
            false,
        );
        let (main, per_segment) = dsq.search(searcher, &fcm, &[fcm, fcm]).unwrap();
        assert_eq!(total(&main), 0);
        assert_eq!(totals(&per_segment), vec![2, 0]);
        // A base with no scorer: nothing.
        let dsq = DrillSidewaysQuery::new(term("id", "nope"), vec![drill("Author", "Lisa")], true);
        let (main, per_segment) = dsq.search(searcher, &fcm, &[fcm]).unwrap();
        assert_eq!(total(&main), 0);
        assert_eq!(totals(&per_segment), vec![0]);
    });
}

#[test]
fn refuses_a_manager_count_other_than_the_dims() {
    let index = basic_index();
    let _ = basic_config();
    index.with(|searcher, _, _| {
        let dsq = DrillSidewaysQuery::new(
            Clause::MatchAllDocs(MatchAllDocsQuery::new(0)),
            vec![drill("Author", "Lisa")],
            false,
        );
        let fcm = FacetsCollectorManager::new(false);
        let err = dsq
            .search(searcher, &fcm, &[fcm, fcm])
            .unwrap_err();
        assert!(
            matches!(&err, Error::IllegalArgument(m) if m == "2 sideways collector managers for 1 dims"),
            "{err:?}"
        );
        // No dim: refused rather than indexing an empty dims array.
        let none = DrillSidewaysQuery::new(Clause::MatchAllDocs(MatchAllDocsQuery::new(0)), vec![], false);
        let empty: [FacetsCollectorManager; 0] = [];
        let err = none.search(searcher, &fcm, &empty).unwrap_err();
        assert!(
            matches!(&err, Error::IllegalArgument(m) if m.contains("at least one dim")),
            "{err:?}"
        );
    });
}
