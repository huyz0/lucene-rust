//! Unit tests, after `TestDrillSideways`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::directory_reader::DirectoryReader;
use crate::facets::taxonomy::directory::DirectoryTaxonomyReader;
use crate::facets::test_index::{basic_config, basic_index, build, FacetIndex};
use crate::query::TermQuery;

fn ia_message<T>(r: Result<T>) -> String {
    match r {
        Err(Error::IllegalArgument(m)) => m,
        Err(other) => panic!("expected IllegalArgument, got {other:?}"),
        Ok(_) => panic!("expected IllegalArgument, got Ok"),
    }
}

/// Runs `f` over `index` with the searcher's slices set to `slices`.
fn with_slices<R>(
    index: &FacetIndex,
    slices: Option<Vec<Vec<usize>>>,
    f: impl FnOnce(&IndexSearcher<'_, '_>, &DirectoryTaxonomyReader<'_>) -> R,
) -> R {
    let reader = DirectoryReader::open(&index.index).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["id".to_string()]).unwrap();
    let norms: Vec<_> = owned.iter().map(Some).collect();
    let mut searcher = IndexSearcher::new(&segments, &norms).unwrap();
    if let Some(s) = slices {
        searcher.set_slices(s).unwrap();
    }
    let taxo = DirectoryTaxonomyReader::open(&index.taxo).unwrap();
    f(&searcher, &taxo)
}

fn top(f: &dyn Facets, dim: &str) -> String {
    match f.get_top_children(10, dim, &[]).unwrap() {
        Some(r) => r.to_string(),
        None => "null".into(),
    }
}

fn hits(t: &TopDocs) -> Vec<(i32, f32)> {
    t.score_docs.iter().map(|d| (d.doc, d.score)).collect()
}

fn digest(fc: &FacetsCollector<'_>) -> (i32, usize) {
    let hits = fc.matching_docs().iter().map(|m| m.total_hits).sum();
    (hits, fc.matching_docs().len())
}

/// Every way of running a drill-sideways search: sequentially, with an
/// executor, with `scoreSubDocsAtOnce`.
fn modes<'r, 's, 'a, 't>(
    searcher: &'r IndexSearcher<'s, 'a>,
    config: &'t FacetsConfig,
    taxo: &'t dyn TaxonomyReader,
) -> Vec<DrillSideways<'r, 's, 'a, 't>> {
    vec![
        DrillSideways::new(searcher, config, taxo),
        DrillSideways::new(searcher, config, taxo).with_executor(),
        DrillSideways::new(searcher, config, taxo).with_score_sub_docs_at_once(true),
    ]
}

#[test]
fn one_dim_counts_its_siblings() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, _, taxo| {
        for ds in modes(searcher, &config, taxo) {
            let mut q = DrillDownQuery::new(&config);
            q.add("Author", &["Lisa"]).unwrap();
            let r = ds.search(&q, 10).unwrap();
            assert_eq!(r.hits.total_hits.value, 2);
            assert_eq!(r.hits.score_docs.len(), 2);
            // The drilled dim counts as if it were not drilled.
            assert_eq!(
                top(&r.facets, "Author"),
                "dim=Author path=[] value=5 childCount=4\n  Lisa (2)\n  Bob (1)\n  Susan (1)\n  \
                 Frank (1)\n"
            );
            // The others count the hits.
            assert_eq!(
                top(&r.facets, "Publish Date"),
                "dim=Publish Date path=[] value=2 childCount=2\n  2010 (1)\n  2012 (1)\n"
            );
            assert_eq!(
                r.drill_sideways_dims.as_deref(),
                Some(&["Author".to_string()][..])
            );
            assert_eq!(digest(&r.drill_down_facets_collector).0, 2);
            let sw = r.drill_sideways_facets_collector.as_ref().unwrap();
            assert_eq!(digest(&sw[0]).0, 5);
            // The `Facets` calls go to the sideways counts for the drilled dim.
            assert_eq!(
                r.facets.get_specific_value("Author", &["Bob"]).unwrap(),
                FacetValue::Int(1)
            );
            let all = r.facets.get_all_children("Author", &[]).unwrap().unwrap();
            assert_eq!(all.child_count, 4);
            let dims = r.facets.get_all_dims(10).unwrap();
            assert_eq!(dims.len(), 2);
        }
    });
}

#[test]
fn two_dims_count_each_without_itself() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, _, taxo| {
        for ds in modes(searcher, &config, taxo) {
            let mut q = DrillDownQuery::new(&config);
            q.add("Author", &["Lisa"]).unwrap();
            q.add("Publish Date", &["2010"]).unwrap();
            let r = ds.search(&q, 10).unwrap();
            assert_eq!(
                hits(&r.hits).iter().map(|h| h.0).collect::<Vec<_>>(),
                vec![1]
            );
            // Author, without the Author drill-down: the 2010 documents.
            assert_eq!(
                top(&r.facets, "Author"),
                "dim=Author path=[] value=2 childCount=2\n  Bob (1)\n  Lisa (1)\n"
            );
            // Publish Date, without its own: Lisa's documents.
            assert_eq!(
                top(&r.facets, "Publish Date"),
                "dim=Publish Date path=[] value=2 childCount=2\n  2010 (1)\n  2012 (1)\n"
            );
        }
    });
}

#[test]
fn a_pure_browse_counts_every_document() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, _, taxo| {
        for ds in modes(searcher, &config, taxo) {
            let q = DrillDownQuery::new(&config);
            let r = ds.search(&q, 3).unwrap();
            assert_eq!(r.hits.total_hits.value, 5);
            assert_eq!(r.hits.score_docs.len(), 3);
            assert!(r.drill_sideways_dims.is_none());
            assert!(r.drill_sideways_facets_collector.is_none());
            assert_eq!(
                top(&r.facets, "Author"),
                "dim=Author path=[] value=5 childCount=4\n  Lisa (2)\n  Bob (1)\n  Susan (1)\n  \
                 Frank (1)\n"
            );
            assert_eq!(
                r.facets.get_specific_value("Author", &["Lisa"]).unwrap(),
                FacetValue::Int(2)
            );
            assert!(r.facets.get_all_children("Nope", &[]).unwrap().is_none());
            assert_eq!(r.facets.get_all_dims(1).unwrap().len(), 2);
        }
    });
}

#[test]
fn a_base_query_scores_the_hits() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, _, taxo| {
        let base = Clause::Term(TermQuery::new("id", "d2"));
        let mut q = DrillDownQuery::with_base(&config, base);
        q.add("Author", &["Lisa"]).unwrap();
        let seq = DrillSideways::new(searcher, &config, taxo)
            .search(&q, 10)
            .unwrap();
        let conc = DrillSideways::new(searcher, &config, taxo)
            .with_executor()
            .search(&q, 10)
            .unwrap();
        assert_eq!(hits(&seq.hits), hits(&conc.hits));
        assert_eq!(seq.hits.score_docs[0].doc, 2);
        assert!(seq.hits.score_docs[0].score > 0.0);
        // Sideways: the base's one document.
        assert_eq!(
            top(&seq.facets, "Author"),
            "dim=Author path=[] value=1 childCount=1\n  Lisa (1)\n"
        );
    });
}

#[test]
fn paging_and_filters() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, _, taxo| {
        let ds = DrillSideways::new(searcher, &config, taxo);
        let mut q = DrillDownQuery::new(&config);
        q.add("Author", &["Lisa"]).unwrap();
        let first = ds.search(&q, 1).unwrap();
        assert_eq!(hits(&first.hits), vec![(1, 1.0)]);
        let after = ScoreDoc {
            doc_id: 1,
            score: 1.0,
        };
        let next = ds.search_after(Some(after), &q, 1).unwrap();
        assert_eq!(hits(&next.hits), vec![(2, 1.0)]);
        assert_eq!(next.hits.total_hits.value, 2);

        let filter = Clause::Term(TermQuery::new("id", "d2"));
        let f = ds
            .search_filtered(&q, Some(filter.clone()), None, 10)
            .unwrap();
        assert_eq!(f.hits.total_hits.value, 1);
        assert_eq!(f.hits.score_docs[0].doc, 2);
        // The filter narrows the sideways counts too.
        assert_eq!(
            top(&f.facets, "Author"),
            "dim=Author path=[] value=1 childCount=1\n  Lisa (1)\n"
        );
        let none = ds.search_filtered(&q, None, None, 10).unwrap();
        assert_eq!(none.hits.total_hits.value, 2);
        // TopN past maxDoc is clamped.
        let big = ds.search(&q, i32::MAX).unwrap();
        assert_eq!(big.hits.score_docs.len(), 2);
    });
}

#[test]
fn slices_give_the_same_results() {
    let index = basic_index();
    let config = basic_config();
    let run = |slices: Option<Vec<Vec<usize>>>, executor: bool| {
        with_slices(&index, slices, |searcher, taxo| {
            let mut ds = DrillSideways::new(searcher, &config, taxo);
            if executor {
                ds = ds.with_executor();
            }
            let mut q = DrillDownQuery::new(&config);
            q.add("Author", &["Lisa"]).unwrap();
            q.add("Publish Date", &["2012"]).unwrap();
            let r = ds.search(&q, 10).unwrap();
            (
                hits(&r.hits),
                r.hits.total_hits.value,
                top(&r.facets, "Author"),
                top(&r.facets, "Publish Date"),
            )
        })
    };
    let want = run(None, false);
    assert_eq!(want.0, vec![(2, 1.0)]);
    assert_eq!(run(Some(vec![vec![2], vec![1, 0]]), false), want);
    // With an executor a pure browse's hits are the filters', scoring 0.
    let want_executor = run(None, true);
    assert_eq!(want_executor.0, vec![(2, 0.0)]);
    assert_eq!(want_executor.2, want.2);
    assert_eq!(run(Some(vec![vec![2], vec![1, 0]]), true), want_executor);
    // No slice: no segment is searched.
    let empty = run(Some(Vec::new()), false);
    assert_eq!(empty.1, 0);
}

#[test]
fn an_empty_index() {
    let config = basic_config();
    let index = build(&config, Vec::new(), &[], &[]);
    index.with(|searcher, _, taxo| {
        let mut q = DrillDownQuery::new(&config);
        q.add("Author", &["Lisa"]).unwrap();
        let r = DrillSideways::new(searcher, &config, taxo)
            .search(&q, 10)
            .unwrap();
        assert_eq!(r.hits.total_hits.value, 0);
        assert_eq!(top(&r.facets, "Author"), "null");
    });
}

#[test]
fn refusals() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, _, taxo| {
        let ds = DrillSideways::new(searcher, &config, taxo);
        let mut q = DrillDownQuery::new(&config);
        q.add("Author", &["Lisa"]).unwrap();
        let numhits = "numHits must be > 0; please use TotalHitCountCollectorManager if you just \
                       need the total hit count";
        assert_eq!(ia_message(ds.search(&q, 0)), numhits);
        assert_eq!(ia_message(ds.search(&q, -1)), numhits);
        // The sorted search: `TopFieldCollectorManager`'s wording.
        let sort = [SortField::numeric(
            "num",
            crate::top_field::SortType::Long,
            false,
        )];
        // An empty sort stands in for Java's `null`, refused after `numHits`.
        assert_eq!(
            ia_message(ds.search_sorted(&[], &q, None, None, 5, &[], false)),
            "sort must not be null"
        );
        for (n, s) in [(0, &sort[..]), (-1, &sort[..]), (0, &[][..])] {
            assert_eq!(
                ia_message(ds.search_sorted(&[], &q, None, None, n, s, false)),
                "numHits must be > 0; please use TotalHitCountCollector if you just need \
                 the total hit count"
            );
        }

        let fcm = FacetsCollectorManager::new(false);
        assert_eq!(
            ia_message(ds.search_managers::<_, FacetsCollectorManager>(&q, &fcm, None)),
            "The query requires not null drillSidewaysCollectorManagers"
        );
        assert_eq!(
            ia_message(ds.search_managers::<_, FacetsCollectorManager>(&q, &fcm, Some(&[][..]))),
            "drillSidewaysCollectorManagers size must be equal to number of dimensions in the \
             query."
        );
        let r = ds.search_managers(&q, &fcm, Some(&[fcm][..])).unwrap();
        assert_eq!(digest(&r.drill_down_result).0, 2);
        assert_eq!(digest(&r.drill_sideways_results.unwrap()[0]).0, 5);
        let browse = DrillDownQuery::new(&config);
        for ds in modes(searcher, &config, taxo) {
            let r = ds
                .search_managers::<_, FacetsCollectorManager>(&browse, &fcm, None)
                .unwrap();
            assert_eq!(digest(&r.drill_down_result).0, 5);
            assert!(r.drill_sideways_results.is_none_or(|v| v.is_empty()));
        }
    });
}

#[test]
fn any_hit_collector_manager() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, _, taxo| {
        let ds = DrillSideways::new(searcher, &config, taxo);
        let mut q = DrillDownQuery::new(&config);
        q.add("Publish Date", &["2012"]).unwrap();
        // Both collectors want no scores: so does the pair.
        let fcm = FacetsCollectorManager::new(false);
        let pair = PairManager {
            first: &fcm,
            second: &fcm,
        };
        let c = GroupingCollectorManager::new_collector(&pair).unwrap();
        assert_eq!(c.score_mode(), ScoreMode::CompleteNoScores);
        let r = ds.search_with(&q, &fcm).unwrap();
        assert_eq!(digest(&r.hits).0, 2);
        // A scoring collector beside it: `COMPLETE`.
        let top = TopScoreDocCollectorManager::new(3, None).unwrap();
        let pair = PairManager {
            first: &fcm,
            second: &top,
        };
        let c = GroupingCollectorManager::new_collector(&pair).unwrap();
        assert_eq!(c.score_mode(), ScoreMode::Complete);
    });
}
