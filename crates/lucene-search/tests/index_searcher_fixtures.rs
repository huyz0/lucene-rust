#![allow(clippy::arithmetic_side_effects)]
//! **`IndexSearcher` as an object, and the collector combinators, against
//! real Lucene.**
//!
//! Reuses `fixtures/src/GenMinScore.java`'s index (three segments, two with
//! deletions) and its record of Lucene's top 10 per query with the total
//! counted exactly (`threshold=max`, `min=0`: every match passes). Each
//! query then runs through [`IndexSearcher`] six ways -- `search(query, n)`,
//! `search(query, collector)` with a `TopScoreDocCollector`, a
//! `MultiCollector` of two of them, a `PositiveScoresOnlyCollector` (every
//! score here is positive), a `CachingCollector` replayed into a fresh one,
//! and `search(query, collectorManager)` over two concurrent slices through
//! a `MultiCollectorManager` -- and every way must return Lucene's hits,
//! score bits and total. `count(query)` must equal Lucene's total.

mod m7support;

use std::collections::HashMap;

use lucene_search::collector::{ScoreDoc, TopDocsCollector};
use lucene_search::collectors::{
    BoxCollector, CachingCollector, CollectorManager, MultiCollector, MultiCollectorManager,
    PositiveScoresOnlyCollector,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::Result;
use lucene_store::FsDirectory;
use m7support::{fixture, scored_hits, Grammar, Manifest};

struct TopManager;

impl CollectorManager for TopManager {
    type Collector = TopDocsCollector;
    type Output = (Vec<ScoreDoc>, u64);
    fn new_collector(&self) -> Result<TopDocsCollector> {
        Ok(TopDocsCollector::with_total_hits_threshold(10, u64::MAX))
    }
    fn reduce(&self, collectors: Vec<TopDocsCollector>) -> Result<Self::Output> {
        let total = collectors.iter().map(|c| c.total_hits().value).sum();
        let mut all: Vec<ScoreDoc> = collectors
            .iter()
            .flat_map(|c| c.top_docs().to_vec())
            .collect();
        all.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap()
                .then(a.doc_id.cmp(&b.doc_id))
        });
        all.truncate(10);
        Ok((all, total))
    }
}

fn bits(h: &[ScoreDoc]) -> Vec<(i32, u32)> {
    h.iter().map(|h| (h.doc_id, h.score.to_bits())).collect()
}

#[test]
fn index_searcher_and_collectors_match_real_lucene() {
    let dir = fixture("min_score_index");
    let m = Manifest::load(&format!("{dir}/manifest.properties"));
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).expect("open reader");
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let mut searcher = IndexSearcher::new(&segments, &norms).unwrap();
    assert!(IndexSearcher::new(&segments, &norms[..1]).is_err());
    assert_eq!(searcher.max_doc(), 9000);
    assert_eq!(searcher.segment_of(3000), Some(1));
    assert_eq!(searcher.segment_of(9000), None);
    assert!(searcher.set_slices(vec![vec![7]]).is_err());
    searcher.set_slices(vec![vec![0], vec![2, 1]]).unwrap();
    assert_eq!(searcher.slices().len(), 2);
    let grammar = Grammar {
        text: "body",
        range: "r",
    };

    let runs: usize = m.get("run_count").parse().unwrap();
    let mut checked = 0;
    for r in 0..runs {
        let k = format!("run.{r}");
        if m.get(&format!("{k}.min")) != "0" || m.get(&format!("{k}.threshold")) != "max" {
            continue;
        }
        checked += 1;
        let text = m.get(&format!("{k}.query"));
        let q = grammar.query(text);
        let want = scored_hits(m.get(&format!("{k}.hits")));
        let total: u64 = m.get(&format!("{k}.total")).parse().unwrap();

        // search(query, n): exact up to 1000.
        let td = searcher.search(&q, 10).unwrap();
        let got: Vec<(i32, u32)> = td
            .score_docs
            .iter()
            .map(|h| (h.doc, h.score.to_bits()))
            .collect();
        assert_eq!(got, want, "{text}: search(query, 10)");
        assert_eq!(td.total_hits.value.min(1000), total.min(1000), "{text}");

        // search(query, collector).
        let mut c = TopDocsCollector::with_total_hits_threshold(10, u64::MAX);
        searcher.search_collector(&q, &mut c).unwrap();
        assert_eq!(bits(c.top_docs()), want, "{text}: collector");
        assert_eq!(c.total_hits().value, total, "{text}: collector total");

        // MultiCollector of two.
        let mut a = TopDocsCollector::with_total_hits_threshold(10, u64::MAX);
        let mut b = TopDocsCollector::with_total_hits_threshold(3, u64::MAX);
        {
            let mut multi = MultiCollector::wrap(vec![
                Some(Box::new(&mut a) as BoxCollector<'_>),
                Some(Box::new(&mut b)),
            ])
            .unwrap();
            searcher.search_collector(&q, &mut multi).unwrap();
        }
        assert_eq!(bits(a.top_docs()), want, "{text}: multi a");
        assert_eq!(
            bits(b.top_docs()),
            want.iter().take(3).copied().collect::<Vec<_>>(),
            "{text}: multi b"
        );

        // PositiveScoresOnlyCollector.
        let mut p = PositiveScoresOnlyCollector::new(TopDocsCollector::with_total_hits_threshold(
            10,
            u64::MAX,
        ));
        searcher.search_collector(&q, &mut p).unwrap();
        assert_eq!(bits(p.inner().top_docs()), want, "{text}: positive");

        // CachingCollector, replayed.
        let mut cc = CachingCollector::create(true, 64.0);
        searcher.search_collector(&q, &mut cc).unwrap();
        assert!(cc.is_cached());
        let mut replayed = TopDocsCollector::with_total_hits_threshold(10, u64::MAX);
        cc.replay(&mut replayed).unwrap();
        assert_eq!(bits(replayed.top_docs()), want, "{text}: replay");
        assert_eq!(replayed.total_hits().value, total);

        // search(query, collectorManager): two slices, concurrently.
        let (top, top2) = (TopManager, TopManager);
        let (hits, t) = searcher.search_manager(&q, &top).unwrap();
        assert_eq!(bits(&hits), want, "{text}: manager");
        assert_eq!(t, total);
        let mm = MultiCollectorManager::new(vec![&top, &top2]).unwrap();
        let out = searcher.search_manager(&q, &mm).unwrap();
        for o in &out {
            let (hits, t) = o.downcast_ref::<(Vec<ScoreDoc>, u64)>().unwrap();
            assert_eq!(bits(hits), want, "{text}: multi manager");
            assert_eq!(*t, total);
        }

        assert_eq!(searcher.count(&q).unwrap(), total, "{text}: count");
    }
    // Explanations score with the segment's own statistics, not the
    // reader's: refused over several segments.
    assert!(searcher.explain(&grammar.query("(t w0)"), 9000).is_err());
    assert!(checked >= 9, "one run per query: {checked}");
}
