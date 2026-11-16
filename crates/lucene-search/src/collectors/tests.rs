#![cfg(test)]
use super::*;
use crate::collector::TopDocsCollector;

/// Records what it is handed, in a fixed mode, with an optional
/// threshold.
#[derive(Default)]
struct Rec {
    hits: Vec<(i32, f32)>,
    mode: Option<ScoreMode>,
    min: Option<f32>,
}

impl ScoringCollector for Rec {
    fn collect(&mut self, doc_id: i32, score: f32) {
        self.hits.push((doc_id, score));
    }
    fn score_mode(&self) -> ScoreMode {
        self.mode.unwrap_or(ScoreMode::Complete)
    }
    fn min_competitive_score(&self) -> Option<f32> {
        self.min
    }
}

fn rec(mode: ScoreMode, min: Option<f32>) -> Rec {
    Rec {
        hits: Vec::new(),
        mode: Some(mode),
        min,
    }
}

#[test]
fn wrap_needs_one_collector() {
    assert!(MultiCollector::wrap(vec![None, None]).is_err());
    assert!(MultiCollector::wrap(vec![]).is_err());
}

#[test]
fn every_collector_sees_every_hit() {
    let (mut a, mut b) = (Rec::default(), Rec::default());
    {
        let mut m = MultiCollector::wrap(vec![
            Some(Box::new(&mut a) as BoxCollector<'_>),
            None,
            Some(Box::new(&mut b)),
        ])
        .unwrap();
        assert_eq!(m.collectors().len(), 2);
        m.collect(3, 1.5);
        m.collect(7, 0.5);
    }
    assert_eq!(a.hits, vec![(3, 1.5), (7, 0.5)]);
    assert_eq!(a.hits, b.hits);
}

#[test]
fn score_mode_combines_as_lucene() {
    // `TestMultiCollector.testSetScorerAfterCollectionTerminated`'s
    // neighbours: equal modes stay, mixed ones fall back to COMPLETE or
    // COMPLETE_NO_SCORES.
    let cases = [
        (
            ScoreMode::TopScores,
            ScoreMode::TopScores,
            ScoreMode::TopScores,
        ),
        (
            ScoreMode::TopScores,
            ScoreMode::Complete,
            ScoreMode::Complete,
        ),
        (
            ScoreMode::CompleteNoScores,
            ScoreMode::TopDocs,
            ScoreMode::CompleteNoScores,
        ),
        (
            ScoreMode::CompleteNoScores,
            ScoreMode::TopDocsWithScores,
            ScoreMode::Complete,
        ),
    ];
    for (x, y, want) in cases {
        let (mut a, mut b) = (rec(x, None), rec(y, None));
        let m = MultiCollector::wrap(vec![
            Some(Box::new(&mut a) as BoxCollector<'_>),
            Some(Box::new(&mut b)),
        ])
        .unwrap();
        assert_eq!(m.score_mode(), want, "{x:?} + {y:?}");
    }
}

#[test]
fn min_competitive_score_is_the_lowest_under_top_scores_only() {
    // `TestMultiCollector.testMinCompetitiveScore`.
    let (mut a, mut b) = (
        rec(ScoreMode::TopScores, Some(2.0)),
        rec(ScoreMode::TopScores, None),
    );
    {
        let m = MultiCollector::wrap(vec![
            Some(Box::new(&mut a) as BoxCollector<'_>),
            Some(Box::new(&mut b)),
        ])
        .unwrap();
        assert_eq!(m.min_competitive_score(), None, "b has not published");
    }
    b.min = Some(1.0);
    {
        let m = MultiCollector::wrap(vec![
            Some(Box::new(&mut a) as BoxCollector<'_>),
            Some(Box::new(&mut b)),
        ])
        .unwrap();
        assert_eq!(m.min_competitive_score(), Some(1.0));
        assert_eq!(m.pruning_threshold(), Some(1.0));
    }
    let mut c = rec(ScoreMode::Complete, Some(5.0));
    let m = MultiCollector::wrap(vec![
        Some(Box::new(&mut a) as BoxCollector<'_>),
        Some(Box::new(&mut c)),
    ])
    .unwrap();
    assert_eq!(m.min_competitive_score(), None, "not TOP_SCORES");
    let mut d = rec(ScoreMode::TopScores, Some(4.0));
    let single = MultiCollector::wrap(vec![Some(Box::new(&mut d) as BoxCollector<'_>)]).unwrap();
    assert_eq!(single.min_competitive_score(), Some(4.0));
    assert_eq!(single.constant_score_hits_needed(), None);
}

#[test]
fn positive_scores_only() {
    // `TestPositiveScoresOnlyCollector`: negative, zero and NaN scores
    // never reach the wrapped collector.
    let mut p = PositiveScoresOnlyCollector::new(Rec::default());
    for (doc, score) in [(0, -1.0), (1, 0.0), (2, -0.0), (3, f32::NAN), (4, 0.25)] {
        p.collect(doc, score);
    }
    assert_eq!(p.inner().hits, vec![(4, 0.25)]);
    assert_eq!(p.score_mode(), ScoreMode::Complete);
    let p = PositiveScoresOnlyCollector::new(rec(ScoreMode::CompleteNoScores, None));
    assert_eq!(p.score_mode(), ScoreMode::Complete);
    let p = PositiveScoresOnlyCollector::new(rec(ScoreMode::TopDocs, Some(1.0)));
    assert_eq!(p.score_mode(), ScoreMode::TopDocsWithScores);
    assert_eq!(p.min_competitive_score(), Some(1.0));
    assert!(p.into_inner().hits.is_empty());
}

#[test]
fn caching_collector_replays_docs_and_scores() {
    // `TestCachingCollector.testBasic`.
    let mut cc = CachingCollector::with_max_docs(Rec::default(), true, 1000);
    for doc in 0..1000 {
        cc.collect(doc, doc as f32 / 2.0);
    }
    assert!(cc.is_cached());
    assert_eq!(cc.score_mode(), ScoreMode::Complete);
    let mut out = Rec::default();
    cc.replay(&mut out).unwrap();
    assert_eq!(out.hits.len(), 1000);
    assert_eq!(out.hits[10], (10, 5.0));
    assert_eq!(
        cc.inner().hits.len(),
        1000,
        "the wrapped collector saw them"
    );
}

#[test]
fn caching_collector_without_scores_replays_docs() {
    let mut cc = CachingCollector::create(false, 1.0);
    assert_eq!(cc.score_mode(), ScoreMode::Complete);
    cc.collect(4, 9.0);
    let mut out = Rec::default();
    cc.replay(&mut out).unwrap();
    assert_eq!(out.hits, vec![(4, 0.0)]);
    let cc = CachingCollector::with_max_docs(rec(ScoreMode::TopDocs, Some(3.0)), false, 4);
    assert_eq!(cc.score_mode(), ScoreMode::TopDocs);
    assert_eq!(cc.min_competitive_score(), Some(3.0));
    assert!(cc.into_inner().hits.is_empty());
}

#[test]
fn caching_collector_overflow_refuses_replay() {
    // `TestCachingCollector.testIllegalStateOnReplay`: one document past
    // the budget clears the cache, but the wrapped collector still
    // collects everything.
    let mut cc = CachingCollector::with_max_docs(Rec::default(), true, 50);
    for doc in 0..51 {
        cc.collect(doc, 1.0);
    }
    assert!(!cc.is_cached());
    assert!(matches!(
        cc.replay(&mut Rec::default()),
        Err(Error::IllegalState(_))
    ));
    assert_eq!(cc.inner().hits.len(), 51);
    // A RAM budget: 4 bytes a document without scores, 8 with.
    let tiny = CachingCollector::create(true, 16.0 / (1024.0 * 1024.0));
    assert_eq!(tiny.max_docs_to_cache, 2);
    let tiny = CachingCollector::create(false, 16.0 / (1024.0 * 1024.0));
    assert_eq!(tiny.max_docs_to_cache, 4);
    let none = CachingCollector::create(false, -1.0);
    assert_eq!(none.max_docs_to_cache, 0);
    let huge = CachingCollector::create(false, 1e12);
    assert_eq!(huge.max_docs_to_cache, i32::MAX as usize);
}

struct TopN(usize);

impl CollectorManager for TopN {
    type Collector = TopDocsCollector;
    type Output = Vec<crate::collector::ScoreDoc>;
    fn new_collector(&self) -> Result<TopDocsCollector> {
        Ok(TopDocsCollector::new(self.0))
    }
    fn reduce(&self, collectors: Vec<TopDocsCollector>) -> Result<Self::Output> {
        let mut all: Vec<_> = collectors
            .iter()
            .flat_map(|c| c.top_docs().to_vec())
            .collect();
        all.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));
        all.truncate(self.0);
        Ok(all)
    }
}

struct Counting;

impl CollectorManager for Counting {
    type Collector = Rec;
    type Output = usize;
    fn new_collector(&self) -> Result<Rec> {
        Ok(rec(ScoreMode::CompleteNoScores, None))
    }
    fn reduce(&self, collectors: Vec<Rec>) -> Result<usize> {
        Ok(collectors.iter().map(|c| c.hits.len()).sum())
    }
}

#[test]
fn multi_collector_manager_reduces_each_manager() {
    // `TestMultiCollectorManager.testCollection`.
    assert!(MultiCollectorManager::new(vec![]).is_err());
    let (top, count) = (TopN(2), Counting);
    let m = MultiCollectorManager::new(vec![&top, &count]).unwrap();
    let mut slices = vec![m.new_collector().unwrap(), m.new_collector().unwrap()];
    assert_eq!(slices[0].score_mode(), ScoreMode::Complete);
    assert_eq!(slices[0].min_competitive_score(), None);
    slices[0].collect(1, 1.0);
    slices[0].collect(2, 3.0);
    slices[1].collect(10, 2.0);
    let out = m.reduce(slices).unwrap();
    let hits = out[0]
        .downcast_ref::<Vec<crate::collector::ScoreDoc>>()
        .unwrap();
    assert_eq!(
        hits.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        vec![2, 10]
    );
    assert_eq!(*out[1].downcast_ref::<usize>().unwrap(), 3);
    // A collector from another manager is refused at reduce time.
    let wrong: Vec<Box<dyn AnyCollector>> = vec![Box::new(Rec::default())];
    assert!(top.reduce_dyn(wrong).is_err());
    let single = MultiCollectorManager::new(vec![&top]).unwrap();
    let c = single.new_collector().unwrap();
    assert_eq!(c.min_competitive_score(), None);
}
