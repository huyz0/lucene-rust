//! Lucene's general-purpose collector combinators, over this port's
//! [`ScoringCollector`]: `MultiCollector`, `PositiveScoresOnlyCollector`,
//! `CachingCollector`, and the `CollectorManager` pair a concurrent search
//! drives (`CollectorManager`, `MultiCollectorManager`).
//!
//! # How the collector model maps
//!
//! Java hands a `LeafCollector` a `Scorable` and pulls the score from it; a
//! collector that wants to prune calls `Scorable.setMinCompetitiveScore`.
//! Here the scorer pushes `(doc, score)` into [`ScoringCollector::collect`]
//! and *asks* the collector for its threshold
//! ([`ScoringCollector::min_competitive_score`]). So:
//!
//! * `MultiCollector`'s `MinCompetitiveScoreAwareScorable` -- which forwards
//!   the minimum of its sub-collectors' thresholds, each defaulting to `0` --
//!   becomes [`MultiCollector::min_competitive_score`] returning the minimum of
//!   theirs, and `None` while any of them has none (Java's `0`, which prunes
//!   nothing);
//! * `ScoreCachingWrappingScorer` has nothing to do: a score is computed once
//!   and handed to every sub-collector by value;
//! * leaves are not visible: a collector sees global document ids through
//!   [`crate::collector::LeafCollector`], so [`CachingCollector`] caches one
//!   list rather than one per leaf, and replays it into one collector. The
//!   per-leaf budget of Java's cache (`maxDocsToCache` decremented as each
//!   leaf finishes) is the same total budget.
//!
//! One Java behaviour has no counterpart: `CollectionTerminatedException`.
//! [`ScoringCollector`] has no way to end collection early, so a
//! sub-collector of a [`MultiCollector`] is never dropped mid-search.

use std::any::Any;

use crate::collector::{ScoreMode, ScoringCollector};
use crate::{Error, Result};

/// A mutable borrow of a collector collects into it: what lets a caller keep
/// its collectors and hand [`MultiCollector`] references to them.
impl<C: ScoringCollector + ?Sized> ScoringCollector for &mut C {
    #[inline]
    fn collect(&mut self, doc_id: i32, score: f32) {
        (**self).collect(doc_id, score);
    }
    #[inline]
    fn min_competitive_score(&self) -> Option<f32> {
        (**self).min_competitive_score()
    }
    #[inline]
    fn score_mode(&self) -> ScoreMode {
        (**self).score_mode()
    }
    #[inline]
    fn pruning_threshold(&self) -> Option<f32> {
        (**self).pruning_threshold()
    }
    #[inline]
    fn constant_score_hits_needed(&self) -> Option<u64> {
        (**self).constant_score_hits_needed()
    }
    #[inline]
    fn count_losing_hits(&mut self, n: u64, score: f32) -> bool {
        (**self).count_losing_hits(n, score)
    }
    #[inline]
    fn add_hits(&mut self, n: u64) -> bool {
        (**self).add_hits(n)
    }
    #[inline]
    fn collect_many(&mut self, docs: &[i32], scores: &[f32], doc_base: i32) -> usize {
        (**self).collect_many(docs, scores, doc_base)
    }
}

/// A boxed collector, as [`MultiCollector`] holds them.
pub type BoxCollector<'c> = Box<dyn ScoringCollector + Send + 'c>;

/// `MultiCollector`: every hit goes to each of several collectors.
pub struct MultiCollector<'c> {
    collectors: Vec<BoxCollector<'c>>,
}

impl<'c> MultiCollector<'c> {
    /// `MultiCollector.wrap(collectors)`: the collectors that are `Some`, in
    /// order. Java returns a lone collector unwrapped; this wrapper then
    /// behaves exactly as that collector does.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when every collector is `None` ("At least 1
    /// collector must not be null").
    pub fn wrap(collectors: Vec<Option<BoxCollector<'c>>>) -> Result<Self> {
        let collectors: Vec<BoxCollector<'c>> = collectors.into_iter().flatten().collect();
        if collectors.is_empty() {
            return Err(Error::IllegalArgument(
                "At least 1 collector must not be null".to_string(),
            ));
        }
        Ok(Self { collectors })
    }

    /// `getCollectors()`.
    pub fn collectors(&self) -> &[BoxCollector<'c>] {
        &self.collectors
    }

    /// The wrapped collectors, back.
    pub fn into_collectors(self) -> Vec<BoxCollector<'c>> {
        self.collectors
    }
}

/// `MultiCollector.scoreMode()`: the collectors' common mode, else `COMPLETE`
/// if any needs scores, else `COMPLETE_NO_SCORES`.
fn combined_score_mode<'a>(modes: impl Iterator<Item = ScoreMode> + 'a) -> Option<ScoreMode> {
    let mut mode: Option<ScoreMode> = None;
    for m in modes {
        mode = Some(match mode {
            None => m,
            Some(prev) if prev == m => prev,
            Some(prev) => {
                if prev.needs_scores() || m.needs_scores() {
                    ScoreMode::Complete
                } else {
                    ScoreMode::CompleteNoScores
                }
            }
        });
    }
    mode
}

impl ScoringCollector for MultiCollector<'_> {
    #[inline]
    fn collect(&mut self, doc_id: i32, score: f32) {
        for c in &mut self.collectors {
            c.collect(doc_id, score);
        }
    }

    fn score_mode(&self) -> ScoreMode {
        combined_score_mode(self.collectors.iter().map(|c| c.score_mode()))
            .unwrap_or(ScoreMode::Complete)
    }

    /// Only a `TOP_SCORES` multi-collector passes thresholds on
    /// (`skipNonCompetitiveScores`); otherwise its `FilterScorable` swallows
    /// them. Then the lowest of the collectors' thresholds, since a document
    /// any of them still wants must reach it.
    fn min_competitive_score(&self) -> Option<f32> {
        if self.collectors.len() == 1 {
            return self.collectors[0].min_competitive_score();
        }
        if self.score_mode() != ScoreMode::TopScores {
            return None;
        }
        let mut min = f32::MAX;
        for c in &self.collectors {
            min = min.min(c.min_competitive_score()?);
        }
        Some(min)
    }

    fn constant_score_hits_needed(&self) -> Option<u64> {
        if self.collectors.len() == 1 {
            self.collectors[0].constant_score_hits_needed()
        } else {
            None
        }
    }
}

/// `PositiveScoresOnlyCollector`: only hits scoring above zero reach `inner`.
pub struct PositiveScoresOnlyCollector<C> {
    inner: C,
}

impl<C: ScoringCollector> PositiveScoresOnlyCollector<C> {
    pub fn new(inner: C) -> Self {
        Self { inner }
    }

    /// The wrapped collector.
    pub fn into_inner(self) -> C {
        self.inner
    }

    pub fn inner(&self) -> &C {
        &self.inner
    }
}

impl<C: ScoringCollector> ScoringCollector for PositiveScoresOnlyCollector<C> {
    #[inline]
    fn collect(&mut self, doc_id: i32, score: f32) {
        // `scorer.score() > 0`: NaN and both zeros are dropped.
        if score > 0.0 {
            self.inner.collect(doc_id, score);
        }
    }

    fn score_mode(&self) -> ScoreMode {
        // `FilterCollector.scoreMode()` is the wrapped collector's; the
        // filter reads a score whatever it is, which the scorer here always
        // computes for a scoring mode. A no-scores mode would hand `0` to
        // every document, so ask for scores.
        match self.inner.score_mode() {
            ScoreMode::CompleteNoScores => ScoreMode::Complete,
            ScoreMode::TopDocs => ScoreMode::TopDocsWithScores,
            m => m,
        }
    }

    fn min_competitive_score(&self) -> Option<f32> {
        self.inner.min_competitive_score()
    }
}

/// A collector that ignores everything: `CachingCollector.create(cacheScores,
/// maxRAMMB)`'s wrapped `SimpleCollector`, whose mode is `COMPLETE`.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpCollector;

impl ScoringCollector for NoOpCollector {
    #[inline]
    fn collect(&mut self, _doc_id: i32, _score: f32) {}
}

/// `CachingCollector`: collects into `inner` while caching every document
/// (and, if asked, its score) up to a budget, to be [replayed](Self::replay)
/// into another collector without searching again.
pub struct CachingCollector<C> {
    inner: C,
    cache_scores: bool,
    max_docs_to_cache: usize,
    docs: Vec<i32>,
    scores: Vec<f32>,
    cached: bool,
}

/// `CachingCollector.INITIAL_ARRAY_SIZE`.
const INITIAL_ARRAY_SIZE: usize = 128;

impl CachingCollector<NoOpCollector> {
    /// `CachingCollector.create(cacheScores, maxRAMMB)`: caching only.
    pub fn create(cache_scores: bool, max_ram_mb: f64) -> Self {
        CachingCollector::with_ram(NoOpCollector, cache_scores, max_ram_mb)
    }
}

impl<C: ScoringCollector> CachingCollector<C> {
    /// `CachingCollector.create(other, cacheScores, maxRAMMB)`: 4 bytes a
    /// document, 8 with its score.
    pub fn with_ram(inner: C, cache_scores: bool, max_ram_mb: f64) -> Self {
        let bytes_per_doc = if cache_scores { 8.0 } else { 4.0 };
        // `(int) (maxRAMMB * 1024 * 1024 / bytesPerDoc)`: saturating, and a
        // negative budget caches nothing.
        let max = (max_ram_mb * 1024.0 * 1024.0) / bytes_per_doc;
        let max = if max.is_nan() || max < 0.0 {
            0
        } else {
            max.min(f64::from(i32::MAX)) as usize
        };
        Self::with_max_docs(inner, cache_scores, max)
    }

    /// `CachingCollector.create(other, cacheScores, maxDocsToCache)`.
    pub fn with_max_docs(inner: C, cache_scores: bool, max_docs_to_cache: usize) -> Self {
        let initial = max_docs_to_cache.min(INITIAL_ARRAY_SIZE);
        Self {
            inner,
            cache_scores,
            max_docs_to_cache,
            docs: Vec::with_capacity(initial),
            scores: if cache_scores {
                Vec::with_capacity(initial)
            } else {
                Vec::new()
            },
            cached: true,
        }
    }

    /// `isCached()`: whether every collected document fit, so
    /// [`Self::replay`] may run.
    pub fn is_cached(&self) -> bool {
        self.cached
    }

    /// The wrapped collector.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    pub fn into_inner(self) -> C {
        self.inner
    }

    /// `replay(other)`: the cached documents, in collection order, into
    /// `other` -- with their scores when they were cached, else `0`
    /// (Java sets no scorer then, so a collector reading one fails).
    ///
    /// # Errors
    /// [`Error::IllegalState`] when the budget overflowed ("cannot replay:
    /// cache was cleared because too much RAM was required").
    pub fn replay<O: ScoringCollector + ?Sized>(&self, other: &mut O) -> Result<()> {
        if !self.cached {
            return Err(Error::IllegalState(
                "cannot replay: cache was cleared because too much RAM was required".to_string(),
            ));
        }
        if self.cache_scores {
            for (&doc, &score) in self.docs.iter().zip(&self.scores) {
                other.collect(doc, score);
            }
        } else {
            for &doc in &self.docs {
                other.collect(doc, 0.0);
            }
        }
        Ok(())
    }

    fn invalidate(&mut self) {
        self.cached = false;
        self.docs = Vec::new();
        self.scores = Vec::new();
    }
}

impl<C: ScoringCollector> ScoringCollector for CachingCollector<C> {
    fn collect(&mut self, doc_id: i32, score: f32) {
        if self.cached {
            if self.docs.len() >= self.max_docs_to_cache {
                self.invalidate();
            } else {
                self.docs.push(doc_id);
                if self.cache_scores {
                    self.scores.push(score);
                }
            }
        }
        self.inner.collect(doc_id, score);
    }

    /// `ScoreCachingCollector.scoreMode()` is `COMPLETE`, so the scores are
    /// there to cache; otherwise the wrapped collector's.
    fn score_mode(&self) -> ScoreMode {
        if self.cache_scores {
            ScoreMode::Complete
        } else {
            self.inner.score_mode()
        }
    }

    fn min_competitive_score(&self) -> Option<f32> {
        self.inner.min_competitive_score()
    }
}

/// `CollectorManager<C, T>`: makes one collector per slice of a concurrent
/// search and reduces them to a result.
pub trait CollectorManager: Sync {
    type Collector: ScoringCollector + Send;
    type Output;
    /// `newCollector()`.
    fn new_collector(&self) -> Result<Self::Collector>;
    /// `reduce(collectors)`: the collectors in slice order.
    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<Self::Output>;
}

/// A collector whose concrete type a [`DynCollectorManager`] can recover.
pub trait AnyCollector: ScoringCollector + Send {
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

impl<T: ScoringCollector + Send + 'static> AnyCollector for T {
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

/// A [`CollectorManager`] with its types erased, as
/// [`MultiCollectorManager`] holds several of them (Java's
/// `CollectorManager<Collector, ?>`). Every `CollectorManager` whose
/// collector and output are `'static` is one.
pub trait DynCollectorManager: Sync {
    fn new_dyn_collector(&self) -> Result<Box<dyn AnyCollector>>;
    fn reduce_dyn(&self, collectors: Vec<Box<dyn AnyCollector>>) -> Result<Box<dyn Any + Send>>;
}

impl<M> DynCollectorManager for M
where
    M: CollectorManager,
    M::Collector: 'static,
    M::Output: Send + 'static,
{
    fn new_dyn_collector(&self) -> Result<Box<dyn AnyCollector>> {
        Ok(Box::new(self.new_collector()?))
    }

    fn reduce_dyn(&self, collectors: Vec<Box<dyn AnyCollector>>) -> Result<Box<dyn Any + Send>> {
        let typed = collectors
            .into_iter()
            .map(|c| {
                c.into_any()
                    .downcast::<M::Collector>()
                    .map(|b| *b)
                    .map_err(|_| {
                        Error::IllegalArgument("collector was not made by this manager".to_string())
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Box::new(self.reduce(typed)?))
    }
}

/// `MultiCollectorManager`: one [`MultiCollector`] per slice over a
/// collector from each manager; reduces to each manager's own result, in
/// order (Java's `Object[]`), to be downcast by the caller.
pub struct MultiCollectorManager<'m> {
    managers: Vec<&'m dyn DynCollectorManager>,
}

impl<'m> MultiCollectorManager<'m> {
    /// # Errors
    /// [`Error::IllegalArgument`] for no managers ("There must be at least one
    /// collector manager").
    pub fn new(managers: Vec<&'m dyn DynCollectorManager>) -> Result<Self> {
        if managers.is_empty() {
            return Err(Error::IllegalArgument(
                "There must be at least one collector manager".to_string(),
            ));
        }
        Ok(Self { managers })
    }
}

/// The collector a [`MultiCollectorManager`] makes: one of each manager's.
pub struct ManagedMultiCollector {
    collectors: Vec<Box<dyn AnyCollector>>,
}

impl ScoringCollector for ManagedMultiCollector {
    #[inline]
    fn collect(&mut self, doc_id: i32, score: f32) {
        for c in &mut self.collectors {
            c.collect(doc_id, score);
        }
    }

    fn score_mode(&self) -> ScoreMode {
        combined_score_mode(self.collectors.iter().map(|c| c.score_mode()))
            .unwrap_or(ScoreMode::Complete)
    }

    fn min_competitive_score(&self) -> Option<f32> {
        if self.collectors.len() == 1 {
            return self.collectors[0].min_competitive_score();
        }
        if self.score_mode() != ScoreMode::TopScores {
            return None;
        }
        let mut min = f32::MAX;
        for c in &self.collectors {
            min = min.min(c.min_competitive_score()?);
        }
        Some(min)
    }
}

impl CollectorManager for MultiCollectorManager<'_> {
    type Collector = ManagedMultiCollector;
    type Output = Vec<Box<dyn Any + Send>>;

    fn new_collector(&self) -> Result<ManagedMultiCollector> {
        Ok(ManagedMultiCollector {
            collectors: self
                .managers
                .iter()
                .map(|m| m.new_dyn_collector())
                .collect::<Result<_>>()?,
        })
    }

    fn reduce(&self, collectors: Vec<ManagedMultiCollector>) -> Result<Self::Output> {
        let mut per_manager: Vec<Vec<Box<dyn AnyCollector>>> =
            self.managers.iter().map(|_| Vec::new()).collect();
        for multi in collectors {
            for (i, c) in multi.collectors.into_iter().enumerate() {
                if let Some(slot) = per_manager.get_mut(i) {
                    slot.push(c);
                }
            }
        }
        self.managers
            .iter()
            .zip(per_manager)
            .map(|(m, cs)| m.reduce_dyn(cs))
            .collect()
    }
}

#[cfg(test)]
mod tests {
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
        let single =
            MultiCollector::wrap(vec![Some(Box::new(&mut d) as BoxCollector<'_>)]).unwrap();
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
}
