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
    #[inline]
    fn collect_range(&mut self, min: i32, max: i32, score: f32) {
        (**self).collect_range(min, max, score);
    }
}

/// A boxed collector, as [`MultiCollector`] holds them.
pub type BoxCollector<'c> = Box<dyn ScoringCollector + Send + 'c>;

/// `MultiCollector`: every hit goes to each of several collectors.
pub struct MultiCollector<'c> {
    collectors: Vec<BoxCollector<'c>>,
    /// `scoreMode()`, computed once as Java's constructor does.
    mode: ScoreMode,
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
        let mode = combined_score_mode(collectors.iter().map(|c| c.score_mode()))
            .unwrap_or(ScoreMode::Complete);
        Ok(Self { collectors, mode })
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

    #[inline]
    fn score_mode(&self) -> ScoreMode {
        self.mode
    }

    #[inline]
    fn collect_range(&mut self, min: i32, max: i32, score: f32) {
        for c in &mut self.collectors {
            c.collect_range(min, max, score);
        }
    }

    // A lone collector is Java's unwrapped one: its batch paths stay live.
    fn collect_many(&mut self, docs: &[i32], scores: &[f32], doc_base: i32) -> usize {
        match self.collectors.as_mut_slice() {
            [c] => c.collect_many(docs, scores, doc_base),
            _ => 0,
        }
    }

    fn count_losing_hits(&mut self, n: u64, score: f32) -> bool {
        match self.collectors.as_mut_slice() {
            [c] => c.count_losing_hits(n, score),
            _ => false,
        }
    }

    fn add_hits(&mut self, n: u64) -> bool {
        match self.collectors.as_mut_slice() {
            [c] => c.add_hits(n),
            _ => false,
        }
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
mod tests;
