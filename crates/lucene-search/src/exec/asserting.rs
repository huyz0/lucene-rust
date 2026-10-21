//! lucene-test-framework's asserting search wrappers, at the level they need
//! to live at here: `AssertingScorer` (around every scorer
//! [`super::build::build`] returns), `AssertingBulkScorer` (a segment scored
//! in random windows) and `AssertingLeafCollector` (each window's collector).
//!
//! Java installs them through `AssertingIndexSearcher.createWeight`; this
//! port has no `Weight` to wrap, so a thread-local switch
//! ([`AssertingGuard`], from [`crate::test_support::asserting`]) makes the
//! scorer tree builder and [`super::score_segment`] install them for the
//! searches run on that thread while it is held. Every check is an
//! `assert!`: a broken contract panics, as Java's `assert` throws.
//!
//! Compiled only under `cfg(test)` or the `asserting` feature, which only
//! `lucene-test-framework` enables: a production build, and the benchmark
//! runner (`test-support` only; check-port-invariants.py's
//! `bench-not-asserting` rule), have neither the module nor the checks at
//! its call sites.

use std::cell::RefCell;

use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::java_random::JavaRandom;

use super::{cache, BoxScorer, Bulk, Mode, Scorer, NO_MORE_DOCS};
use crate::collector::{ScoreMode, ScoringCollector};
use crate::Result;

thread_local! {
    /// The searcher's `random`, while asserting wrappers are installed.
    static ACTIVE: RefCell<Option<JavaRandom>> = const { RefCell::new(None) };
}

/// Installs the asserting wrappers on this thread until dropped
/// (`new AssertingIndexSearcher(random, reader)`); `random` seeds the bulk
/// scorers' windows. Guards nest: the outer one's state comes back.
pub(crate) struct AssertingGuard {
    previous: Option<JavaRandom>,
}

impl AssertingGuard {
    pub(crate) fn new(seed: i64) -> Self {
        let previous = ACTIVE.with(|a| a.borrow_mut().replace(JavaRandom::new(seed)));
        Self { previous }
    }
}

impl Drop for AssertingGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        ACTIVE.with(|a| *a.borrow_mut() = previous);
    }
}

/// `random.nextLong()` from the installed searcher random, `None` when no
/// [`AssertingGuard`] is held.
fn next_seed() -> Option<i64> {
    ACTIVE.with(|a| a.borrow_mut().as_mut().map(JavaRandom::next_long))
}

/// Whether an [`AssertingGuard`] is held on this thread.
pub(crate) fn active() -> bool {
    ACTIVE.with(|a| a.borrow().is_some())
}

/// `AssertingWeight.scorerSupplier(...).get(leadCost)`: `scorer` wrapped
/// when the wrappers are installed, as is otherwise.
pub(crate) fn wrap<'a>(scorer: BoxScorer<'a>, mode: Mode, top_level: bool) -> BoxScorer<'a> {
    if !active() {
        return scorer;
    }
    Box::new(AssertingScorer::new(
        scorer,
        mode.needs_scores(),
        mode == Mode::TopScores && top_level,
    ))
}

/// `AssertingScorer.IteratorState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Approximating,
    Iterating,
    ShallowAdvancing,
    Finished,
}

/// `AssertingScorer`: forwards everything to `inner`, checking the
/// iteration contract around it. This port's scorer is the iterator and the
/// two-phase iterator in one, so `next_doc`/`advance` of a two-phase scorer
/// are its approximation's (`APPROXIMATING` until `matches`), and of any
/// other its iterator's (`ITERATING`).
pub(crate) struct AssertingScorer<'a> {
    inner: BoxScorer<'a>,
    can_score: bool,
    can_set_min_competitive_score: bool,
    state: State,
    doc: i32,
    min_competitive_score: f32,
    last_shallow_target: i32,
}

impl<'a> AssertingScorer<'a> {
    pub(crate) fn new(
        inner: BoxScorer<'a>,
        can_score: bool,
        can_set_min_competitive_score: bool,
    ) -> Self {
        let doc = inner.doc_id();
        Self {
            inner,
            can_score,
            can_set_min_competitive_score,
            state: State::Iterating,
            doc,
            min_competitive_score: 0.0,
            last_shallow_target: -1,
        }
    }

    /// `iterating()`: positioned on a document known to match.
    fn iterating(&self) -> bool {
        match self.doc_id() {
            -1 | NO_MORE_DOCS => false,
            _ => self.state == State::Iterating,
        }
    }

    /// The state a move to `doc` leaves: past the end, or on a candidate
    /// (two-phase) or a match.
    fn moved_to(&mut self, doc: i32) {
        self.state = if doc == NO_MORE_DOCS {
            State::Finished
        } else if self.inner.two_phase() {
            State::Approximating
        } else {
            State::Iterating
        };
        self.doc = doc;
    }
}

impl Scorer for AssertingScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }

    fn next_doc(&mut self) -> Result<i32> {
        assert!(
            self.state != State::Finished,
            "nextDoc() called after NO_MORE_DOCS"
        );
        assert!(self.doc_id().saturating_add(1) >= self.last_shallow_target);
        let next = self.inner.next_doc()?;
        assert!(
            next > self.doc,
            "backwards nextDoc from {} to {next}",
            self.doc
        );
        assert_eq!(self.inner.doc_id(), next);
        self.moved_to(next);
        Ok(next)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        assert!(
            self.state != State::Finished,
            "advance() called after NO_MORE_DOCS"
        );
        assert!(
            target > self.doc,
            "target must be > docID(), got {target} <= {}",
            self.doc
        );
        assert!(target >= self.last_shallow_target);
        let advanced = self.inner.advance(target)?;
        assert!(
            advanced >= target,
            "backwards advance from: {target} to: {advanced}"
        );
        assert_eq!(self.inner.doc_id(), advanced);
        self.moved_to(advanced);
        Ok(advanced)
    }

    fn cost(&self) -> i64 {
        self.inner.cost()
    }

    fn two_phase(&self) -> bool {
        self.inner.two_phase()
    }

    fn matches(&mut self) -> Result<bool> {
        if !self.inner.two_phase() {
            // No `TwoPhaseIterator`: every document the iterator stops on
            // matches, and asking again changes nothing.
            return self.inner.matches();
        }
        assert!(
            self.state == State::Approximating,
            "matches() in state {:?}",
            self.state
        );
        let matches = self.inner.matches()?;
        if matches {
            assert_eq!(
                self.inner.doc_id(),
                self.doc,
                "Approximation and scorer don't advance synchronously"
            );
            self.state = State::Iterating;
        }
        Ok(matches)
    }

    fn match_cost(&self) -> f32 {
        let cost = self.inner.match_cost();
        assert!(!cost.is_nan());
        assert!(cost >= 0.0);
        cost
    }

    fn score(&mut self) -> Result<f32> {
        assert!(self.can_score, "score() on a scorer built without scores");
        assert!(self.iterating(), "score() in state {:?}", self.state);
        let score = self.inner.score()?;
        assert!(!score.is_nan(), "NaN score");
        if self.last_shallow_target != -1 {
            let doc = self.doc_id();
            let max = self.inner.max_score(doc)?;
            assert!(score <= max, "score {score} > getMaxScore({doc}) = {max}");
        }
        assert!(score.total_cmp(&0.0).is_ge(), "negative score {score}");
        Ok(score)
    }

    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        assert!(
            self.can_score,
            "advanceShallow on a scorer built without scores"
        );
        assert!(
            target >= self.last_shallow_target,
            "called on decreasing targets: target = {target} < last target = {}",
            self.last_shallow_target
        );
        assert!(
            target >= self.doc_id(),
            "target = {target} < docID = {}",
            self.doc_id()
        );
        let up_to = self.inner.advance_shallow(target)?;
        assert!(up_to >= target, "upTo = {up_to} < target = {target}");
        self.last_shallow_target = target;
        if target != self.doc {
            self.state = State::ShallowAdvancing;
        }
        Ok(up_to)
    }

    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        assert!(
            self.can_score,
            "getMaxScore on a scorer built without scores"
        );
        assert!(
            up_to >= self.last_shallow_target,
            "upTo = {up_to} < last target = {}",
            self.last_shallow_target
        );
        assert!(
            self.doc_id() >= 0 || self.last_shallow_target >= 0,
            "Cannot get max scores until the iterator is positioned or advanceShallow has been called"
        );
        self.inner.max_score(up_to)
    }

    fn max_score_unbounded(&self) -> bool {
        self.inner.max_score_unbounded()
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        assert!(self.can_score, "setMinCompetitiveScore without scores");
        assert!(
            self.can_set_min_competitive_score,
            "setMinCompetitiveScore on a scorer that is not the top-level scoring clause of a TOP_SCORES search"
        );
        assert!(!min.is_nan());
        assert!(
            min >= self.min_competitive_score,
            "min competitive score went down: {min} < {}",
            self.min_competitive_score
        );
        self.inner.set_min_competitive_score(min)?;
        self.min_competitive_score = min;
        Ok(())
    }

    fn doc_id_run_end(&self) -> i32 {
        let end = self.inner.doc_id_run_end();
        assert!(
            end > self.doc_id(),
            "docIDRunEnd={end}, docID={}",
            self.doc_id()
        );
        end
    }

    fn constant_bits(&self) -> Option<(std::sync::Arc<cache::CachedSet>, f32)> {
        self.inner.constant_bits()
    }

    fn constant_score(&self) -> Option<f32> {
        self.inner.constant_score()
    }

    fn contains(&self, doc: i32) -> Option<bool> {
        self.inner.contains(doc)
    }

    fn indri_boost(&self) -> Option<f32> {
        self.inner.indri_boost()
    }

    fn smoothing_score(&mut self, doc: i32) -> Result<f32> {
        self.inner.smoothing_score(doc)
    }

    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&FixedBitSet>,
        out: &mut crate::bulk_scorer::DocScores,
    ) -> Result<()> {
        assert!(
            self.doc != -1,
            "nextDocsAndScores on an unpositioned scorer"
        );
        self.inner.next_docs_and_scores(up_to, live_docs, out)?;
        let doc = self.inner.doc_id();
        if doc != self.doc {
            self.doc = doc;
            self.state = if doc == NO_MORE_DOCS {
                State::Finished
            } else {
                State::Iterating
            };
        }
        Ok(())
    }

    fn prefers_batches(&self) -> bool {
        self.inner.prefers_batches()
    }

    fn constant_scores(&self) -> bool {
        self.inner.constant_scores()
    }

    fn batch_matches(&self) -> bool {
        self.inner.batch_matches()
    }

    fn matches_batch(&mut self, docs: &[i32], keep: &mut [bool]) -> Result<()> {
        if let Some(&first) = docs.first() {
            assert!(
                self.state != State::Finished,
                "matches_batch after NO_MORE_DOCS"
            );
            assert!(
                first > self.doc,
                "matches_batch from {first} <= {}",
                self.doc
            );
            assert!(docs.windows(2).all(|w| w[0] < w[1]), "unsorted batch");
        }
        self.inner.matches_batch(docs, keep)?;
        if let (Some(&last), Some(&kept)) = (docs.last(), keep.get(docs.len().saturating_sub(1))) {
            assert_eq!(self.inner.doc_id(), last);
            self.doc = last;
            self.state = if kept {
                State::Iterating
            } else {
                State::Approximating
            };
        }
        Ok(())
    }
}

/// `AssertingLeafCollector`: the documents a bulk scorer's window hands
/// over are in order and inside `[min, max)`.
pub(crate) struct AssertingLeafCollector<'c> {
    inner: &'c mut dyn ScoringCollector,
    min: i32,
    max: i32,
    last_collected: i32,
}

impl<'c> AssertingLeafCollector<'c> {
    pub(crate) fn new(inner: &'c mut dyn ScoringCollector, min: i32, max: i32) -> Self {
        Self {
            inner,
            min,
            max,
            last_collected: -1,
        }
    }

    fn check(&self, doc: i32) {
        assert!(
            doc > self.last_collected,
            "Out of order : {} {doc}",
            self.last_collected
        );
        assert!(doc >= self.min, "Out of range: {doc} < {}", self.min);
        assert!(doc < self.max, "Out of range: {doc} >= {}", self.max);
    }
}

impl ScoringCollector for AssertingLeafCollector<'_> {
    fn collect(&mut self, doc: i32, score: f32) {
        self.check(doc);
        // `AssertingScorable.score()`: `score >= 0`, which NaN fails.
        assert!(score >= 0.0, "score={score} for doc {doc}");
        self.inner.collect(doc, score);
        self.last_collected = doc;
    }
    fn min_competitive_score(&self) -> Option<f32> {
        self.inner.min_competitive_score()
    }
    fn score_mode(&self) -> ScoreMode {
        self.inner.score_mode()
    }
    fn pruning_threshold(&self) -> Option<f32> {
        self.inner.pruning_threshold()
    }
    fn constant_score_hits_needed(&self) -> Option<u64> {
        self.inner.constant_score_hits_needed()
    }
    fn count_losing_hits(&mut self, n: u64, score: f32) -> bool {
        self.inner.count_losing_hits(n, score)
    }
    fn add_hits(&mut self, n: u64) -> bool {
        self.inner.add_hits(n)
    }
    fn collect_many(&mut self, docs: &[i32], scores: &[f32], doc_base: i32) -> usize {
        let mut last = self.last_collected;
        for (&d, &score) in docs.iter().zip(scores) {
            let doc = d.saturating_add(doc_base);
            assert!(score >= 0.0, "score={score} for doc {doc}");
            assert!(doc > last, "Out of order : {last} {doc}");
            assert!(doc >= self.min, "Out of range: {doc} < {}", self.min);
            assert!(doc < self.max, "Out of range: {doc} >= {}", self.max);
            last = doc;
        }
        let n = self.inner.collect_many(docs, scores, doc_base);
        if let Some(i) = n.checked_sub(1) {
            self.last_collected = docs[i].saturating_add(doc_base);
        }
        n
    }
    fn collect_range(&mut self, min: i32, max: i32, score: f32) {
        assert!(min > self.last_collected);
        assert!(max > min);
        assert!(min >= self.min, "Out of range: {min} < {}", self.min);
        assert!(max <= self.max, "Out of range: {} >= {}", max - 1, self.max);
        assert!(score >= 0.0, "score={score} for docs [{min}, {max})");
        self.inner.collect_range(min, max, score);
        self.last_collected = max - 1;
    }
}

/// `AssertingBulkScorer`: one bulk scorer's calls, each checked to move
/// forward, and each split into random windows with its own
/// [`AssertingLeafCollector`].
pub(crate) struct AssertingBulk {
    random: JavaRandom,
    max_doc: Option<i32>,
    max: i32,
}

impl AssertingBulk {
    /// `AssertingBulkScorer.wrap(new Random(random.nextLong()), ...)`, or
    /// `None` when no [`AssertingGuard`] is held.
    pub(crate) fn installed(max_doc: Option<i32>) -> Option<Self> {
        next_seed().map(|seed| Self::new(JavaRandom::new(seed), max_doc))
    }

    pub(crate) fn new(random: JavaRandom, max_doc: Option<i32>) -> Self {
        Self {
            random,
            max_doc,
            max: 0,
        }
    }

    /// `AssertingBulkScorer.score(collector, acceptDocs, min, max)`.
    pub(crate) fn score(
        &mut self,
        bulk: &mut Bulk<'_>,
        mode: Mode,
        live_docs: Option<&FixedBitSet>,
        collector: &mut dyn ScoringCollector,
        min: i32,
        max: i32,
    ) -> Result<i32> {
        assert!(
            min >= self.max,
            "Scoring backward: min={min} while previous max was max={}",
            self.max
        );
        assert!(
            min <= max,
            "max must be greater than min, got min={min}, and max={max}"
        );
        self.max = max;
        let mut outer = AssertingLeafCollector::new(collector, min, max);
        let mut next = min;
        loop {
            let up_to = if self.random.next_boolean() {
                max
            } else {
                let interval = if self.random.next_int_bounded(100) <= 5 {
                    1 + i64::from(self.random.next_int_bounded(10))
                } else {
                    let bound = if self.random.next_boolean() {
                        100
                    } else {
                        5000
                    };
                    1 + i64::from(self.random.next_int_bounded(bound))
                };
                let up_to = (i64::from(next) + interval).min(i64::from(max));
                i32::try_from(up_to).unwrap_or(max)
            };
            let mut window = AssertingLeafCollector::new(&mut outer, next, up_to);
            next = bulk.score(mode, live_docs, &mut window, next, up_to)?;
            if next >= max {
                break;
            }
        }
        // A leaf of unknown `maxDoc` is taken as the widest (every real
        // `maxDoc` is below `NO_MORE_DOCS`): a whole-segment call
        // (`max == NO_MORE_DOCS`) then ends here as Java's, without the
        // `randomIntBetween` draw below.
        let max_doc = self.max_doc.unwrap_or(NO_MORE_DOCS);
        if max >= max_doc || next >= max_doc {
            assert_eq!(next, NO_MORE_DOCS, "a bulk scorer past maxDoc must be done");
            return Ok(NO_MORE_DOCS);
        }
        // `RandomNumbers.randomIntBetween(random, max, next)`.
        Ok(random_int_between(&mut self.random, max, next))
    }
}

/// `RandomNumbers.randomIntBetween(random, min, max)`, as
/// `lucene-test-framework`'s `random_numbers` draws it (this crate cannot
/// depend on that one).
pub(super) fn random_int_between(r: &mut JavaRandom, min: i32, max: i32) -> i32 {
    let range = i64::from(max) - i64::from(min);
    if range < i64::from(i32::MAX) {
        min.wrapping_add(r.next_int_bounded(1 + range as i32))
    } else {
        (i64::from(min) + next_long(r, 1 + range)) as i32
    }
}

/// `RandomNumbers.nextLong(rnd, n)`: uniform in `0..n`, by rejection.
fn next_long(r: &mut JavaRandom, n: i64) -> i64 {
    let mut value = r.next_long();
    let range = n - 1;
    if n & range == 0 {
        value &= range;
    } else {
        let mut u = ((value as u64) >> 1) as i64;
        loop {
            value = u % n;
            if u.wrapping_add(range).wrapping_sub(value) >= 0 {
                break;
            }
            u = ((r.next_long() as u64) >> 1) as i64;
        }
    }
    value
}

#[cfg(test)]
mod tests;
