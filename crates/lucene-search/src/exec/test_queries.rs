//! The scorers of [`crate::test_support::TestQuery`]: lucene-test-framework's
//! `RandomApproximationQuery` (a scorer turned into a two-phase one whose
//! approximation adds random false positives) and the deliberately broken
//! scorers of `TestQueryUtils`, which `QueryUtils` must catch.
//!
//! Compiled only under `cfg(test)` or the `asserting` feature.

use lucene_util::java_random::JavaRandom;

use super::asserting::random_int_between;
use super::{build, exact_advance, exact_next, BoxScorer, LeafContext, Mode, Scorer, NO_MORE_DOCS};
use crate::test_support::{BrokenKind, TestKind, TestQuery};
use crate::Result;

/// `Weight.scorerSupplier(leaf).get(...)` for a [`TestQuery`]: the wrapped
/// query's scorer (`in.scorer(context)`, not top-level), wrapped.
pub(crate) fn build<'a>(
    ctx: &LeafContext<'a>,
    q: &TestQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let Some(inner) = build::build(ctx, q.query(), boost, mode, false)? else {
        return Ok(None);
    };
    Ok(Some(match q.kind() {
        // `new RandomApproximationScorer(subScorer, new Random(random.nextLong()))`.
        TestKind::RandomApproximation(random) => {
            let seed = random
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .next_long();
            Box::new(RandomApproximationScorer::new(inner, JavaRandom::new(seed)))
        }
        TestKind::Broken(kind) => Box::new(BrokenScorer {
            inner,
            kind: *kind,
            doc: -1,
            last_was_advance: false,
            exhausted: false,
            last_scored_doc: -2,
            score_calls: 0,
        }),
    }))
}

/// `RandomApproximationScorer` with its `RandomTwoPhaseView` and
/// `RandomApproximation`: the approximation lands anywhere in
/// `[target, next match]`, and `matches()` confirms it is on the match.
struct RandomApproximationScorer<'a> {
    /// The wrapped scorer, iterated exactly (`scorer.iterator()`).
    inner: BoxScorer<'a>,
    random: JavaRandom,
    /// The approximation's document.
    doc: i32,
    /// The document `matches()` was last called on.
    last_doc: i32,
    match_cost: f32,
}

impl<'a> RandomApproximationScorer<'a> {
    fn new(inner: BoxScorer<'a>, mut random: JavaRandom) -> Self {
        let match_cost = random.next_float() * 200.0;
        Self {
            inner,
            random,
            doc: -1,
            last_doc: -1,
            match_cost,
        }
    }
}

impl Scorer for RandomApproximationScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.doc.saturating_add(1))
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        if self.inner.doc_id() < target {
            exact_advance(&mut *self.inner, target)?;
        }
        let disi = self.inner.doc_id();
        self.doc = if disi == NO_MORE_DOCS {
            NO_MORE_DOCS
        } else {
            random_int_between(&mut self.random, target, disi)
        };
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
    fn two_phase(&self) -> bool {
        true
    }
    fn matches(&mut self) -> Result<bool> {
        assert!(
            self.doc != -1 && self.doc != NO_MORE_DOCS,
            "matches() should not be called on doc ID {}",
            self.doc
        );
        assert!(
            self.last_doc != self.doc,
            "matches() has been called twice on doc ID {}",
            self.doc
        );
        self.last_doc = self.doc;
        Ok(self.doc == self.inner.doc_id())
    }
    fn match_cost(&self) -> f32 {
        self.match_cost
    }
    fn score(&mut self) -> Result<f32> {
        self.inner.score()
    }
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        // The approximation's false positives sit before the wrapped
        // scorer's next match, which is already ahead of them.
        let inner = self.inner.doc_id();
        let target = if inner > target && self.doc != inner {
            inner
        } else {
            target
        };
        self.inner.advance_shallow(target)
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        self.inner.max_score(up_to)
    }
    fn doc_id_run_end(&self) -> i32 {
        // `disi.docIDRunEnd()` holds for the exact iterator only, which a
        // two-phase wrapped scorer's approximation is not.
        if self.doc == self.inner.doc_id() && !self.inner.two_phase() {
            self.inner.doc_id_run_end()
        } else {
            self.doc.saturating_add(1)
        }
    }
}

/// `TestQueryUtils`' broken scorers, one per [`BrokenKind`], over the
/// wrapped scorer's exact iterator.
struct BrokenScorer<'a> {
    inner: BoxScorer<'a>,
    kind: BrokenKind,
    doc: i32,
    /// `ScoreMismatchQuery`: whether the current document was reached by
    /// `advance`.
    last_was_advance: bool,
    /// `EarlyTerminationQuery`: whether the iterator has moved once.
    exhausted: bool,
    /// `UnstableScoreQuery`: the document scored last and the calls on it.
    last_scored_doc: i32,
    score_calls: u32,
}

impl Scorer for BrokenScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.last_was_advance = false;
        if self.kind == BrokenKind::EarlyTermination {
            if self.exhausted {
                self.doc = NO_MORE_DOCS;
                return Ok(self.doc);
            }
            self.exhausted = true;
        }
        self.doc = exact_next(&mut *self.inner)?;
        Ok(self.doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.last_was_advance = true;
        self.exhausted = true;
        self.doc = match self.kind {
            BrokenKind::Advance => NO_MORE_DOCS,
            BrokenKind::SkipTooFar => {
                let doc = exact_advance(&mut *self.inner, target)?;
                if doc == NO_MORE_DOCS {
                    doc
                } else {
                    exact_next(&mut *self.inner)?
                }
            }
            _ => exact_advance(&mut *self.inner, target)?,
        };
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
    fn score(&mut self) -> Result<f32> {
        let score = self.inner.score()?;
        Ok(match self.kind {
            BrokenKind::UnstableScore => {
                if self.doc != self.last_scored_doc {
                    self.last_scored_doc = self.doc;
                    self.score_calls = 0;
                }
                self.score_calls += 1;
                if self.score_calls.is_multiple_of(2) {
                    score + 1.0
                } else {
                    score
                }
            }
            BrokenKind::ScoreMismatch if self.last_was_advance => score + 100.0,
            _ => score,
        })
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let max = self.inner.max_score(up_to)?;
        Ok(match self.kind {
            BrokenKind::UnstableScore => max + 1.0,
            BrokenKind::ScoreMismatch => max + 100.0,
            _ => max,
        })
    }
}
