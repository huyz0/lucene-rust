//! Leaf scorers and the single-child wrappers: `TermScorer`,
//! `ConstantScoreScorer`, the zero-score `FilterScorer` a lone `FILTER` clause
//! becomes, `MatchAllDocsQuery`'s range, and a materialized list for clause
//! kinds that have no streaming scorer yet.

use super::{BoxScorer, Scorer, NO_MORE_DOCS};
use crate::bulk_scorer::TermLeg;
use crate::Result;

/// `TermScorer`: a [`TermLeg`] iterated through `ImpactsDISI` once the tree,
/// running in `TOP_SCORES`, hands it a minimum competitive score (so its
/// non-competitive blocks are skipped), and plainly otherwise.
///
/// Until that first threshold `ImpactsDISI` passes every call straight
/// through (its `upTo` is `NO_MORE_DOCS`), so this does too, without the
/// indirection: `impacts` turns on with the first threshold.
pub(crate) struct TermScorer<'a> {
    leg: TermLeg<'a>,
    /// `TOP_SCORES`: a threshold may be set.
    top_scores: bool,
    /// A threshold has been set: iterate through the impacts.
    impacts: bool,
}

impl<'a> TermScorer<'a> {
    pub(crate) fn new(leg: TermLeg<'a>, top_scores: bool) -> Self {
        Self {
            leg,
            top_scores,
            impacts: false,
        }
    }
}

impl Scorer for TermScorer<'_> {
    #[inline]
    fn doc_id(&self) -> i32 {
        self.leg.doc_id()
    }

    #[inline]
    fn next_doc(&mut self) -> Result<i32> {
        if self.impacts {
            self.leg.impacts_next_doc()
        } else {
            self.leg.next_doc()
        }
    }

    #[inline]
    fn advance(&mut self, target: i32) -> Result<i32> {
        if self.impacts {
            self.leg.impacts_advance(target)
        } else {
            self.leg.advance(target)
        }
    }

    fn cost(&self) -> i64 {
        self.leg.cost
    }

    #[inline]
    fn score(&mut self) -> Result<f32> {
        self.leg.score()
    }

    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        self.leg.shallow_advance(target)
    }

    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        Ok(self.leg.max_score(up_to))
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        if self.top_scores && min > 0.0 {
            self.leg.set_min_competitive_score(min);
            self.impacts = true;
        }
        Ok(())
    }

    fn doc_id_run_end(&self) -> i32 {
        self.leg.doc_id_run_end()
    }
}

/// `ConstantScoreScorer`: the wrapped scorer's matches, each scoring `score`.
/// In `TOP_SCORES` a minimum competitive score above `score` empties it --
/// `DocIdSetIteratorWrapper.delegate = DocIdSetIterator.empty()` -- which is
/// what lets a filter-only query stop once the top hits fill.
pub(crate) struct ConstantScorer<'a> {
    inner: BoxScorer<'a>,
    score: f32,
    top_scores: bool,
    emptied: bool,
    doc: i32,
}

impl<'a> ConstantScorer<'a> {
    pub(crate) fn new(inner: BoxScorer<'a>, score: f32, top_scores: bool) -> Self {
        let doc = inner.doc_id();
        Self {
            inner,
            score,
            top_scores,
            emptied: false,
            doc,
        }
    }
}

impl Scorer for ConstantScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        self.doc = if self.emptied {
            NO_MORE_DOCS
        } else {
            self.inner.next_doc()?
        };
        Ok(self.doc)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        self.doc = if self.emptied {
            NO_MORE_DOCS
        } else {
            self.inner.advance(target)?
        };
        Ok(self.doc)
    }

    fn cost(&self) -> i64 {
        self.inner.cost()
    }

    fn two_phase(&self) -> bool {
        self.inner.two_phase()
    }

    fn matches(&mut self) -> Result<bool> {
        self.inner.matches()
    }

    fn match_cost(&self) -> f32 {
        self.inner.match_cost()
    }

    fn score(&mut self) -> Result<f32> {
        Ok(self.score)
    }

    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(self.score)
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        if self.top_scores && min > self.score {
            self.emptied = true;
        }
        Ok(())
    }

    fn contains(&self, doc: i32) -> Option<bool> {
        if self.emptied {
            return None;
        }
        self.inner.contains(doc)
    }
}

/// The anonymous `FilterScorer` `BooleanScorerSupplier.req` wraps a lone
/// `FILTER` clause in when scores are needed: its matches, scoring `0`.
pub(crate) struct ZeroScorer<'a>(pub(crate) BoxScorer<'a>);

impl Scorer for ZeroScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.0.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.0.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.0.advance(target)
    }
    fn cost(&self) -> i64 {
        self.0.cost()
    }
    fn two_phase(&self) -> bool {
        self.0.two_phase()
    }
    fn matches(&mut self) -> Result<bool> {
        self.0.matches()
    }
    fn match_cost(&self) -> f32 {
        self.0.match_cost()
    }
    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        self.0.advance_shallow(target)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }

    fn contains(&self, doc: i32) -> Option<bool> {
        self.0.contains(doc)
    }
}

/// Every document in `[0, max_doc)`: `MatchAllDocsQuery`'s iterator
/// (`DocIdSetIterator.all`), scored by the [`ConstantScorer`] around it.
pub(crate) struct AllDocs {
    doc: i32,
    max_doc: i32,
}

impl AllDocs {
    pub(crate) fn new(max_doc: i32) -> Self {
        Self { doc: -1, max_doc }
    }
}

impl Scorer for AllDocs {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.doc.saturating_add(1))
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.doc = if target >= self.max_doc {
            NO_MORE_DOCS
        } else {
            target
        };
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        i64::from(self.max_doc)
    }
    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }

    fn contains(&self, doc: i32) -> Option<bool> {
        Some((0..self.max_doc).contains(&doc))
    }
}

/// A clause resolved up front to its ascending matches and their scores --
/// the fallback for clause kinds without a streaming scorer here yet. Correct
/// for any clause; as expensive as the materializing path it replaces, but
/// only for the one clause, not the whole query.
pub(crate) struct DocList {
    docs: Vec<i32>,
    /// Aligned with `docs`; empty when scores are not needed.
    scores: Vec<f32>,
    max: f32,
    /// Index of the current document in `docs`; `-1` before the first.
    at: isize,
}

impl DocList {
    pub(crate) fn new(docs: Vec<i32>, scores: Vec<f32>) -> Self {
        debug_assert!(scores.is_empty() || scores.len() == docs.len());
        debug_assert!(docs.windows(2).all(|w| w[0] < w[1]));
        let max = scores.iter().copied().fold(0.0f32, f32::max);
        Self {
            docs,
            scores,
            max,
            at: -1,
        }
    }

    // SENTINEL: `-1` = "not yet positioned", `DocIdSetIterator`'s own
    // unpositioned doc id. Its callers are `doc_id` (which must report it)
    // and `next_doc`/`advance` (which only return it before a first move,
    // which they never do: both move `at` to 0 or past it first).
    fn current(&self) -> i32 {
        match usize::try_from(self.at) {
            Ok(i) => self.docs.get(i).copied().unwrap_or(NO_MORE_DOCS),
            Err(_) => -1,
        }
    }
}

impl Scorer for DocList {
    fn doc_id(&self) -> i32 {
        self.current()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.at = self.at.saturating_add(1);
        Ok(self.current())
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let from = usize::try_from(self.at.saturating_add(1)).unwrap_or(0);
        let rest = self.docs.get(from..).unwrap_or(&[]);
        let skip = rest.partition_point(|&d| d < target);
        self.at = isize::try_from(from.saturating_add(skip)).unwrap_or(isize::MAX);
        Ok(self.current())
    }
    fn cost(&self) -> i64 {
        i64::try_from(self.docs.len()).unwrap_or(i64::MAX)
    }
    fn score(&mut self) -> Result<f32> {
        Ok(usize::try_from(self.at)
            .ok()
            .and_then(|i| self.scores.get(i).copied())
            .unwrap_or(0.0))
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(self.max)
    }
}
