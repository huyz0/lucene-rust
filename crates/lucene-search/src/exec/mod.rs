//! A Lucene-style scorer tree: ports of Lucene 10.5.0's `Scorer` composition
//! (`BooleanScorerSupplier.getInternal`, `ConjunctionScorer`,
//! `BlockMaxConjunctionScorer`, `DisjunctionSumScorer`, `DisjunctionMaxScorer`,
//! `WANDScorer`, `ReqExclScorer`, `ReqOptSumScorer`, `ConstantScoreScorer`)
//! and the `DefaultBulkScorer` loop that drives one into a collector.
//!
//! # Why this module exists
//!
//! [`crate::search_boolean_query_scored`] had fast paths for three shapes --
//! one term, a conjunction of terms, a disjunction of terms -- and one
//! exhaustive path for everything else: resolve every clause to a complete
//! doc list, intersect and subtract those lists, then look every match up in a
//! `HashMap` of summed clause scores. That path reads every posting of every
//! clause and cannot skip anything, so a `must` + `should` query, a
//! `must_not`, a boost or a `constant_score` wrapper cost 4-8x Lucene's time
//! on dense terms (`docs/benchmarks/m2-opensearch-e2e.md`). Lucene never
//! materializes a clause: it composes iterators, advances the rarest one, and
//! prunes with block maxima once the collector publishes a threshold. This is
//! that composition.
//!
//! # The trait
//!
//! [`Scorer`] folds Lucene's `Scorer`, its `iterator()` and its
//! `twoPhaseIterator()` into one object. `next_doc`/`advance` move the
//! **approximation**; a scorer whose [`Scorer::two_phase`] is `true` must then
//! be asked [`Scorer::matches`] before its document counts (Lucene's
//! `TwoPhaseIterator.matches`), and one whose `two_phase` is `false` matches
//! every document its approximation stops on. [`exact_advance`] is
//! `TwoPhaseIterator.asDocIdSetIterator(..).advance`.
//!
//! Scores follow Lucene's arithmetic, including where it is `f32` and where it
//! is `f64` (a conjunction and a disjunction sum in `double`; `ReqOptSumScorer`
//! adds its optional clause in `float`), and boosts are folded into the leaves'
//! weights the way `createWeight(searcher, scoreMode, boost)` passes them down,
//! so a boosted term scores `(boost * idf)`-weighted, not `boost * score`.

pub(crate) mod build;
mod bulk;
pub(crate) mod cache;
mod conjunction;
mod disi_approx;
mod disjunction;
pub(crate) mod extended;
pub(crate) mod function;
pub(crate) mod join;
mod leaf;
pub(crate) mod multi_term;
mod phrase;
pub(crate) mod query_join;
pub(crate) mod ranges;
mod req;
pub(crate) mod span;
mod term_dismax;
mod wand;

pub(crate) use build::LeafContext;
pub(crate) use bulk::{
    bulk_boolean, bulk_clause, score_segment, score_segment_below, score_segment_time_limited,
    score_segment_with_deadline, Bulk,
};

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::ScoringCollector;
use crate::Result;

/// `DocIdSetIterator.NO_MORE_DOCS`.
pub(crate) const NO_MORE_DOCS: i32 = i32::MAX;

/// Lucene's `ScoreMode`, as far as a scorer tree needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// `TOP_SCORES`: scores are needed, and the collector may publish a
    /// minimum competitive score the tree can prune against.
    TopScores,
    /// `COMPLETE`: every match, with its score.
    Complete,
    /// `COMPLETE_NO_SCORES`: every match; scores are never read.
    NoScores,
}

impl Mode {
    pub(crate) fn needs_scores(self) -> bool {
        self != Mode::NoScores
    }

    /// The mode a collector asks for.
    pub(crate) fn of<C: ScoringCollector + ?Sized>(collector: &C) -> Self {
        use crate::collector::ScoreMode;
        match collector.score_mode() {
            ScoreMode::CompleteNoScores => Mode::NoScores,
            m if m.is_exhaustive() => Mode::Complete,
            _ => Mode::TopScores,
        }
    }
}

/// Lucene's `Scorer` + `DocIdSetIterator` + `TwoPhaseIterator`; see the module
/// doc for how the three fold into one.
pub(crate) trait Scorer {
    /// The approximation's current document: `-1` before the first call,
    /// [`NO_MORE_DOCS`] once exhausted.
    fn doc_id(&self) -> i32;
    /// Moves the approximation to its next document.
    fn next_doc(&mut self) -> Result<i32>;
    /// Moves the approximation to the first document at or after `target`,
    /// which must be past the current one.
    fn advance(&mut self, target: i32) -> Result<i32>;
    /// `DocIdSetIterator.cost()`: an upper bound on the number of documents
    /// the approximation visits.
    fn cost(&self) -> i64;
    /// Whether [`Self::matches`] must confirm the approximation's documents.
    fn two_phase(&self) -> bool {
        false
    }
    /// `TwoPhaseIterator.matches()` for the current document.
    fn matches(&mut self) -> Result<bool> {
        Ok(true)
    }
    /// `TwoPhaseIterator.matchCost()`.
    fn match_cost(&self) -> f32 {
        0.0
    }
    /// The current document's score; only valid once it is known to match.
    fn score(&mut self) -> Result<f32>;
    /// `Scorer.advanceShallow`: moves any block-max state to `target` and
    /// returns the last document the bounds it now holds cover.
    fn advance_shallow(&mut self, _target: i32) -> Result<i32> {
        Ok(NO_MORE_DOCS)
    }
    /// `Scorer.getMaxScore(upTo)`: a bound on every score up to and including
    /// `up_to`.
    fn max_score(&mut self, up_to: i32) -> Result<f32>;
    /// `Scorable.setMinCompetitiveScore`: documents scoring below `min` may be
    /// skipped from now on.
    fn set_min_competitive_score(&mut self, _min: f32) -> Result<()> {
        Ok(())
    }
    /// `DocIdSetIterator.docIDRunEnd()` of the approximation: one past the
    /// end of the run of consecutive matches starting at the current document.
    fn doc_id_run_end(&self) -> i32 {
        self.doc_id().saturating_add(1)
    }
    /// A constant-scored iterator over a cached bit set: the set and the
    /// score every document gets, for a bulk scorer to walk the set's words
    /// itself instead of stepping the iterator document by document (Lucene
    /// hands such a range over as a `DocIdStream`). `None` for anything else.
    fn constant_bits(&self) -> Option<(std::sync::Arc<cache::CachedSet>, f32)> {
        None
    }
    /// Random access, for an iterator backed by a bit set (or matching every
    /// document): whether `doc` matches, without moving. `None` for an
    /// iterator that can only be advanced. `ConjunctionDISI` checks such
    /// iterators by membership (`BitSetConjunctionDISI`).
    fn contains(&self, _doc: i32) -> Option<bool> {
        None
    }
    /// `IndriScorer.getBoost()` for a scorer that is an `IndriScorer`
    /// (`IndriAndScorer`); `None` for every other scorer, which
    /// `IndriAndScorer.scoreDoc`'s `instanceof` check skips.
    fn indri_boost(&self) -> Option<f32> {
        None
    }
    /// `Scorable.smoothingScore(docId)`: `0` unless overridden.
    fn smoothing_score(&mut self, _doc: i32) -> Result<f32> {
        Ok(0.0)
    }
    /// `Scorer.nextDocsAndScores`: up to 64 matches from the current document
    /// on, below `up_to`, live ones only, over the *exact* iterator; leaves
    /// the scorer on the first document not returned.
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&FixedBitSet>,
        out: &mut crate::bulk_scorer::DocScores,
    ) -> Result<()> {
        out.docs.clear();
        out.scores.clear();
        let mut doc = self.doc_id();
        while doc < up_to && out.docs.len() < NEXT_DOCS_BATCH {
            if live_docs.is_none_or(|l| l.get_doc(doc)) {
                out.docs.push(doc);
                out.scores.push(self.score()?);
            }
            doc = self.next_doc()?;
            if self.two_phase() {
                while doc != NO_MORE_DOCS && !self.matches()? {
                    doc = self.next_doc()?;
                }
            }
        }
        Ok(())
    }
}

/// The batch [`Scorer::next_docs_and_scores`] fills.
pub(crate) const NEXT_DOCS_BATCH: usize = 64;

pub(crate) type BoxScorer<'a> = Box<dyn Scorer + 'a>;

/// A scorer whose documents end before `end`: at or past it, it reports
/// [`NO_MORE_DOCS`]. A sorted search's last segment under `terminate_after`
/// iterates only up to the cut this way, instead of past it with every later
/// document masked out.
pub(crate) struct Below<'a> {
    inner: BoxScorer<'a>,
    end: i32,
    done: bool,
}

impl<'a> Below<'a> {
    pub(crate) fn new(inner: BoxScorer<'a>, end: i32) -> Self {
        Self {
            inner,
            end,
            done: false,
        }
    }

    fn clip(&mut self, doc: i32) -> i32 {
        if doc >= self.end {
            self.done = true;
            NO_MORE_DOCS
        } else {
            doc
        }
    }
}

impl Scorer for Below<'_> {
    fn doc_id(&self) -> i32 {
        if self.done {
            NO_MORE_DOCS
        } else {
            self.inner.doc_id()
        }
    }
    fn next_doc(&mut self) -> Result<i32> {
        if self.done {
            return Ok(NO_MORE_DOCS);
        }
        let d = self.inner.next_doc()?;
        Ok(self.clip(d))
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        if self.done || target >= self.end {
            self.done = true;
            return Ok(NO_MORE_DOCS);
        }
        let d = self.inner.advance(target)?;
        Ok(self.clip(d))
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
        self.inner.score()
    }
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        self.inner.advance_shallow(target)
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        self.inner.max_score(up_to)
    }
    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        self.inner.set_min_competitive_score(min)
    }
    fn doc_id_run_end(&self) -> i32 {
        self.inner.doc_id_run_end().min(self.end)
    }
    fn contains(&self, doc: i32) -> Option<bool> {
        self.inner.contains(doc).map(|m| m && doc < self.end)
    }
}

/// `TwoPhaseIterator.asDocIdSetIterator(...).advance(target)`.
pub(crate) fn exact_advance(s: &mut dyn Scorer, target: i32) -> Result<i32> {
    let doc = s.advance(target)?;
    confirm(s, doc)
}

/// `TwoPhaseIterator.asDocIdSetIterator(...).nextDoc()`.
pub(crate) fn exact_next(s: &mut dyn Scorer) -> Result<i32> {
    let doc = s.next_doc()?;
    confirm(s, doc)
}

fn confirm(s: &mut dyn Scorer, mut doc: i32) -> Result<i32> {
    if s.two_phase() {
        while doc != NO_MORE_DOCS && !s.matches()? {
            doc = s.next_doc()?;
        }
    }
    Ok(doc)
}

#[cfg(test)]
mod tests;
