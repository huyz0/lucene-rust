//! `DrillSidewaysScorer`: one segment of a drill-sideways search. Scores the
//! base query against every drill-down dim at once, handing each document
//! that matches every dim (a hit) to the main collector and to every dim's
//! sideways collector, and each document that misses exactly one dim (a near
//! miss) to that dim's sideways collector only.
//!
//! Three strategies, chosen from the costs as Java chooses them:
//! * query first ([`DrillSidewaysScorer::do_query_first_scoring`]): the base
//!   query leads and each dim is advanced to it -- when the base is far
//!   cheaper than the dims, or when every sub-scorer must be positioned on
//!   the collected document (`scoreSubDocsAtOnce`);
//! * drill-down advance ([`DrillSidewaysScorer::do_drill_down_advance_scoring`]):
//!   the two cheapest dims fill a 2,048-document window and the base is
//!   advanced to their documents -- when the second dim is far cheaper than
//!   the base;
//! * union ([`DrillSidewaysScorer::do_union_scoring`]): the base fills the
//!   window and each dim marks its documents in it.
//!
//! Differs: a collector here takes its document's score with the document
//! (there is no `Scorable`). The score is the base scorer's, read once per
//! collected document, and `0` when the main collector does not need scores
//! (the base scorer was then built without them).

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::exec::{exact_advance, exact_next, BoxScorer, Scorer, NO_MORE_DOCS};
use crate::leaf_collector::SegmentCollector;
use crate::{Error, Result};

/// `CHUNK`: the window of the two windowed strategies.
const CHUNK: usize = 2048;
/// `MASK`.
const MASK: i32 = 2047;

/// `DocsAndCost`: a dim's scorer (its approximation and, when
/// [`Self::two_phase`], its confirmation) and which sideways collector its
/// near misses go to.
pub(crate) struct DocsAndCost<'a> {
    scorer: BoxScorer<'a>,
    /// `cost`: the approximation's.
    pub(crate) cost: i64,
    /// Whether `twoPhase != null`.
    two_phase: bool,
    /// `matchCost`.
    match_cost: f32,
    /// The index of this dim's sideways collector.
    collector: usize,
}

impl<'a> DocsAndCost<'a> {
    /// `new DocsAndCost(scorer, sidewaysLeafCollector)`.
    pub(crate) fn new(scorer: BoxScorer<'a>, collector: usize) -> Self {
        let two_phase = scorer.two_phase();
        let match_cost = if two_phase { scorer.match_cost() } else { 0.0 };
        DocsAndCost {
            cost: scorer.cost(),
            scorer,
            two_phase,
            match_cost,
            collector,
        }
    }

    fn doc_id(&self) -> i32 {
        self.scorer.doc_id()
    }

    /// `twoPhase == null || twoPhase.matches()`.
    fn confirmed(&mut self) -> Result<bool> {
        if self.two_phase {
            self.scorer.matches()
        } else {
            Ok(true)
        }
    }
}

/// `DocIdSetIterator.empty()` under a `ConstantScoreScorer`: what a dim
/// without a scorer in the segment is given.
pub(crate) struct EmptyScorer;

impl Scorer for EmptyScorer {
    fn doc_id(&self) -> i32 {
        NO_MORE_DOCS
    }
    fn next_doc(&mut self) -> Result<i32> {
        Ok(NO_MORE_DOCS)
    }
    fn advance(&mut self, _target: i32) -> Result<i32> {
        Ok(NO_MORE_DOCS)
    }
    fn cost(&self) -> i64 {
        0
    }
    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }
}

/// The strategy a [`DrillSidewaysScorer::score`] took, which the tests
/// record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Strategy {
    QueryFirstSingle,
    QueryFirst,
    DrillDownAdvance,
    Union,
}

#[cfg(test)]
thread_local! {
    /// Every strategy taken on this thread, in order.
    pub(crate) static STRATEGIES: std::cell::RefCell<Vec<Strategy>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
fn record(s: Strategy) {
    STRATEGIES.with(|v| v.borrow_mut().push(s));
}

#[cfg(not(test))]
#[inline(always)]
fn record(_: Strategy) {}

/// `DrillSidewaysScorer`.
pub(crate) struct DrillSidewaysScorer<'a> {
    dims: Vec<DocsAndCost<'a>>,
    base: BoxScorer<'a>,
    base_two_phase: bool,
    /// `context.reader().maxDoc()`.
    max_doc: i32,
    score_sub_docs_at_once: bool,
    /// Whether the main collector reads scores (the base scorer has them).
    needs_scores: bool,
}

impl<'a> DrillSidewaysScorer<'a> {
    /// `new DrillSidewaysScorer(context, baseScorer, dims,
    /// scoreSubDocsAtOnce)`.
    pub(crate) fn new(
        max_doc: i32,
        base: BoxScorer<'a>,
        dims: Vec<DocsAndCost<'a>>,
        score_sub_docs_at_once: bool,
        needs_scores: bool,
    ) -> Self {
        DrillSidewaysScorer {
            base_two_phase: base.two_phase(),
            dims,
            base,
            max_doc,
            score_sub_docs_at_once,
            needs_scores,
        }
    }

    /// The base scorer's score on its current document (when it is needed).
    fn base_score(&mut self) -> Result<f32> {
        if self.needs_scores {
            self.base.score()
        } else {
            Ok(0.0)
        }
    }

    /// `score(collector, acceptDocs, min, maxDoc)`: the whole segment, then
    /// `finish` on every sideways collector (whatever the scoring reported).
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] unless `min` is `0` and `max` is
    /// `Integer.MAX_VALUE` (Java's two refusals), and whatever a scorer or a
    /// collector reports.
    pub(crate) fn score<'c, C, K>(
        &mut self,
        collector: &mut C,
        sideways: &mut [K],
        accept_docs: Option<&FixedBitSet>,
        min: i32,
        max: i32,
    ) -> Result<i32>
    where
        C: SegmentCollector<'c> + ?Sized,
        K: SegmentCollector<'c>,
    {
        if min != 0 {
            return Err(Error::IllegalArgument(format!("min must be 0, got {min}")));
        }
        if max != i32::MAX {
            return Err(Error::IllegalArgument(
                "maxDoc must be Integer.MAX_VALUE".into(),
            ));
        }
        let scored = self.score_all(collector, sideways, accept_docs);
        // `finally { finish(dims) }`.
        let mut finished = Ok(());
        for dim in &self.dims {
            if let Some(c) = sideways.get_mut(dim.collector) {
                let f = c.finish();
                if finished.is_ok() {
                    finished = f;
                }
            }
        }
        scored?;
        finished?;
        Ok(i32::MAX)
    }

    fn score_all<'c, C, K>(
        &mut self,
        collector: &mut C,
        sideways: &mut [K],
        accept_docs: Option<&FixedBitSet>,
    ) -> Result<()>
    where
        C: SegmentCollector<'c> + ?Sized,
        K: SegmentCollector<'c>,
    {
        let base_query_cost = self.base.cost();
        let num_dims = self.dims.len();
        let drill_down_cost = self
            .dims
            .iter()
            .fold(0i64, |sum, d| sum.saturating_add(d.cost));
        let drill_down_advanced_cost = if num_dims > 1 { self.dims[1].cost } else { 0 };
        // Position the dims' scorers on their first documents.
        for dim in &mut self.dims {
            dim.scorer.next_doc()?;
        }
        if self.score_sub_docs_at_once || base_query_cost < drill_down_cost / 10 {
            self.base.next_doc()?;
            self.do_query_first_scoring(accept_docs, collector, sideways)
        } else if num_dims > 1 && drill_down_advanced_cost < base_query_cost / 10 {
            exact_next(&mut *self.base)?;
            self.do_drill_down_advance_scoring(accept_docs, collector, sideways)
        } else {
            exact_next(&mut *self.base)?;
            self.do_union_scoring(accept_docs, collector, sideways)
        }
    }

    /// `doQueryFirstScoringSingleDim`.
    fn do_query_first_scoring_single_dim<'c, C, K>(
        &mut self,
        accept_docs: Option<&FixedBitSet>,
        collector: &mut C,
        sideways: &mut [K],
    ) -> Result<()>
    where
        C: SegmentCollector<'c> + ?Sized,
        K: SegmentCollector<'c>,
    {
        record(Strategy::QueryFirstSingle);
        let mut doc = self.base.doc_id();
        while doc != NO_MORE_DOCS {
            if accept_docs.is_some_and(|live| !live.get_doc(doc)) {
                doc = self.base.next_doc()?;
                continue;
            }
            if self.base_two_phase && !self.base.matches()? {
                doc = self.base.next_doc()?;
                continue;
            }
            // A near miss or a hit: the sideways dim tells which.
            let dim = &mut self.dims[0];
            let missed = advance_if_behind(doc, &mut *dim.scorer)? != doc || !dim.confirmed()?;
            let target = dim.collector;
            let score = self.base_score()?;
            if !missed {
                collector.collect(doc, score)?;
            }
            collect_in(sideways, target, doc, score)?;
            doc = self.base.next_doc()?;
        }
        Ok(())
    }

    /// `doQueryFirstScoring`.
    fn do_query_first_scoring<'c, C, K>(
        &mut self,
        accept_docs: Option<&FixedBitSet>,
        collector: &mut C,
        sideways: &mut [K],
    ) -> Result<()>
    where
        C: SegmentCollector<'c> + ?Sized,
        K: SegmentCollector<'c>,
    {
        if self.dims.len() == 1 {
            return self.do_query_first_scoring_single_dim(accept_docs, collector, sideways);
        }
        record(Strategy::QueryFirst);
        // The dims by approximation cost (a stable sort), so the cheaper
        // ones advance first.
        let mut sideways_dims: Vec<usize> = (0..self.dims.len()).collect();
        sideways_dims.sort_by_key(|&i| self.dims[i].cost);
        // The two-phase dims, by match cost.
        let mut two_phase_dims: Vec<usize> = (0..self.dims.len())
            .filter(|&i| self.dims[i].two_phase)
            .collect();
        two_phase_dims
            .sort_by(|&a, &b| self.dims[a].match_cost.total_cmp(&self.dims[b].match_cost));
        // `runawayDim`: a one-slot priority queue holding the dim that
        // advanced past the document the others are on.
        let mut runaway_dim: Option<usize> = None;
        let mut doc = self.base.doc_id();
        'next_doc: while doc != NO_MORE_DOCS {
            if accept_docs.is_some_and(|live| !live.get_doc(doc)) {
                doc = self.base.next_doc()?;
                continue;
            }
            // A runaway carried over from the last document: caught up yet?
            let mut runaway = runaway_dim;
            if let Some(r) = runaway {
                if self.dims[r].doc_id() <= doc {
                    runaway_dim = None;
                    runaway = None;
                }
            }
            // At most one dim may miss; when a second one does, move on to
            // the closer of the two.
            for &k in &sideways_dims {
                let dim_doc = advance_if_behind(doc, &mut *self.dims[k].scorer)?;
                if dim_doc != doc && Some(k) != runaway {
                    // `insertWithOverflow`.
                    if let Some(top) = runaway_dim {
                        let evicted = if self.dims[top].doc_id() < self.dims[k].doc_id() {
                            runaway_dim = Some(k);
                            top
                        } else {
                            k
                        };
                        let next = self.dims[evicted].doc_id();
                        doc = self.base.advance(next)?;
                        continue 'next_doc;
                    }
                    runaway_dim = Some(k);
                }
            }
            // The base's confirmation is always required.
            if self.base_two_phase && !self.base.matches()? {
                doc = self.base.next_doc()?;
                continue;
            }
            // Then the dims': one may fail only when no approximation did.
            let mut failed_dim = runaway_dim;
            match failed_dim {
                None => {
                    for &k in &two_phase_dims {
                        if !self.dims[k].scorer.matches()? {
                            if failed_dim.is_some() {
                                doc = self.base.next_doc()?;
                                continue 'next_doc;
                            }
                            failed_dim = Some(k);
                        }
                    }
                }
                Some(failed) => {
                    for &k in &two_phase_dims {
                        if k != failed && !self.dims[k].scorer.matches()? {
                            doc = self.base.next_doc()?;
                            continue 'next_doc;
                        }
                    }
                }
            }
            let score = self.base_score()?;
            match failed_dim {
                // A hit.
                None => {
                    collector.collect(doc, score)?;
                    for &k in &sideways_dims {
                        collect_in(sideways, self.dims[k].collector, doc, score)?;
                    }
                }
                // A near miss of exactly one dim.
                Some(f) => collect_in(sideways, self.dims[f].collector, doc, score)?,
            }
            doc = self.base.next_doc()?;
        }
        Ok(())
    }

    /// `doDrillDownAdvanceScoring`: used when the drill-downs constrain far
    /// more than the base query.
    fn do_drill_down_advance_scoring<'c, C, K>(
        &mut self,
        accept_docs: Option<&FixedBitSet>,
        collector: &mut C,
        sideways: &mut [K],
    ) -> Result<()>
    where
        C: SegmentCollector<'c> + ?Sized,
        K: SegmentCollector<'c>,
    {
        record(Strategy::DrillDownAdvance);
        let max_doc = self.max_doc;
        let mut w = Window::new();
        let mut seen = FixedBitSet::new(CHUNK);
        let seen_len = seen.len();
        let mut next_chunk_start = CHUNK as i32;
        loop {
            // The first dim.
            let dc = &mut self.dims[0];
            let mut doc = dc.doc_id();
            while doc < next_chunk_start {
                if accept_docs.is_none_or(|live| live.get_doc(doc)) {
                    let slot = slot_of(doc);
                    if w.doc_ids[slot] != doc && dc.confirmed()? && slot < seen_len {
                        seen.set(slot);
                        w.doc_ids[slot] = doc;
                        w.missing_dims[slot] = 1;
                        w.counts[slot] = 1;
                    }
                }
                doc = dc.scorer.next_doc()?;
            }
            // The second dim. Java's precedence is kept: without deletions
            // (`acceptDocs == null`) its confirmation is not asked.
            let dc = &mut self.dims[1];
            let mut doc = dc.doc_id();
            while doc < next_chunk_start {
                let accepted = match accept_docs {
                    None => true,
                    Some(live) => live.get_doc(doc) && dc.confirmed()?,
                };
                if accepted {
                    let slot = slot_of(doc);
                    if w.doc_ids[slot] != doc {
                        if slot < seen_len {
                            seen.set(slot);
                        }
                        w.doc_ids[slot] = doc;
                        w.missing_dims[slot] = 0;
                        w.counts[slot] = 1;
                    } else if w.missing_dims[slot] >= 1 {
                        w.missing_dims[slot] = 2;
                        w.counts[slot] = 2;
                    } else {
                        w.counts[slot] = 1;
                    }
                }
                doc = dc.scorer.next_doc()?;
            }
            // A document neither of the first two dims has cannot be a hit
            // or a near miss: fold in the base, by advancing it.
            let mut filled_count = 0usize;
            let mut from = 0usize;
            while let Some(slot) = seen.next_set_bit(from) {
                let dd_doc = w.doc_ids[slot];
                let mut base_doc = self.base.doc_id();
                if base_doc < dd_doc {
                    base_doc = exact_advance(&mut *self.base, dd_doc)?;
                }
                if base_doc == dd_doc {
                    w.scores[slot] = self.base_score()?;
                    w.filled_slots[filled_count] = slot;
                    filled_count = filled_count.saturating_add(1);
                    w.counts[slot] = w.counts[slot].saturating_add(1);
                } else {
                    w.doc_ids[slot] = -1;
                }
                from = slot.saturating_add(1);
                if from >= seen_len {
                    break;
                }
            }
            seen.clear_all();
            if filled_count > 0 {
                self.mark_dims(&mut w, 2, next_chunk_start)?;
                self.collect_window(&w, filled_count, collector, sideways)?;
            }
            if next_chunk_start >= max_doc {
                break;
            }
            next_chunk_start = next_chunk_start.saturating_add(CHUNK as i32);
        }
        Ok(())
    }

    /// `doUnionScoring`.
    fn do_union_scoring<'c, C, K>(
        &mut self,
        accept_docs: Option<&FixedBitSet>,
        collector: &mut C,
        sideways: &mut [K],
    ) -> Result<()>
    where
        C: SegmentCollector<'c> + ?Sized,
        K: SegmentCollector<'c>,
    {
        record(Strategy::Union);
        let max_doc = self.max_doc;
        let mut w = Window::new();
        let mut next_chunk_start = CHUNK as i32;
        loop {
            let mut filled_count = 0usize;
            let mut doc = self.base.doc_id();
            while doc < next_chunk_start {
                if accept_docs.is_none_or(|live| live.get_doc(doc)) {
                    let slot = slot_of(doc);
                    w.doc_ids[slot] = doc;
                    w.scores[slot] = self.base_score()?;
                    w.filled_slots[filled_count] = slot;
                    filled_count = filled_count.saturating_add(1);
                    w.missing_dims[slot] = 0;
                    w.counts[slot] = 1;
                }
                doc = exact_next(&mut *self.base)?;
            }
            if filled_count > 0 {
                // The first dim: a `SHOULD` on the base.
                let dc = &mut self.dims[0];
                let mut doc = dc.doc_id();
                while doc < next_chunk_start {
                    let slot = slot_of(doc);
                    // `docIDs[slot] == docID` also says it is not deleted.
                    if w.doc_ids[slot] == doc && dc.confirmed()? {
                        w.missing_dims[slot] = 1;
                        w.counts[slot] = 2;
                    }
                    doc = dc.scorer.next_doc()?;
                }
                self.mark_dims(&mut w, 1, next_chunk_start)?;
                self.collect_window(&w, filled_count, collector, sideways)?;
            }
            if next_chunk_start >= max_doc {
                break;
            }
            next_chunk_start = next_chunk_start.saturating_add(CHUNK as i32);
        }
        Ok(())
    }

    /// The dims from `first` on, over the window ending before
    /// `next_chunk_start`: a document still in the running (every earlier
    /// dim but at most one matched) counts this dim's match, and a document
    /// with no miss so far records this dim as its miss when it does not.
    fn mark_dims(&mut self, w: &mut Window, first: usize, next_chunk_start: i32) -> Result<()> {
        for dim in first..self.dims.len() {
            let dc = &mut self.dims[dim];
            let mut doc = dc.doc_id();
            while doc < next_chunk_start {
                let slot = slot_of(doc);
                if w.doc_ids[slot] == doc && w.counts[slot] >= dim && dc.confirmed()? {
                    if w.missing_dims[slot] >= dim {
                        w.missing_dims[slot] = dim.saturating_add(1);
                        w.counts[slot] = dim.saturating_add(2);
                    } else {
                        w.counts[slot] = dim.saturating_add(1);
                    }
                }
                doc = dc.scorer.next_doc()?;
            }
        }
        Ok(())
    }

    /// The window's documents, in order: a document every dim matched is a
    /// hit, one that missed exactly one dim that dim's near miss.
    fn collect_window<'c, C, K>(
        &self,
        w: &Window,
        filled_count: usize,
        collector: &mut C,
        sideways: &mut [K],
    ) -> Result<()>
    where
        C: SegmentCollector<'c> + ?Sized,
        K: SegmentCollector<'c>,
    {
        let num_dims = self.dims.len();
        for &slot in &w.filled_slots[..filled_count] {
            let doc = w.doc_ids[slot];
            let score = w.scores[slot];
            if w.counts[slot] == num_dims.saturating_add(1) {
                collector.collect(doc, score)?;
                for dim in &self.dims {
                    collect_in(sideways, dim.collector, doc, score)?;
                }
            } else if w.counts[slot] == num_dims {
                if let Some(dim) = self.dims.get(w.missing_dims[slot]) {
                    collect_in(sideways, dim.collector, doc, score)?;
                }
            }
        }
        Ok(())
    }
}

/// The parallel arrays of a window (`docIDs`, `scores`, `missingDims`,
/// `counts`, `filledSlots`).
struct Window {
    doc_ids: Vec<i32>,
    scores: Vec<f32>,
    missing_dims: Vec<usize>,
    counts: Vec<usize>,
    filled_slots: Vec<usize>,
}

impl Window {
    fn new() -> Self {
        let mut doc_ids = vec![0; CHUNK];
        doc_ids[0] = -1;
        Window {
            doc_ids,
            scores: vec![0.0; CHUNK],
            missing_dims: vec![0; CHUNK],
            counts: vec![0; CHUNK],
            filled_slots: vec![0; CHUNK],
        }
    }
}

/// `docID & MASK`.
#[inline]
fn slot_of(doc: i32) -> usize {
    // In `0..CHUNK` for any document.
    usize::try_from(doc & MASK).unwrap_or(0)
}

/// `advanceIfBehind(docID, iterator)`.
fn advance_if_behind(doc: i32, it: &mut dyn Scorer) -> Result<i32> {
    if it.doc_id() < doc {
        it.advance(doc)
    } else {
        Ok(it.doc_id())
    }
}

/// `collect(doc)` on sideways collector `i`.
#[inline]
fn collect_in<'a, K: SegmentCollector<'a>>(
    sideways: &mut [K],
    i: usize,
    doc: i32,
    score: f32,
) -> Result<()> {
    match sideways.get_mut(i) {
        Some(c) => c.collect(doc, score),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests;
