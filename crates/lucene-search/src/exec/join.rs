//! Per-segment scorers for the block joins ([`crate::join`]): Lucene 10.5.0's
//! `ToParentBlockJoinQuery.BlockJoinScorer` (with `ParentApproximation`,
//! `ParentTwoPhase` and `Score`) and `BlockJoinBulkScorer`,
//! `ToChildBlockJoinQuery.ToChildBlockJoinScorer`, the scorer of
//! `ParentChildrenBlockJoinQuery`, and
//! `ParentsChildrenBlockJoinQuery.ParentsChildrenBlockJoinScorer`, each
//! method by method.
//!
//! As in Lucene, none of them reads live documents: a deleted child scores
//! into its parent through [`BlockJoinScorer`] (the scorer a join gets inside
//! a boolean), while the bulk scorer a top-level `ToParentBlockJoinQuery`
//! runs ([`BlockJoinBulk`]) hands the live documents to its child's bulk
//! scorer and so skips deleted children -- and collects their parent without
//! asking whether it is live. Both follow from where Java applies
//! `acceptDocs`, and both are kept.

use std::sync::Arc;

use lucene_util::fixed_bit_set::FixedBitSet;

use super::build::{self, LeafContext};
use super::{exact_advance, exact_next, BoxScorer, Mode, Scorer, NO_MORE_DOCS};
use crate::collector::ScoringCollector;
use crate::join::{
    ParentChildrenBlockJoinQuery, ParentsChildrenBlockJoinQuery, ScoreMode, ToChildBlockJoinQuery,
    ToParentBlockJoinQuery,
};
use crate::multi_segment::OpenSegment;
use crate::query::{Clause, ConstantScoreQuery};
use crate::{Error, Result};

/// `ToChildBlockJoinQuery.INVALID_QUERY_MESSAGE`.
const INVALID_QUERY_MESSAGE: &str = "Parent query must not match any docs besides parent filter. \
     Combine them as must (+) and must-not (-) clauses to find a problem doc. docID=";

/// The segment a [`LeafContext`] reads, as a [`crate::join::BitSetProducer`]
/// takes it.
fn leaf_segment<'s>(ctx: &LeafContext<'s>) -> OpenSegment<'s> {
    OpenSegment {
        fields: ctx.fields,
        doc_in: ctx.doc_in,
        pos_in: ctx.pos_in,
        pay_in: ctx.pay_in,
        live_docs: ctx.live_docs,
        doc_base: ctx.reader.map_or(0, |r| r.doc_base),
        max_doc: ctx.max_doc.or(ctx.reader.map(|r| r.max_doc)),
        cache: ctx.cache,
        points: ctx.points,
        reader: ctx.reader,
        index_sort_prefix: false,
        termless: None,
        similarity: None,
    }
}

/// `parentsFilter.getBitSet(context)`.
fn parent_bits(
    ctx: &LeafContext<'_>,
    producer: &dyn crate::join::BitSetProducer,
) -> Result<Option<Arc<FixedBitSet>>> {
    producer.bit_set(&leaf_segment(ctx))
}

/// `BitSet.prevSetBit(index) + 1`: where the block holding `index + 1`
/// starts -- one past the last set bit at or before `index`, `0` when there is
/// none (Java's `-1 + 1`). Every caller wants that sum, so the `-1` never
/// leaves here. `index < 0` is `0` too (Java's callers never pass one; a
/// corrupt filter could).
fn block_start(bits: &FixedBitSet, index: i32) -> i32 {
    let at = match usize::try_from(index) {
        Ok(i) => i.min(bits.len().saturating_sub(1)),
        Err(_) => return 0,
    };
    if bits.is_empty() {
        return 0;
    }
    bits.prev_set_bit(at)
        .and_then(|p| i32::try_from(p).ok())
        .map_or(0, |p| p.saturating_add(1))
}

/// `BitSet.nextSetBit(index)` with Java's `NO_MORE_DOCS` for none.
fn next_set_bit(bits: &FixedBitSet, index: i32) -> i32 {
    usize::try_from(index)
        .ok()
        .and_then(|i| bits.next_set_bit(i))
        .and_then(|p| i32::try_from(p).ok())
        .unwrap_or(NO_MORE_DOCS)
}

/// `BitSet.length()`.
fn bits_len(bits: &FixedBitSet) -> i32 {
    i32::try_from(bits.len()).unwrap_or(i32::MAX)
}

/// `Math.min(double, double)`: `NaN` if either is, `-0.0` below `0.0`.
fn java_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a <= b {
        a
    } else {
        b
    }
}

/// `Math.max(double, double)`: `NaN` if either is, `0.0` above `-0.0`.
fn java_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_positive() {
            a
        } else {
            b
        }
    } else if a >= b {
        a
    } else {
        b
    }
}

/// `ToParentBlockJoinQuery.Score`: a parent's score accumulated from its
/// children's, in `double`.
#[derive(Debug, Clone, Copy)]
struct ParentScore {
    mode: ScoreMode,
    score: f64,
    freq: i32,
}

impl ParentScore {
    fn new(mode: ScoreMode) -> Self {
        Self {
            mode,
            score: 0.0,
            freq: 0,
        }
    }

    /// `reset(firstChildScorer)`.
    fn reset(&mut self, child_score: f32) {
        self.score = if self.mode == ScoreMode::None {
            0.0
        } else {
            f64::from(child_score)
        };
        self.freq = 1;
    }

    /// `addChildScore(childScorer)`.
    fn add(&mut self, child_score: f32) {
        let child = if self.mode == ScoreMode::None {
            0.0
        } else {
            child_score
        };
        self.freq = self.freq.saturating_add(1);
        match self.mode {
            ScoreMode::Total | ScoreMode::Avg => self.score += f64::from(child),
            ScoreMode::Min => self.score = java_min(self.score, f64::from(child)),
            ScoreMode::Max => self.score = java_max(self.score, f64::from(child)),
            ScoreMode::None => {}
        }
    }

    /// `score()`.
    fn score(&self) -> f32 {
        let mut score = self.score;
        if self.mode == ScoreMode::Avg {
            score /= f64::from(self.freq);
        }
        score as f32
    }
}

fn child_matches_parent(doc: i32) -> Error {
    Error::IllegalState(format!(
        "Child query must not match same docs with parent filter. Combine them as must clauses \
         (+) to find a problem doc. docId={doc}"
    ))
}

/// `ToParentBlockJoinQuery.createWeight` for the child: `None` scores need
/// no child scores, so the child runs as a `ConstantScoreQuery` with a zero
/// boost; otherwise with the boost, `COMPLETE` unless the mode is `Max`
/// (only a maximum lets the child skip non-competitive documents).
fn child_weight(score_mode: ScoreMode, mode: Mode) -> (ScoreMode, Mode) {
    let child_score_mode = if mode.needs_scores() {
        score_mode
    } else {
        ScoreMode::None
    };
    let child_mode = if child_score_mode != ScoreMode::None && child_score_mode != ScoreMode::Max {
        Mode::Complete
    } else {
        mode
    };
    (child_score_mode, child_mode)
}

/// The child clause as `createWeight` runs it.
fn child_clause(
    q: &ToParentBlockJoinQuery,
    child_score_mode: ScoreMode,
) -> std::borrow::Cow<'_, Clause> {
    if child_score_mode == ScoreMode::None {
        std::borrow::Cow::Owned(Clause::ConstantScore(Box::new(ConstantScoreQuery {
            inner: q.child.clone(),
            score: 1.0,
        })))
    } else {
        std::borrow::Cow::Borrowed(q.child.as_ref())
    }
}

/// `BlockJoinWeight.scorerSupplier(context).get(leadCost)`.
pub(crate) fn to_parent<'a>(
    ctx: &LeafContext<'a>,
    q: &ToParentBlockJoinQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let (child_score_mode, child_mode) = child_weight(q.score_mode, mode);
    let child_boost = if child_score_mode == ScoreMode::None {
        0.0
    } else {
        boost
    };
    let Some(child) = build::build(
        ctx,
        &child_clause(q, child_score_mode),
        child_boost,
        child_mode,
        false,
    )?
    else {
        return Ok(None);
    };
    let Some(parents) = parent_bits(ctx, q.parents.as_ref())? else {
        return Ok(None);
    };
    Ok(Some(Box::new(BlockJoinScorer::new(
        child,
        parents,
        child_score_mode,
    ))))
}

/// `BlockJoinWeight.scorerSupplier(context).bulkScorer()` when it is not
/// `DefaultBulkScorer`: the child's own bulk scorer, wrapped
/// ([`BlockJoinBulk`]). `None` -- use the scorer -- when no scores are read
/// (`ScoreMode.None`).
pub(crate) fn to_parent_bulk<'a>(
    ctx: &LeafContext<'a>,
    q: &ToParentBlockJoinQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<Option<super::Bulk<'a>>>> {
    let (child_score_mode, child_mode) = child_weight(q.score_mode, mode);
    if child_score_mode == ScoreMode::None {
        return Ok(None);
    }
    let Some(child) = super::bulk::bulk_clause(ctx, &q.child, boost, child_mode)? else {
        return Ok(Some(None));
    };
    let Some(parents) = parent_bits(ctx, q.parents.as_ref())? else {
        return Ok(Some(None));
    };
    Ok(Some(Some(super::Bulk::BlockJoin(Box::new(
        BlockJoinBulk {
            child,
            child_mode,
            parents,
            mode: child_score_mode,
        },
    )))))
}

/// `ToParentBlockJoinQuery.BlockJoinScorer`: its iterator is
/// `ParentApproximation` (exact unless the child is two-phase, when
/// `ParentTwoPhase` confirms a parent by finding a matching child).
pub(crate) struct BlockJoinScorer<'a> {
    child: BoxScorer<'a>,
    parents: Arc<FixedBitSet>,
    mode: ScoreMode,
    /// `parentApproximation.docID()`.
    doc: i32,
    parent_score: ParentScore,
}

impl<'a> BlockJoinScorer<'a> {
    fn new(child: BoxScorer<'a>, parents: Arc<FixedBitSet>, mode: ScoreMode) -> Self {
        Self {
            child,
            parents,
            mode,
            doc: -1,
            parent_score: ParentScore::new(mode),
        }
    }

    /// `ParentApproximation.advance(target)`.
    fn approximate(&mut self, target: i32) -> Result<i32> {
        let len = bits_len(&self.parents);
        if target >= len {
            self.doc = NO_MORE_DOCS;
            return Ok(self.doc);
        }
        let first_child_target = if target == 0 {
            0
        } else {
            block_start(&self.parents, target.saturating_sub(1))
        };
        let mut child_doc = self.child.doc_id();
        if child_doc < first_child_target {
            child_doc = self.child.advance(first_child_target)?;
        }
        if child_doc >= len.saturating_sub(1) {
            self.doc = NO_MORE_DOCS;
            return Ok(self.doc);
        }
        self.doc = next_set_bit(&self.parents, child_doc.saturating_add(1));
        Ok(self.doc)
    }

    /// `scoreChildDocs()`.
    fn score_child_docs(&mut self) -> Result<f32> {
        if self.child.doc_id() >= self.doc {
            return Ok(self.parent_score.score());
        }
        let mut score = 0.0f32;
        let two_phase = self.child.two_phase();
        if self.mode != ScoreMode::None {
            let first = self.child.score()?;
            self.parent_score.reset(first);
            while self.child.next_doc()? < self.doc {
                if !two_phase || self.child.matches()? {
                    let s = self.child.score()?;
                    self.parent_score.add(s);
                }
            }
            score = self.parent_score.score();
        }
        if self.child.doc_id() == self.doc && (!two_phase || self.child.matches()?) {
            return Err(child_matches_parent(self.doc));
        }
        Ok(score)
    }

    /// The children of the current parent the child scorer matches, and the
    /// range they lie in -- what `BlockJoinScorer.explain` reports.
    pub(crate) fn current_range(&self) -> (i32, i32) {
        let start = block_start(&self.parents, self.doc.saturating_sub(1));
        (start, self.doc.saturating_sub(1))
    }
}

impl Scorer for BlockJoinScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        let target = self.doc.saturating_add(1);
        self.approximate(target)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.approximate(target)
    }
    fn cost(&self) -> i64 {
        self.child.cost()
    }
    fn two_phase(&self) -> bool {
        self.child.two_phase()
    }
    /// `ParentTwoPhase.matches()`.
    fn matches(&mut self) -> Result<bool> {
        loop {
            if self.child.matches()? {
                return Ok(true);
            }
            if self.child.next_doc()? >= self.doc {
                return Ok(false);
            }
        }
    }
    fn match_cost(&self) -> f32 {
        self.child.match_cost() + 10.0
    }
    fn score(&mut self) -> Result<f32> {
        self.score_child_docs()
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        if self.mode == ScoreMode::None {
            return self.child.max_score(up_to);
        }
        Ok(f32::INFINITY)
    }
    fn max_score_unbounded(&self) -> bool {
        self.mode != ScoreMode::None
    }
    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        if matches!(self.mode, ScoreMode::None | ScoreMode::Max) {
            self.child.set_min_competitive_score(min)?;
        }
        Ok(())
    }
}

/// `BlockJoinBulkScorer`: the child's bulk scorer over whole blocks, its
/// hits folded into their parents by a collector wrapper.
pub(crate) struct BlockJoinBulk<'a> {
    child: super::Bulk<'a>,
    child_mode: Mode,
    parents: Arc<FixedBitSet>,
    mode: ScoreMode,
}

impl BlockJoinBulk<'_> {
    /// `BlockJoinBulkScorer.cost()`: the child bulk scorer's.
    #[cfg_attr(not(any(test, feature = "asserting")), allow(dead_code))]
    pub(crate) fn cost(&self) -> i64 {
        self.child.cost()
    }

    /// `score(collector, acceptDocs, min, max)`.
    pub(crate) fn score<C: ScoringCollector + ?Sized>(
        &mut self,
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        min: i32,
        max: i32,
    ) -> Result<i32> {
        let len = bits_len(&self.parents);
        let complete = |inner: i32, returned: i32| {
            if inner >= len {
                NO_MORE_DOCS
            } else {
                returned
            }
        };
        if min == max {
            return Ok(complete(max, max));
        }
        // `max` is exclusive for scoring, inclusive for `prevSetBit`.
        // Both one past the parent Java names (`lastParent + 1`,
        // `prevParent + 1`), so no `-1` is held.
        let after_last_parent = block_start(&self.parents, len.min(max).saturating_sub(1));
        let after_prev_parent = if min == 0 {
            0
        } else {
            block_start(&self.parents, min.saturating_sub(1))
        };
        if after_last_parent == after_prev_parent {
            return Ok(complete(max, max));
        }
        // The child collects into one concrete type, whatever `C` is -- a
        // child hit is a static call, a parent a dynamic one -- and a nested
        // block join's child collects into that same type again, so the
        // instantiations stop there.
        let mut forward = Forward(collector);
        let mut wrapped = BlockJoinCollector::<dyn ScoringCollector> {
            inner: &mut forward,
            parents: &self.parents,
            mode: self.mode,
            current_parent: -1,
            score: ParentScore::new(self.mode),
            error: None,
        };
        self.child.score(
            self.child_mode,
            live_docs,
            &mut wrapped,
            after_prev_parent,
            after_last_parent,
        )?;
        if let Some(e) = wrapped.error.take() {
            return Err(e);
        }
        wrapped.end_batch();
        Ok(complete(after_last_parent, max))
    }
}

/// A collector of any size behind a sized one, so it can be a
/// `dyn ScoringCollector`: every call passed through.
struct Forward<'c, C: ?Sized>(&'c mut C);

impl<C: ScoringCollector + ?Sized> ScoringCollector for Forward<'_, C> {
    fn collect(&mut self, doc: i32, score: f32) {
        self.0.collect(doc, score);
    }
    fn min_competitive_score(&self) -> Option<f32> {
        self.0.min_competitive_score()
    }
    fn score_mode(&self) -> crate::collector::ScoreMode {
        self.0.score_mode()
    }
    fn pruning_threshold(&self) -> Option<f32> {
        self.0.pruning_threshold()
    }
}

/// `BlockJoinBulkScorer.wrapCollector`'s `BatchAwareLeafCollector`.
struct BlockJoinCollector<'c, 'p, C: ?Sized> {
    inner: &'c mut C,
    parents: &'p FixedBitSet,
    mode: ScoreMode,
    current_parent: i32,
    score: ParentScore,
    /// The `IllegalStateException` `collect` would throw, raised once the
    /// child's bulk scorer returns (`collect` cannot fail here).
    error: Option<Error>,
}

impl<C: ScoringCollector + ?Sized> BlockJoinCollector<'_, '_, C> {
    /// `endBatch()`.
    fn end_batch(&mut self) {
        if self.current_parent >= 0 {
            let s = self.score.score();
            self.inner.collect(self.current_parent, s);
        }
    }
}

impl<C: ScoringCollector + ?Sized> ScoringCollector for BlockJoinCollector<'_, '_, C> {
    fn collect(&mut self, doc: i32, score: f32) {
        if self.error.is_some() {
            return;
        }
        if doc > self.current_parent {
            if self.current_parent >= 0 {
                let s = self.score.score();
                self.inner.collect(self.current_parent, s);
            }
            self.current_parent = next_set_bit(self.parents, doc);
            self.score.reset(score);
        } else if doc == self.current_parent {
            self.error = Some(child_matches_parent(doc));
        } else {
            self.score.add(score);
        }
    }
    // `setMinCompetitiveScore` reaches the child only for `None`/`Max`; any
    // other mode's child runs `COMPLETE` and is never offered a threshold.
    fn min_competitive_score(&self) -> Option<f32> {
        if self.mode == ScoreMode::Max {
            self.inner.min_competitive_score()
        } else {
            None
        }
    }
    fn score_mode(&self) -> crate::collector::ScoreMode {
        if self.mode == ScoreMode::Max {
            self.inner.score_mode()
        } else {
            crate::collector::ScoreMode::Complete
        }
    }
    fn pruning_threshold(&self) -> Option<f32> {
        if self.mode == ScoreMode::Max {
            self.inner.pruning_threshold()
        } else {
            None
        }
    }
}

/// `ToChildBlockJoinWeight.scorerSupplier(context)`.
pub(crate) fn to_child<'a>(
    ctx: &LeafContext<'a>,
    q: &ToChildBlockJoinQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let Some(parent) = build::build(ctx, &q.parent, boost, mode, false)? else {
        return Ok(None);
    };
    let Some(parents) = parent_bits(ctx, q.parents.as_ref())? else {
        return Ok(None);
    };
    Ok(Some(Box::new(ToChildScorer {
        parent,
        parents,
        do_scores: mode.needs_scores(),
        parent_score: 0.0,
        child_doc: -1,
        parent_doc: 0,
    })))
}

/// `ToChildBlockJoinQuery.ToChildBlockJoinScorer`.
pub(crate) struct ToChildScorer<'a> {
    /// Walked through its exact iterator (`parentScorer.iterator()`).
    parent: BoxScorer<'a>,
    parents: Arc<FixedBitSet>,
    do_scores: bool,
    parent_score: f32,
    child_doc: i32,
    parent_doc: i32,
}

impl ToChildScorer<'_> {
    /// `validateParentDoc()`.
    fn validate(&self) -> Result<()> {
        if self.parent_doc != NO_MORE_DOCS && !self.parents.get_doc(self.parent_doc) {
            return Err(Error::IllegalState(format!(
                "{INVALID_QUERY_MESSAGE}{}",
                self.parent_doc
            )));
        }
        Ok(())
    }

    /// `getParentDoc()`.
    pub(crate) fn parent_doc(&self) -> i32 {
        self.parent_doc
    }
}

impl Scorer for ToChildScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.child_doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        if self.child_doc.saturating_add(1) != self.parent_doc {
            debug_assert!(self.child_doc < self.parent_doc);
            self.child_doc = self.child_doc.saturating_add(1);
            return Ok(self.child_doc);
        }
        loop {
            self.parent_doc = exact_next(&mut *self.parent)?;
            self.validate()?;
            if self.parent_doc == 0 {
                // Degenerate but allowed: the first parent has no children.
                self.parent_doc = exact_next(&mut *self.parent)?;
                self.validate()?;
            }
            if self.parent_doc == NO_MORE_DOCS {
                self.child_doc = NO_MORE_DOCS;
                return Ok(self.child_doc);
            }
            self.child_doc = block_start(&self.parents, self.parent_doc.saturating_sub(1));
            if self.child_doc == self.parent_doc {
                // This parent has no children.
                continue;
            }
            if self.child_doc < self.parent_doc {
                if self.do_scores {
                    self.parent_score = self.parent.score()?;
                }
                return Ok(self.child_doc);
            }
        }
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let mut child_target = target;
        if child_target >= self.parent_doc {
            if child_target == NO_MORE_DOCS {
                self.child_doc = NO_MORE_DOCS;
                self.parent_doc = NO_MORE_DOCS;
                return Ok(self.child_doc);
            }
            self.parent_doc = exact_advance(&mut *self.parent, child_target.saturating_add(1))?;
            self.validate()?;
            if self.parent_doc == NO_MORE_DOCS {
                self.child_doc = NO_MORE_DOCS;
                return Ok(self.child_doc);
            }
            // The first parent that has children.
            loop {
                let first_child = block_start(&self.parents, self.parent_doc.saturating_sub(1));
                if first_child != self.parent_doc {
                    child_target = child_target.max(first_child);
                    break;
                }
                self.parent_doc = exact_next(&mut *self.parent)?;
                self.validate()?;
                if self.parent_doc == NO_MORE_DOCS {
                    self.child_doc = NO_MORE_DOCS;
                    return Ok(self.child_doc);
                }
            }
            if self.do_scores {
                self.parent_score = self.parent.score()?;
            }
        }
        self.child_doc = child_target;
        Ok(self.child_doc)
    }
    fn cost(&self) -> i64 {
        self.parent.cost()
    }
    fn score(&mut self) -> Result<f32> {
        Ok(self.parent_score)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(f32::INFINITY)
    }
}

/// `ParentChildrenBlockJoinQuery`'s weight's `scorerSupplier(context)`: only
/// the segment holding `parent_doc` has a scorer, over the children before it.
pub(crate) fn parent_children<'a>(
    ctx: &LeafContext<'a>,
    q: &ParentChildrenBlockJoinQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let Some(reader) = ctx.reader else {
        return Err(Error::MissingSegmentReader(
            "ParentChildrenBlockJoinQuery".into(),
        ));
    };
    // `ReaderUtil.subIndex(parentDocId, leaves)`: the leaf holding it.
    let local = q.parent_doc.saturating_sub(reader.doc_base);
    if q.parent_doc < reader.doc_base || local >= reader.max_doc {
        return Ok(None);
    }
    // A parent at doc 0 has no children before it.
    if local == 0 {
        return Ok(None);
    }
    // Java dereferences the producer's `null` here; no parents is no match.
    let Some(parents) = parent_bits(ctx, q.parents.as_ref())? else {
        return Ok(None);
    };
    let first_child = block_start(&parents, local.saturating_sub(1));
    if first_child == local {
        return Ok(None);
    }
    let Some(children) = build::build(ctx, &q.child, boost, mode, false)? else {
        return Ok(None);
    };
    Ok(Some(Box::new(ParentChildrenScorer {
        children,
        first_child,
        parent: local,
        doc: -1,
    })))
}

/// The anonymous scorer of `ParentChildrenBlockJoinQuery`: the child
/// scorer's exact iterator clipped to `[first_child, parent)`.
struct ParentChildrenScorer<'a> {
    children: BoxScorer<'a>,
    first_child: i32,
    parent: i32,
    doc: i32,
}

impl Scorer for ParentChildrenScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        let target = self.doc.saturating_add(1);
        self.advance(target)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let target = self.first_child.max(target);
        if target >= self.parent {
            self.doc = NO_MORE_DOCS;
            return Ok(self.doc);
        }
        let advanced = exact_advance(&mut *self.children, target)?;
        self.doc = if advanced >= self.parent {
            NO_MORE_DOCS
        } else {
            advanced
        };
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.children
            .cost()
            .min(i64::from(self.parent) - i64::from(self.first_child))
    }
    fn score(&mut self) -> Result<f32> {
        self.children.score()
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(f32::INFINITY)
    }
}

/// `ParentsChildrenBlockJoinWeight.scorerSupplier(context).get(leadCost)`.
pub(crate) fn parents_children<'a>(
    ctx: &LeafContext<'a>,
    q: &ParentsChildrenBlockJoinQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    Ok(parents_children_scorer(ctx, q, boost, mode)?.map(|s| -> BoxScorer<'a> { Box::new(s) }))
}

fn parents_children_scorer<'a>(
    ctx: &LeafContext<'a>,
    q: &ParentsChildrenBlockJoinQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<ParentsChildrenScorer<'a>>> {
    let Some(parents) = parent_bits(ctx, q.parents.as_ref())? else {
        return Ok(None);
    };
    let Some(parent) = build::build(ctx, &q.parent, boost, mode, false)? else {
        return Ok(None);
    };
    let Some(child) = build::build(ctx, &q.child, boost, mode, false)? else {
        return Ok(None);
    };
    Ok(Some(ParentsChildrenScorer {
        parents,
        parent,
        child,
        limit: q.child_limit_per_parent,
        do_scores: mode.needs_scores(),
        combiner: q.combiner.clone(),
        parent_score: 0.0,
        child_score: 0.0,
        parent_doc: 0,
        child_doc: -1,
        child_count: 0,
    }))
}

/// `ParentsChildrenBlockJoinQuery.ParentsChildrenBlockJoinScorer`.
struct ParentsChildrenScorer<'a> {
    parents: Arc<FixedBitSet>,
    /// Both walked through their exact iterators.
    parent: BoxScorer<'a>,
    child: BoxScorer<'a>,
    limit: i32,
    do_scores: bool,
    combiner: crate::join::ScoreCombiner,
    parent_score: f32,
    child_score: f32,
    parent_doc: i32,
    child_doc: i32,
    child_count: i32,
}

impl ParentsChildrenScorer<'_> {
    fn validate(&self) -> Result<()> {
        if self.parent_doc != NO_MORE_DOCS && !self.parents.get_doc(self.parent_doc) {
            return Err(Error::IllegalState(format!(
                "{INVALID_QUERY_MESSAGE}{}",
                self.parent_doc
            )));
        }
        Ok(())
    }

    /// `getParentDoc()` and `docID()`.
    fn docs(&self) -> (i32, i32) {
        (self.parent_doc, self.child_doc)
    }

    /// `exhausted()`: either iterator is done.
    fn exhausted(&self) -> bool {
        self.child.doc_id() == NO_MORE_DOCS || self.parent.doc_id() == NO_MORE_DOCS
    }

    /// `alignParentAndChildIterator()`.
    fn align(&mut self) -> Result<()> {
        while !self.exhausted() {
            let first_child = block_start(&self.parents, self.parent_doc.saturating_sub(1));
            if self.child_doc >= first_child && self.child_doc < self.parent_doc {
                break;
            } else if self.child_doc < first_child {
                self.child_doc = exact_advance(&mut *self.child, first_child)?;
            } else {
                self.parent_doc = if self.child_doc == self.parent_doc {
                    exact_next(&mut *self.parent)?
                } else {
                    exact_advance(&mut *self.parent, self.child_doc)?
                };
                self.validate()?;
            }
        }
        Ok(())
    }

    /// The tail both `nextDoc` and `advance` end with.
    fn land(&mut self) -> Result<i32> {
        if self.exhausted() {
            self.child_doc = NO_MORE_DOCS;
            self.parent_doc = NO_MORE_DOCS;
            return Ok(self.child_doc);
        }
        if self.do_scores {
            self.child_score = self.child.score()?;
            self.parent_score = self.parent.score()?;
        }
        self.child_count = self.child_count.saturating_add(1);
        Ok(self.child_doc)
    }
}

impl Scorer for ParentsChildrenScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.child_doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        if self.child_count < self.limit {
            self.child_doc = exact_next(&mut *self.child)?;
        }
        if self.child_count >= self.limit || self.child_doc >= self.parent_doc {
            self.child_count = 0;
            self.parent_doc = exact_next(&mut *self.parent)?;
            if self.parent_doc == 0 {
                // The first parent has no children.
                self.parent_doc = exact_next(&mut *self.parent)?;
            }
            self.validate()?;
        }
        self.align()?;
        self.land()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        if target <= self.child_doc {
            return Ok(self.child_doc);
        }
        self.child_doc = exact_advance(&mut *self.child, target)?;
        if self.child_count >= self.limit || self.child_doc >= self.parent_doc {
            self.child_count = 0;
            self.parent_doc = if self.child_doc <= self.parent_doc {
                exact_next(&mut *self.parent)?
            } else {
                exact_advance(&mut *self.parent, self.child_doc)?
            };
            self.validate()?;
            self.align()?;
        }
        self.land()
    }
    fn cost(&self) -> i64 {
        if self.limit == crate::join::DEFAULT_CHILD_LIMIT_PER_PARENT {
            self.child.cost()
        } else {
            self.child
                .cost()
                .min(self.parent.cost().saturating_mul(i64::from(self.limit)))
        }
    }
    fn score(&mut self) -> Result<f32> {
        if self.do_scores {
            return Ok(self.combiner.apply(self.parent_score, self.child_score));
        }
        Ok(1.0)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(f32::INFINITY)
    }
}

// ---------------------------------------------------------------------------
// explain
// ---------------------------------------------------------------------------

/// `childWeight.explain(context, doc)`: a sub-clause explained in the same
/// segment, with the same statistics.
fn explain_in(
    ctx: &LeafContext<'_>,
    clause: &Clause,
    doc: i32,
) -> Result<crate::explain::Explanation> {
    crate::explain::explain_clause_with_stats(
        ctx.fields,
        ctx.doc_in,
        ctx.pos_in,
        ctx.pay_in,
        ctx.live_docs,
        ctx.points,
        clause,
        doc,
        ctx.norms,
        ctx.global,
    )
}

/// `Weight.explain(context, doc)` of a block join (`doc` leaf-local,
/// `doc_base` the leaf's), or `None` for any other extended query.
pub(crate) fn explain(
    ctx: &LeafContext<'_>,
    q: &crate::extended_query::ExtendedQuery,
    doc: i32,
    doc_base: i32,
) -> Result<Option<crate::explain::Explanation>> {
    use crate::explain::Explanation;
    use crate::extended_query::ExtendedQuery;
    let not_a_match = || Explanation::no_match("Not a match");
    if let Some(e) = super::query_join::explain(ctx, q, doc)? {
        return Ok(Some(e));
    }
    if let Some(e) = super::function::explain(ctx, q, doc)? {
        return Ok(Some(e));
    }
    Ok(Some(match q {
        // `BlockJoinWeight.explain` -> `BlockJoinScorer.explain`.
        ExtendedQuery::ToParentBlockJoin(q) => {
            let (child_score_mode, child_mode) = child_weight(q.score_mode, Mode::Complete);
            let child_boost = if child_score_mode == ScoreMode::None {
                0.0
            } else {
                1.0
            };
            let clause = child_clause(q, child_score_mode);
            let Some(child) = build::build(ctx, &clause, child_boost, child_mode, false)? else {
                return Ok(Some(not_a_match()));
            };
            let Some(parents) = parent_bits(ctx, q.parents.as_ref())? else {
                return Ok(Some(not_a_match()));
            };
            let mut s = BlockJoinScorer::new(child, parents, child_score_mode);
            if exact_advance(&mut s, doc)? != doc {
                return Ok(Some(not_a_match()));
            }
            let (start, end) = s.current_range();
            let (mut best, mut worst): (Option<Explanation>, Option<Explanation>) = (None, None);
            let mut matches = 0;
            for child_doc in start..=end {
                let e = explain_in(ctx, &clause, child_doc)?;
                if e.matched {
                    matches += 1;
                    if best
                        .as_ref()
                        .is_none_or(|b| f64::from(e.value) > f64::from(b.value))
                    {
                        best = Some(e.clone());
                    }
                    if worst
                        .as_ref()
                        .is_none_or(|w| f64::from(e.value) < f64::from(w.value))
                    {
                        worst = Some(e);
                    }
                }
            }
            let sub = if child_score_mode == ScoreMode::Min {
                worst
            } else {
                best
            };
            Explanation::match_(
                s.score()?,
                format!(
                    "Score based on {matches} child docs in range from {} to {}, using score \
                     mode {}",
                    start.saturating_add(doc_base),
                    end.saturating_add(doc_base),
                    child_score_mode
                ),
            )
            .with_details(sub.into_iter().collect())
        }
        // `ToChildBlockJoinWeight.explain`.
        ExtendedQuery::ToChildBlockJoin(q) => {
            let Some(parent) = build::build(ctx, &q.parent, 1.0, Mode::Complete, false)? else {
                return Ok(Some(not_a_match()));
            };
            let Some(parents) = parent_bits(ctx, q.parents.as_ref())? else {
                return Ok(Some(not_a_match()));
            };
            let mut s = ToChildScorer {
                parent,
                parents,
                do_scores: true,
                parent_score: 0.0,
                child_doc: -1,
                parent_doc: 0,
            };
            if s.advance(doc)? != doc {
                return Ok(Some(not_a_match()));
            }
            let parent_doc = s.parent_doc();
            Explanation::match_(
                s.score()?,
                format!(
                    "Score based on parent document {}",
                    parent_doc.saturating_add(doc_base)
                ),
            )
            .with_details(vec![explain_in(ctx, &q.parent, parent_doc)?])
        }
        ExtendedQuery::ParentChildrenBlockJoin(_) => Explanation::no_match(
            "Not implemented, use ToParentBlockJoinQuery explain why a document matched",
        ),
        // `ParentsChildrenBlockJoinWeight.explain`.
        ExtendedQuery::ParentsChildrenBlockJoin(q) => {
            let Some(mut s) = parents_children_scorer(ctx, q, 1.0, Mode::Complete)? else {
                return Ok(Some(not_a_match()));
            };
            if s.advance(doc)? != doc {
                return Ok(Some(not_a_match()));
            }
            let (parent_doc, child_doc) = s.docs();
            Explanation::match_(
                s.score()?,
                format!(
                    "Score based on parent document {} and child document {} ",
                    parent_doc.saturating_add(doc_base),
                    child_doc.saturating_add(doc_base)
                ),
            )
            .with_details(vec![
                explain_in(ctx, &q.parent, parent_doc)?,
                explain_in(ctx, &q.child, child_doc)?,
            ])
        }
        _ => return Ok(None),
    }))
}

/// [`explain`] from `explain_clause_with_stats`' arguments, the segment's
/// size and base read from [`crate::explain::leaf`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn explain_extended(
    fields: &lucene_codecs::blocktree::BlockTreeFields,
    doc_in: Option<&lucene_codecs::postings::DocInput<'_>>,
    pos_in: Option<&lucene_codecs::postings::PosInput<'_>>,
    pay_in: Option<&lucene_codecs::postings::PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&crate::points_query::PointsInput<'_>>,
    norms: Option<&std::collections::HashMap<String, crate::FieldNorms<'_>>>,
    global: Option<&crate::GlobalStats>,
    q: &crate::extended_query::ExtendedQuery,
    doc: i32,
) -> Result<Option<crate::explain::Explanation>> {
    let leaf = crate::explain::leaf();
    let ctx = LeafContext {
        fields,
        doc_in,
        pos_in,
        pay_in,
        live_docs,
        points,
        norms,
        global,
        max_doc: leaf.map(|(max_doc, _)| max_doc),
        cache: None,
        reader: None,
        similarity: None,
    };
    explain(&ctx, q, doc, leaf.map_or(0, |(_, base)| base))
}

#[cfg(test)]
mod tests;
