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
mod tests {
    // Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::exec::leaf::DocList;

    fn bits(len: usize, set: &[usize]) -> Arc<FixedBitSet> {
        let mut b = FixedBitSet::new(len);
        for &i in set {
            b.set(i);
        }
        Arc::new(b)
    }

    fn list(docs: &[i32], scores: &[f32]) -> BoxScorer<'static> {
        Box::new(DocList::new(docs.to_vec(), scores.to_vec()))
    }

    #[test]
    fn block_starts_and_java_math() {
        let b = bits(8, &[2, 5]);
        assert_eq!(block_start(&b, -1), 0);
        assert_eq!(block_start(&b, 1), 0);
        assert_eq!(block_start(&b, 3), 3);
        assert_eq!(block_start(&b, 100), 6);
        assert_eq!(block_start(&FixedBitSet::new(0), 3), 0);
        assert_eq!(next_set_bit(&b, -4), NO_MORE_DOCS);
        assert_eq!(next_set_bit(&b, 6), NO_MORE_DOCS);
        assert!(java_min(f64::NAN, 1.0).is_nan() && java_max(1.0, f64::NAN).is_nan());
        assert!(java_min(0.0, -0.0).is_sign_negative() && java_min(-0.0, 0.0).is_sign_negative());
        assert!(java_max(-0.0, 0.0).is_sign_positive() && java_max(0.0, -0.0).is_sign_positive());
        assert_eq!((java_min(1.0, 2.0), java_max(1.0, 2.0)), (1.0, 2.0));
        let mut none = ParentScore::new(ScoreMode::None);
        none.reset(3.0);
        none.add(4.0);
        assert_eq!(none.score(), 0.0);
        let mut avg = ParentScore::new(ScoreMode::Avg);
        avg.reset(1.0);
        avg.add(2.0);
        assert_eq!(avg.score(), 1.5);
    }

    /// `BlockJoinScorer` over children 0, 1 (parent 2) and 3 (parent 5).
    #[test]
    fn the_block_join_scorer_folds_children_into_parents() {
        for (mode, want) in [
            (ScoreMode::Total, [3.0, 4.0]),
            (ScoreMode::Max, [2.0, 4.0]),
            (ScoreMode::Min, [1.0, 4.0]),
        ] {
            let mut s =
                BlockJoinScorer::new(list(&[0, 1, 3], &[1.0, 2.0, 4.0]), bits(6, &[2, 5]), mode);
            assert_eq!(s.cost(), 3);
            assert_eq!(s.match_cost(), 10.0);
            assert_eq!(s.next_doc().unwrap(), 2);
            assert_eq!(s.score().unwrap(), want[0]);
            assert_eq!(s.score().unwrap(), want[0], "read again, not recomputed");
            assert_eq!(s.current_range(), (0, 1));
            assert_eq!(s.max_score(5).unwrap(), f32::INFINITY);
            s.set_min_competitive_score(0.5).unwrap();
            assert_eq!(s.advance(3).unwrap(), 5);
            assert_eq!(s.score().unwrap(), want[1]);
            assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
        }
        let mut s = BlockJoinScorer::new(list(&[0], &[1.0]), bits(2, &[1]), ScoreMode::None);
        s.next_doc().unwrap();
        s.max_score(1).unwrap();
        s.set_min_competitive_score(1.0).unwrap();
        // A child on a parent is Lucene's `IllegalStateException`.
        let mut s = BlockJoinScorer::new(list(&[0, 2], &[1.0, 1.0]), bits(3, &[2]), ScoreMode::Avg);
        assert_eq!(s.next_doc().unwrap(), 2);
        assert!(matches!(s.score(), Err(Error::IllegalState(_))));
    }

    struct Keep(Vec<(i32, f32)>, Option<f32>);
    impl ScoringCollector for Keep {
        fn collect(&mut self, doc: i32, score: f32) {
            self.0.push((doc, score));
        }
        fn min_competitive_score(&self) -> Option<f32> {
            self.1
        }
        fn pruning_threshold(&self) -> Option<f32> {
            self.1
        }
    }

    #[test]
    fn the_bulk_scorer_scores_whole_blocks_per_window() {
        let bulk = |mode| BlockJoinBulk {
            child: super::super::Bulk::scorer(list(&[0, 1, 3, 6], &[1.0, 2.0, 4.0, 8.0])),
            child_mode: Mode::Complete,
            parents: bits(8, &[2, 5, 7]),
            mode,
        };
        let mut b = bulk(ScoreMode::Total);
        let mut c = Keep(Vec::new(), None);
        assert_eq!(b.score(None, &mut c, 4, 4).unwrap(), 4, "an empty window");
        assert_eq!(b.score(None, &mut c, 3, 5).unwrap(), 5, "no parent in it");
        assert_eq!(b.score(None, &mut c, 0, 4).unwrap(), 4);
        assert_eq!(
            b.score(None, &mut c, 4, NO_MORE_DOCS).unwrap(),
            NO_MORE_DOCS
        );
        assert_eq!(c.0, [(2, 3.0), (5, 4.0), (7, 8.0)]);
        // The threshold reaches the child only for `Max`.
        let mut c = Keep(Vec::new(), Some(2.0));
        let w = BlockJoinCollector {
            inner: &mut c,
            parents: &bits(3, &[2]),
            mode: ScoreMode::Max,
            current_parent: -1,
            score: ParentScore::new(ScoreMode::Max),
            error: None,
        };
        assert_eq!(
            (w.min_competitive_score(), w.pruning_threshold()),
            (Some(2.0), Some(2.0))
        );
        let _ = w.score_mode();
        let w = BlockJoinCollector {
            mode: ScoreMode::Avg,
            ..w
        };
        assert_eq!(
            (w.min_competitive_score(), w.pruning_threshold()),
            (None, None)
        );
        assert_eq!(w.score_mode(), crate::collector::ScoreMode::Complete);
        // A child on a parent fails the scoring once the child is done.
        let mut b = BlockJoinBulk {
            child: super::super::Bulk::scorer(list(&[0, 2], &[1.0, 1.0])),
            child_mode: Mode::Complete,
            parents: bits(3, &[2]),
            mode: ScoreMode::Avg,
        };
        let mut c = Keep(Vec::new(), None);
        assert!(b.score(None, &mut c, 0, NO_MORE_DOCS).is_err());
    }

    #[test]
    fn the_to_child_scorer_walks_each_parents_children() {
        let to_child = |parents: &[i32], set: Arc<FixedBitSet>| ToChildScorer {
            parent: list(parents, &vec![2.0; parents.len()]),
            parents: set,
            do_scores: true,
            parent_score: 0.0,
            child_doc: -1,
            parent_doc: 0,
        };
        // Parents 0 (no children), 3 (children 1, 2), 4 (none), 7 (5, 6).
        let set = bits(8, &[0, 3, 4, 7]);
        let mut s = to_child(&[0, 3, 4, 7], Arc::clone(&set));
        let mut got = Vec::new();
        while s.next_doc().unwrap() != NO_MORE_DOCS {
            got.push((s.doc_id(), s.score().unwrap()));
        }
        assert_eq!(got, [(1, 2.0), (2, 2.0), (5, 2.0), (6, 2.0)]);
        assert_eq!((s.cost(), s.max_score(1).unwrap()), (4, f32::INFINITY));
        let mut s = to_child(&[3, 4, 7], Arc::clone(&set));
        assert_eq!(s.advance(3).unwrap(), 5, "past a parent without children");
        assert_eq!(s.parent_doc(), 7);
        assert_eq!(s.advance(NO_MORE_DOCS).unwrap(), NO_MORE_DOCS);
        let mut s = to_child(&[3, 4], Arc::clone(&set));
        assert_eq!(s.advance(3).unwrap(), NO_MORE_DOCS);
        let mut s = to_child(&[3], Arc::clone(&set));
        assert_eq!(s.advance(3).unwrap(), NO_MORE_DOCS);
        // A parent query matching a child.
        let mut s = to_child(&[2], set);
        assert!(matches!(s.next_doc(), Err(Error::IllegalState(_))));
    }

    #[test]
    fn the_parent_children_scorer_clips_to_one_block() {
        let mut s = ParentChildrenScorer {
            children: list(&[0, 2, 3, 6], &[1.0, 2.0, 3.0, 4.0]),
            first_child: 2,
            parent: 5,
            doc: -1,
        };
        assert_eq!((s.cost(), s.max_score(9).unwrap()), (3, f32::INFINITY));
        assert_eq!(s.next_doc().unwrap(), 2);
        assert_eq!(s.score().unwrap(), 2.0);
        assert_eq!(s.next_doc().unwrap(), 3);
        assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn the_parents_children_scorer_limits_and_combines() {
        let scorer = |limit, parents: &[i32], children: &[i32]| ParentsChildrenScorer {
            parents: bits(9, &[3, 8]),
            parent: list(parents, &vec![1.0; parents.len()]),
            child: list(children, &vec![2.0; children.len()]),
            limit,
            do_scores: false,
            combiner: crate::join::ScoreCombiner::Sum,
            parent_score: 0.0,
            child_score: 0.0,
            parent_doc: 0,
            child_doc: -1,
            child_count: 0,
        };
        let mut s = scorer(1, &[3, 8], &[0, 1, 4, 5]);
        assert_eq!(s.cost(), 2);
        assert_eq!(s.next_doc().unwrap(), 0);
        assert_eq!(s.score().unwrap(), 1.0);
        assert_eq!(s.next_doc().unwrap(), 4, "one child a parent");
        assert_eq!(s.docs(), (8, 4));
        assert_eq!(s.max_score(9).unwrap(), f32::INFINITY);
        assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
        let mut s = scorer(
            crate::join::DEFAULT_CHILD_LIMIT_PER_PARENT,
            &[3, 8],
            &[0, 1, 4, 5],
        );
        assert_eq!(s.cost(), 4);
        assert_eq!(s.advance(1).unwrap(), 1);
        assert_eq!(s.advance(1).unwrap(), 1, "not past the current one");
        assert_eq!(s.advance(5).unwrap(), 5);
        // A parent query matching a child.
        let mut s = scorer(2, &[1], &[0]);
        assert!(matches!(s.next_doc(), Err(Error::IllegalState(_))));
    }
}
