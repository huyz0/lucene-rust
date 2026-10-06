//! The function queries over one segment: `FunctionQuery`'s `AllScorer`,
//! `FunctionRangeQuery`'s `ValueSourceScorer`, `FunctionMatchQuery`'s
//! constant-scored two-phase iterator, `FunctionScoreQuery`'s
//! `FilterScorer`, and their weights' `explain`.
//!
//! A query's reader-wide state (its sources' `createWeight` contexts, its
//! values source's `rewrite(searcher)`) comes from the statistics pass
//! ([`crate::GlobalStats`]), which every entry point runs over all its
//! segments when the query holds a function query
//! (`crate::multi_segment::global_function_stats`); a segment reached without
//! it is an error, never one segment read as the whole index.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use super::leaf::ConstantScorer;
use super::{build, BoxScorer, LeafContext, Mode, Scorer, NEXT_DOCS_BATCH, NO_MORE_DOCS};
use crate::explain::Explanation;
use crate::extended_query::ExtendedQuery;
use crate::function::{
    BoxValues, FunctionContext, FunctionMatchQuery, FunctionQuery, FunctionRangeQuery,
    FunctionScoreQuery, ValueLeaf, ValueSource, ValueSourceScorer,
};
use crate::values_source::{BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext};
use crate::{Error, Result};

/// The segment's `maxDoc`.
fn max_doc(ctx: &LeafContext<'_>) -> Result<i32> {
    match (ctx.max_doc, ctx.reader) {
        (Some(m), _) => Ok(m),
        (None, Some(r)) => Ok(r.max_doc),
        (None, None) => Err(Error::MissingSegmentReader(
            "a function query needs the segment's maxDoc".into(),
        )),
    }
}

/// `source`'s `createWeight` context for this search, from the statistics
/// pass.
///
/// # Errors
/// [`Error::IllegalState`] when the entry point that built this leaf did not
/// prepare the query's function queries: Java creates the weight over the
/// whole searcher, and a context built from this segment alone would read
/// one segment's statistics as the index's.
fn context(ctx: &LeafContext<'_>, source: &Arc<dyn ValueSource>) -> Result<Arc<FunctionContext>> {
    ctx.global
        .and_then(|g| g.functions().context(source.as_ref()))
        .map(Arc::clone)
        .ok_or_else(|| unprepared(&source.description()))
}

/// `source.rewrite(searcher)` for this search, from the statistics pass.
///
/// # Errors
/// As [`context`].
fn rewritten(
    ctx: &LeafContext<'_>,
    source: &Arc<dyn DoubleValuesSource>,
) -> Result<Arc<dyn DoubleValuesSource>> {
    ctx.global
        .and_then(|g| g.functions().source(source.as_ref()))
        .map(Arc::clone)
        .ok_or_else(|| unprepared(&source.describe()))
}

/// A function query reached a leaf whose entry point did not prepare it
/// (`crate::multi_segment::global_function_stats`).
fn unprepared(what: &str) -> Error {
    Error::IllegalState(format!(
        "function query over {what} reached a segment without its reader-wide \
         preparation: the search must prepare its function queries over every segment"
    ))
}

// ---------------------------------------------------------------------------
// FunctionQuery
// ---------------------------------------------------------------------------

/// `FunctionQuery.AllScorer`: every document, scored `boost * floatVal`
/// (`0` for a negative or `NaN` value).
struct AllScorer<'a> {
    vals: BoxValues<'a>,
    boost: f32,
    doc: i32,
    max_doc: i32,
}

/// The documents a batch of an every-document scorer reads: the live ones
/// from `doc` up to the next multiple of 64 (below `end`), so that the
/// values are read as aligned windows (`NumericColumn::get_batch`) -- or
/// further, a window at a time, while none is live (a batch is empty only
/// at `end`). Returns where the walk stopped, the first document not taken.
fn live_run(
    doc: i32,
    end: i32,
    live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
    docs: &mut Vec<i32>,
) -> i32 {
    docs.clear();
    let mut d = doc;
    while d < end {
        // ARITH: `d < end <= max_doc`, an `i32`, so `(d | 63) + 1` is at
        // most `max_doc + 63`, which `min` brings back below `end`.
        #[allow(clippy::arithmetic_side_effects)]
        let stop = end.min((d | 63).saturating_add(1));
        match live_docs {
            None => docs.extend(d..stop),
            Some(l) => docs.extend((d..stop).filter(|&x| l.get_doc(x))),
        }
        d = stop;
        if !docs.is_empty() {
            break;
        }
    }
    d
}

impl Scorer for AllScorer<'_> {
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
        let val = self.vals.float_val(self.doc)?;
        // `val >= 0 == false` covers `NaN` too.
        Ok(if val >= 0.0 { self.boost * val } else { 0.0 })
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(f32::INFINITY)
    }
    /// The live documents from the current one on, their values read in
    /// one batch.
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
        out: &mut crate::bulk_scorer::DocScores,
    ) -> Result<()> {
        let end = up_to.min(self.max_doc);
        let next = live_run(self.doc, end, live_docs, &mut out.docs);
        out.scores.clear();
        out.scores.resize(out.docs.len(), 0.0);
        self.vals.float_val_batch(&out.docs, &mut out.scores)?;
        for s in &mut out.scores {
            *s = if *s >= 0.0 { self.boost * *s } else { 0.0 };
        }
        self.advance(next)?;
        Ok(())
    }
    fn prefers_batches(&self) -> bool {
        true
    }
}

/// `FunctionWeight.scorerSupplier(context).get(...)`.
pub(crate) fn function_query<'a>(
    ctx: &LeafContext<'a>,
    q: &FunctionQuery,
    boost: f32,
) -> Result<Option<BoxScorer<'a>>> {
    let fcx = context(ctx, &q.source)?;
    let vals = q.source.get_values(&fcx, &ValueLeaf::new(*ctx))?;
    Ok(Some(Box::new(AllScorer {
        vals,
        boost,
        doc: -1,
        max_doc: max_doc(ctx)?,
    })))
}

// ---------------------------------------------------------------------------
// FunctionRangeQuery
// ---------------------------------------------------------------------------

impl Scorer for ValueSourceScorer<'_> {
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
    fn two_phase(&self) -> bool {
        true
    }
    fn matches(&mut self) -> Result<bool> {
        let doc = self.doc;
        self.matches_doc(doc)
    }
    fn match_cost(&self) -> f32 {
        self.match_cost_of()
    }
    fn score(&mut self) -> Result<f32> {
        let doc = self.doc;
        self.score_doc(doc)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(f32::INFINITY)
    }
    /// Runs of live documents from the current one (a match) on, each
    /// matched and scored in one batch, until at least [`NEXT_DOCS_BATCH`]
    /// matches (fewer than twice that: a run is at most 64 documents); then,
    /// as the document-at-a-time default does, on to the next match.
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
        out: &mut crate::bulk_scorer::DocScores,
    ) -> Result<()> {
        out.docs.clear();
        out.scores.clear();
        let end = up_to.min(self.max_doc);
        let mut candidates = std::mem::take(&mut self.candidates);
        let mut doc = self.doc;
        while doc < end && out.docs.len() < NEXT_DOCS_BATCH {
            doc = live_run(doc, end, live_docs, &mut candidates);
            self.values.range_batch(
                self.range.as_ref(),
                &candidates,
                &mut out.docs,
                Some(&mut out.scores),
            )?;
        }
        self.candidates = candidates;
        // `score()`'s floor for negative infinity and `NaN`.
        for s in &mut out.scores {
            if s.is_nan() || *s == f32::NEG_INFINITY {
                *s = -f32::MAX;
            }
        }
        self.advance(doc)?;
        while self.doc != NO_MORE_DOCS && !self.matches()? {
            self.next_doc()?;
        }
        Ok(())
    }
    fn prefers_batches(&self) -> bool {
        true
    }
    fn batch_matches(&self) -> bool {
        true
    }
    /// The range over the batch at once (no value read beyond what
    /// `matches` reads).
    fn matches_batch(&mut self, docs: &[i32], keep: &mut [bool]) -> Result<()> {
        let mut matched = std::mem::take(&mut self.candidates);
        matched.clear();
        self.values
            .range_batch(self.range.as_ref(), docs, &mut matched, None)?;
        // `matched` is the documents of `docs` that match, in order.
        let mut j = 0;
        for (&doc, k) in docs.iter().zip(keep.iter_mut()) {
            *k = matched.get(j) == Some(&doc);
            j += usize::from(*k);
        }
        self.candidates = matched;
        if let Some(&last) = docs.last() {
            self.advance(last)?;
        }
        Ok(())
    }
}

/// `FunctionRangeWeight.scorerSupplier(context).get(...)`: the values'
/// range scorer (never boosted: the weight ignores its boost).
pub(crate) fn function_range<'a>(
    ctx: &LeafContext<'a>,
    q: &FunctionRangeQuery,
) -> Result<Option<BoxScorer<'a>>> {
    Ok(Some(Box::new(range_scorer(ctx, q)?)))
}

fn range_scorer<'a>(
    ctx: &LeafContext<'a>,
    q: &FunctionRangeQuery,
) -> Result<ValueSourceScorer<'a>> {
    let fcx = context(ctx, &q.source)?;
    let vals = q.source.get_values(&fcx, &ValueLeaf::new(*ctx))?;
    ValueSourceScorer::range(
        vals,
        max_doc(ctx)?,
        q.lower_val.as_deref(),
        q.upper_val.as_deref(),
        q.include_lower,
        q.include_upper,
    )
}

// ---------------------------------------------------------------------------
// FunctionMatchQuery
// ---------------------------------------------------------------------------

/// `FunctionMatchQuery`'s two-phase iterator: every document, confirmed
/// where the source has a value that passes the predicate.
struct MatchIterator<'a> {
    values: BoxDoubleValues<'a>,
    filter: Arc<crate::function::DoublePredicate>,
    match_cost: f32,
    doc: i32,
    max_doc: i32,
    /// A batch's values ([`Scorer::matches_batch`]), and the zero scores
    /// handed with them.
    batch_values: Vec<f64>,
    batch_scores: Vec<f32>,
}

impl Scorer for MatchIterator<'_> {
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
    fn two_phase(&self) -> bool {
        true
    }
    fn matches(&mut self) -> Result<bool> {
        Ok(self.values.advance_exact(self.doc)? && (self.filter)(self.values.double_value()?))
    }
    fn batch_matches(&self) -> bool {
        true
    }
    /// The values for the whole batch ([`DoubleValues::fill_batch`] when
    /// they can, else one document at a time), then the predicate.
    fn matches_batch(&mut self, docs: &[i32], keep: &mut [bool]) -> Result<()> {
        let n = docs.len();
        if self.values.batch_capable() {
            self.batch_values.resize(n, 0.0);
            // No scorer's scores: the source was opened without them.
            self.batch_scores.resize(n, 0.0);
            self.values.fill_batch(
                docs,
                &self.batch_scores,
                &mut self.batch_values,
                &mut keep[..n],
            )?;
            for (k, &v) in keep[..n].iter_mut().zip(&self.batch_values) {
                *k = *k && (self.filter)(v);
            }
        } else {
            for (&doc, k) in docs.iter().zip(keep.iter_mut()) {
                *k = self.values.advance_exact(doc)? && (self.filter)(self.values.double_value()?);
            }
        }
        if let Some(&last) = docs.last() {
            self.advance(last)?;
        }
        Ok(())
    }
    fn match_cost(&self) -> f32 {
        self.match_cost
    }
    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }
}

fn match_iterator<'a>(ctx: &LeafContext<'a>, q: &FunctionMatchQuery) -> Result<MatchIterator<'a>> {
    let vs = rewritten(ctx, &q.source)?;
    let values = vs.get_values(&ValuesContext::for_leaf(*ctx), 0, None)?;
    Ok(MatchIterator {
        values,
        filter: Arc::clone(&q.filter),
        match_cost: q.match_cost,
        doc: -1,
        max_doc: max_doc(ctx)?,
        batch_values: Vec::new(),
        batch_scores: Vec::new(),
    })
}

/// `FunctionMatchQuery`'s `ConstantScoreWeight`: the iterator,
/// constant-scored at the boost.
pub(crate) fn function_match<'a>(
    ctx: &LeafContext<'a>,
    q: &FunctionMatchQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let it: BoxScorer<'a> = Box::new(match_iterator(ctx, q)?);
    Ok(Some(Box::new(ConstantScorer::new(
        it,
        boost,
        mode == Mode::TopScores,
    ))))
}

// ---------------------------------------------------------------------------
// FunctionScoreQuery
// ---------------------------------------------------------------------------

/// `DoubleValuesSource.fromScorer(in)`: the wrapped scorer's score at the
/// document, set before the source's values are read.
struct ScoreCellValues(Rc<Cell<f32>>);

impl DoubleValues for ScoreCellValues {
    fn advance_exact(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(f64::from(self.0.get()))
    }
    /// A batch reads the scores it is handed, not the cell.
    fn batch_capable(&self) -> bool {
        true
    }
    fn fill_batch(
        &mut self,
        docs: &[i32],
        scores: &[f32],
        out: &mut [f64],
        has: &mut [bool],
    ) -> Result<()> {
        let n = docs.len();
        for (o, &s) in out[..n].iter_mut().zip(&scores[..n]) {
            *o = f64::from(s);
        }
        has[..n].fill(true);
        if let Some(&last) = scores[..n].last() {
            self.0.set(last);
        }
        Ok(())
    }
}

/// `FunctionScoreWeight`'s `FilterScorer`: the wrapped query's iteration,
/// scored `(float) (value * boost)` (`0` for a missing, negative or `NaN`
/// value).
struct FunctionScoreScorer<'a> {
    inner: BoxScorer<'a>,
    values: BoxDoubleValues<'a>,
    score: Rc<Cell<f32>>,
    needs_scores: bool,
    boost: f32,
    /// `values` can be read a batch at a time ([`DoubleValues::fill_batch`]).
    batch: bool,
    /// A batch's values and whether each document has one.
    batch_values: Vec<f64>,
    batch_has: Vec<bool>,
}

impl FunctionScoreScorer<'_> {
    /// `(float) (value * boost)` for a value, `0` without one or for a
    /// negative or `NaN` one.
    #[inline]
    fn combine(&self, has: bool, factor: f64) -> f32 {
        if has && factor >= 0.0 {
            (factor * f64::from(self.boost)) as f32
        } else {
            0.0
        }
    }
}

impl Scorer for FunctionScoreScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.inner.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.inner.advance(target)
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
        if self.needs_scores {
            self.score.set(self.inner.score()?);
        }
        let doc = self.inner.doc_id();
        if self.values.advance_exact(doc)? {
            let factor = self.values.double_value()?;
            return Ok(self.combine(true, factor));
        }
        Ok(0.0)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(f32::INFINITY)
    }
    /// The wrapped scorer's batch (its scores, when the source reads
    /// them), then the source's values for the whole batch.
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
        out: &mut crate::bulk_scorer::DocScores,
    ) -> Result<()> {
        if !self.batch {
            return super::docs_and_scores_one_by_one(self, up_to, live_docs, out);
        }
        self.inner.next_docs_and_scores(up_to, live_docs, out)?;
        let n = out.docs.len();
        self.batch_values.resize(n, 0.0);
        self.batch_has.resize(n, false);
        self.values.fill_batch(
            &out.docs,
            &out.scores,
            &mut self.batch_values,
            &mut self.batch_has,
        )?;
        for i in 0..n {
            out.scores[i] = self.combine(self.batch_has[i], self.batch_values[i]);
        }
        Ok(())
    }
    fn prefers_batches(&self) -> bool {
        self.batch
    }
}

/// The wrapped query's weight: `COMPLETE` when the source reads its scores,
/// `COMPLETE_NO_SCORES` otherwise, boost `1`.
fn inner_mode(source: &dyn DoubleValuesSource) -> Mode {
    if source.needs_scores() {
        Mode::Complete
    } else {
        Mode::NoScores
    }
}

/// The wrapped query's scorer, the source's values, the cell they read the
/// scorer's score from, and whether they read it.
type ScoreParts<'a> = (BoxScorer<'a>, BoxDoubleValues<'a>, Rc<Cell<f32>>, bool);

/// The scorer over the wrapped query, the source's values reading its
/// scores.
fn score_parts<'a>(
    ctx: &LeafContext<'a>,
    q: &FunctionScoreQuery,
    sub_boost: f32,
) -> Result<Option<ScoreParts<'a>>> {
    let source = rewritten(ctx, &q.source)?;
    let needs_scores = source.needs_scores();
    let Some(inner) = build::build(
        ctx,
        &q.in_query,
        sub_boost,
        inner_mode(source.as_ref()),
        false,
    )?
    else {
        return Ok(None);
    };
    let cell = Rc::new(Cell::new(0.0f32));
    let scores: BoxDoubleValues<'a> = Box::new(ScoreCellValues(Rc::clone(&cell)));
    let values = source.get_values(&ValuesContext::for_leaf(*ctx), 0, Some(scores))?;
    Ok(Some((inner, values, cell, needs_scores)))
}

/// `FunctionScoreQuery.createWeight(...)`'s scorer: without scores, the
/// wrapped query's own (`createWeight` returns its weight).
pub(crate) fn function_score<'a>(
    ctx: &LeafContext<'a>,
    q: &FunctionScoreQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    if !mode.needs_scores() {
        return build::build(ctx, &q.in_query, 1.0, Mode::NoScores, top_level);
    }
    // A source that takes the weight's boost onto the wrapped query
    // (OpenSearch's function score) scores its own value unboosted.
    let (sub_boost, boost) = if q.source.boosts_wrapped_query() {
        (boost, 1.0)
    } else {
        (1.0, boost)
    };
    let Some((inner, values, score, needs_scores)) = score_parts(ctx, q, sub_boost)? else {
        return Ok(None);
    };
    // Without its scores the wrapped scorer's batch must not compute any:
    // the document-at-a-time `score()` never asks for them.
    let batch =
        values.batch_capable() && (needs_scores || inner.constant_scores()) && super::batches_on();
    Ok(Some(Box::new(FunctionScoreScorer {
        inner,
        values,
        score,
        needs_scores,
        boost,
        batch,
        batch_values: Vec::new(),
        batch_has: Vec::new(),
    })))
}

// ---------------------------------------------------------------------------
// explain
// ---------------------------------------------------------------------------

/// The explanation of `clause` for `doc` over this segment.
fn explain_in(
    ctx: &LeafContext<'_>,
    clause: &crate::query::Clause,
    doc: i32,
) -> Result<Explanation> {
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

/// `Explanation`s of a value that is no score: `truncated score` for a
/// negative one, the `NaN` rule for a `NaN`.
fn illegal_score(value: f64, expl: Explanation) -> Option<Explanation> {
    if value < 0.0 {
        Some(
            Explanation::match_long(0, "truncated score, max of:")
                .with_details(vec![Explanation::match_(0.0, "minimum score"), expl]),
        )
    } else if value.is_nan() {
        Some(
            Explanation::match_long(
                0,
                "score, computed as (score == NaN ? 0 : score) since NaN is an illegal score \
                 from:",
            )
            .with_details(vec![expl]),
        )
    } else {
        None
    }
}

/// The function queries' weights' `explain(context, doc)` (the weight
/// unboosted, as `IndexSearcher.explain` creates it); `None` for any other
/// query.
pub(crate) fn explain(
    ctx: &LeafContext<'_>,
    q: &ExtendedQuery,
    doc: i32,
) -> Result<Option<Explanation>> {
    if !matches!(
        q,
        ExtendedQuery::Function(_)
            | ExtendedQuery::FunctionRange(_)
            | ExtendedQuery::FunctionMatch(_)
            | ExtendedQuery::FunctionScore(_)
    ) {
        return Ok(None);
    }
    crate::explain::with_leaf_reader(|reader, similarity| {
        let mut ctx = *ctx;
        if ctx.reader.is_none() {
            ctx.reader = reader;
        }
        if ctx.similarity.is_none() {
            ctx.similarity = similarity;
        }
        explain_function(&ctx, q, doc, 1.0).map(Some)
    })
}

/// `explain` of a function query under `BoostQuery`s (`createWeight(...,
/// boost)`: the boost is the weight's, which each function query's
/// explanation shows in its own way -- or ignores, as `FunctionRangeQuery`
/// does); `None` when `clause` is not a (boosted) function query.
#[allow(clippy::too_many_arguments)]
pub(crate) fn explain_boosted(
    fields: &lucene_codecs::blocktree::BlockTreeFields,
    doc_in: Option<&lucene_codecs::postings::DocInput<'_>>,
    pos_in: Option<&lucene_codecs::postings::PosInput<'_>>,
    pay_in: Option<&lucene_codecs::postings::PayInput<'_>>,
    live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
    points: Option<&crate::points_query::PointsInput<'_>>,
    norms: Option<&std::collections::HashMap<String, crate::FieldNorms<'_>>>,
    global: Option<&crate::GlobalStats>,
    clause: &crate::query::Clause,
    boost: f32,
    doc: i32,
) -> Result<Option<Explanation>> {
    use crate::query::Clause;
    let (mut inner, mut boost) = (clause, boost);
    while let Clause::Boost(b) = inner {
        boost *= b.boost;
        inner = &b.inner;
    }
    let Clause::Extended(q) = inner else {
        return Ok(None);
    };
    if !matches!(
        q.as_ref(),
        ExtendedQuery::Function(_)
            | ExtendedQuery::FunctionRange(_)
            | ExtendedQuery::FunctionMatch(_)
            | ExtendedQuery::FunctionScore(_)
    ) {
        return Ok(None);
    }
    let leaf = crate::explain::leaf();
    crate::explain::with_leaf_reader(|reader, similarity| {
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
            reader,
            similarity,
        };
        explain_function(&ctx, q, doc, boost).map(Some)
    })
}

fn explain_function(
    ctx: &LeafContext<'_>,
    q: &ExtendedQuery,
    doc: i32,
    boost: f32,
) -> Result<Explanation> {
    match q {
        // `AllScorer.explain(doc)`.
        ExtendedQuery::Function(q) => {
            let fcx = context(ctx, &q.source)?;
            let mut vals = q.source.get_values(&fcx, &ValueLeaf::new(*ctx))?;
            let raw = vals.explain(doc)?;
            let value = raw.value;
            let expl = illegal_score(f64::from(value), raw.clone()).unwrap_or(raw);
            Ok(Explanation::match_(
                boost * expl.value,
                format!("FunctionQuery({}), product of:", q.source.description()),
            )
            .with_details(vec![
                vals.explain(doc)?,
                Explanation::match_(boost, "boost"),
            ]))
        }
        // `FunctionRangeWeight.explain`.
        ExtendedQuery::FunctionRange(q) => {
            let fcx = context(ctx, &q.source)?;
            let mut function_values = q.source.get_values(&fcx, &ValueLeaf::new(*ctx))?;
            let mut scorer = range_scorer(ctx, q)?;
            let description = format!("{q:?}");
            if scorer.matches_doc(doc)? {
                super::exact_advance(&mut scorer, doc)?;
                let score = scorer.score()?;
                Ok(Explanation::match_(score, description)
                    .with_details(vec![function_values.explain(doc)?]))
            } else {
                Ok(Explanation::no_match(description)
                    .with_details(vec![function_values.explain(doc)?]))
            }
        }
        // `ConstantScoreWeight.explain`.
        ExtendedQuery::FunctionMatch(q) => {
            let mut it = match_iterator(ctx, q)?;
            let exists = it.advance(doc)? == doc && it.matches()?;
            let description = format!("{q:?}");
            Ok(if exists {
                let suffix = if boost == 1.0 {
                    String::new()
                } else {
                    format!("^{}", crate::function::java_float(boost))
                };
                Explanation::match_(boost, format!("{description}{suffix}"))
            } else {
                Explanation::no_match(format!("{description} doesn't match id {doc}"))
            })
        }
        // `FunctionScoreWeight.explain`.
        ExtendedQuery::FunctionScore(q) => {
            // As `function_score`: such a source's boost is its wrapped query's.
            let (sub_boost, boost) = if q.source.boosts_wrapped_query() {
                (boost, 1.0)
            } else {
                (1.0, boost)
            };
            let score_explanation = if sub_boost == 1.0 {
                explain_in(ctx, &q.in_query, doc)?
            } else {
                let boosted = crate::query::Clause::Boost(Box::new(crate::query::BoostQuery::new(
                    (*q.in_query).clone(),
                    sub_boost,
                )));
                explain_in(ctx, &boosted, doc)?
            };
            if !score_explanation.matched {
                return Ok(score_explanation);
            }
            let source = rewritten(ctx, &q.source)?;
            let Some((mut inner, mut values, cell, needs_scores)) = score_parts(ctx, q, sub_boost)?
            else {
                return Ok(score_explanation);
            };
            inner.advance(doc)?;
            if needs_scores {
                cell.set(inner.score()?);
            }
            let vctx = ValuesContext::for_leaf(*ctx);
            let mut value;
            let mut expl;
            if values.advance_exact(doc)? {
                value = values.double_value()?;
                expl = source.explain(&vctx, 0, doc, &score_explanation)?;
                if let Some(e) = illegal_score(value, expl.clone()) {
                    value = 0.0;
                    expl = e;
                }
            } else {
                value = 0.0;
                expl = source.explain(&vctx, 0, doc, &score_explanation)?;
            }
            let query = format!("{q:?}");
            Ok(if !expl.matched {
                Explanation::match_(
                    0.0,
                    format!(
                        "weight({query}) using default score of 0 because the function \
                         produced no value:"
                    ),
                )
                .with_details(vec![expl])
            } else if boost != 1.0 {
                Explanation::match_(
                    (value * f64::from(boost)) as f32,
                    format!("weight({query}), product of:"),
                )
                .with_details(vec![Explanation::match_(boost, "boost"), expl])
            } else {
                Explanation::match_value_of(&expl, format!("weight({query}), result of:"))
                    .with_details(vec![expl])
            })
        }
        _ => Err(Error::IllegalState(format!(
            "{} is not a function query",
            q.name()
        ))),
    }
}
