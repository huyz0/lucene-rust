//! Per-segment scorers for the query-time joins ([`crate::join`]'s
//! `query_time`): Lucene 10.5.0's `TermsIncludingScoreQuery`
//! (`SVInOrderScorer`, `MVInOrderScorer`), `PointInSetIncludingScoreQuery`
//! (its `MergePointVisitor` and scorer), and `GlobalOrdinalsQuery` /
//! `GlobalOrdinalsWithScoreQuery` (`BaseGlobalOrdinalScorer` with its
//! `OrdinalMapScorer`/`SegmentOrdinalScorer`), each with its weight's
//! `explain`.
//!
//! As in Lucene the term and point scorers fill a bit set and a score per
//! document of the segment up front, deleted documents included; the
//! collector applies deletions. The global-ordinal scorers are two-phase over
//! the to-query's documents (its exact iterator, built without scores), each
//! confirmed by reading its join ordinal.

use std::sync::Arc;

use lucene_codecs::points::{IntersectVisitor, Relation};
use lucene_codecs::postings::PostingsFlags;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::build::{self, LeafContext};
use super::{exact_advance, exact_next, BoxScorer, Mode, Scorer, NO_MORE_DOCS};
use crate::explain::Explanation;
use crate::extended_query::ExtendedQuery;
use crate::join::query_time::{
    bit, global_ord, segment_map, CollectedOrdinals, GlobalOrdinalsQuery,
    GlobalOrdinalsWithScoreQuery, PointInSetIncludingScoreQuery, TermsIncludingScoreQuery,
};
use crate::reader::doc_values::SortedOrds;
use crate::reader::SortedDocValues;
use crate::{Error, Result};

/// The segment's `maxDoc`.
fn max_doc(ctx: &LeafContext<'_>, what: &str) -> Result<i32> {
    ctx.max_doc
        .or(ctx.reader.map(|r| r.max_doc))
        .ok_or_else(|| Error::MissingSegmentReader(what.to_string()))
}

/// A segment's matches as a bit set with a score per document, iterated in
/// order (`BitSetIterator` under the scorers of `TermsIncludingScoreQuery`
/// and `PointInSetIncludingScoreQuery`).
struct BitsScorer {
    bits: FixedBitSet,
    scores: Vec<f32>,
    boost: f32,
    doc: i32,
    cost: i64,
}

impl Scorer for BitsScorer {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        let from = self.doc.saturating_add(1);
        self.advance(from)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.doc = usize::try_from(target)
            .ok()
            .filter(|&t| t < self.bits.len())
            .and_then(|t| self.bits.next_set_bit(t))
            .and_then(|d| i32::try_from(d).ok())
            .unwrap_or(NO_MORE_DOCS);
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.cost
    }
    fn score(&mut self) -> Result<f32> {
        let s = usize::try_from(self.doc)
            .ok()
            .and_then(|d| self.scores.get(d).copied())
            .unwrap_or(0.0);
        Ok(s * self.boost)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(f32::INFINITY)
    }
    fn contains(&self, doc: i32) -> Option<bool> {
        // FBS: bounded by `bits.len()` here.
        Some(usize::try_from(doc).is_ok_and(|d| d < self.bits.len() && self.bits.get(d)))
    }
}

// ---------------------------------------------------------------------------
// TermsIncludingScoreQuery
// ---------------------------------------------------------------------------

/// `fillDocsAndScores`: every document of each join term the segment has,
/// in term order, scored its term's score -- the last term's (`SV`) or the
/// first's (`MV`).
fn terms_docs_and_scores(
    ctx: &LeafContext<'_>,
    q: &TermsIncludingScoreQuery,
    max_doc: i32,
) -> Result<Option<(FixedBitSet, Vec<f32>, i64)>> {
    let Some(ft) = ctx.fields.field(&q.to_field) else {
        return Ok(None);
    };
    let len = usize::try_from(max_doc).unwrap_or(0);
    let mut bits = FixedBitSet::new(len);
    let mut scores = vec![0.0f32; len];
    // `context.reader().maxDoc() * terms.size()`.
    let cost = i64::from(max_doc).saturating_mul(ft.num_terms);
    let Some(doc_in) = ctx.doc_in else {
        return Ok(Some((bits, scores, cost)));
    };
    // `postingsEnum = segmentTermsEnum.postings(postingsEnum, NONE)`: one
    // cursor, reset onto each term (a pulsed single-document term reads its
    // one document from the term metadata).
    let mut reuse = None;
    let mut single;
    for (term, score) in q.terms_and_scores() {
        let Some(seeked) = ft.seek_term_state(term)? else {
            continue;
        };
        let docs = if seeked.stats.doc_freq <= 1 {
            single = ft.lazy_postings_for(&seeked, doc_in, PostingsFlags::DocsOnly)?;
            &mut single
        } else {
            ft.reuse_postings_for(&seeked, doc_in, PostingsFlags::DocsOnly, &mut reuse)?
        };
        loop {
            let doc = docs.next_doc()?;
            if doc == NO_MORE_DOCS {
                break;
            }
            // FBS: a posting of this segment is below its `maxDoc`, the
            // set's length; the check keeps a corrupt one from panicking.
            let Some(d) = usize::try_from(doc).ok().filter(|&d| d < bits.len()) else {
                continue;
            };
            if q.multiple_values_per_document {
                if !bits.get(d) {
                    bits.set(d);
                    scores[d] = score;
                }
            } else {
                bits.set(d);
                scores[d] = score;
            }
        }
    }
    Ok(Some((bits, scores, cost)))
}

/// `TermsIncludingScoreQuery.createWeight(...).scorerSupplier(context)`.
pub(crate) fn terms_including_score<'a>(
    ctx: &LeafContext<'a>,
    q: &TermsIncludingScoreQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    if !mode.needs_scores() {
        // `searcher.rewrite(termsQuery).createWeight(COMPLETE_NO_SCORES, boost)`.
        return build::build(ctx, &q.as_terms_query(), boost, Mode::NoScores, top_level);
    }
    let max_doc = max_doc(ctx, "TermsIncludingScoreQuery")?;
    let Some((bits, scores, cost)) = terms_docs_and_scores(ctx, q, max_doc)? else {
        return Ok(None);
    };
    Ok(Some(Box::new(BitsScorer {
        bits,
        scores,
        boost,
        doc: -1,
        cost,
    })))
}

// ---------------------------------------------------------------------------
// PointInSetIncludingScoreQuery
// ---------------------------------------------------------------------------

/// `MergePointVisitor`: merges the query's sorted points with the points of
/// the cells the tree hands over, which come in ascending order for one
/// dimension.
struct MergePointVisitor<'q> {
    points: &'q [Vec<u8>],
    scores: &'q [f32],
    multiple_values_per_document: bool,
    /// `nextQueryPoint` (`points.len()` once exhausted) and `nextScore`.
    next: usize,
    result: FixedBitSet,
    doc_scores: Vec<f32>,
    /// `visit(int)` -- which Java answers with an `IllegalStateException`.
    misuse: bool,
}

impl MergePointVisitor<'_> {
    fn current(&self) -> Option<&[u8]> {
        self.points.get(self.next).map(Vec::as_slice)
    }

    fn next_score(&self) -> f32 {
        // `nextScore` keeps the last score once the scores run out, and is
        // `0` when there never was one.
        self.scores
            .get(self.next)
            .or(self.scores.last())
            .copied()
            .unwrap_or(0.0)
    }
}

impl IntersectVisitor for MergePointVisitor<'_> {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        while let Some(p) = self.current() {
            if p < min_packed {
                self.next = self.next.saturating_add(1);
                continue;
            }
            if p > max_packed {
                return Relation::CellOutsideQuery;
            }
            return Relation::CellCrossesQuery;
        }
        Relation::CellOutsideQuery
    }

    fn visit(&mut self, _doc_id: i32) {
        self.misuse = true;
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        while let Some(p) = self.current() {
            match p.cmp(packed_value) {
                std::cmp::Ordering::Equal => {
                    let score = self.next_score();
                    if let Ok(d) = usize::try_from(doc_id) {
                        // FBS: a point of this segment names one of its
                        // documents, below `maxDoc`, the set's length.
                        if d < self.result.len() {
                            if self.multiple_values_per_document {
                                if !self.result.get(d) {
                                    self.result.set(d);
                                    self.doc_scores[d] = score;
                                }
                            } else {
                                self.result.set(d);
                                self.doc_scores[d] = score;
                            }
                        }
                    }
                    break;
                }
                std::cmp::Ordering::Less => self.next = self.next.saturating_add(1),
                std::cmp::Ordering::Greater => break,
            }
        }
    }
}

/// `PointInSetIncludingScoreQuery`'s weight over one segment: the matching
/// documents with their scores, or `None` without the field's points.
fn point_docs_and_scores(
    ctx: &LeafContext<'_>,
    q: &PointInSetIncludingScoreQuery,
) -> Result<Option<(FixedBitSet, Vec<f32>)>> {
    let infos = match (ctx.points, ctx.reader) {
        (Some(p), _) => p.field_infos,
        (None, Some(r)) => r.field_infos(),
        (None, None) => return Err(Error::MissingPointsInput(q.field.clone())),
    };
    let Some(fi) = infos.field_by_name(&q.field) else {
        return Ok(None);
    };
    if fi.point_dimension_count != 1 {
        return Err(Error::IllegalArgument(format!(
            "field=\"{}\" was indexed with numDims={} but this query has numDims=1",
            q.field, fi.point_dimension_count
        )));
    }
    if usize::try_from(fi.point_num_bytes).ok() != Some(q.bytes_per_dim) {
        return Err(Error::IllegalArgument(format!(
            "field=\"{}\" was indexed with bytesPerDim={} but this query has bytesPerDim={}",
            q.field, fi.point_num_bytes, q.bytes_per_dim
        )));
    }
    let Some(points) = ctx.points else {
        return Err(Error::MissingPointsInput(q.field.clone()));
    };
    if points.reader.field(fi.number).is_none() {
        return Ok(None);
    }
    let len = usize::try_from(max_doc(ctx, "PointInSetIncludingScoreQuery")?).unwrap_or(0);
    let mut visitor = MergePointVisitor {
        points: &q.points,
        scores: &q.scores,
        multiple_values_per_document: q.multiple_values_per_document,
        next: 0,
        result: FixedBitSet::new(len),
        doc_scores: vec![0.0; len],
        misuse: false,
    };
    points.reader.intersect(fi.number, &mut visitor)?;
    if visitor.misuse {
        return Err(Error::IllegalState(
            "shouldn't get here, since CELL_INSIDE_QUERY isn't emitted".into(),
        ));
    }
    Ok(Some((visitor.result, visitor.doc_scores)))
}

/// `PointInSetIncludingScoreQuery.createWeight(...).scorerSupplier(context)`:
/// scored without the boost, as Java's scorer is.
pub(crate) fn point_in_set_including_score<'a>(
    ctx: &LeafContext<'a>,
    q: &PointInSetIncludingScoreQuery,
) -> Result<Option<BoxScorer<'a>>> {
    let Some((bits, scores)) = point_docs_and_scores(ctx, q)? else {
        return Ok(None);
    };
    Ok(Some(Box::new(BitsScorer {
        bits,
        scores,
        boost: 1.0,
        doc: -1,
        cost: 10,
    })))
}

// ---------------------------------------------------------------------------
// Global ordinals
// ---------------------------------------------------------------------------

/// What a global-ordinal scorer accepts and scores, by **segment** ordinal:
/// the collected global ordinals translated through the segment's map once,
/// when the scorer is made. Java maps every document's ordinal to its global
/// one (`segmentOrdToGlobalOrdLookup.get(ord)`) and asks the collector; the
/// matches and scores are the same (stage 3: the per-document lookups were
/// most of the scorer's time).
enum OrdinalMatch {
    /// `GlobalOrdinalsQuery`: the found ordinals, at the constant score.
    Found(FixedBitSet, f32),
    /// `GlobalOrdinalsWithScoreQuery`: the collector's `match`, its
    /// `score` per ordinal, and the boost it is multiplied by.
    Scored(FixedBitSet, Vec<f32>, f32),
    /// Java's per-document lookup, for a to-query that visits few
    /// documents of a large dictionary (where translating every ordinal
    /// would cost more than it saves): the segment's map and the collected
    /// global ordinals.
    PerDocument {
        map: Option<Arc<crate::ordinal_map::OrdinalMap>>,
        ord: usize,
        global: GlobalAccept,
    },
}

/// The collected global ordinals [`OrdinalMatch::PerDocument`] asks.
enum GlobalAccept {
    Found(Arc<FixedBitSet>, f32),
    Collected(Arc<CollectedOrdinals>, f32),
}

/// Whether to translate the segment's ordinals up front: unless the
/// to-query visits fewer than a quarter as many documents as the segment
/// has ordinals.
fn translate_up_front(approximation: &BoxScorer<'_>, values: &dyn SortedDocValues) -> bool {
    approximation.cost().saturating_mul(4) >= i64::from(values.value_count())
}

/// `BaseGlobalOrdinalScorer` with `OrdinalMapScorer`/`SegmentOrdinalScorer`:
/// two-phase over the to-query's documents.
struct GlobalOrdinalScorer<'a> {
    approximation: BoxScorer<'a>,
    /// The join field, its ordinals read straight off the column.
    values: SortedOrds<'a>,
    accept: OrdinalMatch,
    score: f32,
    /// A batch of the approximation's documents.
    buf: crate::bulk_scorer::DocScores,
}

impl GlobalOrdinalScorer<'_> {
    /// `matches()` for `doc`, the approximation's document: whether its
    /// join ordinal is accepted, the score set when it is.
    #[inline]
    fn matches_doc(&mut self, doc: i32) -> Result<bool> {
        let Some(seg_ord) = self.values.ord(doc)? else {
            return Ok(false);
        };
        let ord = i64::from(seg_ord);
        Ok(match &self.accept {
            OrdinalMatch::Found(found, score) => {
                self.score = *score;
                bit(found, ord)
            }
            OrdinalMatch::Scored(found, scores, boost) => {
                if bit(found, ord) {
                    let s = usize::try_from(ord)
                        .ok()
                        .and_then(|o| scores.get(o))
                        .copied()
                        .unwrap_or(0.0);
                    self.score = s * boost;
                    true
                } else {
                    false
                }
            }
            OrdinalMatch::PerDocument {
                map,
                ord: seg,
                global,
            } => {
                let map = segment_map(map.as_deref(), *seg)?;
                let g = global_ord(map, seg_ord)?;
                match global {
                    GlobalAccept::Found(found, score) => {
                        self.score = *score;
                        bit(found, g)
                    }
                    GlobalAccept::Collected(c, boost) => {
                        if c.matches(g) {
                            self.score = c.score(g) * boost;
                            true
                        } else {
                            false
                        }
                    }
                }
            }
        })
    }
}

/// The segment's ordinals that `accept` takes (and, with `score`, their
/// scores), through the segment's part of the ordinal map.
fn segment_ordinals(
    values: &dyn SortedDocValues,
    map: Option<&Arc<crate::ordinal_map::OrdinalMap>>,
    ord: usize,
    accept: impl Fn(i64) -> bool,
    mut score: Option<&mut Vec<f32>>,
    score_of: impl Fn(i64) -> f32,
) -> Result<FixedBitSet> {
    let map = segment_map(map.map(Arc::as_ref), ord)?;
    let count = usize::try_from(values.value_count()).unwrap_or(0);
    let mut bits = FixedBitSet::new(count);
    if let Some(s) = score.as_mut() {
        s.resize(count, 0.0);
    }
    for o in 0..count {
        let g = match map {
            Some(m) => m.get(o).copied().unwrap_or(-1),
            None => i64::try_from(o).unwrap_or(-1),
        };
        if accept(g) {
            // FBS: `o` is below `count`, the set's length.
            bits.set(o);
            if let Some(s) = score.as_mut() {
                s[o] = score_of(g);
            }
        }
    }
    Ok(bits)
}

impl Scorer for GlobalOrdinalScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.approximation.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        exact_next(self.approximation.as_mut())
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        exact_advance(self.approximation.as_mut(), target)
    }
    fn cost(&self) -> i64 {
        self.approximation.cost()
    }
    fn two_phase(&self) -> bool {
        true
    }
    fn matches(&mut self) -> Result<bool> {
        let doc = self.approximation.doc_id();
        self.matches_doc(doc)
    }
    fn match_cost(&self) -> f32 {
        100.0
    }
    fn score(&mut self) -> Result<f32> {
        Ok(self.score)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(f32::INFINITY)
    }
    /// The approximation's documents a batch at a time (its exact
    /// iterator's, as `next_doc` steps), each confirmed by its ordinal;
    /// then, as the document-at-a-time default does, on to the next match.
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&FixedBitSet>,
        out: &mut crate::bulk_scorer::DocScores,
    ) -> Result<()> {
        if !self.approximation.constant_scores() {
            // Its batches would score documents nobody asked it to.
            return super::docs_and_scores_one_by_one(self, up_to, live_docs, out);
        }
        out.docs.clear();
        out.scores.clear();
        let mut buf = std::mem::take(&mut self.buf);
        loop {
            self.approximation
                .next_docs_and_scores(up_to, live_docs, &mut buf)?;
            if buf.docs.is_empty() {
                break;
            }
            for &doc in &buf.docs {
                if self.matches_doc(doc)? {
                    out.docs.push(doc);
                    out.scores.push(self.score);
                }
            }
            if !out.docs.is_empty() {
                break;
            }
        }
        self.buf = buf;
        let mut doc = self.approximation.doc_id();
        while doc != NO_MORE_DOCS && !self.matches_doc(doc)? {
            doc = exact_next(self.approximation.as_mut())?;
        }
        Ok(())
    }
    fn prefers_batches(&self) -> bool {
        self.approximation.constant_scores()
    }
}

/// The segment's position among the searcher's the query was built over
/// (`context.ord`), for the ordinal map. A segment that was not among them
/// is Java's `IllegalStateException` -- with or without a map: a query built
/// over one segment (no map) and run over a refreshed reader would
/// otherwise read another segment's ordinals as that one's.
fn leaf_ord(ctx: &LeafContext<'_>, leaves: &crate::join::query_time::LeafOrds) -> Result<usize> {
    let reader = ctx
        .reader
        .ok_or_else(|| Error::MissingSegmentReader("GlobalOrdinalsQuery".into()))?;
    leaves.ord_of(reader).ok_or_else(|| {
        Error::IllegalState(
            "Creating the weight against a different index reader than this query has been \
             built for."
                .into(),
        )
    })
}

/// `GlobalOrdinalsQuery.W.scorerSupplier(context)`: a `ConstantScoreWeight`'s
/// `ConstantScoreScorer` over the two-phase iterator -- which, collecting
/// the top hits, empties itself once the minimum competitive score passes
/// its constant.
pub(crate) fn global_ordinals<'a>(
    ctx: &LeafContext<'a>,
    q: &GlobalOrdinalsQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let reader = ctx
        .reader
        .ok_or_else(|| Error::MissingSegmentReader("GlobalOrdinalsQuery".into()))?;
    let mut values = SortedOrds::open(reader, &q.join_field)?;
    let Some(approximation) = build::build(ctx, &q.to_query, 1.0, Mode::NoScores, false)? else {
        return Ok(None);
    };
    let ord = leaf_ord(ctx, &q.leaves)?;
    let accept = if translate_up_front(&approximation, values.dict()) {
        let found = segment_ordinals(
            values.dict(),
            q.ordinal_map.as_ref(),
            ord,
            |g| bit(&q.found_ords, g),
            None,
            |_| 0.0,
        )?;
        OrdinalMatch::Found(found, boost)
    } else {
        OrdinalMatch::PerDocument {
            map: q.ordinal_map.clone(),
            ord,
            global: GlobalAccept::Found(Arc::clone(&q.found_ords), boost),
        }
    };
    let inner: BoxScorer<'a> = Box::new(GlobalOrdinalScorer {
        approximation,
        values,
        accept,
        score: 0.0,
        buf: Default::default(),
    });
    Ok(Some(Box::new(super::leaf::ConstantScorer::new(
        inner,
        boost,
        mode == Mode::TopScores,
    ))))
}

/// `GlobalOrdinalsWithScoreQuery.createWeight(...).scorerSupplier(context)`.
pub(crate) fn global_ordinals_with_score<'a>(
    ctx: &LeafContext<'a>,
    q: &GlobalOrdinalsWithScoreQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let no_min_max = q.min <= 1 && q.max == i32::MAX;
    if !mode.needs_scores() && no_min_max {
        return global_ordinals(ctx, &q.as_global_ordinals_query(), boost, mode);
    }
    use crate::reader::LeafReader;
    let reader = ctx
        .reader
        .ok_or_else(|| Error::MissingSegmentReader("GlobalOrdinalsWithScoreQuery".into()))?;
    let Some(values) = reader.sorted_doc_values(&q.join_field)? else {
        return Ok(None);
    };
    let mut values = SortedOrds::with(values, reader, &q.join_field);
    let Some(approximation) = build::build(ctx, &q.to_query, 1.0, Mode::NoScores, false)? else {
        return Ok(None);
    };
    let ord = leaf_ord(ctx, &q.leaves)?;
    let c = &q.collected;
    let accept = if translate_up_front(&approximation, values.dict()) {
        let mut scores = Vec::new();
        let found = segment_ordinals(
            values.dict(),
            q.ordinal_map.as_ref(),
            ord,
            |g| c.matches(g),
            Some(&mut scores),
            |g| c.score(g),
        )?;
        OrdinalMatch::Scored(found, scores, boost)
    } else {
        OrdinalMatch::PerDocument {
            map: q.ordinal_map.clone(),
            ord,
            global: GlobalAccept::Collected(Arc::clone(c), boost),
        }
    };
    Ok(Some(Box::new(GlobalOrdinalScorer {
        approximation,
        values,
        accept,
        score: 0.0,
        buf: Default::default(),
    })))
}

// ---------------------------------------------------------------------------
// explain
// ---------------------------------------------------------------------------

/// The weights' `explain(context, doc)` for the query-time joins; `None`
/// for any other query.
pub(crate) fn explain(
    ctx: &LeafContext<'_>,
    q: &ExtendedQuery,
    doc: i32,
) -> Result<Option<Explanation>> {
    let not_a_match = || Explanation::no_match("Not a match");
    Ok(Some(match q {
        // `TermsIncludingScoreQuery`'s weight: the first join term (in term
        // order) whose postings hold the document, at the boost of `1`
        // (`IndexSearcher.explain` creates the weight unboosted).
        ExtendedQuery::TermsIncludingScore(q) => {
            let Some(ft) = ctx.fields.field(&q.to_field) else {
                return Ok(Some(not_a_match()));
            };
            let Some(doc_in) = ctx.doc_in else {
                return Ok(Some(not_a_match()));
            };
            for (term, score) in q.terms_and_scores() {
                let Some(seeked) = ft.seek_term_state(term)? else {
                    continue;
                };
                let mut docs = ft.lazy_postings_for(&seeked, doc_in, PostingsFlags::DocsOnly)?;
                if docs.advance(doc)? == doc {
                    return Ok(Some(Explanation::match_(
                        score,
                        format!(
                            "Score based on join value {}",
                            String::from_utf8_lossy(term)
                        ),
                    )));
                }
            }
            not_a_match()
        }
        ExtendedQuery::PointInSetIncludingScore(q) => {
            let Some(mut s) = point_in_set_including_score(ctx, q)? else {
                return Ok(Some(not_a_match()));
            };
            if s.advance(doc)? == doc {
                Explanation::match_(s.score()?, "A match")
            } else {
                not_a_match()
            }
        }
        ExtendedQuery::GlobalOrdinals(_) | ExtendedQuery::GlobalOrdinalsWithScore(_) => {
            if ctx.reader.is_none() {
                return Err(Error::MissingSegmentReader(
                    "explaining a global-ordinals join".into(),
                ));
            }
            return Ok(None);
        }
        _ => return Ok(None),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bit_set_scorer_walks_its_documents() {
        let mut bits = FixedBitSet::new(10);
        bits.set(2);
        bits.set(7);
        let mut scores = vec![0.0; 10];
        scores[2] = 1.5;
        scores[7] = 3.0;
        let mut s = BitsScorer {
            bits,
            scores,
            boost: 2.0,
            doc: -1,
            cost: 4,
        };
        assert_eq!(s.cost(), 4);
        assert_eq!(s.contains(7), Some(true));
        assert_eq!(s.contains(3), Some(false));
        assert_eq!(s.contains(-1), Some(false));
        assert_eq!(s.contains(99), Some(false));
        assert_eq!(s.next_doc().unwrap(), 2);
        assert_eq!(s.score().unwrap(), 3.0);
        assert_eq!(s.max_score(9).unwrap(), f32::INFINITY);
        assert_eq!(s.advance(3).unwrap(), 7);
        assert_eq!(s.score().unwrap(), 6.0);
        assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(s.score().unwrap(), 0.0);
        assert_eq!(s.advance(NO_MORE_DOCS).unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn the_merge_visitor_walks_query_points_forward() {
        let points = vec![vec![1u8], vec![3], vec![5]];
        let scores = vec![1.0, 3.0, 5.0];
        let mut v = MergePointVisitor {
            points: &points,
            scores: &scores,
            multiple_values_per_document: true,
            next: 0,
            result: FixedBitSet::new(4),
            doc_scores: vec![0.0; 4],
            misuse: false,
        };
        // A cell below every query point, then one past the first.
        assert_eq!(v.compare(&[0], &[0]), Relation::CellOutsideQuery);
        assert_eq!(v.compare(&[2], &[4]), Relation::CellCrossesQuery);
        assert_eq!(v.next, 1);
        v.visit_with_value(0, &[3]);
        v.visit_with_value(0, &[5]);
        assert_eq!(v.doc_scores[0], 3.0, "the first matching value wins");
        v.visit_with_value(1, &[4]);
        assert!(!v.result.get(1));
        v.visit_with_value(9, &[5]);
        v.visit_with_value(2, &[6]);
        assert!(v.current().is_none());
        assert_eq!(v.next_score(), 5.0);
        assert_eq!(v.compare(&[6], &[7]), Relation::CellOutsideQuery);
        v.visit(1);
        assert!(v.misuse);
        v.multiple_values_per_document = false;
        v.next = 1;
        v.visit_with_value(3, &[3]);
        v.next = 1;
        v.visit_with_value(3, &[5]);
        assert_eq!(v.doc_scores[3], 5.0, "the last matching value wins");
        let none: Vec<f32> = Vec::new();
        let v = MergePointVisitor {
            points: &points,
            scores: &none,
            multiple_values_per_document: false,
            next: 0,
            result: FixedBitSet::new(1),
            doc_scores: vec![0.0],
            misuse: false,
        };
        assert_eq!(v.next_score(), 0.0);
    }
}
