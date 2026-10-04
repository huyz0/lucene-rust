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
    bit, global_ord, GlobalOrdinalsQuery, GlobalOrdinalsWithScoreQuery,
    PointInSetIncludingScoreQuery, TermsIncludingScoreQuery,
};
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
    for (term, score) in q.terms_and_scores() {
        let Some(seeked) = ft.seek_term_state(term)? else {
            continue;
        };
        let mut docs = ft.lazy_postings_for(&seeked, doc_in, PostingsFlags::DocsOnly)?;
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

/// What a global-ordinal scorer accepts and scores.
enum OrdinalMatch {
    /// `GlobalOrdinalsQuery`: the found ordinals, at the constant score.
    Found(Arc<FixedBitSet>, f32),
    /// `GlobalOrdinalsWithScoreQuery`: the collector's `match` and `score`.
    Collected(Arc<crate::join::query_time::CollectedOrdinals>, f32),
}

/// `BaseGlobalOrdinalScorer` with `OrdinalMapScorer`/`SegmentOrdinalScorer`:
/// two-phase over the to-query's documents.
struct GlobalOrdinalScorer<'a> {
    approximation: BoxScorer<'a>,
    values: Box<dyn SortedDocValues + 'a>,
    /// The segment's ordinals to global ones; `None` with a single segment.
    map: Option<Arc<crate::ordinal_map::OrdinalMap>>,
    ord: usize,
    accept: OrdinalMatch,
    score: f32,
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
        if !self.values.advance_exact(doc)? {
            return Ok(false);
        }
        let map = match &self.map {
            Some(m) => Some(m.segment_ords(self.ord).ok_or_else(|| {
                Error::IllegalArgument(format!("the ordinal map has no segment {}", self.ord))
            })?),
            None => None,
        };
        let g = global_ord(map, self.values.ord_value())?;
        Ok(match &self.accept {
            OrdinalMatch::Found(found, score) => {
                self.score = *score;
                bit(found, g)
            }
            OrdinalMatch::Collected(c, boost) => {
                if c.matches(g) {
                    self.score = c.score(g) * boost;
                    true
                } else {
                    false
                }
            }
        })
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
}

/// The segment's position among the searcher's (`context.ord`), for the
/// ordinal map.
fn leaf_ord(
    ctx: &LeafContext<'_>,
    map: Option<&Arc<crate::ordinal_map::OrdinalMap>>,
    leaves: &crate::join::query_time::LeafOrds,
) -> Result<usize> {
    if map.is_none() {
        return Ok(0);
    }
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

/// `GlobalOrdinalsQuery.W.scorerSupplier(context)`.
pub(crate) fn global_ordinals<'a>(
    ctx: &LeafContext<'a>,
    q: &GlobalOrdinalsQuery,
    boost: f32,
) -> Result<Option<BoxScorer<'a>>> {
    let reader = ctx
        .reader
        .ok_or_else(|| Error::MissingSegmentReader("GlobalOrdinalsQuery".into()))?;
    let values = crate::reader::doc_values::get_sorted(reader, &q.join_field)?;
    let Some(approximation) = build::build(ctx, &q.to_query, 1.0, Mode::NoScores, false)? else {
        return Ok(None);
    };
    let ord = leaf_ord(ctx, q.ordinal_map.as_ref(), &q.leaves)?;
    Ok(Some(Box::new(GlobalOrdinalScorer {
        approximation,
        values,
        map: q.ordinal_map.clone(),
        ord,
        accept: OrdinalMatch::Found(Arc::clone(&q.found_ords), boost),
        score: 0.0,
    })))
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
        return global_ordinals(ctx, &q.as_global_ordinals_query(), boost);
    }
    use crate::reader::LeafReader;
    let reader = ctx
        .reader
        .ok_or_else(|| Error::MissingSegmentReader("GlobalOrdinalsWithScoreQuery".into()))?;
    let Some(values) = reader.sorted_doc_values(&q.join_field)? else {
        return Ok(None);
    };
    let Some(approximation) = build::build(ctx, &q.to_query, 1.0, Mode::NoScores, false)? else {
        return Ok(None);
    };
    let ord = leaf_ord(ctx, q.ordinal_map.as_ref(), &q.leaves)?;
    Ok(Some(Box::new(GlobalOrdinalScorer {
        approximation,
        values,
        map: q.ordinal_map.clone(),
        ord,
        accept: OrdinalMatch::Collected(Arc::clone(&q.collected), boost),
        score: 0.0,
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
