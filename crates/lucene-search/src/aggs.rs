//! OpenSearch's numeric metric aggregations -- `min`, `max`, `sum`, `avg`,
//! `value_count` and `stats` on a numeric field, at the top level -- over a
//! query's matching documents, as a shard computes them before the
//! coordinator reduces them.
//!
//! One pass over the query's live matches in document order, segment by
//! segment, keeps for each field what every one of those aggregators needs
//! (the aggregators themselves differ only in which parts they report; a
//! field asked for twice is read once). Within a segment each field is read
//! on its own -- a field's state depends on its column alone, in document
//! order, so reading the columns one after another adds the values in
//! Java's order -- straight down the column for a match-all, else over the
//! matches collected once:
//!
//! * the number of values (`ValueCountAggregator`: `docValueCount` per
//!   document);
//! * their sum, compensated exactly as OpenSearch's `CompensatedSum` does
//!   (`SumAggregator`, `AvgAggregator`, `StatsAggregator`), in the same order;
//! * the minimum and maximum over every value (`StatsAggregator`);
//! * the minimum of each document's smallest value and the maximum of each
//!   document's largest (`MinAggregator`/`MaxAggregator` read
//!   `MultiValueMode.MIN`/`MAX` of a document's values) -- not the same as
//!   the pair above when a value is `NaN`, which `SORTED_NUMERIC` keeps last.
//!
//! Values are read as `SortedNumericDoubleValues` reads them: a long field's
//! value converted to `double`, a `double` field's sortable long decoded, a
//! `float` field's sortable int decoded and widened. `Math.min`/`Math.max`
//! are Java's, including `NaN` and signed zeros.
//!
//! A top-level `min`/`max` under a bare match-all reads the field's points
//! instead ([`Source::PointsMin`]/[`Source::PointsMax`]): a segment's bound
//! comes from `MinAggregator.findLeafMinValue`/`MaxAggregator.findLeafMaxValue`
//! (ported below) and only a segment they cannot answer is read document by
//! document. Which requests qualify is the caller's to decide, as
//! `AggregatorBase.pointReaderIfAvailable` decides it.
//!
//! Under concurrent segment search every slice keeps its own states
//! ([`metric_states_sliced`], slices in parallel on rayon's pool); reducing
//! them is the caller's, as it is OpenSearch's `NonGlobalAggCollectorManager`'s.

use lucene_codecs::doc_values::{NumericReader, SortedNumericReader};
use lucene_codecs::points::{IntersectVisitor, Relation};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::directory_reader::SegmentReader;
use crate::exec::{self, Mode, NO_MORE_DOCS};
use crate::multi_segment::OpenSegment;
use crate::query::{BooleanQuery, Clause};
use crate::Result;

/// How a field's stored longs become the `double`s an aggregation reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    /// `long`, `integer`, `short`, `byte`, `date`: the value itself.
    Long,
    /// `double`: `NumericUtils.sortableLongToDouble`.
    Double,
    /// `float`: `NumericUtils.sortableIntToFloat`, widened.
    Float,
}

/// Where a field's per-document minimum or maximum comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Source {
    /// The matches' doc values, document by document.
    #[default]
    DocValues,
    /// `MinAggregator`'s points shortcut: each segment's smallest live
    /// point, into [`MetricState::min_of_mins`]; nothing else is kept.
    PointsMin,
    /// `MaxAggregator`'s: the largest, into [`MetricState::max_of_maxes`].
    PointsMax,
}

/// One field to aggregate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricSpec {
    pub field: String,
    pub kind: ValueKind,
    pub source: Source,
}

/// A field's state after the pass; see the module doc for what each part
/// feeds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricState {
    /// Values seen (`value_count`, and `avg`'s and `stats`'s count).
    pub count: u64,
    /// `CompensatedSum.value()`.
    pub sum: f64,
    /// `CompensatedSum.delta()`.
    pub delta: f64,
    /// The minimum and maximum over every value (`stats`).
    pub min: f64,
    pub max: f64,
    /// The minimum of each document's smallest value (`min`) and the maximum
    /// of each document's largest (`max`).
    pub min_of_mins: f64,
    pub max_of_maxes: f64,
}

impl Default for MetricState {
    fn default() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            delta: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            min_of_mins: f64::INFINITY,
            max_of_maxes: f64::NEG_INFINITY,
        }
    }
}

impl MetricState {
    /// `CompensatedSum.add(value)`.
    fn add(&mut self, value: f64) {
        // "If the value is Inf or NaN, just add it to the running tally."
        if !value.is_finite() {
            self.sum += value;
        }
        if self.sum.is_finite() {
            let corrected = value + self.delta;
            let updated = self.sum + corrected;
            self.delta = corrected - (updated - self.sum);
            self.sum = updated;
        }
    }

    /// A document with the one value `v`.
    #[inline]
    fn one(&mut self, v: f64) {
        self.count += 1;
        self.add(v);
        self.min = java_min(self.min, v);
        self.max = java_max(self.max, v);
        self.min_of_mins = java_min(self.min_of_mins, v);
        self.max_of_maxes = java_max(self.max_of_maxes, v);
    }

    /// A document's stored values, ascending, read as `kind`.
    #[inline]
    fn many(&mut self, kind: ValueKind, values: &[i64]) {
        let (Some(&first), Some(&last)) = (values.first(), values.last()) else {
            return;
        };
        self.count += values.len() as u64;
        for &v in values {
            let v = to_double(kind, v);
            self.add(v);
            self.min = java_min(self.min, v);
            self.max = java_max(self.max, v);
        }
        self.min_of_mins = java_min(self.min_of_mins, to_double(kind, first));
        self.max_of_maxes = java_max(self.max_of_maxes, to_double(kind, last));
    }
}

/// Java's `Math.min(double, double)`: `NaN` wins, and `-0.0 < 0.0`.
#[inline]
pub fn java_min(a: f64, b: f64) -> f64 {
    // The ordinary case first: one compare each way decides it.
    if a < b {
        return a;
    }
    if b < a {
        return b;
    }
    java_min_tie(a, b)
}

/// [`java_min`] when neither is below the other: equal (signed zeros), or a
/// `NaN`.
#[cold]
fn java_min_tie(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    if a == 0.0 && b.to_bits() == (-0.0f64).to_bits() {
        return b;
    }
    a
}

/// Java's `Math.max(double, double)`.
#[inline]
pub fn java_max(a: f64, b: f64) -> f64 {
    if a > b {
        return a;
    }
    if b > a {
        return b;
    }
    java_max_tie(a, b)
}

/// [`java_max`] when neither is above the other.
#[cold]
fn java_max_tie(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    if a == 0.0 && a.to_bits() == (-0.0f64).to_bits() {
        return b;
    }
    a
}

/// `NumericUtils.sortableLongToDouble`.
fn sortable_long_to_double(v: i64) -> f64 {
    f64::from_bits((v ^ ((v >> 63) & 0x7fff_ffff_ffff_ffff)) as u64)
}

/// `NumericUtils.sortableIntToFloat`.
fn sortable_int_to_float(v: i32) -> f32 {
    f32::from_bits((v ^ ((v >> 31) & 0x7fff_ffff)) as u32)
}

fn to_double(kind: ValueKind, v: i64) -> f64 {
    match kind {
        ValueKind::Long => v as f64,
        ValueKind::Double => sortable_long_to_double(v),
        ValueKind::Float => f64::from(sortable_int_to_float(v as i32)),
    }
}

/// `MinAggregator.MAX_BKD_LOOKUPS`: deleted points `findLeafMinValue` walks
/// past before it gives the segment up to document-by-document collection.
const MAX_BKD_LOOKUPS: u32 = 1024;

/// `findLeafMinValue`'s visitor: the first live point in the tree's order.
struct FirstLive<'a> {
    live: &'a FixedBitSet,
    found: Option<Vec<u8>>,
    lookups: u32,
    done: bool,
}

impl IntersectVisitor for FirstLive<'_> {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
        // Java stops the walk by throwing; here every cell after the stop is
        // pruned instead.
        if self.done {
            Relation::CellOutsideQuery
        } else {
            Relation::CellCrossesQuery
        }
    }

    fn visit(&mut self, _doc_id: i32) {}

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        if self.done {
            return;
        }
        if self.live.get_doc(doc_id) {
            self.found = Some(packed_value.to_vec());
            self.done = true;
            return;
        }
        self.lookups = self.lookups.saturating_add(1);
        if self.lookups > MAX_BKD_LOOKUPS {
            self.done = true;
        }
    }
}

/// `findLeafMaxValue`'s visitor: the last live point of the cells holding the
/// segment's maximum.
struct LastLiveOfMax<'a> {
    live: &'a FixedBitSet,
    max: &'a [u8],
    found: Option<Vec<u8>>,
}

impl IntersectVisitor for LastLiveOfMax<'_> {
    fn compare(&mut self, _min: &[u8], max: &[u8]) -> Relation {
        if max.get(..self.max.len()) == Some(self.max) {
            Relation::CellCrossesQuery
        } else {
            Relation::CellOutsideQuery
        }
    }

    fn visit(&mut self, _doc_id: i32) {}

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        if self.live.get_doc(doc_id) {
            let found = self.found.get_or_insert_with(Vec::new);
            found.clear();
            found.extend_from_slice(packed_value);
        }
    }
}

/// A point's value as the field's `pointReaderIfPossible` converter reads it
/// (`IntPoint`/`LongPoint`/`FloatPoint`/`DoublePoint.decodeDimension`),
/// widened to `double`; `None` for a width no numeric field has.
fn decode_point(kind: ValueKind, packed: &[u8]) -> Option<f64> {
    if let Ok(b) = <[u8; 8]>::try_from(packed) {
        let v = i64::from_be_bytes(b) ^ i64::MIN;
        return Some(match kind {
            ValueKind::Long => v as f64,
            ValueKind::Double | ValueKind::Float => sortable_long_to_double(v),
        });
    }
    let b = <[u8; 4]>::try_from(packed).ok()?;
    let v = i32::from_be_bytes(b) ^ i32::MIN;
    Some(match kind {
        ValueKind::Long => f64::from(v),
        ValueKind::Double | ValueKind::Float => f64::from(sortable_int_to_float(v)),
    })
}

/// `findLeafMinValue`/`findLeafMaxValue` for one segment: its smallest or
/// largest live point, or `None` when the points cannot tell (no points for
/// the field, or -- for the minimum -- no live point among the first
/// [`MAX_BKD_LOOKUPS`] deleted ones).
fn leaf_point_bound(seg: &OpenSegment<'_>, spec: &MetricSpec) -> Result<Option<f64>> {
    let Some(points) = seg.points else {
        return Ok(None);
    };
    let Some(number) = points.field_number(&spec.field) else {
        return Ok(None);
    };
    let Some(field) = points.reader.field(number) else {
        return Ok(None);
    };
    let width = usize::try_from(field.bytes_per_dim).unwrap_or(0);
    let packed = match (seg.live_docs, spec.source) {
        (_, Source::DocValues) => None,
        (None, Source::PointsMin) => Some(field.min_packed_value.clone()),
        (None, Source::PointsMax) => Some(field.max_packed_value.clone()),
        (Some(live), Source::PointsMin) => {
            let mut v = FirstLive {
                live,
                found: None,
                lookups: 0,
                done: false,
            };
            points.reader.intersect(number, &mut v)?;
            v.found
        }
        (Some(live), Source::PointsMax) => {
            let max = field.max_packed_value.get(..width).unwrap_or(&[]);
            let mut v = LastLiveOfMax {
                live,
                max,
                found: None,
            };
            points.reader.intersect(number, &mut v)?;
            v.found
        }
    };
    Ok(packed.and_then(|p| decode_point(spec.kind, p.get(..width)?)))
}

/// A segment's column for one field.
enum Values<'a> {
    Absent,
    Single(Box<NumericReader<'a>>),
    Multi(SortedNumericReader<'a>),
}

fn open_values<'a>(reader: &'a SegmentReader, field: &str) -> Result<Values<'a>> {
    let Some(info) = reader.field_infos().fields.iter().find(|i| i.name == field) else {
        return Ok(Values::Absent);
    };
    let Some((meta, data)) = reader.doc_values_for_field(info.number) else {
        return Ok(Values::Absent);
    };
    Ok(if let Some(e) = meta.sorted_numeric_entry(info.number) {
        if e.addresses.is_none() {
            Values::Single(Box::new(NumericReader::new(data, &e.numeric)))
        } else {
            Values::Multi(SortedNumericReader::new(data, e))
        }
    } else if let Some(e) = meta.numeric_entry(info.number) {
        Values::Single(Box::new(NumericReader::new(data, e)))
    } else {
        Values::Absent
    })
}

/// The metrics of `specs` over `query`'s live matches.
pub fn metric_states(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    specs: &[MetricSpec],
) -> Result<Vec<MetricState>> {
    let all: Vec<usize> = (0..segments.len().min(readers.len())).collect();
    let mut sliced = metric_states_sliced(segments, readers, query, specs, &[all])?;
    Ok(sliced.pop().unwrap_or_default())
}

/// [`metric_states`] as a concurrent segment search computes them: each
/// slice -- segment indices, in the order its collector visits them -- keeps
/// its own states from scratch, one `Vec` per slice. (OpenSearch then reduces
/// the slices' shard results, dropping each sum's delta; that is the
/// caller's.)
///
/// # Errors
/// A segment index out of range, or what [`metric_states`] reports.
pub fn metric_states_sliced(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    specs: &[MetricSpec],
    slices: &[Vec<usize>],
) -> Result<Vec<Vec<MetricState>>> {
    // A field asked for twice the same way (`sum` and `avg` of one field,
    // say) is read once: the state is the same.
    let mut unique: Vec<MetricSpec> = Vec::with_capacity(specs.len());
    let slot: Vec<usize> = specs
        .iter()
        .map(|s| {
            unique.iter().position(|u| u == s).unwrap_or_else(|| {
                unique.push(s.clone());
                unique.len() - 1
            })
        })
        .collect();
    let states = unique_states(segments, readers, query, &unique, slices)?;
    Ok(states
        .into_iter()
        .map(|per| slot.iter().map(|&i| per[i]).collect())
        .collect())
}

/// [`metric_states_sliced`] over distinct specs.
fn unique_states(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    specs: &[MetricSpec],
    slices: &[Vec<usize>],
) -> Result<Vec<Vec<MetricState>>> {
    let rewritten = crate::multi_segment::rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    let clause = lone_clause(query);
    // Slices are independent, as a concurrent search's are: they run on
    // rayon's pool, one task each, as Lucene hands each to its executor.
    if slices.len() > 1 {
        use rayon::prelude::*;
        return slices
            .par_iter()
            .map(|slice| slice_states(segments, readers, &clause, specs, slice))
            .collect();
    }
    slices
        .iter()
        .map(|slice| slice_states(segments, readers, &clause, specs, slice))
        .collect()
}

/// One slice's states: its segments, in order, from fresh states.
fn slice_states(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    clause: &Clause,
    specs: &[MetricSpec],
    slice: &[usize],
) -> Result<Vec<MetricState>> {
    let clause = clause.clone();
    let mut raw = Vec::new();
    let mut docs_buf = Vec::new();
    let mut precomputed = Vec::with_capacity(specs.len());
    let mut states = vec![MetricState::default(); specs.len()];
    for &i in slice {
        let (Some(seg), Some(reader)) = (segments.get(i), readers.get(i)) else {
            return Err(crate::Error::SliceOutOfRange {
                segment: i,
                segments: segments.len().min(readers.len()),
            });
        };
        let ctx = exec::LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: seg.pos_in,
            pay_in: seg.pay_in,
            live_docs: seg.live_docs,
            points: seg.points,
            norms: None,
            global: None,
            max_doc: seg.max_doc,
            cache: seg.cache,
        };
        // The points shortcut first (AggregatorBase.getLeafCollector asks
        // tryPrecomputeAggregationForLeaf before collecting): a field it
        // answers is not read document by document in this segment.
        let mut answered = 0;
        precomputed.clear();
        for (spec, state) in specs.iter().zip(&mut states) {
            let bound = match spec.source {
                Source::DocValues => None,
                _ => leaf_point_bound(seg, spec)?,
            };
            if let Some(v) = bound {
                if spec.source == Source::PointsMin {
                    state.min_of_mins = java_min(state.min_of_mins, v);
                } else {
                    state.max_of_maxes = java_max(state.max_of_maxes, v);
                }
                answered += 1;
            }
            precomputed.push(bound.is_some());
        }
        if answered == specs.len() {
            continue;
        }
        // The matches: every live document for a match-all (read straight
        // down each column below), else the scorer's, collected once.
        let live: Option<&FixedBitSet> = seg.live_docs;
        let docs = if matches_everything(&clause) {
            None
        } else {
            let Some(child) = exec::build::child(&ctx, &clause, 1.0, Mode::NoScores, true)? else {
                continue;
            };
            let mut scorer = child.into_scorer(Mode::NoScores);
            docs_buf.clear();
            let mut doc = exec::exact_next(&mut *scorer)?;
            while doc != NO_MORE_DOCS {
                if live.is_none_or(|l| l.get_doc(doc)) {
                    docs_buf.push(doc);
                }
                doc = exec::exact_next(&mut *scorer)?;
            }
            Some(&docs_buf[..])
        };
        let is_live = |doc: i32| live.is_none_or(|l| l.get_doc(doc));
        // Field by field: each state depends on its own column alone, read
        // in document order, so the order of the sums is Java's.
        for ((spec, state), &done) in specs.iter().zip(&mut states).zip(&precomputed) {
            if done {
                continue;
            }
            let kind = spec.kind;
            match (open_values(reader, &spec.field)?, docs) {
                (Values::Absent, _) => {}
                (Values::Single(mut r), None) => {
                    r.for_each_value(0, reader.max_doc, |doc, v| {
                        if is_live(doc) {
                            state.one(to_double(kind, v));
                        }
                    })?
                }
                (Values::Single(mut r), Some(docs)) => {
                    for &doc in docs {
                        if let Some(v) = r.value(doc)? {
                            state.one(to_double(kind, v));
                        }
                    }
                }
                (Values::Multi(mut r), None) => {
                    r.for_each_doc(0, reader.max_doc, |doc, vals| {
                        if is_live(doc) {
                            state.many(kind, vals);
                        }
                    })?
                }
                (Values::Multi(mut r), Some(docs)) => {
                    for &doc in docs {
                        r.values(doc, &mut raw)?;
                        state.many(kind, &raw);
                    }
                }
            }
        }
    }
    Ok(states)
}

/// Whether `clause` matches every document, whatever wraps the match-all
/// (a constant score, a boost, a boolean whose required clauses all match
/// everything): its matches are then every live document, in order, and the
/// columns can be read straight down.
fn matches_everything(clause: &Clause) -> bool {
    match clause {
        Clause::MatchAllDocs(_) => true,
        Clause::ConstantScore(c) => matches_everything(&c.inner),
        Clause::Boost(b) => matches_everything(&b.inner),
        Clause::Boolean(b) if b.must_not.is_empty() => {
            let required: Vec<&Clause> = b.must.iter().chain(&b.filter).collect();
            if required.is_empty() {
                b.minimum_should_match <= 1 && b.should.iter().any(matches_everything)
            } else {
                b.minimum_should_match == 0 && required.into_iter().all(matches_everything)
            }
        }
        _ => false,
    }
}

/// `BooleanQuery.rewrite`: a boolean of one clause is that clause.
fn lone_clause(query: &BooleanQuery) -> Clause {
    let clauses = query.must.len() + query.filter.len() + query.should.len() + query.must_not.len();
    match (&query.must[..], &query.filter[..], &query.should[..]) {
        ([only], _, _) if clauses == 1 && query.minimum_should_match == 0 => only.clone(),
        (_, [only], _) if clauses == 1 && query.minimum_should_match == 0 => only.clone(),
        (_, _, [only]) if clauses == 1 && query.minimum_should_match <= 1 => only.clone(),
        _ => Clause::Boolean(Box::new(query.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `NumericUtils.doubleToSortableLong`: how a `double` field stores `d`.
    fn enc(d: f64) -> i64 {
        let bits = d.to_bits() as i64;
        bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
    }

    /// A document's values, as a `double` column holds them.
    fn doc(s: &mut MetricState, values: &[f64]) {
        let stored: Vec<i64> = values.iter().map(|&d| enc(d)).collect();
        s.many(ValueKind::Double, &stored);
    }

    #[test]
    fn java_min_and_max_keep_nan_and_signed_zeros() {
        assert!(java_min(f64::NAN, 1.0).is_nan());
        assert!(java_min(1.0, f64::NAN).is_nan());
        assert!(java_max(f64::NAN, 1.0).is_nan());
        assert!(java_max(1.0, f64::NAN).is_nan());
        assert_eq!(java_min(0.0, -0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(java_min(-0.0, 0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(java_max(-0.0, 0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(java_max(0.0, -0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(java_min(2.0, 3.0), 2.0);
        assert_eq!(java_max(2.0, 3.0), 3.0);
    }

    #[test]
    fn the_sum_is_compensated_as_opensearch_compensates_it() {
        // 1 + 1e-16 ten times: a naive sum stays 1.0, Kahan's does not.
        let mut s = MetricState::default();
        doc(&mut s, &[1.0]);
        for _ in 0..10 {
            doc(&mut s, &[1e-16]);
        }
        assert!(s.sum > 1.0);
        assert_eq!(s.count, 11);
        // A non-finite value turns the tally non-finite, and it stays so.
        let mut s = MetricState::default();
        doc(&mut s, &[1.0, f64::INFINITY]);
        assert_eq!(s.sum, f64::INFINITY);
        doc(&mut s, &[5.0]);
        assert_eq!(s.sum, f64::INFINITY);
        let mut s = MetricState::default();
        doc(&mut s, &[f64::NAN]);
        assert!(s.sum.is_nan());
        // A single-valued document feeds every part the same way.
        let (mut one, mut many) = (MetricState::default(), MetricState::default());
        for v in [3.0, -0.0, 0.0, 1e300, f64::NEG_INFINITY] {
            one.one(v);
            doc(&mut many, &[v]);
        }
        assert_eq!(format!("{one:?}"), format!("{many:?}"));
    }

    #[test]
    fn a_match_all_is_seen_through_its_wrappers() {
        let all = || Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(0));
        let term = || Clause::Term(crate::TermQuery::new("f", b"x".to_vec()));
        assert!(matches_everything(&all()));
        assert!(matches_everything(&Clause::ConstantScore(Box::new(
            crate::query::ConstantScoreQuery::new(all(), 1.0)
        ))));
        assert!(matches_everything(&Clause::Boost(Box::new(
            crate::query::BoostQuery::new(all(), 2.0)
        ))));
        let mut b = BooleanQuery::new();
        b.filter.push(all());
        b.must.push(all());
        assert!(matches_everything(&Clause::Boolean(Box::new(b.clone()))));
        b.must_not.push(term());
        assert!(!matches_everything(&Clause::Boolean(Box::new(b))));
        let mut b = BooleanQuery::new();
        b.should.push(term());
        b.should.push(all());
        assert!(matches_everything(&Clause::Boolean(Box::new(b.clone()))));
        b.minimum_should_match = 2;
        assert!(!matches_everything(&Clause::Boolean(Box::new(b))));
        let mut b = BooleanQuery::new();
        b.must.push(all());
        b.filter.push(term());
        assert!(!matches_everything(&Clause::Boolean(Box::new(b))));
        assert!(!matches_everything(&term()));
    }

    #[test]
    fn a_documents_own_minimum_and_maximum_feed_min_and_max() {
        // `SORTED_NUMERIC` keeps NaN last: `min` reads the first value and
        // never sees it; `stats` reads every value and does.
        let mut s = MetricState::default();
        doc(&mut s, &[1.0, f64::NAN]);
        doc(&mut s, &[]);
        doc(&mut s, &[-2.0, 4.0]);
        assert_eq!(s.min_of_mins, -2.0);
        assert!(s.max_of_maxes.is_nan());
        assert!(s.min.is_nan());
        assert_eq!(s.count, 4);
    }

    #[test]
    fn stored_longs_become_the_doubles_an_aggregation_reads() {
        assert_eq!(to_double(ValueKind::Long, -7), -7.0);
        let d = -1.5f64;
        let bits = d.to_bits() as i64;
        let sortable = bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff);
        assert_eq!(to_double(ValueKind::Double, sortable), d);
        let f = 2.25f32;
        let ibits = f.to_bits() as i32;
        let sortable = ibits ^ ((ibits >> 31) & 0x7fff_ffff);
        assert_eq!(to_double(ValueKind::Float, i64::from(sortable)), 2.25);
    }
}
