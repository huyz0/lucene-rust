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
use lucene_util::numeric_utils::{sortable_int_to_float, sortable_long_to_double};

use lucene_search::directory_reader::SegmentReader;

use crate::terms_agg::{segment_counts, select_with, TermsResult, TermsScratch, TermsSpec};
use lucene_search::global_ords::GlobalOrds;
use lucene_search::java_math::{java_max, java_min, JAVA_NAN};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::segment_match::{
    column_read, lone_clause, plain_context, segment_match_read, segment_matches_scoring, Accept,
    ColumnRead, MinScore,
};
use lucene_search::Result;

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
    /// The parts of [`MetricState`] the caller reads ([`NEED_COUNT`] and the
    /// rest, or [`NEED_ALL`]); the others may be left at their defaults.
    pub needs: u8,
}

/// [`MetricState::count`] (`value_count`, `avg`, `stats`).
pub const NEED_COUNT: u8 = 1;
/// [`MetricState::sum`] and its delta (`sum`, `avg`, `stats`).
pub const NEED_SUM: u8 = 2;
/// [`MetricState::min`]/[`MetricState::max`] over every value (`stats`).
pub const NEED_MIN_MAX: u8 = 4;
/// [`MetricState::min_of_mins`] (`min`).
pub const NEED_MIN: u8 = 8;
/// [`MetricState::max_of_maxes`] (`max`).
pub const NEED_MAX: u8 = 16;
/// Every part.
pub const NEED_ALL: u8 = NEED_COUNT | NEED_SUM | NEED_MIN_MAX | NEED_MIN | NEED_MAX;

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
        compensated_add(&mut self.sum, &mut self.delta, value);
    }

    /// A document with the one value `v`, keeping the parts in `N`.
    #[inline]
    pub(crate) fn one<const N: u8>(&mut self, v: f64) {
        if N & NEED_COUNT != 0 {
            self.count += 1;
        }
        if N & NEED_SUM != 0 {
            self.add(v);
        }
        if N & NEED_MIN_MAX != 0 {
            self.min = java_min(self.min, v);
            self.max = java_max(self.max, v);
        }
        if N & NEED_MIN != 0 {
            self.min_of_mins = java_min(self.min_of_mins, v);
        }
        if N & NEED_MAX != 0 {
            self.max_of_maxes = java_max(self.max_of_maxes, v);
        }
    }

    /// A document's stored values, ascending, read as `kind`, keeping the
    /// parts in `N`.
    #[inline]
    pub(crate) fn many<const N: u8>(&mut self, kind: ValueKind, values: &[i64]) {
        let (Some(&first), Some(&last)) = (values.first(), values.last()) else {
            return;
        };
        if N & NEED_COUNT != 0 {
            self.count += values.len() as u64;
        }
        if N & (NEED_SUM | NEED_MIN_MAX) != 0 {
            for &v in values {
                let v = to_double(kind, v);
                if N & NEED_SUM != 0 {
                    self.add(v);
                }
                if N & NEED_MIN_MAX != 0 {
                    self.min = java_min(self.min, v);
                    self.max = java_max(self.max, v);
                }
            }
        }
        if N & NEED_MIN != 0 {
            self.min_of_mins = java_min(self.min_of_mins, to_double(kind, first));
        }
        if N & NEED_MAX != 0 {
            self.max_of_maxes = java_max(self.max_of_maxes, to_double(kind, last));
        }
    }
}

/// `CompensatedSum.add(value)` on its `value()` and `delta()`.
#[inline]
fn compensated_add(sum_ref: &mut f64, delta_ref: &mut f64, value: f64) {
    let (mut sum, mut delta) = (*sum_ref, *delta_ref);
    // "If the value is Inf or NaN, just add it to the running tally."
    if !value.is_finite() {
        // Java's `value = v + value`, in Java's operand order, and the
        // result canonicalised to Java's `Double.NaN`. The bits of a NaN
        // this addition makes are not Java's to promise nor Rust's: x86
        // gives `inf + -inf` the negative default NaN (`0xfff8...`),
        // aarch64 the positive one, and with two NaN operands keeps
        // whichever the optimiser put first -- so a release build
        // differed from a debug one. OpenSearch only ever sees the value
        // through `doubleToLongBits` (`StreamOutput.writeDouble`), which
        // is Java's `Double.NaN`; so is this.
        let tally = value + sum;
        sum = if tally.is_nan() { JAVA_NAN } else { tally };
    }
    if sum.is_finite() {
        let corrected = value + delta;
        let updated = sum + corrected;
        delta = corrected - (updated - sum);
        sum = updated;
    }
    *sum_ref = sum;
    *delta_ref = delta;
}

/// One owning bucket's metric in the form a bucketed state keeps it: only
/// the parts its node reads (`NEED_*`), read back as a [`MetricState`]
/// whose other parts are their defaults.
pub(crate) trait MetricSlot: Default + Clone {
    /// The parts kept.
    const NEEDS: u8;
    /// A document with the one value `v`.
    fn one(&mut self, v: f64);
    /// A document's stored values, ascending, read as `kind`.
    fn many(&mut self, kind: ValueKind, values: &[i64]);
    /// A bound read from the points, for a `min` (`is_min`) or a `max`.
    fn bound(&mut self, is_min: bool, v: f64);
    /// The state as the result reads it.
    fn state(&self) -> MetricState;
}

impl MetricSlot for MetricState {
    const NEEDS: u8 = NEED_ALL;
    #[inline]
    fn one(&mut self, v: f64) {
        MetricState::one::<NEED_ALL>(self, v);
    }
    #[inline]
    fn many(&mut self, kind: ValueKind, values: &[i64]) {
        MetricState::many::<NEED_ALL>(self, kind, values);
    }
    fn bound(&mut self, is_min: bool, v: f64) {
        if is_min {
            self.min_of_mins = java_min(self.min_of_mins, v);
        } else {
            self.max_of_maxes = java_max(self.max_of_maxes, v);
        }
    }
    fn state(&self) -> MetricState {
        *self
    }
}

/// `value_count`, `sum` and `avg`: the count and the compensated sum.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct CountSum {
    count: u64,
    sum: f64,
    delta: f64,
}

impl MetricSlot for CountSum {
    const NEEDS: u8 = NEED_COUNT | NEED_SUM;
    #[inline]
    fn one(&mut self, v: f64) {
        self.count += 1;
        compensated_add(&mut self.sum, &mut self.delta, v);
    }
    #[inline]
    fn many(&mut self, kind: ValueKind, values: &[i64]) {
        self.count += values.len() as u64;
        for &v in values {
            compensated_add(&mut self.sum, &mut self.delta, to_double(kind, v));
        }
    }
    fn bound(&mut self, _: bool, _: f64) {}
    fn state(&self) -> MetricState {
        MetricState {
            count: self.count,
            sum: self.sum,
            delta: self.delta,
            ..MetricState::default()
        }
    }
}

/// `value_count` alone.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Count(u64);

impl MetricSlot for Count {
    const NEEDS: u8 = NEED_COUNT;
    #[inline]
    fn one(&mut self, _: f64) {
        self.0 += 1;
    }
    #[inline]
    fn many(&mut self, _: ValueKind, values: &[i64]) {
        self.0 += values.len() as u64;
    }
    fn bound(&mut self, _: bool, _: f64) {}
    fn state(&self) -> MetricState {
        MetricState {
            count: self.0,
            ..MetricState::default()
        }
    }
}

/// `min`: the smallest of each document's smallest value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MinOf(f64);

impl Default for MinOf {
    fn default() -> Self {
        Self(f64::INFINITY)
    }
}

impl MetricSlot for MinOf {
    const NEEDS: u8 = NEED_MIN;
    #[inline]
    fn one(&mut self, v: f64) {
        self.0 = java_min(self.0, v);
    }
    #[inline]
    fn many(&mut self, kind: ValueKind, values: &[i64]) {
        if let Some(&first) = values.first() {
            self.0 = java_min(self.0, to_double(kind, first));
        }
    }
    fn bound(&mut self, is_min: bool, v: f64) {
        if is_min {
            self.0 = java_min(self.0, v);
        }
    }
    fn state(&self) -> MetricState {
        MetricState {
            min_of_mins: self.0,
            ..MetricState::default()
        }
    }
}

/// `max`: the largest of each document's largest value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MaxOf(f64);

impl Default for MaxOf {
    fn default() -> Self {
        Self(f64::NEG_INFINITY)
    }
}

impl MetricSlot for MaxOf {
    const NEEDS: u8 = NEED_MAX;
    #[inline]
    fn one(&mut self, v: f64) {
        self.0 = java_max(self.0, v);
    }
    #[inline]
    fn many(&mut self, kind: ValueKind, values: &[i64]) {
        if let Some(&last) = values.last() {
            self.0 = java_max(self.0, to_double(kind, last));
        }
    }
    fn bound(&mut self, is_min: bool, v: f64) {
        if !is_min {
            self.0 = java_max(self.0, v);
        }
    }
    fn state(&self) -> MetricState {
        MetricState {
            max_of_maxes: self.0,
            ..MetricState::default()
        }
    }
}

pub(crate) fn to_double(kind: ValueKind, v: i64) -> f64 {
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
pub(crate) fn leaf_point_bound(seg: &OpenSegment<'_>, spec: &MetricSpec) -> Result<Option<f64>> {
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
pub(crate) enum Values<'a> {
    Absent,
    Single(Box<NumericReader<'a>>),
    Multi(Box<SortedNumericReader<'a>>),
}

pub(crate) fn open_values<'a>(reader: &'a SegmentReader, field: &str) -> Result<Values<'a>> {
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
            Values::Multi(Box::new(SortedNumericReader::new(data, e)))
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
    let sliced = aggregate_sliced(segments, readers, query, specs, &[], &[], slices)?;
    Ok(sliced.into_iter().map(|(states, _)| states).collect())
}

/// A request's native aggregations in one pass per slice: the metrics of
/// `specs` and the `terms` of `terms` (each counted into its field's
/// [`GlobalOrds`], `globals` in the same order), each segment's matches collected once
/// and every column read over them -- per slice, as [`metric_states_sliced`]
/// and [`crate::terms_agg::terms_sliced`] give them, which it serves.
///
/// # Errors
/// What either reports.
pub fn aggregate_sliced(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    specs: &[MetricSpec],
    terms: &[TermsSpec],
    globals: &[std::sync::Arc<GlobalOrds>],
    slices: &[Vec<usize>],
) -> Result<Vec<(Vec<MetricState>, Vec<TermsResult>)>> {
    aggregate_sliced_min_score(
        segments, readers, query, specs, terms, globals, slices, None,
    )
}

/// [`aggregate_sliced`] behind `min_score` (`None`: every match). The
/// `MinimumScoreCollector` passes no weight on, so no aggregator answers a
/// segment from its postings' statistics; the points bound of a `min`/`max`
/// does not ask for one, and still answers.
///
/// # Errors
/// What [`aggregate_sliced`] reports.
#[allow(clippy::too_many_arguments)]
pub fn aggregate_sliced_min_score(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    specs: &[MetricSpec],
    terms: &[TermsSpec],
    globals: &[std::sync::Arc<GlobalOrds>],
    slices: &[Vec<usize>],
    min_score: Option<&MinScore<'_, '_>>,
) -> Result<Vec<(Vec<MetricState>, Vec<TermsResult>)>> {
    aggregate_sliced_counting(
        segments, readers, query, specs, terms, globals, slices, min_score,
    )
    .map(|(sliced, _)| sliced)
}

/// [`aggregate_sliced_min_score`], and beside it each segment's live match
/// count as the aggregations saw it (the documents passing `min_score`,
/// behind one) -- `None` for a segment whose aggregations were all answered
/// without visiting its matches (points bounds). What a `size: 0` search's
/// hit count would otherwise count again (Lucene counts in the same pass).
///
/// # Errors
/// What [`aggregate_sliced`] reports.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn aggregate_sliced_counting(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    specs: &[MetricSpec],
    terms: &[TermsSpec],
    globals: &[std::sync::Arc<GlobalOrds>],
    slices: &[Vec<usize>],
    min_score: Option<&MinScore<'_, '_>>,
) -> Result<(Vec<(Vec<MetricState>, Vec<TermsResult>)>, Vec<Option<u64>>)> {
    if globals.len() != terms.len() {
        return Err(lucene_search::Error::TermsAggType(format!(
            "{} terms aggregations with {} global ordinal maps",
            terms.len(),
            globals.len()
        )));
    }
    // A field asked for twice the same way (`sum` and `avg` of one field,
    // say) is read once: the state is the same.
    let mut unique: Vec<MetricSpec> = Vec::with_capacity(specs.len());
    let slot: Vec<usize> = specs
        .iter()
        .map(|s| {
            let same =
                |u: &MetricSpec| u.field == s.field && u.kind == s.kind && u.source == s.source;
            match unique.iter().position(same) {
                Some(i) => {
                    unique[i].needs |= s.needs;
                    i
                }
                None => {
                    unique.push(s.clone());
                    unique.len() - 1
                }
            }
        })
        .collect();
    let sliced = unique_states(
        segments, readers, query, &unique, terms, globals, slices, min_score,
    )?;
    let mut counts = vec![None; segments.len()];
    let mut out = Vec::with_capacity(sliced.len());
    for (per, t, seen) in sliced {
        for (i, c) in seen {
            if let Some(slot) = counts.get_mut(i) {
                *slot = Some(c);
            }
        }
        out.push((slot.iter().map(|&i| per[i]).collect(), t));
    }
    Ok((out, counts))
}

/// A slice's states, terms, and the live match count of each segment whose
/// matches it visited.
type SliceStates = (Vec<MetricState>, Vec<TermsResult>, Vec<(usize, u64)>);

/// [`aggregate_sliced`] over distinct specs.
#[allow(clippy::too_many_arguments)]
fn unique_states(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    specs: &[MetricSpec],
    terms: &[TermsSpec],
    globals: &[std::sync::Arc<GlobalOrds>],
    slices: &[Vec<usize>],
    min_score: Option<&MinScore<'_, '_>>,
) -> Result<Vec<SliceStates>> {
    let rewritten = lucene_search::multi_segment::rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    let clause = lone_clause(query);
    // Scoring, for `min_score`: the reader's statistics, once; without it,
    // the function queries' reader-wide preparation alone.
    let global =
        lucene_search::multi_segment::leaf_stats(segments, query, min_score.is_some(), None)?;
    let scoring = min_score.zip(global.as_ref());
    // Slices are independent, as a concurrent search's are: they run
    // concurrently ([`lucene_search::slices::run_slices`]) -- unless every
    // aggregation is a points bound (one read per segment), where handing
    // slices to other threads costs more than it saves.
    let one = |slice: &[usize]| {
        slice_states(
            segments,
            readers,
            query,
            &clause,
            specs,
            terms,
            globals,
            slice,
            scoring,
            global.as_ref(),
        )
    };
    let points_only = terms.is_empty() && specs.iter().all(|s| s.source != Source::DocValues);
    // Nor when every document matches and each terms aggregation reads its
    // counts off the postings' `docFreq`s: no document is visited at all.
    let stats_only = min_score.is_none()
        && matches!(clause, Clause::MatchAllDocs(_))
        && specs.iter().all(|s| s.source != Source::DocValues)
        && segments.iter().all(|s| {
            s.live_docs.is_none()
                && terms
                    .iter()
                    .all(|t| crate::terms_agg::precomputable(s.fields, &t.field))
        });
    if points_only || stats_only {
        return slices.iter().map(|s| one(s)).collect();
    }
    let parallel = lucene_search::slices::estimated_matches(segments, query)
        >= lucene_search::slices::SEQUENTIAL_BELOW;
    lucene_search::slices::run_slices_if(parallel, slices, one)
        .into_iter()
        .collect()
}

/// One slice's states and terms: its segments, in order, from scratch.
#[allow(clippy::too_many_arguments)]
fn slice_states(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    clause: &Clause,
    specs: &[MetricSpec],
    terms: &[TermsSpec],
    globals: &[std::sync::Arc<GlobalOrds>],
    slice: &[usize],
    scoring: Option<(&MinScore<'_, '_>, &lucene_search::GlobalStats)>,
    global: Option<&lucene_search::GlobalStats>,
) -> Result<SliceStates> {
    let mut seen = Vec::with_capacity(slice.len());
    // A count per global ordinal: as large as the field's cardinality, so
    // asked of the request's memory bound before it is allocated.
    lucene_util::memory::reserve(
        globals
            .iter()
            .map(|g| g.value_count().saturating_mul(8))
            .fold(0, usize::saturating_add),
    )?;
    let mut term_counts: Vec<Vec<u64>> = globals.iter().map(|g| vec![0; g.value_count()]).collect();
    let mut term_extra: Vec<std::collections::HashMap<Vec<u8>, u64>> =
        vec![std::collections::HashMap::new(); globals.len()];
    let mut terms_scratch = TermsScratch::default();
    let clause = clause.clone();
    let mut raw = Vec::new();
    let mut docs_buf = Vec::new();
    let mut words = Vec::new();
    let mut precomputed = Vec::with_capacity(specs.len());
    let mut states = vec![MetricState::default(); specs.len()];
    for &i in slice {
        let (Some(seg), Some(reader)) = (segments.get(i), readers.get(i)) else {
            return Err(lucene_search::Error::SliceOutOfRange {
                segment: i,
                segments: segments.len().min(readers.len()),
            });
        };
        let ctx = plain_context(seg).with_global(global);
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
        if answered == specs.len() && terms.is_empty() {
            continue;
        }
        // The matches: every live document for a match-all (read straight
        // down each column below), else the scorer's, collected once.
        let live: Option<&FixedBitSet> = seg.live_docs;
        let read = match scoring {
            Some((m, global)) => {
                let scored = ctx.with_scoring(m.norms.get(i).copied().flatten(), global);
                let Some(docs) =
                    segment_matches_scoring(&scored, query, live, m.min, &mut docs_buf)?
                else {
                    seen.push((i, 0));
                    continue;
                };
                let count = docs.map_or(0, <[i32]>::len);
                seen.push((i, u64::try_from(count).unwrap_or(u64::MAX)));
                column_read(docs, live, reader.max_doc, &mut words)
            }
            None => {
                let Some((read, count)) = segment_match_read(
                    &ctx,
                    query,
                    &clause,
                    live,
                    reader.max_doc,
                    &mut words,
                    &mut docs_buf,
                )?
                else {
                    seen.push((i, 0));
                    continue;
                };
                seen.push((i, count));
                read
            }
        };
        // Field by field: each state depends on its own column alone, read
        // in document order, so the order of the sums is Java's.
        for ((spec, state), &done) in specs.iter().zip(&mut states).zip(&precomputed) {
            if done {
                continue;
            }
            let values = open_values(reader, &spec.field)?;
            // The fold, compiled for the parts asked for: a `min` alone skips
            // the compensated sum. More parts than asked is never wrong.
            let (kind, max_doc) = (spec.kind, reader.max_doc);
            match spec.needs {
                NEED_MIN => fold::<NEED_MIN>(values, &read, state, kind, max_doc, &mut raw)?,
                NEED_MAX => fold::<NEED_MAX>(values, &read, state, kind, max_doc, &mut raw)?,
                NEED_COUNT => fold::<NEED_COUNT>(values, &read, state, kind, max_doc, &mut raw)?,
                n if n & !(NEED_COUNT | NEED_SUM) == 0 => fold::<{ NEED_COUNT | NEED_SUM }>(
                    values, &read, state, kind, max_doc, &mut raw,
                )?,
                _ => fold::<NEED_ALL>(values, &read, state, kind, max_doc, &mut raw)?,
            }
        }
        for (((t, g), counts), extra) in terms
            .iter()
            .zip(globals)
            .zip(&mut term_counts)
            .zip(&mut term_extra)
        {
            segment_counts(
                reader,
                seg.fields,
                i,
                &t.field,
                g,
                &read,
                counts,
                extra,
                &mut terms_scratch,
            )?;
        }
    }
    let terms = terms
        .iter()
        .zip(globals)
        .zip(&term_counts)
        .zip(&term_extra)
        .map(|(((t, g), counts), extra)| {
            select_with(counts, extra, t.shard_size, g, readers, &t.field)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((states, terms, seen))
}

/// Documents per [`fold_windows`] window: a multiple of 64, so each window's
/// presence words line up with the segment's bit sets, and one
/// [`lucene_codecs::adaptive_zones`] zone, so a whole accepted window can be
/// folded from its statistics ([`fold_zone`]).
const FOLD_WINDOW: usize = 1024;
const _: () = assert!(FOLD_WINDOW == lucene_codecs::adaptive_zones::ZONE_SIZE);

/// `2^53`: every integer up to this magnitude is exact as an `f64`, and so is
/// every sum of such integers that stays within it.
const EXACT_INTEGERS: u64 = 1 << 53;

/// Whether every one of documents `64 base..64 base + len` is accepted.
// ARITH: `len <= FOLD_WINDOW`, so `base + w` and the shift stay small.
#[allow(clippy::arithmetic_side_effects)]
fn all_accepted(accept: &Accept<'_>, base: usize, len: usize) -> bool {
    let full = len / 64;
    if !(0..full).all(|w| accept.word(base + w) == u64::MAX) {
        return false;
    }
    let tail = len % 64;
    tail == 0 || {
        let mask = (1u64 << tail) - 1;
        accept.word(base + full) & mask == mask
    }
}

/// Folds a whole window of `count` single-valued documents from its zone
/// statistics into `state`, when that gives exactly what folding its values
/// one by one would: the count; the extremes, unless one is a NaN (a stored
/// value's conversion to `double` is monotonic, and a window holding a NaN
/// holds it at one of its extremes); and, for `long`s, while the running sum
/// is an exact integer that stays within [`EXACT_INTEGERS`] whatever the
/// window's values, the compensated sum, whose every step is then exact with
/// a zero correction. `None` when the window must be read.
// ARITH: `bound` is at most `2^63 * 1024` in `u128`; `EXACT_INTEGERS -
// current` is taken only when `current <= EXACT_INTEGERS`.
#[allow(clippy::arithmetic_side_effects)]
fn fold_zone<const N: u8>(
    mut state: MetricState,
    stats: lucene_codecs::adaptive_zones::ZoneStats,
    count: usize,
    kind: ValueKind,
) -> Option<MetricState> {
    // A `float` column's sortable ints, sign-extended: anything else would
    // not convert as the values do.
    if kind == ValueKind::Float
        && (i32::try_from(stats.min).is_err() || i32::try_from(stats.max).is_err())
    {
        return None;
    }
    let (lo, hi) = (to_double(kind, stats.min), to_double(kind, stats.max));
    if N & (NEED_MIN_MAX | NEED_MIN | NEED_MAX) != 0 && (lo.is_nan() || hi.is_nan()) {
        return None;
    }
    if N & NEED_SUM != 0 {
        let sum = stats.sum.filter(|_| kind == ValueKind::Long)?;
        let exact = state.delta == 0.0 && state.sum.is_finite() && state.sum.fract() == 0.0;
        if !exact {
            return None;
        }
        let current = state.sum.abs() as u64;
        if current > EXACT_INTEGERS {
            return None;
        }
        let largest = stats.min.unsigned_abs().max(stats.max.unsigned_abs());
        let bound = u128::from(largest) * count as u128;
        if bound > u128::from(EXACT_INTEGERS - current) {
            return None;
        }
        state.sum += sum as f64;
    }
    if N & NEED_COUNT != 0 {
        state.count += count as u64;
    }
    if N & NEED_MIN_MAX != 0 {
        state.min = java_min(state.min, lo);
        state.max = java_max(state.max, hi);
    }
    if N & NEED_MIN != 0 {
        state.min_of_mins = java_min(state.min_of_mins, lo);
    }
    if N & NEED_MAX != 0 {
        state.max_of_maxes = java_max(state.max_of_maxes, hi);
    }
    Some(state)
}

/// A single-valued column streamed into `state` a window at a time
/// ([`NumericReader::fill_window`]: a sparse column's `IndexedDISI` walked
/// once per window rather than probed per document), each window's
/// documents with a value and `accept`ed taken a word at a time, in
/// ascending order -- the order the sum's compensation depends on. A `min`
/// or `max` of longs is folded as integers and handed over once per window:
/// `long -> double` is monotonic, so the extreme of the converted values is
/// the converted extreme.
fn fold_windows<const N: u8>(
    r: &mut NumericReader<'_>,
    accept: &Accept<'_>,
    state: &mut MetricState,
    kind: ValueKind,
    max_doc: i32,
    values: &mut Vec<i64>,
) -> Result<()> {
    values.resize(FOLD_WINDOW, 0);
    let mut present = [0u64; FOLD_WINDOW / 64];
    let mut start = 0i32;
    let zoned = r.has_window_stats();
    while start < max_doc {
        // ARITH: `start < max_doc`, so the difference is positive; `start`
        // stays a multiple of `FOLD_WINDOW` below `max_doc + FOLD_WINDOW`,
        // and `base + w` indexes the segment's words.
        #[allow(clippy::arithmetic_side_effects)]
        {
            let len = ((max_doc - start) as usize).min(FOLD_WINDOW);
            let base = start as usize >> 6;
            if zoned && all_accepted(accept, base, len) {
                if let Some((stats, count)) = r.window_stats(start)? {
                    if let Some(next) = (count == len)
                        .then(|| fold_zone::<N>(*state, stats, len, kind))
                        .flatten()
                    {
                        *state = next;
                        start += len as i32;
                        continue;
                    }
                }
            }
            r.fill_window(start, &mut values[..len], &mut present)?;
            fold_window_values::<N>(&values[..len], &present, accept, base, state, kind);
            start += len as i32;
        }
    }
    Ok(())
}

/// One window's `values` (documents `64 base..`) that are `present` and
/// `accept`ed into `state`, in ascending order. Kept out of line: inlined
/// into [`fold_windows`], the compensated sum's running value went through
/// the stack on every add.
// ARITH: `w < 16` and `bits != 0`, so `w * 64 + tz < values.len()` and
// `bits - 1` does not wrap.
#[allow(clippy::arithmetic_side_effects)]
#[inline(never)]
fn fold_window_values<const N: u8>(
    values: &[i64],
    present: &[u64],
    accept: &Accept<'_>,
    base: usize,
    state: &mut MetricState,
    kind: ValueKind,
) {
    let integer_extreme = kind == ValueKind::Long && (N == NEED_MIN || N == NEED_MAX);
    let mut best = if N == NEED_MIN { i64::MAX } else { i64::MIN };
    let mut any = false;
    for (w, &have) in present[..values.len().div_ceil(64)].iter().enumerate() {
        let mut bits = have & accept.word(base + w);
        while bits != 0 {
            let v = values[w * 64 + bits.trailing_zeros() as usize];
            bits &= bits - 1;
            if integer_extreme {
                any = true;
                best = if N == NEED_MIN {
                    best.min(v)
                } else {
                    best.max(v)
                };
            } else {
                state.one::<N>(to_double(kind, v));
            }
        }
    }
    if any {
        state.one::<N>(best as f64);
    }
}

/// A single-valued column at the matches `docs` (ascending, distinct) into
/// `state`, one value at a time, except that a run of matches covering a
/// whole zone is folded from its statistics
/// ([`fold_zone`]).
// ARITH: `i < docs.len()`; `doc + len - 1` is checked against `max_doc`
// before use, and `len <= FOLD_WINDOW`.
#[allow(clippy::arithmetic_side_effects)]
fn fold_seek<const N: u8>(
    r: &mut NumericReader<'_>,
    docs: &[i32],
    state: &mut MetricState,
    kind: ValueKind,
    max_doc: i32,
) -> Result<()> {
    let mut i = 0;
    let zoned = r.has_window_stats();
    while let Some(&doc) = docs.get(i) {
        if zoned && (doc as usize).is_multiple_of(FOLD_WINDOW) && doc < max_doc {
            let len = ((max_doc - doc) as usize).min(FOLD_WINDOW);
            // Distinct ascending matches: the run is whole when its last is
            // `len - 1` past its first.
            let whole = docs.get(i + len - 1) == Some(&(doc + len as i32 - 1));
            if whole {
                if let Some((stats, count)) = r.window_stats(doc)? {
                    if let Some(next) = (count == len)
                        .then(|| fold_zone::<N>(*state, stats, len, kind))
                        .flatten()
                    {
                        *state = next;
                        i += len;
                        continue;
                    }
                }
            }
        }
        if let Some(v) = r.value(doc)? {
            state.one::<N>(to_double(kind, v));
        }
        i += 1;
    }
    Ok(())
}

/// One field's column over a segment's matches into `state`, keeping `N`.
fn fold<const N: u8>(
    values: Values<'_>,
    read: &ColumnRead<'_>,
    state: &mut MetricState,
    kind: ValueKind,
    max_doc: i32,
    raw: &mut Vec<i64>,
) -> Result<()> {
    match (values, read) {
        (Values::Absent, _) => {}
        (Values::Single(mut r), ColumnRead::Stream(accept)) => {
            fold_windows::<N>(&mut r, accept, state, kind, max_doc, raw)?
        }
        (Values::Single(mut r), ColumnRead::Seek(docs)) => {
            fold_seek::<N>(&mut r, docs, state, kind, max_doc)?
        }
        (Values::Multi(mut r), ColumnRead::Stream(accept)) => r.for_each_accepted(
            0,
            max_doc,
            |doc| accept.test(doc),
            |_, vals| {
                state.many::<N>(kind, vals);
            },
        )?,
        (Values::Multi(mut r), ColumnRead::Seek(docs)) => {
            for &doc in *docs {
                r.values(doc, raw)?;
                state.many::<N>(kind, raw);
            }
        }
    }
    Ok(())
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
        s.many::<NEED_ALL>(ValueKind::Double, &stored);
    }

    /// **A NaN sum is Java's `Double.NaN`, in every build profile.** On x86
    /// `inf + -inf` is the negative default NaN, and a release build (which
    /// may commute the addition) returned `0xfff8...` where Java's fixture
    /// records `0x7ff8...`; `metric_aggs_fixtures` failed under `--release`
    /// only. The values go through `black_box` so the additions run at run
    /// time, not folded; CI runs this test in release as well.
    #[test]
    fn a_nan_sum_has_javas_nan_bits() {
        let neg_nan = f64::from_bits(0xfff8_0000_0000_0000);
        let payload_nan = f64::from_bits(0x7ff8_0000_0000_0001);
        for values in [
            [f64::INFINITY, f64::NEG_INFINITY, 1.0],
            [f64::NEG_INFINITY, 2.0, f64::INFINITY],
            [neg_nan, 1.0, 1.0],
            [f64::INFINITY, neg_nan, f64::NEG_INFINITY],
            [payload_nan, f64::INFINITY, neg_nan],
            [f64::NAN, f64::NEG_INFINITY, f64::INFINITY],
        ] {
            let mut s = MetricState::default();
            for v in values {
                s.one::<NEED_SUM>(std::hint::black_box(v));
            }
            assert_eq!(s.sum.to_bits(), 0x7ff8_0000_0000_0000, "{values:?}");
        }
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
            one.one::<NEED_ALL>(v);
            doc(&mut many, &[v]);
        }
        assert_eq!(format!("{one:?}"), format!("{many:?}"));
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

    /// A window folded from its statistics against the same values one by
    /// one: equal bit for bit when it folds, untouched when it must not.
    fn zone_vs_values<const N: u8>(start: MetricState, kind: ValueKind, values: &[i64]) -> bool {
        use lucene_codecs::adaptive_zones::ZoneStats;
        let stats = ZoneStats {
            min: *values.iter().min().unwrap(),
            max: *values.iter().max().unwrap(),
            sum: values.iter().try_fold(0i64, |a, &v| a.checked_add(v)),
        };
        let mut one = start;
        for &v in values {
            one.one::<N>(to_double(kind, v));
        }
        match fold_zone::<N>(start, stats, values.len(), kind) {
            Some(zone) => {
                assert_eq!(format!("{zone:?}"), format!("{one:?}"));
                assert_eq!(zone.sum.to_bits(), one.sum.to_bits());
                assert_eq!(zone.min.to_bits(), one.min.to_bits());
                assert_eq!(zone.min_of_mins.to_bits(), one.min_of_mins.to_bits());
                assert_eq!(zone.max_of_maxes.to_bits(), one.max_of_maxes.to_bits());
                true
            }
            None => false,
        }
    }

    #[test]
    fn a_window_folds_from_its_statistics_exactly_or_not_at_all() {
        let longs: Vec<i64> = (0..1024).map(|i| i * 3 - 700).collect();
        assert!(zone_vs_values::<NEED_ALL>(
            MetricState::default(),
            ValueKind::Long,
            &longs
        ));
        // The exact-integer bound, at the edge and one past it.
        let big = vec![1i64 << 40; 1024];
        let edge = MetricState {
            sum: ((1u64 << 53) - (1u64 << 50)) as f64,
            ..MetricState::default()
        };
        assert!(zone_vs_values::<NEED_SUM>(edge, ValueKind::Long, &big));
        let past = MetricState {
            sum: edge.sum + 1.0,
            ..edge
        };
        assert!(!zone_vs_values::<NEED_SUM>(past, ValueKind::Long, &big));
        // ...but its count and extremes still fold.
        assert!(zone_vs_values::<{ NEED_COUNT | NEED_MIN_MAX }>(
            past,
            ValueKind::Long,
            &big
        ));
        // A sum already carrying a correction, or not an integer.
        let carried = MetricState {
            delta: 1e-17,
            ..MetricState::default()
        };
        assert!(!zone_vs_values::<NEED_SUM>(
            carried,
            ValueKind::Long,
            &longs
        ));
        let half = MetricState {
            sum: 0.5,
            ..MetricState::default()
        };
        assert!(!zone_vs_values::<NEED_SUM>(half, ValueKind::Long, &longs));

        // Doubles: count and extremes, both zeros on either side, never a
        // sum, and no NaN of either sign.
        let doubles = |ds: &[f64]| ds.iter().map(|&d| enc(d)).collect::<Vec<i64>>();
        for ds in [
            [-0.0, 0.0, 1.5, 2.0],
            [0.0, -0.0, -1.5, -2.0],
            [0.0, -0.0, 0.0, -0.0],
        ] {
            let v = doubles(&ds);
            assert!(zone_vs_values::<
                { NEED_COUNT | NEED_MIN_MAX | NEED_MIN | NEED_MAX },
            >(MetricState::default(), ValueKind::Double, &v));
            assert!(!zone_vs_values::<NEED_SUM>(
                MetricState::default(),
                ValueKind::Double,
                &v
            ));
        }
        let negative_nan = f64::from_bits(0xfff8_0000_0000_0001);
        for nan in [f64::NAN, negative_nan] {
            let v = doubles(&[1.0, nan, -3.0]);
            assert!(!zone_vs_values::<NEED_MIN>(
                MetricState::default(),
                ValueKind::Double,
                &v
            ));
            assert!(!zone_vs_values::<NEED_MAX>(
                MetricState::default(),
                ValueKind::Double,
                &v
            ));
            // A count alone needs neither extreme.
            assert!(zone_vs_values::<NEED_COUNT>(
                MetricState::default(),
                ValueKind::Double,
                &v
            ));
        }

        // Floats: sign-extended sortable ints fold; anything wider does not.
        let float = |f: f32| {
            let bits = f.to_bits() as i32;
            i64::from(bits ^ ((bits >> 31) & 0x7fff_ffff))
        };
        let v: Vec<i64> = [-2.5f32, -0.0, 0.0, 7.25]
            .iter()
            .map(|&f| float(f))
            .collect();
        assert!(!zone_vs_values::<NEED_ALL>(
            MetricState::default(),
            ValueKind::Float,
            &v
        ));
        assert!(zone_vs_values::<{ NEED_MIN | NEED_MAX }>(
            MetricState::default(),
            ValueKind::Float,
            &v
        ));
        assert!(!zone_vs_values::<NEED_MIN>(
            MetricState::default(),
            ValueKind::Float,
            &[1, i64::from(u32::MAX)]
        ));
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
