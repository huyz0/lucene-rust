//! OpenSearch's numeric metric aggregations -- `min`, `max`, `sum`, `avg`,
//! `value_count` and `stats` on a numeric field, at the top level -- over a
//! query's matching documents, as a shard computes them before the
//! coordinator reduces them.
//!
//! One pass over the query's live matches in document order, segment by
//! segment, keeps for each field what every one of those aggregators needs
//! (the aggregators themselves differ only in which parts they report):
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

    /// One document's values, ascending as `SORTED_NUMERIC` stores them.
    fn doc(&mut self, values: &[f64]) {
        let (Some(&first), Some(&last)) = (values.first(), values.last()) else {
            return;
        };
        self.count += values.len() as u64;
        for &v in values {
            self.add(v);
            self.min = java_min(self.min, v);
            self.max = java_max(self.max, v);
        }
        self.min_of_mins = java_min(self.min_of_mins, first);
        self.max_of_maxes = java_max(self.max_of_maxes, last);
    }
}

/// Java's `Math.min(double, double)`: `NaN` wins, and `-0.0 < 0.0`.
pub fn java_min(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && b.to_bits() == (-0.0f64).to_bits() {
        return b;
    }
    if a <= b {
        a
    } else {
        b
    }
}

/// Java's `Math.max(double, double)`.
pub fn java_max(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && a.to_bits() == (-0.0f64).to_bits() {
        return b;
    }
    if a >= b {
        a
    } else {
        b
    }
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
    let mut states = vec![MetricState::default(); specs.len()];
    if specs.is_empty() {
        return Ok(states);
    }
    let rewritten = crate::multi_segment::rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    let clause = lone_clause(query);
    let mut raw = Vec::new();
    let mut doubles = Vec::new();
    let mut precomputed = Vec::with_capacity(specs.len());
    for (i, seg) in segments.iter().enumerate() {
        let Some(reader) = readers.get(i) else {
            break;
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
        let Some(child) = exec::build::child(&ctx, &clause, 1.0, Mode::NoScores, true)? else {
            continue;
        };
        let mut scorer = child.into_scorer(Mode::NoScores);
        let mut columns = specs
            .iter()
            .zip(&precomputed)
            .map(|(s, &done)| {
                if done {
                    Ok(Values::Absent)
                } else {
                    open_values(reader, &s.field)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let live: Option<&FixedBitSet> = seg.live_docs;
        let mut doc = exec::exact_next(&mut *scorer)?;
        while doc != NO_MORE_DOCS {
            if live.is_none_or(|l| l.get_doc(doc)) {
                for ((col, spec), state) in columns.iter_mut().zip(specs).zip(&mut states) {
                    raw.clear();
                    match col {
                        Values::Absent => {}
                        Values::Single(r) => {
                            if let Some(v) = r.value(doc).map_err(crate::Error::from)? {
                                raw.push(v);
                            }
                        }
                        Values::Multi(r) => r.values(doc, &mut raw).map_err(crate::Error::from)?,
                    }
                    doubles.clear();
                    doubles.extend(raw.iter().map(|&v| to_double(spec.kind, v)));
                    state.doc(&doubles);
                }
            }
            doc = exec::exact_next(&mut *scorer)?;
        }
    }
    Ok(states)
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
        s.doc(&[1.0]);
        for _ in 0..10 {
            s.doc(&[1e-16]);
        }
        assert!(s.sum > 1.0);
        assert_eq!(s.count, 11);
        // A non-finite value turns the tally non-finite, and it stays so.
        let mut s = MetricState::default();
        s.doc(&[1.0, f64::INFINITY]);
        assert_eq!(s.sum, f64::INFINITY);
        s.doc(&[5.0]);
        assert_eq!(s.sum, f64::INFINITY);
        let mut s = MetricState::default();
        s.doc(&[f64::NAN]);
        assert!(s.sum.is_nan());
    }

    #[test]
    fn a_documents_own_minimum_and_maximum_feed_min_and_max() {
        // `SORTED_NUMERIC` keeps NaN last: `min` reads the first value and
        // never sees it; `stats` reads every value and does.
        let mut s = MetricState::default();
        s.doc(&[1.0, f64::NAN]);
        s.doc(&[]);
        s.doc(&[-2.0, 4.0]);
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
