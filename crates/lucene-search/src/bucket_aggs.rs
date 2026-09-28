//! OpenSearch's bucket aggregations, and whatever sits under them, as a shard
//! computes them before any reduce (read path R5): `histogram`,
//! `date_histogram`, `range` (and `date_range`), `filter`, `filters`,
//! `global`, keyword `terms` with sub-aggregations, `cardinality`, and the
//! metrics of [`crate::aggs`] at any depth.
//!
//! The shape is OpenSearch's `BucketsAggregator` tree, collected document by
//! document: every aggregator receives `collect(doc, owningBucketOrd)`, and a
//! bucketing one hands the document on to its sub-aggregators under the
//! ordinal of each bucket the document falls in. A bucket's ordinal is
//! assigned when its first document arrives (`LongKeyedBucketOrds.add`) for
//! the keyed aggregations (histograms, `terms`), and is `owning * width +
//! index` for the fixed ones (`range`, `filters`, `filter`, `global`). The
//! documents come segment by segment, in document order, so each bucket's
//! metric sums are added in Java's order.
//!
//! What is ported, per aggregator:
//!
//! * `histogram` (`NumericHistogramAggregator`): each distinct `Math.floor((v -
//!   offset) / interval)` of a document's values, ascending, in the hard
//!   bounds (`key * interval`, inclusive both ends); the bucket keyed by that
//!   double's bits (`Double.doubleToLongBits`: one `NaN`, and `-0.0` apart
//!   from `0.0`).
//! * `date_histogram` (`DateHistogramAggregator`): each distinct rounded value
//!   ([`DateRounding`]: `Rounding.Prepared.round` for a calendar unit or a
//!   fixed interval, in a fixed-offset zone, with an offset), in the hard
//!   bounds (`[min, max)`).
//! * `range` (`RangeAggregator`): the ranges sorted as OpenSearch sorts them,
//!   each value binary-searched from where the previous value's search ended
//!   (`MatchedRange`), a range matching `from <= v < to`.
//! * `filters`/`filter` (`FiltersAggregator`/`FilterAggregator`): a bucket
//!   per filter a document matches (deletions aside: only live documents
//!   arrive), and with `other_bucket` one for a document matching none.
//! * `global` (`GlobalAggregator`): every document of the pass (the caller
//!   runs it over every live document, as `DefaultAggregationProcessor`
//!   searches a match-all for it).
//! * `terms` (`GlobalOrdinalsStringTermsAggregator`, remapping ordinals under
//!   a parent): per owning bucket, the live documents per distinct term, the
//!   top `shard_size` by count then term, the rest into `otherDocCount`.
//! * `cardinality`: the distinct values per bucket -- a keyword's terms, a
//!   whole number field's longs, a floating-point field's doubles as
//!   `Double.doubleToLongBits` -- which the caller hashes into
//!   `HyperLogLogPlusPlus` (a set of hashes, so the order is immaterial).
//! * the metrics: [`MetricState`] per owning bucket, and at the top level the
//!   `min`/`max` points shortcut of [`crate::aggs`].

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use lucene_codecs::terms_dict::TermsDict;

use crate::aggs::{self, MetricState, Source, ValueKind, Values};
use crate::directory_reader::SegmentReader;
use crate::exec;
use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::terms_agg::{GlobalOrds, Ords};
use crate::Result;
use lucene_util::fixed_bit_set::FixedBitSet;

/// An FxHash-style hasher for the bucket ordinals and the distinct values:
/// integer keys, hashed once per document, where SipHash's cost shows.
#[derive(Default, Clone, Copy)]
struct Fx(u64);

impl Fx {
    #[inline]
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for Fx {
    /// MurmurHash3's `fmix64`: the multiply alone leaves a key's zero low
    /// bits zero (a `float` widened to a `double`, a histogram key), and the
    /// table buckets by the low bits.
    fn finish(&self) -> u64 {
        let mut h = self.0;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^ (h >> 33)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(u64::from(b));
        }
    }

    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    fn write_i64(&mut self, i: i64) {
        self.add(i as u64);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;
type FastSet<K> = HashSet<K, BuildHasherDefault<Fx>>;

/// A calendar unit of `Rounding.DateTimeUnit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DateUnit {
    Week,
    Year,
    Quarter,
    Month,
    Day,
    Hour,
    Minute,
    Second,
}

/// A `date_histogram`'s rounding: a calendar unit or a fixed interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingKind {
    /// `TimeUnitRounding`.
    Unit(DateUnit),
    /// `TimeIntervalRounding`, in milliseconds (at least 1).
    Interval(i64),
}

/// `Rounding.Prepared` for a zone whose offset is fixed (`UTC`, `+05:30`):
/// `FixedToMidnightRounding`/`FixedNotToMidnightRounding`/`FixedRounding`
/// behind `OffsetRounding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DateRounding {
    pub kind: RoundingKind,
    /// The zone's offset from UTC, in milliseconds.
    pub zone_ms: i64,
    /// `OffsetRounding`'s offset, in milliseconds.
    pub offset: i64,
}

const MILLIS_PER_DAY: i64 = 86_400_000;
const DAYS_0000_TO_1970: i64 = 719_527;
const MILLIS_PER_YEAR: i64 = 31_556_952_000;
const MIN_TOTAL_MILLIS_BY_MONTH: [i64; 12] = month_starts(false);
const MAX_TOTAL_MILLIS_BY_MONTH: [i64; 12] = month_starts(true);

/// `DateUtilsRounding`'s `MIN_`/`MAX_TOTAL_MILLIS_BY_MONTH_ARRAY`.
const fn month_starts(leap: bool) -> [i64; 12] {
    let days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut out = [0i64; 12];
    let mut i = 0;
    while i < 11 {
        out[i + 1] = out[i] + days[i] * MILLIS_PER_DAY;
        i += 1;
    }
    out
}

/// `DateUtils.roundFloor` (units of a day or less).
fn round_floor(utc: i64, unit: i64) -> i64 {
    if utc >= 0 {
        utc - utc % unit
    } else {
        let u = utc.wrapping_add(1);
        u.wrapping_sub(u % unit).wrapping_sub(unit)
    }
}

/// `DateUtilsRounding.isLeapYear`.
fn is_leap_year(year: i32) -> bool {
    if year & 3 != 0 {
        return false;
    }
    if year % 100 != 0 {
        return true;
    }
    (year / 100) & 3 == 0
}

/// `DateUtilsRounding.utcMillisAtStartOfYear`.
fn utc_millis_at_start_of_year(year: i32) -> i64 {
    let mut leap_years = year / 100;
    if year < 0 {
        leap_years = ((year + 3) >> 2) - leap_years + ((leap_years + 3) >> 2) - 1;
    } else {
        leap_years = (year >> 2) - leap_years + (leap_years >> 2);
        if is_leap_year(year) {
            leap_years -= 1;
        }
    }
    // Java's long arithmetic, which wraps at the far ends of the range.
    i64::from(year)
        .wrapping_mul(365)
        .wrapping_add(i64::from(leap_years) - DAYS_0000_TO_1970)
        .wrapping_mul(MILLIS_PER_DAY)
}

/// `DateUtilsRounding.getYear`.
fn get_year(utc: i64) -> i32 {
    let unit = MILLIS_PER_YEAR / 2;
    let mut i2 = (utc >> 1) + (1970 * MILLIS_PER_YEAR) / 2;
    if i2 < 0 {
        i2 = i2 - unit + 1;
    }
    let mut year = (i2 / unit) as i32;
    let mut year_start = utc_millis_at_start_of_year(year);
    let diff = utc.wrapping_sub(year_start);
    if diff < 0 {
        year -= 1;
    } else if diff >= MILLIS_PER_DAY * 365 {
        let one_year = if is_leap_year(year) {
            MILLIS_PER_DAY * 366
        } else {
            MILLIS_PER_DAY * 365
        };
        year_start = year_start.wrapping_add(one_year);
        if year_start <= utc {
            year += 1;
        }
    }
    year
}

/// `DateUtilsRounding.getMonthOfYear`: 1 to 12.
fn get_month_of_year(utc: i64, year: i32) -> usize {
    let i = (utc.wrapping_sub(utc_millis_at_start_of_year(year)) >> 10) as i32;
    let d = 84_375;
    let bounds: [i32; 11] = if is_leap_year(year) {
        [31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    } else {
        [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    };
    1 + bounds.iter().take_while(|&&b| i >= b * d).count()
}

/// `DateUtils.of(year, month)`: the first millisecond of the month.
fn start_of_month(year: i32, month: usize) -> i64 {
    let table = if is_leap_year(year) {
        &MAX_TOTAL_MILLIS_BY_MONTH
    } else {
        &MIN_TOTAL_MILLIS_BY_MONTH
    };
    utc_millis_at_start_of_year(year).wrapping_add(table.get(month - 1).copied().unwrap_or(0))
}

impl DateUnit {
    /// `DateTimeUnit.roundFloor`.
    pub fn round_floor(self, utc: i64) -> i64 {
        match self {
            DateUnit::Week => {
                let three_days = 3 * MILLIS_PER_DAY;
                round_floor(utc.wrapping_add(three_days), 7 * MILLIS_PER_DAY)
                    .wrapping_sub(three_days)
            }
            DateUnit::Year => utc_millis_at_start_of_year(get_year(utc)),
            DateUnit::Quarter => {
                let year = get_year(utc);
                let month = get_month_of_year(utc, year);
                start_of_month(year, (month - 1) / 3 * 3 + 1)
            }
            DateUnit::Month => {
                let year = get_year(utc);
                start_of_month(year, get_month_of_year(utc, year))
            }
            DateUnit::Day => round_floor(utc, MILLIS_PER_DAY),
            DateUnit::Hour => round_floor(utc, 3_600_000),
            DateUnit::Minute => round_floor(utc, 60_000),
            DateUnit::Second => round_floor(utc, 1_000),
        }
    }
}

impl DateUnit {
    /// The first millisecond of the unit after the one starting at `start`
    /// (a value [`Self::round_floor`] returned); `None` past the range.
    fn next_start(self, start: i64) -> Option<i64> {
        let months = match self {
            DateUnit::Week => return start.checked_add(7 * MILLIS_PER_DAY),
            DateUnit::Day => return start.checked_add(MILLIS_PER_DAY),
            DateUnit::Hour => return start.checked_add(3_600_000),
            DateUnit::Minute => return start.checked_add(60_000),
            DateUnit::Second => return start.checked_add(1_000),
            DateUnit::Month => 1,
            DateUnit::Quarter => 3,
            DateUnit::Year => 12,
        };
        let year = get_year(start);
        let month = get_month_of_year(start, year) + months;
        let (year, month) = if month > 12 {
            (year.checked_add(1)?, month - 12)
        } else {
            (year, month)
        };
        Some(start_of_month(year, month))
    }
}

impl DateRounding {
    /// `utc`'s bucket: its rounded value and the first value past it that
    /// rounds elsewhere (every value in between rounds the same); `None`
    /// where that bound would overflow.
    pub fn bucket(&self, utc: i64) -> Option<(i64, i64)> {
        let shift = self.zone_ms.checked_sub(self.offset)?;
        let local = utc.checked_add(shift)?;
        let (floor, next) = match self.kind {
            RoundingKind::Unit(u) => {
                let floor = u.round_floor(local);
                (floor, u.next_start(floor)?)
            }
            RoundingKind::Interval(interval) => {
                let floor = self.round(utc).checked_add(shift)?;
                (floor, floor.checked_add(interval)?)
            }
        };
        Some((floor.checked_sub(shift)?, next.checked_sub(shift)?))
    }

    /// `Prepared.round`: `offset.localToUtcInThisOffset(roundFloor(
    /// offset.utcToLocalTime(utc - offset))) + offset`.
    pub fn round(&self, utc: i64) -> i64 {
        let local = utc.wrapping_sub(self.offset).wrapping_add(self.zone_ms);
        let rounded = match self.kind {
            RoundingKind::Unit(u) => u.round_floor(local),
            RoundingKind::Interval(interval) => {
                // `TimeIntervalRounding.roundKey`.
                let key = if local < 0 {
                    local.wrapping_sub(interval).wrapping_add(1) / interval
                } else {
                    local / interval
                };
                key.wrapping_mul(interval)
            }
        };
        rounded.wrapping_sub(self.zone_ms).wrapping_add(self.offset)
    }
}

/// What a `cardinality` counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardinalityKind {
    /// A keyword field's terms.
    Keyword,
    /// A numeric field's values: longs as they are, doubles and floats as
    /// the `double`'s bits (`MurmurHash3Values.hash(doubleValues)`).
    Numeric(ValueKind),
}

/// One aggregation of a request, with its sub-aggregations.
#[derive(Debug, Clone)]
pub enum AggNode {
    /// A metric of [`crate::aggs`]; `source` other than
    /// [`Source::DocValues`] only at the top level.
    Metric {
        field: String,
        kind: ValueKind,
        source: Source,
        /// The parts of [`MetricState`] the caller reads (`aggs::NEED_*`).
        needs: u8,
    },
    Cardinality {
        field: String,
        kind: CardinalityKind,
        /// `HyperLogLogPlusPlus`'s precision: the result is each bucket's
        /// sketch ([`AggResult::CardinalitySketch`]). `None` answers the
        /// distinct values themselves ([`AggResult::Cardinality`]).
        precision: Option<u32>,
    },
    /// Keyword `terms`, default order, `min_doc_count` of at least 1.
    Terms {
        field: String,
        shard_size: usize,
        subs: Vec<AggNode>,
    },
    Histogram {
        field: String,
        kind: ValueKind,
        interval: f64,
        offset: f64,
        /// `hard_bounds`, each end optional.
        hard_bounds: (Option<f64>, Option<f64>),
        subs: Vec<AggNode>,
    },
    DateHistogram {
        field: String,
        rounding: DateRounding,
        hard_bounds: (Option<i64>, Option<i64>),
        subs: Vec<AggNode>,
    },
    /// The ranges as `(from, to)`, in `RangeAggregator`'s (sorted) order.
    Range {
        field: String,
        kind: ValueKind,
        ranges: Vec<(f64, f64)>,
        subs: Vec<AggNode>,
    },
    /// `filters` (`other` adds the bucket for documents matching none);
    /// `filter` is one filter without it.
    Filters {
        filters: Vec<BooleanQuery>,
        other: bool,
        subs: Vec<AggNode>,
    },
    /// `global`: one bucket holding every document of its pass.
    Global { subs: Vec<AggNode> },
}

impl AggNode {
    fn subs(&self) -> &[AggNode] {
        match self {
            AggNode::Metric { .. } | AggNode::Cardinality { .. } => &[],
            AggNode::Terms { subs, .. }
            | AggNode::Histogram { subs, .. }
            | AggNode::DateHistogram { subs, .. }
            | AggNode::Range { subs, .. }
            | AggNode::Filters { subs, .. }
            | AggNode::Global { subs } => subs,
        }
    }

    /// The keyword fields whose global ordinals the tree reads.
    pub fn keyword_fields<'n>(&'n self, out: &mut Vec<&'n str>) {
        match self {
            AggNode::Terms { field, .. }
            | AggNode::Cardinality {
                field,
                kind: CardinalityKind::Keyword,
                ..
            } if !out.contains(&field.as_str()) => out.push(field),
            _ => {}
        }
        for s in self.subs() {
            s.keyword_fields(out);
        }
    }

    /// Whether the tree reads points (a filter's range query, a metric's
    /// points bound).
    pub fn filter_queries<'n>(&'n self, out: &mut Vec<&'n BooleanQuery>) {
        if let AggNode::Filters { filters, .. } = self {
            out.extend(filters.iter());
        }
        for s in self.subs() {
            s.filter_queries(out);
        }
    }

    /// Whether any metric of the tree reads a points bound.
    pub fn reads_points(&self) -> bool {
        matches!(self, AggNode::Metric { source, .. } if *source != Source::DocValues)
            || self.subs().iter().any(AggNode::reads_points)
    }
}

/// A distinct value a `cardinality` saw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CardinalityValue {
    Term(Vec<u8>),
    Long(i64),
}

/// One owning bucket's `terms`: `otherDocCount`, and `(term, doc count)` of
/// each kept bucket, by term.
pub type TermsBuckets = (u64, Vec<(Vec<u8>, u64)>);

/// One aggregation's shard result, per owning bucket (in the order the
/// parent lists its buckets; one owning bucket at the top level).
#[derive(Debug, Clone, PartialEq)]
pub enum AggResult {
    Metric(Vec<MetricState>),
    Cardinality(Vec<Vec<CardinalityValue>>),
    /// Per owning bucket, its `HyperLogLogPlusPlus` as `writeTo` writes it
    /// (`crate::cardinality_sketch`); empty for a bucket that saw nothing.
    CardinalitySketch(Vec<Vec<u8>>),
    /// Per owning bucket: `otherDocCount` and the kept buckets by term;
    /// `subs[i]`'s owning buckets are the kept buckets in that order.
    Terms {
        buckets: Vec<TermsBuckets>,
        subs: Vec<AggResult>,
    },
    /// Per owning bucket: `(floor((v - offset) / interval), doc count)`, by
    /// key ascending (`Double.compare`).
    Histogram {
        buckets: Vec<Vec<(f64, u64)>>,
        subs: Vec<AggResult>,
    },
    /// Per owning bucket: `(rounded key, doc count)`, by key ascending.
    DateHistogram {
        buckets: Vec<Vec<(i64, u64)>>,
        subs: Vec<AggResult>,
    },
    /// `width` buckets per owning bucket (a range, a filter), every one
    /// listed, `docs[owning * width + i]`.
    Fixed {
        width: usize,
        docs: Vec<u64>,
        subs: Vec<AggResult>,
    },
}

/// What a collection pass reads besides the tree: each keyword field's
/// global ordinals.
pub struct Globals<'g> {
    pub ords: &'g HashMap<String, Arc<GlobalOrds>>,
}

impl<'g> Globals<'g> {
    fn get(&self, field: &str) -> Result<&'g GlobalOrds> {
        self.ords
            .get(field)
            .map(|g| &**g)
            .ok_or_else(|| crate::Error::TermsAggType(format!("{field}: no global ordinals")))
    }
}

/// The collected state of one aggregation (`BucketsAggregator` and kin).
enum State {
    Metric(Vec<MetricState>),
    /// Distinct values per owning bucket: global ordinals for a keyword, raw
    /// longs otherwise.
    Cardinality(Vec<FastSet<i64>>),
    /// A keyword `cardinality`: each owning bucket's global ordinals as bits
    /// (`OrdinalsCollector`'s per-bucket `BitArray`), read back in term order.
    OrdBits(Vec<Vec<u64>>),
    /// A numeric `cardinality` with a precision: each owning bucket's
    /// `HyperLogLogPlusPlus`, fed each value's hash as it is collected --
    /// `CardinalityAggregator.DirectCollector`, in document order.
    Sketches {
        p: u32,
        sketches: Vec<Option<crate::cardinality_sketch::Sketch>>,
    },
    /// `LongKeyedBucketOrds`: `(owning, key)` to bucket ordinal, the keys in
    /// ordinal order, each bucket's document count.
    Keyed {
        ords: FastMap<(u32, i64), u32>,
        keys: Vec<(u32, i64)>,
        docs: Vec<u64>,
        subs: Vec<State>,
    },
    Fixed {
        width: usize,
        docs: Vec<u64>,
        subs: Vec<State>,
    },
}

impl State {
    fn new(node: &AggNode) -> State {
        let subs = || node.subs().iter().map(State::new).collect();
        match node {
            AggNode::Metric { .. } => State::Metric(Vec::new()),
            AggNode::Cardinality {
                kind: CardinalityKind::Numeric(_),
                precision: Some(p),
                ..
            } => State::Sketches {
                p: *p,
                sketches: Vec::new(),
            },
            AggNode::Cardinality {
                kind: CardinalityKind::Keyword,
                ..
            } => State::OrdBits(Vec::new()),
            AggNode::Cardinality { .. } => State::Cardinality(Vec::new()),
            AggNode::Terms { .. } | AggNode::Histogram { .. } | AggNode::DateHistogram { .. } => {
                State::Keyed {
                    ords: FastMap::default(),
                    keys: Vec::new(),
                    docs: Vec::new(),
                    subs: subs(),
                }
            }
            AggNode::Range { ranges, .. } => State::Fixed {
                width: ranges.len(),
                docs: Vec::new(),
                subs: subs(),
            },
            AggNode::Filters { filters, other, .. } => State::Fixed {
                width: filters.len() + usize::from(*other),
                docs: Vec::new(),
                subs: subs(),
            },
            AggNode::Global { .. } => State::Fixed {
                width: 1,
                docs: Vec::new(),
                subs: subs(),
            },
        }
    }
}

/// A slot of `v`, grown to hold it.
fn slot<T: Default>(v: &mut Vec<T>, i: usize) -> &mut T {
    if v.len() <= i {
        v.resize_with(i + 1, T::default);
    }
    &mut v[i]
}

/// The documents an aggregation reads its columns ahead for at a time.
const WINDOW: i32 = 1024;

#[cfg(test)]
thread_local! {
    /// Tests: read every column and collect every node document by
    /// document, the path the windows must agree with.
    static NO_WINDOWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn no_windows() -> bool {
    #[cfg(test)]
    {
        NO_WINDOWS.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// One aggregation's columns over one segment.
struct Leaf<'a> {
    col: Col<'a>,
    /// The document's values, read by this node (its subs have their own).
    buf: Vec<i64>,
    /// A `range`'s `maxTo`.
    max_to: Vec<f64>,
    /// A `date_histogram`'s last bucket: `[from, to)` rounds to `rounded`,
    /// whose ordinal under owning bucket `.3` is `.4` (`u32::MAX`: none yet).
    memo: (i64, i64, i64, u32, u32),
    /// This segment was answered from the points ([`count_from_points`]).
    done: bool,
    /// A window of a single-valued numeric column read ahead
    /// ([`Leaf::prefetch`]): document `pre_base + i`'s value is `pre_vals[i]`
    /// when bit `i` of `pre_bits` is set, for `i < pre_len`.
    pre_base: i32,
    pre_len: usize,
    pre_vals: Vec<i64>,
    pre_bits: Vec<u64>,
    /// A multi-valued column's window instead: document `pre_base + i`'s
    /// values are `pre_vals[pre_offsets[i]..pre_offsets[i + 1]]`.
    pre_multi: bool,
    pre_offsets: Vec<u32>,
    /// The documents this node placed in a bucket during a window, and the
    /// buckets: what its sub-aggregations collect next ([`collect_window`]).
    win_docs: Vec<i32>,
    win_ords: Vec<u32>,
    subs: Vec<Leaf<'a>>,
}

enum Col<'a> {
    /// A numeric column; `done` when the points answered this segment.
    Values(Values<'a>, bool),
    /// A keyword column and its segment-to-global ordinal map.
    Ords(Ords<'a>, &'a [i64]),
    /// Each filter's matching documents (`None`: all of them), as bits.
    Filters(Vec<Option<Vec<u64>>>),
    None,
}

impl Leaf<'_> {
    /// Reads the documents `base..base + len` of this node's single-valued
    /// numeric column ahead, and its sub-aggregations' -- one pass over each
    /// column per window (`NumericReader::fill_window`) instead of a lookup
    /// per document. [`Self::read`] answers from the window what it covers.
    fn prefetch(&mut self, base: i32, len: usize) -> Result<()> {
        self.pre_len = 0;
        self.pre_multi = false;
        match &mut self.col {
            // A keyword's segment ordinals are the same column shapes.
            Col::Values(Values::Single(r), false) | Col::Ords(Ords::Single(r), _) => {
                self.pre_vals.resize(len, 0);
                self.pre_bits.resize(len.div_ceil(64), 0);
                r.fill_window(base, &mut self.pre_vals, &mut self.pre_bits)
                    .map_err(crate::Error::from)?;
                self.pre_base = base;
                self.pre_len = len;
            }
            Col::Values(Values::Multi(r), false) | Col::Ords(Ords::Multi(r), _) => {
                // A column that cannot be read a window at a time is read
                // document by document, for the matching ones only.
                if r.fill_window(base, len, &mut self.pre_offsets, &mut self.pre_vals)
                    .map_err(crate::Error::from)?
                {
                    self.pre_multi = true;
                    self.pre_base = base;
                    self.pre_len = len;
                }
            }
            _ => {}
        }
        for s in &mut self.subs {
            s.prefetch(base, len)?;
        }
        Ok(())
    }

    /// The value a window read ahead holds for `doc`: `Some(None)` when the
    /// document has none, `None` when the window does not cover it.
    #[inline]
    fn prefetched(&self, doc: i32) -> Option<Option<i64>> {
        let i = doc.wrapping_sub(self.pre_base) as u32 as usize;
        if i >= self.pre_len || self.pre_multi {
            return None;
        }
        let has = self
            .pre_bits
            .get(i >> 6)
            .is_some_and(|w| w >> (i & 63) & 1 == 1);
        Some(has.then(|| self.pre_vals[i]))
    }

    /// Forgets the window read ahead, here and in the sub-aggregations.
    fn drop_prefetch(&mut self) {
        self.pre_len = 0;
        for s in &mut self.subs {
            s.drop_prefetch();
        }
    }

    /// The document's values into `buf` (ascending); false without any.
    fn read(&mut self, doc: i32) -> Result<bool> {
        self.buf.clear();
        // The window read ahead, when it covers the document.
        let i = doc.wrapping_sub(self.pre_base) as u32 as usize;
        if i < self.pre_len {
            if self.pre_multi {
                let range = self.pre_offsets.get(i).zip(self.pre_offsets.get(i + 1));
                if let Some((&from, &to)) = range {
                    if let Some(v) = self.pre_vals.get(from as usize..to as usize) {
                        self.buf.extend_from_slice(v);
                    }
                }
            } else if self
                .pre_bits
                .get(i >> 6)
                .is_some_and(|w| w >> (i & 63) & 1 == 1)
            {
                self.buf.push(self.pre_vals[i]);
            }
            return Ok(!self.buf.is_empty());
        }
        match &mut self.col {
            Col::Values(Values::Absent, _) | Col::Ords(Ords::Absent, _) => {}
            Col::Values(Values::Single(r), _) => {
                // A dense column's value inline; any other through `value`.
                if let Some(v) = r.dense_value(doc) {
                    self.buf.push(v);
                } else if let Some(v) = r.value(doc)? {
                    self.buf.push(v);
                }
            }
            Col::Values(Values::Multi(r), _) => r.values(doc, &mut self.buf)?,
            Col::Ords(Ords::Single(r), _) => {
                if let Some(v) = r.dense_value(doc) {
                    self.buf.push(v);
                } else if let Some(v) = r.value(doc)? {
                    self.buf.push(v);
                }
            }
            Col::Ords(Ords::Multi(r), _) => r.values(doc, &mut self.buf)?,
            Col::Filters(_) | Col::None => {}
        }
        Ok(!self.buf.is_empty())
    }
}

fn bit(words: &[u64], doc: i32) -> bool {
    let d = doc as u32 as usize;
    words.get(d >> 6).is_some_and(|w| w >> (d & 63) & 1 == 1)
}

/// One segment's view: what opening a node's columns needs.
struct SegmentView<'s, 'a> {
    seg: &'s OpenSegment<'a>,
    reader: &'a SegmentReader,
    index: usize,
    top: bool,
}

fn open_leaf<'a>(
    node: &AggNode,
    view: &SegmentView<'_, 'a>,
    globals: &Globals<'a>,
    state: &mut State,
) -> Result<Leaf<'a>> {
    let reader = view.reader;
    let col = match node {
        AggNode::Metric {
            field,
            kind,
            source,
            ..
        } => {
            // The points shortcut (a top-level `min`/`max` over a match-all)
            // answers the segment without visiting a document.
            let bound = if view.top && *source != Source::DocValues {
                let spec = aggs::MetricSpec {
                    field: field.clone(),
                    kind: *kind,
                    source: *source,
                    needs: aggs::NEED_ALL,
                };
                aggs::leaf_point_bound(view.seg, &spec)?
            } else {
                None
            };
            if let (Some(v), State::Metric(states)) = (bound, &mut *state) {
                let s = slot(states, 0);
                if *source == Source::PointsMin {
                    s.min_of_mins = aggs::java_min(s.min_of_mins, v);
                } else {
                    s.max_of_maxes = aggs::java_max(s.max_of_maxes, v);
                }
            }
            Col::Values(aggs::open_values(reader, field)?, bound.is_some())
        }
        AggNode::Cardinality {
            field,
            kind: CardinalityKind::Keyword,
            ..
        }
        | AggNode::Terms { field, .. } => {
            let map = globals.get(field)?.segment_map(view.index).unwrap_or(&[]);
            Col::Ords(crate::terms_agg::open_ords(reader, field)?.0, map)
        }
        AggNode::Cardinality { field, .. }
        | AggNode::Histogram { field, .. }
        | AggNode::DateHistogram { field, .. }
        | AggNode::Range { field, .. } => Col::Values(aggs::open_values(reader, field)?, false),
        AggNode::Filters { filters, .. } => {
            let mut bits = Vec::with_capacity(filters.len());
            for f in filters {
                bits.push(filter_bits(view.seg, f)?);
            }
            Col::Filters(bits)
        }
        AggNode::Global { .. } => Col::None,
    };
    let mut subs = Vec::with_capacity(node.subs().len());
    let sub_view = SegmentView {
        top: false,
        ..*view
    };
    let sub_states: &mut [State] = match state {
        State::Keyed { subs, .. } | State::Fixed { subs, .. } => subs,
        _ => &mut [],
    };
    for (s, st) in node.subs().iter().zip(sub_states) {
        subs.push(open_leaf(s, &sub_view, globals, st)?);
    }
    let max_to = match node {
        AggNode::Range { ranges, .. } => max_to(ranges),
        _ => Vec::new(),
    };
    Ok(Leaf {
        col,
        buf: Vec::new(),
        max_to,
        memo: (1, 0, 0, 0, u32::MAX),
        done: false,
        pre_base: 0,
        pre_len: 0,
        pre_vals: Vec::new(),
        pre_bits: Vec::new(),
        pre_multi: false,
        pre_offsets: Vec::new(),
        win_docs: Vec::new(),
        win_ords: Vec::new(),
        subs,
    })
}

/// A filter's matches in one segment, deletions ignored (`FiltersAggregator`
/// asks the weight for every document; only live ones are collected), as
/// bits; `None` when it matches every document.
fn filter_bits(seg: &OpenSegment<'_>, filter: &BooleanQuery) -> Result<Option<Vec<u64>>> {
    let rewritten = crate::multi_segment::rewrite_points_ranges(filter, std::slice::from_ref(seg));
    let filter = rewritten.as_ref().unwrap_or(filter);
    let clause = aggs::lone_clause(filter);
    let ctx = aggs::plain_context(seg);
    let mut docs = Vec::new();
    let max = usize::try_from(seg.max_doc.unwrap_or(0)).unwrap_or(0);
    let words = match aggs::segment_matches(&ctx, filter, &clause, None, &mut docs)? {
        None => vec![0; max.div_ceil(64)],
        Some(None) => return Ok(None),
        Some(Some(d)) => {
            let mut words = vec![0u64; max.div_ceil(64)];
            for &doc in d {
                let doc = doc as u32 as usize;
                if let Some(w) = words.get_mut(doc >> 6) {
                    *w |= 1 << (doc & 63);
                }
            }
            words
        }
    };
    Ok(Some(words))
}

/// Adds one to bucket `ord`'s count (`collectBucket`/`collectExistingBucket`)
/// and hands the document to the sub-aggregations under it.
fn collect_bucket(
    node: &AggNode,
    docs: &mut Vec<u64>,
    subs: &mut [State],
    leaf_subs: &mut [Leaf<'_>],
    doc: i32,
    ord: u32,
) -> Result<()> {
    *slot(docs, ord as usize) += 1;
    for ((n, s), l) in node.subs().iter().zip(subs).zip(leaf_subs) {
        collect(n, s, l, doc, ord)?;
    }
    Ok(())
}

/// `LongKeyedBucketOrds.add`: the bucket of `(owning, key)`, new or not.
fn keyed_ord(
    ords: &mut FastMap<(u32, i64), u32>,
    keys: &mut Vec<(u32, i64)>,
    owning: u32,
    key: i64,
) -> u32 {
    let next = keys.len() as u32;
    *ords.entry((owning, key)).or_insert_with(|| {
        keys.push((owning, key));
        next
    })
}

/// The hash `CardinalityAggregator` collects for a numeric value: `mix64` of
/// a long, or of a double's `doubleToLongBits`.
#[inline]
fn numeric_hash(kind: ValueKind, v: i64) -> i64 {
    crate::cardinality_sketch::mix64(match kind {
        ValueKind::Long => v,
        k => java_double_bits(aggs::to_double(k, v)),
    })
}

/// An owning bucket's sketch, created at its first value.
fn sketch_for(
    s: &mut Option<crate::cardinality_sketch::Sketch>,
    p: u32,
) -> Result<&mut crate::cardinality_sketch::Sketch> {
    if s.is_none() {
        *s = Some(
            crate::cardinality_sketch::Sketch::new(p)
                .ok_or_else(|| crate::Error::TermsAggType(format!("cardinality precision {p}")))?,
        );
    }
    Ok(s.as_mut().expect("just set"))
}

/// [`collect`] for each of `docs` (ascending, inside the window read ahead)
/// under owning bucket `owners[i]` -- the same calls in the same order, per
/// node: a bucket aggregation places the whole window, then hands its
/// sub-aggregations the documents it placed and their buckets in one call,
/// so each sub-aggregation still sees its documents in ascending order. A
/// metric, and a `histogram` or `date_histogram` over a single-valued column,
/// run without the per-document dispatch; everything else document by
/// document.
fn collect_window(
    node: &AggNode,
    state: &mut State,
    leaf: &mut Leaf<'_>,
    docs: &[i32],
    owners: &[u32],
) -> Result<()> {
    // The single-valued fast paths below read `Leaf::prefetched`; a
    // multi-valued window goes document by document (through `Leaf::read`).
    let prefetched = leaf.pre_len > 0 && !leaf.pre_multi;
    match (node, &mut *state) {
        (AggNode::Metric { kind, needs, .. }, State::Metric(states)) if prefetched => {
            // `collect_needs`'s dispatch, once for the window.
            use aggs::{NEED_ALL, NEED_COUNT, NEED_MAX, NEED_MIN, NEED_SUM};
            fn run<const N: u8>(
                states: &mut Vec<aggs::MetricState>,
                leaf: &Leaf<'_>,
                kind: ValueKind,
                docs: &[i32],
                owners: &[u32],
            ) {
                for (&doc, &owner) in docs.iter().zip(owners) {
                    if let Some(Some(v)) = leaf.prefetched(doc) {
                        slot(states, owner as usize).many::<N>(kind, &[v]);
                    }
                }
            }
            match *needs {
                NEED_MIN => run::<NEED_MIN>(states, leaf, *kind, docs, owners),
                NEED_MAX => run::<NEED_MAX>(states, leaf, *kind, docs, owners),
                NEED_COUNT => run::<NEED_COUNT>(states, leaf, *kind, docs, owners),
                n if n & !(NEED_COUNT | NEED_SUM) == 0 => {
                    run::<{ NEED_COUNT | NEED_SUM }>(states, leaf, *kind, docs, owners)
                }
                _ => run::<NEED_ALL>(states, leaf, *kind, docs, owners),
            }
            return Ok(());
        }
        (
            AggNode::Cardinality {
                kind: CardinalityKind::Numeric(k),
                ..
            },
            State::Sketches { p, sketches },
        ) if prefetched => {
            for (&doc, &owner) in docs.iter().zip(owners) {
                if let Some(Some(v)) = leaf.prefetched(doc) {
                    sketch_for(slot(sketches, owner as usize), *p)?.collect(numeric_hash(*k, v));
                }
            }
            return Ok(());
        }
        (
            AggNode::DateHistogram {
                rounding,
                hard_bounds,
                ..
            },
            State::Keyed {
                ords,
                keys,
                docs: counts,
                ..
            },
        ) if prefetched => {
            let (mut win_docs, mut win_ords) = (
                std::mem::take(&mut leaf.win_docs),
                std::mem::take(&mut leaf.win_ords),
            );
            win_docs.clear();
            win_ords.clear();
            for (&doc, &owning) in docs.iter().zip(owners) {
                let Some(Some(v)) = leaf.prefetched(doc) else {
                    continue;
                };
                // As `collect`'s single-valued case.
                let rounded = if v >= leaf.memo.0 && v < leaf.memo.1 {
                    leaf.memo.2
                } else {
                    let bucket = rounding.bucket(v);
                    if let Some((from, to)) = bucket {
                        leaf.memo = (from, to, from, 0, u32::MAX);
                    }
                    bucket.map_or_else(|| rounding.round(v), |b| b.0)
                };
                let contained = !hard_bounds.1.is_some_and(|max| rounded >= max)
                    && !hard_bounds.0.is_some_and(|min| rounded < min);
                if !contained {
                    continue;
                }
                let ord =
                    if leaf.memo.2 == rounded && leaf.memo.3 == owning && leaf.memo.4 != u32::MAX {
                        leaf.memo.4
                    } else {
                        let o = keyed_ord(ords, keys, owning, rounded);
                        if leaf.memo.2 == rounded {
                            leaf.memo.3 = owning;
                            leaf.memo.4 = o;
                        }
                        o
                    };
                *slot(counts, ord as usize) += 1;
                win_docs.push(doc);
                win_ords.push(ord);
            }
            let out = collect_subs_window(node, state, leaf, &win_docs, &win_ords);
            leaf.win_docs = win_docs;
            leaf.win_ords = win_ords;
            return out;
        }
        (
            AggNode::Histogram {
                kind,
                interval,
                offset,
                hard_bounds,
                ..
            },
            State::Keyed {
                ords,
                keys,
                docs: counts,
                ..
            },
        ) if prefetched => {
            let (mut win_docs, mut win_ords) = (
                std::mem::take(&mut leaf.win_docs),
                std::mem::take(&mut leaf.win_ords),
            );
            win_docs.clear();
            win_ords.clear();
            for (&doc, &owning) in docs.iter().zip(owners) {
                let Some(Some(v)) = leaf.prefetched(doc) else {
                    continue;
                };
                // As `collect`'s case for one value: its `previous` starts at
                // negative infinity, so a key of negative infinity is skipped.
                let key = ((aggs::to_double(*kind, v) - offset) / interval).floor();
                if key == f64::NEG_INFINITY {
                    continue;
                }
                let bound = key * interval;
                let contained = !hard_bounds.1.is_some_and(|max| bound > max)
                    && !hard_bounds.0.is_some_and(|min| bound < min);
                if contained {
                    let ord = keyed_ord(ords, keys, owning, java_double_bits(key));
                    *slot(counts, ord as usize) += 1;
                    win_docs.push(doc);
                    win_ords.push(ord);
                }
            }
            let out = collect_subs_window(node, state, leaf, &win_docs, &win_ords);
            leaf.win_docs = win_docs;
            leaf.win_ords = win_ords;
            return out;
        }
        // `global`: every document in its owner's bucket, the window handed
        // on as it came.
        (AggNode::Global { .. }, State::Fixed { docs: counts, .. }) => {
            for &owner in owners {
                *slot(counts, owner as usize) += 1;
            }
            return collect_subs_window(node, state, leaf, docs, owners);
        }
        _ => {}
    }
    for (&doc, &owner) in docs.iter().zip(owners) {
        collect(node, state, leaf, doc, owner)?;
    }
    Ok(())
}

/// A bucket aggregation's sub-aggregations over the documents it placed in
/// a window ([`collect_window`]).
fn collect_subs_window(
    node: &AggNode,
    state: &mut State,
    leaf: &mut Leaf<'_>,
    docs: &[i32],
    ords: &[u32],
) -> Result<()> {
    let subs: &mut [State] = match state {
        State::Keyed { subs, .. } | State::Fixed { subs, .. } => subs,
        _ => &mut [],
    };
    for ((n, s), l) in node.subs().iter().zip(subs).zip(&mut leaf.subs) {
        collect_window(n, s, l, docs, ords)?;
    }
    Ok(())
}

/// `LeafBucketCollector.collect(doc, owningBucketOrd)` for `node`.
fn collect(
    node: &AggNode,
    state: &mut State,
    leaf: &mut Leaf<'_>,
    doc: i32,
    owning: u32,
) -> Result<()> {
    match (node, state) {
        (AggNode::Metric { kind, needs, .. }, State::Metric(states)) => {
            if matches!(leaf.col, Col::Values(_, true)) || !leaf.read(doc)? {
                return Ok(());
            }
            slot(states, owning as usize).collect_needs(*needs, *kind, &leaf.buf);
        }
        (AggNode::Cardinality { .. }, State::OrdBits(bits)) => {
            if !leaf.read(doc)? {
                return Ok(());
            }
            let Col::Ords(_, map) = &leaf.col else {
                return Err(crate::Error::TermsAggType(
                    "keyword cardinality without an ordinal column".to_string(),
                ));
            };
            let words = slot(bits, owning as usize);
            for &o in &leaf.buf {
                // A global ordinal is below the reader's term count; a
                // negative one never reaches the bit set.
                let Ok(g) = usize::try_from(global_ord(map, o)?) else {
                    return Err(crate::Error::TermsAggType(format!(
                        "negative global ordinal for segment ordinal {o}"
                    )));
                };
                *slot(words, g >> 6) |= 1 << (g & 63);
            }
        }
        (AggNode::Cardinality { kind, .. }, State::Sketches { p, sketches }) => {
            if !leaf.read(doc)? {
                return Ok(());
            }
            let CardinalityKind::Numeric(k) = kind else {
                return Err(crate::Error::TermsAggType(
                    "a keyword cardinality has no direct sketch".to_string(),
                ));
            };
            let sketch = sketch_for(slot(sketches, owning as usize), *p)?;
            for &v in &leaf.buf {
                sketch.collect(numeric_hash(*k, v));
            }
        }
        (AggNode::Cardinality { kind, .. }, State::Cardinality(sets)) => {
            if !leaf.read(doc)? {
                return Ok(());
            }
            // A numeric one without a precision (a keyword's is `OrdBits`,
            // one with a precision `Sketches`).
            let set = slot(sets, owning as usize);
            match kind {
                CardinalityKind::Numeric(ValueKind::Long) => set.extend(leaf.buf.iter().copied()),
                CardinalityKind::Numeric(k) => {
                    for &v in &leaf.buf {
                        set.insert(java_double_bits(aggs::to_double(*k, v)));
                    }
                }
                CardinalityKind::Keyword => {
                    return Err(crate::Error::TermsAggType(
                        "keyword cardinality counted as values".to_string(),
                    ))
                }
            }
        }
        (
            AggNode::Terms { .. },
            State::Keyed {
                ords,
                keys,
                docs,
                subs,
                ..
            },
        ) => {
            if !leaf.read(doc)? {
                return Ok(());
            }
            let values = std::mem::take(&mut leaf.buf);
            let map = match &leaf.col {
                Col::Ords(_, map) => *map,
                _ => &[],
            };
            let mut out = Ok(());
            for &o in &values {
                let ord = match global_ord(map, o) {
                    Ok(g) => keyed_ord(ords, keys, owning, g),
                    Err(e) => {
                        out = Err(e);
                        break;
                    }
                };
                if let Err(e) = collect_bucket(node, docs, subs, &mut leaf.subs, doc, ord) {
                    out = Err(e);
                    break;
                }
            }
            leaf.buf = values;
            out?;
        }
        (
            AggNode::Histogram {
                kind,
                interval,
                offset,
                hard_bounds,
                ..
            },
            State::Keyed {
                ords,
                keys,
                docs,
                subs,
                ..
            },
        ) => {
            if !leaf.read(doc)? {
                return Ok(());
            }
            let values = std::mem::take(&mut leaf.buf);
            let mut previous = f64::NEG_INFINITY;
            let mut out = Ok(());
            for &v in &values {
                let key = ((aggs::to_double(*kind, v) - offset) / interval).floor();
                if key == previous {
                    continue;
                }
                let bound = key * interval;
                let contained = !hard_bounds.1.is_some_and(|max| bound > max)
                    && !hard_bounds.0.is_some_and(|min| bound < min);
                if contained {
                    let ord = keyed_ord(ords, keys, owning, java_double_bits(key));
                    if let Err(e) = collect_bucket(node, docs, subs, &mut leaf.subs, doc, ord) {
                        out = Err(e);
                        break;
                    }
                }
                previous = key;
            }
            leaf.buf = values;
            out?;
        }
        (
            AggNode::DateHistogram {
                rounding,
                hard_bounds,
                ..
            },
            State::Keyed {
                ords,
                keys,
                docs,
                subs,
                ..
            },
        ) => {
            if !leaf.read(doc)? {
                return Ok(());
            }
            let values = std::mem::take(&mut leaf.buf);
            let mut previous = i64::MIN;
            let mut out = Ok(());
            // A single-valued column's value is rounded as is; of a
            // multi-valued one's, a rounded value equal to the last one
            // collected is skipped.
            let single = matches!(leaf.col, Col::Values(Values::Single(_), _));
            for &v in &values {
                let hit = v >= leaf.memo.0 && v < leaf.memo.1;
                let rounded = if hit {
                    leaf.memo.2
                } else {
                    // The bucket's `from` is `round(v)`: the floor, back in
                    // UTC. Only an overflowing value has no bucket.
                    let bucket = rounding.bucket(v);
                    if let Some((from, to)) = bucket {
                        leaf.memo = (from, to, from, 0, u32::MAX);
                    }
                    bucket.map_or_else(|| rounding.round(v), |b| b.0)
                };
                if !single && rounded == previous {
                    continue;
                }
                let contained = !hard_bounds.1.is_some_and(|max| rounded >= max)
                    && !hard_bounds.0.is_some_and(|min| rounded < min);
                if contained {
                    // The last bucket's ordinal again, when the owner is the same.
                    let ord = if leaf.memo.2 == rounded
                        && leaf.memo.3 == owning
                        && leaf.memo.4 != u32::MAX
                    {
                        leaf.memo.4
                    } else {
                        let o = keyed_ord(ords, keys, owning, rounded);
                        if leaf.memo.2 == rounded {
                            leaf.memo.3 = owning;
                            leaf.memo.4 = o;
                        }
                        o
                    };
                    if let Err(e) = collect_bucket(node, docs, subs, &mut leaf.subs, doc, ord) {
                        out = Err(e);
                        break;
                    }
                }
                previous = rounded;
            }
            leaf.buf = values;
            out?;
        }
        (
            AggNode::Range { kind, ranges, .. },
            State::Fixed {
                width, docs, subs, ..
            },
        ) => {
            if !leaf.read(doc)? {
                return Ok(());
            }
            let values = std::mem::take(&mut leaf.buf);
            let mut lo = 0usize;
            let mut out = Ok(());
            'values: for &v in &values {
                let value = aggs::to_double(*kind, v);
                let (start, end) = matched_range(ranges, lo, value, &leaf.max_to);
                for (i, &(from, to)) in ranges.iter().enumerate().take(end).skip(start) {
                    if value >= from && value < to {
                        let ord = (owning as usize * *width + i) as u32;
                        if let Err(e) = collect_bucket(node, docs, subs, &mut leaf.subs, doc, ord) {
                            out = Err(e);
                            break 'values;
                        }
                    }
                }
                lo = end;
            }
            leaf.buf = values;
            out?;
        }
        (
            AggNode::Filters { other, .. },
            State::Fixed {
                width, docs, subs, ..
            },
        ) => {
            let base = owning as usize * *width;
            let bits = match &leaf.col {
                Col::Filters(b) => b,
                _ => return Ok(()),
            };
            let hits: Vec<usize> = bits
                .iter()
                .enumerate()
                .filter(|(_, b)| b.as_ref().is_none_or(|w| bit(w, doc)))
                .map(|(i, _)| i)
                .collect();
            for &i in &hits {
                collect_bucket(node, docs, subs, &mut leaf.subs, doc, (base + i) as u32)?;
            }
            if *other && hits.is_empty() {
                collect_bucket(
                    node,
                    docs,
                    subs,
                    &mut leaf.subs,
                    doc,
                    (base + bits.len()) as u32,
                )?;
            }
        }
        (AggNode::Global { .. }, State::Fixed { docs, subs, .. }) => {
            collect_bucket(node, docs, subs, &mut leaf.subs, doc, owning)?;
        }
        _ => {
            return Err(crate::Error::TermsAggType(
                "aggregation state does not match its node".to_string(),
            ))
        }
    }
    Ok(())
}

/// `Double.doubleToLongBits`: every `NaN` is the canonical one.
fn java_double_bits(d: f64) -> i64 {
    if d.is_nan() {
        f64::NAN.to_bits() as i64
    } else {
        d.to_bits() as i64
    }
}

fn global_ord(map: &[i64], ord: i64) -> Result<i64> {
    usize::try_from(ord)
        .ok()
        .and_then(|o| map.get(o))
        .copied()
        .ok_or_else(|| {
            crate::Error::from(lucene_codecs::doc_values::Error::from(
                lucene_store::Error::Corrupted(format!(
                    "ordinal {ord} outside the segment's dictionary"
                )),
            ))
        })
}

/// `RangeAggregator.maxTo`.
fn max_to(ranges: &[(f64, f64)]) -> Vec<f64> {
    let mut out = Vec::with_capacity(ranges.len());
    for (i, &(_, to)) in ranges.iter().enumerate() {
        out.push(match i {
            0 => to,
            _ => aggs::java_max(to, out[i - 1]),
        });
    }
    out
}

/// `RangeAggregator.MatchedRange`: the ranges `[start, end)` that may hold
/// `value`, searching from `low`.
fn matched_range(ranges: &[(f64, f64)], low: usize, value: f64, max_to: &[f64]) -> (usize, usize) {
    // Signed, as Java's ints are: `hi` may reach -1.
    let from = |i: i64| ranges[i as usize].0;
    let max = |i: i64| max_to[i as usize];
    let (mut lo, mut hi) = (low as i64, ranges.len() as i64 - 1);
    let mut mid = (lo + hi) >> 1;
    while lo <= hi {
        if value < from(mid) {
            hi = mid - 1;
        } else if value >= max(mid) {
            lo = mid + 1;
        } else {
            break;
        }
        mid = (lo + hi) >> 1;
    }
    if lo > hi {
        return (lo as usize, lo as usize);
    }
    let (mut start_lo, mut start_hi) = (lo, mid);
    while start_lo <= start_hi {
        let m = (start_lo + start_hi) >> 1;
        if value >= max(m) {
            start_lo = m + 1;
        } else {
            start_hi = m - 1;
        }
    }
    let (mut end_lo, mut end_hi) = (mid, hi);
    while end_lo <= end_hi {
        let m = (end_lo + end_hi) >> 1;
        if value < from(m) {
            end_hi = m - 1;
        } else {
            end_lo = m + 1;
        }
    }
    (start_lo as usize, (end_hi + 1) as usize)
}

/// How a points field's packed values read as its doc values do (the
/// sortable longs of a `double`, the sortable ints of a `float`).
#[derive(Debug, Clone, Copy)]
enum PointDecode {
    Long8,
    Int4,
    Float4,
}

impl PointDecode {
    fn of(kind: ValueKind, bytes: i32) -> Option<Self> {
        match (kind, bytes) {
            (ValueKind::Long | ValueKind::Double, 8) => Some(PointDecode::Long8),
            (ValueKind::Long, 4) => Some(PointDecode::Int4),
            (ValueKind::Float, 4) => Some(PointDecode::Float4),
            _ => None,
        }
    }

    fn decode(self, packed: &[u8]) -> Option<i64> {
        match self {
            PointDecode::Long8 => {
                let b: [u8; 8] = packed.get(..8)?.try_into().ok()?;
                Some(i64::from_be_bytes(b) ^ i64::MIN)
            }
            PointDecode::Int4 | PointDecode::Float4 => {
                let b: [u8; 4] = packed.get(..4)?.try_into().ok()?;
                Some(i64::from(i32::from_be_bytes(b) ^ i32::MIN))
            }
        }
    }
}

/// Counts the points whose value `side` places in its interval
/// (`Ordering::Equal`); `side` must be monotone in the points' order.
struct CountIn<'f> {
    decode: PointDecode,
    side: &'f dyn Fn(i64) -> std::cmp::Ordering,
    count: u64,
}

impl lucene_codecs::points::IntersectVisitor for CountIn<'_> {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> lucene_codecs::points::Relation {
        use lucene_codecs::points::Relation;
        use std::cmp::Ordering::Equal;
        match (self.decode.decode(min), self.decode.decode(max)) {
            (Some(a), Some(b)) => match ((self.side)(a), (self.side)(b)) {
                (Equal, Equal) => Relation::CellInsideQuery,
                (x, y) if x == y => Relation::CellOutsideQuery,
                _ => Relation::CellCrossesQuery,
            },
            _ => Relation::CellCrossesQuery,
        }
    }

    fn visit(&mut self, _doc_id: i32) {
        self.count += 1;
    }

    fn visit_many(&mut self, doc_ids: &[i32]) {
        self.count += doc_ids.len() as u64;
    }

    fn visit_with_value(&mut self, _doc_id: i32, packed: &[u8]) {
        if self
            .decode
            .decode(packed)
            .is_some_and(|v| (self.side)(v) == std::cmp::Ordering::Equal)
        {
            self.count += 1;
        }
    }
}

/// What [`count_from_points`] counted: a `date_histogram`'s non-empty
/// buckets by key, or a `range`'s every range.
enum PointCounts {
    Keyed(Vec<(i64, u64)>),
    Fixed(Vec<u64>),
}

/// At most this many `date_histogram` buckets are counted from the points
/// in one segment; past it the segment is read document by document.
const MAX_POINT_BUCKETS: usize = 1024;

/// A top-level `date_histogram` or `range` without sub-aggregations, counted
/// from a segment's points -- valid when every document of the segment is a
/// match (the caller's check) and the field has one point per document, so
/// that the points hold each document's one value. `None` otherwise.
fn count_from_points(node: &AggNode, seg: &OpenSegment<'_>) -> Result<Option<PointCounts>> {
    let (field, kind) = match node {
        AggNode::DateHistogram { field, subs, .. } if subs.is_empty() => (field, ValueKind::Long),
        AggNode::Range {
            field,
            kind,
            subs,
            ranges,
        } if subs.is_empty() && ranges.iter().all(|r| !r.0.is_nan() && !r.1.is_nan()) => {
            (field, *kind)
        }
        _ => return Ok(None),
    };
    let Some(points) = seg.points else {
        return Ok(None);
    };
    let Some(num) = points.field_number(field) else {
        return Ok(None);
    };
    let Some(pf) = points.reader.field(num) else {
        return Ok(None);
    };
    if pf.num_dims != 1 || i64::from(pf.doc_count) != pf.point_count {
        return Ok(None);
    }
    let Some(decode) = PointDecode::of(kind, pf.bytes_per_dim) else {
        return Ok(None);
    };
    let count = |side: &dyn Fn(i64) -> std::cmp::Ordering| -> Result<u64> {
        let mut v = CountIn {
            decode,
            side,
            count: 0,
        };
        points.reader.intersect(num, &mut v)?;
        Ok(v.count)
    };
    match node {
        AggNode::DateHistogram {
            rounding,
            hard_bounds,
            ..
        } => {
            let (Some(min), Some(max)) = (
                decode.decode(&pf.min_packed_value),
                decode.decode(&pf.max_packed_value),
            ) else {
                return Ok(None);
            };
            let mut out = Vec::new();
            let mut at = min;
            while at <= max {
                let Some((from, to)) = rounding.bucket(at) else {
                    return Ok(None);
                };
                if out.len() >= MAX_POINT_BUCKETS || to <= from {
                    return Ok(None);
                }
                let contained = !hard_bounds.1.is_some_and(|m| from >= m)
                    && !hard_bounds.0.is_some_and(|m| from < m);
                if contained {
                    let n = count(&|v: i64| {
                        if v < from {
                            std::cmp::Ordering::Less
                        } else if v < to {
                            std::cmp::Ordering::Equal
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    })?;
                    if n > 0 {
                        out.push((from, n));
                    }
                }
                at = to;
            }
            Ok(Some(PointCounts::Keyed(out)))
        }
        AggNode::Range { ranges, .. } => {
            let mut out = Vec::with_capacity(ranges.len());
            for &(from, to) in ranges {
                out.push(count(&|v: i64| {
                    let d = aggs::to_double(kind, v);
                    if d.is_nan() {
                        std::cmp::Ordering::Greater
                    } else if d < from {
                        std::cmp::Ordering::Less
                    } else if d < to {
                        std::cmp::Ordering::Equal
                    } else {
                        std::cmp::Ordering::Greater
                    }
                })?);
            }
            Ok(Some(PointCounts::Fixed(out)))
        }
        _ => Ok(None),
    }
}

/// Adds [`count_from_points`]' counts to the top-level state.
fn apply_counts(state: &mut State, counts: PointCounts) {
    match (state, counts) {
        (
            State::Keyed {
                ords, keys, docs, ..
            },
            PointCounts::Keyed(buckets),
        ) => {
            for (key, n) in buckets {
                let ord = keyed_ord(ords, keys, 0, key);
                *slot(docs, ord as usize) += n;
            }
        }
        (State::Fixed { docs, .. }, PointCounts::Fixed(counts)) => {
            for (i, n) in counts.into_iter().enumerate() {
                *slot(docs, i) += n;
            }
        }
        _ => {}
    }
}

/// The documents a pass collects: `min_score` behind the query, or none.
pub struct PassScoring<'a, 'n> {
    pub min_score: &'a aggs::MinScore<'a, 'n>,
}

/// One pass of aggregations: `nodes` over `query`'s live matches in the
/// segments of each slice, a state per slice from scratch (a concurrent
/// search's aggregators, reduced by the caller); slices run concurrently.
///
/// # Errors
/// A column of the wrong kind, an unknown keyword field, a slice naming a
/// segment the reader lacks, or what reading the index reports.
#[allow(clippy::too_many_arguments)]
pub fn aggregate_tree(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    nodes: &[AggNode],
    globals: &Globals<'_>,
    slices: &[Vec<usize>],
    min_score: Option<&aggs::MinScore<'_, '_>>,
) -> Result<Vec<Vec<AggResult>>> {
    let rewritten = crate::multi_segment::rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    let clause = aggs::lone_clause(query);
    let global = match min_score {
        Some(_) => Some(crate::multi_segment::global_boolean_stats(segments, query)?),
        None => None,
    };
    let scoring = min_score.zip(global.as_ref());
    let one = |slice: &[usize]| -> Result<Vec<AggResult>> {
        let mut states: Vec<State> = nodes.iter().map(State::new).collect();
        let mut docs_buf = Vec::new();
        for &i in slice {
            let (Some(seg), Some(reader)) = (segments.get(i), readers.get(i)) else {
                return Err(crate::Error::SliceOutOfRange {
                    segment: i,
                    segments: segments.len().min(readers.len()),
                });
            };
            let view = SegmentView {
                seg,
                reader,
                index: i,
                top: true,
            };
            let mut leaves = Vec::with_capacity(nodes.len());
            for (n, s) in nodes.iter().zip(&mut states) {
                leaves.push(open_leaf(n, &view, globals, s)?);
            }
            let ctx = aggs::plain_context(seg);
            let live: Option<&FixedBitSet> = seg.live_docs;
            let matched = match scoring {
                Some((m, g)) => {
                    let scored = exec::LeafContext {
                        norms: m.norms.get(i).copied().flatten(),
                        global: Some(g),
                        ..ctx
                    };
                    aggs::segment_matches_scoring(&scored, query, live, m.min, &mut docs_buf)?
                }
                None => aggs::segment_matches(&ctx, query, &clause, live, &mut docs_buf)?,
            };
            // OpenSearch's filter rewrite: a top-level `date_histogram` or
            // `range` without sub-aggregations over a segment every document
            // of which matches is counted from the points.
            if scoring.is_none() && seg.live_docs.is_none() && matches!(matched, Some(None)) {
                for ((n, s), l) in nodes.iter().zip(&mut states).zip(&mut leaves) {
                    if let Some(counts) = count_from_points(n, seg)? {
                        apply_counts(s, counts);
                        l.done = true;
                    }
                }
                if leaves.iter().all(|l| l.done) {
                    continue;
                }
            }
            let zeros = vec![0u32; WINDOW as usize];
            let mut live_window: Vec<i32> = Vec::with_capacity(WINDOW as usize);
            let visit_window =
                |states: &mut [State], leaves: &mut [Leaf<'_>], docs: &[i32]| -> Result<()> {
                    for ((n, s), l) in nodes.iter().zip(states).zip(leaves) {
                        if l.done {
                        } else if no_windows() {
                            for &doc in docs {
                                collect(n, s, l, doc, 0)?;
                            }
                        } else {
                            collect_window(n, s, l, docs, &zeros[..docs.len()])?;
                        }
                    }
                    Ok(())
                };
            // Windows of documents, each column read ahead for the window
            // where the matches are dense enough to repay reading every
            // document's value (one in eight).
            let window = |leaves: &mut [Leaf<'_>], base: i32, end: i32, matches: usize| {
                // ARITH: `base < end <= max_doc`.
                #[allow(clippy::arithmetic_side_effects)]
                let len = (end - base) as usize;
                for l in leaves.iter_mut().filter(|l| !l.done) {
                    if matches.saturating_mul(8) >= len && !no_windows() {
                        l.prefetch(base, len)?;
                    } else {
                        l.drop_prefetch();
                    }
                }
                Ok::<(), crate::Error>(())
            };
            let max_doc = reader.max_doc;
            match matched {
                None => {}
                Some(Some(docs)) => {
                    let mut i = 0;
                    while let Some(&first) = docs.get(i) {
                        // Windows aligned as the whole-segment ones are, so
                        // a column's blocks can be read in one piece.
                        let base = first - first.rem_euclid(WINDOW);
                        let end = base.saturating_add(WINDOW).min(max_doc).max(first + 1);
                        let j = i + docs[i..].partition_point(|&d| d < end);
                        window(&mut leaves, base, end, j - i)?;
                        visit_window(&mut states, &mut leaves, &docs[i..j])?;
                        i = j;
                    }
                }
                Some(None) => {
                    let mut base = 0;
                    while base < max_doc {
                        let end = base.saturating_add(WINDOW).min(max_doc);
                        window(&mut leaves, base, end, (end - base) as usize)?;
                        live_window.clear();
                        live_window
                            .extend((base..end).filter(|&d| live.is_none_or(|l| l.get_doc(d))));
                        visit_window(&mut states, &mut leaves, &live_window)?;
                        base = end;
                    }
                }
            }
        }
        let mut terms = TermLookup::new(readers);
        nodes
            .iter()
            .zip(states)
            .map(|(n, s)| finish(n, s, &[0], globals, &mut terms))
            .collect()
    };
    let parallel =
        crate::slices::estimated_matches(segments, query) >= crate::slices::SEQUENTIAL_BELOW;
    crate::slices::run_slices_if(parallel, slices, one)
        .into_iter()
        .collect()
}

/// A global ordinal's term, read from the first segment holding it.
struct TermLookup<'r> {
    readers: &'r [SegmentReader],
    dicts: HashMap<(String, usize), Option<TermsDict<'r>>>,
}

impl<'r> TermLookup<'r> {
    fn new(readers: &'r [SegmentReader]) -> Self {
        TermLookup {
            readers,
            dicts: HashMap::new(),
        }
    }

    fn term(&mut self, global: &GlobalOrds, field: &str, g: i64) -> Result<Vec<u8>> {
        let (Some(seg), Some(ord)) = (global.first_segment(g), global.first_segment_ord(g)) else {
            return Err(crate::Error::TermsAggType(format!(
                "{field}: global ordinal {g} has no segment"
            )));
        };
        let readers = self.readers;
        let dict = match self.dicts.entry((field.to_string(), seg)) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let d = match readers.get(seg) {
                    Some(r) => crate::terms_agg::open_ords(r, field)?.1,
                    None => None,
                };
                e.insert(d)
            }
        };
        let Some(dict) = dict.as_mut() else {
            return Err(crate::Error::TermsAggType(format!(
                "{field}: segment {seg} has no dictionary"
            )));
        };
        Ok(dict
            .seek_ord(ord)
            .map_err(|e| crate::Error::from(lucene_codecs::doc_values::Error::from(e)))?
            .to_vec())
    }
}

/// `buildAggregations(owningBucketOrds)`: the results for the owning buckets
/// `owners` (in the parent's listed order).
fn finish(
    node: &AggNode,
    state: State,
    owners: &[u32],
    globals: &Globals<'_>,
    terms: &mut TermLookup<'_>,
) -> Result<AggResult> {
    let at = |v: &[u64], i: usize| v.get(i).copied().unwrap_or(0);
    match (node, state) {
        (AggNode::Metric { .. }, State::Metric(states)) => Ok(AggResult::Metric(
            owners
                .iter()
                .map(|&o| states.get(o as usize).copied().unwrap_or_default())
                .collect(),
        )),
        (
            AggNode::Cardinality {
                field, precision, ..
            },
            State::OrdBits(bits),
        ) => {
            // Each owner's global ordinals in ascending -- term -- order.
            let ords_of = |o: u32| -> Vec<i64> {
                let mut out = Vec::new();
                if let Some(words) = bits.get(o as usize) {
                    for (w, &word) in words.iter().enumerate() {
                        let mut word = word;
                        while word != 0 {
                            out.push((w * 64 + word.trailing_zeros() as usize) as i64);
                            word &= word - 1;
                        }
                    }
                }
                out
            };
            let g = globals.get(field)?;
            match precision {
                Some(p) => {
                    let mut out = Vec::with_capacity(owners.len());
                    for &o in owners {
                        let values = ords_of(o);
                        if values.is_empty() {
                            out.push(Vec::new());
                            continue;
                        }
                        let mut sketch =
                            crate::cardinality_sketch::Sketch::new(*p).ok_or_else(|| {
                                crate::Error::TermsAggType(format!("cardinality precision {p}"))
                            })?;
                        for &v in &values {
                            let term = terms.term(g, field, v)?;
                            sketch.collect(crate::cardinality_sketch::murmur3_h1(&term, 0));
                        }
                        let mut bytes = Vec::new();
                        sketch.write_to(&mut bytes);
                        out.push(bytes);
                    }
                    Ok(AggResult::CardinalitySketch(out))
                }
                None => {
                    let mut out = Vec::with_capacity(owners.len());
                    for &o in owners {
                        out.push(
                            ords_of(o)
                                .into_iter()
                                .map(|v| terms.term(g, field, v).map(CardinalityValue::Term))
                                .collect::<Result<Vec<_>>>()?,
                        );
                    }
                    Ok(AggResult::Cardinality(out))
                }
            }
        }
        (AggNode::Cardinality { .. }, State::Sketches { sketches, .. }) => {
            Ok(AggResult::CardinalitySketch(
                owners
                    .iter()
                    .map(|&o| {
                        let mut bytes = Vec::new();
                        if let Some(Some(s)) = sketches.get(o as usize) {
                            s.write_to(&mut bytes);
                        }
                        bytes
                    })
                    .collect(),
            ))
        }
        // A numeric one without a precision: its distinct values, ascending.
        (AggNode::Cardinality { .. }, State::Cardinality(sets)) => {
            let mut out = Vec::with_capacity(owners.len());
            for &o in owners {
                let Some(set) = sets.get(o as usize) else {
                    out.push(Vec::new());
                    continue;
                };
                let mut values: Vec<i64> = set.iter().copied().collect();
                values.sort_unstable();
                out.push(values.into_iter().map(CardinalityValue::Long).collect());
            }
            Ok(AggResult::Cardinality(out))
        }
        (
            AggNode::Terms {
                field,
                shard_size,
                subs,
            },
            State::Keyed {
                keys,
                docs,
                subs: sub_states,
                ..
            },
        ) => {
            let g = globals.get(field)?;
            let by_owner = group(&keys);
            let mut buckets = Vec::with_capacity(owners.len());
            let mut child = Vec::new();
            for &o in owners {
                let mut kept: Vec<(i64, u64, u32)> = by_owner
                    .get(&o)
                    .map(|ords| {
                        ords.iter()
                            .map(|&b| (keys[b as usize].1, at(&docs, b as usize), b))
                            .collect()
                    })
                    .unwrap_or_default();
                let total: u64 = kept.iter().map(|k| k.1).sum();
                kept.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                kept.truncate(*shard_size);
                let kept_docs: u64 = kept.iter().map(|k| k.1).sum();
                kept.sort_unstable_by_key(|k| k.0);
                let mut list = Vec::with_capacity(kept.len());
                for (term, n, b) in kept {
                    list.push((terms.term(g, field, term)?, n));
                    child.push(b);
                }
                buckets.push((total - kept_docs, list));
            }
            let subs = finish_subs(subs, sub_states, &child, globals, terms)?;
            Ok(AggResult::Terms { buckets, subs })
        }
        (
            AggNode::Histogram { subs, .. },
            State::Keyed {
                keys,
                docs,
                subs: sub_states,
                ..
            },
        ) => {
            let by_owner = group(&keys);
            let mut buckets = Vec::with_capacity(owners.len());
            let mut child = Vec::new();
            for &o in owners {
                let mut list: Vec<(f64, u64, u32)> = by_owner
                    .get(&o)
                    .map(|ords| {
                        ords.iter()
                            .map(|&b| {
                                let key = f64::from_bits(keys[b as usize].1 as u64);
                                (key, at(&docs, b as usize), b)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                list.sort_by(|a, b| a.0.total_cmp(&b.0));
                child.extend(list.iter().map(|k| k.2));
                buckets.push(list.into_iter().map(|(k, n, _)| (k, n)).collect());
            }
            let subs = finish_subs(subs, sub_states, &child, globals, terms)?;
            Ok(AggResult::Histogram { buckets, subs })
        }
        (
            AggNode::DateHistogram { subs, .. },
            State::Keyed {
                keys,
                docs,
                subs: sub_states,
                ..
            },
        ) => {
            let by_owner = group(&keys);
            let mut buckets = Vec::with_capacity(owners.len());
            let mut child = Vec::new();
            for &o in owners {
                let mut list: Vec<(i64, u64, u32)> = by_owner
                    .get(&o)
                    .map(|ords| {
                        ords.iter()
                            .map(|&b| (keys[b as usize].1, at(&docs, b as usize), b))
                            .collect()
                    })
                    .unwrap_or_default();
                list.sort_by_key(|k| k.0);
                child.extend(list.iter().map(|k| k.2));
                buckets.push(list.into_iter().map(|(k, n, _)| (k, n)).collect());
            }
            let subs = finish_subs(subs, sub_states, &child, globals, terms)?;
            Ok(AggResult::DateHistogram { buckets, subs })
        }
        (
            AggNode::Range { subs, .. } | AggNode::Filters { subs, .. } | AggNode::Global { subs },
            State::Fixed {
                width,
                docs,
                subs: sub_states,
            },
        ) => {
            let mut out = Vec::with_capacity(owners.len() * width);
            let mut child = Vec::with_capacity(owners.len() * width);
            for &o in owners {
                for i in 0..width {
                    let b = o as usize * width + i;
                    out.push(at(&docs, b));
                    child.push(b as u32);
                }
            }
            let subs = finish_subs(subs, sub_states, &child, globals, terms)?;
            Ok(AggResult::Fixed {
                width,
                docs: out,
                subs,
            })
        }
        _ => Err(crate::Error::TermsAggType(
            "aggregation state does not match its node".to_string(),
        )),
    }
}

fn finish_subs(
    nodes: &[AggNode],
    states: Vec<State>,
    owners: &[u32],
    globals: &Globals<'_>,
    terms: &mut TermLookup<'_>,
) -> Result<Vec<AggResult>> {
    nodes
        .iter()
        .zip(states)
        .map(|(n, s)| finish(n, s, owners, globals, terms))
        .collect()
}

/// A keyed aggregation's buckets by owning bucket, each in ordinal order.
fn group(keys: &[(u32, i64)]) -> HashMap<u32, Vec<u32>> {
    let mut out: HashMap<u32, Vec<u32>> = HashMap::new();
    for (b, &(o, _)) in keys.iter().enumerate() {
        out.entry(o).or_default().push(b as u32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hasher's byte and word paths agree with its per-word mixing, and
    /// the month tables are `DateUtilsRounding`'s (checked at run time too:
    /// the constants are built by a `const fn`).
    #[test]
    fn the_fx_hasher_and_month_tables() {
        let mut a = Fx::default();
        a.write(&[1, 2]);
        let mut b = Fx::default();
        b.write_u64(1);
        b.write_u64(2);
        assert_eq!(a.finish(), b.finish());
        let mut c = Fx::default();
        c.write_u32(7);
        let mut d = Fx::default();
        d.write_i64(7);
        assert_eq!(c.finish(), d.finish());
        assert_eq!(month_starts(false), MIN_TOTAL_MILLIS_BY_MONTH);
        assert_eq!(month_starts(true), MAX_TOTAL_MILLIS_BY_MONTH);
        assert_eq!(
            month_starts(true)[2] - month_starts(false)[2],
            MILLIS_PER_DAY
        );
    }
    use crate::aggs::{MetricSpec, NEED_ALL};
    use crate::directory_reader::DirectoryReader;
    use crate::query::{Clause, MatchAllDocsQuery, PointsRangeQuery, TermQuery};
    use lucene_store::FsDirectory;

    const DAY: i64 = MILLIS_PER_DAY;

    /// Days since 1970-01-01 of a proleptic Gregorian date (Hinnant's
    /// `days_from_civil`): an oracle independent of the Joda-style port.
    fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
        let y = if m <= 2 { y - 1 } else { y };
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let mp = (m + 9) % 12;
        let doy = (153 * mp + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    /// `(year, month)` of a day count (`civil_from_days`).
    fn civil_from_days(z: i64) -> (i64, i64) {
        let z = z + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        (yoe + era * 400 + i64::from(m <= 2), m)
    }

    #[test]
    fn calendar_units_round_as_opensearch_does() {
        let t = 1_700_000_000_000; // 2023-11-14T22:13:20Z, a Tuesday
        let r = |unit| DateRounding {
            kind: RoundingKind::Unit(unit),
            zone_ms: 0,
            offset: 0,
        };
        assert_eq!(r(DateUnit::Second).round(t), t);
        assert_eq!(r(DateUnit::Minute).round(t), 1_699_999_980_000);
        assert_eq!(r(DateUnit::Hour).round(t), 1_699_999_200_000);
        assert_eq!(r(DateUnit::Day).round(t), 1_699_920_000_000);
        assert_eq!(r(DateUnit::Week).round(t), 1_699_833_600_000, "Monday 13th");
        assert_eq!(r(DateUnit::Month).round(t), 1_698_796_800_000);
        assert_eq!(r(DateUnit::Quarter).round(t), 1_696_118_400_000);
        assert_eq!(r(DateUnit::Year).round(t), 1_672_531_200_000);
        // Before the epoch, and across a leap day.
        assert_eq!(r(DateUnit::Day).round(-1), -DAY);
        assert_eq!(r(DateUnit::Second).round(-1), -1_000);
        assert_eq!(r(DateUnit::Year).round(-1), -365 * DAY);
        assert_eq!(r(DateUnit::Month).round(-1), -31 * DAY);
        assert_eq!(
            r(DateUnit::Week).round(0),
            -3 * DAY,
            "1970-01-01 was a Thursday"
        );
        let leap = days_from_civil(2024, 2, 29) * DAY + 43_200_000;
        assert_eq!(
            r(DateUnit::Month).round(leap),
            days_from_civil(2024, 2, 1) * DAY
        );
        // Every unit against the oracle, over four centuries either side.
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        for _ in 0..20_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let t = (x % (800 * 366 * DAY as u64)) as i64 - 400 * 366 * DAY;
            let days = t.div_euclid(DAY);
            let (y, m) = civil_from_days(days);
            assert_eq!(r(DateUnit::Day).round(t), days * DAY, "{t}");
            assert_eq!(
                r(DateUnit::Month).round(t),
                days_from_civil(y, m, 1) * DAY,
                "{t}"
            );
            let q = (m - 1) / 3 * 3 + 1;
            assert_eq!(
                r(DateUnit::Quarter).round(t),
                days_from_civil(y, q, 1) * DAY,
                "{t}"
            );
            assert_eq!(
                r(DateUnit::Year).round(t),
                days_from_civil(y, 1, 1) * DAY,
                "{t}"
            );
            // Monday on or before: 1970-01-05 was a Monday.
            let monday = days - (days - 4).rem_euclid(7);
            assert_eq!(r(DateUnit::Week).round(t), monday * DAY, "{t}");
        }
    }

    #[test]
    fn intervals_zones_and_offsets_round_as_opensearch_does() {
        let t = 1_700_000_000_000;
        // A fixed interval of 90 minutes, 17 minutes in.
        let fixed = DateRounding {
            kind: RoundingKind::Interval(90 * 60_000),
            zone_ms: 0,
            offset: 17 * 60_000,
        };
        let v = t - 17 * 60_000;
        assert_eq!(fixed.round(t), v - v.rem_euclid(90 * 60_000) + 17 * 60_000);
        assert_eq!(fixed.round(fixed.round(t)), fixed.round(t));
        // Negative values round down, not toward zero.
        let neg = DateRounding {
            kind: RoundingKind::Interval(1_000),
            zone_ms: 0,
            offset: 0,
        };
        assert_eq!(neg.round(-1), -1_000);
        assert_eq!(neg.round(-1_000), -1_000);
        // A day in +05:30 starts at 18:30 UTC the day before.
        let india = DateRounding {
            kind: RoundingKind::Unit(DateUnit::Day),
            zone_ms: 19_800_000,
            offset: 0,
        };
        assert_eq!(india.round(t), 1_699_986_600_000);
        let hour = DateRounding {
            kind: RoundingKind::Unit(DateUnit::Hour),
            zone_ms: 19_800_000,
            offset: 0,
        };
        assert_eq!(hour.round(t), 1_699_999_200_000 - 1_800_000);
    }

    #[test]
    fn a_bucket_is_the_run_of_values_rounding_the_same() {
        let mut x: u64 = 0x2545_f491_4f6c_dd1d;
        let kinds = [
            RoundingKind::Unit(DateUnit::Week),
            RoundingKind::Unit(DateUnit::Year),
            RoundingKind::Unit(DateUnit::Quarter),
            RoundingKind::Unit(DateUnit::Month),
            RoundingKind::Unit(DateUnit::Day),
            RoundingKind::Unit(DateUnit::Hour),
            RoundingKind::Unit(DateUnit::Minute),
            RoundingKind::Unit(DateUnit::Second),
            RoundingKind::Interval(7_777),
        ];
        for kind in kinds {
            for (zone_ms, offset) in [(0, 0), (19_800_000, 0), (-3_600_000, 17 * 60_000)] {
                let r = DateRounding {
                    kind,
                    zone_ms,
                    offset,
                };
                for _ in 0..500 {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    let t = (x % (400 * 366 * DAY as u64)) as i64 - 200 * 366 * DAY;
                    let (from, to) = r.bucket(t).unwrap();
                    assert_eq!(from, r.round(t), "{kind:?} {t}");
                    assert!(from <= t && t < to, "{kind:?} {t}");
                    assert_eq!(r.round(to - 1), from, "{kind:?} {t}: the last value in it");
                    assert_eq!(
                        r.round(to),
                        to,
                        "{kind:?} {t}: the next bucket starts at to"
                    );
                }
            }
        }
        // At the edges of the range there is no bucket to name.
        let day = DateRounding {
            kind: RoundingKind::Unit(DateUnit::Day),
            zone_ms: 0,
            offset: 0,
        };
        assert_eq!(day.bucket(i64::MAX - 5), None);
        let far = DateRounding {
            kind: RoundingKind::Unit(DateUnit::Day),
            zone_ms: 5,
            offset: i64::MIN,
        };
        assert_eq!(far.bucket(0), None);
    }

    #[test]
    fn matched_ranges_are_the_candidates_and_nothing_else() {
        let ranges = [
            (f64::NEG_INFINITY, 0.0),
            (0.0, 100.0),
            (50.0, 250.0),
            (250.0, f64::INFINITY),
        ];
        let mt = max_to(&ranges);
        assert_eq!(mt, vec![0.0, 100.0, 250.0, f64::INFINITY]);
        let hits = |v: f64, low: usize| {
            let (s, e) = matched_range(&ranges, low, v, &mt);
            (s..e)
                .filter(|&i| v >= ranges[i].0 && v < ranges[i].1)
                .collect::<Vec<_>>()
        };
        assert_eq!(hits(-5.0, 0), vec![0]);
        assert_eq!(hits(0.0, 0), vec![1]);
        assert_eq!(hits(75.0, 0), vec![1, 2]);
        assert_eq!(hits(75.0, 2), vec![2], "searching on from a previous value");
        assert_eq!(hits(250.0, 0), vec![3]);
        assert_eq!(hits(f64::NAN, 0), Vec::<usize>::new());
        // Nothing can hold a value past every range.
        let bounded = [(0.0, 1.0), (2.0, 3.0)];
        let bt = max_to(&bounded);
        assert_eq!(matched_range(&bounded, 0, 5.0, &bt), (2, 2));
        assert_eq!(matched_range(&bounded, 0, -5.0, &bt), (0, 0));
        assert_eq!(matched_range(&bounded, 0, 1.5, &bt), (1, 1));
    }

    /// A state's parts as bits (`NaN` equal to itself).
    fn bits(s: &MetricState) -> [u64; 7] {
        [
            s.count,
            s.sum.to_bits(),
            s.delta.to_bits(),
            s.min.to_bits(),
            s.max.to_bits(),
            s.min_of_mins.to_bits(),
            s.max_of_maxes.to_bits(),
        ]
    }

    fn fixture(name: &str) -> DirectoryReader {
        let dir = format!("{}/../../fixtures/data/{name}", env!("CARGO_MANIFEST_DIR"));
        DirectoryReader::open(&FsDirectory::open(dir)).expect("open fixture")
    }

    fn all() -> BooleanQuery {
        BooleanQuery {
            must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(0))],
            ..Default::default()
        }
    }

    fn body(term: &str) -> BooleanQuery {
        BooleanQuery {
            must: vec![Clause::Term(TermQuery::new(
                "body",
                term.as_bytes().to_vec(),
            ))],
            ..Default::default()
        }
    }

    fn metric(field: &str, kind: ValueKind) -> AggNode {
        AggNode::Metric {
            field: field.to_string(),
            kind,
            source: Source::DocValues,
            needs: NEED_ALL,
        }
    }

    /// Every live document matching `query` with its values of `field`,
    /// segment by segment: the oracle the bucketed results are held to.
    fn scan(
        reader: &DirectoryReader,
        query: &BooleanQuery,
        field: &str,
    ) -> Vec<(usize, i32, Vec<i64>)> {
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let mut out = Vec::new();
        let mut buf = Vec::new();
        for (i, (seg, r)) in segments.iter().zip(reader.segment_readers()).enumerate() {
            let clause = aggs::lone_clause(query);
            let docs: Vec<i32> = match aggs::segment_matches(
                &aggs::plain_context(seg),
                query,
                &clause,
                seg.live_docs,
                &mut buf,
            )
            .unwrap()
            {
                None => Vec::new(),
                Some(Some(d)) => d.to_vec(),
                Some(None) => (0..r.max_doc)
                    .filter(|&d| seg.live_docs.is_none_or(|l| l.get_doc(d)))
                    .collect(),
            };
            let mut values = aggs::open_values(r, field).unwrap();
            for doc in docs {
                let mut v = Vec::new();
                match &mut values {
                    Values::Absent => {}
                    Values::Single(c) => v.extend(c.value(doc).unwrap()),
                    Values::Multi(c) => c.values(doc, &mut v).unwrap(),
                }
                out.push((i, doc, v));
            }
        }
        out
    }

    fn run(
        reader: &DirectoryReader,
        query: &BooleanQuery,
        nodes: &[AggNode],
        slices: &[Vec<usize>],
    ) -> Result<Vec<Vec<AggResult>>> {
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let mut fields = Vec::new();
        for n in nodes {
            n.keyword_fields(&mut fields);
        }
        let ords: HashMap<String, Arc<GlobalOrds>> = fields
            .into_iter()
            .map(|f| (f.to_string(), reader.global_ords(f).unwrap()))
            .collect();
        aggregate_tree(
            &segments,
            reader.segment_readers(),
            query,
            nodes,
            &Globals { ords: &ords },
            slices,
            None,
        )
    }

    fn whole(reader: &DirectoryReader) -> Vec<Vec<usize>> {
        vec![(0..reader.segment_readers().len()).collect()]
    }

    /// The filter rewrite counts from points only what it can count
    /// exactly: a top-level `date_histogram` or `range` without
    /// sub-aggregations over a single-valued, one-dimensional points field
    /// of the node's value type; anything else is left to the scan.
    #[test]
    fn counting_from_points_declines_what_it_cannot_count_exactly() {
        let reader = fixture("metric_aggs_index");
        let range = |field: &str, kind, subs: Vec<AggNode>| AggNode::Range {
            field: field.to_string(),
            kind,
            ranges: vec![(f64::NEG_INFINITY, 0.0), (0.0, f64::INFINITY)],
            subs,
        };
        let mut plain = reader.open_segments().unwrap();
        {
            let segments = plain.as_open_segments();
            // No points opened for the segment at all.
            let r = range("d", ValueKind::Double, vec![]);
            assert!(count_from_points(&r, &segments[0]).unwrap().is_none());
        }
        plain.open_points().unwrap();
        let segments = plain.as_open_segments();
        let seg = &segments[0];
        let terms = AggNode::Terms {
            field: "d".to_string(),
            shard_size: 1,
            subs: vec![],
        };
        let declined = [
            terms,
            range(
                "d",
                ValueKind::Double,
                vec![AggNode::Global { subs: vec![] }],
            ),
            AggNode::Range {
                field: "d".to_string(),
                kind: ValueKind::Double,
                ranges: vec![(f64::NAN, 1.0)],
                subs: vec![],
            },
            range("no-such-field", ValueKind::Double, vec![]),
            // Multi-valued: more points than documents.
            range("md", ValueKind::Double, vec![]),
            // A double field read as a float: the widths disagree.
            range("d", ValueKind::Float, vec![]),
        ];
        for node in &declined {
            assert!(count_from_points(node, seg).unwrap().is_none(), "{node:?}");
        }
        assert!(
            count_from_points(&range("d", ValueKind::Double, vec![]), seg)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn a_match_all_filter_and_global_see_what_the_metrics_see() {
        let reader = fixture("metric_aggs_index");
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        for q in [all(), body("a"), body("b")] {
            let specs: Vec<MetricSpec> = [
                ("d", ValueKind::Double),
                ("md", ValueKind::Double),
                ("ml", ValueKind::Long),
                ("f", ValueKind::Float),
            ]
            .iter()
            .map(|&(f, kind)| MetricSpec {
                field: f.to_string(),
                kind,
                source: Source::DocValues,
                needs: NEED_ALL,
            })
            .collect();
            let want = crate::aggs::metric_states(&segments, reader.segment_readers(), &q, &specs)
                .unwrap();
            let subs: Vec<AggNode> = specs.iter().map(|s| metric(&s.field, s.kind)).collect();
            let nodes = vec![
                AggNode::Filters {
                    filters: vec![all()],
                    other: true,
                    subs: subs.clone(),
                },
                AggNode::Global { subs: subs.clone() },
            ];
            let got = run(&reader, &q, &nodes, &whole(&reader)).unwrap().remove(0);
            let AggResult::Fixed {
                width: 2,
                docs,
                subs: f,
            } = &got[0]
            else {
                panic!("{got:?}")
            };
            let matched = scan(&reader, &q, "d").len() as u64;
            assert_eq!(
                docs,
                &vec![matched, 0],
                "every match in the filter, none in other"
            );
            for (i, s) in f.iter().enumerate() {
                let AggResult::Metric(states) = s else {
                    panic!()
                };
                assert_eq!(
                    bits(&states[0]),
                    bits(&want[i]),
                    "{i}: bit for bit, the documents in the same order"
                );
                assert_eq!(states[1], MetricState::default(), "the empty other bucket");
            }
            let AggResult::Fixed {
                width: 1,
                docs,
                subs: g,
            } = &got[1]
            else {
                panic!()
            };
            assert_eq!(docs, &vec![matched]);
            for (i, s) in g.iter().enumerate() {
                let AggResult::Metric(g) = s else { panic!() };
                assert_eq!(bits(&g[0]), bits(&want[i]));
            }
        }
    }

    #[test]
    fn histograms_and_ranges_bucket_what_a_scan_buckets() {
        let reader = fixture("metric_aggs_index");
        for (q, field, kind) in [
            (all(), "d", ValueKind::Double),
            (body("a"), "md", ValueKind::Double),
            (all(), "ml", ValueKind::Long),
            (body("b"), "f", ValueKind::Float),
            (all(), "i", ValueKind::Long),
        ] {
            let rows = scan(&reader, &q, field);
            for (interval, offset, bounds) in [
                (10.0, 0.0, (None, None)),
                (3.5, 1.25, (Some(-20.0), Some(40.0))),
                (1e9, 0.0, (None, None)),
            ] {
                let node = AggNode::Histogram {
                    field: field.to_string(),
                    kind,
                    interval,
                    offset,
                    hard_bounds: bounds,
                    subs: vec![
                        metric("ml", ValueKind::Long),
                        AggNode::Cardinality {
                            field: field.to_string(),
                            kind: CardinalityKind::Numeric(kind),
                            precision: None,
                        },
                    ],
                };
                // The oracle: each distinct key of a document's values once.
                let mut want: std::collections::BTreeMap<i64, u64> =
                    std::collections::BTreeMap::new();
                let mut distinct: HashMap<i64, HashSet<i64>> = HashMap::new();
                for (_, _, values) in &rows {
                    // Java's loop: a key equal to the previous one -- which
                    // starts at -Infinity, so a -Infinity value never counts,
                    // and moves on past out-of-bounds keys too -- is skipped.
                    let mut previous = f64::NEG_INFINITY;
                    for &v in values {
                        let k = ((aggs::to_double(kind, v) - offset) / interval).floor();
                        if k == previous {
                            continue;
                        }
                        previous = k;
                        let b = k * interval;
                        if bounds.0.is_some_and(|m| b < m) || bounds.1.is_some_and(|m| b > m) {
                            continue;
                        }
                        *want.entry(java_double_bits(k)).or_default() += 1;
                        let set = distinct.entry(java_double_bits(k)).or_default();
                        // A whole number field's longs as they are, not widened.
                        for &v in values {
                            set.insert(match kind {
                                ValueKind::Long => v,
                                _ => java_double_bits(aggs::to_double(kind, v)),
                            });
                        }
                    }
                }
                let got = run(&reader, &q, std::slice::from_ref(&node), &whole(&reader))
                    .unwrap()
                    .remove(0)
                    .remove(0);
                let AggResult::Histogram { buckets, subs } = got else {
                    panic!()
                };
                let mut got_counts: Vec<(i64, u64)> = buckets[0]
                    .iter()
                    .map(|&(k, n)| (java_double_bits(k), n))
                    .collect();
                got_counts.sort_unstable();
                let mut want_counts: Vec<(i64, u64)> = want.into_iter().collect();
                want_counts.sort_unstable();
                assert_eq!(got_counts, want_counts, "{field} / {interval}");
                assert!(
                    buckets[0]
                        .windows(2)
                        .all(|w| w[0].0.total_cmp(&w[1].0).is_lt()),
                    "listed by key"
                );
                let AggResult::Cardinality(sets) = &subs[1] else {
                    panic!()
                };
                for ((k, _), values) in buckets[0].iter().zip(sets) {
                    assert_eq!(values.len(), distinct[&java_double_bits(*k)].len());
                }
            }
            // Ranges: overlapping, open-ended, one past every value.
            let ranges = vec![
                (f64::NEG_INFINITY, 0.0),
                (-5.0, 5.0),
                (0.0, 1e9),
                (1e9, f64::INFINITY),
            ];
            let node = AggNode::Range {
                field: field.to_string(),
                kind,
                ranges: ranges.clone(),
                subs: vec![],
            };
            let mut want = vec![0u64; ranges.len()];
            for (_, _, values) in &rows {
                for (i, &(from, to)) in ranges.iter().enumerate() {
                    if values.iter().any(|&v| {
                        let d = aggs::to_double(kind, v);
                        d >= from && d < to
                    }) {
                        want[i] += 1;
                    }
                }
            }
            let got = run(&reader, &q, &[node], &whole(&reader))
                .unwrap()
                .remove(0)
                .remove(0);
            assert_eq!(
                got,
                AggResult::Fixed {
                    width: 4,
                    docs: want,
                    subs: vec![]
                },
                "{field}"
            );
        }
    }

    #[test]
    fn date_histograms_bucket_each_rounded_value_once() {
        let reader = fixture("metric_aggs_index");
        // `l` has one point per document: its segment without deletions is
        // counted from the points; `ml`, multi-valued, never is.
        for field in ["ml", "l"] {
            let rows = scan(&reader, &all(), field);
            for kind in [
                RoundingKind::Interval(7),
                RoundingKind::Unit(DateUnit::Second),
                RoundingKind::Interval(1),
                // Few enough buckets to count from the points.
                RoundingKind::Interval(1 << 56),
                RoundingKind::Unit(DateUnit::Year),
            ] {
                let rounding = DateRounding {
                    kind,
                    zone_ms: 0,
                    offset: 3,
                };
                for bounds in [(None, None), (Some(0), Some(500))] {
                    let subs = if field == "l" {
                        vec![]
                    } else {
                        vec![metric("d", ValueKind::Double)]
                    };
                    let node = AggNode::DateHistogram {
                        field: field.to_string(),
                        rounding,
                        hard_bounds: bounds,
                        subs,
                    };
                    let mut want: std::collections::BTreeMap<i64, u64> =
                        std::collections::BTreeMap::new();
                    for (_, _, values) in &rows {
                        let mut keys: Vec<i64> =
                            values.iter().map(|&v| rounding.round(v)).collect();
                        keys.dedup();
                        for k in keys {
                            if bounds.0.is_some_and(|m| k < m) || bounds.1.is_some_and(|m| k >= m) {
                                continue;
                            }
                            *want.entry(k).or_default() += 1;
                        }
                    }
                    let got = run(&reader, &all(), &[node], &whole(&reader))
                        .unwrap()
                        .remove(0)
                        .remove(0);
                    let AggResult::DateHistogram { buckets, subs } = got else {
                        panic!()
                    };
                    assert_eq!(buckets[0], want.into_iter().collect::<Vec<_>>());
                    if let Some(AggResult::Metric(states)) = subs.first() {
                        assert_eq!(states.len(), buckets[0].len());
                    }
                }
            }
        }
    }

    /// Reading columns a window at a time and collecting a window per node
    /// (`collect_window`) answers what document-at-a-time collection does:
    /// nested buckets with metrics, a cardinality and multi-valued columns,
    /// over every document (windows) and a sparse query (documents one by
    /// one), with sparse single-valued columns and `-Infinity` values.
    #[test]
    fn windows_answer_what_documents_one_at_a_time_answer() {
        let reader = fixture("metric_aggs_index");
        let rounding = DateRounding {
            kind: RoundingKind::Interval(7),
            zone_ms: 0,
            offset: 3,
        };
        let date = |field: &str, subs| AggNode::DateHistogram {
            field: field.to_string(),
            rounding,
            hard_bounds: (None, None),
            subs,
        };
        let histogram = |field: &str, kind, subs| AggNode::Histogram {
            field: field.to_string(),
            kind,
            interval: 3.5,
            offset: 1.25,
            hard_bounds: (Some(-20.0), Some(40.0)),
            subs,
        };
        let trees = vec![
            date(
                "l",
                vec![
                    metric("d", ValueKind::Double),
                    metric("ml", ValueKind::Long),
                ],
            ),
            histogram(
                "d",
                ValueKind::Double,
                vec![
                    date("i", vec![metric("f", ValueKind::Float)]),
                    AggNode::Cardinality {
                        field: "e".to_string(),
                        kind: CardinalityKind::Numeric(ValueKind::Double),
                        precision: None,
                    },
                ],
            ),
            histogram(
                "e",
                ValueKind::Double,
                vec![
                    metric("l", ValueKind::Long),
                    AggNode::Cardinality {
                        field: "d".to_string(),
                        kind: CardinalityKind::Numeric(ValueKind::Double),
                        precision: Some(5),
                    },
                    AggNode::Cardinality {
                        field: "ml".to_string(),
                        kind: CardinalityKind::Numeric(ValueKind::Long),
                        precision: Some(14),
                    },
                ],
            ),
            histogram("md", ValueKind::Double, vec![metric("i", ValueKind::Long)]),
            AggNode::Range {
                field: "d".to_string(),
                kind: ValueKind::Double,
                ranges: vec![(f64::NEG_INFINITY, 0.0), (0.0, 10.0), (5.0, f64::INFINITY)],
                subs: vec![date("l", vec![metric("e", ValueKind::Double)])],
            },
            metric("f", ValueKind::Float),
            AggNode::Global {
                subs: vec![
                    metric("l", ValueKind::Long),
                    date("l", vec![metric("d", ValueKind::Double)]),
                ],
            },
        ];
        let both = |reader: &DirectoryReader, q: &BooleanQuery, tree: &AggNode| {
            let nodes = std::slice::from_ref(tree);
            NO_WINDOWS.with(|w| w.set(true));
            let one = run(reader, q, nodes, &whole(reader)).unwrap();
            NO_WINDOWS.with(|w| w.set(false));
            let windowed = run(reader, q, nodes, &whole(reader)).unwrap();
            assert_eq!(format!("{windowed:?}"), format!("{one:?}"), "{tree:?}");
        };
        for q in [all(), body("a"), body("b")] {
            for tree in &trees {
                both(&reader, &q, tree);
            }
        }
        // Keyword columns: single- and multi-valued ordinals, as terms and as
        // cardinalities, with and without a precision.
        let reader = fixture("terms_aggs_index");
        for (field, other) in [("kw", "mkw"), ("mkw", "kw")] {
            let tree = AggNode::Terms {
                field: field.to_string(),
                shard_size: 5,
                subs: vec![
                    AggNode::Cardinality {
                        field: other.to_string(),
                        kind: CardinalityKind::Keyword,
                        precision: None,
                    },
                    AggNode::Cardinality {
                        field: other.to_string(),
                        kind: CardinalityKind::Keyword,
                        precision: Some(12),
                    },
                ],
            };
            for q in [all(), body("a")] {
                both(&reader, &q, &tree);
            }
        }
    }

    /// `SortedNumericReader::fill_window` is `values` document by document,
    /// on the fixtures' multi-valued columns (numbers and keyword ordinals,
    /// dense and sparse): aligned windows across each segment, windows that
    /// are not aligned, one running past the documents, and a rewind.
    #[test]
    fn a_multi_valued_window_is_each_documents_values() {
        let (mut checked, mut declined) = (0, 0);
        for (index, field, keyword) in [
            ("metric_aggs_index", "ml", false),
            ("metric_aggs_index", "md", false),
            ("terms_aggs_index", "mkw", true),
        ] {
            let reader = fixture(index);
            for r in reader.segment_readers() {
                let open = || -> Option<Box<lucene_codecs::doc_values::SortedNumericReader<'_>>> {
                    if keyword {
                        match crate::terms_agg::open_ords(r, field).unwrap().0 {
                            Ords::Multi(m) => Some(m),
                            _ => None,
                        }
                    } else {
                        match aggs::open_values(r, field).unwrap() {
                            Values::Multi(m) => Some(m),
                            _ => None,
                        }
                    }
                };
                let Some(mut windowed) = open() else {
                    continue;
                };
                let max = r.max_doc;
                let mut windows: Vec<(i32, usize)> = (0..max)
                    .step_by(WINDOW as usize)
                    .map(|b| (b, WINDOW as usize))
                    .collect();
                windows.extend([(3, 100), (max - 5, 64), (0, 64), (max, 0)]);
                let (mut offsets, mut values) = (Vec::new(), Vec::new());
                for (base, len) in windows {
                    if !windowed
                        .fill_window(base, len, &mut offsets, &mut values)
                        .unwrap()
                    {
                        declined += 1;
                        continue;
                    }
                    assert_eq!(offsets.len(), len + 1);
                    let mut by_doc = open().unwrap();
                    let mut want = Vec::new();
                    for i in 0..len {
                        let doc = base + i as i32;
                        if doc < max {
                            by_doc.values(doc, &mut want).unwrap();
                        } else {
                            want.clear();
                        }
                        let got = &values[offsets[i] as usize..offsets[i + 1] as usize];
                        assert_eq!(got, &want[..], "{field} doc {doc} window {base}+{len}");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 1000, "{checked}");
        // Unaligned windows over a sparse column are left to the caller.
        assert!(declined > 0);
    }

    /// With a precision, a `cardinality` answers each bucket's sketch: the
    /// one its distinct values build when hashed as `CardinalityAggregator`
    /// hashes them (`cardinality_sketch`'s tests hold the sketch to Java's).
    #[test]
    fn a_cardinality_with_a_precision_is_its_values_sketch() {
        use crate::cardinality_sketch::{mix64, murmur3_h1, Sketch};
        let sketch = |p: u32, values: &[CardinalityValue]| -> Vec<u8> {
            if values.is_empty() {
                return Vec::new();
            }
            let mut s = Sketch::new(p).unwrap();
            for v in values {
                s.collect(match v {
                    CardinalityValue::Term(t) => murmur3_h1(t, 0),
                    CardinalityValue::Long(l) => mix64(*l),
                });
            }
            let mut out = Vec::new();
            s.write_to(&mut out);
            out
        };
        for (index, field, kind) in [
            ("terms_aggs_index", "kw", CardinalityKind::Keyword),
            ("terms_aggs_index", "mkw", CardinalityKind::Keyword),
            (
                "metric_aggs_index",
                "ml",
                CardinalityKind::Numeric(ValueKind::Long),
            ),
            (
                "metric_aggs_index",
                "d",
                CardinalityKind::Numeric(ValueKind::Double),
            ),
        ] {
            let reader = fixture(index);
            // A numeric one is fed every value in document order, as
            // `DirectCollector` feeds it; a keyword one its distinct terms
            // in term order, as `OrdinalsCollector` does.
            let rows = match kind {
                CardinalityKind::Numeric(_) => scan(&reader, &all(), field),
                CardinalityKind::Keyword => Vec::new(),
            };
            let in_doc_order = |p: u32| -> Vec<u8> {
                let CardinalityKind::Numeric(k) = kind else {
                    unreachable!()
                };
                let mut s = Sketch::new(p).unwrap();
                for (_, _, values) in &rows {
                    for &v in values {
                        s.collect(numeric_hash(k, v));
                    }
                }
                let mut out = Vec::new();
                s.write_to(&mut out);
                out
            };
            let want = |p: u32, values: &[CardinalityValue]| match kind {
                CardinalityKind::Numeric(_) if !values.is_empty() => in_doc_order(p),
                _ => sketch(p, values),
            };
            for p in [4, 10, 14] {
                let node = |precision| AggNode::Cardinality {
                    field: field.to_string(),
                    kind,
                    precision,
                };
                // Under a filter bucket that matches nothing too: an empty
                // bucket answers no sketch.
                let nodes = vec![
                    node(None),
                    node(Some(p)),
                    AggNode::Filters {
                        filters: vec![all(), body("no-such-term")],
                        other: false,
                        subs: vec![node(None), node(Some(p))],
                    },
                ];
                let got = run(&reader, &all(), &nodes, &whole(&reader))
                    .unwrap()
                    .remove(0);
                let (AggResult::Cardinality(values), AggResult::CardinalitySketch(sketches)) =
                    (&got[0], &got[1])
                else {
                    panic!("{got:?}")
                };
                assert_eq!(sketches[0], want(p, &values[0]), "{field} p {p}");
                assert!(!sketches[0].is_empty());
                let AggResult::Fixed { subs, .. } = &got[2] else {
                    panic!()
                };
                let (AggResult::Cardinality(values), AggResult::CardinalitySketch(sketches)) =
                    (&subs[0], &subs[1])
                else {
                    panic!()
                };
                assert_eq!(sketches.len(), 2);
                for (v, s) in values.iter().zip(sketches) {
                    assert_eq!(s, &want(p, v), "{field} p {p} under filters");
                }
                assert!(sketches[1].is_empty(), "the empty filter bucket");
            }
        }
        // A precision OpenSearch would refuse is an error, not a panic.
        let reader = fixture("terms_aggs_index");
        let bad = AggNode::Cardinality {
            field: "kw".to_string(),
            kind: CardinalityKind::Keyword,
            precision: Some(40),
        };
        assert!(run(&reader, &all(), &[bad], &whole(&reader)).is_err());
    }

    #[test]
    fn terms_under_a_bucket_are_the_top_level_terms_of_its_documents() {
        let reader = fixture("terms_aggs_index");
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let n = reader.segment_readers().len();
        for q in [all(), body("a")] {
            for (field, shard_size) in [("kw", 5), ("mkw", 3), ("bk", 1000), ("sk", 2)] {
                let want = crate::terms_agg::terms(
                    &segments,
                    reader.segment_readers(),
                    &q,
                    field,
                    shard_size,
                )
                .unwrap();
                let terms = AggNode::Terms {
                    field: field.to_string(),
                    shard_size,
                    subs: vec![AggNode::Cardinality {
                        field: field.to_string(),
                        kind: CardinalityKind::Keyword,
                        precision: None,
                    }],
                };
                let nodes = vec![
                    terms.clone(),
                    AggNode::Filters {
                        filters: vec![
                            all(),
                            BooleanQuery {
                                must: vec![Clause::PointsRange(PointsRangeQuery::new("r", 0, 99))],
                                ..Default::default()
                            },
                        ],
                        other: false,
                        subs: vec![terms],
                    },
                ];
                let got = run(&reader, &q, &nodes, &whole(&reader)).unwrap().remove(0);
                let AggResult::Terms { buckets, subs } = &got[0] else {
                    panic!()
                };
                assert_eq!(buckets[0].0, want.other_doc_count, "{field}");
                assert_eq!(buckets[0].1, want.buckets, "{field}");
                // A bucket's cardinality of its own field is its one term.
                let AggResult::Cardinality(sets) = &subs[0] else {
                    panic!()
                };
                for ((term, _), set) in buckets[0].1.iter().zip(sets) {
                    if field != "mkw" {
                        assert_eq!(set, &vec![CardinalityValue::Term(term.clone())]);
                    } else {
                        assert!(set.contains(&CardinalityValue::Term(term.clone())));
                    }
                }
                let AggResult::Fixed { subs, .. } = &got[1] else {
                    panic!()
                };
                let AggResult::Terms { buckets: under, .. } = &subs[0] else {
                    panic!()
                };
                assert_eq!(under[0].1, want.buckets, "under a match-all filter");
                assert_eq!(under[0].0, want.other_doc_count);
                assert_eq!(under.len(), 2, "one owning bucket per filter");
            }
            // Sliced: every slice from scratch; the counts add up.
            let node = AggNode::Terms {
                field: "kw".to_string(),
                shard_size: 10_000,
                subs: vec![],
            };
            let slices: Vec<Vec<usize>> = (0..n).rev().map(|i| vec![i]).collect();
            let per = run(&reader, &q, std::slice::from_ref(&node), &slices).unwrap();
            let total: u64 = per
                .iter()
                .map(|r| match &r[0] {
                    AggResult::Terms { buckets, .. } => {
                        buckets[0].1.iter().map(|b| b.1).sum::<u64>()
                    }
                    _ => 0,
                })
                .sum();
            let one = run(&reader, &q, &[node], &whole(&reader))
                .unwrap()
                .remove(0)
                .remove(0);
            let AggResult::Terms { buckets, .. } = one else {
                panic!()
            };
            assert_eq!(total, buckets[0].1.iter().map(|b| b.1).sum::<u64>());
        }
    }

    #[test]
    fn a_bad_tree_or_slice_is_an_error() {
        let reader = fixture("terms_aggs_index");
        let nodes = [AggNode::Terms {
            field: "kw".to_string(),
            shard_size: 1,
            subs: vec![],
        }];
        assert!(
            run(&reader, &all(), &nodes, &[vec![99]]).is_err(),
            "no such segment"
        );
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let empty = HashMap::new();
        let missing = aggregate_tree(
            &segments,
            reader.segment_readers(),
            &all(),
            &nodes,
            &Globals { ords: &empty },
            &whole(&reader),
            None,
        );
        assert!(missing.is_err(), "no global ordinals for the field");
        // A numeric field under terms is a mapping error.
        let numeric = [AggNode::Terms {
            field: "r".to_string(),
            shard_size: 1,
            subs: vec![],
        }];
        let _ = run(&reader, &all(), &numeric, &whole(&reader));
        assert!(AggNode::Metric {
            field: "x".into(),
            kind: ValueKind::Long,
            source: Source::PointsMin,
            needs: NEED_ALL,
        }
        .reads_points());
        let mut qs = Vec::new();
        let g = AggNode::Global {
            subs: vec![AggNode::Filters {
                filters: vec![all()],
                other: false,
                subs: vec![],
            }],
        };
        g.filter_queries(&mut qs);
        assert_eq!(qs.len(), 1);
    }
}
