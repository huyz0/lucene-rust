//! `org.apache.lucene.sandbox.facet.cutters.ranges`: a document's long (or
//! double) values as the ordinals of the ranges they fall in -- each range's
//! ordinal is its index in the requested ranges.
//!
//! The ranges are cut into elementary intervals that cover every `long`; a
//! value is binary-searched to its interval. Disjoint ranges map an interval
//! straight to its range (or to a gap); overlapping ones roll the hit
//! intervals up a binary tree whose nodes carry the ranges they lie in.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::directory_reader::SegmentReader;
use crate::facets::multi_double_values_source::{
    self, to_sortable_multi_long_values_source, MultiDoubleValuesSource,
};
use crate::facets::multi_long_values_source::{self, BoxMultiLongValues, MultiLongValuesSource};
use crate::facets::range::DoubleRange;
use crate::facets::range::LongRange;
use crate::values_source::{
    double_to_sortable_long, to_sortable_long_values_source, BoxLongValues, LongValuesSource,
    ValuesContext,
};
use crate::Result;

use super::{FacetCutter, LeafFacetCutter, OrdinalIterator, NO_MORE_ORDS};

/// `LongRangeFacetCutter.SKIP_INTERVAL_POSITION`: an interval in no range.
const SKIP_INTERVAL_POSITION: i32 = -1;

/// `LongRangeFacetCutter`: a document's values as the ordinals of the
/// [`LongRange`]s holding them, each counted once per document.
pub struct LongRangeFacetCutter {
    values_source: Option<Arc<dyn MultiLongValuesSource>>,
    single_values: Option<Arc<dyn LongValuesSource>>,
    requested_range_count: usize,
    /// Each elementary interval's (inclusive) end.
    boundaries: Arc<[i64]>,
    /// Each elementary interval's range, or [`SKIP_INTERVAL_POSITION`]
    /// (disjoint ranges only).
    pos: Arc<[i32]>,
    /// The interval tree, for overlapping ranges.
    root: Option<Arc<LongRangeNode>>,
}

/// `LongRangeFacetCutter.InclusiveRange`.
#[derive(Debug, Clone, Copy)]
pub(super) struct InclusiveRange {
    pub(super) start: i64,
    pub(super) end: i64,
}

impl LongRangeFacetCutter {
    /// `LongRangeFacetCutter.create(longValuesSource, longRanges)`: the
    /// source's single-valued form when it has one.
    pub fn create(values_source: Arc<dyn MultiLongValuesSource>, ranges: &[LongRange]) -> Self {
        Self::create_single_or_multi_valued(Some(values_source), None, ranges)
    }

    /// `createSingleOrMultiValued(longValuesSource, singleLongValuesSource,
    /// longRanges)`.
    fn create_single_or_multi_valued(
        values_source: Option<Arc<dyn MultiLongValuesSource>>,
        single: Option<Arc<dyn LongValuesSource>>,
        ranges: &[LongRange],
    ) -> Self {
        let single = single.or_else(|| {
            values_source
                .as_deref()
                .and_then(multi_long_values_source::unwrap_singleton)
        });
        // Stable, as Java's `Arrays.sort` of objects.
        let mut sorted: Vec<(&LongRange, i32)> = ranges
            .iter()
            .enumerate()
            .map(|(i, r)| (r, i32::try_from(i).unwrap_or(i32::MAX)))
            .collect();
        sorted.sort_by_key(|(r, _)| r.min);
        let overlapping = are_overlapping(&sorted);
        let intervals = if overlapping {
            overlapping_intervals(&sorted)
        } else {
            disjoint_intervals(&sorted)
        };
        let mut boundaries = Vec::with_capacity(intervals.len());
        let mut pos = vec![SKIP_INTERVAL_POSITION; intervals.len()];
        let mut curr = 0usize;
        for (i, interval) in intervals.iter().enumerate() {
            boundaries.push(interval.end);
            if let Some((range, p)) = sorted.get(curr) {
                if interval.end == range.max {
                    pos[i] = *p;
                    curr += 1;
                }
            }
        }
        let root = if overlapping {
            let mut root = split(0, intervals.len(), &intervals);
            for (range, p) in &sorted {
                root.add_outputs(range, *p);
            }
            Some(Arc::new(root))
        } else {
            None
        };
        LongRangeFacetCutter {
            values_source,
            single_values: single,
            requested_range_count: ranges.len(),
            boundaries: boundaries.into(),
            pos: pos.into(),
            root,
        }
    }
}

/// `areOverlappingRanges`: some range starts at or before the previous
/// one's end (closed ranges).
fn are_overlapping(sorted: &[(&LongRange, i32)]) -> bool {
    let Some((first, _)) = sorted.first() else {
        return false;
    };
    let mut previous_max = first.max;
    for (r, _) in &sorted[1..] {
        if r.min <= previous_max {
            return true;
        }
        previous_max = r.max;
    }
    false
}

/// `NonOverlappingLongRangeFacetCutter.buildElementaryIntervals`: each
/// range, the gaps before them, and the tail to `Long.MAX_VALUE`.
pub(super) fn disjoint_intervals(sorted: &[(&LongRange, i32)]) -> Vec<InclusiveRange> {
    let mut out = Vec::new();
    let mut prev = i64::MIN;
    for (r, _) in sorted {
        if r.min > prev {
            out.push(InclusiveRange {
                start: prev,
                end: r.min - 1,
            });
        }
        out.push(InclusiveRange {
            start: r.min,
            end: r.max,
        });
        // Wraps for a range ending at `Long.MAX_VALUE`, as Java's.
        prev = r.max.wrapping_add(1);
    }
    match out.last() {
        Some(last) if last.end < i64::MAX => out.push(InclusiveRange {
            start: last.end + 1,
            end: i64::MAX,
        }),
        Some(_) => {}
        None => out.push(InclusiveRange {
            start: i64::MIN,
            end: i64::MAX,
        }),
    }
    out
}

/// `OverlappingLongRangeFacetCutter.buildElementaryIntervals`: a 1D Venn
/// diagram of the ranges' ends (1 = a start, 2 = an end, 3 = both, which
/// is its own interval).
pub(super) fn overlapping_intervals(sorted: &[(&LongRange, i32)]) -> Vec<InclusiveRange> {
    let mut ends: BTreeMap<i64, u8> = BTreeMap::new();
    ends.insert(i64::MIN, 1);
    ends.insert(i64::MAX, 2);
    for (r, _) in sorted {
        *ends.entry(r.min).or_insert(0) |= 1;
        *ends.entry(r.max).or_insert(0) |= 2;
    }
    let mut out = Vec::new();
    let mut it = ends.into_iter();
    let Some((v, flags)) = it.next() else {
        return out;
    };
    let mut prev = if flags == 3 {
        out.push(InclusiveRange { start: v, end: v });
        v.wrapping_add(1)
    } else {
        v
    };
    for (v, flags) in it {
        match flags {
            3 => {
                if v > prev {
                    out.push(InclusiveRange {
                        start: prev,
                        end: v - 1,
                    });
                }
                out.push(InclusiveRange { start: v, end: v });
                prev = v.wrapping_add(1);
            }
            1 => {
                if v > prev {
                    out.push(InclusiveRange {
                        start: prev,
                        end: v - 1,
                    });
                }
                prev = v;
            }
            _ => {
                out.push(InclusiveRange {
                    start: prev,
                    end: v,
                });
                prev = v.wrapping_add(1);
            }
        }
    }
    out
}

/// `LongRangeNode`: a node of the elementary intervals' binary tree, with
/// the ranges ([`outputs`](Self::outputs)) that contain it but not its
/// parent.
#[derive(Debug)]
struct LongRangeNode {
    start: i64,
    end: i64,
    children: Option<(Box<LongRangeNode>, Box<LongRangeNode>)>,
    outputs: Vec<i32>,
    /// The elementary intervals (leaves) under this node.
    leaves: usize,
}

/// `OverlappingLongRangeFacetCutter.split`.
fn split(start: usize, end: usize, intervals: &[InclusiveRange]) -> LongRangeNode {
    if start + 1 >= end {
        let r = intervals.get(start).copied().unwrap_or(InclusiveRange {
            start: i64::MIN,
            end: i64::MAX,
        });
        return LongRangeNode {
            start: r.start,
            end: r.end,
            children: None,
            outputs: Vec::new(),
            leaves: 1,
        };
    }
    let mid = (start + end) >> 1;
    let left = split(start, mid, intervals);
    let right = split(mid, end, intervals);
    LongRangeNode {
        start: left.start,
        end: right.end,
        leaves: left.leaves + right.leaves,
        children: Some((Box::new(left), Box::new(right))),
        outputs: Vec::new(),
    }
}

impl LongRangeNode {
    /// `addOutputs(range)`.
    fn add_outputs(&mut self, range: &LongRange, pos: i32) {
        if self.start >= range.min && self.end <= range.max {
            self.outputs.push(pos);
        } else if let Some((left, right)) = &mut self.children {
            left.add_outputs(range, pos);
            right.add_outputs(range, pos);
        }
    }

    /// `rollupMultiValued`/`rollupSingleValued`: whether a leaf under this
    /// node was hit, setting the ranges of every hit node in `requested`.
    /// `hit(lo, hi)` tells whether a leaf in `[lo, hi)` was hit, so a
    /// subtree without one is passed over whole -- Java walks it to the
    /// same answer: no hit, no outputs.
    fn rollup<F: Fn(usize, usize) -> bool>(
        &self,
        upto: &mut usize,
        hit: &F,
        requested: &mut IntervalTracker,
    ) -> bool {
        if !hit(*upto, *upto + self.leaves) {
            *upto += self.leaves;
            return false;
        }
        match &self.children {
            Some((left, right)) => {
                left.rollup(upto, hit, requested);
                right.rollup(upto, hit, requested);
            }
            None => *upto += 1,
        }
        for &p in &self.outputs {
            requested.set(p);
        }
        true
    }
}

/// `IntervalTracker.MultiIntervalTracker`: which intervals a document hit,
/// read back in order once frozen -- a `FixedBitSet` of `u64` words, as
/// Java's.
#[derive(Debug, Clone)]
pub(super) struct IntervalTracker {
    words: Vec<u64>,
    size: usize,
    /// The next bit [`next_ord`](Self::next_ord) looks at.
    bit_from: usize,
    /// The set bits not read back yet (Java's `withHit - state`).
    remaining: u32,
}

impl IntervalTracker {
    pub(super) fn new(size: usize) -> Self {
        IntervalTracker {
            words: vec![0; size.div_ceil(64)],
            size,
            bit_from: 0,
            remaining: 0,
        }
    }

    pub(super) fn set(&mut self, i: i32) {
        if let Ok(i) = usize::try_from(i) {
            if i < self.size {
                self.words[i >> 6] |= 1u64 << (i & 63);
            }
        }
    }

    /// Whether a bit in `[lo, hi)` is set.
    pub(super) fn any_in(&self, lo: usize, hi: usize) -> bool {
        let hi = hi.min(self.size);
        if lo >= hi {
            return false;
        }
        let (first, last) = (lo >> 6, (hi - 1) >> 6);
        (first..=last).any(|w| {
            let mut m = self.words[w];
            if w == first {
                m &= u64::MAX << (lo & 63);
            }
            if w == last {
                m &= u64::MAX >> (63 - ((hi - 1) & 63));
            }
            m != 0
        })
    }

    pub(super) fn clear(&mut self) {
        self.words.fill(0);
        self.bit_from = 0;
        self.remaining = 0;
    }

    pub(super) fn freeze(&mut self) {
        self.remaining = self.words.iter().map(|w| w.count_ones()).sum();
    }

    /// The next set bit, ascending; [`NO_MORE_ORDS`] once every bit set
    /// at [`freeze`](Self::freeze) was read.
    pub(super) fn next_ord(&mut self) -> i32 {
        if self.remaining == 0 {
            return NO_MORE_ORDS;
        }
        self.remaining -= 1;
        let mut w = self.bit_from >> 6;
        let mut word = self.words[w] & (u64::MAX << (self.bit_from & 63));
        while word == 0 {
            w += 1;
            word = self.words[w];
        }
        let next = (w << 6) + word.trailing_zeros() as usize;
        self.bit_from = next + 1;
        i32::try_from(next).unwrap_or(NO_MORE_ORDS)
    }
}

/// `boundaries[i]`, `Long.MAX_VALUE` past the end (the last boundary).
fn boundary(boundaries: &[i64], i: usize) -> i64 {
    boundaries.get(i).copied().unwrap_or(i64::MAX)
}

/// `processValue(v, lastIntervalSeen)`: the elementary interval of `v`,
/// searching after `last` (a sorted document's previous value's interval).
pub(super) fn process_value(boundaries: &[i64], v: i64, last: Option<usize>) -> usize {
    let mut lo = 0usize;
    let mut hi = boundaries.len().saturating_sub(1);
    if let Some(last) = last {
        if v <= boundary(boundaries, last) {
            return last;
        }
        lo = last + 1;
        if lo == boundaries.len() {
            return last;
        }
    }
    let lower = lo;
    loop {
        let mid = (lo + hi) >> 1;
        if v <= boundary(boundaries, mid) {
            if mid == lower {
                return mid;
            }
            hi = mid - 1;
        } else if v > boundary(boundaries, mid + 1) {
            lo = mid + 1;
        } else {
            return mid + 1;
        }
    }
}

impl FacetCutter for LongRangeFacetCutter {
    fn create_leaf_cutter<'a>(
        &self,
        reader: &'a SegmentReader,
    ) -> Result<Box<dyn LeafFacetCutter + 'a>> {
        let ctx = ValuesContext::for_reader(reader);
        let requested = self
            .root
            .as_ref()
            .map(|_| IntervalTracker::new(self.requested_range_count));
        if let Some(single) = &self.single_values {
            return Ok(Box::new(SingleValuedLeafCutter {
                values: single.get_values(&ctx, 0, None)?,
                boundaries: self.boundaries.clone(),
                pos: self.pos.clone(),
                elementary_ord: 0,
                requested,
                root: self.root.clone(),
            }));
        }
        let source = self.values_source.as_ref().ok_or_else(|| {
            crate::Error::IllegalState("a range cutter without a values source".into())
        })?;
        Ok(Box::new(MultiValuedLeafCutter {
            values: source.get_values(&ctx, 0)?,
            elementary: IntervalTracker::new(self.boundaries.len()),
            boundaries: self.boundaries.clone(),
            pos: self.pos.clone(),
            requested,
            root: self.root.clone(),
        }))
    }
}

/// `LongRangeMultivaluedLeafFacetCutter` and its two subclasses.
struct MultiValuedLeafCutter<'a> {
    values: BoxMultiLongValues<'a>,
    boundaries: Arc<[i64]>,
    pos: Arc<[i32]>,
    elementary: IntervalTracker,
    requested: Option<IntervalTracker>,
    root: Option<Arc<LongRangeNode>>,
}

impl LeafFacetCutter for MultiValuedLeafCutter<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        if !self.values.advance_exact(doc)? {
            return Ok(false);
        }
        self.elementary.clear();
        if let Some(r) = &mut self.requested {
            r.clear();
        }
        let n = self.values.get_value_count();
        let mut last = None;
        let mut i = 0i64;
        while i < n {
            let interval = process_value(&self.boundaries, self.values.next_value()?, last);
            last = Some(interval);
            self.elementary.set(i32::try_from(interval).unwrap_or(-1));
            if interval + 1 == self.boundaries.len() {
                // The last interval: nothing further can be hit.
                break;
            }
            i += 1;
        }
        if let (Some(root), Some(requested)) = (&self.root, &mut self.requested) {
            let elementary = &self.elementary;
            let mut upto = 0;
            root.rollup(&mut upto, &|lo, hi| elementary.any_in(lo, hi), requested);
        }
        self.elementary.freeze();
        if let Some(r) = &mut self.requested {
            r.freeze();
        }
        Ok(true)
    }
}

impl OrdinalIterator for MultiValuedLeafCutter<'_> {
    fn next_ord(&mut self) -> Result<i32> {
        if let Some(requested) = &mut self.requested {
            return Ok(requested.next_ord());
        }
        loop {
            let ord = self.elementary.next_ord();
            if ord == NO_MORE_ORDS {
                return Ok(NO_MORE_ORDS);
            }
            let result = usize::try_from(ord)
                .ok()
                .and_then(|o| self.pos.get(o))
                .copied()
                .unwrap_or(SKIP_INTERVAL_POSITION);
            if result != SKIP_INTERVAL_POSITION {
                return Ok(result);
            }
        }
    }
}

/// `LongRangeSingleValuedLeafFacetCutter` and its two subclasses.
struct SingleValuedLeafCutter<'a> {
    values: BoxLongValues<'a>,
    boundaries: Arc<[i64]>,
    pos: Arc<[i32]>,
    /// The document's elementary interval; [`NO_MORE_ORDS`] once read
    /// (disjoint ranges).
    elementary_ord: i32,
    requested: Option<IntervalTracker>,
    root: Option<Arc<LongRangeNode>>,
}

impl LeafFacetCutter for SingleValuedLeafCutter<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        if !self.values.advance_exact(doc)? {
            return Ok(false);
        }
        if let Some(r) = &mut self.requested {
            r.clear();
        }
        let interval = process_value(&self.boundaries, self.values.long_value()?, None);
        self.elementary_ord = i32::try_from(interval).unwrap_or(NO_MORE_ORDS);
        if let (Some(root), Some(requested)) = (&self.root, &mut self.requested) {
            let mut upto = 0;
            root.rollup(&mut upto, &|lo, hi| (lo..hi).contains(&interval), requested);
            requested.freeze();
        }
        Ok(true)
    }
}

impl OrdinalIterator for SingleValuedLeafCutter<'_> {
    fn next_ord(&mut self) -> Result<i32> {
        if let Some(requested) = &mut self.requested {
            return Ok(requested.next_ord());
        }
        if self.elementary_ord == NO_MORE_ORDS {
            return Ok(NO_MORE_ORDS);
        }
        let result = usize::try_from(self.elementary_ord)
            .ok()
            .and_then(|o| self.pos.get(o))
            .copied()
            .unwrap_or(SKIP_INTERVAL_POSITION);
        self.elementary_ord = NO_MORE_ORDS;
        Ok(if result != SKIP_INTERVAL_POSITION {
            result
        } else {
            NO_MORE_ORDS
        })
    }
}

/// `DoubleRangeFacetCutter`: double ranges as [`LongRangeFacetCutter`]
/// ranges over `NumericUtils.doubleToSortableLong` values.
pub struct DoubleRangeFacetCutter {
    inner: LongRangeFacetCutter,
}

impl DoubleRangeFacetCutter {
    /// `new DoubleRangeFacetCutter(multiDoubleValuesSource, doubleRanges)`.
    pub fn new(source: Arc<dyn MultiDoubleValuesSource>, ranges: &[DoubleRange]) -> Self {
        let (single, multi) = match multi_double_values_source::unwrap_singleton(source.as_ref()) {
            Some(single) => (Some(to_sortable_long_values_source(single)), None),
            None => (None, Some(to_sortable_multi_long_values_source(source))),
        };
        let long_ranges: Vec<LongRange> = ranges
            .iter()
            .map(|r| LongRange {
                label: r.label.clone(),
                min: double_to_sortable_long(r.min),
                max: double_to_sortable_long(r.max),
            })
            .collect();
        DoubleRangeFacetCutter {
            inner: LongRangeFacetCutter::create_single_or_multi_valued(multi, single, &long_ranges),
        }
    }
}

impl FacetCutter for DoubleRangeFacetCutter {
    fn create_leaf_cutter<'a>(
        &self,
        reader: &'a SegmentReader,
    ) -> Result<Box<dyn LeafFacetCutter + 'a>> {
        self.inner.create_leaf_cutter(reader)
    }
}
