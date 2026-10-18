//! `LongRangeCounter`, `ExclusiveLongRangeCounter` and
//! `OverlappingLongRangeCounter`: counting values into [`LongRange`]s
//! through the elementary intervals the ranges' ends cut `long` space into.
//!
//! Java's abstract class and its two subclasses are one struct here, the
//! subclass a [`Kind`]; the binary searches over the boundaries
//! (`addSingleValued`, `addMultiValued`) are the base class's, unchanged.
//! The segment tree's nodes live in an arena, `FixedBitSet`s are `Vec<bool>`.

use super::LongRange;

/// `LongRangeCounter`.
#[derive(Debug)]
pub(crate) struct LongRangeCounter {
    /// `countBuffer`, one count per range.
    counts: Vec<i32>,
    /// `multiValuedDocLastSeenElementaryInterval`.
    last_seen: i32,
    /// The elementary intervals' (inclusive) ends.
    boundaries: Vec<i64>,
    kind: Kind,
}

#[derive(Debug)]
enum Kind {
    /// `ExclusiveLongRangeCounter`: no two ranges overlap.
    Exclusive {
        /// The range each elementary interval is, `-1` for a gap.
        range_nums: Vec<i32>,
        missing_count: i32,
        multi_valued_doc_matched_range: bool,
    },
    /// `OverlappingLongRangeCounter`: a segment tree over the elementary
    /// intervals, each node listing the ranges it lies wholly inside.
    Overlapping(Box<Overlapping>),
}

#[derive(Debug)]
struct Overlapping {
    nodes: Vec<Node>,
    root: usize,
    has_unflushed_counts: bool,
    single_valued_counts: Option<Vec<i32>>,
    interval_hits: Vec<bool>,
    range_hits: Vec<bool>,
    upto: usize,
    missing_count: i32,
}

/// `LongRangeNode`.
#[derive(Debug)]
struct Node {
    start: i64,
    end: i64,
    /// `(left, right)`, `None` for a leaf.
    children: Option<(usize, usize)>,
    outputs: Vec<usize>,
}

impl LongRangeCounter {
    /// `LongRangeCounter.create(ranges, countBuffer)`.
    pub(crate) fn create(ranges: &[LongRange]) -> Self {
        if has_overlapping_ranges(ranges) {
            Self::overlapping(ranges)
        } else {
            Self::exclusive(ranges)
        }
    }

    fn exclusive(ranges: &[LongRange]) -> Self {
        let mut sorted: Vec<(&LongRange, usize)> =
            ranges.iter().enumerate().map(|(i, r)| (r, i)).collect();
        // `Arrays.sort` on objects is stable, as is `sort_by_key`.
        sorted.sort_by_key(|(r, _)| r.min);

        let mut intervals: Vec<(i64, i64)> = Vec::new();
        let mut prev = i64::MIN;
        for (r, _) in &sorted {
            if r.min > prev {
                intervals.push((prev, r.min.wrapping_sub(1)));
            }
            intervals.push((r.min, r.max));
            prev = r.max.wrapping_add(1);
        }
        match intervals.last() {
            Some(&(_, last_end)) => {
                if last_end < i64::MAX {
                    intervals.push((last_end.wrapping_add(1), i64::MAX));
                }
            }
            None => intervals.push((i64::MIN, i64::MAX)),
        }

        let boundaries: Vec<i64> = intervals.iter().map(|&(_, end)| end).collect();
        let mut range_nums = vec![-1i32; boundaries.len()];
        let mut curr = 0usize;
        for (i, &b) in boundaries.iter().enumerate() {
            if let Some(&(r, pos)) = sorted.get(curr) {
                if b == r.max {
                    range_nums[i] = pos as i32;
                    curr += 1;
                }
            }
        }
        LongRangeCounter {
            counts: vec![0; ranges.len()],
            last_seen: -1,
            boundaries,
            kind: Kind::Exclusive {
                range_nums,
                missing_count: 0,
                multi_valued_doc_matched_range: false,
            },
        }
    }

    fn overlapping(ranges: &[LongRange]) -> Self {
        let intervals = build_elementary_intervals(ranges);
        let mut nodes = Vec::new();
        let root = split(0, intervals.len(), &intervals, &mut nodes);
        for (i, r) in ranges.iter().enumerate() {
            add_outputs(&mut nodes, root, i, r);
        }
        let boundaries: Vec<i64> = intervals.iter().map(|&(_, end)| end).collect();
        LongRangeCounter {
            counts: vec![0; ranges.len()],
            last_seen: -1,
            boundaries,
            kind: Kind::Overlapping(Box::new(Overlapping {
                nodes,
                root,
                has_unflushed_counts: false,
                single_valued_counts: None,
                interval_hits: Vec::new(),
                range_hits: Vec::new(),
                upto: 0,
                missing_count: 0,
            })),
        }
    }

    /// The per-range counts.
    pub(crate) fn into_counts(self) -> Vec<i32> {
        self.counts
    }

    fn range_count(&self) -> usize {
        self.counts.len()
    }

    /// `startMultiValuedDoc()`.
    pub(crate) fn start_multi_valued_doc(&mut self) {
        self.last_seen = -1;
        let n = self.boundaries.len();
        match &mut self.kind {
            Kind::Exclusive {
                multi_valued_doc_matched_range,
                ..
            } => *multi_valued_doc_matched_range = false,
            Kind::Overlapping(o) => {
                o.interval_hits.clear();
                o.interval_hits.resize(n, false);
            }
        }
    }

    /// `endMultiValuedDoc()`: whether the document's values fell in any
    /// range.
    pub(crate) fn end_multi_valued_doc(&mut self) -> bool {
        let range_count = self.range_count();
        match &mut self.kind {
            Kind::Exclusive {
                multi_valued_doc_matched_range,
                ..
            } => *multi_valued_doc_matched_range,
            Kind::Overlapping(o) => {
                if range_count == 0 {
                    return false;
                }
                o.range_hits.clear();
                o.range_hits.resize(range_count, false);
                o.upto = 0;
                let root = o.root;
                o.rollup_multi_valued(root);
                let mut contributed = false;
                for (i, &hit) in o.range_hits.iter().enumerate() {
                    if hit {
                        self.counts[i] = self.counts[i].wrapping_add(1);
                        contributed = true;
                    }
                }
                contributed
            }
        }
    }

    /// `addSingleValued(v)`.
    pub(crate) fn add_single_valued(&mut self, v: i64) {
        if let Kind::Exclusive { missing_count, .. } = &mut self.kind {
            if self.counts.is_empty() {
                *missing_count = missing_count.wrapping_add(1);
                return;
            }
        }
        // Java's binary search finds the interval `i` with
        // `boundaries[i - 1] < v <= boundaries[i]`: the first boundary not
        // below `v`, which `partition_point` finds without the hard-to-predict
        // branches (the last boundary is `Long.MAX_VALUE`, so there is one).
        let interval = self.boundaries.partition_point(|&b| b < v);
        self.process_single_valued_hit(interval);
    }

    /// `addMultiValued(v)`.
    pub(crate) fn add_multi_valued(&mut self, v: i64) {
        if self.range_count() == 0 {
            return;
        }
        let b = &self.boundaries;
        if self.last_seen != -1 && v <= b[self.last_seen as usize] {
            return;
        }
        let next = (self.last_seen + 1) as usize;
        if next == b.len() {
            return;
        }
        let mut lo = next;
        let mut hi = b.len() - 1;
        loop {
            let mid = (lo + hi) >> 1;
            if v <= b[mid] {
                if mid == next {
                    self.process_multi_valued_hit(mid);
                    self.last_seen = mid as i32;
                    return;
                }
                hi = mid - 1;
            } else if v > b[mid + 1] {
                lo = mid + 1;
            } else {
                self.process_multi_valued_hit(mid + 1);
                self.last_seen = (mid + 1) as i32;
                return;
            }
        }
    }

    /// `finish()`: the single-valued counts rolled up, returning how many
    /// values fell in no range.
    pub(crate) fn finish(&mut self) -> i32 {
        match &mut self.kind {
            Kind::Exclusive { missing_count, .. } => *missing_count,
            Kind::Overlapping(o) => {
                if !o.has_unflushed_counts {
                    return 0;
                }
                o.missing_count = 0;
                o.upto = 0;
                let root = o.root;
                o.rollup_single_valued(root, false, &mut self.counts);
                o.missing_count
            }
        }
    }

    fn process_single_valued_hit(&mut self, interval: usize) {
        let n = self.boundaries.len();
        match &mut self.kind {
            Kind::Exclusive {
                range_nums,
                missing_count,
                ..
            } => {
                let range = range_nums[interval];
                if range != -1 {
                    let c = &mut self.counts[range as usize];
                    *c = c.wrapping_add(1);
                } else {
                    *missing_count = missing_count.wrapping_add(1);
                }
            }
            Kind::Overlapping(o) => {
                let counts = o.single_valued_counts.get_or_insert_with(|| vec![0; n]);
                counts[interval] = counts[interval].wrapping_add(1);
                o.has_unflushed_counts = true;
            }
        }
    }

    fn process_multi_valued_hit(&mut self, interval: usize) {
        match &mut self.kind {
            Kind::Exclusive {
                range_nums,
                multi_valued_doc_matched_range,
                ..
            } => {
                let range = range_nums[interval];
                if range != -1 {
                    let c = &mut self.counts[range as usize];
                    *c = c.wrapping_add(1);
                    *multi_valued_doc_matched_range = true;
                }
            }
            Kind::Overlapping(o) => o.interval_hits[interval] = true,
        }
    }
}

impl Overlapping {
    fn rollup_single_valued(&mut self, node: usize, saw_outputs: bool, counts: &mut [i32]) -> i32 {
        let saw_outputs = saw_outputs || !self.nodes[node].outputs.is_empty();
        let count = match self.nodes[node].children {
            Some((left, right)) => self
                .rollup_single_valued(left, saw_outputs, counts)
                .wrapping_add(self.rollup_single_valued(right, saw_outputs, counts)),
            None => {
                let c = self
                    .single_valued_counts
                    .as_ref()
                    .map_or(0, |v| v[self.upto]);
                self.upto += 1;
                if !saw_outputs {
                    self.missing_count = self.missing_count.wrapping_add(c);
                }
                c
            }
        };
        for &r in &self.nodes[node].outputs {
            counts[r] = counts[r].wrapping_add(count);
        }
        count
    }

    fn rollup_multi_valued(&mut self, node: usize) -> bool {
        let contained_hit = match self.nodes[node].children {
            Some((left, right)) => {
                let l = self.rollup_multi_valued(left);
                let r = self.rollup_multi_valued(right);
                l | r
            }
            None => {
                let hit = self.interval_hits[self.upto];
                self.upto += 1;
                hit
            }
        };
        if contained_hit {
            for &r in &self.nodes[node].outputs {
                self.range_hits[r] = true;
            }
        }
        contained_hit
    }
}

/// `hasOverlappingRanges`: sorted by `min`, does any range start at or
/// before the previous range's `max`? (Java compares with the previous
/// range's `max`, not the running maximum, and so does this.)
fn has_overlapping_ranges(ranges: &[LongRange]) -> bool {
    if ranges.is_empty() {
        return false;
    }
    let mut sorted: Vec<&LongRange> = ranges.iter().collect();
    sorted.sort_by_key(|r| r.min);
    let mut previous_max = sorted[0].max;
    for r in &sorted[1..] {
        if r.min <= previous_max {
            return true;
        }
        previous_max = r.max;
    }
    false
}

/// `OverlappingLongRangeCounter.buildElementaryIntervals`: every range end
/// flagged `1` (a start), `2` (an end) or `3` (both), walked in order.
fn build_elementary_intervals(ranges: &[LongRange]) -> Vec<(i64, i64)> {
    let mut ends: std::collections::BTreeMap<i64, u8> = std::collections::BTreeMap::new();
    ends.insert(i64::MIN, 1);
    ends.insert(i64::MAX, 2);
    for r in ranges {
        *ends.entry(r.min).or_insert(0) |= 1;
        *ends.entry(r.max).or_insert(0) |= 2;
    }
    let mut intervals = Vec::new();
    let mut it = ends.into_iter();
    // The map holds `Long.MIN_VALUE` at least.
    let Some((v, flags)) = it.next() else {
        return intervals;
    };
    let mut prev = if flags == 3 {
        intervals.push((v, v));
        v.wrapping_add(1)
    } else {
        v
    };
    for (v, flags) in it {
        if flags == 3 {
            if v > prev {
                intervals.push((prev, v.wrapping_sub(1)));
            }
            intervals.push((v, v));
            prev = v.wrapping_add(1);
        } else if flags == 1 {
            if v > prev {
                intervals.push((prev, v.wrapping_sub(1)));
            }
            prev = v;
        } else {
            intervals.push((prev, v));
            prev = v.wrapping_add(1);
        }
    }
    intervals
}

/// `OverlappingLongRangeCounter.split`.
fn split(start: usize, end: usize, intervals: &[(i64, i64)], nodes: &mut Vec<Node>) -> usize {
    let node = if start + 1 == end {
        let (s, e) = intervals[start];
        Node {
            start: s,
            end: e,
            children: None,
            outputs: Vec::new(),
        }
    } else {
        let mid = (start + end) >> 1;
        let left = split(start, mid, intervals, nodes);
        let right = split(mid, end, intervals, nodes);
        Node {
            start: nodes[left].start,
            end: nodes[right].end,
            children: Some((left, right)),
            outputs: Vec::new(),
        }
    };
    nodes.push(node);
    nodes.len() - 1
}

/// `LongRangeNode.addOutputs`.
fn add_outputs(nodes: &mut [Node], node: usize, index: usize, range: &LongRange) {
    let n = &nodes[node];
    if n.start >= range.min && n.end <= range.max {
        nodes[node].outputs.push(index);
    } else if let Some((left, right)) = n.children {
        add_outputs(nodes, left, index, range);
        add_outputs(nodes, right, index, range);
    }
}

#[cfg(test)]
mod tests;
