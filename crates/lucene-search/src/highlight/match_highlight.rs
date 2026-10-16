//! `org.apache.lucene.search.matchhighlight`: highlighting driven by the
//! Matches API. A query's matches in a document give offset ranges per field
//! ([`MatchRegionRetriever`], by whichever [`OffsetsRetrievalStrategy`] the
//! field's index options allow); [`PassageSelector`] picks the best windows of
//! the field's text around them, [`BreakIteratorShrinkingAdjuster`] pulls the
//! windows in to word boundaries, and [`PassageFormatter`] marks the ranges up.
//! [`MatchHighlighter`] runs it for a set of documents and queries.
//!
//! Text is UTF-16, as Java's offsets are: values are held as `Vec<u16>` while
//! a passage is cut, and a cut inside a surrogate pair reads U+FFFD.
//!
//! Differences from Java: documents are processed one after the other (no
//! `TaskExecutor` blocks); a passage's markers are plain ranges, so the
//! `QueryOffsetRange.query` of a marker is not reachable from a formatter;
//! only the `Locale.ROOT` word iterator is available for the adjuster.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use lucene_analysis::{Analyzer, TokenStream};
use lucene_codecs::field_infos::IndexOptions;

use super::{Error, Result, WordBreakIterator};
use crate::directory_reader::DirectoryReader;
use crate::index_searcher::IndexSearcher;
use crate::matches::{BoxMatchesIterator, MatchesIterator};
use crate::query::Clause;
use crate::query_visitor::{self, QueryVisitor, TermCollector};

fn io(e: impl std::fmt::Display) -> Error {
    Error::Io(e.to_string())
}

/// `OffsetRange`: `[from, to)` in UTF-16 code units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OffsetRange {
    pub from: usize,
    pub to: usize,
}

impl OffsetRange {
    /// `new OffsetRange(from, to)`.
    pub fn new(from: usize, to: usize) -> Self {
        debug_assert!(
            from <= to,
            "A non-empty offset range is required: {from}-{to}"
        );
        OffsetRange { from, to }
    }

    /// `length()`.
    pub fn length(&self) -> usize {
        self.to - self.from
    }

    /// `contains(other)`.
    pub fn contains(&self, other: &OffsetRange) -> bool {
        self.from <= other.from && self.to >= other.to
    }
}

impl std::fmt::Display for OffsetRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[from={}, to={}]", self.from, self.to)
    }
}

/// `Passage`: a window of the text and the marked ranges inside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Passage {
    pub from: usize,
    pub to: usize,
    pub markers: Vec<OffsetRange>,
}

impl Passage {
    /// `new Passage(from, to, markers)`.
    pub fn new(from: usize, to: usize, markers: Vec<OffsetRange>) -> Self {
        Passage { from, to, markers }
    }

    /// `length()`.
    pub fn length(&self) -> usize {
        self.to - self.from
    }
}

/// `MatchHighlighter.QueryOffsetRange`: a range and the query that matched it.
#[derive(Debug, Clone)]
pub struct QueryOffsetRange {
    pub query: Arc<Clause>,
    pub range: OffsetRange,
}

// ---------------------------------------------------------------------------
// PassageAdjuster
// ---------------------------------------------------------------------------

/// `PassageAdjuster`.
pub trait PassageAdjuster {
    /// `currentValue(value)`.
    fn current_value(&mut self, value: &[u16]);
    /// `adjust(passage)`: a range inside the passage's.
    fn adjust(&mut self, passage: &Passage) -> OffsetRange;
}

/// `BreakIteratorShrinkingAdjuster`: a passage drawn in to word boundaries
/// (`BreakIterator.getWordInstance(Locale.ROOT)`) without losing a marker.
pub struct BreakIteratorShrinkingAdjuster {
    value: Vec<u16>,
    bi: WordBreakIterator,
}

impl Default for BreakIteratorShrinkingAdjuster {
    fn default() -> Self {
        BreakIteratorShrinkingAdjuster {
            value: Vec::new(),
            bi: WordBreakIterator::from_utf16(Vec::new()),
        }
    }
}

fn is_whitespace_unit(u: u16) -> bool {
    lucene_analysis::java_character::is_whitespace(u32::from(u))
}

impl PassageAdjuster for BreakIteratorShrinkingAdjuster {
    fn current_value(&mut self, value: &[u16]) {
        self.value = value.to_vec();
        self.bi = WordBreakIterator::from_utf16(value.to_vec());
    }

    fn adjust(&mut self, passage: &Passage) -> OffsetRange {
        let len = self.value.len();
        let mut from = passage.from;
        if from > 0 {
            while !self.bi.is_boundary(from) || (from < len && is_whitespace_unit(self.value[from]))
            {
                match self.bi.following(from) {
                    None => {
                        from = passage.from;
                        break;
                    }
                    Some(f) => from = f,
                }
            }
            if from == len {
                from = passage.from;
            }
        }
        let mut to = passage.to;
        if to != len {
            while !self.bi.is_boundary(to) || (to > 0 && is_whitespace_unit(self.value[to - 1])) {
                match self.bi.preceding(to) {
                    None => {
                        to = passage.to;
                        break;
                    }
                    Some(t) => to = t,
                }
            }
            if to == 0 {
                to = passage.to;
            }
        }
        for r in &passage.markers {
            from = from.min(r.from);
            to = to.max(r.to);
        }
        if from > to {
            from = to;
        }
        OffsetRange::new(from, to)
    }
}

// ---------------------------------------------------------------------------
// PassageSelector
// ---------------------------------------------------------------------------

/// `PassageSelector.DEFAULT_SCORER`: more markers, then more marked
/// characters, then the earlier passage.
pub fn default_scorer(a: &Passage, b: &Passage) -> Ordering {
    let v = a.markers.len().cmp(&b.markers.len());
    if v != Ordering::Equal {
        return v;
    }
    let len1: usize = a.markers.iter().map(OffsetRange::length).sum();
    let len2: usize = b.markers.iter().map(OffsetRange::length).sum();
    if len1 != len2 {
        return len1.cmp(&len2);
    }
    b.from.cmp(&a.from)
}

/// Lucene's `PriorityQueue`, for `insertWithOverflow`'s exact tie behaviour.
struct PriorityQueue<T> {
    heap: Vec<Option<T>>,
    size: usize,
    max_size: usize,
}

impl<T> PriorityQueue<T> {
    fn new(max_size: usize) -> Self {
        let mut heap = Vec::with_capacity(max_size + 1);
        heap.resize_with(max_size + 1, || None);
        PriorityQueue {
            heap,
            size: 0,
            max_size,
        }
    }

    fn less_than(&self, a: usize, b: usize, lt: &dyn Fn(&T, &T) -> bool) -> bool {
        match (&self.heap[a], &self.heap[b]) {
            (Some(x), Some(y)) => lt(x, y),
            _ => false,
        }
    }

    fn up_heap(&mut self, mut i: usize, lt: &dyn Fn(&T, &T) -> bool) {
        while i > 1 && self.less_than(i, i >> 1, lt) {
            self.heap.swap(i, i >> 1);
            i >>= 1;
        }
    }

    fn down_heap(&mut self, mut i: usize, lt: &dyn Fn(&T, &T) -> bool) {
        loop {
            let mut j = i << 1;
            if j > self.size {
                break;
            }
            let k = j + 1;
            if k <= self.size && self.less_than(k, j, lt) {
                j = k;
            }
            if self.less_than(j, i, lt) {
                self.heap.swap(i, j);
                i = j;
            } else {
                break;
            }
        }
    }

    /// `insertWithOverflow(element)`.
    fn insert_with_overflow(&mut self, element: T, lt: &dyn Fn(&T, &T) -> bool) {
        if self.size < self.max_size {
            self.size += 1;
            self.heap[self.size] = Some(element);
            self.up_heap(self.size, lt);
        } else if self.size > 0 {
            let replaces = self.heap[1].as_ref().is_some_and(|top| !lt(&element, top));
            if replaces {
                self.heap[1] = Some(element);
                self.down_heap(1, lt);
            }
        }
    }

    /// `pop()`: the least element.
    fn pop(&mut self, lt: &dyn Fn(&T, &T) -> bool) -> Option<T> {
        if self.size == 0 {
            return None;
        }
        let result = self.heap[1].take();
        self.heap[1] = self.heap[self.size].take();
        self.size -= 1;
        if self.size > 0 {
            self.down_heap(1, lt);
        }
        result
    }
}

/// How two passages compare: the greater is the better.
type PassageScorerFn = Box<dyn Fn(&Passage, &Passage) -> Ordering>;

/// `PassageSelector`: the best passages of a value.
pub struct PassageSelector {
    passage_scorer: PassageScorerFn,
    passage_adjuster: Option<Box<dyn PassageAdjuster>>,
}

impl Default for PassageSelector {
    /// `new PassageSelector()`.
    fn default() -> Self {
        PassageSelector {
            passage_scorer: Box::new(default_scorer),
            passage_adjuster: None,
        }
    }
}

impl PassageSelector {
    /// `new PassageSelector(passageScorer, passageAdjuster)`.
    pub fn new(
        passage_scorer: impl Fn(&Passage, &Passage) -> Ordering + 'static,
        passage_adjuster: Option<Box<dyn PassageAdjuster>>,
    ) -> Self {
        PassageSelector {
            passage_scorer: Box::new(passage_scorer),
            passage_adjuster,
        }
    }

    /// `pickBest(value, markers, maxPassageWindow, maxPassages)`.
    pub fn pick_best_whole(
        &mut self,
        value: &[u16],
        markers: &[OffsetRange],
        max_passage_window: usize,
        max_passages: usize,
    ) -> Vec<Passage> {
        self.pick_best(
            value,
            markers,
            max_passage_window,
            max_passages,
            &[OffsetRange::new(0, value.len())],
        )
    }

    /// `pickBest(value, markers, maxPassageWindow, maxPassages,
    /// permittedPassageRanges)`: passages sorted by position.
    pub fn pick_best(
        &mut self,
        value: &[u16],
        markers: &[OffsetRange],
        max_passage_window: usize,
        max_passages: usize,
        permitted: &[OffsetRange],
    ) -> Vec<Passage> {
        if value.is_empty() || max_passage_window == 0 {
            return Vec::new();
        }
        let pq_size = max_passages.max(16);
        let scorer = &self.passage_scorer;
        let lt = |a: &Passage, b: &Passage| scorer(a, b) == Ordering::Less;
        let mut pq: PriorityQueue<Passage> = PriorityQueue::new(pq_size);

        let mut markers = split_or_truncate_to_windows(markers, max_passage_window, permitted);
        markers.sort_by_key(|r| (r.from, r.to));
        let max = markers.len();
        let mut marker_index = 0usize;
        let value_len = value.len();
        'next_range: for range in permitted {
            let range_to = range.to.min(value_len);
            if range.from >= range_to {
                continue;
            }
            while marker_index < max {
                let m = markers[marker_index];
                if m.from >= range_to {
                    continue 'next_range;
                }
                if m.from >= range.from && m.to <= range_to && m.length() <= max_passage_window {
                    let (mf, mt, w) = (m.from as i64, m.to as i64, max_passage_window as i64);
                    let mut from = (mf + mt - w) / 2;
                    let mut to = (mf + mt + w) / 2;
                    let (rf, rt) = (range.from as i64, range_to as i64);
                    if from < rf {
                        to += rf - from;
                        from = rf;
                    }
                    if to > rt {
                        from -= to - rt;
                        to = rt;
                        if from < rf {
                            from = rf;
                        }
                    }
                    if from < to && to <= value_len as i64 {
                        let (from, to) = (from as usize, to as usize);
                        let mut inside = Vec::new();
                        let mut i = marker_index;
                        while i > 0 && markers[i - 1].from >= from {
                            i -= 1;
                        }
                        while i < max && markers[i].from < to {
                            if markers[i].to <= to {
                                inside.push(markers[i]);
                            }
                            i += 1;
                        }
                        if !inside.is_empty() {
                            pq.insert_with_overflow(Passage::new(from, to, inside), &lt);
                        }
                    }
                }
                marker_index += 1;
            }
        }

        let mut passages: Vec<Option<Passage>> = if pq.size > 0 {
            let n = pq.size;
            let mut v: Vec<Option<Passage>> = (0..n).map(|_| None).collect();
            for i in (0..n).rev() {
                v[i] = pq.pop(&lt);
            }
            v
        } else {
            pick_default_passage(value_len, max_passage_window, max_passages, permitted)
                .into_iter()
                .map(Some)
                .collect()
        };

        if let Some(adjuster) = self.passage_adjuster.as_mut() {
            adjuster.current_value(value);
            for slot in &mut passages {
                if let Some(p) = slot {
                    let new_range = adjuster.adjust(p);
                    if new_range.from != p.from || new_range.to != p.to {
                        debug_assert!(new_range.from >= p.from && new_range.to <= p.to);
                        *slot = Some(Passage::new(
                            new_range.from,
                            new_range.to,
                            p.markers.clone(),
                        ));
                    }
                }
            }
        }

        // drop empty passages and those adjacent to or overlapping a better one
        let mut last = 0;
        for i in 0..passages.len() {
            let Some(a) = passages[i].clone() else {
                continue;
            };
            if a.length() > 0 {
                passages[last] = Some(a.clone());
                last += 1;
                for slot in passages.iter_mut().skip(i + 1) {
                    if slot
                        .as_ref()
                        .is_some_and(|b| adjacent_or_overlapping(&a, b))
                    {
                        *slot = None;
                    }
                }
            }
        }
        last = last.min(max_passages);
        let mut out: Vec<Passage> = passages.into_iter().take(last).flatten().collect();
        out.sort_by_key(|p| p.from);
        out
    }
}

fn split_or_truncate_to_windows(
    markers: &[OffsetRange],
    max_passage_window: usize,
    permitted: &[OffsetRange],
) -> Vec<OffsetRange> {
    let mut out = Vec::with_capacity(markers.len());
    for marker in markers {
        for p in permitted {
            let from = marker.from.max(p.from);
            let mut to = marker.to.min(p.to);
            let mut new_slice = from != marker.from || to != marker.to;
            if from >= to {
                continue;
            }
            if to - from > max_passage_window {
                to = from + max_passage_window;
                new_slice = true;
            }
            out.push(if new_slice {
                OffsetRange::new(from, to)
            } else {
                *marker
            });
        }
    }
    out
}

fn pick_default_passage(
    value_len: usize,
    max_character_window: usize,
    max_passages: usize,
    permitted: &[OffsetRange],
) -> Vec<Passage> {
    let mut out = Vec::new();
    for o in permitted {
        if out.len() >= max_passages {
            break;
        }
        let to = value_len.min(o.to);
        if o.from < to {
            out.push(Passage::new(
                o.from,
                o.from + max_character_window.min(o.length()),
                Vec::new(),
            ));
        }
    }
    out
}

fn adjacent_or_overlapping(a: &Passage, b: &Passage) -> bool {
    if a.from >= b.from {
        a.from < b.to
    } else {
        a.to > b.from
    }
}

// ---------------------------------------------------------------------------
// PassageFormatter
// ---------------------------------------------------------------------------

type MarkerFn = Box<dyn Fn(&OffsetRange) -> String>;

/// `PassageFormatter`: passages as text, with marker strings around the
/// marked ranges and an ellipsis where a passage does not reach the edge of
/// its range.
pub struct PassageFormatter {
    ellipsis: String,
    marker_start: MarkerFn,
    marker_end: MarkerFn,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SliceType {
    End,
    Start,
}

impl PassageFormatter {
    /// `new PassageFormatter(ellipsis, markerStart, markerEnd)` with fixed
    /// marker strings.
    pub fn new(
        ellipsis: impl Into<String>,
        marker_start: impl Into<String>,
        marker_end: impl Into<String>,
    ) -> Self {
        let (s, e) = (marker_start.into(), marker_end.into());
        Self::with_markers(ellipsis, move |_| s.clone(), move |_| e.clone())
    }

    /// `new PassageFormatter(ellipsis, Function markerStart, Function
    /// markerEnd)`.
    pub fn with_markers(
        ellipsis: impl Into<String>,
        marker_start: impl Fn(&OffsetRange) -> String + 'static,
        marker_end: impl Fn(&OffsetRange) -> String + 'static,
    ) -> Self {
        PassageFormatter {
            ellipsis: ellipsis.into(),
            marker_start: Box::new(marker_start),
            marker_end: Box::new(marker_end),
        }
    }

    /// `format(value, passages, ranges)`: one string per passage, with the
    /// ellipsis where the passage stops short of its range.
    pub fn format(
        &self,
        value: &[u16],
        passages: &[Passage],
        ranges: &[OffsetRange],
    ) -> Vec<String> {
        if ranges.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut range_index = 0;
        let mut range = ranges[range_index];
        for passage in passages {
            while passage.from >= range.to {
                range_index += 1;
                if range_index == ranges.len() {
                    return result;
                }
                range = ranges[range_index];
            }
            let mut buf = String::new();
            if range.from < passage.from {
                buf.push_str(&self.ellipsis);
            }
            self.format_passage(&mut buf, value, passage);
            if range.to > passage.to {
                buf.push_str(&self.ellipsis);
            }
            result.push(buf);
        }
        result
    }

    /// `format(buf, value, passage)`.
    pub fn format_passage(&self, buf: &mut String, value: &[u16], passage: &Passage) {
        let text = |from: usize, to: usize| String::from_utf16_lossy(&value[from..to.max(from)]);
        match passage.markers.len() {
            0 => buf.push_str(&text(passage.from, passage.to)),
            1 => {
                let m = passage.markers[0];
                buf.push_str(&text(passage.from, m.from));
                buf.push_str(&(self.marker_start)(&m));
                buf.push_str(&text(m.from, m.to));
                buf.push_str(&(self.marker_end)(&m));
                buf.push_str(&text(m.to, passage.to));
            }
            _ => self.multiple_markers(value, passage, buf),
        }
    }

    fn multiple_markers(&self, value: &[u16], p: &Passage, b: &mut String) {
        let text = |from: usize, to: usize| String::from_utf16_lossy(&value[from..to.max(from)]);
        let mut marker_stack: Vec<OffsetRange> = Vec::new();
        let mut at = p.from;
        let max = p.to;
        let mut points: Vec<(usize, SliceType, OffsetRange)> = Vec::new();
        for m in &p.markers {
            points.push((m.from, SliceType::Start, *m));
            points.push((m.to, SliceType::End, *m));
        }
        // offset, then ends before starts, then the longer marker first
        points.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| {
                    let rank = |t: SliceType| if t == SliceType::Start { 2 } else { 1 };
                    rank(a.1).cmp(&rank(b.1))
                })
                .then_with(|| {
                    if a.1 == SliceType::Start {
                        b.2.to.cmp(&a.2.to)
                    } else {
                        b.2.from.cmp(&a.2.from)
                    }
                })
        });
        for (offset, kind, marker) in points {
            b.push_str(&text(at, offset));
            match kind {
                SliceType::Start => {
                    marker_stack.push(marker);
                    b.push_str(&(self.marker_start)(&marker));
                }
                SliceType::End => {
                    let idx = marker_stack.iter().rposition(|m| *m == marker).unwrap_or(0);
                    for m in &marker_stack[idx..] {
                        b.push_str(&(self.marker_end)(m));
                    }
                    marker_stack.remove(idx);
                    for m in &marker_stack[idx..] {
                        b.push_str(&(self.marker_start)(m));
                    }
                }
            }
            at = offset;
        }
        if at < max {
            b.push_str(&text(at, max));
        }
    }
}

// ---------------------------------------------------------------------------
// OffsetsRetrievalStrategy
// ---------------------------------------------------------------------------

/// `MatchRegionRetriever.FieldValueProvider`: a document's loaded stored
/// fields.
pub trait FieldValueProvider {
    /// `getValues(field)`.
    fn get_values(&self, field: &str) -> &[String];
    /// The fields with values, in stored order.
    fn fields(&self) -> Vec<String>;
}

/// The stored values the retriever loaded, in the order they were stored.
#[derive(Debug, Default, Clone)]
pub struct StoredValues {
    values: Vec<(String, Vec<String>)>,
}

impl StoredValues {
    /// Values for a field, for the tests' hand-built documents.
    #[cfg(test)]
    pub(crate) fn push_for_test(&mut self, field: &str, values: &[&str]) {
        self.values.push((
            field.to_string(),
            values.iter().map(|v| (*v).to_string()).collect(),
        ));
    }
}

impl FieldValueProvider for StoredValues {
    fn get_values(&self, field: &str) -> &[String] {
        self.values
            .iter()
            .find(|(f, _)| f == field)
            .map_or(&[][..], |(_, v)| v.as_slice())
    }

    fn fields(&self) -> Vec<String> {
        self.values.iter().map(|(f, _)| f.clone()).collect()
    }
}

/// `OffsetsRetrievalStrategy`: the offsets of a field's matches.
pub trait OffsetsRetrievalStrategy {
    /// `get(matchesIterator, doc)`.
    fn get(
        &self,
        matches: &mut dyn MatchesIterator,
        doc: &dyn FieldValueProvider,
    ) -> Result<Vec<OffsetRange>>;
}

/// `OffsetsFromPositions`: the offsets of the tokens at the matched
/// positions, found by analyzing the stored values again.
pub struct OffsetsFromPositions {
    field: String,
    analyzer: Arc<Analyzer>,
}

impl OffsetsFromPositions {
    /// `new OffsetsFromPositions(field, analyzer)`.
    pub fn new(field: impl Into<String>, analyzer: Arc<Analyzer>) -> Self {
        OffsetsFromPositions {
            field: field.into(),
            analyzer,
        }
    }

    fn convert_positions_to_offsets(
        &self,
        position_ranges: &[OffsetRange],
        values: &[String],
    ) -> Result<Vec<OffsetRange>> {
        if position_ranges.is_empty() {
            return Ok(Vec::new());
        }
        struct Span {
            from: i64,
            to: i64,
            left: i64,
            right: i64,
        }
        let mut spans: Vec<Span> = position_ranges
            .iter()
            .map(|r| Span {
                from: r.from as i64,
                to: r.to as i64,
                left: i64::from(i32::MAX),
                right: i64::from(i32::MIN),
            })
            .collect();
        let min_position = spans
            .iter()
            .map(|s| s.from)
            .min()
            .unwrap_or(i64::from(i32::MAX));
        let max_position = spans
            .iter()
            .map(|s| s.to)
            .max()
            .unwrap_or(i64::from(i32::MIN));
        let mut table: Vec<usize> = (0..spans.len()).collect();
        let mut span_count = table.len();
        let mut position = -1i64;
        let mut value_offset = 0i64;
        let analysis = |e: lucene_analysis::AnalysisError| Error::Analysis(e);
        let last_index = values.len().saturating_sub(1);
        for (value_index, value) in values.iter().enumerate() {
            let last_value = value_index == last_index;
            let mut ts = self
                .analyzer
                .token_stream(&self.field, value)
                .map_err(analysis)?;
            ts.reset().map_err(analysis)?;
            while ts.increment_token().map_err(analysis)? {
                let a = ts.attributes();
                position += i64::from(a.position_increment());
                if position >= min_position {
                    let start = value_offset + i64::from(a.start_offset());
                    let end = value_offset + i64::from(a.end_offset());
                    let mut j = 0;
                    for i in 0..span_count {
                        table[j] = table[i];
                        let span = &mut spans[table[j]];
                        if position >= span.from {
                            if position <= span.to {
                                span.left = span.left.min(start);
                                span.right = span.right.max(end);
                            } else {
                                continue;
                            }
                        }
                        j += 1;
                    }
                    span_count = j;
                    if position > max_position && last_value {
                        break;
                    }
                }
            }
            ts.end().map_err(analysis)?;
            position += i64::from(ts.attributes().position_increment())
                + i64::from(self.analyzer.position_increment_gap_for_field(&self.field));
            value_offset += i64::from(ts.attributes().end_offset())
                + i64::from(self.analyzer.offset_gap_for_field(&self.field));
            ts.close().map_err(analysis)?;
        }
        let mut converted = Vec::with_capacity(spans.len());
        for span in &spans {
            if span.left == i64::from(i32::MAX) || span.right == i64::from(i32::MIN) {
                return Err(Error::Runtime(format!(
                    "One of the offsets missing for position range: [from={}, to={}, L: {}, R: {}]",
                    span.from, span.to, span.left, span.right
                )));
            }
            converted.push(OffsetRange::new(
                usize::try_from(span.left).unwrap_or(0),
                usize::try_from(span.right).unwrap_or(0),
            ));
        }
        Ok(converted)
    }
}

impl OffsetsRetrievalStrategy for OffsetsFromPositions {
    fn get(
        &self,
        matches: &mut dyn MatchesIterator,
        doc: &dyn FieldValueProvider,
    ) -> Result<Vec<OffsetRange>> {
        let mut position_ranges = Vec::new();
        while matches.next().map_err(io)? {
            let (from, to) = (matches.start_position(), matches.end_position());
            if from < 0 || to < 0 {
                return Err(Error::Io(format!(
                    "Matches API returned negative positions for field: {}",
                    self.field
                )));
            }
            position_ranges.push(OffsetRange::new(from as usize, to as usize));
        }
        self.convert_positions_to_offsets(&position_ranges, doc.get_values(&self.field))
    }
}

/// `OffsetsFromMatchIterator`: the offsets the matches themselves carry,
/// else those of [`OffsetsFromPositions`].
pub struct OffsetsFromMatchIterator {
    field: String,
    no_offsets_fallback: OffsetsFromPositions,
}

impl OffsetsFromMatchIterator {
    /// `new OffsetsFromMatchIterator(field, noOffsetsFallback)`.
    pub fn new(field: impl Into<String>, no_offsets_fallback: OffsetsFromPositions) -> Self {
        OffsetsFromMatchIterator {
            field: field.into(),
            no_offsets_fallback,
        }
    }
}

impl OffsetsRetrievalStrategy for OffsetsFromMatchIterator {
    fn get(
        &self,
        matches: &mut dyn MatchesIterator,
        doc: &dyn FieldValueProvider,
    ) -> Result<Vec<OffsetRange>> {
        let mut position_ranges = Vec::new();
        let mut offset_ranges: Option<Vec<OffsetRange>> = Some(Vec::new());
        while matches.next().map_err(io)? {
            let (from_position, to_position) = (matches.start_position(), matches.end_position());
            if from_position < 0 || to_position < 0 {
                return Err(Error::Io(format!(
                    "Matches API returned negative positions for field: {}",
                    self.field
                )));
            }
            position_ranges.push(OffsetRange::new(
                from_position as usize,
                to_position as usize,
            ));
            if let Some(ranges) = offset_ranges.as_mut() {
                let (from, to) = (matches.start_offset(), matches.end_offset());
                if from < 0 || to < 0 {
                    offset_ranges = None;
                } else {
                    ranges.push(OffsetRange::new(from as usize, to as usize));
                }
            }
        }
        match offset_ranges {
            None => self
                .no_offsets_fallback
                .convert_positions_to_offsets(&position_ranges, doc.get_values(&self.field)),
            Some(r) => Ok(r),
        }
    }
}

/// `OffsetsFromTokens`: the stored values analyzed again, the tokens whose
/// term the matching queries name.
pub struct OffsetsFromTokens {
    field: String,
    analyzer: Arc<Analyzer>,
}

impl OffsetsFromTokens {
    /// `new OffsetsFromTokens(field, analyzer)`.
    pub fn new(field: impl Into<String>, analyzer: Arc<Analyzer>) -> Self {
        OffsetsFromTokens {
            field: field.into(),
            analyzer,
        }
    }
}

impl OffsetsRetrievalStrategy for OffsetsFromTokens {
    fn get(
        &self,
        matches: &mut dyn MatchesIterator,
        doc: &dyn FieldValueProvider,
    ) -> Result<Vec<OffsetRange>> {
        let values = doc.get_values(&self.field);
        let mut match_terms: HashSet<Vec<u8>> = HashSet::new();
        while matches.next().map_err(io)? {
            let mut collector = TermCollector::default();
            query_visitor::visit(matches.query(), &mut collector);
            for t in collector.terms {
                if t.field == self.field {
                    match_terms.insert(t.bytes);
                }
            }
        }
        let analysis = |e: lucene_analysis::AnalysisError| Error::Analysis(e);
        let mut ranges = Vec::new();
        let mut value_offset = 0i64;
        for value in values {
            let mut ts = self
                .analyzer
                .token_stream(&self.field, value)
                .map_err(analysis)?;
            ts.reset().map_err(analysis)?;
            while ts.increment_token().map_err(analysis)? {
                let a = ts.attributes();
                if match_terms.contains(a.term_bytes()) {
                    let start = value_offset + i64::from(a.start_offset());
                    let end = value_offset + i64::from(a.end_offset());
                    ranges.push(OffsetRange::new(
                        usize::try_from(start).unwrap_or(0),
                        usize::try_from(end).unwrap_or(0),
                    ));
                }
            }
            ts.end().map_err(analysis)?;
            value_offset += i64::from(ts.attributes().end_offset())
                + i64::from(self.analyzer.offset_gap_for_field(&self.field));
            ts.close().map_err(analysis)?;
        }
        Ok(ranges)
    }
}

/// `OffsetsFromValues`: the whole of each stored value.
pub struct OffsetsFromValues {
    field: String,
    analyzer: Arc<Analyzer>,
}

impl OffsetsFromValues {
    /// `new OffsetsFromValues(field, analyzer)`.
    pub fn new(field: impl Into<String>, analyzer: Arc<Analyzer>) -> Self {
        OffsetsFromValues {
            field: field.into(),
            analyzer,
        }
    }
}

impl OffsetsRetrievalStrategy for OffsetsFromValues {
    fn get(
        &self,
        _matches: &mut dyn MatchesIterator,
        doc: &dyn FieldValueProvider,
    ) -> Result<Vec<OffsetRange>> {
        let analysis = |e: lucene_analysis::AnalysisError| Error::Analysis(e);
        let mut ranges = Vec::new();
        let mut value_offset = 0i64;
        for value in doc.get_values(&self.field) {
            let mut ts = self
                .analyzer
                .token_stream(&self.field, value)
                .map_err(analysis)?;
            ts.reset().map_err(analysis)?;
            let start = value_offset;
            while ts.increment_token().map_err(analysis)? {}
            ts.end().map_err(analysis)?;
            value_offset += i64::from(ts.attributes().end_offset());
            ranges.push(OffsetRange::new(
                usize::try_from(start).unwrap_or(0),
                usize::try_from(value_offset).unwrap_or(0),
            ));
            value_offset += i64::from(self.analyzer.offset_gap_for_field(&self.field));
            ts.close().map_err(analysis)?;
        }
        Ok(ranges)
    }
}

/// A strategy that always fails with the given message.
struct Failing(String);

impl OffsetsRetrievalStrategy for Failing {
    fn get(
        &self,
        _matches: &mut dyn MatchesIterator,
        _doc: &dyn FieldValueProvider,
    ) -> Result<Vec<OffsetRange>> {
        Err(Error::Io(self.0.clone()))
    }
}

/// `OffsetsRetrievalStrategySupplier`.
pub type OffsetsRetrievalStrategySupplier = Box<dyn Fn(&str) -> Box<dyn OffsetsRetrievalStrategy>>;

/// `MatchRegionRetriever.computeOffsetRetrievalStrategies(reader, analyzer)`:
/// the strategy each field's index options allow.
pub fn compute_offset_retrieval_strategies(
    reader: &DirectoryReader,
    analyzer: Arc<Analyzer>,
) -> OffsetsRetrievalStrategySupplier {
    // `FieldInfos.getMergedFieldInfos`: the first segment to name a field
    let mut infos: HashMap<String, IndexOptions> = HashMap::new();
    for seg in reader.segment_readers() {
        for fi in &seg.field_infos().fields {
            infos.entry(fi.name.clone()).or_insert(fi.index_options);
        }
    }
    Box::new(move |field: &str| -> Box<dyn OffsetsRetrievalStrategy> {
        let Some(options) = infos.get(field) else {
            return Box::new(Failing(format!("FieldInfo is null for field: {field}")));
        };
        match options {
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets => {
                Box::new(OffsetsFromMatchIterator::new(
                    field,
                    OffsetsFromPositions::new(field, Arc::clone(&analyzer)),
                ))
            }
            IndexOptions::DocsAndFreqsAndPositions => {
                Box::new(OffsetsFromPositions::new(field, Arc::clone(&analyzer)))
            }
            IndexOptions::DocsAndCustomFreqs | IndexOptions::DocsAndFreqs | IndexOptions::Docs => {
                Box::new(OffsetsFromTokens::new(field, Arc::clone(&analyzer)))
            }
            IndexOptions::None => Box::new(Failing(format!(
                "Field is indexed without positions and/or offsets: {field}, {options:?}"
            ))),
        }
    })
}

// ---------------------------------------------------------------------------
// MatchRegionRetriever
// ---------------------------------------------------------------------------

/// `MatchesIteratorWithLimit`.
struct MatchesIteratorWithLimit {
    inner: BoxMatchesIterator,
    limit: usize,
}

impl MatchesIterator for MatchesIteratorWithLimit {
    fn next(&mut self) -> crate::Result<bool> {
        if self.limit == 0 {
            return Ok(false);
        }
        self.limit -= 1;
        self.inner.next()
    }
    fn start_position(&self) -> i32 {
        self.inner.start_position()
    }
    fn end_position(&self) -> i32 {
        self.inner.end_position()
    }
    fn start_offset(&self) -> i32 {
        self.inner.start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.inner.end_offset()
    }
    fn query(&self) -> &Clause {
        self.inner.query()
    }
}

/// `MatchRegionRetriever`.
pub struct MatchRegionRetriever {
    query: Clause,
    offset_strategies: HashMap<String, Box<dyn OffsetsRetrievalStrategy>>,
    query_affected_highlighted_fields: BTreeSet<String>,
    fields_to_load_unconditionally: Box<dyn Fn(&str) -> bool>,
}

/// The consumer of `highlightDocuments`: the document id, its leaf-local id,
/// the values of the fields loaded and the offsets of its hits per field.
pub type MatchOffsetsConsumer<'c> = dyn FnMut(i32, i32, &dyn FieldValueProvider, &BTreeMap<String, Vec<OffsetRange>>) -> Result<()>
    + 'c;

struct AcceptFieldCollector<'f> {
    test: &'f dyn Fn(&str) -> bool,
    fields: RefCell<BTreeSet<String>>,
}

impl QueryVisitor for AcceptFieldCollector<'_> {
    fn accept_field(&self, field: &str) -> bool {
        if (self.test)(field) {
            self.fields.borrow_mut().insert(field.to_string());
        }
        false
    }
}

impl MatchRegionRetriever {
    /// `new MatchRegionRetriever(searcher, query, fieldOffsetStrategySupplier,
    /// fieldsToLoadUnconditionally, fieldsToLoadIfWithHits)`.
    pub fn new(
        query: &Clause,
        strategies: &dyn Fn(&str) -> Box<dyn OffsetsRetrievalStrategy>,
        fields_to_load_unconditionally: impl Fn(&str) -> bool + 'static,
        fields_to_load_if_with_hits: &dyn Fn(&str) -> bool,
    ) -> Self {
        let mut collector = AcceptFieldCollector {
            test: fields_to_load_if_with_hits,
            fields: RefCell::new(BTreeSet::new()),
        };
        query_visitor::visit(query, &mut collector);
        let fields = collector.fields.into_inner();
        let offset_strategies = fields.iter().map(|f| (f.clone(), strategies(f))).collect();
        MatchRegionRetriever {
            query: query.clone(),
            offset_strategies,
            query_affected_highlighted_fields: fields,
            fields_to_load_unconditionally: Box::new(fields_to_load_unconditionally),
        }
    }

    fn should_load_stored_field(&self, field: &str) -> bool {
        (self.fields_to_load_unconditionally)(field)
            || self.query_affected_highlighted_fields.contains(field)
    }

    /// `highlightDocuments(docIds, consumer, maxHitsPerField)`: `doc_ids`
    /// must be increasing.
    pub fn highlight_documents(
        &self,
        searcher: &IndexSearcher<'_, '_>,
        reader: &DirectoryReader,
        doc_ids: &[i32],
        consumer: &mut MatchOffsetsConsumer<'_>,
        max_hits_per_field: &dyn Fn(&str) -> usize,
    ) -> Result<()> {
        let mut previous = -1;
        for &doc_id in doc_ids {
            if doc_id < previous {
                return Err(Error::Runtime(
                    "Input document IDs must be sorted (increasing).".into(),
                ));
            }
            previous = doc_id;
            let leaf = searcher
                .segment_of(doc_id)
                .ok_or_else(|| Error::IllegalArgument(format!("doc {doc_id} is in no segment")))?;
            let doc_base = searcher.segments()[leaf].doc_base;
            let local = doc_id - doc_base;
            let seg = &reader.segment_readers()[leaf];
            let mut values = StoredValues::default();
            if let Some(doc) = seg.stored_document(local).map_err(io)? {
                for f in &doc.fields {
                    let Some(fi) = seg.field_infos().field_by_number(f.field_number) else {
                        continue;
                    };
                    if !self.should_load_stored_field(&fi.name) {
                        continue;
                    }
                    let Some(text) = stored_string_value(&f.value) else {
                        continue;
                    };
                    match values.values.iter_mut().find(|(n, _)| *n == fi.name) {
                        Some((_, v)) => v.push(text),
                        None => values.values.push((fi.name.clone(), vec![text])),
                    }
                }
            }
            let mut hits = BTreeMap::new();
            self.highlight_document(searcher, doc_id, &values, max_hits_per_field, &mut hits)?;
            consumer(doc_id, local, &values, &hits)?;
        }
        Ok(())
    }

    /// `highlightDocument(...)`: the offsets of each affected field's matches.
    pub fn highlight_document(
        &self,
        searcher: &IndexSearcher<'_, '_>,
        doc_id: i32,
        doc: &dyn FieldValueProvider,
        max_hits_per_field: &dyn Fn(&str) -> usize,
        output: &mut BTreeMap<String, Vec<OffsetRange>>,
    ) -> Result<()> {
        let Some(matches) = crate::matches::matches(searcher, &self.query, doc_id).map_err(io)?
        else {
            return Ok(());
        };
        for field in &self.query_affected_highlighted_fields {
            let Some(mut it) = matches.get_matches(field).map_err(io)? else {
                continue;
            };
            let Some(strategy) = self.offset_strategies.get(field) else {
                return Err(Error::Io(format!(
                    "Non-empty matches but no offset retrieval strategy for field: {field}"
                )));
            };
            let max_hits = max_hits_per_field(field);
            let ranges = if max_hits == usize::MAX {
                strategy.get(it.as_mut(), doc)?
            } else {
                let mut limited = MatchesIteratorWithLimit {
                    inner: it,
                    limit: max_hits,
                };
                strategy.get(&mut limited, doc)?
            };
            if !ranges.is_empty() {
                output.insert(field.clone(), ranges);
            }
        }
        Ok(())
    }
}

/// `StoredFieldsVisitor.addField`.
fn stored_string_value(value: &lucene_codecs::stored_fields::FieldValue) -> Option<String> {
    use lucene_codecs::stored_fields::FieldValue;
    match value {
        FieldValue::String(s) => Some(s.clone()),
        FieldValue::Int(v) => Some(v.to_string()),
        FieldValue::Long(v) => Some(v.to_string()),
        FieldValue::Float(v) => Some(crate::function::java_float(*v)),
        FieldValue::Double(v) => Some(crate::function::java_double(*v)),
        FieldValue::Binary(_) => None,
    }
}

// ---------------------------------------------------------------------------
// MatchHighlighter
// ---------------------------------------------------------------------------

/// `MatchHighlighter.FieldValueHighlighter`.
pub trait FieldValueHighlighter {
    /// `isApplicable(field, hasMatches)`.
    fn is_applicable(&self, field: &str, has_matches: bool) -> bool;
    /// `format(field, values, contiguousValue, valueRanges, matchOffsets)`;
    /// `None` leaves the field out.
    fn format(
        &self,
        field: &str,
        values: &[String],
        contiguous_value: &[u16],
        value_ranges: &[OffsetRange],
        match_offsets: Option<&[QueryOffsetRange]>,
    ) -> Option<Vec<String>>;
    /// `alwaysFetchedFields()`.
    fn always_fetched_fields(&self) -> Vec<String> {
        Vec::new()
    }
}

/// `FieldValueHighlighter.or(other)`.
pub struct OrHighlighter {
    first: Box<dyn FieldValueHighlighter>,
    second: Box<dyn FieldValueHighlighter>,
}

impl OrHighlighter {
    /// `first.or(second)`.
    pub fn new(
        first: impl FieldValueHighlighter + 'static,
        second: impl FieldValueHighlighter + 'static,
    ) -> Self {
        OrHighlighter {
            first: Box::new(first),
            second: Box::new(second),
        }
    }
}

impl FieldValueHighlighter for OrHighlighter {
    fn is_applicable(&self, field: &str, has_matches: bool) -> bool {
        self.first.is_applicable(field, has_matches)
            || self.second.is_applicable(field, has_matches)
    }

    fn format(
        &self,
        field: &str,
        values: &[String],
        contiguous_value: &[u16],
        value_ranges: &[OffsetRange],
        match_offsets: Option<&[QueryOffsetRange]>,
    ) -> Option<Vec<String>> {
        let has = match_offsets.is_some_and(|m| !m.is_empty());
        let delegate = if self.first.is_applicable(field, has) {
            &self.first
        } else {
            &self.second
        };
        delegate.format(field, values, contiguous_value, value_ranges, match_offsets)
    }

    fn always_fetched_fields(&self) -> Vec<String> {
        let mut union: BTreeSet<String> = self.first.always_fetched_fields().into_iter().collect();
        union.extend(self.second.always_fetched_fields());
        union.into_iter().collect()
    }
}

type FormatFn = Box<
    dyn Fn(
        &str,
        &[String],
        &[u16],
        &[OffsetRange],
        Option<&[QueryOffsetRange]>,
    ) -> Option<Vec<String>>,
>;

type ApplicableFn = Box<dyn Fn(&str, bool) -> bool>;

/// The highlighters of `FieldValueHighlighters`.
struct SimpleFieldHighlighter {
    applicable: ApplicableFn,
    always: Vec<String>,
    format: FormatFn,
}

impl FieldValueHighlighter for SimpleFieldHighlighter {
    fn is_applicable(&self, field: &str, has_matches: bool) -> bool {
        (self.applicable)(field, has_matches)
    }

    fn format(
        &self,
        field: &str,
        values: &[String],
        contiguous_value: &[u16],
        value_ranges: &[OffsetRange],
        match_offsets: Option<&[QueryOffsetRange]>,
    ) -> Option<Vec<String>> {
        (self.format)(field, values, contiguous_value, value_ranges, match_offsets)
    }

    fn always_fetched_fields(&self) -> Vec<String> {
        self.always.clone()
    }
}

/// `FieldValueHighlighters`.
pub struct FieldValueHighlighters;

impl FieldValueHighlighters {
    /// `defaultPassageSelector()`.
    pub fn default_passage_selector() -> PassageSelector {
        PassageSelector::new(
            default_scorer,
            Some(Box::new(BreakIteratorShrinkingAdjuster::default())),
        )
    }

    /// `maxLeadingCharacters(maxLeadingCharacters, ellipsis, fields)`.
    pub fn max_leading_characters(
        max_leading_characters: usize,
        ellipsis: &str,
        fields: HashSet<String>,
    ) -> impl FieldValueHighlighter {
        let selector = RefCell::new(Self::default_passage_selector());
        let formatter = PassageFormatter::new(ellipsis, "", "");
        let always: Vec<String> = fields.iter().cloned().collect();
        SimpleFieldHighlighter {
            applicable: Box::new(move |field, _| fields.contains(field)),
            always,
            format: Box::new(move |_field, _values, contiguous, ranges, _offsets| {
                let best = selector.borrow_mut().pick_best(
                    contiguous,
                    &[],
                    max_leading_characters,
                    1,
                    ranges,
                );
                Some(formatter.format(contiguous, &best, ranges))
            }),
        }
    }

    /// `highlighted(maxPassageWindow, maxPassages, passageFormatter,
    /// matchFields)`.
    pub fn highlighted(
        max_passage_window: usize,
        max_passages: usize,
        passage_formatter: PassageFormatter,
        match_fields: impl Fn(&str) -> bool + 'static,
    ) -> impl FieldValueHighlighter {
        let selector = RefCell::new(Self::default_passage_selector());
        SimpleFieldHighlighter {
            applicable: Box::new(move |field, has_matches| match_fields(field) && has_matches),
            always: Vec::new(),
            format: Box::new(move |_field, _values, contiguous, ranges, offsets| {
                let markers: Vec<OffsetRange> =
                    offsets.unwrap_or(&[]).iter().map(|o| o.range).collect();
                let best = selector.borrow_mut().pick_best(
                    contiguous,
                    &markers,
                    max_passage_window,
                    max_passages,
                    ranges,
                );
                Some(passage_formatter.format(contiguous, &best, ranges))
            }),
        }
    }

    /// `verbatimValue(field, moreFields...)`.
    pub fn verbatim_value(fields: &[&str]) -> impl FieldValueHighlighter {
        let set: HashSet<String> = fields.iter().map(|f| (*f).to_string()).collect();
        let always: Vec<String> = set.iter().cloned().collect();
        SimpleFieldHighlighter {
            applicable: Box::new(move |field, _| set.contains(field)),
            always,
            format: Box::new(|_f, values, _c, _r, _o| Some(values.to_vec())),
        }
    }

    /// `skipRemaining()`.
    pub fn skip_remaining() -> impl FieldValueHighlighter {
        SimpleFieldHighlighter {
            applicable: Box::new(|_, _| true),
            always: Vec::new(),
            format: Box::new(|_f, _v, _c, _r, _o| None),
        }
    }
}

/// `MatchHighlighter.DocHighlights`.
#[derive(Debug, Clone, PartialEq)]
pub struct DocHighlights {
    pub doc_id: i32,
    /// The highlighted fields, in the order the values were stored.
    pub fields: Vec<(String, Vec<String>)>,
}

/// One document's hits across the queries.
struct DocHit {
    doc_id: i32,
    match_ranges: Vec<(String, Vec<QueryOffsetRange>)>,
    field_values: Vec<(String, Vec<String>)>,
}

/// `MatchHighlighter`.
pub struct MatchHighlighter {
    offsets_retrieval_strategies: OffsetsRetrievalStrategySupplier,
    analyzer: Arc<Analyzer>,
    fields_always_returned: HashSet<String>,
    field_highlighters: Vec<Box<dyn FieldValueHighlighter>>,
}

impl MatchHighlighter {
    /// `new MatchHighlighter(searcher, analyzer)`.
    pub fn new(reader: &DirectoryReader, analyzer: Arc<Analyzer>) -> Self {
        let strategies = compute_offset_retrieval_strategies(reader, Arc::clone(&analyzer));
        Self::with_strategies(analyzer, strategies)
    }

    /// `new MatchHighlighter(searcher, analyzer, offsetsRetrievalStrategies)`.
    pub fn with_strategies(
        analyzer: Arc<Analyzer>,
        offsets_retrieval_strategies: OffsetsRetrievalStrategySupplier,
    ) -> Self {
        MatchHighlighter {
            offsets_retrieval_strategies,
            analyzer,
            fields_always_returned: HashSet::new(),
            field_highlighters: Vec::new(),
        }
    }

    /// `appendFieldHighlighter(highlighter)`.
    pub fn append_field_highlighter(&mut self, highlighter: impl FieldValueHighlighter + 'static) {
        self.fields_always_returned
            .extend(highlighter.always_fetched_fields());
        self.field_highlighters.push(Box::new(highlighter));
    }

    /// `alwaysFetchFields(fields...)`.
    pub fn always_fetch_fields(&mut self, fields: &[&str]) {
        self.fields_always_returned
            .extend(fields.iter().map(|f| (*f).to_string()));
    }

    /// `highlight(topDocs, queries...)`: `top_docs` in the order to return.
    pub fn highlight(
        &self,
        searcher: &IndexSearcher<'_, '_>,
        reader: &DirectoryReader,
        top_docs: &[i32],
        queries: &[Clause],
    ) -> Result<Vec<DocHighlights>> {
        let mut doc_hits: Vec<(i32, Option<DocHit>)> = Vec::new();
        for &d in top_docs {
            if !doc_hits.iter().any(|(id, _)| *id == d) {
                doc_hits.push((d, None));
            }
        }
        let always = self.fields_always_returned.clone();
        let applicable_somewhere = |field: &str| {
            self.field_highlighters
                .iter()
                .any(|h| h.is_applicable(field, true) || h.is_applicable(field, false))
        };
        let mut sorted: Vec<i32> = top_docs.to_vec();
        sorted.sort_unstable();
        for q in queries {
            let rewritten = q.clone().rewrite();
            let always = always.clone();
            let retriever = MatchRegionRetriever::new(
                &rewritten,
                self.offsets_retrieval_strategies.as_ref(),
                move |f: &str| always.contains(f),
                &applicable_somewhere,
            );
            let shared_query = Arc::new(q.clone());
            retriever.highlight_documents(
                searcher,
                reader,
                &sorted,
                &mut |doc_id, _local, values, hits| {
                    let slot = doc_hits
                        .iter_mut()
                        .find(|(id, _)| *id == doc_id)
                        .map(|(_, s)| s);
                    let Some(slot) = slot else {
                        return Ok(());
                    };
                    let hit = slot.get_or_insert_with(|| DocHit {
                        doc_id,
                        match_ranges: Vec::new(),
                        field_values: values
                            .fields()
                            .into_iter()
                            .map(|f| {
                                let v = values.get_values(&f).to_vec();
                                (f, v)
                            })
                            .collect(),
                    });
                    for (field, offsets) in hits {
                        let target = match hit.match_ranges.iter_mut().find(|(f, _)| f == field) {
                            Some((_, t)) => t,
                            None => {
                                hit.match_ranges.push((field.clone(), Vec::new()));
                                &mut hit.match_ranges.last_mut().expect("just pushed").1
                            }
                        };
                        for o in offsets {
                            target.push(QueryOffsetRange {
                                query: Arc::clone(&shared_query),
                                range: *o,
                            });
                        }
                    }
                    Ok(())
                },
                &|_| usize::MAX,
            )?;
        }
        doc_hits
            .into_iter()
            .filter_map(|(_, h)| h)
            .map(|h| self.compute_doc_field_values(h))
            .collect()
    }

    fn compute_doc_field_values(&self, hit: DocHit) -> Result<DocHighlights> {
        let mut out = DocHighlights {
            doc_id: hit.doc_id,
            fields: Vec::new(),
        };
        for (field, values) in &hit.field_values {
            let contiguous = self.contiguous_field_value(field, values);
            let value_ranges = self.compute_value_ranges(field, values);
            let offsets = hit
                .match_ranges
                .iter()
                .find(|(f, _)| f == field)
                .map(|(_, o)| o.as_slice());
            let highlighter = self
                .field_highlighters
                .iter()
                .find(|h| h.is_applicable(field, offsets.is_some()))
                .ok_or_else(|| {
                    Error::Runtime(format!(
                        "No field highlighter could be matched to field: {field}"
                    ))
                })?;
            if let Some(formatted) =
                highlighter.format(field, values, &contiguous, &value_ranges, offsets)
            {
                out.fields.push((field.clone(), formatted));
            }
        }
        Ok(out)
    }

    fn compute_value_ranges(&self, field: &str, values: &[String]) -> Vec<OffsetRange> {
        let gap = usize::try_from(self.analyzer.offset_gap_for_field(field)).unwrap_or(0);
        let mut ranges = Vec::new();
        let mut offset = 0;
        for v in values {
            let len = v.encode_utf16().count();
            ranges.push(OffsetRange::new(offset, offset + len));
            offset += len + gap;
        }
        ranges
    }

    fn contiguous_field_value(&self, field: &str, values: &[String]) -> Vec<u16> {
        let gap = usize::try_from(self.analyzer.offset_gap_for_field(field)).unwrap_or(0);
        let padding = " ".repeat(gap);
        values.join(&padding).encode_utf16().collect()
    }
}
