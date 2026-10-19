//! `DocValuesMultiRangeQuery` and the two queries its builders make:
//! `SortedNumericDocValuesMultiRangeQuery` (a value of a `NUMERIC`/
//! `SORTED_NUMERIC` field in any of several `long` ranges) and
//! `SortedSetDocValuesMultiRangeQuery` (an ordinal of a `SORTED`/
//! `SORTED_SET` field in any of several term ranges), both at a constant
//! score.
//!
//! The numeric query resolves its ranges once, in the constructor
//! (`resolveOverlaps`): every range's two edges and every one-value range's
//! point go through Lucene's `PriorityQueue` by value, points first on
//! ties, and a sweep over them keeps the outermost spans; reversed ranges
//! are dropped -- by the sign of `lower - upper` in wrapping `long`
//! arithmetic, as Java computes it. A document matches when one of its
//! values has a span starting at or below it (the floor of a sorted set)
//! that reaches it. The term query turns its sorted ranges into ordinal
//! ranges per segment (`createOrdRanges`), merging adjacent and overlapping
//! ones.
//!
//! With a doc-values skipper the numeric query walks Java's skip-block
//! approximation and reproduces its `matches()` reading a stale iterator
//! (see `StaleCursor`): Java returns extra documents there, and so does
//! this port. The term query's skipper only rules a segment out
//! (`skipper.minValue() > maxOrd || skipper.maxValue() < minOrd`); its
//! hits do not depend on the approximation.
//!
//! Deliberate differences: a numeric query whose every range was reversed fails
//! with an error where Java's `getFirst()` throws `NoSuchElementException`;
//! a stale read past the field's last value ends the match (Java reads past
//! the column: an `EOFException`, or a value decoded from `DirectWriter`'s
//! padding that may match).
//! `toString`, `equals`/`hashCode` and `visit` are not ported.

use std::collections::BTreeMap;

use lucene_codecs::doc_values::{
    DocValuesSkipIndex, DocValuesSkipper, NumericReader, SortedNumericReader,
};
use lucene_codecs::terms_dict::{SeekStatus, TermsDict};

use super::saturated;
use crate::collector::ScoringCollector;
use crate::document::doc_values_queries::{
    numeric_column, ords_column, skip_index, store_err, NumericColumn, OrdsColumn,
};
use crate::document::{
    collect_live, field_info, reader, DocumentQuery, MatchNoDocs, SortedNumericDocValuesRangeQuery,
    SortedSetDocValuesRangeQuery,
};
use crate::java_priority_queue::JavaPriorityQueue;
use crate::multi_segment::OpenSegment;
use crate::{Error, Result};

/// `DocValuesMultiRangeQuery.LongRange`: inclusive `long` bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LongRange {
    pub lower: i64,
    pub upper: i64,
}

impl LongRange {
    pub fn new(lower: i64, upper: i64) -> Self {
        LongRange { lower, upper }
    }
}

/// `DocValuesMultiRangeQuery.ByteRange`: inclusive term bounds.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ByteRange {
    pub lower: Vec<u8>,
    pub upper: Vec<u8>,
}

impl ByteRange {
    /// `new ByteRange(lowerValue, upperValue)`.
    pub fn new(lower: impl Into<Vec<u8>>, upper: impl Into<Vec<u8>>) -> Self {
        ByteRange {
            lower: lower.into(),
            upper: upper.into(),
        }
    }

    /// `new ByteRange(singleValue)`.
    pub fn single(value: impl Into<Vec<u8>>) -> Self {
        let value = value.into();
        ByteRange {
            lower: value.clone(),
            upper: value,
        }
    }
}

/// `DocValuesMultiRangeQuery.SortedNumericStabbingBuilder`.
#[derive(Debug, Clone)]
pub struct SortedNumericStabbingBuilder {
    field_name: String,
    clauses: Vec<LongRange>,
}

impl SortedNumericStabbingBuilder {
    pub fn new(field_name: impl Into<String>) -> Self {
        SortedNumericStabbingBuilder {
            field_name: field_name.into(),
            clauses: Vec::new(),
        }
    }

    /// `add(lowerValue, upperValue)`.
    pub fn add(&mut self, lower: i64, upper: i64) -> &mut Self {
        self.clauses.push(LongRange::new(lower, upper));
        self
    }

    /// `build()`: no clauses is `MatchNoDocsQuery`, one is
    /// `SortedNumericDocValuesField.newSlowRangeQuery`, more the multi-range
    /// query.
    pub fn build(&self) -> Box<dyn DocumentQuery> {
        match self.clauses.as_slice() {
            [] => Box::new(MatchNoDocs),
            [only] => Box::new(SortedNumericDocValuesRangeQuery::new(
                self.field_name.clone(),
                only.lower,
                only.upper,
            )),
            clauses => Box::new(SortedNumericDocValuesMultiRangeQuery::new(
                self.field_name.clone(),
                clauses,
            )),
        }
    }
}

/// `DocValuesMultiRangeQuery.SortedSetStabbingBuilder`.
#[derive(Debug, Clone)]
pub struct SortedSetStabbingBuilder {
    field_name: String,
    clauses: Vec<ByteRange>,
}

impl SortedSetStabbingBuilder {
    pub fn new(field_name: impl Into<String>) -> Self {
        SortedSetStabbingBuilder {
            field_name: field_name.into(),
            clauses: Vec::new(),
        }
    }

    /// `add(lowerValue, upperValue)`.
    pub fn add(&mut self, lower: impl Into<Vec<u8>>, upper: impl Into<Vec<u8>>) -> &mut Self {
        self.clauses.push(ByteRange::new(lower, upper));
        self
    }

    /// `add(singleValue)`.
    pub fn add_single(&mut self, value: impl Into<Vec<u8>>) -> &mut Self {
        self.clauses.push(ByteRange::single(value));
        self
    }

    /// `build()`: no clauses is `MatchNoDocsQuery`, one is
    /// `SortedSetDocValuesField.newSlowRangeQuery(.., true, true)`, more the
    /// multi-range query.
    pub fn build(&self) -> Box<dyn DocumentQuery> {
        match self.clauses.as_slice() {
            [] => Box::new(MatchNoDocs),
            [only] => Box::new(SortedSetDocValuesRangeQuery::new(
                self.field_name.clone(),
                Some(only.lower.clone()),
                Some(only.upper.clone()),
                true,
                true,
            )),
            clauses => Box::new(SortedSetDocValuesMultiRangeQuery::new(
                self.field_name.clone(),
                clauses.to_vec(),
            )),
        }
    }
}

/// `SortedNumericDocValuesMultiRangeQuery`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortedNumericDocValuesMultiRangeQuery {
    pub field_name: String,
    /// The resolved spans, `lower -> upper`, ascending and disjoint (Java's
    /// `TreeSet` ordered by `lower` alone).
    pub sorted_clauses: BTreeMap<i64, i64>,
}

/// `Edge`: a range's lower or upper end, or a one-value range's point.
#[derive(Debug, Clone, Copy)]
struct Edge {
    range: usize,
    value: i64,
    point: bool,
    upper: bool,
}

impl SortedNumericDocValuesMultiRangeQuery {
    /// `new SortedNumericDocValuesMultiRangeQuery(fieldName, clauses)`.
    pub fn new(field_name: impl Into<String>, clauses: &[LongRange]) -> Self {
        SortedNumericDocValuesMultiRangeQuery {
            field_name: field_name.into(),
            sorted_clauses: resolve_overlaps(clauses),
        }
    }
}

/// `resolveOverlaps(clauses)`.
fn resolve_overlaps(clauses: &[LongRange]) -> BTreeMap<i64, i64> {
    let mut sorted = BTreeMap::new();
    let mut heap = JavaPriorityQueue::new(clauses.len().saturating_mul(2), |a: &Edge, b: &Edge| {
        a.value < b.value || (a.value == b.value && a.point)
    });
    for (i, r) in clauses.iter().enumerate() {
        let cmp = r.lower.wrapping_sub(r.upper);
        let edge = |value, point, upper| Edge {
            range: i,
            value,
            point,
            upper,
        };
        if cmp == 0 {
            heap.insert_with_overflow(edge(r.lower, true, false));
        } else if cmp < 0 {
            heap.insert_with_overflow(edge(r.lower, false, false));
            heap.insert_with_overflow(edge(r.upper, false, true));
        } // else a reversed range: dropped.
    }
    let total_edges = heap.size();
    let mut depth = 0i32;
    let mut started: Option<Edge> = None;
    for i in 0..total_edges {
        let Some(smallest) = heap.pop() else {
            break;
        };
        if depth == 0 && smallest.point {
            // A point on the edge of a range is that range's.
            if i + 1 < total_edges && heap.top().is_some_and(|t| t.value == smallest.value) {
                continue;
            }
            let r = clauses[smallest.range];
            sorted.entry(r.lower).or_insert(r.upper);
        }
        if !smallest.point {
            if !smallest.upper {
                depth = depth.wrapping_add(1);
                if depth == 1 {
                    started = Some(smallest);
                }
            } else {
                depth = depth.wrapping_sub(1);
                if depth == 0 {
                    // `started` is set whenever `depth` has risen to 1 since
                    // it last fell to 0.
                    if let Some(s) = started.take() {
                        let (lower, upper) = if s.range == smallest.range {
                            let r = clauses[smallest.range];
                            (r.lower, r.upper)
                        } else {
                            (s.value, smallest.value)
                        };
                        sorted.entry(lower).or_insert(upper);
                    }
                }
            }
        }
    }
    sorted
}

impl DocumentQuery for SortedNumericDocValuesMultiRangeQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(info) = field_info(leaf, &self.field_name)? else {
            return Ok(());
        };
        let (Some((&lower_value, _)), Some((_, &upper_value))) = (
            self.sorted_clauses.first_key_value(),
            self.sorted_clauses.last_key_value(),
        ) else {
            return Err(Error::DocumentQuery(format!(
                "field=\"{}\": every range of this SortedNumericDocValuesMultiRangeQuery is \
                 reversed, so it has no first range (Java: NoSuchElementException)",
                self.field_name
            )));
        };
        let skip = skip_index(leaf, info)?;
        if let Some(s) = skip {
            if s.min_value > upper_value || s.max_value < lower_value {
                return Ok(());
            }
        }
        let Some((data, column)) = numeric_column(leaf, info)? else {
            return Ok(());
        };
        if saturated(collector, boost) {
            return Ok(());
        }
        let max_doc = reader(leaf)?.max_doc;
        let ranges = Ranges {
            clauses: self.sorted_clauses.iter().map(|(&l, &u)| (l, u)).collect(),
            lower: lower_value,
            upper: upper_value,
        };
        let (dense, docs_with_field) = match column {
            NumericColumn::Numeric(e) => (e.is_dense(), e.num_values),
            NumericColumn::SortedNumeric(e) => {
                (e.numeric.is_dense(), i64::from(e.num_docs_with_field))
            }
        };
        // `SkipBlockRangeIterator.cost()` is `NO_MORE_DOCS`; the column's is
        // its documents.
        let cost = if skip.is_some() {
            i64::from(i32::MAX)
        } else {
            docs_with_field
        };
        let mut bulk = Bulk::new(collector.score_mode().needs_scores(), max_doc, cost);
        let mut forward = Forward::new(data, column);
        let mut values = Vec::new();
        let Some(s) = skip else {
            // The approximation is the column: its documents with a value.
            for doc in 0..max_doc {
                forward.values(doc, &mut values)?;
                if values.is_empty() {
                    continue;
                }
                match bulk.candidate(leaf, doc, boost, collector) {
                    Step::Stop => return Ok(()),
                    Step::Skip => continue,
                    Step::Check => {}
                }
                if ranges.matches(values.len(), values.iter().map(|&v| Ok(Some(v))))?
                    && bulk.hit(leaf, doc, boost, collector)
                {
                    return Ok(());
                }
            }
            return Ok(());
        };
        let mut approximation = SkipBlocks::new(s, lower_value, upper_value);
        // Every document has a value when the column is dense, so `values`
        // is never ahead of the candidate and `matches()` reads each
        // document's own: no stale state to keep.
        let mut cursor = (!dense).then(|| StaleCursor::new(Forward::new(data, column), max_doc));
        loop {
            let doc = approximation.next_doc();
            if doc >= max_doc {
                return Ok(());
            }
            if cursor.is_none() {
                // Each candidate reads its own values here, so none in a
                // block whose values miss every range can match: the block
                // is skipped, its candidates seen only by the window
                // bookkeeping, which they may move.
                let (last, min, max) = approximation.block();
                if !ranges.may_match(min, max) {
                    if bulk.pass(doc, last, boost, collector) {
                        return Ok(());
                    }
                    approximation.skip_to(last);
                    continue;
                }
            }
            match bulk.candidate(leaf, doc, boost, collector) {
                Step::Stop => return Ok(()),
                Step::Skip => continue,
                Step::Check => {}
            }
            let matched = match cursor.as_mut() {
                Some(c) => c.matches(doc, &ranges)?,
                None => {
                    forward.values(doc, &mut values)?;
                    ranges.matches(values.len(), values.iter().map(|&v| Ok(Some(v))))?
                }
            };
            if matched && bulk.hit(leaf, doc, boost, collector) {
                return Ok(());
            }
        }
    }
}

/// The resolved ranges and `matches()`'s walk over a document's values.
struct Ranges {
    /// `sortedClauses`, ascending and disjoint.
    clauses: Vec<(i64, i64)>,
    lower: i64,
    upper: i64,
}

impl Ranges {
    /// Whether a value in `[min, max]` can fall in a range. The resolved
    /// ranges ascend and do not overlap, so the last one starting at or
    /// before `max` reaches furthest.
    fn may_match(&self, min: i64, max: i64) -> bool {
        let at = self.clauses.partition_point(|&(lower, _)| lower <= max);
        at > 0 && self.clauses[at - 1].1 >= min
    }

    /// `matches()`'s loop over `count` values: a value inside `[lower,
    /// upper]` is looked up as the floor of the remaining tree; one past its
    /// floor's end cuts the tree to the ranges after it (`tailSet(lessOrEq,
    /// false)`) when values remain. Values ascend within a document, so the
    /// cut only matters when a stale read runs into the next document's.
    #[inline]
    fn matches(
        &self,
        count: usize,
        mut values: impl Iterator<Item = Result<Option<i64>>>,
    ) -> Result<bool> {
        let mut start = 0;
        for i in 0..count {
            let Some(v) = values.next().transpose()?.flatten() else {
                return Ok(false);
            };
            if v < self.lower || v > self.upper {
                continue;
            }
            let at = start + self.clauses[start..].partition_point(|&(lower, _)| lower <= v);
            if at > start {
                if self.clauses[at - 1].1 >= v {
                    return Ok(true);
                }
                if i + 1 < count {
                    start = at;
                }
            }
        }
        Ok(false)
    }
}

/// What the bulk scorer does with a candidate.
enum Step {
    Stop,
    Skip,
    Check,
}

/// The numeric query goes through `ConstantScoreScorerSupplier`, whose bulk
/// scorer decides when `matches()` runs -- which, with a stale iterator,
/// decides the hits -- and when the search stops:
///
/// * `DenseConjunctionBulkScorer` (a segment of at least 2048 documents whose
///   approximation costs at least `maxDoc / 32`): windows of 4096 documents
///   from a candidate on, every candidate confirmed before the live
///   documents are masked, the collector's threshold checked only between
///   windows;
/// * `ConstantScoreBulkScorer` (no scores needed): every candidate
///   confirmed, then masked; nothing to stop for;
/// * `DefaultBulkScorer` (otherwise): only live candidates confirmed, the
///   search ending at the hit that saturates the collector.
struct Bulk {
    windows: bool,
    live_first: bool,
    window_end: i32,
}

impl Bulk {
    fn new(needs_scores: bool, max_doc: i32, cost: i64) -> Self {
        let windows = max_doc >= 2048 && cost >= i64::from(max_doc / 32);
        Bulk {
            windows,
            live_first: needs_scores && !windows,
            window_end: -1,
        }
    }

    fn candidate(
        &mut self,
        leaf: &OpenSegment<'_>,
        doc: i32,
        boost: f32,
        collector: &dyn ScoringCollector,
    ) -> Step {
        if self.windows && doc >= self.window_end {
            if saturated(collector, boost) {
                return Step::Stop;
            }
            self.window_end = doc.saturating_add(4096);
        }
        if self.live_first && leaf.live_docs.is_some_and(|b| !b.get_doc(doc)) {
            return Step::Skip;
        }
        Step::Check
    }

    /// Candidates `first..=last` that cannot match: only a window starting
    /// among them checks the collector. Whether to stop.
    fn pass(
        &mut self,
        first: i32,
        last: i32,
        boost: f32,
        collector: &dyn ScoringCollector,
    ) -> bool {
        if !self.windows {
            return false;
        }
        loop {
            let start = first.max(self.window_end);
            if start > last {
                return false;
            }
            if saturated(collector, boost) {
                return true;
            }
            self.window_end = start.saturating_add(4096);
        }
    }

    /// Collects a confirmed candidate if it is live; whether to stop.
    fn hit(
        &self,
        leaf: &OpenSegment<'_>,
        doc: i32,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> bool {
        collect_live(leaf, doc, boost, collector);
        !self.windows && saturated(collector, boost)
    }
}

/// `SkipBlockRangeIterator`'s documents: every document of the skip blocks
/// whose value range meets `[min, max]`.
struct SkipBlocks<'a> {
    skipper: DocValuesSkipper<'a>,
    min: i64,
    max: i64,
    doc: i32,
}

impl<'a> SkipBlocks<'a> {
    fn new(index: &'a DocValuesSkipIndex, min: i64, max: i64) -> Self {
        SkipBlocks {
            skipper: DocValuesSkipper::new(index),
            min,
            max,
            doc: -1,
        }
    }

    /// The current level-0 block: its last document and its value range.
    fn block(&self) -> (i32, i64, i64) {
        (
            self.skipper.max_doc_id(0),
            self.skipper.min_value(0),
            self.skipper.max_value(0),
        )
    }

    /// Moves to `doc`, a document of the current block.
    fn skip_to(&mut self, doc: i32) {
        self.doc = doc;
    }

    /// `nextDoc()`: `advance(doc + 1)`.
    fn next_doc(&mut self) -> i32 {
        let target = self.doc.saturating_add(1);
        if target <= self.skipper.max_doc_id(0) {
            if self.doc > -1 {
                self.doc = target;
                return target;
            }
        } else {
            self.skipper.advance(target);
        }
        self.skipper.advance_range(self.min, self.max);
        self.doc = target.max(self.skipper.min_doc_id(0));
        self.doc
    }
}

/// A numeric column read in ascending document order.
enum Forward<'a> {
    Numeric(NumericReader<'a>),
    Sorted(SortedNumericReader<'a>),
}

impl<'a> Forward<'a> {
    fn new(data: &'a [u8], column: NumericColumn<'a>) -> Self {
        match column {
            NumericColumn::Numeric(e) => Forward::Numeric(NumericReader::new(data, e)),
            NumericColumn::SortedNumeric(e) => Forward::Sorted(SortedNumericReader::new(data, e)),
        }
    }

    /// A `SORTED`/`SORTED_SET` column's ordinals.
    fn ords(data: &'a [u8], column: OrdsColumn<'a>) -> Self {
        match column {
            OrdsColumn::Sorted(e) => Forward::Numeric(NumericReader::new(data, &e.ords)),
            OrdsColumn::Multi(e, _) => Forward::Sorted(SortedNumericReader::new(data, e)),
        }
    }

    /// `DocValues.unwrapSingleton` succeeds.
    fn is_singleton(&self) -> bool {
        match self {
            Forward::Numeric(_) => true,
            Forward::Sorted(r) => r.entry().addresses.is_none(),
        }
    }

    /// The document's values, ascending; documents must come in order.
    fn values(&mut self, doc: i32, out: &mut Vec<i64>) -> Result<()> {
        out.clear();
        match self {
            Forward::Numeric(r) => out.extend(r.value(doc)?),
            Forward::Sorted(r) => r.values(doc, out)?,
        }
        Ok(())
    }
}

/// The `SortedNumericDocValues` that `matches()` reads, with Java's state.
///
/// Under a skip-block approximation `matches()` advances `values` only when
/// it is behind the candidate (`values.docID() < doc`); a candidate before
/// the document `values` already sits on reads *that* document's
/// `docValueCount()` values, and `nextValue()` carries on from wherever the
/// last call stopped -- into the following documents' values. Java returns
/// extra hits for it on a multi-valued field; so does this port. A singleton
/// column's `nextValue()` repeats its one value. Reading past the field's
/// last value ends the match here; Java reads past the column -- an
/// `EOFException`, or a value decoded from `DirectWriter`'s padding that may
/// match.
///
/// The column is read forward, once: `pending` holds the values from the
/// current document's first on, with every document read ahead of it.
struct StaleCursor<'a> {
    column: Forward<'a>,
    singleton: bool,
    max_doc: i32,
    /// `(doc, index of its first value in pending)` from the current
    /// document on; empty before the first `advance` and once exhausted.
    docs: Vec<(i32, usize)>,
    pending: Vec<i64>,
    /// `values.docID()`.
    doc: i32,
    /// The next `nextValue()`'s index into `pending`.
    next: usize,
    /// The last document read from the column.
    read_to: i32,
    buf: Vec<i64>,
}

impl<'a> StaleCursor<'a> {
    fn new(column: Forward<'a>, max_doc: i32) -> Self {
        StaleCursor {
            singleton: column.is_singleton(),
            column,
            max_doc,
            docs: Vec::new(),
            pending: Vec::new(),
            doc: -1,
            next: 0,
            read_to: -1,
            buf: Vec::new(),
        }
    }

    /// Reads the column on to the next document with a value, appending it.
    fn read_next(&mut self) -> Result<bool> {
        while self.read_to < self.max_doc - 1 {
            self.read_to += 1;
            self.column.values(self.read_to, &mut self.buf)?;
            if !self.buf.is_empty() {
                self.docs.push((self.read_to, self.pending.len()));
                self.pending.extend_from_slice(&self.buf);
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// `values.advance(target)`.
    fn advance(&mut self, target: i32) -> Result<()> {
        let keep = self.docs.partition_point(|&(d, _)| d < target);
        if keep < self.docs.len() {
            let base = self.docs[keep].1;
            self.docs.drain(..keep);
            self.pending.drain(..base);
            for d in &mut self.docs {
                d.1 -= base;
            }
        } else {
            self.docs.clear();
            self.pending.clear();
            self.read_to = self.read_to.max(target - 1);
            if !self.read_next()? {
                self.doc = i32::MAX;
                return Ok(());
            }
        }
        self.doc = self.docs[0].0;
        self.next = 0;
        Ok(())
    }

    /// `nextValue()`, `None` past the column's last value.
    fn next_value(&mut self) -> Result<Option<i64>> {
        if self.singleton {
            return Ok(self.pending.first().copied());
        }
        if self.next == self.pending.len() && !self.read_next()? {
            return Ok(None);
        }
        let v = self.pending[self.next];
        self.next += 1;
        Ok(Some(v))
    }

    /// `matches()`.
    fn matches(&mut self, doc: i32, ranges: &Ranges) -> Result<bool> {
        if self.doc < doc {
            self.advance(doc)?;
            if self.doc != doc {
                return Ok(false);
            }
        }
        if self.doc == i32::MAX {
            return Ok(false);
        }
        let count = self.docs.get(1).map_or(self.pending.len(), |&(_, s)| s);
        ranges.matches(count, std::iter::from_fn(|| Some(self.next_value())))
    }
}

/// `SortedSetDocValuesMultiRangeQuery`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortedSetDocValuesMultiRangeQuery {
    pub field_name: String,
    /// Sorted by `lower`, then `upper`.
    pub range_clauses: Vec<ByteRange>,
}

impl SortedSetDocValuesMultiRangeQuery {
    /// `new SortedSetDocValuesMultiRangeQuery(fieldName, clauses)`.
    pub fn new(field_name: impl Into<String>, mut clauses: Vec<ByteRange>) -> Self {
        clauses.sort_by(|a, b| a.lower.cmp(&b.lower).then_with(|| a.upper.cmp(&b.upper)));
        SortedSetDocValuesMultiRangeQuery {
            field_name: field_name.into(),
            range_clauses: clauses,
        }
    }

    /// `createOrdRanges(values, ordRanges)`: the clauses as ascending,
    /// disjoint, non-adjacent inclusive ordinal ranges of this segment's
    /// terms.
    fn create_ord_ranges(&self, dict: &mut TermsDict<'_>) -> Result<Vec<(i64, i64)>> {
        let value_count = dict.size();
        let mut out: Vec<(i64, i64)> = Vec::new();
        for range in &self.range_clauses {
            let status = dict.seek_ceil(&range.lower).map_err(store_err)?;
            if status == SeekStatus::End {
                break;
            }
            let lower_ord = dict.ord();
            let status = if range.lower == range.upper {
                status
            } else {
                dict.seek_ceil(&range.upper).map_err(store_err)?
            };
            let upper_ord = match status {
                SeekStatus::End => value_count.saturating_sub(1),
                SeekStatus::Found => dict.ord(),
                SeekStatus::NotFound => {
                    if dict.ord() == 0 {
                        // The range lies before every term.
                        continue;
                    }
                    dict.ord().saturating_sub(1)
                }
            };
            if lower_ord <= upper_ord {
                match out.last_mut() {
                    Some(previous) if previous.1 >= lower_ord.saturating_sub(1) => {
                        previous.1 = previous.1.max(upper_ord);
                    }
                    _ => out.push((lower_ord, upper_ord)),
                }
            }
        }
        Ok(out)
    }
}

impl DocumentQuery for SortedSetDocValuesMultiRangeQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(info) = field_info(leaf, &self.field_name)? else {
            return Ok(());
        };
        let Some((data, column)) = ords_column(leaf, info)? else {
            return Ok(());
        };
        if saturated(collector, boost) {
            return Ok(());
        }
        let mut dict = TermsDict::open(data, column.terms()).map_err(store_err)?;
        let ord_ranges = self.create_ord_ranges(&mut dict)?;
        let (Some(&(min_ord, _)), Some(&(_, max_ord))) = (ord_ranges.first(), ord_ranges.last())
        else {
            return Ok(());
        };
        if let Some(s) = skip_index(leaf, info)? {
            if s.min_value > max_ord || s.max_value < min_ord {
                return Ok(());
            }
        }
        // `matchingOrds`: a bit per ordinal up to `maxOrd`.
        let mut matching = vec![0u64; usize::try_from(max_ord / 64 + 1).unwrap_or(0)];
        for &(lo, hi) in &ord_ranges {
            set_bits(&mut matching, lo, hi);
        }
        let mut forward = Forward::ords(data, column);
        let mut ords = Vec::new();
        for doc in 0..reader(leaf)?.max_doc {
            forward.values(doc, &mut ords)?;
            // `DocValuesRangeIterator.forOrdinalRange`/`forOrdinalSet`: an
            // ordinal inside one of the ranges.
            let matched = ords.iter().any(|&o| {
                o >= min_ord && o <= max_ord && (matching[(o / 64) as usize] >> (o % 64)) & 1 == 1
            });
            if matched {
                collect_live(leaf, doc, boost, collector);
                if saturated(collector, boost) {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

/// `LongBitSet.set(lo, hi + 1)` over `0 <= lo <= hi < bits.len() * 64`.
fn set_bits(bits: &mut [u64], lo: i64, hi: i64) {
    let (lo, hi) = (lo as usize, hi as usize);
    let (low_word, high_word) = (lo / 64, hi / 64);
    let low_mask = !0u64 << (lo % 64);
    let high_mask = !0u64 >> (63 - hi % 64);
    if low_word == high_word {
        bits[low_word] |= low_mask & high_mask;
        return;
    }
    bits[low_word] |= low_mask;
    for w in &mut bits[low_word + 1..high_word] {
        *w = !0;
    }
    bits[high_word] |= high_mask;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved(ranges: &[(i64, i64)]) -> Vec<(i64, i64)> {
        let clauses: Vec<LongRange> = ranges.iter().map(|&(l, u)| LongRange::new(l, u)).collect();
        resolve_overlaps(&clauses).into_iter().collect()
    }

    #[test]
    fn resolves_overlaps_like_java() {
        assert_eq!(
            resolved(&[(-10, 10), (5, 20), (40, 45), (0, 0), (45, 45), (-50, -50)]),
            vec![(-50, -50), (-10, 20), (40, 45)]
        );
        assert_eq!(
            resolved(&[(3, 3), (3, 3), (7, 7), (10, 2)]),
            vec![(3, 3), (7, 7)]
        );
        assert!(resolved(&[(10, 2), (30, 20)]).is_empty());
        // `Long.MIN_VALUE - 1` wraps positive: Java drops the range.
        assert_eq!(resolved(&[(i64::MIN, 1), (5, 6)]), vec![(5, 6)]);
        // Touching ranges: ties between edges fall as Lucene's heap orders them.
        assert_eq!(resolved(&[(1, 5), (5, 9), (9, 9)]), vec![(1, 5), (5, 9)]);
    }

    #[test]
    fn sets_bit_ranges_like_long_bit_set() {
        for (lo, hi) in [
            (0, 0),
            (3, 3),
            (5, 63),
            (60, 64),
            (1, 191),
            (64, 127),
            (190, 191),
        ] {
            let mut bits = vec![0u64; 3];
            set_bits(&mut bits, lo, hi);
            for o in 0..192i64 {
                let set = (bits[(o / 64) as usize] >> (o % 64)) & 1 == 1;
                assert_eq!(set, (lo..=hi).contains(&o), "{lo}..={hi} at {o}");
            }
        }
    }

    #[test]
    fn builders() {
        let mut b = SortedNumericStabbingBuilder::new("f");
        assert!(format!("{:?}", b.build()).contains("MatchNoDocs"));
        b.add(1, 2);
        assert!(format!("{:?}", b.build()).contains("SortedNumericDocValuesRangeQuery"));
        b.add(5, 6);
        assert!(format!("{:?}", b.build()).contains("MultiRangeQuery"));
        let mut s = SortedSetStabbingBuilder::new("f");
        assert!(format!("{:?}", s.build()).contains("MatchNoDocs"));
        s.add_single("b");
        assert!(format!("{:?}", s.build()).contains("SortedSetDocValuesRangeQuery"));
        s.add("a", "c");
        let q = SortedSetDocValuesMultiRangeQuery::new(
            "f",
            vec![
                ByteRange::new("b", "c"),
                ByteRange::single("a"),
                ByteRange::new("b", "a"),
            ],
        );
        assert_eq!(
            q.range_clauses,
            vec![
                ByteRange::single("a"),
                ByteRange::new("b", "a"),
                ByteRange::new("b", "c")
            ]
        );
        assert!(format!("{:?}", s.build()).contains("MultiRangeQuery"));
    }

    #[test]
    fn no_range_left_is_an_error() {
        let q = SortedNumericDocValuesMultiRangeQuery::new(
            "f",
            &[LongRange::new(3, 1), LongRange::new(9, 2)],
        );
        assert!(q.sorted_clauses.is_empty());
    }
}
