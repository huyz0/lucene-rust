//! Doc-values, index-sort and point ranges as scorer-tree leaves:
//! `SortedNumericDocValuesRangeQuery` (behind `NumericDocValuesRangeQuery`),
//! `IndexSortSortedNumericDocValuesRangeQuery`, `PointRangeQuery` over any
//! number of dimensions and any width, `PointInSetQuery`, and
//! `DocValuesRewriteMethod`. Every one is a `ConstantScoreWeight`: its
//! documents score the boost.

use lucene_codecs::adaptive_zones::ZONE_SHIFT;
use lucene_codecs::doc_values::{self, NumericReader, SortedNumericEntry, SortedNumericReader};
use lucene_codecs::points::{IntersectVisitor, Relation};
use lucene_util::fixed_bit_set::FixedBitSet;

use super::build::{self, LeafContext};
use super::leaf::{ConstantScorer, DocList};
use super::{BoxScorer, Mode, Scorer, NO_MORE_DOCS};
use crate::directory_reader::SegmentReader;
use crate::extended_query::*;
use crate::query::{Clause, FieldExistsQuery};
use crate::Result;

/// A document's numeric doc values, read in ascending document order:
/// `DocValues.getSortedNumeric`, a NUMERIC column being its singleton.
enum Column<'a> {
    Numeric(NumericReader<'a>),
    Sorted(SortedNumericReader<'a>),
}

/// The random-access twin of [`Column`], for a binary search.
enum Values<'a> {
    Numeric(&'a [u8], &'a doc_values::NumericEntry),
    Sorted(&'a [u8], &'a SortedNumericEntry),
}

impl<'a> Values<'a> {
    fn open(reader: &'a SegmentReader, field: &str) -> Option<Self> {
        let fi = reader.field_infos().field_by_name(field)?;
        let (meta, data) = reader.doc_values_for_field(fi.number)?;
        if let Some(e) = meta.numeric_entry(fi.number) {
            return Some(Values::Numeric(data, e));
        }
        meta.sorted_numeric_entry(fi.number)
            .map(|e| Values::Sorted(data, e))
    }

    /// `DocValues.unwrapSingleton` succeeds: at most one value a document.
    fn is_single(&self) -> bool {
        match self {
            Values::Numeric(..) => true,
            Values::Sorted(_, e) => e.addresses.is_none(),
        }
    }

    /// The single value of `doc`, for a singleton column.
    fn value(&self, doc: i32) -> Result<Option<i64>> {
        Ok(match self {
            Values::Numeric(data, e) => doc_values::numeric_value(data, e, doc)?,
            Values::Sorted(data, e) => doc_values::sorted_numeric_values(data, e, doc)?
                .first()
                .copied(),
        })
    }

    fn column(&self) -> Column<'a> {
        match *self {
            Values::Numeric(data, e) => Column::Numeric(NumericReader::new(data, e)),
            Values::Sorted(data, e) => Column::Sorted(SortedNumericReader::new(data, e)),
        }
    }
}

impl Column<'_> {
    fn values(&mut self, doc: i32, out: &mut Vec<i64>) -> Result<()> {
        out.clear();
        match self {
            Column::Numeric(r) => out.extend(r.value(doc)?),
            Column::Sorted(r) => r.values(doc, out)?,
        }
        Ok(())
    }
}

/// A two-phase leaf over every document of the segment: the approximation
/// visits each one, `matches` asks `accept`. `DocValuesRangeIterator`'s
/// shape for a field without a skip index, except that its approximation is
/// the whole segment where Java's is the column's documents -- the same
/// matches, found by asking more documents.
struct TwoPhaseDocs<F> {
    doc: i32,
    max_doc: i32,
    accept: F,
    match_cost: f32,
}

impl<F: FnMut(i32) -> Result<bool>> Scorer for TwoPhaseDocs<F> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        let target = self.doc.saturating_add(1);
        self.advance(target)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.doc = if target >= self.max_doc {
            NO_MORE_DOCS
        } else {
            target
        };
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        i64::from(self.max_doc)
    }
    fn two_phase(&self) -> bool {
        true
    }
    fn matches(&mut self) -> Result<bool> {
        (self.accept)(self.doc)
    }
    fn match_cost(&self) -> f32 {
        self.match_cost
    }
    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }
}

/// `SkipBlockRangeIterator.Match`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockMatch {
    /// Every document of the block has a value, and every value is in range.
    Yes,
    /// Every value of the block is in range; a document matches if it has
    /// one.
    YesIfPresent,
    /// The document's value must be checked.
    Maybe,
}

/// `SkipBlockRangeIterator` as the approximation of
/// `DocValuesRangeIterator`'s bulk range iterators: it visits only the
/// skip-index blocks whose value range meets `[min, max]`, and classifies
/// each so `matches` reads a value only where it must.
struct SkipBlockRange<'a, F> {
    skipper: lucene_codecs::doc_values::DocValuesSkipper<'a>,
    min: i64,
    max: i64,
    doc: i32,
    matched: BlockMatch,
    /// `check(doc, presence_only)`: whether `doc` has a value (presence) or
    /// a value in range.
    check: F,
}

impl<'a, F: FnMut(i32, bool) -> Result<bool>> SkipBlockRange<'a, F> {
    fn new(
        skipper: lucene_codecs::doc_values::DocValuesSkipper<'a>,
        min: i64,
        max: i64,
        check: F,
    ) -> Self {
        Self {
            skipper,
            min,
            max,
            doc: -1,
            matched: BlockMatch::Maybe,
            check,
        }
    }

    /// `classifyBlock()`.
    fn classify(&self) -> BlockMatch {
        let s = &self.skipper;
        if s.min_value(0) >= self.min && s.max_value(0) <= self.max {
            if i64::from(s.max_doc_id(0)) - i64::from(s.min_doc_id(0))
                == i64::from(s.doc_count(0)) - 1
            {
                return BlockMatch::Yes;
            }
            return BlockMatch::YesIfPresent;
        }
        BlockMatch::Maybe
    }
}

impl<F: FnMut(i32, bool) -> Result<bool>> Scorer for SkipBlockRange<'_, F> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        let target = self.doc.saturating_add(1);
        self.advance(target)
    }
    /// `SkipBlockRangeIterator.advance`.
    fn advance(&mut self, target: i32) -> Result<i32> {
        if target <= self.skipper.max_doc_id(0) {
            if self.doc > -1 {
                self.doc = target;
                return Ok(target);
            }
        } else {
            self.skipper.advance(target);
        }
        self.skipper.advance_range(self.min, self.max);
        let next = target.max(self.skipper.min_doc_id(0));
        self.matched = if next == NO_MORE_DOCS {
            BlockMatch::Maybe
        } else {
            self.classify()
        };
        self.doc = next;
        Ok(next)
    }
    fn cost(&self) -> i64 {
        i64::from(NO_MORE_DOCS)
    }
    fn two_phase(&self) -> bool {
        true
    }
    /// `BulkBlockRangeIterator.matches`.
    fn matches(&mut self) -> Result<bool> {
        match self.matched {
            BlockMatch::Yes => Ok(true),
            BlockMatch::YesIfPresent => (self.check)(self.doc, true),
            BlockMatch::Maybe => (self.check)(self.doc, false),
        }
    }
    fn match_cost(&self) -> f32 {
        2.0
    }
    /// `DocValuesBlockRangeIterator.docIDRunEnd()` (Lucene 10.6,
    /// branch_10_6, #16307): only a `YES` block proves a run of matches --
    /// `SkipBlockRangeIterator.docIDRunEnd()`, the block extended to every
    /// coarser enclosing `YES` interval; any other block reports none.
    fn two_phase_run_end(&self) -> i32 {
        if self.matched != BlockMatch::Yes {
            return self.doc;
        }
        let s = &self.skipper;
        let mut max_doc = s.max_doc_id(0);
        let mut level = 1;
        while level < s.num_levels()
            && s.min_value(level) >= self.min
            && s.max_value(level) <= self.max
            && i64::from(s.max_doc_id(level)) - i64::from(s.min_doc_id(level))
                == i64::from(s.doc_count(level)) - 1
        {
            max_doc = s.max_doc_id(level);
            level = level.saturating_add(1);
        }
        max_doc.saturating_add(1)
    }
    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }
}

/// The approximation of a range over a dense single-valued `Adaptive90`
/// field written with per-1024-document statistics
/// ([`lucene_codecs::adaptive_zones`]) and no skip index: it visits only the
/// windows whose `[min, max]` meets the range, and a window inside the range
/// matches whole without reading a value.
struct ZoneRange<'a> {
    stats: NumericReader<'a>,
    values: NumericReader<'a>,
    /// The values of the window starting at `window_start`, once read.
    window: Vec<i64>,
    window_start: i32,
    min: i64,
    max: i64,
    max_doc: i32,
    doc: i32,
    matched: BlockMatch,
}

impl ZoneRange<'_> {
    /// How window `window` meets the range: `None` when it cannot match.
    fn classify(&self, window: i32) -> Result<Option<BlockMatch>> {
        let start = window << ZONE_SHIFT;
        let Some((s, _)) = self.stats.window_stats(start)? else {
            return Ok(Some(BlockMatch::Maybe));
        };
        Ok(if s.max < self.min || s.min > self.max {
            None
        } else if s.min >= self.min && s.max <= self.max {
            Some(BlockMatch::Yes)
        } else {
            Some(BlockMatch::Maybe)
        })
    }
}

impl Scorer for ZoneRange<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        let target = self.doc.saturating_add(1);
        self.advance(target)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        if self.doc == NO_MORE_DOCS {
            return Ok(NO_MORE_DOCS);
        }
        if self.doc > -1
            && target < self.max_doc
            && target < ((self.doc >> ZONE_SHIFT).saturating_add(1) << ZONE_SHIFT)
        {
            self.doc = target.max(self.doc);
            return Ok(self.doc);
        }
        let mut window = target.max(0) >> ZONE_SHIFT;
        while target < self.max_doc && (window << ZONE_SHIFT) < self.max_doc {
            if let Some(m) = self.classify(window)? {
                self.matched = m;
                self.doc = target.max(window << ZONE_SHIFT);
                return Ok(self.doc);
            }
            window += 1;
        }
        self.doc = NO_MORE_DOCS;
        Ok(NO_MORE_DOCS)
    }
    fn cost(&self) -> i64 {
        i64::from(NO_MORE_DOCS)
    }
    fn two_phase(&self) -> bool {
        true
    }
    fn matches(&mut self) -> Result<bool> {
        match self.matched {
            BlockMatch::Yes | BlockMatch::YesIfPresent => Ok(true),
            BlockMatch::Maybe => {
                let start = (self.doc >> ZONE_SHIFT) << ZONE_SHIFT;
                if self.window_start != start {
                    let end = start.saturating_add(1 << ZONE_SHIFT).min(self.max_doc);
                    let window = &mut self.window;
                    window.clear();
                    self.values
                        .for_each_value(start, end, |_, v| window.push(v))?;
                    self.window_start = start;
                }
                let at = usize::try_from(self.doc.saturating_sub(start)).unwrap_or(0);
                Ok(self
                    .window
                    .get(at)
                    .is_some_and(|&v| v >= self.min && v <= self.max))
            }
        }
    }
    fn match_cost(&self) -> f32 {
        2.0
    }
    fn two_phase_run_end(&self) -> i32 {
        if self.matched != BlockMatch::Yes || self.doc == NO_MORE_DOCS {
            return self.doc;
        }
        let mut window = (self.doc >> ZONE_SHIFT).saturating_add(1);
        while (window << ZONE_SHIFT) < self.max_doc {
            if !matches!(self.classify(window), Ok(Some(BlockMatch::Yes))) {
                break;
            }
            window += 1;
        }
        (window << ZONE_SHIFT).min(self.max_doc)
    }
    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }
}

/// The documents `q` matches in a segment, as a bit set, read straight from
/// a dense single-valued field: a window of an `Adaptive90` field whose
/// statistics put every value in range sets its bits in one run, one whose
/// values all miss is skipped, and the rest are decoded a window at a time.
/// `None` where the scorer tree must answer: a field that is not dense and
/// single-valued, or one a skip index serves better than a full decode
/// (skip index, no statistics).
// ARITH: window starts and document ids below `max_doc`, an `i32`.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn dense_range_bits(
    reader: &SegmentReader,
    q: &NumericDocValuesRangeQuery,
) -> Result<Option<FixedBitSet>> {
    if q.lower > q.upper || (q.lower == i64::MIN && q.upper == i64::MAX) {
        return Ok(None);
    }
    let Some(values) = Values::open(reader, &q.field) else {
        return Ok(None);
    };
    let (data, entry) = match &values {
        Values::Numeric(data, e) => (*data, *e),
        Values::Sorted(data, e) if e.addresses.is_none() => (*data, &e.numeric),
        Values::Sorted(..) => return Ok(None),
    };
    if !entry.is_dense() || entry.num_values != i64::from(reader.max_doc) {
        return Ok(None);
    }
    let stats = NumericReader::new(data, entry);
    let zoned = stats.has_window_stats();
    let has_skip = match reader.field_infos().field_by_name(&q.field) {
        Some(fi) => reader.doc_values_skip_index(fi.number)?.is_some(),
        None => false,
    };
    if has_skip && !zoned {
        return Ok(None);
    }
    let max_doc = reader.max_doc;
    let (lo, hi) = (q.lower, q.upper);
    let mut bits = FixedBitSet::new(usize::try_from(max_doc).unwrap_or(0));
    let mut decode = NumericReader::new(data, entry);
    let mut start = 0i32;
    while start < max_doc {
        let end = start.saturating_add(1 << ZONE_SHIFT).min(max_doc);
        let class = if zoned {
            stats.window_stats(start)?.map(|(s, _)| s)
        } else {
            None
        };
        match class {
            Some(s) if s.max < lo || s.min > hi => {}
            Some(s) if s.min >= lo && s.max <= hi => {
                bits.set_range(start as usize, end as usize);
            }
            _ => decode.for_each_value(start, end, |doc, v| {
                // FBS: `for_each_value` yields documents in `[start, end)`,
                // below `max_doc`, the set's length.
                if v >= lo && v <= hi {
                    bits.set(doc as usize);
                }
            })?,
        }
        start = end;
    }
    Ok(Some(bits))
}

fn reader<'a>(ctx: &LeafContext<'a>, what: &str) -> Result<&'a SegmentReader> {
    ctx.reader
        .ok_or_else(|| crate::Error::MissingSegmentReader(what.to_string()))
}

fn constant<'a>(inner: BoxScorer<'a>, boost: f32, mode: Mode) -> Option<BoxScorer<'a>> {
    Some(Box::new(ConstantScorer::new(
        inner,
        boost,
        mode == Mode::TopScores,
    )))
}

/// `SortedNumericDocValuesRangeQuery`: a document matches when one of its
/// values is in `[lower, upper]` -- its values ascend, so the first not
/// below `lower` decides.
pub(crate) fn numeric_range<'a>(
    ctx: &LeafContext<'a>,
    q: &NumericDocValuesRangeQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    // `rewrite`: the whole range is `FieldExistsQuery`, an empty one nothing.
    if q.lower == i64::MIN && q.upper == i64::MAX {
        let exists = Clause::Exists(FieldExistsQuery::new(q.field.clone()));
        return build::build(ctx, &exists, boost, mode, false);
    }
    if q.lower > q.upper {
        return Ok(None);
    }
    let reader = reader(ctx, &q.field)?;
    let Some(values) = Values::open(reader, &q.field) else {
        return Ok(None);
    };
    let (lower, upper) = (q.lower, q.upper);
    let skip_index = match reader.field_infos().field_by_name(&q.field) {
        Some(fi) => reader.doc_values_skip_index(fi.number)?,
        None => None,
    };
    if let Some(index) = skip_index {
        // `docCountIgnoringDeletes`: the skipper's global bounds answer a
        // range that misses every value, or holds every document.
        let skipper = lucene_codecs::doc_values::DocValuesSkipper::new(index);
        if skipper.global_min_value() > upper || skipper.global_max_value() < lower {
            return Ok(None);
        }
        if skipper.global_doc_count() == reader.max_doc
            && skipper.global_min_value() >= lower
            && skipper.global_max_value() <= upper
        {
            let all: BoxScorer<'a> = Box::new(super::leaf::AllDocs::new(reader.max_doc));
            return Ok(constant(all, boost, mode));
        }
        // `DocValuesRangeIterator.forRange` with a skipper.
        let mut column = values.column();
        let mut buf = Vec::new();
        let check = move |doc: i32, presence: bool| -> Result<bool> {
            column.values(doc, &mut buf)?;
            if presence {
                return Ok(!buf.is_empty());
            }
            Ok(buf
                .iter()
                .find(|&&v| v >= lower)
                .is_some_and(|&v| v <= upper))
        };
        return Ok(constant(
            Box::new(SkipBlockRange::new(skipper, lower, upper, check)),
            boost,
            mode,
        ));
    }
    let zoned = match &values {
        Values::Numeric(data, e) => Some((*data, *e)),
        Values::Sorted(data, e) if e.addresses.is_none() => Some((*data, &e.numeric)),
        Values::Sorted(..) => None,
    }
    .filter(|&(data, e)| NumericReader::new(data, e).has_window_stats());
    let mut column = values.column();
    let mut buf = Vec::new();
    let accept = move |doc: i32| -> Result<bool> {
        column.values(doc, &mut buf)?;
        Ok(buf
            .iter()
            .find(|&&v| v >= lower)
            .is_some_and(|&v| v <= upper))
    };
    if let Some((data, entry)) = zoned {
        return Ok(constant(
            Box::new(ZoneRange {
                stats: NumericReader::new(data, entry),
                values: NumericReader::new(data, entry),
                window: Vec::with_capacity(1 << ZONE_SHIFT),
                window_start: -1,
                min: lower,
                max: upper,
                max_doc: reader.max_doc,
                doc: -1,
                matched: BlockMatch::Maybe,
            }),
            boost,
            mode,
        ));
    }
    Ok(constant(
        Box::new(TwoPhaseDocs {
            doc: -1,
            max_doc: reader.max_doc,
            accept,
            match_cost: 2.0,
        }),
        boost,
        mode,
    ))
}

/// `IndexSortSortedNumericDocValuesRangeQuery`: on a segment sorted first by
/// the field (an `INT` or `LONG` key over a single-valued column), the
/// matches are one contiguous run found by two binary searches with the
/// sort's comparator; anywhere else, `fallback` -- including a segment with
/// document blocks (`getDocIdSetIteratorOrNull`'s `hasBlocks` check, Lucene
/// 10.6, branch_10_6, #16893 fixing #16886), where only the blocks follow
/// the sort.
pub(crate) fn index_sort_range<'a>(
    ctx: &LeafContext<'a>,
    q: &IndexSortSortedNumericDocValuesRangeQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    if q.lower == i64::MIN && q.upper == i64::MAX {
        let exists = Clause::Exists(FieldExistsQuery::new(q.field.clone()));
        return build::build(ctx, &exists, boost, mode, false);
    }
    if q.lower > q.upper {
        return Ok(None);
    }
    let fallback = || build::build(ctx, &q.fallback, boost, mode, top_level);
    let Some(reader) = ctx.reader else {
        return fallback();
    };
    if reader.has_blocks() {
        // With blocks only the parent documents are sorted, so the doc values
        // do not follow the doc-ID order and the binary search does not apply.
        return fallback();
    }
    let Some(sort) = reader.index_sort().and_then(|s| s.first()) else {
        return fallback();
    };
    if sort.field != q.field {
        return fallback();
    }
    use lucene_index::segment_info::{IndexSortKind, NumericSortKey};
    let (int, missing) = match &sort.kind {
        IndexSortKind::Numeric(NumericSortKey::Long(m))
        | IndexSortKind::SortedNumeric {
            key: NumericSortKey::Long(m),
            ..
        } => (false, m.unwrap_or(0)),
        IndexSortKind::Numeric(NumericSortKey::Int(m))
        | IndexSortKind::SortedNumeric {
            key: NumericSortKey::Int(m),
            ..
        } => (true, i64::from(m.unwrap_or(0))),
        _ => return fallback(),
    };
    let Some(values) = Values::open(reader, &q.field) else {
        return fallback();
    };
    if !values.is_single() {
        return fallback();
    }
    let reverse = sort.reverse;
    let direction: i32 = if reverse { -1 } else { 1 };
    // `loadComparator(...).compare(doc)`: `direction * compareTop(doc)`,
    // `compareTop` being `Long.compare(topValue, value)` (an `int` compare of
    // `(int) topValue` for an `INT` key), the missing value standing in for a
    // document without one.
    let compare = |top: i64, doc: i32| -> Result<i32> {
        let v = values.value(doc)?.unwrap_or(missing);
        let ord = if int {
            (top as i32).cmp(&(v as i32))
        } else {
            top.cmp(&v)
        };
        Ok(direction * ord as i32)
    };
    let (lower, upper) = if reverse {
        (q.upper, q.lower)
    } else {
        (q.lower, q.upper)
    };
    let max_doc = reader.max_doc;
    let (mut low, mut high) = (0i32, max_doc - 1);
    while low <= high {
        let mid = ((low + high) as u32 >> 1) as i32;
        if compare(lower, mid)? <= 0 {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    let first = high + 1;
    let (mut low, mut high) = (first, max_doc - 1);
    while low <= high {
        let mid = ((low + high) as u32 >> 1) as i32;
        if compare(upper, mid)? < 0 {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    let last = high + 1;
    if first == last {
        return Ok(None);
    }
    // `denseRange` when a document without a value cannot sort inside the
    // run; `sparseRange` (the documents with a value) otherwise.
    let dense = missing < q.lower || missing > q.upper;
    if dense {
        return Ok(constant(
            Box::new(DocRange {
                doc: -1,
                min: first,
                max: last,
            }),
            boost,
            mode,
        ));
    }
    let mut docs = Vec::new();
    for doc in first..last {
        if values.value(doc)?.is_some() {
            docs.push(doc);
        }
    }
    if docs.is_empty() {
        return Ok(None);
    }
    Ok(constant(
        Box::new(DocList::new(docs, Vec::new())),
        boost,
        mode,
    ))
}

/// Every document in `min..max` (`min < max`) as a constant-scored leaf.
pub(crate) fn doc_range<'a>(min: i32, max: i32, boost: f32, mode: Mode) -> Option<BoxScorer<'a>> {
    constant(Box::new(DocRange { doc: -1, min, max }), boost, mode)
}

/// The documents of `bits` in `min..max` (`min < max`) as a constant-scored
/// leaf: a segment of document blocks' parents in a run of whole blocks.
pub(crate) fn bits_range<'a>(
    bits: &'a FixedBitSet,
    min: i32,
    max: i32,
    boost: f32,
    mode: Mode,
) -> Option<BoxScorer<'a>> {
    let cost = bits.cardinality_range(min as usize, max as usize) as i64;
    if cost == 0 {
        return None;
    }
    constant(
        Box::new(BitsRange {
            bits,
            doc: -1,
            min,
            max,
            cost,
        }),
        boost,
        mode,
    )
}

/// [`bits_range`]'s iterator.
struct BitsRange<'a> {
    bits: &'a FixedBitSet,
    doc: i32,
    min: i32,
    max: i32,
    cost: i64,
}

impl Scorer for BitsRange<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.doc.saturating_add(1))
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let from = target.max(self.min);
        self.doc = if from >= self.max {
            NO_MORE_DOCS
        } else {
            self.bits
                .next_set_bit_in_range(from as usize, self.max as usize)
                .map_or(NO_MORE_DOCS, |d| d as i32)
        };
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.cost
    }
    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }
    fn contains(&self, doc: i32) -> Option<bool> {
        Some((self.min..self.max).contains(&doc) && self.bits.get_doc(doc))
    }
}

/// `DocIdSetIterator.range(min, max)`: every document in `min..max`
/// (`min < max`).
struct DocRange {
    doc: i32,
    min: i32,
    max: i32,
}

impl Scorer for DocRange {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.doc.saturating_add(1))
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.doc = if target >= self.max {
            NO_MORE_DOCS
        } else {
            target.max(self.min)
        };
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        i64::from(self.max) - i64::from(self.min)
    }
    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }
    fn contains(&self, doc: i32) -> Option<bool> {
        Some((self.min..self.max).contains(&doc))
    }
    /// The range's own end: every document to `max` matches.
    fn doc_id_run_end(&self) -> i32 {
        self.max
    }
}

/// The documents `docs` (unsorted, possibly repeated) as a constant-scored
/// leaf: a bit set when dense, a list otherwise (`DocIdSetBuilder`).
///
/// No matching document still gives a scorer: Lucene's supplier exists once
/// the segment's points overlap the query's bounds ([`disjoint`]), and a
/// boolean counts its clauses by supplier (`BooleanWeight.scorerSupplier`
/// turns `minimumNumberShouldMatch` optional clauses into required ones).
fn doc_set<'a>(docs: Vec<i32>, max_doc: i32, boost: f32, mode: Mode) -> Option<BoxScorer<'a>> {
    constant(RangeDocs::Docs(docs).into_scorer(max_doc), boost, mode)
}

/// A point range's matches over one segment, before deletions.
pub(crate) enum RangeDocs {
    /// Unsorted, possibly repeated doc ids (a sparse result).
    Docs(Vec<i32>),
    /// The matching documents as a set of `maxDoc` bits, and how many.
    Bits(FixedBitSet, i64),
}

impl RangeDocs {
    /// Whether nothing matched.
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            RangeDocs::Docs(d) => d.is_empty(),
            RangeDocs::Bits(_, n) => *n == 0,
        }
    }

    /// The matches as a scorer's iterator: the set as it is, or the doc ids
    /// as `DocIdSetBuilder` keeps them (a bit set past `maxDoc / 128`, a
    /// sorted list below).
    pub(crate) fn into_scorer<'a>(self, max_doc: i32) -> BoxScorer<'a> {
        let bits_scorer = |bits, cardinality| -> BoxScorer<'a> {
            Box::new(super::cache::CachedScorer::new(std::sync::Arc::new(
                super::cache::CachedSet::Bits { bits, cardinality },
            )))
        };
        match self {
            RangeDocs::Bits(bits, n) => bits_scorer(bits, n),
            RangeDocs::Docs(mut docs) => {
                let len = usize::try_from(max_doc).unwrap_or(0);
                if docs.len() > len / 128 {
                    let mut bits =
                        crate::bit_set_pool::take(len).unwrap_or_else(|| FixedBitSet::new(len));
                    let mut cardinality = 0i64;
                    for &d in &docs {
                        if let Ok(i) = usize::try_from(d) {
                            // FBS: a BKD walk only returns doc ids below
                            // `maxDoc`; `i < len` holds by construction.
                            if i < len && !bits.get_and_set(i) {
                                cardinality += 1;
                            }
                        }
                    }
                    bits_scorer(bits, cardinality)
                } else {
                    lucene_util::doc_id_sort::sort_dedup_doc_ids(&mut docs);
                    Box::new(DocList::new(docs, Vec::new()))
                }
            }
        }
    }
}

/// `PointRangeQuery`'s scorer supplier's walk: the documents of segment
/// field `number` (`info`) with a point inside `[lower, upper]`.
///
/// - A dense single-valued field (`docCount == maxDoc == size`) whose
///   estimated match count passes half the segment starts from every
///   document and clears the ones whose point is outside the box
///   (`getInverseIntersectVisitor`), walking the cells outside the box
///   rather than the larger set inside it.
/// - An estimate past `maxDoc / 128` (`DocIdSetBuilder`'s bit-set threshold)
///   sets the matches straight into a bit set, counting them as it goes.
/// - Anything smaller collects doc ids.
pub(crate) fn range_docs(
    points: &crate::points_query::PointsInput<'_>,
    number: i32,
    info: &lucene_codecs::points::PointsField,
    max_doc: i32,
    lower: &[u8],
    upper: &[u8],
) -> Result<RangeDocs> {
    #[cfg(test)]
    RANGE_WALKS.with(|n| n.set(n.get() + 1));
    let len = usize::try_from(max_doc).unwrap_or(0);
    if len == 0 {
        return Ok(RangeDocs::Docs(
            points.reader.range_query(number, lower, upper)?,
        ));
    }
    let estimate = points
        .reader
        .estimate_range_point_count(number, lower, upper)?;
    let single_valued = info.doc_count == max_doc && i64::from(info.doc_count) == info.point_count;
    let inverse = single_valued && estimate > i64::from(max_doc / 2);
    if !inverse && estimate <= i64::from(max_doc >> 7) {
        return Ok(RangeDocs::Docs(
            points.reader.range_query(number, lower, upper)?,
        ));
    }
    let matcher = points.reader.range_matcher(number, lower, upper)?;
    let words = crate::bit_set_pool::take(len)
        .map(FixedBitSet::into_words)
        .unwrap_or_else(|| vec![0u64; lucene_util::fixed_bit_set::bits2words(len)]);
    if inverse {
        let mut v = BitsRangeVisitor::<true> {
            matcher,
            words,
            len,
        };
        v.words.fill(u64::MAX);
        if let Some(last) = v.words.last_mut() {
            *last &= tail_mask(len);
        }
        points.reader.intersect(number, &mut v)?;
        Ok(bits_with_cardinality(v.words, len))
    } else {
        let mut v = BitsRangeVisitor::<false> {
            matcher,
            words,
            len,
        };
        points.reader.intersect(number, &mut v)?;
        Ok(bits_with_cardinality(v.words, len))
    }
}

/// `PointRangeQuery`'s `Weight.count` on a segment without deletions, the
/// caller's check: no match when the segment has no points for the field or
/// its points miss the box, the field's `docCount` when they all fall
/// inside it, and for a one-dimension single-valued field (`docCount ==
/// size`) the points inside counted off the tree (`pointCount`: inside
/// cells by their size, only the crossing leaves read). `None` -- iterate --
/// for anything else, including a field of another shape than the box's,
/// which the scorer reports.
pub(crate) fn point_range_count(
    points: &crate::points_query::PointsInput<'_>,
    field: &str,
    lower: &[u8],
    upper: &[u8],
    num_dims: usize,
    bytes_per_dim: usize,
) -> Result<Option<u64>> {
    let Some(number) = points.field_number(field) else {
        return Ok(Some(0));
    };
    let Some(info) = points.reader.field(number) else {
        return Ok(Some(0));
    };
    if usize::try_from(info.num_dims).ok() != Some(num_dims)
        || usize::try_from(info.bytes_per_dim).ok() != Some(bytes_per_dim)
        || lower.len() != num_dims * bytes_per_dim
        || upper.len() != num_dims * bytes_per_dim
    {
        return Ok(None);
    }
    if disjoint(info, lower, upper, num_dims, bytes_per_dim) {
        return Ok(Some(0));
    }
    let matcher = points.reader.range_matcher(number, lower, upper)?;
    if matcher.compare(&info.min_packed_value, &info.max_packed_value) == Relation::CellInsideQuery
    {
        return Ok(Some(u64::try_from(info.doc_count).unwrap_or(0)));
    }
    if num_dims != 1 || i64::from(info.doc_count) != info.point_count {
        return Ok(None);
    }
    struct Count {
        matcher: lucene_codecs::points::RangeMatcher,
        n: u64,
    }
    impl IntersectVisitor for Count {
        fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
            self.matcher.compare(min_packed, max_packed)
        }
        fn visit(&mut self, _doc_id: i32) {
            self.n += 1;
        }
        fn visit_many(&mut self, doc_ids: &[i32]) {
            self.n += doc_ids.len() as u64;
        }
        fn visit_with_value(&mut self, _doc_id: i32, packed_value: &[u8]) {
            if self.matcher.matches(packed_value) {
                self.n += 1;
            }
        }
    }
    let mut v = Count { matcher, n: 0 };
    let inside = points.reader.count_points(number, &mut v)?;
    Ok(Some(v.n.saturating_add(u64::try_from(inside).unwrap_or(0))))
}

/// The bits of the last word of a `len`-bit set that are inside it.
fn tail_mask(len: usize) -> u64 {
    match len & 63 {
        0 => u64::MAX,
        r => (1u64 << r) - 1,
    }
}

#[cfg(test)]
thread_local! {
    /// How many times [`range_docs`] walked a field's tree on this thread,
    /// for tests.
    pub(crate) static RANGE_WALKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The set a range walk filled and how many documents it holds, counted
/// once over its words rather than bit by bit as the walk sets them.
fn bits_with_cardinality(words: Vec<u64>, len: usize) -> RangeDocs {
    let bits = FixedBitSet::from_words(words, len);
    let n = i64::try_from(bits.cardinality()).unwrap_or(i64::MAX);
    RangeDocs::Bits(bits, n)
}

/// A point range's visitor over a set's words. `INVERSE == false` sets every
/// matching document; `INVERSE == true` starts from a full set, walks the
/// cells outside the box and clears every document with a point outside
/// it. Doc ids at or past `len` are ignored.
struct BitsRangeVisitor<const INVERSE: bool> {
    matcher: lucene_codecs::points::RangeMatcher,
    words: Vec<u64>,
    len: usize,
}

impl<const INVERSE: bool> BitsRangeVisitor<INVERSE> {
    /// Applies `mask` to word `w`: sets its bits, or clears them.
    #[inline(always)]
    fn apply(&mut self, w: usize, mask: u64) {
        if let Some(word) = self.words.get_mut(w) {
            if INVERSE {
                *word &= !mask;
            } else {
                *word |= mask;
            }
        }
    }

    #[inline(always)]
    fn doc(&mut self, doc_id: i32) {
        let i = doc_id as u32 as usize;
        if i < self.len {
            self.apply(i >> 6, 1u64 << (i & 63));
        }
    }
}

impl<const INVERSE: bool> IntersectVisitor for BitsRangeVisitor<INVERSE> {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        let r = self.matcher.compare(min_packed, max_packed);
        if INVERSE {
            match r {
                Relation::CellInsideQuery => Relation::CellOutsideQuery,
                Relation::CellOutsideQuery => Relation::CellInsideQuery,
                Relation::CellCrossesQuery => Relation::CellCrossesQuery,
            }
        } else {
            r
        }
    }

    fn visit(&mut self, doc_id: i32) {
        self.doc(doc_id);
    }

    fn visit_many(&mut self, doc_ids: &[i32]) {
        // One read-modify-write per id, nothing carried from one to the
        // next: the count is taken once the walk is done.
        let words = &mut self.words[..];
        let len = self.len;
        for &d in doc_ids {
            let i = d as u32 as usize;
            if i < len {
                if let Some(word) = words.get_mut(i >> 6) {
                    if INVERSE {
                        *word &= !(1u64 << (i & 63));
                    } else {
                        *word |= 1u64 << (i & 63);
                    }
                }
            }
        }
    }

    fn visit_range(&mut self, start: i32, count: usize) -> bool {
        let from = usize::try_from(start).unwrap_or(usize::MAX).min(self.len);
        let to = from.saturating_add(count).min(self.len);
        if from >= to {
            return true;
        }
        let (fw, tw) = (from >> 6, (to - 1) >> 6);
        let head = u64::MAX << (from & 63);
        let tail = u64::MAX >> (63 - ((to - 1) & 63));
        if fw == tw {
            self.apply(fw, head & tail);
            return true;
        }
        self.apply(fw, head);
        for w in fw + 1..tw {
            self.apply(w, u64::MAX);
        }
        self.apply(tw, tail);
        true
    }

    fn visit_bitset(&mut self, words: &[u64], base: i32, _count: usize) -> bool {
        let base = base as u32 as usize;
        if base & 63 != 0 {
            for (i, &word) in words.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let d = base + i * 64 + w.trailing_zeros() as usize;
                    if d < self.len {
                        self.apply(d >> 6, 1u64 << (d & 63));
                    }
                    w &= w - 1;
                }
            }
            return true;
        }
        let last = (self.len - 1) >> 6;
        let tail = tail_mask(self.len);
        for (i, &word) in words.iter().enumerate() {
            let w = (base >> 6) + i;
            if w > last {
                break;
            }
            self.apply(w, if w == last { word & tail } else { word });
        }
        true
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        if self.matcher.matches(packed_value) != INVERSE {
            self.doc(doc_id);
        }
    }
}

/// The field's points and the segment's `maxDoc`, after
/// `PointRangeQuery`/`PointInSetQuery`'s configuration checks.
fn points_field<'a>(
    ctx: &LeafContext<'a>,
    field: &str,
    num_dims: usize,
    bytes_per_dim: usize,
    range: bool,
) -> Result<Option<(&'a crate::points_query::PointsInput<'a>, i32, i32)>> {
    let Some(points) = ctx.points else {
        return Err(crate::Error::MissingPointsInput(field.to_string()));
    };
    let max_doc = match ctx.max_doc.or(ctx.reader.map(|r| r.max_doc)) {
        Some(m) => m,
        None => return Err(crate::Error::MissingPointsInput(field.to_string())),
    };
    let Some(number) = points.field_number(field) else {
        return Ok(None);
    };
    let Some(info) = points.reader.field(number) else {
        return Ok(None);
    };
    // `PointRangeQuery.checkValidPointValues` checks the dimension count, then
    // the bytes per dimension, each with its own message; the set queries
    // check both at once
    if range {
        if usize::try_from(info.num_dims).ok() != Some(num_dims) {
            return Err(crate::Error::IllegalArgument(format!(
                "field=\"{field}\" was indexed with numIndexDimensions={} but this query has numDims={num_dims}",
                info.num_dims
            )));
        }
        if usize::try_from(info.bytes_per_dim).ok() != Some(bytes_per_dim) {
            return Err(crate::Error::IllegalArgument(format!(
                "field=\"{field}\" was indexed with bytesPerDim={} but this query has bytesPerDim={bytes_per_dim}",
                info.bytes_per_dim
            )));
        }
    } else if usize::try_from(info.num_dims).ok() != Some(num_dims)
        || usize::try_from(info.bytes_per_dim).ok() != Some(bytes_per_dim)
    {
        return Err(crate::Error::InvalidQuery(format!(
            "field=\"{field}\" was indexed with numDims={} bytesPerDim={} but this query has numDims={num_dims} bytesPerDim={bytes_per_dim}",
            info.num_dims, info.bytes_per_dim
        )));
    }
    Ok(Some((points, number, max_doc)))
}

/// `PointRangeQuery`'s and `PointInSetQuery`'s null supplier: a segment
/// without documents for the field, or whose `[min, max]` packed values miss
/// `[lower, upper]` in some dimension.
fn disjoint(
    info: &lucene_codecs::points::PointsField,
    lower: &[u8],
    upper: &[u8],
    num_dims: usize,
    bpd: usize,
) -> bool {
    info.doc_count == 0
        || (0..num_dims).any(|d| {
            let r = d * bpd..(d + 1) * bpd;
            match (
                lower.get(r.clone()),
                upper.get(r.clone()),
                info.min_packed_value.get(r.clone()),
                info.max_packed_value.get(r),
            ) {
                (Some(lo), Some(hi), Some(min), Some(max)) => lo > max || hi < min,
                _ => false,
            }
        })
}

/// [`disjoint`] for a `PointInSetQuery`, whose bounds are its first and last
/// packed points (none for an empty set, which only a segment without
/// documents for the field leaves without a supplier).
pub(crate) fn point_set_disjoint(
    points: &crate::points_query::PointsInput<'_>,
    number: i32,
    q: &PointInSetQuery,
) -> bool {
    let Some(info) = points.reader.field(number) else {
        return true;
    };
    match (q.points.first(), q.points.last()) {
        (Some(lo), Some(hi)) => disjoint(info, lo, hi, q.num_dims, q.bytes_per_dim),
        _ => info.doc_count == 0,
    }
}

/// `PointRangeQuery`: every document with a point inside the box.
pub(crate) fn point_range<'a>(
    ctx: &LeafContext<'a>,
    q: &PointRangeQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let Some((points, number, max_doc)) =
        points_field(ctx, &q.field, q.num_dims, q.bytes_per_dim, true)?
    else {
        return Ok(None);
    };
    let Some(info) = points.reader.field(number) else {
        return Ok(None);
    };
    if disjoint(info, &q.lower, &q.upper, q.num_dims, q.bytes_per_dim) {
        return Ok(None);
    }
    let docs = range_docs(points, number, info, max_doc, &q.lower, &q.upper)?;
    Ok(constant(docs.into_scorer(max_doc), boost, mode))
}

/// `PointRangeQuery`'s `ScorerSupplier.cost()` without building its scorer:
/// `maxDoc` when every document has a value inside the box (`allDocsMatch`),
/// else `PointValues.estimateDocCount` of its visitor. `Absent` when the
/// segment has no points for the field.
pub(crate) fn point_range_cost(
    ctx: &LeafContext<'_>,
    q: &PointRangeQuery,
) -> Result<super::extended::Supplier> {
    use super::extended::Supplier;
    let Some((points, number, max_doc)) =
        points_field(ctx, &q.field, q.num_dims, q.bytes_per_dim, true)?
    else {
        return Ok(Supplier::Absent);
    };
    let Some(info) = points.reader.field(number) else {
        return Ok(Supplier::Absent);
    };
    if disjoint(info, &q.lower, &q.upper, q.num_dims, q.bytes_per_dim) {
        return Ok(Supplier::Absent);
    }
    if info.doc_count == max_doc {
        let bpd = q.bytes_per_dim;
        let all = (0..q.num_dims).all(|d| {
            let r = d * bpd..(d + 1) * bpd;
            match (
                q.lower.get(r.clone()),
                q.upper.get(r.clone()),
                info.min_packed_value.get(r.clone()),
                info.max_packed_value.get(r),
            ) {
                (Some(lo), Some(hi), Some(min), Some(max)) => lo <= min && hi >= max,
                _ => false,
            }
        });
        if all {
            return Ok(Supplier::Present(Some(i64::from(max_doc))));
        }
    }
    let points_estimate = points
        .reader
        .estimate_range_point_count(number, &q.lower, &q.upper)?;
    Ok(Supplier::Present(Some(
        crate::points_query::estimate_doc_count(points_estimate, info.point_count, info.doc_count),
    )))
}

/// `PointInSetQuery`: every document with a point equal to one of the set.
pub(crate) fn point_in_set<'a>(
    ctx: &LeafContext<'a>,
    q: &PointInSetQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let Some((points, number, max_doc)) =
        points_field(ctx, &q.field, q.num_dims, q.bytes_per_dim, false)?
    else {
        return Ok(None);
    };
    if point_set_disjoint(points, number, q) {
        return Ok(None);
    }
    let docs = if q.num_dims == 1 {
        // `PointInSetQuery.MergePointVisitor`: one walk of the tree, the
        // sorted query points merged with the cells' ascending values.
        let mut visitor = MergePointVisitor {
            points: &q.points,
            next: 0,
            docs: Vec::new(),
        };
        points.reader.intersect(number, &mut visitor)?;
        visitor.docs
    } else {
        // `SinglePointVisitor`: a walk per point.
        let mut docs = Vec::new();
        for p in &q.points {
            docs.extend(points.reader.range_query(number, p, p)?);
        }
        docs
    };
    Ok(doc_set(docs, max_doc, boost, mode))
}

/// `PointInSetQuery.MergePointVisitor` (one dimension): the query's sorted
/// points, advanced as the tree hands over cells and values in ascending
/// order.
struct MergePointVisitor<'q> {
    points: &'q [Vec<u8>],
    /// `nextQueryPoint`: `points.len()` once exhausted.
    next: usize,
    docs: Vec<i32>,
}

impl MergePointVisitor<'_> {
    /// `matches(packedValue)`.
    fn matches(&mut self, packed: &[u8]) -> bool {
        while let Some(p) = self.points.get(self.next) {
            match p.as_slice().cmp(packed) {
                std::cmp::Ordering::Equal => return true,
                std::cmp::Ordering::Less => self.next = self.next.saturating_add(1),
                std::cmp::Ordering::Greater => break,
            }
        }
        false
    }
}

impl IntersectVisitor for MergePointVisitor<'_> {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        while let Some(p) = self.points.get(self.next) {
            let p = p.as_slice();
            let cmp_min = p.cmp(min_packed);
            if cmp_min == std::cmp::Ordering::Less {
                // The query point is before the cell: try the next one.
                self.next = self.next.saturating_add(1);
                continue;
            }
            let cmp_max = p.cmp(max_packed);
            if cmp_max == std::cmp::Ordering::Greater {
                return Relation::CellOutsideQuery;
            }
            if cmp_min == std::cmp::Ordering::Equal && cmp_max == std::cmp::Ordering::Equal {
                // A cell of one value, the query point's.
                return Relation::CellInsideQuery;
            }
            return Relation::CellCrossesQuery;
        }
        Relation::CellOutsideQuery
    }

    fn visit(&mut self, doc_id: i32) {
        self.docs.push(doc_id);
    }

    fn visit_many(&mut self, doc_ids: &[i32]) {
        self.docs.extend_from_slice(doc_ids);
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        if self.matches(packed_value) {
            self.docs.push(doc_id);
        }
    }

    fn grow(&mut self, count: usize) {
        self.docs.reserve(count);
    }
}

/// A [`DocumentClause`]: the query's `ConstantScoreWeight` over this
/// segment -- its matches, before deletions, collected into a set (as the geo
/// queries' own scorers collect theirs into a `FixedBitSet` or
/// `DocIdSetBuilder`) and iterated at the boost.
pub(crate) fn document<'a>(
    ctx: &LeafContext<'a>,
    q: &DocumentClause,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let Some(reader) = ctx.reader else {
        return Err(crate::Error::MissingSegmentReader(q.key.clone()));
    };
    let leaf = crate::multi_segment::OpenSegment {
        fields: ctx.fields,
        doc_in: ctx.doc_in,
        pos_in: ctx.pos_in,
        pay_in: ctx.pay_in,
        // The tree applies deletions; the set is the segment core's. A
        // query that searches for its best live documents sees them.
        live_docs: if q.query.needs_live_docs() {
            ctx.live_docs
        } else {
            None
        },
        doc_base: 0,
        max_doc: Some(reader.max_doc),
        cache: None,
        points: ctx.points,
        reader: Some(reader),
        index_sort_prefix: false,
        termless: None,
        similarity: None,
    };
    if let Some(fallback) = &q.fallback {
        if !q.query.scores_leaf(&leaf)? {
            return build::build(ctx, fallback, boost, mode, false);
        }
    }
    let filter = match &q.filter {
        None => None,
        Some(filter) => match build::build(ctx, filter, 1.0, Mode::NoScores, false)? {
            // `Weight.scorer` returned null: the query searches unfiltered,
            // or, for a KNN query, finds nothing.
            None if q.query.empty_filter_matches_nothing() => return Ok(None),
            None => None,
            Some(mut scorer) => {
                let len = usize::try_from(reader.max_doc).unwrap_or(0);
                let mut bits = FixedBitSet::new(len);
                let mut doc = super::exact_advance(scorer.as_mut(), 0)?;
                while doc != NO_MORE_DOCS {
                    if ctx.live_docs.is_none_or(|live| live.get_doc(doc)) {
                        if let Some(d) = usize::try_from(doc).ok().filter(|&d| d < bits.len()) {
                            bits.set(d);
                        }
                    }
                    doc = super::exact_advance(scorer.as_mut(), doc.saturating_add(1))?;
                }
                Some(bits)
            }
        },
    };
    if !q.query.constant_score() {
        let mut hits = ScoredCollector::default();
        q.query
            .score_leaf_filtered(&leaf, boost, filter.as_ref(), &mut hits)?;
        if hits.docs.is_empty() {
            return Ok(None);
        }
        // `boost` bounds a distance feature; other queries' scores can exceed it.
        let max = hits.scores.iter().copied().fold(boost, f32::max);
        return Ok(Some(Box::new(ScoredDocs {
            docs: hits.docs,
            scores: hits.scores,
            at: 0,
            doc: -1,
            max,
            min: 0.0,
            prune: mode == Mode::TopScores,
        })));
    }
    let len = usize::try_from(reader.max_doc).unwrap_or(0);
    let mut hits = SetCollector {
        bits: crate::bit_set_pool::take(len).unwrap_or_else(|| FixedBitSet::new(len)),
        cardinality: 0,
    };
    q.query
        .score_leaf_filtered(&leaf, boost, filter.as_ref(), &mut hits)?;
    let SetCollector { bits, cardinality } = hits;
    if cardinality == 0 {
        crate::bit_set_pool::give(bits);
        return Ok(None);
    }
    let inner: BoxScorer<'a> = Box::new(super::cache::CachedScorer::new(std::sync::Arc::new(
        super::cache::CachedSet::Bits { bits, cardinality },
    )));
    Ok(constant(inner, boost, mode))
}

/// Collects a scoring document query's matches with their scores.
#[derive(Default)]
struct ScoredCollector {
    docs: Vec<i32>,
    scores: Vec<f32>,
}

impl crate::collector::ScoringCollector for ScoredCollector {
    fn collect(&mut self, doc_id: i32, score: f32) {
        // Ascending, each document once.
        if self.docs.last().is_none_or(|&last| doc_id > last) {
            self.docs.push(doc_id);
            self.scores.push(score);
        }
    }
}

/// A scoring document query's matches and scores, every score at most
/// `max` (the boost). Lucene's own scorer (`DistanceScorer` of the distance
/// feature queries) iterates the doc values and narrows itself to the points
/// near the origin once a minimum competitive score is set; this one has
/// scored every document up front, so a threshold just skips those below it.
struct ScoredDocs {
    docs: Vec<i32>,
    scores: Vec<f32>,
    /// The index of the current document in `docs`.
    at: usize,
    doc: i32,
    max: f32,
    /// The minimum competitive score, honoured in `TOP_SCORES` only.
    min: f32,
    prune: bool,
}

impl ScoredDocs {
    /// Positions on the first document from index `at` that can compete.
    fn settle(&mut self, mut at: usize) -> i32 {
        while at < self.docs.len() && self.scores[at] < self.min {
            at += 1;
        }
        self.at = at;
        self.doc = self.docs.get(at).copied().unwrap_or(NO_MORE_DOCS);
        self.doc
    }
}

impl Scorer for ScoredDocs {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        let from = if self.doc < 0 {
            0
        } else {
            self.at.saturating_add(1)
        };
        Ok(self.settle(from))
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let from =
            self.at + self.docs[self.at.min(self.docs.len())..].partition_point(|&d| d < target);
        Ok(self.settle(from))
    }
    fn cost(&self) -> i64 {
        i64::try_from(self.docs.len()).unwrap_or(i64::MAX)
    }
    fn score(&mut self) -> Result<f32> {
        Ok(self.scores.get(self.at).copied().unwrap_or(0.0))
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(self.max)
    }
    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        if self.prune {
            self.min = min;
        }
        Ok(())
    }
}

/// Collects a document query's matches into a set.
struct SetCollector {
    bits: FixedBitSet,
    cardinality: i64,
}

impl crate::collector::ScoringCollector for SetCollector {
    fn collect(&mut self, doc_id: i32, _score: f32) {
        if let Ok(i) = usize::try_from(doc_id) {
            // FBS: a document query collects its own segment's documents,
            // below `maxDoc`, the set's length; the check keeps a corrupt
            // one from panicking.
            if i < self.bits.len() && !self.bits.get(i) {
                self.bits.set(i);
                self.cardinality += 1;
            }
        }
    }
}

/// `DocValuesRewriteMethod`: the multi-term query's terms looked up in the
/// field's `SORTED_SET` (or `SORTED`) doc-values dictionary, and every
/// document holding one of those ordinals.
pub(crate) fn doc_values_rewrite<'a>(
    ctx: &LeafContext<'a>,
    q: &MultiTermQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    use lucene_codecs::doc_values::SortedSetKind;
    let field = q.field();
    let reader = reader(ctx, field)?;
    let Some(fi) = reader.field_infos().field_by_name(field) else {
        return Ok(None);
    };
    let Some((meta, data)) = reader.doc_values_for_field(fi.number) else {
        return Ok(None);
    };
    enum Ords<'e> {
        Single(&'e doc_values::SortedEntry),
        Multi(&'e SortedNumericEntry),
    }
    let (ords, terms) = if let Some(e) = meta.sorted_set_entry(fi.number) {
        match &e.kind {
            SortedSetKind::Single(s) => (Ords::Single(s), &s.terms),
            SortedSetKind::Multi { ords, terms } => (Ords::Multi(ords), terms),
        }
    } else if let Some(s) = meta.sorted_entry(fi.number) {
        (Ords::Single(s), &s.terms)
    } else {
        return Ok(None);
    };
    let accepts = term_matcher(&q.source)?;
    let store = |e| crate::Error::from(lucene_codecs::blocktree::Error::Store(e));
    // `query.getTermsEnum(values.termsEnum())`: the dictionary from the
    // first term the source can match (`seekCeil`), until it can match no
    // later one -- not every ordinal of the field.
    let (from, within) = term_range(&q.source)?;
    let mut dict = lucene_codecs::terms_dict::TermsDict::open(data, terms).map_err(store)?;
    let size = dict.size();
    // `TermsDict::seek_ceil`, spelled out: the doc-values dictionary's own,
    // already fallible -- not the block tree's retired infallible one.
    let mut ord = match &from {
        Some(lo) => {
            match lucene_codecs::terms_dict::TermsDict::seek_ceil(&mut dict, lo).map_err(store)? {
                lucene_codecs::terms_dict::SeekStatus::End => size,
                _ => dict.ord(),
            }
        }
        None => 0,
    };
    // `buildOrdinalSet` (Lucene 10.6, branch_10_6, #16297): the accepted
    // ordinals collected in order, then held as their `[min, max]` range when
    // contiguous, else as a bit set spanning only that range -- memory and
    // clearing that scale with the matching terms, not the field's ordinal
    // count. The same ordinals either way, so the same matches.
    let mut collected: Vec<i64> = Vec::new();
    while ord < size {
        let term = dict.seek_ord(ord).map_err(store)?;
        if !within(term) {
            break;
        }
        if accepts(term) {
            collected.push(ord);
        }
        ord = ord.saturating_add(1);
    }
    let (Some(&min_ord), Some(&max_ord)) = (collected.first(), collected.last()) else {
        return Ok(None);
    };
    let span = usize::try_from(max_ord.saturating_sub(min_ord))
        .unwrap_or(0)
        .saturating_add(1);
    // `contiguous`: one ordinal per slot of the span (ascending, distinct).
    let ord_set = if collected.len() == span {
        None
    } else {
        let mut bits = FixedBitSet::new(span);
        for &o in &collected {
            // FBS: `min_ord <= o <= max_ord`, so `o - min_ord < span`.
            bits.set(usize::try_from(o.saturating_sub(min_ord)).unwrap_or(0));
        }
        Some(bits)
    };
    let mut buf = Vec::new();
    let mut single = match &ords {
        Ords::Single(s) => Some(NumericReader::new(data, &s.ords)),
        Ords::Multi(_) => None,
    };
    let mut multi = match &ords {
        Ords::Multi(m) => Some(SortedNumericReader::new(data, m)),
        Ords::Single(_) => None,
    };
    let accept = move |doc: i32| -> Result<bool> {
        buf.clear();
        if let Some(r) = single.as_mut() {
            buf.extend(r.value(doc)?);
        }
        if let Some(r) = multi.as_mut() {
            r.values(doc, &mut buf)?;
        }
        Ok(buf.iter().any(|&o| {
            (min_ord..=max_ord).contains(&o)
                && ord_set.as_ref().is_none_or(|b| {
                    // FBS: `o` is in `[min_ord, max_ord]`, the set's span.
                    b.get(usize::try_from(o.saturating_sub(min_ord)).unwrap_or(0))
                })
        }))
    };
    Ok(constant(
        Box::new(TwoPhaseDocs {
            doc: -1,
            max_doc: reader.max_doc,
            accept,
            match_cost: 1.0,
        }),
        boost,
        mode,
    ))
}

/// A predicate over a term's bytes.
type TermPredicate = Box<dyn Fn(&[u8]) -> bool>;

/// Where in sorted term order `source` can match: the term to seek to
/// (`None`: the first), and while a term is still in range -- the source's
/// literal prefix, or a range's upper bound. A source without either reads
/// to the end.
fn term_range(source: &MultiTermSource) -> Result<(Option<Vec<u8>>, TermPredicate)> {
    let prefix = |p: Vec<u8>| -> (Option<Vec<u8>>, TermPredicate) {
        if p.is_empty() {
            (None, Box::new(|_: &[u8]| true))
        } else {
            (Some(p.clone()), Box::new(move |t: &[u8]| t.starts_with(&p)))
        }
    };
    Ok(match source {
        MultiTermSource::Prefix(p) => prefix(p.prefix.clone()),
        MultiTermSource::Wildcard(w) => {
            prefix(lucene_codecs::wildcard::WildcardPattern::new(&w.pattern).literal_prefix())
        }
        MultiTermSource::Regexp(r) => prefix(
            lucene_codecs::regexp::RegexpPattern::new(r.pattern.as_bytes())?.literal_prefix(),
        ),
        MultiTermSource::TermRange(r) => {
            let upper = r.upper.clone();
            let include_upper = r.include_upper;
            (
                r.lower.clone(),
                Box::new(move |t: &[u8]| match &upper {
                    None => true,
                    Some(hi) if include_upper => t <= hi.as_slice(),
                    Some(hi) => t < hi.as_slice(),
                }),
            )
        }
        MultiTermSource::Automaton(_) => (None, Box::new(|_: &[u8]| true)),
        MultiTermSource::TermSet(q) => {
            let last = q.terms.last().cloned();
            (
                q.terms.first().cloned(),
                Box::new(move |t: &[u8]| last.as_deref().is_some_and(|l| t <= l)),
            )
        }
    })
}

/// Whether a term is one `source` enumerates.
fn term_matcher(source: &MultiTermSource) -> Result<TermPredicate> {
    Ok(match source {
        MultiTermSource::Prefix(p) => {
            let prefix = p.prefix.clone();
            Box::new(move |t: &[u8]| t.starts_with(&prefix))
        }
        MultiTermSource::Wildcard(w) => {
            let pattern = lucene_codecs::wildcard::WildcardPattern::new(&w.pattern);
            Box::new(move |t: &[u8]| pattern.matches(t))
        }
        MultiTermSource::Regexp(r) => {
            let pattern = lucene_codecs::regexp::RegexpPattern::new(r.pattern.as_bytes())?;
            Box::new(move |t: &[u8]| pattern.matches(t))
        }
        MultiTermSource::TermRange(r) => {
            let r = r.clone();
            Box::new(move |t: &[u8]| r.accepts(t))
        }
        MultiTermSource::Automaton(a) => {
            use lucene_util::automaton::{AutomatonType, CompiledAutomaton};
            let compiled = CompiledAutomaton::with_options(&a.automaton, false, true, a.binary)
                .map_err(|e| crate::Error::InvalidQuery(format!("automaton: {e:?}")))?;
            Box::new(move |t: &[u8]| match compiled.automaton_type {
                AutomatonType::NONE => false,
                AutomatonType::ALL => true,
                AutomatonType::SINGLE => compiled.term.as_deref() == Some(t),
                AutomatonType::NORMAL => compiled.byte_runnable().is_some_and(|r| r.run(t)),
            })
        }
        MultiTermSource::TermSet(q) => {
            let q = q.clone();
            Box::new(move |t: &[u8]| q.contains(t))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_merge_visitor_walks_the_set_with_the_tree() {
        let points = vec![vec![2u8], vec![4], vec![9]];
        let mut v = MergePointVisitor {
            points: &points,
            next: 0,
            docs: Vec::new(),
        };
        // Below every point, a single-valued cell of one point, a cell that
        // spans one: outside, inside, crossing.
        assert_eq!(v.compare(&[0], &[1]), Relation::CellOutsideQuery);
        assert_eq!(v.compare(&[2], &[2]), Relation::CellInsideQuery);
        v.grow(2);
        v.visit_many(&[5, 6]);
        v.visit(7);
        assert_eq!(v.compare(&[3], &[5]), Relation::CellCrossesQuery);
        assert_eq!(v.next, 1, "2 is behind the cell");
        v.visit_with_value(8, &[3]);
        v.visit_with_value(9, &[4]);
        v.visit_with_value(10, &[5]);
        assert_eq!(v.compare(&[10], &[11]), Relation::CellOutsideQuery);
        v.visit_with_value(11, &[12]);
        assert_eq!(v.docs, vec![5, 6, 7, 9]);
    }
}
