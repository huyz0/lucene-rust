//! Doc-values, index-sort and point ranges as scorer-tree leaves:
//! `SortedNumericDocValuesRangeQuery` (behind `NumericDocValuesRangeQuery`),
//! `IndexSortSortedNumericDocValuesRangeQuery`, `PointRangeQuery` over any
//! number of dimensions and any width, `PointInSetQuery`, and
//! `DocValuesRewriteMethod`. Every one is a `ConstantScoreWeight`: its
//! documents score the boost.

use lucene_codecs::doc_values::{self, NumericReader, SortedNumericEntry, SortedNumericReader};
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
/// shape without a skipper (this port's `SegmentReader` opens no `.dvs`), so
/// its approximation is the whole segment where Java's is the column's
/// documents -- the same matches, found by asking more documents.
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
    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }
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
    let mut column = values.column();
    let mut buf = Vec::new();
    let accept = move |doc: i32| -> Result<bool> {
        column.values(doc, &mut buf)?;
        Ok(buf
            .iter()
            .find(|&&v| v >= lower)
            .is_some_and(|&v| v <= upper))
    };
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
/// sort's comparator; anywhere else, `fallback`.
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
    let mut docs = Vec::with_capacity(usize::try_from(last - first).unwrap_or(0));
    for doc in first..last {
        if dense || values.value(doc)?.is_some() {
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

/// The documents `docs` (unsorted, possibly repeated) as a constant-scored
/// leaf: a bit set when dense, a list otherwise (`DocIdSetBuilder`).
fn doc_set<'a>(mut docs: Vec<i32>, max_doc: i32, boost: f32, mode: Mode) -> Option<BoxScorer<'a>> {
    if docs.is_empty() {
        return None;
    }
    let len = usize::try_from(max_doc).unwrap_or(0);
    let inner: BoxScorer<'a> = if docs.len() > len / 128 {
        let mut bits = FixedBitSet::new(len);
        for &d in &docs {
            if let Ok(i) = usize::try_from(d) {
                // FBS: a BKD walk only returns doc ids below `maxDoc`.
                if i < len {
                    bits.set(i);
                }
            }
        }
        let cardinality = bits.cardinality() as i64;
        Box::new(super::cache::CachedScorer::new(std::sync::Arc::new(
            super::cache::CachedSet::Bits { bits, cardinality },
        )))
    } else {
        lucene_util::doc_id_sort::sort_dedup_doc_ids(&mut docs);
        Box::new(DocList::new(docs, Vec::new()))
    };
    constant(inner, boost, mode)
}

/// The field's points and the segment's `maxDoc`, after
/// `PointRangeQuery`/`PointInSetQuery`'s configuration checks.
fn points_field<'a>(
    ctx: &LeafContext<'a>,
    field: &str,
    num_dims: usize,
    bytes_per_dim: usize,
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
    if usize::try_from(info.num_dims).ok() != Some(num_dims)
        || usize::try_from(info.bytes_per_dim).ok() != Some(bytes_per_dim)
    {
        return Err(crate::Error::InvalidQuery(format!(
            "field=\"{field}\" was indexed with numDims={} bytesPerDim={} but this query has numDims={num_dims} bytesPerDim={bytes_per_dim}",
            info.num_dims, info.bytes_per_dim
        )));
    }
    Ok(Some((points, number, max_doc)))
}

/// `PointRangeQuery`: every document with a point inside the box.
pub(crate) fn point_range<'a>(
    ctx: &LeafContext<'a>,
    q: &PointRangeQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let Some((points, number, max_doc)) = points_field(ctx, &q.field, q.num_dims, q.bytes_per_dim)?
    else {
        return Ok(None);
    };
    let docs = points.reader.range_query(number, &q.lower, &q.upper)?;
    Ok(doc_set(docs, max_doc, boost, mode))
}

/// `PointInSetQuery`: every document with a point equal to one of the set.
pub(crate) fn point_in_set<'a>(
    ctx: &LeafContext<'a>,
    q: &PointInSetQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    if q.points.is_empty() {
        return Ok(None);
    }
    let Some((points, number, max_doc)) = points_field(ctx, &q.field, q.num_dims, q.bytes_per_dim)?
    else {
        return Ok(None);
    };
    let mut docs = Vec::new();
    for p in &q.points {
        docs.extend(points.reader.range_query(number, p, p)?);
    }
    Ok(doc_set(docs, max_doc, boost, mode))
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
    let mut cursor = lucene_codecs::terms_dict::TermsCursor::open(data, terms).map_err(store)?;
    let mut accepted = Vec::new();
    let mut any = false;
    while let Some(term) = cursor.next_term().map_err(store)? {
        let yes = accepts(term);
        any |= yes;
        accepted.push(yes);
    }
    if !any {
        return Ok(None);
    }
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
            usize::try_from(o)
                .ok()
                .and_then(|i| accepted.get(i))
                .copied()
                .unwrap_or(false)
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

/// Whether a term is one `source` enumerates.
fn term_matcher(source: &MultiTermSource) -> Result<Box<dyn Fn(&[u8]) -> bool>> {
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
                AutomatonType::NORMAL => compiled.get_byte_runnable().is_some_and(|r| r.run(t)),
            })
        }
    })
}
