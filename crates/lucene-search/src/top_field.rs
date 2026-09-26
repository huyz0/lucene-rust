//! `TopFieldCollector`: the top hits of a query by a sort -- one or more
//! keys among the score, the document id and a numeric doc-values field --
//! with `searchAfter` paging and `NumericComparator`'s points-based skipping
//! of documents that can no longer compete.
//!
//! A port of Lucene 10.5.0's `TopFieldCollector` (`SimpleFieldCollector`,
//! `PagingFieldCollector`), `FieldValueHitQueue`, `MultiLeafFieldComparator`,
//! `FieldComparator.RelevanceComparator`, `DocComparator` and
//! `NumericComparator` with its `PointsCompetitiveDISIBuilder`, driven the way
//! `Weight.DefaultBulkScorer` drives a collector that has a competitive
//! iterator.
//!
//! # Values
//!
//! Every comparator here orders `i64`s ascending, with the sort's `reverse`
//! flipping the sign of each comparison as `FieldValueHitQueue.reverseMul`
//! does. What the `i64` is depends on the key:
//!
//! * a numeric field: its *comparable long*, which is what
//!   `NumericComparator` itself compares with points
//!   (`missingValueAsComparableLong`, `bottomAsComparableLong`): the value for
//!   `LONG`/`INT`, `NumericUtils.doubleToSortableLong` for `DOUBLE` and
//!   `floatToSortableInt` for `FLOAT` -- which is exactly the number a
//!   `SortedNumericDocValuesField` stores for those types, and orders as
//!   `Double.compare`/`Float.compare` do;
//! * the document: its global id (`DocComparator`);
//! * the score: `-floatToSortableInt(score)` -- `RelevanceComparator` orders
//!   `Float.compare(b, a)`, highest first.
//!
//! [`FieldDoc::values`] carries a hit's values in the same encoding, except
//! that a score is its `f32` bits, so a caller can hand back the float as it
//! was computed.
//!
//! # Deviations
//!
//! None in which hits are returned or in what order. Two in how much work is
//! skipped: a segment whose sort field has no points but a doc-values skip
//! index is scanned without skipping (Lucene's
//! `DVSkipperCompetitiveDISIBuilder`). That only changes how many documents
//! are counted past the total-hits threshold -- a count Lucene reports as a
//! lower bound for exactly that reason.

use std::collections::HashMap;

use lucene_codecs::doc_values::{NumericReader, SortedNumericReader};
use lucene_codecs::points::{IntersectVisitor, PointsReader, PointsScratch, Relation};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::{ScoreMode, ScoringCollector, TotalHits, TotalHitsRelation};
use crate::directory_reader::SegmentReader;
use crate::exec::{self, Mode, Scorer, NO_MORE_DOCS};
use crate::field_norms::FieldNorms;
use crate::multi_segment::OpenSegment;
use crate::query::{BooleanQuery, Clause};
use crate::Result;

/// `SortField.Type`, for the keys this collector sorts by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortType {
    /// `SortField.FIELD_SCORE`'s type: by relevance, highest first.
    Score,
    /// `SortField.FIELD_DOC`'s type: by document id, lowest first.
    Doc,
    Long,
    Int,
    Double,
    Float,
}

/// `SortedNumericSelector.Type`: which of a document's values it sorts by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selector {
    Min,
    Max,
}

/// One sort key: `SortField` for the score or the document, or a
/// `SortedNumericSortField` (which also reads a single-valued `NUMERIC`
/// column, as `DocValues.getSortedNumeric` does).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortField {
    /// The doc-values field; unused for [`SortType::Score`]/[`SortType::Doc`].
    pub field: String,
    pub ty: SortType,
    pub reverse: bool,
    pub selector: Selector,
    /// The comparable long a document without a value sorts as (see the
    /// module doc); Lucene's default missing value is `0`.
    pub missing: i64,
}

impl SortField {
    /// `SortField.FIELD_SCORE`.
    pub fn score() -> Self {
        Self::of_type(SortType::Score)
    }

    /// `SortField.FIELD_DOC`.
    pub fn doc() -> Self {
        Self::of_type(SortType::Doc)
    }

    /// A numeric key on `field`, `MIN` selector, missing value `0`.
    pub fn numeric(field: &str, ty: SortType, reverse: bool) -> Self {
        Self {
            field: field.to_string(),
            ty,
            reverse,
            selector: Selector::Min,
            missing: 0,
        }
    }

    fn of_type(ty: SortType) -> Self {
        Self {
            field: String::new(),
            ty,
            reverse: false,
            selector: Selector::Min,
            missing: 0,
        }
    }

    /// The width of the field's points, which `NumericComparator` insists
    /// match (`bytesCount`); `None` for the score and the document.
    fn point_bytes(&self) -> Option<usize> {
        match self.ty {
            SortType::Long | SortType::Double => Some(8),
            SortType::Int | SortType::Float => Some(4),
            SortType::Score | SortType::Doc => None,
        }
    }
}

/// A hit: its global document id and its sort values (see the module doc).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDoc {
    pub doc: i32,
    pub values: Vec<i64>,
}

/// `TopFieldDocs`: the hits, best first, and the total.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopFieldDocs {
    pub hits: Vec<FieldDoc>,
    pub total: TotalHits,
}

/// Why a sorted search could not run.
#[derive(Debug, thiserror::Error)]
pub enum SortError {
    #[error("a sort needs at least one key")]
    NoKeys,
    #[error("the search-after document has {got} sort values, the sort {want}")]
    AfterArity { got: usize, want: usize },
    /// `NumericComparator`'s `IllegalArgumentException`.
    #[error("field {field} is indexed with {dims} dimensions of {bytes} bytes; sorting needs one of {want}")]
    PointsShape {
        field: String,
        dims: i32,
        bytes: i32,
        want: usize,
    },
    #[error("field {0} has doc values of a type that cannot be sorted numerically")]
    DocValuesType(String),
}

/// `NumericUtils.floatToSortableInt`, over `Float.floatToIntBits` (one NaN).
fn float_to_sortable_int(f: f32) -> i32 {
    let bits = if f.is_nan() {
        0x7fc0_0000
    } else {
        f.to_bits() as i32
    };
    bits ^ ((bits >> 31) & 0x7fff_ffff)
}

/// A score as the relevance comparator's value (see the module doc).
fn score_value(score: f32) -> i64 {
    -i64::from(float_to_sortable_int(score))
}

/// `NumericUtils.sortableBytesToLong`/`sortableBytesToInt`.
fn sortable_bytes_to_long(b: &[u8]) -> i64 {
    match b.len() {
        8 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(b);
            (u64::from_be_bytes(a) ^ (1 << 63)) as i64
        }
        4 => {
            let mut a = [0u8; 4];
            a.copy_from_slice(b);
            i64::from((u32::from_be_bytes(a) ^ 0x8000_0000) as i32)
        }
        _ => 0,
    }
}

/// `Pruning`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pruning {
    None,
    GreaterThan,
    GreaterThanOrEqualTo,
}

/// A `FieldComparator`: one key's slot values, bottom and top, and (for a
/// numeric key) the state `NumericComparator` keeps across segments.
struct Comparator {
    field: SortField,
    /// `reverseMul`.
    mul: i32,
    values: Vec<i64>,
    bottom: i64,
    top: i64,
    top_set: bool,
    pruning: Pruning,
    single_sort: bool,
    hits_threshold_reached: bool,
    queue_full: bool,
}

impl Comparator {
    fn compare(&self, a: usize, b: usize) -> i32 {
        self.mul * cmp(self.values[a], self.values[b])
    }
}

fn cmp(a: i64, b: i64) -> i32 {
    match a.cmp(&b) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// `FieldValueHitQueue.Entry`.
#[derive(Debug, Clone, Copy)]
struct Entry {
    slot: usize,
    doc: i32,
}

/// `FieldValueHitQueue.lessThan`: whether `a` sorts after `b` -- the queue's
/// top is its worst hit.
fn less_than(comps: &[Comparator], a: Entry, b: Entry) -> bool {
    for c in comps {
        let r = c.compare(a.slot, b.slot);
        if r != 0 {
            return r > 0;
        }
    }
    a.doc > b.doc
}

/// Lucene's `PriorityQueue` over [`Entry`], ordered by [`less_than`].
struct HitQueue {
    heap: Vec<Entry>,
}

impl HitQueue {
    fn top(&self) -> Option<Entry> {
        self.heap.first().copied()
    }

    fn add(&mut self, comps: &[Comparator], e: Entry) {
        self.heap.push(e);
        let mut i = self.heap.len() - 1;
        while i > 0 {
            let parent = (i - 1) / 2;
            if less_than(comps, self.heap[i], self.heap[parent]) {
                self.heap.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    }

    /// `updateTop`: the top changed; restore the heap.
    fn update_top(&mut self, comps: &[Comparator]) {
        let n = self.heap.len();
        let mut i = 0;
        loop {
            let l = 2 * i + 1;
            if l >= n {
                break;
            }
            let r = l + 1;
            let child = if r < n && less_than(comps, self.heap[r], self.heap[l]) {
                r
            } else {
                l
            };
            if less_than(comps, self.heap[child], self.heap[i]) {
                self.heap.swap(child, i);
                i = child;
            } else {
                break;
            }
        }
    }

    fn pop(&mut self, comps: &[Comparator]) -> Option<Entry> {
        if self.heap.is_empty() {
            return None;
        }
        let last = self.heap.len() - 1;
        self.heap.swap(0, last);
        let e = self.heap.pop();
        self.update_top(comps);
        e
    }
}

/// `TopFieldCollector`'s reader-wide state.
struct TopField {
    comps: Vec<Comparator>,
    queue: HitQueue,
    num_hits: usize,
    total_hits: u64,
    threshold: u64,
    relation: TotalHitsRelation,
    queue_full: bool,
    needs_scores: bool,
    can_set_min_score: bool,
    min_competitive_score: f32,
    /// `searchSortPartOfIndexSort`: here only a sort led by the document id.
    doc_first: bool,
    after: Option<FieldDoc>,
    /// `PagingFieldCollector.collectedHits`.
    collected_hits: usize,
    /// `scoreMode.isExhaustive()`, fixed at construction.
    exhaustive: bool,
}

impl TopField {
    fn new(sort: &[SortField], num_hits: usize, threshold: u64, after: Option<&FieldDoc>) -> Self {
        let n = sort.len();
        let comps: Vec<Comparator> = sort
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut pruning = if i > 0 {
                    Pruning::None
                } else if n > 1 {
                    Pruning::GreaterThan
                } else {
                    Pruning::GreaterThanOrEqualTo
                };
                if f.point_bytes().is_none() {
                    pruning = Pruning::None;
                }
                Comparator {
                    field: f.clone(),
                    mul: if f.reverse { -1 } else { 1 },
                    values: vec![0; num_hits],
                    bottom: 0,
                    top: 0,
                    top_set: false,
                    pruning,
                    single_sort: false,
                    hits_threshold_reached: false,
                    queue_full: false,
                }
            })
            .collect();
        let needs_scores = sort.iter().any(|f| f.ty == SortType::Score);
        let threshold = threshold.max(num_hits as u64);
        let can_set_min_score =
            sort[0].ty == SortType::Score && !sort[0].reverse && threshold != u64::MAX;
        let doc_first = sort[0].ty == SortType::Doc && !sort[0].reverse;
        let mut tf = Self {
            comps,
            queue: HitQueue {
                heap: Vec::with_capacity(num_hits),
            },
            num_hits,
            total_hits: 0,
            threshold,
            relation: TotalHitsRelation::EqualTo,
            queue_full: false,
            needs_scores,
            can_set_min_score,
            min_competitive_score: 0.0,
            doc_first,
            after: after.cloned(),
            collected_hits: 0,
            exhaustive: false,
        };
        tf.exhaustive = tf.score_mode().is_exhaustive();
        match after {
            None if tf.comps.len() == 1 => tf.comps[0].single_sort = true,
            None => {}
            Some(a) => {
                for (c, &v) in tf.comps.iter_mut().zip(&a.values) {
                    c.top_set = true;
                    c.top = match c.field.ty {
                        SortType::Score => score_value(f32::from_bits(v as u32)),
                        _ => v,
                    };
                }
            }
        }
        if tf.doc_first {
            // `firstComparator.disableSkipping()`.
            tf.comps[0].pruning = Pruning::None;
        }
        tf
    }

    /// `TopFieldCollector.scoreMode`.
    fn score_mode(&self) -> ScoreMode {
        if self.can_set_min_score {
            ScoreMode::TopScores
        } else if self.threshold != u64::MAX {
            if self.needs_scores {
                ScoreMode::TopDocsWithScores
            } else {
                ScoreMode::TopDocs
            }
        } else if self.needs_scores {
            ScoreMode::Complete
        } else {
            ScoreMode::CompleteNoScores
        }
    }

    /// `populateResults`/`newTopDocs`.
    fn top_docs(mut self) -> TopFieldDocs {
        let mut hits = Vec::with_capacity(self.queue.heap.len());
        while let Some(e) = self.queue.pop(&self.comps) {
            let values = self
                .comps
                .iter()
                .map(|c| {
                    let v = c.values[e.slot];
                    match c.field.ty {
                        SortType::Score => i64::from(sortable_int_to_float(-v).to_bits()),
                        _ => v,
                    }
                })
                .collect();
            hits.push(FieldDoc { doc: e.doc, values });
        }
        hits.reverse();
        TopFieldDocs {
            hits,
            total: TotalHits {
                value: self.total_hits,
                relation: self.relation,
            },
        }
    }
}

/// `NumericUtils.sortableIntToFloat`, from a relevance value's negation.
fn sortable_int_to_float(v: i64) -> f32 {
    let i = v as i32;
    f32::from_bits((i ^ ((i >> 31) & 0x7fff_ffff)) as u32)
}

/// A numeric doc-values column as `NumericDocValues`: a `NUMERIC` column
/// directly, a `SORTED_NUMERIC` one through `SortedNumericSelector`.
enum Column<'a> {
    Absent,
    Single(NumericReader<'a>),
    Multi(SortedNumericReader<'a>, Vec<i64>, Selector),
}

impl Column<'_> {
    fn value(&mut self, doc: i32) -> Result<Option<i64>> {
        Ok(match self {
            Column::Absent => None,
            Column::Single(r) => r.value(doc).map_err(crate::Error::from)?,
            Column::Multi(r, buf, selector) => {
                r.values(doc, buf).map_err(crate::Error::from)?;
                match selector {
                    Selector::Min => buf.first().copied(),
                    Selector::Max => buf.last().copied(),
                }
            }
        })
    }
}

/// `NumericComparator.NumericLeafComparator`: one segment's column, and the
/// competitive iterator when points allow one.
struct LeafNumeric<'a> {
    column: Column<'a>,
    int: bool,
    missing: i64,
    /// The last document read, so `compareBottom` and `copy` read once.
    cached: (i32, i64),
    competitive: Option<Competitive<'a>>,
}

impl LeafNumeric<'_> {
    fn value(&mut self, doc: i32) -> Result<i64> {
        if self.cached.0 == doc {
            return Ok(self.cached.1);
        }
        let v = match self.column.value(doc)? {
            Some(v) if self.int => i64::from(v as i32),
            Some(v) => v,
            None => self.missing,
        };
        self.cached = (doc, v);
        Ok(v)
    }
}

/// `MIN_SKIP_INTERVAL`, `MAX_SKIP_INTERVAL`.
const MIN_SKIP_INTERVAL: u32 = 32;
const MAX_SKIP_INTERVAL: u32 = 8192;

/// `UpdateableDocIdSetIterator` over what `PointsCompetitiveDISIBuilder`
/// hands it: all documents, or a materialized set -- sorted ids or, past
/// `maxDoc / 128` of them, a bit set (`DocIdSetBuilder`'s two forms).
enum Iter {
    All {
        max_doc: i32,
        doc: i32,
    },
    Docs {
        docs: Vec<i32>,
        next: usize,
        doc: i32,
    },
    Bits {
        bits: FixedBitSet,
        doc: i32,
    },
}

impl Iter {
    fn doc_id(&self) -> i32 {
        match self {
            Iter::All { doc, .. } | Iter::Docs { doc, .. } | Iter::Bits { doc, .. } => *doc,
        }
    }

    fn advance(&mut self, target: i32) -> i32 {
        match self {
            Iter::All { max_doc, doc } => {
                *doc = if target >= *max_doc {
                    NO_MORE_DOCS
                } else {
                    target
                };
                *doc
            }
            Iter::Docs { docs, next, doc } => {
                // Targets only grow, and usually by little: gallop from the
                // current entry, then search the bracket found.
                let rest = &docs[*next..];
                let mut hi = 1;
                while hi < rest.len() && rest[hi - 1] < target {
                    hi *= 2;
                }
                let lo = hi / 2;
                let hi = hi.min(rest.len());
                *next += lo + rest[lo..hi].partition_point(|&d| d < target);
                *doc = docs.get(*next).copied().unwrap_or(NO_MORE_DOCS);
                *doc
            }
            Iter::Bits { bits, doc } => {
                *doc = usize::try_from(target)
                    .ok()
                    .and_then(|t| bits.next_set_bit(t))
                    .map_or(NO_MORE_DOCS, |d| d as i32);
                *doc
            }
        }
    }
}

/// `NumericComparator.PointsCompetitiveDISIBuilder`.
struct Competitive<'a> {
    points: &'a PointsReader<'a>,
    field_number: i32,
    bytes: usize,
    point_doc_count: i32,
    max_doc: i32,
    leaf_top_set: bool,
    iter: Iter,
    min_value: i64,
    max_value: i64,
    max_doc_visited: i32,
    update_counter: u32,
    current_skip_interval: u32,
    iterator_cost: i64,
    try_update_fail_count: u32,
    /// The column's documents with a value, for when the points are too
    /// dense to narrow anything (`getNumericDocValues` as the iterator).
    with_value: WithValue<'a>,
    /// A cleared id buffer for the next update to fill.
    scratch: Vec<i32>,
    /// A cleared bit set for the next dense update.
    spare_bits: Option<FixedBitSet>,
    /// The tree walk's buffers, kept across updates (Lucene keeps its
    /// `PointTree` for the estimate the same way).
    walk: PointsScratch,
}

/// Where a column's docs-with-values set is: every document, or an
/// `IndexedDISI` region of the `.dvd`.
#[derive(Clone, Copy)]
enum WithValue<'a> {
    None,
    All,
    Disi {
        region: &'a [u8],
        dense_rank_power: u8,
    },
}

impl<'a> WithValue<'a> {
    fn of(data: &'a [u8], e: &lucene_codecs::doc_values::NumericEntry) -> Self {
        if e.is_empty_field() {
            return WithValue::None;
        }
        if e.is_dense() {
            return WithValue::All;
        }
        let region = usize::try_from(e.docs_with_field_offset)
            .ok()
            .zip(usize::try_from(e.docs_with_field_length).ok())
            .and_then(|(start, len)| data.get(start..start.checked_add(len)?));
        match region {
            Some(region) => WithValue::Disi {
                region,
                dense_rank_power: e.dense_rank_power,
            },
            None => WithValue::None,
        }
    }
}

impl Competitive<'_> {
    /// `encodeBottom`.
    fn encode_bottom(&mut self, c: &Comparator) {
        if !c.field.reverse {
            self.max_value = c.bottom;
            if c.pruning == Pruning::GreaterThanOrEqualTo && self.max_value != i64::MIN {
                self.max_value -= 1;
            }
        } else {
            self.min_value = c.bottom;
            if c.pruning == Pruning::GreaterThanOrEqualTo && self.min_value != i64::MAX {
                self.min_value += 1;
            }
        }
    }

    /// `encodeTop`.
    fn encode_top(&mut self, c: &Comparator) {
        let tighten = c.single_sort && c.pruning == Pruning::GreaterThanOrEqualTo && c.queue_full;
        if !c.field.reverse {
            self.min_value = c.top;
            if tighten && self.min_value != i64::MAX {
                self.min_value += 1;
            }
        } else {
            self.max_value = c.top;
            if tighten && self.max_value != i64::MIN {
                self.max_value -= 1;
            }
        }
    }

    /// `isMissingValueCompetitive`.
    fn missing_competitive(&self, c: &Comparator) -> bool {
        let missing = c.field.missing;
        if c.queue_full {
            let r = cmp(missing, c.bottom);
            let competitive = if c.field.reverse {
                if c.pruning == Pruning::GreaterThanOrEqualTo {
                    r > 0
                } else {
                    r >= 0
                }
            } else if c.pruning == Pruning::GreaterThanOrEqualTo {
                r < 0
            } else {
                r <= 0
            };
            if !competitive {
                return false;
            }
        }
        if self.leaf_top_set {
            let r = cmp(missing, c.top);
            return if c.field.reverse { r <= 0 } else { r >= 0 };
        }
        true
    }

    /// `updateCompetitiveIterator`.
    fn update(&mut self, c: &Comparator) -> Result<()> {
        if !c.hits_threshold_reached {
            return Ok(());
        }
        if !self.leaf_top_set && !c.queue_full {
            return Ok(());
        }
        if self.point_doc_count != self.max_doc && self.missing_competitive(c) {
            return Ok(());
        }
        self.update_counter += 1;
        if self.update_counter > 256
            && (self.update_counter & (self.current_skip_interval - 1))
                != self.current_skip_interval - 1
        {
            return Ok(());
        }
        if c.queue_full {
            self.encode_bottom(c);
        }
        self.do_update()
    }

    /// `PointsCompetitiveDISIBuilder.doUpdateCompetitiveIterator`.
    fn do_update(&mut self) -> Result<()> {
        let threshold = ((self.iterator_cost as u64) >> 3) as i64;
        let mut visitor = CompetitiveVisitor {
            min: self.min_value,
            max: self.max_value,
            max_doc_visited: self.max_doc_visited,
            docs: std::mem::take(&mut self.scratch),
            bits: None,
            upgrade_at: ((self.max_doc as usize) >> 7).max(1),
            added: 0,
            spare: self.spare_bits.take(),
            max_doc: self.max_doc as usize,
        };
        let estimate = self
            .points
            .estimate_point_count_bounded_in(
                self.field_number,
                &mut visitor,
                threshold,
                &mut self.walk,
            )
            .map_err(crate::Error::from)?;
        if estimate >= threshold {
            self.scratch = visitor.docs;
            self.spare_bits = visitor.spare;
            self.update_skip_interval(false);
            if i64::from(self.point_doc_count) < self.iterator_cost {
                // Use the set of documents with values to drive iteration.
                self.iter = match self.with_value {
                    WithValue::None => Iter::Docs {
                        docs: Vec::new(),
                        next: 0,
                        doc: -1,
                    },
                    WithValue::All => Iter::All {
                        max_doc: self.max_doc,
                        doc: -1,
                    },
                    WithValue::Disi {
                        region,
                        dense_rank_power,
                    } => Iter::Docs {
                        docs: lucene_codecs::indexed_disi::decode_doc_ids(region, dense_rank_power)
                            .map_err(|e| {
                                crate::Error::from(lucene_codecs::doc_values::Error::from(e))
                            })?,
                        next: 0,
                        doc: -1,
                    },
                };
                self.iterator_cost = i64::from(self.point_doc_count);
            }
            return Ok(());
        }
        self.points
            .intersect_in(self.field_number, &mut visitor, &mut self.walk)
            .map_err(crate::Error::from)?;
        self.spare_bits = visitor.spare.take();
        let mut docs = visitor.docs;
        let new_iter = match visitor.bits {
            Some(bits) => {
                self.iterator_cost = visitor.added as i64;
                self.scratch = docs;
                Iter::Bits { bits, doc: -1 }
            }
            None => {
                lucene_util::doc_id_sort::sort_dedup_doc_ids(&mut docs);
                self.iterator_cost = docs.len() as i64;
                Iter::Docs {
                    docs,
                    next: 0,
                    doc: -1,
                }
            }
        };
        // The set being replaced gives its storage to the next update.
        match std::mem::replace(&mut self.iter, new_iter) {
            Iter::Docs { mut docs, .. } if self.scratch.capacity() == 0 => {
                docs.clear();
                self.scratch = docs;
            }
            Iter::Bits { mut bits, .. } if self.spare_bits.is_none() => {
                bits.clear_all();
                self.spare_bits = Some(bits);
            }
            _ => {}
        }
        self.update_skip_interval(true);
        Ok(())
    }

    fn update_skip_interval(&mut self, success: bool) {
        if self.update_counter > 256 {
            if success {
                self.current_skip_interval =
                    (self.current_skip_interval / 2).max(MIN_SKIP_INTERVAL);
                self.try_update_fail_count = 0;
            } else if self.try_update_fail_count >= 3 {
                self.current_skip_interval =
                    (self.current_skip_interval * 2).min(MAX_SKIP_INTERVAL);
                self.try_update_fail_count = 0;
            } else {
                self.try_update_fail_count += 1;
            }
        }
    }
}

/// The `IntersectVisitor` `doUpdateCompetitiveIterator` builds: documents past
/// `maxDocVisited` whose value is in `[min, max]`.
struct CompetitiveVisitor {
    min: i64,
    max: i64,
    max_doc_visited: i32,
    docs: Vec<i32>,
    /// `DocIdSetBuilder`'s dense form: once `docs` would pass `upgrade_at`
    /// ids, they move here and every later one is set directly.
    bits: Option<FixedBitSet>,
    upgrade_at: usize,
    /// Ids added (`DocIdSetBuilder`'s cost), duplicates included.
    added: usize,
    /// A cleared bit set to upgrade into, when one is spare.
    spare: Option<FixedBitSet>,
    max_doc: usize,
}

impl CompetitiveVisitor {
    #[inline]
    fn add(&mut self, doc: i32) {
        self.added += 1;
        match &mut self.bits {
            Some(b) => b.set(doc as usize),
            None => {
                self.docs.push(doc);
                if self.docs.len() >= self.upgrade_at {
                    self.upgrade();
                }
            }
        }
    }

    /// `DocIdSetBuilder.upgradeToBitSet`.
    fn upgrade(&mut self) {
        let mut b = self
            .spare
            .take()
            .unwrap_or_else(|| FixedBitSet::new(self.max_doc));
        for &d in &self.docs {
            b.set(d as usize);
        }
        self.docs.clear();
        self.bits = Some(b);
    }
}

impl IntersectVisitor for CompetitiveVisitor {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        let min = sortable_bytes_to_long(min_packed);
        let max = sortable_bytes_to_long(max_packed);
        if min > self.max || max < self.min {
            Relation::CellOutsideQuery
        } else if min < self.min || max > self.max {
            Relation::CellCrossesQuery
        } else {
            Relation::CellInsideQuery
        }
    }

    fn visit(&mut self, doc_id: i32) {
        if doc_id > self.max_doc_visited {
            self.add(doc_id);
        }
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        if doc_id <= self.max_doc_visited {
            return;
        }
        let v = sortable_bytes_to_long(packed_value);
        if v >= self.min && v <= self.max {
            self.add(doc_id);
        }
    }
}

/// Where a collected document's score comes from: its scorer, asked lazily,
/// or `None` when the bulk scorer handed it over already.
type Sc<'s> = Option<&'s mut dyn Scorer>;

/// One key's per-segment source.
enum LeafKey<'a> {
    Score,
    Doc,
    Numeric(LeafNumeric<'a>),
}

/// `TopFieldLeafCollector` for one segment.
struct Leaf<'a> {
    doc_base: i32,
    keys: Vec<LeafKey<'a>>,
    collected_all_competitive: bool,
    /// `PagingFieldCollector`'s `afterDoc`, in this segment's space.
    after_doc: i32,
    /// `CollectionTerminatedException`.
    terminated: bool,
    /// The current document's score, when the sort reads it, and the
    /// document it belongs to.
    score: f32,
    score_doc: i32,
}

/// The segment's column and points for a numeric key.
fn open_leaf<'a>(
    tf: &TopField,
    reader: &'a SegmentReader,
    points: Option<&'a crate::points_query::PointsInput<'a>>,
    doc_base: i32,
    cost: i64,
) -> Result<Leaf<'a>> {
    let max_doc = reader.max_doc;
    let mut keys = Vec::with_capacity(tf.comps.len());
    for c in &tf.comps {
        let f = &c.field;
        keys.push(match f.ty {
            SortType::Score => LeafKey::Score,
            SortType::Doc => LeafKey::Doc,
            _ => {
                let info = reader
                    .field_infos()
                    .fields
                    .iter()
                    .find(|i| i.name == f.field);
                let (column, with_value) = match info {
                    None => (Column::Absent, WithValue::None),
                    Some(info) => match reader.doc_values_for_field(info.number) {
                        None => (Column::Absent, WithValue::None),
                        Some((meta, data)) => {
                            if let Some(e) = meta.sorted_numeric_entry(info.number) {
                                (
                                    Column::Multi(
                                        SortedNumericReader::new(data, e),
                                        Vec::new(),
                                        f.selector,
                                    ),
                                    WithValue::of(data, &e.numeric),
                                )
                            } else if let Some(e) = meta.numeric_entry(info.number) {
                                (
                                    Column::Single(NumericReader::new(data, e)),
                                    WithValue::of(data, e),
                                )
                            } else if meta.binary_entry(info.number).is_some()
                                || meta.sorted_entry(info.number).is_some()
                                || meta.sorted_set_entry(info.number).is_some()
                            {
                                return Err(SortError::DocValuesType(f.field.clone()).into());
                            } else {
                                (Column::Absent, WithValue::None)
                            }
                        }
                    },
                };
                let competitive = match (c.pruning, points, f.point_bytes()) {
                    (Pruning::None, _, _) | (_, None, _) | (_, _, None) => None,
                    (_, Some(p), Some(bytes)) => match p.field_number(&f.field) {
                        None => None,
                        Some(num) => match p.reader.field(num) {
                            None => None,
                            Some(pf) => {
                                if pf.num_dims != 1 || pf.bytes_per_dim as usize != bytes {
                                    return Err(SortError::PointsShape {
                                        field: f.field.clone(),
                                        dims: pf.num_dims,
                                        bytes: pf.bytes_per_dim,
                                        want: bytes,
                                    }
                                    .into());
                                }
                                let mut comp = Competitive {
                                    points: &p.reader,
                                    field_number: num,
                                    bytes,
                                    point_doc_count: pf.doc_count,
                                    max_doc,
                                    leaf_top_set: c.top_set,
                                    iter: Iter::All { max_doc, doc: -1 },
                                    min_value: i64::MIN,
                                    max_value: i64::MAX,
                                    max_doc_visited: -1,
                                    update_counter: 0,
                                    current_skip_interval: MIN_SKIP_INTERVAL,
                                    iterator_cost: -1,
                                    try_update_fail_count: 0,
                                    with_value,
                                    scratch: Vec::new(),
                                    spare_bits: None,
                                    walk: PointsScratch::default(),
                                };
                                if comp.leaf_top_set {
                                    comp.encode_top(c);
                                }
                                // `setScorer`: the scorer's cost, or `maxDoc`
                                // behind `ScoreCachingWrappingScorer`.
                                comp.iterator_cost = if tf.needs_scores {
                                    i64::from(max_doc)
                                } else {
                                    cost
                                };
                                comp.update(c)?;
                                Some(comp)
                            }
                        },
                    },
                };
                debug_assert!(competitive
                    .as_ref()
                    .is_none_or(|c| c.bytes == f.point_bytes().unwrap_or(0)));
                LeafKey::Numeric(LeafNumeric {
                    column,
                    int: f.ty == SortType::Int,
                    missing: f.missing,
                    cached: (-1, 0),
                    competitive,
                })
            }
        });
    }
    let after_doc = tf
        .after
        .as_ref()
        .map_or(0, |a| a.doc.saturating_sub(doc_base));
    Ok(Leaf {
        doc_base,
        keys,
        collected_all_competitive: false,
        after_doc,
        terminated: false,
        score: 0.0,
        score_doc: -1,
    })
}

impl Leaf<'_> {
    /// Key `i`'s value for `doc`. A score is read from `scorer` the first
    /// time it is needed for a document and kept (`ScoreCachingWrappingScorer`);
    /// with no scorer, [`Self::score`] already holds it.
    fn value(&mut self, i: usize, doc: i32, scorer: &mut Sc<'_>) -> Result<i64> {
        Ok(match &mut self.keys[i] {
            LeafKey::Score => {
                if let Some(s) = scorer.as_deref_mut() {
                    if self.score_doc != doc {
                        self.score = s.score()?;
                        self.score_doc = doc;
                    }
                }
                score_value(self.score)
            }
            LeafKey::Doc => i64::from(self.doc_base + doc),
            LeafKey::Numeric(n) => n.value(doc)?,
        })
    }

    fn compare_bottom(&mut self, tf: &TopField, doc: i32, scorer: &mut Sc<'_>) -> Result<i32> {
        for (i, c) in tf.comps.iter().enumerate() {
            let r = c.mul * cmp(c.bottom, self.value(i, doc, scorer)?);
            if r != 0 {
                return Ok(r);
            }
        }
        Ok(0)
    }

    fn compare_top(&mut self, tf: &TopField, doc: i32, scorer: &mut Sc<'_>) -> Result<i32> {
        for (i, c) in tf.comps.iter().enumerate() {
            let r = c.mul * cmp(c.top, self.value(i, doc, scorer)?);
            if r != 0 {
                return Ok(r);
            }
        }
        Ok(0)
    }

    fn copy(
        &mut self,
        tf: &mut TopField,
        slot: usize,
        doc: i32,
        scorer: &mut Sc<'_>,
    ) -> Result<()> {
        for i in 0..tf.comps.len() {
            let v = self.value(i, doc, scorer)?;
            tf.comps[i].values[slot] = v;
            if let LeafKey::Numeric(n) = &mut self.keys[i] {
                if let Some(comp) = n.competitive.as_mut() {
                    comp.max_doc_visited = doc;
                }
            }
        }
        Ok(())
    }

    fn set_bottom(&mut self, tf: &mut TopField, slot: usize) -> Result<()> {
        for i in 0..tf.comps.len() {
            let c = &mut tf.comps[i];
            c.bottom = c.values[slot];
            if let LeafKey::Numeric(n) = &mut self.keys[i] {
                c.queue_full = true;
                if let Some(comp) = n.competitive.as_mut() {
                    comp.update(c)?;
                }
            }
        }
        Ok(())
    }

    fn set_hits_threshold_reached(&mut self, tf: &mut TopField) -> Result<()> {
        let c = &mut tf.comps[0];
        if let LeafKey::Numeric(n) = &mut self.keys[0] {
            c.hits_threshold_reached = true;
            if let Some(comp) = n.competitive.as_mut() {
                comp.update(c)?;
            }
        }
        Ok(())
    }

    fn competitive(&mut self) -> Option<&mut Iter> {
        match self.keys.first_mut() {
            Some(LeafKey::Numeric(n)) => n.competitive.as_mut().map(|c| &mut c.iter),
            _ => None,
        }
    }

    /// `countHit`.
    fn count_hit(&mut self, tf: &mut TopField) -> Result<()> {
        tf.total_hits += 1;
        if !tf.exhaustive
            && tf.relation == TotalHitsRelation::EqualTo
            && tf.total_hits > tf.threshold
        {
            self.set_hits_threshold_reached(tf)?;
            tf.relation = TotalHitsRelation::GreaterThanOrEqualTo;
        }
        Ok(())
    }

    /// `thresholdCheck`: true when `doc` cannot enter the queue.
    fn threshold_check(
        &mut self,
        tf: &mut TopField,
        doc: i32,
        scorer: &mut Sc<'_>,
    ) -> Result<bool> {
        if self.collected_all_competitive || self.compare_bottom(tf, doc, scorer)? <= 0 {
            if tf.doc_first {
                if tf.total_hits > tf.threshold {
                    tf.relation = TotalHitsRelation::GreaterThanOrEqualTo;
                    self.terminated = true;
                } else {
                    self.collected_all_competitive = true;
                }
            } else if tf.relation == TotalHitsRelation::EqualTo {
                update_min_competitive_score(tf);
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn collect_competitive_hit(
        &mut self,
        tf: &mut TopField,
        doc: i32,
        scorer: &mut Sc<'_>,
    ) -> Result<()> {
        let Some(bottom) = tf.queue.top() else {
            return Ok(());
        };
        self.copy(tf, bottom.slot, doc, scorer)?;
        tf.queue.heap[0].doc = self.doc_base + doc;
        tf.queue.update_top(&tf.comps);
        let slot = tf.queue.top().map_or(0, |e| e.slot);
        self.set_bottom(tf, slot)?;
        update_min_competitive_score(tf);
        Ok(())
    }

    fn collect_any_hit(
        &mut self,
        tf: &mut TopField,
        doc: i32,
        hits_collected: usize,
        scorer: &mut Sc<'_>,
    ) -> Result<()> {
        let slot = hits_collected - 1;
        self.copy(tf, slot, doc, scorer)?;
        tf.queue.add(
            &tf.comps,
            Entry {
                slot,
                doc: self.doc_base + doc,
            },
        );
        tf.queue_full = slot == tf.num_hits - 1;
        if tf.queue_full {
            let slot = tf.queue.top().map_or(0, |e| e.slot);
            self.set_bottom(tf, slot)?;
            update_min_competitive_score(tf);
        }
        Ok(())
    }

    /// `SimpleFieldCollector`/`PagingFieldCollector`'s `collect`.
    fn collect(&mut self, tf: &mut TopField, doc: i32, mut scorer: Sc<'_>) -> Result<()> {
        let scorer = &mut scorer;
        self.count_hit(tf)?;
        if tf.after.is_none() {
            if tf.queue_full {
                if self.threshold_check(tf, doc, scorer)? {
                    return Ok(());
                }
                self.collect_competitive_hit(tf, doc, scorer)
            } else {
                let n = tf.total_hits as usize;
                self.collect_any_hit(tf, doc, n, scorer)
            }
        } else {
            if tf.queue_full && self.threshold_check(tf, doc, scorer)? {
                return Ok(());
            }
            let top_cmp = self.compare_top(tf, doc, scorer)?;
            if top_cmp > 0 || (top_cmp == 0 && doc <= self.after_doc) {
                if tf.relation == TotalHitsRelation::EqualTo {
                    update_min_competitive_score(tf);
                }
                return Ok(());
            }
            if tf.queue_full {
                self.collect_competitive_hit(tf, doc, scorer)
            } else {
                tf.collected_hits += 1;
                let n = tf.collected_hits;
                self.collect_any_hit(tf, doc, n, scorer)
            }
        }
    }
}

/// `updateMinCompetitiveScore`.
fn update_min_competitive_score(tf: &mut TopField) {
    if tf.can_set_min_score && tf.queue_full && tf.total_hits > tf.threshold {
        let Some(bottom) = tf.queue.top() else {
            return;
        };
        let min = sortable_int_to_float(-tf.comps[0].values[bottom.slot]);
        if min > tf.min_competitive_score {
            tf.min_competitive_score = min;
            tf.relation = TotalHitsRelation::GreaterThanOrEqualTo;
        }
    }
}

/// The leaf collector as a [`ScoringCollector`], for the bulk scorers: a sort
/// led by the score has no competitive iterator, and prunes by score.
struct BulkLeaf<'l, 'a> {
    tf: &'l mut TopField,
    leaf: &'l mut Leaf<'a>,
    error: Option<crate::Error>,
}

impl ScoringCollector for BulkLeaf<'_, '_> {
    fn collect(&mut self, doc_id: i32, score: f32) {
        if self.error.is_some() {
            return;
        }
        self.leaf.score = score;
        if let Err(e) = self.leaf.collect(self.tf, doc_id, None) {
            self.error = Some(e);
        }
    }

    /// The bottom's score is still competitive (a tie can win on a later
    /// key), so the threshold handed out is the float just below it: the
    /// scorers skip only what scores strictly less, as Lucene's
    /// `setMinCompetitiveScore(bottom)` means.
    fn min_competitive_score(&self) -> Option<f32> {
        (self.tf.min_competitive_score > 0.0).then(|| self.tf.min_competitive_score.next_down())
    }

    fn score_mode(&self) -> ScoreMode {
        match self.tf.score_mode() {
            ScoreMode::TopScores => ScoreMode::TopScores,
            _ if self.tf.needs_scores => ScoreMode::Complete,
            _ => ScoreMode::CompleteNoScores,
        }
    }
}

/// `IndexSearcher.search(query, TopFieldCollectorManager(sort, topN, after,
/// totalHitsThreshold))` over the reader's segments, in order.
///
/// `readers[i]` is the segment `segments[i]` was opened from (its doc values
/// and field infos); the segments' points must be open when a numeric key is
/// to skip with them. `norms` as for
/// [`crate::multi_segment::search_boolean_query_multi_segment_maxscore_counting`],
/// read only when the sort has a score key. `after` is `searchAfter`, its
/// values encoded as [`FieldDoc::values`] are.
#[allow(clippy::too_many_arguments)]
pub fn search_sorted(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    sort: &[SortField],
    top_n: usize,
    total_hits_threshold: u64,
    after: Option<&FieldDoc>,
) -> Result<TopFieldDocs> {
    if sort.is_empty() {
        return Err(SortError::NoKeys.into());
    }
    if let Some(a) = after {
        if a.values.len() != sort.len() {
            return Err(SortError::AfterArity {
                got: a.values.len(),
                want: sort.len(),
            }
            .into());
        }
    }
    let empty = TotalHits {
        value: 0,
        relation: TotalHitsRelation::EqualTo,
    };
    if top_n == 0 {
        return Ok(TopFieldDocs {
            hits: Vec::new(),
            total: empty,
        });
    }
    let rewritten = crate::multi_segment::rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    let mut tf = TopField::new(sort, top_n, total_hits_threshold, after);
    let global = if tf.needs_scores {
        Some(crate::multi_segment::global_boolean_stats(segments, query)?)
    } else {
        None
    };
    // `BooleanQuery.rewrite`: a boolean of one clause is that clause (so a
    // lone term is a term to the query cache, not a composite).
    let clauses = query.must.len() + query.filter.len() + query.should.len() + query.must_not.len();
    let clause = match (&query.must[..], &query.should[..]) {
        ([only], _) if clauses == 1 && query.minimum_should_match == 0 => only.clone(),
        (_, [only]) if clauses == 1 && query.minimum_should_match <= 1 => only.clone(),
        _ => Clause::Boolean(Box::new(query.clone())),
    };
    let mode = match tf.score_mode() {
        ScoreMode::TopScores => Mode::TopScores,
        _ if tf.needs_scores => Mode::Complete,
        _ => Mode::NoScores,
    };
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
            norms: norms.get(i).copied().flatten(),
            global: global.as_ref(),
            max_doc: seg.max_doc,
            cache: seg.cache,
        };
        if sort[0].ty == SortType::Score {
            // No competitive iterator: the bulk scorers, pruning by score.
            let Some(mut bulk) = exec::bulk_boolean(&ctx, query, 1.0, mode)? else {
                continue;
            };
            let mut leaf = open_leaf(
                &tf,
                reader,
                seg.points,
                seg.doc_base,
                i64::from(reader.max_doc),
            )?;
            let mut c = BulkLeaf {
                tf: &mut tf,
                leaf: &mut leaf,
                error: None,
            };
            exec::score_segment(&mut bulk, mode, seg.live_docs, &mut c)?;
            if let Some(e) = c.error {
                return Err(e);
            }
            continue;
        }
        // The score is not the first key, so it only breaks ties: the
        // documents are iterated without scores (the cheaper tree, and the
        // one the query cache serves), and a scoring tree of the same query
        // is advanced only to the documents whose score a comparison reads.
        let Some(child) = exec::build::child(&ctx, &clause, 1.0, Mode::NoScores, true)? else {
            continue;
        };
        let mut scorer = child.into_scorer(Mode::NoScores);
        let mut scores = if tf.needs_scores {
            exec::build::child(&ctx, &clause, 1.0, mode, true)?.map(|c| ScoreAt {
                inner: c.into_scorer(mode),
                doc: -1,
            })
        } else {
            None
        };
        let mut leaf = open_leaf(&tf, reader, seg.points, seg.doc_base, scorer.cost())?;
        score_competitive(
            &mut *scorer,
            scores.as_mut(),
            &mut tf,
            &mut leaf,
            seg.live_docs,
        )?;
    }
    Ok(tf.top_docs())
}

/// `DefaultBulkScorer.score` for a collector that may have a competitive
/// iterator (`scoreCompetitiveIterator`, `scoreTwoPhaseOrCompetitiveIterator`):
/// the scorer is advanced past documents the iterator has ruled out.
fn score_competitive(
    scorer: &mut dyn Scorer,
    mut scores: Option<&mut ScoreAt<'_>>,
    tf: &mut TopField,
    leaf: &mut Leaf<'_>,
    live_docs: Option<&FixedBitSet>,
) -> Result<()> {
    let two_phase = scorer.two_phase();
    // Membership instead of leapfrog: when the scorer can say whether a
    // document matches without moving (a cached bit set, match-all) and no
    // score is read, a narrowed competitive set is walked on its own and each
    // of its documents tested -- one move per candidate instead of two.
    let by_membership = !two_phase && scorer.contains(0).is_some();
    let mut doc = match leaf.competitive() {
        Some(it) if it.doc_id() > 0 => scorer.advance(it.doc_id())?,
        _ => scorer.next_doc()?,
    };
    while doc != NO_MORE_DOCS {
        if by_membership {
            if let Some(it) = leaf
                .competitive()
                .filter(|it| !matches!(it, Iter::All { .. }))
            {
                let d = if it.doc_id() < doc {
                    it.advance(doc)
                } else {
                    it.doc_id()
                };
                if d == NO_MORE_DOCS {
                    return Ok(());
                }
                if scorer.contains(d) == Some(true) && live_docs.is_none_or(|l| l.get_doc(d)) {
                    leaf.collect(tf, d, score_at(&mut scores, d))?;
                    if leaf.terminated {
                        return Ok(());
                    }
                }
                doc = d.saturating_add(1);
                continue;
            }
            if scorer.doc_id() < doc {
                doc = scorer.advance(doc)?;
                continue;
            }
        }
        if let Some(it) = leaf.competitive() {
            if it.doc_id() < doc {
                let next = it.advance(doc);
                if next != doc {
                    doc = scorer.advance(next)?;
                    continue;
                }
            }
        }
        if live_docs.is_none_or(|l| l.get_doc(doc)) && (!two_phase || scorer.matches()?) {
            leaf.collect(tf, doc, score_at(&mut scores, doc))?;
            if leaf.terminated {
                return Ok(());
            }
            if leaf.collected_all_competitive {
                return count_rest(scorer, tf, leaf, live_docs);
            }
        }
        doc = scorer.next_doc()?;
    }
    Ok(())
}

/// The scoring tree, pointed at `doc` for [`Leaf::value`] to read lazily.
fn score_at<'s>(scores: &'s mut Option<&mut ScoreAt<'_>>, doc: i32) -> Sc<'s> {
    scores.as_deref_mut().map(|s| {
        s.doc = doc;
        s as &mut dyn Scorer
    })
}

/// A scoring tree read only for the documents whose score is compared:
/// `score()` first moves it to [`Self::doc`], which the non-scoring tree has
/// already matched, so it is there.
struct ScoreAt<'a> {
    inner: exec::BoxScorer<'a>,
    doc: i32,
}

impl Scorer for ScoreAt<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.inner.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.inner.advance(target)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
    fn score(&mut self) -> Result<f32> {
        if self.inner.doc_id() < self.doc {
            exec::exact_advance(&mut *self.inner, self.doc)?;
        }
        debug_assert_eq!(
            self.inner.doc_id(),
            self.doc,
            "the scoring tree matches what the plain one did"
        );
        self.inner.score()
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        self.inner.max_score(up_to)
    }
}

/// The rest of a segment once a sort led by the document id has filled its
/// queue: every later document is non-competitive, so `collect` would only
/// count it (`countHit`, then `thresholdCheck` stopping the segment once the
/// count passes the threshold). That is done here directly, a run of
/// consecutive matches at a time where the scorer reports one.
fn count_rest(
    scorer: &mut dyn Scorer,
    tf: &mut TopField,
    leaf: &mut Leaf<'_>,
    live_docs: Option<&FixedBitSet>,
) -> Result<()> {
    let two_phase = scorer.two_phase();
    let limit = if tf.exhaustive {
        u64::MAX
    } else {
        tf.threshold
    };
    let mut doc = scorer.next_doc()?;
    while doc != NO_MORE_DOCS {
        if !two_phase && live_docs.is_none() {
            // Every document up to the run's end matches: count them at once,
            // but no further than the one that passes the threshold.
            let end = scorer.doc_id_run_end();
            let run = u64::try_from(i64::from(end) - i64::from(doc))
                .unwrap_or(1)
                .max(1);
            let room = limit.saturating_sub(tf.total_hits).saturating_add(1);
            let n = run.min(room);
            tf.total_hits += n;
            if n < run || tf.total_hits > limit {
                break;
            }
            doc = if end == NO_MORE_DOCS {
                NO_MORE_DOCS
            } else {
                scorer.advance(end)?
            };
            continue;
        }
        if live_docs.is_none_or(|l| l.get_doc(doc)) && (!two_phase || scorer.matches()?) {
            tf.total_hits += 1;
            if tf.total_hits > limit {
                break;
            }
        }
        doc = scorer.next_doc()?;
    }
    if tf.total_hits > limit {
        // `countHit`'s switch to a lower bound, then `thresholdCheck`'s
        // `CollectionTerminatedException`.
        tf.relation = TotalHitsRelation::GreaterThanOrEqualTo;
        leaf.terminated = true;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory_reader::DirectoryReader;
    use lucene_store::FsDirectory;

    fn fixture(name: &str) -> DirectoryReader {
        let dir = format!("{}/../../fixtures/data/{name}", env!("CARGO_MANIFEST_DIR"));
        DirectoryReader::open(&FsDirectory::open(dir)).expect("open fixture")
    }

    fn all() -> BooleanQuery {
        let mut q = BooleanQuery::new();
        q.must
            .push(Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(
                0,
            )));
        q
    }

    fn run(
        reader: &DirectoryReader,
        q: &BooleanQuery,
        sort: &[SortField],
        top_n: usize,
        after: Option<&FieldDoc>,
    ) -> Result<TopFieldDocs> {
        let mut opened = reader.open_segments().expect("open");
        opened.open_points().expect("points");
        let segments = opened.as_open_segments();
        let norms = vec![None; segments.len()];
        search_sorted(
            &segments,
            reader.segment_readers(),
            q,
            &norms,
            sort,
            top_n,
            u64::MAX,
            after,
        )
    }

    #[test]
    fn a_sort_needs_keys_and_an_after_of_its_arity() {
        let r = fixture("sorted_search_index");
        assert!(matches!(
            run(&r, &all(), &[], 10, None),
            Err(crate::Error::Sort(SortError::NoKeys))
        ));
        let after = FieldDoc {
            doc: 3,
            values: vec![1, 2],
        };
        assert!(matches!(
            run(&r, &all(), &[SortField::doc()], 10, Some(&after)),
            Err(crate::Error::Sort(SortError::AfterArity {
                got: 2,
                want: 1
            }))
        ));
        let none = run(&r, &all(), &[SortField::doc()], 0, None).unwrap();
        assert!(none.hits.is_empty());
        assert_eq!(none.total.value, 0);
    }

    #[test]
    fn points_of_another_width_are_refused_as_lucene_refuses_them() {
        // `i` is an `IntPoint`; sorting it as a long asks for 8-byte points.
        let r = fixture("sorted_search_index");
        let e = run(
            &r,
            &all(),
            &[SortField::numeric("i", SortType::Long, false)],
            10,
            None,
        );
        assert!(matches!(
            e,
            Err(crate::Error::Sort(SortError::PointsShape { want: 8, .. }))
        ));
    }

    #[test]
    fn a_non_numeric_column_cannot_be_sorted_numerically() {
        let r = fixture("sorted_dv_index");
        let e = run(
            &r,
            &all(),
            &[SortField::numeric("sorted", SortType::Long, false)],
            3,
            None,
        );
        assert!(matches!(
            e,
            Err(crate::Error::Sort(SortError::DocValuesType(f))) if f == "sorted"
        ));
        // A field with no doc values at all sorts every document as missing.
        let ok = run(
            &r,
            &all(),
            &[SortField::numeric("nope", SortType::Long, true)],
            3,
            None,
        )
        .unwrap();
        assert_eq!(
            ok.hits
                .iter()
                .map(|h| (h.doc, h.values[0]))
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 0), (2, 0)]
        );
    }

    #[test]
    fn values_order_as_lucene_compares_them() {
        // Float.compare: -0.0 before 0.0, NaN after everything.
        let order = [
            f32::NEG_INFINITY,
            -1.5,
            -0.0,
            0.0,
            1e-30,
            2.0,
            f32::INFINITY,
            f32::NAN,
        ];
        for w in order.windows(2) {
            assert!(
                float_to_sortable_int(w[0]) < float_to_sortable_int(w[1]),
                "{w:?}"
            );
            // Relevance runs the other way.
            assert!(score_value(w[0]) > score_value(w[1]), "{w:?}");
        }
        for f in [0.25f32, -3.0, 0.0, -0.0, 7.5e10] {
            assert_eq!(
                sortable_int_to_float(-score_value(f)).to_bits(),
                f.to_bits()
            );
        }
        assert_eq!(
            sortable_bytes_to_long(&crate::points_query::pack_i64(-42)),
            -42
        );
        assert_eq!(sortable_bytes_to_long(&[0x80, 0, 0, 5]), 5);
        assert_eq!(sortable_bytes_to_long(&[0x7f, 0xff, 0xff, 0xff]), -1);
        assert_eq!(
            sortable_bytes_to_long(&[1, 2]),
            0,
            "no other width is a sortable number"
        );
    }

    #[test]
    fn every_competitive_iterator_form_advances_the_same_way() {
        let docs = vec![3, 9, 20];
        let mut bits = FixedBitSet::new(32);
        for &d in &docs {
            bits.set(d as usize);
        }
        let mut forms = [
            Iter::Docs {
                docs: docs.clone(),
                next: 0,
                doc: -1,
            },
            Iter::Bits { bits, doc: -1 },
        ];
        for it in &mut forms {
            assert_eq!(it.doc_id(), -1);
            assert_eq!(it.advance(0), 3);
            assert_eq!(it.advance(4), 9);
            assert_eq!(it.advance(9), 9);
            assert_eq!(it.advance(21), NO_MORE_DOCS);
        }
        let mut all = Iter::All {
            max_doc: 5,
            doc: -1,
        };
        assert_eq!(all.advance(2), 2);
        assert_eq!(all.advance(5), NO_MORE_DOCS);
    }

    #[test]
    fn the_skip_interval_widens_on_failures_and_narrows_on_success() {
        let points = PointsReader::empty();
        let mut c = Competitive {
            points: &points,
            field_number: 0,
            bytes: 8,
            point_doc_count: 0,
            max_doc: 10,
            leaf_top_set: false,
            iter: Iter::All {
                max_doc: 10,
                doc: -1,
            },
            min_value: i64::MIN,
            max_value: i64::MAX,
            max_doc_visited: -1,
            update_counter: 0,
            current_skip_interval: MIN_SKIP_INTERVAL,
            iterator_cost: 10,
            try_update_fail_count: 0,
            with_value: WithValue::None,
            scratch: Vec::new(),
            spare_bits: None,
            walk: PointsScratch::default(),
        };
        // Below 257 updates the interval never moves.
        c.update_skip_interval(false);
        assert_eq!(c.current_skip_interval, MIN_SKIP_INTERVAL);
        c.update_counter = 300;
        for _ in 0..4 {
            c.update_skip_interval(false);
        }
        assert_eq!(
            c.current_skip_interval,
            2 * MIN_SKIP_INTERVAL,
            "the fourth failure doubles it"
        );
        c.current_skip_interval = MAX_SKIP_INTERVAL;
        for _ in 0..4 {
            c.update_skip_interval(false);
        }
        assert_eq!(c.current_skip_interval, MAX_SKIP_INTERVAL, "capped");
        c.update_skip_interval(true);
        assert_eq!(c.current_skip_interval, MAX_SKIP_INTERVAL / 2);
        c.current_skip_interval = MIN_SKIP_INTERVAL;
        c.update_skip_interval(true);
        assert_eq!(c.current_skip_interval, MIN_SKIP_INTERVAL, "floored");
    }

    #[test]
    fn the_visitor_moves_to_a_bit_set_past_the_upgrade_size() {
        let mut v = CompetitiveVisitor {
            min: 10,
            max: 20,
            max_doc_visited: 2,
            docs: Vec::new(),
            bits: None,
            upgrade_at: 3,
            added: 0,
            spare: None,
            max_doc: 64,
        };
        v.visit(1); // already visited: dropped
        v.visit(9);
        v.visit_with_value(5, &crate::points_query::pack_i64(30)); // out of range
        v.visit_with_value(7, &crate::points_query::pack_i64(15));
        assert!(v.bits.is_none());
        assert_eq!(v.docs, [9, 7]);
        v.visit(40); // the third id upgrades
        v.visit(41);
        let bits = v.bits.as_ref().expect("upgraded");
        assert!(v.docs.is_empty());
        assert_eq!(v.added, 4);
        for d in [7, 9, 40, 41] {
            assert!(bits.get(d), "{d}");
        }
        assert_eq!(bits.cardinality(), 4);
    }

    #[test]
    fn the_empty_queue_pops_nothing() {
        let mut q = HitQueue { heap: Vec::new() };
        assert!(q.top().is_none());
        assert!(q.pop(&[]).is_none());
    }
}
