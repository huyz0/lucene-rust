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
//! None in which hits are returned or in what order. Three in how much is
//! counted past the total-hits threshold -- a count both report as a lower
//! bound, and OpenSearch caps at `track_total_hits` anyway:
//!
//! * a segment whose sort field has no points but a doc-values skip index is
//!   scanned without skipping (Lucene's `DVSkipperCompetitiveDISIBuilder`);
//! * `DocComparator`'s competitive iterator is not ported: a sort led by the
//!   document id stops each segment once its count passes the threshold,
//!   where Lucene skips the later segments whole, so the lower bound here is
//!   one hit higher per later matching segment;
//! * the collector consults the competitive iterator per document, where
//!   Lucene's match-all and filter conjunctions (`DenseConjunctionBulkScorer`)
//!   collect whole 4,096-document windows first, so Lucene's bound is usually
//!   the higher one there.
//!
//! And one in when a sort is refused: Lucene builds every segment's
//! comparators before it searches, so a field with points of the wrong width
//! or doc values of the wrong type fails the search even in a segment the
//! query does not match; here only the segments searched are opened.

use std::collections::HashMap;

use lucene_codecs::doc_values::{NumericReader, SortedNumericReader};
use lucene_codecs::field_infos::IndexOptions;
use lucene_codecs::points::{IntersectVisitor, PointsReader, PointsScratch, Relation};
use lucene_codecs::postings::{DocInput, LazyDocsCursor, PostingsFlags};
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
    /// `SortedSetSortField` (and `SortField.Type.STRING` over a `SORTED`
    /// column): by term, through `TermOrdValComparator`. `missing` is `1` for
    /// `STRING_LAST`, `0` for `STRING_FIRST`.
    String,
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
            SortType::Score | SortType::Doc | SortType::String => None,
        }
    }

    /// A keyword key on `field` (`SortedSetSortField`, `MIN`), missing first.
    pub fn string(field: &str, reverse: bool) -> Self {
        Self {
            field: field.to_string(),
            ty: SortType::String,
            reverse,
            selector: Selector::Min,
            missing: 0,
        }
    }
}

/// A hit: its global document id and its sort values (see the module doc).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDoc {
    pub doc: i32,
    pub values: Vec<i64>,
    /// A keyword key's value, parallel to `values` (whose entry is then
    /// `0`): the term, or `None` for a document without one. Empty when the
    /// sort has no keyword key.
    pub terms: Vec<Option<Vec<u8>>>,
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
    /// The scoring tree of a sort that reads scores did not match a document
    /// the non-scoring tree of the same query did.
    #[error("the scoring tree does not match document {0}, which the query matched")]
    ScoringTree(i32),
    /// A points leaf named a document outside the segment.
    #[error(
        "points field {field_number} names document {doc}, outside the segment's 0..{max_doc}"
    )]
    PointsDoc {
        field_number: i32,
        doc: i32,
        max_doc: i32,
    },
    #[error("field {0} has doc values of a type that cannot be sorted numerically")]
    DocValuesType(String),
    #[error("ordinal {0} out of range")]
    Ordinal(i64),
    #[error("field {0} has doc values of a type that cannot be sorted by term")]
    KeywordType(String),
    /// `TermOrdValComparator`: a doc-values term the terms index lacks, or
    /// fewer indexed terms than doc-values ones.
    #[error("doc-values term {0:?} and the terms index disagree")]
    TermsIndex(Vec<u8>),
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
    /// A keyword key's `TermOrdValComparator` state.
    strs: Option<Box<StrSlots>>,
}

/// `TermOrdValComparator`'s reader-wide state: each slot's ordinal, term and
/// the segment (`readerGen`) the ordinal belongs to.
struct StrSlots {
    ords: Vec<i32>,
    values: Vec<Option<Vec<u8>>>,
    reader_gen: Vec<i32>,
    current_gen: i32,
    /// `missingSortCmp`: `1` for `STRING_LAST`, `-1` for `STRING_FIRST`.
    missing_cmp: i32,
    bottom_slot: Option<usize>,
    /// `topValue`: the search-after term; `None` also for a missing one.
    top: Option<Vec<u8>>,
    /// Slots copied in the current segment whose term is not read yet:
    /// within a segment slots compare by ordinal, so the term (Java's
    /// `lookupOrd` in `copy`) is read once, when the segment ends, and only
    /// for the slots still in the queue.
    pending: Vec<bool>,
    /// Filled slots from earlier segments. The queue compares those with
    /// this segment's by term, so while there are any, `copy` reads the
    /// term at once, as Java does.
    old_slots: usize,
}

/// `readerGen` of a slot never filled.
const UNUSED_GEN: i32 = i32::MIN;

impl StrSlots {
    /// `compareValues`.
    fn compare_values(&self, a: &Option<Vec<u8>>, b: &Option<Vec<u8>>) -> i32 {
        match (a, b) {
            (None, None) => 0,
            (None, Some(_)) => self.missing_cmp,
            (Some(_), None) => -self.missing_cmp,
            (Some(x), Some(y)) => x.as_slice().cmp(y.as_slice()) as i32,
        }
    }
}

impl Comparator {
    fn compare(&self, a: usize, b: usize) -> i32 {
        match &self.strs {
            None => self.mul * cmp(self.values[a], self.values[b]),
            Some(st) => {
                let r = if st.reader_gen[a] == st.reader_gen[b] {
                    st.ords[a].wrapping_sub(st.ords[b]).signum()
                } else {
                    st.compare_values(&st.values[a], &st.values[b])
                };
                self.mul * r
            }
        }
    }
}

/// `Long.compare`, as the sign of an `i32`.
fn cmp(a: i64, b: i64) -> i32 {
    a.cmp(&b) as i32
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
                if f.point_bytes().is_none() && f.ty != SortType::String {
                    pruning = Pruning::None;
                }
                let strs = (f.ty == SortType::String).then(|| {
                    Box::new(StrSlots {
                        ords: vec![0; num_hits],
                        values: vec![None; num_hits],
                        reader_gen: vec![UNUSED_GEN; num_hits],
                        current_gen: -1,
                        missing_cmp: if f.missing != 0 { 1 } else { -1 },
                        bottom_slot: None,
                        top: None,
                        pending: vec![false; num_hits],
                        old_slots: 0,
                    })
                });
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
                    strs,
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
                for (i, (c, &v)) in tf.comps.iter_mut().zip(&a.values).enumerate() {
                    c.top_set = true;
                    c.top = match c.field.ty {
                        SortType::Score => score_value(f32::from_bits(v as u32)),
                        _ => v,
                    };
                    if let Some(st) = c.strs.as_mut() {
                        st.top = a.terms.get(i).cloned().flatten();
                    }
                }
            }
        }
        if tf.doc_first {
            // Never a points iterator for the document id (`DocComparator`'s
            // own competitive iterator is not ported; see the module doc).
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
        let any_terms = self.comps.iter().any(|c| c.strs.is_some());
        while let Some(e) = self.queue.pop(&self.comps) {
            let values = self
                .comps
                .iter()
                .map(|c| {
                    let v = c.values[e.slot];
                    match c.field.ty {
                        SortType::Score => i64::from(sortable_int_to_float(-v).to_bits()),
                        SortType::String => 0,
                        _ => v,
                    }
                })
                .collect();
            let terms = if any_terms {
                self.comps
                    .iter()
                    .map(|c| c.strs.as_ref().and_then(|st| st.values[e.slot].clone()))
                    .collect()
            } else {
                Vec::new()
            };
            hits.push(FieldDoc {
                doc: e.doc,
                values,
                terms,
            });
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
    /// [`Self::value`] when a dense single-valued column answers directly
    /// (kept for [`Self::value`]), or `None` for "ask it".
    #[inline]
    fn quick_value(&mut self, doc: i32) -> Option<i64> {
        let Column::Single(r) = &self.column else {
            return None;
        };
        let v = r.dense_value(doc)?;
        let v = if self.int { i64::from(v as i32) } else { v };
        self.cached = (doc, v);
        Some(v)
    }

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
/// `maxDoc / 128` of them, a bit set (`DocIdSetBuilder`'s two forms) -- or,
/// for a keyword key, `TermOrdValComparator`'s disjunction of the competitive
/// terms' postings.
enum Iter<'a> {
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
    Union(Box<Union<'a>>),
}

impl Iter<'_> {
    fn doc_id(&self) -> i32 {
        match self {
            Iter::All { doc, .. } | Iter::Docs { doc, .. } | Iter::Bits { doc, .. } => *doc,
            Iter::Union(u) => u.doc,
        }
    }

    /// `cost`: an upper bound of the documents left, for choosing the
    /// iterator that leads (unknown for a bit set: never it).
    fn cost(&self) -> i64 {
        match self {
            Iter::All { max_doc, .. } => i64::from(*max_doc),
            Iter::Docs { docs, .. } => docs.len() as i64,
            Iter::Bits { .. } => i64::MAX,
            Iter::Union(u) => u.cost,
        }
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        Ok(match self {
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
            Iter::Union(u) => u.advance(target)?,
        })
    }
}

/// `PostingsBasedCompetitiveState`'s disjunction: the postings of a run of
/// consecutive ordinals, in ordinal order, and a heap of them by document.
struct Union<'a> {
    /// `postings`: each competitive term's cursor and ordinal; the live ones
    /// are `legs[lo..hi]`.
    legs: Vec<(LazyDocsCursor<'a>, i32)>,
    /// Each leg's `docFreq`, and the live legs' sum (`cost`).
    doc_freqs: Vec<i64>,
    cost: i64,
    lo: usize,
    hi: usize,
    /// `disjunction`: indices into `legs`, least document first.
    heap: Vec<usize>,
    doc: i32,
}

impl Union<'_> {
    /// `disjunction.clear(); disjunction.addAll(postings)`, the cursors where
    /// they are.
    fn rebuild(&mut self) {
        self.cost = self.doc_freqs[self.lo..self.hi].iter().sum();
        self.heap.clear();
        self.heap.extend(self.lo..self.hi);
        let legs = &self.legs;
        self.heap.sort_by_key(|&i| legs[i].0.doc_id());
    }

    /// `updateTop`: the top moved; sink it.
    fn sift_down(&mut self) {
        let n = self.heap.len();
        let key = |u: &Self, at: usize| u.legs[u.heap[at]].0.doc_id();
        let mut i = 0;
        loop {
            let l = 2 * i + 1;
            if l >= n {
                break;
            }
            let r = l + 1;
            let c = if r < n && key(self, r) < key(self, l) {
                r
            } else {
                l
            };
            if key(self, c) >= key(self, i) {
                break;
            }
            self.heap.swap(i, c);
            i = c;
        }
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        loop {
            let Some(&top) = self.heap.first() else {
                self.doc = NO_MORE_DOCS;
                return Ok(self.doc);
            };
            let d = self.legs[top].0.doc_id();
            if d >= target {
                self.doc = d;
                return Ok(d);
            }
            self.legs[top]
                .0
                .advance(target)
                .map_err(|e| crate::Error::from(crate::blocktree::Error::Postings(e)))?;
            self.sift_down();
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
    iter: Iter<'a>,
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
    /// The set as an iterator.
    fn iter(self, max_doc: i32) -> Result<Iter<'a>> {
        Ok(match self {
            WithValue::None => Iter::Docs {
                docs: Vec::new(),
                next: 0,
                doc: -1,
            },
            WithValue::All => Iter::All { max_doc, doc: -1 },
            WithValue::Disi {
                region,
                dense_rank_power,
            } => Iter::Docs {
                docs: lucene_codecs::indexed_disi::decode_doc_ids(region, dense_rank_power)
                    .map_err(|e| crate::Error::from(lucene_codecs::doc_values::Error::from(e)))?,
                next: 0,
                doc: -1,
            },
        })
    }

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
            corrupt: None,
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
                self.iter = self.with_value.iter(self.max_doc)?;
                self.iterator_cost = i64::from(self.point_doc_count);
            }
            return Ok(());
        }
        self.points
            .intersect_in(self.field_number, &mut visitor, &mut self.walk)
            .map_err(crate::Error::from)?;
        if let Some(doc) = visitor.corrupt {
            return Err(SortError::PointsDoc {
                field_number: self.field_number,
                doc,
                max_doc: self.max_doc,
            }
            .into());
        }
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
    /// The first out-of-segment doc id a leaf named, if any.
    corrupt: Option<i32>,
}

impl CompetitiveVisitor {
    #[inline]
    fn add(&mut self, doc: i32) {
        if !self.in_segment(doc) {
            return;
        }
        self.added += 1;
        match &mut self.bits {
            // FBS: `in_segment` bounded `doc` to `0..max_doc`, the length
            // every set here is built with.
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
            // FBS: every id in `docs` passed `in_segment` (`0..max_doc`), and
            // `b` is `max_doc` bits (new, or a spare of the same segment).
            b.set(d as usize);
        }
        self.docs.clear();
        self.bits = Some(b);
    }

    /// Whether `doc` is a document of the segment; a points leaf naming one
    /// outside it is corrupt, remembered for `do_update` to report.
    #[inline]
    fn in_segment(&mut self, doc: i32) -> bool {
        let ok = doc >= 0 && (doc as usize) < self.max_doc;
        if !ok && self.corrupt.is_none() {
            self.corrupt = Some(doc);
        }
        ok
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

    fn visit_many(&mut self, doc_ids: &[i32]) {
        if self.bits.is_none() && self.docs.len() + doc_ids.len() >= self.upgrade_at {
            self.upgrade();
        }
        let floor = self.max_doc_visited;
        if let Some(&bad) = doc_ids
            .iter()
            .find(|&&d| d < 0 || d as usize >= self.max_doc)
        {
            self.corrupt.get_or_insert(bad);
            return;
        }
        match &mut self.bits {
            Some(b) => {
                for &d in doc_ids {
                    if d > floor {
                        // FBS: every id was checked against `0..max_doc`
                        // above, and `b` is `max_doc` bits.
                        b.set(d as usize);
                        self.added += 1;
                    }
                }
            }
            None => {
                let before = self.docs.len();
                self.docs
                    .extend(doc_ids.iter().copied().filter(|&d| d > floor));
                self.added += self.docs.len() - before;
            }
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
    Numeric(Box<LeafNumeric<'a>>),
    Str(Box<LeafStr<'a>>),
}

/// A keyword column as `SortedDocValues`: a `SORTED` column, or a
/// `SORTED_SET` one through `SortedSetSelector` (`MIN` the first ordinal,
/// `MAX` the last).
enum OrdColumn<'a> {
    Absent,
    Single(NumericReader<'a>),
    Multi(SortedNumericReader<'a>, Vec<i64>, Selector),
}

impl OrdColumn<'_> {
    /// `advanceExact` + `ordValue`, `-1` for a document without a value.
    //
    // SENTINEL: `-1` = "no value", outside the domain of an ordinal (Java's
    // `getOrdForDoc`). Its callers, `LeafStr::{compare_bottom, compare_top,
    // copy}` through `LeafStr::ord`, test `== -1`.
    #[inline]
    fn ord(&mut self, doc: i32) -> Result<i32> {
        let v = match self {
            OrdColumn::Absent => None,
            OrdColumn::Single(r) => r.value(doc).map_err(crate::Error::from)?,
            OrdColumn::Multi(r, buf, selector) => {
                r.values(doc, buf).map_err(crate::Error::from)?;
                match selector {
                    Selector::Min => buf.first().copied(),
                    Selector::Max => buf.last().copied(),
                }
            }
        };
        Ok(match v {
            Some(o) => i32::try_from(o).map_err(|_| crate::Error::from(SortError::Ordinal(o)))?,
            None => -1,
        })
    }
}

/// `TermOrdValComparator.TermOrdValLeafComparator`.
struct LeafStr<'a> {
    column: OrdColumn<'a>,
    dict: Option<lucene_codecs::terms_dict::TermsDict<'a>>,
    bottom_same_reader: bool,
    bottom_ord: i32,
    top_same_reader: bool,
    top_ord: i32,
    missing_ord: i32,
    /// The last document read, so `compareBottom` and `copy` read once.
    cached: (i32, i32),
    /// `competitiveState`, when this key may skip documents.
    competitive: Option<StrCompetitive<'a>>,
}

/// `TermOrdValComparator`'s `CompetitiveState`: `EmptyCompetitiveState` for
/// a segment without the field, `PostingsBasedCompetitiveState` for an
/// indexed one.
struct StrCompetitive<'a> {
    /// The field's terms and postings; `None` for `EmptyCompetitiveState`.
    postings: Option<(&'a crate::blocktree::FieldTerms, &'a DocInput<'a>)>,
    /// Every document has a term.
    dense: bool,
    /// The column's documents with a value (`getSortedDocValues` as an
    /// iterator), for when too many terms compete.
    with_value: WithValue<'a>,
    docs_with_field_set: bool,
    max_doc: i32,
    /// `postings != null`.
    initialized: bool,
    iter: Iter<'a>,
}

/// `PostingsBasedCompetitiveState.MAX_TERMS`, capped (as Lucene caps it) by
/// `IndexSearcher.getMaxClauseCount()`'s default, which is the same.
const MAX_COMPETITIVE_TERMS: i64 = 1024;

impl<'a> LeafStr<'a> {
    #[inline]
    fn ord(&mut self, doc: i32) -> Result<i32> {
        if self.cached.0 != doc {
            self.cached = (doc, self.column.ord(doc)?);
        }
        Ok(self.cached.1)
    }

    /// `lookupTerm`: the term's ordinal here, or `-insertion - 1`; `-1` in a
    /// segment without the field (`DocValues.emptySorted`).
    //
    // SENTINEL: none -- `-1` is `lookupTerm`'s "insert before ordinal 0", in
    // its domain. Every caller tests `< 0`/`>= 0` for found.
    fn lookup_term(&mut self, term: &[u8]) -> Result<i32> {
        let Some(d) = self.dict.as_mut() else {
            return Ok(-1);
        };
        let o = d.lookup_term(term).map_err(store_err)?;
        i32::try_from(o).map_err(|_| SortError::Ordinal(o).into())
    }

    /// `lookupOrd`.
    fn term(&mut self, ord: i32) -> Result<Vec<u8>> {
        match self.dict.as_mut() {
            Some(d) => Ok(d.seek_ord(i64::from(ord)).map_err(store_err)?.to_vec()),
            None => Err(SortError::Ordinal(i64::from(ord)).into()),
        }
    }

    /// `compareBottom`.
    //
    // SENTINEL: none -- `-1` is a comparison result, in the domain.
    #[inline]
    fn compare_bottom(&mut self, doc: i32) -> Result<i32> {
        let mut o = self.ord(doc)?;
        if o == -1 {
            o = self.missing_ord;
        }
        Ok(if self.bottom_same_reader {
            self.bottom_ord.wrapping_sub(o).signum()
        } else if self.bottom_ord >= o {
            1
        } else {
            -1
        })
    }

    /// [`Self::compare_bottom`] when a dense column answers directly, or
    /// `None` for "ask it".
    //
    // SENTINEL: none -- `-1` is a comparison result, in the domain.
    #[inline]
    fn quick_compare_bottom(&self, doc: i32) -> Option<i32> {
        let OrdColumn::Single(r) = &self.column else {
            return None;
        };
        let o = i32::try_from(r.dense_value(doc)?).ok()?;
        Some(if self.bottom_same_reader {
            self.bottom_ord.wrapping_sub(o).signum()
        } else if self.bottom_ord >= o {
            1
        } else {
            -1
        })
    }

    /// `compareTop`.
    //
    // SENTINEL: none -- `-1` is a comparison result, in the domain.
    fn compare_top(&mut self, doc: i32) -> Result<i32> {
        let mut o = self.ord(doc)?;
        if o == -1 {
            o = self.missing_ord;
        }
        Ok(if self.top_same_reader {
            self.top_ord.wrapping_sub(o).signum()
        } else if o <= self.top_ord {
            1
        } else {
            -1
        })
    }

    /// `copy`.
    fn copy(&mut self, st: &mut StrSlots, slot: usize, doc: i32) -> Result<()> {
        let o = self.ord(doc)?;
        if st.reader_gen[slot] != st.current_gen && st.reader_gen[slot] != UNUSED_GEN {
            st.old_slots = st.old_slots.saturating_sub(1);
        }
        st.values[slot] = None;
        st.pending[slot] = false;
        if o == -1 {
            st.ords[slot] = self.missing_ord;
        } else {
            st.ords[slot] = o;
            if st.old_slots > 0 {
                st.values[slot] = Some(self.term(o)?);
            } else {
                st.pending[slot] = true;
            }
        }
        st.reader_gen[slot] = st.current_gen;
        Ok(())
    }

    /// The terms of the slots [`Self::copy`] left pending, read in ordinal
    /// order (the dictionary scans forward inside a block).
    fn materialize(&mut self, st: &mut StrSlots) -> Result<()> {
        let mut slots: Vec<usize> = (0..st.pending.len()).filter(|&i| st.pending[i]).collect();
        slots.sort_unstable_by_key(|&i| st.ords[i]);
        for i in slots {
            st.values[i] = Some(self.term(st.ords[i])?);
            st.pending[i] = false;
        }
        Ok(())
    }

    /// `setBottom`.
    fn set_bottom(&mut self, st: &mut StrSlots, slot: usize) -> Result<()> {
        st.bottom_slot = Some(slot);
        if st.current_gen == st.reader_gen[slot] {
            self.bottom_ord = st.ords[slot];
            self.bottom_same_reader = true;
        } else if st.values[slot].is_none() {
            self.bottom_ord = self.missing_ord;
            self.bottom_same_reader = true;
            st.reader_gen[slot] = st.current_gen;
            st.old_slots = st.old_slots.saturating_sub(1);
        } else {
            let value = st.values[slot].clone().unwrap_or_default();
            let o = self.lookup_term(&value)?;
            if o < 0 {
                self.bottom_ord = -o - 2;
                self.bottom_same_reader = false;
            } else {
                self.bottom_ord = o;
                self.bottom_same_reader = true;
                st.reader_gen[slot] = st.current_gen;
                st.ords[slot] = o;
                st.old_slots = st.old_slots.saturating_sub(1);
            }
        }
        Ok(())
    }

    /// `updateCompetitiveIterator`: the ordinals that can still compete,
    /// from the top (search after) to the bottom, or nothing while missing
    /// values still compete.
    fn update_competitive(&mut self, c: &Comparator) -> Result<()> {
        let Some(st) = c.strs.as_deref() else {
            return Ok(());
        };
        let Some(comp) = self.competitive.as_ref() else {
            return Ok(());
        };
        if !c.hits_threshold_reached || st.bottom_slot.is_none() {
            return Ok(());
        }
        let dense = comp.dense;
        let missing_last = self.missing_ord == i32::MAX;
        let value_count = self.value_count();
        let single = c.single_sort;
        let top_set = st.top.is_some();
        let (min_ord, max_ord): (i64, i64) = if !c.field.reverse {
            let min_ord = if top_set {
                if self.top_same_reader {
                    i64::from(self.top_ord)
                } else {
                    i64::from(self.top_ord) + 1
                }
            } else if missing_last || dense {
                0
            } else {
                -1
            };
            let max_ord = if self.bottom_ord == self.missing_ord {
                if single {
                    value_count - 1
                } else {
                    i64::from(i32::MAX)
                }
            } else if self.bottom_same_reader && single {
                i64::from(self.bottom_ord) - 1
            } else {
                i64::from(self.bottom_ord)
            };
            (min_ord, max_ord)
        } else {
            let min_ord = if self.bottom_ord == self.missing_ord {
                if single {
                    0
                } else {
                    -1
                }
            } else if self.bottom_same_reader {
                i64::from(self.bottom_ord) + i64::from(single)
            } else {
                i64::from(self.bottom_ord) + 1
            };
            let max_ord = if top_set {
                i64::from(self.top_ord)
            } else if !missing_last || dense {
                value_count - 1
            } else {
                i64::from(i32::MAX)
            };
            (min_ord, max_ord)
        };
        if min_ord == -1 || max_ord == i64::from(i32::MAX) {
            // Missing values still compete: nothing can be skipped yet.
            return Ok(());
        }
        self.update_state(min_ord, max_ord)
    }

    /// `getValueCount`.
    fn value_count(&self) -> i64 {
        self.dict.as_ref().map_or(0, |d| d.size())
    }

    /// `CompetitiveState.update(minOrd, maxOrd)`.
    fn update_state(&mut self, min_ord: i64, max_ord: i64) -> Result<()> {
        let Some(comp) = self.competitive.as_mut() else {
            return Ok(());
        };
        let Some((terms, doc_in)) = comp.postings else {
            // `EmptyCompetitiveState`.
            comp.iter = Iter::Docs {
                docs: Vec::new(),
                next: 0,
                doc: -1,
            };
            return Ok(());
        };
        let size = (max_ord - min_ord + 1).max(0);
        if size > MAX_COMPETITIVE_TERMS {
            if !comp.dense && !comp.docs_with_field_set {
                comp.docs_with_field_set = true;
                comp.iter = comp.with_value.iter(comp.max_doc)?;
            }
            return Ok(());
        }
        if !comp.initialized {
            comp.initialized = true;
            let mut legs = Vec::with_capacity(size as usize);
            let mut doc_freqs = Vec::with_capacity(size as usize);
            if size > 0 {
                // `init`: the doc-values term of `minOrd`, found in the terms
                // index, and the next `size - 1` terms after it.
                let dict = self.dict.as_mut().ok_or(SortError::Ordinal(min_ord))?;
                let min_term = dict.seek_ord(min_ord).map_err(store_err)?.to_vec();
                let mut te = terms.iter();
                let pe = |e| crate::Error::from(e);
                if te.try_seek_ceil(&min_term).map_err(pe)? != crate::blocktree::SeekStatus::Found {
                    return Err(SortError::TermsIndex(min_term).into());
                }
                let mut ord = min_ord;
                loop {
                    let seeked = te
                        .try_seeked_term()
                        .map_err(pe)?
                        .ok_or_else(|| SortError::TermsIndex(min_term.clone()))?;
                    let cursor = terms
                        .lazy_postings_for(&seeked, doc_in, PostingsFlags::DocsOnly)
                        .map_err(pe)?;
                    legs.push((cursor, ord as i32));
                    doc_freqs.push(i64::from(seeked.stats.doc_freq));
                    if ord == max_ord {
                        break;
                    }
                    ord += 1;
                    if te.try_next_term().map_err(pe)?.is_none() {
                        return Err(SortError::TermsIndex(min_term).into());
                    }
                }
            }
            let hi = legs.len();
            let mut u = Union {
                legs,
                doc_freqs,
                cost: 0,
                lo: 0,
                hi,
                heap: Vec::with_capacity(hi),
                doc: -1,
            };
            u.rebuild();
            comp.iter = Iter::Union(Box::new(u));
            return Ok(());
        }
        if let Iter::Union(u) = &mut comp.iter {
            if (size as usize) < u.hi - u.lo {
                // One or more ordinals left the range.
                while u.lo < u.hi && i64::from(u.legs[u.lo].1) < min_ord {
                    u.lo += 1;
                }
                while u.lo < u.hi && i64::from(u.legs[u.hi - 1].1) > max_ord {
                    u.hi -= 1;
                }
                u.rebuild();
            }
        }
        Ok(())
    }
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

fn store_err(e: lucene_store::Error) -> crate::Error {
    crate::Error::from(lucene_codecs::doc_values::Error::from(e))
}

/// A keyword key's column and terms dictionary in one segment: a
/// `SORTED_SET` column (single-valued or through the selector) or a `SORTED`
/// one; none at all sorts every document as missing (`DocValues.emptySorted`).
fn open_str<'a>(
    reader: &'a SegmentReader,
    seg: &OpenSegment<'a>,
    c: &Comparator,
) -> Result<LeafStr<'a>> {
    let f = &c.field;
    use lucene_codecs::doc_values::SortedSetKind;
    use lucene_codecs::terms_dict::TermsDict;
    let info = reader
        .field_infos()
        .fields
        .iter()
        .find(|i| i.name == f.field);
    let dv = info.and_then(|i| {
        reader
            .doc_values_for_field(i.number)
            .map(|dv| (i.number, dv))
    });
    let (column, dict, with_value) = match dv {
        None => (OrdColumn::Absent, None, WithValue::None),
        Some((num, (meta, data))) => {
            if let Some(e) = meta.sorted_set_entry(num) {
                match &e.kind {
                    SortedSetKind::Single(se) => (
                        OrdColumn::Single(NumericReader::new(data, &se.ords)),
                        Some(TermsDict::open(data, &se.terms).map_err(store_err)?),
                        WithValue::of(data, &se.ords),
                    ),
                    SortedSetKind::Multi { ords, terms } => (
                        OrdColumn::Multi(
                            SortedNumericReader::new(data, ords),
                            Vec::new(),
                            f.selector,
                        ),
                        Some(TermsDict::open(data, terms).map_err(store_err)?),
                        WithValue::of(data, &ords.numeric),
                    ),
                }
            } else if let Some(se) = meta.sorted_entry(num) {
                (
                    OrdColumn::Single(NumericReader::new(data, &se.ords)),
                    Some(TermsDict::open(data, &se.terms).map_err(store_err)?),
                    WithValue::of(data, &se.ords),
                )
            } else if meta.numeric_entry(num).is_some()
                || meta.sorted_numeric_entry(num).is_some()
                || meta.binary_entry(num).is_some()
            {
                return Err(SortError::KeywordType(f.field.clone()).into());
            } else {
                (OrdColumn::Absent, None, WithValue::None)
            }
        }
    };
    let missing_ord = if f.missing != 0 { i32::MAX } else { -1 };
    // `canSkipDocuments`, then the competitive state the field allows.
    let max_doc = reader.max_doc;
    let state = |postings, dense| StrCompetitive {
        postings,
        dense,
        with_value,
        docs_with_field_set: false,
        max_doc,
        initialized: false,
        iter: Iter::All { max_doc, doc: -1 },
    };
    let missing_last = f.missing != 0;
    let top_set = c.strs.as_ref().is_some_and(|st| st.top.is_some());
    let should_skip = |dense: bool| dense || top_set || f.reverse != missing_last;
    let competitive = match info {
        _ if c.pruning == Pruning::None => None,
        // `EmptyCompetitiveState`: a segment without the field.
        None => Some(state(None, false)),
        Some(i) if i.index_options != IndexOptions::None => {
            match (seg.fields.field(&f.field), seg.doc_in) {
                (Some(terms), Some(doc_in)) => {
                    let dense = terms.doc_count == max_doc;
                    should_skip(dense).then(|| state(Some((terms, doc_in)), dense))
                }
                // Indexed, but no terms here: nothing to build a
                // disjunction from, so nothing is skipped.
                _ => None,
            }
        }
        // A doc-values skip index is not used (no
        // `SkipperBasedCompetitiveState`): the column is scanned.
        Some(_) => None,
    };
    Ok(LeafStr {
        column,
        dict,
        bottom_same_reader: false,
        bottom_ord: 0,
        top_same_reader: true,
        top_ord: missing_ord,
        missing_ord,
        cached: (-1, 0),
        competitive,
    })
}

/// The segment's column and points for a numeric key.
fn open_leaf<'a>(
    tf: &mut TopField,
    reader: &'a SegmentReader,
    seg: &OpenSegment<'a>,
    cost: i64,
) -> Result<Leaf<'a>> {
    let points = seg.points;
    let doc_base = seg.doc_base;
    let max_doc = reader.max_doc;
    let mut keys = Vec::with_capacity(tf.comps.len());
    for c in &tf.comps {
        let f = &c.field;
        keys.push(match f.ty {
            SortType::Score => LeafKey::Score,
            SortType::Doc => LeafKey::Doc,
            SortType::String => LeafKey::Str(Box::new(open_str(reader, seg, c)?)),
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
                LeafKey::Numeric(Box::new(LeafNumeric {
                    column,
                    int: f.ty == SortType::Int,
                    missing: f.missing,
                    cached: (-1, 0),
                    competitive,
                }))
            }
        });
    }
    // `getLeafComparator` for a keyword key: a new reader generation, the
    // search-after term and the bottom looked up in this segment.
    for (i, key) in keys.iter_mut().enumerate() {
        let c = &mut tf.comps[i];
        if let (LeafKey::Str(k), Some(st)) = (key, c.strs.as_mut()) {
            st.current_gen += 1;
            // Every filled slot is now from an earlier segment.
            st.old_slots = st.reader_gen.iter().filter(|&&g| g != UNUSED_GEN).count();
            match st.top.clone() {
                Some(top) => {
                    let o = k.lookup_term(&top)?;
                    if o >= 0 {
                        k.top_same_reader = true;
                        k.top_ord = o;
                    } else {
                        k.top_same_reader = false;
                        k.top_ord = -o - 2;
                    }
                }
                None => {
                    k.top_same_reader = true;
                    k.top_ord = k.missing_ord;
                }
            }
            if let Some(b) = st.bottom_slot {
                k.set_bottom(st, b)?;
            }
            k.update_competitive(c)?;
        }
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

impl<'a> Leaf<'a> {
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
            // A keyword key compares by ordinal in `compare_*`, never here.
            LeafKey::Str(_) => 0,
        })
    }

    #[inline]
    fn compare_bottom(&mut self, tf: &TopField, doc: i32, scorer: &mut Sc<'_>) -> Result<i32> {
        self.compare_bottom_from(tf, 0, doc, scorer)
    }

    /// [`Self::compare_bottom`] over the keys from `first` on.
    fn compare_bottom_from(
        &mut self,
        tf: &TopField,
        first: usize,
        doc: i32,
        scorer: &mut Sc<'_>,
    ) -> Result<i32> {
        for (i, c) in tf.comps.iter().enumerate().skip(first) {
            let r = match &mut self.keys[i] {
                LeafKey::Str(k) => c.mul * k.compare_bottom(doc)?,
                _ => c.mul * cmp(c.bottom, self.value(i, doc, scorer)?),
            };
            if r != 0 {
                return Ok(r);
            }
        }
        Ok(0)
    }

    fn compare_top(&mut self, tf: &TopField, doc: i32, scorer: &mut Sc<'_>) -> Result<i32> {
        for (i, c) in tf.comps.iter().enumerate() {
            let r = match &mut self.keys[i] {
                LeafKey::Str(k) => c.mul * k.compare_top(doc)?,
                _ => c.mul * cmp(c.top, self.value(i, doc, scorer)?),
            };
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
            if let (LeafKey::Str(k), Some(st)) = (&mut self.keys[i], tf.comps[i].strs.as_mut()) {
                k.copy(st, slot, doc)?;
                continue;
            }
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
            if let (LeafKey::Str(k), Some(st)) = (&mut self.keys[i], c.strs.as_mut()) {
                k.set_bottom(st, slot)?;
                k.update_competitive(c)?;
                continue;
            }
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
        match &mut self.keys[0] {
            LeafKey::Numeric(n) => {
                c.hits_threshold_reached = true;
                if let Some(comp) = n.competitive.as_mut() {
                    comp.update(c)?;
                }
            }
            LeafKey::Str(k) => {
                c.hits_threshold_reached = true;
                k.update_competitive(c)?;
            }
            _ => {}
        }
        Ok(())
    }

    #[inline]
    fn competitive(&mut self) -> Option<&mut Iter<'a>> {
        match self.keys.first_mut() {
            Some(LeafKey::Numeric(n)) => n.competitive.as_mut().map(|c| &mut c.iter),
            Some(LeafKey::Str(k)) => k.competitive.as_mut().map(|c| &mut c.iter),
            _ => None,
        }
    }

    /// The whole of [`Self::collect`] for a document the leading key alone
    /// rules out, when nothing else would happen on the way: the queue is
    /// full, there is no search-after page, the count no longer changes
    /// state, and no score bound is kept. `true` when `doc` was counted and
    /// dropped; `false` means [`Self::collect`] must look at it.
    #[inline]
    fn quick_reject(&mut self, tf: &mut TopField, doc: i32, scorer: &mut Sc<'_>) -> bool {
        if !tf.queue_full
            || tf.after.is_some()
            || tf.can_set_min_score
            || tf.doc_first
            || !(tf.exhaustive || tf.relation == TotalHitsRelation::GreaterThanOrEqualTo)
        {
            return false;
        }
        let r = match self.keys.first_mut() {
            Some(LeafKey::Str(k)) => k.quick_compare_bottom(doc),
            Some(LeafKey::Numeric(n)) => n.quick_value(doc).map(|v| cmp(tf.comps[0].bottom, v)),
            _ => None,
        };
        // `thresholdCheck` drops a document that does not beat the bottom;
        // a tie on the first key goes to the others (a score read here is
        // kept for `collect`, and an error is left for it to raise).
        let drop = match r.map(|r| tf.comps[0].mul * r) {
            Some(r) if r < 0 || (r == 0 && tf.comps.len() == 1) => true,
            // The common tie-break, the score, read and kept directly.
            Some(0) if tf.comps.len() == 2 && matches!(self.keys[1], LeafKey::Score) => {
                match scorer.as_deref_mut().map(|s| s.score()) {
                    Some(Ok(score)) => {
                        self.score = score;
                        self.score_doc = doc;
                        let c = &tf.comps[1];
                        c.mul * cmp(c.bottom, score_value(score)) <= 0
                    }
                    _ => false,
                }
            }
            Some(0) => matches!(self.compare_bottom_from(tf, 1, doc, scorer), Ok(r) if r <= 0),
            _ => false,
        };
        if drop {
            tf.total_hits += 1;
        }
        drop
    }

    /// The end of the segment: keyword slots read their terms.
    fn finish(&mut self, tf: &mut TopField) -> Result<()> {
        for (key, c) in self.keys.iter_mut().zip(tf.comps.iter_mut()) {
            if let (LeafKey::Str(k), Some(st)) = (key, c.strs.as_mut()) {
                k.materialize(st)?;
            }
        }
        Ok(())
    }

    /// Whether the competitive iterator, if any, still lets every document
    /// through.
    #[inline]
    fn visits_all(&mut self) -> bool {
        self.competitive()
            .is_none_or(|it| matches!(it, Iter::All { .. }))
    }

    /// `countHit`.
    #[inline]
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
    #[inline]
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
    #[inline]
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
            let mut leaf = open_leaf(&mut tf, reader, seg, i64::from(reader.max_doc))?;
            let mut c = BulkLeaf {
                tf: &mut tf,
                leaf: &mut leaf,
                error: None,
            };
            exec::score_segment(&mut bulk, mode, seg.live_docs, &mut c)?;
            if let Some(e) = c.error {
                return Err(e);
            }
            leaf.finish(&mut tf)?;
            continue;
        }
        // The score is not the first key, so it only breaks ties: the
        // documents are iterated without scores (the cheaper tree, and the
        // one the query cache serves), and a scoring tree of the same query
        // is advanced only to the documents whose score a comparison reads.
        // A lone term is the exception: its scoring cursor iterates as fast
        // as a bare one, and scores where it stands, as Lucene's one scorer
        // does -- no second walk over the same postings.
        let self_scores = tf.needs_scores && matches!(clause, Clause::Term(_));
        let iter_mode = if self_scores { mode } else { Mode::NoScores };
        let Some(child) = exec::build::child(&ctx, &clause, 1.0, iter_mode, true)? else {
            continue;
        };
        let mut scorer = child.into_scorer(iter_mode);
        let mut scores = if tf.needs_scores && !self_scores {
            // The same query, so the same segment answer: a tree here and
            // none there would be a bug, reported rather than scored as 0.
            let Some(c) = exec::build::child(&ctx, &clause, 1.0, mode, true)? else {
                return Err(SortError::ScoringTree(-1).into());
            };
            Some(ScoreAt {
                inner: c.into_scorer(mode),
                doc: -1,
            })
        } else {
            None
        };
        let mut leaf = open_leaf(&mut tf, reader, seg, scorer.cost())?;
        score_competitive(
            &mut *scorer,
            scores.as_mut(),
            self_scores,
            &mut tf,
            &mut leaf,
            seg.live_docs,
        )?;
        leaf.finish(&mut tf)?;
    }
    Ok(tf.top_docs())
}

/// `DefaultBulkScorer.score` for a collector that may have a competitive
/// iterator (`scoreCompetitiveIterator`, `scoreTwoPhaseOrCompetitiveIterator`):
/// the scorer is advanced past documents the iterator has ruled out.
fn score_competitive(
    scorer: &mut dyn Scorer,
    mut scores: Option<&mut ScoreAt<'_>>,
    self_scores: bool,
    tf: &mut TopField,
    leaf: &mut Leaf<'_>,
    live_docs: Option<&FixedBitSet>,
) -> Result<()> {
    let two_phase = scorer.two_phase();
    // Membership instead of leapfrog: when the scorer can say whether a
    // document matches without moving (a cached bit set, match-all), a
    // narrowed competitive set is walked on its own and each of its documents
    // tested -- one move per candidate instead of two. Scores, when a key
    // reads them, come from the separate scoring tree either way.
    // (Scores read off the iterating scorer need it on each collected
    // document: leapfrog only.)
    let mut by_membership = !two_phase && !self_scores && scorer.contains(0).is_some();
    let lead_cost = scorer.cost();
    // Runs are asked for until this many documents in a row start none: an
    // iterator without runs then stops paying for the question.
    const RUN_PATIENCE: u32 = 64;
    let mut run_misses = 0u32;
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
                    it.advance(doc)?
                } else {
                    it.doc_id()
                };
                if d == NO_MORE_DOCS {
                    return Ok(());
                }
                match scorer.contains(d) {
                    Some(true) if live_docs.is_none_or(|l| l.get_doc(d)) => {
                        leaf.collect(tf, d, score_at(&mut scores, d))?;
                        if leaf.terminated {
                            return Ok(());
                        }
                    }
                    Some(_) => {}
                    // The scorer stopped answering membership: leapfrog
                    // from here, with `d` still to be looked at.
                    None => {
                        by_membership = false;
                        doc = scorer.advance(d)?;
                        continue;
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
                let next = it.advance(doc)?;
                if next != doc {
                    doc = scorer.advance(next)?;
                    continue;
                }
            }
        }
        // A run of matches (match-all): walked here, one document at a
        // time, without moving the scorer, while nothing narrows the
        // documents to visit.
        // (Asked only then: a postings list computes its run end per call.)
        let run_end =
            if !two_phase && !self_scores && run_misses < RUN_PATIENCE && leaf.visits_all() {
                scorer.doc_id_run_end()
            } else {
                doc
            };
        if run_end <= doc.saturating_add(1) {
            run_misses = run_misses.saturating_add(1);
        } else {
            run_misses = 0;
            let mut d = doc;
            while d < run_end {
                if !live_docs.is_none_or(|l| l.get_doc(d)) {
                    d += 1;
                    continue;
                }
                let mut sc = score_at(&mut scores, d);
                if !leaf.quick_reject(tf, d, &mut sc) {
                    leaf.collect(tf, d, sc)?;
                    if leaf.terminated {
                        return Ok(());
                    }
                    if leaf.collected_all_competitive {
                        if d > doc {
                            scorer.advance(d)?;
                        }
                        return count_rest(scorer, tf, leaf, live_docs);
                    }
                    if !leaf.visits_all() {
                        d += 1;
                        break;
                    }
                }
                d += 1;
            }
            doc = if d == NO_MORE_DOCS {
                NO_MORE_DOCS
            } else if d > doc {
                scorer.advance(d)?
            } else {
                scorer.next_doc()?
            };
            continue;
        }
        if live_docs.is_none_or(|l| l.get_doc(doc)) && (!two_phase || scorer.matches()?) {
            let mut sc: Sc<'_> = if self_scores {
                Some(&mut *scorer)
            } else {
                score_at(&mut scores, doc)
            };
            if leaf.quick_reject(tf, doc, &mut sc) {
                doc = step(scorer, leaf, doc, lead_cost)?;
                continue;
            }
            leaf.collect(tf, doc, sc)?;
            if leaf.terminated {
                return Ok(());
            }
            if leaf.collected_all_competitive {
                return count_rest(scorer, tf, leaf, live_docs);
            }
        }
        doc = step(scorer, leaf, doc, lead_cost)?;
    }
    Ok(())
}

/// The next candidate after `doc`: the scorer's next document, or -- when
/// the competitive iterator is the sparser of the two -- the scorer advanced
/// to the competitive iterator's next. Either way the loop then checks the
/// other side, so the same documents are collected in the same order.
#[inline]
fn step(scorer: &mut dyn Scorer, leaf: &mut Leaf<'_>, doc: i32, lead_cost: i64) -> Result<i32> {
    if let Some(it) = leaf.competitive() {
        if it.cost() < lead_cost && doc < NO_MORE_DOCS {
            let next = it.advance(doc.saturating_add(1))?;
            return if next == NO_MORE_DOCS {
                Ok(NO_MORE_DOCS)
            } else {
                scorer.advance(next)
            };
        }
    }
    scorer.next_doc()
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
        if self.inner.doc_id() != self.doc {
            return Err(SortError::ScoringTree(self.doc).into());
        }
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
        run_t(reader, q, sort, top_n, u64::MAX, after)
    }

    fn run_t(
        reader: &DirectoryReader,
        q: &BooleanQuery,
        sort: &[SortField],
        top_n: usize,
        threshold: u64,
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
            threshold,
            after,
        )
    }

    fn keyword(field: &str, reverse: bool, missing_last: bool) -> SortField {
        SortField {
            missing: i64::from(missing_last),
            ..SortField::string(field, reverse)
        }
    }

    #[test]
    fn skipping_by_term_never_changes_the_keyword_hits() {
        // Every competitive-state form: postings (sparse `k`, multi-valued
        // `km`, dense `kd`), an indexed field without doc values (`id`: no
        // terms to compete, so nothing does), a field no segment has
        // (`EmptyCompetitiveState`), and the missing-first/last cases that
        // keep everything.
        let reader = fixture("keyword_sort_index");
        let mut sorts = Vec::new();
        for field in ["k", "km", "kd", "id", "nosuch"] {
            for reverse in [false, true] {
                for last in [false, true] {
                    sorts.push(vec![keyword(field, reverse, last)]);
                    sorts.push(vec![keyword(field, reverse, last), SortField::doc()]);
                }
            }
        }
        sorts.push(vec![SortField {
            selector: Selector::Max,
            ..keyword("km", true, true)
        }]);
        let mut checked = 0;
        for sort in &sorts {
            for top_n in [1, 7] {
                let exact = run(&reader, &all(), sort, top_n, None).unwrap();
                let pruned = run_t(&reader, &all(), sort, top_n, 0, None).unwrap();
                assert_eq!(pruned.hits, exact.hits, "{sort:?} top {top_n}");
                let Some(last) = exact.hits.last() else {
                    continue;
                };
                let exact2 = run(&reader, &all(), sort, top_n, Some(last)).unwrap();
                let pruned2 = run_t(&reader, &all(), sort, top_n, 0, Some(last)).unwrap();
                assert_eq!(pruned2.hits, exact2.hits, "{sort:?} top {top_n} after");
                checked += 1;
            }
        }
        assert_eq!(checked, sorts.len() * 2);
        // A field no segment has sorts every document as missing: the first
        // documents, in order, and a search-after term looked up nowhere.
        let absent = [keyword("nosuch", false, true)];
        let got = run_t(&reader, &all(), &absent, 3, 0, None).unwrap();
        let docs: Vec<i32> = got.hits.iter().map(|h| h.doc).collect();
        assert!(
            docs.len() == 3 && docs.windows(2).all(|w| w[0] < w[1]),
            "{docs:?}"
        );
        assert!(got.hits.iter().all(|h| h.terms == [None]));
        assert_eq!(got.total.relation, TotalHitsRelation::GreaterThanOrEqualTo);
        let after = FieldDoc {
            doc: 5,
            values: vec![0],
            terms: vec![Some(b"m".to_vec())],
        };
        // Missing sorts last, so every document is after the term.
        let got = run_t(&reader, &all(), &absent, 2, 0, Some(&after)).unwrap();
        assert_eq!(
            got.hits.iter().map(|h| h.doc).collect::<Vec<_>>(),
            docs[..2]
        );
    }

    #[test]
    fn a_keyword_key_needs_a_sorted_column_and_a_numeric_key_a_numeric_one() {
        let reader = fixture("keyword_sort_index");
        let err = run(&reader, &all(), &[keyword("i", false, true)], 3, None).unwrap_err();
        assert!(
            matches!(err, crate::Error::Sort(SortError::KeywordType(ref f)) if f == "i"),
            "{err}"
        );
        let err = run(
            &reader,
            &all(),
            &[SortField::numeric("k", SortType::Long, false)],
            3,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, crate::Error::Sort(SortError::DocValuesType(ref f)) if f == "k"),
            "{err}"
        );
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
            terms: Vec::new(),
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
            assert_eq!(it.advance(0).unwrap(), 3);
            assert_eq!(it.advance(4).unwrap(), 9);
            assert_eq!(it.advance(9).unwrap(), 9);
            assert_eq!(it.advance(21).unwrap(), NO_MORE_DOCS);
        }
        let mut all = Iter::All {
            max_doc: 5,
            doc: -1,
        };
        assert_eq!(all.advance(2).unwrap(), 2);
        assert_eq!(all.advance(5).unwrap(), NO_MORE_DOCS);
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
            corrupt: None,
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

        // In bulk: below the floor dropped, the list kept until a run would
        // pass the upgrade size, then set directly.
        let mut v = CompetitiveVisitor {
            min: 0,
            max: 0,
            max_doc_visited: 4,
            docs: Vec::new(),
            bits: None,
            upgrade_at: 4,
            added: 0,
            spare: Some(FixedBitSet::new(64)),
            max_doc: 64,
            corrupt: None,
        };
        v.visit_many(&[3, 5, 6]);
        assert_eq!((v.docs.as_slice(), v.added), (&[5, 6][..], 2));
        v.visit_many(&[2, 8, 9]);
        let bits = v.bits.as_ref().expect("upgraded");
        assert!(v.spare.is_none(), "the spare set was used");
        assert_eq!(v.added, 4);
        assert_eq!(bits.cardinality(), 4);

        // A leaf naming a document outside the segment is remembered, not
        // indexed: singly or in bulk.
        v.visit(64);
        assert_eq!(v.corrupt, Some(64));
        let mut w = CompetitiveVisitor {
            min: 0,
            max: 0,
            max_doc_visited: -1,
            docs: Vec::new(),
            bits: None,
            upgrade_at: 100,
            added: 0,
            spare: None,
            max_doc: 8,
            corrupt: None,
        };
        w.visit_many(&[1, -3, 2]);
        assert_eq!((w.corrupt, w.added), (Some(-3), 0));
    }

    fn comparator(reverse: bool, pruning: Pruning) -> Comparator {
        let mut f = SortField::numeric("l", SortType::Long, reverse);
        f.missing = 7;
        Comparator {
            field: f,
            mul: if reverse { -1 } else { 1 },
            values: vec![0; 4],
            bottom: 10,
            top: 3,
            top_set: false,
            pruning,
            single_sort: true,
            hits_threshold_reached: false,
            queue_full: false,
            strs: None,
        }
    }

    fn competitive<'a>(
        points: &'a PointsReader<'a>,
        field_number: i32,
        max_doc: i32,
    ) -> Competitive<'a> {
        Competitive {
            points,
            field_number,
            bytes: 8,
            point_doc_count: max_doc,
            max_doc,
            leaf_top_set: false,
            iter: Iter::All { max_doc, doc: -1 },
            min_value: i64::MIN,
            max_value: i64::MAX,
            max_doc_visited: -1,
            update_counter: 0,
            current_skip_interval: MIN_SKIP_INTERVAL,
            iterator_cost: i64::from(max_doc),
            try_update_fail_count: 0,
            with_value: WithValue::None,
            scratch: Vec::new(),
            spare_bits: None,
            walk: PointsScratch::default(),
        }
    }

    #[test]
    fn bounds_follow_the_bottom_the_top_and_the_pruning() {
        let points = PointsReader::empty();
        // Ascending, `>=` pruning: the bottom itself cannot compete.
        let mut c = comparator(false, Pruning::GreaterThanOrEqualTo);
        let mut comp = competitive(&points, 0, 100);
        comp.encode_bottom(&c);
        assert_eq!((comp.min_value, comp.max_value), (i64::MIN, 9));
        // Descending with `>` pruning: the bottom still can (a later key).
        let d = comparator(true, Pruning::GreaterThan);
        comp.encode_bottom(&d);
        assert_eq!(comp.min_value, 10);
        // encodeTop tightens only for a single sort with a full queue.
        c.queue_full = true;
        comp.encode_top(&c);
        assert_eq!(comp.min_value, 4);
        let mut d = comparator(true, Pruning::GreaterThanOrEqualTo);
        d.queue_full = true;
        comp.encode_top(&d);
        assert_eq!(comp.max_value, 2);
        d.single_sort = false;
        comp.encode_top(&d);
        assert_eq!(comp.max_value, 3);

        // isMissingValueCompetitive: missing 7 against bottom 10 and top 3.
        let mut asc = comparator(false, Pruning::GreaterThanOrEqualTo);
        asc.queue_full = true;
        assert!(comp.missing_competitive(&asc), "7 < 10 ascending");
        asc.bottom = 7;
        assert!(!comp.missing_competitive(&asc), "a tie loses under >=");
        asc.pruning = Pruning::GreaterThan;
        assert!(comp.missing_competitive(&asc), "a tie can win under >");
        let mut desc = comparator(true, Pruning::GreaterThanOrEqualTo);
        desc.queue_full = true;
        assert!(!comp.missing_competitive(&desc), "7 < 10 descending");
        desc.pruning = Pruning::GreaterThan;
        desc.bottom = 7;
        assert!(comp.missing_competitive(&desc));
        // With a top value: ascending needs missing >= top, descending <= top.
        comp.leaf_top_set = true;
        asc.bottom = 20;
        assert!(comp.missing_competitive(&asc));
        desc.bottom = 1;
        assert!(!comp.missing_competitive(&desc));
        asc.queue_full = false;
        asc.top = 8;
        assert!(!comp.missing_competitive(&asc));
    }

    #[test]
    fn an_update_waits_for_the_threshold_and_a_full_queue_then_falls_back_to_values() {
        let reader = fixture("sorted_search_index");
        let mut opened = reader.open_segments().expect("open");
        opened.open_points().expect("points");
        let segments = opened.as_open_segments();
        let p = segments[0].points.expect("points opened");
        let num = p.field_number("l").expect("l has points");
        let max_doc = reader.segment_readers()[0].max_doc;
        let mut c = comparator(false, Pruning::GreaterThanOrEqualTo);
        c.bottom = i64::MAX - 1;
        let mut comp = competitive(&p.reader, num, max_doc);
        comp.point_doc_count = max_doc; // no missing documents
        comp.update(&c).unwrap();
        assert_eq!(comp.update_counter, 0, "before the threshold, nothing");
        c.hits_threshold_reached = true;
        comp.update(&c).unwrap();
        assert_eq!(comp.update_counter, 0, "no top and no full queue, nothing");
        c.queue_full = true;
        // A range this wide cannot narrow 8-fold: with fewer points than the
        // iterator costs, the documents with values drive it instead.
        comp.point_doc_count = max_doc - 1;
        comp.iterator_cost = i64::from(max_doc) + 1;
        c.field.missing = i64::MAX; // missing sorts last: not competitive
        comp.with_value = WithValue::All;
        comp.update(&c).unwrap();
        assert!(matches!(comp.iter, Iter::All { .. }));
        assert_eq!(comp.iterator_cost, i64::from(max_doc - 1));
        comp.iterator_cost = i64::from(max_doc) + 1;
        comp.with_value = WithValue::None;
        comp.update(&c).unwrap();
        assert!(matches!(&comp.iter, Iter::Docs { docs, .. } if docs.is_empty()));
        // A missing value that could still compete stops the update.
        c.field.missing = i64::MIN;
        let before = comp.update_counter;
        comp.update(&c).unwrap();
        assert_eq!(comp.update_counter, before);
    }

    #[test]
    fn docs_with_values_come_from_the_entry() {
        let reader = fixture("sorted_search_index");
        let r = &reader.segment_readers()[0];
        let info = r
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == "l")
            .unwrap();
        let (meta, data) = r.doc_values_for_field(info.number).unwrap();
        let entry = meta
            .sorted_numeric_entry(info.number)
            .unwrap()
            .numeric
            .clone();
        assert!(
            matches!(WithValue::of(data, &entry), WithValue::Disi { .. }),
            "l is sparse"
        );
        let mut e = entry.clone();
        e.docs_with_field_offset = -1;
        assert!(matches!(WithValue::of(data, &e), WithValue::All));
        e.docs_with_field_offset = -2;
        assert!(matches!(WithValue::of(data, &e), WithValue::None));
        e.docs_with_field_offset = data.len() as i64;
        e.docs_with_field_length = 10;
        assert!(
            matches!(WithValue::of(data, &e), WithValue::None),
            "past the file"
        );
    }

    #[test]
    fn the_scoring_tree_passes_through_and_reports_a_document_it_lacks() {
        let reader = fixture("sorted_search_index");
        let opened = reader.open_segments().expect("open");
        let segments = opened.as_open_segments();
        let seg = &segments[0];
        let ctx = exec::LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: seg.pos_in,
            pay_in: seg.pay_in,
            live_docs: None,
            points: None,
            norms: None,
            global: None,
            max_doc: seg.max_doc,
            cache: None,
        };
        let all = Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(0));
        let tree = exec::build::child(&ctx, &all, 1.0, Mode::Complete, true)
            .unwrap()
            .unwrap()
            .into_scorer(Mode::Complete);
        let mut s = ScoreAt {
            inner: tree,
            doc: 5,
        };
        assert_eq!(s.cost(), i64::from(seg.max_doc.unwrap()));
        assert_eq!(s.next_doc().unwrap(), 0);
        assert_eq!(s.doc_id(), 0);
        assert_eq!(s.score().unwrap(), 1.0, "moved to document 5 first");
        assert_eq!(s.doc_id(), 5);
        assert_eq!(s.advance(7).unwrap(), 7);
        assert!(s.max_score(10).unwrap() >= 1.0);
        s.doc = seg.max_doc.unwrap() + 3;
        assert!(matches!(
            s.score(),
            Err(crate::Error::Sort(SortError::ScoringTree(_)))
        ));
    }

    #[test]
    fn counting_the_rest_stops_just_past_the_threshold() {
        let reader = fixture("sorted_search_index");
        let opened = reader.open_segments().expect("open");
        let segments = opened.as_open_segments();
        let seg = &segments[1];
        let ctx = exec::LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: seg.pos_in,
            pay_in: seg.pay_in,
            live_docs: None,
            points: None,
            norms: None,
            global: None,
            max_doc: seg.max_doc,
            cache: None,
        };
        let all = Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(0));
        let scorer = || {
            exec::build::child(&ctx, &all, 1.0, Mode::NoScores, true)
                .unwrap()
                .unwrap()
                .into_scorer(Mode::NoScores)
        };
        let leaf = || Leaf {
            doc_base: 0,
            keys: vec![LeafKey::Doc],
            collected_all_competitive: true,
            after_doc: 0,
            terminated: false,
            score: 0.0,
            score_doc: -1,
        };
        let max_doc = u64::try_from(seg.max_doc.unwrap()).unwrap();
        // A run: counted at once, to one past the threshold.
        let mut tf = TopField::new(&[SortField::doc()], 5, 100, None);
        let mut l = leaf();
        count_rest(&mut *scorer(), &mut tf, &mut l, None).unwrap();
        assert_eq!(tf.total_hits, 101);
        assert!(l.terminated);
        assert_eq!(tf.relation, TotalHitsRelation::GreaterThanOrEqualTo);
        // Exhaustive: every document but the one already on (next_doc skips it).
        let mut tf = TopField::new(&[SortField::doc()], 5, u64::MAX, None);
        let mut l = leaf();
        count_rest(&mut *scorer(), &mut tf, &mut l, None).unwrap();
        assert_eq!(tf.total_hits, max_doc);
        assert!(!l.terminated);
        // With live documents, one at a time.
        let mut live = FixedBitSet::new(max_doc as usize);
        for d in (0..max_doc as usize).step_by(2) {
            live.set(d);
        }
        let mut tf = TopField::new(&[SortField::doc()], 5, 3, None);
        let mut l = leaf();
        count_rest(&mut *scorer(), &mut tf, &mut l, Some(&live)).unwrap();
        assert_eq!(tf.total_hits, 6, "threshold max(3, 5 hits) + 1");
        assert!(l.terminated);
    }

    #[test]
    fn the_bulk_collector_keeps_its_first_error_and_asks_for_no_scores_by_document() {
        let mut tf = TopField::new(&[SortField::doc()], 2, 10, None);
        let mut l = Leaf {
            doc_base: 0,
            keys: vec![LeafKey::Doc],
            collected_all_competitive: false,
            after_doc: 0,
            terminated: false,
            score: 0.0,
            score_doc: -1,
        };
        let mut c = BulkLeaf {
            tf: &mut tf,
            leaf: &mut l,
            error: Some(SortError::NoKeys.into()),
        };
        assert_eq!(c.score_mode(), ScoreMode::CompleteNoScores);
        assert_eq!(c.min_competitive_score(), None);
        c.collect(1, 0.5);
        assert_eq!(c.tf.total_hits, 0, "nothing collected past an error");
        let mut tf = TopField::new(&[SortField::doc(), SortField::score()], 2, 10, None);
        let mut l = Leaf {
            doc_base: 0,
            keys: vec![LeafKey::Doc, LeafKey::Score],
            collected_all_competitive: false,
            after_doc: 0,
            terminated: false,
            score: 0.0,
            score_doc: -1,
        };
        let c = BulkLeaf {
            tf: &mut tf,
            leaf: &mut l,
            error: None,
        };
        assert_eq!(c.score_mode(), ScoreMode::Complete);
    }

    #[test]
    fn the_empty_queue_pops_nothing() {
        let mut q = HitQueue { heap: Vec::new() };
        assert!(q.top().is_none());
        assert!(q.pop(&[]).is_none());
    }
}
