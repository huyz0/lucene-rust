//! The doc-values queries of the document package:
//! `SortedNumericDocValuesRangeQuery` and `SortedNumericDocValuesSetQuery`
//! (with its `DocValuesLongHashSet`) over `NUMERIC`/`SORTED_NUMERIC` values,
//! `SortedSetDocValuesRangeQuery` and `TermInSetQuery`'s doc-values rewrite
//! over `SORTED`/`SORTED_SET` ordinals, the index-sort fast path both range
//! queries take (`SortedSkipperScorerSupplier` + `RangeBulkScorer`), and
//! `KeywordField`'s term queries -- with the factories of
//! `NumericDocValuesField`, `SortedNumericDocValuesField`,
//! `SortedDocValuesField`, `SortedSetDocValuesField` and `KeywordField`.
//!
//! A `NUMERIC` field reads as a single-valued `SORTED_NUMERIC` one and a
//! `SORTED` field as a single-valued `SORTED_SET` one
//! (`DocValues.getSortedNumeric`/`getSortedSet`), as in Java.
//!
//! Deliberate difference: `NumericFieldStats`' rewrite of a range the whole
//! reader's value range excludes (to `MatchNoDocsQuery`) or covers in every
//! document (to `MatchAllDocsQuery`) is not applied; each segment finds the
//! same hits itself.

use lucene_codecs::doc_values::{
    self, DocValuesMeta, DocValuesSkipIndex, DocValuesSkipper, NumericEntry, SortedEntry,
    SortedNumericEntry, SortedSetKind, NO_MORE_DOCS,
};
use lucene_codecs::field_infos::{DocValuesType, FieldInfo};
use lucene_codecs::terms_dict::{TermsDict, TermsDictEntry};
use lucene_index::segment_info::IndexSortField;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::{collect_live, field_info, reader, DocumentQuery, FieldExists, MatchNoDocs};
use crate::collector::{ScoringCollector, VecCollector};
use crate::multi_segment::OpenSegment;
use crate::query::TermQuery;
use crate::top_field::{Selector, SortField};
use crate::{search_term_query, Error, Result};

fn illegal(message: impl Into<String>) -> Error {
    Error::DocumentQuery(message.into())
}

fn store_err(e: lucene_store::Error) -> Error {
    Error::DocValues(doc_values::Error::Store(e))
}

/// `DocValues.getSortedNumeric(reader, field)`'s column: a `NUMERIC` field
/// (a singleton) or a `SORTED_NUMERIC` one.
#[derive(Debug, Clone, Copy)]
enum NumericColumn<'a> {
    Numeric(&'a NumericEntry),
    SortedNumeric(&'a SortedNumericEntry),
}

impl NumericColumn<'_> {
    /// `DocValues.unwrapSingleton` succeeds.
    fn is_singleton(&self) -> bool {
        match self {
            NumericColumn::Numeric(_) => true,
            NumericColumn::SortedNumeric(e) => e.addresses.is_none(),
        }
    }

    /// The document's values, ascending (one at most for a singleton).
    fn values(&self, data: &[u8], doc: i32, out: &mut Vec<i64>) -> Result<()> {
        out.clear();
        match self {
            NumericColumn::Numeric(e) => {
                if let Some(v) = doc_values::numeric_value(data, e, doc)? {
                    out.push(v);
                }
            }
            NumericColumn::SortedNumeric(e) => {
                out.extend(doc_values::sorted_numeric_values(data, e, doc)?);
            }
        }
        Ok(())
    }
}

/// The unexpected-type message `DocValues.checkField` throws.
fn unexpected(info: &FieldInfo, expected: &str) -> Error {
    illegal(format!(
        "unexpected docvalues type {} for field '{}' (expected one of {expected}). Re-index with \
         correct docvalues type.",
        lucene_index::document::doc_values_type_name(info.doc_values_type),
        info.name
    ))
}

/// The segment's numeric column for `field`, `None` when the field has no
/// doc values here.
fn numeric_column<'a>(
    leaf: &OpenSegment<'a>,
    info: &FieldInfo,
) -> Result<Option<(&'a [u8], NumericColumn<'a>)>> {
    let r = reader(leaf)?;
    match info.doc_values_type {
        DocValuesType::None => return Ok(None),
        DocValuesType::Numeric | DocValuesType::SortedNumeric => {}
        _ => return Err(unexpected(info, "[SORTED_NUMERIC, NUMERIC]")),
    }
    let Some((meta, data)) = r.doc_values_for_field(info.number) else {
        return Ok(None);
    };
    let col = column_of(meta, info.number);
    Ok(col.map(|c| (data, c)))
}

fn column_of(meta: &DocValuesMeta, number: i32) -> Option<NumericColumn<'_>> {
    if let Some(e) = meta.numeric_entry(number) {
        return Some(NumericColumn::Numeric(e));
    }
    meta.sorted_numeric_entry(number)
        .map(NumericColumn::SortedNumeric)
}

/// The field's skip index in this segment, if it has one.
fn skip_index(leaf: &OpenSegment<'_>, info: &FieldInfo) -> Result<Option<DocValuesSkipIndex>> {
    reader(leaf)?.doc_values_skip_index(info.number)
}

/// `densePrimarySort(reader, skipper)`: the segment's primary sort, when it is
/// on `field` and every document has a value.
fn dense_primary_sort<'a>(
    leaf: &OpenSegment<'a>,
    field: &str,
    skip: &DocValuesSkipIndex,
) -> Result<Option<&'a IndexSortField>> {
    let r = reader(leaf)?;
    if skip.doc_count != r.max_doc {
        return Ok(None);
    }
    Ok(r.index_sort()
        .and_then(|s| s.first())
        .filter(|s| s.field == field))
}

/// `SortedNumericDocValuesRangeQuery` (`NumericDocValuesField` /
/// `SortedNumericDocValuesField.newSlowRangeQuery`): documents with a value
/// in `[lower_value, upper_value]`, at a constant score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortedNumericDocValuesRangeQuery {
    pub field: String,
    pub lower_value: i64,
    pub upper_value: i64,
}

impl SortedNumericDocValuesRangeQuery {
    pub fn new(field: impl Into<String>, lower_value: i64, upper_value: i64) -> Self {
        SortedNumericDocValuesRangeQuery {
            field: field.into(),
            lower_value,
            upper_value,
        }
    }
}

impl DocumentQuery for SortedNumericDocValuesRangeQuery {
    /// `rewrite`: the whole `long` range is `FieldExistsQuery`, an empty one
    /// `MatchNoDocsQuery`.
    fn rewrite(&self, _leaves: &[OpenSegment<'_>]) -> Result<Option<Box<dyn DocumentQuery>>> {
        if self.lower_value == i64::MIN && self.upper_value == i64::MAX {
            return Ok(Some(Box::new(FieldExists {
                field: self.field.clone(),
            })));
        }
        if self.lower_value > self.upper_value {
            return Ok(Some(Box::new(MatchNoDocs)));
        }
        Ok(None)
    }

    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(info) = field_info(leaf, &self.field)? else {
            return Ok(());
        };
        let Some((data, column)) = numeric_column(leaf, info)? else {
            return Ok(());
        };
        let r = reader(leaf)?;
        let (lo, hi) = (self.lower_value, self.upper_value);
        let skip = skip_index(leaf, info)?;
        // `docCountIgnoringDeletes`.
        if let Some(s) = &skip {
            if s.min_value > hi || s.max_value < lo {
                return Ok(());
            }
            if s.doc_count == r.max_doc && s.min_value >= lo && s.max_value <= hi {
                for doc in 0..r.max_doc {
                    collect_live(leaf, doc, boost, collector);
                }
                return Ok(());
            }
        }
        if column.is_singleton() {
            if let Some(s) = &skip {
                if let Some(sort) = dense_primary_sort(leaf, &self.field, s)? {
                    let mut supplier = SortedSkipperScorerSupplier::new(s, sort.reverse, lo, hi);
                    let mut values = Vec::new();
                    let mut cursor = -1;
                    let range = supplier.range(|start, pred| {
                        advance_until(start, &mut cursor, r.max_doc, |doc| {
                            column.values(data, doc, &mut values)?;
                            Ok(values.first().is_some_and(|&v| pred(v)))
                        })
                    })?;
                    RangeBulkScorer::score_range(range, leaf.live_docs, boost, collector);
                    return Ok(());
                }
            }
        }
        // `DocValuesRangeIterator.forRange`: the skip index only decides which
        // blocks to read; the matches are the values in range.
        let mut values = Vec::new();
        for doc in 0..r.max_doc {
            column.values(data, doc, &mut values)?;
            // Values are ascending: the first one not below `lo` decides.
            if values.iter().find(|&&v| v >= lo).is_some_and(|&v| v <= hi) {
                collect_live(leaf, doc, boost, collector);
            }
        }
        Ok(())
    }
}

/// `nextDoc(startDocID, predicate)` of the skipper scorer suppliers over a
/// dense column: from `start` (or where the column already is, if that is
/// further), the first document whose value passes; the column's position
/// is kept across calls as Java's iterator's is.
pub(super) fn advance_until(
    start: i32,
    cursor: &mut i32,
    max_doc: i32,
    mut passes: impl FnMut(i32) -> Result<bool>,
) -> Result<i32> {
    if start > *cursor {
        *cursor = start;
    }
    while *cursor < max_doc {
        if passes(*cursor)? {
            return Ok(*cursor);
        }
        *cursor = cursor.saturating_add(1);
    }
    *cursor = NO_MORE_DOCS;
    Ok(NO_MORE_DOCS)
}

/// `DocValuesLongHashSet`: an open-addressing set of longs, with the sorted
/// input's minimum and maximum kept for a cheap range check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocValuesLongHashSet {
    table: Vec<i64>,
    mask: usize,
    has_missing_value: bool,
    size: usize,
    pub min_value: i64,
    pub max_value: i64,
}

/// `DocValuesLongHashSet.MISSING`: the empty-slot marker.
const MISSING: i64 = i64::MIN;

impl DocValuesLongHashSet {
    /// `DocValuesLongHashSet(values)`: `values` must be sorted.
    pub fn new(values: &[i64]) -> Self {
        debug_assert!(
            values.windows(2).all(|w| w[0] <= w[1]),
            "values must be sorted"
        );
        // `1 << PackedInts.bitsRequired(values.length * 3L / 2)`.
        let want = values.len().saturating_mul(3) / 2;
        let bits = if want == 0 {
            1
        } else {
            usize::BITS - want.leading_zeros()
        };
        let table_size = 1usize.checked_shl(bits).unwrap_or(usize::MAX);
        let mut set = DocValuesLongHashSet {
            table: vec![MISSING; table_size],
            mask: table_size.saturating_sub(1),
            has_missing_value: false,
            size: 0,
            min_value: values.first().copied().unwrap_or(i64::MAX),
            max_value: values.last().copied().unwrap_or(i64::MIN),
        };
        for &v in values {
            if v == MISSING {
                if !set.has_missing_value {
                    set.size = set.size.saturating_add(1);
                }
                set.has_missing_value = true;
            } else if set.add(v) {
                set.size = set.size.saturating_add(1);
            }
        }
        set
    }

    /// `Long.hashCode(l) & mask`.
    fn slot(&self, l: i64) -> usize {
        let h = (l ^ ((l as u64) >> 32) as i64) as i32;
        (h as u32 as usize) & self.mask
    }

    fn add(&mut self, l: i64) -> bool {
        let mut i = self.slot(l);
        loop {
            if self.table[i] == MISSING {
                self.table[i] = l;
                return true;
            } else if self.table[i] == l {
                return false;
            }
            i = i.wrapping_add(1) & self.mask;
        }
    }

    /// `contains(l)`.
    pub fn contains(&self, l: i64) -> bool {
        if l == MISSING {
            return self.has_missing_value;
        }
        let mut i = self.slot(l);
        loop {
            if self.table[i] == MISSING {
                return false;
            } else if self.table[i] == l {
                return true;
            }
            i = i.wrapping_add(1) & self.mask;
        }
    }

    /// `size()`: distinct values.
    pub fn size(&self) -> usize {
        self.size
    }

    /// `stream()`: the missing marker (if held) and then the table's values
    /// in slot order.
    pub fn values(&self) -> impl Iterator<Item = i64> + '_ {
        self.has_missing_value
            .then_some(MISSING)
            .into_iter()
            .chain(self.table.iter().copied().filter(|&v| v != MISSING))
    }
}

/// `SortedNumericDocValuesSetQuery` (`newSlowSetQuery`): documents with a
/// value in the set, at a constant score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortedNumericDocValuesSetQuery {
    pub field: String,
    pub numbers: DocValuesLongHashSet,
}

impl SortedNumericDocValuesSetQuery {
    pub fn new(field: impl Into<String>, mut numbers: Vec<i64>) -> Self {
        numbers.sort_unstable();
        SortedNumericDocValuesSetQuery {
            field: field.into(),
            numbers: DocValuesLongHashSet::new(&numbers),
        }
    }
}

impl DocumentQuery for SortedNumericDocValuesSetQuery {
    fn rewrite(&self, _leaves: &[OpenSegment<'_>]) -> Result<Option<Box<dyn DocumentQuery>>> {
        if self.numbers.size() == 0 {
            return Ok(Some(Box::new(MatchNoDocs)));
        }
        Ok(None)
    }

    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(info) = field_info(leaf, &self.field)? else {
            return Ok(());
        };
        let Some((data, column)) = numeric_column(leaf, info)? else {
            return Ok(());
        };
        let n = &self.numbers;
        let mut values = Vec::new();
        for doc in 0..reader(leaf)?.max_doc {
            column.values(data, doc, &mut values)?;
            let mut matched = false;
            for &v in &values {
                if v < n.min_value {
                    continue;
                } else if v > n.max_value {
                    break;
                } else if n.contains(v) {
                    matched = true;
                    break;
                }
            }
            if matched {
                collect_live(leaf, doc, boost, collector);
            }
        }
        Ok(())
    }
}

/// `DocValues.getSortedSet(reader, field)`'s column: a `SORTED` field (a
/// singleton) or a `SORTED_SET` one, with its terms dictionary.
#[derive(Debug, Clone, Copy)]
enum OrdsColumn<'a> {
    Sorted(&'a SortedEntry),
    Multi(&'a SortedNumericEntry, &'a TermsDictEntry),
}

impl<'a> OrdsColumn<'a> {
    fn terms(&self) -> &'a TermsDictEntry {
        match self {
            OrdsColumn::Sorted(e) => &e.terms,
            OrdsColumn::Multi(_, t) => t,
        }
    }

    /// The document's ordinals, ascending.
    fn ords(&self, data: &[u8], doc: i32, out: &mut Vec<i64>) -> Result<()> {
        out.clear();
        match self {
            OrdsColumn::Sorted(e) => {
                if let Some(o) = doc_values::sorted_ord(data, e, doc)? {
                    out.push(o);
                }
            }
            OrdsColumn::Multi(e, _) => out.extend(doc_values::sorted_numeric_values(data, e, doc)?),
        }
        Ok(())
    }
}

fn ords_column<'a>(
    leaf: &OpenSegment<'a>,
    info: &FieldInfo,
) -> Result<Option<(&'a [u8], OrdsColumn<'a>)>> {
    let r = reader(leaf)?;
    match info.doc_values_type {
        DocValuesType::None => return Ok(None),
        DocValuesType::Sorted | DocValuesType::SortedSet => {}
        _ => return Err(unexpected(info, "[SORTED, SORTED_SET]")),
    }
    let Some((meta, data)) = r.doc_values_for_field(info.number) else {
        return Ok(None);
    };
    if let Some(e) = meta.sorted_entry(info.number) {
        return Ok(Some((data, OrdsColumn::Sorted(e))));
    }
    Ok(meta.sorted_set_entry(info.number).map(|e| {
        (
            data,
            match &e.kind {
                SortedSetKind::Single(s) => OrdsColumn::Sorted(s),
                SortedSetKind::Multi { ords, terms } => OrdsColumn::Multi(ords, terms),
            },
        )
    }))
}

/// `SortedSetDocValues.lookupTerm`.
fn lookup_term(data: &[u8], terms: &TermsDictEntry, key: &[u8]) -> Result<i64> {
    let mut dict = TermsDict::open(data, terms).map_err(store_err)?;
    dict.lookup_term(key).map_err(store_err)
}

/// `SortedSetDocValuesRangeQuery` (`SortedDocValuesField` /
/// `SortedSetDocValuesField.newSlowRangeQuery`): documents with a term in
/// the range, bounds optional and inclusive or not, at a constant score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortedSetDocValuesRangeQuery {
    pub field: String,
    pub lower_value: Option<Vec<u8>>,
    pub upper_value: Option<Vec<u8>>,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
}

impl SortedSetDocValuesRangeQuery {
    /// An absent bound is never inclusive.
    pub fn new(
        field: impl Into<String>,
        lower_value: Option<Vec<u8>>,
        upper_value: Option<Vec<u8>>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Self {
        SortedSetDocValuesRangeQuery {
            field: field.into(),
            lower_inclusive: lower_inclusive && lower_value.is_some(),
            upper_inclusive: upper_inclusive && upper_value.is_some(),
            lower_value,
            upper_value,
        }
    }

    /// `minOrd(values)`.
    fn min_ord(&self, data: &[u8], terms: &TermsDictEntry) -> Result<i64> {
        let Some(lower) = &self.lower_value else {
            return Ok(0);
        };
        let ord = lookup_term(data, terms, lower)?;
        Ok(if ord < 0 {
            ord.saturating_neg().saturating_sub(1)
        } else if self.lower_inclusive {
            ord
        } else {
            ord.saturating_add(1)
        })
    }

    /// `maxOrd(values)`.
    fn max_ord(&self, data: &[u8], terms: &TermsDictEntry) -> Result<i64> {
        let Some(upper) = &self.upper_value else {
            return Ok(terms.terms_dict_size.saturating_sub(1));
        };
        let ord = lookup_term(data, terms, upper)?;
        Ok(if ord < 0 {
            (-2i64).saturating_sub(ord)
        } else if self.upper_inclusive {
            ord
        } else {
            ord.saturating_sub(1)
        })
    }
}

impl DocumentQuery for SortedSetDocValuesRangeQuery {
    /// `rewrite`: no bounds at all is `FieldExistsQuery`.
    fn rewrite(&self, _leaves: &[OpenSegment<'_>]) -> Result<Option<Box<dyn DocumentQuery>>> {
        if self.lower_value.is_none() && self.upper_value.is_none() {
            return Ok(Some(Box::new(FieldExists {
                field: self.field.clone(),
            })));
        }
        Ok(None)
    }

    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(info) = field_info(leaf, &self.field)? else {
            return Ok(());
        };
        let Some((data, column)) = ords_column(leaf, info)? else {
            return Ok(());
        };
        let r = reader(leaf)?;
        let skip = skip_index(leaf, info)?;
        let terms = column.terms();
        let min_ord = self.min_ord(data, terms)?;
        let max_ord = self.max_ord(data, terms)?;
        if let (OrdsColumn::Sorted(_), Some(s)) = (&column, &skip) {
            if let Some(sort) = dense_primary_sort(leaf, &self.field, s)? {
                let mut supplier =
                    SortedSkipperScorerSupplier::new(s, sort.reverse, min_ord, max_ord);
                let mut ords = Vec::new();
                let mut cursor = -1;
                let range = supplier.range(|start, pred| {
                    advance_until(start, &mut cursor, r.max_doc, |doc| {
                        column.ords(data, doc, &mut ords)?;
                        Ok(ords.first().is_some_and(|&o| pred(o)))
                    })
                })?;
                RangeBulkScorer::score_range(range, leaf.live_docs, boost, collector);
                return Ok(());
            }
        }
        if min_ord > max_ord
            || skip
                .as_ref()
                .is_some_and(|s| min_ord > s.max_value || max_ord < s.min_value)
        {
            return Ok(());
        }
        if skip.as_ref().is_some_and(|s| {
            s.doc_count == r.max_doc && s.min_value >= min_ord && s.max_value <= max_ord
        }) {
            for doc in 0..r.max_doc {
                collect_live(leaf, doc, boost, collector);
            }
            return Ok(());
        }
        let mut ords = Vec::new();
        for doc in 0..r.max_doc {
            column.ords(data, doc, &mut ords)?;
            if ords
                .iter()
                .find(|&&o| o >= min_ord)
                .is_some_and(|&o| o <= max_ord)
            {
                collect_live(leaf, doc, boost, collector);
            }
        }
        Ok(())
    }
}

/// `TermInSetQuery` rewritten with `MultiTermQuery.DOC_VALUES_REWRITE`
/// (`SortedDocValuesField`/`SortedSetDocValuesField.newSlowSetQuery`, and
/// `KeywordField.newSetQuery`'s doc-values side): documents whose `SORTED`/
/// `SORTED_SET` values include one of `terms`, at a constant score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocValuesTermInSetQuery {
    pub field: String,
    /// Sorted, deduplicated.
    pub terms: Vec<Vec<u8>>,
}

impl DocValuesTermInSetQuery {
    pub fn new(field: impl Into<String>, mut terms: Vec<Vec<u8>>) -> Self {
        terms.sort_unstable();
        terms.dedup();
        DocValuesTermInSetQuery {
            field: field.into(),
            terms,
        }
    }
}

impl DocumentQuery for DocValuesTermInSetQuery {
    fn rewrite(&self, _leaves: &[OpenSegment<'_>]) -> Result<Option<Box<dyn DocumentQuery>>> {
        if self.terms.is_empty() {
            return Ok(Some(Box::new(MatchNoDocs)));
        }
        Ok(None)
    }

    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(info) = field_info(leaf, &self.field)? else {
            return Ok(());
        };
        let Some((data, column)) = ords_column(leaf, info)? else {
            return Ok(());
        };
        // `MultiTermQueryDocValuesWrapper`: the ordinals of the terms the
        // dictionary holds.
        let mut wanted: Vec<i64> = Vec::new();
        for t in &self.terms {
            let ord = lookup_term(data, column.terms(), t)?;
            if ord >= 0 {
                wanted.push(ord);
            }
        }
        if wanted.is_empty() {
            return Ok(());
        }
        let mut ords = Vec::new();
        for doc in 0..reader(leaf)?.max_doc {
            column.ords(data, doc, &mut ords)?;
            if ords.iter().any(|o| wanted.binary_search(o).is_ok()) {
                collect_live(leaf, doc, boost, collector);
            }
        }
        Ok(())
    }
}

/// `SortedSkipperScorerSupplier`: over a segment sorted by the queried
/// field, with every document holding one value, the matches of a value
/// range are one run of doc ids -- found from the skip index where it is
/// exact, and by stepping the values where it is not.
#[derive(Debug)]
pub struct SortedSkipperScorerSupplier<'a> {
    skipper: DocValuesSkipper<'a>,
    index: &'a DocValuesSkipIndex,
    reverse: bool,
    lower: i64,
    upper: i64,
    skipper_min_doc_id: i32,
    skipper_max_doc_id: i32,
    skipper_min_doc_id_exact: bool,
    skipper_max_doc_id_exact: bool,
    computed: bool,
}

impl<'a> SortedSkipperScorerSupplier<'a> {
    /// Over `index`, the segment's primary sort `reverse` or not, for values
    /// in `[lower, upper]`.
    pub fn new(index: &'a DocValuesSkipIndex, reverse: bool, lower: i64, upper: i64) -> Self {
        SortedSkipperScorerSupplier {
            skipper: DocValuesSkipper::new(index),
            index,
            reverse,
            lower,
            upper,
            skipper_min_doc_id: -1,
            skipper_max_doc_id: -1,
            skipper_min_doc_id_exact: false,
            skipper_max_doc_id_exact: false,
            computed: false,
        }
    }

    /// `computeSkipperDocIds()`.
    fn compute_skipper_doc_ids(&mut self) {
        self.computed = true;
        let (min_ord, max_ord) = (self.lower, self.upper);
        let (g_min, g_max) = (self.index.min_value, self.index.max_value);
        if min_ord > max_ord || min_ord > g_max || max_ord < g_min {
            self.skipper_min_doc_id = NO_MORE_DOCS;
            self.skipper_max_doc_id = NO_MORE_DOCS;
            self.skipper_min_doc_id_exact = true;
            self.skipper_max_doc_id_exact = true;
            return;
        }
        if g_min >= min_ord && g_max <= max_ord {
            self.skipper_min_doc_id = 0;
            self.skipper_max_doc_id = self.index.doc_count;
            self.skipper_min_doc_id_exact = true;
            self.skipper_max_doc_id_exact = true;
            return;
        }
        if self.reverse {
            if g_max <= max_ord {
                self.skipper_min_doc_id = 0;
                self.skipper_min_doc_id_exact = true;
            } else {
                self.skipper.advance_range(i64::MIN, max_ord);
                self.skipper_min_doc_id = self.skipper.min_doc_id(0);
                self.skipper_min_doc_id_exact = self.skipper.max_value(0) == max_ord;
            }
            if g_min >= min_ord {
                self.skipper_max_doc_id = self.index.doc_count;
                self.skipper_max_doc_id_exact = true;
            } else {
                self.skipper
                    .advance_range(i64::MIN, min_ord.saturating_sub(1));
                self.skipper_max_doc_id = self.skipper.min_doc_id(0);
                self.skipper_max_doc_id_exact = self.skipper.max_value(0) < min_ord;
            }
        } else {
            if g_min >= min_ord {
                self.skipper_min_doc_id = 0;
                self.skipper_min_doc_id_exact = true;
            } else {
                self.skipper.advance_range(min_ord, i64::MAX);
                self.skipper_min_doc_id = self.skipper.min_doc_id(0);
                self.skipper_min_doc_id_exact = self.skipper.min_value(0) == min_ord;
            }
            if g_max <= max_ord {
                self.skipper_max_doc_id = self.index.doc_count;
                self.skipper_max_doc_id_exact = true;
            } else {
                self.skipper
                    .advance_range(max_ord.saturating_add(1), i64::MAX);
                self.skipper_max_doc_id = self.skipper.min_doc_id(0);
                self.skipper_max_doc_id_exact = self.skipper.min_value(0) > max_ord;
            }
        }
    }

    /// `range()`: `[minDocID, maxDocID)`, stepping the values through
    /// `next_doc(start, predicate)` where the skip index is not exact.
    pub fn range(
        &mut self,
        mut next_doc: impl FnMut(i32, &dyn Fn(i64) -> bool) -> Result<i32>,
    ) -> Result<(i32, i32)> {
        if !self.computed {
            self.compute_skipper_doc_ids();
        }
        let (min_ord, max_ord) = (self.lower, self.upper);
        let (min_doc, max_doc) = if self.reverse {
            let lo = if self.skipper_min_doc_id_exact {
                self.skipper_min_doc_id
            } else {
                next_doc(self.skipper_min_doc_id, &|l| l <= max_ord)?
            };
            let hi = if self.skipper_max_doc_id_exact {
                self.skipper_max_doc_id
            } else {
                next_doc(self.skipper_max_doc_id, &|l| l < min_ord)?
            };
            (lo, hi)
        } else {
            let lo = if self.skipper_min_doc_id_exact {
                self.skipper_min_doc_id
            } else {
                next_doc(self.skipper_min_doc_id, &|l| l >= min_ord)?
            };
            let hi = if self.skipper_max_doc_id_exact {
                self.skipper_max_doc_id
            } else {
                next_doc(self.skipper_max_doc_id, &|l| l > max_ord)?
            };
            (lo, hi)
        };
        Ok((min_doc, max_doc))
    }

    /// `cost()`: the skip index's document span, plus one level-0 interval
    /// when its upper end is not exact.
    pub fn cost(&mut self) -> i64 {
        if !self.computed {
            self.compute_skipper_doc_ids();
        }
        let span =
            i64::from(self.skipper_max_doc_id).saturating_sub(i64::from(self.skipper_min_doc_id));
        if self.skipper_max_doc_id_exact {
            span
        } else {
            span.saturating_add(i64::from(self.skipper.doc_count(0)))
        }
    }
}

/// `RangeBulkScorer`: every document of `[min_doc_id, max_doc_id)` at one
/// score, collected run by run (`LeafCollector.collectRange`), deleted
/// documents splitting the runs.
#[derive(Debug, Clone, Copy)]
pub struct RangeBulkScorer {
    pub min_doc_id: i32,
    pub max_doc_id: i32,
    pub score: f32,
}

impl RangeBulkScorer {
    /// `RangeBulkScorer(iterator, score, minDocID, maxDocID)`: `min < max`.
    pub fn new(min_doc_id: i32, max_doc_id: i32, score: f32) -> Result<Self> {
        if min_doc_id >= max_doc_id {
            return Err(illegal("minDocID must be less than maxDocID"));
        }
        Ok(RangeBulkScorer {
            min_doc_id,
            max_doc_id,
            score,
        })
    }

    /// `score(collector, acceptDocs, min, max)`: the documents of
    /// `[min, max)` in the range, returning where scoring stopped (the next
    /// document the range holds, or `NO_MORE_DOCS`).
    pub fn score(
        &self,
        collector: &mut dyn ScoringCollector,
        accept_docs: Option<&FixedBitSet>,
        min: i32,
        max: i32,
    ) -> i32 {
        let advance = |target: i32| {
            if target >= self.max_doc_id {
                NO_MORE_DOCS
            } else {
                target.max(self.min_doc_id)
            }
        };
        if max <= self.min_doc_id {
            return advance(self.min_doc_id);
        }
        if min >= self.max_doc_id {
            return advance(self.max_doc_id);
        }
        let lo = min.max(self.min_doc_id);
        let hi = max.min(self.max_doc_id);
        for doc in lo..hi {
            if accept_docs.is_none_or(|bits| bits.get_doc(doc)) {
                collector.collect(doc, self.score);
            }
        }
        advance(hi)
    }

    /// `SortedSkipperScorerSupplier.bulkScorer()` driven over a whole
    /// segment: nothing for an empty range.
    pub fn score_range(
        (min, max): (i32, i32),
        accept_docs: Option<&FixedBitSet>,
        score: f32,
        collector: &mut dyn ScoringCollector,
    ) {
        if let Ok(s) = RangeBulkScorer::new(min, max, score) {
            s.score(collector, accept_docs, 0, NO_MORE_DOCS);
        }
    }

    /// `cost()`.
    pub fn cost(&self) -> i64 {
        i64::from(self.max_doc_id).saturating_sub(i64::from(self.min_doc_id))
    }
}

/// `ConstantScoreQuery(TermQuery)` (`KeywordField.newExactQuery`): the
/// documents holding the term, at a constant score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermConstantScoreQuery {
    pub field: String,
    pub term: Vec<u8>,
}

fn term_docs(leaf: &OpenSegment<'_>, field: &str, term: &[u8]) -> Result<Vec<i32>> {
    let mut docs = VecCollector::default();
    search_term_query(
        leaf.fields,
        leaf.doc_in,
        leaf.live_docs,
        &TermQuery::new(field, term.to_vec()),
        &mut docs,
    )?;
    Ok(docs.docs)
}

impl DocumentQuery for TermConstantScoreQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        for doc in term_docs(leaf, &self.field, &self.term)? {
            collector.collect(doc, boost);
        }
        Ok(())
    }
}

/// `TermInSetQuery.newIndexOrDocValuesQuery` (`KeywordField.newSetQuery`):
/// the documents holding any of the terms, from the postings (the doc-values
/// side, [`DocValuesTermInSetQuery`], finds the same), at a constant score.
/// A segment whose field lacks postings or doc values matches nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermInSetConstantScoreQuery {
    pub field: String,
    /// Sorted, deduplicated.
    pub terms: Vec<Vec<u8>>,
}

impl DocumentQuery for TermInSetConstantScoreQuery {
    fn rewrite(&self, _leaves: &[OpenSegment<'_>]) -> Result<Option<Box<dyn DocumentQuery>>> {
        if self.terms.is_empty() {
            return Ok(Some(Box::new(MatchNoDocs)));
        }
        Ok(None)
    }

    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(info) = field_info(leaf, &self.field)? else {
            return Ok(());
        };
        if info.doc_values_type == DocValuesType::None {
            return Ok(());
        }
        let mut docs: Vec<i32> = Vec::new();
        for t in &self.terms {
            docs.extend(term_docs(leaf, &self.field, t)?);
        }
        docs.sort_unstable();
        docs.dedup();
        for doc in docs {
            collector.collect(doc, boost);
        }
        Ok(())
    }
}

fn check_set_bytes(values: &[&[u8]]) -> Vec<Vec<u8>> {
    values.iter().map(|v| v.to_vec()).collect()
}

macro_rules! numeric_dv_factories {
    ($(#[$m:meta])* $module:ident) => {
        $(#[$m])*
        pub mod $module {
            use super::*;

            /// `newSlowRangeQuery(field, lowerValue, upperValue)`: inclusive.
            pub fn new_slow_range_query(
                field: &str,
                lower: i64,
                upper: i64,
            ) -> SortedNumericDocValuesRangeQuery {
                SortedNumericDocValuesRangeQuery::new(field, lower, upper)
            }

            /// `newSlowExactQuery(field, value)`.
            pub fn new_slow_exact_query(field: &str, value: i64) -> SortedNumericDocValuesRangeQuery {
                new_slow_range_query(field, value, value)
            }

            /// `newSlowSetQuery(field, values...)`.
            pub fn new_slow_set_query(field: &str, values: &[i64]) -> SortedNumericDocValuesSetQuery {
                SortedNumericDocValuesSetQuery::new(field, values.to_vec())
            }
        }
    };
}

numeric_dv_factories!(
    /// `NumericDocValuesField`'s slow queries.
    numeric_doc_values_field
);
numeric_dv_factories!(
    /// `SortedNumericDocValuesField`'s slow queries.
    sorted_numeric_doc_values_field
);

macro_rules! ords_dv_factories {
    ($(#[$m:meta])* $module:ident) => {
        $(#[$m])*
        pub mod $module {
            use super::*;

            /// `newSlowRangeQuery(field, lower, upper, lowerInclusive,
            /// upperInclusive)`: `None` is an open bound.
            pub fn new_slow_range_query(
                field: &str,
                lower: Option<&[u8]>,
                upper: Option<&[u8]>,
                lower_inclusive: bool,
                upper_inclusive: bool,
            ) -> SortedSetDocValuesRangeQuery {
                SortedSetDocValuesRangeQuery::new(
                    field,
                    lower.map(<[u8]>::to_vec),
                    upper.map(<[u8]>::to_vec),
                    lower_inclusive,
                    upper_inclusive,
                )
            }

            /// `newSlowExactQuery(field, value)`.
            pub fn new_slow_exact_query(field: &str, value: &[u8]) -> SortedSetDocValuesRangeQuery {
                new_slow_range_query(field, Some(value), Some(value), true, true)
            }

            /// `newSlowSetQuery(field, values)`.
            pub fn new_slow_set_query(field: &str, values: &[&[u8]]) -> DocValuesTermInSetQuery {
                DocValuesTermInSetQuery::new(field, check_set_bytes(values))
            }
        }
    };
}

ords_dv_factories!(
    /// `SortedDocValuesField`'s slow queries.
    sorted_doc_values_field
);
ords_dv_factories!(
    /// `SortedSetDocValuesField`'s slow queries.
    sorted_set_doc_values_field
);

/// `SortedSetSelector.Type`, as `KeywordField.newSortField` takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortedSetSelector {
    Min,
    Max,
}

/// `KeywordField`'s queries and sort field.
pub mod keyword_field {
    use super::*;

    /// `newExactQuery(field, value)`: `ConstantScoreQuery(TermQuery)`.
    pub fn new_exact_query(field: &str, value: &[u8]) -> TermConstantScoreQuery {
        TermConstantScoreQuery {
            field: field.to_string(),
            term: value.to_vec(),
        }
    }

    /// `newSetQuery(field, values)`.
    pub fn new_set_query(field: &str, values: &[&[u8]]) -> TermInSetConstantScoreQuery {
        let mut terms = check_set_bytes(values);
        terms.sort_unstable();
        terms.dedup();
        TermInSetConstantScoreQuery {
            field: field.to_string(),
            terms,
        }
    }

    /// `newSortField(field, reverse, selector)`: a `SortedSetSortField`,
    /// missing values first.
    pub fn new_sort_field(field: &str, reverse: bool, selector: SortedSetSelector) -> SortField {
        let mut s = SortField::string(field, reverse);
        s.selector = match selector {
            SortedSetSelector::Min => Selector::Min,
            SortedSetSelector::Max => Selector::Max,
        };
        s
    }
}
