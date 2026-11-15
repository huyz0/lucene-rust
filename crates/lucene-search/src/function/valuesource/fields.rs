//! The doc-values field sources: `FieldCacheSource`, `IntFieldSource`,
//! `LongFieldSource`, `FloatFieldSource`, `DoubleFieldSource`, their
//! `MultiValued*FieldSource` siblings over `SORTED_NUMERIC` (through
//! `SortedNumericSelector`), `EnumFieldSource`, `BytesRefFieldSource`,
//! `SortedSetFieldSource` (through `SortedSetSelector`) and
//! `JoinDocFreqValueSource`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::function::docvalues::{
    DocTermsIndexDocValues, Double, DoubleDocValues, Float, FloatDocValues, Int, IntDocValues,
    Long, LongDocValues,
};
use crate::function::{
    not_weighted, out_of_order, parse_java_int, BoxValues, FunctionContext, FunctionValues,
    MutableValue, ObjectVal, RangeMatcher, TopLevel, ValueLeaf, ValueSource,
};
use crate::reader::doc_values as dv;
use crate::reader::{
    BinaryDocValues, DocIdSetIterator, DocValuesIterator, NumericDocValues, SortedDocValues,
    SortedNumericDocValues, SortedSetDocValues, NO_MORE_DOCS,
};
use crate::top_field::{SortField, SortType};
use crate::{Error, Result};
use lucene_codecs::doc_values::NumericReader;

/// `FieldCacheSource`: a source reading one field (its description is the
/// field's name).
pub trait FieldCacheSource: ValueSource {
    /// `getField()`.
    fn field(&self) -> &str;
}

/// A `NUMERIC` iterator positioned as the field sources' `exists(doc)`
/// moves it: forward only, `docs were sent out-of-order` otherwise.
pub(crate) struct NumericColumn<'a> {
    arr: Arr<'a>,
    last_doc: i32,
    /// The last document asked about, and its value (`None` without one).
    at: i32,
    value: Option<i64>,
    /// A window's values and presence bits ([`Self::get_batch`]).
    window: Vec<i64>,
    present: Vec<u64>,
}

/// The widest run of documents [`NumericColumn::get_batch`] reads as one
/// window; a batch spread wider (many deleted documents between) is read
/// one document at a time.
const MAX_WINDOW: i32 = 4096;

/// Whether `n` documents over a window `span + 1` wide are worth decoding
/// the whole window for: more than three in four of its documents asked
/// (an every-document scorer's run). Sparser batches -- a term's postings
/// -- are cheaper looked up one by one: measured, a window decode lost to
/// per-document reads even at half the window asked.
fn dense_enough(n: usize, span: i32) -> bool {
    usize::try_from(span).is_ok_and(|s| n.saturating_mul(4) > s.saturating_mul(3))
}

/// Where a column's values come from: any `NumericDocValues` (a
/// selector's view, an empty column), or a segment's own `NUMERIC` field
/// read through the codec's reader directly (stage 3: no iterator object
/// between the source and the values; the same values, read forward as the
/// iterator reads them).
enum Arr<'a> {
    Iter(Box<dyn NumericDocValues + 'a>),
    Direct(Box<NumericReader<'a>>),
}

impl<'a> NumericColumn<'a> {
    pub(crate) fn new(arr: Box<dyn NumericDocValues + 'a>) -> Self {
        Self::of(Arr::Iter(arr))
    }

    fn of(arr: Arr<'a>) -> Self {
        Self {
            arr,
            last_doc: 0,
            at: -1,
            value: None,
            window: Vec::new(),
            present: Vec::new(),
        }
    }

    /// `DocValues.getNumeric(reader, field)` for a field source.
    fn open(leaf: &ValueLeaf<'a>, field: &str) -> Result<Self> {
        let reader = leaf.reader()?;
        let direct = reader.field_infos().field_by_name(field).and_then(|fi| {
            let (meta, data) = reader.doc_values_for_field(fi.number)?;
            Some((meta.numeric_entry(fi.number)?, data))
        });
        Ok(match direct {
            Some((entry, data)) => Self::of(Arr::Direct(Box::new(NumericReader::new(data, entry)))),
            None => Self::new(dv::get_numeric(reader, field)?),
        })
    }

    /// Java's `if (doc > arr.docID()) arr.advance(doc); return doc ==
    /// arr.docID()`: the document's value is looked up once.
    pub(crate) fn exists(&mut self, doc: i32) -> Result<bool> {
        Ok(self.get(doc)?.is_some())
    }

    /// [`Self::get`] for each of `docs` (ascending): `has[i]` whether `docs[i]`
    /// has a value and `out[i]` that value (`0` without one); both as long
    /// as `docs`. A segment's own column is read as one window over the
    /// 64-aligned documents covering the batch
    /// ([`NumericReader::fill_window`]: a chunked decode for a dense column,
    /// one walk of the `IndexedDISI` and a run of consecutive ordinals for a
    /// sparse one); the values are [`Self::get`]'s, and the column is left
    /// as the last `get` would leave it.
    ///
    /// # Errors
    /// As [`Self::get`]: a batch before the last document asked about, or
    /// whatever reading the column reports.
    pub(crate) fn get_batch(
        &mut self,
        docs: &[i32],
        out: &mut [i64],
        has: &mut [bool],
    ) -> Result<()> {
        let (Some(&first), Some(&last)) = (docs.first(), docs.last()) else {
            return Ok(());
        };
        let start = first & !63;
        let direct = match &mut self.arr {
            Arr::Direct(r)
                if first >= self.last_doc
                    && last.saturating_sub(start) < MAX_WINDOW
                    && dense_enough(docs.len(), last.saturating_sub(start)) =>
            {
                Some(r)
            }
            _ => None,
        };
        let Some(r) = direct else {
            for ((&doc, o), h) in docs.iter().zip(out.iter_mut()).zip(has.iter_mut()) {
                let v = self.get(doc)?;
                *h = v.is_some();
                *o = v.unwrap_or(0);
            }
            return Ok(());
        };
        // A sparse column's window is whole words of 64 documents (its
        // `IndexedDISI` is read a word at a time; documents past the last
        // have no value); a dense one's ends at the batch's last document,
        // never past the column's.
        // ARITH: `0 <= last - start < MAX_WINDOW`, so either length is
        // positive and no larger than `MAX_WINDOW`.
        #[allow(clippy::arithmetic_side_effects)]
        let len = if r.entry().is_dense() {
            (last - start) as usize + 1
        } else {
            ((last - start) as usize / 64 + 1) * 64
        };
        self.window.resize(len, 0);
        self.present.resize(len.div_ceil(64), 0);
        r.fill_window(start, &mut self.window, &mut self.present)?;
        for ((&doc, o), h) in docs.iter().zip(out.iter_mut()).zip(has.iter_mut()) {
            // ARITH: `start <= doc <= last`.
            #[allow(clippy::arithmetic_side_effects)]
            let i = (doc - start) as usize;
            *h = self.present[i >> 6] >> (i & 63) & 1 == 1;
            *o = if *h { self.window[i] } else { 0 };
        }
        self.last_doc = last;
        self.at = last;
        let n = docs.len() - 1;
        self.value = has[n].then_some(out[n]);
        Ok(())
    }

    /// The value at `doc`, or `None` without one.
    #[inline]
    pub(crate) fn get(&mut self, doc: i32) -> Result<Option<i64>> {
        if doc < self.last_doc {
            return Err(out_of_order(self.last_doc, doc));
        }
        self.last_doc = doc;
        if doc != self.at {
            self.value = match &mut self.arr {
                Arr::Direct(r) => r.value(doc)?,
                Arr::Iter(arr) => {
                    if arr.advance_exact(doc)? {
                        Some(arr.long_value())
                    } else {
                        None
                    }
                }
            };
            self.at = doc;
        }
        Ok(self.value)
    }
}

// ---------------------------------------------------------------------------
// SortedNumericSelector / SortedSetSelector
// ---------------------------------------------------------------------------

/// `SortedNumericSelector.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericSelector {
    Min,
    Max,
}

impl NumericSelector {
    /// `name()`.
    pub fn name(self) -> &'static str {
        match self {
            NumericSelector::Min => "MIN",
            NumericSelector::Max => "MAX",
        }
    }
}

/// How a `SORTED_NUMERIC` value is decoded back from its sortable form
/// (`SortedNumericSelector.wrap`'s `numericType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sortable {
    Plain,
    Float,
    Double,
}

/// `SortedNumericSelector.wrap(sortedNumeric, selector, numericType)`: the
/// minimum or maximum of each document's values (a singleton field's one
/// value either way), with `NumericUtils.sortableFloatBits` /
/// `sortableDoubleBits` undone for a float or double field.
struct SelectedNumeric<'a> {
    inner: Box<dyn SortedNumericDocValues + 'a>,
    selector: NumericSelector,
    sortable: Sortable,
    value: i64,
}

impl SelectedNumeric<'_> {
    fn set_value(&mut self) -> Result<()> {
        self.value = match self.selector {
            NumericSelector::Min => self.inner.next_value()?,
            NumericSelector::Max => {
                let mut v = 0;
                for _ in 0..self.inner.doc_value_count() {
                    v = self.inner.next_value()?;
                }
                v
            }
        };
        Ok(())
    }
}

impl DocIdSetIterator for SelectedNumeric<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        let d = self.inner.next_doc()?;
        if d != NO_MORE_DOCS {
            self.set_value()?;
        }
        Ok(d)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let d = self.inner.advance(target)?;
        if d != NO_MORE_DOCS {
            self.set_value()?;
        }
        Ok(d)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
}

impl DocValuesIterator for SelectedNumeric<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        if self.inner.advance_exact(target)? {
            self.set_value()?;
            return Ok(true);
        }
        Ok(false)
    }
}

impl NumericDocValues for SelectedNumeric<'_> {
    fn long_value(&self) -> i64 {
        match self.sortable {
            Sortable::Plain => self.value,
            // `NumericUtils.sortableFloatBits((int) in.longValue())`.
            Sortable::Float => {
                let bits = self.value as i32;
                i64::from(bits ^ ((bits >> 31) & 0x7fff_ffff))
            }
            Sortable::Double => self.value ^ ((self.value >> 63) & 0x7fff_ffff_ffff_ffff),
        }
    }
}

fn selected_numeric<'a>(
    leaf: &ValueLeaf<'a>,
    field: &str,
    selector: NumericSelector,
    sortable: Sortable,
) -> Result<Box<dyn NumericDocValues + 'a>> {
    let inner = dv::get_sorted_numeric(leaf.reader()?, field)?;
    Ok(Box::new(SelectedNumeric {
        inner,
        selector,
        sortable,
        value: 0,
    }))
}

/// `SortedSetSelector.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetSelector {
    Min,
    Max,
    MiddleMin,
    MiddleMax,
}

impl SetSelector {
    /// `name()` / `toString()`.
    pub fn name(self) -> &'static str {
        match self {
            SetSelector::Min => "MIN",
            SetSelector::Max => "MAX",
            SetSelector::MiddleMin => "MIDDLE_MIN",
            SetSelector::MiddleMax => "MIDDLE_MAX",
        }
    }
}

/// `SortedSetSelector.wrap(sortedSet, selector)`: one ordinal of each
/// document's set.
struct SelectedSet<'a> {
    inner: Box<dyn SortedSetDocValues + 'a>,
    selector: SetSelector,
    ord: i32,
}

impl SelectedSet<'_> {
    fn set_ord(&mut self) -> Result<()> {
        let count = self.inner.doc_value_count();
        let pick = match self.selector {
            SetSelector::Min => 0,
            SetSelector::Max => count - 1,
            SetSelector::MiddleMin => (count - 1) / 2,
            SetSelector::MiddleMax => count / 2,
        };
        let mut ord = -1i64;
        for _ in 0..=pick {
            ord = self.inner.next_ord()?;
        }
        self.ord = ord as i32;
        Ok(())
    }
}

impl DocIdSetIterator for SelectedSet<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        let d = self.inner.next_doc()?;
        if d != NO_MORE_DOCS {
            self.set_ord()?;
        }
        Ok(d)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let d = self.inner.advance(target)?;
        if d != NO_MORE_DOCS {
            self.set_ord()?;
        }
        Ok(d)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
}

impl DocValuesIterator for SelectedSet<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        if self.inner.advance_exact(target)? {
            self.set_ord()?;
            return Ok(true);
        }
        Ok(false)
    }
}

impl SortedDocValues for SelectedSet<'_> {
    fn ord_value(&self) -> i32 {
        self.ord
    }
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
        self.inner.lookup_ord(i64::from(ord))
    }
    fn value_count(&self) -> i32 {
        self.inner.value_count() as i32
    }
    fn lookup_term(&mut self, key: &[u8]) -> Result<i32> {
        let ret = self.inner.lookup_term(key)?;
        i32::try_from(ret).map_err(|_| {
            Error::Unsupported(format!(
                "fields containing more than {} unique terms are unsupported",
                i32::MAX - 1
            ))
        })
    }
}

/// `SortedSetSelector.wrap`'s refusal of a dictionary of `Integer.MAX_VALUE`
/// terms or more.
fn selected_set<'a>(
    leaf: &ValueLeaf<'a>,
    field: &str,
    selector: SetSelector,
) -> Result<Box<dyn SortedDocValues + 'a>> {
    let inner = dv::get_sorted_set(leaf.reader()?, field)?;
    if inner.value_count() >= i64::from(i32::MAX) {
        return Err(Error::Unsupported(format!(
            "fields containing more than {} unique terms are unsupported",
            i32::MAX - 1
        )));
    }
    Ok(Box::new(SelectedSet {
        inner,
        selector,
        ord: -1,
    }))
}

// ---------------------------------------------------------------------------
// Int / Long / Float / Double
// ---------------------------------------------------------------------------

struct IntValues<'a> {
    col: NumericColumn<'a>,
    description: String,
    /// A batch's values ([`NumericColumn::get_batch`]).
    buf: Vec<i64>,
    has: Vec<bool>,
}

impl IntDocValues for IntValues<'_> {
    fn description(&self) -> String {
        self.description.clone()
    }
    fn int_val(&mut self, doc: i32) -> Result<i32> {
        Ok(self.col.get(doc)?.map_or(0, |v| v as i32))
    }
    fn exists(&mut self, doc: i32) -> Result<bool> {
        self.col.exists(doc)
    }
    /// `(float) intVal(doc)` per document.
    fn float_val_batch(&mut self, docs: &[i32], out: &mut [f32]) -> Result<()> {
        self.buf.resize(docs.len(), 0);
        self.has.resize(docs.len(), false);
        self.col.get_batch(docs, &mut self.buf, &mut self.has)?;
        for (o, &v) in out.iter_mut().zip(&self.buf) {
            // `0` without a value, as `intVal` reads it.
            *o = v as i32 as f32;
        }
        Ok(())
    }
    /// `IntDocValues`' getters over one batch of the column: `exists`, the
    /// bounds against the getter the range reads, and `floatVal`.
    fn range_batch_native(
        &mut self,
        range: &RangeMatcher,
        docs: &[i32],
        matched: &mut Vec<i32>,
        mut values: Option<&mut Vec<f32>>,
    ) -> Result<bool> {
        if matches!(range, RangeMatcher::Ord { .. }) {
            return Ok(false);
        }
        self.buf.resize(docs.len(), 0);
        self.has.resize(docs.len(), false);
        self.col.get_batch(docs, &mut self.buf, &mut self.has)?;
        for ((&doc, &v), &h) in docs.iter().zip(&self.buf).zip(&self.has) {
            if !h {
                continue;
            }
            let i = v as i32;
            if range.matches_numeric(i as f32, f64::from(i), i, i64::from(i)) == Some(true) {
                matched.push(doc);
                if let Some(values) = values.as_deref_mut() {
                    values.push(i as f32);
                }
            }
        }
        Ok(true)
    }
}

/// `IntFieldSource`: a `NUMERIC` field's value as an `int`.
#[derive(Debug, Clone)]
pub struct IntFieldSource {
    field: String,
}

impl IntFieldSource {
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }
}

impl ValueSource for IntFieldSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        Ok(Box::new(Int(IntValues {
            col: NumericColumn::open(leaf, &self.field)?,
            description: self.description(),
            buf: Vec::new(),
            has: Vec::new(),
        })))
    }
    fn description(&self) -> String {
        format!("int({})", self.field)
    }
    fn native_sort_field(&self, reverse: bool) -> Option<SortField> {
        let mut f = SortField::numeric(&self.field, SortType::Int, false);
        f.reverse = reverse;
        Some(f)
    }
}

impl FieldCacheSource for IntFieldSource {
    fn field(&self) -> &str {
        &self.field
    }
}

/// `MultiValuedIntFieldSource`: a `SORTED_NUMERIC` field's minimum or
/// maximum as an `int`.
#[derive(Debug, Clone)]
pub struct MultiValuedIntFieldSource {
    field: String,
    selector: NumericSelector,
    missing_value: Option<i32>,
}

impl MultiValuedIntFieldSource {
    pub fn new(field: impl Into<String>, selector: NumericSelector) -> Self {
        Self::with_missing(field, selector, None)
    }

    pub fn with_missing(
        field: impl Into<String>,
        selector: NumericSelector,
        missing_value: Option<i32>,
    ) -> Self {
        Self {
            field: field.into(),
            selector,
            missing_value,
        }
    }
}

impl ValueSource for MultiValuedIntFieldSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        let arr = selected_numeric(leaf, &self.field, self.selector, Sortable::Plain)?;
        Ok(Box::new(Int(IntValues {
            col: NumericColumn::new(arr),
            description: self.description(),
            buf: Vec::new(),
            has: Vec::new(),
        })))
    }
    fn description(&self) -> String {
        format!("int({},{})", self.field, self.selector.name())
    }
    fn native_sort_field(&self, reverse: bool) -> Option<SortField> {
        Some(multi_sort(
            &self.field,
            SortType::Int,
            reverse,
            self.selector,
            self.missing_value.map(i64::from),
        ))
    }
}

impl FieldCacheSource for MultiValuedIntFieldSource {
    fn field(&self) -> &str {
        &self.field
    }
}

/// `SortedNumericSortField(field, type, reverse, selector, missingValue)`.
fn multi_sort(
    field: &str,
    ty: SortType,
    reverse: bool,
    selector: NumericSelector,
    missing: Option<i64>,
) -> SortField {
    let mut f = SortField::numeric(field, ty, false);
    f.reverse = reverse;
    f.selector = match selector {
        NumericSelector::Min => crate::top_field::Selector::Min,
        NumericSelector::Max => crate::top_field::Selector::Max,
    };
    if let Some(m) = missing {
        f.missing = m;
    }
    f
}

struct LongValues<'a> {
    col: NumericColumn<'a>,
    description: String,
    /// A batch's values ([`NumericColumn::get_batch`]).
    buf: Vec<i64>,
    has: Vec<bool>,
}

impl LongDocValues for LongValues<'_> {
    fn description(&self) -> String {
        self.description.clone()
    }
    fn long_val(&mut self, doc: i32) -> Result<i64> {
        Ok(self.col.get(doc)?.unwrap_or(0))
    }
    /// `(float) longVal(doc)` per document.
    fn float_val_batch(&mut self, docs: &[i32], out: &mut [f32]) -> Result<()> {
        self.buf.resize(docs.len(), 0);
        self.has.resize(docs.len(), false);
        self.col.get_batch(docs, &mut self.buf, &mut self.has)?;
        for (o, &v) in out.iter_mut().zip(&self.buf) {
            *o = v as f32;
        }
        Ok(())
    }
    fn exists(&mut self, doc: i32) -> Result<bool> {
        self.col.exists(doc)
    }
    /// `longToObject(value)`, `null` without one.
    fn object_val(&mut self, doc: i32) -> Result<ObjectVal> {
        Ok(match self.col.get(doc)? {
            Some(v) => ObjectVal::Long(v),
            None => ObjectVal::Null,
        })
    }
    /// `longToString(value)`, `null` without one.
    fn str_val(&mut self, doc: i32) -> Result<Option<String>> {
        Ok(self.col.get(doc)?.map(|v| v.to_string()))
    }
}

/// `LongFieldSource`: a `NUMERIC` field's value as a `long`.
#[derive(Debug, Clone)]
pub struct LongFieldSource {
    field: String,
}

impl LongFieldSource {
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }
}

impl ValueSource for LongFieldSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        Ok(Box::new(Long(LongValues {
            col: NumericColumn::open(leaf, &self.field)?,
            description: self.description(),
            buf: Vec::new(),
            has: Vec::new(),
        })))
    }
    fn description(&self) -> String {
        format!("long({})", self.field)
    }
    fn native_sort_field(&self, reverse: bool) -> Option<SortField> {
        let mut f = SortField::numeric(&self.field, SortType::Long, false);
        f.reverse = reverse;
        Some(f)
    }
}

impl FieldCacheSource for LongFieldSource {
    fn field(&self) -> &str {
        &self.field
    }
}

/// `MultiValuedLongFieldSource`.
#[derive(Debug, Clone)]
pub struct MultiValuedLongFieldSource {
    field: String,
    selector: NumericSelector,
    missing_value: Option<i64>,
}

impl MultiValuedLongFieldSource {
    pub fn new(field: impl Into<String>, selector: NumericSelector) -> Self {
        Self::with_missing(field, selector, None)
    }

    pub fn with_missing(
        field: impl Into<String>,
        selector: NumericSelector,
        missing_value: Option<i64>,
    ) -> Self {
        Self {
            field: field.into(),
            selector,
            missing_value,
        }
    }
}

impl ValueSource for MultiValuedLongFieldSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        let arr = selected_numeric(leaf, &self.field, self.selector, Sortable::Plain)?;
        Ok(Box::new(Long(LongValues {
            col: NumericColumn::new(arr),
            description: self.description(),
            buf: Vec::new(),
            has: Vec::new(),
        })))
    }
    fn description(&self) -> String {
        format!("long({},{})", self.field, self.selector.name())
    }
    fn native_sort_field(&self, reverse: bool) -> Option<SortField> {
        Some(multi_sort(
            &self.field,
            SortType::Long,
            reverse,
            self.selector,
            self.missing_value,
        ))
    }
}

impl FieldCacheSource for MultiValuedLongFieldSource {
    fn field(&self) -> &str {
        &self.field
    }
}

struct FloatValues<'a> {
    col: NumericColumn<'a>,
    description: String,
    /// A batch's values ([`NumericColumn::get_batch`]).
    buf: Vec<i64>,
    has: Vec<bool>,
}

impl FloatDocValues for FloatValues<'_> {
    fn description(&self) -> String {
        self.description.clone()
    }
    /// `Float.intBitsToFloat((int) arr.longValue())`.
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        Ok(self
            .col
            .get(doc)?
            .map_or(0.0, |v| f32::from_bits(v as i32 as u32)))
    }
    fn float_val_batch(&mut self, docs: &[i32], out: &mut [f32]) -> Result<()> {
        self.buf.resize(docs.len(), 0);
        self.has.resize(docs.len(), false);
        self.col.get_batch(docs, &mut self.buf, &mut self.has)?;
        for ((o, &v), &h) in out.iter_mut().zip(&self.buf).zip(&self.has) {
            *o = if h {
                f32::from_bits(v as i32 as u32)
            } else {
                0.0
            };
        }
        Ok(())
    }
    /// `doubleVal` is `floatVal` widened (the base's).
    fn double_val_batch(&mut self, docs: &[i32], out: &mut [f64]) -> Result<()> {
        self.buf.resize(docs.len(), 0);
        self.has.resize(docs.len(), false);
        self.col.get_batch(docs, &mut self.buf, &mut self.has)?;
        for ((o, &v), &h) in out.iter_mut().zip(&self.buf).zip(&self.has) {
            *o = f64::from(if h {
                f32::from_bits(v as i32 as u32)
            } else {
                0.0
            });
        }
        Ok(())
    }
    fn exists(&mut self, doc: i32) -> Result<bool> {
        self.col.exists(doc)
    }
}

/// `FloatFieldSource`: a `NUMERIC` field holding float bits
/// (`FloatDocValuesField`).
#[derive(Debug, Clone)]
pub struct FloatFieldSource {
    field: String,
}

impl FloatFieldSource {
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }
}

impl ValueSource for FloatFieldSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        Ok(Box::new(Float(FloatValues {
            col: NumericColumn::open(leaf, &self.field)?,
            description: self.description(),
            buf: Vec::new(),
            has: Vec::new(),
        })))
    }
    fn description(&self) -> String {
        format!("float({})", self.field)
    }
    /// `SortField(field, FLOAT)` over the raw bits a `FloatDocValuesField`
    /// stores, which `top_field`'s numeric keys (sortable bits) do not
    /// read: the value-source comparator instead, which orders the same
    /// (`Double.compare`, a missing value as `0`).
    fn native_sort_field(&self, _reverse: bool) -> Option<SortField> {
        None
    }
}

impl FieldCacheSource for FloatFieldSource {
    fn field(&self) -> &str {
        &self.field
    }
}

/// `MultiValuedFloatFieldSource`: a `SORTED_NUMERIC` field of sortable
/// float bits.
#[derive(Debug, Clone)]
pub struct MultiValuedFloatFieldSource {
    field: String,
    selector: NumericSelector,
    missing_value: Option<f32>,
}

impl MultiValuedFloatFieldSource {
    pub fn new(field: impl Into<String>, selector: NumericSelector) -> Self {
        Self::with_missing(field, selector, None)
    }

    pub fn with_missing(
        field: impl Into<String>,
        selector: NumericSelector,
        missing_value: Option<f32>,
    ) -> Self {
        Self {
            field: field.into(),
            selector,
            missing_value,
        }
    }
}

impl ValueSource for MultiValuedFloatFieldSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        let arr = selected_numeric(leaf, &self.field, self.selector, Sortable::Float)?;
        Ok(Box::new(Float(FloatValues {
            col: NumericColumn::new(arr),
            description: self.description(),
            buf: Vec::new(),
            has: Vec::new(),
        })))
    }
    fn description(&self) -> String {
        format!("float({},{})", self.field, self.selector.name())
    }
    fn native_sort_field(&self, reverse: bool) -> Option<SortField> {
        Some(multi_sort(
            &self.field,
            SortType::Float,
            reverse,
            self.selector,
            self.missing_value
                .map(|m| i64::from(crate::top_field::float_to_sortable_int(m))),
        ))
    }
}

impl FieldCacheSource for MultiValuedFloatFieldSource {
    fn field(&self) -> &str {
        &self.field
    }
}

struct DoubleValues<'a> {
    col: NumericColumn<'a>,
    description: String,
}

impl DoubleDocValues for DoubleValues<'_> {
    fn description(&self) -> String {
        self.description.clone()
    }
    /// `Double.longBitsToDouble(arr.longValue())`.
    fn double_val(&mut self, doc: i32) -> Result<f64> {
        Ok(self.col.get(doc)?.map_or(0.0, |v| f64::from_bits(v as u64)))
    }
    fn exists(&mut self, doc: i32) -> Result<bool> {
        self.col.exists(doc)
    }
}

/// `DoubleFieldSource`: a `NUMERIC` field holding double bits
/// (`DoubleDocValuesField`).
#[derive(Debug, Clone)]
pub struct DoubleFieldSource {
    field: String,
}

impl DoubleFieldSource {
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }
}

impl ValueSource for DoubleFieldSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        Ok(Box::new(Double(DoubleValues {
            col: NumericColumn::open(leaf, &self.field)?,
            description: self.description(),
        })))
    }
    fn description(&self) -> String {
        format!("double({})", self.field)
    }
    /// `SortField(field, DOUBLE)` over the raw bits a `DoubleDocValuesField`
    /// stores, which `top_field`'s numeric keys (sortable bits) do not
    /// read: the value-source comparator instead, which orders the same
    /// (`Double.compare`, a missing value as `0`).
    fn native_sort_field(&self, _reverse: bool) -> Option<SortField> {
        None
    }
}

impl FieldCacheSource for DoubleFieldSource {
    fn field(&self) -> &str {
        &self.field
    }
}

/// `MultiValuedDoubleFieldSource`: a `SORTED_NUMERIC` field of sortable
/// double bits.
#[derive(Debug, Clone)]
pub struct MultiValuedDoubleFieldSource {
    field: String,
    selector: NumericSelector,
    missing_value: Option<f64>,
}

impl MultiValuedDoubleFieldSource {
    pub fn new(field: impl Into<String>, selector: NumericSelector) -> Self {
        Self::with_missing(field, selector, None)
    }

    pub fn with_missing(
        field: impl Into<String>,
        selector: NumericSelector,
        missing_value: Option<f64>,
    ) -> Self {
        Self {
            field: field.into(),
            selector,
            missing_value,
        }
    }
}

impl ValueSource for MultiValuedDoubleFieldSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        let arr = selected_numeric(leaf, &self.field, self.selector, Sortable::Double)?;
        Ok(Box::new(Double(DoubleValues {
            col: NumericColumn::new(arr),
            description: self.description(),
        })))
    }
    fn description(&self) -> String {
        format!("double({},{})", self.field, self.selector.name())
    }
    fn native_sort_field(&self, reverse: bool) -> Option<SortField> {
        Some(multi_sort(
            &self.field,
            SortType::Double,
            reverse,
            self.selector,
            self.missing_value
                .map(crate::values_source::double_to_sortable_long),
        ))
    }
}

impl FieldCacheSource for MultiValuedDoubleFieldSource {
    fn field(&self) -> &str {
        &self.field
    }
}

// ---------------------------------------------------------------------------
// EnumFieldSource
// ---------------------------------------------------------------------------

/// `EnumFieldSource`: a `NUMERIC` field of enum ordinals, named through two
/// maps.
#[derive(Debug, Clone)]
pub struct EnumFieldSource {
    field: String,
    int_to_string: Arc<HashMap<i32, String>>,
    string_to_int: Arc<HashMap<String, i32>>,
}

/// `EnumFieldSource.DEFAULT_VALUE`.
const ENUM_DEFAULT: i32 = -1;

impl EnumFieldSource {
    pub fn new(
        field: impl Into<String>,
        int_to_string: HashMap<i32, String>,
        string_to_int: HashMap<String, i32>,
    ) -> Self {
        Self {
            field: field.into(),
            int_to_string: Arc::new(int_to_string),
            string_to_int: Arc::new(string_to_int),
        }
    }
}

struct EnumValues<'a> {
    col: NumericColumn<'a>,
    int_to_string: Arc<HashMap<i32, String>>,
    string_to_int: Arc<HashMap<String, i32>>,
    description: String,
}

impl EnumValues<'_> {
    /// `intValueToStringValue`: the name, or `"-1"`.
    fn name(&self, v: i32) -> String {
        self.int_to_string
            .get(&v)
            .cloned()
            .unwrap_or_else(|| ENUM_DEFAULT.to_string())
    }

    /// `stringValueToIntValue`: the named value, else a number that names
    /// one, else `-1`.
    fn value_of(&self, s: Option<&str>) -> Option<i32> {
        let s = s?;
        if let Some(&v) = self.string_to_int.get(s) {
            return Some(v);
        }
        let v = parse_java_int(s).unwrap_or(ENUM_DEFAULT);
        Some(if self.int_to_string.contains_key(&v) {
            v
        } else {
            ENUM_DEFAULT
        })
    }
}

impl IntDocValues for EnumValues<'_> {
    fn description(&self) -> String {
        self.description.clone()
    }
    fn int_val(&mut self, doc: i32) -> Result<i32> {
        Ok(self.col.get(doc)?.map_or(0, |v| v as i32))
    }
    fn str_val(&mut self, doc: i32) -> Result<Option<String>> {
        let v = self.int_val(doc)?;
        Ok(Some(self.name(v)))
    }
    fn exists(&mut self, doc: i32) -> Result<bool> {
        self.col.exists(doc)
    }
    /// `EnumFieldSource`'s range scorer: the bounds as enum values.
    fn range_matcher(
        &mut self,
        lower: Option<&str>,
        upper: Option<&str>,
        include_lower: bool,
        include_upper: bool,
    ) -> Result<RangeMatcher> {
        Ok(RangeMatcher::int(
            self.value_of(lower),
            self.value_of(upper),
            include_lower,
            include_upper,
        ))
    }
}

impl ValueSource for EnumFieldSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        Ok(Box::new(Int(EnumValues {
            col: NumericColumn::open(leaf, &self.field)?,
            int_to_string: Arc::clone(&self.int_to_string),
            string_to_int: Arc::clone(&self.string_to_int),
            description: self.description(),
        })))
    }
    fn description(&self) -> String {
        format!("enum({})", self.field)
    }
}

impl FieldCacheSource for EnumFieldSource {
    fn field(&self) -> &str {
        &self.field
    }
}

// ---------------------------------------------------------------------------
// BytesRefFieldSource, SortedSetFieldSource
// ---------------------------------------------------------------------------

/// `BytesRefFieldSource`: a `BINARY` field's bytes, or (any other field) a
/// `SORTED` field's term.
#[derive(Debug, Clone)]
pub struct BytesRefFieldSource {
    field: String,
}

impl BytesRefFieldSource {
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }
}

struct BinaryValues<'a> {
    arr: Box<dyn BinaryDocValues + 'a>,
    last_doc: i32,
    description: String,
}

impl BinaryValues<'_> {
    fn value(&mut self, doc: i32) -> Result<Option<Vec<u8>>> {
        if !self.exists(doc)? {
            return Ok(None);
        }
        let v = self.arr.binary_value();
        Ok((!v.is_empty()).then(|| v.to_vec()))
    }
}

impl FunctionValues for BinaryValues<'_> {
    fn exists(&mut self, doc: i32) -> Result<bool> {
        if doc < self.last_doc {
            return Err(out_of_order(self.last_doc, doc));
        }
        self.last_doc = doc;
        let mut cur = self.arr.doc_id();
        if doc > cur {
            cur = self.arr.advance(doc)?;
        }
        Ok(doc == cur)
    }
    /// An empty value counts as none.
    fn bytes_val(&mut self, doc: i32, target: &mut Vec<u8>) -> Result<bool> {
        match self.value(doc)? {
            Some(v) => {
                target.clear();
                target.extend_from_slice(&v);
                Ok(true)
            }
            None => Ok(false),
        }
    }
    fn str_val(&mut self, doc: i32) -> Result<Option<String>> {
        Ok(self
            .value(doc)?
            .map(|v| String::from_utf8_lossy(&v).into_owned()))
    }
    fn object_val(&mut self, doc: i32) -> Result<ObjectVal> {
        Ok(self.str_val(doc)?.map_or(ObjectVal::Null, ObjectVal::Str))
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let s = self.str_val(doc)?;
        Ok(format!(
            "{}={}",
            self.description,
            crate::function::docvalues::opt_str(s)
        ))
    }
    fn new_value(&self) -> MutableValue {
        MutableValue::Str {
            value: Vec::new(),
            exists: true,
        }
    }
    /// `mval.exists = exists(doc); mval.value.clear(); bytesVal(doc,
    /// mval.value)`.
    fn fill_value(&mut self, doc: i32, out: &mut MutableValue) -> Result<()> {
        let exists = self.exists(doc)?;
        let mut value = Vec::new();
        self.bytes_val(doc, &mut value)?;
        *out = MutableValue::Str { value, exists };
        Ok(())
    }
}

impl ValueSource for BytesRefFieldSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        let reader = leaf.reader()?;
        let binary = reader
            .field_infos()
            .field_by_name(&self.field)
            .is_some_and(|fi| {
                fi.doc_values_type == lucene_codecs::field_infos::DocValuesType::Binary
            });
        if binary {
            return Ok(Box::new(BinaryValues {
                arr: dv::get_binary(reader, &self.field)?,
                last_doc: -1,
                description: self.description(),
            }));
        }
        Ok(Box::new(DocTermsIndexDocValues::open(
            self.description(),
            reader,
            &self.field,
        )?))
    }
    fn description(&self) -> String {
        self.field.clone()
    }
}

impl FieldCacheSource for BytesRefFieldSource {
    fn field(&self) -> &str {
        &self.field
    }
}

/// `SortedSetFieldSource`: one term of a `SORTED_SET` field's set, chosen
/// by a selector.
#[derive(Debug, Clone)]
pub struct SortedSetFieldSource {
    field: String,
    selector: SetSelector,
}

impl SortedSetFieldSource {
    /// `new SortedSetFieldSource(field)`: the `MIN` selector.
    pub fn new(field: impl Into<String>) -> Self {
        Self::with_selector(field, SetSelector::Min)
    }

    pub fn with_selector(field: impl Into<String>, selector: SetSelector) -> Self {
        Self {
            field: field.into(),
            selector,
        }
    }
}

impl ValueSource for SortedSetFieldSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        let view = selected_set(leaf, &self.field, self.selector)?;
        Ok(Box::new(DocTermsIndexDocValues::new(
            self.description(),
            view,
        )))
    }
    fn description(&self) -> String {
        format!(
            "sortedset({},selector={})",
            self.field,
            self.selector.name()
        )
    }
    fn native_sort_field(&self, reverse: bool) -> Option<SortField> {
        let mut f = SortField::string(&self.field, reverse);
        f.selector = match self.selector {
            SetSelector::Min => crate::top_field::Selector::Min,
            SetSelector::Max => crate::top_field::Selector::Max,
            SetSelector::MiddleMin => crate::top_field::Selector::MiddleMin,
            SetSelector::MiddleMax => crate::top_field::Selector::MiddleMax,
        };
        Some(f)
    }
}

impl FieldCacheSource for SortedSetFieldSource {
    fn field(&self) -> &str {
        &self.field
    }
}

// ---------------------------------------------------------------------------
// JoinDocFreqValueSource
// ---------------------------------------------------------------------------

/// `JoinDocFreqValueSource`: for a document's `SORTED` term in `field`, the
/// term's reader-wide `docFreq` in `qfield` (`0` without a term, or a term
/// `qfield` lacks).
#[derive(Debug, Clone)]
pub struct JoinDocFreqValueSource {
    field: String,
    qfield: String,
}

impl JoinDocFreqValueSource {
    /// `JoinDocFreqValueSource.NAME`.
    pub const NAME: &'static str = "joindf";

    pub fn new(field: impl Into<String>, qfield: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            qfield: qfield.into(),
        }
    }
}

/// Every term of `field`'s doc values, with its top-level `docFreq` in
/// `qfield` (`MultiTerms.getTerms(top, qfield)`'s `seekExact` and
/// `docFreq`, computed once instead of per document).
struct JoinDocFreqs(HashMap<Vec<u8>, i32>);

struct JoinValues<'a> {
    terms: Box<dyn SortedDocValues + 'a>,
    doc_freqs: Arc<JoinDocFreqs>,
    last_doc: i32,
    description: String,
}

impl IntDocValues for JoinValues<'_> {
    fn description(&self) -> String {
        self.description.clone()
    }
    fn int_val(&mut self, doc: i32) -> Result<i32> {
        if doc < self.last_doc {
            return Err(out_of_order(self.last_doc, doc));
        }
        self.last_doc = doc;
        let mut cur = self.terms.doc_id();
        if doc > cur {
            cur = self.terms.advance(doc)?;
        }
        if doc == cur {
            let term = self.terms.lookup_ord(self.terms.ord_value())?;
            if let Some(&df) = self.doc_freqs.0.get(&term) {
                return Ok(df);
            }
        }
        Ok(0)
    }
}

impl ValueSource for JoinDocFreqValueSource {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        let terms = dv::get_sorted(leaf.reader()?, &self.field)?;
        let doc_freqs = fcx
            .get::<Arc<JoinDocFreqs>, _>(self)
            .cloned()
            .ok_or_else(|| not_weighted(&self.description()))?;
        Ok(Box::new(Int(JoinValues {
            terms,
            doc_freqs,
            last_doc: -1,
            description: self.description(),
        })))
    }
    fn description(&self) -> String {
        format!("{}({}:({}))", Self::NAME, self.field, self.qfield)
    }
    /// Gathers every term of `field` across the leaves and its `docFreq`
    /// in `qfield`, summed over the leaves (`MultiTermsEnum.docFreq`).
    fn create_weight(&self, fcx: &mut FunctionContext, top: &TopLevel<'_>) -> Result<()> {
        let mut map = HashMap::new();
        for leaf in top.leaves() {
            let mut sorted = dv::get_sorted(leaf.reader()?, &self.field)?;
            for ord in 0..sorted.value_count() {
                let term = sorted.lookup_ord(ord)?;
                if let std::collections::hash_map::Entry::Vacant(slot) = map.entry(term) {
                    let df = top.doc_freq(&self.qfield, slot.key())?;
                    if df > 0 {
                        slot.insert(df);
                    }
                }
            }
        }
        fcx.put(self, Arc::new(JoinDocFreqs(map)));
        Ok(())
    }
}

impl FieldCacheSource for JoinDocFreqValueSource {
    fn field(&self) -> &str {
        &self.field
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The selector views' iterator methods the field sources do not use
    /// (they only `advance`), over the `GenFunction` fixture.
    #[test]
    fn selector_views_iterate_as_their_inner_values() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/data/function/index");
        let reader =
            crate::directory_reader::DirectoryReader::open(&lucene_store::FsDirectory::open(dir))
                .unwrap();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let leaf = ValueLeaf::of_segment(&segments[0]);
        for (field, sortable) in [
            ("mi", Sortable::Plain),
            ("mf", Sortable::Float),
            ("md", Sortable::Double),
        ] {
            for selector in [NumericSelector::Min, NumericSelector::Max] {
                let mut a = selected_numeric(&leaf, field, selector, sortable).unwrap();
                let mut b = selected_numeric(&leaf, field, selector, sortable).unwrap();
                assert!(a.cost() > 0);
                let mut doc = a.next_doc().unwrap();
                while doc != NO_MORE_DOCS {
                    assert!(b.advance_exact(doc).unwrap());
                    assert_eq!(a.long_value(), b.long_value());
                    doc = a.next_doc().unwrap();
                }
                assert_eq!(a.doc_id(), NO_MORE_DOCS);
            }
        }
        for selector in [
            SetSelector::Min,
            SetSelector::Max,
            SetSelector::MiddleMin,
            SetSelector::MiddleMax,
        ] {
            let mut a = selected_set(&leaf, "ss", selector).unwrap();
            let mut b = selected_set(&leaf, "ss", selector).unwrap();
            assert!(a.cost() > 0 && a.value_count() > 0);
            let mut doc = a.next_doc().unwrap();
            while doc != NO_MORE_DOCS {
                assert!(b.advance_exact(doc).unwrap());
                assert_eq!(a.ord_value(), b.ord_value());
                let term = a.lookup_ord(a.ord_value()).unwrap();
                assert_eq!(a.lookup_term(&term).unwrap(), a.ord_value());
                doc = a.next_doc().unwrap();
            }
            assert!(a.lookup_term(b"zzzz").unwrap() < 0);
            assert!(!b.advance_exact(i32::MAX - 1).unwrap_or(false));
        }
        assert_eq!(SetSelector::MiddleMax.name(), "MIDDLE_MAX");
        assert_eq!(NumericSelector::Max.name(), "MAX");
    }
}
