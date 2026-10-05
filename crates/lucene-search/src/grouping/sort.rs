//! The sorts grouping orders groups and documents by: `Sort` and the
//! `FieldComparator`s grouping drives slot by slot (`getComparator(numHits,
//! Pruning.NONE)`: `copy`, `compare`, `setBottom`, `compareBottom`, `value`,
//! `compareValues`).
//!
//! A key is a [`SortField`] of [`crate::top_field`]: the score
//! (`RelevanceComparator`), the document (`DocComparator`), a numeric
//! doc-values field (`LONG`/`INT`/`DOUBLE`/`FLOAT` through `NUMERIC` or
//! `SORTED_NUMERIC` doc values and the key's selector, as
//! `SortedNumericSortField` reads them), or a keyword field (`STRING`,
//! `TermOrdValComparator`, through `SORTED` or `SORTED_SET` doc values and
//! the selector). A slot holds the value Java's `value(slot)` returns
//! ([`GroupSortValue`]), and every comparison is `compareValues` over two of
//! them -- which orders exactly as Java's slot comparisons do: a keyword
//! comparator compares ordinals within a segment and terms across them, and
//! ordinals are in term order.
//!
//! Not supported here: `STRING_VAL` and custom keys
//! ([`Error::Unsupported`]).

use std::cmp::Ordering;

use super::FxHashMap;
use crate::multi_segment::OpenSegment;
use crate::reader::doc_values::{self as dv, SortedOrds};
use crate::reader::{NumericDocValues, SortedNumericDocValues, SortedSetDocValues};
use crate::top_field::{SortField, SortType};
use crate::{Error, Result};
use lucene_codecs::field_infos::DocValuesType;

/// `Sort`: the keys, in priority order. `Sort.RELEVANCE` is
/// [`Sort::relevance`], which [`TopGroupsCollector`](super::TopGroupsCollector)
/// tells apart by identity as Java does (`withinGroupSort ==
/// Sort.RELEVANCE`); everywhere else two sorts are equal when their keys
/// are (`Sort.equals`).
#[derive(Debug, Clone)]
pub struct Sort {
    pub fields: Vec<SortField>,
    relevance: bool,
}

impl PartialEq for Sort {
    fn eq(&self, o: &Self) -> bool {
        self.fields == o.fields
    }
}

impl Sort {
    /// `Sort.RELEVANCE`: by score, highest first.
    pub fn relevance() -> Self {
        Self {
            fields: vec![SortField::score()],
            relevance: true,
        }
    }

    /// `Sort.INDEXORDER`: by document id.
    pub fn index_order() -> Self {
        Self::new(vec![SortField::doc()])
    }

    /// `new Sort(fields...)`.
    pub fn new(fields: Vec<SortField>) -> Self {
        Self {
            fields,
            relevance: false,
        }
    }

    /// Whether this is `Sort.RELEVANCE` itself (Java's `==`).
    pub fn is_relevance_singleton(&self) -> bool {
        self.relevance
    }

    /// `equals(Sort.RELEVANCE)`.
    pub fn is_relevance(&self) -> bool {
        *self == Self::relevance()
    }

    /// `needsScores()`.
    pub fn needs_scores(&self) -> bool {
        self.fields.iter().any(|f| f.ty == SortType::Score)
    }

    /// The `reverseMul`s.
    pub(crate) fn reversed(&self) -> Vec<i32> {
        self.fields
            .iter()
            .map(|f| if f.reverse { -1 } else { 1 })
            .collect()
    }
}

/// A key's value as Java's `FieldComparator.value(slot)` boxes it.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupSortValue {
    /// The score (`Float`), or a `FLOAT` key's value.
    Float(f32),
    Double(f64),
    Long(i64),
    /// The document (`Integer`, its global id), or an `INT` key's value.
    Int(i32),
    /// A keyword key's term, `None` for a document without one.
    Bytes(Option<Vec<u8>>),
}

/// `Float.compare`.
pub(crate) fn float_compare(a: f32, b: f32) -> Ordering {
    float_to_sortable_int(a).cmp(&float_to_sortable_int(b))
}

/// `Double.compare`.
fn double_compare(a: f64, b: f64) -> Ordering {
    double_to_sortable_long(a).cmp(&double_to_sortable_long(b))
}

fn float_to_sortable_int(f: f32) -> i32 {
    let bits = if f.is_nan() {
        0x7fc0_0000
    } else {
        f.to_bits() as i32
    };
    bits ^ ((bits >> 31) & 0x7fff_ffff)
}

fn double_to_sortable_long(d: f64) -> i64 {
    let bits = if d.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        d.to_bits() as i64
    };
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
}

fn sortable_int_to_float(v: i64) -> f32 {
    let i = v as i32;
    f32::from_bits((i ^ ((i >> 31) & 0x7fff_ffff)) as u32)
}

fn sortable_long_to_double(v: i64) -> f64 {
    f64::from_bits((v ^ ((v >> 63) & 0x7fff_ffff_ffff_ffff)) as u64)
}

/// `FieldComparator.compareValues(first, second)` for `field`'s comparator
/// (ascending, before `reverse`).
pub fn compare_values(field: &SortField, a: &GroupSortValue, b: &GroupSortValue) -> Ordering {
    use GroupSortValue as V;
    match (a, b) {
        // `RelevanceComparator.compareValues`: `Float.compare(second, first)`.
        (V::Float(x), V::Float(y)) if field.ty == SortType::Score => float_compare(*y, *x),
        (V::Float(x), V::Float(y)) => float_compare(*x, *y),
        (V::Double(x), V::Double(y)) => double_compare(*x, *y),
        (V::Long(x), V::Long(y)) => x.cmp(y),
        (V::Int(x), V::Int(y)) => x.cmp(y),
        // `TermOrdValComparator.compareValues`: a missing value sorts first
        // (`STRING_FIRST`) or last (`STRING_LAST`).
        (V::Bytes(x), V::Bytes(y)) => {
            let missing_last = field.missing == 1;
            match (x, y) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => {
                    if missing_last {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                }
                (Some(_), None) => {
                    if missing_last {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                }
                (Some(x), Some(y)) => x.cmp(y),
            }
        }
        // Values of different keys never meet; order them stably.
        _ => Ordering::Equal,
    }
}

/// A numeric key's comparable long as the value its comparator boxes.
fn numeric_value(ty: SortType, v: i64) -> GroupSortValue {
    match ty {
        SortType::Long => GroupSortValue::Long(v),
        SortType::Int => GroupSortValue::Int(v as i32),
        SortType::Double => GroupSortValue::Double(sortable_long_to_double(v)),
        _ => GroupSortValue::Float(sortable_int_to_float(v)),
    }
}

/// `compareValues(slot, value)` for a numeric key's comparable long.
fn compare_numeric(ty: SortType, slot: &GroupSortValue, v: i64) -> Ordering {
    match (ty, slot) {
        (SortType::Long, GroupSortValue::Long(b)) => b.cmp(&v),
        (SortType::Int, GroupSortValue::Int(b)) => b.cmp(&(v as i32)),
        (SortType::Double, GroupSortValue::Double(b)) => {
            double_compare(*b, sortable_long_to_double(v))
        }
        (SortType::Float, GroupSortValue::Float(b)) => float_compare(*b, sortable_int_to_float(v)),
        _ => Ordering::Equal,
    }
}

/// One key's reading of one segment (`getLeafComparator(context)`).
enum LeafKey<'a> {
    Score,
    Doc(i32),
    Numeric(Box<dyn SortedNumericDocValues + 'a>, Vec<i64>),
    /// A numeric key over a `NUMERIC` field: one value per document, read
    /// directly (the comparator's `DocValues.getNumeric`), its missing value
    /// substituted.
    Single(Box<dyn NumericDocValues + 'a>),
    /// [`LeafKey::Single`] over a segment's own column, read without an
    /// iterator object between (stage 3: the same values).
    Direct(Box<lucene_codecs::doc_values::NumericReader<'a>>),
    /// The values, and the slot terms placed in this segment with their
    /// ordinals: a term copied from this segment's document keeps its
    /// ordinal, any other is resolved once by `lookupTerm`
    /// (`TermOrdValComparator`'s `bottomOrd`/`bottomSameReader`).
    Str(StrValues<'a>, FxHashMap<Vec<u8>, i64>),
}

/// A keyword key's values: a `SORTED_SET` field through its selector, or a
/// `SORTED` field's one ordinal read directly.
enum StrValues<'a> {
    Set(Box<dyn SortedSetDocValues + 'a>, Vec<i64>),
    Single(Box<SortedOrds<'a>>),
}

impl StrValues<'_> {
    /// The ordinal the key's selector picks for `doc`, if it has one.
    fn doc_ord(&mut self, doc: i32, selector: crate::top_field::Selector) -> Result<Option<i64>> {
        match self {
            StrValues::Single(v) => Ok(v.ord(doc)?.map(i64::from)),
            StrValues::Set(v, buf) => {
                buf.clear();
                if v.advance_exact(doc)? {
                    for _ in 0..v.doc_value_count() {
                        buf.push(v.next_ord()?);
                    }
                }
                Ok(selector.pick_ord(buf))
            }
        }
    }

    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        match self {
            StrValues::Single(v) => v.dict().lookup_ord(i32::try_from(ord).unwrap_or(i32::MAX)),
            StrValues::Set(v, _) => v.lookup_ord(ord),
        }
    }

    // SENTINEL: negative = absent, `-insertionPoint-1` (`lookupTerm`).
    fn lookup_term(&mut self, term: &[u8]) -> Result<i64> {
        match self {
            StrValues::Single(v) => Ok(i64::from(v.dict().lookup_term(term)?)),
            StrValues::Set(v, _) => v.lookup_term(term),
        }
    }
}

/// The keys of a [`Sort`] over one segment: each document's values.
pub(crate) struct LeafKeys<'a> {
    keys: Vec<LeafKey<'a>>,
}

impl<'a> LeafKeys<'a> {
    /// The leaf comparators of `sort` for `leaf`.
    ///
    /// # Errors
    /// A key this module does not sort by, a field with doc values of
    /// another kind, or a segment without a reader.
    pub(crate) fn open(sort: &Sort, leaf: &OpenSegment<'a>) -> Result<Self> {
        let mut keys = Vec::with_capacity(sort.fields.len());
        for f in &sort.fields {
            keys.push(match f.ty {
                SortType::Score => LeafKey::Score,
                SortType::Doc => LeafKey::Doc(leaf.doc_base),
                SortType::Long | SortType::Int | SortType::Double | SortType::Float => {
                    let reader = leaf.reader.ok_or_else(|| {
                        Error::MissingSegmentReader(format!("sort field {}", f.field))
                    })?;
                    let single = reader
                        .field_infos()
                        .field_by_name(&f.field)
                        .is_some_and(|fi| fi.doc_values_type == DocValuesType::Numeric);
                    if single {
                        match dv::direct_numeric(reader, &f.field) {
                            Some(r) => LeafKey::Direct(Box::new(r)),
                            None => LeafKey::Single(dv::get_numeric(reader, &f.field)?),
                        }
                    } else {
                        LeafKey::Numeric(dv::get_sorted_numeric(reader, &f.field)?, Vec::new())
                    }
                }
                SortType::String => {
                    let reader = leaf.reader.ok_or_else(|| {
                        Error::MissingSegmentReader(format!("sort field {}", f.field))
                    })?;
                    let single = reader
                        .field_infos()
                        .field_by_name(&f.field)
                        .is_some_and(|fi| fi.doc_values_type == DocValuesType::Sorted);
                    let values = if single {
                        StrValues::Single(Box::new(SortedOrds::open(reader, &f.field)?))
                    } else {
                        StrValues::Set(dv::get_sorted_set(reader, &f.field)?, Vec::new())
                    };
                    LeafKey::Str(values, FxHashMap::default())
                }
                SortType::StringVal | SortType::Custom(_) => {
                    return Err(Error::Unsupported(format!(
                        "grouping does not sort by {:?} keys",
                        f.ty
                    )))
                }
            });
        }
        Ok(Self { keys })
    }

    /// Key `i`'s value for segment document `doc` scoring `score` (what
    /// `copy(slot, doc)` stores).
    pub(crate) fn value(
        &mut self,
        sort: &Sort,
        i: usize,
        doc: i32,
        score: f32,
    ) -> Result<GroupSortValue> {
        let f = &sort.fields[i];
        Ok(match &mut self.keys[i] {
            LeafKey::Score => GroupSortValue::Float(score),
            LeafKey::Doc(base) => GroupSortValue::Int(base.saturating_add(doc)),
            LeafKey::Single(values) => {
                let v = if values.advance_exact(doc)? {
                    values.long_value()
                } else {
                    f.missing
                };
                numeric_value(f.ty, v)
            }
            LeafKey::Direct(r) => numeric_value(f.ty, r.value(doc)?.unwrap_or(f.missing)),
            LeafKey::Numeric(values, buf) => {
                buf.clear();
                if values.advance_exact(doc)? {
                    for _ in 0..values.doc_value_count() {
                        buf.push(values.next_value()?);
                    }
                }
                let v = f.selector.pick(f.ty, buf).unwrap_or(f.missing);
                numeric_value(f.ty, v)
            }
            LeafKey::Str(values, placed) => match values.doc_ord(doc, f.selector)? {
                Some(ord) => {
                    let term = values.lookup_ord(ord)?;
                    placed.insert(term.clone(), ord);
                    GroupSortValue::Bytes(Some(term))
                }
                None => GroupSortValue::Bytes(None),
            },
        })
    }

    /// `compareValues(slot, doc's value)` for key `i` without boxing the
    /// document's value where the comparator does not (`compareBottom`):
    /// the score and the document compared as numbers, a numeric key as its
    /// selected long, a keyword key by ordinal against the slot term's
    /// position in this segment (looked up once per slot term).
    pub(crate) fn compare_slot(
        &mut self,
        sort: &Sort,
        i: usize,
        slot: &GroupSortValue,
        doc: i32,
        score: f32,
    ) -> Result<Ordering> {
        let f = &sort.fields[i];
        match (&mut self.keys[i], slot) {
            (LeafKey::Score, GroupSortValue::Float(b)) => Ok(float_compare(score, *b)),
            (LeafKey::Doc(base), GroupSortValue::Int(b)) => Ok(b.cmp(&base.saturating_add(doc))),
            (LeafKey::Single(values), _) => {
                let v = if values.advance_exact(doc)? {
                    values.long_value()
                } else {
                    f.missing
                };
                Ok(compare_numeric(f.ty, slot, v))
            }
            (LeafKey::Direct(r), _) => Ok(compare_numeric(
                f.ty,
                slot,
                r.value(doc)?.unwrap_or(f.missing),
            )),
            (LeafKey::Numeric(values, buf), _) => {
                buf.clear();
                if values.advance_exact(doc)? {
                    for _ in 0..values.doc_value_count() {
                        buf.push(values.next_value()?);
                    }
                }
                let v = f.selector.pick(f.ty, buf).unwrap_or(f.missing);
                Ok(compare_numeric(f.ty, slot, v))
            }
            (LeafKey::Str(values, placed), GroupSortValue::Bytes(slot_term)) => {
                let doc_ord = values.doc_ord(doc, f.selector)?;
                let missing_last = f.missing == 1;
                Ok(match (slot_term, doc_ord) {
                    (None, None) => Ordering::Equal,
                    (None, Some(_)) => {
                        if missing_last {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    }
                    (Some(_), None) => {
                        if missing_last {
                            Ordering::Less
                        } else {
                            Ordering::Greater
                        }
                    }
                    (Some(term), Some(ord)) => {
                        let at = match placed.get(term.as_slice()) {
                            Some(&at) => at,
                            None => {
                                let at = values.lookup_term(term)?;
                                placed.insert(term.clone(), at);
                                at
                            }
                        };
                        // SENTINEL: a negative answer is `-insertionPoint - 1`:
                        // the slot term sorts between that point's
                        // neighbours, after every ordinal below it.
                        if at >= 0 {
                            at.cmp(&ord)
                        } else if ord < at.saturating_neg().saturating_sub(1) {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    }
                })
            }
            _ => {
                let v = self.value(sort, i, doc, score)?;
                Ok(compare_values(f, slot, &v))
            }
        }
    }

    /// `reverseMul * compareBottom(doc)` over every key, in order, against
    /// the values of a slot: the first that differs ([`compare_all`] with
    /// the document's values read as the comparators read them).
    #[inline(always)]
    pub(crate) fn compare_doc(
        &mut self,
        sort: &Sort,
        reversed: &[i32],
        slot: &[GroupSortValue],
        doc: i32,
        score: f32,
    ) -> Result<Ordering> {
        // By score alone (`Sort.RELEVANCE`, the default): one float compare.
        if let ([LeafKey::Score], [GroupSortValue::Float(b)], [r]) =
            (self.keys.as_slice(), slot, reversed)
        {
            let c = float_compare(score, *b);
            return Ok(if *r < 0 { c.reverse() } else { c });
        }
        // By one `NUMERIC` field read off its column: one load and compare.
        if let ([LeafKey::Direct(values)], [s], [r], [f]) = (
            self.keys.as_mut_slice(),
            slot,
            reversed,
            sort.fields.as_slice(),
        ) {
            let c = compare_numeric(f.ty, s, values.value(doc)?.unwrap_or(f.missing));
            return Ok(if *r < 0 { c.reverse() } else { c });
        }
        self.compare_doc_keys(sort, reversed, slot, doc, score)
    }

    /// [`Self::compare_doc`] key by key.
    #[inline(never)]
    fn compare_doc_keys(
        &mut self,
        sort: &Sort,
        reversed: &[i32],
        slot: &[GroupSortValue],
        doc: i32,
        score: f32,
    ) -> Result<Ordering> {
        for (i, s) in slot.iter().enumerate().take(sort.fields.len()) {
            let c = self.compare_slot(sort, i, s, doc, score)?;
            let c = if reversed[i] < 0 { c.reverse() } else { c };
            if c != Ordering::Equal {
                return Ok(c);
            }
        }
        Ok(Ordering::Equal)
    }

    /// Every key's value for `doc`, into `out` (its allocation reused).
    pub(crate) fn values_into(
        &mut self,
        sort: &Sort,
        doc: i32,
        score: f32,
        out: &mut Vec<GroupSortValue>,
    ) -> Result<()> {
        out.clear();
        for i in 0..sort.fields.len() {
            out.push(self.value(sort, i, doc, score)?);
        }
        Ok(())
    }

    /// Every key's value for `doc`.
    pub(crate) fn values(
        &mut self,
        sort: &Sort,
        doc: i32,
        score: f32,
    ) -> Result<Vec<GroupSortValue>> {
        (0..sort.fields.len())
            .map(|i| self.value(sort, i, doc, score))
            .collect()
    }
}

/// `reverseMul * compareValues` over every key, in order: the first that
/// differs.
pub(crate) fn compare_all(
    sort: &Sort,
    reversed: &[i32],
    a: &[GroupSortValue],
    b: &[GroupSortValue],
) -> Ordering {
    for (i, f) in sort.fields.iter().enumerate() {
        let c = compare_values(f, &a[i], &b[i]);
        let c = if reversed[i] < 0 { c.reverse() } else { c };
        if c != Ordering::Equal {
            return c;
        }
    }
    Ordering::Equal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_compare_as_java_boxes_do() {
        let score = SortField::score();
        assert_eq!(
            compare_values(
                &score,
                &GroupSortValue::Float(1.0),
                &GroupSortValue::Float(2.0)
            ),
            Ordering::Greater
        );
        let f = SortField::numeric("f", SortType::Float, false);
        assert_eq!(
            compare_values(
                &f,
                &GroupSortValue::Float(-0.0),
                &GroupSortValue::Float(0.0)
            ),
            Ordering::Less
        );
        assert_eq!(
            compare_values(
                &f,
                &GroupSortValue::Float(f32::NAN),
                &GroupSortValue::Float(f32::INFINITY)
            ),
            Ordering::Greater
        );
        let d = SortField::numeric("d", SortType::Double, false);
        assert_eq!(
            compare_values(
                &d,
                &GroupSortValue::Double(1.5),
                &GroupSortValue::Double(1.5)
            ),
            Ordering::Equal
        );
        let mut s = SortField::string("s", false);
        let none = GroupSortValue::Bytes(None);
        let a = GroupSortValue::Bytes(Some(b"a".to_vec()));
        assert_eq!(compare_values(&s, &none, &a), Ordering::Less);
        assert_eq!(compare_values(&s, &a, &none), Ordering::Greater);
        assert_eq!(compare_values(&s, &none, &none), Ordering::Equal);
        s.missing = 1;
        assert_eq!(compare_values(&s, &none, &a), Ordering::Greater);
        assert_eq!(compare_values(&s, &a, &none), Ordering::Less);
        assert_eq!(
            compare_values(&s, &a, &GroupSortValue::Long(1)),
            Ordering::Equal
        );
        assert_eq!(
            sortable_int_to_float(i64::from(float_to_sortable_int(-2.5))),
            -2.5
        );
        assert_eq!(sortable_long_to_double(double_to_sortable_long(-2.5)), -2.5);
    }

    #[test]
    fn relevance_is_told_apart_by_identity_and_by_keys() {
        let r = Sort::relevance();
        let same = Sort::new(vec![SortField::score()]);
        assert!(r.is_relevance_singleton() && !same.is_relevance_singleton());
        assert!(r.is_relevance() && same.is_relevance());
        assert_eq!(r, same);
        assert!(r.needs_scores());
        assert!(!Sort::index_order().needs_scores());
        assert_eq!(
            Sort::new(vec![SortField::doc(), SortField::string("x", true)]).reversed(),
            vec![1, -1]
        );
    }
}
