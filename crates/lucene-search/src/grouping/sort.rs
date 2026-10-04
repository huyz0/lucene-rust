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

use crate::multi_segment::OpenSegment;
use crate::reader::doc_values as dv;
use crate::reader::{SortedNumericDocValues, SortedSetDocValues};
use crate::top_field::{SortField, SortType};
use crate::{Error, Result};

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

/// One key's reading of one segment (`getLeafComparator(context)`).
enum LeafKey<'a> {
    Score,
    Doc(i32),
    Numeric(Box<dyn SortedNumericDocValues + 'a>, Vec<i64>),
    Str(Box<dyn SortedSetDocValues + 'a>, Vec<i64>),
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
                    LeafKey::Numeric(dv::get_sorted_numeric(reader, &f.field)?, Vec::new())
                }
                SortType::String => {
                    let reader = leaf.reader.ok_or_else(|| {
                        Error::MissingSegmentReader(format!("sort field {}", f.field))
                    })?;
                    LeafKey::Str(dv::get_sorted_set(reader, &f.field)?, Vec::new())
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
            LeafKey::Numeric(values, buf) => {
                buf.clear();
                if values.advance_exact(doc)? {
                    for _ in 0..values.doc_value_count() {
                        buf.push(values.next_value()?);
                    }
                }
                let v = f.selector.pick(f.ty, buf).unwrap_or(f.missing);
                match f.ty {
                    SortType::Long => GroupSortValue::Long(v),
                    SortType::Int => GroupSortValue::Int(v as i32),
                    SortType::Double => GroupSortValue::Double(sortable_long_to_double(v)),
                    _ => GroupSortValue::Float(sortable_int_to_float(v)),
                }
            }
            LeafKey::Str(values, buf) => {
                buf.clear();
                if values.advance_exact(doc)? {
                    for _ in 0..values.doc_value_count() {
                        buf.push(values.next_ord()?);
                    }
                }
                match f.selector.pick_ord(buf) {
                    Some(ord) => GroupSortValue::Bytes(Some(values.lookup_ord(ord)?)),
                    None => GroupSortValue::Bytes(None),
                }
            }
        })
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
