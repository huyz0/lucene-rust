//! `org.apache.lucene.index.DocValues`'s leaf getters: a field's doc values
//! of the kind asked for, an **empty** iterator when the leaf has none, and
//! Java's `IllegalStateException` (here [`Error::IllegalState`], with its
//! message) when the field exists but was indexed with doc values of another
//! kind -- or with none.
//!
//! - [`get_sorted`][]: `DocValues.getSorted`;
//! - [`get_sorted_set`]: `DocValues.getSortedSet` (a `SORTED` field read as
//!   a one-value-per-document set, `DocValues.singleton`);
//! - [`get_numeric`][]: `DocValues.getNumeric`;
//! - [`get_sorted_numeric`]: `DocValues.getSortedNumeric` (a `NUMERIC` field
//!   as a singleton).
//!
//! The query-time joins ([`crate::join`]) and the grouping collectors
//! ([`crate::grouping`]) read every leaf through these, as their Java
//! originals do.

use lucene_codecs::field_infos::DocValuesType;

use super::{
    DocIdSetIterator, DocValuesIterator, LeafReader, NumericDocValues, SortedDocValues,
    SortedNumericDocValues, SortedSetDocValues, NO_MORE_DOCS,
};
use crate::{Error, Result};

/// `DocValuesType.toString()`.
pub fn doc_values_type_name(t: DocValuesType) -> &'static str {
    match t {
        DocValuesType::None => "NONE",
        DocValuesType::Numeric => "NUMERIC",
        DocValuesType::Binary => "BINARY",
        DocValuesType::Sorted => "SORTED",
        DocValuesType::SortedSet => "SORTED_SET",
        DocValuesType::SortedNumeric => "SORTED_NUMERIC",
    }
}

/// `DocValues.checkField`: an error when `field` exists in the leaf (its
/// doc values, if any, are not of an `expected` kind), nothing otherwise.
fn check_field<R: LeafReader + ?Sized>(
    reader: &R,
    field: &str,
    expected: &[DocValuesType],
) -> Result<()> {
    let Some(fi) = reader.field_infos().field_by_name(field) else {
        return Ok(());
    };
    let expected = if let [one] = expected {
        format!("(expected={}", doc_values_type_name(*one))
    } else {
        let names: Vec<&str> = expected.iter().map(|t| doc_values_type_name(*t)).collect();
        format!("(expected one of [{}]", names.join(", "))
    };
    Err(Error::IllegalState(format!(
        "unexpected docvalues type {} for field '{field}' {expected}). Re-index with correct \
         docvalues type.",
        doc_values_type_name(fi.doc_values_type)
    )))
}

/// `DocValues.getSorted(reader, field)`.
///
/// # Errors
/// [`Error::IllegalState`] when the field exists without `SORTED` doc values.
pub fn get_sorted<'a, R: LeafReader + ?Sized>(
    reader: &'a R,
    field: &str,
) -> Result<Box<dyn SortedDocValues + 'a>> {
    match reader.sorted_doc_values(field)? {
        Some(dv) => Ok(dv),
        None => {
            check_field(reader, field, &[DocValuesType::Sorted])?;
            Ok(Box::new(Empty::default()))
        }
    }
}

/// `DocValues.getSortedSet(reader, field)`.
///
/// # Errors
/// [`Error::IllegalState`] when the field exists without `SORTED` or
/// `SORTED_SET` doc values.
pub fn get_sorted_set<'a, R: LeafReader + ?Sized>(
    reader: &'a R,
    field: &str,
) -> Result<Box<dyn SortedSetDocValues + 'a>> {
    if let Some(dv) = reader.sorted_set_doc_values(field)? {
        return Ok(dv);
    }
    match reader.sorted_doc_values(field)? {
        Some(sorted) => Ok(Box::new(SingletonSortedSet {
            inner: sorted,
            pending: false,
        })),
        None => {
            check_field(
                reader,
                field,
                &[DocValuesType::Sorted, DocValuesType::SortedSet],
            )?;
            Ok(Box::new(Empty::default()))
        }
    }
}

/// `DocValues.getNumeric(reader, field)`.
///
/// # Errors
/// [`Error::IllegalState`] when the field exists without `NUMERIC` doc
/// values.
pub fn get_numeric<'a, R: LeafReader + ?Sized>(
    reader: &'a R,
    field: &str,
) -> Result<Box<dyn NumericDocValues + 'a>> {
    match reader.numeric_doc_values(field)? {
        Some(dv) => Ok(dv),
        None => {
            check_field(reader, field, &[DocValuesType::Numeric])?;
            Ok(Box::new(Empty::default()))
        }
    }
}

/// `DocValues.getSortedNumeric(reader, field)`.
///
/// # Errors
/// [`Error::IllegalState`] when the field exists without `SORTED_NUMERIC` or
/// `NUMERIC` doc values.
pub fn get_sorted_numeric<'a, R: LeafReader + ?Sized>(
    reader: &'a R,
    field: &str,
) -> Result<Box<dyn SortedNumericDocValues + 'a>> {
    if let Some(dv) = reader.sorted_numeric_doc_values(field)? {
        return Ok(dv);
    }
    match reader.numeric_doc_values(field)? {
        Some(single) => Ok(Box::new(SingletonSortedNumeric {
            inner: single,
            pending: false,
        })),
        None => {
            check_field(
                reader,
                field,
                &[DocValuesType::SortedNumeric, DocValuesType::Numeric],
            )?;
            Ok(Box::new(Empty::default()))
        }
    }
}

/// `DocValues.emptySorted()` / `emptySortedSet()` / `emptyNumeric()` /
/// `emptySortedNumeric()`: no document has a value.
#[derive(Debug, Default)]
struct Empty {
    doc: i32,
    started: bool,
}

impl DocIdSetIterator for Empty {
    // SENTINEL: `-1` = unpositioned, `DocIdSetIterator.docID()`'s contract.
    fn doc_id(&self) -> i32 {
        if self.started {
            self.doc
        } else {
            -1
        }
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.started = true;
        self.doc = NO_MORE_DOCS;
        Ok(NO_MORE_DOCS)
    }
    fn advance(&mut self, _target: i32) -> Result<i32> {
        self.next_doc()
    }
    fn cost(&self) -> i64 {
        0
    }
}

impl DocValuesIterator for Empty {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.started = true;
        self.doc = target;
        Ok(false)
    }
}

impl SortedDocValues for Empty {
    fn ord_value(&self) -> i32 {
        -1
    }
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
        Err(Error::IllegalArgument(format!(
            "ord {ord} is out of bounds of an empty doc values"
        )))
    }
    fn value_count(&self) -> i32 {
        0
    }
}

impl SortedSetDocValues for Empty {
    fn doc_value_count(&self) -> i32 {
        0
    }
    fn next_ord(&mut self) -> Result<i64> {
        Err(Error::IllegalState(
            "nextOrd called more than docValueCount times".into(),
        ))
    }
    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        Err(Error::IllegalArgument(format!(
            "ord {ord} is out of bounds of an empty doc values"
        )))
    }
    fn value_count(&self) -> i64 {
        0
    }
}

impl NumericDocValues for Empty {
    fn long_value(&self) -> i64 {
        0
    }
}

impl SortedNumericDocValues for Empty {
    fn doc_value_count(&self) -> i32 {
        0
    }
    fn next_value(&mut self) -> Result<i64> {
        Err(Error::IllegalState(
            "nextValue called more than docValueCount times".into(),
        ))
    }
}

/// `DocValues.singleton(SortedDocValues)`: a `SORTED` field as a set of one
/// ordinal per document.
struct SingletonSortedSet<'a> {
    inner: Box<dyn SortedDocValues + 'a>,
    /// Whether the current document's one ordinal is still to be read.
    pending: bool,
}

impl DocIdSetIterator for SingletonSortedSet<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        let d = self.inner.next_doc()?;
        self.pending = d != NO_MORE_DOCS;
        Ok(d)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let d = self.inner.advance(target)?;
        self.pending = d != NO_MORE_DOCS;
        Ok(d)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
}

impl DocValuesIterator for SingletonSortedSet<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        let found = self.inner.advance_exact(target)?;
        self.pending = found;
        Ok(found)
    }
}

impl SortedSetDocValues for SingletonSortedSet<'_> {
    fn doc_value_count(&self) -> i32 {
        1
    }
    fn next_ord(&mut self) -> Result<i64> {
        if !self.pending {
            return Err(Error::IllegalState(
                "nextOrd called more than docValueCount times".into(),
            ));
        }
        self.pending = false;
        Ok(i64::from(self.inner.ord_value()))
    }
    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        let ord = i32::try_from(ord)
            .map_err(|_| Error::IllegalArgument(format!("ord {ord} is out of bounds")))?;
        self.inner.lookup_ord(ord)
    }
    fn value_count(&self) -> i64 {
        i64::from(self.inner.value_count())
    }
    fn lookup_term(&mut self, key: &[u8]) -> Result<i64> {
        Ok(i64::from(self.inner.lookup_term(key)?))
    }
}

/// `DocValues.singleton(NumericDocValues)`: a `NUMERIC` field as a set of
/// one value per document.
struct SingletonSortedNumeric<'a> {
    inner: Box<dyn NumericDocValues + 'a>,
    pending: bool,
}

impl DocIdSetIterator for SingletonSortedNumeric<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        let d = self.inner.next_doc()?;
        self.pending = d != NO_MORE_DOCS;
        Ok(d)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let d = self.inner.advance(target)?;
        self.pending = d != NO_MORE_DOCS;
        Ok(d)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
}

impl DocValuesIterator for SingletonSortedNumeric<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        let found = self.inner.advance_exact(target)?;
        self.pending = found;
        Ok(found)
    }
}

impl SortedNumericDocValues for SingletonSortedNumeric<'_> {
    fn doc_value_count(&self) -> i32 {
        1
    }
    fn next_value(&mut self) -> Result<i64> {
        if !self.pending {
            return Err(Error::IllegalState(
                "nextValue called more than docValueCount times".into(),
            ));
        }
        self.pending = false;
        Ok(self.inner.long_value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_values_have_no_documents() {
        let mut e = Empty::default();
        assert_eq!(DocIdSetIterator::doc_id(&e), -1);
        assert!(!e.advance_exact(3).unwrap());
        assert_eq!(DocIdSetIterator::doc_id(&e), 3);
        assert_eq!(e.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(e.advance(9).unwrap(), NO_MORE_DOCS);
        assert_eq!(DocIdSetIterator::cost(&e), 0);
        assert_eq!(SortedDocValues::value_count(&e), 0);
        assert_eq!(SortedSetDocValues::value_count(&e), 0);
        assert_eq!(e.ord_value(), -1);
        assert!(SortedDocValues::lookup_ord(&mut e, 0).is_err());
        assert!(SortedSetDocValues::lookup_ord(&mut e, 0).is_err());
        assert!(e.next_ord().is_err());
        assert!(e.next_value().is_err());
        assert_eq!(e.long_value(), 0);
        assert_eq!(SortedSetDocValues::doc_value_count(&e), 0);
        assert_eq!(SortedNumericDocValues::doc_value_count(&e), 0);
        assert_eq!(SortedDocValues::lookup_term(&mut e, b"x").unwrap(), -1);
        assert_eq!(SortedSetDocValues::lookup_term(&mut e, b"x").unwrap(), -1);
    }

    #[test]
    fn type_names_are_javas() {
        for (t, n) in [
            (DocValuesType::None, "NONE"),
            (DocValuesType::Numeric, "NUMERIC"),
            (DocValuesType::Binary, "BINARY"),
            (DocValuesType::Sorted, "SORTED"),
            (DocValuesType::SortedSet, "SORTED_SET"),
            (DocValuesType::SortedNumeric, "SORTED_NUMERIC"),
        ] {
            assert_eq!(doc_values_type_name(t), n);
        }
    }
}
