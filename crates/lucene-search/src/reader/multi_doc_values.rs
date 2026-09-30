//! Port of `org.apache.lucene.index.MultiDocValues`: one doc-values
//! iterator over every leaf of a composite reader, in top-level doc ids --
//! `getNormValues`, `getNumericValues`, `getBinaryValues`,
//! `getSortedNumericValues`, `getSortedValues` and `getSortedSetValues`.
//!
//! The sorted views map each leaf's ordinals to global ones through an
//! [`OrdinalMap`], as Java's `MultiSortedDocValues`/`MultiSortedSetDocValues`
//! do; `lookupOrd` reads the term from the first leaf holding it.
//!
//! Every leaf's values are read through its own iterator, positioned
//! forward, never through a per-document column lookup
//! (docs/mechanical-gates.md, "doc-values per doc").
//!
//! # What differs from Java
//!
//! Java opens each leaf's numeric/binary/norms iterator only when iteration
//! reaches that leaf; this opens them all up front (as Java already does for
//! the sorted-numeric and sorted views). The documents and values are the
//! same; only when a leaf's open error surfaces moves.

use lucene_codecs::field_infos::{DocValuesType, IndexOptions};

use super::{
    BinaryDocValues, DocIdSetIterator, DocValuesIterator, IndexReader, LeafReader,
    LeafReaderContext, NumericDocValues, SortedDocValues, SortedNumericDocValues,
    SortedSetDocValues, NO_MORE_DOCS,
};
use crate::ordinal_map::OrdinalMap;
use crate::{Error, Result};

/// The leaves' iterators and the walk over them every multi view shares.
pub struct MultiIter<'a, T: ?Sized> {
    subs: Vec<Option<Box<T>>>,
    /// Each leaf's doc base, then the reader's `maxDoc`.
    starts: Vec<i32>,
    /// The leaf the iterator is in.
    cur: usize,
    doc: i32,
    _leaves: std::marker::PhantomData<&'a ()>,
}

impl<'a, T: ?Sized + DocValuesIterator> MultiIter<'a, T> {
    fn new(leaves: &[LeafReaderContext<'a>], max_doc: i32, subs: Vec<Option<Box<T>>>) -> Self {
        let mut starts: Vec<i32> = leaves.iter().map(|l| l.doc_base).collect();
        starts.push(max_doc);
        Self {
            subs,
            starts,
            cur: 0,
            doc: -1,
            _leaves: std::marker::PhantomData,
        }
    }

    /// `ReaderUtil.subIndex(doc, leaves)`: the last leaf starting at or
    /// before `doc` (empty leaves are skipped).
    fn sub_index(&self, doc: i32) -> usize {
        let n = self.subs.len();
        self.starts[..n]
            .partition_point(|&s| s <= doc)
            .saturating_sub(1)
    }

    fn current(&self) -> Option<&T> {
        self.subs.get(self.cur).and_then(|s| s.as_deref())
    }

    fn current_mut(&mut self) -> Option<&mut T> {
        self.subs.get_mut(self.cur).and_then(|s| s.as_deref_mut())
    }

    /// From the current leaf on: the next document of the current leaf, else
    /// the first of a later one.
    fn next_from_current(&mut self) -> Result<i32> {
        while self.cur < self.subs.len() {
            let base = self.starts[self.cur];
            if let Some(sub) = self.subs[self.cur].as_deref_mut() {
                let d = sub.next_doc()?;
                if d != NO_MORE_DOCS {
                    self.doc = base + d;
                    return Ok(self.doc);
                }
            }
            self.cur += 1;
        }
        self.doc = NO_MORE_DOCS;
        Ok(NO_MORE_DOCS)
    }

    fn check_forward(&self, target: i32, strict: bool) -> Result<()> {
        if (strict && target <= self.doc) || (!strict && target < self.doc) {
            return Err(Error::IllegalArgument(format!(
                "can only advance beyond current document: on docID={} but targetDocID={target}",
                self.doc
            )));
        }
        Ok(())
    }
}

impl<T: ?Sized + DocValuesIterator> DocIdSetIterator for MultiIter<'_, T> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        if self.doc == NO_MORE_DOCS {
            return Ok(NO_MORE_DOCS);
        }
        self.next_from_current()
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        self.check_forward(target, true)?;
        if target >= *self.starts.last().unwrap_or(&0) {
            self.cur = self.subs.len();
            self.doc = NO_MORE_DOCS;
            return Ok(NO_MORE_DOCS);
        }
        let i = self.sub_index(target);
        if i > self.cur {
            self.cur = i;
        }
        let base = self.starts[self.cur];
        if let Some(sub) = self.subs[self.cur].as_deref_mut() {
            let d = sub.advance((target - base).max(0))?;
            if d != NO_MORE_DOCS {
                self.doc = base + d;
                return Ok(self.doc);
            }
        }
        self.cur += 1;
        self.next_from_current()
    }

    fn cost(&self) -> i64 {
        self.subs
            .iter()
            .flatten()
            .fold(0i64, |a, s| a.saturating_add(s.cost()))
    }
}

impl<T: ?Sized + DocValuesIterator> DocValuesIterator for MultiIter<'_, T> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.check_forward(target, false)?;
        if target < 0 || target >= *self.starts.last().unwrap_or(&0) {
            return Err(Error::IllegalArgument(format!("Out of range: {target}")));
        }
        self.cur = self.sub_index(target);
        self.doc = target;
        let base = self.starts[self.cur];
        match self.current_mut() {
            None => Ok(false),
            Some(sub) => sub.advance_exact(target - base),
        }
    }
}

impl<'a> NumericDocValues for MultiIter<'a, dyn NumericDocValues + 'a> {
    fn long_value(&self) -> i64 {
        self.current().map_or(0, |s| s.long_value())
    }
}

impl<'a> BinaryDocValues for MultiIter<'a, dyn BinaryDocValues + 'a> {
    fn binary_value(&self) -> &[u8] {
        self.current().map_or(&[], |s| s.binary_value())
    }
}

impl<'a> SortedNumericDocValues for MultiIter<'a, dyn SortedNumericDocValues + 'a> {
    fn doc_value_count(&self) -> i32 {
        self.current().map_or(0, |s| s.doc_value_count())
    }
    fn next_value(&mut self) -> Result<i64> {
        match self.current_mut() {
            Some(s) => s.next_value(),
            None => Err(Error::IllegalState("no current document".into())),
        }
    }
}

fn any_field(
    leaves: &[LeafReaderContext<'_>],
    field: &str,
    ok: impl Fn(&lucene_codecs::field_infos::FieldInfo) -> bool,
) -> bool {
    leaves
        .iter()
        .any(|l| l.reader.field_infos().field_by_name(field).is_some_and(&ok))
}

type Opener<'a, T> = fn(&'a dyn LeafReader, &str) -> Result<Option<Box<T>>>;

fn open_all<'a, T: ?Sized>(
    leaves: &[LeafReaderContext<'a>],
    field: &str,
    open: Opener<'a, T>,
) -> Result<Vec<Option<Box<T>>>> {
    leaves.iter().map(|l| open(l.reader, field)).collect()
}

/// `MultiDocValues.getNormValues(r, field)`.
///
/// # Errors
/// A leaf's norms fail to open.
pub fn norm_values<'a, R: IndexReader + ?Sized>(
    r: &'a R,
    field: &str,
) -> Result<Option<Box<dyn NumericDocValues + 'a>>> {
    let leaves = r.leaves();
    match leaves.len() {
        0 => return Ok(None),
        1 => return leaves[0].reader.norm_values(field),
        _ => {}
    }
    if !any_field(&leaves, field, |fi| {
        fi.index_options != IndexOptions::None && !fi.omit_norms
    }) {
        return Ok(None);
    }
    let subs = open_all(&leaves, field, |l, f| l.norm_values(f))?;
    Ok(Some(Box::new(MultiIter::new(&leaves, r.max_doc(), subs))))
}

/// `MultiDocValues.getNumericValues(r, field)`.
///
/// # Errors
/// A leaf's values fail to open.
pub fn numeric_values<'a, R: IndexReader + ?Sized>(
    r: &'a R,
    field: &str,
) -> Result<Option<Box<dyn NumericDocValues + 'a>>> {
    let leaves = r.leaves();
    match leaves.len() {
        0 => return Ok(None),
        1 => return leaves[0].reader.numeric_doc_values(field),
        _ => {}
    }
    if !any_field(&leaves, field, |fi| {
        fi.doc_values_type == DocValuesType::Numeric
    }) {
        return Ok(None);
    }
    let subs = open_all(&leaves, field, |l, f| l.numeric_doc_values(f))?;
    Ok(Some(Box::new(MultiIter::new(&leaves, r.max_doc(), subs))))
}

/// `MultiDocValues.getBinaryValues(r, field)`.
///
/// # Errors
/// A leaf's values fail to open.
pub fn binary_values<'a, R: IndexReader + ?Sized>(
    r: &'a R,
    field: &str,
) -> Result<Option<Box<dyn BinaryDocValues + 'a>>> {
    let leaves = r.leaves();
    match leaves.len() {
        0 => return Ok(None),
        1 => return leaves[0].reader.binary_doc_values(field),
        _ => {}
    }
    if !any_field(&leaves, field, |fi| {
        fi.doc_values_type == DocValuesType::Binary
    }) {
        return Ok(None);
    }
    let subs = open_all(&leaves, field, |l, f| l.binary_doc_values(f))?;
    Ok(Some(Box::new(MultiIter::new(&leaves, r.max_doc(), subs))))
}

/// `MultiDocValues.getSortedNumericValues(r, field)`.
///
/// # Errors
/// A leaf's values fail to open.
pub fn sorted_numeric_values<'a, R: IndexReader + ?Sized>(
    r: &'a R,
    field: &str,
) -> Result<Option<Box<dyn SortedNumericDocValues + 'a>>> {
    let leaves = r.leaves();
    match leaves.len() {
        0 => return Ok(None),
        1 => return leaves[0].reader.sorted_numeric_doc_values(field),
        _ => {}
    }
    let subs = open_all(&leaves, field, |l, f| l.sorted_numeric_doc_values(f))?;
    if subs.iter().all(Option::is_none) {
        return Ok(None);
    }
    Ok(Some(Box::new(MultiIter::new(&leaves, r.max_doc(), subs))))
}

/// Every term of a leaf's sorted values, by ordinal: what the ordinal map is
/// built from.
fn sorted_terms(v: Option<&mut Box<dyn SortedDocValues + '_>>) -> Result<Vec<Vec<u8>>> {
    match v {
        None => Ok(Vec::new()),
        Some(v) => (0..v.value_count()).map(|o| v.lookup_ord(o)).collect(),
    }
}

fn sorted_set_terms(v: Option<&mut Box<dyn SortedSetDocValues + '_>>) -> Result<Vec<Vec<u8>>> {
    match v {
        None => Ok(Vec::new()),
        Some(v) => (0..v.value_count()).map(|o| v.lookup_ord(o)).collect(),
    }
}

/// `MultiDocValues.MultiSortedDocValues`.
pub struct MultiSortedDocValues<'a> {
    iter: MultiIter<'a, dyn SortedDocValues + 'a>,
    /// `mapping`.
    pub mapping: OrdinalMap,
}

/// `MultiDocValues.getSortedValues(r, field)`.
///
/// # Errors
/// A leaf's values fail to open or to read.
pub fn sorted_values<'a, R: IndexReader + ?Sized>(
    r: &'a R,
    field: &str,
) -> Result<Option<Box<dyn SortedDocValues + 'a>>> {
    let leaves = r.leaves();
    match leaves.len() {
        0 => return Ok(None),
        1 => return leaves[0].reader.sorted_doc_values(field),
        _ => {}
    }
    let mut subs = open_all(&leaves, field, |l, f| l.sorted_doc_values(f))?;
    if subs.iter().all(Option::is_none) {
        return Ok(None);
    }
    let terms = subs
        .iter_mut()
        .map(|s| sorted_terms(s.as_mut()))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(Box::new(MultiSortedDocValues {
        mapping: OrdinalMap::build(&terms),
        iter: MultiIter::new(&leaves, r.max_doc(), subs),
    })))
}

impl DocIdSetIterator for MultiSortedDocValues<'_> {
    fn doc_id(&self) -> i32 {
        self.iter.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.iter.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.iter.advance(target)
    }
    fn cost(&self) -> i64 {
        self.iter.cost()
    }
}

impl DocValuesIterator for MultiSortedDocValues<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.iter.advance_exact(target)
    }
}

impl SortedDocValues for MultiSortedDocValues<'_> {
    fn ord_value(&self) -> i32 {
        let local = self.iter.current().map_or(-1, |s| s.ord_value());
        self.mapping
            .global_ord(self.iter.cur, i64::from(local))
            .map_or(-1, |g| g as i32)
    }
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
        let g = i64::from(ord);
        let (Some(seg), Some(seg_ord)) = (
            self.mapping.first_segment(g),
            self.mapping.first_segment_ord(g),
        ) else {
            return Err(Error::IllegalArgument(format!("ord {ord} out of range")));
        };
        // The map names a leaf only for an ordinal that leaf has.
        let sub = self.iter.subs[seg]
            .as_deref_mut()
            .expect("the map's leaf has values");
        sub.lookup_ord(seg_ord as i32)
    }
    fn value_count(&self) -> i32 {
        i32::try_from(self.mapping.value_count()).unwrap_or(i32::MAX)
    }
}

/// `MultiDocValues.MultiSortedSetDocValues`.
pub struct MultiSortedSetDocValues<'a> {
    iter: MultiIter<'a, dyn SortedSetDocValues + 'a>,
    /// `mapping`.
    pub mapping: OrdinalMap,
}

/// `MultiDocValues.getSortedSetValues(r, field)`.
///
/// # Errors
/// A leaf's values fail to open or to read.
pub fn sorted_set_values<'a, R: IndexReader + ?Sized>(
    r: &'a R,
    field: &str,
) -> Result<Option<Box<dyn SortedSetDocValues + 'a>>> {
    let leaves = r.leaves();
    match leaves.len() {
        0 => return Ok(None),
        1 => return leaves[0].reader.sorted_set_doc_values(field),
        _ => {}
    }
    let mut subs = open_all(&leaves, field, |l, f| l.sorted_set_doc_values(f))?;
    if subs.iter().all(Option::is_none) {
        return Ok(None);
    }
    let terms = subs
        .iter_mut()
        .map(|s| sorted_set_terms(s.as_mut()))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(Box::new(MultiSortedSetDocValues {
        mapping: OrdinalMap::build(&terms),
        iter: MultiIter::new(&leaves, r.max_doc(), subs),
    })))
}

impl DocIdSetIterator for MultiSortedSetDocValues<'_> {
    fn doc_id(&self) -> i32 {
        self.iter.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.iter.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.iter.advance(target)
    }
    fn cost(&self) -> i64 {
        self.iter.cost()
    }
}

impl DocValuesIterator for MultiSortedSetDocValues<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.iter.advance_exact(target)
    }
}

impl SortedSetDocValues for MultiSortedSetDocValues<'_> {
    fn doc_value_count(&self) -> i32 {
        self.iter.current().map_or(0, |s| s.doc_value_count())
    }
    fn next_ord(&mut self) -> Result<i64> {
        let cur = self.iter.cur;
        let local = match self.iter.current_mut() {
            Some(s) => s.next_ord()?,
            None => return Err(Error::IllegalState("no current document".into())),
        };
        self.mapping
            .global_ord(cur, local)
            .ok_or_else(|| Error::IllegalState(format!("leaf ordinal {local} not in the map")))
    }
    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        let (Some(seg), Some(seg_ord)) = (
            self.mapping.first_segment(ord),
            self.mapping.first_segment_ord(ord),
        ) else {
            return Err(Error::IllegalArgument(format!("ord {ord} out of range")));
        };
        // The map names a leaf only for an ordinal that leaf has.
        let sub = self.iter.subs[seg]
            .as_deref_mut()
            .expect("the map's leaf has values");
        sub.lookup_ord(seg_ord)
    }
    fn value_count(&self) -> i64 {
        self.mapping.value_count()
    }
}
