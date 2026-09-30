//! Port of `org.apache.lucene.index.SortingCodecReader`: a [`CodecReader`]
//! whose documents are re-numbered by an index sort -- what `addIndexes`
//! uses to add an unsorted index to a sorted one, and what `OneMerge.reorder`
//! produces.
//!
//! Given a [`DocMap`] (`Sorter.DocMap`), every read of the view is the inner
//! reader's with document ids mapped: postings re-sorted by new doc id
//! (`FreqProxTermsWriter.SortingTerms`), doc values and norms re-laid in new
//! doc order (`NumericDocValuesWriter.SortingNumericDocValues` and the
//! binary/sorted/sorted-numeric/sorted-set equivalents), stored fields and
//! term vectors read at `newToOld(doc)`, points visited with `oldToNew`, live
//! docs as `in.get(newToOld(i))`, vectors re-ordered by new doc.
//!
//! [`sort_doc_map`] is `Sorter.sort(reader)` over this layer's doc values,
//! for every sort kind whose key is one number per document (`NUMERIC`,
//! `SORTED_NUMERIC`, `SORTED`, `SORTED_SET`); a `BinarySortField` is refused
//! with [`Error::Unsupported`]. It returns `None` for a reader already in
//! order, and [`SortingCodecReader::wrap`] then only relabels the reader's
//! sort, as Java does.
//!
//! # What differs from Java
//!
//! The view materializes each field's doc values, and each term's postings,
//! when they are asked for -- which is what Java's sorting wrappers do too
//! (they buffer a field in new-doc order on first use); the difference is
//! only that nothing is cached between two requests for the same field.

use std::sync::Arc;

use lucene_codecs::field_infos::FieldInfos;
use lucene_index::segment_info::{
    IndexSortField, IndexSortKind, SortKeyComparator, SortedNumericSelector, SortedSetSelector,
};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::multi_terms::drain_into;
use crate::{Error, Result};

use super::{
    BinaryDocValues, ByteVectorValues, CacheHelper, CodecReader, DocIdSetIterator,
    DocValuesIterator, FloatVectorValues, ImpactsEnum, IndexReader, IntersectVisitor, LeafReader,
    LeafReaderContext, MaterializedPostings, NumericDocValues, PointValues, Position, PostingsEnum,
    PostingsFlags, Relation, SlowImpactsEnum, SortedDocValues, SortedNumericDocValues,
    SortedSetDocValues, StoredFieldVisitor, TermVectorsDocument, Terms, TermsEnum, NO_MORE_DOCS,
};

/// `Sorter.DocMap`: a permutation of a leaf's documents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocMap {
    old_to_new: Vec<i32>,
    new_to_old: Vec<i32>,
}

impl DocMap {
    /// A map from `new_to_old[new] = old`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when `new_to_old` is not a permutation of
    /// `0..len` (`Sorter.isConsistent`).
    pub fn from_new_to_old(new_to_old: Vec<i32>) -> Result<Self> {
        let n = new_to_old.len();
        let mut old_to_new = vec![-1i32; n];
        for (new, &old) in new_to_old.iter().enumerate() {
            match usize::try_from(old)
                .ok()
                .and_then(|o| old_to_new.get_mut(o))
            {
                Some(slot) if *slot == -1 => *slot = new as i32,
                _ => {
                    return Err(Error::IllegalArgument(format!(
                        "not a permutation: old doc {old} at new doc {new}"
                    )))
                }
            }
        }
        Ok(Self {
            old_to_new,
            new_to_old,
        })
    }

    /// `oldToNew(docID)`.
    pub fn old_to_new(&self, doc: i32) -> i32 {
        self.old_to_new[doc as usize]
    }

    /// `newToOld(docID)`.
    pub fn new_to_old(&self, doc: i32) -> i32 {
        self.new_to_old[doc as usize]
    }

    /// `size()`.
    pub fn size(&self) -> i32 {
        self.new_to_old.len() as i32
    }
}

/// One document's key for one sort field, `None` when it has no value.
fn sort_keys(reader: &dyn LeafReader, sort: &IndexSortField) -> Result<Vec<Option<i64>>> {
    let n = reader.max_doc().max(0) as usize;
    let pairs: Vec<(i32, Option<i64>)> = match &sort.kind {
        IndexSortKind::Numeric(_) => match reader.numeric_doc_values(&sort.field)? {
            Some(mut v) => collect(&mut *v, |v| Ok(Some(v.long_value())))?,
            None => Vec::new(),
        },
        IndexSortKind::SortedNumeric { selector, .. } => {
            match reader.sorted_numeric_doc_values(&sort.field)? {
                Some(mut v) => collect(&mut *v, |v| {
                    let values = (0..v.doc_value_count())
                        .map(|_| v.next_value())
                        .collect::<Result<Vec<_>>>()?;
                    Ok(match selector {
                        SortedNumericSelector::Min => values.first().copied(),
                        SortedNumericSelector::Max => values.last().copied(),
                    })
                })?,
                None => Vec::new(),
            }
        }
        IndexSortKind::String(_) => match reader.sorted_doc_values(&sort.field)? {
            Some(mut v) => collect(&mut *v, |v| Ok(Some(i64::from(v.ord_value()))))?,
            None => Vec::new(),
        },
        IndexSortKind::SortedSet { selector, .. } => {
            match reader.sorted_set_doc_values(&sort.field)? {
                Some(mut v) => collect(&mut *v, |v| {
                    let ords = (0..v.doc_value_count())
                        .map(|_| v.next_ord())
                        .collect::<Result<Vec<_>>>()?;
                    let count = ords.len();
                    Ok(match selector {
                        SortedSetSelector::Min => ords.first(),
                        SortedSetSelector::Max => ords.last(),
                        SortedSetSelector::MiddleMin => ords.get(count.saturating_sub(1) / 2),
                        SortedSetSelector::MiddleMax => ords.get(count / 2),
                    }
                    .copied())
                })?,
                None => Vec::new(),
            }
        }
        // Refused by `sort_doc_map` before any key is read.
        IndexSortKind::Binary(_) => Vec::new(),
    };
    let mut keys = vec![None; n];
    for (d, k) in pairs {
        if let Some(slot) = usize::try_from(d).ok().and_then(|d| keys.get_mut(d)) {
            *slot = k;
        }
    }
    Ok(keys)
}

/// `Sorter.sort(reader)`: the documents of `reader` ordered by `sort`, ties
/// by doc id; `None` when they already are in that order.
///
/// # Errors
/// A binary sort field ([`Error::Unsupported`]) and doc-values read errors.
pub fn sort_doc_map(reader: &dyn LeafReader, sort: &[IndexSortField]) -> Result<Option<DocMap>> {
    let mut columns = Vec::with_capacity(sort.len());
    for field in sort {
        let cmp = SortKeyComparator::new(field).ok_or_else(|| {
            Error::Unsupported(format!(
                "sorting a reader by the binary sort field {:?}",
                field.field
            ))
        })?;
        columns.push((cmp, sort_keys(reader, field)?));
    }
    let n = reader.max_doc().max(0) as usize;
    let mut order: Vec<i32> = (0..n as i32).collect();
    order.sort_by(|&a, &b| {
        columns
            .iter()
            .fold(std::cmp::Ordering::Equal, |acc, (cmp, keys)| {
                acc.then_with(|| cmp.compare(keys[a as usize], keys[b as usize]))
            })
            .then(a.cmp(&b))
    });
    if order.iter().enumerate().all(|(i, &d)| i as i32 == d) {
        return Ok(None);
    }
    DocMap::from_new_to_old(order).map(Some)
}

/// `SortingCodecReader`.
pub struct SortingCodecReader {
    in_: Arc<dyn CodecReader>,
    /// `None`: the reader was already sorted and only its sort is relabelled.
    map: Option<DocMap>,
    sort: Vec<IndexSortField>,
    live_docs: Option<FixedBitSet>,
}

impl SortingCodecReader {
    /// `SortingCodecReader.wrap(reader, sort)`: sorts with [`sort_doc_map`].
    ///
    /// # Errors
    /// What [`sort_doc_map`] reports.
    pub fn wrap_sorted(reader: Arc<dyn CodecReader>, sort: Vec<IndexSortField>) -> Result<Self> {
        let map = sort_doc_map(reader.as_ref(), &sort)?;
        Self::wrap(reader, map, sort)
    }

    /// `SortingCodecReader.wrap(reader, docMap, sort)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when `doc_map` is not the reader's size.
    pub fn wrap(
        reader: Arc<dyn CodecReader>,
        doc_map: Option<DocMap>,
        sort: Vec<IndexSortField>,
    ) -> Result<Self> {
        let live_docs = match &doc_map {
            Some(map) => {
                if reader.max_doc() != map.size() {
                    return Err(Error::IllegalArgument(format!(
                        "reader.maxDoc() should be equal to docMap.size(), got {} != {}",
                        reader.max_doc(),
                        map.size()
                    )));
                }
                // `SortingBits`: new doc `i` is live iff old `newToOld(i)` is.
                reader.live_docs().map(|live| {
                    let mut bits = FixedBitSet::new(live.len());
                    for new in 0..map.size() {
                        if live.get_doc(map.new_to_old(new)) && (new as usize) < bits.len() {
                            bits.set(new as usize);
                        }
                    }
                    bits
                })
            }
            None => None,
        };
        Ok(Self {
            in_: reader,
            map: doc_map,
            sort,
            live_docs,
        })
    }

    /// The map this view applies, `None` when it applies none.
    pub fn doc_map(&self) -> Option<&DocMap> {
        self.map.as_ref()
    }

    /// Lays out `(old doc, value)` pairs by new doc id.
    fn by_new_doc<V>(&self, pairs: Vec<(i32, V)>) -> Vec<Option<V>> {
        let map = self.map.as_ref().expect("only called with a map");
        let mut out: Vec<Option<V>> = (0..map.size()).map(|_| None).collect();
        for (old, v) in pairs {
            out[map.old_to_new(old) as usize] = Some(v);
        }
        out
    }
}

impl IndexReader for SortingCodecReader {
    fn max_doc(&self) -> i32 {
        self.in_.max_doc()
    }
    fn num_docs(&self) -> i32 {
        self.in_.num_docs()
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        vec![LeafReaderContext {
            reader: self,
            ord: 0,
            doc_base: 0,
        }]
    }
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        None
    }
}

/// Collects every `(doc, value)` of a doc-values iterator.
fn collect<V, T: ?Sized + DocIdSetIterator>(
    it: &mut T,
    mut value: impl FnMut(&mut T) -> Result<V>,
) -> Result<Vec<(i32, V)>> {
    let mut out = Vec::new();
    loop {
        let d = it.next_doc()?;
        if d == NO_MORE_DOCS {
            return Ok(out);
        }
        out.push((d, value(it)?));
    }
}

impl LeafReader for SortingCodecReader {
    fn field_infos(&self) -> &FieldInfos {
        self.in_.field_infos()
    }
    fn live_docs(&self) -> Option<&FixedBitSet> {
        match self.map {
            Some(_) => self.live_docs.as_ref(),
            None => self.in_.live_docs(),
        }
    }
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms + '_>>> {
        let Some(t) = self.in_.terms(field)? else {
            return Ok(None);
        };
        match &self.map {
            None => Ok(Some(t)),
            Some(map) => Ok(Some(Box::new(SortingTerms { in_: t, map }))),
        }
    }
    fn numeric_doc_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        let Some(mut v) = self.in_.numeric_doc_values(field)? else {
            return Ok(None);
        };
        if self.map.is_none() {
            return Ok(Some(v));
        }
        let pairs = collect(&mut *v, |v| Ok(v.long_value()))?;
        Ok(Some(Box::new(ArrayDv::new(self.by_new_doc(pairs)))))
    }
    fn binary_doc_values(&self, field: &str) -> Result<Option<Box<dyn BinaryDocValues + '_>>> {
        let Some(mut v) = self.in_.binary_doc_values(field)? else {
            return Ok(None);
        };
        if self.map.is_none() {
            return Ok(Some(v));
        }
        let pairs = collect(&mut *v, |v| Ok(v.binary_value().to_vec()))?;
        Ok(Some(Box::new(ArrayDv::new(self.by_new_doc(pairs)))))
    }
    fn sorted_doc_values(&self, field: &str) -> Result<Option<Box<dyn SortedDocValues + '_>>> {
        let Some(mut v) = self.in_.sorted_doc_values(field)? else {
            return Ok(None);
        };
        if self.map.is_none() {
            return Ok(Some(v));
        }
        let pairs = collect(&mut *v, |v| Ok(v.ord_value()))?;
        Ok(Some(Box::new(SortingSorted {
            dv: ArrayDv::new(self.by_new_doc(pairs)),
            lookup: v,
        })))
    }
    fn sorted_numeric_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        let Some(mut v) = self.in_.sorted_numeric_doc_values(field)? else {
            return Ok(None);
        };
        if self.map.is_none() {
            return Ok(Some(v));
        }
        let pairs = collect(&mut *v, |v| {
            (0..v.doc_value_count()).map(|_| v.next_value()).collect()
        })?;
        Ok(Some(Box::new(ArrayDv::new(self.by_new_doc(pairs)))))
    }
    fn sorted_set_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        let Some(mut v) = self.in_.sorted_set_doc_values(field)? else {
            return Ok(None);
        };
        if self.map.is_none() {
            return Ok(Some(v));
        }
        let pairs = collect(&mut *v, |v| {
            (0..v.doc_value_count()).map(|_| v.next_ord()).collect()
        })?;
        Ok(Some(Box::new(SortingSortedSet {
            dv: ArrayDv::new(self.by_new_doc(pairs)),
            lookup: v,
        })))
    }
    fn norm_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        let Some(mut v) = self.in_.norm_values(field)? else {
            return Ok(None);
        };
        if self.map.is_none() {
            return Ok(Some(v));
        }
        let pairs = collect(&mut *v, |v| Ok(v.long_value()))?;
        Ok(Some(Box::new(ArrayDv::new(self.by_new_doc(pairs)))))
    }
    fn point_values(&self, field: &str) -> Result<Option<Box<dyn PointValues + '_>>> {
        let Some(v) = self.in_.point_values(field)? else {
            return Ok(None);
        };
        match &self.map {
            None => Ok(Some(v)),
            Some(map) => Ok(Some(Box::new(SortingPoints { in_: v, map }))),
        }
    }
    fn float_vector_values(&self, field: &str) -> Result<Option<Box<dyn FloatVectorValues + '_>>> {
        let Some(v) = self.in_.float_vector_values(field)? else {
            return Ok(None);
        };
        match &self.map {
            None => Ok(Some(v)),
            Some(map) => {
                let ords = sorted_ords(map, v.size(), |o| v.ord_to_doc(o))?;
                Ok(Some(Box::new(SortingVectors { in_: v, ords })))
            }
        }
    }
    fn byte_vector_values(&self, field: &str) -> Result<Option<Box<dyn ByteVectorValues + '_>>> {
        let Some(v) = self.in_.byte_vector_values(field)? else {
            return Ok(None);
        };
        match &self.map {
            None => Ok(Some(v)),
            Some(map) => {
                let ords = sorted_ords(map, v.size(), |o| v.ord_to_doc(o))?;
                Ok(Some(Box::new(SortingVectors { in_: v, ords })))
            }
        }
    }
    fn document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        let old = self.map.as_ref().map_or(doc, |m| m.new_to_old(doc));
        self.in_.document(old, visitor)
    }
    fn term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        let old = self.map.as_ref().map_or(doc, |m| m.new_to_old(doc));
        self.in_.term_vectors(old)
    }
    /// The sort it was wrapped with; `None` for an empty one (a reordered
    /// merge's view, which Java wraps with a `null` sort).
    fn index_sort(&self) -> Option<&[IndexSortField]> {
        (!self.sort.is_empty()).then_some(self.sort.as_slice())
    }
    fn core_cache_helper(&self) -> Option<&CacheHelper> {
        None
    }
    fn check_integrity(&self) -> Result<()> {
        self.in_.check_integrity()
    }
}

impl CodecReader for SortingCodecReader {}

// ---------------------------------------------------------------------------
// Postings
// ---------------------------------------------------------------------------

/// `FreqProxTermsWriter.SortingTerms`.
struct SortingTerms<'a> {
    in_: Box<dyn Terms + 'a>,
    map: &'a DocMap,
}

impl Terms for SortingTerms<'_> {
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
        Ok(Box::new(SortingTermsEnum {
            in_: self.in_.iterator()?,
            map: self.map,
            has_positions: self.in_.has_positions(),
            has_offsets: self.in_.has_offsets(),
        }))
    }
    fn intersect<'s>(
        &'s self,
        dfa: &'s lucene_codecs::automaton::ByteDfa,
        start_term: Option<&[u8]>,
    ) -> Result<Box<dyn TermsEnum + 's>> {
        Ok(Box::new(SortingTermsEnum {
            in_: self.in_.intersect(dfa, start_term)?,
            map: self.map,
            has_positions: self.in_.has_positions(),
            has_offsets: self.in_.has_offsets(),
        }))
    }
    fn size(&self) -> i64 {
        self.in_.size()
    }
    fn sum_total_term_freq(&self) -> i64 {
        self.in_.sum_total_term_freq()
    }
    fn sum_doc_freq(&self) -> i64 {
        self.in_.sum_doc_freq()
    }
    fn doc_count(&self) -> i32 {
        self.in_.doc_count()
    }
    fn has_freqs(&self) -> bool {
        self.in_.has_freqs()
    }
    fn has_offsets(&self) -> bool {
        self.in_.has_offsets()
    }
    fn has_positions(&self) -> bool {
        self.in_.has_positions()
    }
    fn has_payloads(&self) -> bool {
        self.in_.has_payloads()
    }
    fn min(&self) -> Result<Option<Vec<u8>>> {
        self.in_.min()
    }
    fn max(&self) -> Result<Option<Vec<u8>>> {
        self.in_.max()
    }
}

/// `FreqProxTermsWriter.SortingTermsEnum`.
struct SortingTermsEnum<'a> {
    in_: Box<dyn TermsEnum + 'a>,
    map: &'a DocMap,
    has_positions: bool,
    has_offsets: bool,
}

impl TermsEnum for SortingTermsEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        self.in_.next()
    }
    fn term(&self) -> Option<&[u8]> {
        self.in_.term()
    }
    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<lucene_codecs::blocktree::SeekStatus> {
        self.in_.try_seek_ceil(target)
    }
    fn try_seek_exact(&mut self, target: &[u8]) -> Result<bool> {
        self.in_.try_seek_exact(target)
    }
    fn doc_freq(&mut self) -> Result<i32> {
        self.in_.doc_freq()
    }
    fn total_term_freq(&mut self) -> Result<i64> {
        self.in_.total_term_freq()
    }
    /// `SortingDocsEnum`/`SortingPostingsEnum`: the term's postings with
    /// every document renumbered and put back in doc-id order.
    fn postings(&mut self, flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        let positional = flags.wants_positions() && self.has_positions;
        let pe = self.in_.postings(flags)?;
        let mut docs = Vec::new();
        let mut freqs = Vec::new();
        let mut positions: Option<Vec<Vec<Position>>> = positional.then(Vec::new);
        drain_into(pe, 0, &mut docs, &mut freqs, positions.as_mut())?;
        let mut order: Vec<usize> = (0..docs.len()).collect();
        order.sort_by_key(|&i| self.map.old_to_new(docs[i]));
        let new_docs = order
            .iter()
            .map(|&i| self.map.old_to_new(docs[i]))
            .collect();
        let new_freqs = order.iter().map(|&i| freqs[i]).collect();
        // Without stored offsets `SortingPostingsEnum` reports the start it
        // was reset to (`-1`) and the end `nextDoc` reset (`0`).
        let no_offsets = !self.has_offsets;
        let new_positions = positions.map(|mut p| {
            order
                .iter()
                .map(|&i| {
                    let mut list = std::mem::take(&mut p[i]);
                    if no_offsets {
                        for pos in &mut list {
                            pos.start_offset = -1;
                            pos.end_offset = 0;
                        }
                    }
                    list
                })
                .collect()
        });
        Ok(Box::new(MaterializedPostings::new(
            new_docs,
            new_freqs,
            new_positions,
        )?))
    }
    fn impacts(&mut self, flags: PostingsFlags) -> Result<Box<dyn ImpactsEnum>> {
        Ok(Box::new(SlowImpactsEnum::new(self.postings(flags)?)))
    }
}

// ---------------------------------------------------------------------------
// Doc values in new-doc order
// ---------------------------------------------------------------------------

/// A doc-values iterator over values already laid out by (new) doc id.
struct ArrayDv<V> {
    values: Vec<Option<V>>,
    doc: i32,
    /// For multi-valued values: the next one to hand out.
    upto: usize,
}

impl<V> ArrayDv<V> {
    fn new(values: Vec<Option<V>>) -> Self {
        Self {
            values,
            doc: -1,
            upto: 0,
        }
    }

    fn value(&self) -> Option<&V> {
        usize::try_from(self.doc)
            .ok()
            .and_then(|d| self.values.get(d))
            .and_then(Option::as_ref)
    }

    fn seek_from(&mut self, from: usize) -> i32 {
        self.upto = 0;
        self.doc = self.values[from.min(self.values.len())..]
            .iter()
            .position(Option::is_some)
            .map_or(NO_MORE_DOCS, |i| (from + i) as i32);
        self.doc
    }
}

impl<V> DocIdSetIterator for ArrayDv<V> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        if self.doc == NO_MORE_DOCS {
            return Ok(NO_MORE_DOCS);
        }
        Ok(self.seek_from((self.doc + 1) as usize))
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        Ok(self.seek_from(target.max(0) as usize))
    }
    fn cost(&self) -> i64 {
        self.values.iter().filter(|v| v.is_some()).count() as i64
    }
}

impl<V> DocValuesIterator for ArrayDv<V> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.doc = target;
        self.upto = 0;
        Ok(self.value().is_some())
    }
}

impl NumericDocValues for ArrayDv<i64> {
    fn long_value(&self) -> i64 {
        self.value().copied().unwrap_or(0)
    }
}

impl BinaryDocValues for ArrayDv<Vec<u8>> {
    fn binary_value(&self) -> &[u8] {
        self.value().map_or(&[], Vec::as_slice)
    }
}

impl SortedNumericDocValues for ArrayDv<Vec<i64>> {
    fn doc_value_count(&self) -> i32 {
        self.value().map_or(0, |v| v.len() as i32)
    }
    fn next_value(&mut self) -> Result<i64> {
        let v = self
            .value()
            .and_then(|v| v.get(self.upto))
            .copied()
            .ok_or_else(|| Error::IllegalState("no more values for this document".into()))?;
        self.upto += 1;
        Ok(v)
    }
}

/// `SortedDocValuesWriter.SortingSortedDocValues`: ordinals by new doc,
/// terms from the inner values.
struct SortingSorted<'a> {
    dv: ArrayDv<i32>,
    lookup: Box<dyn SortedDocValues + 'a>,
}

impl DocIdSetIterator for SortingSorted<'_> {
    fn doc_id(&self) -> i32 {
        self.dv.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.dv.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.dv.advance(target)
    }
    fn cost(&self) -> i64 {
        self.dv.cost()
    }
}

impl DocValuesIterator for SortingSorted<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.dv.advance_exact(target)
    }
}

impl SortedDocValues for SortingSorted<'_> {
    fn ord_value(&self) -> i32 {
        self.dv.value().copied().unwrap_or(-1)
    }
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
        self.lookup.lookup_ord(ord)
    }
    fn value_count(&self) -> i32 {
        self.lookup.value_count()
    }
}

/// `SortedSetDocValuesWriter.SortingSortedSetDocValues`.
struct SortingSortedSet<'a> {
    dv: ArrayDv<Vec<i64>>,
    lookup: Box<dyn SortedSetDocValues + 'a>,
}

impl DocIdSetIterator for SortingSortedSet<'_> {
    fn doc_id(&self) -> i32 {
        self.dv.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.dv.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.dv.advance(target)
    }
    fn cost(&self) -> i64 {
        self.dv.cost()
    }
}

impl DocValuesIterator for SortingSortedSet<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.dv.advance_exact(target)
    }
}

impl SortedSetDocValues for SortingSortedSet<'_> {
    fn doc_value_count(&self) -> i32 {
        self.dv.doc_value_count()
    }
    fn next_ord(&mut self) -> Result<i64> {
        self.dv.next_value()
    }
    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        self.lookup.lookup_ord(ord)
    }
    fn value_count(&self) -> i64 {
        self.lookup.value_count()
    }
}

// ---------------------------------------------------------------------------
// Points and vectors
// ---------------------------------------------------------------------------

/// `SortingPointValues`.
struct SortingPoints<'a> {
    in_: Box<dyn PointValues + 'a>,
    map: &'a DocMap,
}

/// `SortingIntersectVisitor`: doc ids through `oldToNew`.
struct SortingVisitor<'v> {
    inner: &'v mut dyn IntersectVisitor,
    map: &'v DocMap,
}

impl IntersectVisitor for SortingVisitor<'_> {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        self.inner.compare(min_packed, max_packed)
    }
    fn visit(&mut self, doc_id: i32) {
        self.inner.visit(self.map.old_to_new(doc_id));
    }
    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        self.inner
            .visit_with_value(self.map.old_to_new(doc_id), packed_value);
    }
}

impl PointValues for SortingPoints<'_> {
    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()> {
        self.in_.intersect(&mut SortingVisitor {
            inner: visitor,
            map: self.map,
        })
    }
    fn estimate_point_count(&self, visitor: &mut dyn IntersectVisitor) -> Result<i64> {
        self.in_.estimate_point_count(&mut SortingVisitor {
            inner: visitor,
            map: self.map,
        })
    }
    fn min_packed_value(&self) -> &[u8] {
        self.in_.min_packed_value()
    }
    fn max_packed_value(&self) -> &[u8] {
        self.in_.max_packed_value()
    }
    fn num_dimensions(&self) -> i32 {
        self.in_.num_dimensions()
    }
    fn num_index_dimensions(&self) -> i32 {
        self.in_.num_index_dimensions()
    }
    fn bytes_per_dimension(&self) -> i32 {
        self.in_.bytes_per_dimension()
    }
    fn size(&self) -> i64 {
        self.in_.size()
    }
    fn doc_count(&self) -> i32 {
        self.in_.doc_count()
    }
}

/// `SortingCodecReader.iteratorSupplier`: `(new doc, old ordinal)` for every
/// vector, in new-doc order -- the new ordinals.
fn sorted_ords(
    map: &DocMap,
    size: i32,
    ord_to_doc: impl Fn(i32) -> Result<i32>,
) -> Result<Vec<(i32, i32)>> {
    let mut ords = (0..size)
        .map(|o| Ok((map.old_to_new(ord_to_doc(o)?), o)))
        .collect::<Result<Vec<_>>>()?;
    ords.sort_unstable();
    Ok(ords)
}

/// `SortingFloatVectorValues`/`SortingByteVectorValues`.
struct SortingVectors<T: ?Sized> {
    in_: Box<T>,
    /// By new ordinal: `(new doc, old ordinal)`.
    ords: Vec<(i32, i32)>,
}

impl<T: ?Sized> SortingVectors<T> {
    fn get(&self, ord: i32) -> Result<(i32, i32)> {
        usize::try_from(ord)
            .ok()
            .and_then(|o| self.ords.get(o))
            .copied()
            .ok_or_else(|| Error::IllegalArgument(format!("vector ordinal {ord} out of range")))
    }
}

impl<'a> FloatVectorValues for SortingVectors<dyn FloatVectorValues + 'a> {
    fn dimension(&self) -> usize {
        self.in_.dimension()
    }
    fn size(&self) -> i32 {
        self.in_.size()
    }
    fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        Ok(self.get(ord)?.0)
    }
    fn vector_value(&self, ord: i32) -> Result<Vec<f32>> {
        self.in_.vector_value(self.get(ord)?.1)
    }
}

impl<'a> ByteVectorValues for SortingVectors<dyn ByteVectorValues + 'a> {
    fn dimension(&self) -> usize {
        self.in_.dimension()
    }
    fn size(&self) -> i32 {
        self.in_.size()
    }
    fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        Ok(self.get(ord)?.0)
    }
    fn vector_value(&self, ord: i32) -> Result<Vec<u8>> {
        self.in_.vector_value(self.get(ord)?.1)
    }
}
