//! Ports of the delegating reader wrappers: `FilterLeafReader` (with its
//! `FilterTerms`/`FilterTermsEnum`/`FilterPostingsEnum`), `FilterCodecReader`,
//! `FilterDirectoryReader` and the five `Filter*DocValues`.
//!
//! # How overriding maps
//!
//! Java's filters are abstract classes a subclass overrides method by method.
//! A Rust type cannot inherit, so the overridable part is a hook trait with
//! identity defaults: a [`FilterLeafReader`] owns its inner reader and a
//! [`LeafFilter`], and every access object the inner reader hands out passes
//! through the matching `wrap_*` hook before it reaches the caller.
//! Implementing only the hooks one needs is overriding only the methods one
//! needs. `FilterDirectoryReader.SubReaderWrapper` is [`SubReaderWrapper`],
//! the same idea one level up.
//!
//! The access-object filters ([`FilterTerms`], [`FilterTermsEnum`],
//! [`FilterPostingsEnum`], [`FilterNumericDocValues`], ...) delegate every
//! method to a public `in_`; a wrapper that changes one behaviour (as
//! `ExitableDirectoryReader`'s do) implements the trait itself and forwards
//! the rest to `in_` through them.
//!
//! Cache helpers: Java makes `getCoreCacheHelper`/`getReaderCacheHelper`
//! abstract so each filter decides whether it may be cached as its inner
//! reader. [`LeafFilter::caches_like_inner`] is that decision, `false` (no
//! helper) by default -- the safe answer for a filter that changes content.

use std::sync::Arc;

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::FieldInfos;
use lucene_index::segment_info::IndexSortField;
use lucene_store::directory::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::{
    BinaryDocValues, ByteVectorValues, CacheHelper, CodecReader, CompositeReader, DocIdSetIterator,
    DocValuesIterator, FloatVectorValues, ImpactsEnum, IndexReader, LeafReader, LeafReaderContext,
    NumericDocValues, PointValues, PostingsEnum, PostingsFlags, SortedDocValues,
    SortedNumericDocValues, SortedSetDocValues, StoredFieldVisitor, SubReader, TermVectorsDocument,
    Terms, TermsEnum,
};
use crate::directory_reader::DirectoryReader;
use crate::Result;

/// The overridable half of `FilterLeafReader`: every hook receives what the
/// inner reader returned and may replace it. All default to passing it
/// through unchanged.
#[allow(unused_variables)]
pub trait LeafFilter: Send + Sync {
    /// `terms(field)`.
    fn wrap_terms<'a>(
        &'a self,
        field: &str,
        terms: Box<dyn Terms + 'a>,
    ) -> Result<Box<dyn Terms + 'a>> {
        Ok(terms)
    }
    /// `getNumericDocValues(field)`.
    fn wrap_numeric<'a>(
        &'a self,
        field: &str,
        values: Box<dyn NumericDocValues + 'a>,
    ) -> Result<Box<dyn NumericDocValues + 'a>> {
        Ok(values)
    }
    /// `getBinaryDocValues(field)`.
    fn wrap_binary<'a>(
        &'a self,
        field: &str,
        values: Box<dyn BinaryDocValues + 'a>,
    ) -> Result<Box<dyn BinaryDocValues + 'a>> {
        Ok(values)
    }
    /// `getSortedDocValues(field)`.
    fn wrap_sorted<'a>(
        &'a self,
        field: &str,
        values: Box<dyn SortedDocValues + 'a>,
    ) -> Result<Box<dyn SortedDocValues + 'a>> {
        Ok(values)
    }
    /// `getSortedNumericDocValues(field)`.
    fn wrap_sorted_numeric<'a>(
        &'a self,
        field: &str,
        values: Box<dyn SortedNumericDocValues + 'a>,
    ) -> Result<Box<dyn SortedNumericDocValues + 'a>> {
        Ok(values)
    }
    /// `getSortedSetDocValues(field)`.
    fn wrap_sorted_set<'a>(
        &'a self,
        field: &str,
        values: Box<dyn SortedSetDocValues + 'a>,
    ) -> Result<Box<dyn SortedSetDocValues + 'a>> {
        Ok(values)
    }
    /// `getNormValues(field)`.
    fn wrap_norms<'a>(
        &'a self,
        field: &str,
        values: Box<dyn NumericDocValues + 'a>,
    ) -> Result<Box<dyn NumericDocValues + 'a>> {
        Ok(values)
    }
    /// `getPointValues(field)`.
    fn wrap_points<'a>(
        &'a self,
        field: &str,
        values: Box<dyn PointValues + 'a>,
    ) -> Result<Box<dyn PointValues + 'a>> {
        Ok(values)
    }
    /// `getFloatVectorValues(field)`.
    fn wrap_float_vectors<'a>(
        &'a self,
        field: &str,
        values: Box<dyn FloatVectorValues + 'a>,
    ) -> Result<Box<dyn FloatVectorValues + 'a>> {
        Ok(values)
    }
    /// `getByteVectorValues(field)`.
    fn wrap_byte_vectors<'a>(
        &'a self,
        field: &str,
        values: Box<dyn ByteVectorValues + 'a>,
    ) -> Result<Box<dyn ByteVectorValues + 'a>> {
        Ok(values)
    }
    /// `getLiveDocs()`: the inner reader's unless overridden.
    fn live_docs<'a>(&'a self, inner: Option<&'a FixedBitSet>) -> Option<&'a FixedBitSet> {
        inner
    }
    /// `storedFields().document(doc, visitor)`.
    fn document(
        &self,
        inner: &dyn LeafReader,
        doc: i32,
        visitor: &mut dyn StoredFieldVisitor,
    ) -> Result<()> {
        inner.document(doc, visitor)
    }
    /// `termVectors().get(doc)`.
    fn term_vectors(
        &self,
        inner: &dyn LeafReader,
        doc: i32,
    ) -> Result<Option<TermVectorsDocument>> {
        inner.term_vectors(doc)
    }
    /// Whether this filter's reader may share its inner reader's cache
    /// helpers (`getCoreCacheHelper`/`getReaderCacheHelper` returning
    /// `in`'s): only a filter that changes no content may say yes.
    fn caches_like_inner(&self) -> bool {
        false
    }
}

/// The identity [`LeafFilter`]: a `FilterLeafReader` that overrides nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoFilter;

impl LeafFilter for NoFilter {
    fn caches_like_inner(&self) -> bool {
        true
    }
}

/// `FilterLeafReader` (and, over a [`CodecReader`], `FilterCodecReader`):
/// `in_` read through `filter`.
pub struct FilterLeafReader<R: ?Sized + LeafReader + AsDynLeaf = dyn LeafReader> {
    in_: Arc<R>,
    filter: Box<dyn LeafFilter>,
}

/// `FilterCodecReader`: a [`FilterLeafReader`] over a [`CodecReader`], which
/// is itself one.
pub type FilterCodecReader = FilterLeafReader<dyn CodecReader>;

impl<R: ?Sized + LeafReader + AsDynLeaf> FilterLeafReader<R> {
    /// `new FilterLeafReader(in)` with `filter`'s overrides.
    pub fn new(in_: Arc<R>, filter: Box<dyn LeafFilter>) -> Self {
        Self { in_, filter }
    }

    /// `getDelegate()`.
    pub fn delegate(&self) -> &Arc<R> {
        &self.in_
    }
}

impl<R: ?Sized + LeafReader + AsDynLeaf> IndexReader for FilterLeafReader<R> {
    fn max_doc(&self) -> i32 {
        self.in_.max_doc()
    }
    fn num_docs(&self) -> i32 {
        // The live docs as the filter presents them.
        super::num_docs_of(self.in_.max_doc(), LeafReader::live_docs(self))
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        vec![LeafReaderContext {
            reader: self,
            ord: 0,
            doc_base: 0,
        }]
    }
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        if self.filter.caches_like_inner() {
            self.in_.reader_cache_helper()
        } else {
            None
        }
    }
}

impl<R: ?Sized + LeafReader + AsDynLeaf> LeafReader for FilterLeafReader<R> {
    fn field_infos(&self) -> &FieldInfos {
        self.in_.field_infos()
    }
    fn live_docs(&self) -> Option<&FixedBitSet> {
        self.filter.live_docs(self.in_.live_docs())
    }
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms + '_>>> {
        match self.in_.terms(field)? {
            None => Ok(None),
            Some(t) => self.filter.wrap_terms(field, t).map(Some),
        }
    }
    fn numeric_doc_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        match self.in_.numeric_doc_values(field)? {
            None => Ok(None),
            Some(v) => self.filter.wrap_numeric(field, v).map(Some),
        }
    }
    fn binary_doc_values(&self, field: &str) -> Result<Option<Box<dyn BinaryDocValues + '_>>> {
        match self.in_.binary_doc_values(field)? {
            None => Ok(None),
            Some(v) => self.filter.wrap_binary(field, v).map(Some),
        }
    }
    fn sorted_doc_values(&self, field: &str) -> Result<Option<Box<dyn SortedDocValues + '_>>> {
        match self.in_.sorted_doc_values(field)? {
            None => Ok(None),
            Some(v) => self.filter.wrap_sorted(field, v).map(Some),
        }
    }
    fn sorted_numeric_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        match self.in_.sorted_numeric_doc_values(field)? {
            None => Ok(None),
            Some(v) => self.filter.wrap_sorted_numeric(field, v).map(Some),
        }
    }
    fn sorted_set_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        match self.in_.sorted_set_doc_values(field)? {
            None => Ok(None),
            Some(v) => self.filter.wrap_sorted_set(field, v).map(Some),
        }
    }
    fn norm_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        match self.in_.norm_values(field)? {
            None => Ok(None),
            Some(v) => self.filter.wrap_norms(field, v).map(Some),
        }
    }
    fn point_values(&self, field: &str) -> Result<Option<Box<dyn PointValues + '_>>> {
        match self.in_.point_values(field)? {
            None => Ok(None),
            Some(v) => self.filter.wrap_points(field, v).map(Some),
        }
    }
    fn float_vector_values(&self, field: &str) -> Result<Option<Box<dyn FloatVectorValues + '_>>> {
        match self.in_.float_vector_values(field)? {
            None => Ok(None),
            Some(v) => self.filter.wrap_float_vectors(field, v).map(Some),
        }
    }
    fn byte_vector_values(&self, field: &str) -> Result<Option<Box<dyn ByteVectorValues + '_>>> {
        match self.in_.byte_vector_values(field)? {
            None => Ok(None),
            Some(v) => self.filter.wrap_byte_vectors(field, v).map(Some),
        }
    }
    fn document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        self.filter.document(self.in_.as_dyn_leaf(), doc, visitor)
    }
    fn term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        self.filter.term_vectors(self.in_.as_dyn_leaf(), doc)
    }
    fn index_sort(&self) -> Option<&[IndexSortField]> {
        self.in_.index_sort()
    }
    fn core_cache_helper(&self) -> Option<&CacheHelper> {
        if self.filter.caches_like_inner() {
            self.in_.core_cache_helper()
        } else {
            None
        }
    }
    fn doc_values_skipper(
        &self,
        field: &str,
    ) -> Result<Option<lucene_codecs::doc_values::DocValuesSkipper<'_>>> {
        self.in_.doc_values_skipper(field)
    }
    fn check_integrity(&self) -> Result<()> {
        self.in_.check_integrity()
    }
}

impl CodecReader for FilterLeafReader<dyn CodecReader> {}

/// Upcasting to `&dyn LeafReader`, for every sized reader and the trait
/// objects the wrappers hold.
pub trait AsDynLeaf {
    fn as_dyn_leaf(&self) -> &dyn LeafReader;
}

impl<T: LeafReader> AsDynLeaf for T {
    fn as_dyn_leaf(&self) -> &dyn LeafReader {
        self
    }
}

impl AsDynLeaf for dyn LeafReader {
    fn as_dyn_leaf(&self) -> &dyn LeafReader {
        self
    }
}

impl AsDynLeaf for dyn CodecReader {
    fn as_dyn_leaf(&self) -> &dyn LeafReader {
        self
    }
}

// ---------------------------------------------------------------------------
// FilterDirectoryReader
// ---------------------------------------------------------------------------

/// `FilterDirectoryReader.SubReaderWrapper`: wraps each leaf of a
/// [`DirectoryReader`].
pub trait SubReaderWrapper: Send + Sync {
    /// `wrap(reader)`.
    fn wrap(&self, reader: Arc<dyn LeafReader>) -> Result<Arc<dyn LeafReader>>;
}

/// `FilterDirectoryReader`: a [`DirectoryReader`] whose leaves are wrapped by
/// a [`SubReaderWrapper`]; reopening reopens the inner reader and wraps the
/// result again (`doWrapDirectoryReader`).
pub struct FilterDirectoryReader {
    in_: Arc<DirectoryReader>,
    leaves: Vec<Arc<dyn LeafReader>>,
    wrapper: Arc<dyn SubReaderWrapper>,
    /// `DelegatingCacheHelper`: its own key.
    reader_cache: CacheHelper,
}

impl FilterDirectoryReader {
    /// `new FilterDirectoryReader(in, wrapper)`.
    ///
    /// # Errors
    /// What `wrapper` reports.
    pub fn new(in_: Arc<DirectoryReader>, wrapper: Arc<dyn SubReaderWrapper>) -> Result<Self> {
        let leaves = in_
            .segment_readers()
            .iter()
            .map(|s| wrapper.wrap(Arc::new(s.clone())))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            in_,
            leaves,
            wrapper,
            reader_cache: CacheHelper::new(),
        })
    }

    /// `getDelegate()`.
    pub fn delegate(&self) -> &Arc<DirectoryReader> {
        &self.in_
    }

    /// `FilterDirectoryReader.unwrap`: the innermost [`DirectoryReader`].
    pub fn unwrap(&self) -> &Arc<DirectoryReader> {
        &self.in_
    }

    /// The wrapped leaves, in order.
    pub fn wrapped_leaves(&self) -> &[Arc<dyn LeafReader>] {
        &self.leaves
    }

    /// `openIfChanged(this)`: the inner reader reopened and wrapped with the
    /// same wrapper, `None` when nothing changed.
    ///
    /// # Errors
    /// Reopening or wrapping fails.
    pub fn open_if_changed(&self, dir: &dyn Directory) -> Result<Option<Self>> {
        match self.in_.open_if_changed(dir)? {
            None => Ok(None),
            Some(r) => Self::new(Arc::new(r), Arc::clone(&self.wrapper)).map(Some),
        }
    }

    /// `getVersion()`: the inner reader's commit generation.
    pub fn version(&self) -> i64 {
        self.in_.segment_infos.version
    }
}

impl IndexReader for FilterDirectoryReader {
    fn max_doc(&self) -> i32 {
        self.leaves
            .iter()
            .fold(0i32, |a, l| a.saturating_add(l.max_doc()))
    }
    fn num_docs(&self) -> i32 {
        self.leaves
            .iter()
            .fold(0i32, |a, l| a.saturating_add(l.num_docs()))
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        super::composite_leaves(&self.sequential_sub_readers())
    }
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        Some(&self.reader_cache)
    }
}

impl CompositeReader for FilterDirectoryReader {
    fn sequential_sub_readers(&self) -> Vec<SubReader<'_>> {
        self.leaves
            .iter()
            .map(|l| SubReader::Leaf(l.as_ref()))
            .collect()
    }
    fn leaf_handles(&self) -> Vec<Arc<dyn LeafReader>> {
        self.leaves.clone()
    }
}

// ---------------------------------------------------------------------------
// Access-object filters
// ---------------------------------------------------------------------------

/// `FilterLeafReader.FilterTerms`.
pub struct FilterTerms<'a> {
    pub in_: Box<dyn Terms + 'a>,
}

impl Terms for FilterTerms<'_> {
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
        self.in_.iterator()
    }
    fn intersect<'s>(
        &'s self,
        dfa: &'s lucene_codecs::automaton::ByteDfa,
        start_term: Option<&[u8]>,
    ) -> Result<Box<dyn TermsEnum + 's>> {
        self.in_.intersect(dfa, start_term)
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

/// `FilterLeafReader.FilterTermsEnum`.
pub struct FilterTermsEnum<'a> {
    pub in_: Box<dyn TermsEnum + 'a>,
}

impl TermsEnum for FilterTermsEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        self.in_.next()
    }
    fn term(&self) -> Option<&[u8]> {
        self.in_.term()
    }
    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
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
    fn postings(&mut self, flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        self.in_.postings(flags)
    }
    fn impacts(&mut self, flags: PostingsFlags) -> Result<Box<dyn ImpactsEnum>> {
        self.in_.impacts(flags)
    }
    fn ord(&self) -> Result<i64> {
        self.in_.ord()
    }
    fn seek_exact_ord(&mut self, ord: i64) -> Result<()> {
        self.in_.seek_exact_ord(ord)
    }
}

/// `FilterLeafReader.FilterPostingsEnum`.
pub struct FilterPostingsEnum {
    pub in_: Box<dyn PostingsEnum>,
}

impl DocIdSetIterator for FilterPostingsEnum {
    fn doc_id(&self) -> i32 {
        self.in_.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.in_.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.in_.advance(target)
    }
    fn cost(&self) -> i64 {
        self.in_.cost()
    }
}

impl PostingsEnum for FilterPostingsEnum {
    fn freq(&self) -> i32 {
        self.in_.freq()
    }
    fn next_position(&mut self) -> Result<i32> {
        self.in_.next_position()
    }
    fn start_offset(&self) -> i32 {
        self.in_.start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.in_.end_offset()
    }
    fn payload(&self) -> Option<&[u8]> {
        self.in_.payload()
    }
}

/// The `DocIdSetIterator`/`DocValuesIterator` half every `Filter*DocValues`
/// shares: delegation to `in_`.
macro_rules! filter_dv {
    ($name:ident, $tr:ident) => {
        impl DocIdSetIterator for $name<'_> {
            fn doc_id(&self) -> i32 {
                self.in_.doc_id()
            }
            fn next_doc(&mut self) -> Result<i32> {
                self.in_.next_doc()
            }
            fn advance(&mut self, target: i32) -> Result<i32> {
                self.in_.advance(target)
            }
            fn cost(&self) -> i64 {
                self.in_.cost()
            }
        }
        impl DocValuesIterator for $name<'_> {
            fn advance_exact(&mut self, target: i32) -> Result<bool> {
                self.in_.advance_exact(target)
            }
        }
    };
}

/// `FilterNumericDocValues`.
pub struct FilterNumericDocValues<'a> {
    pub in_: Box<dyn NumericDocValues + 'a>,
}
filter_dv!(FilterNumericDocValues, NumericDocValues);

impl NumericDocValues for FilterNumericDocValues<'_> {
    fn long_value(&self) -> i64 {
        self.in_.long_value()
    }
}

/// `FilterBinaryDocValues`.
pub struct FilterBinaryDocValues<'a> {
    pub in_: Box<dyn BinaryDocValues + 'a>,
}
filter_dv!(FilterBinaryDocValues, BinaryDocValues);

impl BinaryDocValues for FilterBinaryDocValues<'_> {
    fn binary_value(&self) -> &[u8] {
        self.in_.binary_value()
    }
}

/// `FilterSortedDocValues`.
pub struct FilterSortedDocValues<'a> {
    pub in_: Box<dyn SortedDocValues + 'a>,
}
filter_dv!(FilterSortedDocValues, SortedDocValues);

impl SortedDocValues for FilterSortedDocValues<'_> {
    fn ord_value(&self) -> i32 {
        self.in_.ord_value()
    }
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
        self.in_.lookup_ord(ord)
    }
    fn value_count(&self) -> i32 {
        self.in_.value_count()
    }
    fn lookup_term(&mut self, key: &[u8]) -> Result<i32> {
        self.in_.lookup_term(key)
    }
}

/// `FilterSortedNumericDocValues`.
pub struct FilterSortedNumericDocValues<'a> {
    pub in_: Box<dyn SortedNumericDocValues + 'a>,
}
filter_dv!(FilterSortedNumericDocValues, SortedNumericDocValues);

impl SortedNumericDocValues for FilterSortedNumericDocValues<'_> {
    fn doc_value_count(&self) -> i32 {
        self.in_.doc_value_count()
    }
    fn next_value(&mut self) -> Result<i64> {
        self.in_.next_value()
    }
}

/// `FilterSortedSetDocValues`.
pub struct FilterSortedSetDocValues<'a> {
    pub in_: Box<dyn SortedSetDocValues + 'a>,
}
filter_dv!(FilterSortedSetDocValues, SortedSetDocValues);

impl SortedSetDocValues for FilterSortedSetDocValues<'_> {
    fn doc_value_count(&self) -> i32 {
        self.in_.doc_value_count()
    }
    fn next_ord(&mut self) -> Result<i64> {
        self.in_.next_ord()
    }
    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        self.in_.lookup_ord(ord)
    }
    fn value_count(&self) -> i64 {
        self.in_.value_count()
    }
    fn lookup_term(&mut self, key: &[u8]) -> Result<i64> {
        self.in_.lookup_term(key)
    }
}
