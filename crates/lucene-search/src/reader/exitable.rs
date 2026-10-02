//! Ports of `org.apache.lucene.index.QueryTimeout`, `QueryTimeoutImpl` and
//! `ExitableDirectoryReader`: a reader wrapper whose term, doc-values, point
//! and vector enumerations ask a [`QueryTimeout`] whether to stop, and fail
//! with [`Error::ExitingReader`] (`ExitingReaderException`) when it says so.
//!
//! The sampling is Java's: a terms enumeration asks on every 16th `next()`
//! (`(calls++ & 15) == 0`), a doc-values iterator whenever it reaches a
//! document at least 1000 past the last check (`DOCS_BETWEEN_TIMEOUT_CHECK`),
//! a point intersection on every `compare` and every 16th `visit`.
//!
//! # What differs from Java
//!
//! - `Thread.interrupted()` has no Rust counterpart; only the timeout stops a
//!   wrapped reader.
//! - A points `IntersectVisitor` cannot fail. Once the timeout fires inside
//!   an intersection, the wrapper answers every later cell with
//!   `CellOutsideQuery` and drops every later visit -- so the walk unwinds
//!   without reading another block -- and `intersect` then reports the exit.
//!   Java checks the same points (`compare`, sampled `visit`s) plus the
//!   `PointTree` moves, which this port's intersect does not expose.
//! - `PointValues`' metadata getters are infallible here and do not check.
//! - Vector values are read by ordinal ([`super::FloatVectorValues`]), so the
//!   1000-document check runs on `ord_to_doc` as the documents it returns
//!   advance, where Java's runs on the values' `DocIndexIterator`.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lucene_codecs::blocktree::SeekStatus;

use super::filter::{FilterDirectoryReader, FilterLeafReader, LeafFilter, SubReaderWrapper};
use super::{
    BinaryDocValues, ByteVectorValues, DocIdSetIterator, DocValuesIterator, FloatVectorValues,
    ImpactsEnum, IntersectVisitor, LeafReader, NumericDocValues, PointValues, PostingsEnum,
    PostingsFlags, Relation, SortedDocValues, SortedNumericDocValues, SortedSetDocValues, Terms,
    TermsEnum,
};
use crate::directory_reader::DirectoryReader;
use crate::{Error, Result};

/// `QueryTimeout`: whether a query should stop now.
pub trait QueryTimeout: Send + Sync + std::fmt::Debug {
    /// `shouldExit()`.
    fn should_exit(&self) -> bool;
}

/// `QueryTimeoutImpl`: exits once a wall-clock deadline has passed.
#[derive(Debug, Clone, Copy)]
pub struct QueryTimeoutImpl {
    /// `timeoutAt`; `None` never times out.
    timeout_at: Option<Instant>,
}

impl QueryTimeoutImpl {
    /// `new QueryTimeoutImpl(timeAllowed)`: `time_allowed_ms` from now. A
    /// negative allowance is Java's `Long.MAX_VALUE`: never.
    pub fn new(time_allowed_ms: i64) -> Self {
        let timeout_at = u64::try_from(time_allowed_ms)
            .ok()
            .and_then(|ms| Instant::now().checked_add(Duration::from_millis(ms)));
        Self { timeout_at }
    }

    /// `getTimeoutAt()`.
    pub fn timeout_at(&self) -> Option<Instant> {
        self.timeout_at
    }
}

impl QueryTimeout for QueryTimeoutImpl {
    fn should_exit(&self) -> bool {
        self.timeout_at.is_some_and(|at| Instant::now() > at)
    }
}

/// `DOCS_BETWEEN_TIMEOUT_CHECK`.
pub const DOCS_BETWEEN_TIMEOUT_CHECK: i32 = 1000;
/// `ExitableTermsEnum.NUM_CALLS_PER_TIMEOUT_CHECK`: a check every 16 calls.
const NUM_CALLS_PER_TIMEOUT_CHECK: u32 = (1 << 4) - 1;
/// `MAX_CALLS_BEFORE_QUERY_TIMEOUT_CHECK` of the point visitor.
const MAX_CALLS_BEFORE_QUERY_TIMEOUT_CHECK: u32 = 16;

fn check(timeout: &dyn QueryTimeout, what: &str) -> Result<()> {
    if timeout.should_exit() {
        return Err(Error::ExitingReader(format!(
            "The request took too long to {what}. Timeout: {timeout:?}"
        )));
    }
    Ok(())
}

/// `ExitableDirectoryReader`: `in` with every leaf an [`ExitableLeafReader`].
pub struct ExitableDirectoryReader;

impl ExitableDirectoryReader {
    /// `ExitableDirectoryReader.wrap(in, queryTimeout)`.
    ///
    /// # Errors
    /// None in practice; wrapping is infallible.
    pub fn wrap(
        in_: Arc<DirectoryReader>,
        timeout: Arc<dyn QueryTimeout>,
    ) -> Result<FilterDirectoryReader> {
        FilterDirectoryReader::new(in_, Arc::new(ExitableSubReaderWrapper { timeout }))
    }
}

/// `ExitableSubReaderWrapper`.
pub struct ExitableSubReaderWrapper {
    timeout: Arc<dyn QueryTimeout>,
}

impl ExitableSubReaderWrapper {
    pub fn new(timeout: Arc<dyn QueryTimeout>) -> Self {
        Self { timeout }
    }
}

impl SubReaderWrapper for ExitableSubReaderWrapper {
    fn wrap(&self, reader: Arc<dyn LeafReader>) -> Result<Arc<dyn LeafReader>> {
        Ok(Arc::new(exitable_leaf_reader(
            reader,
            Arc::clone(&self.timeout),
        )))
    }
}

/// `ExitableFilterAtomicReader`.
pub type ExitableLeafReader = FilterLeafReader<dyn LeafReader>;

/// `new ExitableFilterAtomicReader(in, queryTimeout)`.
pub fn exitable_leaf_reader(
    in_: Arc<dyn LeafReader>,
    timeout: Arc<dyn QueryTimeout>,
) -> ExitableLeafReader {
    FilterLeafReader::new(in_, Box::new(ExitableFilter { timeout }))
}

/// The overrides of `ExitableFilterAtomicReader`.
pub struct ExitableFilter {
    timeout: Arc<dyn QueryTimeout>,
}

impl ExitableFilter {
    fn t(&self) -> &dyn QueryTimeout {
        &*self.timeout
    }
}

impl LeafFilter for ExitableFilter {
    fn wrap_terms<'a>(
        &'a self,
        _field: &str,
        terms: Box<dyn Terms + 'a>,
    ) -> Result<Box<dyn Terms + 'a>> {
        Ok(Box::new(ExitableTerms {
            in_: terms,
            timeout: self.t(),
        }))
    }
    fn wrap_numeric<'a>(
        &'a self,
        _field: &str,
        values: Box<dyn NumericDocValues + 'a>,
    ) -> Result<Box<dyn NumericDocValues + 'a>> {
        Ok(Box::new(ExitableDv::new(values, self.t(), false)))
    }
    fn wrap_binary<'a>(
        &'a self,
        _field: &str,
        values: Box<dyn BinaryDocValues + 'a>,
    ) -> Result<Box<dyn BinaryDocValues + 'a>> {
        // Java's binary wrapper checks `advance` against the target, not the
        // document it landed on.
        Ok(Box::new(ExitableDv::new(values, self.t(), true)))
    }
    fn wrap_sorted<'a>(
        &'a self,
        _field: &str,
        values: Box<dyn SortedDocValues + 'a>,
    ) -> Result<Box<dyn SortedDocValues + 'a>> {
        Ok(Box::new(ExitableDv::new(values, self.t(), false)))
    }
    fn wrap_sorted_numeric<'a>(
        &'a self,
        _field: &str,
        values: Box<dyn SortedNumericDocValues + 'a>,
    ) -> Result<Box<dyn SortedNumericDocValues + 'a>> {
        Ok(Box::new(ExitableDv::new(values, self.t(), false)))
    }
    fn wrap_sorted_set<'a>(
        &'a self,
        _field: &str,
        values: Box<dyn SortedSetDocValues + 'a>,
    ) -> Result<Box<dyn SortedSetDocValues + 'a>> {
        Ok(Box::new(ExitableDv::new(values, self.t(), false)))
    }
    fn wrap_points<'a>(
        &'a self,
        _field: &str,
        values: Box<dyn PointValues + 'a>,
    ) -> Result<Box<dyn PointValues + 'a>> {
        Ok(Box::new(ExitablePointValues {
            in_: values,
            timeout: self.t(),
        }))
    }
    fn wrap_float_vectors<'a>(
        &'a self,
        _field: &str,
        values: Box<dyn FloatVectorValues + 'a>,
    ) -> Result<Box<dyn FloatVectorValues + 'a>> {
        Ok(Box::new(ExitableVectors {
            in_: values,
            timeout: self.t(),
            next_check: std::sync::atomic::AtomicI32::new(0),
        }))
    }
    fn wrap_byte_vectors<'a>(
        &'a self,
        _field: &str,
        values: Box<dyn ByteVectorValues + 'a>,
    ) -> Result<Box<dyn ByteVectorValues + 'a>> {
        Ok(Box::new(ExitableVectors {
            in_: values,
            timeout: self.t(),
            next_check: std::sync::atomic::AtomicI32::new(0),
        }))
    }
    fn caches_like_inner(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Terms
// ---------------------------------------------------------------------------

/// `ExitableTerms`.
pub struct ExitableTerms<'a> {
    in_: Box<dyn Terms + 'a>,
    timeout: &'a dyn QueryTimeout,
}

impl Terms for ExitableTerms<'_> {
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
        Ok(Box::new(ExitableTermsEnum::new(
            self.in_.iterator()?,
            self.timeout,
        )?))
    }
    fn intersect<'s>(
        &'s self,
        dfa: &'s lucene_codecs::automaton::ByteDfa,
        start_term: Option<&[u8]>,
    ) -> Result<Box<dyn TermsEnum + 's>> {
        Ok(Box::new(ExitableTermsEnum::new(
            self.in_.intersect(dfa, start_term)?,
            self.timeout,
        )?))
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

/// `ExitableTermsEnum`: checks the timeout on every 16th `next()`.
pub struct ExitableTermsEnum<'a> {
    in_: Box<dyn TermsEnum + 'a>,
    timeout: &'a dyn QueryTimeout,
    calls: u32,
}

impl<'a> ExitableTermsEnum<'a> {
    /// `new ExitableTermsEnum(termsEnum, queryTimeout)`, which already
    /// counts as the first (checked) call.
    ///
    /// # Errors
    /// [`Error::ExitingReader`] when the timeout has already fired.
    pub fn new(in_: Box<dyn TermsEnum + 'a>, timeout: &'a dyn QueryTimeout) -> Result<Self> {
        let mut te = Self {
            in_,
            timeout,
            calls: 0,
        };
        te.check_with_sampling()?;
        Ok(te)
    }

    /// `checkTimeoutWithSampling()`: every 16th call asks the timeout.
    fn check_with_sampling(&mut self) -> Result<()> {
        let calls = self.calls;
        self.calls = self.calls.wrapping_add(1);
        if calls & NUM_CALLS_PER_TIMEOUT_CHECK == 0 {
            check(self.timeout, "iterate over terms")?;
        }
        Ok(())
    }
}

impl TermsEnum for ExitableTermsEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        self.check_with_sampling()?;
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

// ---------------------------------------------------------------------------
// Doc values
// ---------------------------------------------------------------------------

/// The five doc-values wrappers of `ExitableFilterAtomicReader`: a check
/// whenever iteration reaches `docToCheck`.
pub struct ExitableDv<'a, T: ?Sized> {
    in_: Box<T>,
    timeout: &'a dyn QueryTimeout,
    doc_to_check: i32,
    /// The binary wrapper compares `advance`'s target, the others the
    /// document `advance` returned.
    advance_by_target: bool,
}

impl<'a, T: ?Sized + DocValuesIterator> ExitableDv<'a, T> {
    fn new(in_: Box<T>, timeout: &'a dyn QueryTimeout, advance_by_target: bool) -> Self {
        Self {
            in_,
            timeout,
            doc_to_check: 0,
            advance_by_target,
        }
    }

    fn maybe_check(&mut self, doc: i32) -> Result<()> {
        if doc >= self.doc_to_check {
            check(self.timeout, "iterate over doc values")?;
            self.doc_to_check = doc.saturating_add(DOCS_BETWEEN_TIMEOUT_CHECK);
        }
        Ok(())
    }
}

impl<T: ?Sized + DocValuesIterator> DocIdSetIterator for ExitableDv<'_, T> {
    fn doc_id(&self) -> i32 {
        self.in_.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        let d = self.in_.next_doc()?;
        self.maybe_check(d)?;
        Ok(d)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let d = self.in_.advance(target)?;
        self.maybe_check(if self.advance_by_target { target } else { d })?;
        Ok(d)
    }
    fn cost(&self) -> i64 {
        self.in_.cost()
    }
}

impl<T: ?Sized + DocValuesIterator> DocValuesIterator for ExitableDv<'_, T> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        let found = self.in_.advance_exact(target)?;
        self.maybe_check(target)?;
        Ok(found)
    }
}

impl<'a> NumericDocValues for ExitableDv<'a, dyn NumericDocValues + 'a> {
    fn long_value(&self) -> i64 {
        self.in_.long_value()
    }
}

impl<'a> BinaryDocValues for ExitableDv<'a, dyn BinaryDocValues + 'a> {
    fn binary_value(&self) -> &[u8] {
        self.in_.binary_value()
    }
}

impl<'a> SortedDocValues for ExitableDv<'a, dyn SortedDocValues + 'a> {
    fn ord_value(&self) -> i32 {
        self.in_.ord_value()
    }
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
        self.in_.lookup_ord(ord)
    }
    fn value_count(&self) -> i32 {
        self.in_.value_count()
    }
}

impl<'a> SortedNumericDocValues for ExitableDv<'a, dyn SortedNumericDocValues + 'a> {
    fn doc_value_count(&self) -> i32 {
        self.in_.doc_value_count()
    }
    fn next_value(&mut self) -> Result<i64> {
        self.in_.next_value()
    }
}

impl<'a> SortedSetDocValues for ExitableDv<'a, dyn SortedSetDocValues + 'a> {
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
}

// ---------------------------------------------------------------------------
// Points
// ---------------------------------------------------------------------------

/// `ExitablePointValues`.
pub struct ExitablePointValues<'a> {
    in_: Box<dyn PointValues + 'a>,
    timeout: &'a dyn QueryTimeout,
}

/// `ExitableIntersectVisitor`, which records an exit it cannot throw.
struct ExitableVisitor<'v> {
    in_: &'v mut dyn IntersectVisitor,
    timeout: &'v dyn QueryTimeout,
    calls: u32,
    exited: bool,
}

impl ExitableVisitor<'_> {
    fn exited(&self) -> bool {
        self.exited
    }

    fn check_now(&mut self) -> bool {
        if !self.exited && self.timeout.should_exit() {
            self.exited = true;
        }
        self.exited()
    }

    fn check_sampled(&mut self) -> bool {
        let calls = self.calls;
        self.calls = self.calls.wrapping_add(1);
        if calls.is_multiple_of(MAX_CALLS_BEFORE_QUERY_TIMEOUT_CHECK) {
            self.check_now()
        } else {
            self.exited()
        }
    }
}

impl IntersectVisitor for ExitableVisitor<'_> {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        if self.check_now() {
            return Relation::CellOutsideQuery;
        }
        self.in_.compare(min_packed, max_packed)
    }
    fn visit(&mut self, doc_id: i32) {
        if !self.check_sampled() {
            self.in_.visit(doc_id);
        }
    }
    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        if !self.check_sampled() {
            self.in_.visit_with_value(doc_id, packed_value);
        }
    }
    /// `grow(count)`: `checkAndThrow()`, then forwarded.
    fn grow(&mut self, count: usize) {
        if !self.check_now() {
            self.in_.grow(count);
        }
    }
}

impl ExitablePointValues<'_> {
    fn run(
        &self,
        visitor: &mut dyn IntersectVisitor,
        f: impl FnOnce(&dyn PointValues, &mut dyn IntersectVisitor) -> Result<i64>,
    ) -> Result<i64> {
        check(self.timeout, "intersect point values")?;
        let mut v = ExitableVisitor {
            in_: visitor,
            timeout: self.timeout,
            calls: 0,
            exited: false,
        };
        let out = f(&*self.in_, &mut v)?;
        if v.exited() {
            return Err(Error::ExitingReader(format!(
                "The request took too long to intersect point values. Timeout: {:?}",
                self.timeout
            )));
        }
        Ok(out)
    }
}

impl PointValues for ExitablePointValues<'_> {
    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()> {
        self.run(visitor, |p, v| p.intersect(v).map(|()| 0))
            .map(|_| ())
    }
    fn estimate_point_count(&self, visitor: &mut dyn IntersectVisitor) -> Result<i64> {
        self.run(visitor, |p, v| p.estimate_point_count(v))
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

// ---------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------

/// `ExitableFloatVectorValues`/`ExitableByteVectorValues`.
pub struct ExitableVectors<'a, T: ?Sized> {
    in_: Box<T>,
    timeout: &'a dyn QueryTimeout,
    next_check: std::sync::atomic::AtomicI32,
}

impl<T: ?Sized> ExitableVectors<'_, T> {
    fn maybe_check(&self, doc: i32) -> Result<()> {
        if doc >= self.next_check.load(Ordering::Relaxed) {
            check(self.timeout, "iterate over knn vector values")?;
            self.next_check.store(
                doc.saturating_add(DOCS_BETWEEN_TIMEOUT_CHECK),
                Ordering::Relaxed,
            );
        }
        Ok(())
    }
}

impl<'a> FloatVectorValues for ExitableVectors<'a, dyn FloatVectorValues + 'a> {
    fn dimension(&self) -> usize {
        self.in_.dimension()
    }
    fn size(&self) -> i32 {
        self.in_.size()
    }
    fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        let doc = self.in_.ord_to_doc(ord)?;
        self.maybe_check(doc)?;
        Ok(doc)
    }
    fn vector_value(&self, ord: i32) -> Result<Vec<f32>> {
        self.in_.vector_value(ord)
    }
}

impl<'a> ByteVectorValues for ExitableVectors<'a, dyn ByteVectorValues + 'a> {
    fn dimension(&self) -> usize {
        self.in_.dimension()
    }
    fn size(&self) -> i32 {
        self.in_.size()
    }
    fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        let doc = self.in_.ord_to_doc(ord)?;
        self.maybe_check(doc)?;
        Ok(doc)
    }
    fn vector_value(&self, ord: i32) -> Result<Vec<u8>> {
        self.in_.vector_value(ord)
    }
}
