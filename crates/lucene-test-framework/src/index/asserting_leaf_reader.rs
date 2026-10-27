//! `AssertingLeafReader` and `AssertingDirectoryReader`: a leaf (and a
//! directory reader whose every leaf is one) that checks how its terms,
//! postings and doc values are used and what they answer, panicking where
//! Java's `assert` would throw.
//!
//! Each Java check that has a counterpart here is kept, with Java's message
//! where it has one. Differs:
//!
//! - `assertThread` has no counterpart: the iterators borrow their reader
//!   and are not `Send`, so the compiler already refuses to hand one to
//!   another thread.
//! - Not wrapped (they pass through unchecked): points (`AssertingPointValues`,
//!   `AssertingPointTree`, `AssertingIntersectVisitor`) and vectors
//!   (`searchNearestVectors`). Stored fields and term vectors pass through
//!   too: Java checks only the calling thread of the former, and a
//!   `TermVectorsDocument` is data with no `Terms` to wrap.
//! - `hasDeletions()` is `liveDocs != null` rather than Java's
//!   `numDeletedDocs() > 0`, so a leaf exposing a live-docs bitset with no
//!   deletions fails the constructor's check -- a deliberate strengthening.
//! - `prepareSeekExact`, `termState`, `intoBitSet`, `docIDRunEnd`,
//!   `nextPostings` and `longValues` have no Rust counterpart, nor has the
//!   `TWO_PHASE_SEEKING` state they need. `BytesRef.isValid` is a slice here.
//! - Impacts are owned snapshots, so Java's stale `AssertingImpacts` view
//!   guard is unnecessary. The infallible `ImpactsEnum::impacts` API reports
//!   Java's `CheckIndexException` validation failures as assertion panics.
//! - [`SortedNumericDocValues::take_values`] and
//!   [`SortedSetDocValues::take_ords`], the bulk reads Java lacks, are
//!   checked as `nextValue`/`nextOrd` are and count as reading every value.

use std::sync::Arc;

use lucene_codecs::automaton::ByteDfa;
use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, IndexOptions,
};
use lucene_codecs::postings::Impact;
use lucene_codecs::stored_fields::StoredFieldVisitor;
use lucene_index::segment_info::IndexSortField;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::filter::{FilterDirectoryReader, SubReaderWrapper};
use lucene_search::reader::{
    BinaryDocValues, ByteVectorValues, CacheHelper, DocIdSetIterator, DocValuesIterator,
    FloatVectorValues, ImpactsEnum, IndexReader, LeafReader, LeafReaderContext, NumericDocValues,
    PointValues, PostingsEnum, PostingsFlags, SortedDocValues, SortedNumericDocValues,
    SortedSetDocValues, TermVectorsDocument, Terms, TermsEnum, NO_MORE_DOCS,
};
use lucene_search::reader::{DocValuesSkipper, SkipperImpl};
use lucene_search::Result;
use lucene_util::fixed_bit_set::FixedBitSet;

// ---------------------------------------------------------------------------
// The reader
// ---------------------------------------------------------------------------

/// `AssertingLeafReader`.
pub struct AssertingLeafReader {
    in_: Arc<dyn LeafReader>,
}

impl AssertingLeafReader {
    /// `new AssertingLeafReader(in)`: checks the reader's document counts and
    /// registers closed listeners that check the key they are told.
    pub fn new(in_: Arc<dyn LeafReader>) -> Self {
        let max_doc = in_.max_doc();
        let num_docs = in_.num_docs();
        assert!(max_doc >= 0, "maxDoc={max_doc}");
        assert!(num_docs <= max_doc, "numDocs={num_docs} > maxDoc={max_doc}");
        let deleted = max_doc - num_docs;
        let has_deletions = in_.live_docs().is_some();
        assert!(
            !has_deletions || deleted > 0 && num_docs < max_doc,
            "hasDeletions() but numDeletedDocs={deleted}"
        );
        for (kind, helper) in [
            ("Core", in_.core_cache_helper()),
            ("Reader", in_.reader_cache_helper()),
        ] {
            if let Some(helper) = helper {
                let expected = helper.key_id();
                helper.add_closed_listener(Box::new(move |key| {
                    assert!(
                        expected == key,
                        "{kind} closed listener called on a different key {expected:#x} <> {key:#x}"
                    );
                }));
            }
        }
        Self { in_ }
    }

    /// The wrapped reader.
    pub fn delegate(&self) -> &Arc<dyn LeafReader> {
        &self.in_
    }

    fn field(&self, field: &str) -> Option<&FieldInfo> {
        self.in_.field_infos().field_by_name(field)
    }

    /// Java's `getXDocValues` checks: values exactly when the field has
    /// that doc-values type.
    fn check_dv_type(&self, field: &str, present: bool, ty: DocValuesType) {
        let fi = self.field(field);
        if present {
            assert!(
                fi.is_some_and(|fi| fi.doc_values_type == ty),
                "{ty:?} doc values for {field}, whose field info says {:?}",
                fi.map(|fi| fi.doc_values_type)
            );
        } else {
            assert!(
                fi.is_none_or(|fi| fi.doc_values_type != ty),
                "no {ty:?} doc values for {field}, whose field info has them"
            );
        }
    }
}

/// `FieldInfo.hasNorms()`.
fn has_norms(fi: &FieldInfo) -> bool {
    fi.index_options != IndexOptions::None && !fi.omit_norms
}

impl IndexReader for AssertingLeafReader {
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
        self.in_.reader_cache_helper()
    }
}

impl LeafReader for AssertingLeafReader {
    fn field_infos(&self) -> &FieldInfos {
        self.in_.field_infos()
    }

    fn live_docs(&self) -> Option<&FixedBitSet> {
        let live = self.in_.live_docs();
        match live {
            Some(bits) => assert!(
                usize::try_from(self.max_doc()).is_ok_and(|m| m == bits.len()),
                "maxDoc={} but live docs hold {} bits",
                self.max_doc(),
                bits.len()
            ),
            None => assert!(
                self.max_doc() == self.num_docs(),
                "no live docs but maxDoc={} numDocs={}",
                self.max_doc(),
                self.num_docs()
            ),
        }
        live
    }

    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms + '_>>> {
        Ok(self
            .in_
            .terms(field)?
            .map(|t| Box::new(AssertingTerms::new(t)) as Box<dyn Terms + '_>))
    }

    fn numeric_doc_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        let dv = self.in_.numeric_doc_values(field)?;
        self.check_dv_type(field, dv.is_some(), DocValuesType::Numeric);
        let max_doc = self.max_doc();
        Ok(dv.map(|dv| {
            Box::new(AssertingNumericDocValues::new(dv, max_doc)) as Box<dyn NumericDocValues + '_>
        }))
    }

    fn binary_doc_values(&self, field: &str) -> Result<Option<Box<dyn BinaryDocValues + '_>>> {
        let dv = self.in_.binary_doc_values(field)?;
        self.check_dv_type(field, dv.is_some(), DocValuesType::Binary);
        let max_doc = self.max_doc();
        Ok(dv.map(|dv| {
            Box::new(AssertingBinaryDocValues::new(dv, max_doc)) as Box<dyn BinaryDocValues + '_>
        }))
    }

    fn sorted_doc_values(&self, field: &str) -> Result<Option<Box<dyn SortedDocValues + '_>>> {
        let dv = self.in_.sorted_doc_values(field)?;
        self.check_dv_type(field, dv.is_some(), DocValuesType::Sorted);
        let max_doc = self.max_doc();
        Ok(dv.map(|dv| {
            Box::new(AssertingSortedDocValues::new(dv, max_doc)) as Box<dyn SortedDocValues + '_>
        }))
    }

    fn sorted_numeric_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        let dv = self.in_.sorted_numeric_doc_values(field)?;
        self.check_dv_type(field, dv.is_some(), DocValuesType::SortedNumeric);
        let max_doc = self.max_doc();
        Ok(dv.map(|dv| {
            Box::new(AssertingSortedNumericDocValues::new(dv, max_doc))
                as Box<dyn SortedNumericDocValues + '_>
        }))
    }

    fn sorted_set_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        let dv = self.in_.sorted_set_doc_values(field)?;
        self.check_dv_type(field, dv.is_some(), DocValuesType::SortedSet);
        let max_doc = self.max_doc();
        Ok(dv.map(|dv| {
            Box::new(AssertingSortedSetDocValues::new(dv, max_doc))
                as Box<dyn SortedSetDocValues + '_>
        }))
    }

    fn norm_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        let dv = self.in_.norm_values(field)?;
        let fi = self.field(field);
        if dv.is_some() {
            assert!(
                fi.is_some_and(has_norms),
                "norms for {field}, whose field info has none"
            );
        } else {
            assert!(
                !fi.is_some_and(has_norms),
                "no norms for {field}, whose field info has them"
            );
        }
        let max_doc = self.max_doc();
        Ok(dv.map(|dv| {
            Box::new(AssertingNumericDocValues::new(dv, max_doc)) as Box<dyn NumericDocValues + '_>
        }))
    }

    fn point_values(&self, field: &str) -> Result<Option<Box<dyn PointValues + '_>>> {
        self.in_.point_values(field)
    }

    fn float_vector_values(&self, field: &str) -> Result<Option<Box<dyn FloatVectorValues + '_>>> {
        self.in_.float_vector_values(field)
    }

    fn byte_vector_values(&self, field: &str) -> Result<Option<Box<dyn ByteVectorValues + '_>>> {
        self.in_.byte_vector_values(field)
    }

    fn document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        self.in_.document(doc, visitor)
    }

    fn term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        self.in_.term_vectors(doc)
    }

    fn index_sort(&self) -> Option<&[IndexSortField]> {
        self.in_.index_sort()
    }

    fn core_cache_helper(&self) -> Option<&CacheHelper> {
        self.in_.core_cache_helper()
    }

    fn doc_values_skipper(&self, field: &str) -> Result<Option<DocValuesSkipper<'_>>> {
        let skipper = self.in_.doc_values_skipper(field)?;
        let indexed = self
            .field(field)
            .is_some_and(|fi| fi.doc_values_skip_index_type != DocValuesSkipIndexType::None);
        match skipper {
            Some(skipper) => {
                assert!(
                    indexed,
                    "a skipper for {field}, whose field info has no skip index"
                );
                Ok(Some(DocValuesSkipper::from_impl(Box::new(
                    AssertingDocValuesSkipper::new(skipper),
                ))))
            }
            None => {
                assert!(
                    !indexed,
                    "no skipper for {field}, whose field info has a skip index"
                );
                Ok(None)
            }
        }
    }

    /// Java's `AssertingSortedNumericDocValues.create` keeps a singleton a
    /// singleton.
    fn sorted_numeric_is_singleton(&self, field: &str) -> bool {
        self.in_.sorted_numeric_is_singleton(field)
    }

    fn check_integrity(&self) -> Result<()> {
        self.in_.check_integrity()
    }
}

/// `AssertingDirectoryReader`: `in` with every leaf an
/// [`AssertingLeafReader`].
///
/// # Errors
/// What wrapping the leaves returns (nothing, as the wrapper cannot fail).
pub fn asserting_directory_reader(in_: Arc<DirectoryReader>) -> Result<FilterDirectoryReader> {
    FilterDirectoryReader::new(in_, Arc::new(AssertingSubReaderWrapper))
}

/// `AssertingDirectoryReader.AssertingSubReaderWrapper`.
#[derive(Debug, Default, Clone, Copy)]
pub struct AssertingSubReaderWrapper;

impl SubReaderWrapper for AssertingSubReaderWrapper {
    fn wrap(&self, reader: Arc<dyn LeafReader>) -> Result<Arc<dyn LeafReader>> {
        Ok(Arc::new(AssertingLeafReader::new(reader)))
    }
}

// ---------------------------------------------------------------------------
// Terms
// ---------------------------------------------------------------------------

/// `AssertingTerms`.
pub struct AssertingTerms<'a> {
    in_: Box<dyn Terms + 'a>,
}

impl<'a> AssertingTerms<'a> {
    /// Wraps `in_`.
    pub fn new(in_: Box<dyn Terms + 'a>) -> Self {
        Self { in_ }
    }
}

impl Terms for AssertingTerms<'_> {
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
        Ok(Box::new(AssertingTermsEnum::new(
            self.in_.iterator()?,
            self.in_.has_freqs(),
        )))
    }
    fn intersect<'s>(
        &'s self,
        dfa: &'s ByteDfa,
        start_term: Option<&[u8]>,
    ) -> Result<Box<dyn TermsEnum + 's>> {
        Ok(Box::new(AssertingTermsEnum::new(
            self.in_.intersect(dfa, start_term)?,
            self.in_.has_freqs(),
        )))
    }
    fn size(&self) -> i64 {
        self.in_.size()
    }
    fn sum_total_term_freq(&self) -> i64 {
        // Java: AssertingLeafReader.AssertingTerms.getSumTotalTermFreq.
        let sum_ttf = self.in_.sum_total_term_freq();
        if !self.in_.has_freqs() {
            let sum_df = self.in_.sum_doc_freq();
            assert!(
                sum_ttf == sum_df,
                "sumTotalTermFreq={sum_ttf} != sumDocFreq={sum_df} without freqs"
            );
        }
        let sum_df = self.sum_doc_freq();
        assert!(
            sum_ttf >= sum_df,
            "sumTotalTermFreq={sum_ttf} < sumDocFreq={sum_df}"
        );
        sum_ttf
    }
    fn sum_doc_freq(&self) -> i64 {
        let sum_df = self.in_.sum_doc_freq();
        let doc_count = self.doc_count();
        assert!(
            sum_df >= i64::from(doc_count),
            "sumDocFreq={sum_df} < docCount={doc_count}"
        );
        sum_df
    }
    fn doc_count(&self) -> i32 {
        let doc_count = self.in_.doc_count();
        assert!(doc_count > 0, "docCount={doc_count}");
        doc_count
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TermsState {
    Initial,
    Positioned,
    Unpositioned,
}

/// `AssertingTermsEnum`: the INITIAL/POSITIONED/UNPOSITIONED state machine.
pub struct AssertingTermsEnum<'a> {
    in_: Box<dyn TermsEnum + 'a>,
    has_freqs: bool,
    state: TermsState,
}

impl<'a> AssertingTermsEnum<'a> {
    /// Wraps `in_`, a field's terms whose `hasFreqs()` is `has_freqs`.
    pub fn new(in_: Box<dyn TermsEnum + 'a>, has_freqs: bool) -> Self {
        Self {
            in_,
            has_freqs,
            state: TermsState::Initial,
        }
    }

    fn positioned(&self, method: &str) {
        assert!(
            self.state == TermsState::Positioned,
            "{method} called on unpositioned TermsEnum"
        );
    }
}

impl TermsEnum for AssertingTermsEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        assert!(
            matches!(self.state, TermsState::Initial | TermsState::Positioned),
            "next() called on unpositioned TermsEnum"
        );
        let result = self.in_.next()?;
        self.state = if result.is_some() {
            TermsState::Positioned
        } else {
            TermsState::Unpositioned
        };
        Ok(result)
    }
    fn term(&self) -> Option<&[u8]> {
        self.positioned("term()");
        self.in_.term()
    }
    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        let status = self.in_.try_seek_ceil(target)?;
        self.state = if status == SeekStatus::End {
            TermsState::Unpositioned
        } else {
            TermsState::Positioned
        };
        Ok(status)
    }
    fn try_seek_exact(&mut self, target: &[u8]) -> Result<bool> {
        let found = self.in_.try_seek_exact(target)?;
        self.state = if found {
            TermsState::Positioned
        } else {
            TermsState::Unpositioned
        };
        Ok(found)
    }
    fn doc_freq(&mut self) -> Result<i32> {
        self.positioned("docFreq()");
        let df = self.in_.doc_freq()?;
        assert!(df > 0, "docFreq={df}");
        Ok(df)
    }
    fn total_term_freq(&mut self) -> Result<i64> {
        self.positioned("totalTermFreq()");
        let ttf = self.in_.total_term_freq()?;
        let df = i64::from(self.doc_freq()?);
        if self.has_freqs {
            assert!(ttf >= df, "totalTermFreq={ttf} < docFreq={df}");
        } else {
            assert!(
                ttf == df,
                "totalTermFreq={ttf} != docFreq={df} without freqs"
            );
        }
        Ok(ttf)
    }
    fn postings(&mut self, flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        self.positioned("docs(...)");
        Ok(Box::new(AssertingPostingsEnum::new(
            self.in_.postings(flags)?,
        )))
    }
    fn impacts(&mut self, flags: PostingsFlags) -> Result<Box<dyn ImpactsEnum>> {
        self.positioned("docs(...)");
        assert!(
            flags >= PostingsFlags::Freqs,
            "Freqs should be requested on impacts"
        );
        Ok(Box::new(AssertingImpactsEnum::new(
            self.in_.impacts(flags)?,
        )))
    }
    fn ord(&self) -> Result<i64> {
        // SENTINEL-OK: the "ord()" is a state label, not a call; the
        // wrapped `ord` goes back to the caller unchanged.
        self.positioned("ord()");
        self.in_.ord()
    }
    fn seek_exact_ord(&mut self, ord: i64) -> Result<()> {
        self.in_.seek_exact_ord(ord)?;
        self.state = TermsState::Positioned;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Postings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DocsState {
    Start,
    Iterating,
    Finished,
}

/// `AssertingPostingsEnum`: the START/ITERATING/FINISHED state machine and
/// the position budget `freq()` sets. The wrapper is neither `Send` nor `Sync`,
/// even when its concrete delegate is: Java's creation-thread checks are
/// enforced by this thread restriction. The public constructor keeps its
/// original erased delegate type; the marker preserves the same restriction
/// for internal concrete delegate shapes too.
///
/// The concrete wrapper type fails the `Send` bound required by thread spawn,
/// even though its delegate passes that bound:
///
/// ```compile_fail
/// use lucene_search::reader::MaterializedPostings;
/// use lucene_test_framework::index::asserting_leaf_reader::AssertingPostingsEnum;
///
/// fn require_send<T: Send>() {}
/// require_send::<MaterializedPostings>();
/// require_send::<AssertingPostingsEnum<MaterializedPostings>>();
/// ```
///
/// The public constructor erases different concrete delegates identically:
///
/// ```
/// use lucene_search::reader::{DocIdSetIterator, MaterializedPostings, SlowImpactsEnum};
/// use lucene_test_framework::index::asserting_leaf_reader::AssertingPostingsEnum;
///
/// let first = MaterializedPostings::new(vec![0], vec![1], None).unwrap();
/// let mut checked = AssertingPostingsEnum::new(Box::new(first));
/// assert_eq!(checked.cost(), 1);
/// let second = MaterializedPostings::new(vec![0], vec![1], None).unwrap();
/// checked = AssertingPostingsEnum::new(Box::new(SlowImpactsEnum::new(Box::new(second))));
/// assert_eq!(checked.cost(), 1);
/// ```
pub struct AssertingPostingsEnum<P: PostingsEnum + ?Sized = dyn PostingsEnum> {
    in_: Box<P>,
    // Preserve the non-thread-transferable trait-object wrapper's contract.
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
    state: DocsState,
    position_count: i32,
    position_max: i32,
    doc: i32,
}

impl AssertingPostingsEnum {
    /// Wraps `in_`, preserving the original erased-delegate constructor API.
    pub fn new(in_: Box<dyn PostingsEnum>) -> Self {
        Self::new_inner(in_)
    }
}

impl<P: PostingsEnum + ?Sized> AssertingPostingsEnum<P> {
    fn new_inner(in_: Box<P>) -> Self {
        let doc = in_.doc_id();
        Self {
            in_,
            _thread_bound: std::marker::PhantomData,
            state: DocsState::Start,
            position_count: 0,
            position_max: 0,
            doc,
        }
    }

    fn moved_to(&mut self, doc: i32) -> i32 {
        if doc == NO_MORE_DOCS {
            self.state = DocsState::Finished;
            self.position_max = 0;
        } else {
            self.state = DocsState::Iterating;
            self.position_max = self.in_.freq();
        }
        self.position_count = 0;
        assert!(self.in_.doc_id() == doc, "docID() disagrees with {doc}");
        self.doc = doc;
        doc
    }

    fn on_a_doc(&self, method: &str) {
        assert!(
            self.state != DocsState::Start,
            "{method}() called before nextDoc()/advance()"
        );
        assert!(
            self.state != DocsState::Finished,
            "{method}() called after NO_MORE_DOCS"
        );
    }

    fn on_a_position(&self, method: &str) {
        self.on_a_doc(method);
        assert!(
            self.position_count > 0,
            "{method}() called before nextPosition()!"
        );
    }
}

impl<P: PostingsEnum + ?Sized> DocIdSetIterator for AssertingPostingsEnum<P> {
    fn doc_id(&self) -> i32 {
        let inner = self.in_.doc_id();
        assert!(
            self.doc == inner,
            " invalid docID() {inner} instead of {}",
            self.doc
        );
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        assert!(
            self.state != DocsState::Finished,
            "nextDoc() called after NO_MORE_DOCS"
        );
        let next = self.in_.next_doc()?;
        assert!(
            next > self.doc,
            "backwards nextDoc from {} to {next}",
            self.doc
        );
        Ok(self.moved_to(next))
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        assert!(
            self.state != DocsState::Finished,
            "advance() called after NO_MORE_DOCS"
        );
        assert!(
            target > self.doc,
            "target must be > docID(), got {target} <= {}",
            self.doc
        );
        let advanced = self.in_.advance(target)?;
        assert!(
            advanced >= target,
            "backwards advance from: {target} to: {advanced}"
        );
        Ok(self.moved_to(advanced))
    }
    fn cost(&self) -> i64 {
        self.in_.cost()
    }
}

impl<P: PostingsEnum + ?Sized> PostingsEnum for AssertingPostingsEnum<P> {
    fn freq(&self) -> i32 {
        self.on_a_doc("freq");
        let freq = self.in_.freq();
        assert!(freq > 0, "freq={freq}");
        freq
    }
    fn next_position(&mut self) -> Result<i32> {
        self.on_a_doc("nextPosition");
        assert!(
            self.position_count < self.position_max,
            "nextPosition() called more than freq() times!"
        );
        let position = self.in_.next_position()?;
        assert!(
            position >= 0 || position == -1,
            "invalid position: {position}"
        );
        self.position_count += 1;
        Ok(position)
    }
    fn start_offset(&self) -> i32 {
        self.on_a_position("startOffset");
        self.in_.start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.on_a_position("endOffset");
        self.in_.end_offset()
    }
    fn payload(&self) -> Option<&[u8]> {
        self.on_a_position("getPayload");
        let payload = self.in_.payload();
        assert!(
            payload.is_none_or(|p| !p.is_empty()),
            "getPayload() returned payload with invalid length!"
        );
        payload
    }
}

/// Java's `AssertingLeafReader.AssertingImpactsEnum`: postings checks plus
/// shallow-position ordering and `CheckIndex.checkImpacts` validation.
pub struct AssertingImpactsEnum {
    postings: AssertingPostingsEnum<dyn ImpactsEnum>,
    // SENTINEL: `-1` means advanceShallow has not been called; compared only.
    last_shallow_target: i32,
}

impl AssertingImpactsEnum {
    /// Wraps the same impacts iterator in the inherited postings checks.
    pub fn new(inner: Box<dyn ImpactsEnum>) -> Self {
        Self {
            postings: AssertingPostingsEnum::new_inner(inner),
            last_shallow_target: -1,
        }
    }
}

impl DocIdSetIterator for AssertingImpactsEnum {
    fn doc_id(&self) -> i32 {
        self.postings.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        // Java int addition wraps when docID is NO_MORE_DOCS.
        assert!(
            self.doc_id().wrapping_add(1) >= self.last_shallow_target,
            "target = {} < last shallow target = {}",
            self.doc_id().wrapping_add(1),
            self.last_shallow_target
        );
        self.postings.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        assert!(
            target >= self.last_shallow_target,
            "target = {target} < last shallow target = {}",
            self.last_shallow_target
        );
        self.postings.advance(target)
    }
    fn cost(&self) -> i64 {
        self.postings.cost()
    }
}

impl PostingsEnum for AssertingImpactsEnum {
    fn freq(&self) -> i32 {
        self.postings.freq()
    }
    fn next_position(&mut self) -> Result<i32> {
        self.postings.next_position()
    }
    fn start_offset(&self) -> i32 {
        self.postings.start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.postings.end_offset()
    }
    fn payload(&self) -> Option<&[u8]> {
        self.postings.payload()
    }
}

impl ImpactsEnum for AssertingImpactsEnum {
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        assert!(
            target >= self.last_shallow_target,
            "called on decreasing targets: target = {target} < last target = {}",
            self.last_shallow_target
        );
        assert!(
            target >= self.doc_id(),
            "target = {target} < docID = {}",
            self.doc_id()
        );
        self.last_shallow_target = target;
        self.postings.in_.advance_shallow(target)
    }
    fn impacts(&self) -> Vec<(i32, Vec<Impact>)> {
        assert!(
            self.doc_id() >= 0 || self.last_shallow_target >= 0,
            "Cannot get impacts until the iterator is positioned or advanceShallow has been called"
        );
        let impacts = self.postings.in_.impacts();
        lucene_index::check_index::check_impacts(
            &impacts,
            self.doc_id().max(self.last_shallow_target),
        )
        .unwrap_or_else(|message| panic!("{message}"));
        impacts
    }
}

// ---------------------------------------------------------------------------
// Doc values
// ---------------------------------------------------------------------------

/// What every asserting doc-values iterator tracks: Java's `lastDocID`,
/// `exists` and `maxDoc`.
struct DvCheck {
    kind: &'static str,
    max_doc: i32,
    last_doc_id: i32,
    exists: bool,
}

impl DvCheck {
    fn new(kind: &'static str, max_doc: i32) -> Self {
        Self {
            kind,
            max_doc,
            last_doc_id: -1,
            exists: false,
        }
    }

    fn in_range(&self, doc: i32) {
        assert!(
            doc == NO_MORE_DOCS || doc < self.max_doc,
            "{} doc values went to doc {doc} of maxDoc={}",
            self.kind,
            self.max_doc
        );
    }

    fn after_next(&mut self, doc: i32, in_doc: impl FnOnce() -> i32) {
        assert!(
            doc > self.last_doc_id,
            "{} doc values: backwards nextDoc from {} to {doc}",
            self.kind,
            self.last_doc_id
        );
        self.in_range(doc);
        assert!(
            doc == in_doc(),
            "{} doc values: docID() disagrees",
            self.kind
        );
        self.last_doc_id = doc;
        self.exists = doc != NO_MORE_DOCS;
    }

    fn before_advance(&self, target: i32, in_doc: impl FnOnce() -> i32) {
        assert!(
            target >= 0,
            "{} doc values: advance({target}) on negative target",
            self.kind
        );
        let in_doc = in_doc();
        assert!(
            target > in_doc,
            "{} doc values: advance({target}) from doc {in_doc}",
            self.kind
        );
    }

    fn after_advance(&mut self, target: i32, doc: i32) {
        assert!(
            doc >= target,
            "{} doc values: backwards advance from: {target} to: {doc}",
            self.kind
        );
        self.in_range(doc);
        self.last_doc_id = doc;
        self.exists = doc != NO_MORE_DOCS;
    }

    fn before_advance_exact(&self, target: i32, in_doc: impl FnOnce() -> i32) {
        assert!(
            target >= 0,
            "{} doc values: advanceExact({target}) on negative target",
            self.kind
        );
        let in_doc = in_doc();
        assert!(
            target >= in_doc,
            "{} doc values: advanceExact({target}) from doc {in_doc}, maxDoc={}",
            self.kind,
            self.max_doc
        );
        assert!(
            target < self.max_doc,
            "{} doc values: advanceExact({target}) from doc {in_doc}, maxDoc={}",
            self.kind,
            self.max_doc
        );
    }

    fn after_advance_exact(&mut self, target: i32, in_doc: i32) {
        assert!(
            in_doc == target,
            "{} doc values: advanceExact({target}) left the iterator on {in_doc}",
            self.kind
        );
        self.last_doc_id = target;
    }

    fn exists(&self, method: &str) {
        assert!(
            self.exists,
            "{} doc values: {method}() called on a document without a value",
            self.kind
        );
    }

    fn cost(&self, cost: i64) -> i64 {
        assert!(cost >= 0, "{} doc values: cost={cost}", self.kind);
        cost
    }
}

/// `DocIdSetIterator` and `DocValuesIterator` for an asserting wrapper with
/// `in_` and `check: DvCheck`, running `$reset` (if any) whenever it moves.
macro_rules! asserting_dv_iterator {
    ($ty:ident, $check_advanced_doc:literal $(, |$s:ident| $reset:expr)?) => {
        impl DocIdSetIterator for $ty<'_> {
            fn doc_id(&self) -> i32 {
                self.in_.doc_id()
            }
            fn next_doc(&mut self) -> Result<i32> {
                let doc = self.in_.next_doc()?;
                // Java checks monotonicity and bounds before calling docID().
                let in_ = &self.in_;
                self.check.after_next(doc, || in_.doc_id());
                $(
                    let $s = &mut *self;
                    $reset;
                )?
                Ok(doc)
            }
            fn advance(&mut self, target: i32) -> Result<i32> {
                self.check.before_advance(target, || self.in_.doc_id());
                let doc = self.in_.advance(target)?;
                // Only Java's multi-valued wrappers check docID() after advance.
                if $check_advanced_doc {
                    assert!(
                        doc == self.in_.doc_id(),
                        "{} doc values: docID() disagrees",
                        self.check.kind
                    );
                }
                self.check.after_advance(target, doc);
                $(
                    let $s = &mut *self;
                    $reset;
                )?
                Ok(doc)
            }
            fn cost(&self) -> i64 {
                self.check.cost(self.in_.cost())
            }
        }

        impl DocValuesIterator for $ty<'_> {
            fn advance_exact(&mut self, target: i32) -> Result<bool> {
                self.check.before_advance_exact(target, || self.in_.doc_id());
                // Java commits exists before the post-call assertion, but leaves
                // lastDocID and per-document counters unchanged if it fails.
                self.check.exists = self.in_.advance_exact(target)?;
                self.check
                    .after_advance_exact(target, self.in_.doc_id());
                $(
                    let $s = &mut *self;
                    $reset;
                )?
                Ok(self.check.exists)
            }
        }
    };
}

/// `AssertingNumericDocValues` (norms too).
pub struct AssertingNumericDocValues<'a> {
    in_: Box<dyn NumericDocValues + 'a>,
    check: DvCheck,
}

impl<'a> AssertingNumericDocValues<'a> {
    /// Wraps `in_`, an unpositioned iterator over a leaf of `max_doc`.
    pub fn new(in_: Box<dyn NumericDocValues + 'a>, max_doc: i32) -> Self {
        let in_doc = in_.doc_id();
        assert!(in_doc == -1, "Numeric doc values start on doc {in_doc}");
        let check = DvCheck::new("Numeric", max_doc);
        Self { in_, check }
    }
}

asserting_dv_iterator!(AssertingNumericDocValues, false);

impl NumericDocValues for AssertingNumericDocValues<'_> {
    fn long_value(&self) -> i64 {
        self.check.exists("longValue");
        self.in_.long_value()
    }
}

/// `AssertingBinaryDocValues`.
pub struct AssertingBinaryDocValues<'a> {
    in_: Box<dyn BinaryDocValues + 'a>,
    check: DvCheck,
}

impl<'a> AssertingBinaryDocValues<'a> {
    /// Wraps `in_`, an unpositioned iterator over a leaf of `max_doc`.
    pub fn new(in_: Box<dyn BinaryDocValues + 'a>, max_doc: i32) -> Self {
        let in_doc = in_.doc_id();
        assert!(in_doc == -1, "Binary doc values start on doc {in_doc}");
        let check = DvCheck::new("Binary", max_doc);
        Self { in_, check }
    }
}

asserting_dv_iterator!(AssertingBinaryDocValues, false);

impl BinaryDocValues for AssertingBinaryDocValues<'_> {
    fn binary_value(&self) -> &[u8] {
        self.check.exists("binaryValue");
        self.in_.binary_value()
    }
}

/// `AssertingSortedDocValues`.
pub struct AssertingSortedDocValues<'a> {
    in_: Box<dyn SortedDocValues + 'a>,
    check: DvCheck,
    value_count: i32,
}

impl<'a> AssertingSortedDocValues<'a> {
    /// Wraps `in_` over a leaf of `max_doc`, checking its value count.
    pub fn new(in_: Box<dyn SortedDocValues + 'a>, max_doc: i32) -> Self {
        let check = DvCheck::new("Sorted", max_doc);
        let value_count = in_.value_count();
        assert!(
            value_count >= 0 && value_count <= max_doc,
            "Sorted doc values: valueCount={value_count} of maxDoc={max_doc}"
        );
        Self {
            in_,
            check,
            value_count,
        }
    }
}

asserting_dv_iterator!(AssertingSortedDocValues, false);

impl SortedDocValues for AssertingSortedDocValues<'_> {
    fn ord_value(&self) -> i32 {
        self.check.exists("ordValue");
        let ord = self.in_.ord_value();
        assert!(
            ord >= -1 && ord < self.value_count,
            "ord={ord} of valueCount={}",
            self.value_count
        );
        ord
    }
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
        assert!(
            ord >= 0 && ord < self.value_count,
            "lookupOrd({ord}) of valueCount={}",
            self.value_count
        );
        self.in_.lookup_ord(ord)
    }
    fn value_count(&self) -> i32 {
        let value_count = self.in_.value_count();
        assert!(
            value_count == self.value_count,
            "valueCount changed from {} to {value_count}",
            self.value_count
        );
        value_count
    }
    fn lookup_term(&mut self, key: &[u8]) -> Result<i32> {
        let result = self.in_.lookup_term(key)?;
        assert!(
            result < self.value_count,
            "lookupTerm returned {result} of valueCount={}",
            self.value_count
        );
        Ok(result)
    }
}

/// `AssertingSortedNumericDocValues`.
pub struct AssertingSortedNumericDocValues<'a> {
    in_: Box<dyn SortedNumericDocValues + 'a>,
    check: DvCheck,
    value_upto: i32,
}

impl<'a> AssertingSortedNumericDocValues<'a> {
    /// Wraps `in_` over a leaf of `max_doc` without inspecting its position.
    pub fn new(in_: Box<dyn SortedNumericDocValues + 'a>, max_doc: i32) -> Self {
        let check = DvCheck::new("Sorted numeric", max_doc);
        Self {
            in_,
            check,
            value_upto: 0,
        }
    }
}

asserting_dv_iterator!(AssertingSortedNumericDocValues, true, |s| s.value_upto = 0);

impl SortedNumericDocValues for AssertingSortedNumericDocValues<'_> {
    fn doc_value_count(&self) -> i32 {
        self.check.exists("docValueCount");
        let count = self.in_.doc_value_count();
        assert!(count > 0, "docValueCount={count}");
        self.in_.doc_value_count()
    }
    fn next_value(&mut self) -> Result<i64> {
        self.check.exists("nextValue");
        assert!(
            self.value_upto < self.in_.doc_value_count(),
            "valueUpto={} in.docValueCount()={}",
            self.value_upto,
            self.in_.doc_value_count()
        );
        self.value_upto += 1;
        self.in_.next_value()
    }
    fn take_values(&mut self) -> Option<&[i64]> {
        self.check.exists("takeValues");
        self.value_upto = self.in_.doc_value_count();
        self.in_.take_values()
    }
}

/// `AssertingSortedSetDocValues`.
pub struct AssertingSortedSetDocValues<'a> {
    in_: Box<dyn SortedSetDocValues + 'a>,
    check: DvCheck,
    value_count: i64,
    ords_retrieved: i32,
}

impl<'a> AssertingSortedSetDocValues<'a> {
    /// Wraps `in_` over a leaf of `max_doc`, checking its value count.
    pub fn new(in_: Box<dyn SortedSetDocValues + 'a>, max_doc: i32) -> Self {
        let check = DvCheck::new("Sorted set", max_doc);
        let value_count = in_.value_count();
        assert!(
            value_count >= 0,
            "Sorted set doc values: valueCount={value_count}"
        );
        Self {
            in_,
            check,
            value_count,
            ords_retrieved: 0,
        }
    }
}

asserting_dv_iterator!(AssertingSortedSetDocValues, true, |s| s.ords_retrieved = 0);

impl SortedSetDocValues for AssertingSortedSetDocValues<'_> {
    fn doc_value_count(&self) -> i32 {
        self.in_.doc_value_count()
    }
    fn next_ord(&mut self) -> Result<i64> {
        self.check.exists("nextOrd");
        assert!(
            self.ords_retrieved < self.doc_value_count(),
            "nextOrd() called more than docValueCount() times"
        );
        self.ords_retrieved += 1;
        let ord = self.in_.next_ord()?;
        assert!(
            ord < self.value_count,
            "ord={ord} of valueCount={}",
            self.value_count
        );
        Ok(ord)
    }
    fn take_ords(&mut self) -> Option<&[i64]> {
        self.check.exists("takeOrds");
        self.ords_retrieved = self.in_.doc_value_count();
        let value_count = self.value_count;
        let ords = self.in_.take_ords();
        if let Some(ords) = ords {
            assert!(
                ords.iter().all(|&o| o >= 0 && o < value_count),
                "ords {ords:?} of valueCount={value_count}"
            );
        }
        ords
    }
    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        assert!(
            ord >= 0 && ord < self.value_count,
            "lookupOrd({ord}) of valueCount={}",
            self.value_count
        );
        self.in_.lookup_ord(ord)
    }
    fn value_count(&self) -> i64 {
        let value_count = self.in_.value_count();
        assert!(
            value_count == self.value_count,
            "valueCount changed from {} to {value_count}",
            self.value_count
        );
        value_count
    }
    fn lookup_term(&mut self, key: &[u8]) -> Result<i64> {
        let result = self.in_.lookup_term(key)?;
        assert!(
            result < self.value_count,
            "lookupTerm returned {result} of valueCount={}",
            self.value_count
        );
        Ok(result)
    }
}

#[cfg(test)]
mod doc_values_callback_tests {
    use super::*;
    use std::cell::RefCell;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::rc::Rc;

    struct State {
        calls: Vec<&'static str>,
        doc: i32,
        returned_doc: i32,
        reported_doc: i32,
        exists: bool,
        counts: Vec<i32>,
        count_calls: usize,
        panic_count_at: Option<usize>,
        fail_value: bool,
    }
    impl Default for State {
        fn default() -> Self {
            Self {
                calls: Vec::new(),
                doc: -1,
                returned_doc: 0,
                reported_doc: 0,
                exists: true,
                counts: vec![1],
                count_calls: 0,
                panic_count_at: None,
                fail_value: false,
            }
        }
    }
    type Shared = Rc<RefCell<State>>;
    struct Probe(Shared);
    impl Probe {
        fn step(&self, call: &'static str) -> i32 {
            let mut state = self.0.borrow_mut();
            state.calls.push(call);
            state.doc = state.reported_doc;
            state.returned_doc
        }
        fn count(&self) -> i32 {
            let mut state = self.0.borrow_mut();
            state.calls.push("count");
            state.count_calls += 1;
            assert_ne!(
                Some(state.count_calls),
                state.panic_count_at,
                "count callback"
            );
            state.counts[(state.count_calls - 1).min(state.counts.len() - 1)]
        }
        fn value(&self, call: &'static str) -> Result<i64> {
            let mut state = self.0.borrow_mut();
            state.calls.push(call);
            if state.fail_value {
                return Err(lucene_store::Error::Io(std::io::Error::other("value failure")).into());
            }
            Ok(0)
        }
    }
    impl DocIdSetIterator for Probe {
        fn doc_id(&self) -> i32 {
            let mut state = self.0.borrow_mut();
            state.calls.push("docID");
            state.doc
        }
        fn next_doc(&mut self) -> Result<i32> {
            Ok(self.step("next"))
        }
        fn advance(&mut self, _: i32) -> Result<i32> {
            Ok(self.step("advance"))
        }
        fn cost(&self) -> i64 {
            1
        }
    }
    impl DocValuesIterator for Probe {
        fn advance_exact(&mut self, _: i32) -> Result<bool> {
            self.step("exact");
            Ok(self.0.borrow().exists)
        }
    }
    impl NumericDocValues for Probe {
        fn long_value(&self) -> i64 {
            7
        }
    }
    impl BinaryDocValues for Probe {
        fn binary_value(&self) -> &[u8] {
            b"v"
        }
    }
    impl SortedDocValues for Probe {
        fn ord_value(&self) -> i32 {
            0
        }
        fn lookup_ord(&mut self, _: i32) -> Result<Vec<u8>> {
            Ok(vec![0])
        }
        fn value_count(&self) -> i32 {
            2
        }
    }
    impl SortedNumericDocValues for Probe {
        fn doc_value_count(&self) -> i32 {
            self.count()
        }
        fn next_value(&mut self) -> Result<i64> {
            self.value("value")
        }
    }
    impl SortedSetDocValues for Probe {
        fn doc_value_count(&self) -> i32 {
            self.count()
        }
        fn next_ord(&mut self) -> Result<i64> {
            self.value("ord")
        }
        fn lookup_ord(&mut self, _: i64) -> Result<Vec<u8>> {
            Ok(vec![0])
        }
        fn value_count(&self) -> i64 {
            2
        }
    }
    fn iterator(kind: usize, state: &Shared) -> Box<dyn DocValuesIterator> {
        let probe = Box::new(Probe(state.clone()));
        let iterator: Box<dyn DocValuesIterator> = match kind {
            0 => Box::new(AssertingNumericDocValues::new(probe, 4)),
            1 => Box::new(AssertingBinaryDocValues::new(probe, 4)),
            2 => Box::new(AssertingSortedDocValues::new(probe, 4)),
            3 => Box::new(AssertingSortedNumericDocValues::new(probe, 4)),
            4 => Box::new(AssertingSortedSetDocValues::new(probe, 4)),
            _ => unreachable!(),
        };
        state.borrow_mut().calls.clear();
        iterator
    }
    fn panics(action: impl FnOnce()) {
        assert!(catch_unwind(AssertUnwindSafe(action)).is_err());
    }

    #[test]
    fn movement_short_circuits_before_observing_delegate_position() {
        for kind in 0..5 {
            for returned_doc in [-1, 4] {
                let state = Shared::default();
                let mut values = iterator(kind, &state);
                state.borrow_mut().returned_doc = returned_doc;
                panics(|| {
                    values.next_doc().unwrap();
                });
                assert_eq!(state.borrow().calls, ["next"], "kind={kind}");
            }
            for exact in [false, true] {
                let state = Shared::default();
                let mut values = iterator(kind, &state);
                panics(|| {
                    if exact {
                        values.advance_exact(-1).unwrap();
                    } else {
                        values.advance(-1).unwrap();
                    }
                });
                assert!(state.borrow().calls.is_empty(), "kind={kind}");
            }
            let state = Shared::default();
            let mut values = iterator(kind, &state);
            state.borrow_mut().reported_doc = 1;
            if kind < 3 {
                // Java accepts the returned document without querying docID again.
                assert_eq!(values.advance(0).unwrap(), 0);
                assert_eq!(state.borrow().calls, ["docID", "advance"]);
            } else {
                panics(|| {
                    values.advance(0).unwrap();
                });
                assert_eq!(state.borrow().calls, ["docID", "advance", "docID"]);
            }
        }
    }

    #[test]
    fn advance_exact_commits_exists_before_failed_postcondition_only() {
        macro_rules! check {
            ($wrapper:ident) => {
                for exists in [false, true] {
                    let state = Shared::default();
                    let mut values = $wrapper::new(Box::new(Probe(state.clone())), 4);
                    values.check.exists = !exists;
                    state.borrow_mut().reported_doc = 1;
                    state.borrow_mut().exists = exists;
                    panics(|| {
                        values.advance_exact(0).unwrap();
                    });
                    assert_eq!(values.check.exists, exists);
                    assert_eq!(values.check.last_doc_id, -1);
                }
            };
        }
        check!(AssertingNumericDocValues);
        check!(AssertingBinaryDocValues);
        check!(AssertingSortedDocValues);
        check!(AssertingSortedNumericDocValues);
        check!(AssertingSortedSetDocValues);

        let state = Shared::default();
        let mut numeric = AssertingSortedNumericDocValues::new(Box::new(Probe(state.clone())), 4);
        numeric.value_upto = 1;
        state.borrow_mut().reported_doc = 1;
        panics(|| {
            numeric.advance_exact(0).unwrap();
        });
        assert_eq!(numeric.value_upto, 1);
        assert!(numeric.advance_exact(1).unwrap());
        assert_eq!(numeric.value_upto, 0);

        let state = Shared::default();
        let mut set = AssertingSortedSetDocValues::new(Box::new(Probe(state.clone())), 4);
        set.ords_retrieved = 1;
        state.borrow_mut().reported_doc = 1;
        panics(|| {
            set.advance_exact(0).unwrap();
        });
        assert_eq!(set.ords_retrieved, 1);
        assert!(set.advance_exact(1).unwrap());
        assert_eq!(set.ords_retrieved, 0);
    }

    #[test]
    fn per_kind_count_getters_preserve_duplicate_and_unchecked_calls() {
        let state = Shared::default();
        let mut numeric = AssertingSortedNumericDocValues::new(Box::new(Probe(state.clone())), 4);
        numeric.next_doc().unwrap();
        state.borrow_mut().counts = vec![1, 0];
        state.borrow_mut().calls.clear();
        assert_eq!(numeric.doc_value_count(), 0);
        assert_eq!(state.borrow().calls, ["count", "count"]);

        state.borrow_mut().count_calls = 0;
        state.borrow_mut().counts = vec![0];
        state.borrow_mut().calls.clear();
        panics(|| {
            numeric.doc_value_count();
        });
        assert_eq!(state.borrow().calls, ["count"]);

        let state = Shared::default();
        state.borrow_mut().counts = vec![0];
        let mut set = AssertingSortedSetDocValues::new(Box::new(Probe(state.clone())), 4);
        assert_eq!(set.doc_value_count(), 0);
        assert_eq!(state.borrow().calls, ["count"]);
        set.next_doc().unwrap();
        state.borrow_mut().calls.clear();
        assert_eq!(set.doc_value_count(), 0);
        assert_eq!(state.borrow().calls, ["count"]);
    }

    #[test]
    fn failed_value_callbacks_keep_increment_and_assertion_message_reads() {
        let state = Shared::default();
        let mut numeric = AssertingSortedNumericDocValues::new(Box::new(Probe(state.clone())), 4);
        numeric.next_doc().unwrap();
        state.borrow_mut().fail_value = true;
        state.borrow_mut().calls.clear();
        assert!(numeric.next_value().is_err());
        assert_eq!(numeric.value_upto, 1);
        assert_eq!(state.borrow().calls, ["count", "value"]);
        state.borrow_mut().calls.clear();
        // The failing condition reads count, then its message reads count again.
        state.borrow_mut().panic_count_at = Some(3);
        let failure = catch_unwind(AssertUnwindSafe(|| numeric.next_value())).unwrap_err();
        assert!(failure
            .downcast_ref::<String>()
            .unwrap()
            .contains("count callback"));
        assert_eq!(state.borrow().calls, ["count", "count"]);
        assert_eq!(numeric.value_upto, 1);

        let state = Shared::default();
        let mut set = AssertingSortedSetDocValues::new(Box::new(Probe(state.clone())), 4);
        set.next_doc().unwrap();
        state.borrow_mut().fail_value = true;
        state.borrow_mut().calls.clear();
        assert!(set.next_ord().is_err());
        assert_eq!(set.ords_retrieved, 1);
        assert_eq!(state.borrow().calls, ["count", "ord"]);
        state.borrow_mut().calls.clear();
        panics(|| {
            set.next_ord().unwrap();
        });
        assert_eq!(set.ords_retrieved, 1);
        assert_eq!(state.borrow().calls, ["count"]);
    }
}

// ---------------------------------------------------------------------------
// Doc values skipper
// ---------------------------------------------------------------------------

/// `AssertingDocValuesSkipper`: unpositioned at -1 when created, advanced
/// only beyond its current interval, each level nested in the one above,
/// and value statistics read only while positioned. Java's
/// `Objects.checkIndex(level, numLevels())` is an index panic here too.
#[derive(Debug, Clone)]
pub struct AssertingDocValuesSkipper<'a> {
    in_: DocValuesSkipper<'a>,
}

impl<'a> AssertingDocValuesSkipper<'a> {
    /// `new AssertingDocValuesSkipper(in)`.
    ///
    /// # Panics
    /// When `in_` is already positioned.
    pub fn new(in_: DocValuesSkipper<'a>) -> Self {
        let s = Self { in_ };
        assert_eq!(s.min_doc_id(0), -1);
        assert_eq!(s.max_doc_id(0), -1);
        s
    }

    fn check_index(&self, level: usize) {
        let n = self.in_.num_levels();
        assert!(level < n, "Index {level} out of bounds for length {n}");
    }

    fn iterating(&self) -> bool {
        let (min, max) = (self.min_doc_id(0), self.max_doc_id(0));
        max != -1 && min != -1 && max != NO_MORE_DOCS && min != NO_MORE_DOCS
    }
}

impl SkipperImpl for AssertingDocValuesSkipper<'_> {
    fn advance(&mut self, target: i32) {
        assert!(
            target > self.max_doc_id(0),
            "Illegal to call advance() on a target that is not beyond the current interval"
        );
        self.in_.advance(target);
        assert!(self.in_.min_doc_id(0) <= self.in_.max_doc_id(0));
    }

    fn num_levels(&self) -> usize {
        self.in_.num_levels()
    }

    fn min_doc_id(&self, level: usize) -> i32 {
        self.check_index(level);
        let min = self.in_.min_doc_id(level);
        assert!(min <= self.in_.max_doc_id(level));
        if level > 0 {
            assert!(min <= self.in_.min_doc_id(level - 1));
        }
        min
    }

    fn max_doc_id(&self, level: usize) -> i32 {
        self.check_index(level);
        let max = self.in_.max_doc_id(level);
        assert!(max >= self.in_.min_doc_id(level));
        if level > 0 {
            assert!(max >= self.in_.max_doc_id(level - 1));
        }
        max
    }

    fn min_value(&self, level: usize) -> i64 {
        assert!(self.iterating(), "Unpositioned iterator");
        self.check_index(level);
        self.in_.min_value(level)
    }

    fn max_value(&self, level: usize) -> i64 {
        assert!(self.iterating(), "Unpositioned iterator");
        self.check_index(level);
        self.in_.max_value(level)
    }

    fn doc_count(&self, level: usize) -> i32 {
        assert!(self.iterating(), "Unpositioned iterator");
        self.check_index(level);
        self.in_.doc_count(level)
    }

    fn global_min_value(&self) -> i64 {
        self.in_.global_min_value()
    }

    fn global_max_value(&self) -> i64 {
        self.in_.global_max_value()
    }

    fn global_doc_count(&self) -> i32 {
        self.in_.global_doc_count()
    }

    fn max_value_count(&self) -> i32 {
        self.in_.max_value_count()
    }

    fn clone_box<'s>(&self) -> Box<dyn SkipperImpl + 's>
    where
        Self: 's,
    {
        Box::new(self.clone())
    }
}
