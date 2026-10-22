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
//! - Not wrapped (they pass through unchecked): impacts
//!   (`AssertingImpactsEnum`), the doc-values skipper
//!   (`AssertingDocValuesSkipper`), points (`AssertingPointValues`,
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
//! - [`SortedNumericDocValues::take_values`] and
//!   [`SortedSetDocValues::take_ords`], the bulk reads Java lacks, are
//!   checked as `nextValue`/`nextOrd` are and count as reading every value.

use std::sync::Arc;

use lucene_codecs::automaton::ByteDfa;
use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::doc_values::DocValuesSkipper;
use lucene_codecs::field_infos::{DocValuesType, FieldInfo, FieldInfos, IndexOptions};
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
        self.in_.doc_values_skipper(field)
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
        let sum_ttf = self.in_.sum_total_term_freq();
        let sum_df = self.sum_doc_freq();
        if !self.in_.has_freqs() {
            assert!(
                sum_ttf == sum_df,
                "sumTotalTermFreq={sum_ttf} != sumDocFreq={sum_df} without freqs"
            );
        }
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
        self.in_.impacts(flags)
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
/// the position budget `freq()` sets.
pub struct AssertingPostingsEnum {
    in_: Box<dyn PostingsEnum>,
    state: DocsState,
    position_count: i32,
    position_max: i32,
    doc: i32,
}

impl AssertingPostingsEnum {
    /// Wraps `in_`.
    pub fn new(in_: Box<dyn PostingsEnum>) -> Self {
        let doc = in_.doc_id();
        Self {
            in_,
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

impl DocIdSetIterator for AssertingPostingsEnum {
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

impl PostingsEnum for AssertingPostingsEnum {
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
    fn new(kind: &'static str, max_doc: i32, in_doc: i32) -> Self {
        assert!(in_doc == -1, "{kind} doc values start on doc {in_doc}");
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

    fn after_next(&mut self, doc: i32, in_doc: i32) {
        assert!(
            doc > self.last_doc_id,
            "{} doc values: backwards nextDoc from {} to {doc}",
            self.kind,
            self.last_doc_id
        );
        self.in_range(doc);
        assert!(doc == in_doc, "{} doc values: docID() disagrees", self.kind);
        self.last_doc_id = doc;
        self.exists = doc != NO_MORE_DOCS;
    }

    fn before_advance(&self, target: i32, in_doc: i32) {
        assert!(
            target >= 0 && target > in_doc,
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

    fn before_advance_exact(&self, target: i32, in_doc: i32) {
        assert!(
            target >= 0 && target >= in_doc && target < self.max_doc,
            "{} doc values: advanceExact({target}) from doc {in_doc}, maxDoc={}",
            self.kind,
            self.max_doc
        );
    }

    fn after_advance_exact(&mut self, target: i32, exists: bool, in_doc: i32) {
        assert!(
            in_doc == target,
            "{} doc values: advanceExact({target}) left the iterator on {in_doc}",
            self.kind
        );
        self.last_doc_id = target;
        self.exists = exists;
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
    ($ty:ident $(, |$s:ident| $reset:expr)?) => {
        impl DocIdSetIterator for $ty<'_> {
            fn doc_id(&self) -> i32 {
                self.in_.doc_id()
            }
            fn next_doc(&mut self) -> Result<i32> {
                let doc = self.in_.next_doc()?;
                self.check.after_next(doc, self.in_.doc_id());
                $(
                    let $s = &mut *self;
                    $reset;
                )?
                Ok(doc)
            }
            fn advance(&mut self, target: i32) -> Result<i32> {
                self.check.before_advance(target, self.in_.doc_id());
                let doc = self.in_.advance(target)?;
                assert!(
                    doc == self.in_.doc_id(),
                    "{} doc values: docID() disagrees",
                    self.check.kind
                );
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
                self.check.before_advance_exact(target, self.in_.doc_id());
                let exists = self.in_.advance_exact(target)?;
                self.check
                    .after_advance_exact(target, exists, self.in_.doc_id());
                $(
                    let $s = &mut *self;
                    $reset;
                )?
                Ok(exists)
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
        let check = DvCheck::new("Numeric", max_doc, in_.doc_id());
        Self { in_, check }
    }
}

asserting_dv_iterator!(AssertingNumericDocValues);

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
        let check = DvCheck::new("Binary", max_doc, in_.doc_id());
        Self { in_, check }
    }
}

asserting_dv_iterator!(AssertingBinaryDocValues);

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
    /// Wraps `in_`, an unpositioned iterator over a leaf of `max_doc`.
    pub fn new(in_: Box<dyn SortedDocValues + 'a>, max_doc: i32) -> Self {
        let check = DvCheck::new("Sorted", max_doc, in_.doc_id());
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

asserting_dv_iterator!(AssertingSortedDocValues);

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
    /// Wraps `in_`, an unpositioned iterator over a leaf of `max_doc`.
    pub fn new(in_: Box<dyn SortedNumericDocValues + 'a>, max_doc: i32) -> Self {
        let check = DvCheck::new("Sorted numeric", max_doc, in_.doc_id());
        Self {
            in_,
            check,
            value_upto: 0,
        }
    }
}

asserting_dv_iterator!(AssertingSortedNumericDocValues, |s| s.value_upto = 0);

impl SortedNumericDocValues for AssertingSortedNumericDocValues<'_> {
    fn doc_value_count(&self) -> i32 {
        self.check.exists("docValueCount");
        let count = self.in_.doc_value_count();
        assert!(count > 0, "docValueCount={count}");
        count
    }
    fn next_value(&mut self) -> Result<i64> {
        self.check.exists("nextValue");
        let count = self.in_.doc_value_count();
        assert!(
            self.value_upto < count,
            "valueUpto={} in.docValueCount()={count}",
            self.value_upto
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
    /// Wraps `in_`, an unpositioned iterator over a leaf of `max_doc`.
    pub fn new(in_: Box<dyn SortedSetDocValues + 'a>, max_doc: i32) -> Self {
        let check = DvCheck::new("Sorted set", max_doc, in_.doc_id());
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

asserting_dv_iterator!(AssertingSortedSetDocValues, |s| s.ords_retrieved = 0);

impl SortedSetDocValues for AssertingSortedSetDocValues<'_> {
    fn doc_value_count(&self) -> i32 {
        self.check.exists("docValueCount");
        let count = self.in_.doc_value_count();
        assert!(count > 0, "docValueCount={count}");
        count
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
