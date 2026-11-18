//! A unifying facade over this port's already-built write-side primitives --
//! analogous in spirit to real Lucene's `org.apache.lucene.index.IndexWriter`
//! as the single entry point for add/update/delete/commit, **not** a
//! reimplementation of any of the pieces it composes.
//!
//! # What this is
//!
//! Every write-side capability this facade exposes already existed as a
//! standalone primitive before this module:
//! - [`crate::segment_writer::flush_stored_only_segment`] -- flush a batch
//!   of buffered documents to a new segment.
//! - [`crate::update_document::update_document`] -- atomic delete-by-term +
//!   add-document, committed as one `segments_N`.
//! - [`crate::term_delete::resolve_and_apply_term_delete`] -- resolve a term
//!   to live doc IDs in one segment and apply the delete.
//! - [`crate::segment_infos::write`]/[`crate::segment_infos::read_latest`] --
//!   the `segments_N` commit file itself.
//!
//! What none of those modules provide on their own is a single stateful
//! object a caller can hold onto across several `add_document`/`commit`
//! calls without hand-threading a `SegmentInfos`, a segment-name counter, and
//! a buffered-document list through every call itself. [`IndexWriter`] is
//! exactly that: it owns the buffered-document list, the current committed
//! [`SegmentInfos`], and the next segment-name counter, and calls the
//! existing functions above in the right order with the right state at each
//! lifecycle point (`add_document` buffers; `commit` flushes the buffer via
//! `flush_stored_only_segment` and appends the result to `segment_infos`
//! before writing it; `update_document`/`delete_documents` delegate straight
//! to the existing atomic primitives).
//!
//! # Automatic merge triggering
//!
//! [`IndexWriter::set_merge_policy`] lets a caller opt this writer into
//! automatic merging: once a [`MergePolicyConfig`] is set, every
//! [`IndexWriter::commit`] call, right after writing its own `segments_N`,
//! synchronously asks [`crate::merge_policy::find_merges`] whether any of
//! this writer's *now-committed* segments should merge, and if so executes
//! each proposed group via [`crate::merge::merge_stored_only_segments`] and
//! folds the result in via [`IndexWriter::apply_merge`] -- reusing exactly
//! those two existing functions, not reimplementing either one. This repeats
//! (re-querying `find_merges` against the post-merge segment list) until it
//! proposes nothing further; each merge strictly reduces the total segment
//! count by at least one, so this loop is guaranteed to terminate. By
//! default (no [`MergePolicyConfig`] set), `commit()` behaves exactly as
//! before: no merge-policy consultation at all, matching every existing
//! caller of `commit()` from before this feature existed.
//!
//! This is deliberately synchronous, inside `commit()` itself -- this port
//! has no background-thread/`ConcurrentMergeScheduler`-equivalent
//! infrastructure, so "run it right there" is the only shape that fits
//! everything else here. [`IndexWriter::apply_merge`] remains public and
//! usable on its own for a caller that wants to drive a merge manually
//! instead (e.g. with different sources, or a policy this module doesn't
//! model).
//!
//! Still out of scope: no per-writer merge-policy *tuning* beyond whatever
//! [`MergePolicyConfig`] itself exposes, no background merging on this writer
//! (a [`crate::concurrent_writer::ConcurrentIndexWriter`] runs merges on a
//! merge thread instead), no merge-scheduling across many tiers beyond what
//! [`crate::merge_policy`] itself already does in one
//! [`crate::merge_policy::find_merges`] call, and
//! [`IndexWriter::update_document`]/[`IndexWriter::delete_documents_by_term`]
//! do not trigger this check (only [`IndexWriter::commit`] does, matching where
//! this port's flush/commit work already lived before this feature).
//!
//! # `IndexOptions::DocsAndCustomFreqs` postings
//!
//! [`IndexWriter::set_custom_freq_postings_field`] +
//! [`IndexWriter::add_document_with_custom_freq_terms`] are a second,
//! separate postings entry point alongside
//! [`IndexWriter::set_postings_field`]/[`IndexWriter::add_postings_field`]'s
//! analyzed-text one: instead of tokenizing a stored field's text to derive
//! term-occurrence-count freqs, a caller supplies each pending doc's exact
//! `(term, custom_freq)` pairs directly, and those values are written
//! verbatim as `IndexOptions::DocsAndCustomFreqs` postings (wire-identical to
//! `DocsAndFreqs` -- see `crate::postings_writer`'s module doc comment). This
//! exists because there is no way to derive a genuinely *arbitrary*
//! per-doc-per-term similarity score from analyzed text -- it has to come
//! from the caller. **Scope decision:** the two postings paths are mutually
//! exclusive per writer (never both active in the same commit) -- see
//! [`IndexWriter::set_custom_freq_postings_field`]'s doc comment for exactly
//! why and what error a caller sees if they try to combine them.
//!
//! # What this deliberately is not
//!
//! - **No RAM-based flush triggering.** Real `IndexWriter` auto-flushes once
//!   buffered documents exceed `ramBufferSizeMB`; this facade only flushes
//!   on an explicit [`IndexWriter::commit`] call, matching
//!   `segment_writer.rs`'s own documented stance that this port has "no RAM
//!   accounting or automatic flush-triggering" yet.
//! - **No multi-threaded `DocumentsWriterPerThread` pooling, no
//!   `IndexWriterConfig`-style tunable object** -- one caller, one
//!   `Directory`, sequential calls, exactly like every primitive this
//!   facade composes.
//! - **`update_document`/`delete_documents` only resolve against segments
//!   the caller explicitly supplies an opened [`SegmentDeleteSource`] for**
//!   (same limitation `update_document.rs` already documents) -- there is no
//!   reader pool that automatically opens every existing segment's postings
//!   for the caller. In particular, a document sitting only in this
//!   writer's own *unflushed* buffer can never be matched by a delete/update
//!   term (it isn't a segment yet), matching real Lucene's own
//!   `BufferedUpdates` timing (a delete only ever resolves against segments
//!   that exist *at delete time*).
//! - **[`IndexWriter::rollback`] and [`IndexWriter::prepare_commit`]/
//!   [`IndexWriter::finish_commit`] are implemented on real Lucene's
//!   `pending_segments_N` protocol** -- see those methods' own doc comments.
//!   What remains out of scope is *cross-process* handoff: a crashed process's
//!   `pending_segments_N` is inert and correct to ignore (the previous commit
//!   stays current), but nothing here discovers it and rolls it forward, and
//!   nothing deletes it either, because this port has no `IndexFileDeleter`.
//!
//! # Segment/commit-file lifecycle
//!
//! [`IndexWriter::open`] looks for an existing `segments_N` in `dir` (via
//! [`lucene_store::directory::last_commit_generation`], not
//! [`crate::segment_infos::read_latest`] directly, so "no commit yet" is
//! distinguished from "a commit file exists but is corrupt" -- the latter
//! still surfaces as an `Err`, matching this port's stance elsewhere of
//! never treating corruption as an empty index). If none is found, it starts
//! from a fresh, empty [`SegmentInfos`] (generation/version/counter all `0`,
//! no segments) -- the first [`IndexWriter::commit`] then writes `segments_1`
//! (`SegmentInfos::write` picks a `generation` field the caller controls;
//! this facade always writes the *next* generation, matching real Lucene's
//! monotonic commit-generation counter). Segment names follow the real
//! `_0`, `_1`, ... convention (`IndexFileNames.segmentFileName`'s counter),
//! driven off `segment_infos.counter` so a writer resumed on an
//! already-committed directory doesn't collide with segment names an
//! earlier writer session already used.

use crate::buffered_updates::{
    BufferedUpdatesStream, DeleteQuery, DeleteQueue, DocValuesUpdate, FrozenBufferedUpdates, SeqNo,
    Term, UpdateValue,
};
use crate::compound_reader::CompoundReader;
use crate::deletes;
use crate::index_file_deleter::{self, DeletionPolicy, IndexFileDeleter};
use crate::indexing_chain::{
    invert_documents_with_payloads, InMemoryInvertedIndex, PayloadContext,
};
use crate::merge;
use crate::merge_policy;
use crate::segment_info::{self, LuceneVersion};
use crate::segment_infos::{self, SegmentCommitInfo, SegmentInfos};
use crate::segment_writer::{self};
use crate::similarity::{FieldInvertState, NormSimilarity};
use crate::term_delete;
use crate::update_document::{self, SegmentDeleteSource};

use lucene_analysis::Analyzer;
use lucene_codecs::backward_codecs::hnsw_vectors::{RetiredHnswFormat, RetiredHnswVectorsReader};
use lucene_codecs::backward_codecs::quantized_vectors::{
    QuantizedFiles, QuantizedFormat, QuantizedVectorsReader,
};
use lucene_codecs::doc_values;
use lucene_codecs::field_infos::{
    DocValuesType, FieldInfo, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::hnsw;
use lucene_codecs::hnsw_vectors::{self, HnswVectorsField};
use lucene_codecs::norms;
use lucene_codecs::per_field_doc_values;
use lucene_codecs::per_field_knn_vectors;
use lucene_codecs::per_field_postings;
use lucene_codecs::postings_writer::{self, FieldPostingsInput, TermPostings};
use lucene_codecs::stored_fields::{self, Document, FieldValue};
use lucene_codecs::term_vectors::{self, TermVectorField, TermVectorTerm, TermVectorsDocument};
use lucene_codecs::vectors::{self, FieldVectorData, FlatVectorsField};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::data_output::DataOutput;
use lucene_store::directory::{Directory, Input};
use lucene_util::fixed_bit_set::FixedBitSet;

mod add_indexes;
#[cfg(test)]
mod block_tests;
mod dv_columns;
mod explicit;
mod inverted_terms;
mod lifecycle;
mod merge_reading;
pub(crate) mod packed_fields;
pub(crate) mod pending_postings;
mod pluggable_merge;
mod segment_attributes;
mod segment_partitions;
pub(crate) use segment_partitions::set_attribute;
pub use segment_partitions::SegmentPartitioning;
mod try_modify;
mod upgrade;
pub(crate) use explicit::explicit_error;
pub use explicit::{ExplicitDocument, ExplicitFields, InvertedField, InvertedTerm, LeanParts};
pub use inverted_terms::{InvertedTerms, InvertedTermsIter, Payloads, TermBuilder, TermRef};
pub use lifecycle::MergedSegmentWarmer;

pub use crate::merge_policy::MergePolicyConfig;
pub use crate::update_document::SegmentDeleteSource as DeleteSource;
mod soft_deletes;
use soft_deletes::count_soft_deletes;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    /// An [`ExplicitDocument`] or field registration the explicit path cannot
    /// accept -- see `index_writer/explicit.rs`.
    #[error("explicit document: {0}")]
    Explicit(String),
    /// A search over the writer's buffers that the buffers cannot answer --
    /// see [`crate::concurrent_writer::ConcurrentIndexWriter::live_snapshot`].
    #[error("live search: {0}")]
    LiveSearch(String),
    /// A [`crate::document::Document`] the document API refuses -- Java's
    /// `IllegalArgumentException` from `IndexingChain` or a field.
    #[error("document: {0}")]
    Document(#[from] crate::document::Error),
    /// `IndexingChain.PerField.finish`'s `IllegalStateException`: the
    /// configured similarity's `computeNorm` returned `0` for a field that
    /// has tokens, a value only an empty field may carry.
    #[error("the similarity returned a norm of 0 for non-empty field {0:?}")]
    ZeroNorm(String),
    /// A `OneMerge`'s `wrapForMerge`/`reorder`
    /// ([`crate::merge_policy::MergeHooks`]) failed, or answered something
    /// the merge cannot honour.
    #[error("merge hooks: {0}")]
    MergeHook(String),
    #[error(transparent)]
    SegmentWriter(#[from] segment_writer::Error),
    #[error(transparent)]
    SegmentInfos(#[from] segment_infos::Error),
    #[error(transparent)]
    UpdateDocument(#[from] update_document::Error),
    #[error(transparent)]
    TermDelete(#[from] term_delete::Error),
    #[error(transparent)]
    Deletes(#[from] deletes::Error),
    #[error(transparent)]
    FieldUpdates(#[from] crate::field_updates::Error),
    #[error(transparent)]
    Deleter(#[from] index_file_deleter::Error),
    #[error(transparent)]
    Merge(#[from] merge::Error),
    /// `MergePolicy.MergeAbortedException`: the merge's
    /// [`crate::merge_rate_limiter::OneMergeProgress`] was aborted while it
    /// ran; its output is discarded.
    #[error("merge aborted")]
    MergeAborted,
    /// `MergeRateLimiter.setMBPerSec`'s `IllegalArgumentException("mbPerSec
    /// must be positive")`.
    #[error("mbPerSec must be positive; got: {0}")]
    InvalidMergeRate(f64),
    #[error(transparent)]
    SegmentInfo(#[from] segment_info::Error),
    #[error(transparent)]
    StoredFields(#[from] lucene_codecs::stored_fields::Error),
    #[error(transparent)]
    LiveDocs(#[from] lucene_codecs::live_docs::Error),
    #[error(transparent)]
    PostingsWriter(#[from] postings_writer::Error),
    /// `IndexWriter::open`'s field list did not survive Java's `FieldInfo`
    /// constructor / `FieldInfos(FieldInfo[])` constructor. Java makes these
    /// combinations unrepresentable by throwing from the constructor; a Rust
    /// public-field struct cannot, so the writer -- the port's own
    /// caller-facing door for a hand-built field list -- checks them at
    /// `open`, rather than letting the mistake surface as an unreadable
    /// `.fnm` several thousand documents later.
    #[error(transparent)]
    FieldInfos(#[from] lucene_codecs::field_infos::Error),
    #[error(transparent)]
    SimpleText(#[from] lucene_codecs::simpletext::Error),
    /// A source segment's `.tim`/`.tip`/`.tmd` term dictionary couldn't be
    /// opened while [`IndexWriter::execute_merge`] was assembling that
    /// segment's [`crate::merge::SourcePostings`].
    #[error(transparent)]
    Blocktree(#[from] lucene_codecs::blocktree::Error),
    /// A source segment's `.doc` file couldn't be opened while
    /// [`IndexWriter::execute_merge`] was assembling that segment's
    /// [`crate::merge::SourcePostings`].
    #[error(transparent)]
    Postings(#[from] lucene_codecs::postings::Error),
    #[error(transparent)]
    TermVectors(#[from] term_vectors::Error),
    #[error(transparent)]
    DocValues(#[from] doc_values::WriteError),
    /// A source segment's `.dvm`/`.dvd` could not be read while
    /// [`IndexWriter::execute_merge`] was assembling that segment's
    /// doc-values merge inputs, or while reading an index-sort tier's column
    /// back out of it.
    #[error(transparent)]
    DocValuesRead(#[from] doc_values::Error),
    /// A source segment's `.nvm`/`.nvd` could not be read while
    /// [`IndexWriter::execute_merge`] was assembling that segment's norms
    /// merge inputs.
    #[error(transparent)]
    NormsRead(#[from] norms::Error),
    /// `IndexWriter.validateIndexSort`, at merge time: every segment in one
    /// merge must declare the same index sort (or none). Merging segments
    /// that disagree could only produce a segment in one order described as
    /// being in another, so it is refused.
    #[error(
        "cannot merge segments with different index sorts: expected {expected}, but segment \
         {segment} has {found}"
    )]
    MergeSortDisagreement {
        expected: String,
        found: String,
        segment: String,
    },
    /// A source segment declares an index sort in its `.si` but carries no
    /// NUMERIC doc-values column for one of its tiers, so there is nothing to
    /// re-derive its order from. `set_index_sort` makes the column mandatory,
    /// so this means the segment does not satisfy the sort it claims.
    #[error("segment declares an index sort on field {0:?} but has no doc-values column for it")]
    MergeSortColumnMissing(String),
    #[error(transparent)]
    Norms(#[from] norms::WriteError),
    /// Every vector-format error -- `vectors`, `hnsw` and `hnsw_vectors` all
    /// share one `Error` type, the way they share one on-disk format family.
    #[error(transparent)]
    Vectors(#[from] vectors::Error),
    /// The IVF index of a `IvfVectorsFormat` instance.
    #[error(transparent)]
    Ivf(#[from] lucene_ivf::Error),
    /// The disk files of a `DiskHnswVectorsFormat` instance.
    #[error(transparent)]
    DiskHnsw(#[from] lucene_disk_hnsw::Error),
    #[error("set_vector_field: no field named {0:?} in this writer's field list")]
    UnknownVectorField(String),
    #[error(transparent)]
    Points(#[from] lucene_codecs::points::Error),
    #[error("add_points_field: no field named {0:?} in this writer's field list")]
    UnknownPointsField(String),
    /// The field's `FieldInfo` must declare its point shape the way
    /// `FieldType.setDimensions` does: `1..=16` dimensions (`PointValues.MAX_DIMENSIONS`),
    /// `1..=num_dims` (and at most 8, `MAX_INDEX_DIMENSIONS`) of them indexed,
    /// `1..=16` bytes each (`PointValues.MAX_NUM_BYTES`).
    #[error(
        "add_points_field: field {0:?} declares {1} dimension(s), {2} indexed, {3} byte(s) \
         each; a points field needs 1..=16 dimensions, 1..=min(dims, 8) indexed and 1..=16 \
         bytes per dimension"
    )]
    UnsupportedPointsField(String, i32, i32, i32),
    #[error("add_points_field: field {0:?} is already in this writer's points-field list")]
    DuplicatePointsField(String),
    /// A document's value for a points field has the wrong shape: a number
    /// whose width is not the field's `bytesPerDim` (or a field with more
    /// than one dimension, which only packed `Binary` can fill), or packed
    /// bytes that are not `numDims * bytesPerDim` long -- what
    /// `Field(name, packedPoint, type)` rejects in Java.
    #[error(
        "document {1}, points field {0:?}: a {2} value does not fit a point of {3} dimension(s) \
         x {4} byte(s)"
    )]
    PointValueShape(String, usize, &'static str, i32, i32),
    #[error(
        "set_vector_field: field {0:?} declares vector_dimension {1}; a vector field's FieldInfo \
         must carry a positive dimension (real Lucene's KnnFloatVectorField/KnnByteVectorField \
         set it from the field's own vector length)"
    )]
    UnsupportedVectorField(String, i32),
    #[error(
        "add_vector_field: field {0:?} is already in this writer's vector-field list -- a field \
         can only carry one vector value per document"
    )]
    DuplicateVectorField(String),
    #[error(
        "document {1}, vector field {0:?}: expected {2} components (the field's declared \
         vector_dimension), got {3}"
    )]
    VectorDimensionMismatch(String, usize, i32, usize),
    #[error(
        "document {1}, vector field {0:?}: the field is declared {2:?}-encoded but the document \
         supplied a {3:?} vector"
    )]
    VectorEncodingMismatch(String, usize, VectorEncoding, VectorEncoding),
    #[error("set_postings_field: no field named {0:?} in this writer's field list")]
    UnknownPostingsField(String),
    #[error(
        "set_postings_field: field {0:?} has index_options {1:?}; only Docs/DocsAndFreqs/\
         DocsAndFreqsAndPositions/DocsAndFreqsAndPositionsAndOffsets are supported by this \
         writer's postings write-side"
    )]
    UnsupportedPostingsIndexOptions(String, IndexOptions),
    #[error(
        "set_payload_source: no field opted into postings has store_payloads set, so every \
         payload this source produced would be silently discarded -- set store_payloads on the \
         FieldInfo of a positions-indexing field first (payload presence is a per-field \
         property, FieldInfo.hasPayloads(), never a per-token one)"
    )]
    NoPayloadFields,
    #[error(
        "add_postings_field: field {0:?} is already opted into postings for this writer \
         (via set_postings_field or an earlier add_postings_field call) -- each field number \
         may only be added once per commit"
    )]
    DuplicatePostingsField(String),
    /// A segment about to be merged claims one document count in its `.si`
    /// and a different one in its stored-fields metadata.
    ///
    /// Java's `SegmentMerger` takes `maxDoc` from `SegmentReader.maxDoc()`,
    /// i.e. from `SegmentInfo` -- the `.si`. This port's merge reads it off
    /// the `.fdm` instead, where `stored_fields::open` only checks that it is
    /// non-negative, so a four-byte edit claiming `maxDoc = i32::MAX` sizes
    /// the merge's per-source live-id list and doc-id map at ~8.6 GB each --
    /// an allocation abort, the one failure `catch_unwind` at the FFI
    /// boundary cannot intercept.
    #[error(
        "segment {segment:?}: the .si records {si_doc_count} documents but its stored-fields \
         metadata records {stored_fields_max_doc}"
    )]
    SegmentDocCountMismatch {
        segment: String,
        si_doc_count: i32,
        stored_fields_max_doc: i32,
    },
    #[error("set_custom_freq_postings_field: no field named {0:?} in this writer's field list")]
    UnknownCustomFreqPostingsField(String),
    #[error(
        "set_custom_freq_postings_field: field {0:?} has index_options {1:?}; only \
         IndexOptions::DocsAndCustomFreqs is supported by this entry point -- use \
         set_postings_field/add_postings_field for an analyzed-text postings field instead"
    )]
    UnsupportedCustomFreqPostingsIndexOptions(String, IndexOptions),
    #[error(
        "{0}: a writer cannot have both an analyzed-text postings field (set_postings_field/\
         add_postings_field) and a custom-freq postings field (set_custom_freq_postings_field) \
         active at the same time -- see IndexWriter's module doc comment for why this scope is \
         deliberate; clear the other one first (set_postings_field(None) / \
         set_custom_freq_postings_field(None))"
    )]
    PostingsAndCustomFreqPostingsMutuallyExclusive(&'static str),
    #[error("set_term_vector_field: no field named {0:?} in this writer's field list")]
    UnknownTermVectorField(String),
    #[error(
        "set_term_vector_field: field {0:?} does not have store_term_vectors set; \
         this writer's term-vector write-side only builds term vectors for a field \
         whose FieldInfo already advertises them"
    )]
    UnsupportedTermVectorField(String),
    #[error(
        "add_term_vector_field: field {0:?} is already opted into term vectors for this writer \
         (via set_term_vector_field or an earlier add_term_vector_field call) -- each field \
         number may only be added once per commit"
    )]
    DuplicateTermVectorField(String),
    #[error("set_doc_values_field: no field named {0:?} in this writer's field list")]
    UnknownDocValuesField(String),
    #[error(
        "set_doc_values_field: field {0:?} has doc_values_type {1:?}; only \
         DocValuesType::Numeric/DocValuesType::Binary/DocValuesType::Sorted/\
         DocValuesType::SortedNumeric/DocValuesType::SortedSet are supported by this writer's \
         doc-values write-side"
    )]
    UnsupportedDocValuesType(String, DocValuesType),
    #[error(
        "commit: doc-values field {0:?} is dense-only (write_single_dense_numeric_field/\
         write_single_dense_sorted_field have no missing-value encoding) but pending doc {1} \
         has no value for it"
    )]
    MissingDenseDocValue(String, usize),
    #[error(
        "commit: doc-values field {0:?} requires FieldValue::Int or FieldValue::Long on every \
         pending doc, but doc {1} has a {2} value"
    )]
    NonNumericDocValue(String, usize, &'static str),
    #[error(
        "commit: doc-values field {0:?} requires FieldValue::String or FieldValue::Binary on \
         every pending doc, but doc {1} has a {2} value"
    )]
    NonBinaryDocValue(String, usize, &'static str),
    #[error("omit_norms_field: no field named {0:?} in this writer's field list")]
    UnknownNormsField(String),
    #[error(
        "omit_norms_field: field {0:?} is not indexed (index_options == None); norms only \
         exist for an indexed field, matching real Lucene's IndexOptions.NONE gate on \
         Similarity.computeNorm"
    )]
    UnsupportedNormsField(String),
    #[error(
        "finish_commit: no prepared commit pending -- call prepare_commit() first \
         (or use commit(), which calls both)"
    )]
    NoPreparedCommit,
    #[error(
        "prepare_commit was already called with no corresponding call to commit \
         (call finish_commit() to activate it, or rollback() to discard it)"
    )]
    PrepareCommitAlreadyCalled,
    #[error(
        "{0}: a commit is already prepared (prepare_commit() was called with no corresponding \
         finish_commit()); this operation writes its own segments_N at the same generation the \
         prepared commit will claim, which would silently revert it -- call finish_commit() or \
         rollback() first"
    )]
    PreparedCommitPending(&'static str),
    /// An operation [`crate::concurrent_writer::ConcurrentIndexWriter`] does
    /// not offer; do it on the single-threaded writer
    /// ([`crate::concurrent_writer::ConcurrentIndexWriter::into_writer`]).
    #[error("{0} is not supported by the concurrent writer")]
    ConcurrentUnsupported(&'static str),
    /// `LiveIndexWriterConfig.setRAMBufferSizeMB`'s
    /// `IllegalArgumentException("ramBufferSize should be > 0.0 MB when enabled")`.
    #[error(
        "set_ram_buffer_size_mb: ramBufferSize should be > 0.0 MB when enabled (got {0}); pass \
         DISABLE_AUTO_FLUSH_MB to turn RAM-based flushing off"
    )]
    InvalidRamBufferSize(f64),
    /// `LiveIndexWriterConfig.setMaxBufferedDocs`'s
    /// `IllegalArgumentException("maxBufferedDocs must at least be 2 when enabled")`.
    #[error(
        "set_max_buffered_docs: maxBufferedDocs must at least be 2 when enabled (got {0}); pass \
         DISABLE_AUTO_FLUSH to turn document-count-based flushing off"
    )]
    InvalidMaxBufferedDocs(i32),
    /// `LiveIndexWriterConfig`'s `IllegalArgumentException("at least one of ramBufferSize and
    /// maxBufferedDocs must be enabled")` -- disabling both would restore exactly the unbounded
    /// peak memory this writer's flush trigger exists to bound, so Java refuses it and so does
    /// this.
    #[error(
        "at least one of ramBufferSize and maxBufferedDocs must be enabled -- disabling both \
         makes peak memory O(every document added since the last commit)"
    )]
    BothAutoFlushTriggersDisabled,
    /// `IndexWriter.softUpdateDocument`'s
    /// `IllegalArgumentException("at least one soft delete must be present")`.
    #[error("soft_update_document: at least one soft delete must be present")]
    NoSoftDeletesSupplied,
    /// `IndexWriter.tooManyDocs`' `IllegalArgumentException`: the documents
    /// already in the index -- deleted ones included, until a merge drops
    /// them -- plus the ones being added would exceed the writer's limit
    /// ([`MAX_DOCS`] unless [`IndexWriter::set_max_docs`] lowered it).
    #[error("number of documents in the index cannot exceed {0}")]
    TooManyDocs(usize),
    /// `IndexWriter.forceMerge`'s `IllegalArgumentException("maxNumSegments
    /// must be >= 1; got 0")`.
    #[error("force_merge: max_num_segments must be >= 1; got {0}")]
    InvalidMaxNumSegments(usize),
    /// `IndexWriter.updateDocValues` with an empty `updates` array buffers a
    /// node that can never match anything; Java's `DocValuesUpdate[0]` is a
    /// silent no-op that still burns a sequence number. Rejected here, because
    /// a caller reaching this has a bug and nothing downstream can tell them.
    #[error("update_doc_values: at least one doc-values update must be supplied")]
    NoDocValuesUpdatesSupplied,
    /// Java's `verifyOrCreateDvOnlyField` *creates* the field if it is absent;
    /// this port's field list is fixed at [`IndexWriter::open`], so an unknown
    /// field is an error instead of an implicit schema change.
    #[error("update_doc_values: no field named {0:?} in this writer's field list")]
    UnknownDocValuesUpdateField(String),
    /// Java's `verifyOrCreateDvOnlyField` throws when the existing field's
    /// `DocValuesType` does not match the update's.
    #[error(
        "update_doc_values: field {field:?} is declared with doc_values_type {declared}; only a \
         NUMERIC field accepts a numeric update and only a BINARY field a binary one"
    )]
    WrongDocValuesUpdateType { field: String, declared: String },
    /// A buffered delete or doc-values update named a field this writer's
    /// segments have no term dictionary for, so it can never be resolved. Java
    /// silently resolves to "no documents" (`TermDocsIterator.nextTerm`
    /// returns null); this port does the same and this variant is reserved for
    /// the case where the segment's postings files are *listed* but unreadable.
    #[error("apply_deletes: segment {segment:?} lists postings files that could not be opened")]
    UnreadableSegmentPostings { segment: String },
    /// `IndexWriterConfig.setIndexSort(new Sort())` -- a `Sort` with no
    /// `SortField`s. Java's `Sort()` no-arg constructor sorts by relevance,
    /// which is not an index sort at all; `SegmentInfo`'s `numSortFields`
    /// likewise uses `0` to mean *unsorted*, so an empty list is
    /// indistinguishable from `None` on disk.
    #[error(
        "set_index_sort: an index sort must have at least one sort field (use None to disable)"
    )]
    EmptyIndexSort,
    /// This port's field list is fixed at [`IndexWriter::open`], so a sort
    /// field that is not in it can never receive a value.
    #[error("set_index_sort: no field named {0:?} in this writer's field list")]
    UnknownIndexSortField(String),
    /// `IndexingChain.validateIndexSortDVType`: Java refuses a sort field
    /// whose `DocValuesType` is not the one the `SortField.Type` reads
    /// (`"SortField <..> expected field [x] to be NUMERIC but it is [BINARY]"`):
    /// NUMERIC for a numeric `SortField`, SORTED_NUMERIC for a
    /// `SortedNumericSortField`, SORTED for a `STRING` one, SORTED_SET for a
    /// `SortedSetSortField` and BINARY for a `BinarySortField`.
    #[error(
        "set_index_sort: sort field {0:?} is declared with doc_values_type {1:?}, which is not \
         the doc-values type its kind of sort reads"
    )]
    UnsupportedIndexSortField(String, DocValuesType),
    /// A sort field with no doc values written for it makes every
    /// sort-order check downstream vacuous: real Lucene's
    /// `DocValues.getNumeric` returns an all-missing instance rather than
    /// failing, so `CheckIndex.testSort` compares `maxDoc` equal keys and
    /// passes, and this port's own `check_index` reports "no NUMERIC
    /// doc-values entry". The sort key has to be a column a reader can
    /// actually read.
    #[error(
        "set_index_sort: sort field {0:?} is not opted into doc values -- call \
         set_doc_values_field/add_doc_values_field for it first, or nothing will write the \
         column the sort is defined over"
    )]
    IndexSortFieldWithoutDocValues(String),
    /// `IndexWriter.validateIndexSort`: every segment already in the index
    /// must carry a sort the incoming one is a prefix of
    /// (`isCongruentSort`), or the index would hold segments ordered by two
    /// different keys and every sort-preserving merge over them would be
    /// wrong.
    #[error(
        "set_index_sort: cannot change previous indexSort={existing:?} (from segment={segment}) \
         to new indexSort={incoming:?}"
    )]
    IncongruentIndexSort {
        segment: String,
        existing: String,
        incoming: String,
    },
    /// The sort is read at flush, so changing it with documents already
    /// buffered would order part of the batch by one key and part by
    /// another. Java cannot reach this state at all -- `IndexWriterConfig` is
    /// snapshotted when the `IndexWriter` is constructed -- so this guard is
    /// the faithful lowering of "the sort is fixed for the writer's life",
    /// not an extra rule.
    #[error(
        "set_index_sort: {0} document(s) are already buffered; flush or commit before changing \
         the index sort (real Lucene fixes it on IndexWriterConfig before the writer exists)"
    )]
    IndexSortChangedMidBuffer(usize),
    /// `IndexingChain.maybeSortSegment`'s
    /// `CorruptIndexException("parent field is not set but the index has
    /// blocks and uses index sorting")`. A block of documents added by
    /// `add_documents`/`update_documents` must stay physically contiguous
    /// and in order; an index sort would scatter it. Java allows the
    /// combination only when a *parent* field marks each block's last
    /// document, so the sort can be applied to whole blocks
    /// ([`IndexWriter::set_parent_field`]). The add paths refuse such a block
    /// up front ([`Error::BlocksWithIndexSortNeedParentField`]); this is the
    /// flush's backstop, so no segment with shredded blocks is ever written.
    #[error(
        "flush: this segment carries document blocks (add_documents/update_documents) and an \
         index sort is configured, but no parent field is set -- real Lucene raises \
         CorruptIndexException for exactly this combination"
    )]
    IndexSortWithBlocksAndNoParentField,
    /// `DocumentsWriterPerThread.updateDocuments`'
    /// `IllegalArgumentException`: a block of more than one document added
    /// to a writer with an index sort and no parent field
    /// ([`IndexWriter::set_parent_field`]), refused before anything is
    /// buffered.
    #[error(
        "a parent field must be set in order to use document blocks with index sorting; see \
         IndexWriterConfig#setParentField"
    )]
    BlocksWithIndexSortNeedParentField,
    /// `IndexWriter.updateDocValues`'s `IllegalArgumentException("cannot
    /// update docvalues field involved in the index sort")`. Rewriting the
    /// column the segment's physical order is defined over would leave the
    /// segment claiming a sort it no longer has.
    #[error(
        "update_doc_values: cannot update doc-values field {field:?} -- it is part of this \
         writer's index sort ({sort})"
    )]
    DocValuesUpdateOnIndexSortField { field: String, sort: String },
    /// `IndexWriter.addIndexes`' `IllegalArgumentException`: a field of an
    /// incoming segment whose schema conflicts with this writer's.
    #[error("add_indexes: {0}")]
    AddIndexes(String),
    /// `ConcurrentMergeScheduler.setMaxMergesAndThreads`'
    /// `IllegalArgumentException`.
    #[error("merge scheduler: {0}")]
    InvalidMergeScheduler(String),
    /// A merge run on a merge thread panicked; the scheduler caught it and
    /// reports it as this merge's failure (Java: an uncaught `Throwable`
    /// wrapped in `MergePolicy.MergeException`).
    #[error("a merge thread panicked: {0}")]
    MergeThreadPanicked(String),
    /// `add_doc_values_field` twice for the same field, the doc-values
    /// analogue of [`Error::DuplicatePostingsField`]/
    /// [`Error::DuplicateVectorField`].
    #[error("add_doc_values_field: field {0:?} is already opted into doc values for this writer")]
    DuplicateDocValuesField(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One entry of Java's `DocumentsWriterDeleteQueue` node hierarchy
/// (`TermArrayNode`/`QueryArrayNode`/`DocValuesUpdatesNode`), as the enum Rust
/// spells a sealed hierarchy as. Purely internal: the public API takes the
/// terms/queries/updates directly, exactly as Java's does.
#[derive(Debug, Clone)]
pub(crate) enum DeleteNode {
    Terms(Vec<std::sync::Arc<Term>>),
    Queries(Vec<DeleteQuery>),
    DocValuesUpdates(Vec<DocValuesUpdate>),
}

impl DeleteNode {
    /// `TermArrayNode`, each term shared by every buffer it reaches.
    pub(crate) fn terms(terms: Vec<Term>) -> Self {
        DeleteNode::Terms(terms.into_iter().map(std::sync::Arc::new).collect())
    }

    /// The RAM this delete adds to a buffer it is recorded in -- each entry
    /// with its `docIDUpto` -- what the flush policy weighs as
    /// `getDeleteBytesUsed()`. A sum of live allocation sizes, as
    /// [`document_ram_bytes`] is, where Java's `BufferedUpdates` counts its
    /// byte-block pools and per-entry constants.
    pub(crate) fn ram_bytes(&self) -> usize {
        let limit = std::mem::size_of::<i32>();
        let entries = match self {
            DeleteNode::Terms(terms) => terms
                .iter()
                .map(|t| {
                    // The shared allocation: the term and its two counts.
                    let shared = t
                        .ram_bytes()
                        .saturating_add(std::mem::size_of::<[usize; 2]>());
                    shared
                        .saturating_add(std::mem::size_of::<std::sync::Arc<Term>>())
                        .saturating_add(limit)
                })
                .fold(0usize, usize::saturating_add),
            DeleteNode::Queries(queries) => queries
                .iter()
                .map(|q| q.ram_bytes().saturating_add(limit))
                .fold(0usize, usize::saturating_add),
            DeleteNode::DocValuesUpdates(updates) => updates
                .iter()
                .map(DocValuesUpdate::ram_bytes)
                .fold(0usize, usize::saturating_add),
        };
        std::mem::size_of::<Self>().saturating_add(entries)
    }
}

/// Records `node` in `buffer`, reaching the documents below `doc_id_upto` --
/// `DocumentsWriterDeleteQueue`'s `Node.apply(BufferedUpdates, docIDUpto)`.
pub(crate) fn buffer_node(
    buffer: &mut crate::buffered_updates::BufferedUpdates,
    node: &DeleteNode,
    doc_id_upto: i32,
) {
    match node {
        DeleteNode::Terms(terms) => {
            for term in terms {
                buffer.add_shared_term(term, doc_id_upto);
            }
        }
        DeleteNode::Queries(queries) => {
            for query in queries {
                buffer.add_query(query.clone(), doc_id_upto);
            }
        }
        DeleteNode::DocValuesUpdates(updates) => {
            for update in updates {
                buffer.add_doc_values_update(update, doc_id_upto);
            }
        }
    }
}

/// `IndexWriter.buildDocValuesUpdate(term, updates)`: every supplied field
/// becomes an update keyed by the caller's `term`, whatever term the caller
/// happened to put in the [`DocValuesUpdate`] they built.
pub(crate) fn retarget_update(update: &DocValuesUpdate, term: &Term) -> DocValuesUpdate {
    match update {
        DocValuesUpdate::Numeric { field, value, .. } => DocValuesUpdate::Numeric {
            term: term.clone(),
            field: field.clone(),
            value: *value,
        },
        DocValuesUpdate::Binary { field, value, .. } => DocValuesUpdate::Binary {
            term: term.clone(),
            field: field.clone(),
            value: value.clone(),
        },
    }
}

/// A flushing buffer's own term deletes, for
/// [`IndexingConfig::build_and_write_segment`] to resolve against the terms it
/// has just inverted, before they are written -- `FreqProxTermsWriter.applyDeletes`,
/// which Java runs during the flush over the in-RAM postings for the same
/// reason: most of a buffer's delete terms (an update issued on another
/// thread, say) match nothing in it, and a miss in memory is a binary search
/// where a miss against the written segment is a term-dictionary seek.
pub(crate) struct FlushDeletes<'a> {
    /// Term -> the buffer position the delete was issued at.
    pub(crate) terms: &'a std::collections::HashMap<std::sync::Arc<Term>, i32>,
    /// Whether a document of the built segment is below a delete's limit
    /// (through the flush's sort map, when it sorted).
    pub(crate) below_limit: &'a dyn Fn(i32, i32) -> bool,
    /// The documents the terms delete; meaningful only once `resolved`.
    pub(crate) deleted: Vec<i32>,
    /// Set when the build resolved `terms` -- only the fast inversion path
    /// does; after any other the caller resolves them against the segment.
    pub(crate) resolved: bool,
}

impl FlushDeletes<'_> {
    /// Resolves the delete terms naming `field` against its postings, sorted
    /// by term bytes as [`crate::inverter::FieldInverter::into_term_postings`]
    /// returns them.
    fn resolve_field(&mut self, field: &str, terms: &[TermPostings]) {
        for (term, &limit) in self.terms.iter().filter(|(t, _)| t.field == field) {
            if let Ok(i) = terms.binary_search_by(|p| p.term.as_slice().cmp(&term.bytes)) {
                let docs = terms[i].docs.iter().map(|&(doc, _)| doc);
                self.deleted
                    .extend(docs.filter(|&doc| (self.below_limit)(doc, limit)));
            }
        }
    }
}

/// Everything about **how** a segment is built that a writer fixes before its
/// first document: the schema, the per-field opt-ins, the analyzer gaps, the
/// index sort, the graph parameters. Split out of [`IndexWriter`] so the
/// build of a segment (`IndexingConfig::build_and_write_segment`) needs only
/// this, a document buffer and a directory -- which is what lets a
/// [`crate::concurrent_writer::ConcurrentIndexWriter`] build segments on
/// several threads at once, each from its own buffer, without the writer's
/// lock (`DocumentsWriterPerThread.flush`, which likewise reads only its
/// `FieldInfos.Builder` and `LiveIndexWriterConfig`).
/// Positions in [`IndexingConfig`]'s field list by name and by number. The
/// first field with a given name or number wins, as the `iter().find` scans
/// these replace did: they ran per document and per value, so a wide schema
/// paid O(fields) for every one.
///
/// Field numbers are small and dense (`FieldInfos.Builder` hands them out
/// from 0), so the number lookup is an array indexed by number: it runs for
/// every field of every document added, where a hash per lookup was a
/// measurable share of the add. A number past [`DENSE_FIELD_NUMBERS`] (only
/// a caller registering explicit numbers could make one) goes to a map.
#[derive(Clone, Default)]
pub(crate) struct FieldLookup {
    by_name: std::collections::HashMap<String, usize>,
    /// Position by number, [`NO_FIELD`] where none.
    by_number: Vec<u32>,
    sparse_by_number: std::collections::HashMap<i32, usize>,
}

/// Field numbers below this are looked up by index.
const DENSE_FIELD_NUMBERS: usize = 1 << 16;
/// [`FieldLookup::by_number`]'s empty slot.
const NO_FIELD: u32 = u32::MAX;

impl FieldLookup {
    fn build(fields: &[FieldInfo]) -> Self {
        let mut lookup = FieldLookup::default();
        for (at, info) in fields.iter().enumerate() {
            lookup.insert(at, info);
        }
        lookup
    }

    fn insert(&mut self, at: usize, info: &FieldInfo) {
        self.by_name.entry(info.name.clone()).or_insert(at);
        match (usize::try_from(info.number), u32::try_from(at)) {
            (Ok(n), Ok(at32)) if n < DENSE_FIELD_NUMBERS && at32 != NO_FIELD => {
                if self.by_number.len() <= n {
                    self.by_number.resize(n.saturating_add(1), NO_FIELD);
                }
                if let Some(slot) = self.by_number.get_mut(n) {
                    if *slot == NO_FIELD {
                        *slot = at32;
                    }
                }
            }
            _ => {
                self.sparse_by_number.entry(info.number).or_insert(at);
            }
        }
    }

    fn by_number(&self, number: i32) -> Option<usize> {
        match usize::try_from(number) {
            Ok(n) if n < DENSE_FIELD_NUMBERS => self
                .by_number
                .get(n)
                .filter(|&&at| at != NO_FIELD)
                .map(|&at| at as usize),
            _ => self.sparse_by_number.get(&number).copied(),
        }
    }
}

#[cfg(test)]
mod field_lookup_tests {
    use super::{FieldInfo, FieldLookup};

    #[test]
    fn numbers_resolve_dense_or_sparse_and_the_first_field_wins() {
        let numbers = [0, 65_535, 65_536, i32::MAX, -1, 7, 65_536, 0];
        let fields: Vec<FieldInfo> = numbers
            .iter()
            .enumerate()
            .map(|(i, &n)| FieldInfo::new(format!("f{i}"), n))
            .collect();
        let lookup = FieldLookup::build(&fields);
        assert_eq!(lookup.by_number(0), Some(0));
        assert_eq!(lookup.by_number(65_535), Some(1));
        assert_eq!(lookup.by_number(65_536), Some(2));
        assert_eq!(lookup.by_number(i32::MAX), Some(3));
        assert_eq!(lookup.by_number(-1), Some(4));
        assert_eq!(lookup.by_number(7), Some(5));
        assert_eq!(lookup.by_number(6), None);
        assert_eq!(lookup.by_number(8), None);
        assert_eq!(lookup.by_number(1 << 20), None);
        assert_eq!(lookup.by_number(i32::MIN), None);
    }
}

#[derive(Clone)]
pub(crate) struct IndexingConfig {
    /// `LiveIndexWriterConfig.getMergedSegmentWarmer()`; see
    /// [`IndexWriter::set_merged_segment_warmer`].
    merged_segment_warmer: Option<std::sync::Arc<dyn MergedSegmentWarmer>>,
    /// `IndexWriter.readerPool`: segments opened for delete resolution, kept
    /// for the next round (see [`crate::reader_pool`]). Shared, not copied,
    /// when the configuration is.
    reader_pool: std::sync::Arc<crate::reader_pool::ReaderPool>,
    /// Each IVF field's clusters from the last flush that wrote it, which
    /// live search routes buffered vectors to. Shared, not copied, when the
    /// configuration is.
    ivf_clusters: std::sync::Arc<crate::ivf_vectors::ClusterSink>,
    fields: Vec<FieldInfo>,
    /// Name and number lookups into `fields`, kept in step by
    /// [`IndexingConfig::push_field`] (the only way `fields` grows).
    field_lookup: FieldLookup,
    /// Writer-lifetime reservations, including fields in opened segments that
    /// explicit documents have not registered yet.
    explicit_field_numbers: explicit::FieldNumbers,
    codec_name: String,
    lucene_version: LuceneVersion,
    /// Every field this writer is currently opted into building real
    /// postings for, in insertion order -- see [`IndexWriter::set_postings_field`]/
    /// [`IndexWriter::add_postings_field`] for how entries are added/replaced.
    /// All entries here are batched into a **single**
    /// [`postings_writer::write_fields`] call per `commit()`, so a commit can
    /// carry postings for any number of distinct fields at once (see module
    /// doc comment).
    postings_fields: Vec<PostingsFieldConfig>,
    /// `Analyzer.getPositionIncrementGap(String)` for every field this writer
    /// analyses -- see [`IndexWriter::set_position_increment_gap`]. Java's
    /// default, `0`.
    position_increment_gap: i32,
    /// `Analyzer.getOffsetGap(String)` -- see
    /// [`IndexWriter::set_offset_gap`]. Java's default, `1`.
    offset_gap: i32,
    /// `IndexWriterConfig.getAnalyzer()` for the document API -- see
    /// [`IndexWriter::set_analyzer`]. `None` is the standard analyzer with
    /// the two gaps above.
    document_analyzer: Option<std::sync::Arc<Analyzer>>,
    /// The token-payload supplier for the `store_payloads` fields among
    /// [`Self::postings_fields`] -- this port's stand-in for the
    /// `PayloadAttribute` a real Lucene `TokenFilter` sets, see
    /// [`IndexWriter::set_payload_source`] and
    /// [`crate::indexing_chain::PayloadSource`].
    ///
    /// `None` (the default) means every occurrence of a `store_payloads` field
    /// gets a zero-length payload -- a valid segment (Lucene treats a `null`
    /// payload and a zero-length one identically) that still carries the
    /// `.pay` payload-length stream its `.fnm` promises.
    payload_source: Option<SharedPayloadSource>,
    /// The single field this writer is currently opted into building real
    /// `IndexOptions::DocsAndCustomFreqs` postings for, driven by explicit
    /// per-document caller-supplied `(term, custom_freq)` pairs (see
    /// [`IndexWriter::pending_custom_freq_terms`]) rather than analyzed text -- see
    /// [`IndexWriter::set_custom_freq_postings_field`]. **Mutually exclusive**
    /// with [`Self::postings_fields`] (see that method's doc comment for why):
    /// at most one of the two is ever non-empty/`Some` at a time.
    custom_freq_postings_field: Option<CustomFreqPostingsFieldConfig>,
    /// Every field this writer is currently opted into building real term
    /// vectors for -- same "batched into one write call per commit" shape as
    /// [`Self::postings_fields`], via [`term_vectors::write_best_speed`]
    /// (which already accepts multiple fields per document).
    term_vector_fields: Vec<TermVectorFieldConfig>,
    /// Every field this writer is currently opted into building real doc
    /// values for, in insertion order -- see
    /// [`IndexWriter::set_doc_values_field`]/
    /// [`IndexWriter::add_doc_values_field`]. All of them are written into
    /// **one** `.dvm`/`.dvd`/`.dvs` triple per segment via
    /// [`doc_values::write_dense_fields`], exactly as a real multi-field
    /// segment's `Lucene90DocValuesFormat` files interleave their per-field
    /// meta entries. A single configured field additionally keeps the sparse
    /// write path (see [`IndexWriter::build_doc_values_output`]).
    doc_values_fields: Vec<DocValuesFieldConfig>,
    /// `IndexWriterConfig.getIndexSort()`: the (possibly multi-field)
    /// priority-ordered key every segment this writer flushes is physically
    /// ordered by, and that its `.si` records. `None` -- the default -- means
    /// unsorted, which is what `SegmentInfo`'s `numSortFields == 0` says on
    /// disk. See [`IndexWriter::set_index_sort`].
    index_sort: Option<Vec<segment_info::IndexSortField>>,
    /// `IndexWriterConfig.getParentField()`: the NUMERIC doc-values field
    /// that marks the last document of every `addDocument(s)` call -- the
    /// parent of a block -- with `-1`, which is what lets an index sort move
    /// whole blocks. See [`IndexWriter::set_parent_field`].
    parent_field: Option<String>,
    /// Documents arrive as [`ExplicitDocument`]s: flushes write what each
    /// document states rather than deriving postings, norms, doc values and
    /// points from stored values. See [`IndexWriter::enable_explicit_documents`].
    explicit: bool,
    /// Every field this writer is currently opted into indexing **vectors**
    /// for, in insertion order -- real Lucene's per-field
    /// `KnnFieldVectorsWriter`, which `IndexingChain` creates the first time a
    /// document carries a `KnnFloatVectorField`/`KnnByteVectorField` for that
    /// field. Same "resolve once against the fixed `fields` list, reuse every
    /// flush" shape as [`Self::postings_fields`], and like it a *list*: all of
    /// them are written into one `.vec`/`.vemf`/`.vem`/`.vex` quadruple per
    /// segment, exactly as `Lucene99HnswVectorsFormat` does.
    vector_fields: Vec<VectorFieldConfig>,
    /// The fields whose values this writer indexes as points
    /// (`.kdm`/`.kdi`/`.kdd`, `Lucene90PointsFormat`), in the order they were
    /// opted in -- see [`IndexWriter::add_points_field`].
    points_fields: Vec<PointsFieldConfig>,
    /// `Lucene99HnswVectorsFormat(maxConn, beamWidth)`: the graph parameters
    /// every vector field flushed from here is built with. Defaults are
    /// Lucene's own ([`hnsw::DEFAULT_MAX_CONN`] / [`hnsw::DEFAULT_BEAM_WIDTH`]).
    hnsw_m: i32,
    hnsw_beam_width: i32,
    /// The IO rate, in MB/s, every merge's outputs are held to
    /// (`ConcurrentMergeScheduler`'s per-merge `MergeRateLimiter`); `None`,
    /// the default, is unthrottled. See [`IndexWriter::set_merge_mb_per_sec`].
    merge_mb_per_sec: Option<f64>,
    /// `LiveIndexWriterConfig.useCompoundFile`: whether a flushed segment is
    /// packed into `.cfs`/`.cfe`. See [`IndexWriter::set_use_compound_file`]
    /// for why this writer's default is `false` where Java's is `true`.
    use_compound_file: bool,
    /// The merge policy's `noCFSRatio`/`maxCFSSegmentSize` for every merge
    /// not decided by a pluggable policy; `None` keeps
    /// [`IndexWriter::use_compound_file`] as the switch, with
    /// `TieredMergePolicy`'s defaults. See
    /// [`IndexWriter::set_merge_compound_file_settings`].
    merge_compound_file: Option<merge_policy::CompoundFileSettings>,
    /// `ShuffleForcedMergePolicy` (OpenSearch): forced merges interleave
    /// their sources, oldest with newest. See
    /// [`IndexWriter::set_shuffle_forced_merges`].
    shuffle_forced_merges: bool,
    /// See [`IndexWriter::set_segment_partitioning`].
    segment_partitioning: Option<SegmentPartitioning>,
    /// See [`IndexWriter::add_segment_list_value`].
    segment_list_attributes: Vec<(String, Vec<String>)>,
    /// See [`IndexWriter::set_merge_union_attributes`].
    merge_union_attributes: Vec<String>,
    /// See [`IndexWriter::set_segment_attribute`].
    segment_attributes: Vec<(String, String)>,
    /// `IndexWriterConfig.getSimilarity()`, as far as the writer reads it
    /// (`computeNorm`): `None` is Lucene's default, `BM25Similarity`. See
    /// [`IndexWriter::set_similarity`].
    similarity: Option<std::sync::Arc<dyn NormSimilarity>>,
    /// `PerFieldPostingsFormat.getPostingsFormatForField`: the fields routed
    /// to a postings format other than the default `Lucene104PostingsFormat()`
    /// -- see [`IndexWriter::set_postings_format_for_field`].
    postings_formats: Vec<(String, per_field_postings::Lucene104PostingsFormat)>,
    /// `PerFieldDocValuesFormat.getDocValuesFormatForField`: the fields routed
    /// to a doc-values format other than the default
    /// `Lucene90DocValuesFormat()` -- see
    /// [`IndexWriter::set_doc_values_format_for_field`].
    doc_values_formats: Vec<(String, per_field_doc_values::Lucene90DocValuesFormat)>,
    /// The format a field gets when neither [`Self::doc_values_formats`] nor
    /// the codec routes it anywhere but the default `Lucene90DocValuesFormat()`
    /// -- see [`IndexWriter::set_default_doc_values_format`].
    default_doc_values_format: per_field_doc_values::Lucene90DocValuesFormat,
    /// The Lucene release whose index formats this writer produces -- see
    /// [`IndexWriter::set_index_format`].
    index_format: lucene_codecs::index_format::IndexFormat,
    /// `index.codec.adaptive_doc_values: auto`: the choice between stock and
    /// adaptive doc values for the fields that get the default format, made
    /// from the index's first documents -- see
    /// [`IndexWriter::set_adaptive_doc_values_auto`].
    adaptive_auto: Option<std::sync::Arc<crate::adaptive_auto::AdaptiveAuto>>,
    /// The format a field gets when neither [`Self::postings_formats`] nor
    /// the codec routes it anywhere but the default `Lucene104PostingsFormat()`
    /// -- see [`IndexWriter::set_default_postings_format`].
    default_postings_format: per_field_postings::Lucene104PostingsFormat,
    /// Whether merges write `BEST_SPEED` stored fields with the
    /// high-compression LZ4 match finder -- see
    /// [`IndexWriter::set_merge_lz4_high_compression`].
    merge_lz4_high_compression: bool,
    /// With [`Self::merge_lz4_high_compression`] on, leave the forced merges
    /// (`force_merge`, `_upgrade`) on the fast compressor, see
    /// [`IndexWriter::set_merge_lz4_high_compression_non_force`].
    merge_lz4_hc_non_force: bool,
    /// The fewest vectors an IVF field clusters a partition of the index
    /// sort's first field alone with, `None` for no per-partition clusters
    /// -- see [`IndexWriter::set_ivf_partition_clusters`].
    ivf_partition_clusters: Option<usize>,
    /// The least source `.fdt` bytes for which a merge trains a paged zstd
    /// dictionary -- see [`IndexWriter::set_zstd_dictionary_min_bytes`].
    zstd_dictionary_min_bytes: u64,
    /// `PerFieldKnnVectorsFormat.getKnnVectorsFormatForField`: the fields
    /// routed to a KNN format other than the default
    /// `Lucene99HnswVectorsFormat(hnsw_m, hnsw_beam_width)` -- see
    /// [`IndexWriter::set_knn_vectors_format_for_field`].
    knn_vectors_formats: Vec<(String, per_field_knn_vectors::KnnVectorsFormat)>,
    /// `IndexWriterConfig.setCodec`: the per-field routing every field
    /// without a per-field setting of its own follows -- see
    /// [`IndexWriter::set_codec`]. `None` is `Lucene104Codec()`.
    pub(crate) codec: Option<std::sync::Arc<dyn lucene_codecs::codec::Lucene104Codec>>,
}

/// The doc-values type an index-sort field reads (`SortField.getIndexSorter`'s
/// `NumericDocValues`/`SortedNumericDocValues`/`SortedDocValues`/...).
pub(crate) fn sort_doc_values_type(sf: &segment_info::IndexSortField) -> DocValuesType {
    match &sf.kind {
        segment_info::IndexSortKind::Numeric(_) => DocValuesType::Numeric,
        segment_info::IndexSortKind::SortedNumeric { .. } => DocValuesType::SortedNumeric,
        segment_info::IndexSortKind::String(_) => DocValuesType::Sorted,
        segment_info::IndexSortKind::SortedSet { .. } => DocValuesType::SortedSet,
        segment_info::IndexSortKind::Binary(_) => DocValuesType::Binary,
    }
}

/// One `PerFieldDocValuesFormat` instance's files for a flushed segment:
/// its suffix number, its fields, and its `.dvm`/`.dvd`/`.dvs`.
pub(crate) struct DocValuesGroupOutput {
    /// The instance's format name (`PerFieldDocValuesFormat.format`).
    pub(crate) format_name: &'static str,
    pub(crate) suffix: u32,
    pub(crate) field_numbers: Vec<i32>,
    pub(crate) dvm: Vec<u8>,
    pub(crate) dvd: Vec<u8>,
    pub(crate) dvs: Vec<u8>,
    /// The `.dvp` of a version-3 group (Lucene 10.6).
    pub(crate) dvp: Option<Vec<u8>>,
}

/// [`IndexWriter::begin_merge`]'s snapshot: the sources as the merge reads
/// them, and the merged segment's name and id.
pub(crate) struct MergePlan {
    pub(crate) names: Vec<String>,
    sources: Vec<SegmentCommitInfo>,
    pub(crate) merged_name: String,
    merged_id: [u8; ID_LENGTH],
    /// The sources' files, held by the deleter until the merge ends.
    held: Vec<String>,
    /// Which soft-deleted documents survive this merge -- see
    /// [`IndexWriter::set_soft_deletes_retention`]. `None` keeps them all.
    retention: Option<SoftDeletesRetention>,
    /// The `OneMerge`'s `wrapForMerge`/`reorder`, with the sources as the
    /// segment infos the hooks open them from.
    hooks: Option<(
        std::sync::Arc<dyn crate::merge_policy::MergeHooks>,
        crate::segment_infos::SegmentInfos,
    )>,
    /// SORTED_NUMERIC values rewritten on the way through
    /// ([`IndexWriter::add_indexes_merged_remapping`]).
    sorted_numeric_remap: Option<std::sync::Arc<merge::SortedNumericRemap>>,
    /// What the merged segment's diagnostics record.
    pub(crate) kind: merge::MergeKind,
    /// With a merged-segment warmer, the compound-file decision is made on
    /// the merge thread, before the warmer sees the segment (`mergeMiddle`):
    /// `Some(None)` when the segment cannot be compound. `None` leaves it to
    /// [`IndexWriter::finish_merge`].
    early_compound: Option<Option<pluggable_merge::CompoundFileDecider>>,
    /// `.si` attributes the merged segment records besides the writer's own:
    /// its sources' common group ([`IndexWriter::set_segment_partitioning`]).
    attributes: Vec<(String, String)>,
}

/// `SoftDeletesRetentionMergePolicy` with OpenSearch's retention query,
/// `LongPoint.newRangeQuery("_seq_no", minRetainedSeqNo, Long.MAX_VALUE)`
/// (`SoftDeletesPolicy.getRetentionQuery`): a soft-deleted document survives a
/// merge only while its sequence number is at least `min_retained_seq_no`.
///
/// The range is read off the field's NUMERIC doc values rather than its
/// points; OpenSearch's `SeqNoFieldMapper` indexes the same value both ways.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoftDeletesRetention {
    /// The NUMERIC doc-values field carrying each document's sequence number.
    pub seq_no_field: String,
    /// The lowest sequence number whose soft-deleted documents are kept.
    pub min_retained_seq_no: i64,
    /// OpenSearch's `PrunePostingsMergePolicy`, which `InternalEngine` nests
    /// inside the retention policy: a merge drops this field's postings for
    /// every soft-deleted document it keeps, so a retained old version or
    /// delete tombstone no longer answers a lookup by `_id`. `None` keeps them.
    pub prune_postings_field: Option<String>,
    /// OpenSearch's `RecoverySourcePruneMergePolicy`: a merge keeps this
    /// stored field, and the NUMERIC doc-values field of the same name that
    /// flags it, only on documents whose sequence number is still retained.
    /// `None` keeps it everywhere.
    pub prune_recovery_source_field: Option<String>,
    /// Elasticsearch's `index.disable_sequence_numbers`: a merge rewrites the
    /// NUMERIC doc values of these fields (`_seq_no` to `UNASSIGNED_SEQ_NO`,
    /// `_primary_term` to `UNASSIGNED_PRIMARY_TERM`) on every document whose
    /// sequence number is below `min_retained_seq_no`, so the merged columns
    /// hold constant blocks for history nothing replays any more. Empty
    /// keeps every value.
    pub seq_no_sentinel: Vec<(String, i64)>,
}

/// A source's NUMERIC doc-values column for the first field `pred` accepts,
/// with the field's number in that source.
fn numeric_column<'c>(
    infos: &lucene_codecs::field_infos::FieldInfos,
    columns: &'c SourceDocValueColumns,
    pred: &dyn Fn(&FieldInfo) -> bool,
) -> Option<(&'c [u8], &'c doc_values::NumericEntry, i32)> {
    let field = infos
        .fields
        .iter()
        .find(|f| pred(f) && f.doc_values_type == DocValuesType::Numeric)?;
    let &(_, at) = columns.per_field.iter().find(|(n, _)| *n == field.number)?;
    let (meta, data) = &columns.columns[at];
    Some((
        data.as_slice(),
        meta.numeric_entry(field.number)?,
        field.number,
    ))
}

/// `RecoverySourcePruneMergePolicy.wrapReader` for one source: the field's
/// number and the documents that keep it -- those the retention query
/// (`_seq_no >= min_retained_seq_no`) matches, deleted or not -- or `None`
/// when nothing is pruned: no field configured, no document carrying it, or
/// every document retained.
fn recovery_source_pruning(
    max_doc: i32,
    infos: &lucene_codecs::field_infos::FieldInfos,
    columns: &SourceDocValueColumns,
    retention: &SoftDeletesRetention,
) -> Result<Option<(i32, Option<FixedBitSet>)>> {
    let Some(name) = retention.prune_recovery_source_field.as_deref() else {
        return Ok(None);
    };
    let Some((data, entry, number)) = numeric_column(infos, columns, &|f| f.name == name) else {
        return Ok(None);
    };
    // A corrupt column is a read failure -- tragic, not a refused argument.
    let read_err = Error::DocValuesRead;
    let mut flags = doc_values::NumericReader::new(data, entry);
    let mut flagged = false;
    for doc in 0..max_doc {
        if flags.value(doc).map_err(read_err)?.is_some() {
            flagged = true;
            break;
        }
    }
    if !flagged {
        return Ok(None);
    }
    let len = usize::try_from(max_doc).unwrap_or(0);
    let mut keep = FixedBitSet::new(len);
    if let Some((seq_data, seq_entry, _)) =
        numeric_column(infos, columns, &|f| f.name == retention.seq_no_field)
    {
        let mut seq = doc_values::NumericReader::new(seq_data, seq_entry);
        for doc in 0..len {
            let id = i32::try_from(doc).unwrap_or(i32::MAX);
            if seq
                .value(id)
                .map_err(read_err)?
                .is_some_and(|s| s >= retention.min_retained_seq_no)
            {
                // FBS: `keep` was built with `len` bits and `doc < len`.
                keep.set(doc);
            }
        }
    }
    Ok(match keep.cardinality() {
        n if n == len => None,
        0 => Some((number, None)),
        _ => Some((number, Some(keep))),
    })
}

/// The documents of one source a merge writes sentinels for, and each
/// sentinel field's number in that source with its value.
type SentinelDocs = (FixedBitSet, Vec<(i32, i64)>);

/// [`SoftDeletesRetention::seq_no_sentinel`] for one source: the documents
/// whose sequence number is below the retained minimum, and each sentinel
/// field's number in this source with its value -- or `None` when nothing is
/// rewritten.
fn seq_no_sentinels(
    max_doc: i32,
    infos: &lucene_codecs::field_infos::FieldInfos,
    columns: &SourceDocValueColumns,
    retention: &SoftDeletesRetention,
) -> Result<Option<SentinelDocs>> {
    if retention.seq_no_sentinel.is_empty() {
        return Ok(None);
    }
    let Some((data, entry, _)) =
        numeric_column(infos, columns, &|f| f.name == retention.seq_no_field)
    else {
        return Ok(None);
    };
    let fields: Vec<(i32, i64)> = retention
        .seq_no_sentinel
        .iter()
        .filter_map(|(name, value)| {
            infos
                .fields
                .iter()
                .find(|f| f.name == *name && f.doc_values_type == DocValuesType::Numeric)
                .map(|f| (f.number, *value))
        })
        .collect();
    let len = usize::try_from(max_doc).unwrap_or(0);
    let mut docs = FixedBitSet::new(len);
    let mut seq = doc_values::NumericReader::new(data, entry);
    for doc in 0..len {
        let id = i32::try_from(doc).unwrap_or(i32::MAX);
        if seq
            .value(id)
            .map_err(Error::DocValuesRead)?
            .is_some_and(|s| s < retention.min_retained_seq_no)
        {
            // FBS: `docs` was built with `len` bits and `doc < len`.
            docs.set(doc);
        }
    }
    Ok((docs.cardinality() > 0).then_some((docs, fields)))
}

/// The live documents a merge should read from one source under `retention`:
/// `live` (its hard deletes) minus every soft-deleted document whose sequence
/// number is absent or below the retained minimum. A segment without the
/// soft-deletes column comes back unchanged.
///
/// The second half is the documents whose postings
/// [`SoftDeletesRetention::prune_postings_field`] keeps -- `live` minus every
/// soft-deleted document -- or `None` when nothing is pruned.
fn apply_soft_deletes_retention(
    live: Option<FixedBitSet>,
    max_doc: i32,
    infos: &lucene_codecs::field_infos::FieldInfos,
    columns: &SourceDocValueColumns,
    retention: &SoftDeletesRetention,
) -> Result<(Option<FixedBitSet>, Option<FixedBitSet>)> {
    let Some((soft_data, soft_entry, _)) =
        numeric_column(infos, columns, &|f| f.soft_deletes_field)
    else {
        return Ok((live, None));
    };
    let seq = numeric_column(infos, columns, &|f| f.name == retention.seq_no_field);
    let len = usize::try_from(max_doc).unwrap_or(0);
    let mut out = live.unwrap_or_else(|| {
        let mut all = FixedBitSet::new(len);
        for doc in 0..len {
            // FBS: `all` was just built with `len` bits and `doc < len`.
            all.set(doc);
        }
        all
    });
    let mut soft = doc_values::NumericReader::new(soft_data, soft_entry);
    let mut seq = seq.map(|(data, entry, _)| doc_values::NumericReader::new(data, entry));
    // A corrupt column is a read failure -- tragic, not a refused argument.
    let read_err = Error::DocValuesRead;
    let mut postings_live = retention.prune_postings_field.as_ref().map(|_| out.clone());
    for doc in 0..out.len() {
        if !out.get(doc) {
            continue;
        }
        let id = i32::try_from(doc).unwrap_or(i32::MAX);
        if soft.value(id).map_err(read_err)?.is_none() {
            continue;
        }
        if let Some(postings_live) = postings_live.as_mut() {
            // FBS: a clone of `out`, so `doc < out.len()` bounds it too.
            postings_live.clear(doc);
        }
        let seq_no = match seq.as_mut() {
            Some(reader) => reader.value(id).map_err(read_err)?,
            None => None,
        };
        if seq_no.is_none_or(|s| s < retention.min_retained_seq_no) {
            out.clear(doc);
        }
    }
    Ok((Some(out), postings_live))
}

/// What [`IndexingConfig::run_merge`] produced.
pub(crate) enum MergeOutcome {
    /// No source held a live document: nothing was written.
    AllDeleted,
    Merged {
        /// Boxed: the other variant is empty, and this one is carried across
        /// the unlocked half of a concurrent merge.
        merged: Box<merge::MergedSegment>,
        /// Each source's live documents as the merge read them.
        source_live: Vec<Option<FixedBitSet>>,
        /// Whether the merge thread already made the compound-file decision.
        compound_decided: bool,
    },
}

/// [`IndexWriter::begin_segment`]'s claim on the next segment -- Java's
/// `DocumentsWriterFlushQueue.FlushTicket`.
pub(crate) struct SegmentTicket {
    pub(crate) segment_name: String,
    pub(crate) segment_id: [u8; ID_LENGTH],
    /// What the already-written segments owed when the ticket was taken.
    global: Option<crate::buffered_updates::FrozenBufferedUpdates>,
}

/// A borrowed view of one buffer of documents -- the writer's own, or a
/// [`crate::concurrent_writer`] thread's -- to build a segment from.
pub(crate) struct DocumentBuffer<'b> {
    pub(crate) docs: &'b [Document],
    pub(crate) custom_freq_terms: &'b [Vec<(String, i32)>],
    pub(crate) vectors: &'b [Vec<DocumentVector>],
    /// Aligned with `docs` when the writer takes explicit documents; read only
    /// then.
    pub(crate) explicit: ExplicitDocs<'b>,
    /// The explicit documents' postings, inverted as they were buffered, in
    /// insertion-order doc ids (the flush renumbers them by `old_to_new`);
    /// their `InvertedField`s then hold no terms. `None` for a buffer whose
    /// `explicit` still carries every term.
    pub(crate) postings: Option<&'b pending_postings::PendingPostings>,
    /// Under an index sort, the flush's permutation: `old_to_new[d]` is
    /// where buffered document `d` lands -- what the insertion-order doc
    /// ids of `postings`, or of terms still in `explicit`, are remapped by.
    /// `explicit` is already permuted.
    pub(crate) old_to_new: Option<&'b [usize]>,
    pub(crate) has_blocks: bool,
    /// The stored fields, already compressed as the documents arrived; the
    /// documents in `docs` then hold no values.
    pub(crate) stored: Option<&'b FinishedStored>,
    /// Writes only some of the buffer's documents: see [`BufferLimit`].
    pub(crate) limit: Option<&'b BufferLimit>,
    /// The group the documents were added under
    /// ([`IndexWriter::set_segment_partitioning`]).
    pub(crate) partition: Option<&'b str>,
    /// Vector fields' graphs built as the documents arrived, in their
    /// order: written instead of built when they are the graph the flush
    /// would build ([`IndexWriter::build_vectors_output`]).
    pub(crate) graphs: &'b [BufferGraphField],
}

/// The segments whose flushes wrote a graph their buffer kept
/// ([`IndexWriter::kept_graph`]).
#[cfg(test)]
pub(crate) static KEPT_GRAPHS: std::sync::Mutex<Vec<[u8; ID_LENGTH]>> =
    std::sync::Mutex::new(Vec::new());

/// A vector field's HNSW graph over a buffer's vectors, built as they
/// arrive.
#[derive(Debug, Clone)]
pub(crate) struct BufferGraphField {
    pub(crate) field_number: i32,
    pub(crate) graph: KeptGraph,
}

/// A buffer's graph: one live search reads while it grows
/// ([`lucene_codecs::buffer_vectors::BufferGraph`]), or one only the
/// buffer's flush reads ([`lucene_codecs::buffer_vectors::FlushGraph`]).
#[derive(Debug, Clone)]
pub(crate) enum KeptGraph {
    Live(std::sync::Arc<lucene_codecs::buffer_vectors::BufferGraph>),
    Flush(std::sync::Arc<lucene_codecs::buffer_vectors::FlushGraph>),
}

impl KeptGraph {
    pub(crate) fn params(&self) -> (i32, i32) {
        match self {
            KeptGraph::Live(g) => g.params(),
            KeptGraph::Flush(g) => g.params(),
        }
    }

    pub(crate) fn len(&self) -> i32 {
        match self {
            KeptGraph::Live(g) => g.len(),
            KeptGraph::Flush(g) => g.len(),
        }
    }

    pub(crate) fn shape(
        &self,
    ) -> (
        lucene_codecs::field_infos::VectorEncoding,
        lucene_codecs::field_infos::VectorSimilarityFunction,
        usize,
    ) {
        match self {
            KeptGraph::Live(g) => g.shape(),
            KeptGraph::Flush(g) => g.shape(),
        }
    }

    /// The buffer's documents by ordinal.
    pub(crate) fn docs(&self) -> Option<Vec<i32>> {
        match self {
            KeptGraph::Live(g) => (0..g.len()).map(|ord| g.doc(ord)).collect(),
            KeptGraph::Flush(g) => Some(g.docs()),
        }
    }

    /// The graph over its `n` vectors.
    pub(crate) fn graph(&self, n: i32) -> lucene_codecs::vectors::Result<hnsw::OnHeapHnswGraph> {
        match self {
            KeptGraph::Live(g) => g.graph(n),
            KeptGraph::Flush(g) => g.graph(n),
        }
    }

    /// See [`lucene_codecs::buffer_vectors::BufferGraph::ram_bytes`].
    pub(crate) fn ram_bytes(&self) -> usize {
        match self {
            KeptGraph::Live(g) => g.ram_bytes(),
            KeptGraph::Flush(g) => g.ram_bytes(),
        }
    }

    /// Stops the graph growing and lets its vectors go once nothing else
    /// holds them.
    pub(crate) fn retire(&self) {
        match self {
            KeptGraph::Live(g) => g.retire(),
            KeptGraph::Flush(g) => g.retire(),
        }
    }
}

/// A flush of a buffer's documents `start..docs` only, numbered from
/// `start` (unsorted buffers; `docs`, `explicit` and the per-document
/// slices of the [`DocumentBuffer`] already hold just those): the postings
/// of other documents are left out, and each field's document count is
/// `doc_counts`'s.
#[derive(Debug, Clone, Default)]
pub(crate) struct BufferLimit {
    pub(crate) start: i32,
    pub(crate) docs: i32,
    /// By field number: how many of those documents have the field.
    pub(crate) doc_counts: std::collections::HashMap<i32, i32>,
    /// Whether the postings are written: without them the segment has every
    /// field's norms, doc values, points and stored fields but no terms,
    /// for a caller that reads the documents' terms elsewhere.
    pub(crate) postings: bool,
    /// The documents' fields as columns, numbered from `start`, when the
    /// caller kept them: written as they are instead of being read from
    /// each document.
    pub(crate) columns: Option<ChunkColumns>,
    /// Whether the doc values are written: without them a reader takes
    /// them from [`LazyDocValues`], one field when it is first read.
    pub(crate) doc_values: bool,
    /// Whether the points are written.
    pub(crate) points: bool,
}

/// A run of a buffer's documents as columns, for writing them without
/// decoding each document.
#[derive(Debug, Clone, Default)]
pub(crate) struct ChunkColumns {
    /// The fields some document has in any role but stored, and none
    /// stores term vectors.
    pub(crate) present: Vec<i32>,
    /// Each norms field's norm by document, `None` where it has none.
    pub(crate) norms: Vec<(i32, Vec<Option<i64>>)>,
    pub(crate) doc_values: Vec<(i32, PrebuiltDocValues)>,
    /// The points fields some document has.
    pub(crate) points: Vec<lucene_codecs::points::PackedPointsField>,
}

/// Appends `value` as `info`'s packed point; `false` when it is not one.
pub(crate) fn encode_live_point(info: &FieldInfo, value: &FieldValue, out: &mut Vec<u8>) -> bool {
    let config = PointsFieldConfig {
        name: String::new(),
        field_number: info.number,
        num_dims: info.point_dimension_count,
        num_index_dims: info.point_index_dimension_count,
        bytes_per_dim: info.point_num_bytes,
    };
    let len = out.len();
    let ok = info.point_dimension_count > 0 && encode_point(&config, 0, value, out).is_ok();
    if !ok {
        out.truncate(len);
    }
    ok
}

/// A run of buffered documents' doc values, kept as columns and encoded a
/// field at a time when a reader first reads that field.
pub struct LazyDocValues {
    cfg: std::sync::Arc<IndexingConfig>,
    columns: Vec<(i32, crate::concurrent_writer::DocValuesRun)>,
    max_doc: i32,
}

/// One field's doc values as a `.dvm`/`.dvd` pair of their own.
#[derive(Debug)]
pub struct EncodedDocValues {
    pub dvm: Vec<u8>,
    pub dvd: Vec<u8>,
    /// The `.dvp` of a version-3 pair (Lucene 10.6), `None` before.
    pub dvp: Option<Vec<u8>>,
    /// The pair's codec suffix (`Lucene90_0`), for parsing the `.dvm`.
    pub segment_suffix: String,
}

impl LazyDocValues {
    /// Whether `columns` hold every doc-values field of `present` in its
    /// own type, so that each encodes from its column as a written segment
    /// would hold it; and the writer's codec checks no doc values (a check
    /// would fail a read).
    pub(crate) fn covers(
        cfg: &IndexingConfig,
        columns: &[(i32, crate::concurrent_writer::DocValuesRun)],
        present: &[i32],
    ) -> bool {
        cfg.codec.is_none()
            && cfg
                .fields
                .iter()
                .filter(|f| present.contains(&f.number) && f.doc_values_type != DocValuesType::None)
                .all(|f| {
                    columns
                        .iter()
                        .any(|(n, c)| *n == f.number && c.doc_values_type() == f.doc_values_type)
                })
    }

    pub(crate) fn new(
        cfg: std::sync::Arc<IndexingConfig>,
        columns: Vec<(i32, crate::concurrent_writer::DocValuesRun)>,
        max_doc: i32,
    ) -> Self {
        Self {
            cfg,
            columns,
            max_doc,
        }
    }

    /// Keyword field `field_number`'s values as they arrived, read in
    /// place; `None` for a field of another type or without a column.
    pub fn keyword_run(
        &self,
        field_number: i32,
    ) -> Option<crate::concurrent_writer::KeywordRun<'_>> {
        self.columns
            .iter()
            .find(|(n, _)| *n == field_number)
            .and_then(|(_, r)| r.keyword())
    }

    /// The fields with a column.
    pub fn fields(&self) -> impl Iterator<Item = i32> + '_ {
        self.columns.iter().map(|(n, _)| *n)
    }

    /// Field `field_number`'s doc values as a segment `segment_id` would
    /// hold them, `None` when no document has it.
    pub fn encode(
        &self,
        field_number: i32,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<Option<EncodedDocValues>> {
        let Some(info) = self.cfg.fields.iter().find(|f| f.number == field_number) else {
            return Ok(None);
        };
        let config = DocValuesFieldConfig {
            name: info.name.clone(),
            field_number,
            doc_values_type: info.doc_values_type,
            skip_index: info.doc_values_skip_index_type
                != lucene_codecs::field_infos::DocValuesSkipIndexType::None,
        };
        let Some((_, run)) = self.columns.iter().find(|(n, _)| *n == field_number) else {
            return Ok(None);
        };
        let column = run.build().ok_or_else(|| {
            Error::LiveSearch(format!(
                "field {field_number}'s buffered doc values do not match their documents"
            ))
        })?;
        let Some(groups) = self.cfg.build_prebuilt_doc_values_groups(
            &[(field_number, column)],
            self.max_doc,
            std::slice::from_ref(&config),
            &[info.name.as_str()],
            segment_id,
            false,
        ) else {
            return Ok(None);
        };
        Ok(groups?.into_iter().next().map(|g| EncodedDocValues {
            segment_suffix: format!("{}_{}", g.format_name, g.suffix),
            dvm: g.dvm,
            dvd: g.dvd,
            dvp: g.dvp,
        }))
    }
}

/// One field's doc values for a run of documents, in the shape the
/// doc-values writer takes.
#[derive(Debug, Clone)]
pub(crate) enum PrebuiltDocValues {
    Numeric(Vec<i64>),
    SparseNumeric(Vec<(i32, i64)>),
    SortedNumeric(doc_values::SortedNumericColumn),
    Binary(doc_values::BinaryColumn),
    Sorted(doc_values::SortedOrdsColumn),
    SortedSet(doc_values::SortedSetOrdsColumn),
}

impl PrebuiltDocValues {
    fn doc_values_type(&self) -> DocValuesType {
        match self {
            PrebuiltDocValues::Numeric(_) | PrebuiltDocValues::SparseNumeric(_) => {
                DocValuesType::Numeric
            }
            PrebuiltDocValues::SortedNumeric(_) => DocValuesType::SortedNumeric,
            PrebuiltDocValues::Binary(_) => DocValuesType::Binary,
            PrebuiltDocValues::Sorted(_) => DocValuesType::Sorted,
            PrebuiltDocValues::SortedSet(_) => DocValuesType::SortedSet,
        }
    }

    fn as_dense_field(&self, n: i32) -> doc_values::DenseField<'_> {
        match self {
            PrebuiltDocValues::Numeric(v) => doc_values::DenseField::Numeric(n, v),
            PrebuiltDocValues::SparseNumeric(v) => doc_values::DenseField::SparseNumeric(n, v),
            PrebuiltDocValues::SortedNumeric(c) => {
                doc_values::DenseField::SortedNumericColumn(n, c)
            }
            PrebuiltDocValues::Binary(c) => doc_values::DenseField::BinaryColumn(n, c),
            PrebuiltDocValues::Sorted(c) => doc_values::DenseField::SortedOrds(n, c),
            PrebuiltDocValues::SortedSet(c) => doc_values::DenseField::SortedSetOrds(n, c),
        }
    }
}

/// A buffer's explicit document fields: as structs, or packed as the
/// concurrent writer buffers them.
#[derive(Clone, Copy)]
pub(crate) enum ExplicitDocs<'b> {
    Fields(&'b [ExplicitFields]),
    Packed(&'b packed_fields::PackedFields),
    /// Documents `start..end` of a packed buffer, numbered from `start`.
    PackedRange(&'b packed_fields::PackedFields, usize, usize),
}

impl ExplicitDocs<'_> {
    pub(crate) fn len(&self) -> usize {
        match self {
            ExplicitDocs::Fields(f) => f.len(),
            ExplicitDocs::Packed(p) => p.len(),
            ExplicitDocs::PackedRange(p, s, e) => p.len().min(*e).saturating_sub(*s),
        }
    }

    fn ram_bytes(&self) -> usize {
        match self {
            ExplicitDocs::Fields(f) => f
                .iter()
                .map(ExplicitFields::ram_bytes)
                .fold(0, usize::saturating_add),
            ExplicitDocs::Packed(p) | ExplicitDocs::PackedRange(p, _, _) => p.ram_bytes(),
        }
    }

    /// Calls `f` with each document's fields in order. Packed fields are
    /// decoded one document at a time into a reused struct.
    pub(crate) fn for_each(
        &self,
        mut f: impl FnMut(usize, &ExplicitFields) -> Result<()>,
    ) -> Result<()> {
        match self {
            ExplicitDocs::Fields(fields) => {
                fields.iter().enumerate().try_for_each(|(i, d)| f(i, d))
            }
            ExplicitDocs::Packed(p) => p.for_each(|i, d| f(i, d)),
            ExplicitDocs::PackedRange(p, s, e) => p.for_each_range(*s, *e, |i, d| f(i, d)),
        }
    }

    /// [`Self::for_each`], gathering each document's doc values and points
    /// on the way: borrowed from the structs, or taken out of packed
    /// documents into one flat list per role, so the packed documents are
    /// read once.
    pub(crate) fn for_each_with_roles(
        &self,
        mut f: impl FnMut(usize, &ExplicitFields) -> Result<()>,
    ) -> Result<[RoleValues<'_>; 2]> {
        match self {
            ExplicitDocs::Fields(fields) => {
                fields.iter().enumerate().try_for_each(|(i, d)| f(i, d))?;
                Ok([
                    RoleValues::Borrowed(fields, Role::DocValues),
                    RoleValues::Borrowed(fields, Role::Points),
                ])
            }
            ExplicitDocs::Packed(p) => Self::packed_roles(p, 0, p.len(), f),
            ExplicitDocs::PackedRange(p, s, e) => Self::packed_roles(p, *s, *e, f),
        }
    }

    /// [`Self::for_each_with_roles`] over a packed buffer's documents
    /// `start..end`.
    fn packed_roles<'b>(
        p: &'b packed_fields::PackedFields,
        start: usize,
        end: usize,
        mut f: impl FnMut(usize, &ExplicitFields) -> Result<()>,
    ) -> Result<[RoleValues<'b>; 2]> {
        let mut roles = [Role::DocValues, Role::Points].map(|_| {
            (
                Vec::new(),
                Vec::with_capacity(p.len().min(end).saturating_sub(start)),
            )
        });
        let docs = p.len().min(end).saturating_sub(start);
        // Documents mostly share a shape, so after the first one each role
        // makes room for every document's values like its, rather than
        // regrowing (and refaulting) a flush-sized array by doubling. The
        // estimate is bounded twice, so an odd first document cannot size
        // the array: per document by `PACKED_ROLE_VALUES_PER_DOC`, and in
        // all by what the packed bytes can hold (a packed value takes
        // several bytes; past the bound the array grows as before).
        let bound = p.ram_bytes() / PACKED_BYTES_PER_ROLE_VALUE;
        p.for_each_range(start, end, |i, d| {
            f(i, d)?;
            for ((values, ends), role) in roles.iter_mut().zip([Role::DocValues, Role::Points]) {
                values.append(role.values_mut(d));
                if ends.is_empty() {
                    let per_doc = values.len().min(PACKED_ROLE_VALUES_PER_DOC);
                    values.reserve(per_doc.saturating_mul(docs.saturating_sub(1)).min(bound));
                }
                ends.push(values.len());
            }
            Ok(())
        })?;
        Ok(roles.map(|(values, ends)| RoleValues::Flat { values, ends }))
    }
}

/// [`ExplicitDocs::packed_roles`]' presizing: at most this many values per
/// document, however many the first document has, ...
const PACKED_ROLE_VALUES_PER_DOC: usize = 64;
/// ... and at most one value per this many bytes of the packed buffer.
const PACKED_BYTES_PER_ROLE_VALUE: usize = 4;

/// Which per-document values a [`RoleValues`] holds.
#[derive(Clone, Copy)]
pub(crate) enum Role {
    DocValues,
    Points,
}

impl Role {
    fn values(self, f: &ExplicitFields) -> &[lucene_codecs::stored_fields::StoredField] {
        match self {
            Role::DocValues => &f.doc_values,
            Role::Points => &f.points,
        }
    }

    fn values_mut(
        self,
        f: &mut ExplicitFields,
    ) -> &mut Vec<lucene_codecs::stored_fields::StoredField> {
        match self {
            Role::DocValues => &mut f.doc_values,
            Role::Points => &mut f.points,
        }
    }
}

/// One role's values per document: see [`ExplicitDocs::for_each_with_roles`].
pub(crate) enum RoleValues<'b> {
    Borrowed(&'b [ExplicitFields], Role),
    Flat {
        values: Vec<lucene_codecs::stored_fields::StoredField>,
        /// Per document, where its values end.
        ends: Vec<usize>,
    },
}

impl RoleValues<'_> {
    /// Each document's values.
    pub(crate) fn per_doc(&self) -> Vec<&[lucene_codecs::stored_fields::StoredField]> {
        match self {
            RoleValues::Borrowed(fields, role) => fields.iter().map(|f| role.values(f)).collect(),
            RoleValues::Flat { values, ends } => {
                let mut from = 0;
                ends.iter()
                    .map(|&end| {
                        let s = values.get(from..end).unwrap_or_default();
                        from = end;
                        s
                    })
                    .collect()
            }
        }
    }
}

/// A buffer's stored fields, finished for its segment, and the field numbers
/// they hold.
pub(crate) struct FinishedStored {
    pub(crate) fields: std::collections::BTreeSet<i32>,
    /// Taken by the one write of the segment's stored fields.
    pub(crate) bytes: std::sync::Mutex<Option<segment_writer::StoredFieldsBytes>>,
}

impl DocumentBuffer<'_> {
    /// `FlushInfo.estimatedSegmentSize`: the buffered documents' heap bytes
    /// (Java passes the flushing `DocumentsWriterPerThread`'s `bytesUsed`).
    pub(crate) fn estimated_bytes(&self) -> u64 {
        let docs = self
            .docs
            .iter()
            .map(document_ram_bytes)
            .chain(std::iter::once(self.explicit.ram_bytes()))
            .chain(self.postings.map(pending_postings::PendingPostings::bytes))
            .fold(0usize, usize::saturating_add);
        u64::try_from(docs).unwrap_or(u64::MAX)
    }
}

/// One buffered document's doc-value (or point) fields, as the doc-values
/// and points builders read them: a native [`Document`]'s fields, or an
/// explicit document's borrowed in place.
pub(crate) trait DocValueFields {
    fn dv_fields(&self) -> &[lucene_codecs::stored_fields::StoredField];
}

impl DocValueFields for Document {
    fn dv_fields(&self) -> &[lucene_codecs::stored_fields::StoredField] {
        &self.fields
    }
}

impl DocValueFields for &[lucene_codecs::stored_fields::StoredField] {
    fn dv_fields(&self) -> &[lucene_codecs::stored_fields::StoredField] {
        self
    }
}

impl IndexingConfig {
    /// The configured field named `name`.
    pub(crate) fn field_by_name(&self, name: &str) -> Option<&FieldInfo> {
        self.field_lookup
            .by_name
            .get(name)
            .and_then(|&at| self.fields.get(at))
    }

    /// The configured fields, in the order they were added.
    pub(crate) fn field_list(&self) -> &[FieldInfo] {
        &self.fields
    }

    /// The configured soft-deletes field, if any.
    pub(crate) fn soft_deletes_field(&self) -> Option<&FieldInfo> {
        self.fields.iter().find(|f| f.soft_deletes_field)
    }

    /// The configured field numbered `number`.
    pub(crate) fn field_by_number(&self, number: i32) -> Option<&FieldInfo> {
        self.field_lookup
            .by_number(number)
            .and_then(|at| self.fields.get(at))
    }

    /// Adds a field to the schema, keeping the lookups in step.
    pub(crate) fn push_field(&mut self, info: FieldInfo) {
        self.field_lookup.insert(self.fields.len(), &info);
        self.fields.push(info);
    }

    /// `DocumentsWriterPerThread.updateDocuments`' refusal: a block of `docs`
    /// documents cannot be added to a writer with an index sort and no
    /// parent field. Every entry point that adds a block checks this before
    /// it reserves or buffers anything (`check-port-invariants.py`'s
    /// block-guard rule).
    pub(crate) fn check_block(&self, docs: usize) -> Result<()> {
        if docs > 1 && self.index_sort.is_some() && self.parent_field.is_none() {
            return Err(Error::BlocksWithIndexSortNeedParentField);
        }
        Ok(())
    }

    /// What the concurrent writer refuses of this configuration: a parent
    /// field for native documents, whose building threads have no parent
    /// marker to give them. Explicit documents carry theirs
    /// (`ConcurrentIndexWriter::add_explicit`).
    pub(crate) fn concurrent_unsupported(&self) -> Option<&'static str> {
        (self.parent_field.is_some() && !self.explicit)
            .then_some("a parent field (set_parent_field) without explicit documents")
    }

    /// `IndexWriter._mergeInit`'s `estimatedMergeBytes`: each source's
    /// `sizeInBytes()` pro-rated by its live documents, the size estimate of
    /// the merge's `IOContext(MergeInfo)`. A source whose `.si` cannot be
    /// read counts as empty; the merge itself reports that failure.
    pub(crate) fn estimated_merge_bytes(dir: &dyn Directory, plan: &MergePlan) -> u64 {
        let mut total = 0u64;
        for sci in &plan.sources {
            let Ok(si_bytes) = dir.open(&format!("{}.si", sci.segment_name)) else {
                continue;
            };
            let Ok(si) = segment_info::parse_for_commit(&si_bytes, sci) else {
                continue;
            };
            if si.doc_count <= 0 {
                continue;
            }
            let size = sci
                .files(&si.files)
                .iter()
                .filter_map(|f| dir.file_length(f).ok())
                .fold(0u64, u64::saturating_add);
            let del_ratio = f64::from(sci.del_count) / f64::from(si.doc_count);
            // `(long) (info.sizeInBytes() * (1.0 - delRatio))`: `as` saturates
            // as Java's cast does.
            let live = (size as f64 * (1.0 - del_ratio)) as u64;
            total = total.saturating_add(live);
        }
        total
    }

    /// Java's `globalFieldNumberMap.verifyOrCreateDvOnlyField(field, dvType,
    /// …)`, against this port's fixed field list: the field must exist and its
    /// declared `DocValuesType` must match the update's kind. Java can *create*
    /// the field; this port cannot (the field list is fixed at
    /// [`IndexWriter::open`], see the module doc), so an unknown field is an
    /// error rather than an implicit schema change.
    pub(crate) fn verify_doc_values_update_field(&self, update: &DocValuesUpdate) -> Result<()> {
        let Some(info) = self.field_by_name(update.field()) else {
            return Err(Error::UnknownDocValuesUpdateField(
                update.field().to_string(),
            ));
        };
        let ok = match update {
            DocValuesUpdate::Numeric { .. } => info.doc_values_type == DocValuesType::Numeric,
            DocValuesUpdate::Binary { .. } => info.doc_values_type == DocValuesType::Binary,
        };
        if !ok {
            return Err(Error::WrongDocValuesUpdateType {
                field: update.field().to_string(),
                declared: format!("{:?}", info.doc_values_type),
            });
        }
        self.verify_not_index_sort_field(update.field())
    }

    /// `IndexWriter.updateNumericDocValue`/`updateBinaryDocValue`/
    /// `buildDocValuesUpdate`: `if (config.getIndexSortFields().contains(field))
    /// throw new IllegalArgumentException("cannot update docvalues field
    /// involved in the index sort, ...")`. The segment's *physical* order is
    /// defined by this column, and a doc-values update rewrites the column
    /// without moving any document, so the segment would keep claiming a sort
    /// it no longer satisfies -- silently, since nothing re-checks the order
    /// after an update.
    pub(crate) fn verify_not_index_sort_field(&self, field: &str) -> Result<()> {
        if let Some(sort) = &self.index_sort {
            if sort.iter().any(|sf| sf.field == field) {
                return Err(Error::DocValuesUpdateOnIndexSortField {
                    field: field.to_string(),
                    sort: segment_info::describe_index_sort(Some(sort)),
                });
            }
        }
        Ok(())
    }

    /// [`IndexWriter::update_doc_values`]'s validation, as the node it would
    /// buffer.
    pub(crate) fn doc_values_update_node(
        &self,
        term: &Term,
        updates: &[DocValuesUpdate],
    ) -> Result<DeleteNode> {
        if updates.is_empty() {
            return Err(Error::NoDocValuesUpdatesSupplied);
        }
        let mut retargeted = Vec::with_capacity(updates.len());
        for update in updates {
            self.verify_doc_values_update_field(update)?;
            retargeted.push(retarget_update(update, term));
        }
        Ok(DeleteNode::DocValuesUpdates(retargeted))
    }

    /// The middle of a merge, `IndexWriter.mergeMiddle`: read every source of
    /// `plan` and write the merged segment -- everything that takes time, and
    /// nothing that needs the writer. [`IndexWriter::execute_merge`] runs it
    /// between [`IndexWriter::begin_merge`] and [`IndexWriter::finish_merge`];
    /// a [`crate::concurrent_writer::ConcurrentIndexWriter`] runs it with the
    /// writer's lock released, so indexing carries on.
    ///
    /// Points are opened like every other format: each source's
    /// `.kdm`/`.kdi`/`.kdd` (when its `.si` lists them) becomes one
    /// [`crate::merge::SourcePoints`] per field, and
    /// [`crate::merge::merge_points`] remaps and rebuilds the trees.
    ///
    /// With [`IndexWriter::set_merge_mb_per_sec`] in force, every file the
    /// merge writes goes through a [`crate::merge_rate_limiter::MergeDirectory`]
    /// held to that rate, as `IndexWriter.mergeMiddle` writes through its
    /// rate-limited merge directory.
    pub(crate) fn run_merge(&self, dir: &dyn Directory, plan: &MergePlan) -> Result<MergeOutcome> {
        let progress = std::sync::Arc::new(crate::merge_rate_limiter::OneMergeProgress::new());
        self.run_merge_with_progress(dir, plan, &progress)
    }

    /// [`Self::run_merge`], aborted through `progress`: a throttled merge
    /// stops at its next write, an unthrottled one (no rate to check against)
    /// once its writes are done. Either way an aborted merge fails with
    /// [`Error::MergeAborted`].
    pub(crate) fn run_merge_with_progress(
        &self,
        dir: &dyn Directory,
        plan: &MergePlan,
        progress: &std::sync::Arc<crate::merge_rate_limiter::OneMergeProgress>,
    ) -> Result<MergeOutcome> {
        // Every file of the merge is written under one
        // `IOContext(merge.getStoreMergeInfo())`.
        let estimate = Self::estimated_merge_bytes(dir, plan);
        let outcome = match self.merge_mb_per_sec {
            None => {
                self.run_merge_unthrottled(&lucene_store::EstimatedWrites::new(dir, estimate), plan)
            }
            Some(mb_per_sec) => {
                let limiter =
                    std::sync::Arc::new(crate::merge_rate_limiter::MergeRateLimiter::new(
                        std::sync::Arc::clone(progress),
                    ));
                lucene_store::RateLimiter::set_mb_per_sec(&*limiter, mb_per_sec);
                let merge_dir = crate::merge_rate_limiter::MergeDirectory::new(dir, limiter);
                self.run_merge_unthrottled(
                    &lucene_store::EstimatedWrites::new(&merge_dir, estimate),
                    plan,
                )
            }
        };
        if progress.is_aborted() {
            return Err(Error::MergeAborted);
        }
        outcome
    }

    /// The postings format `field` is routed to.
    pub(crate) fn postings_format_for(
        &self,
        field: &str,
    ) -> per_field_postings::Lucene104PostingsFormat {
        crate::merge::route_postings_format(
            field,
            &self.postings_formats,
            self.codec.as_deref(),
            self.default_postings_format,
        )
    }

    /// `PerFieldPostingsFormat.FieldsWriter.write`: this segment's postings,
    /// one set of files per postings format its fields are routed to.
    pub(crate) fn write_postings_groups<T: postings_writer::TermSource>(
        &self,
        inputs: &[FieldPostingsInput<'_, T>],
        norms: &[postings_writer::FieldNorms<'_>],
        state: lucene_codecs::postings_consumer::PostingsWriteState<'_>,
    ) -> Result<PostingsGroups> {
        let name_of = |n: i32| {
            self.fields
                .iter()
                .find(|f| f.number == n)
                .map(|f| f.name.clone())
        };
        let format_for = |name: &str| self.postings_format_for(name);
        if let Some(factory) = self
            .codec
            .as_deref()
            .and_then(|c| c.postings_writer_factory())
        {
            return Ok(crate::postings_writer::write_installed(
                factory,
                state,
                inputs,
                norms,
                &name_of,
                &format_for,
                false,
            )?);
        }
        Ok(per_field_postings::write(
            inputs,
            state.field_infos,
            norms,
            &name_of,
            &format_for,
            state.id,
        )?)
    }

    /// [`Self::run_merge_with_progress`], with every output written through
    /// `limiter` when a merge scheduler supplies one
    /// (`MergeScheduler.wrapForMerge`) instead of the writer's own configured
    /// rate. A supplied limiter carries its own progress.
    pub(crate) fn run_merge_limited(
        &self,
        dir: &dyn Directory,
        plan: &MergePlan,
        limiter: Option<std::sync::Arc<crate::merge_rate_limiter::MergeRateLimiter>>,
        progress: &std::sync::Arc<crate::merge_rate_limiter::OneMergeProgress>,
    ) -> Result<MergeOutcome> {
        let Some(limiter) = limiter else {
            return self.run_merge_with_progress(dir, plan, progress);
        };
        // The same `IOContext(merge.getStoreMergeInfo())` as `run_merge`.
        let estimate = Self::estimated_merge_bytes(dir, plan);
        let progress = std::sync::Arc::clone(limiter.merge_progress());
        let merge_dir = crate::merge_rate_limiter::MergeDirectory::new(dir, limiter);
        let outcome = self.run_merge_unthrottled(
            &lucene_store::EstimatedWrites::new(&merge_dir, estimate),
            plan,
        );
        if progress.is_aborted() {
            return Err(Error::MergeAborted);
        }
        outcome
    }

    /// [`Self::run_merge`] with no rate limit.
    /// [`Self::run_merge_reading`], failed when a page of a file under
    /// `dir` could not be read or decoded while it ran: zeros may have
    /// stood in for it (`lucene_store::read_failures`), and the merged
    /// segment would keep them.
    fn run_merge_unthrottled(&self, dir: &dyn Directory, plan: &MergePlan) -> Result<MergeOutcome> {
        let since = lucene_store::read_failures::epoch();
        let outcome = self.run_merge_reading(dir, plan)?;
        if let Some(root) = dir.fs_directory_path() {
            if let Some(message) =
                lucene_store::read_failures::failure_under(&root.to_string_lossy(), since)
            {
                return Err(Error::Store(lucene_store::Error::Io(
                    std::io::Error::other(format!("merge read a page that failed: {message}")),
                )));
            }
        }
        Ok(outcome)
    }

    /// One sort tier's key for every buffered document, in buffer order --
    /// what `IndexingChain.maybeSortSegment` gets from each field's
    /// `DocValuesWriter.getDocComparator`.
    ///
    /// The numeric kinds read the document's value(s) directly; the three
    /// byte-keyed kinds (`STRING`, `SortedSetSortField`, `BinarySortField`)
    /// collect each document's bytes exactly as the column builder will
    /// ([`IndexWriter::collect_binary_values`]/[`IndexWriter::collect_sorted_set_values`])
    /// and rank them ([`crate::index_sorter::rank_terms`]) -- the ordinals
    /// `SortedDocValuesWriter`'s sorted `BytesRefHash` hands Java's
    /// comparator, so the key and the column are one fact.
    fn buffer_sort_keys(
        &self,
        docs: &[Document],
        sf: &segment_info::IndexSortField,
    ) -> Result<Vec<Option<i64>>> {
        let field_number = self
            .fields
            .iter()
            .find(|f| f.name == sf.field)
            .map(|f| f.number)
            .expect("set_index_sort resolved every sort field against this fixed list");
        let selector = match &sf.kind {
            segment_info::IndexSortKind::SortedNumeric { selector, .. } => Some(*selector),
            segment_info::IndexSortKind::Numeric(_) => None,
            segment_info::IndexSortKind::String(_)
            | segment_info::IndexSortKind::SortedSet { .. }
            | segment_info::IndexSortKind::Binary(_) => {
                return self.buffer_term_sort_keys(docs, sf, field_number);
            }
        };
        Ok(docs
            .iter()
            .map(|doc| {
                // A SORTED_NUMERIC column is "repeat the field on the
                // document" here (see
                // `build_sorted_numeric_doc_values_output`), and the
                // values are stored **sorted**
                // (`SortedNumericDocValuesWriter.finishCurrentDoc`),
                // so the selector has to be applied to the sorted
                // form -- `SortedNumericSelector.MIN`/`MAX` are the
                // first and the last *stored* value, not the first
                // and last the caller happened to write. Sorting
                // here rather than reading the column keeps the sort
                // key and the column one fact, which is what stops
                // the two from drifting.
                let mut values: Vec<i64> = doc
                    .fields
                    .iter()
                    .filter(|f| f.field_number == field_number)
                    .filter_map(|f| match &f.value {
                        FieldValue::Int(v) => Some(*v as i64),
                        FieldValue::Long(v) => Some(*v),
                        // Not numeric: `build_doc_values_output` is
                        // about to fail the whole flush with
                        // `NonNumericDocValue` naming the document, a
                        // better message than anything this could
                        // raise. Treated as missing until then.
                        _ => None,
                    })
                    .collect();
                match selector {
                    // Single-valued NUMERIC: the document's one value.
                    None => values.into_iter().next(),
                    Some(segment_info::SortedNumericSelector::Min) => {
                        values.sort_unstable();
                        values.into_iter().next()
                    }
                    Some(segment_info::SortedNumericSelector::Max) => {
                        values.sort_unstable();
                        values.into_iter().last()
                    }
                }
            })
            .collect())
    }

    /// [`Self::buffer_sort_keys`] for the byte-keyed kinds.
    fn buffer_term_sort_keys(
        &self,
        docs: &[Document],
        sf: &segment_info::IndexSortField,
        field_number: i32,
    ) -> Result<Vec<Option<i64>>> {
        let config = self
            .doc_values_fields
            .iter()
            .find(|c| c.field_number == field_number)
            .ok_or_else(|| Error::IndexSortFieldWithoutDocValues(sf.field.clone()))?;
        let mut terms: Vec<Option<Vec<u8>>> = vec![None; docs.len()];
        if let segment_info::IndexSortKind::SortedSet { selector, .. } = &sf.kind {
            for (doc, mut values) in IndexWriter::collect_sorted_set_values(docs, config)? {
                // `SortedSetDocValuesWriter` stores a document's values
                // deduplicated and ascending; the selector picks among those.
                values.sort_unstable();
                values.dedup();
                terms[doc as usize] =
                    crate::index_sorter::select_sorted_set(&values, *selector).cloned();
            }
        } else {
            for (doc, value) in IndexWriter::collect_binary_values(docs, config)? {
                terms[doc as usize] = Some(value);
            }
        }
        Ok(crate::index_sorter::rank_terms(&[terms])
            .pop()
            .unwrap_or_default())
    }

    /// Puts one document buffer into index-sort order and returns the
    /// `new -> old` permutation (`None`: unsorted index, or already in order)
    /// -- the half of `IndexWriter::flush` a
    /// [`crate::concurrent_writer::ConcurrentIndexWriter`] thread runs on its
    /// own buffer. See `IndexWriter::sort_pending_buffer`.
    pub(crate) fn sort_buffer(
        &self,
        docs: &mut [Document],
        custom_freq_terms: &mut [Vec<(String, i32)>],
        vectors: &mut [Vec<DocumentVector>],
        explicit: &mut [ExplicitFields],
        has_blocks: bool,
    ) -> Result<Option<Vec<usize>>> {
        let Some(sort) = self.index_sort.clone() else {
            return Ok(None);
        };
        // `IndexingChain.maybeSortSegment`: a segment with blocks is sorted
        // block by block, each document by its parent's key, which needs the
        // parent field to find the parents; without one it is refused.
        let parents: Option<Vec<bool>> = match (has_blocks, self.parent_field_number()) {
            (false, _) => None,
            (true, Some(parent)) if self.explicit => Some(
                explicit
                    .iter()
                    .map(|f| f.doc_values.iter().any(|v| v.field_number == parent))
                    .collect(),
            ),
            (true, _) => return Err(Error::IndexSortWithBlocksAndNoParentField),
        };

        let mut keys: Vec<Vec<Option<i64>>> = sort
            .iter()
            .map(|sf| {
                if self.explicit {
                    Ok(self.explicit_sort_keys(explicit, sf))
                } else {
                    self.buffer_sort_keys(docs, sf)
                }
            })
            .collect::<Result<_>>()?;
        if let Some(parents) = &parents {
            for tier in &mut keys {
                explicit::key_of_parent(tier, &|doc| parents[doc]);
            }
        }

        let specs: Vec<segment_writer::SortKeySpec<'_>> = sort
            .iter()
            .zip(keys.iter())
            .map(|(sf, k)| segment_writer::SortKeySpec { sort: sf, keys: k })
            .collect();
        let new_to_old = segment_writer::sort_permutation(docs.len(), &specs);
        drop(specs);

        // `Sorter.sortAndLeaveUnpacked` returns null when the documents are
        // already in order; the segment is still *recorded* as sorted, there
        // is simply nothing to permute and no map any delete limit needs.
        if new_to_old.iter().enumerate().all(|(new, &old)| new == old) {
            return Ok(None);
        }

        segment_writer::permute_in_place(docs, &new_to_old);
        segment_writer::permute_in_place(custom_freq_terms, &new_to_old);
        segment_writer::permute_in_place(vectors, &new_to_old);
        if self.explicit {
            segment_writer::permute_in_place(explicit, &new_to_old);
        }
        Ok(Some(new_to_old))
    }

    /// `starts` with the writer's partition size, for IVF fields clustered
    /// per partition.
    pub(crate) fn ivf_partitions<'a>(
        &self,
        starts: Option<&'a [i32]>,
    ) -> Option<crate::ivf_vectors::Partitions<'a>> {
        Some(crate::ivf_vectors::Partitions {
            starts: starts?,
            min_vectors: self.ivf_partition_clusters?,
        })
    }

    /// Each partition's first document in a sorted buffer, for IVF fields
    /// clustered per partition ([`IndexWriter::set_ivf_partition_clusters`]):
    /// where the index sort's first key changes. `None` when the setting is
    /// off, the index is unsorted, or the buffer holds blocks or a range of
    /// a packed buffer.
    pub(crate) fn ivf_partition_starts(
        &self,
        buf: &DocumentBuffer<'_>,
    ) -> Result<Option<Vec<i32>>> {
        let Some(first) = self.index_sort.as_ref().and_then(|s| s.first()) else {
            return Ok(None);
        };
        if self.ivf_partition_clusters.is_none() || buf.has_blocks || buf.limit.is_some() {
            return Ok(None);
        }
        let keys = match buf.explicit {
            ExplicitDocs::Fields(fields) if self.explicit => self.explicit_sort_keys(fields, first),
            _ if self.explicit => return Ok(None),
            _ => self.buffer_sort_keys(buf.docs, first)?,
        };
        let mut starts = Vec::new();
        for (doc, key) in keys.iter().enumerate() {
            if doc == 0 || keys.get(doc.wrapping_sub(1)) != Some(key) {
                starts.push(i32::try_from(doc).unwrap_or(i32::MAX));
            }
        }
        Ok(Some(starts))
    }

    /// [`Self::buffer_sort_keys`] for explicit documents: each document's
    /// value of the sort field out of its doc values -- the column the
    /// segment will carry -- through the tier's selector; byte-keyed tiers
    /// ranked. A field no document has registered sorts every document as
    /// missing, as Java's empty `DocValues.getNumeric` does.
    fn explicit_sort_keys(
        &self,
        explicit: &[ExplicitFields],
        sf: &segment_info::IndexSortField,
    ) -> Vec<Option<i64>> {
        let Some(number) = self
            .fields
            .iter()
            .find(|f| f.name == sf.field)
            .map(|f| f.number)
        else {
            return vec![None; explicit.len()];
        };
        let longs = |f: &ExplicitFields| -> Vec<i64> {
            f.doc_values
                .iter()
                .filter(|v| v.field_number == number)
                .filter_map(|v| match &v.value {
                    FieldValue::Long(x) => Some(*x),
                    FieldValue::Int(x) => Some(i64::from(*x)),
                    _ => None,
                })
                .collect()
        };
        let bytes = |f: &ExplicitFields| -> Vec<Vec<u8>> {
            f.doc_values
                .iter()
                .filter(|v| v.field_number == number)
                .filter_map(|v| match &v.value {
                    FieldValue::Binary(b) => Some(b.clone()),
                    _ => None,
                })
                .collect()
        };
        match &sf.kind {
            segment_info::IndexSortKind::Numeric(_) => explicit
                .iter()
                .map(|f| longs(f).into_iter().next())
                .collect(),
            segment_info::IndexSortKind::SortedNumeric { selector, .. } => explicit
                .iter()
                .map(|f| {
                    let mut values = longs(f);
                    values.sort_unstable();
                    match selector {
                        segment_info::SortedNumericSelector::Min => values.first().copied(),
                        segment_info::SortedNumericSelector::Max => values.last().copied(),
                    }
                })
                .collect(),
            segment_info::IndexSortKind::SortedSet { selector, .. } => {
                let terms: Vec<Option<Vec<u8>>> = explicit
                    .iter()
                    .map(|f| {
                        let mut values = bytes(f);
                        values.sort_unstable();
                        values.dedup();
                        crate::index_sorter::select_sorted_set(&values, *selector).cloned()
                    })
                    .collect();
                crate::index_sorter::rank_terms(&[terms])
                    .pop()
                    .unwrap_or_default()
            }
            segment_info::IndexSortKind::String(_) | segment_info::IndexSortKind::Binary(_) => {
                let terms: Vec<Option<Vec<u8>>> = explicit
                    .iter()
                    .map(|f| bytes(f).into_iter().next())
                    .collect();
                crate::index_sorter::rank_terms(&[terms])
                    .pop()
                    .unwrap_or_default()
            }
        }
    }

    /// The one `Analyzer` this writer analyses every field with -- Java's
    /// `IndexWriterConfig.getAnalyzer()`. This facade has no per-field
    /// analyzer configuration (see the module doc comment), so it is a plain
    /// `Analyzer::standard` carrying the writer's two gap settings, which are
    /// the only part of it a caller can configure
    /// ([`IndexWriter::set_position_increment_gap`] /
    /// [`IndexWriter::set_offset_gap`]).
    fn analyzer(&self) -> Analyzer {
        Analyzer::standard(None)
            .with_position_increment_gap(self.position_increment_gap)
            .with_offset_gap(self.offset_gap)
    }

    /// Whether this flush can take the `IndexingChain`-shaped inverter
    /// ([`crate::inverter::FieldInverter`]) instead of the general
    /// [`IndexWriter::invert_pending_fields`]: no term vectors and no payloads (the
    /// consumers only the general path's `InMemoryInvertedIndex` serves), and
    /// every norms column belonging to a postings field, so the inverter's
    /// own length counts cover all of them.
    fn fast_inversion_applies(&self, norms: &[NormsFieldConfig]) -> bool {
        !self.postings_fields.is_empty()
            && self.term_vector_fields.is_empty()
            && self.payload_source.is_none()
            && self.postings_fields.iter().all(|p| !p.store_payloads)
            && norms.iter().all(|n| {
                self.postings_fields
                    .iter()
                    .any(|p| p.field_number == n.field_number)
            })
    }

    /// `getDocValuesFormatForField(name)`: the format
    /// [`IndexWriter::set_doc_values_format_for_field`] routed `name` to, or
    /// the default `Lucene90DocValuesFormat()`.
    /// `getKnnVectorsFormatForField(name)`: the routed format, or the
    /// default `Lucene99HnswVectorsFormat(hnsw_m, hnsw_beam_width)`.
    /// [`lucene_codecs::codec::Lucene104Codec::knn_vectors_format_is_direct`]
    /// of the configured codec.
    pub(crate) fn knn_vectors_direct(&self) -> bool {
        self.codec
            .as_ref()
            .is_some_and(|c| c.knn_vectors_format_is_direct())
    }

    pub(crate) fn knn_vectors_format_for(
        &self,
        name: &str,
    ) -> per_field_knn_vectors::KnnVectorsFormat {
        self.knn_vectors_formats
            .iter()
            .find(|(field, _)| field == name)
            .map(|(_, format)| *format)
            .or_else(|| {
                self.codec
                    .as_ref()
                    .map(|c| c.knn_vectors_format_for_field(name))
            })
            .unwrap_or(per_field_knn_vectors::KnnVectorsFormat::Hnsw {
                max_conn: self.hnsw_m,
                beam_width: self.hnsw_beam_width,
            })
    }

    /// The doc-values format field `name` is written in: its explicit
    /// route, else the codec's, else `default`, which the caller resolves
    /// once per segment so every field sees the same `auto` choice.
    fn doc_values_format_with_default(
        &self,
        name: &str,
        default: per_field_doc_values::Lucene90DocValuesFormat,
    ) -> per_field_doc_values::Lucene90DocValuesFormat {
        let disi_file = self.index_format.doc_values_disi_file();
        // A field routed explicitly keeps its format, even the default one.
        if let Some((_, format)) = self
            .doc_values_formats
            .iter()
            .find(|(field, _)| field == name)
        {
            return format.with_disi_file(disi_file);
        }
        let routed = self
            .codec
            .as_ref()
            .map(|c| c.doc_values_format_for_field(name))
            .unwrap_or_default();
        if routed == per_field_doc_values::Lucene90DocValuesFormat::default() {
            default
        } else {
            routed
        }
        .with_disi_file(disi_file)
    }

    /// [`Self::default_doc_values_format`], or under `auto` its choice:
    /// adaptive once chosen so, stock until then or when chosen not.
    fn resolved_default_doc_values_format(&self) -> per_field_doc_values::Lucene90DocValuesFormat {
        let disi_file = self.index_format.doc_values_disi_file();
        match &self.adaptive_auto {
            None => self.default_doc_values_format,
            Some(auto) if auto.decided() == Some(true) => {
                per_field_doc_values::Lucene90DocValuesFormat::rust_adaptive()
            }
            Some(_) => per_field_doc_values::Lucene90DocValuesFormat::default(),
        }
        .with_disi_file(disi_file)
    }

    /// Feeds `auto`'s sample with a segment about to be written from
    /// buffered documents: each numeric field every document has exactly one
    /// value of. Nothing once the choice is made.
    fn observe_adaptive_auto_docs<D: DocValueFields>(
        &self,
        docs: &[D],
        configs: &[DocValuesFieldConfig],
    ) {
        let Some(auto) = self
            .adaptive_auto
            .as_ref()
            .filter(|a| a.decided().is_none())
        else {
            return;
        };
        let numbers: Vec<i32> = configs
            .iter()
            .filter(|c| {
                sampled_field(c)
                    && matches!(
                        c.doc_values_type,
                        DocValuesType::Numeric | DocValuesType::SortedNumeric
                    )
            })
            .map(|c| c.field_number)
            .collect();
        let mut columns: Vec<Vec<i64>> = vec![Vec::with_capacity(docs.len()); numbers.len()];
        let mut single = vec![true; numbers.len()];
        let mut counts = vec![0u32; numbers.len()];
        for doc in docs {
            counts.fill(0);
            for f in doc.dv_fields() {
                let (Some(i), lucene_codecs::stored_fields::FieldValue::Long(v)) =
                    (numbers.iter().position(|&n| n == f.field_number), &f.value)
                else {
                    continue;
                };
                counts[i] = counts[i].saturating_add(1);
                columns[i].push(*v);
            }
            for (ok, &c) in single.iter_mut().zip(&counts) {
                *ok &= c == 1;
            }
        }
        auto.observe(
            docs.len(),
            numbers
                .iter()
                .zip(&columns)
                .zip(&single)
                .filter(|(_, &ok)| ok)
                .map(|((&n, c), _)| (n, c.as_slice())),
        );
    }

    /// [`Self::observe_adaptive_auto_docs`] for a segment written from
    /// columns already built.
    fn observe_adaptive_auto_columns(
        &self,
        columns: &[(i32, PrebuiltDocValues)],
        max_doc: i32,
        configs: &[DocValuesFieldConfig],
    ) {
        let Some(auto) = self
            .adaptive_auto
            .as_ref()
            .filter(|a| a.decided().is_none())
        else {
            return;
        };
        let max = usize::try_from(max_doc).unwrap_or(0);
        let sampled = |n: i32| {
            configs
                .iter()
                .any(|c| c.field_number == n && sampled_field(c))
        };
        auto.observe(
            max,
            columns
                .iter()
                .filter(|(n, _)| sampled(*n))
                .filter_map(|(n, c)| match c {
                    PrebuiltDocValues::Numeric(v) => Some((*n, v.as_slice())),
                    PrebuiltDocValues::SortedNumeric(c)
                        if c.docs.len() == max && c.values.len() == max =>
                    {
                        Some((*n, c.values.as_slice()))
                    }
                    _ => None,
                }),
        );
    }

    /// `IndexingChain.writeDocValues` through `PerFieldDocValuesFormat`: the
    /// fields of `configs`, visited in `IndexingChain`'s field-hash order
    /// ([`crate::indexing_chain::field_hash_order`] over `segment_fields`,
    /// the segment's fields in field-number order -- the order a writer whose
    /// documents share one field order first sees them), each go to the
    /// instance of the format they are routed to, instances numbered in
    /// that order; each instance's fields are written, in that order, into
    /// its own `.dvm`/`.dvd`/`.dvs`.
    ///
    /// `observe`: the segment is a flush of the writer's own documents,
    /// which feeds `auto`'s sample (not a search's view of buffered ones).
    fn build_doc_values_groups<D: DocValueFields>(
        &self,
        docs: &[D],
        configs: &[DocValuesFieldConfig],
        segment_fields: &[&str],
        segment_id: &[u8; ID_LENGTH],
        observe: bool,
    ) -> Result<Vec<DocValuesGroupOutput>> {
        if observe {
            self.observe_adaptive_auto_docs(docs, configs);
        }
        self.doc_values_groups_with(
            i32::try_from(docs.len()).unwrap_or(i32::MAX),
            configs,
            segment_fields,
            segment_id,
            |group_configs, suffix, format| {
                IndexWriter::build_doc_values_files(docs, group_configs, segment_id, suffix, format)
            },
        )
    }

    /// [`Self::build_doc_values_groups`] from columns already built, one
    /// per field of `configs` (by field number); `None` when one is missing
    /// or of another type. `observe`: a flush of the writer's own
    /// documents, which feeds `auto`'s sample (not one field's columns
    /// encoded on their own, nor a search's view of buffered documents).
    fn build_prebuilt_doc_values_groups(
        &self,
        columns: &[(i32, PrebuiltDocValues)],
        max_doc: i32,
        configs: &[DocValuesFieldConfig],
        segment_fields: &[&str],
        segment_id: &[u8; ID_LENGTH],
        observe: bool,
    ) -> Option<Result<Vec<DocValuesGroupOutput>>> {
        let column = |n: i32| columns.iter().find(|(f, _)| *f == n).map(|(_, c)| c);
        let covered = configs.iter().all(|c| {
            column(c.field_number).is_some_and(|col| col.doc_values_type() == c.doc_values_type)
        });
        if !covered {
            return None;
        }
        if observe {
            self.observe_adaptive_auto_columns(columns, max_doc, configs);
        }
        Some(self.doc_values_groups_with(
            max_doc,
            configs,
            segment_fields,
            segment_id,
            |group_configs, suffix, format| {
                let skip_indexes: Vec<i32> = group_configs
                    .iter()
                    .filter(|c| c.skip_index)
                    .map(|c| c.field_number)
                    .collect();
                let fields: Vec<doc_values::DenseField<'_>> = group_configs
                    .iter()
                    .filter_map(|c| {
                        column(c.field_number).map(|col| col.as_dense_field(c.field_number))
                    })
                    .collect();
                Ok(doc_values::write_fields_with_format_files(
                    &fields,
                    &skip_indexes,
                    max_doc,
                    segment_id,
                    &format!("{}_{suffix}", format.name()),
                    format,
                )?)
            },
        ))
    }

    /// Every doc-values field's files for `docs`, written from each
    /// document's values and from `columns`, as `(dvm, dvd, dvs)` per
    /// group.
    #[cfg(test)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn doc_values_both_ways(
        &self,
        docs: &[&[lucene_codecs::stored_fields::StoredField]],
        columns: &[(i32, PrebuiltDocValues)],
    ) -> (
        Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
        Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    ) {
        let configs: Vec<DocValuesFieldConfig> = self
            .fields
            .iter()
            .filter(|f| f.doc_values_type != DocValuesType::None)
            .map(|f| DocValuesFieldConfig {
                name: f.name.clone(),
                field_number: f.number,
                doc_values_type: f.doc_values_type,
                skip_index: f.doc_values_skip_index_type
                    != lucene_codecs::field_infos::DocValuesSkipIndexType::None,
            })
            .collect();
        let names: Vec<&str> = self.fields.iter().map(|f| f.name.as_str()).collect();
        let id = [7u8; ID_LENGTH];
        let files = |g: Vec<DocValuesGroupOutput>| {
            g.into_iter()
                .map(|g| (g.dvm, g.dvd, g.dvs))
                .collect::<Vec<_>>()
        };
        let decoded = self
            .build_doc_values_groups(docs, &configs, &names, &id, false)
            .unwrap();
        let max_doc = i32::try_from(docs.len()).unwrap();
        let prebuilt = self
            .build_prebuilt_doc_values_groups(columns, max_doc, &configs, &names, &id, false)
            .expect("a column per field")
            .unwrap();
        (files(decoded), files(prebuilt))
    }

    /// Groups `configs` by doc-values format and writes each group with
    /// `build`, checked against the installed codec.
    fn doc_values_groups_with(
        &self,
        max_doc: i32,
        configs: &[DocValuesFieldConfig],
        segment_fields: &[&str],
        segment_id: &[u8; ID_LENGTH],
        build: impl Fn(
            &[DocValuesFieldConfig],
            u32,
            per_field_doc_values::Lucene90DocValuesFormat,
        ) -> Result<doc_values::DocValuesFiles>,
    ) -> Result<Vec<DocValuesGroupOutput>> {
        let walk = crate::indexing_chain::field_hash_order(segment_fields);
        let rank = |name: &str| {
            walk.iter()
                .position(|&i| segment_fields[i] == name)
                .unwrap_or(usize::MAX)
        };
        let mut ordered: Vec<&DocValuesFieldConfig> = configs.iter().collect();
        ordered.sort_by_key(|c| rank(&c.name));
        let fields: Vec<(i32, &str)> = ordered
            .iter()
            .map(|c| (c.field_number, c.name.as_str()))
            .collect();
        let default = self.resolved_default_doc_values_format();
        let format_for = |name: &str| self.doc_values_format_with_default(name, default);
        per_field_doc_values::group_fields(&fields, &format_for)
            .into_iter()
            .map(|group| {
                let group_configs: Vec<DocValuesFieldConfig> = ordered
                    .iter()
                    .filter(|c| group.field_numbers.contains(&c.field_number))
                    .map(|c| (*c).clone())
                    .collect();
                let doc_values::DocValuesFiles { dvm, dvd, dvs, dvp } =
                    build(&group_configs, group.suffix, group.format)?;
                let field_infos = lucene_codecs::field_infos::FieldInfos {
                    fields: self
                        .fields
                        .iter()
                        .filter(|f| group.field_numbers.contains(&f.number))
                        .cloned()
                        .collect(),
                };
                check_doc_values(
                    self.codec.as_deref(),
                    &lucene_codecs::codec::DocValuesWrite {
                        max_doc,
                        segment_id,
                        segment_suffix: &group.codec_suffix(),
                        format_name: group.format.name(),
                        field_infos: &field_infos,
                        dvm: &dvm,
                        dvd: &dvd,
                        dvs: &dvs,
                    },
                )?;
                Ok(DocValuesGroupOutput {
                    format_name: group.format.name(),
                    suffix: group.suffix,
                    field_numbers: group.field_numbers,
                    dvm,
                    dvd,
                    dvs,
                    dvp,
                })
            })
            .collect()
    }

    /// Stamp the `PerField*Format` attributes real Lucene's codec writes at
    /// flush time onto the fields this commit actually produced files for, so
    /// the `.fnm` points at the suffixed names
    /// [`IndexWriter::write_postings_files`]/[`IndexWriter::write_doc_values_files`] used.
    /// Lucene does this inside `PerFieldPostingsFormat.fieldsConsumer`, which
    /// mutates the `FieldInfo` it is handed; this port's `.fnm` writer takes
    /// its fields by shared reference, so the decorated list is built here.
    ///
    /// Only fields that really got postings (indexed) or doc values are
    /// stamped: an attribute naming a format whose files were never written
    /// would send a reader looking for a file that does not exist.
    fn fields_with_per_field_attributes(
        &self,
        postings: Option<&PostingsGroups>,
        doc_values: Option<&[DocValuesGroupOutput]>,
        wrote_norms: bool,
        vector_fields_written: &[WrittenVectorField],
        points_fields_written: &[String],
    ) -> Vec<FieldInfo> {
        let postings_group =
            |number: i32| postings?.iter().find(|g| g.field_numbers.contains(&number));
        // A direct group's fields carry no per-field attributes.
        let postings_suffix = |number: i32| -> Option<(&'static str, u32)> {
            postings_group(number)
                .filter(|g| !g.direct)
                .map(|g| (g.format_name, g.suffix))
        };
        let dv_suffix = |number: i32| -> Option<(&'static str, u32)> {
            doc_values?
                .iter()
                .find(|g| g.field_numbers.contains(&number))
                .map(|g| (g.format_name, g.suffix))
        };

        self.fields
            .iter()
            // Java discovers point-only fields from actual documents. An
            // opted-in but absent point-only field must not manufacture a
            // zero-dimensional schema that conflicts with a later pointful segment.
            // A field not opted into points still belongs to stored fields.
            .filter(|field| {
                field.point_dimension_count == 0
                    || !self
                        .points_fields
                        .iter()
                        .any(|points| points.field_number == field.number)
                    || points_fields_written.iter().any(|name| name == &field.name)
                    || field.index_options != IndexOptions::None
                    || field.doc_values_type != DocValuesType::None
                    || field.vector_dimension != 0
            })
            .map(|f| {
                let mut f = f.clone();
                // The configured facade retains its declared indexed schema
                // even in segments without this field's text. Keep hasVectors
                // stable across those segments, and emit empty native streams
                // for configured fields rather than advertising missing files.
                f.store_term_vectors &= self
                    .term_vector_fields
                    .iter()
                    .any(|config| config.field_number == f.number);
                // No norms coercion: `norms_field_configs` writes a column for exactly
                // the fields whose `.fnm` claims one, so the two cannot disagree.
                // (`DirectoryReader.open` throws on a `.fnm` claiming norms the
                // segment does not carry.)
                debug_assert!(
                    !wrote_norms
                        || f.omit_norms
                        || f.index_options == IndexOptions::None
                        || self.norms_field_configs().iter().any(|c| c.name == f.name),
                    "every indexed non-omitNorms field must have a norm column"
                );
                // The custom-frequency field is a field of a document only
                // when the document gave it terms
                // (`add_document_with_custom_freq_terms`): a segment none of
                // whose documents did -- a thread's buffer of other documents
                // -- must not claim it indexed without a term dictionary, as
                // Java's segment would not list it. (With norms its column
                // is written regardless, so the claim stays.)
                if f.omit_norms
                    && postings_group(f.number).is_none()
                    && self
                        .custom_freq_postings_field
                        .as_ref()
                        .is_some_and(|c| c.field_number == f.number)
                {
                    f.index_options = IndexOptions::None;
                }
                if let Some((format_name, suffix)) = postings_suffix(f.number) {
                    f.attributes.push((
                        per_field_postings::PER_FIELD_FORMAT_KEY.to_string(),
                        format_name.to_string(),
                    ));
                    f.attributes.push((
                        per_field_postings::PER_FIELD_SUFFIX_KEY.to_string(),
                        suffix.to_string(),
                    ));
                }
                // Same rule as norms: a `.fnm` must not claim what the
                // segment's files do not hold. A positive `vector_dimension`
                // makes `FieldInfo.hasVectorValues()` true, which is what
                // `IncrementalHnswGraphMerger` and `CheckIndex` key off, while
                // `PerFieldKnnVectorsFormat` registers no reader for a field
                // with no format attribute -- so the field reads back as
                // vector-capable and yields nothing. (Measured: real Lucene
                // tolerates that combination without an error, which is exactly
                // why it has to be got right here rather than left to fail
                // loudly.) Java never reaches the state at all, because
                // `IndexingChain` sets `vectorDimension` from the first document
                // that carries the field, so a segment's `.fnm` and its `.vemf`
                // are written from the same fact.
                // The same rule again for doc values, which had it missing:
                // a `.fnm` entry with a non-NONE `DocValuesType` and no
                // `PerFieldDocValuesFormat.format` attribute makes
                // `PerFieldDocValuesFormat.FieldsReader` register no producer
                // for the field, so `getNumeric` returns `null` and real
                // `CheckIndex.testDocValues` -- which iterates every field
                // whose `.fnm` claims doc values -- dereferences it. This
                // port's own `check_index` reports the same thing
                // (`doc_values.entry_present:<field>`). Java never reaches
                // the state: `IndexingChain` creates a `DocValuesWriter` for
                // every field whose `FieldType` declares a type, so the
                // `.fnm` and the `.dvm` are written from one fact.
                if dv_suffix(f.number).is_none() {
                    f.doc_values_type = DocValuesType::None;
                    f.doc_values_skip_index_type =
                        lucene_codecs::field_infos::DocValuesSkipIndexType::None;
                }
                if let Some(written) = vector_fields_written.iter().find(|w| w.name == f.name) {
                    // A direct `knnVectorsFormat()` records no attributes.
                    if !written.direct {
                        f.attributes.push((
                            per_field_knn_vectors::PER_FIELD_FORMAT_KEY.to_string(),
                            written.format.to_string(),
                        ));
                        f.attributes.push((
                            per_field_knn_vectors::PER_FIELD_SUFFIX_KEY.to_string(),
                            written.suffix.to_string(),
                        ));
                    }
                } else {
                    f.vector_dimension = 0;
                }
                // PointsOutput retains the declared shape for an opted-in
                // field needed by another feature, even without point values.
                // Its empty points triple makes the null BKD entry readable
                // and avoids a schema mismatch with later pointful segments.
                if !points_fields_written.iter().any(|n| n == &f.name) {
                    f.point_dimension_count = 0;
                    f.point_index_dimension_count = 0;
                    f.point_num_bytes = 0;
                }
                if let Some((format_name, suffix)) = dv_suffix(f.number) {
                    f.attributes.push((
                        per_field_doc_values::PER_FIELD_FORMAT_KEY.to_string(),
                        format_name.to_string(),
                    ));
                    f.attributes.push((
                        per_field_doc_values::PER_FIELD_SUFFIX_KEY.to_string(),
                        suffix.to_string(),
                    ));
                }
                f
            })
            .collect()
    }

    /// Every field this writer writes a norm column for, ascending by field
    /// number -- this port's `IndexingChain.writeNorms` loop condition.
    ///
    /// Java writes norms for **every** indexed field whose `omitNorms` is
    /// false, with no per-writer opt-in anywhere:
    ///
    /// ```java
    /// for (FieldInfo fi : state.fieldInfos) {
    ///   if (fi.omitsNorms() == false && fi.getIndexOptions() != IndexOptions.NONE) {
    ///     perField.norms.finish(state.segmentInfo.maxDoc());
    ///     perField.norms.flush(state, sortMap, normsConsumer);
    ///   }
    /// }
    /// ```
    ///
    /// -- so that is what this returns. [`IndexWriter::omit_norms_field`] is
    /// the opt-*out* (`FieldType.setOmitNorms(true)`); there is no opt-in
    /// because Lucene has none.
    ///
    /// Ascending field number, because that is the order
    /// [`lucene_codecs::norms::write_fields`] wants its meta entries in and
    /// it makes the `.nvm` a function of the schema rather than of the
    /// order the caller declared the fields.
    /// The similarity norms are computed with: the configured one, or
    /// Lucene's default (`BM25Similarity`'s `computeNorm`).
    pub(crate) fn norm_similarity(&self) -> &dyn NormSimilarity {
        match &self.similarity {
            Some(s) => s.as_ref(),
            None => &crate::similarity::DefaultNormSimilarity,
        }
    }

    fn norms_field_configs(&self) -> Vec<NormsFieldConfig> {
        let mut configs: Vec<NormsFieldConfig> = self
            .fields
            .iter()
            .filter(|f| !f.omit_norms && !matches!(f.index_options, IndexOptions::None))
            .map(|f| NormsFieldConfig {
                name: f.name.clone(),
                field_number: f.number,
                docs_only: f.index_options == IndexOptions::Docs,
            })
            .collect();
        configs.sort_by_key(|c| c.field_number);
        configs
    }

    /// The `store_payloads` subset of this writer's declared fields, by name --
    /// exactly the fields
    /// [`crate::indexing_chain::invert_documents_with_payloads`] should
    /// allocate payload slots for.
    ///
    /// Taken from `self.fields` rather than from `self.postings_fields`
    /// because payloads are consumed by two writers, not one: the postings
    /// (`.pay`'s payload-length run) and the term vectors (a per-occurrence
    /// payload in the `.tvd` chunk). A declared field that is opted into
    /// neither never appears in the invert pass's input at all, so listing it
    /// here costs nothing.
    fn payload_field_names(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|f| f.store_payloads)
            .map(|f| f.name.as_str())
            .collect()
    }

    /// Everything `IndexWriter::flush` builds and writes for one segment,
    /// from the shared invert pass to the `.si` patch -- extracted so that
    /// **every** way it can fail is one `Err` at one call site.
    ///
    /// That matters because the caller has, by this point, already permuted
    /// the document buffer into index-sort order, and a failure it did not
    /// catch would leave the buffer permuted: every buffered delete's
    /// `docIDUpto` is a position in the *insertion*-ordered buffer
    /// (`pending_doc_id_upto`), so a retry over a permuted buffer would
    /// compare limits against the wrong doc space -- silently, and with a
    /// valid `.liv` and a clean `CheckIndex`. It is also why this takes
    /// `&self`: nothing in here may mutate the buffer.
    pub(crate) fn build_and_write_segment(
        &self,
        dir: &dyn Directory,
        buf: &DocumentBuffer<'_>,
        segment_name: &str,
        segment_id: [u8; ID_LENGTH],
        flush_deletes: Option<&mut FlushDeletes<'_>>,
    ) -> Result<(SegmentCommitInfo, Vec<String>)> {
        // `DocumentsWriterPerThread.flush`: every file of the segment is
        // written under one `IOContext(new FlushInfo(numDocs, bytesUsed))`,
        // whose size estimate an `NRTCachingDirectory` caches small segments
        // by.
        let flush_dir = lucene_store::EstimatedWrites::new(dir, buf.estimated_bytes());
        let dir: &dyn Directory = &flush_dir;
        if self.explicit {
            return self.build_and_write_explicit_segment(dir, buf, segment_name, segment_id);
        }
        // `IndexingChain.writeNorms`' loop condition, resolved once: every
        // indexed field that has not opted out gets a norm column, and the
        // shared invert pass has to analyze all of them.
        let norms_configs = self.norms_field_configs();
        let installed_norms = self
            .codec
            .as_deref()
            .and_then(|codec| codec.norms_writer_factory())
            .is_some();
        let norms_state = lucene_codecs::norms_writer::NormsWriteState {
            directory: dir,
            segment: segment_name,
            id: &segment_id,
            suffix: "",
            max_doc: buf.docs.len() as i32,
        };
        let postings_state = lucene_codecs::postings_consumer::PostingsWriteState {
            directory: dir,
            segment: segment_name,
            id: &segment_id,
            suffix: "",
            max_doc: buf.docs.len() as i32,
            field_infos: &self.fields,
            format: per_field_postings::Lucene104PostingsFormat::default(),
        };
        let build_doc_values = || -> Result<_> {
            Ok(if self.doc_values_fields.is_empty() {
                None
            } else {
                let names: Vec<&str> = self.fields.iter().map(|f| f.name.as_str()).collect();
                Some(self.build_doc_values_groups(
                    buf.docs,
                    &self.doc_values_fields,
                    &names,
                    &segment_id,
                    buf.limit.is_none(),
                )?)
            })
        };
        let mut doc_values_output = if installed_norms {
            None
        } else {
            build_doc_values()?
        };
        let build_points = || -> Result<_> {
            if self.points_fields.is_empty() {
                Ok(None)
            } else {
                IndexWriter::build_points_output(
                    self,
                    buf.docs,
                    &self.points_fields,
                    &segment_id,
                    dir,
                    segment_name,
                )
            }
        };
        // `PerFieldKnnVectorsFormat` picks a field's format when the field
        // is first added (Java: during indexing, before any flush), so the
        // codec is asked once per field here, in first-seen order, before the
        // graphs build alongside inversion -- never from a worker thread, in
        // whatever order the stages happen to run.
        let mut vector_formats: Vec<(&str, per_field_knn_vectors::KnnVectorsFormat)> = Vec::new();
        for v in buf.vectors.iter().flatten() {
            if !vector_formats.iter().any(|(n, _)| *n == v.field_name) {
                vector_formats.push((&v.field_name, self.knn_vectors_format_for(&v.field_name)));
            }
        }
        let vector_format_for = |name: &str| {
            vector_formats
                .iter()
                .find(|(n, _)| *n == name)
                .map_or_else(|| self.knn_vectors_format_for(name), |(_, f)| *f)
        };
        let partition_starts = self.ivf_partition_starts(buf)?;
        let partitions = self.ivf_partitions(partition_starts.as_deref());
        let build_vectors = || -> Result<_> {
            Ok(if self.vector_fields.is_empty() {
                None
            } else {
                IndexWriter::build_vectors_output(
                    partitions,
                    buf.vectors,
                    &self.vector_fields,
                    buf.docs.len() as i32,
                    &vector_format_for,
                    self.knn_vectors_direct(),
                    buf.limit.is_none(),
                    buf.graphs,
                    buf.old_to_new,
                    &self.ivf_clusters,
                    &segment_id,
                )?
            })
        };
        let (points_output, vectors_output);
        let (norms_output, term_vectors_output, postings_output);
        let stored_fields_bytes;
        if self.fast_inversion_applies(&norms_configs) {
            // `IndexingChain`'s own shape: a term hash per field, lengths
            // counted as tokens go by, postings handed to the writer in term
            // order -- see `crate::inverter`. Taken whenever nothing needs the
            // general pass's `InMemoryInvertedIndex` (term vectors, payloads).
            // Stored fields' compression shares nothing with inversion, so
            // it runs alongside it; only the file writes stay in order below.
            // Vector graphs (HNSW building is often a flush's largest single
            // cost) are a pure function of the buffer too; their result is
            // only looked at where it always was, so errors surface in the
            // same order.
            let ((inverters, stored), vectors) = rayon::join(
                || {
                    rayon::join(
                        || {
                            IndexWriter::invert_pending_fields_fast(
                                buf.docs,
                                &self.postings_fields,
                                &self.analyzer(),
                            )
                        },
                        || {
                            segment_writer::build_native_stored_fields(
                                buf.docs,
                                &segment_id,
                                self.codec.as_deref(),
                            )
                        },
                    )
                },
                build_vectors,
            );
            stored_fields_bytes = stored;
            norms_output = if norms_configs.is_empty() {
                None
            } else {
                Some(IndexWriter::build_norms_output_from_lengths(
                    buf.docs,
                    &norms_configs,
                    &inverters,
                    self.norm_similarity(),
                    self,
                    norms_state,
                )?)
            };
            if installed_norms {
                doc_values_output = build_doc_values()?;
            }
            points_output = build_points()?;
            vectors_output = vectors?;

            term_vectors_output = None;
            let impact_norms: &[(i32, Vec<i64>)] = norms_output
                .as_ref()
                .map(|output| output.impact_norms.as_slice())
                .unwrap_or(&[]);
            postings_output = IndexWriter::build_postings_output_fast(
                self,
                &self.postings_fields,
                inverters,
                impact_norms,
                postings_state,
                flush_deletes,
            )?;
        } else {
            let ((inverted, stored), vectors) = rayon::join(
                || {
                    rayon::join(
                        || {
                            IndexWriter::invert_pending_fields(
                                buf.docs,
                                &self.postings_fields,
                                &self.term_vector_fields,
                                &norms_configs,
                                &self.payload_field_names(),
                                self.payload_source.as_deref(),
                                &self.analyzer(),
                            )
                        },
                        || {
                            segment_writer::build_native_stored_fields(
                                buf.docs,
                                &segment_id,
                                self.codec.as_deref(),
                            )
                        },
                    )
                },
                build_vectors,
            );
            stored_fields_bytes = stored;
            // Norms and term vectors only *read* the shared invert pass;
            // postings **consumes** it. Ordering them this way means the whole
            // inverted index -- by far the largest transient structure in a
            // flush -- is freed term by term as the postings are built,
            // instead of staying live alongside the postings copy, the
            // stored-fields copy and every output file's byte buffer.
            norms_output = if norms_configs.is_empty() {
                None
            } else {
                Some(IndexWriter::build_norms_output(
                    buf.docs,
                    &norms_configs,
                    &inverted,
                    self.custom_freq_postings_field
                        .as_ref()
                        .map(|cfg| (cfg.field_number, buf.custom_freq_terms)),
                    self.norm_similarity(),
                    self,
                    norms_state,
                )?)
            };
            if installed_norms {
                doc_values_output = build_doc_values()?;
            }
            points_output = build_points()?;
            vectors_output = vectors?;

            // Borrowed out of `norms_output` before the postings consume
            // `inverted`, because the postings writer needs them for its
            // impacts.
            let impact_norms: &[(i32, Vec<i64>)] = norms_output
                .as_ref()
                .map(|output| output.impact_norms.as_slice())
                .unwrap_or(&[]);
            term_vectors_output = if self.term_vector_fields.is_empty() {
                None
            } else {
                IndexWriter::build_term_vectors_output(
                    buf.docs,
                    &self.term_vector_fields,
                    &inverted,
                )
                .map(|docs| {
                    self.write_term_vectors_output(
                        &docs,
                        &self.fields,
                        lucene_codecs::term_vectors_writer::TermVectorsWriteState {
                            directory: dir,
                            segment: segment_name,
                            id: &segment_id,
                            suffix: self.term_vectors_format().map_or("", |f| f.segment_suffix),
                            max_doc: buf.docs.len() as i32,
                            format: self.term_vectors_format(),
                        },
                    )
                })
                .transpose()?
            };
            postings_output = if !self.postings_fields.is_empty() {
                IndexWriter::build_postings_output(
                    self,
                    &self.postings_fields,
                    inverted,
                    impact_norms,
                    postings_state,
                )?
            } else {
                drop(inverted);
                match &self.custom_freq_postings_field {
                    Some(cfg) => IndexWriter::build_custom_freq_postings_output(
                        self,
                        buf.docs,
                        buf.custom_freq_terms,
                        cfg,
                        postings_state,
                    )?,
                    None => None,
                }
            };
        }

        let fnm_fields = self.partition_fields(
            self.fields_with_per_field_attributes(
                postings_output.as_ref(),
                doc_values_output.as_deref(),
                norms_output.is_some(),
                vectors_output
                    .as_ref()
                    .map(|o| o.written.as_slice())
                    .unwrap_or(&[]),
                points_output
                    .as_ref()
                    .map(|o| o.written_fields.as_slice())
                    .unwrap_or(&[]),
            ),
            buf.partition,
        );

        // `IndexWriter.sealFlushedSegment`: every format writes its own files
        // into the segment, each one adding them to the *in-memory*
        // `SegmentInfo.files`, and the `.si` is written once at the end from
        // that accumulated set.
        //
        // Rewriting it per file group would re-open, re-parse, extend and
        // re-fsync a file whose entire content is already in memory, and only
        // the last write would survive.
        before_write(
            self.codec.as_deref(),
            lucene_codecs::codec::WritePoint::FieldInfos,
        )?;
        let mut flushed = segment_writer::write_stored_only_segment_files_with_codec(
            dir,
            segment_name,
            segment_id,
            &self.codec_name,
            self.lucene_version,
            &fnm_fields,
            buf.docs,
            // `DocumentsWriterPerThread.updateDocuments`: set once any
            // `add_documents`/`update_documents` call in this buffer
            // carried more than one document.
            buf.has_blocks,
            self.codec.as_deref(),
            stored_fields_bytes,
        )?;

        let mut record = |names: Vec<String>| {
            flushed.info.files.extend(names);
        };
        if let Some(output) = postings_output {
            record(IndexWriter::write_postings_files(
                dir,
                segment_name,
                &output,
            )?);
        }
        if let Some(output) = term_vectors_output {
            record(output.files);
            if let Some((tvd, tvx, tvm)) = output.bytes {
                record(IndexWriter::write_term_vector_files(
                    dir,
                    segment_name,
                    self.term_vectors_format().map_or("", |f| f.segment_suffix),
                    &tvd,
                    &tvx,
                    &tvm,
                )?);
            }
        }
        for group in doc_values_output.iter().flatten() {
            record(with_doc_values_faults(
                self.codec.as_deref(),
                group.field_numbers.len(),
                || IndexWriter::write_doc_values_files(dir, segment_name, group),
            )?);
        }
        if let Some(output) = norms_output {
            record(output.files);
            if let Some((nvm, nvd)) = output.bytes {
                record(IndexWriter::write_norms_files(
                    dir,
                    segment_name,
                    &nvm,
                    &nvd,
                )?);
            }
        }
        if let Some(output) = &vectors_output {
            record(IndexWriter::write_vector_files(dir, segment_name, output)?);
        }
        if let Some(output) = &points_output {
            record(IndexWriter::write_points_files(dir, segment_name, output)?);
        }
        // `SegmentInfo`'s `numSortFields` block is what a reader surfaces as
        // `LeafMetaData.getSort()`, and what `CheckIndex.testSort` re-derives
        // its comparators from. It goes into the same in-memory `SegmentInfo`
        // as the file names, so it costs nothing extra.
        if let Some(sort) = &self.index_sort {
            flushed.info.index_sort = Some(sort.clone());
        }
        let extra = write_codec_segment_files(
            self.codec.as_deref(),
            lucene_codecs::codec::SegmentWrite {
                directory: dir,
                segment: segment_name,
                id: &segment_id,
                max_doc: flushed.info.doc_count,
                field_infos: &fnm_fields,
                files: &flushed.info.files,
            },
        )?;
        flushed.info.files.extend(extra);
        if self.use_compound_file {
            // Before the `.si`, so the `.si` records the compound layout and
            // is not itself packed.
            before_write(
                self.codec.as_deref(),
                lucene_codecs::codec::WritePoint::Compound,
            )?;
            segment_writer::create_compound_file(dir, segment_name, &mut flushed.info)?;
        }
        before_write(
            self.codec.as_deref(),
            lucene_codecs::codec::WritePoint::SegmentInfo,
        )?;
        self.partition_segment(&mut flushed.info.attributes, buf.partition);
        self.record_list_attributes(&mut flushed.info.attributes);
        segment_writer::seal_flushed_segment(dir, segment_name, flushed).map_err(Error::from)
    }

    /// Java: `TermVectorsConsumer.flush`. Installed sinks keep the native
    /// wire format; codecs without a sink retain the direct encoder path.
    /// The installed codec's `CompressingCodec` format, which its term
    /// vectors (and stored fields) are written in.
    fn term_vectors_format(
        &self,
    ) -> Option<&'static lucene_codecs::compression::CompressingFormat> {
        self.codec
            .as_deref()
            .and_then(|codec| codec.compressing_format())
    }

    fn write_term_vectors_output(
        &self,
        documents: &[TermVectorsDocument],
        infos: &[FieldInfo],
        state: lucene_codecs::term_vectors_writer::TermVectorsWriteState<'_>,
    ) -> Result<TermVectorsOutput> {
        match self
            .codec
            .as_deref()
            .and_then(|codec| codec.term_vectors_writer_factory())
        {
            Some(factory) => Ok(TermVectorsOutput {
                files: crate::term_vectors_writer::flush(factory, state, infos, documents)?,
                bytes: None,
            }),
            None => Ok(TermVectorsOutput {
                files: Vec::new(),
                bytes: Some(match state.format {
                    None => term_vectors::write_best_speed(documents, state.id, state.suffix),
                    Some(format) => {
                        let mut writer =
                            term_vectors::TermVectorsWriter::with_format(state.id, format);
                        for document in documents {
                            writer.add_document(document);
                        }
                        writer.finish()
                    }
                }),
            }),
        }
    }

    fn finish_norms_output(
        &self,
        columns: Vec<NormsColumn>,
        configs: &[NormsFieldConfig],
        state: lucene_codecs::norms_writer::NormsWriteState<'_>,
    ) -> Result<NormsOutput> {
        let fields: Vec<_> = configs
            .iter()
            .zip(&columns)
            .map(|(config, column)| match &column.sparse {
                None => norms::NormsField::Dense(config.field_number, &column.dense),
                Some(pairs) => norms::NormsField::Sparse(config.field_number, pairs),
            })
            .collect();
        let factory = self
            .codec
            .as_deref()
            .and_then(|codec| codec.norms_writer_factory());
        let (bytes, files) = if let Some(factory) = factory {
            (
                None,
                crate::norms_writer::write_installed(factory, state, &self.fields, &fields)?,
            )
        } else {
            (
                Some(norms::write_fields(
                    &fields,
                    state.max_doc,
                    state.id,
                    state.suffix,
                )?),
                Vec::new(),
            )
        };
        let impact_norms = configs
            .iter()
            .zip(columns)
            .map(|(config, column)| (config.field_number, column.dense))
            .collect();
        Ok(NormsOutput {
            bytes,
            files,
            impact_norms,
        })
    }
}

/// A single, coherent entry point over this port's write-side primitives.
/// See the module doc comment for the exact lifecycle and scope.
/// What a segment's soft-delete count depends on: its deletes and its doc
/// values, each by generation.
type SoftDeleteKey = (i64, i64, i64);

fn soft_delete_key(sci: &SegmentCommitInfo) -> SoftDeleteKey {
    (sci.del_gen, sci.doc_values_gen, sci.field_infos_gen)
}

/// `BitSet.of(reader.getNumericDocValues(fieldInfos.getParentField()), maxDoc)`:
/// the documents of a source that carry the parent field, `None` when the
/// source has no parent field.
fn source_parents(
    fields: &[FieldInfo],
    columns: &SourceDocValueColumns,
    max_doc: i32,
) -> Result<Option<FixedBitSet>> {
    let Some(field) = fields
        .iter()
        .find(|f| f.parent_field && f.doc_values_type == DocValuesType::Numeric)
    else {
        return Ok(None);
    };
    let mut parents = FixedBitSet::new(usize::try_from(max_doc).unwrap_or(0));
    let Some(&(_, at)) = columns.per_field.iter().find(|(n, _)| *n == field.number) else {
        return Ok(Some(parents));
    };
    let (meta, data) = &columns.columns[at];
    let Some(entry) = meta.numeric_entry(field.number) else {
        return Ok(Some(parents));
    };
    let mut reader = doc_values::NumericReader::new(data, entry);
    for doc in 0..max_doc {
        if reader.value(doc)?.is_some() {
            // FBS: `doc < max_doc`, the set's length.
            parents.set(doc as usize);
        }
    }
    Ok(Some(parents))
}

/// `IndexWriter.readFieldInfos(sci)`: a segment's current `FieldInfos`,
/// through its compound file when it has one.
pub(crate) fn segment_field_infos(
    dir: &dyn Directory,
    sci: &SegmentCommitInfo,
) -> Result<lucene_codecs::field_infos::FieldInfos> {
    // Java reads updated FieldInfos outside the compound file, without
    // opening the base archive. Updates take precedence for both codecs.
    if sci.codec_name == "SimpleText" {
        use lucene_codecs::simpletext::{compound, field_infos};
        let suffix = if sci.field_infos_gen == -1 {
            String::new()
        } else {
            lucene_util::base36::to_base36(sci.field_infos_gen)
        };
        let name = field_infos::field_infos_file_name(&sci.segment_name, &suffix);
        if sci.field_infos_gen != -1 {
            return Ok(field_infos::read(&dir.open(&name)?)?);
        }
        let si = segment_info::parse_simpletext(
            &dir.open(&format!("{}.si", sci.segment_name))?,
            &sci.segment_name,
            &sci.segment_id,
        )?;
        if si.is_compound_file {
            let data = dir.open(&compound::compound_file_name(&sci.segment_name))?;
            let archive = compound::SimpleTextCompoundReader::open(&data, &sci.segment_name)?;
            return Ok(field_infos::read(archive.open_input(&data, &name)?)?);
        }
        return Ok(field_infos::read(&dir.open(&name)?)?);
    }
    if sci.field_infos_gen != -1 {
        return crate::field_updates::read_current_field_infos(dir, sci, &[]).map_err(Error::from);
    }
    let si = segment_info::parse_for_commit(&dir.open(&format!("{}.si", sci.segment_name))?, sci)?;
    if si.is_compound_file {
        let compound = CompoundReader::open(dir, &sci.segment_name, &sci.segment_id)?;
        let files = compound.member_files();
        return crate::field_updates::read_current_field_infos(&compound, sci, &files)
            .map_err(Error::from);
    }
    crate::field_updates::read_current_field_infos(dir, sci, &si.files).map_err(Error::from)
}

/// `IndexWriter.getFieldNumberMap` for an `IndexWriterConfig` whose parent
/// and soft-deletes fields are unset. Validate before the deleter can remove
/// files, and compare every occurrence with the first schema for its name.
fn validate_default_field_config(dir: &dyn Directory, infos: &SegmentInfos) -> Result<()> {
    let mut schemas = std::collections::HashMap::<String, FieldInfo>::new();
    for sci in &infos.segments {
        for field in segment_field_infos(dir, sci)?.fields {
            if field.soft_deletes_field {
                return Err(explicit::explicit_error(format!(
                    "this index has [{}] as soft-deletes already but soft-deletes field is not \
                     configured in IWC",
                    field.name
                )));
            }
            if field.parent_field {
                return Err(explicit::explicit_error(format!(
                    "can't add field [{}] as parent document field; this IndexWriter has no \
                     parent document field configured",
                    field.name
                )));
            }
            if let Some(previous) = schemas.get(&field.name) {
                if let Some(message) = explicit::schema_change(previous, &field) {
                    return Err(explicit::explicit_error(message));
                }
            } else {
                schemas.insert(field.name.clone(), field);
            }
        }
    }
    Ok(())
}

/// `IndexWriter.MAX_DOCS`: the most documents one index may hold, deleted
/// ones included -- `Integer.MAX_VALUE - 128`, leaving headroom for the
/// arrays Java sizes by `maxDoc`.
pub const MAX_DOCS: usize = i32::MAX as usize - 128;

/// `TieredMergePolicy`'s default `forceMergeDeletesPctAllowed`: a forced
/// deletes merge takes a segment only when more than this percentage of its
/// documents can be reclaimed.
pub const FORCE_MERGE_DELETES_PCT_ALLOWED: i64 = 10;

/// `IndexWriter.WRITE_LOCK_NAME`: the lock every writer holds for as long as
/// it is open, so no two writers -- in this process, another Rust process,
/// or a Java `IndexWriter` -- ever modify one index at once.
pub const WRITE_LOCK_NAME: &str = "write.lock";

/// One merge as a [`MergeListener`] sees it: `OneMerge` as
/// `OpenSearchConcurrentMergeScheduler.doMerge` reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeEvent {
    /// Unique among this process's merges (`OnGoingMerge.getId`).
    pub id: u64,
    /// The merged segments' names, in merge order.
    pub segments: Vec<String>,
    /// `OneMerge.totalNumDocs()`: the sources' `maxDoc`s summed.
    pub total_docs: i64,
    /// `OneMerge.totalBytesSize()`: the sources' sizes on disk summed.
    pub total_bytes: i64,
    /// `OneMerge.maxNumSegments`: `-1` for a natural merge.
    pub max_num_segments: i32,
}

/// How long a finished merge took and how long it was held back
/// (`OneMergeProgress.getPauseTimes`), in milliseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MergeTimes {
    pub took_ms: u64,
    pub stopped_ms: u64,
    pub throttled_ms: u64,
}

/// `OpenSearchConcurrentMergeScheduler.beforeMerge`/`afterMerge` (and
/// `handleMergeException`): told about every merge the writer runs, on the
/// thread running it, before it starts and after it ends -- `failure` set
/// when it failed. Called without the writer's lock only for merges a merge
/// scheduler runs; an inline merge (`force_merge`) calls it under the lock,
/// so a listener must not call back into the writer.
pub trait MergeListener: Send + Sync {
    fn before_merge(&self, merge: &MergeEvent);
    fn after_merge(&self, merge: &MergeEvent, times: MergeTimes, failure: Option<&Error>);
}

static NEXT_MERGE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// One merge as its [`MergeListener`] hears it
/// (`OpenSearchConcurrentMergeScheduler.doMerge`): `before_merge` when
/// started, `after_merge` when finished -- with its pause times when it ran
/// under a rate limiter -- and, should the merge unwind instead, `after_merge`
/// as failed, so the listener never keeps a merge that ended.
pub(crate) struct MergeReport {
    listened: Option<(std::sync::Arc<dyn MergeListener>, MergeEvent)>,
    progress: Option<std::sync::Arc<crate::merge_rate_limiter::OneMergeProgress>>,
    started: std::time::Instant,
}

impl MergeReport {
    pub(crate) fn start(
        listened: Option<(std::sync::Arc<dyn MergeListener>, MergeEvent)>,
        progress: Option<std::sync::Arc<crate::merge_rate_limiter::OneMergeProgress>>,
    ) -> Self {
        if let Some((listener, event)) = &listened {
            listener.before_merge(event);
        }
        MergeReport {
            listened,
            progress,
            started: std::time::Instant::now(),
        }
    }

    /// The merge ended with `result`; an aborted merge is not a failure
    /// (`MergeAbortedException`).
    pub(crate) fn finish<T>(mut self, result: &Result<T>) {
        let failure = result
            .as_ref()
            .err()
            .filter(|e| !matches!(e, Error::MergeAborted));
        self.send(failure);
    }

    fn send(&mut self, failure: Option<&Error>) {
        let Some((listener, event)) = self.listened.take() else {
            return;
        };
        let ms = |ns: u64| ns / 1_000_000;
        let (stopped_ms, throttled_ms) = self.progress.as_ref().map_or((0, 0), |p| {
            (
                ms(p.pause_time_ns(crate::merge_rate_limiter::PauseReason::Stopped)),
                ms(p.pause_time_ns(crate::merge_rate_limiter::PauseReason::Paused)),
            )
        });
        let times = MergeTimes {
            took_ms: u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX),
            stopped_ms,
            throttled_ms,
        };
        listener.after_merge(&event, times, failure);
    }
}

impl Drop for MergeReport {
    fn drop(&mut self) {
        if self.listened.is_some() {
            self.send(Some(&Error::MergeThreadPanicked(
                "a merge panicked".to_string(),
            )));
        }
    }
}

pub struct IndexWriter<'d> {
    dir: &'d dyn Directory,
    /// What every segment this writer builds is built with -- see
    /// [`IndexingConfig`].
    cfg: std::sync::Arc<IndexingConfig>,
    /// The standard analyzer [`IndexWriter::writer_analyzer`] last built,
    /// with the `(position_increment_gap, offset_gap)` it was built for:
    /// built once, not once per document.
    default_analyzer: Option<(i32, i32, std::sync::Arc<Analyzer>)>,
    segment_infos: SegmentInfos,
    pending_docs: Vec<Document>,
    /// Segments already flushed to `dir` (by [`IndexWriter::flush`], including
    /// an automatic flush from [`IndexWriter::add_document`]) but not yet named
    /// by any `segments_N`. Real Lucene keeps these in `segmentInfos` itself,
    /// which is its *in-memory* view rather than the last commit; this facade
    /// documents `segment_infos` as "the last commit", so the not-yet-published
    /// tail lives here instead and is folded in by
    /// [`IndexWriter::prepare_commit`]. Their files are kept alive across that
    /// window by the deleter's non-commit checkpoint (Java's
    /// `IndexFileDeleter.checkpoint(infos, false)` / `lastFiles`), and dropped
    /// -- files and all -- by [`IndexWriter::rollback`].
    flushed_segments: Vec<SegmentCommitInfo>,
    /// Reference-counted reclamation of every file no live commit (and no
    /// not-yet-committed flush) names: real Lucene's `IndexFileDeleter`. See
    /// [`crate::index_file_deleter`].
    deleter: IndexFileDeleter<'d>,
    /// `LiveIndexWriterConfig.ramBufferSizeMB`, default
    /// [`DEFAULT_RAM_BUFFER_SIZE_MB`]. Negative means
    /// [`DISABLE_AUTO_FLUSH_MB`].
    ram_buffer_size_mb: f64,
    /// `LiveIndexWriterConfig.maxBufferedDocs`, default
    /// [`DEFAULT_MAX_BUFFERED_DOCS`] ([`DISABLE_AUTO_FLUSH`]).
    max_buffered_docs: i32,
    /// Running heap footprint of `pending_docs` + `pending_custom_freq_terms`,
    /// maintained incrementally so [`IndexWriter::add_document`] never walks
    /// the buffer. See [`IndexWriter::ram_bytes_used`] for exactly what this
    /// counts and how it relates to Java's `DocumentsWriterPerThread.bytesUsed`.
    ram_bytes_used: usize,
    merge_policy: Option<MergePolicyConfig>,
    /// See [`IndexWriter::set_pluggable_merge_policy`]; decides instead of
    /// `merge_policy` when set.
    pluggable_merge_policy: Option<std::sync::Arc<dyn merge_policy::MergePolicy>>,
    /// See [`IndexWriter::set_full_flush_merge_policy`].
    full_flush_merge_policy: Option<std::sync::Arc<dyn merge_policy::MergePolicy>>,
    /// See [`IndexWriter::set_merges_by_caller`].
    merges_by_caller: bool,
    /// Whether the merge being planned interleaves its sources
    /// ([`IndexWriter::set_shuffle_forced_merges`]): recorded in its
    /// diagnostics.
    merge_shuffled: bool,
    /// See [`IndexWriter::set_merge_listener`].
    merge_listener: Option<std::sync::Arc<dyn MergeListener>>,
    /// The merged-size cap forced merges use in place of the policy's
    /// `max_merged_segment_size`; see
    /// [`IndexWriter::set_forced_merge_max_segment_size`].
    forced_merge_max_segment_size: Option<u64>,
    /// `IndexWriter.mergeMaxNumSegments`: [`IndexWriter::force_merge`]'s
    /// argument while it runs, which its merges record as
    /// `OneMerge.maxNumSegments`; `-1` otherwise.
    merge_max_num_segments: i32,
    /// `LiveIndexWriterConfig.maxFullFlushMergeWaitMillis` -- see
    /// [`IndexWriter::set_max_full_flush_merge_wait_millis`].
    max_full_flush_merge_wait_millis: i64,
    /// `IndexWriterConfig.getCommitOnClose()`; see [`IndexWriter::close`].
    commit_on_close: bool,

    /// Per-pending-doc explicit `(term, custom_freq)` pairs for
    /// [`IndexingConfig::custom_freq_postings_field`], aligned 1:1 by index
    /// with `pending_docs` (index `i` here is doc ID `i` in the next flush,
    /// same convention
    /// [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment)
    /// already uses for `pending_docs` itself) -- kept in lockstep by
    /// [`IndexWriter::add_document`] (pushes an empty `Vec`) and
    /// [`IndexWriter::add_document_with_custom_freq_terms`] (pushes the
    /// caller's terms), and cleared together with `pending_docs` by
    /// `commit`/`prepare_commit`/`rollback`.
    pending_custom_freq_terms: Vec<Vec<(String, i32)>>,
    /// `Sorter.DocMap.newToOld` for the segment the current
    /// [`IndexWriter::flush`] just wrote, kept only for the duration of that
    /// call: `pending_sort_map.1[new_doc_id]` is the position that document
    /// held in the buffer before the sort.
    ///
    /// A segment-private delete packet's `docIDUpto` limits are recorded in
    /// the **pre-sort** doc space (they are buffer positions), so once the
    /// flush reorders the buffer those limits have to be compared against
    /// `new_to_old(doc)`, not against `doc`. Java keeps the same map for the
    /// same reason, on the pooled `ReadersAndUpdates.sortMap`
    /// (`IndexWriter.publishFlushedSegment` -> `FrozenBufferedUpdates`'s two
    /// `sortMap.newToOld(...) < limit` branches). Only the packet whose
    /// generation *equals* the segment's has a limit below
    /// `MAX_DOC_ID_UPTO`, and that packet is applied inside the very
    /// `flush()` that created the segment, so this port can scope the map to
    /// that call instead of pooling a reader for the writer's lifetime.
    ///
    /// A list, keyed by segment name, rather than one slot: when that apply
    /// fails, the private packet goes back into the stream with its
    /// pre-sort limits, and a later flush must not overwrite the map the
    /// retry still needs. Cleared once an apply drains the stream.
    pending_sort_map: Vec<(String, Vec<usize>)>,
    /// Set by [`IndexWriter::prepare_commit`], consumed by
    /// [`IndexWriter::finish_commit`] -- see [`IndexWriter::prepare_commit`]'s
    /// doc comment for exactly what "prepared" does and does not mean on
    /// this port's on-disk format (in short: every file the commit names is
    /// synced and `pending_segments_N` is written when this is `Some`, but
    /// the `segments_N` commit file that would make them discoverable to a
    /// fresh [`IndexWriter::open`]/reader has deliberately not been written
    /// yet).
    prepared_commit: Option<SegmentInfos>,
    /// `IndexWriter.changeCount`: bumped by every change that has to reach the
    /// next commit even though it leaves the segment list alone
    /// ([`IndexWriter::delete_all`], [`IndexWriter::set_live_commit_data`],
    /// `addIndexes`), and starting one above
    /// [`Self::last_commit_change_count`] on a directory with no commit yet, so
    /// a new index's first commit is always written. Changes to the segment
    /// list itself -- a flush, a merge, an applied delete or doc-values
    /// update -- are seen by comparing it with [`Self::rollback_segments`]
    /// instead (see [`IndexWriter::prepare_commit`]).
    change_count: u64,
    /// `IndexWriter.lastCommitChangeCount`: [`Self::change_count`] as of the
    /// last commit this writer published.
    last_commit_change_count: u64,
    /// `IndexWriter.pendingCommitChangeCount`: [`Self::change_count`] when the
    /// outstanding [`IndexWriter::prepare_commit`] ran, which
    /// [`IndexWriter::finish_commit`] records as committed -- so a change made
    /// between the two still counts for the commit after.
    pending_commit_change_count: u64,
    /// [`IndexWriter::try_delete_document`]s not yet written; see
    /// `try_modify`'s module documentation.
    pending_try_deletes: Vec<try_modify::PendingTryDeletes>,
    /// [`IndexWriter::try_update_doc_value`]s not yet written.
    pending_try_updates: Vec<try_modify::PendingTryUpdates>,
    /// The last [`IndexWriter::prepare_commit`] found nothing to commit and
    /// wrote nothing (Java's "skip startCommit(): no changes pending"), so
    /// [`IndexWriter::finish_commit`] has nothing to publish and succeeds
    /// without writing (Java's `finishCommit` with `pendingCommit == null`).
    prepared_nothing: bool,
    /// `DocumentsWriterDeleteQueue`: the sequence-number source, and the two
    /// buffers every delete/update lands in -- one for the segment currently
    /// being built, one for the segments already written. See
    /// [`crate::buffered_updates`].
    delete_queue: DeleteQueue,
    /// `BufferedUpdatesStream`: frozen delete packets waiting to be resolved
    /// against segments, each stamped with the generation that decides which
    /// segments it may touch.
    updates_stream: BufferedUpdatesStream,
    /// Near-real-time snapshots handed out and not yet returned, and their
    /// version -- see [`crate::nrt`].
    pub(crate) nrt: crate::nrt::NrtState,
    /// `IndexWriter.rollbackSegments`: the segment list of the last *durable*
    /// commit, captured at [`IndexWriter::open`] and refreshed every time a
    /// commit is installed. [`IndexWriter::rollback`] restores it
    /// (`SegmentInfos.rollbackSegmentInfos`).
    ///
    /// It became load-bearing when deletes became buffered: applying a buffered
    /// delete bumps a *committed* segment's `del_gen` in `segment_infos` and
    /// writes a new `.liv` that no commit yet names, so without this the
    /// in-memory view could survive a rollback pointing at a `.liv` the
    /// deleter's `refresh()` had just reclaimed.
    rollback_segments: Vec<SegmentCommitInfo>,
    /// Per-pending-doc vector values, aligned 1:1 by index with `pending_docs`
    /// (index `i` here is doc ID `i` in the next flush -- the same convention
    /// [`Self::pending_custom_freq_terms`] uses).
    ///
    /// Vectors do not live in [`Document`] because [`Document`] is this port's
    /// *stored-fields* document: everything in it is serialized into `.fdt`.
    /// Java's `Document` is a list of `IndexableField`s of which
    /// `KnnFloatVectorField` is one **non-stored** kind, so putting a vector
    /// there would store every embedding in the stored-fields file, which
    /// Lucene does not do.
    pending_vectors: Vec<Vec<DocumentVector>>,
    /// Each pending document's [`ExplicitFields`], aligned 1:1 with
    /// `pending_docs` like [`Self::pending_vectors`]; empty for a document
    /// added through a native [`Document`] entry point.
    pending_explicit: Vec<ExplicitFields>,
    /// The buffered explicit documents' postings, inverted as each was
    /// buffered -- see [`pending_postings`].
    pub(crate) pending_postings: pending_postings::PendingPostings,
    /// `SegmentCommitInfo.softDelCount` as last computed, per segment, keyed by
    /// the `(del_gen, doc_values_gen, field_infos_gen)` it was computed at --
    /// see [`IndexWriter::stamp_soft_delete_counts`].
    soft_delete_counts: std::collections::HashMap<String, (SoftDeleteKey, i32)>,
    /// See [`IndexWriter::set_soft_deletes_retention`].
    soft_deletes_retention: Option<SoftDeletesRetention>,
    /// See [`IndexWriter::set_max_docs`].
    max_docs: usize,
    /// Each segment's `maxDoc`, read off its `.si` once -- what
    /// [`IndexWriter::reserve_docs`] sums when the segment set changes.
    segment_max_docs: std::collections::HashMap<String, usize>,
    /// [`IndexWriter::reserve_docs`]' sum over the segments, and the segment
    /// set it was taken over: the commit generation (a commit's segments
    /// never change) and the flushed-but-uncommitted tail, which only grows
    /// between commits.
    segment_docs_total: Option<(i64, usize, Option<String>, usize)>,
    /// `SegmentInfo.setHasBlocks()` for the segment currently being buffered:
    /// set by any [`IndexWriter::add_documents`]/
    /// [`IndexWriter::update_documents`] call that buffers more than one
    /// document, cleared with the buffer at flush.
    pending_has_blocks: bool,
    /// Each segment's own Lucene version (its `.si`'s), by segment name, read
    /// once -- what [`IndexWriter::stamp_min_segment_version`] takes the
    /// minimum of.
    segment_versions: std::collections::HashMap<String, segment_infos::LuceneVersion>,
    /// `IndexWriter.writeLock`, obtained first thing in [`IndexWriter::open`]
    /// and held until the writer is dropped (Java's `close()`/`rollback()`
    /// release it). Declared last, so it is released after every other
    /// field -- the deleter included -- has been dropped, as Java releases it
    /// last in `rollbackInternal`.
    ///
    /// Java also wraps the directory in a `LockValidatingDirectoryWrapper`
    /// so every write checks the lock first. This writer borrows its
    /// directory (`&'d dyn Directory`, shared with its deleter), so it cannot
    /// own a wrapper around it; it checks at the points that make changes
    /// visible instead -- [`IndexWriter::flush`] and
    /// [`IndexWriter::prepare_commit`]/[`IndexWriter::finish_commit`] -- see
    /// [`IndexWriter::ensure_write_lock_valid`].
    write_lock: Box<dyn lucene_store::Lock>,
}

/// One field this writer has been opted into also indexing real postings
/// An owned token-payload supplier, as [`IndexWriter::set_payload_source`]
/// takes it and [`IndexWriter`] stores it -- this port's stand-in for a
/// `PayloadAttribute`-setting `TokenFilter`, see
/// [`crate::indexing_chain::PayloadSource`] for the borrowed form the invert
/// pass consumes.
/// `Send + Sync` is not decoration. `lucene-ffi`'s `WriterHandle` holds the
/// writer in a registry shared across threads, and it is `Send + Sync` by the
/// compiler's own derivation, not an `unsafe impl`. A caller-supplied closure
/// capturing an `Rc` or a `Cell` would break that derivation, in a crate where
/// `forbid(unsafe_code)` would not stop it. The bound keeps the derivation
/// true by construction, and costs nothing: every source here captures `Copy`
/// data or nothing.
pub type BoxedPayloadSource = Box<dyn Fn(&PayloadContext<'_>) -> Option<Vec<u8>> + Send + Sync>;

/// A [`BoxedPayloadSource`] once handed to a writer: shared, so the
/// [`IndexingConfig`] carrying it can be shared with a
/// [`crate::concurrent_writer::ConcurrentIndexWriter`]'s building threads.
type SharedPayloadSource =
    std::sync::Arc<dyn Fn(&PayloadContext<'_>) -> Option<Vec<u8>> + Send + Sync>;

/// The borrowed view of a [`BoxedPayloadSource`], which is what the invert
/// pass is handed.
type PayloadSourceRef<'a> = &'a (dyn Fn(&PayloadContext<'_>) -> Option<Vec<u8>> + Send + Sync);

/// A direct encoder's bytes or files already written by an installed sink.
struct TermVectorsOutput {
    bytes: Option<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    files: Vec<String>,
}

/// [`IndexWriter::build_norms_output`]'s result: the `.nvm`/`.nvd` bytes,
/// plus each norms field's per-document column as `(field number, norms by
/// doc id)` -- the input the postings writer's impacts need. See that
/// method's doc comment for why the columns come back at all.
struct NormsOutput {
    bytes: Option<(Vec<u8>, Vec<u8>)>,
    files: Vec<String>,
    impact_norms: Vec<(i32, Vec<i64>)>,
}

/// for, resolved once by [`IndexWriter::set_postings_field`]/
/// [`IndexWriter::add_postings_field`] against this writer's fixed `fields`
/// list -- `self.cfg.postings_fields` may hold any number of these at once (see
/// [`IndexWriter::set_postings_field`]'s doc comment for exactly how entries
/// are added/replaced, and [`IndexWriter::build_postings_output`] for how
/// they're all batched into one [`postings_writer::write_fields`] call).
#[derive(Debug, Clone)]
struct PostingsFieldConfig {
    name: String,
    field_number: i32,
    index_options: IndexOptions,
    /// `FieldInfo.storePayloads` for this field, copied at resolve time.
    ///
    /// This is what decides whether the segment gets a `.pay` payload-length
    /// stream, and it has to agree with the `STORE_PAYLOADS` bit the `.fnm`
    /// carries for the same field: real Lucene's `Lucene104PostingsReader`
    /// opens `.pay` whenever `fieldInfos.hasPayloads() || hasOffsets()` and
    /// frames every block's payload-length run off `fieldInfo.hasPayloads()`,
    /// so a `.fnm` claiming payloads over postings written without them is a
    /// segment Lucene either cannot open (no `.pay` at all) or reads garbage
    /// from (a `.pay` framed for offsets alone).
    store_payloads: bool,
}

/// The single field this writer has been opted into building real
/// `IndexOptions::DocsAndCustomFreqs` postings for, resolved once by
/// [`IndexWriter::set_custom_freq_postings_field`] against this writer's
/// fixed `fields` list -- see that method's doc comment for the exact
/// contract (in particular, the "at most one of `postings_fields`/this field"
/// mutual-exclusivity scope decision). Unlike [`PostingsFieldConfig`], only
/// `field_number` is kept: [`IndexWriter::build_custom_freq_postings_output`]
/// never needs the field's name (it never re-tokenizes any stored text, so
/// there is no `crate::indexing_chain::invert_documents`-style `(doc_id,
/// field_name, text)` triple to build).
#[derive(Debug, Clone)]
struct CustomFreqPostingsFieldConfig {
    field_number: i32,
}

/// One field this writer has been opted into also building real term
/// vectors for, resolved once by [`IndexWriter::set_term_vector_field`]/
/// [`IndexWriter::add_term_vector_field`] against this writer's fixed
/// `fields` list -- `self.cfg.term_vector_fields` may hold any number of these at
/// once, same "batch every entry into one write call per commit" shape as
/// [`PostingsFieldConfig`].
#[derive(Debug, Clone)]
struct TermVectorFieldConfig {
    name: String,
    field_number: i32,
    /// The field's `IndexOptions`, so the vector records the same axes the
    /// postings do.
    ///
    /// Real Lucene keeps these on the `FieldType`
    /// (`storeTermVectorPositions`/`Offsets`/`Payloads`) rather than on the
    /// `FieldInfo`, because the `.fnm` carries only the single
    /// `STORE_TERMVECTOR` bit -- which axes a vector holds is recorded in the
    /// `.tvd` chunk itself. This facade has no `FieldType`, so the field's own
    /// `IndexOptions` and `store_payloads` stand in. That mapping is not
    /// arbitrary: `CheckIndex.testTermVectors` cross-checks a field's vector
    /// against its postings occurrence by occurrence whenever both carry the
    /// same axis, so deriving the vector's axes from the postings' is what
    /// makes that check bite instead of silently skipping.
    index_options: IndexOptions,
    store_payloads: bool,
}

/// One field this writer has been opted into also building real NUMERIC doc
/// values for, resolved once by [`IndexWriter::set_doc_values_field`] against
/// this writer's fixed `fields` list -- same "resolve once, reuse every
/// commit" shape as [`PostingsFieldConfig`]/[`TermVectorFieldConfig`].
#[derive(Debug, Clone)]
struct DocValuesFieldConfig {
    name: String,
    field_number: i32,
    doc_values_type: DocValuesType,
    /// The field's `FieldInfo` asks for a doc-values skip index.
    skip_index: bool,
}

/// Every doc-values column a source segment currently has, resolved
/// per field to the **newest** generation rather than to the base
/// `.dvm`/`.dvd`.
///
/// A doc-values update rewrites one field's whole column into a new
/// generation (`field_updates`' module comment) and leaves the base
/// pair on disk, superseded; which generation is current is recorded
/// on the field's own `FieldInfo.docValuesGen` in the segment's
/// *newest* `.fnm`. Reading the base pair for such a field would
/// resurrect the pre-update values into the merged segment -- valid,
/// checksummed, `CheckIndex`-clean, wrong.
///
/// `columns` holds each distinct `.dvm`/`.dvd` pair once (a segment's
/// base pair covers every field no update has touched, so a per-field
/// read would copy it once per field), and `per_field` points each
/// field at the pair its current column lives in. The `.dvs` skip
/// index is not opened: nothing in the merge consults it, and the
/// merged one is rebuilt from the merged columns.
#[derive(Debug, Clone)]
struct SourceDocValueColumns {
    columns: Vec<(doc_values::DocValuesMeta, Vec<u8>)>,
    per_field: Vec<(i32, usize)>,
}

/// One field this writer computes and writes real norms (`.nvm`/`.nvd`, via
/// [`lucene_codecs::norms::write_fields`]) for, derived once per flush by
/// [`IndexingConfig::norms_field_configs`] from this writer's fixed `fields`
/// list -- same "resolve once, reuse every commit" shape as
/// [`PostingsFieldConfig`]/[`DocValuesFieldConfig`].
///
/// There is no opt-*in* knob, because Lucene has none: every indexed field
/// gets a norm column unless it says `omitNorms`, so the list is every field
/// with `index_options != IndexOptions::None` that
/// [`IndexWriter::omit_norms_field`] has not been called for, and it is a
/// `Vec`, not the single `Option` this carried before norms stopped being
/// opt-in.
struct NormsFieldConfig {
    name: String,
    field_number: i32,
    /// `FieldInvertState`'s index options: a `DOCS` field's norm counts
    /// distinct terms, not tokens.
    docs_only: bool,
}

/// One field's finished norm column, in whichever of
/// [`norms::NormsField`]'s two shapes `Lucene90NormsConsumer` would pick for
/// it: `Dense` when every document carries the field, `Sparse` otherwise.
/// This owns the values so [`IndexWriter::build_norms_output`] can build all
/// of them before borrowing them into one `norms::write_fields` call.
struct NormsColumn {
    /// The field's norm for **every** doc id, absent ones filled with `1`.
    ///
    /// One column, read twice: `norms::NormsField::Dense` borrows it as-is
    /// (only taken when no doc is absent, so no filler is on the wire), and
    /// `postings_writer::FieldNorms` needs exactly this dense-by-doc-id shape
    /// for its impacts. Keeping one `Vec<i64>` rather than a dense column
    /// plus a clone of it is what stops item 18 from costing 8 bytes per
    /// document per normed field across the window
    /// `build_and_write_segment`'s ordering exists to keep small.
    ///
    /// `1` for an absent doc is Java's own `advanceExact == false` fallback,
    /// and is never read: a document with no norm for the field has no
    /// posting for it either.
    dense: Vec<i64>,
    /// `Some` only when at least one doc is absent -- the sparse `.nvd` shape,
    /// which lists just the present docs.
    sparse: Option<Vec<(i32, i64)>>,
}

impl NormsColumn {
    /// The column for one field's per-document norms (`None`: the document
    /// has no value for the field).
    fn from_values(values: Vec<Option<i64>>) -> Self {
        let sparse = (!values.iter().all(Option::is_some)).then(|| {
            values
                .iter()
                .enumerate()
                .filter_map(|(doc, v)| v.map(|v| (doc as i32, v)))
                .collect()
        });
        NormsColumn {
            // The filler is the norm `1`, **not** `norm(0)`: an absent doc
            // has no norm at all, and `norm(0) == 0` is the one value
            // `CheckIndex.checkImpacts` rejects outright ("First impact had a
            // norm == 0"). A present-but-empty doc still gets its legitimate
            // explicit `0`, which is what Lucene writes for it.
            //
            // The filler never fires for a column with no absent doc, so the
            // `Dense` arm writes exactly the values computed.
            dense: values.into_iter().map(|v| v.unwrap_or(1)).collect(),
            sparse,
        }
    }
}

/// `IndexingChain.PerField.finish`'s norm: `0` for a field present without
/// tokens, `Similarity.computeNorm` otherwise -- which must not be `0` there.
/// A custom-frequency field's number and every buffered document's
/// `(term, custom_freq)` pairs, aligned by doc id.
type CustomFreqTerms<'a> = (i32, &'a [Vec<(String, i32)>]);

/// The `FieldInvertState` `IndexingChain.invert` leaves for a
/// `DOCS_AND_CUSTOM_FREQS` field fed one document's `(term, custom_freq)`
/// pairs, or `None` when the document supplies no term (it then has no such
/// field, so no norm).
///
/// Such a field is a "term-doc field" (`FieldInfo.isTermDocField`): each
/// token adds 1 to the length -- the custom frequency is *not* added, as it
/// is for a `TermFrequencyAttribute` on an ordinary field -- and a term may
/// appear once per document (`DuplicateTermException` otherwise), so the
/// length and the unique-term count are both the number of pairs, and
/// `maxTermFrequency` is the largest frequency. The pairs carry no
/// positions or offsets: each token is taken at an increment of 1 from
/// offset 0, so nothing overlaps, the last position is `pairs - 1` and the
/// offset stays 0.
fn custom_freq_invert_state(terms: &[(String, i32)]) -> Option<FieldInvertState> {
    if terms.is_empty() {
        return None;
    }
    let count = i32::try_from(terms.len()).unwrap_or(i32::MAX);
    // Each token adds 1 to the length; a repeated term counts once in
    // `uniqueTermCount` and its summed freq in `maxTermFrequency`. The terms
    // were validated when the document was buffered.
    let summed = summed_custom_freq_terms(terms).unwrap_or_default();
    Some(FieldInvertState {
        docs_only: false,
        position: count.saturating_sub(1),
        length: count,
        num_overlap: 0,
        offset: 0,
        max_term_frequency: summed.iter().map(|(_, f)| *f).max().unwrap_or(0),
        unique_term_count: i32::try_from(summed.len()).unwrap_or(i32::MAX),
        attribute_source: Some(crate::similarity::end_attributes(0, 0)),
    })
}

fn norm_value(
    similarity: &dyn NormSimilarity,
    field: &str,
    state: &FieldInvertState,
) -> Result<i64> {
    if state.length == 0 {
        return Ok(0);
    }
    match similarity.compute_norm(field, state) {
        0 => Err(Error::ZeroNorm(field.to_string())),
        norm => Ok(norm),
    }
}

/// One points field, resolved once by [`IndexWriter::add_points_field`] from
/// the writer's fixed field list -- the shape `.fnm` records for it.
#[derive(Debug, Clone)]
struct PointsFieldConfig {
    name: String,
    field_number: i32,
    num_dims: i32,
    num_index_dims: i32,
    bytes_per_dim: i32,
}

/// `BKDConfig.DEFAULT_MAX_POINTS_IN_LEAF_NODE`, which `Lucene90PointsFormat`
/// builds every tree with.
const MAX_POINTS_IN_LEAF_NODE: i32 = 512;

/// One field this writer has been opted into indexing vectors for, resolved
/// once by [`IndexWriter::set_vector_field`]/[`IndexWriter::add_vector_field`]
/// against this writer's fixed `fields` list. The dimension, encoding and
/// similarity are read straight off the field's [`FieldInfo`] -- the same
/// three values `.fnm` already carries and that
/// `Lucene99FlatVectorsReader.FieldEntry` cross-checks the `.vemf` against, so
/// they cannot be configured to disagree with the file that records them.
#[derive(Debug, Clone)]
struct VectorFieldConfig {
    name: String,
    field_number: i32,
    dimension: i32,
    encoding: VectorEncoding,
    similarity: VectorSimilarityFunction,
}

/// [`IndexWriter::build_points_output`]'s three files, plus the names of the
/// fields whose point dimensions this flush retains.
struct PointsOutput {
    bytes: Option<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    files: Vec<String>,
    written_fields: Vec<String>,
}

/// One document's value for a points field, in the packed form the BKD tree
/// stores: `NumericUtils.intToSortableBytes`/`longToSortableBytes` (and the
/// float/double `*ToSortable*` bit flips before them) for a one-dimensional
/// numeric field -- big-endian with the sign bit flipped, so unsigned byte
/// order is numeric order -- or the caller's packed bytes.
/// Appends the packed value to `out`, the field's contiguous value buffer.
fn encode_point(
    config: &PointsFieldConfig,
    doc_id: usize,
    value: &FieldValue,
    out: &mut Vec<u8>,
) -> Result<()> {
    let one_dim = |width: i32| config.num_dims == 1 && config.bytes_per_dim == width;
    let shape = |kind| {
        Error::PointValueShape(
            config.name.clone(),
            doc_id,
            kind,
            config.num_dims,
            config.bytes_per_dim,
        )
    };
    match value {
        FieldValue::Int(v) if one_dim(4) => {
            out.extend_from_slice(&((*v as u32) ^ 0x8000_0000).to_be_bytes())
        }
        FieldValue::Long(v) if one_dim(8) => {
            out.extend_from_slice(&((*v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes())
        }
        FieldValue::Float(v) if one_dim(4) => {
            // `NumericUtils.floatToSortableInt`, then `intToSortableBytes`.
            // `floatToSortableInt` goes through `Float.floatToIntBits`, which
            // collapses every NaN -- whatever its sign or payload -- to the
            // canonical `0x7fc00000`, so all NaNs sort above `+inf`.
            let v = if v.is_nan() { f32::NAN } else { *v };
            let bits = v.to_bits() as i32;
            let sortable = bits ^ ((bits >> 31) & 0x7fff_ffff);
            out.extend_from_slice(&((sortable as u32) ^ 0x8000_0000).to_be_bytes())
        }
        FieldValue::Double(v) if one_dim(8) => {
            // `Double.doubleToLongBits`: the same NaN canonicalisation.
            let v = if v.is_nan() { f64::NAN } else { *v };
            let bits = v.to_bits() as i64;
            let sortable = bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff);
            out.extend_from_slice(&((sortable as u64) ^ 0x8000_0000_0000_0000).to_be_bytes())
        }
        FieldValue::Binary(b)
            if i64::try_from(b.len()).ok()
                == i64::from(config.num_dims).checked_mul(i64::from(config.bytes_per_dim)) =>
        {
            out.extend_from_slice(b)
        }
        FieldValue::Int(_) => return Err(shape("Int")),
        FieldValue::Long(_) => return Err(shape("Long")),
        FieldValue::Float(_) => return Err(shape("Float")),
        FieldValue::Double(_) => return Err(shape("Double")),
        FieldValue::Binary(_) => return Err(shape("Binary")),
        FieldValue::String(_) | FieldValue::Shared(_) => return Err(shape("String")),
    }
    Ok(())
}

/// [`IndexWriter::build_vectors_output`]'s four files, plus the names of the
/// fields that actually got vectors in this flush (which is what
/// [`IndexingConfig::fields_with_per_field_attributes`] stamps, and what the rest
/// must have their `.fnm` `vector_dimension` zeroed for).
struct VectorsOutput {
    /// One per `PerFieldKnnVectorsFormat` instance: its codec suffix and its
    /// files, by extension.
    groups: Vec<VectorGroupFiles>,
    /// Every field that got vectors, with the instance it went to.
    written: Vec<WrittenVectorField>,
}

/// One KNN format instance's files for a flushed segment.
struct VectorGroupFiles {
    /// `<formatName>_<n>`.
    suffix: String,
    files: Vec<(&'static str, Vec<u8>)>,
}

/// A field a flush wrote vectors for, and the `PerFieldKnnVectorsFormat`
/// attributes its `.fnm` entry records.
#[derive(Debug, Clone)]
pub(crate) struct WrittenVectorField {
    pub(crate) name: String,
    pub(crate) format: &'static str,
    pub(crate) suffix: u32,
    /// Written by a direct `knnVectorsFormat()`: no per-field attributes.
    pub(crate) direct: bool,
}

/// One doc-values field's whole dense column, owned, so the multi-field
/// branch of [`IndexWriter::build_doc_values_output`] can hold every field's
/// values alive while it hands [`doc_values::write_dense_fields`] a slice of
/// borrowing [`doc_values::DenseField`]s. `DenseField` borrows; this owns.
enum DenseColumn {
    Numeric(i32, Vec<i64>),
    /// A NUMERIC column some documents have no value for -- the normal shape
    /// of an index-sort tier with missing values.
    SparseNumeric(i32, Vec<(i32, i64)>),
    Binary(i32, Vec<Vec<u8>>),
    Sorted(i32, Vec<Vec<u8>>),
    SortedNumeric(i32, Vec<Vec<i64>>),
    SortedSet(i32, Vec<Vec<Vec<u8>>>),
    SparseBinary(i32, Vec<(i32, Vec<u8>)>),
    SparseSorted(i32, Vec<(i32, Vec<u8>)>),
    SparseSortedNumeric(i32, Vec<(i32, Vec<i64>)>),
    SparseSortedSet(i32, Vec<(i32, Vec<Vec<u8>>)>),
    /// A SORTED_NUMERIC column, dense or sparse, in flat form.
    SortedNumericColumn(i32, doc_values::SortedNumericColumn),
    /// A SORTED_SET column, dense or sparse, with its dictionary built.
    SortedSetOrds(i32, doc_values::SortedSetOrdsColumn),
}

impl DenseColumn {
    fn as_dense_field(&self) -> doc_values::DenseField<'_> {
        match self {
            DenseColumn::Numeric(n, v) => doc_values::DenseField::Numeric(*n, v),
            DenseColumn::SparseNumeric(n, v) => doc_values::DenseField::SparseNumeric(*n, v),
            DenseColumn::Binary(n, v) => doc_values::DenseField::Binary(*n, v),
            DenseColumn::Sorted(n, v) => doc_values::DenseField::Sorted(*n, v),
            DenseColumn::SortedNumeric(n, v) => doc_values::DenseField::SortedNumeric(*n, v),
            DenseColumn::SortedSet(n, v) => doc_values::DenseField::SortedSet(*n, v),
            DenseColumn::SparseBinary(n, v) => doc_values::DenseField::SparseBinary(*n, v),
            DenseColumn::SparseSorted(n, v) => doc_values::DenseField::SparseSorted(*n, v),
            DenseColumn::SparseSortedNumeric(n, v) => {
                doc_values::DenseField::SparseSortedNumeric(*n, v)
            }
            DenseColumn::SparseSortedSet(n, v) => doc_values::DenseField::SparseSortedSet(*n, v),
            DenseColumn::SortedNumericColumn(n, c) => {
                doc_values::DenseField::SortedNumericColumn(*n, c)
            }
            DenseColumn::SortedSetOrds(n, c) => doc_values::DenseField::SortedSetOrds(*n, c),
        }
    }
}

/// One document's value for one vector field -- real Lucene's
/// `KnnFloatVectorField`/`KnnByteVectorField`.
///
/// The variant fixes the encoding, so a document can never supply a `f32`
/// vector for a `BYTE` field without [`IndexWriter::add_document_with_vectors`]
/// saying so ([`Error::VectorEncodingMismatch`]).
#[derive(Debug, Clone, PartialEq)]
pub enum VectorValue {
    /// `KnnFloatVectorField`: `dimension` finite `f32` components.
    /// Shared, not copied, with what a realtime buffer's graph reads.
    Float32(std::sync::Arc<Vec<f32>>),
    /// `KnnByteVectorField`: `dimension` signed bytes, held unsigned exactly
    /// as `.vec` stores them (`b as i8` is Java's view of the same byte).
    Byte(std::sync::Arc<Vec<u8>>),
}

impl VectorValue {
    fn encoding(&self) -> VectorEncoding {
        match self {
            VectorValue::Float32(_) => VectorEncoding::Float32,
            VectorValue::Byte(_) => VectorEncoding::Byte,
        }
    }

    fn len(&self) -> usize {
        match self {
            VectorValue::Float32(v) => v.len(),
            VectorValue::Byte(v) => v.len(),
        }
    }

    /// ARITH: every term is a `size_of` or the `capacity()` of an allocation
    /// this process currently holds, so the sum is bounded by the address
    /// space and cannot reach `usize::MAX`. The same proof covers
    /// [`document_ram_bytes`], [`custom_freq_terms_ram_bytes`] and
    /// `IndexWriter::ram_bytes_used`, which is reset to `0` on every flush
    /// and therefore only ever totals the *currently buffered* documents.
    #[allow(clippy::arithmetic_side_effects)]
    fn ram_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + match self {
                VectorValue::Float32(v) => v.capacity() * std::mem::size_of::<f32>(),
                VectorValue::Byte(v) => v.capacity(),
            }
    }
}

/// One `(field, vector)` pair on one document, as handed to
/// [`IndexWriter::add_document_with_vectors`].
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentVector {
    pub field_name: String,
    pub value: VectorValue,
}

impl DocumentVector {
    /// `new KnnFloatVectorField(name, vector)`.
    pub fn float32(field_name: impl Into<String>, vector: Vec<f32>) -> Self {
        DocumentVector {
            field_name: field_name.into(),
            value: VectorValue::Float32(std::sync::Arc::new(vector)),
        }
    }

    /// `new KnnByteVectorField(name, vector)`.
    pub fn byte(field_name: impl Into<String>, vector: Vec<u8>) -> Self {
        DocumentVector {
            field_name: field_name.into(),
            value: VectorValue::Byte(std::sync::Arc::new(vector)),
        }
    }

    /// ARITH: as [`VectorValue::ram_bytes`] -- a sum of live allocation
    /// sizes, bounded by the address space.
    #[allow(clippy::arithmetic_side_effects)]
    pub(crate) fn ram_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.field_name.capacity() + self.value.ram_bytes()
    }
}

/// Real Lucene routes postings and doc values through
/// `PerFieldPostingsFormat`/`PerFieldDocValuesFormat`, which do two things this
/// port must also do: they segregate each format's files under a
/// suffixed segment name, `<segment>_<format>_<suffix>.<ext>`, and they record
/// the format and suffix in each field's `.fnm` attributes so a reader can find
/// them again. Without both, real Lucene opens the segment, reports zero terms
/// and zero doc-values fields, and raises no error -- the fields simply have no
/// format registered against them.
///
/// This port's own reader resolves codec files by extension suffix, so it reads
/// either naming; only real Lucene notices a missing suffix.
/// `IndexWriterConfig.DISABLE_AUTO_FLUSH` (-1): the sentinel that turns one of
/// the two automatic-flush triggers off.
pub const DISABLE_AUTO_FLUSH: i32 = -1;
/// The same sentinel as a RAM-buffer size, matching Java's `ramBufferSizeMB !=
/// IndexWriterConfig.DISABLE_AUTO_FLUSH` comparison against the `int` constant.
pub const DISABLE_AUTO_FLUSH_MB: f64 = DISABLE_AUTO_FLUSH as f64;
/// `IndexWriterConfig.DEFAULT_RAM_BUFFER_SIZE_MB` (16.0).
pub const DEFAULT_RAM_BUFFER_SIZE_MB: f64 = 16.0;
/// `IndexWriterConfig.DEFAULT_MAX_BUFFERED_DOCS` ([`DISABLE_AUTO_FLUSH`]):
/// Lucene flushes on RAM by default, not on document count.
pub const DEFAULT_MAX_BUFFERED_DOCS: i32 = DISABLE_AUTO_FLUSH;
/// `IndexWriterConfig.DEFAULT_MAX_FULL_FLUSH_MERGE_WAIT_MILLIS` (500): how
/// long a commit or a near-real-time reader waits for merge-on-commit/refresh
/// merges -- see [`IndexWriter::set_max_full_flush_merge_wait_millis`].
pub const DEFAULT_MAX_FULL_FLUSH_MERGE_WAIT_MILLIS: i64 = 500;

pub const POSTINGS_FORMAT_NAME: &str = "Lucene104";
pub const DOC_VALUES_FORMAT_NAME: &str = "Lucene90";

/// [`lucene_codecs::codec::Lucene104Codec::before_write`] of the installed
/// codec, if any: the injected failure as a store error.
pub(crate) fn before_write(
    codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
    point: lucene_codecs::codec::WritePoint,
) -> std::result::Result<(), lucene_store::Error> {
    match codec {
        Some(codec) => codec.before_write(point).map_err(lucene_store::Error::Io),
        None => Ok(()),
    }
}

/// [`Lucene104Codec::check_doc_values`](lucene_codecs::codec::Lucene104Codec::check_doc_values)
/// of the writer's codec, if any.
pub(crate) fn check_doc_values(
    codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
    write: &lucene_codecs::codec::DocValuesWrite<'_>,
) -> std::result::Result<(), lucene_store::Error> {
    match codec {
        Some(codec) => codec
            .check_doc_values(write)
            .map_err(lucene_store::Error::Io),
        None => Ok(()),
    }
}

/// [`Lucene104Codec::segment_files_writer`](lucene_codecs::codec::Lucene104Codec::segment_files_writer)
/// of the writer's codec, if any: the names of the files it adds to `segment`.
pub(crate) fn write_codec_segment_files(
    codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
    segment: lucene_codecs::codec::SegmentWrite<'_>,
) -> std::result::Result<Vec<String>, lucene_store::Error> {
    match codec.and_then(|c| c.segment_files_writer()) {
        Some(writer) => writer.write(segment).map_err(lucene_store::Error::Io),
        None => Ok(Vec::new()),
    }
}

/// [`before_write`] around one doc-values instance: the consumer's open,
/// each of its fields, then `write` (its files), then its close.
pub(crate) fn with_doc_values_faults<T, E: From<lucene_store::Error>>(
    codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
    fields: usize,
    write: impl FnOnce() -> std::result::Result<T, E>,
) -> std::result::Result<T, E> {
    use lucene_codecs::codec::WritePoint;
    before_write(codec, WritePoint::DocValuesConsumer)?;
    for _ in 0..fields {
        before_write(codec, WritePoint::DocValuesField)?;
    }
    let out = write()?;
    before_write(codec, WritePoint::DocValuesClose)?;
    Ok(out)
}
/// `PerFieldKnnVectorsFormat`'s per-field format name for
/// `Lucene99HnswVectorsFormat` -- the value that goes in the `.fnm`'s
/// `PerFieldKnnVectorsFormat.format` attribute and, joined with
/// [`PER_FIELD_SUFFIX`], into every `.vec`/`.vemf`/`.vem`/`.vex` file name.
pub const KNN_VECTORS_FORMAT_NAME: &str = "Lucene99HnswVectorsFormat";
/// This writer routes every field to one format, so it only ever needs Lucene's
/// first suffix. This is the value that goes in the `.fnm` attribute; the
/// suffix each *file* carries is the wider [`per_field_codec_suffix`].
pub const PER_FIELD_SUFFIX: &str = "0";

/// A flush's postings: one written group per postings format
/// ([`per_field_postings::write`]).
pub(crate) type PostingsGroups = Vec<per_field_postings::GroupOutput>;

/// `PerFieldPostingsFormat.getSuffix`: the segment suffix a per-field format's
/// files are actually written with -- the format name and the suffix joined,
/// not the suffix alone. It appears both in the file name and, because Lucene
/// passes it straight through as `SegmentWriteState.segmentSuffix`, inside
/// every one of those files' index headers.
pub fn per_field_codec_suffix(format: &str) -> String {
    format!("{format}_{PER_FIELD_SUFFIX}")
}

/// The segment suffix of `segment`'s `.ext` file among `files` -- empty for
/// `_0.fdt`, a `CompressingCodec` format's for `_0_<suffix>.fdt` -- or `None`
/// when the segment has no such file.
pub(crate) fn format_suffix(files: &[String], segment: &str, ext: &str) -> Option<String> {
    files.iter().find_map(|file| {
        let stem = file
            .strip_prefix(segment)?
            .strip_suffix(ext)?
            .strip_suffix('.')?;
        match stem.strip_prefix('_') {
            Some(suffix) => Some(suffix.to_string()),
            None => stem.is_empty().then(String::new),
        }
    })
}

/// `IndexFileNames.segmentFileName(segmentName, segmentSuffix, ext)`:
/// `_0_<suffix>.ext`, or `_0.ext` for an empty suffix (a format that is a
/// codec's own, not a per-field instance).
pub fn segment_file_name(segment_name: &str, suffix: &str, ext: &str) -> String {
    if suffix.is_empty() {
        format!("{segment_name}.{ext}")
    } else {
        format!("{segment_name}_{suffix}.{ext}")
    }
}

/// The suffixed segment name a per-field format's files are written under.
pub fn per_field_segment(segment_name: &str, format: &str) -> String {
    format!("{segment_name}_{}", per_field_codec_suffix(format))
}

/// The per-field segment name and codec suffix a segment's postings files
/// carry, read off its own `.tim`: `_0_Lucene90_0.tim` -> (`_0_Lucene90_0`,
/// `Lucene90_0`). A segment an older Lucene wrote names them after *its*
/// default postings format, which is also the suffix inside every
/// header; this writer's own segments give `Lucene104_0`. Falls back to that
/// when the segment lists no `.tim` at all.
/// Whether a postings group's terms dictionary is not a block tree, so it
/// is read through `per_field_postings::open_groups`' rewrite even alone.
fn is_foreign_postings_group(suffix: &str) -> bool {
    per_field_postings::is_foreign_format(per_field_postings::format_of_suffix(suffix))
}

fn postings_file_base(seg_files: &[String], segment_name: &str) -> (String, String) {
    let prefix = format!("{segment_name}_");
    seg_files
        .iter()
        .filter_map(|f| f.strip_suffix(".tim"))
        .find_map(|stem| {
            if stem == segment_name {
                // A codec's direct postings format (`CheapBastard`): no suffix.
                return Some((stem.to_string(), String::new()));
            }
            stem.strip_prefix(&prefix)
                .map(|suffix| (stem.to_string(), suffix.to_string()))
        })
        .unwrap_or_else(|| {
            (
                per_field_segment(segment_name, POSTINGS_FORMAT_NAME),
                per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            )
        })
}

/// The codec suffixes of every current-format `PerFieldKnnVectorsFormat`
/// instance a segment holds, read off its `.vemf` files
/// (`_0_Lucene99HnswVectorsFormat_1.vemf` -> `Lucene99HnswVectorsFormat_1`),
/// in file-name order: `Lucene99HnswVectorsFormat`,
/// `Lucene104HnswScalarQuantizedVectorsFormat` and
/// `Lucene104ScalarQuantizedVectorsFormat` instances.
pub(crate) fn current_vector_suffixes(seg_files: &[String], segment_name: &str) -> Vec<String> {
    let prefix = format!("{segment_name}_");
    let mut out: Vec<String> = seg_files
        .iter()
        .filter_map(|f| {
            let stem = f.strip_suffix(".vemf")?;
            if stem == segment_name {
                // The format itself, not per-field (`ConfigurableMCodec`).
                return Some(String::new());
            }
            let suffix = stem.strip_prefix(&prefix)?;
            let (format, n) = suffix.rsplit_once('_')?;
            // A current format, or a registered wrapper of one (`Asserting`).
            (per_field_knn_vectors::for_name(format).is_some() && n.parse::<u32>().is_ok())
                .then(|| suffix.to_string())
        })
        .collect();
    out.sort();
    out
}

/// The retired 9.0-9.8 HNSW format a segment's vectors were written with,
/// and the codec suffix of its files, read off its `.vem` name
/// (`_0_lucene92HnswVectorsFormat_0.vem` -> `lucene92HnswVectorsFormat`,
/// `lucene92HnswVectorsFormat_0`); `None` for a current-format segment (it
/// has a `.vemf`) or one without vectors.
fn retired_vector_files(
    seg_files: &[String],
    segment_name: &str,
) -> Option<(RetiredHnswFormat, String)> {
    let prefix = format!("{segment_name}_");
    seg_files.iter().find_map(|f| {
        let suffix = f.strip_suffix(".vem")?.strip_prefix(&prefix)?;
        let (format, _) = suffix.rsplit_once('_')?;
        Some((RetiredHnswFormat::for_name(format)?, suffix.to_string()))
    })
}

/// Every per-field group a retired quantized vector format wrote in a
/// segment, with its codec suffix, read off the quantized metadata files
/// (`_0_Lucene99HnswScalarQuantizedVectorsFormat_0.vemq` ->
/// `Lucene99HnswScalarQuantizedVectorsFormat`,
/// `Lucene99HnswScalarQuantizedVectorsFormat_0`), in file-name order.
fn quantized_vector_files(
    seg_files: &[String],
    segment_name: &str,
) -> Vec<(QuantizedFormat, String)> {
    let prefix = format!("{segment_name}_");
    let mut out: Vec<(QuantizedFormat, String)> = seg_files
        .iter()
        .filter_map(|f| {
            let stem = f
                .strip_suffix(".vemq")
                .or_else(|| f.strip_suffix(".vemb"))?;
            let suffix = stem.strip_prefix(&prefix)?;
            let (format, _) = suffix.rsplit_once('_')?;
            Some((QuantizedFormat::for_name(format)?, suffix.to_string()))
        })
        .collect();
    out.sort_by(|a, b| a.1.cmp(&b.1));
    out
}

impl<'d> IndexWriter<'d> {
    /// Java: FieldInfos.FieldNumbers.verifySameSchema, before a configured flush.
    /// The opt-in selects this segment's vector schema, so it cannot change once
    /// a live segment has recorded the field as indexed.
    pub(crate) fn validate_configured_term_vector_schema(&self) -> Result<()> {
        if self.cfg.explicit
            || self
                .cfg
                .fields
                .iter()
                .all(|field| field.index_options == IndexOptions::None)
        {
            return Ok(());
        }
        for segment in self
            .segment_infos
            .segments
            .iter()
            .chain(&self.flushed_segments)
        {
            for previous in segment_field_infos(self.dir, segment)?.fields {
                if previous.index_options == IndexOptions::None {
                    continue;
                }
                let Some(declared) = self.cfg.fields.iter().find(|field| {
                    field.name == previous.name && field.index_options != IndexOptions::None
                }) else {
                    continue;
                };
                let configured = declared.store_term_vectors
                    && self
                        .cfg
                        .term_vector_fields
                        .iter()
                        .any(|field| field.field_number == declared.number);
                if previous.store_term_vectors != configured {
                    return Err(lucene_codecs::field_infos::Error::Inconsistent(
                        previous.name,
                        "cannot change store_term_vectors across indexed segments",
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    /// `IndexWriterConfig.getAnalyzer()` for the document API.
    pub(crate) fn writer_analyzer(&mut self) -> std::sync::Arc<Analyzer> {
        if let Some(a) = &self.cfg.document_analyzer {
            return std::sync::Arc::clone(a);
        }
        let gaps = (self.cfg.position_increment_gap, self.cfg.offset_gap);
        match &self.default_analyzer {
            Some((p, o, a)) if (*p, *o) == gaps => std::sync::Arc::clone(a),
            _ => {
                let a = std::sync::Arc::new(self.cfg.analyzer());
                self.default_analyzer = Some((gaps.0, gaps.1, std::sync::Arc::clone(&a)));
                a
            }
        }
    }

    /// `new IndexWriterConfig(analyzer)`: the analyzer the document API
    /// ([`crate::document`]) runs every tokenized text field through --
    /// `analyzer.tokenStream(field, text)`, with the analyzer's own
    /// per-field position-increment and offset gaps. Every attribute its
    /// chain sets is read as `IndexingChain` reads it: a `TokenFilter`'s
    /// `PayloadAttribute` becomes the occurrence's payload (and the
    /// segment's field `storePayloads`), its `TermFrequencyAttribute` the
    /// custom frequency. `None` returns to the standard analyzer with
    /// [`IndexWriter::set_position_increment_gap`]'s and
    /// [`IndexWriter::set_offset_gap`]'s gaps, which only that default
    /// reads. The native stored-text path keeps the standard analyzer.
    pub fn set_analyzer(&mut self, analyzer: Option<std::sync::Arc<Analyzer>>) {
        self.cfg_mut().document_analyzer = analyzer;
    }

    /// The configuration as it stands, shared: what the document API reads
    /// the similarity from while it updates the writer's term hash.
    pub(crate) fn config_snapshot(&self) -> std::sync::Arc<IndexingConfig> {
        std::sync::Arc::clone(&self.cfg)
    }
    /// The configuration, to change it: copied first if a
    /// [`crate::concurrent_writer::ConcurrentIndexWriter`] shares it, so a
    /// change never reaches a segment already being built with the old one.
    /// Holds every merge's writes to `mb_per_sec` MB/s -- what
    /// `ConcurrentMergeScheduler` does with a `MergeRateLimiter` per merge
    /// (`setMBPerSec`; this port has no auto-throttle adjusting it). `None`
    /// lifts the limit; `Some(0.0)` stops merges until it changes. A negative
    /// or NaN rate is Java's `IllegalArgumentException`.
    pub fn set_merge_mb_per_sec(&mut self, mb_per_sec: Option<f64>) -> Result<()> {
        if let Some(rate) = mb_per_sec {
            if rate < 0.0 || rate.is_nan() {
                return Err(Error::InvalidMergeRate(rate));
            }
        }
        self.cfg_mut().merge_mb_per_sec = mb_per_sec;
        Ok(())
    }

    /// The rate [`IndexWriter::set_merge_mb_per_sec`] set.
    pub fn merge_mb_per_sec(&self) -> Option<f64> {
        self.cfg.merge_mb_per_sec
    }

    /// `IndexWriterConfig.setReaderPooling(on)`: whether a segment opened to
    /// resolve deletes is kept for the next round ([`crate::reader_pool`]).
    /// On by default; turning it off drops what is pooled.
    pub fn set_reader_pooling(&mut self, on: bool) {
        self.cfg.reader_pool.set_enabled(on);
    }

    /// The writer's reader pool (`IndexWriter.getReaderPool`, for tests and
    /// diagnostics).
    pub fn reader_pool(&self) -> &crate::reader_pool::ReaderPool {
        &self.cfg.reader_pool
    }

    fn cfg_mut(&mut self) -> &mut IndexingConfig {
        std::sync::Arc::make_mut(&mut self.cfg)
    }

    /// `IndexWriterConfig.setUseCompoundFile`: whether segments flushed from
    /// here on are packed into a compound file (`IndexWriter.createCompoundFile`
    /// in `DocumentsWriterPerThread.sealFlushedSegment`).
    ///
    /// **Default `false`, where Java's is `true`**: every caller of this
    /// writer so far (the OpenSearch engine, the tools, the tests) was built
    /// against loose per-format files, and flipping the default silently
    /// would change the layout under all of them; a caller that wants Java's
    /// behaviour says so. A merged segment is compound when the merge
    /// policy's `useCompoundFile` says so (`noCFSRatio`,
    /// `maxCFSSegmentSizeMB`) -- the installed pluggable policy's; with none
    /// installed, `TieredMergePolicy`'s under
    /// [`IndexWriter::set_merge_compound_file_settings`] when set, else its
    /// defaults when this flag is on.
    pub fn set_use_compound_file(&mut self, use_compound_file: bool) {
        self.cfg_mut().use_compound_file = use_compound_file;
    }

    /// `IndexWriterConfig.getUseCompoundFile`.
    pub fn use_compound_file(&self) -> bool {
        self.cfg.use_compound_file
    }

    /// `MergePolicy.setNoCFSRatio`/`setMaxCFSSegmentSizeMB` for the merges
    /// no pluggable policy decides: with `Some`, a merged
    /// segment is compound exactly when `TieredMergePolicy.useCompoundFile`
    /// under these settings says so, whatever
    /// [`IndexWriter::use_compound_file`] (which, as in Java, then governs
    /// flushed segments only). `None` restores this writer's default.
    pub fn set_merge_compound_file_settings(
        &mut self,
        settings: Option<merge_policy::CompoundFileSettings>,
    ) {
        self.cfg_mut().merge_compound_file = settings;
    }

    /// See [`IndexWriter::set_merge_compound_file_settings`].
    pub fn merge_compound_file_settings(&self) -> Option<merge_policy::CompoundFileSettings> {
        self.cfg.merge_compound_file
    }

    /// OpenSearch's `ShuffleForcedMergePolicy` around the
    /// [`merge_policy::MergePolicyConfig`] policy: each forced merge (and
    /// forced-deletes merge) takes its sources sorted by name, then
    /// interleaved oldest with newest, the smaller of each pair first, and
    /// its segment's diagnostics carry `opensearch.shuffle_merge`.
    pub fn set_shuffle_forced_merges(&mut self, shuffle: bool) {
        self.cfg_mut().shuffle_forced_merges = shuffle;
    }

    /// See [`IndexWriter::set_shuffle_forced_merges`].
    pub fn shuffle_forced_merges(&self) -> bool {
        self.cfg.shuffle_forced_merges
    }

    /// Tells `listener` about every merge from now on (`None`: no one).
    pub fn set_merge_listener(&mut self, listener: Option<std::sync::Arc<dyn MergeListener>>) {
        self.merge_listener = listener;
    }

    /// See [`IndexWriter::set_merge_listener`].
    pub fn merge_listener(&self) -> Option<std::sync::Arc<dyn MergeListener>> {
        self.merge_listener.clone()
    }

    /// The [`MergeEvent`] for `plan`, when a listener wants one: `None` for an
    /// `addIndexes` merge, which no merge scheduler runs.
    pub(crate) fn merge_event(
        &self,
        plan: &MergePlan,
    ) -> Option<(std::sync::Arc<dyn MergeListener>, MergeEvent)> {
        let listener = self.merge_listener.clone()?;
        let merge::MergeKind::Merge {
            max_num_segments, ..
        } = plan.kind
        else {
            return None;
        };
        let mut total_docs = 0i64;
        let mut total_bytes = 0i64;
        for sci in &plan.sources {
            // A source whose `.si` cannot be read counts as empty here; the
            // merge itself reports the failure.
            if let Ok(si) = self.cfg.reader_pool.segment_info(self.dir, sci) {
                total_docs = total_docs.saturating_add(i64::from(si.doc_count));
                let bytes = merge_policy::commit_byte_size(self.dir, sci, &si);
                total_bytes = total_bytes.saturating_add(i64::try_from(bytes).unwrap_or(i64::MAX));
            }
        }
        let event = MergeEvent {
            id: NEXT_MERGE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            segments: plan.names.clone(),
            total_docs,
            total_bytes,
            max_num_segments,
        };
        Some((listener, event))
    }

    /// The merged-size cap [`IndexWriter::force_merge`] uses in place of the
    /// [`MergePolicyConfig`]'s `max_merged_segment_size`: OpenSearch's
    /// `OpenSearchTieredMergePolicy` forces with an unlimited one
    /// (`Some(u64::MAX)`). `None`, the default, keeps the policy's.
    /// `force_merge_deletes` always keeps the policy's, as there.
    pub fn set_forced_merge_max_segment_size(&mut self, cap: Option<u64>) {
        self.forced_merge_max_segment_size = cap;
    }

    /// `PerFieldPostingsFormat.getPostingsFormatForField` for one field: from
    /// the next flush and merge on, `field`'s postings go to `format`'s files
    /// (every field on the same format shares one set, under the suffix
    /// `Lucene104_<n>` its fields' `.fnm` attributes record); every other
    /// field stays on the default `Lucene104PostingsFormat()`.
    pub fn set_postings_format_for_field(
        &mut self,
        field: &str,
        format: per_field_postings::Lucene104PostingsFormat,
    ) {
        let formats = &mut self.cfg_mut().postings_formats;
        formats.retain(|(name, _)| name != field);
        formats.push((field.to_string(), format));
    }

    /// `PerFieldDocValuesFormat.getDocValuesFormatForField` for one field:
    /// from the next flush and merge on, `field`'s doc values go to
    /// `format`'s files. Every field on an equal format shares one set,
    /// numbered `Lucene90_<n>` in the order the flush (`IndexingChain`'s
    /// field hash) or the merge (field-number order) first reaches a field
    /// of it, which each field's `.fnm` attributes record; every other field
    /// stays on the default `Lucene90DocValuesFormat()`. A doc-values update
    /// is written as Java writes it: with the format its field's attribute
    /// names, looked up by name (the default instance), under its own
    /// suffix.
    /// `PerFieldKnnVectorsFormat.getKnnVectorsFormatForField` for one field:
    /// from the next flush and merge on, `field`'s vectors go to `format`'s
    /// files -- another `Lucene99HnswVectorsFormat(maxConn, beamWidth)`, a
    /// `Lucene104HnswScalarQuantizedVectorsFormat` or the graph-less
    /// `Lucene104ScalarQuantizedVectorsFormat`. Every field on an equal
    /// format shares one set, numbered `<formatName>_<n>` per format name in
    /// the order a flush's documents first carry a field of it (a merge: in
    /// field-number order), which each field's `.fnm` attributes record.
    /// Every other field stays on the default instance,
    /// `Lucene99HnswVectorsFormat` with [`IndexWriter::set_hnsw_parameters`]'
    /// graph parameters.
    /// `IndexWriterConfig.setCodec(codec)`: from the next flush and merge on,
    /// every field's postings, doc-values and KNN vectors format is the one
    /// `codec` answers (`Lucene104Codec`'s `get*FormatForField`), unless a
    /// per-field setter ([`IndexWriter::set_postings_format_for_field`],
    /// [`IndexWriter::set_doc_values_format_for_field`],
    /// [`IndexWriter::set_knn_vectors_format_for_field`]) routed that field,
    /// which wins. `None` returns to `Lucene104Codec()`'s defaults (with
    /// [`IndexWriter::set_hnsw_parameters`]' graph for vectors).
    /// An optional stored-fields writer factory is also installed for the next
    /// flush and merge. Its writer must produce the existing Lucene90 files;
    /// formats without a factory retain the native bulk encoding path.
    /// An optional same-wire live-docs writer handles subsequent deletions;
    /// the index layer retains generation advancement, sync and cleanup.
    ///
    /// The codec's name must be the one this port writes (`Lucene104`), or
    /// the name of a `FilterCodec` that keeps `Lucene104`'s file formats and
    /// is [registered](lucene_codecs::codecs::register) as such
    /// (`CheapBastard`, `Asserting`, ...): new segments then record that name,
    /// as Java's would, and `None` returns to `Lucene104`. Anything else is
    /// refused, as this writer has no other codec's formats.
    pub fn set_codec(
        &mut self,
        codec: Option<std::sync::Arc<dyn lucene_codecs::codec::Lucene104Codec>>,
    ) -> Result<()> {
        use lucene_codecs::codecs;
        let lucene104_like = |name: &str| {
            codecs::for_name(name).is_some_and(|c| {
                codecs::renamed(c, codecs::lucene104().name) == *codecs::lucene104()
            })
        };
        let name = match &codec {
            Some(c) if c.name() == self.cfg.codec_name => None,
            Some(c) if lucene104_like(&self.cfg.codec_name) && lucene104_like(c.name()) => {
                Some(c.name().to_string())
            }
            Some(c) => {
                return Err(Error::Explicit(format!(
                    "codec {:?} is not the codec this writer writes ({:?})",
                    c.name(),
                    self.cfg.codec_name
                )));
            }
            None if lucene104_like(&self.cfg.codec_name) => {
                Some(codecs::lucene104().name.to_string())
            }
            None => None,
        };
        if let Some(name) = name {
            self.cfg_mut().codec_name = name;
        }
        self.cfg_mut().codec = codec;
        Ok(())
    }

    pub fn set_knn_vectors_format_for_field(
        &mut self,
        field: &str,
        format: per_field_knn_vectors::KnnVectorsFormat,
    ) {
        let formats = &mut self.cfg_mut().knn_vectors_formats;
        formats.retain(|(name, _)| name != field);
        formats.push((field.to_string(), format));
    }

    pub fn set_doc_values_format_for_field(
        &mut self,
        field: &str,
        format: per_field_doc_values::Lucene90DocValuesFormat,
    ) {
        let formats = &mut self.cfg_mut().doc_values_formats;
        formats.retain(|(name, _)| name != field);
        formats.push((field.to_string(), format));
    }

    /// The doc-values format of every field that would otherwise get the
    /// default `Lucene90DocValuesFormat()` -- neither routed by
    /// [`Self::set_doc_values_format_for_field`] nor by the codec to anything
    /// else. Flushes and merges after this call write it; existing segments
    /// keep theirs until merged. [`Lucene90DocValuesFormat::rust_adaptive`]
    /// here is the plugin's `index.codec.adaptive_doc_values`; setting the
    /// default back restores stock-readable segments from then on.
    ///
    /// [`Lucene90DocValuesFormat::rust_adaptive`]: per_field_doc_values::Lucene90DocValuesFormat::rust_adaptive
    /// The Lucene release whose index formats segments and commits are written
    /// in: [`IndexFormat::Lucene105`] (the default) is readable by every
    /// Lucene 10.x, [`IndexFormat::Lucene106`] only by 10.6 and later. Takes
    /// effect for what is written from now on; segments already written keep
    /// the format they have (a reader opens both).
    ///
    /// [`IndexFormat::Lucene105`]: lucene_codecs::index_format::IndexFormat::Lucene105
    /// [`IndexFormat::Lucene106`]: lucene_codecs::index_format::IndexFormat::Lucene106
    pub fn set_index_format(&mut self, format: lucene_codecs::index_format::IndexFormat) {
        self.cfg_mut().index_format = format;
    }

    fn segment_infos_format_version(&self) -> i32 {
        match self.cfg.index_format {
            lucene_codecs::index_format::IndexFormat::Lucene105 => segment_infos::VERSION_86,
            lucene_codecs::index_format::IndexFormat::Lucene106 => segment_infos::VERSION_10_6,
        }
    }

    /// [`Self::set_index_format`]'s value.
    pub fn index_format(&self) -> lucene_codecs::index_format::IndexFormat {
        self.cfg.index_format
    }

    pub fn set_default_doc_values_format(
        &mut self,
        format: per_field_doc_values::Lucene90DocValuesFormat,
    ) {
        self.cfg_mut().default_doc_values_format = format;
    }

    /// `index.codec.adaptive_doc_values: auto`: the fields that get the
    /// default doc-values format get stock or adaptive doc values, one
    /// choice for all of them, made from the index's first documents
    /// ([`crate::adaptive_auto`]): stock until then, then adaptive when
    /// some numeric column is clustered in document order. The choice is
    /// kept in the commit's user data; a writer opened on a commit that
    /// holds one keeps it. Replaces [`Self::set_default_doc_values_format`]'s
    /// format while set.
    pub fn set_adaptive_doc_values_auto(&mut self) {
        let recorded = crate::adaptive_auto::AdaptiveAuto::recorded(&self.segment_infos.user_data);
        self.cfg_mut().adaptive_auto = Some(std::sync::Arc::new(
            crate::adaptive_auto::AdaptiveAuto::new(recorded),
        ));
    }

    /// The choice [`Self::set_adaptive_doc_values_auto`] made (`true`:
    /// adaptive), once made.
    pub fn adaptive_doc_values_choice(&self) -> Option<bool> {
        self.cfg.adaptive_auto.as_ref().and_then(|a| a.decided())
    }

    /// The postings format of every field that would otherwise get the
    /// default `Lucene104PostingsFormat()` -- neither routed by
    /// [`Self::set_postings_format_for_field`] nor by the codec to anything
    /// else. Flushes and merges after this call write it; existing segments
    /// keep theirs until merged.
    /// [`Lucene104PostingsFormat::rust_zstd`] here is the plugin's
    /// `index.codec.terms_compression: zstd`; setting the default back restores
    /// stock-readable segments from then on.
    ///
    /// [`Lucene104PostingsFormat::rust_zstd`]: per_field_postings::Lucene104PostingsFormat::rust_zstd
    pub fn set_default_postings_format(
        &mut self,
        format: per_field_postings::Lucene104PostingsFormat,
    ) {
        self.cfg_mut().default_postings_format = format;
    }

    /// Whether merged segments' `BEST_SPEED` stored fields are written with the
    /// high-compression LZ4 match finder ([`lucene_codecs::lz4_hc`]): the
    /// same LZ4 format, which stock Lucene reads, about 8-33% smaller and
    /// several times slower to compress. Flushes keep the fast compressor.
    /// A merged segment written this way records
    /// [`segment_writer::LZ4_HC_ATTRIBUTE`], so a later merge copies its
    /// chunks as they are instead of compressing them again; a source without
    /// it is recompressed once. The plugin's
    /// `index.codec.stored_fields.recompress_on_merge`.
    pub fn set_merge_lz4_high_compression(&mut self, on: bool) {
        self.cfg_mut().merge_lz4_high_compression = on;
    }

    /// With [`Self::set_merge_lz4_high_compression`] on, keeps the explicitly
    /// forced merges (`force_merge` with a segment target, which `_upgrade`
    /// also uses) on the fast compressor: they copy the sources' chunks
    /// instead of recompressing every one, as stock does, at the cost of the
    /// 8-33% the match finder would save. The plugin's
    /// `index.codec.stored_fields.recompress_on_merge: non_force`.
    pub fn set_merge_lz4_high_compression_non_force(&mut self, on: bool) {
        self.cfg_mut().merge_lz4_hc_non_force = on;
    }

    /// Clusters the vectors of IVF fields ([`KnnVectorsFormat::Ivf`]) per
    /// partition of an index sorted by a partition field (the index sort's
    /// first field), at flush and merge: each partition of at least
    /// `min_vectors` vectors on its own, and the smaller ones between them
    /// together; `None` (the default) clusters each field as a whole. A
    /// search filtered to one partition then probes only its clusters: on
    /// 8M vectors, tenants of 400k and 1.5M searched about 4x and 1.5x
    /// faster at the same or higher recall, with `min_vectors` 20,000,
    /// while unfiltered searches took 2.3x as long (`docs/benchmarks/ivf.md`,
    /// "Clusters per partition"). No effect without an index sort.
    ///
    /// [`KnnVectorsFormat::Ivf`]: per_field_knn_vectors::KnnVectorsFormat::Ivf
    pub fn set_ivf_partition_clusters(&mut self, min_vectors: Option<usize>) {
        self.cfg_mut().ivf_partition_clusters = min_vectors;
    }

    /// The least vectors a partition needs to be clustered on its own; see
    /// [`IndexWriter::set_ivf_partition_clusters`].
    pub fn ivf_partition_clusters(&self) -> Option<usize> {
        self.cfg.ivf_partition_clusters
    }

    /// The least `.fdt` bytes, summed over a merge's sources, for which the
    /// merged segment of a paged zstd codec
    /// ([`lucene_codecs::stored_fields::Mode::ZstdPaged`]) gets a trained
    /// dictionary. Defaults to
    /// [`lucene_codecs::stored_fields::ZSTD_PAGED_DICTIONARY_MIN_BYTES`];
    /// `u64::MAX` never trains one.
    pub fn set_zstd_dictionary_min_bytes(&mut self, bytes: u64) {
        self.cfg_mut().zstd_dictionary_min_bytes = bytes;
    }

    /// Buffers `node` for the segments already published only -- a
    /// [`crate::concurrent_writer::ConcurrentIndexWriter`] keeps each
    /// in-RAM buffer's own share of it with that buffer.
    pub(crate) fn buffer_global_delete(&mut self, node: &DeleteNode) {
        let global = self.delete_queue.global_mut();
        buffer_node(global, node, crate::buffered_updates::MAX_DOC_ID_UPTO);
    }

    /// The sequence number the next operation would take -- where a
    /// [`crate::concurrent_writer::ConcurrentIndexWriter`] wrapping this
    /// writer continues numbering.
    pub(crate) fn next_sequence_number_peek(&self) -> SeqNo {
        self.delete_queue.last_sequence_number().saturating_add(1)
    }

    /// Moves the sequence numbers on so the next one is at least `next` -- a
    /// [`crate::concurrent_writer::ConcurrentIndexWriter`] handing numbering
    /// back, so none is ever issued twice.
    pub(crate) fn advance_sequence_numbers_to(&mut self, next: SeqNo) {
        let jump = next.saturating_sub(self.next_sequence_number_peek());
        self.delete_queue.skip_sequence_numbers(jump);
    }

    /// A ticket that carries no segment, only what the published segments
    /// owe as of now -- `DocumentsWriterFlushQueue`'s global-only
    /// `FlushTicket`, which a full flush adds after the last buffer so the
    /// deletes it covers publish in order with its segments.
    pub(crate) fn begin_deletes_ticket(
        &mut self,
    ) -> Option<crate::buffered_updates::FrozenBufferedUpdates> {
        self.delete_queue.freeze_global_buffer()
    }

    /// Publishes a [`Self::begin_deletes_ticket`] packet in its turn and
    /// applies it.
    pub(crate) fn publish_deletes_ticket(
        &mut self,
        packet: Option<crate::buffered_updates::FrozenBufferedUpdates>,
    ) -> Result<()> {
        self.write_pending_try_changes()?;
        if let Some(packet) = packet {
            self.updates_stream.push(packet);
        }
        self.apply_pending_packets()
    }

    /// Applies every packet in the stream (none frozen from the global
    /// buffer) and checkpoints the files that wrote -- what a ticket whose
    /// segment could not be built still owes, after
    /// [`Self::abandon_segment`].
    pub(crate) fn apply_pending_packets(&mut self) -> Result<()> {
        self.write_pending_try_changes()?;
        self.apply_stream_packets()?;
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
        Ok(())
    }

    /// Hands merging to the caller, or takes it back: while on, a commit runs
    /// no merges of its own and a finished merge is published in memory only
    /// -- `IndexWriter.commitMerge`, which checkpoints and leaves durability
    /// to the next `commit()` -- rather than written as a commit of its own.
    /// A [`crate::concurrent_writer::ConcurrentIndexWriter`] merges on its
    /// merge thread while other threads index, and a merge that wrote
    /// `segments_N` would make whatever they had published durable without
    /// the rest of it.
    pub(crate) fn set_merges_by_caller(&mut self, on: bool) {
        self.merges_by_caller = on;
    }

    /// The natural merge policy's settings, `None` when merges are off.
    pub fn merge_policy(&self) -> Option<&MergePolicyConfig> {
        self.merge_policy.as_ref()
    }

    /// `IndexFileDeleter.deleteNewFiles`: drop files an abandoned build or
    /// merge wrote and nothing references.
    pub(crate) fn delete_new_files(&mut self, files: &[String]) -> Result<()> {
        self.write_pending_try_changes()?;
        self.deleter.delete_new_files(files)?;
        Ok(())
    }

    /// The first merge `config` proposes, the segments in `merging` counted
    /// in the index total but never candidates -- `MergePolicy.findMerges`
    /// with `IndexWriter.getMergingSegments()` as its context.
    #[cfg(test)]
    pub(crate) fn next_merge(
        &self,
        config: &MergePolicyConfig,
        merging: &std::collections::HashSet<String>,
    ) -> Result<Option<Vec<String>>> {
        let stats = self.segment_stats()?;
        Ok(self
            .per_partition_stats(&stats, |s| {
                merge_policy::find_merges_excluding(s, merging, config)
            })?
            .into_iter()
            .next())
    }

    /// `TieredMergePolicy.findMerges` over every segment, the ones in
    /// `merging` included as Java's `MergeContext.getMergingSegments` counts
    /// them (their bytes in the index total, never as candidates): each
    /// proposed merge with its `OneMerge.estimatedMergeBytes`, the sources'
    /// sizes pro-rated by their live documents.
    pub(crate) fn find_merges_sized(
        &self,
        config: &MergePolicyConfig,
        merging: &std::collections::HashSet<String>,
    ) -> Result<Vec<(Vec<String>, u64)>> {
        let stats = self.segment_stats()?;
        let merges = self.per_partition_stats(&stats, |s| {
            merge_policy::find_merges_excluding(s, merging, config)
        })?;
        Ok(merges
            .into_iter()
            .map(|names| {
                let estimated = stats
                    .iter()
                    .filter(|s| names.contains(&s.name))
                    .map(|s| live_pro_rated_bytes(s.size_bytes, s.doc_count, s.del_count))
                    .fold(0u64, u64::saturating_add);
                (names, estimated)
            })
            .collect())
    }

    /// The configuration every segment is built with, shared.
    pub(crate) fn shared_config(&self) -> std::sync::Arc<IndexingConfig> {
        std::sync::Arc::clone(&self.cfg)
    }

    /// This writer's pending documents, as the buffer a segment is built
    /// from, with their buffered postings.
    fn document_buffer<'a>(
        &'a self,
        postings: &'a pending_postings::PendingPostings,
    ) -> DocumentBuffer<'a> {
        DocumentBuffer {
            docs: &self.pending_docs,
            custom_freq_terms: &self.pending_custom_freq_terms,
            vectors: &self.pending_vectors,
            explicit: ExplicitDocs::Fields(&self.pending_explicit),
            postings: Some(postings),
            old_to_new: None,
            has_blocks: self.pending_has_blocks,
            stored: None,
            limit: None,
            partition: None,
            graphs: &[],
        }
    }

    /// Opens a writer over `dir`: resumes the latest existing commit if one is
    /// present, or starts a brand-new, empty index otherwise. `fields`
    /// describes every field [`IndexWriter::add_document`]/
    /// [`IndexWriter::update_document`] documents may use (this facade has no
    /// per-call schema reconciliation the way [`crate::merge`] does across
    /// sources -- every document flushed through one writer shares one fixed
    /// field list, same as every existing caller of
    /// [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment)).
    /// `codec_name`/`lucene_version` are recorded on every segment this writer
    /// flushes, same meaning as the identically-named parameters on
    /// [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment).
    pub fn open(
        dir: &'d dyn Directory,
        fields: Vec<FieldInfo>,
        codec_name: impl Into<String>,
        lucene_version: LuceneVersion,
    ) -> Result<Self> {
        Self::open_inner(dir, fields, codec_name, lucene_version, None, false)
    }

    /// [`IndexWriter::open`] under a pluggable deletion policy --
    /// `IndexWriterConfig.setIndexDeletionPolicy` before `new IndexWriter`:
    /// the policy's `on_init` sees every commit already in `dir` (so a
    /// [`crate::deletion_policy::KeepLastNCommitsDeletionPolicy`] or a
    /// [`crate::deletion_policy::PersistentSnapshotDeletionPolicy`] keeps what
    /// it wants of them), and its `on_commit` runs after each commit.
    pub fn open_with_deletion_policy(
        dir: &'d dyn Directory,
        fields: Vec<FieldInfo>,
        codec_name: impl Into<String>,
        lucene_version: LuceneVersion,
        policy: std::sync::Arc<dyn crate::deletion_policy::IndexDeletionPolicy>,
    ) -> Result<Self> {
        Self::open_inner(dir, fields, codec_name, lucene_version, Some(policy), false)
    }

    /// Opens with Java's default field configuration: no parent or soft-deletes
    /// field, and consistent schemas across the current commit's segments.
    /// Unlike [`Self::open`], this validates those defaults before deleting any
    /// files. Use [`Self::open`] when configuring field roles through setters.
    /// `None` selects the default keep-only-last-commit deletion policy.
    ///
    /// # Errors
    /// An existing field requires a parent/soft-deletes configuration, segment
    /// schemas disagree, or index metadata cannot be read.
    pub fn open_with_default_config(
        dir: &'d dyn Directory,
        codec_name: impl Into<String>,
        lucene_version: LuceneVersion,
        policy: Option<std::sync::Arc<dyn crate::deletion_policy::IndexDeletionPolicy>>,
    ) -> Result<Self> {
        Self::open_inner(dir, Vec::new(), codec_name, lucene_version, policy, true)
    }

    fn open_inner(
        dir: &'d dyn Directory,
        fields: Vec<FieldInfo>,
        codec_name: impl Into<String>,
        lucene_version: LuceneVersion,
        policy: Option<std::sync::Arc<dyn crate::deletion_policy::IndexDeletionPolicy>>,
        validate_default_fields: bool,
    ) -> Result<Self> {
        // `writeLock = d.obtainLock(WRITE_LOCK_NAME)`: before anything reads
        // the index, and held by the writer from here on. A held lock is
        // `LockObtainFailedException` (`lucene_store::Error::LockObtainFailed`).
        // Every early return below drops it, which releases it -- Java's
        // `IOUtils.closeWhileHandlingException(writeLock)` on a failed
        // constructor.
        let write_lock = dir.obtain_lock(WRITE_LOCK_NAME)?;
        // `FieldInfos(FieldInfo[])` over `FieldInfo`'s own constructor: every
        // field is coerced (the three indexed-only flags off a non-indexed
        // field) and then checked, per field and across fields, before this
        // writer will accept it. Java gets this for free -- there is no way to
        // hold a `FieldInfo` that has not been through its constructor.
        let fields = lucene_codecs::field_infos::FieldInfos::new(fields)?.fields;
        let files = dir.list_all()?;
        let generation = lucene_store::directory::last_commit_generation(&files)?;
        let mut segment_infos = if generation < 0 {
            empty_segment_infos(lucene_version)
        } else {
            segment_infos::read_latest(dir)?
        };

        if validate_default_fields {
            validate_default_field_config(dir, &segment_infos)?;
        }

        // Real `IndexWriter`'s constructor builds its `IndexFileDeleter` here,
        // and that construction is what reclaims whatever a previous session
        // leaked: a `pending_segments_N` from a prepare that never finished, the
        // segment files of a flush that was never committed, and every commit
        // generation the deletion policy no longer wants.
        let deleter = match policy {
            None => {
                IndexFileDeleter::open(dir, &segment_infos, DeletionPolicy::KeepOnlyLastCommit)?
            }
            Some(policy) => IndexFileDeleter::open_with_policy(dir, &segment_infos, policy)?,
        };

        // `IndexFileDeleter.inflateGens`, applied *after* the deleter has
        // refcounted the real current commit (so it still recognises it), using
        // the pre-deletion listing: a name that was just reclaimed must still
        // push the counters past it, so a crashed session's generation or
        // segment name can never be handed out a second time.
        // Also pushes each segment's `next_write_*_gen` past every generation
        // that segment's files show on disk, so a `.liv`/`.dvd` written from
        // here never lands on a name a crashed session may already have used.
        IndexFileDeleter::inflate_gens(&files, &mut segment_infos);

        // `IndexWriter`'s constructor: `rollbackSegments =
        // segmentInfos.createBackupSegmentInfos()`.
        let rollback_segments = segment_infos.segments.clone();
        let soft_delete_counts = rollback_segments
            .iter()
            .map(|s| {
                (
                    s.segment_name.clone(),
                    (soft_delete_key(s), s.soft_del_count),
                )
            })
            .collect();

        Ok(IndexWriter {
            dir,
            default_analyzer: None,
            cfg: std::sync::Arc::new(IndexingConfig {
                field_lookup: FieldLookup::build(&fields),
                fields,
                explicit_field_numbers: Default::default(),
                codec_name: codec_name.into(),
                lucene_version,
                postings_fields: Vec::new(),
                position_increment_gap: 0,
                offset_gap: 1,
                document_analyzer: None,
                payload_source: None,
                custom_freq_postings_field: None,
                term_vector_fields: Vec::new(),
                doc_values_fields: Vec::new(),
                index_sort: None,
                parent_field: None,
                explicit: false,
                vector_fields: Vec::new(),
                points_fields: Vec::new(),
                hnsw_m: hnsw::DEFAULT_MAX_CONN,
                hnsw_beam_width: hnsw::DEFAULT_BEAM_WIDTH,
                merge_mb_per_sec: None,
                similarity: None,
                use_compound_file: false,
                merge_compound_file: None,
                shuffle_forced_merges: false,
                segment_partitioning: None,
                segment_list_attributes: Vec::new(),
                merge_union_attributes: Vec::new(),
                segment_attributes: Vec::new(),
                postings_formats: Vec::new(),
                doc_values_formats: Vec::new(),
                default_doc_values_format: Default::default(),
                index_format: Default::default(),
                adaptive_auto: None,
                default_postings_format: Default::default(),
                merge_lz4_high_compression: false,
                merge_lz4_hc_non_force: false,
                ivf_partition_clusters: None,
                zstd_dictionary_min_bytes:
                    lucene_codecs::stored_fields::ZSTD_PAGED_DICTIONARY_MIN_BYTES,
                knn_vectors_formats: Vec::new(),
                codec: None,
                reader_pool: std::sync::Arc::default(),
                ivf_clusters: std::sync::Arc::default(),
                merged_segment_warmer: None,
            }),
            segment_infos,
            pending_docs: Vec::new(),
            flushed_segments: Vec::new(),
            deleter,
            ram_buffer_size_mb: DEFAULT_RAM_BUFFER_SIZE_MB,
            max_buffered_docs: DEFAULT_MAX_BUFFERED_DOCS,
            ram_bytes_used: 0,
            merge_policy: None,
            pluggable_merge_policy: None,
            full_flush_merge_policy: None,
            merges_by_caller: false,
            merge_max_num_segments: -1,
            merge_shuffled: false,
            merge_listener: None,
            forced_merge_max_segment_size: None,
            max_full_flush_merge_wait_millis: DEFAULT_MAX_FULL_FLUSH_MERGE_WAIT_MILLIS,
            commit_on_close: true,
            pending_custom_freq_terms: Vec::new(),
            pending_sort_map: Vec::new(),
            prepared_commit: None,
            // `IndexWriter`'s constructor calls `changed()` for an index it
            // creates, so the first commit of a new index is always written.
            change_count: u64::from(generation < 0),
            last_commit_change_count: 0,
            pending_commit_change_count: 0,
            pending_try_deletes: Vec::new(),
            pending_try_updates: Vec::new(),
            prepared_nothing: false,
            delete_queue: DeleteQueue::new(),
            updates_stream: BufferedUpdatesStream::new(),
            nrt: crate::nrt::NrtState::new(),
            rollback_segments,
            pending_vectors: Vec::new(),
            pending_explicit: Vec::new(),
            pending_postings: pending_postings::PendingPostings::default(),
            soft_delete_counts,
            soft_deletes_retention: None,
            max_docs: MAX_DOCS,
            segment_max_docs: std::collections::HashMap::new(),
            segment_docs_total: None,
            pending_has_blocks: false,
            segment_versions: std::collections::HashMap::new(),
            write_lock,
        })
    }

    /// `writeLock.ensureValid()`: fails with
    /// [`lucene_store::Error::AlreadyClosed`] once this writer's
    /// `write.lock` has been lost (its file deleted or replaced, or the lock
    /// released behind the writer's back). Checked before every flush and
    /// commit step, so a writer that lost its lock stops before it
    /// publishes anything into an index another writer may now own.
    pub fn ensure_write_lock_valid(&self) -> Result<()> {
        self.write_lock.ensure_valid()?;
        Ok(())
    }

    /// Opts this writer into also building and writing real postings
    /// (`.doc`/`.tim`/`.tip`/`.tmd`, via
    /// [`postings_writer::write_fields`]) for one field of every segment
    /// [`IndexWriter::commit`] flushes from here on -- mirroring real
    /// Lucene's per-field `FieldType.setIndexOptions`.
    ///
    /// `Some(field_name)` **replaces** this writer's entire postings-field
    /// list with just `field_name` (matching this method's historical
    /// "reassign, don't accumulate" semantics) -- to index postings for
    /// *more than one* field in the same commit, call
    /// [`IndexWriter::add_postings_field`] afterward for each additional
    /// field, or call it repeatedly on its own (`set_postings_field` is only
    /// needed to establish or replace the first one / to disable postings
    /// entirely). `field_name` is looked up in this writer's fixed `fields`
    /// list (from [`IndexWriter::open`]) and requires its `index_options` to
    /// already be `IndexOptions::Docs`, `IndexOptions::DocsAndFreqs`,
    /// `IndexOptions::DocsAndFreqsAndPositions`, or
    /// `IndexOptions::DocsAndFreqsAndPositionsAndOffsets` (an `Err`
    /// otherwise) -- the same analyzed-field-text convention real Lucene's
    /// own `FieldType` uses to mark a field indexable. For the latter two,
    /// [`IndexWriter::build_postings_output`] also feeds this writer's
    /// [`crate::indexing_chain::invert_documents`] pass's per-occurrence
    /// positions (and, for
    /// `DocsAndFreqsAndPositionsAndOffsets`, offsets) into
    /// [`postings_writer::write_fields`], producing a real `.pos` file (and,
    /// for offsets, `.pay`) a `PhraseQuery` can search against -- payloads
    /// are not wired up here yet (`has_payloads` is always `false`; see
    /// `docs/parity.md`). `None` (the default a freshly [`IndexWriter::open`]ed
    /// writer starts with) turns this back off entirely -- `commit()` then
    /// behaves exactly as it did before this feature existed (stored fields
    /// only, matching every pre-existing caller).
    ///
    /// Only [`FieldValue::String`] values contribute indexable text for an
    /// opted-in field -- a document with no value, or a non-`String` value,
    /// for that field contributes no postings for that document (same "best
    /// effort per document" shape [`crate::indexing_chain::invert_documents`]
    /// already has for a missing `(doc_id, field, text)` triple).
    pub fn set_postings_field(&mut self, field_name: Option<&str>) -> Result<()> {
        if field_name.is_some() && self.cfg.custom_freq_postings_field.is_some() {
            return Err(Error::PostingsAndCustomFreqPostingsMutuallyExclusive(
                "set_postings_field",
            ));
        }
        self.cfg_mut().postings_fields = match field_name {
            None => Vec::new(),
            Some(name) => vec![Self::resolve_postings_field(&self.cfg.fields, name)?],
        };
        Ok(())
    }

    /// `IndexWriterConfig.setSimilarity`: the similarity every segment this
    /// writer flushes computes its norms with (`Similarity.computeNorm`, per
    /// document and field, in `IndexingChain.PerField.finish`). `None`
    /// restores Lucene's default, `BM25Similarity`.
    ///
    /// Takes the index-time half of the contract
    /// ([`crate::similarity::NormSimilarity`]); every
    /// `lucene_search::similarities::Similarity` is one, so the object a
    /// searcher scores with configures the writer too. Merges copy norms
    /// as they are, as Lucene's do: a similarity applies to the documents
    /// flushed after it is set.
    pub fn set_similarity(&mut self, similarity: Option<std::sync::Arc<dyn NormSimilarity>>) {
        self.cfg_mut().similarity = similarity;
    }

    /// `Analyzer.getPositionIncrementGap(String)` for every field this writer
    /// analyses: the positions inserted **between two values of the same
    /// multi-valued field**.
    ///
    /// Java's base `Analyzer` returns `0`, so by default a phrase query can
    /// match across a value boundary -- real Lucene's own behaviour, pinned by
    /// `fixtures/data/analysis/manifest.properties`' `mv_default_gap` case.
    /// Every Lucene consumer exposes an override for it (OpenSearch's
    /// `position_increment_gap`, default 100); Java's override is a subclass
    /// hook, and this writer has no per-field analyzer configuration, so it is
    /// one value for the whole writer.
    pub fn set_position_increment_gap(&mut self, gap: i32) {
        self.cfg_mut().position_increment_gap = gap;
    }

    /// `Analyzer.getOffsetGap(String)`: the character offsets inserted between
    /// two values of the same multi-valued field. Java's default is **`1`**,
    /// which is this writer's default too.
    pub fn set_offset_gap(&mut self, gap: i32) {
        self.cfg_mut().offset_gap = gap;
    }

    /// Opts this writer into building and writing real postings for
    /// **one additional** field, on top of whatever
    /// [`IndexWriter::set_postings_field`]/earlier `add_postings_field` calls
    /// already opted in -- the multi-field entry point this writer needed to
    /// carry more than one distinct postings field through a single
    /// [`IndexWriter::commit`] (see module doc comment and
    /// [`postings_writer::write_fields`], which already accepts any number of
    /// fields in one call; `commit()` batches every entry in this writer's
    /// postings-field list into exactly one `write_fields` call per flush, so
    /// they land in one `.doc`/`.tim`/`.tip`/`.tmd` file set together, never
    /// as separate per-field file sets).
    ///
    /// Same validation as [`IndexWriter::set_postings_field`] (`field_name`
    /// must exist in this writer's fixed `fields` list and have a supported
    /// `index_options`, see that method's doc comment for the accepted set),
    /// plus a new one: `field_name` must not already be opted in (an already
    /// -added field number returns
    /// [`Error::DuplicatePostingsField`] rather than silently duplicating it
    /// in the list, since [`postings_writer::write_fields`] itself has no
    /// defined behavior for two inputs sharing one `field_number`).
    pub fn add_postings_field(&mut self, field_name: &str) -> Result<()> {
        self.write_pending_try_changes()?;
        if self.cfg.custom_freq_postings_field.is_some() {
            return Err(Error::PostingsAndCustomFreqPostingsMutuallyExclusive(
                "add_postings_field",
            ));
        }
        let config = Self::resolve_postings_field(&self.cfg.fields, field_name)?;
        if self
            .cfg
            .postings_fields
            .iter()
            .any(|f| f.field_number == config.field_number)
        {
            return Err(Error::DuplicatePostingsField(field_name.to_string()));
        }
        self.cfg_mut().postings_fields.push(config);
        Ok(())
    }

    /// Installs (or clears, with `None`) the per-token payload supplier used
    /// for every postings field whose `FieldInfo.store_payloads` is set.
    ///
    /// This is this port's seam for what real Lucene does through
    /// `PayloadAttribute`: any `TokenFilter` in the analyzer chain may attach
    /// bytes to a token, and `IndexingChain` records whatever it finds. This
    /// facade has no per-field analyzer configuration (see the module doc
    /// comment), so the supplier is installed here instead of inside an
    /// analyzer -- the layering is the same, the indexing chain still only
    /// *records* what it is handed (see
    /// [`crate::indexing_chain::PayloadSource`]).
    ///
    /// **Which fields it applies to is not this call's decision.** Payload
    /// presence is a per-field property in Lucene (`FieldInfo.hasPayloads()`,
    /// one bit in the `.fnm`), so the gate is `store_payloads` on the
    /// [`FieldInfo`] the field was opened with; the supplier is simply never
    /// consulted for a field without it. Returns [`Error::NoPayloadFields`]
    /// when no opted-in postings field has the flag at all, which would
    /// otherwise discard every payload silently.
    ///
    /// **Divergence from Java, deliberate**: Lucene *promotes* a field to
    /// `storePayloads` the first time the indexing chain sees a token carrying
    /// a payload (`FieldInfo.setStorePayloads`), so the `.fnm` bit is derived
    /// from document content. Here it is declared up front, like every other
    /// per-field opt-in on this facade (postings, norms, doc values, vectors,
    /// term vectors). The wire result is identical for a field that is
    /// declared and used; the difference is only that a field declared with
    /// payloads but never given any writes an all-zero payload-length stream
    /// where Lucene would have written none -- which reads back as "no payload
    /// on any occurrence" either way.
    ///
    /// Call it before `commit`; it affects the next flush, like every other
    /// field opt-in here.
    pub fn set_payload_source(&mut self, source: Option<BoxedPayloadSource>) -> Result<()> {
        // Checked against the declared field list, not against the current
        // opt-ins, so the call is order-independent: a caller may install the
        // source before or after `set_postings_field`.
        if source.is_some() && !self.cfg.fields.iter().any(|f| f.store_payloads) {
            return Err(Error::NoPayloadFields);
        }
        self.cfg_mut().payload_source = source.map(SharedPayloadSource::from);
        Ok(())
    }

    /// Shared lookup/validation [`IndexWriter::set_postings_field`]/
    /// [`IndexWriter::add_postings_field`] both build a
    /// [`PostingsFieldConfig`] from.
    fn resolve_postings_field(fields: &[FieldInfo], name: &str) -> Result<PostingsFieldConfig> {
        let info = fields
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| Error::UnknownPostingsField(name.to_string()))?;
        if !matches!(
            info.index_options,
            IndexOptions::Docs
                | IndexOptions::DocsAndFreqs
                | IndexOptions::DocsAndFreqsAndPositions
                | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        ) {
            return Err(Error::UnsupportedPostingsIndexOptions(
                name.to_string(),
                info.index_options,
            ));
        }
        // Java's `FieldInfo.checkConsistency`' "indexed field 'x' cannot have
        // payloads without positions" is not re-checked here:
        // `IndexWriter::open` puts the whole field list through
        // `FieldInfos::new` (Java's constructor), so a `FieldInfo` this writer
        // holds has already been through the check that makes the combination
        // unrepresentable.
        Ok(PostingsFieldConfig {
            name: name.to_string(),
            field_number: info.number,
            index_options: info.index_options,
            store_payloads: info.store_payloads,
        })
    }

    /// Opts this writer into building and writing real
    /// `IndexOptions::DocsAndCustomFreqs` postings (`.doc`/`.tim`/`.tip`/
    /// `.tmd`, wire-identical to `DocsAndFreqs` -- see
    /// `crate::postings_writer`'s and `crate::postings`'s module doc
    /// comments' `IndexOptions::DocsAndCustomFreqs` sections) for one field of
    /// every segment [`IndexWriter::commit`] flushes from here on, driven by
    /// **explicit caller-supplied per-doc-per-term `custom_freq` values**
    /// (via [`IndexWriter::add_document_with_custom_freq_terms`]) instead of
    /// [`IndexWriter::set_postings_field`]'s analyzed-text pipeline -- there
    /// is no way to derive a genuinely arbitrary "opaque similarity score"
    /// freq value from tokenized text, so this is a separate opt-in with its
    /// own separate document-buffering entry point.
    ///
    /// `field_name` is looked up in this writer's fixed `fields` list (from
    /// [`IndexWriter::open`]) and requires its `index_options` to already be
    /// exactly `IndexOptions::DocsAndCustomFreqs` (an `Err` otherwise) --
    /// unlike [`IndexWriter::set_postings_field`], this is single-field-only
    /// (a single `Option`, not a list): there is no
    /// `add_custom_freq_postings_field` multi-field entry point yet (see
    /// `docs/parity.md`).
    ///
    /// # Scope decision: mutually exclusive with the analyzed-text postings path
    ///
    /// A writer may have **either** [`IndexWriter::set_postings_field`]/
    /// [`IndexWriter::add_postings_field`]'s analyzed-text postings field(s)
    /// **or** this custom-freq postings field active at a time, never both in
    /// the same commit -- calling this with `Some(_)` while
    /// `postings_fields` is non-empty (or calling `set_postings_field`/
    /// `add_postings_field` with a field while this is `Some`) returns
    /// [`Error::PostingsAndCustomFreqPostingsMutuallyExclusive`] rather than
    /// silently combining them. This is a deliberate, honestly-scoped
    /// simplification, not a fundamental limitation of the on-disk format
    /// (real Lucene freely mixes fields with different `IndexOptions` in one
    /// segment): the two paths buffer their per-document input differently
    /// (`pending_docs`' stored `FieldValue::String` re-tokenized by
    /// [`crate::indexing_chain::invert_documents`] vs.
    /// [`Self::pending_custom_freq_terms`]' explicit pairs), and
    /// [`IndexWriter::build_postings_output`]/
    /// [`IndexWriter::build_custom_freq_postings_output`] each build their own
    /// standalone [`postings_writer::write_fields`] input -- merging both
    /// into a single call would need every `FieldPostingsInput` in one
    /// `Vec`, which is possible in principle but not implemented here yet
    /// (would need `commit()`'s postings-output selection to union rather
    /// than branch, plus a test proving the merged multi-field write is
    /// byte-correct). `None` (the default a freshly [`IndexWriter::open`]ed
    /// writer starts with) turns this back off entirely.
    pub fn set_custom_freq_postings_field(&mut self, field_name: Option<&str>) -> Result<()> {
        if field_name.is_some() && !self.cfg.postings_fields.is_empty() {
            return Err(Error::PostingsAndCustomFreqPostingsMutuallyExclusive(
                "set_custom_freq_postings_field",
            ));
        }
        self.cfg_mut().custom_freq_postings_field = match field_name {
            None => None,
            Some(name) => {
                let info = self
                    .cfg
                    .fields
                    .iter()
                    .find(|f| f.name == name)
                    .ok_or_else(|| Error::UnknownCustomFreqPostingsField(name.to_string()))?;
                if info.index_options != IndexOptions::DocsAndCustomFreqs {
                    return Err(Error::UnsupportedCustomFreqPostingsIndexOptions(
                        name.to_string(),
                        info.index_options,
                    ));
                }
                Some(CustomFreqPostingsFieldConfig {
                    field_number: info.number,
                })
            }
        };
        Ok(())
    }

    /// Opts this writer into also building and writing real term vectors
    /// (`.tvd`/`.tvx`/`.tvm`, via [`term_vectors::write_best_speed`]) for one
    /// field of every segment [`IndexWriter::commit`] flushes from here on --
    /// same "replace, don't accumulate" semantics as
    /// [`IndexWriter::set_postings_field`]. To build term vectors for *more
    /// than one* field in the same commit, call
    /// [`IndexWriter::add_term_vector_field`] for each additional field --
    /// [`term_vectors::write_best_speed`] already accepts multiple fields per
    /// document (`TermVectorsDocument::fields` is itself a `Vec`), so every
    /// entry in this writer's term-vector-field list is folded into each
    /// pending doc's own multi-field [`TermVectorsDocument`] before one
    /// `write_best_speed` call per commit -- never one call per field.
    ///
    /// `Some(field_name)` looks `field_name` up in this writer's fixed
    /// `fields` list and requires its `store_term_vectors` flag to already be
    /// `true` (an `Err` otherwise) -- matching real Lucene's own
    /// `FieldType.setStoreTermVectors` convention, and this crate's own
    /// [`lucene_codecs::field_infos::FieldInfo::check_consistency`]
    /// invariant that a non-indexed field can never set that flag. `None`
    /// (the default a freshly [`IndexWriter::open`]ed writer starts with)
    /// turns this back off entirely -- `commit()` then behaves exactly as it
    /// did before this feature existed.
    /// A later flush rejects a vector flag that differs from an existing
    /// indexed segment; restore the setting to retry the buffered documents.
    ///
    /// Only [`FieldValue::String`] values contribute indexable text for an
    /// opted-in field -- a document with no value, or a non-`String` value,
    /// for that field contributes no term vector for that document (same
    /// "best effort per document" shape [`IndexWriter::set_postings_field`]
    /// already has). This is independent of
    /// [`IndexWriter::set_postings_field`]/[`IndexWriter::add_postings_field`]
    /// -- a writer may have both postings and term-vector fields set at once
    /// (to the same fields or different ones); each is built and written
    /// from its own in-memory pass over `pending_docs` before anything
    /// reaches `dir`.
    pub fn set_term_vector_field(&mut self, field_name: Option<&str>) -> Result<()> {
        self.cfg_mut().term_vector_fields = match field_name {
            None => Vec::new(),
            Some(name) => vec![Self::resolve_term_vector_field(&self.cfg.fields, name)?],
        };
        Ok(())
    }

    /// Opts this writer into building and writing real term vectors for
    /// **one additional** field, on top of whatever
    /// [`IndexWriter::set_term_vector_field`]/earlier `add_term_vector_field`
    /// calls already opted in -- see [`IndexWriter::set_term_vector_field`]'s
    /// doc comment for how multiple fields are batched into one
    /// [`term_vectors::write_best_speed`] call per commit.
    ///
    /// Same validation as [`IndexWriter::set_term_vector_field`], plus: an
    /// already-added field number returns
    /// [`Error::DuplicateTermVectorField`] instead of silently duplicating it
    /// in the list.
    pub fn add_term_vector_field(&mut self, field_name: &str) -> Result<()> {
        self.write_pending_try_changes()?;
        let config = Self::resolve_term_vector_field(&self.cfg.fields, field_name)?;
        if self
            .cfg
            .term_vector_fields
            .iter()
            .any(|f| f.field_number == config.field_number)
        {
            return Err(Error::DuplicateTermVectorField(field_name.to_string()));
        }
        self.cfg_mut().term_vector_fields.push(config);
        Ok(())
    }

    /// Shared lookup/validation [`IndexWriter::set_term_vector_field`]/
    /// [`IndexWriter::add_term_vector_field`] both build a
    /// [`TermVectorFieldConfig`] from.
    fn resolve_term_vector_field(
        fields: &[FieldInfo],
        name: &str,
    ) -> Result<TermVectorFieldConfig> {
        let info = fields
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| Error::UnknownTermVectorField(name.to_string()))?;
        if !info.store_term_vectors {
            return Err(Error::UnsupportedTermVectorField(name.to_string()));
        }
        Ok(TermVectorFieldConfig {
            name: name.to_string(),
            field_number: info.number,
            index_options: info.index_options,
            store_payloads: info.store_payloads,
        })
    }

    /// Opts this writer into also building and writing real doc values
    /// (`.dvd`/`.dvm`/`.dvs`) for one field of every segment
    /// [`IndexWriter::commit`] flushes from here on -- real Lucene's
    /// `IndexingChain.PerField.docValuesWriter`, created for any field whose
    /// `FieldType` declares a `DocValuesType`.
    ///
    /// `Some(field_name)` **replaces** this writer's whole doc-values field
    /// list with just `field_name` (the same "reassign, don't accumulate"
    /// shape [`IndexWriter::set_postings_field`]/
    /// [`IndexWriter::set_vector_field`] have); use
    /// [`IndexWriter::add_doc_values_field`] to write more than one
    /// doc-values field into the same segment. `None` (the default a freshly
    /// [`IndexWriter::open`]ed writer starts with) turns doc values back off
    /// entirely.
    ///
    /// **NUMERIC, BINARY, SORTED, SORTED_NUMERIC, and SORTED_SET doc values
    /// are all wired up by this writer** -- see
    /// [`Self::build_doc_values_output`]'s per-type branches for exactly how
    /// each sources its per-doc values from [`FieldValue`]. The field is
    /// looked up in this writer's fixed `fields` list and its
    /// `doc_values_type` must already be one of those five (an `Err`
    /// otherwise), matching real Lucene's own `FieldType.setDocValuesType`
    /// convention.
    ///
    /// **Dense when every pending doc has a value, sparse otherwise,
    /// decided at flush time, not here:** with **one** configured field, if
    /// some pending docs carry a value and some don't,
    /// [`Self::build_doc_values_output`] routes the present docs' values
    /// through the corresponding `write_single_sparse_*_field` writer instead
    /// of the dense one (the missing docs read back as absent, not as a
    /// wrong/zero value). With **more than one** configured field they share
    /// one `.dvm`/`.dvd` triple written by
    /// [`doc_values::write_dense_fields`], which accepts a sparse column only
    /// for every type (a field some documents have no value for is written
    /// through its sparse variant, e.g.
    /// [`doc_values::DenseField::SparseSorted`]). A doc whose
    /// value is present but the wrong [`FieldValue`] variant still fails the
    /// whole flush with [`Error::NonNumericDocValue`]/
    /// [`Error::NonBinaryDocValue`], leaving `dir`/`pending_docs`/
    /// `segment_infos` unchanged; if *no* pending doc has a value for a
    /// configured field the flush fails with [`Error::MissingDenseDocValue`]
    /// (nothing meaningful to encode).
    pub fn set_doc_values_field(&mut self, field_name: Option<&str>) -> Result<()> {
        // This is the one operation that *shrinks* the list, so it is the one
        // that can strand a configured index sort over a column nothing
        // writes -- exactly the state `set_index_sort` refuses to create (see
        // [`Error::IndexSortFieldWithoutDocValues`]). Java cannot reach it:
        // its sort is fixed on `IndexWriterConfig` and its doc values are not
        // opt-in at all. Checked before the mutation, so a rejected call
        // leaves the list untouched.
        if let Some(sort) = &self.cfg.index_sort {
            if let Some(stranded) = sort.iter().find(|sf| Some(sf.field.as_str()) != field_name) {
                return Err(Error::IndexSortFieldWithoutDocValues(
                    stranded.field.clone(),
                ));
            }
        }
        self.cfg_mut().doc_values_fields.clear();
        match field_name {
            None => Ok(()),
            Some(name) => self.add_doc_values_field(name),
        }
    }

    /// Adds one more field to this writer's doc-values field list without
    /// disturbing the ones already there -- see
    /// [`IndexWriter::set_doc_values_field`] for the contract. Every field in
    /// the list is written into the **same** `.dvm`/`.dvd`/`.dvs` triple,
    /// which is what a real `Lucene90DocValuesFormat` segment does: one
    /// `.dvm` holding one meta entry per field, in field order.
    ///
    /// This is what makes a **multi-field index sort** expressible: an index
    /// sort's second tier is a second doc-values column, and before this the
    /// writer could only write one.
    pub fn add_doc_values_field(&mut self, field_name: &str) -> Result<()> {
        self.write_pending_try_changes()?;
        let info = self
            .cfg
            .fields
            .iter()
            .find(|f| f.name == field_name)
            .cloned()
            .ok_or_else(|| Error::UnknownDocValuesField(field_name.to_string()))?;
        if !matches!(
            info.doc_values_type,
            DocValuesType::Numeric
                | DocValuesType::Binary
                | DocValuesType::Sorted
                | DocValuesType::SortedNumeric
                | DocValuesType::SortedSet
        ) {
            return Err(Error::UnsupportedDocValuesType(
                field_name.to_string(),
                info.doc_values_type,
            ));
        }
        if self
            .cfg
            .doc_values_fields
            .iter()
            .any(|c| c.name == field_name)
        {
            return Err(Error::DuplicateDocValuesField(field_name.to_string()));
        }
        self.cfg_mut().doc_values_fields.push(DocValuesFieldConfig {
            name: field_name.to_string(),
            field_number: info.number,
            doc_values_type: info.doc_values_type,
            skip_index: info.doc_values_skip_index_type
                != lucene_codecs::field_infos::DocValuesSkipIndexType::None,
        });
        Ok(())
    }

    /// `IndexWriterConfig.setIndexSort(Sort)`: fixes the (possibly
    /// multi-field) key every segment this writer flushes from here on is
    /// **physically ordered by**, and that the segment's `.si` records
    /// (`SegmentInfo.getIndexSort`, which a reader surfaces as
    /// `LeafMetaData.getSort()`).
    ///
    /// An index-sorted segment is not a segment with a sort recorded beside
    /// it: doc id 0 really is the smallest document by this key, doc id 1 the
    /// next, and so on, for **every** format the flush writes -- stored
    /// fields, postings, term vectors, norms, doc values and vectors alike.
    /// See [`IndexWriter::flush`] for how that is achieved here (one
    /// permutation applied to the document buffer before any format is
    /// built, rather than Java's per-format `Sorter.DocMap` remap).
    ///
    /// # What is validated, and against what in Java
    ///
    /// - **Non-empty** ([`Error::EmptyIndexSort`]). On disk `numSortFields ==
    ///   0` *is* "unsorted", so an empty sort has no encoding distinct from
    ///   `None`.
    /// - **Every sort field exists** in this writer's fixed `fields` list
    ///   ([`Error::UnknownIndexSortField`]). Java creates the field on
    ///   demand; this port's schema is fixed at [`IndexWriter::open`].
    /// - **Every sort field's doc-values type matches the kind of sort**
    ///   ([`Error::UnsupportedIndexSortField`]) --
    ///   `IndexingChain.validateIndexSortDVType`, which asks the
    ///   `SortField`'s own `IndexSorter` which column it reads: a numeric
    ///   `SortField` needs NUMERIC, a `SortedNumericSortField` needs
    ///   SORTED_NUMERIC, a `STRING` `SortField` needs SORTED, a
    ///   `SortedSetSortField` needs SORTED_SET and a `BinarySortField`
    ///   needs BINARY. The byte-keyed three sort by the rank of the
    ///   document's value ([`crate::index_sorter`]), which the flush computes
    ///   from the buffer before it permutes it -- the same ordinals Java's
    ///   `SortedDocValuesWriter` hands its comparator.
    ///
    ///   A `FLOAT`/`DOUBLE` sort *is* supported: Lucene's own
    ///   `FloatDocValuesField`/`DoubleDocValuesField` store
    ///   `Float.floatToRawIntBits`/`Double.doubleToRawLongBits` in a NUMERIC
    ///   column, and `FloatField`/`DoubleField` store
    ///   `NumericUtils.floatToSortableInt` in a SORTED_NUMERIC one --
    ///   [`segment_info::SortKeyComparator`] knows which encoding each kind
    ///   reads, so the caller stores what Lucene stores and nothing else
    ///   changes.
    /// - **Every sort field is actually opted into doc values**
    ///   ([`Error::IndexSortFieldWithoutDocValues`]). Java gets this for
    ///   free -- a field with a `DocValuesType` always gets a
    ///   `DocValuesWriter` -- but this writer's doc values are opt-in, and a
    ///   sort over a column nothing writes is a sort no reader can check:
    ///   real Lucene's `DocValues.getNumeric` substitutes an all-missing
    ///   instance rather than failing, so `CheckIndex.testSort` would compare
    ///   `maxDoc` equal keys and pass.
    /// - **Congruent with every segment already in the index**
    ///   ([`Error::IncongruentIndexSort`]) -- `IndexWriter.validateIndexSort`
    ///   plus `isCongruentSort`: the incoming sort must be a *prefix* of each
    ///   existing segment's, so that a later sort-preserving merge over them
    ///   is well defined. Checked against the last commit's segments and the
    ///   already-flushed-but-uncommitted ones, by reading each `.si`.
    /// - **No documents buffered** ([`Error::IndexSortChangedMidBuffer`]).
    ///   Java cannot reach this state at all: `IndexWriterConfig` is
    ///   snapshotted when the `IndexWriter` is constructed. Since this
    ///   writer's opt-ins are read at flush (see
    ///   [`IndexWriter::set_vector_field`]'s note on timing), changing the
    ///   sort mid-buffer would order one part of the batch by one key and
    ///   another part by another; refusing is the faithful lowering of "the
    ///   sort is fixed for the writer's life".
    ///
    /// Passing `None` clears the sort. That is *also* subject to the
    /// buffered-documents guard, and note it does not make an
    /// already-sorted index unsorted: existing segments keep their `.si`
    /// sort, and a later `set_index_sort(Some(..))` must still be congruent
    /// with them.
    pub fn set_index_sort(&mut self, sort: Option<&[segment_info::IndexSortField]>) -> Result<()> {
        if !self.pending_docs.is_empty() {
            return Err(Error::IndexSortChangedMidBuffer(self.pending_docs.len()));
        }
        let Some(sort) = sort else {
            self.cfg_mut().index_sort = None;
            return Ok(());
        };
        if sort.is_empty() {
            return Err(Error::EmptyIndexSort);
        }
        for sf in sort {
            let wanted = sort_doc_values_type(sf);
            let registered = self.cfg.field_by_name(&sf.field);
            if self.cfg.explicit {
                // Explicit documents register their fields as they arrive, as
                // Java's `IndexWriterConfig.setIndexSort` names fields no
                // document has yet: `register_field` checks the type then
                // (`IndexingChain.validateIndexSortDVType`).
                match registered {
                    Some(info) if info.doc_values_type != wanted => {
                        return Err(Error::UnsupportedIndexSortField(
                            sf.field.clone(),
                            info.doc_values_type,
                        ));
                    }
                    _ => continue,
                }
            }
            let info = registered.ok_or_else(|| Error::UnknownIndexSortField(sf.field.clone()))?;
            if info.doc_values_type != wanted {
                return Err(Error::UnsupportedIndexSortField(
                    sf.field.clone(),
                    info.doc_values_type,
                ));
            }
            if !self
                .cfg
                .doc_values_fields
                .iter()
                .any(|c| c.name == sf.field)
            {
                return Err(Error::IndexSortFieldWithoutDocValues(sf.field.clone()));
            }
        }
        self.validate_index_sort_against_existing_segments(sort)?;
        self.cfg_mut().index_sort = Some(sort.to_vec());
        Ok(())
    }

    /// The sort this writer is configured with, `None` when unsorted.
    pub fn index_sort(&self) -> Option<&[segment_info::IndexSortField]> {
        self.cfg.index_sort.as_deref()
    }

    /// `IndexWriter.validateIndexSort()`: every segment already in the index
    /// must carry a sort `incoming` is a prefix of. A segment with **no**
    /// sort fails just as hard as one with a different sort -- Java's
    /// `segmentIndexSort == null ||` branch -- because its documents are in
    /// insertion order and no merge could put them in key order without
    /// re-sorting them.
    fn validate_index_sort_against_existing_segments(
        &self,
        incoming: &[segment_info::IndexSortField],
    ) -> Result<()> {
        let existing_segments = self
            .segment_infos
            .segments
            .iter()
            .chain(self.flushed_segments.iter());
        for sci in existing_segments {
            let si_bytes = self.dir.open(&format!("{}.si", sci.segment_name))?.to_vec();
            let si = segment_info::parse_for_commit(&si_bytes, sci)?;
            let congruent = match &si.index_sort {
                None => false,
                Some(existing) => {
                    incoming.len() <= existing.len() && incoming[..] == existing[..incoming.len()]
                }
            };
            if !congruent {
                return Err(Error::IncongruentIndexSort {
                    segment: sci.segment_name.clone(),
                    existing: segment_info::describe_index_sort(si.index_sort.as_deref()),
                    incoming: segment_info::describe_index_sort(Some(incoming)),
                });
            }
        }
        Ok(())
    }

    /// `FieldType.setOmitNorms(true)` for one field of this writer's fixed
    /// field list: from here on no segment this writer flushes carries a norm
    /// column for it, and its `.fnm` says `omitNorms` so a reader never looks
    /// for one.
    ///
    /// This is the **only** norms knob, and it is an opt-out, matching
    /// Lucene: an indexed field gets norms unless it says otherwise (see
    /// [`IndexingConfig::norms_field_configs`]). A field that is not indexed
    /// (`index_options == IndexOptions::None`) has no norms to omit and is
    /// rejected rather than silently accepted, so a caller that names the
    /// wrong field hears about it.
    ///
    /// Norms are computed, never supplied: [`Self::build_norms_output`]
    /// derives each document's field length from the one shared invert pass
    /// (real Lucene's `FieldInvertState.length` -- total indexed token count,
    /// not distinct-term count) and encodes it with
    /// [`lucene_util::small_float::int_to_byte4`], the exact inverse of
    /// `lucene_search::similarity::decode_norm`.
    ///
    /// Naming a field twice is a no-op.
    ///
    /// Calling this **between two commits** is allowed and takes effect for
    /// every segment flushed after it -- but the two segments then disagree
    /// about the field's schema, and a merge across them is
    /// `merge::Error::FieldSchemaDisagreement { attribute: "omit_norms" }`,
    /// which is Java's `FieldInfos.verifySameOmitNorms` refusing the same
    /// pair. Set it before the first document unless that is what you want.
    /// (A *document* that simply does not carry the field is a different
    /// thing entirely and needs no configuration: the norm column is
    /// per-document sparse -- see [`Self::build_norms_output`].)
    pub fn omit_norms_field(&mut self, field_name: &str) -> Result<()> {
        self.write_pending_try_changes()?;
        let info = self
            .cfg_mut()
            .fields
            .iter_mut()
            .find(|f| f.name == field_name)
            .ok_or_else(|| Error::UnknownNormsField(field_name.to_string()))?;
        if matches!(info.index_options, IndexOptions::None) {
            return Err(Error::UnsupportedNormsField(field_name.to_string()));
        }
        info.omit_norms = true;
        Ok(())
    }

    /// Opts this writer into indexing **vectors** for one field of every
    /// segment it flushes from here on -- real Lucene's
    /// `KnnFloatVectorField`/`KnnByteVectorField` reaching `IndexingChain`'s
    /// per-field `KnnFieldVectorsWriter`, which
    /// `Lucene99HnswVectorsFormat`'s writer turns into a
    /// `.vec`/`.vemf` flat store plus a `.vem`/`.vex` HNSW graph at flush.
    ///
    /// `Some(field_name)` **replaces** this writer's whole vector-field list
    /// with just `field_name` (the same "reassign, don't accumulate" shape
    /// [`IndexWriter::set_postings_field`] has); use
    /// [`IndexWriter::add_vector_field`] to index more than one vector field
    /// in the same segment. `None` turns vectors back off entirely.
    ///
    /// The field is looked up in this writer's fixed `fields` list and must
    /// declare a positive `vector_dimension`; its `vector_encoding` and
    /// `vector_similarity_function` are taken from the same [`FieldInfo`]. That
    /// is deliberate rather than a separate parameter: Lucene's
    /// `Lucene99FlatVectorsReader.FieldEntry` refuses to open a segment whose
    /// `.vemf` similarity or dimension disagrees with the `.fnm`'s, so the two
    /// must come from one source.
    ///
    /// **Configure before the first document.** This writer's opt-ins are read
    /// at flush, not at `add_document`, so calling `set_vector_field(None)`
    /// with vectors already buffered discards them silently -- the flush writes
    /// no vector files and zeroes every `.fnm` dimension, and nothing reports
    /// it. That is the same timing every other opt-in here has
    /// ([`IndexWriter::set_postings_field`],
    /// [`IndexWriter::set_doc_values_field`], [`IndexWriter::omit_norms_field`]),
    /// and is a consequence of buffering documents and encoding at flush; in
    /// Java the equivalent is fixed on the `FieldType` before the document
    /// exists. Reconfigure after a [`IndexWriter::flush`] or
    /// [`IndexWriter::commit`], not mid-buffer.
    ///
    /// A document supplies its value through
    /// [`IndexWriter::add_document_with_vectors`]; a document added through
    /// plain [`IndexWriter::add_document`] simply has no vector for the field,
    /// which is the sparse case (`OrdToDocDISIReaderConfiguration`'s
    /// `IndexedDISI` + `DirectMonotonicWriter` pair), not an error -- exactly
    /// as in Lucene, where a `Document` need not carry every field.
    pub fn set_vector_field(&mut self, field_name: Option<&str>) -> Result<()> {
        self.cfg_mut().vector_fields.clear();
        match field_name {
            None => Ok(()),
            Some(name) => self.add_vector_field(name),
        }
    }

    /// Adds one more field to this writer's vector-field list without
    /// disturbing the ones already there -- see
    /// [`IndexWriter::set_vector_field`] for the contract. Every field in the
    /// list is written into the **same** `.vec`/`.vemf`/`.vem`/`.vex`
    /// quadruple, which is what `Lucene99HnswVectorsFormat` does for all the
    /// vector fields routed to it.
    pub fn add_vector_field(&mut self, field_name: &str) -> Result<()> {
        self.write_pending_try_changes()?;
        let info = self
            .cfg
            .fields
            .iter()
            .find(|f| f.name == field_name)
            .cloned()
            .ok_or_else(|| Error::UnknownVectorField(field_name.to_string()))?;
        if info.vector_dimension <= 0 {
            return Err(Error::UnsupportedVectorField(
                field_name.to_string(),
                info.vector_dimension,
            ));
        }
        if self.cfg.vector_fields.iter().any(|c| c.name == field_name) {
            return Err(Error::DuplicateVectorField(field_name.to_string()));
        }
        self.cfg_mut().vector_fields.push(VectorFieldConfig {
            name: field_name.to_string(),
            field_number: info.number,
            dimension: info.vector_dimension,
            encoding: info.vector_encoding,
            similarity: info.vector_similarity_function,
        });
        Ok(())
    }

    /// Opts `field_name` into being indexed as **points** --
    /// `Lucene90PointsFormat`'s `.kdm`/`.kdi`/`.kdd` BKD tree, what
    /// `LongPoint`/`IntPoint`/`FloatPoint`/`DoublePoint`/`BinaryPoint` fields
    /// produce in Java. The field's [`FieldInfo`] must declare the shape
    /// (`point_dimension_count`, `point_index_dimension_count`,
    /// `point_num_bytes`), as `FieldType.setDimensions` does.
    ///
    /// A document's value is every [`StoredField`](stored_fields::StoredField)
    /// it carries for the field -- more than one is a multi-valued point
    /// field, as adding several `LongPoint`s is -- encoded as Java encodes
    /// them: a one-dimensional field takes [`FieldValue::Int`]/
    /// [`FieldValue::Float`] (4 bytes) or [`FieldValue::Long`]/
    /// [`FieldValue::Double`] (8 bytes) through `NumericUtils`' sortable
    /// big-endian form, and any field takes [`FieldValue::Binary`] as the
    /// already-packed `numDims * bytesPerDim` bytes. A document without the
    /// field has no point. Like every opt-in here, configure it before the
    /// first document.
    ///
    /// The values are **also stored**, as a doc-values field's are: this
    /// facade's [`Document`] *is* the document's
    /// stored fields, and every entry in it is written to `.fdt`. Java's
    /// `LongPoint` is not stored, so the Java document this one equals is
    /// `LongPoint` plus a `StoredField` of the same name.
    pub fn add_points_field(&mut self, field_name: &str) -> Result<()> {
        self.write_pending_try_changes()?;
        let info = self
            .cfg
            .fields
            .iter()
            .find(|f| f.name == field_name)
            .cloned()
            .ok_or_else(|| Error::UnknownPointsField(field_name.to_string()))?;
        let (dims, index_dims, bytes) = (
            info.point_dimension_count,
            info.point_index_dimension_count,
            info.point_num_bytes,
        );
        if !(1..=16).contains(&dims)
            || !(1..=dims.min(8)).contains(&index_dims)
            || !(1..=16).contains(&bytes)
        {
            return Err(Error::UnsupportedPointsField(
                field_name.to_string(),
                dims,
                index_dims,
                bytes,
            ));
        }
        if self.cfg.points_fields.iter().any(|c| c.name == field_name) {
            return Err(Error::DuplicatePointsField(field_name.to_string()));
        }
        self.cfg_mut().points_fields.push(PointsFieldConfig {
            name: field_name.to_string(),
            field_number: info.number,
            num_dims: dims,
            num_index_dims: index_dims,
            bytes_per_dim: bytes,
        });
        Ok(())
    }

    /// `Lucene99HnswVectorsFormat(maxConn, beamWidth)`: the two graph
    /// parameters every vector field this writer flushes is built with.
    /// Defaults are Lucene's own, `M = 16` / `beamWidth = 100`.
    ///
    /// Both bounds are Lucene's (`M` in `1..=512`, `beamWidth` in `1..=3200`),
    /// enforced here rather than at flush so a bad value is a configuration
    /// error rather than a half-written segment.
    ///
    /// In Java these are constructor arguments to
    /// `Lucene99HnswVectorsFormat`, i.e. fixed for the codec's lifetime. Here
    /// they are read at flush, so changing them mid-buffer changes the graph
    /// the *current* buffer is about to produce -- set them once, before the
    /// first document.
    pub fn set_hnsw_parameters(&mut self, m: i32, beam_width: i32) -> Result<()> {
        if !(1..=hnsw::MAXIMUM_MAX_CONN).contains(&m) {
            return Err(Error::Vectors(vectors::Error::InvalidGraphParameter(
                format!(
                    "M (max connections) must be in 1..={}, got {m}",
                    hnsw::MAXIMUM_MAX_CONN
                ),
            )));
        }
        if !(1..=hnsw::MAXIMUM_BEAM_WIDTH).contains(&beam_width) {
            return Err(Error::Vectors(vectors::Error::InvalidGraphParameter(
                format!(
                    "beamWidth must be in 1..={}, got {beam_width}",
                    hnsw::MAXIMUM_BEAM_WIDTH
                ),
            )));
        }
        self.cfg_mut().hnsw_m = m;
        self.cfg_mut().hnsw_beam_width = beam_width;
        Ok(())
    }

    /// `SoftDeletesRetentionMergePolicy`: from now on, a merge drops a
    /// soft-deleted document unless `retention` still needs it (see
    /// [`SoftDeletesRetention`]). `None` -- the default -- keeps every
    /// soft-deleted document, which never loses history but never reclaims
    /// it either. Hard deletes are always dropped.
    ///
    /// **A deliberate deviation.** Java's `IndexWriter` with a soft-deletes
    /// field and *no* retention policy drops every soft-deleted document at
    /// merge. This writer keeps them: keeping is the side that cannot lose an
    /// operation's history. OpenSearch always installs a retention policy,
    /// and so does the engine (`lucene-ffi`'s `engine_writer`), so the engine
    /// never sees the difference.
    ///
    /// Takes effect for merges that begin after the call, as Java's retention
    /// query is evaluated when a merge opens its readers.
    pub fn set_soft_deletes_retention(&mut self, retention: Option<SoftDeletesRetention>) {
        self.soft_deletes_retention = retention;
    }

    /// Opts this writer into automatic merge triggering (see module doc
    /// comment): `Some(config)` makes every subsequent
    /// [`IndexWriter::commit`] call consult
    /// [`crate::merge_policy::find_merges`] with `config` after writing its
    /// own commit, and execute/fold in whatever it proposes. `None` (the
    /// default a freshly [`IndexWriter::open`]ed writer starts with) turns
    /// this back off -- `commit()` then behaves exactly as it did before
    /// this feature existed.
    pub fn set_merge_policy(&mut self, config: Option<MergePolicyConfig>) {
        self.merge_policy = config;
    }

    /// Read-only access to the directory this writer was opened over.
    /// Exists so a caller that wants to drive
    /// [`IndexWriter::update_document`]/[`IndexWriter::delete_documents_by_term`]
    /// itself can reopen this writer's already-committed segments' files to
    /// build the [`update_document::SegmentDeleteSource`]s those two methods
    /// require (see `crates/lucene-ffi/src/writer.rs`'s
    /// `ffi_writer_update_document`/`ffi_writer_delete_documents` for exactly
    /// that use) -- nothing about the writer's own state is exposed here,
    /// only the same `&dyn Directory` it already holds.
    pub fn dir(&self) -> &'d dyn Directory {
        self.dir
    }

    /// Read-only access to this writer's fixed field list (see
    /// [`IndexWriter::open`]'s doc comment for what "fixed" means here) --
    /// same rationale as [`IndexWriter::dir`]: a caller building
    /// [`update_document::SegmentDeleteSource`]s externally needs the exact
    /// [`FieldInfo`] list this writer's own segments were written against to
    /// reopen their postings via [`lucene_codecs::blocktree::open`].
    pub fn fields(&self) -> &[FieldInfo] {
        &self.cfg.fields
    }

    /// The fields of the segments this writer holds, as their `FieldInfos`
    /// record them, attributes included: per name, the first segment's.
    /// A writer opened on an index learns from these what the plugin's
    /// mapping asked of a field (its per-field formats) before any document
    /// registers it again.
    pub fn segment_fields(&self) -> Result<Vec<FieldInfo>> {
        let mut seen = std::collections::BTreeSet::new();
        let mut out = Vec::new();
        for segment in self
            .segment_infos
            .segments
            .iter()
            .chain(&self.flushed_segments)
        {
            for field in segment_field_infos(self.dir, segment)?.fields {
                if seen.insert(field.name.clone()) {
                    out.push(field);
                }
            }
        }
        Ok(out)
    }

    /// Buffers `doc` for the next [`IndexWriter::commit`] -- real Lucene's
    /// `IndexWriter.addDocument`.
    ///
    /// Like Java's, this may **flush** before it returns: once the buffered
    /// documents exceed [`IndexWriter::set_ram_buffer_size_mb`] (default 16 MB,
    /// exactly Lucene's) or [`IndexWriter::set_max_buffered_docs`], the whole
    /// buffer is written out as a new segment via [`IndexWriter::flush`]. That
    /// segment is on disk and protected by the deleter, but is **not** yet part
    /// of any commit: it becomes visible to a reader at the next
    /// [`IndexWriter::commit`], and is discarded -- files included -- by
    /// [`IndexWriter::rollback`]. This is why the method is fallible;
    /// `addDocument` throws `IOException` in Java for the same reason.
    pub fn add_document(&mut self, doc: Document) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        // Not `add_documents(vec![doc])`: the single-document add is the hot
        // path of the whole write side, and wrapping every document in a
        // one-element `Vec` puts a heap allocation and a move in front of it
        // for nothing. A one-document call can never set `has_blocks` (Java:
        // `numDocs > 1`) and carries no delete node, so there is nothing the
        // block path would do that this does not.
        self.reserve_docs(1)?;
        let seq_no = self.delete_queue.next_sequence_number();
        self.buffer_document(doc);
        self.mark_last_buffered_parent()?;
        self.maybe_flush()?;
        Ok(seq_no)
    }

    /// `IndexWriter.reserveDocs`: refuses an add that would take the index
    /// past [`IndexWriter::set_max_docs`]' limit, before anything -- the
    /// add's own delete included -- is buffered. Java counts
    /// `pendingNumDocs`, every document of every segment plus the buffered
    /// ones; deleted documents count until a merge drops them.
    pub(crate) fn reserve_docs(&mut self, added: usize) -> Result<()> {
        self.write_pending_try_changes()?;
        let segments = self.segment_doc_total()?;
        let total = segments.saturating_add(self.pending_docs.len());
        if total.saturating_add(added) > self.max_docs {
            return Err(Error::TooManyDocs(self.max_docs));
        }
        Ok(())
    }

    /// [`IndexWriter::set_max_docs`]' limit.
    pub(crate) fn max_docs_limit(&self) -> usize {
        self.max_docs
    }

    /// Every committed and flushed segment's `maxDoc`, summed -- cached
    /// until the segment lists change.
    pub(crate) fn segment_doc_total(&mut self) -> Result<usize> {
        self.write_pending_try_changes()?;
        let generation = self.segment_infos.generation;
        let flushed = self.flushed_segments.len();
        let last = self
            .flushed_segments
            .last()
            .map(|s| s.segment_name.as_str());
        let segments = match &self.segment_docs_total {
            Some((g, n, l, total)) if *g == generation && *n == flushed && l.as_deref() == last => {
                *total
            }
            _ => {
                let total = self.sum_segment_max_docs()?;
                self.segment_docs_total = Some((
                    generation,
                    flushed,
                    self.flushed_segments.last().map(|s| s.segment_name.clone()),
                    total,
                ));
                total
            }
        };
        Ok(segments)
    }

    /// Every committed and flushed segment's `maxDoc`, summed.
    fn sum_segment_max_docs(&mut self) -> Result<usize> {
        let mut total = 0usize;
        let segments = self
            .segment_infos
            .segments
            .iter()
            .chain(&self.flushed_segments);
        for sci in segments {
            let max_doc = match self.segment_max_docs.get(&sci.segment_name) {
                Some(&n) => n,
                None => {
                    let si = segment_info::parse_for_commit(
                        &self.dir.open(&format!("{}.si", sci.segment_name))?,
                        sci,
                    )?;
                    let n = usize::try_from(si.doc_count).unwrap_or(0);
                    self.segment_max_docs.insert(sci.segment_name.clone(), n);
                    n
                }
            };
            total = total.saturating_add(max_doc);
        }
        // Merged-away segments leave stale entries behind; drop them once
        // they outnumber the live ones.
        let live = self
            .segment_infos
            .segments
            .len()
            .saturating_add(self.flushed_segments.len());
        if self.segment_max_docs.len() > live.saturating_mul(2) {
            let infos = &self.segment_infos;
            let flushed = &self.flushed_segments;
            self.segment_max_docs.retain(|name, _| {
                infos
                    .segments
                    .iter()
                    .chain(flushed)
                    .any(|s| &s.segment_name == name)
            });
        }
        Ok(total)
    }

    /// `IndexWriter.setMaxDocs`: lowers the limit [`IndexWriter::reserve_docs`]
    /// enforces from [`MAX_DOCS`]. Java keeps it for tests, and so does
    /// this; a caller embedding a JVM passes it the JVM's own
    /// `IndexWriter.getActualMaxDocs()`, so both engines refuse at the same
    /// count. Values above [`MAX_DOCS`] are clamped to it.
    pub fn set_max_docs(&mut self, max_docs: usize) {
        self.max_docs = max_docs.min(MAX_DOCS);
    }

    /// Appends one document to the pending buffer, keeping the parallel
    /// custom-freq list and the RAM counter in step.
    fn buffer_document(&mut self, doc: Document) {
        // ARITH: see [`VectorValue::ram_bytes`] -- a running total of live
        // allocation sizes, reset to `0` on every flush.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.ram_bytes_used += document_ram_bytes(&doc);
        }
        self.pending_docs.push(doc);
        // Keeps `pending_custom_freq_terms` and `pending_vectors` aligned 1:1
        // by index with `pending_docs` regardless of which entry point buffered
        // any given doc -- see those fields' own doc comments.
        self.pending_custom_freq_terms.push(Vec::new());
        self.pending_vectors.push(Vec::new());
        self.pending_explicit.push(ExplicitFields::default());
    }

    /// Buffers `doc` for the next flush, same as
    /// [`IndexWriter::add_document`], and records `vectors` as this document's
    /// values for the fields opted in through
    /// [`IndexWriter::set_vector_field`]/[`IndexWriter::add_vector_field`] --
    /// real Lucene's `document.add(new KnnFloatVectorField(name, vector))`.
    ///
    /// Every entry is validated **here**, not at flush: a vector's length must
    /// equal its field's declared `vector_dimension` and its variant must
    /// match the field's declared `vector_encoding`. Java validates in the
    /// `KnnFloatVectorField` constructor for the same reason -- the caller
    /// still has the offending document in hand, and a mismatch discovered at
    /// flush would fail a `commit` carrying thousands of unrelated documents.
    /// (`VectorUtil.checkFinite`, Java's third constructor check, is applied by
    /// [`vectors::write_flat_vectors`] at flush, which is the last point that
    /// can see the whole field.)
    ///
    /// A named field that is not opted in is rejected too
    /// ([`Error::UnknownVectorField`]): silently dropping the vector would
    /// produce a segment that indexes nothing while looking like it did.
    ///
    /// A field left out of `vectors` simply has no vector on this document --
    /// the sparse case, which is legal and is what the `.vemf`'s
    /// `IndexedDISI`/`DirectMonotonicWriter` pair exists to record.
    pub fn add_document_with_vectors(
        &mut self,
        doc: Document,
        vectors: Vec<DocumentVector>,
    ) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        self.cfg
            .validate_document_vectors(self.pending_docs.len(), &vectors)?;
        let seq_no = self.delete_queue.next_sequence_number();
        // ARITH: see [`VectorValue::ram_bytes`].
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.ram_bytes_used += document_ram_bytes(&doc)
                + vectors.iter().map(DocumentVector::ram_bytes).sum::<usize>();
        }
        self.pending_docs.push(doc);
        self.pending_custom_freq_terms.push(Vec::new());
        self.pending_vectors.push(vectors);
        self.pending_explicit.push(ExplicitFields::default());
        self.mark_last_buffered_parent()?;
        self.maybe_flush()?;
        Ok(seq_no)
    }
}

impl IndexingConfig {
    /// Vector field `name`: its number, dimension, encoding and similarity,
    /// from the registered fields of a writer of explicit documents.
    fn vector_field_shape(
        &self,
        name: &str,
    ) -> Option<(i32, i32, VectorEncoding, VectorSimilarityFunction)> {
        if self.takes_explicit() {
            let f = self.explicit_vector_field(name).ok()?;
            return Some((
                f.number,
                f.vector_dimension,
                f.vector_encoding,
                f.vector_similarity_function,
            ));
        }
        self.vector_fields
            .iter()
            .find(|c| c.name == name)
            .map(|c| (c.field_number, c.dimension, c.encoding, c.similarity))
    }

    /// The number of vector field `name`.
    pub(crate) fn vector_field_number(&self, name: &str) -> Option<i32> {
        self.vector_field_shape(name).map(|(number, ..)| number)
    }

    /// An empty graph for vector field `name` that grows as its vectors
    /// arrive and is the graph its flush writes: `None` for a field whose
    /// format has no graph or scores bytes as Hamming distances. One `live`
    /// search reads as it grows, or one only the flush reads. Under an
    /// index sort the flush renumbers the graph, as
    /// `Lucene99HnswVectorsWriter.writeSortingField` does; live search
    /// keeps none then.
    pub(crate) fn buffer_graph_for(&self, name: &str, live: bool) -> Option<KeptGraph> {
        if self.sorts_documents() && live {
            return None;
        }
        let (_, dimension, encoding, similarity) = self.vector_field_shape(name)?;
        let format = self.knn_vectors_format_for(name);
        if matches!(
            format.files_format(),
            per_field_knn_vectors::KnnVectorsFormat::HnswHamming { .. }
        ) {
            return None;
        }
        let (m, beam_width) = format.graph()?;
        let dimension = usize::try_from(dimension).ok()?;
        let seed = hnsw::DEFAULT_RAND_SEED;
        Some(if live {
            KeptGraph::Live(std::sync::Arc::new(
                lucene_codecs::buffer_vectors::BufferGraph::new(
                    encoding, similarity, dimension, m, beam_width, seed,
                )
                .ok()?,
            ))
        } else {
            KeptGraph::Flush(std::sync::Arc::new(
                lucene_codecs::buffer_vectors::FlushGraph::new(
                    encoding, similarity, dimension, m, beam_width, seed,
                )
                .ok()?,
            ))
        })
    }

    /// Cluster lists for live search over a buffer's vectors of field
    /// `name`, routed to the clusters its last flush wrote: `None` when its
    /// format is not IVF, its vectors are not floats, or no flush wrote it
    /// yet.
    pub(crate) fn buffer_lists_for(
        &self,
        name: &str,
    ) -> Option<lucene_codecs::buffer_lists::BufferLists> {
        let (number, dimension, encoding, similarity) = self.vector_field_shape(name)?;
        if encoding != lucene_codecs::field_infos::VectorEncoding::Float32 {
            return None;
        }
        self.knn_vectors_format_for(name).ivf_params()?;
        let router = self.ivf_clusters.get(number)?;
        Some(lucene_codecs::buffer_lists::BufferLists::new(
            similarity,
            usize::try_from(dimension).ok()?,
            router,
        ))
    }

    /// [`IndexWriter::add_document_with_vectors`]' checks of one document's
    /// vectors (document `doc_index` of its buffer): every field registered
    /// with its encoding and dimension, and named once.
    pub(crate) fn validate_document_vectors(
        &self,
        doc_index: usize,
        vectors: &[DocumentVector],
    ) -> Result<()> {
        for v in vectors {
            let config = self
                .vector_fields
                .iter()
                .find(|c| c.name == v.field_name)
                .ok_or_else(|| Error::UnknownVectorField(v.field_name.clone()))?;
            if v.value.encoding() != config.encoding {
                return Err(Error::VectorEncodingMismatch(
                    v.field_name.clone(),
                    doc_index,
                    config.encoding,
                    v.value.encoding(),
                ));
            }
            if v.value.len() != config.dimension as usize {
                return Err(Error::VectorDimensionMismatch(
                    v.field_name.clone(),
                    doc_index,
                    config.dimension,
                    v.value.len(),
                ));
            }
        }
        // A field named twice on one document would silently write only one of
        // the two vectors; Lucene's `IndexingChain.addField` throws
        // `IllegalArgumentException("VectorValuesField \"...\" appears more
        // than once in this document")` for exactly that.
        for (i, v) in vectors.iter().enumerate() {
            if vectors[..i].iter().any(|w| w.field_name == v.field_name) {
                return Err(Error::DuplicateVectorField(v.field_name.clone()));
            }
        }
        Ok(())
    }
}

impl IndexWriter<'_> {
    /// `IndexWriter.addDocuments(Iterable<Iterable<IndexableField>>)`: adds
    /// `docs` as one **document block** -- a run guaranteed to occupy
    /// contiguous, ascending doc IDs in the segment it lands in, which is what
    /// makes parent/child join queries expressible.
    ///
    /// Two things follow from Java, and both are ported. First, the whole block
    /// takes **one** sequence number, because it is one operation: an external
    /// reader sees all of the documents or none of them. Second, a block of
    /// more than one document sets `SegmentInfo.hasBlocks` on the segment it is
    /// flushed into -- `DocumentsWriterPerThread.updateDocuments` calls
    /// `segmentInfo.setHasBlocks()` whenever `numDocs > 1` -- which is how a
    /// reader knows the segment may contain blocks at all.
    ///
    /// Contiguity is guaranteed by never letting an automatic flush split the
    /// block: the threshold is consulted once, after the whole block is
    /// buffered, exactly as Java's `doAfterDocument` runs once per
    /// `updateDocuments` call rather than once per document.
    pub fn add_documents(&mut self, docs: Vec<Document>) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        self.add_documents_with_delete(None, docs)
    }

    /// `IndexWriter.updateDocuments(Term, Iterable<Iterable<IndexableField>>)`:
    /// atomically deletes every document matching `term` and adds `docs` as one
    /// block. One sequence number for the pair, so the delete and the add can
    /// never be observed apart.
    pub fn update_documents(&mut self, term: Term, docs: Vec<Document>) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        self.add_documents_with_delete(Some(DeleteNode::terms(vec![term])), docs)
    }

    /// The shared body of every add/update entry point -- Java's
    /// `IndexWriter.updateDocuments(Node, Iterable)` /
    /// `DocumentsWriterPerThread.updateDocuments`.
    ///
    /// The ordering here *is* the atomicity guarantee, and it is Java's:
    /// the delete node is applied with `docIDUpto` set to the buffer position
    /// **before** this call's documents (`finishDocuments(deleteNode,
    /// docsInRamBefore)`), so the delete reaches every document that already
    /// existed and none of the ones being added. Java achieves that by
    /// indexing first and then applying the slice at `docsInRamBefore`; with a
    /// single indexing thread, recording the limit up front and buffering
    /// after is the same thing (see `crate::buffered_updates`' module doc).
    fn add_documents_with_delete(
        &mut self,
        delete: Option<DeleteNode>,
        docs: Vec<Document>,
    ) -> Result<SeqNo> {
        // BLOCK-GUARD: `DocumentsWriterPerThread.updateDocuments` refuses a
        // block in a sorted index without a parent field before it reserves
        // or buffers anything; a buffered one could never be flushed.
        self.cfg.check_block(docs.len())?;
        self.reserve_docs(docs.len())?;
        let doc_id_upto = self.pending_doc_id_upto();
        let seq_no = match delete {
            Some(node) => self.buffer_delete_node(node, doc_id_upto),
            None => self.delete_queue.next_sequence_number(),
        };
        if docs.len() > 1 {
            self.pending_has_blocks = true;
        }
        let any = !docs.is_empty();
        for doc in docs {
            self.buffer_document(doc);
        }
        if any {
            self.mark_last_buffered_parent()?;
        }
        // Consulted once per call, never inside the loop: a flush partway
        // through a block would split it across two segments and break the
        // contiguity `has_blocks` promises.
        self.maybe_flush()?;
        Ok(seq_no)
    }

    /// The `docIDUpto` a delete issued right now must carry: the number of
    /// documents already buffered for the next segment. Java's
    /// `docsInRamBefore`.
    /// Counts `bytes` toward (`true`) or back out of the RAM buffer: the
    /// buffered postings a block of the document API grew.
    pub(crate) fn charge_ram_bytes(&mut self, bytes: usize, charge: bool) {
        self.ram_bytes_used = if charge {
            self.ram_bytes_used.saturating_add(bytes)
        } else {
            self.ram_bytes_used.saturating_sub(bytes)
        };
    }

    pub(crate) fn pending_doc_id_upto(&self) -> i32 {
        self.pending_docs.len() as i32
    }

    /// Buffers one delete node against both the private (next segment) and
    /// global (already-written segments) buffers, and returns its sequence
    /// number -- `DocumentsWriterDeleteQueue.add(Node, DeleteSlice)`.
    fn buffer_delete_node(&mut self, node: DeleteNode, doc_id_upto: i32) -> SeqNo {
        self.delete_queue.add_global_bytes(node.ram_bytes());
        match node {
            DeleteNode::Terms(terms) => self
                .delete_queue
                .add_shared_term_deletes(&terms, doc_id_upto),
            DeleteNode::Queries(queries) => {
                self.delete_queue.add_query_deletes(&queries, doc_id_upto)
            }
            DeleteNode::DocValuesUpdates(updates) => self
                .delete_queue
                .add_doc_values_updates(&updates, doc_id_upto),
        }
    }

    /// `IndexWriter.ramBytesUsed()`: the heap this writer is currently holding
    /// on behalf of buffered, not-yet-flushed documents.
    ///
    /// **What it counts, and why that differs from Java.** Java's
    /// `DocumentsWriterPerThread` inverts each document as it arrives, so its
    /// `Counter bytesUsed` measures the *inverted* form -- `ByteBlockPool` /
    /// `IntBlockPool` slices, `BytesRefHash` tables, per-field posting arrays --
    /// and `ramBufferSizeMB` is a bound on that. This port inverts at flush
    /// time instead (one shared pass over the whole batch, see
    /// [`IndexWriter::invert_pending_fields`]), so at `add_document` time the
    /// only structure that exists is the buffered [`Document`] arena. This
    /// therefore counts that arena exactly: the `Vec<Document>` slots plus every
    /// owned `String`/`Vec<u8>` capacity inside them, plus the parallel
    /// custom-freq term lists. It is a real byte count, not a sampled estimate,
    /// and it is O(1) to maintain (accumulated per `add_document`, not
    /// recomputed).
    ///
    /// The consequence for the flush trigger: the threshold bounds the
    /// *buffered document* bytes, and the transient inverted structure built
    /// during the flush is a multiple of that -- measured at 9.4x on
    /// `benchmarks/rust-runner`'s `index-bench` corpus, see
    /// [`crate::indexing_chain::InMemoryInvertedIndex::ram_bytes_used`]. Peak
    /// memory is therefore bounded by configuration and independent of how many
    /// documents the caller adds -- which is the property that was missing --
    /// but the constant is not 1.0 the way Java's is: a 16 MB setting yields a
    /// ~130 MB peak on that corpus, against ~860 MB and rising with no trigger
    /// at all. Closing the constant needs the incremental, per-document invert
    /// Java has, which in turn needs both a borrowed-token analyzer API and a
    /// byte-pool posting representation.
    pub fn ram_bytes_used(&self) -> usize {
        self.ram_bytes_used
    }

    /// `LiveIndexWriterConfig.setRAMBufferSizeMB(double)`: flush the buffered
    /// documents automatically once they occupy this many megabytes (see
    /// [`IndexWriter::ram_bytes_used`] for exactly what is measured).
    ///
    /// Pass [`DISABLE_AUTO_FLUSH_MB`] to turn the RAM trigger off. Java's two
    /// validations are ported verbatim: a non-sentinel value must be `> 0.0`,
    /// and turning *both* triggers off is refused.
    pub fn set_ram_buffer_size_mb(&mut self, mb: f64) -> Result<()> {
        if mb != DISABLE_AUTO_FLUSH_MB && mb <= 0.0 {
            return Err(Error::InvalidRamBufferSize(mb));
        }
        if mb == DISABLE_AUTO_FLUSH_MB && self.max_buffered_docs == DISABLE_AUTO_FLUSH {
            return Err(Error::BothAutoFlushTriggersDisabled);
        }
        self.ram_buffer_size_mb = mb;
        Ok(())
    }

    /// `LiveIndexWriterConfig.getRAMBufferSizeMB()`.
    pub fn ram_buffer_size_mb(&self) -> f64 {
        self.ram_buffer_size_mb
    }

    /// `LiveIndexWriterConfig.setMaxBufferedDocs(int)`: flush automatically once
    /// this many documents are buffered. Pass [`DISABLE_AUTO_FLUSH`] to turn the
    /// document-count trigger off. Java's validations are ported verbatim: a
    /// non-sentinel value must be at least 2, and turning *both* triggers off is
    /// refused.
    pub fn set_max_buffered_docs(&mut self, max_buffered_docs: i32) -> Result<()> {
        if max_buffered_docs != DISABLE_AUTO_FLUSH && max_buffered_docs < 2 {
            return Err(Error::InvalidMaxBufferedDocs(max_buffered_docs));
        }
        if max_buffered_docs == DISABLE_AUTO_FLUSH
            && self.ram_buffer_size_mb == DISABLE_AUTO_FLUSH_MB
        {
            return Err(Error::BothAutoFlushTriggersDisabled);
        }
        self.max_buffered_docs = max_buffered_docs;
        Ok(())
    }

    /// `LiveIndexWriterConfig.getMaxBufferedDocs()`.
    pub fn max_buffered_docs(&self) -> i32 {
        self.max_buffered_docs
    }

    /// Chooses the [`crate::index_file_deleter::DeletionPolicy`] this writer's
    /// deleter applies from here on -- `IndexWriterConfig.setIndexDeletionPolicy`.
    /// Applying it immediately (rather than only at the next commit) matches
    /// Java's `deleter.revisitPolicy()`.
    pub fn set_deletion_policy(&mut self, policy: DeletionPolicy) -> Result<()> {
        self.deleter.set_policy(policy)?;
        Ok(())
    }

    /// [`IndexWriter::set_deletion_policy`] with a pluggable policy: its
    /// `on_commit` runs over the live commits at once (`revisitPolicy`), then
    /// after every commit.
    pub fn set_index_deletion_policy(
        &mut self,
        policy: std::sync::Arc<dyn crate::deletion_policy::IndexDeletionPolicy>,
    ) -> Result<()> {
        self.deleter.set_custom_policy(policy)?;
        Ok(())
    }

    /// An `IndexCommit` for each commit point the deleter still holds, oldest
    /// first: the list a deletion policy is handed.
    pub fn index_commits(&self) -> Vec<crate::deletion_policy::IndexCommit> {
        self.deleter.index_commits()
    }

    /// The generations of the commit points this writer's deleter still
    /// holds, oldest first -- what `IndexWriter`'s deletion policy is handed
    /// as its `List<IndexCommit>`. Under [`DeletionPolicy::KeepAll`] the
    /// caller prunes them with [`IndexWriter::delete_commits`].
    pub fn commit_generations(&self) -> Vec<i64> {
        self.deleter.commit_generations()
    }

    /// `IndexCommit.delete()` for each named generation, from a deletion
    /// policy the caller runs (OpenSearch's `CombinedDeletionPolicy`). The
    /// newest commit always survives; files still named by a surviving commit,
    /// by the writer's uncommitted view or by a [`IndexWriter::hold_commit`]
    /// stay on disk.
    pub fn delete_commits(&mut self, generations: &[i64]) -> Result<()> {
        self.write_pending_try_changes()?;
        self.deleter.drop_commits(generations)?;
        Ok(())
    }

    /// Pins the segment files of commit `generation` -- what an open reader
    /// on that commit does to Java's `IndexFileDeleter` (`incRefDeleter`), so
    /// neither a later merge nor [`IndexWriter::delete_commits`] removes a
    /// file the reader, or a replica copying from it, still needs. The
    /// `segments_N` itself is not pinned: a reader has already read it.
    ///
    /// Returns the pinned files; hand them back to
    /// [`IndexWriter::release_files`] exactly once.
    pub fn hold_commit(&mut self, generation: i64) -> Result<Vec<String>> {
        self.write_pending_try_changes()?;
        let name = lucene_store::directory::segments_file_name(generation).ok_or(
            Error::Explicit(format!("no commit has generation {generation}")),
        )?;
        let infos = segment_infos::parse(&self.dir.open(&name)?, generation)?;
        Ok(self.deleter.hold_segment_files(&infos.segments)?)
    }

    /// Pins segment `sci`'s files, as [`IndexWriter::hold_commit`] pins a
    /// commit's; hand them back to [`IndexWriter::release_files`] once.
    pub(crate) fn hold_segment(&mut self, sci: &SegmentCommitInfo) -> Result<Vec<String>> {
        Ok(self.deleter.hold_segment_files(std::slice::from_ref(sci))?)
    }

    /// Releases a [`IndexWriter::hold_commit`] pin, deleting whatever nothing
    /// else still names.
    pub fn release_files(&mut self, files: &[String]) -> Result<()> {
        self.write_pending_try_changes()?;
        self.deleter.release_files(files)?;
        Ok(())
    }

    /// `IndexWriter.deleteUnusedFiles()`: re-apply the deletion policy
    /// (`revisitPolicy`) and
    /// re-scan `dir` for index files no live commit and no pending flush names,
    /// deleting them.
    ///
    /// This is the fallible counterpart of what [`IndexWriter::rollback`] does
    /// silently -- use it when you want a failure to reclaim disk space to
    /// surface as an error rather than be ignored.
    pub fn delete_unused_files(&mut self) -> Result<()> {
        self.write_pending_try_changes()?;
        self.release_nrt_holds()?;
        self.deleter.revisit_policy()?;
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
        self.deleter.refresh()?;
        Ok(())
    }

    /// The RAM buffer in bytes, `None` when flushing by RAM is off. The
    /// setter guarantees the value is either the sentinel (negative) or
    /// strictly positive, so `> 0.0` is the enabled test without a float
    /// equality comparison.
    fn ram_buffer_bytes(&self) -> Option<usize> {
        (self.ram_buffer_size_mb > 0.0)
            .then_some((self.ram_buffer_size_mb * 1024.0 * 1024.0) as usize)
    }

    /// `FlushByRamOrCountsPolicy.onChange(control, null)` after a buffered
    /// delete (`DocumentsWriterFlushControl.doOnDelete`), then
    /// `DocumentsWriter.applyAllDeletes`: once the buffered deletes alone
    /// reach the RAM buffer they are applied to the segments already written
    /// and stop counting. Deferred while a commit is prepared, as
    /// [`Self::maybe_flush`] is.
    fn maybe_apply_deletes(&mut self) -> Result<()> {
        if self.prepared_commit.is_some() {
            return Ok(());
        }
        if self
            .ram_buffer_bytes()
            .is_some_and(|limit| self.delete_queue.global_ram_bytes() >= limit)
        {
            self.apply_deletes_now()?;
        }
        Ok(())
    }

    /// `DocumentsWriter.applyAllDeletes`: freezes the deletes buffered for
    /// the written segments and applies them now, checkpointing the files
    /// that writes -- what a flush of an empty buffer does.
    fn apply_deletes_now(&mut self) -> Result<()> {
        self.ensure_write_lock_valid()?;
        self.apply_all_deletes_and_updates()?;
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
        Ok(())
    }

    /// `FlushByRamOrCountsPolicy.onChange`: document count first, then RAM
    /// -- the buffered documents and the buffered deletes together --
    /// exactly Java's precedence.
    fn maybe_flush(&mut self) -> Result<()> {
        // **Never while a commit is prepared.** [`IndexWriter::finish_commit`]
        // publishes the `SegmentInfos` snapshot [`IndexWriter::prepare_commit`]
        // took and clears `flushed_segments`, so a flush in between would have
        // its segment *and* every buffered delete it resolved thrown away --
        // silently. Deferring keeps both: the documents stay in the buffer and
        // the deletes stay in the queue, and they land in the next commit,
        // which is where Java puts them too (they are not in `pendingCommit`).
        //
        // The cost is that the RAM/document thresholds do not apply inside the
        // prepare -> finish window, so the buffer can grow past them there. The
        // window is a two-phase commit's activation step, by design a short one;
        // an explicit `flush()` in it is refused outright rather than deferred,
        // since a caller asking for it deserves to be told.
        if self.prepared_commit.is_some() {
            return Ok(());
        }
        if self.max_buffered_docs != DISABLE_AUTO_FLUSH
            && self.pending_docs.len() >= self.max_buffered_docs as usize
        {
            return self.flush();
        }
        if let Some(limit) = self.ram_buffer_bytes() {
            let active = self.ram_bytes_used;
            let deletes = self.delete_queue.global_ram_bytes();
            // Both over: Java applies the deletes and flushes the buffer; the
            // flush freezes and applies them itself.
            if deletes >= limit && active < limit {
                return self.apply_deletes_now();
            }
            if active.saturating_add(deletes) >= limit {
                return self.flush();
            }
        }
        // `hardMaxBytesPerDWPT`, whatever the RAM setting.
        if self.ram_bytes_used > crate::concurrent_writer::HARD_MAX_BYTES_PER_SLOT {
            return self.flush();
        }
        Ok(())
    }

    /// Java's `IndexWriter.checkpoint()` (a *non*-commit checkpoint, which
    /// rolls `lastFiles` forward to the new segment list) followed by the
    /// commit checkpoint for the generation just published.
    ///
    /// Both halves are needed. The commit checkpoint alone would leave the
    /// previous non-commit checkpoint's `lastFiles` still holding a reference on
    /// segments the new commit dropped -- a merge's sources, most visibly --
    /// so their count would never reach zero and they would never be deleted.
    /// Java gets the first half from `commitMerge`/`publishFlushedSegment`
    /// calling `checkpoint()` before the commit ever happens; this port's
    /// `apply_merge`/`delete_documents`/`update_document` publish a commit in
    /// the same call, so both halves land here.
    /// Both halves run against `self.segment_infos`, which every caller
    /// installs *before* calling this: the commit is already durable on disk by
    /// then, so a failure to reclaim files must not leave this writer's
    /// in-memory view behind the directory's. The error still propagates -- the
    /// caller learns the sweep failed -- but what it describes is unreclaimed
    /// disk space, not a lost commit.
    ///
    /// The non-commit half covers the **live** view -- the commit plus every
    /// segment flushed since -- as Java's does: its `segmentInfos` holds
    /// flushed segments too. Covering only the commit took the references of
    /// every flushed-but-uncommitted segment away, and the deleter deleted
    /// their files. A single-threaded writer never noticed, because its merges
    /// only ever committed right after a commit had taken every flushed segment
    /// in; a merge committing while another thread's segment was pending
    /// deleted it.
    fn checkpoint_committed(&mut self) -> Result<()> {
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
        let published = self.segment_infos.clone();
        self.deleter.checkpoint(&published, true)?;
        Ok(())
    }

    /// This writer's in-memory view of the index: the last commit plus every
    /// segment [`IndexWriter::flush`] has written since. Java keeps exactly this
    /// in `segmentInfos`; see [`Self::flushed_segments`] for why this port keeps
    /// the tail separate.
    pub(crate) fn live_infos(&self) -> SegmentInfos {
        let mut infos = self.segment_infos.clone();
        infos.segments.extend(self.flushed_segments.iter().cloned());
        infos
    }

    /// Buffers `doc` for the next [`IndexWriter::commit`], same as
    /// [`IndexWriter::add_document`], and additionally records `terms` -- a
    /// list of `(term, custom_freq)` pairs -- as this doc's explicit input to
    /// [`IndexWriter::set_custom_freq_postings_field`]'s opted-in field (if
    /// any is configured; `terms` is ignored, harmlessly, if
    /// `custom_freq_postings_field` is `None`, matching
    /// [`IndexWriter::set_postings_field`]'s own "best effort per document"
    /// convention of silently contributing nothing when nothing is opted
    /// in).
    ///
    /// `custom_freq` is this port's port of real Lucene's opaque
    /// `DocsAndCustomFreqs` "freq" value -- an arbitrary per-doc-per-term
    /// integer a similarity implementation interprets however it likes,
    /// **not** a literal term-occurrence count (see
    /// `crate::postings_writer`'s module doc comment's
    /// `IndexOptions::DocsAndCustomFreqs` section). It must be `>= 1`
    /// (Java's "Term frequency must be 1 or greater"), checked here before
    /// anything is buffered: the wire encoding has no representation for a
    /// zero-or-negative freq, and a value the codec refused at flush would
    /// leave the buffer unflushable. A `term` repeated in `terms` sums its
    /// freqs, as `FreqProxTermsWriterPerField` does with `Math.addExact`; a
    /// sum past `i32::MAX` is refused the same way.
    ///
    /// A plain [`IndexWriter::add_document`] call is equivalent to calling
    /// this with an empty `terms` list -- both keep
    /// [`Self::pending_custom_freq_terms`] aligned with `pending_docs` by
    /// index.
    pub fn add_document_with_custom_freq_terms(
        &mut self,
        doc: Document,
        terms: Vec<(String, i32)>,
    ) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        summed_custom_freq_terms(&terms)?;
        let seq_no = self.delete_queue.next_sequence_number();
        // ARITH: see [`VectorValue::ram_bytes`].
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.ram_bytes_used += document_ram_bytes(&doc) + custom_freq_terms_ram_bytes(&terms);
        }
        self.pending_docs.push(doc);
        self.pending_custom_freq_terms.push(terms);
        self.pending_vectors.push(Vec::new());
        self.pending_explicit.push(ExplicitFields::default());
        self.mark_last_buffered_parent()?;
        self.maybe_flush()?;
        Ok(seq_no)
    }

    /// `IndexWriter.updateDocument(Term, doc)`: atomically deletes every
    /// document matching `term` and adds `doc`, as one operation with one
    /// sequence number.
    ///
    /// **Buffered**, exactly as Java's is. Both halves land in the delete queue
    /// and the document buffer, and become visible together at the next
    /// [`IndexWriter::commit`] (or at whichever automatic flush the buffer
    /// thresholds trigger first). The delete carries the buffer position it was
    /// issued at, so it reaches every document added before this call and none
    /// added after -- including `doc` itself, which is why "update" is a
    /// replacement rather than a self-cancelling pair.
    ///
    /// The eager, immediately-committed variant this writer had before
    /// sequence numbers existed is still available as
    /// [`IndexWriter::update_document_with_sources`], for a caller that holds
    /// an opened reader over a segment this writer cannot open itself.
    pub fn update_document(&mut self, term: Term, doc: Document) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        self.add_documents_with_delete(Some(DeleteNode::terms(vec![term])), vec![doc])
    }

    /// `IndexWriter.deleteDocuments(Term...)`: buffers a delete for every
    /// document matching any of `terms`, and returns the operation's sequence
    /// number.
    ///
    /// The deletes apply to every segment that exists when this is called and
    /// to every document already buffered, and to nothing added afterwards --
    /// see [`crate::buffered_updates`] for the two mechanisms that produce
    /// that (`docIDUpto` within the in-progress segment, the frozen packet's
    /// generation across segments).
    pub fn delete_documents_by_term(&mut self, terms: &[Term]) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        let doc_id_upto = self.pending_doc_id_upto();
        let seq_no = self.buffer_delete_node(DeleteNode::terms(terms.to_vec()), doc_id_upto);
        self.maybe_apply_deletes()?;
        Ok(seq_no)
    }

    /// `IndexWriter.deleteDocuments(Query...)`.
    ///
    /// Java's LUCENE-6379 specialisation is ported: a `MatchAllDocsQuery`
    /// anywhere in `queries` short-circuits to [`IndexWriter::delete_all`],
    /// which drops whole segments instead of writing an all-zero `.liv` for
    /// each. `delete_all` is not a buffered operation, so -- as in Java -- the
    /// returned sequence number is the one `deleteAll` itself takes.
    ///
    /// See [`DeleteQuery`] for why this port's query type is a closed enum
    /// rather than `lucene_search::Query`.
    pub fn delete_documents_by_query(&mut self, queries: &[DeleteQuery]) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        if queries.iter().any(|q| matches!(q, DeleteQuery::MatchAll)) {
            let seq_no = self.delete_queue.next_sequence_number();
            self.delete_all()?;
            return Ok(seq_no);
        }
        let doc_id_upto = self.pending_doc_id_upto();
        let seq_no = self.buffer_delete_node(DeleteNode::Queries(queries.to_vec()), doc_id_upto);
        self.maybe_apply_deletes()?;
        Ok(seq_no)
    }

    /// `IndexWriter.softUpdateDocument(Term, doc, Field... softDeletes)`:
    /// adds `doc` and, instead of deleting the documents matching `term`,
    /// applies `soft_deletes` to them as **doc-values updates**.
    ///
    /// This is the whole soft-delete mechanism: the previous version of the
    /// document keeps its postings and its live bit, and is instead marked
    /// through a doc-values field a retention policy can consult later
    /// (`SoftDeletesRetentionMergePolicy`, and this port's
    /// `lucene_search::soft_deletes`). Java builds the update nodes with
    /// `buildDocValuesUpdate(term, softDeletes)` and routes them through the
    /// very same `updateDocuments(Node, docs)` path a hard update uses, so the
    /// add and the marking share one sequence number -- ported exactly.
    ///
    /// Java's two argument checks are ported verbatim: the term must be
    /// present (it always is, here -- the type requires it) and at least one
    /// soft-delete field must be supplied
    /// ([`Error::NoSoftDeletesSupplied`] for Java's
    /// `IllegalArgumentException("at least one soft delete must be present")`).
    pub fn soft_update_document(
        &mut self,
        term: Term,
        doc: Document,
        soft_deletes: &[DocValuesUpdate],
    ) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        self.soft_update_documents(term, vec![doc], soft_deletes)
    }

    /// `IndexWriter.softUpdateDocuments(Term, Iterable<Iterable<…>>, Field...)`
    /// -- [`IndexWriter::soft_update_document`] over a whole document block.
    pub fn soft_update_documents(
        &mut self,
        term: Term,
        docs: Vec<Document>,
        soft_deletes: &[DocValuesUpdate],
    ) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        if soft_deletes.is_empty() {
            return Err(Error::NoSoftDeletesSupplied);
        }
        // Java's `buildDocValuesUpdate(term, softDeletes)`: every supplied
        // field becomes an update *keyed by the same term*, so the marking hits
        // exactly the documents a hard `updateDocument` would have deleted.
        // `buildDocValuesUpdate` also runs `verifyOrCreateDvOnlyField` on each
        // one: an index-sort or wrong-typed field is refused here rather than
        // at apply time, where the failure would be restored and fail every
        // later flush. A field a fixed field list does not name stays
        // accepted, as it always was here -- the soft-deletes field of a
        // caller that never declared it (the FFI's soft updates do this).
        self.create_doc_values_only_fields(soft_deletes)?;
        let mut updates: Vec<DocValuesUpdate> = Vec::with_capacity(soft_deletes.len());
        for u in soft_deletes {
            if self.cfg.field_by_name(u.field()).is_some() {
                self.cfg.verify_doc_values_update_field(u)?;
            }
            updates.push(retarget_update(u, &term));
        }
        self.add_documents_with_delete(Some(DeleteNode::DocValuesUpdates(updates)), docs)
    }

    /// `IndexWriter.updateDocValues(Term, Field... updates)`: sets every
    /// document matching `term`'s doc-values value for each updated field.
    ///
    /// A [`DocValuesUpdate`] whose value is `None` **removes** the field's
    /// value from the matched documents rather than setting it to zero --
    /// Java's "If a doc values fields data is `null` the existing value is
    /// removed from all documents matching the term", which reaches
    /// `DocValuesFieldUpdates.reset(doc)`.
    ///
    /// Java also validates that each field exists and is a doc-values-only
    /// NUMERIC/BINARY field (`verifyOrCreateDvOnlyField`) and that it is not
    /// part of the index sort. This port checks the same two things against
    /// its field list: [`Error::UnknownDocValuesUpdateField`] and
    /// [`Error::WrongDocValuesUpdateType`]. With explicit documents (the
    /// document API's mode, whose schema grows as fields arrive) an unknown
    /// field is created as a doc-values-only field first, as Java's
    /// `buildDocValuesUpdate` (`verifyOrCreateDvOnlyField(name, type,
    /// false)`) creates it; a fixed field list refuses it.
    /// [`Self::update_numeric_doc_value`] and [`Self::update_binary_doc_value`]
    /// never create one (Java passes `fieldMustExist`).
    pub fn update_doc_values(&mut self, term: Term, updates: &[DocValuesUpdate]) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        if updates.is_empty() {
            return Err(Error::NoDocValuesUpdatesSupplied);
        }
        self.create_doc_values_only_fields(updates)?;
        self.update_existing_doc_values(term, updates)
    }

    /// [`Self::update_doc_values`] of fields that must exist.
    fn update_existing_doc_values(
        &mut self,
        term: Term,
        updates: &[DocValuesUpdate],
    ) -> Result<SeqNo> {
        let mut retargeted = Vec::with_capacity(updates.len());
        for update in updates {
            self.cfg.verify_doc_values_update_field(update)?;
            retargeted.push(retarget_update(update, &term));
        }
        let doc_id_upto = self.pending_doc_id_upto();
        let seq_no = self.buffer_delete_node(DeleteNode::DocValuesUpdates(retargeted), doc_id_upto);
        self.maybe_apply_deletes()?;
        Ok(seq_no)
    }

    /// `verifyOrCreateDvOnlyField(name, type, false)`'s create half, for a
    /// writer on explicit documents: each update naming a field nothing has
    /// registered registers it, not indexed, with the update's doc-values
    /// type.
    pub(crate) fn create_doc_values_only_fields(
        &mut self,
        updates: &[DocValuesUpdate],
    ) -> Result<()> {
        if !self.cfg.explicit {
            return Ok(());
        }
        for update in updates {
            if self.registered_field(update.field()).is_some() {
                continue;
            }
            let doc_values_type = match update {
                DocValuesUpdate::Numeric { .. } => DocValuesType::Numeric,
                DocValuesUpdate::Binary { .. } => DocValuesType::Binary,
            };
            self.register_field(FieldInfo::new(update.field(), 0).with_doc_values(
                doc_values_type,
                lucene_codecs::field_infos::DocValuesSkipIndexType::None,
                -1,
            ))?;
        }
        Ok(())
    }

    /// `IndexWriter.updateNumericDocValue(Term, String, long)`.
    pub fn update_numeric_doc_value(
        &mut self,
        term: Term,
        field: &str,
        value: i64,
    ) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        let update = DocValuesUpdate::Numeric {
            term: term.clone(),
            field: field.to_string(),
            value: Some(value),
        };
        self.update_existing_doc_values(term, std::slice::from_ref(&update))
    }

    /// `IndexWriter.updateBinaryDocValue(Term, String, BytesRef)`. Java
    /// rejects a null value outright here (unlike `updateDocValues`, which
    /// treats null as a removal), so this takes a `&[u8]` and has no
    /// "remove" spelling -- use [`IndexWriter::update_doc_values`] with a
    /// `None` value for that.
    ///
    /// An index-sort field is refused before the field's existence and
    /// doc-values type are checked, so it surfaces as
    /// [`Error::DocValuesUpdateOnIndexSortField`], not as a type mismatch
    /// (Lucene 10.6, branch_10_6, #16565).
    pub fn update_binary_doc_value(
        &mut self,
        term: Term,
        field: &str,
        value: &[u8],
    ) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        self.cfg.verify_not_index_sort_field(field)?;
        let update = DocValuesUpdate::Binary {
            term: term.clone(),
            field: field.to_string(),
            value: Some(value.to_vec()),
        };
        self.update_existing_doc_values(term, std::slice::from_ref(&update))
    }

    /// The atomic delete-by-term + add-document real Lucene calls
    /// `updateDocument`, committed at once: the term delete is resolved
    /// against every segment `delete_sources` supplies an opened source for
    /// and `doc` is flushed as a brand-new segment
    /// ([`update_document::update_segments`]), then both publish in one
    /// `segments_N` through the same commit tail as
    /// [`IndexWriter::delete_documents_with_sources`] (version above every NRT
    /// snapshot, minimum segment version and soft-delete counts stamped).
    /// Unlike [`IndexWriter::add_document`], this is **not** buffered.
    ///
    /// Live docs in `delete_sources` must be current: a source whose live
    /// docs disagree with the commit's delete count is refused
    /// ([`term_delete::Error::StaleLiveDocs`]).
    ///
    /// On `Err` nothing is published: the segment list, version and commit id
    /// are unchanged. The segment name this attempt claimed is burned, as
    /// `IndexWriter.newSegmentName()` burns it, and a failed `segments_N`
    /// write still advances the generation (`startCommit`), so a retry never
    /// writes over files the failed attempt may have left.
    #[allow(clippy::too_many_arguments)]
    pub fn update_document_with_sources(
        &mut self,
        delete_sources: &[SegmentDeleteSource],
        field: &str,
        term: &[u8],
        new_doc: Document,
    ) -> Result<&SegmentInfos> {
        self.write_pending_try_changes()?;
        if self.prepared_commit.is_some() {
            return Err(Error::PreparedCommitPending("update_document"));
        }
        let new_segment_name = self.new_segment_name();
        let new_segment_id = generate_segment_id(self.segment_infos.counter);

        // `new_segment_name` bumped `segment_infos.counter`, so the commit
        // below carries a counter past the new segment and a fresh writer
        // never hands out its name again.
        let updated_segments = update_document::update_segments(
            self.dir,
            &self.segment_infos,
            delete_sources,
            field,
            term,
            update_document::NewSegment {
                name: &new_segment_name,
                id: new_segment_id,
                codec_name: &self.cfg.codec_name,
                lucene_version: self.cfg.lucene_version,
                fields: &self.cfg.fields,
                docs: std::slice::from_ref(&new_doc),
                use_compound_file: self.cfg.use_compound_file,
            },
        )?;
        self.commit_updated_segments(updated_segments)
    }

    /// Deletes every live doc matching `(field, term)` in whichever of this
    /// writer's current segments `delete_sources` supplies an opened source
    /// for -- delegates to
    /// [`term_delete::resolve_and_apply_term_delete`] per matching segment,
    /// then commits the whole updated segment list as one new `segments_N`
    /// generation (same atomicity shape as
    /// [`IndexWriter::update_document`]: either every targeted segment's
    /// `.liv` update lands in the same commit, or -- on the first failure --
    /// nothing commits and this writer's state is unchanged).
    ///
    /// A segment with no matching entry in `delete_sources` is left
    /// untouched (same "caller supplies whatever it has open" scope as
    /// [`update_document::SegmentDeleteSource`]'s own doc comment).
    pub fn delete_documents_with_sources(
        &mut self,
        delete_sources: &[SegmentDeleteSource],
        field: &str,
        term: &[u8],
    ) -> Result<&SegmentInfos> {
        self.write_pending_try_changes()?;
        if self.prepared_commit.is_some() {
            return Err(Error::PreparedCommitPending("delete_documents"));
        }
        let mut updated_segments = Vec::with_capacity(self.segment_infos.segments.len());
        for sci in &self.segment_infos.segments {
            match delete_sources
                .iter()
                .find(|src| src.segment_name == sci.segment_name)
            {
                Some(src) => {
                    let updated = term_delete::resolve_and_apply_term_delete(
                        self.dir,
                        sci,
                        src.fields,
                        src.doc_in,
                        src.live_docs,
                        src.max_doc,
                        field,
                        term,
                    )?;
                    updated_segments.push(updated);
                }
                None => updated_segments.push(sci.clone()),
            }
        }

        self.commit_updated_segments(updated_segments)
    }

    /// The commit tail of the with-sources update and delete: publishes
    /// `updated_segments` as the next generation, with the version kept above
    /// every NRT snapshot's, the minimum segment version and soft-delete
    /// counts stamped, and the generation advanced on error as on success.
    fn commit_updated_segments(
        &mut self,
        updated_segments: Vec<SegmentCommitInfo>,
    ) -> Result<&SegmentInfos> {
        let mut new_segment_infos = self.segment_infos.clone();
        // ARITH: `segment_infos::parse` rejects any generation, version or
        // counter outside `-1..=MAX_GENERATION` (`i64::MAX / 2`), and
        // `segment_infos::write` refuses to persist one, so every value this
        // writer can be holding is at most `i64::MAX / 2` -- climbing from
        // there to `i64::MAX` would take 2^62 further commits. See
        // `segment_infos::MAX_GENERATION`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            new_segment_infos.generation += 1;
            new_segment_infos.version += 1;
        }
        // Above every version an NRT snapshot was given, so `getVersion`
        // stays monotonic across snapshots and commits (see `crate::nrt`).
        // Never used with NRT readers, `max_version` is -1 and this is a
        // no-op.
        new_segment_infos.version = new_segment_infos
            .version
            .max(self.nrt.max_version.saturating_add(1));
        new_segment_infos.id = generate_segment_id(new_segment_infos.generation);
        new_segment_infos.segments = updated_segments;
        self.stamp_min_segment_version(&mut new_segment_infos)?;
        self.stamp_soft_delete_counts(&mut new_segment_infos)?;
        // This commits on its own: sync what the commit names first
        // (`startCommit`).
        self.deleter.sync_commit_files(&new_segment_infos)?;
        new_segment_infos.format_version = self.segment_infos_format_version();
        let written = segment_infos::write(&new_segment_infos, self.dir);
        // `updateGeneration` on error as on success (`startCommit`).
        self.segment_infos.generation = new_segment_infos.generation;
        written?;
        self.last_commit_change_count = self.change_count;

        self.segment_infos = new_segment_infos;
        // `finishCommit`: `rollbackSegments =
        // pendingCommit.createBackupSegmentInfos()`. This is the one place a
        // commit becomes durable, so it is the one place the rollback snapshot
        // moves forward.
        self.rollback_segments = self.segment_infos.segments.clone();
        self.checkpoint_committed()?;
        Ok(&self.segment_infos)
    }

    /// Flushes every currently-buffered [`IndexWriter::add_document`] call (if
    /// any) to a brand-new segment via
    /// [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment),
    /// appends it to this writer's segment list, and writes the whole updated
    /// list as the next `segments_N` generation via
    /// [`crate::segment_infos::write`] -- real Lucene's `IndexWriter.commit()`
    /// after one or more buffered `DocumentsWriterPerThread.flush()`-worth of
    /// documents. If this writer has a [`MergePolicyConfig`] set via
    /// [`IndexWriter::set_merge_policy`], this also performs real `commit()`'s
    /// automatic merge-triggering step (see module doc comment) right after the
    /// flush commits; with no merge policy set (the default), this method is
    /// unchanged from before that feature existed.
    ///
    /// A `commit()` with nothing to commit -- no buffered documents, deletes
    /// or updates, no segment flushed, merged or changed since the last
    /// commit, and no [`IndexWriter::delete_all`],
    /// [`IndexWriter::set_live_commit_data`] or `add_indexes` call since --
    /// writes nothing and leaves the current generation in place, as Java's
    /// does ("skip startCommit(): no changes pending"); it still succeeds.
    /// The first commit of a writer opened on a directory with no commit
    /// always writes. Returns the current committed [`SegmentInfos`].
    ///
    /// Durability: the commit fsyncs every file it names that this writer
    /// wrote and has not synced before (each file at most once), then
    /// publishes `segments_N` and syncs the directory. Flushes, merges and
    /// `.liv`/doc-values generations are never synced on their own: only
    /// committed data is durable.
    ///
    /// If [`IndexWriter::set_postings_field`]/[`IndexWriter::add_postings_field`]
    /// have opted this writer into postings for one or more fields, this also
    /// builds and writes those fields' real `.doc`/`.tim`/`.tip`/`.tmd` for
    /// the flushed segment in one batched
    /// [`lucene_codecs::postings_writer::write_fields`] call (see
    /// [`IndexWriter::build_postings_output`]/
    /// [`IndexWriter::write_postings_files`]) -- entirely in memory *before*
    /// anything is written to `dir`, so any validation failure across *any*
    /// configured field (see [`postings_writer::write_fields`]'s doc comment
    /// for the current set of rejected shapes) makes the **whole** `commit()`
    /// call fail with `Err` and leaves `dir`/`pending_docs`/`segment_infos`
    /// completely unchanged, exactly like [`IndexWriter::update_document`]'s
    /// own atomicity guarantee -- never a partially-written segment.
    ///
    /// Equivalent to [`IndexWriter::prepare_commit`] immediately followed by
    /// [`IndexWriter::finish_commit`] -- kept as one call for every caller
    /// that doesn't need the two-phase split (see those two methods' own
    /// doc comments for what "two-phase" honestly means on this port's
    /// on-disk format).
    pub fn commit(&mut self) -> Result<&SegmentInfos> {
        self.write_pending_try_changes()?;
        self.prepare_commit()?;
        self.finish_commit()
    }

    /// `SegmentInfos.write`'s `minSegmentLuceneVersion`: the oldest version
    /// any segment of the commit was written by, from each segment's own
    /// `.si`. The value read from the previous `segments_N` is no substitute:
    /// once another engine's or another version's segments are merged away
    /// (or this writer's join an index real Lucene created), it is simply
    /// wrong, and `CheckIndex` rejects a minimum above any segment's version.
    /// `SegmentCommitInfo.softDelCount` for every segment of `infos` about to
    /// be committed: the documents that carry a value in the soft-deletes field
    /// and are not hard-deleted -- what Lucene maintains through
    /// `PendingSoftDeletes` (at flush, `DocumentsWriterPerThread` counts the
    /// documents born soft-deleted; each doc-values update and hard delete
    /// adjusts it) and what `CheckIndex` verifies.
    ///
    /// Recounted from the segment's current soft-deletes column and `.liv`,
    /// but only for a segment whose deletes or doc values changed since the
    /// last count (cached by generation). A count loaded from an existing
    /// commit is trusted until the segment changes. A writer with no
    /// soft-deletes field writes `0`, as before.
    fn stamp_soft_delete_counts(&mut self, infos: &mut SegmentInfos) -> Result<()> {
        let Some(soft) = self.cfg.fields.iter().find(|f| f.soft_deletes_field) else {
            return Ok(());
        };
        let soft_name = soft.name.clone();
        for sci in &mut infos.segments {
            let key = soft_delete_key(sci);
            if let Some(&(k, count)) = self.soft_delete_counts.get(&sci.segment_name) {
                if k == key {
                    sci.soft_del_count = count;
                    continue;
                }
            }
            let count =
                count_soft_deletes(self.dir, sci, &soft_name, None)?.unwrap_or(sci.soft_del_count);
            sci.soft_del_count = count;
            self.soft_delete_counts
                .insert(sci.segment_name.clone(), (key, count));
        }
        let names: std::collections::HashSet<&str> = infos
            .segments
            .iter()
            .map(|s| s.segment_name.as_str())
            .collect();
        self.soft_delete_counts.retain(|name, _| {
            names.contains(name.as_str())
                || self
                    .flushed_segments
                    .iter()
                    .any(|s| &s.segment_name == name)
        });
        Ok(())
    }

    /// `(segments in the view, cached versions, segments the deleter tracks)`:
    /// what a resource-bound test holds the per-segment state to.
    #[cfg(test)]
    pub(crate) fn per_segment_sizes(&self) -> (usize, usize, usize) {
        let view = self.live_infos().segments.len();
        (
            view,
            self.segment_versions.len(),
            self.deleter.tracked_sizes().0,
        )
    }

    /// Drops the versions of segments this writer's view no longer holds.
    /// A merge takes its sources out of the view, and so does applying
    /// deletes that empty a segment: without this their entries lived until
    /// the next `segments_N` was stamped, and the concurrent writer's merges
    /// and flushes stamp nothing, so they piled up until a commit.
    fn prune_segment_versions(&mut self) {
        let in_view: std::collections::HashSet<&str> = self
            .segment_infos
            .segments
            .iter()
            .chain(&self.flushed_segments)
            .map(|s| s.segment_name.as_str())
            .collect();
        self.segment_versions
            .retain(|name, _| in_view.contains(name.as_str()));
    }

    fn stamp_min_segment_version(&mut self, infos: &mut SegmentInfos) -> Result<()> {
        // A cache (a miss re-reads the `.si`), kept to the segments still in
        // play -- the commit being written and this writer's live view -- so
        // it does not grow with every segment ever flushed or merged.
        {
            let in_play: std::collections::HashSet<&str> = infos
                .segments
                .iter()
                .chain(&self.segment_infos.segments)
                .chain(&self.flushed_segments)
                .map(|s| s.segment_name.as_str())
                .collect();
            self.segment_versions
                .retain(|name, _| in_play.contains(name.as_str()));
        }
        let mut min: Option<segment_infos::LuceneVersion> = None;
        for sci in &infos.segments {
            let version = match self.segment_versions.get(&sci.segment_name) {
                Some(&v) => v,
                None => {
                    let si_bytes = self.dir.open(&format!("{}.si", sci.segment_name))?;
                    let v = to_segment_infos_version(
                        segment_info::parse_for_commit(&si_bytes, sci)?.version,
                    );
                    self.segment_versions.insert(sci.segment_name.clone(), v);
                    v
                }
            };
            let key = |v: segment_infos::LuceneVersion| (v.major, v.minor, v.bugfix);
            if min.is_none_or(|m| key(version) < key(m)) {
                min = Some(version);
            }
        }
        infos.min_segment_lucene_version = min;
        Ok(())
    }

    /// The file-writing half of a real two-phase commit, port of real Lucene's
    /// `IndexWriter.prepareCommit()`: does every single thing
    /// [`IndexWriter::commit`] does *except* the final
    /// [`crate::segment_infos::write`] call that actually produces the next
    /// `segments_N` -- flushes `pending_docs` (if any) to a brand-new segment
    /// via
    /// [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment),
    /// builds and writes that segment's postings/term-vector/doc-values files
    /// exactly as [`IndexWriter::commit`] always has, and stashes the resulting
    /// in-memory [`SegmentInfos`] (bumped generation/version, new segment
    /// appended) in `self.prepared_commit` for [`IndexWriter::finish_commit`]
    /// to pick up.
    ///
    /// # The on-disk protocol
    ///
    /// This is Lucene's real two-phase shape, not an in-memory imitation of
    /// it. After flushing, this syncs every file the commit names that this
    /// writer has not synced yet (`IndexWriter.startCommit`'s
    /// `directory.sync(toSync.files(false))`; see
    /// [`crate::index_file_deleter::IndexFileDeleter::sync_commit_files`]),
    /// then serializes the whole commit and writes it -- fsynced -- as
    /// `pending_segments_N` (via [`crate::segment_infos::write_pending`]).
    /// `pending_segments_N` is deliberately not a name
    /// [`lucene_store::directory::last_commit_generation`] scans for, so:
    ///
    /// - after `prepare_commit()` returns, a fresh
    ///   [`IndexWriter::open`]/[`crate::segment_infos::read_latest`] on the
    ///   same `dir` still returns the *previous* commit, unchanged; and
    /// - a crash anywhere during this method, or between it and
    ///   [`IndexWriter::finish_commit`], leaves the previous commit current
    ///   and the pending file an inert orphan. There is no window in which a
    ///   truncated `segments_N` exists under a name a reader would pick up --
    ///   which is exactly why Java never creates a `segments_N` by writing to
    ///   it directly.
    ///
    /// [`IndexWriter::finish_commit`] then performs the single atomic publish:
    /// `rename(pending_segments_N -> segments_N)` plus a directory fsync.
    ///
    /// What is still out of scope, unlike real Lucene:
    /// - **No cross-process/cross-restart handoff.** `prepare_commit()` on one
    ///   `IndexWriter` and `finish_commit()` on another (e.g. after a restart)
    ///   is not supported -- `prepared_commit` lives on `self`, and nothing
    ///   here scans for a leftover `pending_segments_N` to roll forward or
    ///   clean up (Java gives that job to `IndexFileDeleter`, which this port
    ///   does not have). A leftover pending file is harmless but is never
    ///   reclaimed.
    ///
    /// With nothing to commit (see [`IndexWriter::commit`]) this writes
    /// nothing, and the [`IndexWriter::finish_commit`] that follows succeeds
    /// without writing either.
    ///
    /// Calling `prepare_commit()` again while a previous prepared commit has
    /// not been activated by [`IndexWriter::finish_commit`] is an error
    /// ([`Error::PrepareCommitAlreadyCalled`]), matching real
    /// `prepareCommitInternal`'s `IllegalStateException("prepareCommit was
    /// already called with no corresponding call to commit")` -- call
    /// `finish_commit()` to activate it or [`IndexWriter::rollback`] to
    /// discard it (which also deletes the pending file) first.
    pub fn prepare_commit(&mut self) -> Result<()> {
        self.write_pending_try_changes()?;
        self.prepare_commit_excluding(&std::collections::HashSet::new())
    }

    /// [`IndexWriter::prepare_commit`], with merge-on-commit leaving alone
    /// the segments in `merging` -- merges a
    /// [`crate::concurrent_writer::ConcurrentIndexWriter`] is running.
    pub(crate) fn prepare_commit_excluding(
        &mut self,
        merging: &std::collections::HashSet<String>,
    ) -> Result<()> {
        self.write_pending_try_changes()?;
        // Real `prepareCommitInternal` refuses re-entry outright
        // (`IllegalStateException("prepareCommit was already called with no
        // corresponding call to commit")`). Replacing the pending state
        // instead would silently throw away every document the first prepare
        // had already flushed: the second prepare rebuilds its
        // `SegmentInfos` from `self.segment_infos`, which never saw the first
        // prepare's segment, so `finish_commit` published a commit that did not
        // reference it.
        if self.prepared_commit.is_some() {
            return Err(Error::PrepareCommitAlreadyCalled);
        }
        self.prepared_nothing = false;
        self.ensure_write_lock_valid()?;

        // Everything still buffered becomes one last segment; segments an
        // automatic flush already wrote are folded in below.
        self.flush()?;
        // `prepareCommitInternal`'s point-in-time merges
        // (`maxFullFlushMergeWaitMillis > 0`): the commit names the merged
        // segments.
        self.merge_on_full_flush(merge_policy::MergeTrigger::Commit, merging)?;
        // After the flush, which may be the one that makes `auto`'s choice.
        self.stamp_adaptive_doc_values_choice();

        let mut new_segment_infos = self.segment_infos.clone();
        // ARITH: `segment_infos::parse` rejects any generation, version or
        // counter outside `-1..=MAX_GENERATION` (`i64::MAX / 2`), and
        // `segment_infos::write` refuses to persist one, so every value this
        // writer can be holding is at most `i64::MAX / 2` -- climbing from
        // there to `i64::MAX` would take 2^62 further commits. See
        // `segment_infos::MAX_GENERATION`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            new_segment_infos.generation += 1;
            new_segment_infos.version += 1;
        }
        // Above every version an NRT snapshot was given, so `getVersion`
        // stays monotonic across snapshots and commits (see `crate::nrt`).
        // Never used with NRT readers, `max_version` is -1 and this is a
        // no-op.
        new_segment_infos.version = new_segment_infos
            .version
            .max(self.nrt.max_version.saturating_add(1));
        // Java writes a fresh `StringHelper.randomId()` into every
        // `segments_N` header it produces, so two commits of the same index are
        // never confusable by id; cloning the previous commit's id would make
        // every generation of an index report the same one.
        new_segment_infos.id = generate_segment_id(new_segment_infos.generation);
        new_segment_infos
            .segments
            .extend(self.flushed_segments.iter().cloned());
        self.stamp_min_segment_version(&mut new_segment_infos)?;
        self.stamp_soft_delete_counts(&mut new_segment_infos)?;

        // `startCommit`: "skip startCommit(): no changes pending". Nothing
        // that reaches a commit changed since the last one -- no counted
        // change, and the segment list (flushed segments folded in) is what
        // the last commit wrote -- so another generation would only repeat it.
        if self.change_count == self.last_commit_change_count
            && same_committed_segments(&new_segment_infos.segments, &self.rollback_segments)
        {
            self.prepared_nothing = true;
            return Ok(());
        }

        // `startCommit`: `directory.sync(toSync.files(false))` -- every file
        // the commit names that this writer wrote and has not synced yet
        // (flushed, merged and copied segments, `.liv` and doc-values
        // generations), each at most once. Nothing was synced as it was
        // written. On failure nothing is recorded as synced, so a retry syncs
        // the same files again.
        self.deleter.sync_commit_files(&new_segment_infos)?;

        // Phase one of the real two-phase protocol: the commit is fully
        // serialized and fsynced under `pending_segments_N`, a name
        // `last_commit_generation` does not scan for, so a crash between here
        // and `finish_commit` leaves the previous commit current and this one
        // an inert orphan -- never a truncated `segments_N` that makes the
        // whole index unopenable.
        new_segment_infos.format_version = self.segment_infos_format_version();
        let written = segment_infos::write_pending(&new_segment_infos, self.dir);
        // `startCommit`'s `finally`: `segmentInfos.updateGeneration(toSync)`
        // "on error or success so we don't double-write a segments_N file".
        // A pending file a failed write or a rollback could not delete is
        // then never asked for again (`createOutput` refuses an existing
        // name), and the next prepare takes the generation after it.
        self.segment_infos.generation = new_segment_infos.generation;
        written?;
        self.prepared_commit = Some(new_segment_infos);
        self.pending_commit_change_count = self.change_count;
        Ok(())
    }

    /// Writes every currently-buffered document out as one new segment, without
    /// committing it: real Lucene's `DocumentsWriterPerThread.flush()` followed
    /// by `IndexWriter.publishFlushedSegment`.
    ///
    /// The new segment's files are on disk when this returns, and the deleter
    /// has been checkpointed (non-commit) so nothing reclaims them, but no
    /// `segments_N` names them yet -- a fresh reader still sees the previous
    /// commit. They are not fsynced: as in Java, the commit that first names
    /// them syncs them, so a crash before it may lose them.
    /// [`IndexWriter::prepare_commit`] folds every segment flushed this way
    /// into the commit it writes; [`IndexWriter::rollback`] drops them and the
    /// deleter deletes their files.
    ///
    /// A no-op with an empty buffer. Called automatically by
    /// [`IndexWriter::add_document`] once a configured threshold trips (see
    /// [`IndexWriter::set_ram_buffer_size_mb`]), which is what bounds this
    /// writer's peak memory independently of how many documents a caller adds
    /// between commits.
    ///
    /// On failure partway through -- a `postings_writer` validation rejection,
    /// an I/O error -- the partially written segment's files are reclaimed
    /// before the error is returned (Java's `DocumentsWriterPerThread.abort()` +
    /// `IndexFileDeleter.deleteNewFiles`), so a failed flush does not leak.
    pub fn flush(&mut self) -> Result<()> {
        self.write_pending_try_changes()?;
        // See `maybe_flush`: a flush between `prepare_commit` and
        // `finish_commit` would have its segment and its resolved deletes
        // discarded by the publish. The automatic trigger defers; an explicit
        // call is refused, because silently doing nothing would be worse.
        if self.prepared_commit.is_some() {
            return Err(Error::PreparedCommitPending("flush"));
        }
        self.ensure_write_lock_valid()?;
        if !self.pending_docs.is_empty() {
            self.validate_configured_term_vector_schema()?;
        }
        self.release_nrt_holds()?;
        if self.pending_docs.is_empty() {
            // Deletes issued while the document buffer was empty still have to
            // be resolved -- Java's `applyAllDeletes` is not conditional on a
            // segment having been produced.
            self.apply_all_deletes_and_updates()?;
            // The private buffer's entries were issued against an empty
            // document buffer (`docIDUpto` 0): they reach no document, and
            // Java has no per-thread writer to hold them, so they must not
            // keep `anyChanges` (and so `nrtIsCurrent`) true.
            let _ = self.delete_queue.freeze_private_buffer("");
            // ...and the `.liv`/doc-values generations it just wrote are referenced
            // by this writer's in-memory view but by no commit yet, so the
            // deleter has to be told before anything can `refresh()` them away
            // (`IndexFileDeleter.checkpoint(segmentInfos, false)`). The flush
            // path below does the same after its own apply.
            let live = self.live_infos();
            self.deleter.checkpoint(&live, false)?;
            return Ok(());
        }

        // ------------------------------------------------------------------
        // `IndexingChain.maybeSortSegment` + `DocumentsWriterPerThread`'s
        // sort-on-flush. Java threads a `Sorter.DocMap` into *every* format's
        // writer (`SortingStoredFieldsConsumer`, `SortingTermVectorsConsumer`,
        // `NumericDocValuesWriter.flush`'s `sortDocValues`,
        // `Lucene99FlatVectorsWriter.writeSortingField`, ...), each of which
        // then remaps its own doc ids. This port permutes the *buffer*
        // instead, once, before any format is built: every consumer below
        // reads `self.pending_docs` (and the two buffers aligned with it) and
        // takes each entry's index as its doc id, so one permutation puts
        // stored fields, postings, term vectors, norms, doc values and
        // vectors in sort order together and by construction consistently.
        // The remap Java repeats per format is exactly where a format can be
        // forgotten, and points has
        // no flush path here at all.
        //
        // Ahead of the two side effects below (freezing the global packet and
        // burning a segment name) so that a flush this rejects -- document
        // blocks with no parent field -- leaves the writer exactly as it was.
        let sort_map = self.sort_pending_buffer()?;
        // ------------------------------------------------------------------

        // Freeze what the *already written* segments owe **before** the new
        // segment takes a generation. Java's `publishFlushedSegment` publishes
        // the global packet first and only then reads `nextGen` for the new
        // segment, which is exactly what keeps the new segment out of the
        // packets that predate it (`FrozenBufferedUpdates.applies_to`).
        let ticket = self.begin_segment(None);
        let (segment_name, segment_id) = (ticket.segment_name.clone(), ticket.segment_id);

        // Installed consumers can create and write files during the build.
        // The segment remains unpublished until the build succeeds; returned
        // errors below restore the buffer and delete only acquired outputs,
        // preserving unrelated files, the committed view and the original error.
        // `postings_fields`/`custom_freq_postings_field` are enforced
        // mutually exclusive by `set_postings_field`/`add_postings_field`/
        // `set_custom_freq_postings_field` (see those methods' doc
        // comments), so at most one branch below ever produces `Some` --
        // no ambiguity about which one "wins".
        // One call site, so one error path -- see
        // `build_and_write_segment` on why that is load-bearing here.
        // `pending_postings` holds insertion-order doc ids; the sort map is
        // `new_to_old`.
        let old_to_new = sort_map.as_ref().map(|new_to_old| {
            let mut old_to_new = vec![0usize; new_to_old.len()];
            for (new, &old) in new_to_old.iter().enumerate() {
                old_to_new[old] = new;
            }
            old_to_new
        });
        let created = crate::created_outputs::CreatedOutputs::new(self.dir);
        let written = self.cfg.build_and_write_segment(
            &created,
            &DocumentBuffer {
                old_to_new: old_to_new.as_deref(),
                ..self.document_buffer(&self.pending_postings)
            },
            &segment_name,
            segment_id,
            None,
        );

        let (sci, si_files) = match written {
            Ok(pair) => pair,
            Err(e) => {
                // Put the buffer back in insertion order. Every buffered
                // delete's `docIDUpto` is a position in the *insertion*-ordered
                // buffer (`pending_doc_id_upto` reads `pending_docs.len()` at
                // the moment the delete is issued), and this writer keeps the
                // documents on a failure so the caller can retry or roll back.
                // Leaving them permuted would mix the two doc spaces: the
                // retry's `sort_pending_buffer` would see an already-ordered
                // buffer, short-circuit to "no map", and then compare pre-sort
                // limits against sorted doc ids -- silently, with a valid
                // `.liv` and a clean `CheckIndex`. This is also what keeps
                // `set_doc_values_field`'s promise that a failed flush leaves
                // `pending_docs` unchanged.
                if let Some(map) = &sort_map {
                    self.unsort_pending_buffer(map);
                }
                // What the written segments owe is owed whether or not this
                // one could be built.
                self.abandon_segment(ticket);
                // Java: DocumentsWriterPerThread.abort + deleteNewFiles.
                // A failed creation does not own a preexisting collision.
                // Cleanup errors must not replace the original build error.
                let files = created.files().iter().cloned().collect::<Vec<_>>();
                let _ = self.deleter.delete_new_files(&files);
                return Err(e);
            }
        };

        self.pending_docs.clear();
        self.pending_custom_freq_terms.clear();
        self.pending_vectors.clear();
        self.pending_explicit.clear();
        self.pending_postings.clear();
        self.ram_bytes_used = 0;
        self.pending_has_blocks = false;
        let private = self.delete_queue.freeze_private_buffer(&segment_name);
        self.publish_segment(ticket, sci, si_files, private, sort_map)
    }

    /// The first half of a flush, `DocumentsWriter.doFlush`'s
    /// `FlushTicket`: freeze what the **already written** segments owe as of
    /// now, and claim the new segment's name and id.
    ///
    /// The frozen packet is *carried*, not pushed: `publish_segment` pushes it
    /// immediately before the segment's own packet, in ticket order. Java's
    /// `publishFlushedSegment` does exactly that, and it is what keeps
    /// generations honest when builds overlap: a delete issued after this
    /// segment's documents were taken but before its publish is frozen by a
    /// *later* ticket, so it gets a higher generation than this segment and
    /// applies to it. Pushed here instead, it would sort below this segment
    /// and skip it.
    ///
    /// `reserved` is a name and id [`Self::reserve_segment`] handed out
    /// earlier, for a buffer that started writing its segment's files.
    pub(crate) fn begin_segment(
        &mut self,
        reserved: Option<&(String, [u8; ID_LENGTH])>,
    ) -> SegmentTicket {
        let global = self.delete_queue.freeze_global_buffer();
        let (segment_name, segment_id) = match reserved {
            Some((name, id)) => (name.clone(), *id),
            None => self.reserve_segment(),
        };
        SegmentTicket {
            segment_name,
            segment_id,
            global,
        }
    }

    /// A new segment's name and id, taken now for a segment written later.
    pub(crate) fn reserve_segment(&mut self) -> (String, [u8; ID_LENGTH]) {
        let segment_name = self.new_segment_name();
        let segment_id = generate_segment_id(self.segment_infos.counter);
        (segment_name, segment_id)
    }

    /// Pins files a buffer writes before its segment exists against the
    /// deleter's sweeps, until [`Self::release_files`].
    pub(crate) fn hold_new_files(&mut self, files: &[String]) {
        self.deleter.hold_files(files);
    }

    /// A ticket whose segment could not be built: its frozen packet still
    /// joins the stream, since the segments already written owe it regardless.
    pub(crate) fn abandon_segment(&mut self, ticket: SegmentTicket) {
        if let Some(global) = ticket.global {
            self.updates_stream.push(global);
        }
    }

    /// The last half of a flush, `IndexWriter.publishFlushedSegment`: the
    /// segment just written joins this writer's view, its own deletes
    /// (`private`, each still carrying the buffer position it was issued at,
    /// in the pre-sort doc space `sort_map` translates) are resolved against
    /// it, and every other pending packet is applied.
    pub(crate) fn publish_segment(
        &mut self,
        ticket: SegmentTicket,
        sci: SegmentCommitInfo,
        si_files: Vec<String>,
        private: Option<crate::buffered_updates::FrozenBufferedUpdates>,
        sort_map: Option<Vec<usize>>,
    ) -> Result<()> {
        self.write_pending_try_changes()?;
        self.publish_segment_with(ticket, sci, si_files, private, sort_map, true)
    }

    /// [`Self::publish_segment`], with the choice of whether the deletes
    /// buffered *since* the ticket are frozen and applied too.
    /// [`IndexWriter::flush`] does (`true`): nothing is in flight, so this is
    /// Java's `applyAllDeletes`. A
    /// [`crate::concurrent_writer::ConcurrentIndexWriter`] must not (`false`)
    /// while another segment is still being built: those deletes are owed to
    /// that segment too, and only a packet pushed after its own ticket can
    /// reach it -- freezing them now would apply them to every published
    /// segment and drop them before the in-flight one exists. Java routes
    /// every such freeze through its ticket queue for the same reason.
    pub(crate) fn publish_segment_with(
        &mut self,
        ticket: SegmentTicket,
        sci: SegmentCommitInfo,
        si_files: Vec<String>,
        private: Option<crate::buffered_updates::FrozenBufferedUpdates>,
        sort_map: Option<Vec<usize>>,
        freeze_global: bool,
    ) -> Result<()> {
        self.write_pending_try_changes()?;
        let segment_name = sci.segment_name.clone();
        debug_assert_eq!(segment_name, ticket.segment_name);
        // `publishFrozenUpdates` before `publishFlushedSegment`: the packet the
        // ticket froze takes the lower generation.
        if let Some(global) = ticket.global {
            self.updates_stream.push(global);
        }
        // `IndexWriter.publishFlushedSegment`: `rld.sortMap = sortMap`. See
        // [`Self::pending_sort_map`] for why this port scopes it to this call
        // rather than pooling a reader for the writer's lifetime.
        if let Some(m) = sort_map {
            self.pending_sort_map
                .retain(|(name, _)| name != &segment_name);
            self.pending_sort_map.push((segment_name.clone(), m));
        }

        // `IndexWriter.publishFlushedSegment`: the segment-private packet
        // (this segment's own deletes, each still carrying the buffer position
        // it was issued at) is pushed, and the generation it receives becomes
        // the segment's `bufferedDeletesGen`. With no private packet, Java
        // burns a generation anyway (`bufferedUpdatesStream.getNextGen()`), so
        // that every packet pushed *before* this flush sorts strictly below
        // the new segment and therefore cannot touch it.
        let mut sci = sci;
        let gen = match private {
            Some(packet) => self.updates_stream.push(packet),
            None => self.updates_stream.next_gen(),
        };
        sci.set_buffered_deletes_gen(gen);
        // `SegmentCommitInfo` holding its `SegmentInfo` is what lets Java's
        // `SegmentInfos.files()` -- and therefore every `IndexFileDeleter`
        // checkpoint -- read a segment's file set out of memory. This port's
        // `SegmentCommitInfo` deliberately does not own the parsed `.si`, so
        // the list is handed over explicitly here, from the same in-memory
        // `SegmentInfo` `seal_flushed_segment` just encoded.
        self.deleter.record_segment_files(&sci, &si_files);
        // Its `.si` carries this writer's version; recorded here for the same
        // reason, so no commit ever reads it back.
        self.segment_versions.insert(
            sci.segment_name.clone(),
            to_segment_infos_version(self.cfg.lucene_version),
        );
        self.flushed_segments.push(sci);

        // `IndexFileDeleter.checkpoint(segmentInfos, false)`: the new segment's
        // files are now referenced by this writer's in-memory view, so nothing
        // reclaims them before a commit names them.
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;

        if freeze_global {
            self.apply_all_deletes_and_updates()?;
        } else {
            self.apply_stream_packets()?;
        }
        // Every packet with a limit below `MAX_DOC_ID_UPTO` for this segment
        // was applied above (`apply_all_deletes_and_updates` drains the whole
        // stream), so the map has no further reader.
        self.pending_sort_map.clear();

        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
        Ok(())
    }

    /// The activation half of a real two-phase commit, port of real
    /// Lucene's `IndexWriter.commit()` when a `prepareCommit()` already ran
    /// (real Lucene's `finishCommit`): writes the `segments_N` file for the
    /// [`SegmentInfos`] [`IndexWriter::prepare_commit`] stashed in
    /// `self.prepared_commit`, via the exact same
    /// [`crate::segment_infos::write`] call [`IndexWriter::commit`] always
    /// used -- this is the single call that actually makes the new
    /// generation discoverable to a fresh [`IndexWriter::open`]/reader (see
    /// [`IndexWriter::prepare_commit`]'s doc comment for why that one call is
    /// exactly where "prepared" ends and "current" begins on this port's
    /// on-disk format). Runs auto-merge afterward exactly as
    /// [`IndexWriter::commit`] always has.
    ///
    /// Returns `Err(Error::NoPreparedCommit)` if no [`IndexWriter::prepare_commit`]
    /// call is currently pending (nothing to activate) -- distinct from a
    /// no-op, since silently succeeding here could hide a caller bug (calling
    /// `finish_commit()` twice, or before ever calling `prepare_commit()`).
    pub fn finish_commit(&mut self) -> Result<&SegmentInfos> {
        self.write_pending_try_changes()?;
        if self.prepared_commit.is_none() {
            // `finishCommit` with `pendingCommit == null`: the prepare found
            // nothing to commit, so there is nothing to publish.
            if std::mem::take(&mut self.prepared_nothing) {
                return Ok(&self.segment_infos);
            }
            return Err(Error::NoPreparedCommit);
        }
        // The last check before the publish: a lost lock leaves the prepared
        // commit in place for `rollback()`, as any other failure here does.
        self.ensure_write_lock_valid()?;
        let new_segment_infos = self.prepared_commit.take().ok_or(Error::NoPreparedCommit)?;

        // The single atomic publish: rename `pending_segments_N` onto
        // `segments_N`. On failure the prepared state is put back rather than
        // dropped, so the caller can retry (or `rollback()`) instead of losing
        // every document the prepare already flushed.
        if let Err(e) = segment_infos::finish_pending(&new_segment_infos, self.dir) {
            self.prepared_commit = Some(new_segment_infos);
            return Err(e.into());
        }

        // `IndexFileDeleter.checkpoint(pendingCommit, true)`: incRef the new
        // commit (its `segments_N` included), then let the deletion policy drop
        // superseded commit points -- which is what finally reclaims the
        // previous `segments_N`, the `.liv` generation it superseded, and every
        // segment this commit no longer names.
        self.segment_infos = new_segment_infos;
        self.flushed_segments.clear();
        // `finishCommit`: `lastCommitChangeCount = pendingCommitChangeCount`.
        self.last_commit_change_count = self.pending_commit_change_count;
        // `finishCommit`: `rollbackSegments =
        // pendingCommit.createBackupSegmentInfos()`. This is the one place a
        // commit becomes durable, so it is the one place the rollback snapshot
        // moves forward.
        self.rollback_segments = self.segment_infos.segments.clone();
        self.checkpoint_committed()?;

        if (self.merge_policy.is_some() || self.pluggable_merge_policy.is_some())
            && !self.merges_by_caller
        {
            self.auto_merge()?;
        }

        Ok(&self.segment_infos)
    }

    /// `IndexingChain.maybeSortSegment` for this port's buffer-shaped flush:
    /// reorders `pending_docs` and the two buffers aligned with it into index
    /// sort order and returns `Sorter.DocMap.newToOld`, or `None` when no
    /// sort is configured or when the buffer is already in order.
    ///
    /// Why the buffer rather than each format's writer: every consumer in
    /// [`IndexWriter::flush`] derives a document's id from its **index** in
    /// `pending_docs` (postings via
    /// [`crate::indexing_chain::invert_documents`], term vectors and norms
    /// from that same inverted index, doc values from
    /// `docs[i].fields`, vectors from `pending_vectors[i]`, stored fields
    /// from `pending_docs[i]` itself). Permuting the buffer therefore puts
    /// **all** of them in sort order at once, and no format can be forgotten
    /// -- which is the failure mode Java's per-format `Sorter.DocMap`
    /// plumbing has (`Lucene99FlatVectorsWriter.writeSortingField` was
    /// unreachable here for exactly that reason).
    ///
    /// The sort keys are each sort field's NUMERIC doc-values value,
    /// `None` for a document that has none -- the same column
    /// [`Self::build_doc_values_output`] writes, read out of the same
    /// `Document`s, so the order and the column a reader checks it against
    /// cannot come from two different facts. Java reads them back through a
    /// `DocValuesLeafReader` over the in-memory `DocValuesWriter`s, which is
    /// the same thing one indirection later.
    ///
    /// Returns [`Error::IndexSortWithBlocksAndNoParentField`] for a buffer
    /// carrying document blocks without a parent field
    /// ([`IndexWriter::set_parent_field`]) to sort them by: a block must stay
    /// contiguous and in order, and a sort would shred it. Every add path
    /// refuses such a block up front ([`Error::BlocksWithIndexSortNeedParentField`]),
    /// so this is a backstop that no buffer reaches.
    fn sort_pending_buffer(&mut self) -> Result<Option<Vec<usize>>> {
        let cfg = std::sync::Arc::clone(&self.cfg);
        cfg.sort_buffer(
            &mut self.pending_docs,
            &mut self.pending_custom_freq_terms,
            &mut self.pending_vectors,
            &mut self.pending_explicit,
            self.pending_has_blocks,
        )
    }

    /// The inverse of [`Self::sort_pending_buffer`]'s permutation, applied to
    /// the same three buffers: restores insertion order after a flush that
    /// failed between the sort and the publish. `new_to_old` read as an
    /// `old_to_new` map *is* the inverse permutation, which is exactly what
    /// [`segment_writer::permute_in_place`] needs to undo it -- so inverting
    /// is `map[i] -> i` and nothing more.
    fn unsort_pending_buffer(&mut self, new_to_old: &[usize]) {
        let mut old_to_new = vec![0usize; new_to_old.len()];
        for (new, &old) in new_to_old.iter().enumerate() {
            old_to_new[old] = new;
        }
        segment_writer::permute_in_place(&mut self.pending_docs, &old_to_new);
        segment_writer::permute_in_place(&mut self.pending_custom_freq_terms, &old_to_new);
        segment_writer::permute_in_place(&mut self.pending_vectors, &old_to_new);
        segment_writer::permute_in_place(&mut self.pending_explicit, &old_to_new);
    }

    /// Tokenizes every pending document's text **once** for the union of the
    /// fields this commit needs analyzed -- postings, term vectors, and norms
    /// -- and returns the single [`InMemoryInvertedIndex`] all three consumers
    /// then read.
    ///
    /// This is real Lucene's shape: `IndexingChain.processField` runs
    /// `PerField.invert()` exactly once per (document, field) and fans the
    /// result out to `FreqProxTermsWriterPerField` (postings),
    /// `TermVectorsConsumerPerField` (term vectors) and
    /// `NormValuesWriter` (norms) from that one pass; a separate
    /// tokenize-and-invert per consumer would analyze every document up to
    /// three times per commit.
    ///
    /// A field with no [`FieldValue::String`] value on any pending doc simply
    /// contributes no entries.
    fn invert_pending_fields(
        docs: &[Document],
        postings: &[PostingsFieldConfig],
        term_vectors: &[TermVectorFieldConfig],
        norms: &[NormsFieldConfig],
        payload_fields: &[&str],
        payload_source: Option<PayloadSourceRef<'_>>,
        analyzer: &Analyzer,
    ) -> InMemoryInvertedIndex {
        // Union by field number: a field opted into two consumers at once
        // (e.g. postings *and* term vectors) must still be analyzed once, not
        // once per consumer -- that de-duplication is the whole point.
        let mut wanted: Vec<(i32, &str)> = Vec::new();
        for c in postings {
            wanted.push((c.field_number, c.name.as_str()));
        }
        for c in term_vectors {
            wanted.push((c.field_number, c.name.as_str()));
        }
        for c in norms {
            wanted.push((c.field_number, c.name.as_str()));
        }
        wanted.sort_unstable();
        wanted.dedup();

        // **Every** value of the field, not just the first: a document may
        // carry the same field more than once (Java's `Document.add` appends,
        // and `IndexingChain` inverts each value through one
        // `FieldInvertState`), and `invert_documents_with_payloads` groups the
        // consecutive tuples for one (doc, field) into one multi-valued field.
        // Every value is gathered, not only the first: a later value of a
        // multi-valued field would otherwise be stored but never indexed.
        let mut triples: Vec<(i32, &str, &str)> = Vec::new();
        for (field_number, field_name) in wanted {
            for (doc_id, doc) in docs.iter().enumerate() {
                for f in doc.fields.iter().filter(|f| f.field_number == field_number) {
                    if let Some(text) = f.value.as_text() {
                        triples.push((doc_id as i32, field_name, text));
                    }
                }
            }
        }

        // No per-field-analyzer configuration exists on this facade yet (see
        // the module doc comment), so one `Analyzer` covers every field --
        // which is also what makes a single shared pass sound. It carries the
        // writer's `positionIncrementGap`/`offsetGap`, the two knobs Java puts
        // on `Analyzer` and `IndexingChain` reads once per field value.
        // A writer with no payload source still has to allocate payload slots
        // for a `store_payloads` field: the `.fnm` bit is already written, so
        // the `.pay` payload-length stream has to exist for real Lucene's
        // reader to frame the file at all. `no_payload` supplies the
        // zero-length payload that means exactly what a `null`
        // `PayloadAttribute` means in Java.
        let no_payload = |_: &PayloadContext<'_>| None;
        let source: crate::indexing_chain::PayloadSource<'_> = match payload_source {
            Some(source) => source,
            None => &no_payload,
        };

        invert_documents_with_payloads(&triples, analyzer, payload_fields, source)
    }

    /// Builds [`postings_writer::write_fields`]'s input from `docs`'
    /// [`FieldValue::String`] values for **every** field in `configs` (each
    /// pending doc's index into `docs` becomes its doc ID in the new segment,
    /// matching
    /// [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment)'s
    /// own doc-ordering), tokenizes each field independently via
    /// [`crate::indexing_chain::invert_documents`] with a plain
    /// [`Analyzer::standard`] (no stopwords -- this facade has no
    /// per-field-analyzer configuration yet, see module doc comment's scope
    /// notes elsewhere in this crate), and calls
    /// [`postings_writer::write_fields`] **once** with every field's terms
    /// batched together -- so a commit with, say, two distinct indexed text
    /// fields produces exactly one `.doc`/`.tim`/`.tip`/`.tmd` file set
    /// covering both fields, never two separate file sets (mirroring
    /// [`crate::merge::merge_stored_only_segments`]'s own
    /// `merge_postings`-then-single-`write_fields`-call shape in
    /// `crates/lucene-index/src/merge.rs`).
    ///
    /// A field in `configs` with no indexable text across any pending doc is
    /// simply omitted from the `write_fields` call (not an error on its own);
    /// `Ok(None)` is only returned when *every* field in `configs` has
    /// nothing to write, matching this method's previous single-field
    /// "nothing to write this commit" outcome for that case.
    /// Returns `Err` on [`postings_writer::write_fields`]'s own validation
    /// failures -- see that module's doc comment for the current set of
    /// rejected shapes (mismatched per-document position/offset/payload
    /// counts, non-ascending positions or doc ids, `freq < 1`). There is no
    /// `docFreq` ceiling on a positions-indexing term.
    ///
    /// **The offsets it forwards are UTF-16 code units** -- Java `char` indices
    /// into the field text, the unit `OffsetAttribute` reports and the unit
    /// real Lucene reads back out of the `.pos`/`.pay` written here. Nothing here
    /// converts, and nothing on either side would catch a
    /// regression: `CheckIndex` only checks that offsets are ordered and in
    /// range, never that they index the stored text. The unit is pinned where
    /// it is produced, in `crates/lucene-analysis/tests/analysis_fixtures.rs`.
    fn build_postings_output(
        routing: &IndexingConfig,
        configs: &[PostingsFieldConfig],
        inverted: InMemoryInvertedIndex,
        norms: &[(i32, Vec<i64>)],
        state: lucene_codecs::postings_consumer::PostingsWriteState<'_>,
    ) -> Result<Option<PostingsGroups>> {
        struct FieldData {
            config: PostingsFieldConfig,
            /// Which docs carry the field, as bits, and how many: a
            /// `BTreeSet` insert per posting only to count them was
            /// O(postings log docs) with a node allocation each.
            doc_bits: Vec<u64>,
            doc_count: usize,
            terms: Vec<TermPostings>,
        }

        let mut per_field: Vec<FieldData> = configs
            .iter()
            .map(|config| FieldData {
                config: config.clone(),
                doc_bits: Vec::new(),
                doc_count: 0,
                terms: Vec::new(),
            })
            .collect();
        let by_name: std::collections::HashMap<&str, usize> = configs
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.as_str(), i))
            .collect();

        // One pass over the whole term dictionary, **consuming** it. Two things
        // fall out of that: each `(field, term)` key's `Vec<PostingEntry>` (and
        // every `Vec<Occurrence>` inside it) is freed as soon as it has been
        // transformed, instead of the entire inverted index staying live
        // alongside its `TermPostings` copy; and the term bytes are moved out of
        // the key `String` rather than copied. `inverted.terms` is a `BTreeMap`
        // keyed by `(field, term)`, so each field's terms still arrive in
        // ascending byte order -- exactly the per-field ordering
        // `postings_writer::write_fields` requires, with no sort needed.
        for ((field, term), list) in inverted.terms {
            let Some(&idx) = by_name.get(field.as_str()) else {
                continue;
            };
            let data = &mut per_field[idx];
            let has_positions = matches!(
                data.config.index_options,
                IndexOptions::DocsAndFreqsAndPositions
                    | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
            );
            let has_offsets =
                data.config.index_options == IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
            // `store_payloads` on a field without positions is rejected at
            // `set_postings_field` time (Java's `FieldInfo.checkConsistency`),
            // so this `&&` is belt-and-braces rather than a second policy: it
            // keeps `has_payloads` and the `payloads` vectors from ever
            // disagreeing if that guard is ever relaxed.
            let has_payloads = data.config.store_payloads && has_positions;

            let entries = list.entries;
            let mut term_docs: Vec<(i32, i32)> = Vec::with_capacity(entries.len());
            // `postings_writer::write_fields` only consults
            // `positions`/`offsets` when this field's `index_options`
            // indexes them (see `TermPostings`'s doc comment) -- leaving
            // them empty for `Docs`/`DocsAndFreqs` fields matches that
            // contract exactly.
            let mut positions: Vec<Vec<i32>> = Vec::new();
            let mut offsets: Vec<Vec<(i32, i32)>> = Vec::new();
            if has_positions {
                positions.reserve_exact(entries.len());
            }
            if has_offsets {
                offsets.reserve_exact(entries.len());
            }
            for entry in entries {
                // A doc id is an index into this flush's buffer: never
                // negative.
                let doc = usize::try_from(entry.doc_id).unwrap_or(0);
                let word = doc >> 6;
                if word >= data.doc_bits.len() {
                    data.doc_bits.resize(word.saturating_add(1), 0);
                }
                let bit = 1u64 << (doc & 63);
                if data.doc_bits[word] & bit == 0 {
                    data.doc_bits[word] |= bit;
                    data.doc_count = data.doc_count.saturating_add(1);
                }
                term_docs.push((entry.doc_id, entry.term_freq()));
                if has_positions {
                    positions.push(entry.occurrences.iter().map(|o| o.position).collect());
                }
                if has_offsets {
                    offsets.push(
                        entry
                            .occurrences
                            .iter()
                            .map(|o| (o.start_offset, o.end_offset))
                            .collect(),
                    );
                }
            }
            // Moved wholesale, not copied and not re-nested: the inverted
            // index and `TermPostings` hold this term's payloads in the same
            // flat `(bytes, lengths)` layout, so the run changes owner without
            // touching a byte of it.
            //
            // `invert_documents_with_payloads` fills the run for every
            // occurrence of a listed field or for none of them -- the gate is
            // the field, never the token -- so the only mismatch it can
            // produce is a **wholly absent** run, which the pad below turns
            // into an all-zero-length one. That is what a `store_payloads`
            // field with no source is supposed to write anyway
            // (`a_payloads_field_with_no_source_still_writes_the_payload_length_stream`),
            // and it is why the pad is at the end rather than per document:
            // there is no per-document hole for it to fill. A run short in
            // the *middle* would misalign every later document, and cannot
            // arise here; `postings_writer::validate_field` is what would
            // catch it, at the cost of naming a term rather than the field
            // that was mis-declared.
            let (mut payload_bytes, mut payload_lengths) = if has_payloads {
                (list.payload_bytes, list.payload_lengths)
            } else {
                (Vec::new(), Vec::new())
            };
            if has_payloads {
                let occurrences: usize = term_docs.iter().map(|&(_, freq)| freq as usize).sum();
                // The equality is the fast path and the common one -- the
                // invert pass produces exactly one length per occurrence --
                // so the O(occurrences) repair below runs only for a run that
                // is already known to be wrong.
                if payload_lengths.len() != occurrences {
                    payload_lengths.resize(occurrences, 0);
                    let bytes: usize = payload_lengths.iter().map(|&l| l as usize).sum();
                    // `resize` both pads and truncates, which is what a run
                    // that is too long as well as one that is too short
                    // needs.
                    payload_bytes.resize(bytes, 0);
                }
            }
            data.terms.push(TermPostings {
                term: term.into_bytes(),
                docs: term_docs,
                positions,
                offsets,
                payload_bytes,
                payload_lengths,
            });
        }

        // A field in `configs` with no indexable text across any pending doc is
        // simply omitted from the `write_fields` call.
        per_field.retain(|f| !f.terms.is_empty());
        if per_field.is_empty() {
            return Ok(None);
        }

        let inputs: Vec<FieldPostingsInput<'_>> = per_field
            .iter()
            .map(|f| FieldPostingsInput {
                field_number: f.config.field_number,
                index_options: f.config.index_options,
                doc_count: f.doc_count as i32,
                // Must match the `.fnm`'s `STORE_PAYLOADS` bit for this field:
                // it is what makes `postings_writer` emit `.pay` at all and
                // write a payload-length run per block, and it is what real
                // Lucene's reader frames `.pay` with.
                has_payloads: f.config.store_payloads
                    && f.config.index_options.subsumes_positions(),
                terms: &f.terms,
            })
            .collect();
        // The flush's own norms, so the impacts this writes are the real
        // `(freq, norm)` frontier -- see `build_norms_output`'s return value
        // and `postings_writer::write_fields_with_norms` (item 18).
        let norms_for_impacts: Vec<postings_writer::FieldNorms<'_>> = norms
            .iter()
            .map(|(number, values)| postings_writer::FieldNorms {
                field_number: *number,
                values,
            })
            .collect();
        let output = routing.write_postings_groups(&inputs, &norms_for_impacts, state)?;
        Ok(Some(output))
    }

    /// One finished [`crate::inverter::InvertedField`] per postings field, fed every
    /// buffered document in doc-id order.
    fn invert_pending_fields_fast(
        docs: &[Document],
        postings: &[PostingsFieldConfig],
        analyzer: &Analyzer,
    ) -> Vec<(i32, crate::inverter::InvertedField)> {
        use rayon::prelude::*;
        // Fields are independent, and each one's documents are split across
        // the pool again inside `invert_parallel`.
        postings
            .par_iter()
            .map(|config| {
                let inverter = crate::inverter::FieldInverter::invert_parallel(
                    config.index_options,
                    docs.len(),
                    |doc_id, values: &mut Vec<&str>| {
                        values.extend(docs[doc_id].fields.iter().filter_map(|f| {
                            if f.field_number == config.field_number {
                                f.value.as_text()
                            } else {
                                None
                            }
                        }));
                    },
                    analyzer,
                );
                (config.field_number, inverter)
            })
            .collect()
    }

    /// [`Self::build_norms_output`] from field lengths already counted
    /// during inversion, rather than by walking every term's postings again.
    ///
    /// Which documents get a norm is decided exactly as there: a document
    /// whose first value for the field is a string has one (length counted
    /// from its tokens), any other has none.
    fn build_norms_output_from_lengths(
        docs: &[Document],
        configs: &[NormsFieldConfig],
        inverters: &[(i32, crate::inverter::InvertedField)],
        similarity: &dyn NormSimilarity,
        routing: &IndexingConfig,
        state: lucene_codecs::norms_writer::NormsWriteState<'_>,
    ) -> Result<NormsOutput> {
        let columns: Vec<NormsColumn> = configs
            .iter()
            .map(|config| {
                let inverter = inverters
                    .iter()
                    .find(|(n, _)| *n == config.field_number)
                    .map(|(_, inv)| inv);
                let values: Vec<Option<i64>> = docs
                    .iter()
                    .enumerate()
                    .map(|(doc_id, doc)| {
                        let first_is_string = doc
                            .fields
                            .iter()
                            .find(|f| f.field_number == config.field_number)
                            .is_some_and(|f| f.value.as_text().is_some());
                        if !first_is_string {
                            return Ok(None);
                        }
                        // No inverter: no tokens, so `norm_value`'s `0`
                        // without reading the rest.
                        let state = inverter.and_then(|inv| inv.invert_state(doc_id)).unwrap_or(
                            FieldInvertState {
                                docs_only: config.docs_only,
                                ..FieldInvertState::default()
                            },
                        );
                        norm_value(similarity, &config.name, &state).map(Some)
                    })
                    .collect::<Result<_>>()?;
                Ok(NormsColumn::from_values(values))
            })
            .collect::<Result<_>>()?;

        routing.finish_norms_output(columns, configs, state)
    }

    /// [`Self::build_postings_output`] over the inverters' terms, already in
    /// the postings writer's shape.
    fn build_postings_output_fast(
        routing: &IndexingConfig,
        configs: &[PostingsFieldConfig],
        inverters: Vec<(i32, crate::inverter::InvertedField)>,
        norms: &[(i32, Vec<i64>)],
        state: lucene_codecs::postings_consumer::PostingsWriteState<'_>,
        mut flush_deletes: Option<&mut FlushDeletes<'_>>,
    ) -> Result<Option<PostingsGroups>> {
        let mut per_field: Vec<(PostingsFieldConfig, i32, Vec<TermPostings>)> = Vec::new();
        for (config, (field_number, inverter)) in configs.iter().zip(inverters) {
            debug_assert_eq!(config.field_number, field_number);
            let doc_count = inverter.doc_count();
            let terms = inverter.into_term_postings();
            if let Some(deletes) = flush_deletes.as_deref_mut() {
                deletes.resolve_field(&config.name, &terms);
            }
            if !terms.is_empty() {
                per_field.push((config.clone(), doc_count, terms));
            }
        }
        if let Some(deletes) = flush_deletes {
            deletes.resolved = true;
        }
        if per_field.is_empty() {
            return Ok(None);
        }
        let inputs: Vec<FieldPostingsInput<'_>> = per_field
            .iter()
            .map(|(config, doc_count, terms)| FieldPostingsInput {
                field_number: config.field_number,
                index_options: config.index_options,
                doc_count: *doc_count,
                has_payloads: false,
                terms,
            })
            .collect();
        let norms_for_impacts: Vec<postings_writer::FieldNorms<'_>> = norms
            .iter()
            .map(|(number, values)| postings_writer::FieldNorms {
                field_number: *number,
                values,
            })
            .collect();
        Ok(Some(routing.write_postings_groups(
            &inputs,
            &norms_for_impacts,
            state,
        )?))
    }

    /// [`IndexingConfig::custom_freq_postings_field`]'s counterpart to
    /// [`IndexWriter::build_postings_output`]: builds
    /// [`postings_writer::write_fields`]'s input for `config`'s single field
    /// straight from `custom_freq_terms`' explicit `(term, custom_freq)`
    /// pairs -- **no** analyzer, no [`crate::indexing_chain::invert_documents`]
    /// call, no re-tokenizing of any stored field text -- matching
    /// [`IndexWriter::set_custom_freq_postings_field`]'s documented "explicit
    /// caller-supplied freq, not a derived occurrence count" contract.
    ///
    /// `custom_freq_terms[i]` is doc `i`'s explicit term list (`docs[i]`'s
    /// doc ID in the next flush), same 1:1-by-index alignment
    /// [`Self::pending_custom_freq_terms`]'s own doc comment describes;
    /// `docs` itself is only consulted for its length here (each pending doc
    /// contributes whatever `custom_freq_terms` says for it, regardless of
    /// its own stored-field values -- unlike the analyzed-text path, this one
    /// never reads a `Document`'s `FieldValue`s at all). Terms are grouped
    /// into one [`TermPostings`] per distinct term, in ascending byte order
    /// (via a `BTreeMap`, matching [`postings_writer::write_fields`]'s
    /// required per-field term ordering, same as
    /// [`IndexWriter::build_postings_output`]'s own `BTreeMap`-ordering
    /// argument). `IndexOptions::DocsAndCustomFreqs` carries no positions or
    /// offsets (same false `subsumes_positions()`/`subsumes_offsets()` as
    /// `DocsAndFreqs` -- see `crate::postings`'s module doc comment), so
    /// `positions`/`offsets` are always empty and `has_payloads` is always
    /// `false`, exactly like a `Docs`/`DocsAndFreqs` field on the analyzed
    /// path.
    ///
    /// Returns `Ok(None)` when no pending doc has any term for this field
    /// (nothing to write this commit, same "empty is not an error" shape as
    /// [`IndexWriter::build_postings_output`]). Returns `Err` on
    /// [`postings_writer::write_fields`]'s own validation failures -- in
    /// particular, a `custom_freq < 1` anywhere in `custom_freq_terms`
    /// surfaces as [`postings_writer::Error`]'s existing "freq < 1"
    /// rejection (see [`IndexWriter::add_document_with_custom_freq_terms`]'s
    /// doc comment).
    fn build_custom_freq_postings_output(
        routing: &IndexingConfig,
        docs: &[Document],
        custom_freq_terms: &[Vec<(String, i32)>],
        config: &CustomFreqPostingsFieldConfig,
        state: lucene_codecs::postings_consumer::PostingsWriteState<'_>,
    ) -> Result<Option<PostingsGroups>> {
        let mut per_term: std::collections::BTreeMap<Vec<u8>, Vec<(i32, i32)>> =
            std::collections::BTreeMap::new();
        for doc_id in 0..docs.len() {
            let Some(terms) = custom_freq_terms.get(doc_id) else {
                continue;
            };
            for (term, custom_freq) in summed_custom_freq_terms(terms)? {
                per_term
                    .entry(term.as_bytes().to_vec())
                    .or_default()
                    .push((doc_id as i32, custom_freq));
            }
        }

        if per_term.is_empty() {
            return Ok(None);
        }

        let mut doc_ids = std::collections::BTreeSet::new();
        let terms: Vec<TermPostings> = per_term
            .into_iter()
            .map(|(term, term_docs)| {
                for (doc_id, _) in &term_docs {
                    doc_ids.insert(*doc_id);
                }
                TermPostings {
                    term,
                    docs: term_docs,
                    ..Default::default()
                }
            })
            .collect();

        let inputs = [FieldPostingsInput {
            field_number: config.field_number,
            index_options: IndexOptions::DocsAndCustomFreqs,
            doc_count: doc_ids.len() as i32,
            has_payloads: false,
            terms: &terms,
        }];
        let output = routing.write_postings_groups(&inputs, &[], state)?;
        Ok(Some(output))
    }

    /// Builds [`norms::write_fields`]' input for every field in `configs`,
    /// automatically, from `docs`' own indexed text -- this port's
    /// `NormValuesWriter`.
    ///
    /// Real Lucene's `IndexingChain.PerField.finish(docID)` runs for each
    /// document that *contains* the field, and:
    ///
    /// ```java
    /// if (fi.omitsNorms() == false) {
    ///   long normValue;
    ///   if (invertState.length == 0) {
    ///     // the field exists in this document, but it did not have
    ///     // any indexed tokens, so we assign a default value of zero
    ///     normValue = 0;
    ///   } else {
    ///     normValue = similarity.computeNorm(invertState);
    ///   }
    ///   norms.addValue(docID, normValue);
    /// }
    /// ```
    ///
    /// so the column is **sparse**: a document that carries the field but
    /// tokenizes to nothing gets an explicit `0`, and a document that does
    /// not carry the field at all gets *no entry* (`NormValuesWriter`'s
    /// `DocsWithFieldSet`). This reproduces both: a field's per-doc value is
    /// `Some(length)` exactly when that doc has a [`FieldValue::String`] for
    /// it -- the same presence test the shared invert pass
    /// ([`Self::invert_pending_fields`]) uses to decide what to analyze --
    /// and the column is written [`norms::NormsField::Dense`] when every doc
    /// has one and [`norms::NormsField::Sparse`] otherwise, which is the
    /// same `numDocsWithValue == maxDoc` branch `Lucene90NormsConsumer`
    /// takes, so an absent field and an empty one (norm `0`) stay
    /// distinguishable.
    ///
    /// A doc's length is the sum of every matching term's occurrence count
    /// (real Lucene's `FieldInvertState.length` -- total indexed token
    /// count, *not* distinct-term count: "fox fox fox" has length 3, one
    /// distinct term), handed with the distinct-term count to the configured
    /// similarity's `computeNorm` ([`IndexWriter::set_similarity`]; by
    /// default BM25's: [`lucene_util::small_float::int_to_byte4`] of the
    /// length -- of the distinct terms for a `DOCS` field -- sign-extended
    /// into an `i64` the same way `norms::norm_value`'s read side
    /// sign-extends a stored byte back, the exact inverse transformation
    /// `lucene_search::similarity::decode_norm` undoes).
    /// Returns `(.nvm, .nvd, per-field norm columns)`. The third element is
    /// the same per-document norm each column encodes, kept dense and
    /// addressed by doc id, because [`Self::build_postings_output`] needs it:
    /// `Lucene104PostingsWriter` accumulates its level-0/level-1 impacts
    /// against the very norms `Lucene90NormsConsumer` is writing, and without
    /// them every impact is `(maxFreq, 1)` -- sound, but too loose to prune
    /// with (item 18). A document that carries no norm for the field reads as
    /// `1`, Java's own `advanceExact == false` fallback; it is unreachable
    /// through a posting, since a document with an occurrence of the field
    /// has a norm for it.
    ///
    /// `custom_freq` is the writer's [`IndexWriter::set_custom_freq_postings_field`]
    /// field and every document's `(term, custom_freq)` pairs: that field's
    /// postings come from the pairs, not from `inverted`, so its norms must too
    /// (see [`custom_freq_invert_state`]).
    fn build_norms_output(
        docs: &[Document],
        configs: &[NormsFieldConfig],
        inverted: &InMemoryInvertedIndex,
        custom_freq: Option<CustomFreqTerms<'_>>,
        similarity: &dyn NormSimilarity,
        routing: &IndexingConfig,
        state: lucene_codecs::norms_writer::NormsWriteState<'_>,
    ) -> Result<NormsOutput> {
        // One column per configured field, all into the same `.nvm`/`.nvd`
        // pair -- `Lucene90NormsConsumer` gets one `addNormsField` call per
        // field into one pair, and `norms::write_fields` is that shape.
        // `configs` is already ascending by field number
        // ([`Self::norms_field_configs`]), so the meta entry order is a
        // function of the schema.
        let columns: Vec<NormsColumn> = configs
            .iter()
            .map(|config| {
                if let Some((_, terms)) =
                    custom_freq.filter(|(number, _)| *number == config.field_number)
                {
                    let values: Vec<Option<i64>> = (0..docs.len())
                        .map(|doc| {
                            custom_freq_invert_state(terms.get(doc).map_or(&[][..], Vec::as_slice))
                                .map(|state| norm_value(similarity, &config.name, &state))
                                .transpose()
                        })
                        .collect::<Result<_>>()?;
                    return Ok(NormsColumn::from_values(values));
                }
                // `None` == this doc does not carry the field at all, so it
                // gets no norm; `Some(0)` == it carries it but produced no
                // tokens, which is Java's explicit zero.
                // `(length, uniqueTermCount, maxTermFrequency)` per document
                // that carries the field; the rest of its `FieldInvertState`
                // is the inversion's [`crate::indexing_chain::FieldEnd`].
                let mut counts: Vec<Option<(u32, u32, u32)>> = docs
                    .iter()
                    .map(|doc| {
                        doc.fields
                            .iter()
                            .find(|f| f.field_number == config.field_number)
                            .and_then(|f| match &f.value {
                                FieldValue::String(_) | FieldValue::Shared(_) => {
                                    Some((0u32, 0u32, 0u32))
                                }
                                _ => None,
                            })
                    })
                    .collect();
                // Only this field's keys: `(field, term)` sorts by field
                // first, so they are one contiguous range.
                for ((_field, _term), list) in inverted
                    .terms
                    .range((config.name.clone(), String::new())..)
                    .take_while(|((field, _), _)| field == &config.name)
                {
                    let entries = &list.entries;
                    for (i, entry) in entries.iter().enumerate() {
                        // `entry.doc_id` is an index into `docs` that this
                        // writer's own inversion produced, so it addresses
                        // `counts` (which is `docs.len()` long) by
                        // construction -- and only ever for a doc whose
                        // presence test above already said `Some`. One entry
                        // per `(term, doc)`: each is a distinct term.
                        if let Some((length, unique, max_freq)) =
                            counts[entry.doc_id as usize].as_mut()
                        {
                            *length = accumulate_field_length(*length, entry.term_freq());
                            *unique = unique.saturating_add(1);
                            // Without frequencies only `newTerm` -- the
                            // term's first document in the segment -- sets
                            // it, to 1.
                            let freq = if config.docs_only {
                                u32::from(i == 0)
                            } else {
                                u32::try_from(entry.term_freq()).unwrap_or(0)
                            };
                            *max_freq = (*max_freq).max(freq);
                        }
                    }
                }
                let values: Vec<Option<i64>> = counts
                    .into_iter()
                    .enumerate()
                    .map(|(doc, c)| {
                        c.map(|(length, unique, max_freq)| {
                            let clamp = |n: u32| i32::try_from(n).unwrap_or(i32::MAX);
                            let end = inverted
                                .field_end(&config.name, doc as i32)
                                .unwrap_or_default();
                            let state = FieldInvertState {
                                docs_only: config.docs_only,
                                position: end.position,
                                length: clamp(length),
                                num_overlap: end.num_overlap,
                                offset: end.offset,
                                max_term_frequency: clamp(max_freq),
                                unique_term_count: clamp(unique),
                                attribute_source: Some(crate::similarity::end_attributes(
                                    end.final_increment,
                                    end.final_offset,
                                )),
                            };
                            norm_value(similarity, &config.name, &state)
                        })
                        .transpose()
                    })
                    .collect::<Result<_>>()?;
                Ok(NormsColumn::from_values(values))
            })
            .collect::<Result<_>>()?;

        routing.finish_norms_output(columns, configs, state)
    }

    /// Builds one [`TermVectorsDocument`] per entry in `docs` (in the same
    /// doc-ID order
    /// [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment)
    /// uses -- index into `docs` == doc ID in the new segment), sourced from
    /// **every** field in `configs`' [`FieldValue::String`] values, each
    /// tokenized independently via [`crate::indexing_chain::invert_documents`]
    /// with a plain [`Analyzer::standard`] (same analyzer/scope notes as
    /// [`IndexWriter::build_postings_output`]).
    ///
    /// [`crate::indexing_chain::invert_documents`] returns a *term-keyed*
    /// inverted index (postings grouped by `(field, term)`, each entry a
    /// doc-ID-sorted list) -- the shape a postings writer wants. Term
    /// vectors need the transpose: *per-document, per-field* `term -> (freq,
    /// positions)`, so this regroups that same inverted index by doc ID
    /// rather than reimplementing tokenization a second time.
    /// [`term_vectors::write_best_speed`] already accepts multiple
    /// `TermVectorField` entries per [`TermVectorsDocument`], so a doc that
    /// has indexable text for two or more of `configs`' fields gets a
    /// `fields` list with one `TermVectorField` per such field, all built
    /// and written in this single pass -- no per-field
    /// `write_best_speed` call is needed, unlike postings' one
    /// `write_single_field` call per field.
    ///
    /// Returns `None` only when no field is configured. The fixed-schema
    /// facade retains indexed fields even when their text is absent: an empty
    /// native stream keeps hasVectors consistent across segments and permits
    /// Java append/merge. This acquires the writer earlier than Java's dynamic
    /// field discovery. Otherwise returns exactly `docs.len()` entries, one
    /// per doc ID -- a doc with no term-vector data of its own (for any
    /// configured field) still gets an entry with an empty `fields` list (a
    /// legitimate, readable "no term vectors for this doc" shape; see
    /// [`term_vectors::write_best_speed`]'s own tests for this exact case),
    /// never a shorter vector, since [`term_vectors::write_best_speed`]
    /// derives `max_doc` directly from `docs.len()`.
    fn build_term_vectors_output(
        docs: &[Document],
        configs: &[TermVectorFieldConfig],
        inverted: &InMemoryInvertedIndex,
    ) -> Option<Vec<TermVectorsDocument>> {
        // `per_doc[doc_id]` accumulates every configured field's
        // `TermVectorField` that has content for that doc, in ascending
        // field-**name** order.
        //
        // The order is not cosmetic and is not the caller's to choose. Real
        // Lucene's `CheckIndex.checkTermVectors` walks `TVFields.iterator()`,
        // which yields a document's term-vector fields in the order they were
        // written, and `checkFields` requires that iteration to be sorted by
        // field name (`if (lastField != null && field.compareTo(lastField) <=
        // 0) throw new CheckIndexException(...)`). The write side stores field
        // *numbers*, and this writer's numbers come from the caller's field
        // list, so number order and name order need not agree -- and
        // `add_term_vector_field` appends in call order, so before this sort a
        // caller who named "title" after "body" produced a segment real
        // `CheckIndex` rejects. Sorting `configs` here is what makes the
        // per-doc `fields` list ascending by name, because `per_doc` is
        // appended to in this loop's order. (The fix belongs
        // here, where names are known -- `term_vectors::write_best_speed` sees
        // only numbers.)
        let mut ordered_configs: Vec<&TermVectorFieldConfig> = configs.iter().collect();
        ordered_configs.sort_by(|a, b| a.name.encode_utf16().cmp(b.name.encode_utf16()));

        let mut per_doc: Vec<Vec<TermVectorField>> = vec![Vec::new(); docs.len()];

        for config in ordered_configs {
            // Regroup the shared term-keyed inverted index by doc ID: for each
            // doc, collect every term it occurs in *for this field* (ascending
            // term-byte order, since `inverted.terms` is a `BTreeMap` keyed by
            // `(field, term)`, so one field's entries are contiguous and
            // already ordered) into one `TermVectorField` for this field/doc.
            // A field whose text tokenized to nothing on every doc simply
            // contributes no terms and is skipped below, not an error.
            // Which axes this field's vector carries. Java takes these from
            // the `FieldType`'s `storeTermVectorPositions`/`Offsets`/
            // `Payloads`, all three orthogonal to `IndexOptions`; this facade
            // has no `FieldType`, so the field's declared axes stand in --
            // see `TermVectorFieldConfig`'s doc comment for why that mapping
            // is the useful one rather than merely the available one.
            //
            // Positions are unconditional, as they were before offsets and
            // payloads were wired up here: the analyzer resolves a position
            // for every token of every field regardless of `IndexOptions`, and
            // a term vector with positions over a `Docs`-only field is a
            // perfectly legal Lucene index (Java's `verifyFieldType` forbids
            // only vector payloads without vector positions, and any vector
            // axis without `storeTermVectors`). There is deliberately no
            // `has_positions` variable: a `false` case would be dead code, and
            // dead code is what a coverage number cannot tell apart from an
            // untested one.
            let has_offsets = config.index_options.subsumes_offsets();
            let has_payloads = config.store_payloads;

            let mut field_terms_per_doc: Vec<Vec<TermVectorTerm>> = vec![Vec::new(); docs.len()];
            for ((_field, term), list) in inverted
                .terms
                .range((config.name.clone(), String::new())..)
                .take_while(|((field, _), _)| field == &config.name)
            {
                // Running cursors into this term's flat payload run: term
                // vectors want the per-occurrence view, and the run stores
                // lengths rather than offsets, so the only way to it is the
                // in-order walk `Lucene104PostingsWriter` also does over its
                // own `payloadBytes`.
                let (mut byte_at, mut len_at) = (0usize, 0usize);
                for entry in &list.entries {
                    let (start_offsets, end_offsets) = if has_offsets {
                        let mut starts = Vec::with_capacity(entry.occurrences.len());
                        let mut ends = Vec::with_capacity(entry.occurrences.len());
                        for occurrence in &entry.occurrences {
                            starts.push(occurrence.start_offset);
                            ends.push(occurrence.end_offset);
                        }
                        (Some(starts), Some(ends))
                    } else {
                        (None, None)
                    };
                    let payloads = if has_payloads {
                        // The invert pass guarantees one length per occurrence
                        // for a `store_payloads` field; the saturating walk
                        // here is what makes that a checked invariant rather
                        // than an assumed one, since `write_best_speed`
                        // indexes these by occurrence -- a short run pads with
                        // empty payloads instead of panicking on a slice.
                        let mut payloads = Vec::with_capacity(entry.occurrences.len());
                        for _ in 0..entry.occurrences.len() {
                            let length = list
                                .payload_lengths
                                .get(len_at)
                                .map_or(0usize, |&l| l as usize);
                            let end = byte_at.saturating_add(length);
                            payloads.push(
                                list.payload_bytes
                                    .get(byte_at..end)
                                    .map_or_else(Vec::new, <[u8]>::to_vec),
                            );
                            byte_at = end;
                            len_at = len_at.saturating_add(1);
                        }
                        Some(payloads)
                    } else {
                        None
                    };
                    field_terms_per_doc[entry.doc_id as usize].push(TermVectorTerm {
                        term: term.as_bytes().to_vec(),
                        freq: entry.term_freq(),
                        positions: Some(entry.positions()),
                        start_offsets,
                        end_offsets,
                        payloads,
                    });
                }
            }

            for (doc_id, mut terms) in field_terms_per_doc.into_iter().enumerate() {
                if terms.is_empty() {
                    continue;
                }
                // Java: TermVectorsConsumerPerField.writeProx records actual
                // nonempty payloads separately from the field's capability.
                let has_payloads = terms.iter().any(|term| {
                    term.payloads
                        .as_ref()
                        .is_some_and(|values| values.iter().any(|value| !value.is_empty()))
                });
                if !has_payloads {
                    for term in &mut terms {
                        term.payloads = None;
                    }
                }
                per_doc[doc_id].push(TermVectorField {
                    field_number: config.field_number,
                    has_positions: true,
                    has_offsets,
                    has_payloads,
                    terms,
                });
            }
        }

        if configs.is_empty() {
            return None;
        }

        let tv_docs = per_doc
            .into_iter()
            .map(|fields| TermVectorsDocument { fields })
            .collect();
        Some(tv_docs)
    }

    /// Builds [`doc_values::write_single_dense_numeric_field`]'s (or, when some
    /// but not all docs carry a value,
    /// [`doc_values::write_single_sparse_numeric_field`]'s) input from `docs`'
    /// values for `config.field_number` (each pending doc's index into `docs`
    /// becomes its doc ID in the new segment, matching
    /// [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment)'s
    /// own doc-ordering) and calls the appropriate one to actually encode the
    /// bytes.
    ///
    /// A pending doc with no value at all for `config.field_number` is no
    /// longer always an error: if *some but not all* docs in `docs` have it,
    /// the present docs' `(doc_id, value)` pairs are routed to
    /// [`doc_values::write_single_sparse_numeric_field`] instead of the dense
    /// writer, exactly like real Lucene's own dense-vs-sparse choice at flush
    /// time (`numDocsWithValue == maxDoc`). If *no* doc has a value,
    /// [`Error::MissingDenseDocValue`] is still returned (nothing meaningful
    /// to encode, same as before this doc had zero docs to be dense over). A
    /// pending doc whose value isn't [`FieldValue::Int`]/[`FieldValue::Long`]
    /// still fails the whole commit with [`Error::NonNumericDocValue`] -- see
    /// [`IndexWriter::set_doc_values_field`]'s doc comment. Called only when
    /// `docs` is non-empty (see `commit`'s own
    /// `!self.pending_docs.is_empty()` guard), so this never has to decide
    /// what an empty-batch doc-values write even means.
    ///
    /// With **more than one** configured field all of them go into one
    /// `.dvm`/`.dvd`/`.dvs` triple through
    /// [`doc_values::write_dense_fields`] -- what a real multi-field
    /// `Lucene90DocValuesFormat` segment looks like, one meta entry per field
    /// interleaved into the same buffers, each field dense or sparse as its
    /// values require.
    fn build_doc_values_output<D: DocValueFields>(
        docs: &[D],
        configs: &[DocValuesFieldConfig],
        segment_id: &[u8; ID_LENGTH],
        suffix: u32,
        format: per_field_doc_values::Lucene90DocValuesFormat,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let skip_indexes: Vec<i32> = configs
            .iter()
            .filter(|c| c.skip_index)
            .map(|c| c.field_number)
            .collect();
        // The one-field builders write the first instance's suffix.
        if configs.len() == 1
            && skip_indexes.is_empty()
            && suffix == 0
            && format.name() == per_field_doc_values::Lucene90DocValuesFormat::NAME
        {
            let config = &configs[0];
            return match config.doc_values_type {
                DocValuesType::Binary => {
                    Self::build_binary_doc_values_output(docs, config, segment_id)
                }
                DocValuesType::Sorted => {
                    Self::build_sorted_doc_values_output(docs, config, segment_id)
                }
                DocValuesType::SortedNumeric => {
                    Self::build_sorted_numeric_doc_values_output(docs, config, segment_id)
                }
                DocValuesType::SortedSet => {
                    Self::build_sorted_set_doc_values_output(docs, config, segment_id)
                }
                _ => Self::build_numeric_doc_values_output(docs, config, segment_id),
            };
        }

        let files = Self::build_doc_values_files_multi(
            docs,
            configs,
            &skip_indexes,
            segment_id,
            suffix,
            format,
        )?;
        Ok((files.dvm, files.dvd, files.dvs))
    }

    /// [`Self::build_doc_values_output`] for a format that may write a `.dvp`
    /// ([`Lucene90DocValuesFormat::with_disi_file`]): every file the group has.
    /// A single field takes the multi-field writer, the one-field builders
    /// write version 2 only.
    ///
    /// [`Lucene90DocValuesFormat::with_disi_file`]: per_field_doc_values::Lucene90DocValuesFormat::with_disi_file
    fn build_doc_values_files<D: DocValueFields>(
        docs: &[D],
        configs: &[DocValuesFieldConfig],
        segment_id: &[u8; ID_LENGTH],
        suffix: u32,
        format: per_field_doc_values::Lucene90DocValuesFormat,
    ) -> Result<doc_values::DocValuesFiles> {
        if !format.has_disi_file() {
            let (dvm, dvd, dvs) =
                Self::build_doc_values_output(docs, configs, segment_id, suffix, format)?;
            return Ok(doc_values::DocValuesFiles {
                dvm,
                dvd,
                dvs,
                dvp: None,
            });
        }
        let skip_indexes: Vec<i32> = configs
            .iter()
            .filter(|c| c.skip_index)
            .map(|c| c.field_number)
            .collect();
        Self::build_doc_values_files_multi(docs, configs, &skip_indexes, segment_id, suffix, format)
    }

    fn build_doc_values_files_multi<D: DocValueFields>(
        docs: &[D],
        configs: &[DocValuesFieldConfig],
        skip_indexes: &[i32],
        segment_id: &[u8; ID_LENGTH],
        suffix: u32,
        format: per_field_doc_values::Lucene90DocValuesFormat,
    ) -> Result<doc_values::DocValuesFiles> {
        let mut routed = dv_columns::collect_columns(docs, configs)
            .unwrap_or_default()
            .into_iter();
        let columns: Vec<DenseColumn> = configs
            .iter()
            .map(|config| match routed.next().flatten() {
                Some(column) => Ok(column),
                None => Self::collect_dense_column(docs, config),
            })
            .collect::<Result<Vec<_>>>()?;
        let fields: Vec<doc_values::DenseField<'_>> =
            columns.iter().map(DenseColumn::as_dense_field).collect();
        Ok(doc_values::write_fields_with_format_files(
            &fields,
            skip_indexes,
            docs.len() as i32,
            segment_id,
            &format!("{}_{suffix}", format.name()),
            format,
        )?)
    }

    /// One field's column for the multi-field branch of
    /// [`Self::build_doc_values_output`]: dense when every document carries a
    /// value, else the sparse variant of the same type, as
    /// `Lucene90DocValuesConsumer` writes a field with missing values
    /// (`writeValues`' `IndexedDISI` over the documents that have one).
    fn collect_dense_column<D: DocValueFields>(
        docs: &[D],
        config: &DocValuesFieldConfig,
    ) -> Result<DenseColumn> {
        let n = config.field_number;
        let dense = |present_len: usize| present_len == docs.len();
        Ok(match config.doc_values_type {
            DocValuesType::Binary | DocValuesType::Sorted => {
                let present = Self::collect_binary_values(docs, config)?;
                let binary = config.doc_values_type == DocValuesType::Binary;
                match (dense(present.len()), binary) {
                    (true, true) => {
                        DenseColumn::Binary(n, present.into_iter().map(|(_, v)| v).collect())
                    }
                    (true, false) => {
                        DenseColumn::Sorted(n, present.into_iter().map(|(_, v)| v).collect())
                    }
                    (false, true) => DenseColumn::SparseBinary(n, present),
                    (false, false) => DenseColumn::SparseSorted(n, present),
                }
            }
            DocValuesType::SortedNumeric => {
                let present = Self::collect_sorted_numeric_values(docs, config)?;
                if dense(present.len()) {
                    DenseColumn::SortedNumeric(n, present.into_iter().map(|(_, v)| v).collect())
                } else {
                    DenseColumn::SparseSortedNumeric(n, present)
                }
            }
            DocValuesType::SortedSet => {
                let present = Self::collect_sorted_set_values(docs, config)?;
                if dense(present.len()) {
                    DenseColumn::SortedSet(n, present.into_iter().map(|(_, v)| v).collect())
                } else {
                    DenseColumn::SparseSortedSet(n, present)
                }
            }
            _ => {
                let present = Self::collect_numeric_values(docs, config)?;
                if dense(present.len()) {
                    DenseColumn::Numeric(n, present.into_iter().map(|(_, v)| v).collect())
                } else {
                    DenseColumn::SparseNumeric(n, present)
                }
            }
        })
    }

    /// `(doc_id, value)` for every doc carrying a NUMERIC value for
    /// `config`, in doc order. Shared by the dense/sparse single-field
    /// writer and by [`Self::collect_dense_column`], so the two can never
    /// disagree about what a document's value is.
    fn collect_numeric_values<D: DocValueFields>(
        docs: &[D],
        config: &DocValuesFieldConfig,
    ) -> Result<Vec<(i32, i64)>> {
        let mut present: Vec<(i32, i64)> = Vec::with_capacity(docs.len());
        for (doc_id, doc) in docs.iter().enumerate() {
            let Some(field) = doc
                .dv_fields()
                .iter()
                .find(|f| f.field_number == config.field_number)
            else {
                continue;
            };
            let value = match &field.value {
                FieldValue::Int(v) => *v as i64,
                FieldValue::Long(v) => *v,
                other => {
                    return Err(Error::NonNumericDocValue(
                        config.name.clone(),
                        doc_id,
                        field_value_kind(other),
                    ));
                }
            };
            present.push((doc_id as i32, value));
        }
        Ok(present)
    }

    /// `(doc_id, bytes)` for every doc carrying a single BINARY/SORTED value
    /// for `config`, in doc order.
    fn collect_binary_values<D: DocValueFields>(
        docs: &[D],
        config: &DocValuesFieldConfig,
    ) -> Result<Vec<(i32, Vec<u8>)>> {
        let mut present: Vec<(i32, Vec<u8>)> = Vec::with_capacity(docs.len());
        for (doc_id, doc) in docs.iter().enumerate() {
            let Some(field) = doc
                .dv_fields()
                .iter()
                .find(|f| f.field_number == config.field_number)
            else {
                continue;
            };
            let value = match &field.value {
                FieldValue::String(s) => s.as_bytes().to_vec(),
                FieldValue::Shared(s) => s.as_bytes().to_vec(),
                FieldValue::Binary(b) => b.clone(),
                other => {
                    return Err(Error::NonBinaryDocValue(
                        config.name.clone(),
                        doc_id,
                        field_value_kind(other),
                    ));
                }
            };
            present.push((doc_id as i32, value));
        }
        Ok(present)
    }

    /// `(doc_id, values)` for every doc carrying at least one SORTED_NUMERIC
    /// value for `config`, in doc order, **each document's values sorted
    /// ascending**.
    ///
    /// The sort is `SortedNumericDocValuesWriter.finishCurrentDoc`'s
    /// `Arrays.sort(currentValues, 0, currentUpto)`, and it is not cosmetic:
    /// `Lucene90DocValuesConsumer` assumes it, real
    /// `CheckIndex.checkSortedNumericDocValues` rejects a column without it
    /// (`"values out of order: ... for doc: ..."`), and
    /// `SortedNumericSelector.MIN`/`MAX` are literally "the first stored
    /// value" and "the last stored value" -- so an unsorted column makes a
    /// selector pick the wrong one rather than merely looking untidy.
    fn collect_sorted_numeric_values<D: DocValueFields>(
        docs: &[D],
        config: &DocValuesFieldConfig,
    ) -> Result<Vec<(i32, Vec<i64>)>> {
        let mut present: Vec<(i32, Vec<i64>)> = Vec::with_capacity(docs.len());
        for (doc_id, doc) in docs.iter().enumerate() {
            let mut per_doc: Vec<i64> = Vec::new();
            for field in doc
                .dv_fields()
                .iter()
                .filter(|f| f.field_number == config.field_number)
            {
                let value = match &field.value {
                    FieldValue::Int(v) => *v as i64,
                    FieldValue::Long(v) => *v,
                    other => {
                        return Err(Error::NonNumericDocValue(
                            config.name.clone(),
                            doc_id,
                            field_value_kind(other),
                        ));
                    }
                };
                per_doc.push(value);
            }
            if !per_doc.is_empty() {
                per_doc.sort_unstable();
                present.push((doc_id as i32, per_doc));
            }
        }
        Ok(present)
    }

    /// `(doc_id, values)` for every doc carrying at least one SORTED_SET
    /// value for `config`, in doc order.
    fn collect_sorted_set_values<D: DocValueFields>(
        docs: &[D],
        config: &DocValuesFieldConfig,
    ) -> Result<Vec<(i32, Vec<Vec<u8>>)>> {
        let mut present: Vec<(i32, Vec<Vec<u8>>)> = Vec::with_capacity(docs.len());
        for (doc_id, doc) in docs.iter().enumerate() {
            let mut per_doc: Vec<Vec<u8>> = Vec::new();
            for field in doc
                .dv_fields()
                .iter()
                .filter(|f| f.field_number == config.field_number)
            {
                let value = match &field.value {
                    FieldValue::String(s) => s.as_bytes().to_vec(),
                    FieldValue::Shared(s) => s.as_bytes().to_vec(),
                    FieldValue::Binary(b) => b.clone(),
                    other => {
                        return Err(Error::NonBinaryDocValue(
                            config.name.clone(),
                            doc_id,
                            field_value_kind(other),
                        ));
                    }
                };
                per_doc.push(value);
            }
            if !per_doc.is_empty() {
                present.push((doc_id as i32, per_doc));
            }
        }
        Ok(present)
    }

    /// [`Self::build_doc_values_output`]'s `DocValuesType::Numeric` branch --
    /// see that function's own doc comment for the shared dense-only/
    /// missing-value-fails-the-whole-commit contract.
    fn build_numeric_doc_values_output<D: DocValueFields>(
        docs: &[D],
        config: &DocValuesFieldConfig,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let present = Self::collect_numeric_values(docs, config)?;

        let max_doc = docs.len() as i32;
        if present.is_empty() {
            return Err(Error::MissingDenseDocValue(config.name.clone(), 0));
        }
        let output = if present.len() == docs.len() {
            let values: Vec<i64> = present.into_iter().map(|(_, v)| v).collect();
            doc_values::write_single_dense_numeric_field(
                config.field_number,
                &values,
                max_doc,
                segment_id,
                &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            )?
        } else {
            doc_values::write_single_sparse_numeric_field(
                config.field_number,
                &present,
                max_doc,
                segment_id,
                &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            )?
        };
        Ok(output)
    }

    /// [`Self::build_doc_values_output`]'s `DocValuesType::Sorted` branch:
    /// builds [`doc_values::write_single_dense_sorted_field`]'s input from
    /// `docs`' values for `config.field_number` -- same dense-only, "missing
    /// value fails the whole commit" contract as
    /// [`Self::build_numeric_doc_values_output`], except the per-doc value
    /// must be [`FieldValue::String`] (UTF-8 bytes) or [`FieldValue::Binary`]
    /// (raw bytes, real Lucene's own `SortedDocValuesField`/`BytesRef`
    /// convention) rather than numeric -- anything else fails with
    /// [`Error::NonBinaryDocValue`].
    fn build_sorted_doc_values_output<D: DocValueFields>(
        docs: &[D],
        config: &DocValuesFieldConfig,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let present = Self::collect_binary_values(docs, config)?;

        let max_doc = docs.len() as i32;
        if present.is_empty() {
            return Err(Error::MissingDenseDocValue(config.name.clone(), 0));
        }
        let output = if present.len() == docs.len() {
            let values: Vec<Vec<u8>> = present.into_iter().map(|(_, v)| v).collect();
            doc_values::write_single_dense_sorted_field(
                config.field_number,
                &values,
                max_doc,
                segment_id,
                &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            )?
        } else {
            doc_values::write_single_sparse_sorted_field(
                config.field_number,
                &present,
                max_doc,
                segment_id,
                &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            )?
        };
        Ok(output)
    }

    /// [`Self::build_doc_values_output`]'s `DocValuesType::Binary` branch:
    /// builds [`doc_values::write_single_dense_binary_field`]'s input from
    /// `docs`' values for `config.field_number` -- same dense-only, "missing
    /// value fails the whole commit" contract and same accepted
    /// [`FieldValue::String`]/[`FieldValue::Binary`] shape as
    /// [`Self::build_sorted_doc_values_output`] (BINARY has no terms
    /// dictionary/dedup/ordinals -- every doc's raw bytes are stored
    /// verbatim, unlike SORTED).
    fn build_binary_doc_values_output<D: DocValueFields>(
        docs: &[D],
        config: &DocValuesFieldConfig,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let present = Self::collect_binary_values(docs, config)?;

        let max_doc = docs.len() as i32;
        if present.is_empty() {
            return Err(Error::MissingDenseDocValue(config.name.clone(), 0));
        }
        let output = if present.len() == docs.len() {
            let values: Vec<Vec<u8>> = present.into_iter().map(|(_, v)| v).collect();
            doc_values::write_single_dense_binary_field(
                config.field_number,
                &values,
                max_doc,
                segment_id,
                &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            )?
        } else {
            doc_values::write_single_sparse_binary_field(
                config.field_number,
                &present,
                max_doc,
                segment_id,
                &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            )?
        };
        Ok(output)
    }

    /// [`Self::build_doc_values_output`]'s `DocValuesType::SortedNumeric`
    /// branch: builds [`doc_values::write_single_dense_sorted_numeric_field`]'s
    /// input from `docs`' values for `config.field_number` -- same "every doc
    /// opts into multiple values by repeating the field" shape as
    /// [`Self::build_sorted_set_doc_values_output`], except each value must be
    /// [`FieldValue::Int`]/[`FieldValue::Long`] (numeric, not binary), else
    /// [`Error::NonNumericDocValue`]; a doc with zero matching fields fails
    /// the whole commit with [`Error::MissingDenseDocValue`] (real Lucene's
    /// `SortedNumericDocValuesField` requires at least one value per doc in
    /// this dense-only shape, same as [`doc_values::write_single_dense_sorted_numeric_field`]'s
    /// own `EmptyMultiValuedDoc` guard).
    fn build_sorted_numeric_doc_values_output<D: DocValueFields>(
        docs: &[D],
        config: &DocValuesFieldConfig,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let present = Self::collect_sorted_numeric_values(docs, config)?;

        if present.is_empty() {
            return Err(Error::MissingDenseDocValue(config.name.clone(), 0));
        }
        let max_doc = docs.len() as i32;
        let output = if present.len() == docs.len() {
            let values: Vec<Vec<i64>> = present.into_iter().map(|(_, v)| v).collect();
            doc_values::write_single_dense_sorted_numeric_field(
                config.field_number,
                &values,
                segment_id,
                &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            )?
        } else {
            doc_values::write_single_sparse_sorted_numeric_field(
                config.field_number,
                &present,
                max_doc,
                segment_id,
                &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            )?
        };
        Ok(output)
    }

    /// [`Self::build_doc_values_output`]'s `DocValuesType::SortedSet` branch:
    /// builds [`doc_values::write_single_dense_sorted_set_field`]'s input from
    /// `docs`' values for `config.field_number` -- unlike
    /// [`Self::build_sorted_doc_values_output`] (exactly one value per doc), a
    /// doc's value-set here is *every*
    /// [`lucene_codecs::stored_fields::StoredField`] entry in that doc carrying
    /// `config.field_number`, so a doc opts into multiple values by simply
    /// repeating the field (real Lucene's own multi-`add`-calls-per- doc
    /// convention for `SortedSetDocValuesField`). Each such value must be
    /// [`FieldValue::String`] (UTF-8 bytes) or [`FieldValue::Binary`] (raw
    /// bytes), else [`Error::NonBinaryDocValue`]; a doc with zero matching
    /// fields fails the whole commit with [`Error::MissingDenseDocValue`] --
    /// same dense-only, "missing value fails the whole commit" contract as
    /// [`Self::build_sorted_doc_values_output`].
    fn build_sorted_set_doc_values_output<D: DocValueFields>(
        docs: &[D],
        config: &DocValuesFieldConfig,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let present = Self::collect_sorted_set_values(docs, config)?;

        if present.is_empty() {
            return Err(Error::MissingDenseDocValue(config.name.clone(), 0));
        }
        let max_doc = docs.len() as i32;
        let output = if present.len() == docs.len() {
            let values: Vec<Vec<Vec<u8>>> = present.into_iter().map(|(_, v)| v).collect();
            doc_values::write_single_dense_sorted_set_field(
                config.field_number,
                &values,
                max_doc,
                segment_id,
                &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            )?
        } else {
            doc_values::write_single_sparse_sorted_set_field(
                config.field_number,
                &present,
                max_doc,
                segment_id,
                &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            )?
        };
        Ok(output)
    }

    /// Builds this flush's `.vec`/`.vemf`/`.vem`/`.vex` bytes for every field
    /// opted in through [`IndexWriter::set_vector_field`] --
    /// `Lucene99HnswVectorsWriter.flush`, whose two halves are
    /// `Lucene99FlatVectorsWriter.flush` (the flat store) and
    /// `writeField`/`writeGraph` (the graph) on top of it.
    ///
    /// The order is Java's and is load-bearing: the flat store is written and
    /// **reopened** first, and the graph is then built over the reopened
    /// values. Java does the same (`Lucene99FlatVectorsWriter.flush` returns a
    /// `FlatCloseableRandomVectorScorerSupplier` over the bytes it just
    /// wrote), because the graph's neighbour ordinals must address the same
    /// ordinal space the `.vec` file defines. Building the graph over the
    /// in-memory `Vec<f32>` instead would be equivalent only as long as the
    /// two agree -- and an `alignOutput` or ordinal-assignment mistake is
    /// exactly the kind of thing that makes them not.
    ///
    /// A field no pending document carried a vector for is **skipped
    /// entirely**, and [`IndexingConfig::fields_with_per_field_attributes`] then zeroes
    /// its `.fnm` `vector_dimension`. Java reaches the same state from the
    /// other side: `IndexingChain` only creates a field's
    /// `KnnFieldVectorsWriter` (and only sets `FieldInfo.vectorDimension`)
    /// when a document actually carries the field, so a segment's `.fnm`
    /// never claims vectors its files do not hold.
    ///
    /// Returns `None` when no field has any vector, so no files are written.
    ///
    /// Without `indexes` (a realtime chunk of a buffer, read by searches
    /// until the buffer flushes) no graph or IVF clusters are built: every
    /// `.vem` records `numLevels = 0`, Lucene's tiny-segment case, which
    /// both readers search exhaustively with exact scores.
    #[allow(clippy::too_many_arguments)]
    fn build_vectors_output(
        partitions: Option<crate::ivf_vectors::Partitions<'_>>,
        pending_vectors: &[Vec<DocumentVector>],
        configs: &[VectorFieldConfig],
        max_doc: i32,
        format_for: &dyn Fn(&str) -> per_field_knn_vectors::KnnVectorsFormat,
        direct: bool,
        indexes: bool,
        graphs: &[BufferGraphField],
        old_to_new: Option<&[usize]>,
        clusters: &crate::ivf_vectors::ClusterSink,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<Option<VectorsOutput>> {
        // Empty when no buffered document carries a vector (a buffer of
        // explicit documents); otherwise one entry per document.
        debug_assert!(
            pending_vectors.is_empty() || pending_vectors.len() as i32 == max_doc,
            "pending_vectors must stay aligned 1:1 with pending_docs"
        );
        // `IndexingChain` calls `KnnVectorsWriter.addField` the first time a
        // document of the flush carries a field, and
        // `PerFieldKnnVectorsFormat.FieldsWriter.getInstance` numbers the
        // format instances -- and every writer lists its fields -- in that
        // order.
        let mut first_seen: Vec<&str> = Vec::new();
        for doc_vectors in pending_vectors {
            for v in doc_vectors {
                if !first_seen.contains(&v.field_name.as_str()) {
                    first_seen.push(v.field_name.as_str());
                }
            }
        }
        let mut ordered: Vec<&VectorFieldConfig> = configs.iter().collect();
        ordered.sort_by_key(|c| {
            first_seen
                .iter()
                .position(|n| *n == c.name)
                .unwrap_or(usize::MAX)
        });

        let mut flat_fields: Vec<FlatVectorsField> = Vec::new();
        let mut names: Vec<(i32, String)> = Vec::new();
        for config in ordered {
            let mut docs: Vec<i32> = Vec::new();
            let mut floats: Vec<f32> = Vec::new();
            let mut bytes: Vec<u8> = Vec::new();
            for (doc_id, doc_vectors) in pending_vectors.iter().enumerate() {
                let Some(entry) = doc_vectors.iter().find(|v| v.field_name == config.name) else {
                    continue;
                };
                docs.push(doc_id as i32);
                match &entry.value {
                    VectorValue::Float32(v) => floats.extend_from_slice(v),
                    VectorValue::Byte(v) => bytes.extend_from_slice(v),
                }
            }
            if docs.is_empty() {
                continue;
            }
            // Only one of the two accumulators can be non-empty:
            // `add_document_with_vectors` rejects a value whose variant
            // disagrees with the field's declared encoding, so the other is
            // always empty. Asserted rather than assumed, because picking the
            // wrong one here would silently drop every vector of the field.
            debug_assert!(
                floats.is_empty() || bytes.is_empty(),
                "field {:?} accumulated both float and byte components",
                config.name
            );
            let values = match config.encoding {
                VectorEncoding::Float32 => FieldVectorData::Float32(floats),
                VectorEncoding::Byte => FieldVectorData::Byte(bytes),
            };
            names.push((config.field_number, config.name.clone()));
            flat_fields.push(FlatVectorsField {
                field_number: config.field_number,
                similarity: config.similarity,
                dimension: config.dimension,
                docs,
                values,
            });
        }
        if flat_fields.is_empty() {
            return Ok(None);
        }
        let name_refs: Vec<(i32, &str)> = names.iter().map(|(n, s)| (*n, s.as_str())).collect();
        let groups = match name_refs.first() {
            Some(&(_, first)) if direct => {
                per_field_knn_vectors::direct_group(&name_refs, format_for(first))
            }
            _ => per_field_knn_vectors::group_fields(&name_refs, format_for),
        };

        let mut output = VectorsOutput {
            groups: Vec::with_capacity(groups.len()),
            written: Vec::with_capacity(names.len()),
        };
        // Each field is in exactly one group, so the groups take their
        // fields by move: a clone copied every vector a second time.
        let mut flat_fields: Vec<Option<FlatVectorsField>> =
            flat_fields.into_iter().map(Some).collect();
        for group in &groups {
            let suffix = group.codec_suffix();
            let fields: Vec<FlatVectorsField> = group
                .field_numbers
                .iter()
                .filter_map(|n| {
                    flat_fields
                        .iter_mut()
                        .find(|f| f.as_ref().is_some_and(|f| f.field_number == *n))
                        .and_then(Option::take)
                })
                .collect();
            let files = Self::build_vector_group(
                &fields,
                group.format,
                max_doc,
                indexes,
                graphs,
                old_to_new,
                clusters,
                segment_id,
                &suffix,
                partitions,
            )?;
            for (number, name) in &names {
                if group.field_numbers.contains(number) {
                    output.written.push(WrittenVectorField {
                        name: name.clone(),
                        format: group.format.name(),
                        suffix: group.suffix,
                        direct: group.direct,
                    });
                }
            }
            output.groups.push(VectorGroupFiles { suffix, files });
        }
        Ok(Some(output))
    }

    /// The graph the buffer kept over `field`'s vectors as they arrived,
    /// when it is the one this flush writes: the same vectors under the
    /// same ordinals, built with the same parameters and seed by one
    /// inserter in ordinal order, which is what
    /// `HnswGraphBuilder::new(..).build(count)` does. Under an index sort
    /// (`old_to_new`) it is renumbered to the sorted ordinals, as Lucene
    /// writes the graph its `FieldWriter` built in arrival order. A
    /// half-float format's flush graph is built over the vectors as
    /// indexed, as the kept one is.
    fn kept_graph(
        graphs: &[BufferGraphField],
        old_to_new: Option<&[usize]>,
        field: &FlatVectorsField,
        params: (i32, i32),
        #[cfg_attr(not(test), allow(unused_variables))] segment_id: &[u8; ID_LENGTH],
    ) -> Option<hnsw::OnHeapHnswGraph> {
        let kept = graphs
            .iter()
            .find(|g| g.field_number == field.field_number)?;
        let graph = &kept.graph;
        let count = i32::try_from(field.docs.len()).ok()?;
        let (encoding, similarity, dimension) = graph.shape();
        if graph.params() != params
            || graph.len() != count
            || encoding != field.values.encoding()
            || similarity != field.similarity
            || i32::try_from(dimension).ok() != Some(field.dimension)
        {
            return None;
        }
        let docs = graph.docs()?;
        let renumber = match old_to_new {
            // The ordinals name the same documents.
            None => {
                if docs != field.docs {
                    return None;
                }
                None
            }
            // `Lucene99HnswVectorsWriter.writeSortingField`: the graph was
            // built in arrival order, and each ordinal takes the place of
            // its document in the sorted segment's ordinals.
            Some(old_to_new) => {
                let mut ords = Vec::with_capacity(field.docs.len());
                for &old in &docs {
                    let old = usize::try_from(old).ok()?;
                    let new = i32::try_from(*old_to_new.get(old)?).ok()?;
                    ords.push(i32::try_from(field.docs.binary_search(&new).ok()?).ok()?);
                }
                Some(ords)
            }
        };
        let graph = graph.graph(count).ok()?;
        let graph = match renumber {
            None => graph,
            Some(ords) => graph.renumbered(&ords).ok()?,
        };
        #[cfg(test)]
        KEPT_GRAPHS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(*segment_id);
        Some(graph)
    }

    /// One `PerFieldKnnVectorsFormat` instance's flush: the raw vectors
    /// (`Lucene99FlatVectorsWriter`, `.vec`/`.vemf`), the scalar-quantized
    /// codes of its `FLOAT32` fields when the format quantizes
    /// (`Lucene104ScalarQuantizedVectorsWriter.flush`, `.veq`/`.vemq`), and
    /// the graph when it has one (`Lucene99HnswVectorsWriter`, `.vem`/`.vex`)
    /// -- built on the raw vectors, which is what Java's quantized format
    /// hands the graph at flush too.
    #[allow(clippy::too_many_arguments)]
    fn build_vector_group(
        flat_fields: &[FlatVectorsField],
        format: per_field_knn_vectors::KnnVectorsFormat,
        max_doc: i32,
        indexes: bool,
        graphs: &[BufferGraphField],
        old_to_new: Option<&[usize]>,
        clusters: &crate::ivf_vectors::ClusterSink,
        segment_id: &[u8; ID_LENGTH],
        suffix: &str,
        partitions: Option<crate::ivf_vectors::Partitions<'_>>,
    ) -> Result<Vec<(&'static str, Vec<u8>)>> {
        let (vec_bytes, vemf_bytes) =
            vectors::write_flat_vectors(flat_fields, max_doc, segment_id, suffix)?;
        let mut files: Vec<(&'static str, Vec<u8>)> = Vec::new();

        if let Some(encoding) = format.quantization() {
            let mut writer =
                lucene_codecs::scalar_quantized_vectors::ScalarQuantizedVectorsWriter::new(
                    encoding, max_doc, segment_id, suffix,
                );
            for field in flat_fields {
                if let FieldVectorData::Float32(values) = &field.values {
                    writer.write_field(
                        &lucene_codecs::scalar_quantized_vectors::QuantizedVectorsField {
                            field_number: field.field_number,
                            similarity: field.similarity,
                            dimension: field.dimension,
                            docs: &field.docs,
                            vectors: values,
                        },
                    )?;
                }
            }
            let (veq, vemq) = writer.finish();
            files.push(("veq", veq));
            files.push(("vemq", vemq));
        }

        let ivf = if indexes {
            crate::ivf_vectors::IvfGroupWriter::new(format, segment_id, suffix)?
        } else {
            None
        };
        if let Some(mut ivf) = ivf {
            for field in flat_fields {
                if let FieldVectorData::Float32(values) = &field.values {
                    ivf.add(
                        &crate::ivf_vectors::IvfField {
                            number: field.field_number,
                            similarity: field.similarity,
                            dimension: usize::try_from(field.dimension).unwrap_or(0),
                            docs: &field.docs,
                            vectors: values,
                            partitions,
                        },
                        &[],
                    )?;
                }
            }
            clusters.record(ivf.clusters());
            files.extend(ivf.finish().into_iter().flatten());
        }

        if indexes && format.hnsw_quantizer().is_some() {
            let fields: Vec<crate::hnsw_quant_vectors::CodesField<'_>> = flat_fields
                .iter()
                .filter_map(|field| match &field.values {
                    FieldVectorData::Float32(values) => {
                        Some(crate::hnsw_quant_vectors::CodesField {
                            number: field.field_number,
                            similarity: field.similarity,
                            dimension: usize::try_from(field.dimension).unwrap_or(0),
                            vectors: values,
                        })
                    }
                    _ => None,
                })
                .collect();
            if let Some(codes) =
                crate::hnsw_quant_vectors::write(format, &fields, segment_id, suffix)?
            {
                files.extend(codes);
            }
        }

        if let Some((m, beam_width)) = format.graph() {
            // Reopen exactly the bytes just written, and build every graph
            // over *those* -- see this method's doc comment.
            //
            // One graph per field, built in parallel: each depends only on
            // its own vectors and the fixed `DEFAULT_RAND_SEED`, so the bytes
            // do not change. The first error in field order is returned.
            let graphs: Vec<Option<hnsw::OnHeapHnswGraph>> = {
                use rayon::prelude::*;
                // A half-float instance's flush graph is built over the
                // vectors as indexed, as `Lucene99HnswVectorsWriter` builds
                // k-NN's from the field writer's `float[]`s: an in-memory
                // full-precision copy, ordinal for ordinal.
                let full;
                let flat = if vectors::stores_half_floats(suffix) {
                    full = vectors::write_flat_vectors(flat_fields, max_doc, segment_id, "")?;
                    vectors::FlatVectorsReader::open(&full.1, &full.0, segment_id, "")?
                } else {
                    vectors::FlatVectorsReader::open(&vemf_bytes, &vec_bytes, segment_id, suffix)?
                };
                let build = |field: &FlatVectorsField| -> Result<Option<hnsw::OnHeapHnswGraph>> {
                    let count = field.docs.len() as i32;
                    // `Lucene99HnswVectorsWriter.shouldCreateGraph`: a segment
                    // too small for a graph to beat an exhaustive scan gets
                    // none, and its `.vem` records `numLevels = 0`.
                    if !indexes || !hnsw::should_create_graph(hnsw::HNSW_GRAPH_THRESHOLD, count) {
                        return Ok(None);
                    }
                    // A Hamming format scores bytes its own way: never kept.
                    let kept = if vectors::scores_hamming(suffix) {
                        None
                    } else {
                        Self::kept_graph(graphs, old_to_new, field, (m, beam_width), segment_id)
                    };
                    if let Some(graph) = kept {
                        return Ok(Some(graph));
                    }
                    let graph = match field.values.encoding() {
                        VectorEncoding::Float32 => {
                            let values = flat.float_vector_values(field.field_number)?;
                            hnsw::HnswGraphBuilder::new(
                                values.ord_scorer(),
                                m,
                                beam_width,
                                hnsw::DEFAULT_RAND_SEED,
                            )?
                            .build(count)?
                        }
                        VectorEncoding::Byte => {
                            let values = flat.byte_vector_values(field.field_number)?;
                            hnsw::HnswGraphBuilder::new(
                                values.ord_scorer(),
                                m,
                                beam_width,
                                hnsw::DEFAULT_RAND_SEED,
                            )?
                            .build(count)?
                        }
                    };
                    Ok(Some(graph))
                };
                let built: Vec<Result<Option<hnsw::OnHeapHnswGraph>>> =
                    flat_fields.par_iter().map(build).collect();
                built.into_iter().collect::<Result<_>>()?
            };
            let hnsw_fields: Vec<HnswVectorsField<'_>> = flat_fields
                .iter()
                .zip(&graphs)
                .map(|(field, graph)| HnswVectorsField {
                    field_number: field.field_number,
                    encoding: field.values.encoding(),
                    similarity: field.similarity,
                    dimension: field.dimension,
                    count: field.docs.len() as i32,
                    graph: graph.as_ref(),
                    m,
                })
                .collect();
            let (vex_bytes, vem_bytes) =
                hnsw_vectors::write_hnsw_vectors(&hnsw_fields, segment_id, suffix)?;
            files.push(("vex", vex_bytes));
            files.push(("vem", vem_bytes));
            if indexes && format.hnsw_disk_quantizer().is_some() {
                let disk: Vec<crate::disk_hnsw_vectors::DiskField<'_>> = flat_fields
                    .iter()
                    .zip(&graphs)
                    .filter_map(|(field, graph)| match (&field.values, graph) {
                        (FieldVectorData::Float32(values), Some(graph)) => {
                            Some(crate::disk_hnsw_vectors::DiskField {
                                number: field.field_number,
                                similarity: field.similarity,
                                dimension: usize::try_from(field.dimension).unwrap_or(0),
                                docs: &field.docs,
                                vectors: values,
                                graph,
                            })
                        }
                        _ => None,
                    })
                    .collect();
                if let Some(written) =
                    crate::disk_hnsw_vectors::write(format, &disk, segment_id, suffix)?
                {
                    files.extend(written);
                }
            }
        }
        files.insert(0, ("vemf", vemf_bytes));
        files.insert(0, ("vec", vec_bytes));
        Ok(files)
    }

    /// Builds this flush's `.kdm`/`.kdi`/`.kdd` for every points field any
    /// pending document has a value for -- `PointValuesWriter.flush` into
    /// `Lucene90PointsWriter.writeField`. An opted-in field retained by another
    /// feature keeps its shape and an empty triple even without point values.
    fn build_points_output<D: DocValueFields>(
        config: &IndexingConfig,
        docs: &[D],
        configs: &[PointsFieldConfig],
        segment_id: &[u8; ID_LENGTH],
        dir: &dyn Directory,
        segment_name: &str,
    ) -> Result<Option<PointsOutput>> {
        // One pass over the documents, appending each point straight onto
        // its field's flat buffers -- `PointValuesWriter.addPackedValue`'s
        // `ByteBlockPool` plus doc-id array, with no allocation per point.
        let mut fields: Vec<lucene_codecs::points::PackedPointsField> = configs
            .iter()
            .map(|config| lucene_codecs::points::PackedPointsField {
                field_number: config.field_number,
                num_dims: config.num_dims,
                num_index_dims: config.num_index_dims,
                bytes_per_dim: config.bytes_per_dim,
                ..Default::default()
            })
            .collect();
        for (doc_id, doc) in docs.iter().enumerate() {
            for field in doc.dv_fields() {
                let Some(slot) = configs
                    .iter()
                    .position(|c| c.field_number == field.field_number)
                else {
                    continue;
                };
                encode_point(
                    &configs[slot],
                    doc_id,
                    &field.value,
                    &mut fields[slot].values,
                )?;
                fields[slot].docs.push(doc_id as i32);
            }
        }
        Self::write_points_fields(
            config,
            fields,
            configs,
            segment_id,
            dir,
            segment_name,
            docs.len() as i32,
        )
    }

    /// [`Self::build_points_output`] from each field's points already
    /// packed, numbered within the segment; a field of `configs` absent
    /// from `columns` has none.
    fn build_prebuilt_points_output(
        config: &IndexingConfig,
        columns: &[lucene_codecs::points::PackedPointsField],
        configs: &[PointsFieldConfig],
        segment_id: &[u8; ID_LENGTH],
        dir: &dyn Directory,
        segment_name: &str,
        max_doc: i32,
    ) -> Result<Option<PointsOutput>> {
        let fields = configs
            .iter()
            .map(|c| {
                columns
                    .iter()
                    .find(|f| f.field_number == c.field_number)
                    .cloned()
                    .unwrap_or_else(|| lucene_codecs::points::PackedPointsField {
                        field_number: c.field_number,
                        num_dims: c.num_dims,
                        num_index_dims: c.num_index_dims,
                        bytes_per_dim: c.bytes_per_dim,
                        ..Default::default()
                    })
            })
            .collect();
        Self::write_points_fields(
            config,
            fields,
            configs,
            segment_id,
            dir,
            segment_name,
            max_doc,
        )
    }

    /// Writes the points `fields` hold, one per field of `configs`.
    fn write_points_fields(
        config: &IndexingConfig,
        mut fields: Vec<lucene_codecs::points::PackedPointsField>,
        configs: &[PointsFieldConfig],
        segment_id: &[u8; ID_LENGTH],
        dir: &dyn Directory,
        segment_name: &str,
        max_doc: i32,
    ) -> Result<Option<PointsOutput>> {
        let written_fields: Vec<String> = configs
            .iter()
            .zip(&fields)
            .filter(|(points, values)| {
                !values.docs.is_empty()
                    || config.fields.iter().any(|field| {
                        field.number == points.field_number
                            && (field.index_options != IndexOptions::None
                                || field.doc_values_type != DocValuesType::None
                                || field.vector_dimension != 0)
                    })
            })
            .map(|(c, _)| c.name.clone())
            .collect();
        fields.retain(|f| !f.docs.is_empty());
        if fields.is_empty() && written_fields.is_empty() {
            return Ok(None);
        }
        // Empty retained fields have no write_field callback or BKD entry.
        // finish still emits a valid triple for their declared dimensions.
        // Deterministic field-number order. Java flush traverses field-hash
        // buckets; merge uses FieldInfos order. Packed values are unchanged.
        fields.sort_by_key(|f| f.field_number);
        if let Some(factory) = config
            .codec
            .as_deref()
            .and_then(|codec| codec.points_writer_factory())
        {
            let files = crate::points_writer::flush(
                factory,
                lucene_codecs::points_writer::PointsWriteState {
                    directory: dir,
                    segment: segment_name,
                    id: segment_id,
                    suffix: "",
                    max_doc,
                },
                &config.fields,
                fields,
            )?;
            return Ok(Some(PointsOutput {
                bytes: None,
                files,
                written_fields,
            }));
        }
        // The segment's `maxDoc`: `BKDWriter` sizes its doc-id keys from it.
        let (kdm, kdi, kdd) = lucene_codecs::points::write_packed(
            fields,
            MAX_POINTS_IN_LEAF_NODE,
            max_doc,
            segment_id,
            "",
        )?;
        Ok(Some(PointsOutput {
            bytes: Some((kdm, kdi, kdd)),
            files: Vec::new(),
            written_fields,
        }))
    }

    /// Writes [`IndexWriter::build_points_output`]'s three files into `dir`
    /// and returns their names for the segment's `.si`. `Lucene90PointsFormat`
    /// is not a per-field format, so the names carry no suffix.
    fn write_points_files(
        dir: &dyn Directory,
        segment_name: &str,
        output: &PointsOutput,
    ) -> Result<Vec<String>> {
        let Some((kdm, kdi, kdd)) = &output.bytes else {
            return Ok(output.files.clone());
        };
        let names = [
            format!("{segment_name}.kdm"),
            format!("{segment_name}.kdi"),
            format!("{segment_name}.kdd"),
        ];
        for (name, bytes) in names.iter().zip([kdm, kdi, kdd]) {
            let mut out = dir.create_output(name)?;
            out.write_bytes(bytes);
            out.close()?;
        }
        Ok(names.to_vec())
    }

    /// Writes [`IndexWriter::build_vectors_output`]'s four files into `dir`
    /// under `PerFieldKnnVectorsFormat`'s suffixed segment name and returns
    /// their names for the caller to record in the segment's still-unwritten
    /// `.si` -- see [`crate::segment_writer::FlushedSegment`].
    fn write_vector_files(
        dir: &dyn Directory,
        segment_name: &str,
        output: &VectorsOutput,
    ) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for group in &output.groups {
            for (ext, data) in &group.files {
                let name = segment_file_name(segment_name, &group.suffix, ext);
                write_file(dir, &name, data)?;
                names.push(name);
            }
        }
        Ok(names)
    }

    /// Writes [`IndexWriter::build_doc_values_output`]'s three files
    /// (`<segment_name>.dvm`/`.dvd`/`.dvs`) into `dir` and returns their
    /// names for the segment's still-unwritten `.si`.
    fn write_doc_values_files(
        dir: &dyn Directory,
        segment_name: &str,
        group: &DocValuesGroupOutput,
    ) -> Result<Vec<String>> {
        let seg = format!("{segment_name}_{}_{}", group.format_name, group.suffix);
        let mut names = vec![
            format!("{seg}.dvm"),
            format!("{seg}.dvd"),
            format!("{seg}.dvs"),
        ];
        for (name, bytes) in names.iter().zip([&group.dvm, &group.dvd, &group.dvs]) {
            write_file(dir, name, bytes)?;
        }
        if let Some(dvp) = &group.dvp {
            let name = format!("{seg}.dvp");
            write_file(dir, &name, dvp)?;
            names.push(name);
        }
        Ok(names)
    }

    /// Writes [`IndexWriter::build_norms_output`]'s two files
    /// (`<segment_name>.nvm`/`.nvd`) into `dir` and returns their names for
    /// the segment's still-unwritten `.si`.
    fn write_norms_files(
        dir: &dyn Directory,
        segment_name: &str,
        nvm: &[u8],
        nvd: &[u8],
    ) -> Result<Vec<String>> {
        let names = vec![format!("{segment_name}.nvm"), format!("{segment_name}.nvd")];
        for (name, bytes) in names.iter().zip([nvm, nvd]) {
            write_file(dir, name, bytes)?;
        }
        Ok(names)
    }

    /// Writes [`IndexWriter::build_term_vectors_output`]'s three files
    /// (`<segment_name>[_<suffix>].tvd`/`.tvx`/`.tvm`) into `dir` and returns their
    /// names for the segment's still-unwritten `.si`.
    fn write_term_vector_files(
        dir: &dyn Directory,
        segment_name: &str,
        suffix: &str,
        tvd: &[u8],
        tvx: &[u8],
        tvm: &[u8],
    ) -> Result<Vec<String>> {
        let names = vec![
            segment_file_name(segment_name, suffix, "tvd"),
            segment_file_name(segment_name, suffix, "tvx"),
            segment_file_name(segment_name, suffix, "tvm"),
        ];
        for (name, bytes) in names.iter().zip([tvd, tvx, tvm]) {
            write_file(dir, name, bytes)?;
        }
        Ok(names)
    }

    /// Writes [`IndexWriter::build_postings_output`]'s files
    /// (`<segment_name>.doc`/`.psm`/`.tim`/`.tip`/`.tmd`, plus `.pos`/`.pay`
    /// when the fields index them) into `dir` and returns their names for the
    /// segment's still-unwritten `.si` -- see
    /// [`crate::segment_writer::FlushedSegment`].
    fn write_postings_files(
        dir: &dyn Directory,
        segment_name: &str,
        groups: &PostingsGroups,
    ) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for group in groups {
            if !group.written_files.is_empty() {
                names.extend(group.written_files.iter().cloned());
                continue;
            }
            // `.pos`/`.pay` only exist when some field of the segment indexes
            // positions/offsets; `GroupOutput::files` leaves the empty ones
            // out, so no dangling name reaches `si.files`. A completion
            // group's `.lkp`/`.cmp` (`CompletionFieldsConsumer`) come last.
            for (ext, bytes) in group.files() {
                let name = group.file_name(segment_name, ext);
                write_file(dir, &name, bytes)?;
                names.push(name);
            }
        }
        Ok(names)
    }

    /// The automatic-merge-triggering step [`IndexWriter::commit`] runs when
    /// a [`MergePolicyConfig`] is set (see module doc comment). Repeatedly
    /// asks [`crate::merge_policy::find_merges`] for merge candidates among
    /// this writer's current committed segments and, for each proposed
    /// group, executes it via [`crate::merge::merge_stored_only_segments`]
    /// and folds the result in via [`IndexWriter::apply_merge`], until
    /// `find_merges` proposes nothing further. Terminates because every
    /// executed merge strictly reduces this writer's segment count by at
    /// least one (merging >= 2 segments into exactly 1).
    fn auto_merge(&mut self) -> Result<()> {
        if let Some(policy) = self.pluggable_merge_policy.clone() {
            return self.auto_merge_pluggable(&policy);
        }
        let config = self
            .merge_policy
            .clone()
            .expect("auto_merge only called when merge_policy is Some");

        loop {
            let stats = self.segment_stats()?;
            let groups =
                self.per_partition_stats(&stats, |s| merge_policy::find_merges(s, &config))?;
            if groups.is_empty() {
                break;
            }
            for group in groups {
                self.execute_merge(&group)?;
            }
        }
        Ok(())
    }

    /// Builds the [`crate::merge_policy::SegmentStat`] list
    /// [`IndexWriter::auto_merge`] feeds to
    /// [`crate::merge_policy::find_merges`], sourced from this writer's
    /// current committed segments: `doc_count` comes from each segment's own
    /// `.si` file (via [`crate::segment_info::parse`]), the on-disk size from
    /// every file the commit says the segment owns, `.liv` included
    /// ([`crate::merge_policy::commit_byte_size`], Java's
    /// `SegmentCommitInfo.sizeInBytes`), `del_count` from this writer's
    /// own [`SegmentCommitInfo`] (already in memory, no re-read needed).
    ///
    /// **Every format this writer can flush is now mergeable**, so the only
    /// segments held back are the ones `execute_merge` cannot round-trip:
    ///
    /// - Stored fields, postings, term vectors, norms, doc values and KNN
    ///   vectors are all opened by
    ///   [`execute_merge`](IndexWriter::execute_merge) and merged through
    ///   [`crate::merge::merge_segments`], which writes every one of them.
    ///   An **index-sorted** segment merges too, through the same function's
    ///   k-way merge, and the merged `.si` keeps the sort.
    /// - A segment carrying a doc-values **update** (`doc_values_gen != -1`)
    ///   is mergeable too. Its newest column lives in generational
    ///   `.dvm`/`.dvd` files no `.si` lists, so reading the base pair would
    ///   silently resurrect the pre-update values. `execute_merge` resolves every field to
    ///   its **current** generation through
    ///   [`crate::field_updates::read_current_field_infos`] and
    ///   [`crate::field_updates::read_current_column`], the same two
    ///   functions the update path itself reads its base from, and the merged
    ///   segment folds every generation back into one base column
    ///   (`doc_values_gen == -1`), which is what
    ///   `IndexWriter.mergeMiddle` produces.
    ///
    /// **Nothing is withheld.** That is not a claim about this method: it is
    /// what [`crate::merge::check_format_coverage`] enforces on every merge,
    /// from the source segments' own file lists.
    ///
    /// The bar for adding a format here is that losing it must be
    /// impossible, not unlikely: a merged segment that quietly dropped a
    /// format would still be valid, still pass `CheckIndex`, and differ from
    /// the truth only in what it answers.
    fn segment_stats(&self) -> Result<Vec<merge_policy::SegmentStat>> {
        let mut stats = Vec::with_capacity(self.segment_infos.segments.len());
        for sci in &self.segment_infos.segments {
            let si = self.cfg.reader_pool.segment_info(self.dir, sci)?;
            let size_bytes = merge_policy::commit_byte_size(self.dir, sci, &si);
            stats.push(merge_policy::SegmentStat {
                name: sci.segment_name.clone(),
                doc_count: si.doc_count,
                del_count: sci.del_count,
                size_bytes,
            });
        }
        Ok(stats)
    }

    /// Executes one merge group `names` proposed by
    /// [`crate::merge_policy::find_merges`]: opens each named segment's
    /// stored fields (and live-docs bitset, if it has deletions) straight off
    /// `dir`, merges them via [`crate::merge::merge_stored_only_segments`]
    /// into a brand-new segment, and folds the result into this writer's
    /// committed state via [`IndexWriter::apply_merge`] (which itself writes
    /// the next `segments_N` generation -- each executed merge group is its
    /// own commit, same as a caller manually driving
    /// [`IndexWriter::apply_merge`] would produce).
    ///
    /// **Postings.** If a source segment's `.si` lists a `.tim` file (i.e.
    /// it has postings at all -- written when
    /// [`IndexWriter::set_postings_field`]/[`IndexWriter::add_postings_field`]
    /// was configured at flush time), this opens that segment's `.tim`/
    /// `.tip`/`.tmd` term dictionary (via [`lucene_codecs::blocktree::open`])
    /// and `.doc` file (via [`lucene_codecs::postings::DocInput::open`]) and
    /// builds one [`crate::merge::SourcePostings`] per field this writer's
    /// own `self.cfg.fields` schema marks as postings-eligible *and* that
    /// segment's term dictionary actually has an entry for. "Eligible" is
    /// exactly what [`IndexWriter::set_postings_field`] accepts, **positional
    /// options included** -- the `.pos`/`.pay` files are opened alongside when
    /// the source's `.si` lists them, because `merge_postings` needs them for
    /// a field whose merged `index_options` indexes positions. Narrowing this
    /// to `Docs`/`DocsAndFreqs` would drop a positional field's postings from
    /// the merged segment while its `.fnm` still declared them: an indexed
    /// field with no registered postings producer, which reads back as having
    /// no terms and raises nothing.
    ///
    /// **Term vectors.** If a source segment's `.si` lists a `.tvd` file
    /// (written when [`IndexWriter::set_term_vector_field`]/
    /// [`IndexWriter::add_term_vector_field`] was configured at flush time),
    /// this opens that segment's `.tvd`/`.tvx`/`.tvm` via
    /// [`lucene_codecs::term_vectors::open`] and sets the resulting
    /// [`lucene_codecs::term_vectors::TermVectorsReader`] as
    /// [`crate::merge::MergeSource::term_vectors`], which
    /// [`crate::merge::merge_stored_only_segments`]'s existing
    /// `crate::merge::write_merged_term_vectors` plumbing then reads per doc. Since
    /// every segment this writer ever flushes shares this writer's own single
    /// `self.cfg.fields` schema (field numbers are never reassigned per segment),
    /// every source's term-vector field numbers already agree with the
    /// merged field numbers -- the field-number remap
    /// `crate::merge::write_merged_term_vectors` applies is the identity mapping
    /// in practice here, same as it is for postings above.
    ///
    /// Points are opened like every other format: each source's
    /// `.kdm`/`.kdi`/`.kdd` (when its `.si` lists them) becomes one
    /// [`crate::merge::SourcePoints`] per field, and
    /// [`crate::merge::merge_points`] remaps and rebuilds the trees.
    fn execute_merge(&mut self, names: &[String]) -> Result<()> {
        self.execute_merge_hooked(names, None)
    }

    /// [`Self::execute_merge`] of a `OneMerge` whose `wrapForMerge` and
    /// `reorder` are `hooks` (see [`crate::merge_policy::MergeHooks`]).
    pub(crate) fn execute_merge_hooked(
        &mut self,
        names: &[String],
        hooks: Option<std::sync::Arc<dyn crate::merge_policy::MergeHooks>>,
    ) -> Result<()> {
        self.write_pending_try_changes()?;
        self.execute_merge_planned(names, hooks, None, false)
            .map(|_| ())
    }

    /// Makes `plan` run `hooks` (`OneMerge.wrapForMerge`/`reorder`) over its
    /// sources, as they were when the merge began.
    pub(crate) fn attach_merge_hooks(
        &self,
        plan: &mut MergePlan,
        hooks: Option<std::sync::Arc<dyn crate::merge_policy::MergeHooks>>,
    ) {
        if let Some(hooks) = hooks {
            let mut infos = self.segment_infos.clone();
            infos.segments = plan.sources.clone();
            plan.hooks = Some((hooks, infos));
        }
    }

    /// [`Self::execute_merge_hooked`], with SORTED_NUMERIC values rewritten
    /// through `remap`, as `addIndexes(CodecReader...)`'s merge when
    /// `add_indexes`: whether the merge was published.
    pub(crate) fn execute_merge_planned(
        &mut self,
        names: &[String],
        hooks: Option<std::sync::Arc<dyn crate::merge_policy::MergeHooks>>,
        remap: Option<std::sync::Arc<merge::SortedNumericRemap>>,
        add_indexes: bool,
    ) -> Result<bool> {
        self.write_pending_try_changes()?;
        let mut plan = self.begin_merge(names)?;
        plan.sorted_numeric_remap = remap;
        if add_indexes {
            plan.kind = merge::MergeKind::AddIndexes;
        }
        self.attach_merge_hooks(&mut plan, hooks);
        let cfg = std::sync::Arc::clone(&self.cfg);
        let report = MergeReport::start(self.merge_event(&plan), None);
        let result = match cfg.run_merge(self.dir, &plan) {
            Ok(outcome) => self.finish_merge(plan, outcome),
            Err(e) => {
                // Its own failure must not mask the merge's.
                let _ = self.abort_merge(plan);
                Err(e)
            }
        };
        report.finish(&result);
        result
    }

    /// `IndexWriter.forceMerge(maxNumSegments)` over the committed segments.
    /// With a [`MergePolicyConfig`] set it asks `TieredMergePolicy.findForcedMerges`
    /// until nothing is proposed, its merged-size cap replaced by
    /// [`IndexWriter::set_forced_merge_max_segment_size`]'s when one is set;
    /// down to one segment that is every segment merged at once, largest
    /// first (which fixes the merged document order). Without a policy, while
    /// more than `max_num_segments` remain, the smallest are merged into one.
    /// A lone segment left with hard deletes
    /// is rewritten when `max_num_segments` is 1. Each merge publishes its
    /// own commit carrying the current commit data. Buffered and
    /// flushed-but-uncommitted documents are not part of it -- commit
    /// first, as OpenSearch's engine does before a force merge.
    pub fn force_merge(&mut self, max_num_segments: usize) -> Result<()> {
        self.write_pending_try_changes()?;
        if max_num_segments == 0 {
            return Err(Error::InvalidMaxNumSegments(max_num_segments));
        }
        let outer = std::mem::replace(
            &mut self.merge_max_num_segments,
            i32::try_from(max_num_segments).unwrap_or(i32::MAX),
        );
        let merged = self.force_merge_inner(max_num_segments);
        self.merge_max_num_segments = outer;
        merged
    }

    /// [`Self::force_merge`] with `mergeMaxNumSegments` set.
    fn force_merge_inner(&mut self, max_num_segments: usize) -> Result<()> {
        if let Some(policy) = self.pluggable_merge_policy.clone() {
            return self.force_merge_pluggable(&policy, max_num_segments);
        }
        if let Some(mut config) = self.merge_policy.clone() {
            if let Some(cap) = self.forced_merge_max_segment_size {
                config.max_merged_segment_size = cap;
            }
            return self.run_forced_merges(|stats| {
                merge_policy::find_forced_merges(stats, max_num_segments, &config)
            });
        }
        // Every partition on its own: multi-segment merges until none is
        // left, then each single segment with deletes merged once.
        loop {
            let stats = self.segment_stats()?;
            let picks = self.per_partition_stats(&stats, |part| {
                fallback_forced_merge(part, max_num_segments)
                    .into_iter()
                    .collect()
            })?;
            if picks.is_empty() {
                break;
            }
            for names in picks {
                self.execute_merge(&names)?;
            }
        }
        if max_num_segments == 1 {
            let stats = self.segment_stats()?;
            let singles = self.per_partition_stats(&stats, |part| match part {
                [only] => vec![only.clone()],
                _ => Vec::new(),
            })?;
            for only in singles {
                if self.num_deletes_to_merge(&only)? > 0 {
                    self.execute_merge(std::slice::from_ref(&only.name))?;
                }
            }
        }
        Ok(())
    }

    /// [`IndexWriter::segment_stats`] with each `del_count` the deletes a
    /// merge would reclaim ([`IndexWriter::num_deletes_to_merge`]), as
    /// `TieredMergePolicy`'s forced-merge selection counts them.
    fn reclaimable_segment_stats(&self) -> Result<Vec<merge_policy::SegmentStat>> {
        let mut stats = self.segment_stats()?;
        for stat in &mut stats {
            stat.del_count = self.num_deletes_to_merge(stat)?;
        }
        Ok(stats)
    }

    /// The forced-merge loop under a [`MergePolicyConfig`]: asks `find` for
    /// merges over the segments' reclaimable-delete stats and runs them, until
    /// it proposes none.
    fn run_forced_merges(
        &mut self,
        find: impl Fn(&[merge_policy::SegmentStat]) -> Vec<Vec<String>>,
    ) -> Result<()> {
        loop {
            let stats = self.reclaimable_segment_stats()?;
            let groups = self.per_partition_stats(&stats, &find)?;
            if groups.is_empty() {
                return Ok(());
            }
            for group in groups {
                self.execute_forced_merge(group, &stats)?;
            }
        }
    }

    /// Runs one forced merge of `group`, interleaved first when
    /// [`IndexWriter::set_shuffle_forced_merges`] is on.
    fn execute_forced_merge(
        &mut self,
        group: Vec<String>,
        stats: &[merge_policy::SegmentStat],
    ) -> Result<()> {
        if !self.cfg.shuffle_forced_merges {
            return self.execute_merge(&group);
        }
        let names = shuffle_interleave(group, stats);
        let outer = std::mem::replace(&mut self.merge_shuffled, true);
        let merged = self.execute_merge(&names);
        self.merge_shuffled = outer;
        merged
    }

    /// `MergePolicy.numDeletesToMerge` under this writer's retention policy:
    /// the segment's hard deletes plus the soft-deleted documents
    /// [`IndexWriter::set_soft_deletes_retention`] no longer keeps. Without a
    /// retention policy a merge keeps every soft-deleted document, so only
    /// the hard deletes count.
    fn num_deletes_to_merge(&self, stat: &merge_policy::SegmentStat) -> Result<i32> {
        let (Some(retention), Some(soft)) = (
            &self.soft_deletes_retention,
            self.cfg.fields.iter().find(|f| f.soft_deletes_field),
        ) else {
            return Ok(stat.del_count);
        };
        let Some(sci) = self
            .segment_infos
            .segments
            .iter()
            .find(|s| s.segment_name == stat.name)
            .filter(|s| s.soft_del_count > 0)
        else {
            return Ok(stat.del_count);
        };
        let reclaimable =
            count_soft_deletes(self.dir, sci, &soft.name, Some(retention))?.unwrap_or(0);
        Ok(stat.del_count.saturating_add(reclaimable))
    }

    /// `IndexWriter.forceMergeDeletes()` under `TieredMergePolicy.findForcedDeletesMerges`.
    /// With a [`MergePolicyConfig`] set it asks that function, with the
    /// policy's `force_merge_deletes_pct_allowed` and caps, until nothing is
    /// proposed. Without one, every committed segment whose reclaimable
    /// deletes -- [`IndexWriter::num_deletes_to_merge`], so soft deletes
    /// retention still keeps do not count -- exceed
    /// [`FORCE_MERGE_DELETES_PCT_ALLOWED`]% of its documents is merged, in one
    /// merge. A no-op, writing no commit, when no segment qualifies.
    pub fn force_merge_deletes(&mut self) -> Result<()> {
        self.write_pending_try_changes()?;
        if let Some(policy) = self.pluggable_merge_policy.clone() {
            return self.force_merge_deletes_pluggable(&policy);
        }
        if let Some(config) = self.merge_policy.clone() {
            return self.run_forced_merges(|stats| {
                merge_policy::find_forced_delete_merges(stats, &config)
            });
        }
        let mut names: Vec<String> = Vec::new();
        for stat in self.segment_stats()? {
            let reclaimable = i64::from(self.num_deletes_to_merge(&stat)?);
            // `100 * numDeletesToMerge / maxDoc > forceMergeDeletesPctAllowed`,
            // in integers.
            if reclaimable.saturating_mul(100)
                > i64::from(stat.doc_count).saturating_mul(FORCE_MERGE_DELETES_PCT_ALLOWED)
            {
                names.push(stat.name);
            }
        }
        if names.is_empty() {
            return Ok(());
        }
        self.execute_merge(&names)
    }

    /// The start of a merge, `IndexWriter.mergeInit`: claim the merged
    /// segment's name and id, and snapshot each source's commit info -- its
    /// deletion and doc-values generations as the merge will read them.
    pub(crate) fn begin_merge(&mut self, names: &[String]) -> Result<MergePlan> {
        self.write_pending_try_changes()?;
        // A merged segment is published with `buffered_deletes_gen == -1`,
        // i.e. open to every packet, where Java uses `min(sources)`
        // (`IndexWriter.mergeMiddle`). That is only safe while the stream is
        // empty at merge time -- which it is: `flush()` applies and takes
        // every packet before `finish_commit` runs `auto_merge`, and a
        // concurrent writer applies each packet under its control lock in
        // the call that pushed it, so the stream is empty whenever the lock
        // is free. Assert
        // it rather than rely on the reading: if a packet ever outlives the
        // call that pushed it, this is the line that says so.
        debug_assert!(
            !self.updates_stream.any(),
            "a delete packet outlived its flush; a merged segment would take it"
        );
        let merged_name = self.new_segment_name();
        let merged_id = generate_segment_id(self.segment_infos.counter);
        let sources = names
            .iter()
            .map(|name| {
                self.segment_infos
                    .segments
                    .iter()
                    .find(|s| &s.segment_name == name)
                    .expect("merge_policy::find_merges only proposes segment names this writer currently has committed")
                    .clone()
            })
            .collect::<Vec<SegmentCommitInfo>>();
        let early_compound = match self.cfg.merged_segment_warmer {
            Some(_) => Some(self.compound_file_decider()?),
            None => None,
        };
        let mut attributes = self.merged_partition_attributes(&sources)?;
        self.merged_segment_attributes(&sources, &mut attributes)?;
        let held = self.deleter.hold_segment_files(&sources)?;
        Ok(MergePlan {
            early_compound,
            attributes,
            names: names.to_vec(),
            sources,
            merged_name,
            merged_id,
            held,
            retention: self.soft_deletes_retention.clone(),
            hooks: None,
            sorted_numeric_remap: None,
            kind: merge::MergeKind::Merge {
                max_num_segments: self.merge_max_num_segments,
                shuffled: self.merge_shuffled,
            },
        })
    }

    /// A merge that failed before [`Self::finish_merge`]: release its hold on
    /// the sources. Returned merge errors remove outputs owned by that attempt.
    /// A later `delete_unused_files` or reopen reclaims any leftovers, including
    /// files from a completed merge that failed before publication.
    pub(crate) fn abort_merge(&mut self, plan: MergePlan) -> Result<()> {
        self.write_pending_try_changes()?;
        self.deleter.release_files(&plan.held)?;
        Ok(())
    }

    /// The end of a merge, `IndexWriter.commitMerge`: publish the merged
    /// segment in place of its sources -- or retire them, when the merge held
    /// no live document.
    ///
    /// **Deletes made while the merge ran** (`commitMergedDeletesAndUpdates`).
    /// A source may have lost documents after [`IndexWriter::begin_merge`]
    /// snapshotted it -- only with a
    /// [`crate::concurrent_writer::ConcurrentIndexWriter`], whose merges run
    /// without the writer's lock. Those deletes are carried onto the merged
    /// segment through the merge's doc-id maps, as Java carries them. A source
    /// whose *doc values* were updated meanwhile, or that is gone (a
    /// `delete_all`), abandons the merge instead: its files are deleted and
    /// the sources stay, to be merged again later. Java carries doc-values
    /// updates too; this port does not yet, and says so by not merging.
    ///
    /// Returns whether the merge was published (`false`: abandoned).
    pub(crate) fn finish_merge(&mut self, plan: MergePlan, outcome: MergeOutcome) -> Result<bool> {
        self.write_pending_try_changes()?;
        let held = plan.held.clone();
        let result = self.finish_merge_holding(plan, outcome);
        // After the merge is published (or abandoned): released last, so the
        // sources' files go only once nothing else references them.
        // A published merge already dropped its sources' pooled postings in
        // `commit_merge`, before this release can delete their files.
        let released = self.deleter.release_files(&held);
        let published = result?;
        released?;
        Ok(published)
    }

    fn finish_merge_holding(&mut self, plan: MergePlan, outcome: MergeOutcome) -> Result<bool> {
        let source_names: Vec<&str> = plan.names.iter().map(|s| s.as_str()).collect();
        let (merged, source_live, compound_decided) = match outcome {
            MergeOutcome::AllDeleted => {
                self.drop_merge(&source_names)?;
                return Ok(true);
            }
            MergeOutcome::Merged {
                merged,
                source_live,
                compound_decided,
            } => (*merged, source_live, compound_decided),
        };
        let mut merged = merged;
        // `IndexWriter.mergeMiddle`: `mergePolicy.useCompoundFile(segmentInfos,
        // merge.info, this)`, then `createCompoundFile` -- before deletes that
        // arrived during the merge are carried over, as there.
        if !compound_decided && self.merged_segment_uses_compound_file(&merged)? {
            before_write(
                self.cfg.codec.as_deref(),
                lucene_codecs::codec::WritePoint::Compound,
            )?;
            merged.files = segment_writer::convert_to_compound(
                self.dir,
                &merged.info.segment_name,
                &merged.info.segment_id,
            )?;
        }
        let mut info = merged.info;
        let merged_max_doc = merged
            .doc_id_maps
            .iter()
            .flatten()
            .filter(|&&d| d >= 0)
            .count();
        let mut carried: Vec<i32> = Vec::new();
        for (i, snapshot) in plan.sources.iter().enumerate() {
            let current = self
                .segment_infos
                .segments
                .iter()
                .find(|s| s.segment_name == snapshot.segment_name);
            let unchanged_values = current.is_some_and(|c| {
                c.field_infos_gen == snapshot.field_infos_gen
                    && c.doc_values_gen == snapshot.doc_values_gen
            });
            let Some(current) = current.filter(|_| unchanged_values) else {
                // Abandoned: nothing references these files yet.
                self.deleter.delete_new_files(&merged.files)?;
                return Ok(false);
            };
            if current.del_gen == snapshot.del_gen {
                continue;
            }
            let map = &merged.doc_id_maps[i];
            let liv = self.dir.open(&deletes::liv_file_name(
                &current.segment_name,
                current.del_gen,
            ))?;
            let now_live = lucene_codecs::live_docs::parse(
                &liv,
                &current.segment_id,
                current.del_gen,
                map.len(),
                current.del_count as usize,
            )?;
            let was_live = |doc: usize| source_live[i].as_ref().is_none_or(|b| b.get(doc));
            for (doc, &merged_doc) in map.iter().enumerate() {
                if merged_doc >= 0 && was_live(doc) && !now_live.get(doc) {
                    carried.push(merged_doc);
                }
            }
        }
        if !carried.is_empty() {
            info = deletes::apply_deletes_with_codec(
                self.dir,
                &info,
                None,
                merged_max_doc,
                carried,
                self.cfg.codec.as_deref(),
            )?;
        }

        // `merge_segments` stamps the merged `.si` with this writer's version.
        self.segment_versions.insert(
            info.segment_name.clone(),
            to_segment_infos_version(self.cfg.lucene_version),
        );
        self.apply_merge(&source_names, info)?;
        Ok(true)
    }

    /// One source segment's per-document key for one index-sort tier, read
    /// out of that segment's own NUMERIC doc-values column -- the merge-time
    /// counterpart of `sort_pending_buffer`'s per-document read at flush
    /// time, and the port of `IndexSorter.LongSorter.getComparableValues`.
    ///
    /// `None` for a document with no value; the comparator substitutes the
    /// tier's sentinel, exactly as `SortField`'s `missingValue` does.
    ///
    /// A segment whose `.dvm` has no entry for the sort field at all is
    /// refused rather than silently read as all-missing: `set_index_sort`
    /// makes a sort field's column mandatory, so its absence means the
    /// segment does not satisfy the sort its own `.si` declares, and a merge
    /// that quietly ordered every one of its documents by a sentinel would
    /// produce a segment that is `CheckIndex`-clean and in the wrong order.
    ///
    /// This reads the **base** column, not the newest generation, which is
    /// sound twice over: `set_index_sort` refuses a doc-values update against
    /// a sort field (`IndexWriter.updateDocValues`' own guard), and
    /// [`IndexWriter::segment_stats`] withholds any segment with a
    /// doc-values generation from the merge policy at all.
    ///
    /// Deleted documents' keys are read too. They are never compared --
    /// `sorted_doc_order` walks only live doc ids -- but the slice is indexed
    /// by the source's own doc id, so it has to be `max_doc` long.
    fn read_sort_keys(
        columns: &SourceDocValueColumns,
        fields: &[FieldInfo],
        sort: &segment_info::IndexSortField,
        max_doc: i32,
    ) -> Result<Vec<Option<i64>>> {
        let field_name = sort.field.as_str();
        // A source none of whose documents has the field (OpenSearch's
        // delete tombstones, say) sorts every document as missing, as
        // Java's empty `DocValues.getNumeric` does.
        let Some(field_number) = fields
            .iter()
            .find(|f| f.name == field_name)
            .map(|f| f.number)
        else {
            return Ok(vec![None; usize::try_from(max_doc).unwrap_or(0)]);
        };
        let at = columns
            .per_field
            .iter()
            .find(|(n, _)| *n == field_number)
            .map(|&(_, at)| at)
            .ok_or_else(|| Error::MergeSortColumnMissing(field_name.to_string()))?;
        let (meta, dvd) = &columns.columns[at];
        let missing = || Error::MergeSortColumnMissing(field_name.to_string());
        match &sort.kind {
            segment_info::IndexSortKind::SortedNumeric { selector, .. } => {
                let entry = meta
                    .sorted_numeric_entry(field_number)
                    .ok_or_else(missing)?;
                // `SortedNumericSelector.MinValue`/`MaxValue`: the first and
                // the last stored value of the document, the column being
                // ascending by construction.
                (0..max_doc)
                    .map(|doc| {
                        let values = doc_values::sorted_numeric_values(dvd, entry, doc)?;
                        Ok(match selector {
                            segment_info::SortedNumericSelector::Min => values.into_iter().next(),
                            segment_info::SortedNumericSelector::Max => values.into_iter().last(),
                        })
                    })
                    .collect()
            }
            // A single-valued NUMERIC column; the byte-keyed kinds are read
            // by `read_sort_terms` and never reach here.
            _ => {
                let entry = meta.numeric_entry(field_number).ok_or_else(missing)?;
                // One `NumericReader` for the whole column, not a
                // `numeric_value` call per document: that free function
                // re-walks the sparse docs-with-field region from its start on
                // every lookup, and this walks `0..max_doc` once per sort tier
                // per source. `NumericReader`'s own doc comment names a sort
                // as the caller that must hold one.
                let mut reader = doc_values::NumericReader::new(dvd, entry);
                (0..max_doc).map(|doc| Ok(reader.value(doc)?)).collect()
            }
        }
    }

    /// [`Self::read_sort_keys`] for a byte-keyed sort tier: each document's
    /// bytes out of the source's current SORTED / SORTED_SET (through the
    /// tier's selector) / BINARY column, to be ranked across every source
    /// by [`crate::index_sorter::rank_terms`]. A source without the field
    /// and a missing column are as for a numeric tier.
    fn read_sort_terms(
        columns: &SourceDocValueColumns,
        fields: &[FieldInfo],
        sort: &segment_info::IndexSortField,
        max_doc: i32,
    ) -> Result<Vec<Option<Vec<u8>>>> {
        let field_name = sort.field.as_str();
        let Some(field_number) = fields
            .iter()
            .find(|f| f.name == field_name)
            .map(|f| f.number)
        else {
            return Ok(vec![None; usize::try_from(max_doc).unwrap_or(0)]);
        };
        let missing = || Error::MergeSortColumnMissing(field_name.to_string());
        let at = columns
            .per_field
            .iter()
            .find(|(n, _)| *n == field_number)
            .map(|&(_, at)| at)
            .ok_or_else(missing)?;
        let (meta, dvd) = &columns.columns[at];
        crate::index_sorter::read_segment_terms(dvd, meta, sort, field_number, max_doc)?
            .ok_or_else(missing)
    }

    /// `IndexWriter.applyAllDeletesAndUpdates()`: freeze whatever the global
    /// buffer still holds, then resolve every pending
    /// [`FrozenBufferedUpdates`] packet against every segment it is allowed to
    /// touch.
    ///
    /// "Allowed to touch" is the whole contract, and it is one comparison:
    /// a packet applies to a segment iff the segment's
    /// `buffered_deletes_gen` is `<=` the packet's `del_gen`
    /// ([`FrozenBufferedUpdates::applies_to`]). A segment flushed *after* a
    /// delete was issued carries a strictly higher generation and is therefore
    /// skipped -- which is what makes "a delete applies to every document with
    /// a smaller sequence number and to no document with a larger one" true
    /// across segment boundaries. Within the one segment a packet is private
    /// to, the per-entry `docIDUpto` limits do the same job document by
    /// document ([`FrozenBufferedUpdates::limit_for`]).
    ///
    /// Packets are applied oldest generation first, so a later doc-values
    /// update for the same document wins, matching Java's `mergedIterator`
    /// ordering.
    fn apply_all_deletes_and_updates(&mut self) -> Result<()> {
        if let Some(global) = self.delete_queue.freeze_global_buffer() {
            self.updates_stream.push(global);
        }
        self.apply_stream_packets()
    }

    /// Applies every packet already in the stream, without freezing the
    /// deletes buffered since -- see [`Self::publish_segment_with`].
    fn apply_stream_packets(&mut self) -> Result<()> {
        if !self.updates_stream.any() {
            return Ok(());
        }
        let packets: Vec<FrozenBufferedUpdates> = self.updates_stream.take_pending();
        // A packet is fully resolved inside the call that pushed it -- with
        // one indexing thread, or under a concurrent writer's control lock --
        // so Java's `FinishedSegments` in-flight bookkeeping collapses to
        // "everything pending is now done".
        //
        // That "inside the call that pushed it" is also what makes merging safe
        // without Java's `BufferedUpdatesStream.waitApplyForMerge`. A merged
        // segment is published with `buffered_deletes_gen == -1`, i.e. open to
        // every packet, which would be wrong if a packet targeting only the
        // *sources* could still be pending when the merge lands. It cannot:
        // the stream is empty whenever the control lock is free, and a merge
        // is published under it. A packet frozen by a ticket but not yet
        // pushed holds only deletes issued after every committed segment's
        // documents were added, so it rightly reaches the merged segment. A delete
        // still sitting in the *queue* (not yet frozen) is a different and
        // correct case -- it has not been applied to the sources either, and
        // the merged segment carries their live documents forward, so applying
        // it to the merged segment reaches exactly the same documents.

        // `std::mem::take` rather than an index loop: the apply below mutates
        // each segment while borrowing the writer's config, and the segment
        // lists are put back verbatim (plus their new generations) once it is
        // done.
        //
        // On a failed apply the lists still go back, and so do the packets:
        // returning early would drop both, so the next commit would publish a
        // `segments_N` naming none of the committed segments and the deleter
        // would reclaim their files. Java never reaches that state, because a
        // failed packet apply is a tragic event that closes the writer
        // (`onTragicEvent(t, "applyUpdatesPacket")`); this writer stays
        // usable, so a retry re-applies the packets, which is idempotent for
        // the segments the failed round already reached.
        //
        // Segments are applied in parallel: each one reads only its own
        // files and writes only its own `.liv`/doc-values generations, the
        // directory is `Sync`, and the reader pool is behind a lock. The
        // results come back in segment order, and the first error in that
        // order is the one returned.
        use rayon::prelude::*;
        let cfg = std::sync::Arc::clone(&self.cfg);
        let dir = self.dir;
        let sort_maps = &self.pending_sort_map;
        let apply = |sci: &mut SegmentCommitInfo| {
            let sort_map = sort_maps.iter().find(|(name, _)| name == &sci.segment_name);
            cfg.apply_packets_to_segment(dir, &packets, sci, sort_map)
        };
        let mut committed = std::mem::take(&mut self.segment_infos.segments);
        let mut flushed = std::mem::take(&mut self.flushed_segments);
        let (committed_results, flushed_results): (Vec<Result<bool>>, Vec<Result<bool>>) =
            rayon::join(
                || committed.par_iter_mut().map(apply).collect(),
                || flushed.par_iter_mut().map(apply).collect(),
            );
        self.segment_infos.segments = committed;
        self.flushed_segments = flushed;
        let committed_fully_deleted: Result<Vec<bool>> = committed_results.into_iter().collect();
        let flushed_fully_deleted: Result<Vec<bool>> = flushed_results.into_iter().collect();
        let (committed_fully_deleted, flushed_fully_deleted) =
            match (committed_fully_deleted, flushed_fully_deleted) {
                (Ok(c), Ok(f)) => (c, f),
                (Err(e), _) | (_, Err(e)) => {
                    self.updates_stream.restore_pending(packets);
                    return Err(e);
                }
            };

        self.drop_fully_deleted_segments(&committed_fully_deleted, &flushed_fully_deleted);
        // `ReaderPool.drop` for what this round (or anything since the last
        // one) took out of the index -- a segment it left fully deleted
        // included.
        let live = self.live_infos();
        let names: std::collections::HashSet<&str> = live
            .segments
            .iter()
            .map(|s| s.segment_name.as_str())
            .collect();
        self.cfg.reader_pool.retain(&names);
        Ok(())
    }

    /// Drops the flushed segment `name`, which its own buffer's deletes left
    /// with no live document ([`IndexingConfig::apply_flush_private_updates`])
    /// -- as [`Self::drop_fully_deleted_segments`] does when those deletes are
    /// applied at publish instead, and under the same policy.
    pub(crate) fn drop_fully_deleted_flushed(&mut self, name: &str) -> Result<()> {
        self.write_pending_try_changes()?;
        let flushed: Vec<bool> = self
            .flushed_segments
            .iter()
            .map(|s| s.segment_name == name)
            .collect();
        self.drop_fully_deleted_segments(&[], &flushed);
        // Its own deletes pooled its postings: close them before the
        // checkpoint deletes its files.
        self.cfg.reader_pool.drop_segments(&[name]);
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
        Ok(())
    }

    /// `IndexWriter.finishApply`'s second half: drop every segment this apply
    /// left 100% deleted.
    ///
    /// ```java
    /// if (result.allDeleted() != null) {
    ///   for (SegmentCommitInfo info : result.allDeleted()) { dropDeletedSegment(info); }
    ///   checkpoint();
    /// }
    /// ```
    ///
    /// The membership test is `closeSegmentStates`':
    /// `rld.isFullyDeleted() && mergePolicy.keepFullyDeletedSegment(...) == false`,
    /// where `PendingDeletes.isFullyDeleted` is `getDelCount() == maxDoc()` --
    /// **hard deletes only**. A soft-deleted document still counts as live
    /// here, which is why `PendingSoftDeletes` overrides the *policy* hook
    /// rather than the count (see
    /// [`MergePolicyConfig::keep_fully_deleted_segments`]).
    ///
    /// Only segments this apply actually touched are candidates, exactly as in
    /// Java: `openSegmentStates` filters to the segments the packets are
    /// allowed to reach, so a segment that was already fully deleted before
    /// this apply is not reconsidered. [`IndexingConfig::apply_packets_to_segment`] returns
    /// `false` for a segment it skipped, which reproduces that.
    ///
    /// This is an **in-memory** drop plus the deleter checkpoint the caller
    /// already runs, not a new commit: Java's `checkpoint()` is
    /// `changed(); deleter.checkpoint(segmentInfos, false)`, and a dropped
    /// segment only becomes durable at the next `commit()`. A `rollback()`
    /// before that restores it from `rollback_segments`, unchanged.
    ///
    /// *Not modelled*: `dropDeletedSegment`'s `mergingSegments` guard -- a
    /// [`crate::concurrent_writer::ConcurrentIndexWriter`] merge may be
    /// running over the dropped segment, and [`Self::finish_merge`] abandons
    /// a merge whose source is gone instead -- and `adjustPendingNumDocs`
    /// (no such counter -- see `crates/lucene-index/src/deletes.rs`' module
    /// doc). `readerPool.drop` is the caller's, before its checkpoint.
    fn drop_fully_deleted_segments(&mut self, committed: &[bool], flushed: &[bool]) {
        if self
            .merge_policy
            .as_ref()
            .is_some_and(|c| c.keep_fully_deleted_segments)
        {
            return;
        }
        let mut keep = committed.iter();
        self.segment_infos
            .segments
            .retain(|_| !keep.next().copied().unwrap_or(false));
        let mut keep = flushed.iter();
        self.flushed_segments
            .retain(|_| !keep.next().copied().unwrap_or(false));
        self.prune_segment_versions();
    }

    /// `FrozenBufferedUpdates.applyQueryDeletes`' inner
    /// `weight.scorer(readerContext)` walk, for the query shapes this crate
    /// can resolve on its own -- see [`DeleteQuery`] for why the set is closed.
    ///
    /// `max_doc_bound` bounds the doc space [`DeleteQuery::MatchAll`] and
    /// [`DeleteQuery::Not`] enumerate; every other shape derives its docs from
    /// the term dictionary and ignores it.
    fn resolve_delete_query(
        query: &DeleteQuery,
        opened: &DeleteSegmentView<'_>,
        max_doc_bound: usize,
    ) -> Result<Vec<i32>> {
        let live = |doc: i32| opened.live_docs.is_none_or(|b| b.get(doc as usize));
        Ok(match query {
            DeleteQuery::Term(term) => term_delete::resolve_term_doc_ids(
                opened.fields,
                opened.doc_in.as_ref(),
                opened.live_docs,
                &term.field,
                &term.bytes,
            )?,
            DeleteQuery::MatchAll => (0..max_doc_bound.min(opened.max_doc) as i32)
                .filter(|&d| live(d))
                .collect(),
            DeleteQuery::Prefix { field, prefix } => Self::resolve_term_span(opened, field, |t| {
                if t.starts_with(prefix.as_slice()) {
                    TermSpan::Take
                } else if t > prefix.as_slice() {
                    TermSpan::Stop
                } else {
                    TermSpan::Skip
                }
            })?,
            DeleteQuery::TermRange {
                field,
                lower,
                upper,
                include_lower,
                include_upper,
            } => Self::resolve_term_span(opened, field, |t| {
                if let Some(lo) = lower {
                    let below = if *include_lower {
                        t < lo.as_slice()
                    } else {
                        t <= lo.as_slice()
                    };
                    if below {
                        return TermSpan::Skip;
                    }
                }
                if let Some(hi) = upper {
                    let above = if *include_upper {
                        t > hi.as_slice()
                    } else {
                        t >= hi.as_slice()
                    };
                    if above {
                        return TermSpan::Stop;
                    }
                }
                TermSpan::Take
            })?,
            DeleteQuery::Any(clauses) => {
                let mut out: Vec<i32> = Vec::new();
                for clause in clauses {
                    out.extend(Self::resolve_delete_query(clause, opened, max_doc_bound)?);
                }
                out.sort_unstable();
                out.dedup();
                out
            }
            DeleteQuery::All(clauses) => {
                let mut iter = clauses.iter();
                let Some(first) = iter.next() else {
                    // `BooleanQuery` with no MUST clause matches nothing.
                    return Ok(Vec::new());
                };
                let mut out = Self::resolve_delete_query(first, opened, max_doc_bound)?;
                out.sort_unstable();
                out.dedup();
                for clause in iter {
                    let mut next = Self::resolve_delete_query(clause, opened, max_doc_bound)?;
                    next.sort_unstable();
                    next.dedup();
                    out.retain(|d| next.binary_search(d).is_ok());
                }
                out
            }
            DeleteQuery::Not(inner) => {
                let mut excluded = Self::resolve_delete_query(inner, opened, max_doc_bound)?;
                excluded.sort_unstable();
                excluded.dedup();
                (0..max_doc_bound.min(opened.max_doc) as i32)
                    .filter(|&d| live(d) && excluded.binary_search(&d).is_err())
                    .collect()
            }
        })
    }

    /// The shared term-dictionary walk behind `PrefixQuery` and
    /// `TermRangeQuery`: `TermsEnum.next()` from the start of `field`, letting
    /// `classify` decide per term whether it contributes documents, is skipped,
    /// or ends the walk. Ending the walk early is what makes a prefix or an
    /// upper-bounded range cost the matching span rather than the whole field,
    /// matching `TermRangeQuery`'s own `AutomatonQuery` behaviour.
    fn resolve_term_span(
        opened: &DeleteSegmentView<'_>,
        field: &str,
        classify: impl Fn(&[u8]) -> TermSpan,
    ) -> Result<Vec<i32>> {
        let Some(field_terms) = opened.fields.field(field) else {
            return Ok(Vec::new());
        };
        let mut matched_terms: Vec<Vec<u8>> = Vec::new();
        let mut terms = field_terms.iter();
        // `try_next_term`, not `try_next`: this walk classifies on the term
        // bytes and never looks at `docFreq`/`totalTermFreq`, and
        // `term_delete::resolve_term_doc_ids` re-seeks each kept term anyway.
        while let Some(term) = terms.try_next_term()? {
            match classify(term) {
                TermSpan::Take => matched_terms.push(term.to_vec()),
                TermSpan::Skip => {}
                TermSpan::Stop => break,
            }
        }
        let mut out: Vec<i32> = Vec::new();
        for term in matched_terms {
            out.extend(term_delete::resolve_term_doc_ids(
                opened.fields,
                opened.doc_in.as_ref(),
                opened.live_docs,
                field,
                &term,
            )?);
        }
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }

    /// This writer's most recently committed [`SegmentInfos`] -- does not
    /// reflect any not-yet-`commit()`ed [`IndexWriter::add_document`] calls.
    /// Exposed so a caller can drive [`crate::merge_policy::find_merges`]/
    /// [`crate::merge::merge_stored_only_segments`] manually (see module doc
    /// comment: merging is not automatically triggered by this facade).
    pub fn segment_infos(&self) -> &SegmentInfos {
        &self.segment_infos
    }

    /// The segments of the last commit as it was written
    /// (`rollbackSegments`): [`Self::segment_infos`] also takes deletes
    /// applied since.
    pub(crate) fn committed_segments(&self) -> &[SegmentCommitInfo] {
        &self.rollback_segments
    }

    /// `IndexWriter.getSegmentInfosCounter()`: the counter the next
    /// segment's name is drawn from.
    pub fn segment_infos_counter(&self) -> i64 {
        self.segment_infos.counter
    }

    /// `IndexWriter.advanceSegmentInfosCounter(newCounter)`: the next
    /// segment names start at `new_counter`; a lower value changes nothing.
    /// Java's `changed()` bumps the version too; here the next commit
    /// publishes the counter and the next snapshot's version comes from
    /// its segments (see [`crate::nrt`]).
    pub fn advance_segment_infos_counter(&mut self, new_counter: i64) {
        if self.segment_infos.counter < new_counter {
            self.segment_infos.counter = new_counter;
        }
    }

    /// `IndexWriter.getSegmentCount()`: the segments of the writer's
    /// in-memory view -- the last commit's and those flushed since.
    pub fn segment_count(&self) -> usize {
        self.segment_infos
            .segments
            .len()
            .saturating_add(self.flushed_segments.len())
    }

    /// Number of documents buffered by [`IndexWriter::add_document`] but not
    /// yet written to disk by a [`IndexWriter::commit`] call.
    pub fn pending_doc_count(&self) -> usize {
        self.pending_docs.len()
    }

    /// Documents or deletes buffered and not yet in a segment's files --
    /// what `IndexWriter.nrtIsCurrent` checks beside the segment list
    /// (`docWriter.anyChanges()`, `bufferedUpdatesStream.any()`).
    pub(crate) fn has_buffered_changes(&self) -> bool {
        !self.pending_docs.is_empty()
            || !self.pending_try_deletes.is_empty()
            || !self.pending_try_updates.is_empty()
            || self.delete_queue.any_changes()
            || self.updates_stream.any()
    }

    /// Pins `segments`' files in the deleter until
    /// [`IndexWriter::release_files`] -- what an NRT reader holds.
    pub(crate) fn pin_segment_files(
        &mut self,
        segments: &[SegmentCommitInfo],
    ) -> Result<Vec<String>> {
        self.write_pending_try_changes()?;
        Ok(self.deleter.hold_segment_files(segments)?)
    }

    /// `IndexWriter.hasUncommittedChanges()`: whether the next commit would
    /// publish anything -- buffered documents, buffered deletes or updates, a
    /// segment flushed since the last commit, or deletes already applied to a
    /// committed segment.
    pub fn has_uncommitted_changes(&self) -> bool {
        !self.pending_docs.is_empty()
            || !self.pending_try_deletes.is_empty()
            || !self.pending_try_updates.is_empty()
            || self.delete_queue.any_changes()
            || self.updates_stream.any()
            || !self.flushed_segments.is_empty()
            || self.segment_infos.segments != self.rollback_segments
            || self.change_count != self.last_commit_change_count
    }

    /// Total number of documents actually committed to disk right now,
    /// summed across every segment in [`IndexWriter::segment_infos`].
    ///
    /// **Semantics: total, not live.** This sums each segment's `doc_count`
    /// as read from that segment's own `.si` file (via
    /// [`segment_info::parse`], the same source [`IndexWriter::segment_stats`]
    /// already uses for [`crate::merge_policy::find_merges`]) -- it does
    /// **not** subtract `del_count`/soft deletes. A document that was
    /// `delete_documents`/`update_document`-deleted after being committed is
    /// still physically present in its segment's stored-fields file (this
    /// facade never rewrites a segment in place to remove a deleted doc --
    /// only a merge drops it, by omitting it from the merged output) and so
    /// is still counted here. Callers that want the *live* (non-deleted)
    /// count must subtract each [`SegmentCommitInfo::del_count`] themselves
    /// (`self.segment_infos().segments.iter().map(|s| s.del_count as
    /// usize).sum()`), or open each segment's `.liv` file (if any) via
    /// [`lucene_codecs::live_docs::parse`] to count set bits directly.
    ///
    /// **Distinct from [`IndexWriter::pending_doc_count`]:** this method
    /// only reflects documents durably written by a prior
    /// [`IndexWriter::commit`] -- [`IndexWriter::add_document`] calls not
    /// yet committed are never counted here, matching
    /// [`IndexWriter::segment_infos`]'s own "most recently committed" scope.
    pub fn committed_doc_count(&self) -> Result<usize> {
        let mut total = 0usize;
        for sci in &self.segment_infos.segments {
            let si = self.cfg.reader_pool.segment_info(self.dir, sci)?;
            // ARITH: `segment_info::parse` rejects a negative `doc_count`, so
            // each term is in `0..=i32::MAX`, and `segments_N`'s own parse
            // rejects a segment count above the number of bytes left in that
            // file -- so the number of terms is bounded by the `segments_N`
            // length, itself at most `i32::MAX`. The product is below `2^62`,
            // inside `usize` on both targets this workspace builds for.
            #[allow(clippy::arithmetic_side_effects)]
            {
                total += si.doc_count as usize;
            }
        }
        Ok(total)
    }

    /// Discards every document buffered by [`IndexWriter::add_document`]
    /// since the last [`IndexWriter::commit`] -- real Lucene's
    /// `IndexWriter.rollback()`, scoped to what this facade actually has to
    /// roll back.
    ///
    /// **What gets reset:** `pending_docs` (this writer's in-memory buffer of
    /// not-yet-flushed [`Document`]s) and `prepared_commit` (any state
    /// stashed by a prior [`IndexWriter::prepare_commit`] that hasn't been
    /// activated by [`IndexWriter::finish_commit`] yet -- discarding this
    /// too is what makes `rollback()` actually undo "everything not yet
    /// durably committed," matching what its own name implies; leaving a
    /// dangling `prepared_commit` in place would let a *later* unrelated
    /// `finish_commit()` call silently activate segments the caller just
    /// asked to roll back). Discarding a prepared commit also deletes the
    /// `pending_segments_N` file it wrote, which is real
    /// `SegmentInfos.rollbackCommit` -- any failure to delete it is ignored,
    /// exactly as Java's `IOUtils.deleteFilesIgnoringExceptions` does, since
    /// an undeleted pending file is inert and must not turn a rollback into
    /// an error. It also drops every segment an automatic or explicit
    /// [`IndexWriter::flush`] wrote since the last commit, and -- through
    /// `IndexFileDeleter.checkpoint` + `refresh`, Java's own
    /// `rollbackInternal` sequence -- **deletes those segments' files**, along
    /// with anything an operation that failed partway left behind. `rollback`
    /// never reads or writes `segment_infos`'s committed segment list. Calling
    /// `rollback()` when `pending_doc_count()` is already `0` and no commit is
    /// prepared (including right after `IndexWriter::open`) is a safe no-op.
    ///
    /// **What is preserved, deliberately:** this writer's already-committed
    /// [`IndexWriter::segment_infos`] (any *prior* `commit()`'s segments are
    /// on disk and are never touched by this call, so they remain fully
    /// readable/searchable after a rollback -- only documents added *after*
    /// the last commit are discarded) and every writer-configuration field
    /// set via [`IndexWriter::set_postings_field`]/
    /// [`IndexWriter::set_term_vector_field`]/
    /// [`IndexWriter::set_doc_values_field`]/[`IndexWriter::set_merge_policy`].
    /// This matches real Lucene's own split between `IndexWriterConfig`
    /// (survives a `rollback()`, since it belongs to the caller, not to any
    /// one buffered batch of documents) and buffered-but-uncommitted document
    /// state (discarded) -- `rollback()` only ever undoes *documents*, never
    /// *configuration*.
    ///
    /// **Not replicated from real Lucene's `rollback()`:** real
    /// `IndexWriter.rollback()` also closes the writer and permanently
    /// releases its write lock, so the `IndexWriter` instance itself becomes
    /// unusable afterward (any further call throws
    /// `AlreadyClosedException`). This facade's `rollback()` keeps its
    /// `write.lock` ([`WRITE_LOCK_NAME`]) and leaves the writer fully usable for
    /// further [`IndexWriter::add_document`]/[`IndexWriter::commit`] calls
    /// immediately afterward. Drop releases that lock. Cleanup errors are
    /// ignored by this compatibility method; [`Self::try_rollback`] propagates
    /// them as Java does. If restored source metadata cannot be read or its
    /// schema conflicts with fields configured since `delete_all`, rollback
    /// leaves the writer's current document state unchanged; this method
    /// ignores that error, while `try_rollback` reports it.
    pub fn rollback(&mut self) {
        if self.rollback_state().is_err() {
            return;
        }
        let live = self.live_infos();
        let _ = self.deleter.checkpoint(&live, false);
        let _ = self.deleter.refresh();
    }

    /// Discards uncommitted changes and propagates deleter cleanup failures,
    /// as Java's `IndexWriter.rollbackInternal` does. Like [`Self::rollback`],
    /// this keeps the writer usable and retains its lock until it drops.
    ///
    /// # Errors
    /// Restored field metadata cannot be read or reconciled, or the deleter
    /// cannot checkpoint the restored commit or refresh its files.
    pub fn try_rollback(&mut self) -> Result<()> {
        self.rollback_state()?;
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
        self.deleter.refresh()?;
        Ok(())
    }

    fn rollback_state(&mut self) -> Result<()> {
        // Retained configuration may conflict with fields from the restored
        // commit after deleteAll. Reject before changing any document state.
        let field_numbers = self.prepare_rollback_field_numbers()?;
        self.discard_pending_try_changes();
        // `ReaderPool.dropAll()`: nothing opened for the discarded changes
        // survives the rollback.
        self.cfg.reader_pool.clear();
        self.pending_docs.clear();
        self.pending_custom_freq_terms.clear();
        self.pending_vectors.clear();
        self.pending_explicit.clear();
        self.pending_postings.clear();
        self.ram_bytes_used = 0;
        // Only ever non-empty inside `flush()`, and cleared at its end -- but a
        // `flush()` that fails *after* publishing the segment (in
        // `apply_all_deletes_and_updates`) leaves it set, and `rollback` is
        // the caller's answer to that.
        self.pending_sort_map.clear();
        self.pending_has_blocks = false;
        // `IndexWriter.rollbackInternal`: `docWriter.close()` (which discards
        // the delete queue's buffers) and `bufferedUpdatesStream.clear()`.
        // The *sequence number counter* deliberately survives -- Java builds a
        // fresh queue whose seqNos continue past the aborted ones, so a caller
        // can never see the same seqNo twice from one writer.
        self.delete_queue.clear();
        self.updates_stream.clear();
        // Segments an automatic (or explicit) flush already wrote but that no
        // commit names are exactly what a rollback is supposed to undo. Dropping
        // them from the in-memory view is what makes the checkpoint below take
        // their reference counts to zero.
        self.flushed_segments.clear();
        // `SegmentInfos.rollbackSegmentInfos(rollbackSegments)`: restore the
        // *committed* segment list too. Applying a buffered delete bumps a
        // committed segment's `del_gen` in this in-memory view and writes a
        // `.liv` no commit yet names -- without this restore the view would
        // survive the rollback pointing at a file the `refresh()` below
        // reclaims. `generation`/`version`/`counter` deliberately stay put:
        // Java leaves them alone too, and they only ever move forward.
        self.segment_infos.segments = self.rollback_segments.clone();
        self.prepared_nothing = false;
        if let Some(prepared) = self.prepared_commit.take() {
            // `SegmentInfos.rollbackCommit`: drop the `pending_segments_N` the
            // prepare wrote, ignoring any failure (Java uses
            // `IOUtils.deleteFilesIgnoringExceptions` here for exactly the same
            // reason -- an undeleted pending file is inert, so failing to
            // remove it must not turn a rollback into an error).
            segment_infos::rollback_pending(&prepared, self.dir);
        }
        if let Some(numbers) = field_numbers {
            self.cfg_mut().explicit_field_numbers = numbers;
        }
        Ok(())
    }

    /// Port of `IndexWriter.deleteAll()`: drops every buffered document *and*
    /// every segment currently in this writer's view of the index, leaving it
    /// logically empty.
    ///
    /// Like Java's, this does **not** itself write a commit -- it is an
    /// in-memory operation on `segmentInfos` plus an abort of the buffered
    /// documents (Java's `docWriter.lockAndAbortAll()`, then
    /// `segmentInfos.clear()` and `changed()`), so the emptiness only becomes
    /// durable on the next [`IndexWriter::commit`]. Until then a fresh
    /// [`crate::segment_infos::read_latest`] still sees the previous commit,
    /// exactly as in Java.
    ///
    /// Like Java's, this checkpoints the empty view with `IndexFileDeleter`.
    /// Uncommitted segment files are reclaimed; files referenced by a retained
    /// commit survive until that commit is removed by the deletion policy.
    ///
    /// Refused while a [`IndexWriter::prepare_commit`] is outstanding, for the
    /// same reason [`IndexWriter::delete_documents_by_term`] is.
    pub fn delete_all(&mut self) -> Result<()> {
        self.cfg.reader_pool.clear();
        if self.prepared_commit.is_some() {
            return Err(Error::PreparedCommitPending("delete_all"));
        }
        self.discard_pending_try_changes();
        self.pending_docs.clear();
        self.pending_custom_freq_terms.clear();
        self.pending_vectors.clear();
        self.pending_explicit.clear();
        self.pending_postings.clear();
        self.ram_bytes_used = 0;
        self.pending_has_blocks = false;
        // Java's `deleteAll`: `docWriter.lockAndAbortAll()` (which clears the
        // delete queue) and `bufferedUpdatesStream.clear()`. Every buffered
        // delete targets documents that no longer exist, so keeping them would
        // apply them to whatever is added next.
        self.delete_queue.clear();
        self.updates_stream.clear();
        self.flushed_segments.clear();
        self.segment_infos.segments.clear();
        // `deleteAll`: `changed()` -- the next commit is written even when
        // the index was already empty.
        self.changed();
        // Java's `deleteAll` ends with `deleter.checkpoint(segmentInfos,
        // false)`. Segments that only this writer's in-memory view referenced
        // (an uncommitted flush) are reclaimed right here; segments the current
        // commit still names survive until the next commit supersedes it, which
        // is exactly why `deleteAll` is not itself durable.
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
        self.reset_explicit_field_numbers()?;
        Ok(())
    }

    /// Port of `IndexWriter.setLiveCommitData(Iterable<Map.Entry<String,
    /// String>>)`: opaque caller metadata (`SegmentInfos.userData`) carried
    /// verbatim in the next `segments_N` this writer produces, and readable
    /// back from any commit via [`crate::segment_infos::SegmentInfos::user_data`].
    ///
    /// Java's `setLiveCommitData` applies to the *next* commit, and the data
    /// then persists across subsequent commits until replaced -- this matches
    /// that: the value is stored on this writer's current `SegmentInfos` and
    /// every later clone (`commit`, `delete_documents`, `apply_merge`,
    /// `update_document`) carries it forward.
    ///
    /// Like Java's `setLiveCommitData(data)` (whose `doIncrementVersion`
    /// defaults to `true`), setting commit data counts as a change: the next
    /// [`IndexWriter::commit`] writes a new generation even if nothing else
    /// changed.
    pub fn set_live_commit_data(&mut self, data: Vec<(String, String)>) {
        self.segment_infos.user_data = data;
        self.changed();
    }

    /// `auto`'s choice, once made, into the live commit data under
    /// [`crate::adaptive_auto::COMMIT_KEY`], so the commit carries it and a
    /// writer reopened on it keeps it.
    fn stamp_adaptive_doc_values_choice(&mut self) {
        let Some(choice) = self.adaptive_doc_values_choice() else {
            return;
        };
        let key = crate::adaptive_auto::COMMIT_KEY;
        let value = choice.to_string();
        let data = &mut self.segment_infos.user_data;
        match data.iter_mut().find(|(k, _)| k == key) {
            Some((_, v)) => *v = value,
            None => data.push((key.to_string(), value)),
        }
    }

    /// `IndexWriter.changed()`: a change the next commit must write even if
    /// the segment list is unchanged (see [`Self::change_count`]).
    pub(crate) fn changed(&mut self) {
        self.change_count = self.change_count.wrapping_add(1);
    }

    /// Port of `IndexWriter.getLiveCommitData()`: whatever
    /// [`IndexWriter::set_live_commit_data`] last set (or what the resumed
    /// commit carried at [`IndexWriter::open`]).
    pub fn live_commit_data(&self) -> &[(String, String)] {
        &self.segment_infos.user_data
    }

    /// Replaces this writer's committed segment list with `merged` in place
    /// of `source_segment_names` -- the composition point for a caller that
    /// has just run [`crate::merge::merge_stored_only_segments`] against
    /// segments from [`IndexWriter::segment_infos`] and wants the result
    /// folded back into this writer's own view of the index (so a later
    /// `add_document`/`commit` builds on top of the merged state instead of
    /// the pre-merge one). Writes the updated segment list as the next
    /// `segments_N` generation, same commit shape as
    /// [`IndexWriter::commit`]/[`IndexWriter::update_document`].
    ///
    /// This does **not** call [`crate::merge_policy::find_merges`] or
    /// [`crate::merge::merge_stored_only_segments`] itself -- see the module
    /// doc comment's "no automatic merge triggering" scope note. It is
    /// purely the bookkeeping half: fold an already-completed merge's result
    /// into this writer's committed state.
    pub fn apply_merge(
        &mut self,
        source_segment_names: &[&str],
        merged: SegmentCommitInfo,
    ) -> Result<&SegmentInfos> {
        self.write_pending_try_changes()?;
        self.commit_merge("apply_merge", source_segment_names, Some(merged))
    }

    /// `IndexWriter.commitMerge`'s `dropSegment == true` branch: retire
    /// `source_segment_names` **without** publishing anything in their place.
    ///
    /// Java reaches it when the merge's result holds no live document
    /// (`allDeleted`), and its comment says exactly what the call is for --
    /// "Merge would produce a 0-doc segment, so we do nothing except commit
    /// the merge to remove all the 0-doc segments that we merged". The
    /// sources really are removed: `SegmentInfos.applyMergeChanges` deletes
    /// every merged-away entry from the segment list and, because
    /// `dropSegment` is set, inserts nothing, so the commit that follows
    /// names neither the sources nor a merged segment.
    ///
    /// There is nothing to delete on this side that Java's
    /// `deleteNewFiles(merge.info.files())` would delete, because
    /// [`IndexWriter::execute_merge`] takes this path *before* the merge
    /// writes a single file -- the same order `mergeMiddle` uses when it
    /// skips `merger.merge()`.
    pub fn drop_merge(&mut self, source_segment_names: &[&str]) -> Result<&SegmentInfos> {
        self.write_pending_try_changes()?;
        self.commit_merge("drop_merge", source_segment_names, None)
    }

    /// The body both [`IndexWriter::apply_merge`] and
    /// [`IndexWriter::drop_merge`] share -- `IndexWriter.commitMerge` with
    /// `merged == None` standing for its `dropSegment` flag.
    fn commit_merge(
        &mut self,
        caller: &'static str,
        source_segment_names: &[&str],
        merged: Option<SegmentCommitInfo>,
    ) -> Result<&SegmentInfos> {
        if self.prepared_commit.is_some() {
            return Err(Error::PreparedCommitPending(caller));
        }
        // Same invariant `execute_merge` asserts: a merged segment is
        // published open to every packet, which is only safe while there are
        // none pending. A caller driving a merge by hand between a buffered
        // delete and the flush that resolves it would violate it.
        debug_assert!(
            !self.updates_stream.any(),
            "a delete packet outlived its flush; the merged segment would take it"
        );
        // `closeMergeReaders(merge, false, dropSegment)` and `readerPool.drop`
        // before `checkpoint()` -- "Must close before checkpoint, otherwise
        // IFD won't be able to delete the held-open files from the merge
        // readers". A source a delete round pooled keeps its `.doc` open, so
        // dropping it after the deleter reclaims the sources would delete a
        // file still open.
        self.cfg.reader_pool.drop_segments(source_segment_names);
        if self.merges_by_caller {
            // `commitMerge`: swap the segments and checkpoint; the next
            // `commit()` makes it durable. The last commit still names the
            // sources, so the deleter keeps their files until then.
            apply_merge_changes(
                &mut self.segment_infos.segments,
                source_segment_names,
                merged,
            );
            self.prune_segment_versions();
            let live = self.live_infos();
            self.deleter.checkpoint(&live, false)?;
            return Ok(&self.segment_infos);
        }
        let mut new_segment_infos = self.segment_infos.clone();
        // ARITH: `segment_infos::parse` rejects any generation, version or
        // counter outside `-1..=MAX_GENERATION` (`i64::MAX / 2`), and
        // `segment_infos::write` refuses to persist one, so every value this
        // writer can be holding is at most `i64::MAX / 2` -- climbing from
        // there to `i64::MAX` would take 2^62 further commits. See
        // `segment_infos::MAX_GENERATION`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            new_segment_infos.generation += 1;
            new_segment_infos.version += 1;
        }
        // Above every version an NRT snapshot was given, so `getVersion`
        // stays monotonic across snapshots and commits (see `crate::nrt`).
        // Never used with NRT readers, `max_version` is -1 and this is a
        // no-op.
        new_segment_infos.version = new_segment_infos
            .version
            .max(self.nrt.max_version.saturating_add(1));
        new_segment_infos.id = generate_segment_id(new_segment_infos.generation);
        apply_merge_changes(
            &mut new_segment_infos.segments,
            source_segment_names,
            merged,
        );

        self.stamp_min_segment_version(&mut new_segment_infos)?;
        self.stamp_soft_delete_counts(&mut new_segment_infos)?;
        // This commits on its own: sync what the commit names first
        // (`startCommit`).
        self.deleter.sync_commit_files(&new_segment_infos)?;
        new_segment_infos.format_version = self.segment_infos_format_version();
        let written = segment_infos::write(&new_segment_infos, self.dir);
        // `updateGeneration` on error as on success (`startCommit`).
        self.segment_infos.generation = new_segment_infos.generation;
        written?;
        self.last_commit_change_count = self.change_count;
        // The merge's source segments are no longer named by any live commit
        // once the superseded commit point dies here, so this is what actually
        // reclaims them -- `commitMerge` -> `checkpoint` in Java.
        self.segment_infos = new_segment_infos;
        self.prune_segment_versions();
        // `finishCommit`: `rollbackSegments =
        // pendingCommit.createBackupSegmentInfos()`. This is the one place a
        // commit becomes durable, so it is the one place the rollback snapshot
        // moves forward.
        self.rollback_segments = self.segment_infos.segments.clone();
        self.checkpoint_committed()?;
        Ok(&self.segment_infos)
    }

    /// Real `IndexFileNames.segmentFileName`'s `_<counter in base 36>`
    /// convention, driven off this writer's current `segment_infos.counter`
    /// so segment names never collide with an earlier session's, even when
    /// resuming an already-committed directory.
    fn new_segment_name(&mut self) -> String {
        let name = format!(
            "_{}",
            lucene_util::base36::to_base36(self.segment_infos.counter)
        );
        // Java bumps `segmentInfos.counter` in the same synchronized block that
        // hands out the name, precisely so a name can never be handed out twice.
        // ARITH: `counter` is capped at `MAX_GENERATION` (`i64::MAX / 2`) by
        // `segment_infos::parse` and by `segment_infos::write`, and steps once
        // per segment name handed out.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.segment_infos.counter += 1;
        }
        name
    }
}

/// A brand-new, empty [`SegmentInfos`] for a directory with no existing
/// commit -- generation/version/counter all start at `0`, no segments, a
/// freshly generated commit id (see [`generate_segment_id`]'s doc comment on
/// why this facade doesn't use a real CSPRNG here).
/// Everything resolving a buffered delete against one segment needs, owned:
/// this port's slice of Java's `ReadersAndUpdates`. [`OpenedDeleteSegment::view`]
/// turns it into the borrowed form the resolvers take, which is the only way to
/// express "a `DocInput` borrowing the `.doc` bytes next to it" without a
/// self-referential struct.
struct OpenedDeleteSegment {
    max_doc: usize,
    live_docs: Option<FixedBitSet>,
    /// The term dictionary and the `.doc` file's bytes -- the latter held as
    /// the `Input` the directory handed over rather than copied into a `Vec`
    /// (on an `MmapDirectory` that `Input` *is* the mapping, so a `to_vec()`
    /// would heap-copy the whole postings file on every buffered-delete
    /// round). Pooled across rounds ([`crate::reader_pool`]).
    postings: std::sync::Arc<crate::reader_pool::PooledPostings>,
    segment_id: [u8; ID_LENGTH],
    suffix: String,
}

impl IndexingConfig {
    /// `FrozenBufferedUpdates.apply(SegmentState[])` for one segment: resolve
    /// every applicable packet's term deletes, query deletes and doc-values
    /// updates against it, then write the results out as one `.liv` generation
    /// and one doc-values-update generation.
    ///
    /// Java resolves and writes per packet through a pooled
    /// `ReadersAndUpdates`; this port has no reader pool, so it opens the
    /// segment once, resolves *all* applicable packets against that one open,
    /// and writes once. Same outcome (deletes are a set union; doc-values
    /// updates are applied in ascending generation order so the newest still
    /// wins), one open and one generation bump instead of N.
    /// Returns whether this apply left the segment 100% hard-deleted, which is
    /// `closeSegmentStates`' `rld.isFullyDeleted()` -- see
    /// [`IndexWriter::drop_fully_deleted_segments`].
    pub(crate) fn apply_packets_to_segment(
        &self,
        dir: &dyn Directory,
        packets: &[FrozenBufferedUpdates],
        sci: &mut SegmentCommitInfo,
        pending_sort_map: Option<&(String, Vec<usize>)>,
    ) -> Result<bool> {
        let seg_gen = sci.buffered_deletes_gen;
        self.apply_packets_at_gen(dir, packets, sci, seg_gen, pending_sort_map)
    }

    /// `DocumentsWriterPerThread.flush`'s `applyDeletes` half: a segment just
    /// built, not yet published, has its own buffer's deletes resolved against
    /// it -- each limited to the documents added before it (in `sort_map`'s
    /// pre-sort order, when the flush sorted) -- by the thread that built it,
    /// with no writer lock held. Java does this during the flush too, from the
    /// in-RAM postings. The segment is then published with no private packet.
    /// Returns whether it left every document deleted.
    pub(crate) fn apply_flush_private_updates(
        &self,
        dir: &dyn Directory,
        private: &crate::buffered_updates::BufferedUpdates,
        sci: &mut SegmentCommitInfo,
        sort_map: Option<&(String, Vec<usize>)>,
    ) -> Result<bool> {
        // The packet and the segment share generation 0 for this one apply:
        // equal generations are what makes the per-entry limits apply.
        let packet = FrozenBufferedUpdates::new(private, Some(sci.segment_name.clone()))
            .with_flush_generation();
        self.apply_packets_at_gen(dir, std::slice::from_ref(&packet), sci, 0, sort_map)
    }

    fn apply_packets_at_gen(
        &self,
        dir: &dyn Directory,
        packets: &[FrozenBufferedUpdates],
        sci: &mut SegmentCommitInfo,
        seg_gen: i64,
        pending_sort_map: Option<&(String, Vec<usize>)>,
    ) -> Result<bool> {
        let applicable: Vec<&FrozenBufferedUpdates> = packets
            .iter()
            .filter(|p| p.applies_to(seg_gen))
            .filter(|p| {
                // A segment-private packet belongs to exactly one segment;
                // `applies_to` alone would also let it reach an older segment
                // that happens to share the generation boundary.
                p.private_segment
                    .as_deref()
                    .is_none_or(|name| name == sci.segment_name)
            })
            .collect();
        if applicable.is_empty() {
            // Not a candidate for the fully-deleted drop: Java's
            // `openSegmentStates` never opened a state for it.
            return Ok(false);
        }

        // `ReadersAndUpdates.sortMap` for this segment, if this writer's
        // flush just sorted it. Every `docIDUpto` limit below is a position
        // in the **pre-sort** buffer, so a doc id from the sorted segment has
        // to be mapped back through `newToOld` before it is compared -- Java
        // does the same in `FrozenBufferedUpdates`' two `sortMap.newToOld(doc)
        // < limit` branches.
        let sort_map: Option<&Vec<usize>> = pending_sort_map
            .filter(|(name, _)| name == &sci.segment_name)
            .map(|(_, map)| map);
        // "Is this doc id below the packet's pre-sort limit?" -- the one
        // predicate every filter below uses, so the mapping cannot be applied
        // in one place and forgotten in another.
        let owned = self.open_segment_for_deletes(dir, sci)?;
        let opened = owned.view()?;
        let max_doc = opened.max_doc;
        // A doc id the postings decoder produced out of range is the
        // `DocOutOfRange` corruption `deletes::mark_deleted` reports; checking
        // it here, before `below_limit` indexes the sort map or `deleted` sets
        // a bit, keeps it an error rather than an index panic.
        let in_range = |doc_id: i32| -> std::result::Result<usize, deletes::Error> {
            usize::try_from(doc_id)
                .ok()
                .filter(|idx| *idx < max_doc)
                .ok_or(deletes::Error::DocOutOfRange { doc_id, max_doc })
        };
        let below_limit = |doc: usize, limit: i32| -> bool {
            match &sort_map {
                None => i64::try_from(doc).is_ok_and(|d| d < i64::from(limit)),
                Some(map) => map
                    .get(doc)
                    .is_some_and(|&old| i64::try_from(old).is_ok_and(|o| o < i64::from(limit))),
            }
        };
        // `below_limit` indexes the map by doc id. The invariant is
        // maintained by construction (the map is the permutation of the
        // buffer that became this segment, and the name filter above picks
        // the segment it names), but it is maintained by three separate
        // assignments rather than by a type, so pin it here.
        debug_assert!(
            sort_map.as_ref().is_none_or(|m| m.len() == opened.max_doc),
            "the sort map must have one entry per document of the segment it names"
        );
        // A set, not a list: `applyDocValuesUpdates` has to ask whether a doc
        // is already dead, and the same doc can be named by several packets.
        // A bitset over the segment's docs: one bit test per lookup, and it
        // comes out in doc id order for `apply_deletes`.
        let mut deleted = FixedBitSet::new(max_doc);
        let mut any_deleted = false;
        // field number -> updates in ascending packet-generation order, so the
        // last write for a doc wins (`DocValuesFieldUpdates.mergedIterator`).
        let mut numeric_updates: Vec<PerFieldNumericUpdates> = Vec::new();
        let mut binary_updates: Vec<PerFieldBinaryUpdates> = Vec::new();

        for packet in &applicable {
            for (term, entry_limit) in &packet.delete_terms {
                let limit = packet.limit_for(seg_gen, *entry_limit);
                let docs = term_delete::resolve_term_doc_ids(
                    opened.fields,
                    opened.doc_in.as_ref(),
                    opened.live_docs,
                    &term.field,
                    &term.bytes,
                )?;
                for d in docs {
                    let d = in_range(d)?;
                    if below_limit(d, limit) {
                        // FBS: `deleted` is `FixedBitSet::new(max_doc)` and
                        // `in_range` returned `d < max_doc`.
                        deleted.set(d);
                        any_deleted = true;
                    }
                }
            }
            for (query, entry_limit) in &packet.delete_queries {
                let limit = packet.limit_for(seg_gen, *entry_limit);
                // A sorted segment scatters the documents below the limit
                // across the whole doc id range, so the scan cannot stop
                // early -- Java's own sorted branch likewise drops the
                // `docID < limit` loop bound and filters instead.
                let bound = if sort_map.is_some() {
                    opened.max_doc
                } else {
                    query_bound(limit)
                };
                let docs = IndexWriter::resolve_delete_query(query, &opened, bound)?;
                for d in docs {
                    let d = in_range(d)?;
                    if below_limit(d, limit) {
                        // FBS: `deleted` is `FixedBitSet::new(max_doc)` and
                        // `in_range` returned `d < max_doc`.
                        deleted.set(d);
                        any_deleted = true;
                    }
                }
            }
            for (field, buffer) in &packet.field_updates {
                let Some(info) = self.field_by_name(field) else {
                    // Java's `verifyOrCreateDvOnlyField` already rejected an
                    // unknown field at buffer time; reaching here means the
                    // update was buffered before this writer's field list was
                    // in play, and there is nothing to write it against.
                    continue;
                };
                for update in &buffer.updates {
                    let limit = packet.limit_for(seg_gen, update.doc_id_upto);
                    let docs = term_delete::resolve_term_doc_ids(
                        opened.fields,
                        opened.doc_in.as_ref(),
                        opened.live_docs,
                        &update.term.field,
                        &update.term.bytes,
                    )?;
                    // `applyDocValuesUpdates` reads
                    // `segState.rld.getLiveDocs()`, which by then already
                    // reflects *this packet's* term and query deletes (Java
                    // runs the three in that order inside one
                    // `FrozenBufferedUpdates.apply`). So a document this packet
                    // just killed takes no update. `resolve_term_doc_ids` has
                    // already filtered by the segment's pre-pass live docs;
                    // `deleted` is the rest of the answer.
                    let mut matched: Vec<i32> = Vec::with_capacity(docs.len());
                    for d in docs {
                        let idx = in_range(d)?;
                        // FBS: `deleted` is `FixedBitSet::new(max_doc)` and
                        // `in_range` returned `idx < max_doc`.
                        if below_limit(idx, limit) && !deleted.get(idx) {
                            matched.push(d);
                        }
                    }
                    if matched.is_empty() {
                        continue;
                    }
                    if buffer.is_numeric {
                        let value = match &update.value {
                            UpdateValue::Numeric(v) => Some(*v),
                            // `hasValue == false` -> `reset(doc)`.
                            _ => None,
                        };
                        let slot = entry_for(&mut numeric_updates, info.number);
                        slot.extend(matched.into_iter().map(|d| (d, value)));
                    } else {
                        let value = match &update.value {
                            UpdateValue::Binary(v) => Some(v.clone()),
                            _ => None,
                        };
                        let slot = entry_for(&mut binary_updates, info.number);
                        slot.extend(matched.into_iter().map(|d| (d, value.clone())));
                    }
                }
            }
        }

        if any_deleted {
            // Ascending doc id order, straight off the bitset; every set bit
            // is below `max_doc`, which fits an `i32` (it is a `.si` count).
            let deleted_ids = std::iter::successors(deleted.next_set_bit(0), |&d| {
                d.checked_add(1).and_then(|n| deleted.next_set_bit(n))
            })
            .filter_map(|d| i32::try_from(d).ok());
            let updated = deletes::apply_deletes_with_codec(
                dir,
                sci,
                opened.live_docs,
                opened.max_doc,
                deleted_ids,
                self.codec.as_deref(),
            );
            if updated.is_err() {
                // `PendingDeletes.writeLiveDocs`' failure path:
                // `info.advanceNextWriteDelGen()`, so a retry never asks for
                // the name a failed attempt may have left behind.
                let next = sci.next_write_del_gen();
                sci.set_next_write_del_gen(next.saturating_add(1));
            }
            *sci = updated?;
        }

        if !numeric_updates.is_empty() || !binary_updates.is_empty() {
            Self::write_doc_values_update_generation(
                dir,
                sci,
                &numeric_updates,
                &binary_updates,
                &self.fields,
                self.codec.as_deref(),
            )?;
        }
        // `PendingDeletes.isFullyDeleted`: `getDelCount() == info.info.maxDoc()`.
        Ok(i64::from(sci.del_count) == opened.max_doc as i64)
    }

    /// Writes one doc-values-update generation for `sci` --
    /// `ReadersAndUpdates.writeFieldUpdates`, delegated to
    /// [`crate::field_updates`], which owns the file naming, the
    /// `FieldInfos` generation and the `SegmentCommitInfo` bookkeeping.
    ///
    /// The bytes are **real Lucene's**: each updated field's whole column is
    /// rewritten through the `Lucene90DocValuesFormat` writer into a
    /// generation-suffixed `.dvm`/`.dvd`/`.dvs` triple, and a new `.fnm`
    /// generation records the field's `FieldInfo.docValuesGen`.
    fn write_doc_values_update_generation(
        dir: &dyn Directory,
        sci: &mut SegmentCommitInfo,
        numeric: &[PerFieldNumericUpdates],
        binary: &[PerFieldBinaryUpdates],
        schema: &[FieldInfo],
        codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
    ) -> Result<()> {
        crate::field_updates::write_field_updates_with_codec(
            dir,
            sci,
            numeric,
            binary,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            schema,
            codec,
        )?;
        Ok(())
    }

    /// Opens exactly what resolving a buffered delete against one segment
    /// needs: its `max_doc` (from the `.si`, not from a stored-fields open --
    /// the doc count is already recorded there), its current live docs, and
    /// its term dictionary + `.doc` postings when it has any.
    ///
    /// This is the slice of Java's `ReaderPool`/`ReadersAndUpdates` that
    /// buffered deletes actually use. A segment with no postings files opens
    /// with an empty [`lucene_codecs::blocktree::BlockTreeFields`], which
    /// resolves every term to zero documents -- the same outcome as Java's
    /// `TermDocsIterator.nextTerm` returning null, and the reason a delete
    /// against a stored-fields-only segment is a legitimate no-op rather than
    /// an error.
    fn open_segment_for_deletes(
        &self,
        dir: &dyn Directory,
        sci: &SegmentCommitInfo,
    ) -> Result<OpenedDeleteSegment> {
        let si = self.reader_pool.segment_info(dir, sci)?;
        let max_doc = si.doc_count as usize;

        let live_docs = if sci.del_gen >= 0 {
            let liv = dir.open(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))?;
            Some(lucene_codecs::live_docs::parse(
                &liv,
                &sci.segment_id,
                sci.del_gen,
                max_doc,
                sci.del_count as usize,
            )?)
        } else {
            None
        };

        let postings = self.reader_pool.get_or_open(sci, || {
            Self::open_postings_for_deletes(dir, sci, &si, max_doc)
        })?;
        let suffix = postings.suffix.clone();
        Ok(OpenedDeleteSegment {
            max_doc,
            live_docs,
            postings,
            segment_id: sci.segment_id,
            suffix,
        })
    }

    /// The part of [`Self::open_segment_for_deletes`] that does not change
    /// between delete rounds, and so is pooled: the segment's term dictionary
    /// and `.doc` input, from `.cfs` for a compound segment.
    fn open_postings_for_deletes(
        dir: &dyn Directory,
        sci: &SegmentCommitInfo,
        si: &segment_info::SegmentInfo,
        max_doc: usize,
    ) -> Result<crate::reader_pool::PooledPostings> {
        // A compound segment -- one real Lucene flushed -- keeps its term
        // dictionary inside `.cfs`, and its `.si` lists no `.tim` at all:
        // testing `si.files` alone would report "no postings" and every
        // delete against it would silently match nothing.
        let compound = if si.is_compound_file {
            Some(CompoundReader::open(
                dir,
                &sci.segment_name,
                &sci.segment_id,
            )?)
        } else {
            None
        };
        let seg_dir: &dyn Directory = match &compound {
            Some(c) => c,
            None => dir,
        };
        let seg_files: Vec<String> = match &compound {
            Some(c) => c.member_files(),
            None => si.files.clone(),
        };

        // The segment's own postings format: an older Lucene's segment names
        // its files (and frames their headers) after it, not after this
        // writer's.
        let (seg, suffix) = postings_file_base(&seg_files, &sci.segment_name);
        if !per_field_postings::has_postings_files(&seg_files) {
            return Ok(crate::reader_pool::PooledPostings {
                fields: lucene_codecs::blocktree::BlockTreeFields::empty(),
                doc_input: None,
                suffix,
                combined: false,
            });
        }

        // The segment's **own** field infos, every indexed field of them:
        // `blocktree::open` resolves the `.tmd`'s field numbers through this
        // list, and those numbers are the segment's -- a segment another
        // writer (real Lucene) produced numbers its fields its own way, which
        // this writer's schema need not match. Java's `SegmentReader` reads
        // the same `.fnm` for the same reason.
        let own = crate::field_updates::read_current_field_infos(seg_dir, sci, &seg_files)?;
        let field_infos = lucene_codecs::field_infos::FieldInfos {
            fields: own
                .fields
                .into_iter()
                .filter(|f| f.index_options != IndexOptions::None)
                .collect(),
        };
        // `PerFieldPostingsFormat` with more than one format: every group,
        // combined into one dictionary over one concatenated `.doc`.
        let suffixes = per_field_postings::group_suffixes(&seg_files, &sci.segment_name);
        if suffixes.len() > 1 || suffixes.iter().any(|s| is_foreign_postings_group(s)) {
            let mut opened = Vec::with_capacity(suffixes.len());
            for suffix in &suffixes {
                let seg = format!("{}_{suffix}", sci.segment_name);
                let open = |ext: &str| seg_dir.open(&format!("{seg}.{ext}"));
                let format = per_field_postings::format_of_suffix(suffix);
                let dictionary = per_field_postings::dictionary_extensions(format)
                    .iter()
                    .map(|ext| {
                        if !seg_files.contains(&format!("{seg}.{ext}"))
                            && !per_field_postings::dictionary_file_required(format, ext)
                        {
                            return Ok(lucene_store::directory::Input::Owned(Vec::new()));
                        }
                        open(ext)
                    })
                    .collect::<lucene_store::Result<Vec<_>>>()?;
                let doc = if seg_files.contains(&format!("{seg}.doc")) {
                    Some(open("doc")?)
                } else {
                    None
                };
                opened.push((suffix, dictionary, doc));
            }
            let groups: Vec<per_field_postings::GroupFiles<'_>> = opened
                .iter()
                .map(|(suffix, dictionary, doc)| per_field_postings::GroupFiles {
                    suffix,
                    dictionary: dictionary.iter().map(|d| &**d).collect(),
                    doc: doc.as_deref(),
                    pos: None,
                    pay: None,
                })
                .collect();
            let combined = per_field_postings::open_groups(
                &groups,
                &field_infos,
                &sci.segment_id,
                max_doc as i32,
            )?;
            return Ok(crate::reader_pool::PooledPostings {
                fields: combined.fields,
                doc_input: Some(lucene_store::directory::Input::Owned(combined.doc)),
                suffix,
                combined: true,
            });
        }
        // Borrowed from the directory's `Input` (a mapping, under
        // `MmapDirectory`), not copied: `blocktree::open` builds its own
        // structures from these and does not retain the slices, and the `.doc`
        // `Input` is kept alive in the returned struct for `DocInput` to
        // borrow.
        let tim = seg_dir.open(&format!("{seg}.tim"))?;
        let tip = seg_dir.open(&format!("{seg}.tip"))?;
        let tmd = seg_dir.open(&format!("{seg}.tmd"))?;
        let doc_input = seg_dir.open(&format!("{seg}.doc"))?;
        let fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &suffix,
            max_doc as i32,
        )?;
        Ok(crate::reader_pool::PooledPostings {
            fields,
            doc_input: Some(doc_input),
            suffix,
            combined: false,
        })
    }
}

impl OpenedDeleteSegment {
    fn view(&self) -> Result<DeleteSegmentView<'_>> {
        let doc_in = match &self.postings.doc_input {
            Some(bytes) if self.postings.combined => {
                Some(lucene_codecs::postings::DocInput::validated(bytes))
            }
            Some(bytes) => Some(lucene_codecs::postings::DocInput::open(
                bytes,
                &self.segment_id,
                &self.suffix,
            )?),
            None => None,
        };
        Ok(DeleteSegmentView {
            max_doc: self.max_doc,
            live_docs: self.live_docs.as_ref(),
            fields: &self.postings.fields,
            doc_in,
        })
    }
}

/// The borrowed view of an [`OpenedDeleteSegment`] the delete resolvers take.
struct DeleteSegmentView<'a> {
    max_doc: usize,
    live_docs: Option<&'a FixedBitSet>,
    fields: &'a lucene_codecs::blocktree::BlockTreeFields,
    doc_in: Option<lucene_codecs::postings::DocInput<'a>>,
}

/// One field's resolved NUMERIC doc-values updates for one generation:
/// `(field number, (doc id, value-or-reset) pairs)`.
type PerFieldNumericUpdates = (i32, Vec<(i32, Option<i64>)>);

/// One field's resolved BINARY doc-values updates for one generation.
type PerFieldBinaryUpdates = (i32, Vec<(i32, Option<Vec<u8>>)>);

/// What [`IndexWriter::resolve_term_span`]'s classifier decides for one term.
enum TermSpan {
    /// The term is in the query's span: take its documents.
    Take,
    /// Below the span; keep walking.
    Skip,
    /// Past the span; the dictionary is sorted, so nothing after it can match.
    Stop,
}

/// The `docIDUpto` limit as a doc-space bound.
/// [`crate::buffered_updates::MAX_DOC_ID_UPTO`] means "no limit", which is
/// `usize::MAX` here and is then clamped to the segment's own `max_doc` by
/// every caller.
fn query_bound(limit: i32) -> usize {
    if limit < 0 {
        0
    } else {
        limit as usize
    }
}

/// `Map.computeIfAbsent(fieldNumber, …)` over an association list -- a `Vec`
/// rather than a `HashMap` because the number of updated fields in one flush is
/// a handful, and the list keeps a deterministic write order for the resulting
/// files.
fn entry_for<V>(list: &mut Vec<(i32, Vec<V>)>, field_number: i32) -> &mut Vec<V> {
    if let Some(idx) = list.iter().position(|(n, _)| *n == field_number) {
        return &mut list[idx].1;
    }
    list.push((field_number, Vec::new()));
    &mut list.last_mut().expect("just pushed").1
}

fn empty_segment_infos(lucene_version: LuceneVersion) -> SegmentInfos {
    SegmentInfos {
        id: generate_segment_id(0),
        generation: 0,
        format_version: segment_infos::VERSION_86,
        lucene_version: to_segment_infos_version(lucene_version),
        index_created_version_major: lucene_version.major,
        version: 0,
        counter: 0,
        min_segment_lucene_version: None,
        segments: Vec::new(),
        user_data: Vec::new(),
    }
}

fn to_segment_infos_version(v: LuceneVersion) -> segment_infos::LuceneVersion {
    segment_infos::LuceneVersion {
        major: v.major,
        minor: v.minor,
        bugfix: v.bugfix,
    }
}

/// Same minimal `create_output`/`write_bytes`/`close` sequence
/// `crate::segment_writer`'s own private `write_file` helper uses -- kept as
/// a separate copy here rather than made `pub(crate)` there, since this is
/// the only other module that currently needs it and the function is a
/// three-line wrapper, not shared logic worth a cross-module dependency for.
/// The heap `doc` occupies while it sits in [`IndexWriter`]'s pending buffer:
/// its slot in the `Vec<Document>`, the field vector it owns, and every owned
/// `String`/`Vec<u8>` payload inside it.
///
/// A real byte count of the structure, not a sampled estimate -- the analogue of
/// what Java's `DocumentsWriterPerThread` feeds its `Counter bytesUsed`, over the
/// structure this port actually holds at `add_document` time (see
/// [`IndexWriter::ram_bytes_used`] for why those are different structures).
/// `capacity()`, not `len()`: an over-allocated `String` occupies its capacity.
/// ARITH: as [`VectorValue::ram_bytes`] -- a sum of live allocation sizes.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn document_ram_bytes(doc: &Document) -> usize {
    std::mem::size_of::<Document>()
        + doc.fields.capacity() * std::mem::size_of::<stored_fields::StoredField>()
        + doc
            .fields
            .iter()
            .map(|f| match &f.value {
                FieldValue::String(s) => s.capacity(),
                FieldValue::Shared(s) => s.len(),
                FieldValue::Binary(b) => b.capacity(),
                FieldValue::Int(_)
                | FieldValue::Long(_)
                | FieldValue::Float(_)
                | FieldValue::Double(_) => 0,
            })
            .sum::<usize>()
}

/// One document's field length so far plus one term's frequency -- Java's
/// `FieldInvertState.length`, which is what a norm encodes.
///
/// Java steps this with `Math.addExact(invertState.length, 1)` in
/// `IndexingChain.PerField.invert`: it *throws* rather than wrapping past
/// `Integer.MAX_VALUE`. A bare `+=` here wraps in a release build, and a
/// wrapped length encodes to a *small* norm -- the longest document in the
/// index would then score as one of the shortest, silently, in every BM25
/// query that reads the field. It also trips
/// [`lucene_util::small_float::int_to_byte4`]'s own `debug_assert` on the way past
/// `i32::MAX`.
///
/// Saturating at `i32::MAX` is the closest honest analogue of Java's throw:
/// `int_to_byte4` is a lossy 8-bit quantisation whose top bucket already
/// encodes everything near `Integer.MAX_VALUE` as `255`, so a saturated
/// length and Java's exact one produce the *same norm byte* for every input
/// Java accepts at all -- and the values past that are ones Java refuses to
/// index rather than ones it scores differently.
///
/// A negative `term_freq` can only come from an occurrence count above
/// `i32::MAX` (`InvertedEntry::term_freq` is `occurrences.len() as i32`), so
/// it means "longer than the longest length there is", not "shorter than
/// none".
fn accumulate_field_length(so_far: u32, term_freq: i32) -> u32 {
    let freq = if term_freq < 0 {
        u32::MAX
    } else {
        term_freq as u32
    };
    so_far.saturating_add(freq).min(i32::MAX as u32)
}

/// A document's custom-freq terms with each term once, in first-seen order:
/// repeated terms sum their freqs (`Math.addExact`), and a freq below 1 is
/// refused. Checked before a document is buffered, so the flush (which sums
/// the same way) cannot fail on them.
pub(crate) fn summed_custom_freq_terms(terms: &[(String, i32)]) -> Result<Vec<(&str, i32)>> {
    let illegal = |m: String| Error::Document(crate::document::Error::IllegalArgument(m));
    let mut index: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut summed: Vec<(&str, i32)> = Vec::with_capacity(terms.len());
    for (term, freq) in terms {
        let freq = *freq;
        if freq < 1 {
            return Err(illegal(format!(
                "Term frequency must be 1 or greater; got {freq} for term \"{term}\""
            )));
        }
        match index.get(term.as_str()).and_then(|&i| summed.get_mut(i)) {
            Some((_, slot)) => {
                *slot = slot.checked_add(freq).ok_or_else(|| {
                    illegal(format!(
                        "custom term frequency of \"{term}\" overflows an int"
                    ))
                })?;
            }
            None => {
                index.insert(term, summed.len());
                summed.push((term, freq));
            }
        }
    }
    Ok(summed)
}

/// The buffered-side heap cost of one document's explicit
/// [`IndexWriter::add_document_with_custom_freq_terms`] term list.
/// ARITH: as [`VectorValue::ram_bytes`] -- a sum of live allocation sizes.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn custom_freq_terms_ram_bytes(terms: &[(String, i32)]) -> usize {
    std::mem::size_of::<Vec<(String, i32)>>()
        + std::mem::size_of_val(terms)
        + terms.iter().map(|(t, _)| t.capacity()).sum::<usize>()
}

fn write_file(dir: &dyn Directory, name: &str, bytes: &[u8]) -> Result<()> {
    let mut out = dir.create_output(name)?;
    out.write_bytes(bytes);
    out.close()?;
    Ok(())
}

/// The `&'static str` kind name [`Error::NonNumericDocValue`] reports for a
/// [`FieldValue`] that isn't [`FieldValue::Int`]/[`FieldValue::Long`].
fn field_value_kind(value: &FieldValue) -> &'static str {
    match value {
        FieldValue::String(_) | FieldValue::Shared(_) => "String",
        FieldValue::Binary(_) => "Binary",
        FieldValue::Int(_) => "Int",
        FieldValue::Long(_) => "Long",
        FieldValue::Float(_) => "Float",
        FieldValue::Double(_) => "Double",
    }
}

/// Generates a 16-byte segment/commit id, the role of Java's
/// `StringHelper.randomId()`. This workspace has no `rand`-family
/// dependency, so the bytes come from SipHash under per-process random keys
/// (`RandomState`) over the time, `salt` (a writer counter) and a
/// process-wide sequence number. The sequence number makes every id this
/// process draws distinct -- two ids drawn with the same salt in the same
/// clock tick included -- and the random keys keep ids from separate
/// processes (and so separate indexes) from coinciding, which a fixed-key
/// hash of a small counter and the time did not.
pub(crate) fn generate_segment_id(salt: i64) -> [u8; ID_LENGTH] {
    use std::hash::{BuildHasher, RandomState};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    use std::time::{SystemTime, UNIX_EPOCH};

    static KEYS: OnceLock<RandomState> = OnceLock::new();
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let keys = KEYS.get_or_init(RandomState::new);
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let mut id = [0u8; ID_LENGTH];
    id[0..8].copy_from_slice(&keys.hash_one((sequence, nanos, salt, 1u8)).to_le_bytes());
    id[8..16].copy_from_slice(&keys.hash_one((sequence, nanos, salt, 2u8)).to_le_bytes());
    id
}

/// Whether two segment lists would be written identically into a
/// `segments_N`: the same segments, in the same order, each with the same
/// persisted per-commit state. The in-memory-only bookkeeping
/// ([`SegmentCommitInfo::buffered_deletes_gen`], the `next_write_*_gen`
/// counters) is not part of a commit and is ignored.
fn same_committed_segments(a: &[SegmentCommitInfo], b: &[SegmentCommitInfo]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.segment_name == y.segment_name
                && x.segment_id == y.segment_id
                && x.codec_name == y.codec_name
                && x.del_gen == y.del_gen
                && x.del_count == y.del_count
                && x.field_infos_gen == y.field_infos_gen
                && x.doc_values_gen == y.doc_values_gen
                && x.soft_del_count == y.soft_del_count
                && x.sci_id == y.sci_id
                && x.field_infos_files == y.field_infos_files
                && x.dv_update_files == y.dv_update_files
        })
}

/// `SegmentInfos.applyMergeChanges(merge, dropSegment)`: the sources go
/// either way; the merged segment (`None` standing for `dropSegment`) takes
/// the place of the first source, so a policy that reads segment order
/// (`LogMergePolicy`) sees the layout Java's writer would. When no source is
/// present any more it goes first, as in Java.
fn apply_merge_changes(
    segments: &mut Vec<SegmentCommitInfo>,
    source_segment_names: &[&str],
    merged: Option<SegmentCommitInfo>,
) {
    let mut merged = merged;
    let mut out = Vec::with_capacity(segments.len());
    for info in segments.drain(..) {
        if source_segment_names.contains(&info.segment_name.as_str()) {
            if let Some(m) = merged.take() {
                out.push(m);
            }
        } else {
            out.push(info);
        }
    }
    if let Some(m) = merged {
        out.insert(0, m);
    }
    *segments = out;
}

/// `OneMerge.estimatedMergeBytes`'s share for one source: its size on disk
/// pro-rated by its live documents.
pub(crate) fn live_pro_rated_bytes(size_bytes: u64, max_doc: i32, del_count: i32) -> u64 {
    if max_doc <= 0 {
        return 0;
    }
    let live = f64::from(max_doc.saturating_sub(del_count).max(0));
    (size_bytes as f64 * (live / f64::from(max_doc))) as u64
}

/// The forced merge `IndexWriter.forceMerge` runs without a merge policy
/// over `stats`, while they number more than `max_num_segments`: down to one,
/// every segment in one merge, largest first (`findForcedMerges`' merge-down
/// case, no size cap); else the smallest segments, enough to reach the count.
fn fallback_forced_merge(
    stats: &[merge_policy::SegmentStat],
    max_num_segments: usize,
) -> Option<Vec<String>> {
    if stats.len() <= max_num_segments {
        return None;
    }
    if max_num_segments == 1 {
        let config = merge_policy::MergePolicyConfig::default();
        return Some(
            merge_policy::find_forced_merges(stats, 1, &config)
                .into_iter()
                .flatten()
                .collect(),
        );
    }
    let mut stats = stats.to_vec();
    stats.sort_by_key(|s| s.size_bytes);
    // ARITH: `stats.len() > max_num_segments >= 1` here, so the difference
    // is at least 1 and the `+ 1` at most `stats.len()`.
    #[allow(clippy::arithmetic_side_effects)]
    let take = stats.len() - max_num_segments + 1;
    Some(stats.into_iter().take(take).map(|s| s.name).collect())
}

/// `ShuffleForcedMergePolicy.interleaveList`: the sources sorted by name
/// (Java's `String` order, older first), then taken from both ends at once,
/// oldest with newest, the smaller of each pair (by `sizeInBytes`) first.
fn shuffle_interleave(mut names: Vec<String>, stats: &[merge_policy::SegmentStat]) -> Vec<String> {
    let bytes_of = |name: &str| {
        stats
            .iter()
            .find(|s| s.name == name)
            .map_or(0, |s| s.size_bytes)
    };
    // Java's `String.compareTo` is UTF-16 order; segment names are ASCII,
    // where it is byte order.
    names.sort();
    let mut out = Vec::with_capacity(names.len());
    let (mut left, mut right) = (0usize, names.len());
    while left < right {
        // ARITH: `right > left >= 0`, so `right - 1` does not underflow.
        #[allow(clippy::arithmetic_side_effects)]
        let last = right - 1;
        if left == last {
            out.push(names[left].clone());
        } else if bytes_of(&names[left]) < bytes_of(&names[last]) {
            out.push(names[left].clone());
            out.push(names[last].clone());
        } else {
            out.push(names[last].clone());
            out.push(names[left].clone());
        }
        // ARITH: `left < right <= names.len()`, so neither overflows.
        #[allow(clippy::arithmetic_side_effects)]
        {
            left += 1;
            right = last;
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests;

/// Whether `auto`'s sample reads `config`'s field: not the engine's own
/// fields, named with a leading underscore (OpenSearch's `_seq_no` rises with
/// arrival in every index, and says nothing of the documents' own columns).
fn sampled_field(config: &DocValuesFieldConfig) -> bool {
    !config.name.starts_with('_')
}
