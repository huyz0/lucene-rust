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
//! [`MergePolicyConfig`] itself exposes, no concurrent/background merging, no
//! merge-scheduling across many tiers beyond what [`crate::merge_policy`]
//! itself already does in one [`crate::merge_policy::find_merges`] call, and
//! [`IndexWriter::update_document`]/[`IndexWriter::delete_documents_by_term`] do not
//! trigger this check (only [`IndexWriter::commit`] does, matching where
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
use crate::term_delete;
use crate::update_document::{self, SegmentDeleteSource};

use lucene_analysis::Analyzer;
use lucene_codecs::doc_values;
use lucene_codecs::field_infos::{
    DocValuesType, FieldInfo, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::hnsw;
use lucene_codecs::hnsw_vectors::{self, HnswVectorsField};
use lucene_codecs::norms;
use lucene_codecs::postings_writer::{self, FieldPostingsInput, TermPostings};
use lucene_codecs::stored_fields::{self, Document, FieldValue};
use lucene_codecs::term_vectors::{self, TermVectorField, TermVectorTerm, TermVectorsDocument};
use lucene_codecs::vectors::{self, FieldVectorData, FlatVectorsField};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::small_float;

pub use crate::merge_policy::MergePolicyConfig;
pub use crate::update_document::SegmentDeleteSource as DeleteSource;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
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
    /// A merged segment's `.si` names an index-sort field this writer's own
    /// field list does not have -- the index was written by a differently
    /// configured writer.
    #[error("index sort field {0:?} is not in this writer's field list")]
    UnknownSortField(String),
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
    #[error("set_vector_field: no field named {0:?} in this writer's field list")]
    UnknownVectorField(String),
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
        "update_doc_values: field {field:?} is declared with doc_values_type {declared}; only a          NUMERIC field accepts a numeric update and only a BINARY field a binary one"
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
    /// (`"SortField <..> expected field [x] to be NUMERIC but it is [BINARY]"`).
    /// This port's `.si` encoder emits a single-valued `LONG` sort
    /// (`segment_info::write_sort_field`), which real Lucene resolves through
    /// `DocValues.getNumeric`, so the field must be NUMERIC.
    #[error(
        "set_index_sort: sort field {0:?} is declared with doc_values_type {1:?}; an index sort \
         field must be NUMERIC (this port's .si encodes a single-valued LONG sort, which real \
         Lucene reads through DocValues.getNumeric)"
    )]
    UnsupportedIndexSortField(String, DocValuesType),
    #[error(
        "set_index_sort: sort field {0:?} sorts by term ordinal or by raw bytes; this writer \
         assigns ordinals after it permutes the buffer, so the key would not exist when the \
         sort runs, and a BinarySortField has no single-i64 key at all. Such a sort can be \
         read (segment_info parses every SortFieldProvider encoding) but not produced"
    )]
    UnsupportedIndexSortKind(String),
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
    /// document, so the sort can be applied to whole blocks. This port has
    /// no parent-field write path, so the combination is refused rather than
    /// silently producing a segment whose blocks are shredded.
    #[error(
        "flush: this segment carries document blocks (add_documents/update_documents) and an \
         index sort is configured, but no parent field is set -- real Lucene raises \
         CorruptIndexException for exactly this combination"
    )]
    IndexSortWithBlocksAndNoParentField,
    /// `IndexWriter.updateDocValues`'s `IllegalArgumentException("cannot
    /// update docvalues field involved in the index sort")`. Rewriting the
    /// column the segment's physical order is defined over would leave the
    /// segment claiming a sort it no longer has.
    #[error(
        "update_doc_values: cannot update doc-values field {field:?} -- it is part of this \
         writer's index sort ({sort})"
    )]
    DocValuesUpdateOnIndexSortField { field: String, sort: String },
    /// This writer writes every doc-values field of one flush into a single
    /// `.dvm`/`.dvd`/`.dvs` triple through
    /// [`doc_values::write_dense_fields`], which is dense-only. With exactly
    /// one field the sparse writer is used instead, so this only bites a
    /// multi-field configuration.
    #[error(
        "flush: doc-values field {field:?} has no value on {missing} of {max_doc} documents; \
         with more than one doc-values field configured only a NUMERIC field may be sparse \
         (this writer batches them into one .dvm/.dvd, and doc_values::write_dense_fields is \
         dense-only apart from NUMERIC)"
    )]
    SparseFieldInMultiFieldDocValues {
        field: String,
        missing: usize,
        max_doc: usize,
    },
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
enum DeleteNode {
    Terms(Vec<Term>),
    Queries(Vec<DeleteQuery>),
    DocValuesUpdates(Vec<DocValuesUpdate>),
}

/// `IndexWriter.buildDocValuesUpdate(term, updates)`: every supplied field
/// becomes an update keyed by the caller's `term`, whatever term the caller
/// happened to put in the [`DocValuesUpdate`] they built.
fn retarget_update(update: &DocValuesUpdate, term: &Term) -> DocValuesUpdate {
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

/// A single, coherent entry point over this port's write-side primitives.
/// See the module doc comment for the exact lifecycle and scope.
pub struct IndexWriter<'d> {
    dir: &'d dyn Directory,
    fields: Vec<FieldInfo>,
    codec_name: String,
    lucene_version: LuceneVersion,
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
    payload_source: Option<BoxedPayloadSource>,
    /// The single field this writer is currently opted into building real
    /// `IndexOptions::DocsAndCustomFreqs` postings for, driven by explicit
    /// per-document caller-supplied `(term, custom_freq)` pairs (see
    /// [`Self::pending_custom_freq_terms`]) rather than analyzed text -- see
    /// [`IndexWriter::set_custom_freq_postings_field`]. **Mutually exclusive**
    /// with [`Self::postings_fields`] (see that method's doc comment for why):
    /// at most one of the two is ever non-empty/`Some` at a time.
    custom_freq_postings_field: Option<CustomFreqPostingsFieldConfig>,
    /// Per-pending-doc explicit `(term, custom_freq)` pairs for
    /// [`Self::custom_freq_postings_field`], aligned 1:1 by index with
    /// `pending_docs` (index `i` here is doc ID `i` in the next flush, same
    /// convention [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment) already uses for
    /// `pending_docs` itself) -- kept in lockstep by
    /// [`IndexWriter::add_document`] (pushes an empty `Vec`) and
    /// [`IndexWriter::add_document_with_custom_freq_terms`] (pushes the
    /// caller's terms), and cleared together with `pending_docs` by
    /// `commit`/`prepare_commit`/`rollback`.
    pending_custom_freq_terms: Vec<Vec<(String, i32)>>,
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
    /// write path (see [`Self::build_doc_values_output`]).
    doc_values_fields: Vec<DocValuesFieldConfig>,
    /// `IndexWriterConfig.getIndexSort()`: the (possibly multi-field)
    /// priority-ordered key every segment this writer flushes is physically
    /// ordered by, and that its `.si` records. `None` -- the default -- means
    /// unsorted, which is what `SegmentInfo`'s `numSortFields == 0` says on
    /// disk. See [`IndexWriter::set_index_sort`].
    index_sort: Option<Vec<segment_info::IndexSortField>>,
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
    pending_sort_map: Option<(String, Vec<usize>)>,
    /// Set by [`IndexWriter::prepare_commit`], consumed by
    /// [`IndexWriter::finish_commit`] -- see [`IndexWriter::prepare_commit`]'s
    /// doc comment for exactly what "prepared" does and does not mean on
    /// this port's on-disk format (in short: the *segment* files it flushes
    /// are already durable and synced when this is `Some`, but the
    /// `segments_N` commit file that would make them discoverable to a
    /// fresh [`IndexWriter::open`]/reader has deliberately not been written
    /// yet).
    prepared_commit: Option<SegmentInfos>,
    /// `DocumentsWriterDeleteQueue`: the sequence-number source, and the two
    /// buffers every delete/update lands in -- one for the segment currently
    /// being built, one for the segments already written. See
    /// [`crate::buffered_updates`].
    delete_queue: DeleteQueue,
    /// `BufferedUpdatesStream`: frozen delete packets waiting to be resolved
    /// against segments, each stamped with the generation that decides which
    /// segments it may touch.
    updates_stream: BufferedUpdatesStream,
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
    /// Every field this writer is currently opted into indexing **vectors**
    /// for, in insertion order -- real Lucene's per-field
    /// `KnnFieldVectorsWriter`, which `IndexingChain` creates the first time a
    /// document carries a `KnnFloatVectorField`/`KnnByteVectorField` for that
    /// field. Same "resolve once against the fixed `fields` list, reuse every
    /// flush" shape as [`Self::postings_fields`], and like it a *list*: all of
    /// them are written into one `.vec`/`.vemf`/`.vem`/`.vex` quadruple per
    /// segment, exactly as `Lucene99HnswVectorsFormat` does.
    vector_fields: Vec<VectorFieldConfig>,
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
    /// `Lucene99HnswVectorsFormat(maxConn, beamWidth)`: the graph parameters
    /// every vector field flushed from here is built with. Defaults are
    /// Lucene's own ([`hnsw::DEFAULT_MAX_CONN`] / [`hnsw::DEFAULT_BEAM_WIDTH`]).
    hnsw_m: i32,
    hnsw_beam_width: i32,
    /// `SegmentInfo.setHasBlocks()` for the segment currently being buffered:
    /// set by any [`IndexWriter::add_documents`]/
    /// [`IndexWriter::update_documents`] call that buffers more than one
    /// document, cleared with the buffer at flush.
    pending_has_blocks: bool,
}

/// One field this writer has been opted into also indexing real postings
/// An owned token-payload supplier, as [`IndexWriter::set_payload_source`]
/// takes it and [`IndexWriter`] stores it -- this port's stand-in for a
/// `PayloadAttribute`-setting `TokenFilter`, see
/// [`crate::indexing_chain::PayloadSource`] for the borrowed form the invert
/// pass consumes.
/// `Send + Sync` is not decoration. `lucene-ffi`'s `WriterHandle` carries an
/// `unsafe impl Send`/`Sync` whose whole justification is that `IndexWriter`
/// "is a plain aggregate ... with no interior mutability at all"
/// (`lucene-ffi/src/registry.rs`), which a caller-supplied closure could
/// otherwise falsify by capturing an `Rc` or a `Cell` -- in a crate where
/// `forbid(unsafe_code)` would not stop it. The bound keeps that argument true
/// by construction, and costs nothing: every source here captures `Copy` data
/// or nothing.
pub type BoxedPayloadSource = Box<dyn Fn(&PayloadContext<'_>) -> Option<Vec<u8>> + Send + Sync>;

/// The borrowed view of a [`BoxedPayloadSource`], which is what the invert
/// pass is handed.
type PayloadSourceRef<'a> = &'a (dyn Fn(&PayloadContext<'_>) -> Option<Vec<u8>> + Send + Sync);

/// [`IndexWriter::build_norms_output`]'s result: the `.nvm`/`.nvd` bytes,
/// plus each norms field's per-document column as `(field number, norms by
/// doc id)` -- the input the postings writer's impacts need. See that
/// method's doc comment for why the columns come back at all.
type NormsOutput = (Vec<u8>, Vec<u8>, Vec<(i32, Vec<i64>)>);

/// for, resolved once by [`IndexWriter::set_postings_field`]/
/// [`IndexWriter::add_postings_field`] against this writer's fixed `fields`
/// list -- `self.postings_fields` may hold any number of these at once (see
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
/// `fields` list -- `self.term_vector_fields` may hold any number of these at
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
/// checksummed, `CheckIndex`-clean, wrong. Until c26 the containment
/// was `segment_stats` withholding any segment with a doc-values
/// generation from the merge policy entirely.
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
/// [`IndexWriter::norms_field_configs`] from this writer's fixed `fields`
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

/// [`IndexWriter::build_vectors_output`]'s four files, plus the names of the
/// fields that actually got vectors in this flush (which is what
/// [`IndexWriter::fields_with_per_field_attributes`] stamps, and what the rest
/// must have their `.fnm` `vector_dimension` zeroed for).
struct VectorsOutput {
    vec: Vec<u8>,
    vemf: Vec<u8>,
    vex: Vec<u8>,
    vem: Vec<u8>,
    written_fields: Vec<String>,
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
    Float32(Vec<f32>),
    /// `KnnByteVectorField`: `dimension` signed bytes, held unsigned exactly
    /// as `.vec` stores them (`b as i8` is Java's view of the same byte).
    Byte(Vec<u8>),
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
            value: VectorValue::Float32(vector),
        }
    }

    /// `new KnnByteVectorField(name, vector)`.
    pub fn byte(field_name: impl Into<String>, vector: Vec<u8>) -> Self {
        DocumentVector {
            field_name: field_name.into(),
            value: VectorValue::Byte(vector),
        }
    }

    /// ARITH: as [`VectorValue::ram_bytes`] -- a sum of live allocation
    /// sizes, bounded by the address space.
    #[allow(clippy::arithmetic_side_effects)]
    fn ram_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.field_name.capacity() + self.value.ram_bytes()
    }
}

/// Real Lucene routes postings and doc values through
/// `PerFieldPostingsFormat`/`PerFieldDocValuesFormat`, which do two things this
/// port previously did neither of: they segregate each format's files under a
/// suffixed segment name, `<segment>_<format>_<suffix>.<ext>`, and they record
/// the format and suffix in each field's `.fnm` attributes so a reader can find
/// them again. Without both, real Lucene opens the segment, reports zero terms
/// and zero doc-values fields, and raises no error -- the fields simply have no
/// format registered against them.
///
/// This port's own reader resolves codec files by extension suffix, so it reads
/// either naming, which is exactly why the divergence stayed invisible.
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

pub const POSTINGS_FORMAT_NAME: &str = "Lucene104";
pub const DOC_VALUES_FORMAT_NAME: &str = "Lucene90";
/// `PerFieldKnnVectorsFormat`'s per-field format name for
/// `Lucene99HnswVectorsFormat` -- the value that goes in the `.fnm`'s
/// `PerFieldKnnVectorsFormat.format` attribute and, joined with
/// [`PER_FIELD_SUFFIX`], into every `.vec`/`.vemf`/`.vem`/`.vex` file name.
pub const KNN_VECTORS_FORMAT_NAME: &str = "Lucene99HnswVectorsFormat";
/// This writer routes every field to one format, so it only ever needs Lucene's
/// first suffix. This is the value that goes in the `.fnm` attribute; the
/// suffix each *file* carries is the wider [`per_field_codec_suffix`].
pub const PER_FIELD_SUFFIX: &str = "0";

/// `PerFieldPostingsFormat.getSuffix`: the segment suffix a per-field format's
/// files are actually written with -- the format name and the suffix joined,
/// not the suffix alone. It appears both in the file name and, because Lucene
/// passes it straight through as `SegmentWriteState.segmentSuffix`, inside
/// every one of those files' index headers.
pub fn per_field_codec_suffix(format: &str) -> String {
    format!("{format}_{PER_FIELD_SUFFIX}")
}

/// The suffixed segment name a per-field format's files are written under.
pub fn per_field_segment(segment_name: &str, format: &str) -> String {
    format!("{segment_name}_{}", per_field_codec_suffix(format))
}

impl<'d> IndexWriter<'d> {
    /// Opens a writer over `dir`: resumes the latest existing commit if one
    /// is present, or starts a brand-new, empty index otherwise. `fields`
    /// describes every field [`IndexWriter::add_document`]/
    /// [`IndexWriter::update_document`] documents may use (this facade has
    /// no per-call schema reconciliation the way [`crate::merge`] does
    /// across sources -- every document flushed through one writer shares
    /// one fixed field list, same as every existing caller of
    /// [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment)). `codec_name`/`lucene_version` are
    /// recorded on every segment this writer flushes, same meaning as the
    /// identically-named parameters on [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment).
    pub fn open(
        dir: &'d dyn Directory,
        fields: Vec<FieldInfo>,
        codec_name: impl Into<String>,
        lucene_version: LuceneVersion,
    ) -> Result<Self> {
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

        // Real `IndexWriter`'s constructor builds its `IndexFileDeleter` here,
        // and that construction is what reclaims whatever a previous session
        // leaked: a `pending_segments_N` from a prepare that never finished, the
        // segment files of a flush that was never committed, and every commit
        // generation the deletion policy no longer wants.
        let deleter =
            IndexFileDeleter::open(dir, &segment_infos, DeletionPolicy::KeepOnlyLastCommit)?;

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

        Ok(IndexWriter {
            dir,
            fields,
            codec_name: codec_name.into(),
            lucene_version,
            segment_infos,
            pending_docs: Vec::new(),
            flushed_segments: Vec::new(),
            deleter,
            ram_buffer_size_mb: DEFAULT_RAM_BUFFER_SIZE_MB,
            max_buffered_docs: DEFAULT_MAX_BUFFERED_DOCS,
            ram_bytes_used: 0,
            merge_policy: None,
            postings_fields: Vec::new(),
            position_increment_gap: 0,
            offset_gap: 1,
            payload_source: None,
            custom_freq_postings_field: None,
            pending_custom_freq_terms: Vec::new(),
            term_vector_fields: Vec::new(),
            doc_values_fields: Vec::new(),
            index_sort: None,
            pending_sort_map: None,
            prepared_commit: None,
            delete_queue: DeleteQueue::new(),
            updates_stream: BufferedUpdatesStream::new(),
            rollback_segments,
            vector_fields: Vec::new(),
            pending_vectors: Vec::new(),
            hnsw_m: hnsw::DEFAULT_MAX_CONN,
            hnsw_beam_width: hnsw::DEFAULT_BEAM_WIDTH,
            pending_has_blocks: false,
        })
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
        if field_name.is_some() && self.custom_freq_postings_field.is_some() {
            return Err(Error::PostingsAndCustomFreqPostingsMutuallyExclusive(
                "set_postings_field",
            ));
        }
        self.postings_fields = match field_name {
            None => Vec::new(),
            Some(name) => vec![Self::resolve_postings_field(&self.fields, name)?],
        };
        Ok(())
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
        self.position_increment_gap = gap;
    }

    /// `Analyzer.getOffsetGap(String)`: the character offsets inserted between
    /// two values of the same multi-valued field. Java's default is **`1`**,
    /// which is this writer's default too.
    pub fn set_offset_gap(&mut self, gap: i32) {
        self.offset_gap = gap;
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
        if self.custom_freq_postings_field.is_some() {
            return Err(Error::PostingsAndCustomFreqPostingsMutuallyExclusive(
                "add_postings_field",
            ));
        }
        let config = Self::resolve_postings_field(&self.fields, field_name)?;
        if self
            .postings_fields
            .iter()
            .any(|f| f.field_number == config.field_number)
        {
            return Err(Error::DuplicatePostingsField(field_name.to_string()));
        }
        self.postings_fields.push(config);
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
        if source.is_some() && !self.fields.iter().any(|f| f.store_payloads) {
            return Err(Error::NoPayloadFields);
        }
        self.payload_source = source;
        Ok(())
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
        // payloads without positions" used to be re-checked here. It is not
        // any more: `IndexWriter::open` now puts the whole field list through
        // `FieldInfos::new` (Java's constructor), so a `FieldInfo` this writer
        // holds has already been through the check that makes the combination
        // unrepresentable -- and a guard no input can trip is a guard nothing
        // tests.
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
        if field_name.is_some() && !self.postings_fields.is_empty() {
            return Err(Error::PostingsAndCustomFreqPostingsMutuallyExclusive(
                "set_custom_freq_postings_field",
            ));
        }
        self.custom_freq_postings_field = match field_name {
            None => None,
            Some(name) => {
                let info = self
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
        self.term_vector_fields = match field_name {
            None => Vec::new(),
            Some(name) => vec![Self::resolve_term_vector_field(&self.fields, name)?],
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
        let config = Self::resolve_term_vector_field(&self.fields, field_name)?;
        if self
            .term_vector_fields
            .iter()
            .any(|f| f.field_number == config.field_number)
        {
            return Err(Error::DuplicateTermVectorField(field_name.to_string()));
        }
        self.term_vector_fields.push(config);
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
    /// for NUMERIC ([`doc_values::DenseField::SparseNumeric`]) -- so a
    /// NUMERIC field may still be sparse (which is what lets an index-sort
    /// tier have missing values) and any other type must be dense
    /// ([`Error::SparseFieldInMultiFieldDocValues`]). A doc whose
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
        if let Some(sort) = &self.index_sort {
            if let Some(stranded) = sort.iter().find(|sf| Some(sf.field.as_str()) != field_name) {
                return Err(Error::IndexSortFieldWithoutDocValues(
                    stranded.field.clone(),
                ));
            }
        }
        self.doc_values_fields.clear();
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
        let info = self
            .fields
            .iter()
            .find(|f| f.name == field_name)
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
        if self.doc_values_fields.iter().any(|c| c.name == field_name) {
            return Err(Error::DuplicateDocValuesField(field_name.to_string()));
        }
        self.doc_values_fields.push(DocValuesFieldConfig {
            name: field_name.to_string(),
            field_number: info.number,
            doc_values_type: info.doc_values_type,
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
    ///   SORTED_NUMERIC.
    /// - **The kind of sort is one this writer can produce**
    ///   ([`Error::UnsupportedIndexSortKind`]). `segment_info` can now *read*
    ///   every sort `SortFieldProvider` round-trips, which is what lets this
    ///   port open an index someone else wrote; producing one is narrower.
    ///   A `SortField.Type.STRING` or `SortedSetSortField` sorts by **term
    ///   ordinal**, and this writer assigns ordinals inside
    ///   `build_sorted_doc_values_output` *after* the buffer is permuted, so
    ///   the key the sort needs does not exist when the sort runs; a
    ///   `BinarySortField` has no single-`i64` key at all
    ///   ([`segment_info::IndexSortField::key_comparison`]). Both are
    ///   refused here rather than mis-ordered.
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
            self.index_sort = None;
            return Ok(());
        };
        if sort.is_empty() {
            return Err(Error::EmptyIndexSort);
        }
        for sf in sort {
            let info = self
                .fields
                .iter()
                .find(|f| f.name == sf.field)
                .ok_or_else(|| Error::UnknownIndexSortField(sf.field.clone()))?;
            let wanted = match &sf.kind {
                segment_info::IndexSortKind::Numeric(_) => DocValuesType::Numeric,
                segment_info::IndexSortKind::SortedNumeric { .. } => DocValuesType::SortedNumeric,
                segment_info::IndexSortKind::String(_)
                | segment_info::IndexSortKind::SortedSet { .. }
                | segment_info::IndexSortKind::Binary(_) => {
                    return Err(Error::UnsupportedIndexSortKind(sf.field.clone()))
                }
            };
            if info.doc_values_type != wanted {
                return Err(Error::UnsupportedIndexSortField(
                    sf.field.clone(),
                    info.doc_values_type,
                ));
            }
            if !self.doc_values_fields.iter().any(|c| c.name == sf.field) {
                return Err(Error::IndexSortFieldWithoutDocValues(sf.field.clone()));
            }
        }
        self.validate_index_sort_against_existing_segments(sort)?;
        self.index_sort = Some(sort.to_vec());
        Ok(())
    }

    /// The sort this writer is configured with, `None` when unsorted.
    pub fn index_sort(&self) -> Option<&[segment_info::IndexSortField]> {
        self.index_sort.as_deref()
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
            let si = segment_info::parse(&si_bytes, &sci.segment_id)?;
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
    /// -- so that is what this returns. Until c35 this port required a
    /// `set_norms_field`/`add_norms_field` call per field and forced
    /// `omit_norms = true` into the `.fnm` for every other indexed field,
    /// which made BM25 score every un-named field against a constant length
    /// instead of the document's own. [`IndexWriter::omit_norms_field`] is
    /// the opt-*out* (`FieldType.setOmitNorms(true)`); there is no opt-in
    /// because Lucene has none.
    ///
    /// Ascending field number, because that is the order
    /// [`lucene_codecs::norms::write_fields`] wants its meta entries in and
    /// it makes the `.nvm` a function of the schema rather than of the
    /// order the caller declared the fields.
    fn norms_field_configs(&self) -> Vec<NormsFieldConfig> {
        let mut configs: Vec<NormsFieldConfig> = self
            .fields
            .iter()
            .filter(|f| !f.omit_norms && !matches!(f.index_options, IndexOptions::None))
            .map(|f| NormsFieldConfig {
                name: f.name.clone(),
                field_number: f.number,
            })
            .collect();
        configs.sort_by_key(|c| c.field_number);
        configs
    }

    /// `FieldType.setOmitNorms(true)` for one field of this writer's fixed
    /// field list: from here on no segment this writer flushes carries a norm
    /// column for it, and its `.fnm` says `omitNorms` so a reader never looks
    /// for one.
    ///
    /// This is the **only** norms knob, and it is an opt-out, matching
    /// Lucene: an indexed field gets norms unless it says otherwise (see
    /// [`Self::norms_field_configs`]). A field that is not indexed
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
        let info = self
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
        self.vector_fields.clear();
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
        let info = self
            .fields
            .iter()
            .find(|f| f.name == field_name)
            .ok_or_else(|| Error::UnknownVectorField(field_name.to_string()))?;
        if info.vector_dimension <= 0 {
            return Err(Error::UnsupportedVectorField(
                field_name.to_string(),
                info.vector_dimension,
            ));
        }
        if self.vector_fields.iter().any(|c| c.name == field_name) {
            return Err(Error::DuplicateVectorField(field_name.to_string()));
        }
        self.vector_fields.push(VectorFieldConfig {
            name: field_name.to_string(),
            field_number: info.number,
            dimension: info.vector_dimension,
            encoding: info.vector_encoding,
            similarity: info.vector_similarity_function,
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
        self.hnsw_m = m;
        self.hnsw_beam_width = beam_width;
        Ok(())
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
        &self.fields
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
    /// [`IndexWriter::rollback`]. This is why the method is fallible where it
    /// used to return `()`; `addDocument` throws `IOException` in Java for the
    /// same reason.
    pub fn add_document(&mut self, doc: Document) -> Result<SeqNo> {
        // Not `add_documents(vec![doc])`: the single-document add is the hot
        // path of the whole write side, and wrapping every document in a
        // one-element `Vec` puts a heap allocation and a move in front of it
        // for nothing. A one-document call can never set `has_blocks` (Java:
        // `numDocs > 1`) and carries no delete node, so there is nothing the
        // block path would do that this does not.
        let seq_no = self.delete_queue.next_sequence_number();
        self.buffer_document(doc);
        self.maybe_flush()?;
        Ok(seq_no)
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
        let doc_index = self.pending_docs.len();
        for v in &vectors {
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
        self.maybe_flush()?;
        Ok(seq_no)
    }

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
        self.add_documents_with_delete(None, docs)
    }

    /// `IndexWriter.updateDocuments(Term, Iterable<Iterable<IndexableField>>)`:
    /// atomically deletes every document matching `term` and adds `docs` as one
    /// block. One sequence number for the pair, so the delete and the add can
    /// never be observed apart.
    pub fn update_documents(&mut self, term: Term, docs: Vec<Document>) -> Result<SeqNo> {
        self.add_documents_with_delete(Some(DeleteNode::Terms(vec![term])), docs)
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
        let doc_id_upto = self.pending_doc_id_upto();
        let seq_no = match delete {
            Some(node) => self.buffer_delete_node(node, doc_id_upto),
            None => self.delete_queue.next_sequence_number(),
        };
        if docs.len() > 1 {
            self.pending_has_blocks = true;
        }
        for doc in docs {
            self.buffer_document(doc);
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
    fn pending_doc_id_upto(&self) -> i32 {
        self.pending_docs.len() as i32
    }

    /// Buffers one delete node against both the private (next segment) and
    /// global (already-written segments) buffers, and returns its sequence
    /// number -- `DocumentsWriterDeleteQueue.add(Node, DeleteSlice)`.
    fn buffer_delete_node(&mut self, node: DeleteNode, doc_id_upto: i32) -> SeqNo {
        match node {
            DeleteNode::Terms(terms) => self.delete_queue.add_term_deletes(&terms, doc_id_upto),
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
    /// byte-pool posting representation (recorded in
    /// `docs/sweep/m2/LEDGER.md`).
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

    /// `IndexWriter.deleteUnusedFiles()`: re-apply the deletion policy and
    /// re-scan `dir` for index files no live commit and no pending flush names,
    /// deleting them.
    ///
    /// This is the fallible counterpart of what [`IndexWriter::rollback`] does
    /// silently -- use it when you want a failure to reclaim disk space to
    /// surface as an error rather than be ignored.
    pub fn delete_unused_files(&mut self) -> Result<()> {
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
        self.deleter.refresh()?;
        Ok(())
    }

    /// `FlushByRamOrCountsPolicy.onChange`: document count first, then RAM,
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
        // The setter guarantees the value is either the sentinel (negative) or
        // strictly positive, so `> 0.0` is the enabled test without a float
        // equality comparison.
        if self.ram_buffer_size_mb > 0.0 {
            let limit = (self.ram_buffer_size_mb * 1024.0 * 1024.0) as usize;
            if self.ram_bytes_used >= limit {
                return self.flush();
            }
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
    fn checkpoint_committed(&mut self) -> Result<()> {
        let published = self.segment_infos.clone();
        self.deleter.checkpoint(&published, false)?;
        self.deleter.checkpoint(&published, true)?;
        Ok(())
    }

    /// This writer's in-memory view of the index: the last commit plus every
    /// segment [`IndexWriter::flush`] has written since. Java keeps exactly this
    /// in `segmentInfos`; see [`Self::flushed_segments`] for why this port keeps
    /// the tail separate.
    fn live_infos(&self) -> SegmentInfos {
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
    /// `IndexOptions::DocsAndCustomFreqs` section). It must be `>= 1`: the
    /// underlying codec layer ([`postings_writer::write_fields`]) rejects
    /// `freq < 1` for every `IndexOptions` variant that carries a freq at
    /// all, `DocsAndCustomFreqs` included, since the wire encoding has no
    /// representation for a zero-or-negative freq. A `term` repeated more
    /// than once in `terms` for the same doc contributes one postings entry
    /// per occurrence in this list (i.e. the *last* one wins for that
    /// `(doc, term)` pair's freq, since the codec's per-term-per-doc map
    /// only ever keeps one freq value per doc) -- callers should supply each
    /// distinct term at most once per doc to avoid relying on that.
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
        let seq_no = self.delete_queue.next_sequence_number();
        // ARITH: see [`VectorValue::ram_bytes`].
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.ram_bytes_used += document_ram_bytes(&doc) + custom_freq_terms_ram_bytes(&terms);
        }
        self.pending_docs.push(doc);
        self.pending_custom_freq_terms.push(terms);
        self.pending_vectors.push(Vec::new());
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
        self.add_documents_with_delete(Some(DeleteNode::Terms(vec![term])), vec![doc])
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
        let doc_id_upto = self.pending_doc_id_upto();
        Ok(self.buffer_delete_node(DeleteNode::Terms(terms.to_vec()), doc_id_upto))
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
        if queries.iter().any(|q| matches!(q, DeleteQuery::MatchAll)) {
            let seq_no = self.delete_queue.next_sequence_number();
            self.delete_all()?;
            return Ok(seq_no);
        }
        let doc_id_upto = self.pending_doc_id_upto();
        Ok(self.buffer_delete_node(DeleteNode::Queries(queries.to_vec()), doc_id_upto))
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
        if soft_deletes.is_empty() {
            return Err(Error::NoSoftDeletesSupplied);
        }
        // Java's `buildDocValuesUpdate(term, softDeletes)`: every supplied
        // field becomes an update *keyed by the same term*, so the marking hits
        // exactly the documents a hard `updateDocument` would have deleted.
        let updates: Vec<DocValuesUpdate> = soft_deletes
            .iter()
            .map(|u| retarget_update(u, &term))
            .collect();
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
    /// its own fixed field list: [`Error::UnknownDocValuesUpdateField`] and
    /// [`Error::WrongDocValuesUpdateType`].
    pub fn update_doc_values(&mut self, term: Term, updates: &[DocValuesUpdate]) -> Result<SeqNo> {
        if updates.is_empty() {
            return Err(Error::NoDocValuesUpdatesSupplied);
        }
        let mut retargeted = Vec::with_capacity(updates.len());
        for update in updates {
            self.verify_doc_values_update_field(update)?;
            retargeted.push(retarget_update(update, &term));
        }
        let doc_id_upto = self.pending_doc_id_upto();
        Ok(self.buffer_delete_node(DeleteNode::DocValuesUpdates(retargeted), doc_id_upto))
    }

    /// `IndexWriter.updateNumericDocValue(Term, String, long)`.
    pub fn update_numeric_doc_value(
        &mut self,
        term: Term,
        field: &str,
        value: i64,
    ) -> Result<SeqNo> {
        let update = DocValuesUpdate::Numeric {
            term: term.clone(),
            field: field.to_string(),
            value: Some(value),
        };
        self.update_doc_values(term, std::slice::from_ref(&update))
    }

    /// `IndexWriter.updateBinaryDocValue(Term, String, BytesRef)`. Java
    /// rejects a null value outright here (unlike `updateDocValues`, which
    /// treats null as a removal), so this takes a `&[u8]` and has no
    /// "remove" spelling -- use [`IndexWriter::update_doc_values`] with a
    /// `None` value for that.
    pub fn update_binary_doc_value(
        &mut self,
        term: Term,
        field: &str,
        value: &[u8],
    ) -> Result<SeqNo> {
        let update = DocValuesUpdate::Binary {
            term: term.clone(),
            field: field.to_string(),
            value: Some(value.to_vec()),
        };
        self.update_doc_values(term, std::slice::from_ref(&update))
    }

    /// Java's `globalFieldNumberMap.verifyOrCreateDvOnlyField(field, dvType,
    /// …)`, against this port's fixed field list: the field must exist and its
    /// declared `DocValuesType` must match the update's kind. Java can *create*
    /// the field; this port cannot (the field list is fixed at
    /// [`IndexWriter::open`], see the module doc), so an unknown field is an
    /// error rather than an implicit schema change.
    fn verify_doc_values_update_field(&self, update: &DocValuesUpdate) -> Result<()> {
        let Some(info) = self.fields.iter().find(|f| f.name == update.field()) else {
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
        // `IndexWriter.updateNumericDocValue`/`updateDocValues`:
        //   if (config.getIndexSortFields().contains(field)) throw new
        //     IllegalArgumentException("cannot update docvalues field involved
        //       in the index sort, field=" + field + ", sort=" + ...);
        // The segment's *physical* order is defined by this column, and a
        // doc-values update rewrites the column without moving any document,
        // so the segment would keep claiming a sort it no longer satisfies --
        // silently, since nothing re-checks the order after an update.
        if let Some(sort) = &self.index_sort {
            if sort.iter().any(|sf| sf.field == update.field()) {
                return Err(Error::DocValuesUpdateOnIndexSortField {
                    field: update.field().to_string(),
                    sort: segment_info::describe_index_sort(Some(sort)),
                });
            }
        }
        Ok(())
    }

    /// The atomic delete-by-term + add-document real Lucene calls
    /// `updateDocument`: delegates directly to
    /// [`update_document::update_document`], flushing `doc` as a brand-new
    /// segment and applying the term delete to every segment `delete_sources`
    /// supplies an opened source for, all in one commit. Unlike
    /// [`IndexWriter::add_document`], this is **not** buffered -- it commits
    /// immediately (matching [`update_document::update_document`]'s own
    /// all-or-nothing atomicity, which only makes sense as an immediate
    /// commit; buffering it would let a later `commit()` observe a
    /// half-applied update if that call somehow failed in between).
    ///
    /// Bumps this writer's in-memory [`IndexWriter::segment_infos`] to the
    /// new commit on success and returns it; on `Err`, nothing was written
    /// (see [`update_document::update_document`]'s own atomicity guarantee)
    /// and every observable part of this writer's state -- its segment list,
    /// generation, version and commit id -- is unchanged. The one thing that
    /// does advance is `segment_infos.counter`: the segment name this attempt
    /// claimed is burned, exactly as real `IndexWriter.newSegmentName()` burns
    /// it, so a retry never writes over files the failed attempt may have left.
    #[allow(clippy::too_many_arguments)]
    pub fn update_document_with_sources(
        &mut self,
        delete_sources: &[SegmentDeleteSource],
        field: &str,
        term: &[u8],
        new_doc: Document,
    ) -> Result<&SegmentInfos> {
        if self.prepared_commit.is_some() {
            return Err(Error::PreparedCommitPending("update_document"));
        }
        let new_segment_name = self.new_segment_name();
        let new_segment_id = generate_segment_id(self.segment_infos.counter);

        // Real `IndexWriter.newSegmentName()` increments `segmentInfos.counter`
        // *at the moment it hands out a name*, so the counter that gets written
        // into the very next commit already accounts for the segment being
        // flushed. Bumping it only on `self` afterwards (as this method used
        // to) left the on-disk `segments_N` carrying the pre-flush counter, so
        // a fresh `IndexWriter::open` on the same directory handed out the
        // *same* `_N` name again and the next flush overwrote a live segment's
        // files. The bump therefore has to be on the `SegmentInfos` that is
        // about to be written, not on a copy that never reaches disk.
        let mut base = self.segment_infos.clone();
        base.id = generate_segment_id(base.counter);

        let updated = update_document::update_document(
            self.dir,
            &base,
            delete_sources,
            field,
            term,
            &new_segment_name,
            new_segment_id,
            &self.codec_name,
            self.lucene_version,
            &self.fields,
            std::slice::from_ref(&new_doc),
        )?;
        // This wrote its own `segments_N`, so the deleter sees it as a commit:
        // the superseded generation's commit point dies here, taking the `.liv`
        // generation this update replaced with it.
        self.segment_infos = updated;
        // `finishCommit`: `rollbackSegments =
        // pendingCommit.createBackupSegmentInfos()`. This is the one place a
        // commit becomes durable, so it is the one place the rollback snapshot
        // moves forward.
        self.rollback_segments = self.segment_infos.segments.clone();
        self.checkpoint_committed()?;
        Ok(&self.segment_infos)
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
        new_segment_infos.id = generate_segment_id(new_segment_infos.generation);
        new_segment_infos.segments = updated_segments;
        segment_infos::write(&new_segment_infos, self.dir)?;

        self.segment_infos = new_segment_infos;
        // `finishCommit`: `rollbackSegments =
        // pendingCommit.createBackupSegmentInfos()`. This is the one place a
        // commit becomes durable, so it is the one place the rollback snapshot
        // moves forward.
        self.rollback_segments = self.segment_infos.segments.clone();
        self.checkpoint_committed()?;
        Ok(&self.segment_infos)
    }

    /// Flushes every currently-buffered [`IndexWriter::add_document`] call
    /// (if any) to a brand-new segment via
    /// [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment), appends it to this writer's segment
    /// list, and writes the whole updated list as the next `segments_N`
    /// generation via [`crate::segment_infos::write`] -- real Lucene's
    /// `IndexWriter.commit()` after one or more buffered
    /// `DocumentsWriterPerThread.flush()`-worth of documents. If this writer
    /// has a [`MergePolicyConfig`] set via [`IndexWriter::set_merge_policy`],
    /// this also performs real `commit()`'s automatic merge-triggering step
    /// (see module doc comment) right after the flush commits; with no
    /// merge policy set (the default), this method is unchanged from before
    /// that feature existed.
    ///
    /// A `commit()` with an empty pending-document buffer still writes the
    /// next `segments_N` generation (bumping `version`) with no new
    /// segment appended -- matches real Lucene's `commit()` being a valid,
    /// if unusual, no-op-content commit rather than a special "nothing to do"
    /// case that skips writing. Returns the new committed [`SegmentInfos`].
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
        self.prepare_commit()?;
        self.finish_commit()
    }

    /// The file-writing half of a real two-phase commit, port of real
    /// Lucene's `IndexWriter.prepareCommit()`: does every single thing
    /// [`IndexWriter::commit`] used to do *except* the final
    /// [`crate::segment_infos::write`] call that actually produces the next
    /// `segments_N` -- flushes `pending_docs` (if any) to a brand-new
    /// segment via [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment), builds and writes that
    /// segment's postings/term-vector/doc-values files exactly as
    /// [`IndexWriter::commit`] always has, and stashes the resulting
    /// in-memory [`SegmentInfos`] (bumped generation/version, new segment
    /// appended) in `self.prepared_commit` for [`IndexWriter::finish_commit`]
    /// to pick up.
    ///
    /// # The on-disk protocol
    ///
    /// This is Lucene's real two-phase shape, not an in-memory imitation of
    /// it. After flushing and syncing the new segment's data files, this
    /// serializes the whole commit and writes it -- fsynced -- as
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
    ///   which is exactly the failure the older "write `segments_N` directly"
    ///   shape had, and exactly why Java never creates a `segments_N` by
    ///   writing to it.
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
    /// Calling `prepare_commit()` again while a previous prepared commit has
    /// not been activated by [`IndexWriter::finish_commit`] is an error
    /// ([`Error::PrepareCommitAlreadyCalled`]), matching real
    /// `prepareCommitInternal`'s `IllegalStateException("prepareCommit was
    /// already called with no corresponding call to commit")` -- call
    /// `finish_commit()` to activate it or [`IndexWriter::rollback`] to
    /// discard it (which also deletes the pending file) first.
    pub fn prepare_commit(&mut self) -> Result<()> {
        // Real `prepareCommitInternal` refuses re-entry outright
        // (`IllegalStateException("prepareCommit was already called with no
        // corresponding call to commit")`). This used to *replace* the pending
        // state instead, which silently threw away every document the first
        // prepare had already flushed: the second prepare rebuilds its
        // `SegmentInfos` from `self.segment_infos`, which never saw the first
        // prepare's segment, so `finish_commit` published a commit that did not
        // reference it.
        if self.prepared_commit.is_some() {
            return Err(Error::PrepareCommitAlreadyCalled);
        }

        // Everything still buffered becomes one last segment; segments an
        // automatic flush already wrote are folded in below.
        self.flush()?;

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
        // Java writes a fresh `StringHelper.randomId()` into every
        // `segments_N` header it produces, so two commits of the same index are
        // never confusable by id; cloning the previous commit's id would make
        // every generation of an index report the same one.
        new_segment_infos.id = generate_segment_id(new_segment_infos.generation);
        new_segment_infos
            .segments
            .extend(self.flushed_segments.iter().cloned());

        // Phase one of the real two-phase protocol: the commit is fully
        // serialized and fsynced under `pending_segments_N`, a name
        // `last_commit_generation` does not scan for, so a crash between here
        // and `finish_commit` leaves the previous commit current and this one
        // an inert orphan -- never a truncated `segments_N` that makes the
        // whole index unopenable.
        segment_infos::write_pending(&new_segment_infos, self.dir)?;
        self.prepared_commit = Some(new_segment_infos);
        Ok(())
    }

    /// Writes every currently-buffered document out as one new segment, without
    /// committing it: real Lucene's `DocumentsWriterPerThread.flush()` followed
    /// by `IndexWriter.publishFlushedSegment`.
    ///
    /// The new segment's files are on disk and fsynced when this returns, and
    /// the deleter has been checkpointed (non-commit) so nothing reclaims them,
    /// but no `segments_N` names them yet -- a fresh reader still sees the
    /// previous commit. [`IndexWriter::prepare_commit`] folds every segment
    /// flushed this way into the commit it writes; [`IndexWriter::rollback`]
    /// drops them and the deleter deletes their files.
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
        // See `maybe_flush`: a flush between `prepare_commit` and
        // `finish_commit` would have its segment and its resolved deletes
        // discarded by the publish. The automatic trigger defers; an explicit
        // call is refused, because silently doing nothing would be worse.
        if self.prepared_commit.is_some() {
            return Err(Error::PreparedCommitPending("flush"));
        }
        if self.pending_docs.is_empty() {
            // Deletes issued while the document buffer was empty still have to
            // be resolved -- Java's `applyAllDeletes` is not conditional on a
            // segment having been produced.
            self.apply_all_deletes_and_updates()?;
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
        // forgotten -- c10 found the vector one unreachable, and points has
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
        if let Some(global) = self.delete_queue.freeze_global_buffer() {
            self.updates_stream.push(global);
        }

        let segment_name = self.new_segment_name();
        let segment_id = generate_segment_id(self.segment_infos.counter);

        // Built and validated entirely in memory before anything is
        // written to `dir` -- see `prepare_commit`'s own doc comment on why
        // that ordering is what makes a docFreq-too-large rejection
        // atomic.
        // `postings_fields`/`custom_freq_postings_field` are enforced
        // mutually exclusive by `set_postings_field`/`add_postings_field`/
        // `set_custom_freq_postings_field` (see those methods' doc
        // comments), so at most one branch below ever produces `Some` --
        // no ambiguity about which one "wins".
        // One call site, so one error path -- see
        // `build_and_write_segment` on why that is load-bearing here.
        let written = self.build_and_write_segment(&segment_name, segment_id);

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
                // `DocumentsWriterPerThread.abort()`: the files this failed
                // flush created are referenced by nothing, so `refresh()`
                // removes exactly them. Its own failure must not mask the
                // original error.
                let _ = self.deleter.refresh();
                return Err(e);
            }
        };

        self.pending_docs.clear();
        self.pending_custom_freq_terms.clear();
        self.pending_vectors.clear();
        self.ram_bytes_used = 0;
        self.pending_has_blocks = false;
        // `IndexWriter.publishFlushedSegment`: `rld.sortMap = sortMap`. See
        // [`Self::pending_sort_map`] for why this port scopes it to this call
        // rather than pooling a reader for the writer's lifetime.
        self.pending_sort_map = sort_map.map(|m| (segment_name.clone(), m));

        // `IndexWriter.publishFlushedSegment`: the segment-private packet
        // (this segment's own deletes, each still carrying the buffer position
        // it was issued at) is pushed, and the generation it receives becomes
        // the segment's `bufferedDeletesGen`. With no private packet, Java
        // burns a generation anyway (`bufferedUpdatesStream.getNextGen()`), so
        // that every packet pushed *before* this flush sorts strictly below
        // the new segment and therefore cannot touch it.
        let mut sci = sci;
        let private = self.delete_queue.freeze_private_buffer(&segment_name);
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
        self.flushed_segments.push(sci);

        // `IndexFileDeleter.checkpoint(segmentInfos, false)`: the new segment's
        // files are now referenced by this writer's in-memory view, so nothing
        // reclaims them before a commit names them.
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;

        self.apply_all_deletes_and_updates()?;
        // Every packet with a limit below `MAX_DOC_ID_UPTO` for this segment
        // was applied above (`apply_all_deletes_and_updates` drains the whole
        // stream), so the map has no further reader.
        self.pending_sort_map = None;

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
        // `finishCommit`: `rollbackSegments =
        // pendingCommit.createBackupSegmentInfos()`. This is the one place a
        // commit becomes durable, so it is the one place the rollback snapshot
        // moves forward.
        self.rollback_segments = self.segment_infos.segments.clone();
        self.checkpoint_committed()?;

        if self.merge_policy.is_some() {
            self.auto_merge()?;
        }

        Ok(&self.segment_infos)
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
    fn build_and_write_segment(
        &self,
        segment_name: &str,
        segment_id: [u8; ID_LENGTH],
    ) -> Result<(SegmentCommitInfo, Vec<String>)> {
        // `IndexingChain.writeNorms`' loop condition, resolved once: every
        // indexed field that has not opted out gets a norm column, and the
        // shared invert pass has to analyze all of them.
        let norms_configs = self.norms_field_configs();
        let inverted = Self::invert_pending_fields(
            &self.pending_docs,
            &self.postings_fields,
            &self.term_vector_fields,
            &norms_configs,
            &self.payload_field_names(),
            self.payload_source.as_deref(),
            &self.analyzer(),
        );
        // Norms and term vectors only *read* the shared invert pass; postings
        // **consumes** it. Ordering them this way means the whole inverted
        // index -- by far the largest transient structure in a flush -- is
        // freed term by term as the postings are built, instead of staying
        // live alongside the postings copy, the stored-fields copy and every
        // output file's byte buffer.
        let norms_output = if norms_configs.is_empty() {
            None
        } else {
            Some(Self::build_norms_output(
                &self.pending_docs,
                &norms_configs,
                &inverted,
                &segment_id,
            )?)
        };
        // Borrowed out of `norms_output` before the postings consume
        // `inverted`, because the postings writer needs them for its impacts.
        let impact_norms: &[(i32, Vec<i64>)] = norms_output
            .as_ref()
            .map(|(_, _, columns)| columns.as_slice())
            .unwrap_or(&[]);
        let term_vectors_output = if self.term_vector_fields.is_empty() {
            None
        } else {
            Self::build_term_vectors_output(&self.pending_docs, &self.term_vector_fields, &inverted)
                .map(|docs| term_vectors::write_best_speed(&docs, &segment_id, ""))
        };
        let doc_values_output = if self.doc_values_fields.is_empty() {
            None
        } else {
            Some(Self::build_doc_values_output(
                &self.pending_docs,
                &self.doc_values_fields,
                &segment_id,
            )?)
        };
        let postings_output = if !self.postings_fields.is_empty() {
            Self::build_postings_output(&self.postings_fields, inverted, impact_norms, &segment_id)?
        } else {
            drop(inverted);
            match &self.custom_freq_postings_field {
                Some(cfg) => Self::build_custom_freq_postings_output(
                    &self.pending_docs,
                    &self.pending_custom_freq_terms,
                    cfg,
                    &segment_id,
                )?,
                None => None,
            }
        };

        let vectors_output = if self.vector_fields.is_empty() {
            None
        } else {
            Self::build_vectors_output(
                &self.pending_vectors,
                &self.vector_fields,
                self.pending_docs.len() as i32,
                self.hnsw_m,
                self.hnsw_beam_width,
                &segment_id,
            )?
        };

        let fnm_fields = self.fields_with_per_field_attributes(
            postings_output.is_some(),
            doc_values_output.is_some(),
            norms_output.is_some(),
            vectors_output
                .as_ref()
                .map(|o| o.written_fields.as_slice())
                .unwrap_or(&[]),
        );

        // The inner closure the previous shape needed to collect `?`s is
        // gone: this whole method is that scope now.
        // `IndexWriter.sealFlushedSegment`: every format writes its own files
        // into the segment, each one adding them to the *in-memory*
        // `SegmentInfo.files`, and the `.si` is written once at the end from
        // that accumulated set.
        //
        // This used to be a `.si` write per file group: the stored-fields
        // flush wrote one, then postings, term vectors, doc values, norms,
        // vectors and the index-sort descriptor each re-opened it, re-parsed
        // it, extended `files` and rewrote and re-fsynced it -- up to seven
        // writes, six parses and seven fsyncs of a file whose entire content
        // was already in memory, for a segment whose *data* files were written
        // exactly once. Only the last write survived; the six before it were
        // pure I/O.
        let mut flushed = segment_writer::write_stored_only_segment_files(
            self.dir,
            segment_name,
            segment_id,
            &self.codec_name,
            self.lucene_version,
            &fnm_fields,
            &self.pending_docs,
            false,
            // `DocumentsWriterPerThread.updateDocuments`: set once any
            // `add_documents`/`update_documents` call in this buffer
            // carried more than one document.
            self.pending_has_blocks,
        )?;

        let mut record = |names: Vec<String>| {
            flushed.info.files.extend(names.iter().cloned());
            flushed.pending_sync.extend(names);
        };
        if let Some(output) = postings_output {
            record(Self::write_postings_files(self.dir, segment_name, &output)?);
        }
        if let Some((tvd, tvx, tvm)) = term_vectors_output {
            record(Self::write_term_vector_files(
                self.dir,
                segment_name,
                &tvd,
                &tvx,
                &tvm,
            )?);
        }
        if let Some((dvm, dvd, dvs)) = doc_values_output {
            record(Self::write_doc_values_files(
                self.dir,
                segment_name,
                &dvm,
                &dvd,
                &dvs,
            )?);
        }
        if let Some((nvm, nvd, _)) = norms_output {
            record(Self::write_norms_files(self.dir, segment_name, &nvm, &nvd)?);
        }
        if let Some(output) = &vectors_output {
            record(Self::write_vector_files(self.dir, segment_name, output)?);
        }
        // `SegmentInfo`'s `numSortFields` block is what a reader surfaces as
        // `LeafMetaData.getSort()`, and what `CheckIndex.testSort` re-derives
        // its comparators from. It goes into the same in-memory `SegmentInfo`
        // as the file names, so it costs nothing extra.
        if let Some(sort) = &self.index_sort {
            flushed.info.index_sort = Some(sort.clone());
        }
        segment_writer::seal_flushed_segment(self.dir, segment_name, flushed).map_err(Error::from)
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
    /// unreachable here for exactly that reason, c10 finding 11).
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
    /// carrying document blocks: a block must stay contiguous and in order,
    /// and a sort would shred it. Java refuses the same combination unless a
    /// *parent* field marks each block's last document, which this port has
    /// no write path for.
    fn sort_pending_buffer(&mut self) -> Result<Option<Vec<usize>>> {
        let Some(sort) = self.index_sort.clone() else {
            return Ok(None);
        };
        if self.pending_has_blocks {
            return Err(Error::IndexSortWithBlocksAndNoParentField);
        }

        let keys: Vec<Vec<Option<i64>>> = sort
            .iter()
            .map(|sf| {
                let field_number = self
                    .fields
                    .iter()
                    .find(|f| f.name == sf.field)
                    .map(|f| f.number)
                    .expect("set_index_sort resolved every sort field against this fixed list");
                let selector = match &sf.kind {
                    segment_info::IndexSortKind::SortedNumeric { selector, .. } => Some(*selector),
                    // Single-valued NUMERIC: the first (and only) value.
                    // `set_index_sort` has already refused every kind that
                    // is neither of these two.
                    _ => None,
                };
                self.pending_docs
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
                    .collect()
            })
            .collect();

        let specs: Vec<segment_writer::SortKeySpec<'_>> = sort
            .iter()
            .zip(keys.iter())
            .map(|(sf, k)| segment_writer::SortKeySpec { sort: sf, keys: k })
            .collect();
        let new_to_old = segment_writer::sort_permutation(self.pending_docs.len(), &specs);
        drop(specs);

        // `Sorter.sortAndLeaveUnpacked` returns null when the documents are
        // already in order; the segment is still *recorded* as sorted, there
        // is simply nothing to permute and no map any delete limit needs.
        if new_to_old.iter().enumerate().all(|(new, &old)| new == old) {
            return Ok(None);
        }

        segment_writer::permute_in_place(&mut self.pending_docs, &new_to_old);
        segment_writer::permute_in_place(&mut self.pending_custom_freq_terms, &new_to_old);
        segment_writer::permute_in_place(&mut self.pending_vectors, &new_to_old);
        Ok(Some(new_to_old))
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
    /// `NormValuesWriter` (norms) from that one pass. This port previously ran
    /// a *separate* full tokenize-and-invert over the same text per consumer:
    /// a commit with postings and norms on one field analyzed every document
    /// twice, and with term vectors as well, three times. Measured on
    /// `benchmarks/rust-runner`'s `index-bench` (20k docs x 40 tokens,
    /// postings + norms on one field), the redundant norms pass alone cost
    /// 17.6 us/doc out of 43.6 us/doc -- 40% of the whole commit.
    ///
    /// A field with no [`FieldValue::String`] value on any pending doc simply
    /// contributes no entries, exactly as it did when each consumer built its
    /// own `triples` list and bailed on an empty one.
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
        // `find` used to stop at the first value, so every later value of a
        // multi-valued field was stored but never indexed.
        let mut triples: Vec<(i32, &str, &str)> = Vec::new();
        for (field_number, field_name) in wanted {
            for (doc_id, doc) in docs.iter().enumerate() {
                for f in doc.fields.iter().filter(|f| f.field_number == field_number) {
                    if let FieldValue::String(text) = &f.value {
                        triples.push((doc_id as i32, field_name, text.as_str()));
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
    /// matching [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment)'s own doc-ordering), tokenizes
    /// each field independently via
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
    /// `docFreq` ceiling on a positions-indexing term: batch c20 removed the
    /// one that existed (`DocFreqTooLargeForPositions`, which stood in for the
    /// `.doc`-side full-block writer's then-missing pos/pay skip fields) when
    /// it wrote those fields.
    ///
    /// **The offsets it forwards are UTF-16 code units** -- Java `char` indices
    /// into the field text, the unit `OffsetAttribute` reports and the unit
    /// real Lucene reads back out of the `.pos`/`.pay` written here. They were
    /// `lucene_analysis`' UTF-8 **byte** offsets until c33 fixed the producer
    /// (c23's F13, shipped rather than latent once this method started writing
    /// them). Nothing here converts, and nothing on either side would catch a
    /// regression: `CheckIndex` only checks that offsets are ordered and in
    /// range, never that they index the stored text. The unit is pinned where
    /// it is produced, in `crates/lucene-analysis/tests/analysis_fixtures.rs`.
    fn build_postings_output(
        configs: &[PostingsFieldConfig],
        inverted: InMemoryInvertedIndex,
        norms: &[(i32, Vec<i64>)],
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<Option<postings_writer::Output>> {
        struct FieldData {
            config: PostingsFieldConfig,
            doc_ids: std::collections::BTreeSet<i32>,
            terms: Vec<TermPostings>,
        }

        let mut per_field: Vec<FieldData> = configs
            .iter()
            .map(|config| FieldData {
                config: config.clone(),
                doc_ids: std::collections::BTreeSet::new(),
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
            // contract exactly, and is what keeps that path byte-for-byte
            // unchanged from before this task.
            let mut positions: Vec<Vec<i32>> = Vec::new();
            let mut offsets: Vec<Vec<(i32, i32)>> = Vec::new();
            if has_positions {
                positions.reserve_exact(entries.len());
            }
            if has_offsets {
                offsets.reserve_exact(entries.len());
            }
            for entry in entries {
                data.doc_ids.insert(entry.doc_id);
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
                doc_count: f.doc_ids.len() as i32,
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
        let output = postings_writer::write_fields_with_norms(
            &inputs,
            &norms_for_impacts,
            segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )?;
        Ok(Some(output))
    }

    /// [`Self::custom_freq_postings_field`]'s counterpart to
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
        docs: &[Document],
        custom_freq_terms: &[Vec<(String, i32)>],
        config: &CustomFreqPostingsFieldConfig,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<Option<postings_writer::Output>> {
        let mut per_term: std::collections::BTreeMap<Vec<u8>, Vec<(i32, i32)>> =
            std::collections::BTreeMap::new();
        for doc_id in 0..docs.len() {
            let Some(terms) = custom_freq_terms.get(doc_id) else {
                continue;
            };
            for (term, custom_freq) in terms {
                per_term
                    .entry(term.as_bytes().to_vec())
                    .or_default()
                    .push((doc_id as i32, *custom_freq));
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
        let output = postings_writer::write_fields(
            &inputs,
            segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )?;
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
    /// takes. Before c35 every doc got a dense `0`, which is the byte a
    /// present-but-empty field has: an absent field and an empty one were
    /// indistinguishable in the file.
    ///
    /// A doc's length is the sum of every matching term's occurrence count
    /// (real Lucene's `FieldInvertState.length` -- total indexed token
    /// count, *not* distinct-term count: "fox fox fox" has length 3, one
    /// distinct term), encoded into a single norm byte via
    /// [`small_float::int_to_byte4`] and sign-extended into an `i64` the same
    /// way `norms::norm_value`'s read side sign-extends a stored byte back
    /// (`byte as i8 as i64`) -- the exact inverse transformation
    /// `lucene_search::similarity::decode_norm` undoes.
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
    fn build_norms_output(
        docs: &[Document],
        configs: &[NormsFieldConfig],
        inverted: &InMemoryInvertedIndex,
        segment_id: &[u8; ID_LENGTH],
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
                // `None` == this doc does not carry the field at all, so it
                // gets no norm; `Some(0)` == it carries it but produced no
                // tokens, which is Java's explicit zero.
                let mut lengths: Vec<Option<u32>> = docs
                    .iter()
                    .map(|doc| {
                        doc.fields
                            .iter()
                            .find(|f| f.field_number == config.field_number)
                            .and_then(|f| match &f.value {
                                FieldValue::String(_) => Some(0u32),
                                _ => None,
                            })
                    })
                    .collect();
                for ((field, _term), list) in &inverted.terms {
                    let entries = &list.entries;
                    if field != &config.name {
                        continue;
                    }
                    for entry in entries {
                        // `entry.doc_id` is an index into `docs` that this
                        // writer's own inversion produced, so it addresses
                        // `lengths` (which is `docs.len()` long) by
                        // construction -- and only ever for a doc whose
                        // presence test above already said `Some`.
                        if let Some(slot) = lengths[entry.doc_id as usize].as_mut() {
                            *slot = accumulate_field_length(*slot, entry.term_freq());
                        }
                    }
                }
                let norm = |len: u32| small_float::int_to_byte4(len) as i8 as i64;
                let sparse = (!lengths.iter().all(|l| l.is_some())).then(|| {
                    lengths
                        .iter()
                        .enumerate()
                        .filter_map(|(doc, len)| len.map(|len| (doc as i32, norm(len))))
                        .collect()
                });
                NormsColumn {
                    // The filler is the norm `1`, **not** `norm(0)`: an
                    // absent doc has no norm at all, and `norm(0) == 0` is the
                    // one value `CheckIndex.checkImpacts` rejects outright
                    // ("First impact had a norm == 0"). A present-but-empty
                    // doc still gets its legitimate explicit `norm(0)`, which
                    // is what Lucene writes for it.
                    //
                    // The filler never fires for a column with no absent doc,
                    // so the `Dense` arm below writes exactly the bytes it
                    // wrote before this column grew its second reader.
                    dense: lengths
                        .into_iter()
                        .map(|l| l.map(norm).unwrap_or(1))
                        .collect(),
                    sparse,
                }
            })
            .collect();

        let fields: Vec<norms::NormsField<'_>> = configs
            .iter()
            .zip(&columns)
            .map(|(config, column)| match &column.sparse {
                None => norms::NormsField::Dense(config.field_number, &column.dense),
                Some(pairs) => norms::NormsField::Sparse(config.field_number, pairs),
            })
            .collect();

        let (nvm, nvd) = norms::write_fields(&fields, docs.len() as i32, segment_id, "")?;
        // The dense columns move out to the postings writer, which is the
        // whole reason they are built dense -- no copy is made.
        let impact_norms: Vec<(i32, Vec<i64>)> = configs
            .iter()
            .zip(columns)
            .map(|(config, column)| (config.field_number, column.dense))
            .collect();
        Ok((nvm, nvd, impact_norms))
    }

    /// Builds one [`TermVectorsDocument`] per entry in `docs` (in the same
    /// doc-ID order [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment) uses -- index into `docs`
    /// == doc ID in the new segment), sourced from **every** field in
    /// `configs`' [`FieldValue::String`] values, each tokenized independently
    /// via [`crate::indexing_chain::invert_documents`] with a plain
    /// [`Analyzer::standard`] (same analyzer/scope notes as
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
    /// `write_best_speed` call is needed the way postings previously needed
    /// one `write_single_field` call per field.
    ///
    /// Returns `Ok(None)` when no pending doc has any indexable text for
    /// *any* field in `configs` (mirrors
    /// [`IndexWriter::build_postings_output`]'s own "nothing to write this
    /// commit" outcome). Otherwise returns exactly `docs.len()` entries, one
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
        // appended to in this loop's order. (Raised by b7; the fix belongs
        // here, where names are known -- `term_vectors::write_best_speed` sees
        // only numbers.)
        let mut ordered_configs: Vec<&TermVectorFieldConfig> = configs.iter().collect();
        ordered_configs.sort_by(|a, b| a.name.cmp(&b.name));

        let mut per_doc: Vec<Vec<TermVectorField>> = vec![Vec::new(); docs.len()];
        let mut any_content = false;

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
            for ((field, term), list) in &inverted.terms {
                if field != &config.name {
                    continue;
                }
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

            for (doc_id, terms) in field_terms_per_doc.into_iter().enumerate() {
                if terms.is_empty() {
                    continue;
                }
                any_content = true;
                per_doc[doc_id].push(TermVectorField {
                    field_number: config.field_number,
                    has_positions: true,
                    has_offsets,
                    has_payloads,
                    terms,
                });
            }
        }

        if !any_content {
            return None;
        }

        let tv_docs = per_doc
            .into_iter()
            .map(|fields| TermVectorsDocument { fields })
            .collect();
        Some(tv_docs)
    }

    /// Builds [`doc_values::write_single_dense_numeric_field`]'s (or, when
    /// some but not all docs carry a value,
    /// [`doc_values::write_single_sparse_numeric_field`]'s) input from
    /// `docs`' values for `config.field_number` (each pending doc's index
    /// into `docs` becomes its doc ID in the new segment, matching
    /// [`flush_stored_only_segment`](crate::segment_writer::flush_stored_only_segment)'s own doc-ordering) and calls the
    /// appropriate one to actually encode the bytes.
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
    /// interleaved into the same buffers. That writer is dense-only, so a
    /// field missing a value on some doc is
    /// [`Error::SparseFieldInMultiFieldDocValues`] rather than a silent
    /// downgrade; with exactly one field the sparse writers above are still
    /// used.
    fn build_doc_values_output(
        docs: &[Document],
        configs: &[DocValuesFieldConfig],
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        if configs.len() == 1 {
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

        let columns: Vec<DenseColumn> = configs
            .iter()
            .map(|config| Self::collect_dense_column(docs, config))
            .collect::<Result<Vec<_>>>()?;
        let fields: Vec<doc_values::DenseField<'_>> =
            columns.iter().map(DenseColumn::as_dense_field).collect();
        Ok(doc_values::write_dense_fields(
            &fields,
            docs.len() as i32,
            segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        )?)
    }

    /// One field's whole column, dense over `docs`, for the multi-field
    /// branch of [`Self::build_doc_values_output`]. Every doc must carry a
    /// value; the count that do not is reported, because "which document is
    /// missing the sort key" is the first thing a caller wants to know.
    fn collect_dense_column(
        docs: &[Document],
        config: &DocValuesFieldConfig,
    ) -> Result<DenseColumn> {
        let sparse = |missing: usize| Error::SparseFieldInMultiFieldDocValues {
            field: config.name.clone(),
            missing,
            max_doc: docs.len(),
        };
        match config.doc_values_type {
            DocValuesType::Binary | DocValuesType::Sorted => {
                let present = Self::collect_binary_values(docs, config)?;
                if present.len() != docs.len() {
                    // ARITH: `collect_binary_values` pushes at most one
                    // entry per element of `docs` (one `find` per document,
                    // and a document with no value for the field is skipped),
                    // so `present.len() <= docs.len()`.
                    #[allow(clippy::arithmetic_side_effects)]
                    return Err(sparse(docs.len() - present.len()));
                }
                let values: Vec<Vec<u8>> = present.into_iter().map(|(_, v)| v).collect();
                Ok(if config.doc_values_type == DocValuesType::Binary {
                    DenseColumn::Binary(config.field_number, values)
                } else {
                    DenseColumn::Sorted(config.field_number, values)
                })
            }
            DocValuesType::SortedNumeric => {
                let present = Self::collect_sorted_numeric_values(docs, config)?;
                if present.len() != docs.len() {
                    // ARITH: `collect_sorted_numeric_values` pushes at most one
                    // entry per element of `docs` (one `find` per document,
                    // and a document with no value for the field is skipped),
                    // so `present.len() <= docs.len()`.
                    #[allow(clippy::arithmetic_side_effects)]
                    return Err(sparse(docs.len() - present.len()));
                }
                Ok(DenseColumn::SortedNumeric(
                    config.field_number,
                    present.into_iter().map(|(_, v)| v).collect(),
                ))
            }
            DocValuesType::SortedSet => {
                let present = Self::collect_sorted_set_values(docs, config)?;
                if present.len() != docs.len() {
                    // ARITH: `collect_sorted_set_values` pushes at most one
                    // entry per element of `docs` (one `find` per document,
                    // and a document with no value for the field is skipped),
                    // so `present.len() <= docs.len()`.
                    #[allow(clippy::arithmetic_side_effects)]
                    return Err(sparse(docs.len() - present.len()));
                }
                Ok(DenseColumn::SortedSet(
                    config.field_number,
                    present.into_iter().map(|(_, v)| v).collect(),
                ))
            }
            _ => {
                let present = Self::collect_numeric_values(docs, config)?;
                // NUMERIC is the one type the multi-field writer can express
                // sparsely, which is what lets an index-sort tier have
                // missing values.
                if present.len() != docs.len() {
                    return Ok(DenseColumn::SparseNumeric(config.field_number, present));
                }
                Ok(DenseColumn::Numeric(
                    config.field_number,
                    present.into_iter().map(|(_, v)| v).collect(),
                ))
            }
        }
    }

    /// `(doc_id, value)` for every doc carrying a NUMERIC value for
    /// `config`, in doc order. Shared by the dense/sparse single-field
    /// writer and by [`Self::collect_dense_column`], so the two can never
    /// disagree about what a document's value is.
    fn collect_numeric_values(
        docs: &[Document],
        config: &DocValuesFieldConfig,
    ) -> Result<Vec<(i32, i64)>> {
        let mut present: Vec<(i32, i64)> = Vec::with_capacity(docs.len());
        for (doc_id, doc) in docs.iter().enumerate() {
            let Some(field) = doc
                .fields
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
    fn collect_binary_values(
        docs: &[Document],
        config: &DocValuesFieldConfig,
    ) -> Result<Vec<(i32, Vec<u8>)>> {
        let mut present: Vec<(i32, Vec<u8>)> = Vec::with_capacity(docs.len());
        for (doc_id, doc) in docs.iter().enumerate() {
            let Some(field) = doc
                .fields
                .iter()
                .find(|f| f.field_number == config.field_number)
            else {
                continue;
            };
            let value = match &field.value {
                FieldValue::String(s) => s.as_bytes().to_vec(),
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
    /// selector pick the wrong one rather than merely looking untidy. This
    /// port used to keep the caller's order.
    fn collect_sorted_numeric_values(
        docs: &[Document],
        config: &DocValuesFieldConfig,
    ) -> Result<Vec<(i32, Vec<i64>)>> {
        let mut present: Vec<(i32, Vec<i64>)> = Vec::with_capacity(docs.len());
        for (doc_id, doc) in docs.iter().enumerate() {
            let mut per_doc: Vec<i64> = Vec::new();
            for field in doc
                .fields
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
    fn collect_sorted_set_values(
        docs: &[Document],
        config: &DocValuesFieldConfig,
    ) -> Result<Vec<(i32, Vec<Vec<u8>>)>> {
        let mut present: Vec<(i32, Vec<Vec<u8>>)> = Vec::with_capacity(docs.len());
        for (doc_id, doc) in docs.iter().enumerate() {
            let mut per_doc: Vec<Vec<u8>> = Vec::new();
            for field in doc
                .fields
                .iter()
                .filter(|f| f.field_number == config.field_number)
            {
                let value = match &field.value {
                    FieldValue::String(s) => s.as_bytes().to_vec(),
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
    fn build_numeric_doc_values_output(
        docs: &[Document],
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
    fn build_sorted_doc_values_output(
        docs: &[Document],
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
    fn build_binary_doc_values_output(
        docs: &[Document],
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
    fn build_sorted_numeric_doc_values_output(
        docs: &[Document],
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
    /// builds [`doc_values::write_single_dense_sorted_set_field`]'s input
    /// from `docs`' values for `config.field_number` -- unlike
    /// [`Self::build_sorted_doc_values_output`] (exactly one value per doc),
    /// a doc's value-set here is *every* [`lucene_codecs::stored_fields::StoredField`] entry in that doc
    /// carrying `config.field_number`, so a doc opts into multiple values by
    /// simply repeating the field (real Lucene's own multi-`add`-calls-per-
    /// doc convention for `SortedSetDocValuesField`). Each such value must be
    /// [`FieldValue::String`] (UTF-8 bytes) or [`FieldValue::Binary`] (raw
    /// bytes), else [`Error::NonBinaryDocValue`]; a doc with zero matching
    /// fields fails the whole commit with [`Error::MissingDenseDocValue`] --
    /// same dense-only, "missing value fails the whole commit" contract as
    /// [`Self::build_sorted_doc_values_output`].
    fn build_sorted_set_doc_values_output(
        docs: &[Document],
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
    /// entirely**, and [`Self::fields_with_per_field_attributes`] then zeroes
    /// its `.fnm` `vector_dimension`. Java reaches the same state from the
    /// other side: `IndexingChain` only creates a field's
    /// `KnnFieldVectorsWriter` (and only sets `FieldInfo.vectorDimension`)
    /// when a document actually carries the field, so a segment's `.fnm`
    /// never claims vectors its files do not hold.
    ///
    /// Returns `None` when no field has any vector, so no files are written.
    fn build_vectors_output(
        pending_vectors: &[Vec<DocumentVector>],
        configs: &[VectorFieldConfig],
        max_doc: i32,
        m: i32,
        beam_width: i32,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<Option<VectorsOutput>> {
        debug_assert_eq!(
            pending_vectors.len() as i32,
            max_doc,
            "pending_vectors must stay aligned 1:1 with pending_docs"
        );
        let suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);

        let mut flat_fields: Vec<FlatVectorsField> = Vec::new();
        let mut written_fields: Vec<String> = Vec::new();
        for config in configs {
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
                VectorEncoding::Float32 => {
                    debug_assert!(bytes.is_empty());
                    FieldVectorData::Float32(floats)
                }
                VectorEncoding::Byte => {
                    debug_assert!(floats.is_empty());
                    FieldVectorData::Byte(bytes)
                }
            };
            written_fields.push(config.name.clone());
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

        let (vec_bytes, vemf_bytes) =
            vectors::write_flat_vectors(&flat_fields, max_doc, segment_id, &suffix)?;

        // Reopen exactly the bytes just written, and build every graph over
        // *those* -- see this method's doc comment.
        let graphs: Vec<Option<hnsw::OnHeapHnswGraph>> = {
            let flat =
                vectors::FlatVectorsReader::open(&vemf_bytes, &vec_bytes, segment_id, &suffix)?;
            let mut graphs = Vec::with_capacity(flat_fields.len());
            for field in &flat_fields {
                let count = field.docs.len() as i32;
                // `Lucene99HnswVectorsWriter.shouldCreateGraph`: a segment too
                // small for a graph to beat an exhaustive scan gets none, and
                // its `.vem` records `numLevels = 0`.
                if !hnsw::should_create_graph(hnsw::HNSW_GRAPH_THRESHOLD, count) {
                    graphs.push(None);
                    continue;
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
                graphs.push(Some(graph));
            }
            graphs
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
            hnsw_vectors::write_hnsw_vectors(&hnsw_fields, segment_id, &suffix)?;

        Ok(Some(VectorsOutput {
            vec: vec_bytes,
            vemf: vemf_bytes,
            vex: vex_bytes,
            vem: vem_bytes,
            written_fields,
        }))
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
        let seg = per_field_segment(segment_name, KNN_VECTORS_FORMAT_NAME);
        let names = vec![
            format!("{seg}.vec"),
            format!("{seg}.vemf"),
            format!("{seg}.vex"),
            format!("{seg}.vem"),
        ];
        let bytes = [&output.vec, &output.vemf, &output.vex, &output.vem];
        for (name, data) in names.iter().zip(bytes) {
            write_file(dir, name, data)?;
        }
        Ok(names)
    }

    /// Writes [`IndexWriter::build_doc_values_output`]'s three files
    /// (`<segment_name>.dvm`/`.dvd`/`.dvs`) into `dir` and returns their
    /// names for the segment's still-unwritten `.si`.
    fn write_doc_values_files(
        dir: &dyn Directory,
        segment_name: &str,
        dvm: &[u8],
        dvd: &[u8],
        dvs: &[u8],
    ) -> Result<Vec<String>> {
        let seg = per_field_segment(segment_name, DOC_VALUES_FORMAT_NAME);
        let names = vec![
            format!("{seg}.dvm"),
            format!("{seg}.dvd"),
            format!("{seg}.dvs"),
        ];
        for (name, bytes) in names.iter().zip([dvm, dvd, dvs]) {
            write_file(dir, name, bytes)?;
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
    /// (`<segment_name>.tvd`/`.tvx`/`.tvm`) into `dir` and returns their
    /// names for the segment's still-unwritten `.si`.
    fn write_term_vector_files(
        dir: &dyn Directory,
        segment_name: &str,
        tvd: &[u8],
        tvx: &[u8],
        tvm: &[u8],
    ) -> Result<Vec<String>> {
        let names = vec![
            format!("{segment_name}.tvd"),
            format!("{segment_name}.tvx"),
            format!("{segment_name}.tvm"),
        ];
        for (name, bytes) in names.iter().zip([tvd, tvx, tvm]) {
            write_file(dir, name, bytes)?;
        }
        Ok(names)
    }

    /// Stamp the `PerField*Format` attributes real Lucene's codec writes at
    /// flush time onto the fields this commit actually produced files for, so
    /// the `.fnm` points at the suffixed names
    /// [`Self::write_postings_files`]/[`Self::write_doc_values_files`] used.
    /// Lucene does this inside `PerFieldPostingsFormat.fieldsConsumer`, which
    /// mutates the `FieldInfo` it is handed; this port's `.fnm` writer takes
    /// its fields by shared reference, so the decorated list is built here.
    ///
    /// Only fields that really got postings (indexed) or doc values are
    /// stamped: an attribute naming a format whose files were never written
    /// would send a reader looking for a file that does not exist.
    fn fields_with_per_field_attributes(
        &self,
        wrote_postings: bool,
        wrote_doc_values: bool,
        wrote_norms: bool,
        vector_fields_written: &[String],
    ) -> Vec<FieldInfo> {
        let postings_names: Vec<&str> = if wrote_postings {
            self.postings_fields
                .iter()
                .map(|c| c.name.as_str())
                .collect()
        } else {
            Vec::new()
        };
        let dv_names: Vec<&str> = if wrote_doc_values {
            self.doc_values_fields
                .iter()
                .map(|c| c.name.as_str())
                .collect()
        } else {
            Vec::new()
        };

        self.fields
            .iter()
            .map(|f| {
                let mut f = f.clone();
                // No norms coercion here any more. Until c35 this writer's
                // norms were opt-in per field, so every indexed field the
                // caller had not named had to be rewritten as
                // `omit_norms: true` -- a `.fnm` describing a different
                // schema than the caller asked for, because
                // `DirectoryReader.open` throws on the missing `.nvm` rather
                // than degrading when the `.fnm` claims norms the segment
                // does not carry. `norms_field_configs` now writes a column
                // for exactly the fields whose `.fnm` claims one, so the two
                // cannot disagree and there is nothing to coerce.
                debug_assert!(
                    !wrote_norms
                        || f.omit_norms
                        || f.index_options == IndexOptions::None
                        || self.norms_field_configs().iter().any(|c| c.name == f.name),
                    "every indexed non-omitNorms field must have a norm column"
                );
                if postings_names.contains(&f.name.as_str()) {
                    f.attributes.push((
                        "PerFieldPostingsFormat.format".to_string(),
                        POSTINGS_FORMAT_NAME.to_string(),
                    ));
                    f.attributes.push((
                        "PerFieldPostingsFormat.suffix".to_string(),
                        PER_FIELD_SUFFIX.to_string(),
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
                if !dv_names.contains(&f.name.as_str()) {
                    f.doc_values_type = DocValuesType::None;
                    f.doc_values_skip_index_type =
                        lucene_codecs::field_infos::DocValuesSkipIndexType::None;
                }
                if vector_fields_written.iter().any(|n| n == &f.name) {
                    f.attributes.push((
                        "PerFieldKnnVectorsFormat.format".to_string(),
                        KNN_VECTORS_FORMAT_NAME.to_string(),
                    ));
                    f.attributes.push((
                        "PerFieldKnnVectorsFormat.suffix".to_string(),
                        PER_FIELD_SUFFIX.to_string(),
                    ));
                } else {
                    f.vector_dimension = 0;
                }
                if dv_names.contains(&f.name.as_str()) {
                    f.attributes.push((
                        "PerFieldDocValuesFormat.format".to_string(),
                        DOC_VALUES_FORMAT_NAME.to_string(),
                    ));
                    f.attributes.push((
                        "PerFieldDocValuesFormat.suffix".to_string(),
                        PER_FIELD_SUFFIX.to_string(),
                    ));
                }
                f
            })
            .collect()
    }

    /// Writes [`IndexWriter::build_postings_output`]'s files
    /// (`<segment_name>.doc`/`.psm`/`.tim`/`.tip`/`.tmd`, plus `.pos`/`.pay`
    /// when the fields index them) into `dir` and returns their names for the
    /// segment's still-unwritten `.si` -- see
    /// [`crate::segment_writer::FlushedSegment`].
    fn write_postings_files(
        dir: &dyn Directory,
        segment_name: &str,
        output: &postings_writer::Output,
    ) -> Result<Vec<String>> {
        let seg = per_field_segment(segment_name, POSTINGS_FORMAT_NAME);
        let doc_name = format!("{seg}.doc");
        let tim_name = format!("{seg}.tim");
        let tip_name = format!("{seg}.tip");
        let tmd_name = format!("{seg}.tmd");
        let pos_name = format!("{seg}.pos");
        let pay_name = format!("{seg}.pay");

        let psm_name = format!("{seg}.psm");
        let mut written_names = vec![
            doc_name.clone(),
            psm_name.clone(),
            tim_name.clone(),
            tip_name.clone(),
            tmd_name.clone(),
        ];
        let mut written_bytes: Vec<(&String, &Vec<u8>)> = vec![
            (&doc_name, &output.doc),
            (&psm_name, &output.psm),
            (&tim_name, &output.tim),
            (&tip_name, &output.tip),
            (&tmd_name, &output.tmd),
        ];
        // `.pos`/`.pay` only exist when at least one field in this call
        // indexes positions/offsets -- see `postings_writer::Output`'s doc
        // comment ("`pos` is empty when `index_options` doesn't index
        // positions ... no `.pos` file is needed in that case"). Registering
        // an empty, never-written `.pos`/`.pay` name in `si.files` for a
        // Docs/DocsAndFreqs-only commit would leave a dangling filename no
        // reader could ever open.
        if !output.pos.is_empty() {
            written_names.push(pos_name.clone());
            written_bytes.push((&pos_name, &output.pos));
        }
        if !output.pay.is_empty() {
            written_names.push(pay_name.clone());
            written_bytes.push((&pay_name, &output.pay));
        }

        for (name, bytes) in &written_bytes {
            write_file(dir, name, bytes)?;
        }
        Ok(written_names)
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
        let config = self
            .merge_policy
            .clone()
            .expect("auto_merge only called when merge_policy is Some");

        loop {
            let stats = self.segment_stats()?;
            let groups = merge_policy::find_merges(&stats, &config);
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
    /// current committed segments: `doc_count`/on-disk size come from each
    /// segment's own `.si` file (via [`crate::segment_info::parse`] and
    /// [`crate::merge_policy::segment_byte_size`], the byte-accurate path
    /// that module's doc comment describes), `del_count` from this writer's
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
    ///   is mergeable too, as of c26. Its newest column lives in generational
    ///   `.dvm`/`.dvd` files no `.si` lists, and until c26 `execute_merge`
    ///   read the base pair -- so merging one would have silently resurrected
    ///   the pre-update values. `execute_merge` now resolves every field to
    ///   its **current** generation through
    ///   [`crate::field_updates::read_current_field_infos`] and
    ///   [`crate::field_updates::read_current_column`], the same two
    ///   functions the update path itself reads its base from, and the merged
    ///   segment folds every generation back into one base column
    ///   (`doc_values_gen == -1`), which is what
    ///   `IndexWriter.mergeMiddle` produces.
    ///
    /// **Nothing is withheld any more.** That is not a claim about this
    /// method: it is what [`crate::merge::check_format_coverage`] enforces on
    /// every merge, from the source segments' own file lists. Before c26 the
    /// burden of "`execute_merge` opens every format the flush can write" was
    /// carried by reading, and four separate formats had already been dropped
    /// silently (c22 findings 14, 22, 23, 24).
    ///
    /// The bar for adding a format here is that losing it must be
    /// impossible, not unlikely: a merged segment that quietly dropped a
    /// format would still be valid, still pass `CheckIndex`, and differ from
    /// the truth only in what it answers.
    fn segment_stats(&self) -> Result<Vec<merge_policy::SegmentStat>> {
        let mut stats = Vec::with_capacity(self.segment_infos.segments.len());
        for sci in &self.segment_infos.segments {
            let si_bytes = self.dir.open(&format!("{}.si", sci.segment_name))?.to_vec();
            let si = segment_info::parse(&si_bytes, &sci.segment_id)?;
            let size_bytes = merge_policy::segment_byte_size(self.dir, &si);
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
    /// own `self.fields` schema marks as postings-eligible *and* that
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
    /// `self.fields` schema (field numbers are never reassigned per segment),
    /// every source's term-vector field numbers already agree with the
    /// merged field numbers -- the field-number remap
    /// `crate::merge::write_merged_term_vectors` applies is the identity mapping
    /// in practice here, same as it is for postings above.
    ///
    /// **Points are not wired here.** [`crate::merge::SourcePoints`] exists
    /// and [`crate::merge::merge_stored_only_segments`] already merges it
    /// when populated (see that function's/[`crate::merge::merge_points`]'s
    /// doc comments), but this writer has no points write path at flush time
    /// at all yet (no `set_points_field`-equivalent, no points-carrying
    /// `Document` field shape) -- there is currently no real segment this
    /// writer could ever produce with `.kdm`/`.kdi`/`.kdd` files, so there is
    /// nothing for this method to open. Wiring points end-to-end needs
    /// flush-side points support added first; until then, `merge_points` is
    /// exercised only by `merge.rs`'s own hand-built `MergeSource` fixtures.
    /// See `docs/parity.md`.
    fn execute_merge(&mut self, names: &[String]) -> Result<()> {
        // A merged segment is published with `buffered_deletes_gen == -1`,
        // i.e. open to every packet, where Java uses `min(sources)`
        // (`IndexWriter.mergeMiddle`). That is only safe while the stream is
        // empty at merge time -- which it is, because `flush()` applies and
        // takes every packet before `finish_commit` runs `auto_merge`. Assert
        // it rather than rely on the reading: if a packet ever outlives the
        // call that pushed it, this is the line that says so.
        debug_assert!(
            !self.updates_stream.any(),
            "a delete packet outlived its flush; a merged segment would take it"
        );
        // Claimed up front: every later borrow in this method is an immutable
        // one over `self`, and handing out a segment name mutates the counter.
        let merged_segment_name = self.new_segment_name();
        let merged_segment_id = generate_segment_id(self.segment_infos.counter);
        /// Raw `.tim`/`.tip`/`.tmd`/`.doc` bytes for a source that has
        /// postings, plus its `.pos`/`.pay` when the segment has them --
        /// `None` when that source's `.si` lists no `.tim` file.
        ///
        /// A positional field always has a `.pos`; `.pay` exists only when
        /// some term's `total_term_freq` spans a full 256-position block (see
        /// `postings::read_positions`), so it is separately optional.
        type RawPostingsFiles = Option<RawPostings>;
        struct RawPostings {
            tim: Vec<u8>,
            tip: Vec<u8>,
            tmd: Vec<u8>,
            doc: Vec<u8>,
            pos: Option<Vec<u8>>,
            pay: Option<Vec<u8>>,
        }
        /// Raw `.tvd`/`.tvx`/`.tvm` bytes for a source that has term vectors
        /// -- `None` when that source's `.si` lists no `.tvd` file.
        type RawTermVectorFiles = Option<(Vec<u8>, Vec<u8>, Vec<u8>)>;
        /// Raw `.nvm`/`.nvd` bytes for a source that has norms.
        type RawNormsFiles = Option<(Vec<u8>, Vec<u8>)>;
        /// Raw `.vec`/`.vemf` and, when the segment has a graph,
        /// `.vem`/`.vex` bytes.
        type RawVectorFiles = Option<(Vec<u8>, Vec<u8>, Option<(Vec<u8>, Vec<u8>)>)>;

        struct OpenedSegment {
            sci: SegmentCommitInfo,
            fdt: Vec<u8>,
            fdx: Vec<u8>,
            fdm: Vec<u8>,
            live_docs: Option<FixedBitSet>,
            postings: RawPostingsFiles,
            term_vectors: RawTermVectorFiles,
            doc_values: SourceDocValueColumns,
            norms: RawNormsFiles,
            vectors: RawVectorFiles,
            index_sort: Option<Vec<segment_info::IndexSortField>>,
            has_blocks: bool,
            /// This source's `SegmentInfo.minVersion` -- Java's
            /// `LeafMetaData.minVersion()`, which `SegmentMerger` folds into
            /// the merged segment's.
            min_version: Option<LuceneVersion>,
            /// This source's own `SegmentInfo::files`, for
            /// [`merge::check_format_coverage`] -- the segment's own claim
            /// about which formats it has, which is the same set
            /// `IndexFileDeleter` reference-counts.
            files: Vec<String>,
        }

        let mut opened = Vec::with_capacity(names.len());
        for name in names {
            let sci = self
                .segment_infos
                .segments
                .iter()
                .find(|s| &s.segment_name == name)
                .expect("merge_policy::find_merges only proposes segment names this writer currently has committed")
                .clone();

            let fdt = self.dir.open(&format!("{name}.fdt"))?.to_vec();
            let fdx = self.dir.open(&format!("{name}.fdx"))?.to_vec();
            let fdm = self.dir.open(&format!("{name}.fdm"))?.to_vec();

            // The `.si` is opened *before* anything is sized by a document
            // count, because it is the authority on that count -- Java's
            // `SegmentMerger` works from `SegmentReader.maxDoc()`, which is
            // `SegmentInfo.maxDoc()`, never from the stored-fields file.
            let si_bytes = self.dir.open(&format!("{name}.si"))?.to_vec();
            let si = segment_info::parse(&si_bytes, &sci.segment_id)?;

            // `stored_fields::open` checks only that its own `maxDoc` is
            // non-negative, and `merge_segments` sizes this source's live-id
            // list and doc-id map from it. Left uncrossed-checked, a `.fdm`
            // claiming `i32::MAX` turns a merge into two ~8.6 GB allocations
            // -- an abort, and `open_segment_for_deletes` two thousand lines
            // below already takes the count from the `.si` for exactly this
            // reason. Disagreement is corruption whichever file is right.
            let stored_fields_max_doc =
                stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "")?.max_doc();
            if stored_fields_max_doc != si.doc_count {
                return Err(Error::SegmentDocCountMismatch {
                    segment: name.clone(),
                    si_doc_count: si.doc_count,
                    stored_fields_max_doc,
                });
            }

            let live_docs = if sci.del_gen >= 0 {
                let liv = self.dir.open(&deletes::liv_file_name(name, sci.del_gen))?;
                Some(lucene_codecs::live_docs::parse(
                    &liv,
                    &sci.segment_id,
                    sci.del_gen,
                    // `usize::try_from` cannot fail: `si.doc_count` is
                    // non-negative (`segment_info::parse` rejects otherwise).
                    usize::try_from(si.doc_count).unwrap_or(0),
                    sci.del_count as usize,
                )?)
            } else {
                None
            };
            let postings = if si.files.iter().any(|f| f.ends_with(".tim")) {
                let seg = per_field_segment(name, POSTINGS_FORMAT_NAME);
                let read_optional = |ext: &str| -> Result<Option<Vec<u8>>> {
                    let file = format!("{seg}.{ext}");
                    Ok(if si.files.contains(&file) {
                        Some(self.dir.open(&file)?.to_vec())
                    } else {
                        None
                    })
                };
                Some(RawPostings {
                    tim: self.dir.open(&format!("{seg}.tim"))?.to_vec(),
                    tip: self.dir.open(&format!("{seg}.tip"))?.to_vec(),
                    tmd: self.dir.open(&format!("{seg}.tmd"))?.to_vec(),
                    doc: self.dir.open(&format!("{seg}.doc"))?.to_vec(),
                    pos: read_optional("pos")?,
                    pay: read_optional("pay")?,
                })
            } else {
                None
            };

            let term_vectors = if si.files.iter().any(|f| f.ends_with(".tvd")) {
                let tvd = self.dir.open(&format!("{name}.tvd"))?.to_vec();
                let tvx = self.dir.open(&format!("{name}.tvx"))?.to_vec();
                let tvm = self.dir.open(&format!("{name}.tvm"))?.to_vec();
                Some((tvd, tvx, tvm))
            } else {
                None
            };

            // `IndexWriter.readFieldInfos(SegmentCommitInfo)`: the
            // generational `.fnm` when this segment has had a doc-values
            // update, the base one otherwise. Only the *location* of each
            // column is taken from it -- the merged schema stays this
            // writer's own `self.fields`, whose `doc_values_gen` is `-1`,
            // because the merge folds every generation back into one base
            // column.
            let current_infos =
                crate::field_updates::read_current_field_infos(self.dir, &sci, &si.files)?;
            let mut columns: Vec<(doc_values::DocValuesMeta, Vec<u8>)> = Vec::new();
            let mut per_field: Vec<(i32, usize)> = Vec::new();
            // Distinct column locations, so a segment's base pair is read
            // once rather than once per field that lives in it.
            let mut seen: Vec<(i64, String, usize)> = Vec::new();
            for (index, field) in current_infos.fields.iter().enumerate() {
                if field.doc_values_type == DocValuesType::None {
                    continue;
                }
                let per_field_component = crate::field_updates::per_field_component(
                    field,
                    &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
                );
                let key = (field.doc_values_gen, per_field_component.clone());
                if let Some(&(_, _, at)) = seen
                    .iter()
                    .find(|(gen, comp, _)| *gen == key.0 && *comp == key.1)
                {
                    per_field.push((field.number, at));
                    continue;
                }
                let Some((meta, data)) = crate::field_updates::read_current_column(
                    self.dir,
                    &sci,
                    &si.files,
                    &current_infos,
                    index,
                    &per_field_component,
                )?
                else {
                    // The `.fnm` declares a type but the segment never wrote
                    // a column: legitimate, and `merge_segments` writes an
                    // all-missing merged column for it the same way
                    // `Lucene90DocValuesConsumer.writeValues` does.
                    continue;
                };
                columns.push((meta, data));
                let at = columns.len().saturating_sub(1);
                seen.push((key.0, key.1, at));
                per_field.push((field.number, at));
            }
            let doc_values = SourceDocValueColumns { columns, per_field };

            let norms = if si.files.iter().any(|f| f.ends_with(".nvd")) {
                let nvm = self.dir.open(&format!("{name}.nvm"))?.to_vec();
                let nvd = self.dir.open(&format!("{name}.nvd"))?.to_vec();
                Some((nvm, nvd))
            } else {
                None
            };

            let vectors = if si.files.iter().any(|f| f.ends_with(".vec")) {
                let seg = per_field_segment(name, KNN_VECTORS_FORMAT_NAME);
                let vec_bytes = self.dir.open(&format!("{seg}.vec"))?.to_vec();
                let vemf = self.dir.open(&format!("{seg}.vemf"))?.to_vec();
                // A segment can legitimately have the flat pair and no graph
                // files at all if it was written below
                // `HNSW_GRAPH_THRESHOLD`; this writer always writes the
                // `.vem`/`.vex` pair (with `numLevels = 0` in that case), so
                // the absence is tolerated rather than assumed.
                let graph = if si.files.iter().any(|f| f.ends_with(".vem")) {
                    Some((
                        self.dir.open(&format!("{seg}.vem"))?.to_vec(),
                        self.dir.open(&format!("{seg}.vex"))?.to_vec(),
                    ))
                } else {
                    None
                };
                Some((vec_bytes, vemf, graph))
            } else {
                None
            };

            // Computed before the push, because the struct literal moves
            // `sci` in its first field.
            let all_files = sci.files(&si.files);
            opened.push(OpenedSegment {
                sci,
                fdt,
                fdx,
                fdm,
                live_docs,
                postings,
                term_vectors,
                doc_values,
                norms,
                vectors,
                index_sort: si.index_sort.clone(),
                has_blocks: si.has_blocks,
                min_version: si.min_version,
                // `SegmentCommitInfo::files`, not `si.files`: a generational
                // `.dvm`/`.dvd` is never listed in the `.si` (it did not
                // exist when the `.si` was written), so a gate reading the
                // `.si` alone would be blind to exactly the files an update
                // round produced.
                files: all_files,
            });
        }

        // `IndexWriter.validateIndexSort`: every source must agree, and a
        // merge of segments that disagree is refused rather than silently
        // producing an unsorted segment (or one whose `.si` lies).
        let merge_sort = opened.first().and_then(|o| o.index_sort.clone());
        if let Some(bad) = opened
            .iter()
            .find(|o| o.index_sort.as_deref() != merge_sort.as_deref())
        {
            return Err(Error::MergeSortDisagreement {
                expected: segment_info::describe_index_sort(merge_sort.as_deref()),
                found: segment_info::describe_index_sort(bad.index_sort.as_deref()),
                segment: bad.sci.segment_name.clone(),
            });
        }

        let readers: Vec<_> = opened
            .iter()
            .map(|o| stored_fields::open(&o.fdt, &o.fdx, &o.fdm, &o.sci.segment_id, ""))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // `IndexWriter.mergeMiddle`: `if (merger.shouldMerge()) merger.merge();`
        // -- `SegmentMerger.shouldMerge()` is `segmentInfo.maxDoc() > 0`, and
        // that `maxDoc` is the sum of the readers' *live* document counts.
        // When it is zero, Java writes **nothing at all** and goes straight to
        // `commitMerge`, whose `allDeleted` branch drops the merged segment and
        // still removes every source from the commit ("Merge would produce a
        // 0-doc segment, so we do nothing except commit the merge to remove all
        // the 0-doc segments that we merged").
        //
        // This port used to run the merge and publish the empty result: a real
        // zero-document segment that every later open, merge and `CheckIndex`
        // then pays for, and a `.si`/`.fnm`/`.fdt` set nothing will ever read.
        // Placed exactly where Java's is: after the readers exist and before
        // the merge writes anything. It is deliberately *not* hoisted above
        // the `opened` loop, even though each source's live count is
        // `si.doc_count - sci.del_count` and both are known there. Hoisting it
        // would skip the loop's cross-checks -- the `.fdm`-against-`.si`
        // `maxDoc` agreement, the `.liv` parse, `validate_index_sort` and
        // `check_format_coverage` -- so a corrupt or unmergeable source would
        // be *silently dropped from the commit* instead of reported. Refusing
        // to look before deciding to throw the sources away is the wrong
        // trade; the cost is reading files for a merge that will not happen,
        // which only occurs when every source is fully deleted.
        let live_doc_count: usize = opened
            .iter()
            .zip(readers.iter())
            .map(|(o, reader)| match &o.live_docs {
                Some(bits) => bits.cardinality(),
                None => usize::try_from(reader.max_doc()).unwrap_or(0),
            })
            .sum();
        if live_doc_count == 0 {
            let source_names: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            self.drop_merge(&source_names)?;
            return Ok(());
        }

        // Fields this writer's fixed schema marks as postings-eligible --
        // exactly the `IndexOptions` `set_postings_field`/`add_postings_field`
        // accept at write time, **including the positional ones**. Narrowing
        // this to `Docs`/`DocsAndFreqs` would drop a positional field's
        // postings from every merged segment while `describe_written_files`
        // left its `index_options` claiming them: an indexed field with no
        // registered postings producer, which reads back as having no terms.
        // `merge_postings` handles positions, offsets and payloads.
        let postings_field_infos = lucene_codecs::field_infos::FieldInfos {
            fields: self
                .fields
                .iter()
                .filter(|f| {
                    matches!(
                        f.index_options,
                        IndexOptions::Docs
                            | IndexOptions::DocsAndFreqs
                            | IndexOptions::DocsAndFreqsAndPositions
                            | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
                    )
                })
                .cloned()
                .collect(),
        };

        type OpenedPostings<'a> = Option<(
            lucene_codecs::blocktree::BlockTreeFields,
            lucene_codecs::postings::DocInput<'a>,
            Option<lucene_codecs::postings::PosInput<'a>>,
            Option<lucene_codecs::postings::PayInput<'a>>,
        )>;
        let opened_postings: Vec<OpenedPostings> = opened
            .iter()
            .zip(readers.iter())
            .map(|(o, reader)| match &o.postings {
                Some(raw) => {
                    let fields = lucene_codecs::blocktree::open(
                        &raw.tim,
                        &raw.tip,
                        &raw.tmd,
                        &postings_field_infos,
                        &o.sci.segment_id,
                        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
                        reader.max_doc(),
                    )?;
                    let doc_in = lucene_codecs::postings::DocInput::open(
                        &raw.doc,
                        &o.sci.segment_id,
                        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
                    )?;
                    let pos_in = raw
                        .pos
                        .as_ref()
                        .map(|pos| {
                            lucene_codecs::postings::PosInput::open(
                                pos,
                                &o.sci.segment_id,
                                &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
                            )
                        })
                        .transpose()?;
                    let pay_in = raw
                        .pay
                        .as_ref()
                        .map(|pay| {
                            lucene_codecs::postings::PayInput::open(
                                pay,
                                &o.sci.segment_id,
                                &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
                            )
                        })
                        .transpose()?;
                    Ok::<_, Error>(Some((fields, doc_in, pos_in, pay_in)))
                }
                None => Ok(None),
            })
            .collect::<std::result::Result<Vec<_>, Error>>()?;

        // One `Vec<SourcePostings>` per source, holding every
        // postings-eligible field that source's own term dictionary
        // actually has an entry for.
        let per_source_postings: Vec<Vec<merge::SourcePostings>> = opened_postings
            .iter()
            .map(|maybe| match maybe {
                Some((fields, doc_in, pos_in, pay_in)) => postings_field_infos
                    .fields
                    .iter()
                    .filter_map(|f| {
                        fields
                            .field(&f.name)
                            .map(|field_terms| merge::SourcePostings {
                                field_number: f.number,
                                field_terms,
                                doc_in: Some(doc_in),
                                pos_in: pos_in.as_ref(),
                                pay_in: pay_in.as_ref(),
                            })
                    })
                    .collect(),
                None => Vec::new(),
            })
            .collect();

        let opened_term_vectors: Vec<Option<lucene_codecs::term_vectors::TermVectorsReader>> =
            opened
                .iter()
                .map(|o| match &o.term_vectors {
                    Some((tvd, tvx, tvm)) => Ok::<_, Error>(Some(
                        lucene_codecs::term_vectors::open(tvd, tvx, tvm, &o.sci.segment_id, "")?,
                    )),
                    None => Ok(None),
                })
                .collect::<std::result::Result<Vec<_>, Error>>()?;

        // Doc values. Every source flushed by this writer shares
        // `self.fields`, so a source's `.dvm` entry for a field already
        // carries the merged field number.
        // **Every** doc-values type, not just NUMERIC. `segment_stats` now
        // offers any `.dvd`-bearing segment to the merge policy, and this
        // writer can flush all five types (`collect_dense_column`), so a type
        // opened here and not merged would be silently dropped: the merge
        // writes no column, `merge::describe_written_files` then zeroes the
        // field's `DocValuesType` to keep the `.fnm` honest, and what is left
        // is a valid, `CheckIndex`-clean segment with the data gone.
        // One list per type per source. Each entry names a field and the
        // column its *current* generation lives in, so a source with a
        // doc-values update contributes its updated values rather than the
        // superseded base ones.
        macro_rules! per_source_dv {
            ($ty:ident, $bucket:ident) => {
                opened
                    .iter()
                    .map(|o| {
                        o.doc_values
                            .per_field
                            .iter()
                            .filter_map(|&(field_number, at)| {
                                let (meta, data) = &o.doc_values.columns[at];
                                meta.$bucket
                                    .iter()
                                    .find(|e| e.field_number == field_number)
                                    .map(|entry| merge::$ty {
                                        data: data.as_slice(),
                                        entry: entry.clone(),
                                    })
                            })
                            .collect::<Vec<merge::$ty>>()
                    })
                    .collect::<Vec<_>>()
            };
        }
        let per_source_numeric_dv: Vec<Vec<merge::SourceNumericDocValues>> =
            per_source_dv!(SourceNumericDocValues, numeric);
        let per_source_binary_dv: Vec<Vec<merge::SourceBinaryDocValues>> =
            per_source_dv!(SourceBinaryDocValues, binary);
        let per_source_sorted_dv: Vec<Vec<merge::SourceSortedDocValues>> =
            per_source_dv!(SourceSortedDocValues, sorted);
        let per_source_sorted_numeric_dv: Vec<Vec<merge::SourceSortedNumericDocValues>> =
            per_source_dv!(SourceSortedNumericDocValues, sorted_numeric);
        let per_source_sorted_set_dv: Vec<Vec<merge::SourceSortedSetDocValues>> =
            per_source_dv!(SourceSortedSetDocValues, sorted_set);
        // Stated as an assertion rather than left to review: the five lists
        // above are built by one macro over five named `DocValuesMeta`
        // buckets, and a sixth doc-values type would need a sixth line that
        // nothing else would miss.
        debug_assert!(
            opened.iter().all(|o| {
                let total: usize = o
                    .doc_values
                    .columns
                    .iter()
                    .map(|(m, _)| {
                        // ARITH: five in-memory `Vec` lengths; their sum is
                        // bounded by the entries this process holds.
                        #[allow(clippy::arithmetic_side_effects)]
                        {
                            m.numeric.len()
                                + m.binary.len()
                                + m.sorted.len()
                                + m.sorted_numeric.len()
                                + m.sorted_set.len()
                        }
                    })
                    .sum();
                // Every entry the opened columns declare is either reached by
                // one of the five lists or belongs to a field this segment's
                // current schema no longer points at that column for (a
                // superseded base entry for an updated field).
                total >= o.doc_values.per_field.len()
            }),
            "every doc-values entry a source's .dvm declares must reach the merge"
        );

        let opened_norms: Vec<Option<norms::Norms>> = opened
            .iter()
            .map(|o| match &o.norms {
                Some((nvm, _)) => {
                    let (_v, meta) = norms::parse_meta(nvm, &o.sci.segment_id, "")?;
                    Ok::<_, Error>(Some(meta))
                }
                None => Ok(None),
            })
            .collect::<std::result::Result<Vec<_>, Error>>()?;
        let per_source_norms: Vec<Vec<merge::SourceNorms>> = opened
            .iter()
            .zip(&opened_norms)
            .map(|(o, meta)| match (&o.norms, meta) {
                (Some((_, nvd)), Some(meta)) => meta
                    .entries
                    .iter()
                    .map(|entry| merge::SourceNorms {
                        data: nvd,
                        entry: *entry,
                    })
                    .collect(),
                _ => Vec::new(),
            })
            .collect();

        // Vectors. The flat store and the graph are opened separately
        // because a segment may legitimately have the first and not the
        // second.
        let opened_flat_vectors: Vec<Option<vectors::FlatVectorsReader>> = opened
            .iter()
            .map(|o| match &o.vectors {
                Some((vec_bytes, vemf, _)) => {
                    Ok::<_, Error>(Some(vectors::FlatVectorsReader::open(
                        vemf,
                        vec_bytes,
                        &o.sci.segment_id,
                        &per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME),
                    )?))
                }
                None => Ok(None),
            })
            .collect::<std::result::Result<Vec<_>, Error>>()?;
        let opened_vector_graphs: Vec<Option<hnsw_vectors::HnswVectorsReader>> = opened
            .iter()
            .map(|o| match &o.vectors {
                Some((_, _, Some((vem, vex)))) => {
                    Ok::<_, Error>(Some(hnsw_vectors::HnswVectorsReader::open(
                        vem,
                        vex,
                        &o.sci.segment_id,
                        &per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME),
                    )?))
                }
                _ => Ok(None),
            })
            .collect::<std::result::Result<Vec<_>, Error>>()?;
        let per_source_vectors: Vec<Option<merge::SourceVectors>> = opened_flat_vectors
            .iter()
            .zip(&opened_vector_graphs)
            .map(|(flat, graph)| {
                flat.as_ref().map(|flat| merge::SourceVectors {
                    flat,
                    graph: graph.as_ref(),
                })
            })
            .collect();

        // Indexed rather than zipped: eight parallel `Vec`s in one `zip`
        // chain is a tuple nobody can read, and the index is the source's own
        // identity anyway.
        let sources: Vec<merge::MergeSource> = (0..opened.len())
            .map(|i| merge::MergeSource {
                field_infos: &self.fields,
                reader: &readers[i],
                live_docs: opened[i].live_docs.as_ref(),
                numeric_doc_values: &per_source_numeric_dv[i],
                binary_doc_values: &per_source_binary_dv[i],
                sorted_doc_values: &per_source_sorted_dv[i],
                sorted_numeric_doc_values: &per_source_sorted_numeric_dv[i],
                sorted_set_doc_values: &per_source_sorted_set_dv[i],
                norms: &per_source_norms[i],
                term_vectors: opened_term_vectors[i].as_ref(),
                postings: &per_source_postings[i],
                points: &[],
                vectors: per_source_vectors[i].as_ref(),
                // Both read straight off this source's own `.si`, exactly as
                // `SegmentReader.getMetaData()` builds its `LeafMetaData`.
                min_version: opened[i].min_version,
                has_blocks: opened[i].has_blocks,
            })
            .collect();

        // **The merge-completeness gate.** Every format a source's own `.si`
        // lists files for must be a format the code above actually opened
        // onto that source's `MergeSource` -- and every extension in that
        // `.si` must be one some `merge::SegmentFormat` claims.
        //
        // This is the mechanism c22's Tier-2 review asked for after tracing
        // four of its correctness findings (14: norms, silently wrong BM25
        // scores since c4; 22: every doc-values type but NUMERIC; 23:
        // positional postings; 24: `has_blocks`) to one omission: a format
        // this method forgets to open is a format the merge drops, and the
        // merged segment is well-formed, checksummed and `CheckIndex`-clean
        // either way. Refusing the merge is strictly better than performing
        // it, because the loss is otherwise unobservable.
        //
        // See `merge::check_format_coverage` for why a *new* format cannot
        // slip past it: an unclaimed extension is an error, and satisfying
        // that error means adding a `SegmentFormat` variant whose two
        // exhaustive `match`es then force this method to open it.
        let source_names: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let source_files: Vec<&[String]> = opened.iter().map(|o| o.files.as_slice()).collect();
        merge::check_format_coverage(&source_names, &source_files, &sources)?;

        // `MultiSorter.sort`'s `IndexSorter.getComparableValues`: each sort
        // tier's key for every document of every source, read out of the very
        // NUMERIC column the merged segment will carry -- so the order the
        // merge imposes and the column `CheckIndex.testSort` re-derives it
        // from are one fact, exactly as at flush time.
        let per_tier_keys: Vec<Vec<Vec<Option<i64>>>> = match &merge_sort {
            None => Vec::new(),
            Some(sort) => sort
                .iter()
                .map(|tier| {
                    opened
                        .iter()
                        .zip(readers.iter())
                        .map(|(o, reader)| {
                            Self::read_sort_keys(
                                &o.doc_values,
                                &self.fields,
                                tier,
                                reader.max_doc(),
                            )
                        })
                        .collect::<Result<Vec<Vec<Option<i64>>>>>()
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let per_tier_key_slices: Vec<Vec<&[Option<i64>]>> = per_tier_keys
            .iter()
            .map(|per_source| per_source.iter().map(|k| k.as_slice()).collect())
            .collect();
        let sort_specs: Vec<merge::MergeSortKeySpec<'_>> = merge_sort
            .iter()
            .flatten()
            .zip(&per_tier_key_slices)
            .map(|(tier, keys)| merge::MergeSortKeySpec {
                sort: tier,
                per_source_keys: keys,
            })
            .collect();

        let merged_sci = merge::merge_segments(
            self.dir,
            &sources,
            (!sort_specs.is_empty()).then_some(sort_specs.as_slice()),
            &merge::MergeOptions {
                hnsw_m: self.hnsw_m,
                hnsw_beam_width: self.hnsw_beam_width,
            },
            &merged_segment_name,
            merged_segment_id,
            &self.codec_name,
            self.lucene_version,
        )?;

        self.apply_merge(&source_names, merged_sci)?;
        Ok(())
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
        let field_number = fields
            .iter()
            .find(|f| f.name == field_name)
            .map(|f| f.number)
            .ok_or_else(|| Error::UnknownSortField(field_name.to_string()))?;
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
            // Everything else `set_index_sort` allows reads a single-valued
            // NUMERIC column; the ordinal and byte kinds it refuses cannot
            // reach a merge this writer runs.
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
        if !self.updates_stream.any() {
            return Ok(());
        }
        let packets: Vec<FrozenBufferedUpdates> = self.updates_stream.take_pending();
        // With one indexing thread a packet is fully resolved inside the call
        // that pushed it, so Java's `FinishedSegments` in-flight bookkeeping
        // collapses to "everything pending is now done".
        //
        // That "inside the call that pushed it" is also what makes merging safe
        // without Java's `BufferedUpdatesStream.waitApplyForMerge`. A merged
        // segment is published with `buffered_deletes_gen == -1`, i.e. open to
        // every packet, which would be wrong if a packet targeting only the
        // *sources* could still be pending when the merge lands. It cannot:
        // `flush()` applies and clears the stream before `finish_commit` runs
        // `auto_merge`, so the stream is always empty at merge time. A delete
        // still sitting in the *queue* (not yet frozen) is a different and
        // correct case -- it has not been applied to the sources either, and
        // the merged segment carries their live documents forward, so applying
        // it to the merged segment reaches exactly the same documents.

        // `std::mem::take` rather than an index loop: the apply below needs
        // `&mut self` for the `.liv`/doc-values-generation writes, and the segment lists are
        // put back verbatim (plus their new generations) once it is done.
        let mut committed = std::mem::take(&mut self.segment_infos.segments);
        let mut committed_fully_deleted = Vec::with_capacity(committed.len());
        for sci in committed.iter_mut() {
            committed_fully_deleted.push(self.apply_packets_to_segment(&packets, sci)?);
        }
        self.segment_infos.segments = committed;

        let mut flushed = std::mem::take(&mut self.flushed_segments);
        let mut flushed_fully_deleted = Vec::with_capacity(flushed.len());
        for sci in flushed.iter_mut() {
            flushed_fully_deleted.push(self.apply_packets_to_segment(&packets, sci)?);
        }
        self.flushed_segments = flushed;

        self.drop_fully_deleted_segments(&committed_fully_deleted, &flushed_fully_deleted);
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
    /// this apply is not reconsidered. [`Self::apply_packets_to_segment`] returns
    /// `false` for a segment it skipped, which reproduces that.
    ///
    /// This is an **in-memory** drop plus the deleter checkpoint the caller
    /// already runs, not a new commit: Java's `checkpoint()` is
    /// `changed(); deleter.checkpoint(segmentInfos, false)`, and a dropped
    /// segment only becomes durable at the next `commit()`. A `rollback()`
    /// before that restores it from `rollback_segments`, unchanged.
    ///
    /// *Not modelled*: `dropDeletedSegment`'s `mergingSegments` guard (this
    /// port has no concurrent merges -- `execute_merge` runs to completion
    /// inside one call), `readerPool.drop` (no reader pool) and
    /// `adjustPendingNumDocs` (no such counter -- see
    /// `crates/lucene-index/src/deletes.rs`' module doc).
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
    }

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
    fn apply_packets_to_segment(
        &mut self,
        packets: &[FrozenBufferedUpdates],
        sci: &mut SegmentCommitInfo,
    ) -> Result<bool> {
        let applicable: Vec<&FrozenBufferedUpdates> = packets
            .iter()
            .filter(|p| p.applies_to(sci.buffered_deletes_gen))
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
        // < limit` branches. Cloned rather than borrowed because the writes
        // further down need `&mut self`; only the one segment the map names
        // pays for it, and only on a sorted flush.
        let sort_map: Option<Vec<usize>> = self
            .pending_sort_map
            .as_ref()
            .filter(|(name, _)| name == &sci.segment_name)
            .map(|(_, map)| map.clone());
        // "Is this doc id below the packet's pre-sort limit?" -- the one
        // predicate every filter below uses, so the mapping cannot be applied
        // in one place and forgotten in another.
        let below_limit = |doc: i32, limit: i32| -> bool {
            match &sort_map {
                None => doc < limit,
                Some(map) => (map[doc as usize] as i32) < limit,
            }
        };

        let owned = self.open_segment_for_deletes(sci)?;
        let opened = owned.view()?;
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
        let mut deleted: std::collections::HashSet<i32> = std::collections::HashSet::new();
        // field number -> updates in ascending packet-generation order, so the
        // last write for a doc wins (`DocValuesFieldUpdates.mergedIterator`).
        let mut numeric_updates: Vec<PerFieldNumericUpdates> = Vec::new();
        let mut binary_updates: Vec<PerFieldBinaryUpdates> = Vec::new();

        for packet in &applicable {
            let seg_gen = sci.buffered_deletes_gen;
            for (term, entry_limit) in &packet.delete_terms {
                let limit = packet.limit_for(seg_gen, *entry_limit);
                let docs = term_delete::resolve_term_doc_ids(
                    opened.fields,
                    opened.doc_in.as_ref(),
                    opened.live_docs,
                    &term.field,
                    &term.bytes,
                )?;
                deleted.extend(docs.into_iter().filter(|&d| below_limit(d, limit)));
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
                let docs = Self::resolve_delete_query(query, &opened, bound)?;
                deleted.extend(docs.into_iter().filter(|&d| below_limit(d, limit)));
            }
            for (field, buffer) in &packet.field_updates {
                let Some(info) = self.fields.iter().find(|f| &f.name == field) else {
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
                    let matched: Vec<i32> = docs
                        .into_iter()
                        .filter(|&d| below_limit(d, limit) && !deleted.contains(&d))
                        .collect();
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

        if !deleted.is_empty() {
            // Deterministic order so a `.liv` is byte-identical run to run --
            // `apply_deletes` itself is order-insensitive (it clears bits).
            let mut deleted: Vec<i32> = deleted.into_iter().collect();
            deleted.sort_unstable();
            *sci =
                deletes::apply_deletes(self.dir, sci, opened.live_docs, opened.max_doc, deleted)?;
        }

        if !numeric_updates.is_empty() || !binary_updates.is_empty() {
            self.write_doc_values_update_generation(sci, &numeric_updates, &binary_updates)?;
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
    /// generation records the field's `FieldInfo.docValuesGen`. See
    /// `docs/sweep/m2/c14-dv-updates-format.md` for what this replaced.
    fn write_doc_values_update_generation(
        &mut self,
        sci: &mut SegmentCommitInfo,
        numeric: &[PerFieldNumericUpdates],
        binary: &[PerFieldBinaryUpdates],
    ) -> Result<()> {
        crate::field_updates::write_field_updates(
            self.dir,
            sci,
            numeric,
            binary,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
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
    /// with an empty [`lucene_codecs::blocktree::BlockTreeFields`], which resolves every term to zero
    /// documents -- the same outcome as Java's `TermDocsIterator.nextTerm`
    /// returning null, and the reason a delete against a stored-fields-only
    /// segment is a legitimate no-op rather than an error.
    fn open_segment_for_deletes(&self, sci: &SegmentCommitInfo) -> Result<OpenedDeleteSegment> {
        let suffix = per_field_codec_suffix(POSTINGS_FORMAT_NAME);
        let si_bytes = self.dir.open(&format!("{}.si", sci.segment_name))?;
        let si = segment_info::parse(&si_bytes, &sci.segment_id)?;
        let max_doc = si.doc_count as usize;

        let live_docs = if sci.del_gen >= 0 {
            let liv = self
                .dir
                .open(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))?;
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

        if !si.files.iter().any(|f| f.ends_with(".tim")) {
            return Ok(OpenedDeleteSegment {
                max_doc,
                live_docs,
                fields: lucene_codecs::blocktree::BlockTreeFields::empty(),
                doc_input: None,
                segment_id: sci.segment_id,
                suffix,
            });
        }

        // Every indexed field, not just the two `execute_merge` filters to:
        // `blocktree::open` resolves the `.tmd`'s field numbers through this
        // list, so a field the term dictionary carries but the list omits
        // would fail the open.
        let field_infos = lucene_codecs::field_infos::FieldInfos {
            fields: self
                .fields
                .iter()
                .filter(|f| f.index_options != IndexOptions::None)
                .cloned()
                .collect(),
        };
        let seg = per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
        // Borrowed from the directory's `Input` (a mapping, under
        // `MmapDirectory`), not copied: `blocktree::open` builds its own
        // structures from these and does not retain the slices, and the `.doc`
        // `Input` is kept alive in the returned struct for `DocInput` to
        // borrow. Copying them was heap-copying an entire segment's postings
        // per buffered-delete round.
        let tim = self.dir.open(&format!("{seg}.tim"))?;
        let tip = self.dir.open(&format!("{seg}.tip"))?;
        let tmd = self.dir.open(&format!("{seg}.tmd"))?;
        let doc_input = self.dir.open(&format!("{seg}.doc"))?;
        let fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &suffix,
            max_doc as i32,
        )?;
        Ok(OpenedDeleteSegment {
            max_doc,
            live_docs,
            fields,
            doc_input: Some(doc_input),
            segment_id: sci.segment_id,
            suffix,
        })
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

    /// Number of documents buffered by [`IndexWriter::add_document`] but not
    /// yet written to disk by a [`IndexWriter::commit`] call.
    pub fn pending_doc_count(&self) -> usize {
        self.pending_docs.len()
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
            let si_bytes = self.dir.open(&format!("{}.si", sci.segment_name))?.to_vec();
            let si = segment_info::parse(&si_bytes, &sci.segment_id)?;
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
    /// `AlreadyClosedException`). This facade has no open/close lifecycle or
    /// write-lock concept at all (see module doc comment: "one caller, one
    /// `Directory`, sequential calls" -- there is no `IndexWriterConfig`-style
    /// closeable object here to begin with), so this `rollback()` leaves the
    /// writer fully usable for further [`IndexWriter::add_document`]/
    /// [`IndexWriter::commit`] calls immediately afterward -- the same choice
    /// this facade already made for having no `close()` method at all.
    pub fn rollback(&mut self) {
        self.pending_docs.clear();
        self.pending_custom_freq_terms.clear();
        self.pending_vectors.clear();
        self.ram_bytes_used = 0;
        // Only ever `Some` inside `flush()`, and cleared at its end -- but a
        // `flush()` that fails *after* publishing the segment (in
        // `apply_all_deletes_and_updates`) leaves it set, and `rollback` is
        // the caller's answer to that.
        self.pending_sort_map = None;
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
        if let Some(prepared) = self.prepared_commit.take() {
            // `SegmentInfos.rollbackCommit`: drop the `pending_segments_N` the
            // prepare wrote, ignoring any failure (Java uses
            // `IOUtils.deleteFilesIgnoringExceptions` here for exactly the same
            // reason -- an undeleted pending file is inert, so failing to
            // remove it must not turn a rollback into an error).
            segment_infos::rollback_pending(&prepared, self.dir);
        }
        // `IndexWriter.rollbackInternal`: `deleter.checkpoint(segmentInfos,
        // false)` then `deleter.refresh()`. The checkpoint releases the flushed
        // segments' references (deleting their files); the refresh catches
        // anything written by an operation that failed before any checkpoint saw
        // it. Both failures are ignored for the same reason the pending-file
        // deletion above is: an unreclaimed orphan is inert, and must not turn a
        // rollback into an error. Use [`IndexWriter::delete_unused_files`] when
        // you want that failure surfaced.
        let live = self.live_infos();
        let _ = self.deleter.checkpoint(&live, false);
        let _ = self.deleter.refresh();
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
    /// Also like Java's, the dropped segments' files are *not* deleted from
    /// `dir`: Java hands that to `IndexFileDeleter.checkpoint`, which this port
    /// has no equivalent of (see the module doc comment), so the files linger
    /// as unreferenced orphans no commit points at.
    ///
    /// Refused while a [`IndexWriter::prepare_commit`] is outstanding, for the
    /// same reason [`IndexWriter::delete_documents_by_term`] is.
    pub fn delete_all(&mut self) -> Result<()> {
        if self.prepared_commit.is_some() {
            return Err(Error::PreparedCommitPending("delete_all"));
        }
        self.pending_docs.clear();
        self.pending_custom_freq_terms.clear();
        self.pending_vectors.clear();
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
        // Java's `deleteAll` ends with `deleter.checkpoint(segmentInfos,
        // false)`. Segments that only this writer's in-memory view referenced
        // (an uncommitted flush) are reclaimed right here; segments the current
        // commit still names survive until the next commit supersedes it, which
        // is exactly why `deleteAll` is not itself durable.
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
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
    /// Note the one Java behaviour deliberately not copied: real
    /// `setLiveCommitData(data)` defaults `doIncrementVersion` to `true`, i.e.
    /// setting commit data alone counts as a change worth committing. This
    /// facade's `commit()` already always writes a new generation whether or
    /// not anything changed (see [`IndexWriter::commit`]), so there is no
    /// change counter for this to increment.
    pub fn set_live_commit_data(&mut self, data: Vec<(String, String)>) {
        self.segment_infos.user_data = data;
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
        new_segment_infos.id = generate_segment_id(new_segment_infos.generation);
        new_segment_infos
            .segments
            .retain(|s| !source_segment_names.contains(&s.segment_name.as_str()));
        // `SegmentInfos.applyMergeChanges(merge, dropSegment)`: the sources go
        // either way; the merged segment is inserted only when it was not
        // dropped.
        if let Some(merged) = merged {
            new_segment_infos.segments.push(merged);
        }

        segment_infos::write(&new_segment_infos, self.dir)?;
        // The merge's source segments are no longer named by any live commit
        // once the superseded commit point dies here, so this is what actually
        // reclaims them -- `commitMerge` -> `checkpoint` in Java.
        self.segment_infos = new_segment_infos;
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
        // hands out the name, precisely so a name can never be handed out twice
        // -- see F-2 in `docs/sweep/m2/b9-index-write.md`.
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
    fields: lucene_codecs::blocktree::BlockTreeFields,
    /// The `.doc` file's bytes, held as the `Input` the directory handed over
    /// rather than copied into a `Vec`. On an `MmapDirectory` that `Input` *is*
    /// the mapping, so a `to_vec()` here would heap-copy the whole postings
    /// file on every buffered-delete round; `DocInput::open` only needs to
    /// borrow it. `None` when the segment has no postings.
    doc_input: Option<lucene_store::directory::Input>,
    segment_id: [u8; ID_LENGTH],
    suffix: String,
}

impl OpenedDeleteSegment {
    fn view(&self) -> Result<DeleteSegmentView<'_>> {
        let doc_in = match &self.doc_input {
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
            fields: &self.fields,
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

/// The `docIDUpto` limit as a doc-space bound. [`crate::buffered_updates::MAX_DOC_ID_UPTO`] means "no
/// limit", which is `usize::MAX` here and is then clamped to the segment's own
/// `max_doc` by every caller.
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

/// Generates a 16-byte segment/commit id from `salt` (this writer's current
/// segment-name counter) plus the current time -- **not** a
/// cryptographically random id the way real Lucene's
/// `StringHelper.randomId()` (backed by a `SecureRandom`) is. This
/// workspace has no `rand`-family dependency (see `Cargo.toml`'s
/// `[workspace.dependencies]`), and the only property this port's readers
/// actually rely on (verified: `.si`/`segments_N` parsing checks a
/// referenced id *matches*, never that it looks statistically random) is
/// "distinct segments get distinct ids" -- which salting a hash with a
/// monotonically increasing counter already guarantees deterministically,
/// without pulling in a new dependency for a property this scope doesn't
/// need.
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
fn document_ram_bytes(doc: &Document) -> usize {
    std::mem::size_of::<Document>()
        + doc.fields.capacity() * std::mem::size_of::<stored_fields::StoredField>()
        + doc
            .fields
            .iter()
            .map(|f| match &f.value {
                FieldValue::String(s) => s.capacity(),
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
/// [`small_float::int_to_byte4`]'s own `debug_assert` on the way past
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

/// The buffered-side heap cost of one document's explicit
/// [`IndexWriter::add_document_with_custom_freq_terms`] term list.
/// ARITH: as [`VectorValue::ram_bytes`] -- a sum of live allocation sizes.
#[allow(clippy::arithmetic_side_effects)]
fn custom_freq_terms_ram_bytes(terms: &[(String, i32)]) -> usize {
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
        FieldValue::String(_) => "String",
        FieldValue::Binary(_) => "Binary",
        FieldValue::Int(_) => "Int",
        FieldValue::Long(_) => "Long",
        FieldValue::Float(_) => "Float",
        FieldValue::Double(_) => "Double",
    }
}

fn generate_segment_id(salt: i64) -> [u8; ID_LENGTH] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let mut h1 = DefaultHasher::new();
    (nanos, salt, 1u8).hash(&mut h1);
    let mut h2 = DefaultHasher::new();
    (nanos, salt, 2u8).hash(&mut h2);

    let mut id = [0u8; ID_LENGTH];
    id[0..8].copy_from_slice(&h1.finish().to_le_bytes());
    id[8..16].copy_from_slice(&h2.finish().to_le_bytes());
    id
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a fixture builder's
    // own index arithmetic is not one (see `docs/arithmetic-gate.md`).
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::segment_info::IndexSortField;
    use lucene_codecs::blocktree;
    use lucene_codecs::field_infos::{
        self as fi, DocValuesSkipIndexType, DocValuesType, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::hnsw::HnswGraphView;
    use lucene_codecs::postings::DocInput;
    use lucene_codecs::stored_fields::{self, FieldValue, StoredField};
    use lucene_store::directory::FsDirectory;

    fn version() -> LuceneVersion {
        LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        }
    }

    fn stored_only_field(name: &str, number: i32) -> FieldInfo {
        FieldInfo {
            name: name.to_string(),
            number,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        }
    }

    fn doc(id: &str) -> Document {
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            }],
        }
    }

    fn doc_value(d: &Document) -> String {
        match &d.fields[0].value {
            FieldValue::String(s) => s.clone(),
            other => panic!("unexpected field value shape: {other:?}"),
        }
    }

    use lucene_util::test_support::TempDir;

    /// A scratch directory that removes itself when the test ends -- unless
    /// the test is panicking, in which case its bytes stay for inspection.
    fn tempdir(tag: &str) -> TempDir {
        TempDir::new(&format!("index-writer-{tag}"))
    }

    /// Reads every document out of every segment `segment_infos` lists, in
    /// commit order -- the minimal "is this readable/searchable" check this
    /// crate can do on its own (the real reader/searcher stack lives one
    /// layer up in `lucene-search`, which already depends on `lucene-index`
    /// and so cannot be depended on back from here -- see the
    /// `architecture` skill).
    fn read_all_docs(dir: &FsDirectory, segment_infos: &SegmentInfos) -> Vec<String> {
        let mut out = Vec::new();
        for sci in &segment_infos.segments {
            let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
            let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
            let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
            let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
            let live = if sci.del_gen >= 0 {
                let liv = dir
                    .open(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))
                    .unwrap();
                Some(
                    lucene_codecs::live_docs::parse(
                        &liv,
                        &sci.segment_id,
                        sci.del_gen,
                        reader.max_doc() as usize,
                        sci.del_count as usize,
                    )
                    .unwrap(),
                )
            } else {
                None
            };
            for doc_id in 0..reader.max_doc() {
                let is_live = live
                    .as_ref()
                    .map(|bits| bits.get(doc_id as usize))
                    .unwrap_or(true);
                if is_live {
                    out.push(doc_value(&reader.document(doc_id).unwrap()));
                }
            }
        }
        out
    }

    #[test]
    fn open_on_a_fresh_directory_starts_with_no_segments() {
        let tmp = tempdir("fresh");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        assert!(writer.segment_infos().segments.is_empty());
        assert_eq!(writer.pending_doc_count(), 0);
    }

    /// `IndexWriter::open` is the port's caller-facing door for a hand-built
    /// field list, and Java's is the one place a `FieldInfo` can come from --
    /// its constructor, which throws. Before this, an inconsistent field was
    /// only found much later, at `parse` time or (for the combinations
    /// `field_infos::write` coerces) never.
    #[test]
    fn open_rejects_a_field_list_javas_fieldinfo_constructor_would_throw_on() {
        let tmp = tempdir("open-validates-fields");
        let dir = FsDirectory::open(&tmp);

        // Payloads without positions: an *indexed* field, so no coercion
        // rescues it -- `FieldInfo.checkConsistency` throws.
        let bad = FieldInfo::new("body", 0)
            .with_index_options(IndexOptions::DocsAndFreqs)
            .with_store_payloads(true);
        assert!(matches!(
            IndexWriter::open(&dir, vec![bad], "Lucene104", version()).err(),
            Some(Error::FieldInfos(fi::Error::Inconsistent(_, _)))
        ));

        // Cross-field: two fields sharing a number, which
        // `FieldInfos(FieldInfo[])` rejects.
        assert!(matches!(
            IndexWriter::open(
                &dir,
                vec![stored_only_field("a", 0), stored_only_field("b", 0)],
                "Lucene104",
                version()
            )
            .err(),
            Some(Error::FieldInfos(fi::Error::InvalidFieldInfos(_)))
        ));
    }

    /// The other half of Java's constructor: a *non*-indexed field's
    /// `omitNorms`/`storePayloads`/`storeTermVector` are coerced off rather
    /// than rejected, so the writer's stored copy is the one Lucene would
    /// hold -- and the `.fnm` it later writes cannot disagree with it.
    #[test]
    fn open_coerces_the_indexed_only_flags_off_a_non_indexed_field() {
        let tmp = tempdir("open-coerces-fields");
        let dir = FsDirectory::open(&tmp);
        let writer = IndexWriter::open(
            &dir,
            vec![FieldInfo::new("id", 0)
                .with_omit_norms(true)
                .with_store_term_vectors(true)
                .with_store_payloads(true)],
            "Lucene104",
            version(),
        )
        .unwrap();
        assert!(!writer.fields[0].omit_norms);
        assert!(!writer.fields[0].store_term_vectors);
        assert!(!writer.fields[0].store_payloads);
    }

    #[test]
    fn add_documents_then_commit_produces_one_readable_segment() {
        let tmp = tempdir("add-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.add_document(doc("b")).unwrap();
        writer.add_document(doc("c")).unwrap();
        assert_eq!(writer.pending_doc_count(), 3);

        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        assert_eq!(writer.pending_doc_count(), 0);

        // Readable back through the on-disk segments_N this call wrote --
        // not just through the returned struct.
        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.generation, sis.generation);
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b", "c"]);
    }

    #[test]
    fn committed_doc_count_is_zero_on_a_fresh_directory() {
        let tmp = tempdir("committed-doc-count-fresh");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        assert_eq!(writer.committed_doc_count().unwrap(), 0);
    }

    #[test]
    fn committed_doc_count_sums_across_multiple_commits_and_segments_and_is_distinct_from_pending()
    {
        let tmp = tempdir("committed-doc-count-multi");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        // First commit: 3 docs in one segment.
        writer.add_document(doc("a")).unwrap();
        writer.add_document(doc("b")).unwrap();
        writer.add_document(doc("c")).unwrap();
        writer.commit().unwrap();
        assert_eq!(writer.committed_doc_count().unwrap(), 3);
        assert_eq!(writer.pending_doc_count(), 0);

        // Second commit: 2 more docs, producing a second segment. Total
        // committed count must now sum both segments' `.si` doc_count.
        writer.add_document(doc("d")).unwrap();
        writer.add_document(doc("e")).unwrap();
        writer.commit().unwrap();
        assert_eq!(writer.segment_infos().segments.len(), 2);
        assert_eq!(writer.committed_doc_count().unwrap(), 5);
        assert_eq!(writer.pending_doc_count(), 0);

        // Buffer 2 more docs without committing: committed_doc_count must
        // stay at 5 (untouched by uncommitted buffering) while
        // pending_doc_count reports the 2 buffered docs -- the two accessors
        // must never conflate committed-to-disk state with in-memory
        // buffered state.
        writer.add_document(doc("f")).unwrap();
        writer.add_document(doc("g")).unwrap();
        assert_eq!(writer.committed_doc_count().unwrap(), 5);
        assert_eq!(writer.pending_doc_count(), 2);

        // Committing the buffered pair brings committed_doc_count to 7 and
        // drains pending back to 0.
        writer.commit().unwrap();
        assert_eq!(writer.committed_doc_count().unwrap(), 7);
        assert_eq!(writer.pending_doc_count(), 0);
    }

    #[test]
    fn commit_with_no_pending_documents_is_a_valid_no_op_content_commit() {
        let tmp = tempdir("empty-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.generation, 1);
        assert!(sis.segments.is_empty());

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.generation, 1);
    }

    #[test]
    fn prepare_commit_then_finish_commit_matches_commit_directly() {
        // Two writers over two separate directories, fed the identical
        // pending docs -- one calls `commit()`, the other calls
        // `prepare_commit()` then `finish_commit()`. Their on-disk results
        // (the returned `SegmentInfos` and every readable document) must be
        // identical.
        let fields = || vec![stored_only_field("id", 0)];

        let tmp_a = tempdir("two-phase-direct");
        let dir_a = FsDirectory::open(&tmp_a);
        let mut writer_a = IndexWriter::open(&dir_a, fields(), "Lucene104", version()).unwrap();
        writer_a.add_document(doc("a")).unwrap();
        writer_a.add_document(doc("b")).unwrap();
        let sis_a = writer_a.commit().unwrap().clone();

        let tmp_b = tempdir("two-phase-split");
        let dir_b = FsDirectory::open(&tmp_b);
        let mut writer_b = IndexWriter::open(&dir_b, fields(), "Lucene104", version()).unwrap();
        writer_b.add_document(doc("a")).unwrap();
        writer_b.add_document(doc("b")).unwrap();
        writer_b.prepare_commit().unwrap();
        // Not yet visible: the prepared generation hasn't been activated.
        assert_eq!(
            lucene_store::directory::last_commit_generation(&dir_b.list_all().unwrap()).unwrap(),
            -1
        );
        let sis_b = writer_b.finish_commit().unwrap().clone();

        assert_eq!(sis_a.generation, sis_b.generation);
        assert_eq!(sis_a.segments.len(), sis_b.segments.len());
        assert_eq!(read_all_docs(&dir_a, &sis_a), read_all_docs(&dir_b, &sis_b));

        let reopened_b = segment_infos::read_latest(&dir_b).unwrap();
        assert_eq!(reopened_b.generation, sis_b.generation);
        assert_eq!(read_all_docs(&dir_b, &reopened_b), vec!["a", "b"]);
    }

    #[test]
    fn prepare_commit_without_finish_commit_leaves_the_previous_commit_current() {
        // Simulates "prepared but never activated" (e.g. a crash before
        // `finish_commit()`): a fresh reader of `dir` must still see the
        // prior commit, not the prepared-but-unpublished one -- this port
        // provides no crash-recoverable "prepared" marker at all (see
        // `prepare_commit`'s doc comment), so the only honest guarantee is
        // that nothing changes until `finish_commit()` actually runs.
        let tmp = tempdir("prepare-only");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.commit().unwrap();
        let committed_generation = writer.segment_infos().generation;

        writer.add_document(doc("b")).unwrap();
        writer.prepare_commit().unwrap();

        // The new segment's files are on disk, but no segments_N points at
        // them: a fresh open() still sees only the first commit.
        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.generation, committed_generation);
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a"]);

        // `self.segment_infos` on the live writer is likewise untouched --
        // only `finish_commit()` would update it.
        assert_eq!(writer.segment_infos().generation, committed_generation);
        assert_eq!(
            writer.segment_infos().segments.len(),
            reopened.segments.len()
        );
    }

    #[test]
    fn finish_commit_without_a_prior_prepare_commit_is_an_error() {
        let tmp = tempdir("finish-without-prepare");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer.finish_commit().unwrap_err();
        assert!(matches!(err, Error::NoPreparedCommit));
    }

    #[test]
    fn finish_commit_twice_in_a_row_is_an_error_the_second_time() {
        let tmp = tempdir("finish-twice");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.prepare_commit().unwrap();
        writer.finish_commit().unwrap();

        let err = writer.finish_commit().unwrap_err();
        assert!(matches!(err, Error::NoPreparedCommit));
    }

    #[test]
    fn calling_prepare_commit_again_before_finish_commit_is_rejected_and_loses_nothing() {
        // Real `IndexWriter.prepareCommitInternal` throws
        // IllegalStateException("prepareCommit was already called with no
        // corresponding call to commit"). This port used to *replace* the
        // prepared state instead, which silently discarded every document the
        // first prepare had already flushed and synced: the second prepare
        // rebuilt its SegmentInfos from `self.segment_infos`, which had never
        // seen the first prepare's segment.
        let tmp = tempdir("prepare-twice");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.prepare_commit().unwrap();

        writer.add_document(doc("b")).unwrap();
        let err = writer.prepare_commit().unwrap_err();
        assert!(
            matches!(err, Error::PrepareCommitAlreadyCalled),
            "second prepare_commit() must be refused, got {err:?}"
        );

        // The first prepare is untouched and still activatable, and the
        // document buffered after it is still buffered (not flushed, not lost).
        assert_eq!(writer.pending_doc_count(), 1);
        let sis = writer.finish_commit().unwrap().clone();
        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.generation, sis.generation);
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a"]);

        // ...and "b" lands in the next commit, so nothing was dropped anywhere.
        writer.commit().unwrap();
        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b"]);
    }
    /// The real two-phase protocol, on disk: after `prepare_commit()` the
    /// commit exists as `pending_segments_N` -- a name
    /// `last_commit_generation` deliberately does not scan for -- and only
    /// `finish_commit()`'s rename makes it the current `segments_N`. This is
    /// what makes a crash between the two phases recoverable: the previous
    /// commit is still current and the pending file is inert, rather than a
    /// half-written `segments_N` that would make the whole index unopenable.
    #[test]
    fn prepare_commit_writes_a_pending_segments_file_that_finish_commit_renames() {
        let tmp = tempdir("two-phase-pending-file");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.commit().unwrap();
        writer.add_document(doc("b")).unwrap();
        writer.prepare_commit().unwrap();

        let listed = dir.list_all().unwrap();
        assert!(
            listed.iter().any(|f| f == "pending_segments_2"),
            "prepare_commit must leave a pending_segments_N behind: {listed:?}"
        );
        assert!(
            !listed.iter().any(|f| f == "segments_2"),
            "prepare_commit must NOT publish segments_N yet: {listed:?}"
        );
        // A fresh reader still sees the previous commit, exactly as it would
        // after a crash at this point.
        let before = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(before.generation, 1);
        assert_eq!(read_all_docs(&dir, &before), vec!["a"]);

        writer.finish_commit().unwrap();
        let listed = dir.list_all().unwrap();
        assert!(
            !listed.iter().any(|f| f == "pending_segments_2"),
            "finish_commit must rename the pending file away, not copy it: {listed:?}"
        );
        assert!(listed.iter().any(|f| f == "segments_2"), "{listed:?}");
        let after = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(after.generation, 2);
        assert_eq!(read_all_docs(&dir, &after), vec!["a", "b"]);
    }

    /// `rollback()` after a `prepare_commit()` must also remove the pending
    /// commit file (`SegmentInfos.rollbackCommit`), or a later
    /// `prepare_commit()` at the same generation would be overwriting a file
    /// the caller already asked to discard.
    #[test]
    fn rollback_deletes_the_pending_segments_file_prepare_commit_wrote() {
        let tmp = tempdir("two-phase-rollback-pending");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.prepare_commit().unwrap();
        assert!(dir
            .list_all()
            .unwrap()
            .iter()
            .any(|f| f == "pending_segments_1"));

        writer.rollback();
        let listed = dir.list_all().unwrap();
        assert!(
            !listed.iter().any(|f| f == "pending_segments_1"),
            "rollback must delete the pending commit file: {listed:?}"
        );
        assert!(matches!(
            writer.finish_commit().unwrap_err(),
            Error::NoPreparedCommit
        ));
    }

    /// Every operation that writes its own `segments_N` claims
    /// `generation + 1` -- the *same* generation a pending prepared commit has
    /// already claimed. Letting them through meant `finish_commit()` later
    /// overwrote that commit with a segment list built before it, silently
    /// reverting the delete/merge/update.
    #[test]
    fn commit_writing_operations_are_refused_while_a_commit_is_prepared() {
        let tmp = tempdir("prepared-guard");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.prepare_commit().unwrap();

        assert!(matches!(
            writer
                .delete_documents_with_sources(&[], "id", b"a")
                .unwrap_err(),
            Error::PreparedCommitPending("delete_documents")
        ));
        assert!(matches!(
            writer.delete_all().unwrap_err(),
            Error::PreparedCommitPending("delete_all")
        ));
        assert!(matches!(
            writer
                .update_document_with_sources(&[], "id", b"a", doc("x"))
                .unwrap_err(),
            Error::PreparedCommitPending("update_document")
        ));
        let merged = SegmentCommitInfo {
            segment_name: "_9".to_string(),
            segment_id: [0u8; ID_LENGTH],
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
            ..Default::default()
        };
        assert!(matches!(
            writer.apply_merge(&[], merged).unwrap_err(),
            Error::PreparedCommitPending("apply_merge")
        ));

        // The prepared commit is still intact and still activatable.
        writer.finish_commit().unwrap();
        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a"]);
    }

    /// Java writes a fresh `StringHelper.randomId()` into every `segments_N`
    /// header. Cloning the previous commit's `SegmentInfos` (as every commit
    /// path here does) carried the same id forward forever, so no two
    /// generations of an index were distinguishable by id.
    #[test]
    fn every_commit_generation_gets_its_own_commit_id() {
        let tmp = tempdir("commit-id");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        let first = writer.commit().unwrap().id;
        writer.add_document(doc("b")).unwrap();
        let second = writer.commit().unwrap().id;
        assert_ne!(first, second, "each commit must carry its own id");
        assert_eq!(segment_infos::read_latest(&dir).unwrap().id, second);
    }

    /// `IndexWriter.deleteAll()`: buffered docs and every segment go away
    /// in memory; nothing is committed until the next `commit()`.
    #[test]
    fn delete_all_drops_buffered_and_committed_segments_but_only_on_next_commit() {
        let tmp = tempdir("delete-all");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.commit().unwrap();
        writer.add_document(doc("b")).unwrap();
        assert_eq!(writer.pending_doc_count(), 1);

        writer.delete_all().unwrap();
        assert_eq!(writer.pending_doc_count(), 0);
        assert!(writer.segment_infos().segments.is_empty());

        // Not durable yet: a fresh reader still sees the pre-deleteAll commit,
        // matching Java's deleteAll() not writing a commit of its own.
        let still = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(read_all_docs(&dir, &still), vec!["a"]);

        writer.commit().unwrap();
        let after = segment_infos::read_latest(&dir).unwrap();
        assert!(after.segments.is_empty());
        assert_eq!(read_all_docs(&dir, &after), Vec::<String>::new());
    }

    /// `IndexWriter.setLiveCommitData`/`getLiveCommitData`: opaque caller
    /// metadata carried in `SegmentInfos.userData` from the next commit
    /// onwards, and still there when the directory is reopened.
    #[test]
    fn live_commit_data_is_written_into_the_commit_and_survives_reopen() {
        let tmp = tempdir("live-commit-data");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
        assert!(writer.live_commit_data().is_empty());

        writer.set_live_commit_data(vec![("translog".to_string(), "uuid-1".to_string())]);
        writer.add_document(doc("a")).unwrap();
        writer.commit().unwrap();

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(
            reopened.user_data,
            vec![("translog".to_string(), "uuid-1".to_string())]
        );

        // ...and a writer resumed on that directory reports it back, then
        // carries it into its own next commit unless replaced.
        let mut resumed = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        assert_eq!(
            resumed.live_commit_data(),
            [("translog".to_string(), "uuid-1".to_string())]
        );
        resumed.add_document(doc("b")).unwrap();
        resumed.commit().unwrap();
        assert_eq!(
            segment_infos::read_latest(&dir).unwrap().user_data,
            vec![("translog".to_string(), "uuid-1".to_string())]
        );
    }

    /// A failed publish must not swallow the prepared commit: `finish_commit`
    /// used to `take()` it unconditionally, so a failure at the rename left the
    /// already-flushed-and-synced segment orphaned with no way to retry or roll
    /// back. Deleting the pending file out from under it is the cheapest way to
    /// make the rename fail for real.
    #[test]
    fn a_failed_finish_commit_keeps_the_prepared_commit_instead_of_dropping_it() {
        let tmp = tempdir("finish-commit-failure");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.prepare_commit().unwrap();
        dir.delete_file("pending_segments_1").unwrap();

        let err = writer.finish_commit().unwrap_err();
        assert!(
            !matches!(err, Error::NoPreparedCommit),
            "the failure must come from the publish, not from a missing prepared commit: {err:?}"
        );
        // Still prepared: a second attempt reports the publish failure again,
        // not "nothing to activate".
        assert!(!matches!(
            writer.finish_commit().unwrap_err(),
            Error::NoPreparedCommit
        ));
        // ...and rollback still cleans the state up.
        writer.rollback();
        assert!(matches!(
            writer.finish_commit().unwrap_err(),
            Error::NoPreparedCommit
        ));
    }

    #[test]
    fn prepare_commit_and_finish_commit_with_no_pending_documents_is_a_valid_no_op() {
        let tmp = tempdir("two-phase-empty-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.prepare_commit().unwrap();
        let sis = writer.finish_commit().unwrap().clone();
        assert!(sis.segments.is_empty());

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.generation, sis.generation);
        assert!(reopened.segments.is_empty());
    }

    #[test]
    fn multiple_commits_produce_multiple_independent_segments() {
        let tmp = tempdir("multi-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.commit().unwrap();

        writer.add_document(doc("b")).unwrap();
        writer.add_document(doc("c")).unwrap();
        writer.commit().unwrap();

        let sis = writer.segment_infos().clone();
        assert_eq!(sis.segments.len(), 2);
        assert_ne!(sis.segments[0].segment_name, sis.segments[1].segment_name);

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.segments.len(), 2);
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b", "c"]);
    }

    #[test]
    fn rollback_discards_pending_docs_so_next_commit_never_sees_them() {
        let tmp = tempdir("rollback-basic");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.add_document(doc("b")).unwrap();
        assert_eq!(writer.pending_doc_count(), 2);

        writer.rollback();
        assert_eq!(writer.pending_doc_count(), 0);

        // Nothing was ever written to disk -- a subsequent commit is a
        // no-op-content commit, same as if the docs had never been added.
        let sis = writer.commit().unwrap().clone();
        assert!(sis.segments.is_empty());

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert!(reopened.segments.is_empty());
        assert_eq!(read_all_docs(&dir, &reopened), Vec::<String>::new());
    }

    #[test]
    fn rollback_after_prepare_commit_discards_the_prepared_state_too() {
        // Found in review: rollback() previously only cleared pending_docs,
        // leaving `prepared_commit` intact -- so prepare_commit() ->
        // rollback() -> finish_commit() would silently activate the segment
        // the caller just asked to roll back. rollback() must now discard
        // prepared_commit as well, and finish_commit() afterward must fail
        // with NoPreparedCommit, exactly like it would if prepare_commit()
        // had never been called.
        let tmp = tempdir("rollback-after-prepare");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.prepare_commit().unwrap();
        assert!(writer.pending_doc_count() == 0 || writer.pending_doc_count() == 1);

        writer.rollback();

        let err = writer.finish_commit().unwrap_err();
        assert!(matches!(err, Error::NoPreparedCommit));

        // Nothing was ever written to disk -- no segments_N file exists at all.
        assert!(segment_infos::read_latest(&dir).is_err());

        // The writer is still fully usable afterward.
        writer.add_document(doc("b")).unwrap();
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        assert_eq!(read_all_docs(&dir, &sis), vec!["b"]);
    }

    #[test]
    fn rollback_with_nothing_pending_is_a_safe_no_op() {
        let tmp = tempdir("rollback-noop");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        assert_eq!(writer.pending_doc_count(), 0);
        writer.rollback();
        assert_eq!(writer.pending_doc_count(), 0);

        // Still fully usable afterward.
        writer.add_document(doc("a")).unwrap();
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a"]);
    }

    #[test]
    fn rollback_never_affects_a_prior_commits_segments() {
        let tmp = tempdir("rollback-prior-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        // First commit: "a" is on disk for good.
        writer.add_document(doc("a")).unwrap();
        let first = writer.commit().unwrap().clone();
        assert_eq!(first.segments.len(), 1);

        // Second batch: buffered but never committed, then rolled back.
        writer.add_document(doc("b")).unwrap();
        writer.add_document(doc("c")).unwrap();
        assert_eq!(writer.pending_doc_count(), 2);
        writer.rollback();
        assert_eq!(writer.pending_doc_count(), 0);

        // A commit right after the rollback only re-writes segments_N with
        // no new segment appended -- "a"'s segment from the first commit is
        // still exactly as it was, "b"/"c" never appear anywhere.
        let second = writer.commit().unwrap().clone();
        assert_eq!(second.segments.len(), 1);
        assert_eq!(
            second.segments[0].segment_name,
            first.segments[0].segment_name
        );

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a"]);
    }

    #[test]
    fn rollback_preserves_writer_configuration() {
        let tmp = tempdir("rollback-config");
        let dir = FsDirectory::open(&tmp);
        let mut text_field = stored_only_field("text", 0);
        text_field.index_options = IndexOptions::DocsAndFreqs;
        text_field.store_term_vectors = true;
        let mut num_field = stored_only_field("num", 1);
        num_field.doc_values_type = DocValuesType::Numeric;
        let fields = vec![text_field, num_field];

        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("text")).unwrap();
        writer.set_term_vector_field(Some("text")).unwrap();
        writer.set_doc_values_field(Some("num")).unwrap();
        writer.set_merge_policy(Some(MergePolicyConfig {
            max_merge_at_once: 2,
            ..Default::default()
        }));

        // Buffer a doc with no values for the configured fields (so this
        // rolled-back batch doesn't have to satisfy the dense doc-values
        // requirement), then roll it back.
        writer.add_document(doc("placeholder")).unwrap();
        writer.rollback();
        assert_eq!(writer.pending_doc_count(), 0);

        // Configuration must have survived: a doc with real values for every
        // configured field, committed after the rollback, produces postings/
        // term-vector/doc-values files exactly as if the rollback never
        // happened.
        let real_doc = Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("hello world".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(42),
                },
            ],
        };
        writer.add_document(real_doc).unwrap();
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);

        let segment_name = &sis.segments[0].segment_name;
        let si_bytes = dir.open(&format!("{segment_name}.si")).unwrap();
        let si = segment_info::parse(&si_bytes, &sis.segments[0].segment_id).unwrap();
        assert!(si.files.iter().any(|f| f.ends_with(".doc")));
        assert!(si.files.iter().any(|f| f.ends_with(".tvd")));
        assert!(si.files.iter().any(|f| f.ends_with(".dvd")));
    }

    #[test]
    fn reopening_an_existing_directory_resumes_its_committed_state() {
        let tmp = tempdir("resume");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];

        {
            let mut writer =
                IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
            writer.add_document(doc("a")).unwrap();
            writer.commit().unwrap();
        }

        let mut writer2 = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        assert_eq!(writer2.segment_infos().segments.len(), 1);

        writer2.add_document(doc("b")).unwrap();
        writer2.commit().unwrap();

        let sis = writer2.segment_infos();
        assert_eq!(sis.segments.len(), 2);
        // The second session's segment name must not collide with the
        // first's.
        assert_ne!(sis.segments[0].segment_name, sis.segments[1].segment_name);
    }

    // --- update_document/delete_documents: needs a real postings fixture,
    // same one term_delete.rs/update_document.rs's own tests already use. ---

    struct Fixture {
        fields: blocktree::BlockTreeFields,
        doc_bytes: Vec<u8>,
        segment_id: [u8; ID_LENGTH],
        suffix: String,
        max_doc: usize,
    }

    impl Fixture {
        fn doc_in(&self) -> DocInput<'_> {
            DocInput::open(&self.doc_bytes, &self.segment_id, &self.suffix).expect("open .doc")
        }
    }

    fn open_fixture() -> Fixture {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenBlockTree)");
        let get = |key: &str| -> String {
            manifest
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
                .to_string()
        };
        let id_hex = get("id_hex");
        let mut segment_id = [0u8; ID_LENGTH];
        for (i, slot) in segment_id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = get("segment_suffix");
        let max_doc: i32 = get("max_doc").parse().unwrap();

        let read_raw = |name: &str| -> Vec<u8> {
            std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}"))
        };
        let fnm = read_raw(&get("fnm_file_name"));
        let field_infos = fi::parse(&fnm, &segment_id, "").expect("parse .fnm");
        let tim = read_raw(&get("tim_file_name"));
        let tip = read_raw(&get("tip_file_name"));
        let tmd = read_raw(&get("tmd_file_name"));
        let fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &segment_id,
            &suffix,
            max_doc,
        )
        .expect("open blocktree");
        let doc_bytes = read_raw(&get("doc_file_name"));

        Fixture {
            fields,
            doc_bytes,
            segment_id,
            suffix,
            max_doc: max_doc as usize,
        }
    }

    /// Seeds a writer's committed state with the real-Lucene fixture segment
    /// as segment `_0`, without going through `add_document`/`commit`
    /// (the fixture already has real postings; this facade's own
    /// `flush_stored_only_segment` path never writes any).
    ///
    /// The segment's `.si` **is** written into `dir`, listing only itself. That
    /// is not decoration: `IndexFileDeleter` enumerates a segment's files
    /// through its `.si` exactly the way `CheckIndex` and real Lucene's own
    /// `SegmentCommitInfo.files()` do, so a `SegmentCommitInfo` naming a segment
    /// with no `.si` on disk is a corrupt commit, not a shortcut. Listing only
    /// the `.si` (and not the fixture's `.tim`/`.tip`/`.tmd`/`.doc`, which live
    /// in `fixtures/data/` and were never copied here) keeps every file the
    /// deleter refcounts a file that actually exists in `dir`.
    fn writer_seeded_with_fixture<'d>(
        dir: &'d FsDirectory,
        fx: &Fixture,
        fields: Vec<FieldInfo>,
    ) -> IndexWriter<'d> {
        let si = segment_info::SegmentInfo {
            id: fx.segment_id,
            version: version(),
            min_version: None,
            doc_count: fx.max_doc as i32,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec!["_0.si".to_string()],
            attributes: vec![],
            index_sort: None,
        };
        let mut writer = IndexWriter::open(dir, fields, "Lucene104", version()).unwrap();
        // Written *after* the open: the deleter's init sweep reclaims every
        // unreferenced index file in `dir`, and a `.si` no commit names is
        // exactly that. Writing it here and checkpointing below is the same
        // order a real flush uses (write the files, then tell the deleter).
        write_file(dir, "_0.si", &segment_info::write(&si, "")).unwrap();
        writer.segment_infos.segments.push(SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: fx.segment_id,
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
            ..Default::default()
        });
        writer.segment_infos.counter = 1;
        // `IndexFileDeleter.checkpoint(segmentInfos, false)`: without it the
        // seeded segment's `.si` is still unreferenced, and the next
        // checkpoint/refresh would reclaim it.
        let live = writer.live_infos();
        writer.deleter.checkpoint(&live, false).unwrap();
        writer
    }

    #[test]
    fn update_document_replaces_a_matched_doc_and_is_visible_after_commit() {
        let fx = open_fixture();
        let tmp = tempdir("update-doc");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);

        let sources = [SegmentDeleteSource {
            segment_name: "_0",
            fields: &fx.fields,
            doc_in: None, // "id"->"id0" is a singleton term, no .doc needed
            live_docs: None,
            max_doc: fx.max_doc,
        }];

        let sis = writer
            .update_document_with_sources(&sources, "id", b"id0", doc("replacement"))
            .unwrap()
            .clone();
        assert_eq!(sis.segments.len(), 2);
        assert_eq!(sis.segments[0].del_count, 1);

        let reopened = segment_infos::read_latest(&dir).unwrap();
        // The old segment's real postings can't be read back through this
        // crate's stored-fields-only reader helper, so just confirm the new
        // segment (this writer's own flush) is visible and correct.
        let new_sci = reopened
            .segments
            .iter()
            .find(|s| s.segment_name != "_0")
            .unwrap();
        let fdt = dir.open(&format!("{}.fdt", new_sci.segment_name)).unwrap();
        let fdx = dir.open(&format!("{}.fdx", new_sci.segment_name)).unwrap();
        let fdm = dir.open(&format!("{}.fdm", new_sci.segment_name)).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &new_sci.segment_id, "").unwrap();
        assert_eq!(doc_value(&reader.document(0).unwrap()), "replacement");
    }

    /// Real `IndexWriter.newSegmentName()` bumps `segmentInfos.counter` at
    /// the moment it hands out a name, so the counter written into the very
    /// next commit already accounts for the flushed segment. This port used to
    /// bump it only on the in-memory writer *after* `update_document` had
    /// already written its `segments_N`, so the committed counter was stale: a
    /// writer reopened on the same directory handed out the same `_N` again and
    /// the next flush truncated a live segment's files out from under the
    /// commit that referenced them.
    #[test]
    fn update_document_persists_the_bumped_segment_counter_so_a_reopen_never_reuses_a_name() {
        let fx = open_fixture();
        let tmp = tempdir("update-doc-counter");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = writer_seeded_with_fixture(&dir, &fx, fields.clone());

        let sources = [SegmentDeleteSource {
            segment_name: "_0",
            fields: &fx.fields,
            doc_in: None,
            live_docs: None,
            max_doc: fx.max_doc,
        }];
        let sis = writer
            .update_document_with_sources(&sources, "id", b"id0", doc("replacement"))
            .unwrap()
            .clone();
        let new_name = sis
            .segments
            .iter()
            .find(|s| s.segment_name != "_0")
            .unwrap()
            .segment_name
            .clone();

        let committed = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(
            committed.counter, sis.counter,
            "the committed counter must match the writer's in-memory one"
        );

        // A fresh writer over the same directory must not hand out `new_name`
        // again.
        let mut resumed = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        assert_ne!(
            resumed.new_segment_name(),
            new_name,
            "a reopened writer reused the segment name update_document just flushed"
        );
        resumed.add_document(doc("later")).unwrap();
        let after = resumed.commit().unwrap().clone();
        let names: Vec<&str> = after
            .segments
            .iter()
            .map(|s| s.segment_name.as_str())
            .collect();
        assert_eq!(
            names.len(),
            3,
            "the reopened writer's flush must be a third, distinct segment: {names:?}"
        );
        // The replacement document is still readable, i.e. its files were not
        // overwritten by the resumed writer's flush.
        let replacement = after
            .segments
            .iter()
            .find(|s| s.segment_name == new_name)
            .unwrap();
        let fdt = dir.open(&format!("{new_name}.fdt")).unwrap();
        let fdx = dir.open(&format!("{new_name}.fdx")).unwrap();
        let fdm = dir.open(&format!("{new_name}.fdm")).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &replacement.segment_id, "").unwrap();
        assert_eq!(doc_value(&reader.document(0).unwrap()), "replacement");
    }

    #[test]
    fn delete_documents_marks_matching_docs_dead_and_is_visible_after_commit() {
        let fx = open_fixture();
        let doc_in = fx.doc_in();
        let tmp = tempdir("delete-doc");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);

        // `writer_seeded_with_fixture` only seeds this writer's in-memory
        // `SegmentCommitInfo` -- it never writes a real `_0.si` to `dir`
        // (the fixture's real files are `.tim`/`.tip`/`.tmd`/`.doc` only).
        // `committed_doc_count()` reads doc counts straight off each
        // segment's `.si` file, so write a minimal one here for `_0` --
        // same pattern `check_index.rs`'s own fixture setup uses.
        std::fs::write(
            tmp.join("_0.si"),
            segment_info::write(
                &segment_info::SegmentInfo {
                    id: fx.segment_id,
                    version: LuceneVersion {
                        major: 10,
                        minor: 0,
                        bugfix: 0,
                    },
                    min_version: None,
                    doc_count: fx.max_doc as i32,
                    is_compound_file: false,
                    has_blocks: false,
                    diagnostics: vec![],
                    files: vec![],
                    attributes: vec![],
                    index_sort: None,
                },
                "",
            ),
        )
        .unwrap();

        let sources = [SegmentDeleteSource {
            segment_name: "_0",
            fields: &fx.fields,
            doc_in: Some(&doc_in),
            live_docs: None,
            max_doc: fx.max_doc,
        }];

        // "body" -> "cat" matches docs [0, 2] per the checked-in fixture
        // (same contents `term_delete.rs`'s own tests document).
        let sis = writer
            .delete_documents_with_sources(&sources, "body", b"cat")
            .unwrap()
            .clone();
        assert_eq!(sis.segments.len(), 1);
        assert_eq!(sis.segments[0].del_count, 2);
        assert_eq!(sis.segments[0].del_gen, 1);

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.segments[0].del_count, 2);

        let liv = dir.open("_0_1.liv").unwrap();
        let parsed =
            lucene_codecs::live_docs::parse(&liv, &fx.segment_id, 1, fx.max_doc, 2).unwrap();
        assert!(!parsed.get(0));
        assert!(parsed.get(1));
        assert!(!parsed.get(2));

        // `committed_doc_count()` is total-including-deleted, not live: the
        // segment's `.si` doc_count (the fixture's full `max_doc`) is
        // unchanged by a delete -- only a merge ever actually drops a
        // deleted doc from disk (see that method's doc comment) -- so the
        // count stays at `fx.max_doc` even though 2 of those docs are now
        // dead.
        assert_eq!(writer.committed_doc_count().unwrap(), fx.max_doc);
    }

    #[test]
    fn delete_documents_with_no_matching_source_leaves_segment_untouched() {
        let fx = open_fixture();
        let tmp = tempdir("delete-doc-no-match");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);

        // No source targets "_0" at all, so the segment must pass through
        // unmodified (no .liv written, del_count stays 0) even though
        // segment_infos itself still advances a generation/version.
        let sources: [SegmentDeleteSource; 0] = [];
        let sis = writer
            .delete_documents_with_sources(&sources, "body", b"cat")
            .unwrap()
            .clone();
        assert_eq!(sis.segments.len(), 1);
        assert_eq!(sis.segments[0].segment_name, "_0");
        assert_eq!(sis.segments[0].del_count, 0);
        assert_eq!(sis.segments[0].del_gen, -1);
    }

    #[test]
    fn a_failing_update_document_leaves_the_writer_state_unchanged() {
        let fx = open_fixture();
        let doc_in = fx.doc_in();
        let tmp = tempdir("update-fail");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);
        let before = writer.segment_infos().clone();

        let sources = [SegmentDeleteSource {
            segment_name: "_0",
            fields: &fx.fields,
            doc_in: Some(&doc_in),
            live_docs: None,
            max_doc: 1, // bogus: doc id 2 (from "cat") is out of range
        }];

        let result = writer.update_document_with_sources(&sources, "body", b"cat", doc("nope"));
        assert!(result.is_err());
        assert!(!tmp.join("segments_1").exists());

        // Everything a reader could observe is unchanged...
        let after = writer.segment_infos().clone();
        assert_eq!(after.segments, before.segments);
        assert_eq!(after.generation, before.generation);
        assert_eq!(after.version, before.version);
        assert_eq!(after.id, before.id);

        // ...but the segment name this attempt consumed is gone for good. Real
        // `IndexWriter.newSegmentName()` bumps `segmentInfos.counter` in the
        // same synchronized block that hands the name out, precisely so a name
        // can never be handed out twice -- a failed attempt burning a name is
        // the intended, safe direction (reusing it would let the next flush
        // create_output over files a previous attempt may have left behind).
        assert_eq!(
            after.counter,
            before.counter + 1,
            "a failed update_document must not make its segment name reusable"
        );
    }

    /// A tight [`MergePolicyConfig`] whose threshold (`segments_per_tier`)
    /// this test suite deliberately crosses/stays-under, so `commit()`'s
    /// automatic-merge behavior is exercised deterministically rather than
    /// relying on the (much larger) real-Lucene-shaped defaults.
    fn tight_merge_policy() -> MergePolicyConfig {
        MergePolicyConfig {
            max_merge_at_once: 10,
            segments_per_tier: 2,
            max_merged_segment_size: 1_000_000,
            reclaim_weight: 1.0,
            // Above every segment these tests flush, mirroring real Lucene's
            // 16MB `floorSegmentBytes` sitting far above a freshly-flushed
            // tiny segment: the level walk in `find_merges` then budgets one
            // level only, so `segments_per_tier` alone decides when a merge
            // fires. With a zero floor the budget instead grows level by
            // level and three tiny segments are considered in-budget.
            floor_segment_size: 100_000,
            force_merge_deletes_pct_allowed: 10.0,
            ..MergePolicyConfig::default()
        }
    }

    #[test]
    fn commit_with_no_merge_policy_set_never_auto_merges() {
        let tmp = tempdir("no-merge-policy");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        for doc_id in 0..5 {
            writer.add_document(doc(&doc_id.to_string())).unwrap();
            writer.commit().unwrap();
        }

        // 5 commits, no merge policy set => still 5 independent segments,
        // exactly as before automatic merge triggering existed.
        assert_eq!(writer.segment_infos().segments.len(), 5);
    }

    #[test]
    fn commit_below_merge_threshold_stays_unmerged() {
        let tmp = tempdir("below-threshold");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        // tight_merge_policy's segments_per_tier is 2, so 2 commits (2
        // segments) must stay below/at threshold and remain unmerged.
        writer.add_document(doc("a")).unwrap();
        writer.commit().unwrap();
        writer.add_document(doc("b")).unwrap();
        writer.commit().unwrap();

        assert_eq!(writer.segment_infos().segments.len(), 2);
        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.segments.len(), 2);
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b"]);
    }

    #[test]
    fn commit_above_merge_threshold_automatically_merges_and_stays_readable() {
        let tmp = tempdir("above-threshold");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        // tight_merge_policy's segments_per_tier is 2; crossing it (5
        // one-document commits) must trigger at least one automatic merge,
        // ending with fewer segments than commits.
        let ids = ["a", "b", "c", "d", "e"];
        for id in ids {
            writer.add_document(doc(id)).unwrap();
            writer.commit().unwrap();
        }

        let final_count = writer.segment_infos().segments.len();
        assert!(
            final_count < ids.len(),
            "expected automatic merging to reduce segment count below {}, got {final_count}",
            ids.len()
        );

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.segments.len(), final_count);
        let mut docs = read_all_docs(&dir, &reopened);
        docs.sort();
        assert_eq!(docs, vec!["a", "b", "c", "d", "e"]);
    }

    #[test]
    fn repeated_commits_with_auto_merge_converge_without_panicking_or_looping_forever() {
        let tmp = tempdir("converge");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        // Many small single-document commits in a row: each commit() call
        // must return (no infinite auto-merge loop), and the segment count
        // must never run away unboundedly.
        for i in 0..20 {
            writer.add_document(doc(&i.to_string())).unwrap();
            writer.commit().unwrap();
            assert!(
                writer.segment_infos().segments.len() <= 20,
                "segment count should never exceed the number of commits made so far"
            );
        }

        let reopened = segment_infos::read_latest(&dir).unwrap();
        let mut docs = read_all_docs(&dir, &reopened);
        docs.sort();
        let mut expected: Vec<String> = (0..20).map(|i| i.to_string()).collect();
        expected.sort();
        assert_eq!(docs, expected);
    }

    #[test]
    fn auto_merge_correctly_carries_forward_a_segments_existing_deletions() {
        let tmp = tempdir("auto-merge-with-deletions");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();

        // Two ordinary commits (no merge policy yet), so segment "_0" has a
        // real, on-disk, flushed segment to apply a deletion to directly via
        // `deletes::apply_deletes` (the same primitive `delete_documents`
        // itself calls), independent of term resolution.
        writer.add_document(doc("a")).unwrap();
        writer.add_document(doc("b")).unwrap();
        writer.commit().unwrap();

        let sci = writer.segment_infos().segments[0].clone();
        assert_eq!(sci.segment_name, "_0");
        let fdt = dir.open("_0.fdt").unwrap();
        let fdx = dir.open("_0.fdx").unwrap();
        let fdm = dir.open("_0.fdm").unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        let max_doc = reader.max_doc() as usize;

        // Delete doc 0 ("a") directly, bypassing term resolution entirely --
        // this is exactly what `execute_merge`'s `sci.del_gen >= 0` branch
        // must read back correctly during an automatic merge.
        let updated_sci = deletes::apply_deletes(&dir, &sci, None, max_doc, [0]).unwrap();
        assert_eq!(updated_sci.del_gen, 1);
        assert_eq!(updated_sci.del_count, 1);

        let mut new_segment_infos = writer.segment_infos().clone();
        new_segment_infos.segments[0] = updated_sci;
        new_segment_infos.generation += 1;
        new_segment_infos.version += 1;
        segment_infos::write(&new_segment_infos, &dir).unwrap();

        // Reopen the writer against this on-disk state (one segment with a
        // real deletion already applied), enable the merge policy, and cross
        // its threshold so the deleted segment gets folded into an automatic
        // merge.
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));
        writer.add_document(doc("c")).unwrap();
        writer.commit().unwrap();
        writer.add_document(doc("d")).unwrap();
        writer.commit().unwrap();

        let final_count = writer.segment_infos().segments.len();
        assert!(
            final_count < 3,
            "expected the deleted-doc segment to merge away, got {final_count} segments"
        );

        let reopened = segment_infos::read_latest(&dir).unwrap();
        let mut docs = read_all_docs(&dir, &reopened);
        docs.sort();
        // "a" was deleted before the merge, so only "b", "c", "d" survive.
        assert_eq!(docs, vec!["b", "c", "d"]);
    }

    /// A [`Directory`] that records every `create_output`, `open` and `sync`
    /// by name -- the only way to see how many times one commit rewrites the
    /// same file, which no assertion on the resulting bytes can.
    struct CountingDirectory<'a> {
        inner: &'a FsDirectory,
        created: std::cell::RefCell<Vec<String>>,
        opened: std::cell::RefCell<Vec<String>>,
        synced: std::cell::RefCell<Vec<String>>,
    }

    impl<'a> CountingDirectory<'a> {
        fn new(inner: &'a FsDirectory) -> Self {
            CountingDirectory {
                inner,
                created: std::cell::RefCell::new(Vec::new()),
                opened: std::cell::RefCell::new(Vec::new()),
                synced: std::cell::RefCell::new(Vec::new()),
            }
        }
        fn count(list: &std::cell::RefCell<Vec<String>>, name: &str) -> usize {
            list.borrow().iter().filter(|n| n.as_str() == name).count()
        }
    }

    impl Directory for CountingDirectory<'_> {
        fn list_all(&self) -> lucene_store::Result<Vec<String>> {
            self.inner.list_all()
        }
        fn open(&self, name: &str) -> lucene_store::Result<lucene_store::directory::Input> {
            self.opened.borrow_mut().push(name.to_string());
            self.inner.open(name)
        }
        fn create_output(&self, name: &str) -> lucene_store::Result<lucene_store::FsIndexOutput> {
            self.created.borrow_mut().push(name.to_string());
            self.inner.create_output(name)
        }
        fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
            self.synced.borrow_mut().extend(names.iter().cloned());
            self.inner.sync(names)
        }
        fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
            self.inner.rename(source, dest)
        }
        fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
            self.inner.delete_file(name)
        }
        fn sync_meta_data(&self) -> lucene_store::Result<()> {
            self.inner.sync_meta_data()
        }
    }

    /// `IndexWriter.sealFlushedSegment` writes a segment's `.si` **once**,
    /// from the in-memory `SegmentInfo` every format's writer has added its
    /// files to. This port used to write it once per file group: the
    /// stored-fields flush, then postings, then term vectors, then doc
    /// values, then norms, then the index-sort descriptor each re-opened,
    /// re-parsed, extended, rewrote and re-fsynced it. Only the last write
    /// survived.
    ///
    /// Counted rather than inferred: every intermediate rewrite produces the
    /// same final bytes, so nothing about the resulting segment can see them.
    #[test]
    fn one_commit_writes_the_segments_si_exactly_once() {
        let tmp = tempdir("si-written-once");
        let fs = FsDirectory::open(&tmp);
        let dir = CountingDirectory::new(&fs);
        let fields = vec![
            stored_only_field("id", 0),
            // Indexed, not omitting norms, and advertising term vectors ->
            // postings *and* norms *and* term vectors; the numeric column
            // below adds doc values. Four of the five per-format file groups
            // in one commit, on top of the stored-fields flush itself.
            FieldInfo {
                store_term_vectors: true,
                ..body_field(1)
            },
            numeric_dv_field("num", 2),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();
        writer.set_doc_values_field(Some("num")).unwrap();

        for i in 0..8 {
            writer
                .add_document(Document {
                    fields: vec![
                        StoredField {
                            field_number: 0,
                            value: FieldValue::String(format!("d{i}")),
                        },
                        StoredField {
                            field_number: 1,
                            value: FieldValue::String(format!("term{i} shared")),
                        },
                        StoredField {
                            field_number: 2,
                            value: FieldValue::Long(i as i64),
                        },
                    ],
                })
                .unwrap();
        }
        writer.commit().unwrap();

        // Every format really did run, so the count below is over the full
        // set of writers, not a degenerate stored-fields-only flush.
        let created = dir.created.borrow().clone();
        for ext in [".fdt", ".tim", ".tvd", ".dvd", ".nvd"] {
            assert!(
                created.iter().any(|n| n.ends_with(ext)),
                "no {ext} written; the counted commit did not exercise every format: {created:?}"
            );
        }

        assert_eq!(
            CountingDirectory::count(&dir.created, "_0.si"),
            1,
            "the segment's .si was written more than once: {created:?}"
        );
        // **Zero** reads. c36 removed the read-modify-write per file group
        // during the flush and left one: `IndexFileDeleter`'s checkpoint
        // re-opened and re-parsed the finished `.si` (header check and CRC
        // over the whole file included) to reference-count the segment's
        // files. Java never does that -- `SegmentCommitInfo` holds its
        // `SegmentInfo`, so `SegmentInfos.files()` reads the list out of
        // memory -- and since `c43-final-cleanup` neither does this port:
        // `flush` hands the deleter the same in-memory list
        // `seal_flushed_segment` encoded
        // (`IndexFileDeleter::record_segment_files`).
        assert_eq!(
            CountingDirectory::count(&dir.opened, "_0.si"),
            0,
            "the segment's .si was read back during its own flush: {:?}",
            dir.opened.borrow()
        );
        assert_eq!(
            CountingDirectory::count(&dir.synced, "_0.si"),
            1,
            "the segment's .si was fsynced more than once"
        );

        // ...and the one `.si` that was written still lists every file.
        let sci = &writer.segment_infos().segments[0];
        let si_bytes = fs.open("_0.si").unwrap().to_vec();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for ext in [
            ".fdt", ".fdx", ".fdm", ".fnm", ".si", ".tim", ".tvd", ".dvd", ".nvd",
        ] {
            assert!(
                si.files.iter().any(|f| f.ends_with(ext)),
                "the single .si write lost {ext}: {:?}",
                si.files
            );
        }
    }

    /// `IndexWriter.mergeMiddle`: a merge whose result holds no live
    /// document writes nothing and drops both the result and its sources
    /// ("Merge would produce a 0-doc segment, so we do nothing except commit
    /// the merge to remove all the 0-doc segments that we merged"). This port
    /// used to run the merge and publish the empty segment, so the commit
    /// carried a real zero-document segment that every later open, merge and
    /// `CheckIndex` had to pay for.
    #[test]
    fn a_merge_whose_sources_are_all_deleted_is_dropped_not_committed() {
        let tmp = tempdir("zero-doc-merge-dropped");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();

        // Three one-document segments, then every document deleted through
        // `deletes::apply_deletes` (the field is stored-only, so a term
        // delete has no postings to resolve against). All three `.liv` files
        // are on disk and all three segments are 100% deleted before any
        // merge is proposed.
        for id in ["a", "b", "c"] {
            writer.add_document(doc(id)).unwrap();
            writer.commit().unwrap();
        }
        let mut infos = writer.segment_infos().clone();
        assert_eq!(infos.segments.len(), 3);
        for i in 0..infos.segments.len() {
            let sci = infos.segments[i].clone();
            infos.segments[i] = deletes::apply_deletes(&dir, &sci, None, 1, [0]).unwrap();
        }
        infos.generation += 1;
        infos.version += 1;
        segment_infos::write(&infos, &dir).unwrap();

        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));
        writer.commit().unwrap();

        let committed = segment_infos::read_latest(&dir).unwrap();
        // Nothing is left: the sources were retired and no empty segment took
        // their place. Before the fix the commit named one brand-new
        // zero-document segment instead.
        assert!(
            committed.segments.is_empty(),
            "expected the fully-deleted sources to be dropped with nothing published, got {:?}",
            committed
                .segments
                .iter()
                .map(|s| s.segment_name.clone())
                .collect::<Vec<_>>()
        );
        assert!(read_all_docs(&dir, &committed).is_empty());
        // And no files were written for the merge that did not happen.
        let leftovers: Vec<String> = index_files(&dir)
            .into_iter()
            .filter(|f| !f.starts_with("segments"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "a dropped merge left files behind: {leftovers:?}"
        );
    }

    #[test]
    fn apply_merge_folds_a_merge_result_into_the_writers_committed_state() {
        let tmp = tempdir("apply-merge");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.commit().unwrap();
        writer.add_document(doc("b")).unwrap();
        writer.commit().unwrap();
        assert_eq!(writer.segment_infos().segments.len(), 2);

        let seg0_name = writer.segment_infos().segments[0].segment_name.clone();
        let seg1_name = writer.segment_infos().segments[1].segment_name.clone();

        let fdt0 = dir.open(&format!("{seg0_name}.fdt")).unwrap().to_vec();
        let fdx0 = dir.open(&format!("{seg0_name}.fdx")).unwrap().to_vec();
        let fdm0 = dir.open(&format!("{seg0_name}.fdm")).unwrap().to_vec();
        let fdt1 = dir.open(&format!("{seg1_name}.fdt")).unwrap().to_vec();
        let fdx1 = dir.open(&format!("{seg1_name}.fdx")).unwrap().to_vec();
        let fdm1 = dir.open(&format!("{seg1_name}.fdm")).unwrap().to_vec();

        // Segment ids are generated internally, so re-derive them from the
        // committed SegmentInfos rather than hard-coding a value.
        let seg0_id = writer.segment_infos().segments[0].segment_id;
        let seg1_id = writer.segment_infos().segments[1].segment_id;
        let reader0 = stored_fields::open(&fdt0, &fdx0, &fdm0, &seg0_id, "").unwrap();
        let reader1 = stored_fields::open(&fdt1, &fdx1, &fdm1, &seg1_id, "").unwrap();

        let sources = vec![
            merge::MergeSource::stored_only(&fields, &reader0, None, Some(version())),
            merge::MergeSource::stored_only(&fields, &reader1, None, Some(version())),
        ];
        let merged_sci = merge::merge_stored_only_segments(
            &dir,
            &sources,
            "_merged",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let sis = writer
            .apply_merge(&[seg0_name.as_str(), seg1_name.as_str()], merged_sci)
            .unwrap()
            .clone();
        assert_eq!(sis.segments.len(), 1);
        assert_eq!(sis.segments[0].segment_name, "_merged");

        let reopened = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(reopened.segments.len(), 1);
        assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b"]);
    }

    // --- set_postings_field / commit()'s postings-writing path ---

    fn body_field(number: i32) -> FieldInfo {
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            ..stored_only_field("body", number)
        }
    }

    fn doc_with_body(id: &str, body: &str) -> Document {
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(id.to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String(body.to_string()),
                },
            ],
        }
    }

    #[test]
    fn set_postings_field_rejects_an_unknown_field_name() {
        let tmp = tempdir("unknown-postings-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer.set_postings_field(Some("nonexistent")).unwrap_err();
        assert!(matches!(err, Error::UnknownPostingsField(name) if name == "nonexistent"));
    }

    #[test]
    fn set_postings_field_rejects_a_field_with_no_index_options() {
        let tmp = tempdir("unindexed-postings-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer.set_postings_field(Some("id")).unwrap_err();
        assert!(matches!(
            err,
            Error::UnsupportedPostingsIndexOptions(name, IndexOptions::None) if name == "id"
        ));
    }

    #[test]
    fn add_postings_field_rejects_an_unknown_field_name() {
        let tmp = tempdir("unknown-add-postings-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        let err = writer.add_postings_field("nonexistent").unwrap_err();
        assert!(matches!(err, Error::UnknownPostingsField(name) if name == "nonexistent"));
    }

    #[test]
    fn add_postings_field_rejects_a_duplicate_field() {
        let tmp = tempdir("duplicate-add-postings-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        let err = writer.add_postings_field("body").unwrap_err();
        assert!(matches!(err, Error::DuplicatePostingsField(name) if name == "body"));
    }

    #[test]
    fn add_postings_field_accumulates_on_top_of_set_postings_field() {
        let tmp = tempdir("accumulate-add-postings-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            body_field(1),
            FieldInfo {
                index_options: IndexOptions::DocsAndFreqs,
                ..stored_only_field("title", 2)
            },
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.add_postings_field("title").unwrap();

        assert_eq!(writer.postings_fields.len(), 2);
        assert_eq!(writer.postings_fields[0].name, "body");
        assert_eq!(writer.postings_fields[1].name, "title");

        // `set_postings_field(None)` still clears the whole accumulated list,
        // not just the last entry.
        writer.set_postings_field(None).unwrap();
        assert!(writer.postings_fields.is_empty());
    }

    #[test]
    fn commit_with_postings_field_writes_readable_postings_for_multiple_docs_and_terms() {
        let tmp = tempdir("postings-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        writer
            .add_document(doc_with_body("a", "the quick fox"))
            .unwrap();
        writer
            .add_document(doc_with_body("b", "the lazy fox"))
            .unwrap();
        writer
            .add_document(doc_with_body("c", "the fox runs"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];

        // Stored fields are still intact (backward-compatible).
        assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b", "c"]);

        // The postings files exist and are listed in `.si`.
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for ext in ["doc", "tim", "tip", "tmd"] {
            let name = format!(
                "{}.{ext}",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            );
            assert!(si.files.contains(&name), "missing {name} in .si files");
            assert!(
                dir.list_all().unwrap().contains(&name),
                "missing {name} on disk"
            );
        }

        // Readable via the existing, unmodified read side: `fox` occurs in
        // all 3 docs, `quick`/`lazy`/`runs` are singletons, `the` occurs in
        // all 3 too but is not a singleton either.
        let tim = dir
            .open(&format!(
                "{}.tim",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let tip = dir
            .open(&format!(
                "{}.tip",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let tmd = dir
            .open(&format!(
                "{}.tmd",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let doc_bytes = dir
            .open(&format!(
                "{}.doc",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![
                fi::FieldInfo {
                    index_options: IndexOptions::None,
                    ..stored_only_field("id", 0)
                },
                body_field(1),
            ],
        };
        let block_fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            3,
        )
        .expect("blocktree::open on IndexWriter-produced .tim/.tip/.tmd");
        let doc_in = DocInput::open(
            &doc_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .expect("open .doc");
        let field = block_fields.field("body").unwrap();

        let postings = field.postings(b"fox", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0, 1, 2]);
        let postings = field.postings(b"the", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0, 1, 2]);
        let postings = field.postings(b"quick", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0]);
        let postings = field.postings(b"lazy", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![1]);
        let postings = field.postings(b"runs", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![2]);
        assert!(field.seek_exact(b"missing").is_none());
    }

    /// Task #211's headline proof at the `lucene-index` level (the fuller
    /// `PhraseQuery`-based proof lives in
    /// `lucene-search/src/directory_reader.rs`'s `phrase_query_e2e` module,
    /// since `lucene-index` can't depend on `lucene-search`): a field opted
    /// into `IndexOptions::DocsAndFreqsAndPositions` postings via
    /// `set_postings_field` produces a real `.pos` file (registered in `.si`,
    /// present on disk), and `blocktree::FieldTerms::positions` -- the same
    /// read path a query layer would use -- decodes each doc's occurrences
    /// back out correctly, not just the doc-ID/freq shape
    /// `commit_with_postings_field_writes_readable_postings_for_multiple_docs_and_terms`
    /// already covers for `DocsAndFreqs`.
    #[test]
    fn commit_with_positions_index_options_writes_a_readable_pos_file() {
        let tmp = tempdir("postings-positions-commit");
        let dir = FsDirectory::open(&tmp);
        let positions_body_field = fi::FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            ..stored_only_field("body", 1)
        };
        let fields = vec![stored_only_field("id", 0), positions_body_field.clone()];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        writer
            .add_document(doc_with_body("a", "quick fox jumps"))
            .unwrap();
        writer
            .add_document(doc_with_body("b", "the fox sleeps"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        let pos_name = format!(
            "{}.pos",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        );
        assert!(
            si.files.contains(&pos_name),
            "missing {pos_name} in .si files"
        );
        assert!(
            dir.list_all().unwrap().contains(&pos_name),
            "missing {pos_name} on disk"
        );
        // No offsets were requested, so `.pay` must not exist at all.
        assert!(!si.files.iter().any(|f| f.ends_with(".pay")));

        let tim = dir
            .open(&format!(
                "{}.tim",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let tip = dir
            .open(&format!(
                "{}.tip",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let tmd = dir
            .open(&format!(
                "{}.tmd",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let doc_bytes = dir
            .open(&format!(
                "{}.doc",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let pos_bytes = dir.open(&pos_name).unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![
                fi::FieldInfo {
                    index_options: IndexOptions::None,
                    ..stored_only_field("id", 0)
                },
                positions_body_field,
            ],
        };
        let block_fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            2,
        )
        .expect("blocktree::open on IndexWriter-produced .tim/.tip/.tmd");
        let doc_in = DocInput::open(
            &doc_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .expect("open .doc");
        let pos_in = lucene_codecs::postings::PosInput::open(
            &pos_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .expect("open .pos");
        let field = block_fields.field("body").unwrap();

        // "fox" occurs in both docs: doc 0 at position 1 ("quick fox
        // jumps"), doc 1 at position 1 ("the fox sleeps").
        let fox_positions = field
            .positions(b"fox", Some(&doc_in), &pos_in, None)
            .unwrap()
            .unwrap();
        assert_eq!(fox_positions.len(), 2);
        assert_eq!(fox_positions[0].len(), 1);
        assert_eq!(fox_positions[0][0].position, 1);
        assert_eq!(fox_positions[1].len(), 1);
        assert_eq!(fox_positions[1][0].position, 1);

        // "quick" is a doc-0-only singleton at position 0.
        let quick_positions = field
            .positions(b"quick", Some(&doc_in), &pos_in, None)
            .unwrap()
            .unwrap();
        assert_eq!(quick_positions.len(), 1);
        assert_eq!(
            quick_positions[0],
            vec![lucene_codecs::postings::Position {
                position: 0,
                start_offset: -1,
                end_offset: -1,
                payload: Vec::new(),
            }]
        );
    }

    /// Same shape as
    /// `commit_with_positions_index_options_writes_a_readable_pos_file`, but
    /// with `IndexOptions::DocsAndFreqsAndPositionsAndOffsets` -- proves the
    /// `has_offsets` branch in `IndexWriter::build_postings_output` also
    /// feeds real character offsets through to a readable `.pos`/`.si` file
    /// set (small enough here to stay inline in `.pos` rather than needing a
    /// `.pay` file at all -- see `blocktree::FieldTerms::positions`'s doc
    /// comment on when `.pay` is actually required).
    #[test]
    fn commit_with_offsets_index_options_writes_readable_offsets() {
        let tmp = tempdir("postings-offsets-commit");
        let dir = FsDirectory::open(&tmp);
        let offsets_body_field = fi::FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            ..stored_only_field("body", 1)
        };
        let fields = vec![stored_only_field("id", 0), offsets_body_field.clone()];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        writer
            .add_document(doc_with_body("a", "quick fox"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert!(si.files.iter().any(|f| f.ends_with(".pos")));

        let tim = dir
            .open(&format!(
                "{}.tim",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let tip = dir
            .open(&format!(
                "{}.tip",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let tmd = dir
            .open(&format!(
                "{}.tmd",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let doc_bytes = dir
            .open(&format!(
                "{}.doc",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let pos_bytes = dir
            .open(&format!(
                "{}.pos",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![
                fi::FieldInfo {
                    index_options: IndexOptions::None,
                    ..stored_only_field("id", 0)
                },
                offsets_body_field,
            ],
        };
        let block_fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            1,
        )
        .expect("blocktree::open on IndexWriter-produced .tim/.tip/.tmd");
        let doc_in = DocInput::open(
            &doc_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .expect("open .doc");
        let pos_in = lucene_codecs::postings::PosInput::open(
            &pos_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .expect("open .pos");
        let field = block_fields.field("body").unwrap();

        // "quick fox" -> "quick" spans bytes [0, 5), "fox" spans [6, 9).
        let fox_positions = field
            .positions(b"fox", Some(&doc_in), &pos_in, None)
            .unwrap()
            .unwrap();
        assert_eq!(fox_positions[0][0].start_offset, 6);
        assert_eq!(fox_positions[0][0].end_offset, 9);
        let quick_positions = field
            .positions(b"quick", Some(&doc_in), &pos_in, None)
            .unwrap()
            .unwrap();
        assert_eq!(quick_positions[0][0].start_offset, 0);
        assert_eq!(quick_positions[0][0].end_offset, 5);
    }

    /// A helper for the payload tests below: opens `.tim`/`.tip`/`.tmd`,
    /// `.doc`, `.pos` and (when present) `.pay` from a committed segment and
    /// returns each document's occurrences for one term of one field, read
    /// through the same path a query layer would take.
    fn read_occurrences(
        dir: &FsDirectory,
        sci: &SegmentCommitInfo,
        field_infos: &fi::FieldInfos,
        field: &str,
        term: &[u8],
        max_doc: i32,
    ) -> Vec<Vec<lucene_codecs::postings::Position>> {
        let seg = per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
        let open = |ext: &str| dir.open(&format!("{seg}.{ext}"));
        let tim = open("tim").unwrap();
        let tip = open("tip").unwrap();
        let tmd = open("tmd").unwrap();
        let doc_bytes = open("doc").unwrap();
        let pos_bytes = open("pos").unwrap();
        let pay_bytes = open("pay").ok();
        let suffix = per_field_codec_suffix(POSTINGS_FORMAT_NAME);
        let block_fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            field_infos,
            &sci.segment_id,
            &suffix,
            max_doc,
        )
        .expect("blocktree::open");
        let doc_in = DocInput::open(&doc_bytes, &sci.segment_id, &suffix).expect("open .doc");
        let pos_in = lucene_codecs::postings::PosInput::open(&pos_bytes, &sci.segment_id, &suffix)
            .expect("open .pos");
        let pay_in = pay_bytes.as_ref().map(|b| {
            lucene_codecs::postings::PayInput::open(b, &sci.segment_id, &suffix).expect("open .pay")
        });
        block_fields
            .field(field)
            .unwrap()
            .positions(term, Some(&doc_in), &pos_in, pay_in.as_ref())
            .unwrap()
            .unwrap()
    }

    /// A `store_payloads` field whose `index_options` indexes positions but
    /// **not** offsets is the shape that used to be unwritable: `.pay` exists
    /// only when a field has offsets or payloads, and before this batch
    /// `build_postings_output` hardcoded `has_payloads: false`, so the `.fnm`
    /// promised payloads real Lucene would then look for in a `.pay` that was
    /// never written. Lucene's `Lucene104PostingsReader` opens `.pay`
    /// unconditionally when `fieldInfos.hasPayloads()`, so that segment could
    /// not be opened at all.
    ///
    /// The negative control is the sibling test below: the same field without
    /// `store_payloads` must produce no `.pay` at all, so this test cannot
    /// pass merely because something always writes one.
    #[test]
    fn a_payloads_field_without_offsets_writes_a_pay_file_and_round_trips_its_bytes() {
        let tmp = tempdir("postings-payloads-no-offsets");
        let dir = FsDirectory::open(&tmp);
        let body = fi::FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            store_payloads: true,
            ..stored_only_field("body", 1)
        };
        let fields = vec![stored_only_field("id", 0), body.clone()];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer
            .set_payload_source(Some(Box::new(|ctx| {
                Some(format!("{}@{}", ctx.term, ctx.position).into_bytes())
            })))
            .unwrap();

        writer
            .add_document(doc_with_body("a", "quick fox jumps"))
            .unwrap();
        writer
            .add_document(doc_with_body("b", "the fox sleeps"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        let pay_name = format!(
            "{}.pay",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        );
        assert!(
            si.files.contains(&pay_name),
            "a store_payloads field must produce a .pay even with no offsets: {:?}",
            si.files
        );
        assert!(dir.list_all().unwrap().contains(&pay_name));

        let field_infos = fi::FieldInfos {
            fields: vec![
                fi::FieldInfo {
                    index_options: IndexOptions::None,
                    ..stored_only_field("id", 0)
                },
                body,
            ],
        };
        let fox = read_occurrences(&dir, sci, &field_infos, "body", b"fox", 2);
        assert_eq!(fox.len(), 2);
        assert_eq!(fox[0][0].position, 1);
        assert_eq!(fox[0][0].payload, b"fox@1".to_vec());
        assert_eq!(fox[1][0].position, 1);
        assert_eq!(fox[1][0].payload, b"fox@1".to_vec());
        // Offsets are not indexed, so both are Lucene's -1 sentinel -- a
        // writer that emitted an offset region here would show up as a
        // decoded value instead.
        assert_eq!(fox[0][0].start_offset, -1);
        assert_eq!(fox[0][0].end_offset, -1);
    }

    /// The negative control for the test above: the same field shape with
    /// `store_payloads` off must produce **no** `.pay` file, and every
    /// occurrence must read back with an empty payload. Without this, "a .pay
    /// exists" would be evidence of nothing.
    #[test]
    fn a_positions_field_without_payloads_writes_no_pay_file() {
        let tmp = tempdir("postings-no-payloads");
        let dir = FsDirectory::open(&tmp);
        let body = fi::FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            ..stored_only_field("body", 1)
        };
        let fields = vec![stored_only_field("id", 0), body.clone()];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer
            .add_document(doc_with_body("a", "quick fox jumps"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert!(
            !si.files.iter().any(|f| f.ends_with(".pay")),
            "a field with neither offsets nor payloads needs no .pay: {:?}",
            si.files
        );

        let field_infos = fi::FieldInfos {
            fields: vec![
                fi::FieldInfo {
                    index_options: IndexOptions::None,
                    ..stored_only_field("id", 0)
                },
                body,
            ],
        };
        let fox = read_occurrences(&dir, sci, &field_infos, "body", b"fox", 1);
        assert!(fox[0][0].payload.is_empty());
    }

    /// The payload bytes have to be the ones the source produced, not merely
    /// *some* bytes: this indexes the same corpus twice with two different
    /// sources and requires the read-back payloads to differ. A writer that
    /// wrote a constant, or dropped the source and wrote zero-length payloads
    /// everywhere, passes every "a .pay exists" assertion and fails this one.
    #[test]
    fn the_payload_source_decides_the_bytes_that_land_on_disk() {
        let read_with = |tag: &str, prefix: &'static str| -> Vec<Vec<u8>> {
            let tmp = tempdir(tag);
            let dir = FsDirectory::open(&tmp);
            let body = fi::FieldInfo {
                index_options: IndexOptions::DocsAndFreqsAndPositions,
                store_payloads: true,
                ..stored_only_field("body", 1)
            };
            let fields = vec![stored_only_field("id", 0), body.clone()];
            let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
            writer.set_postings_field(Some("body")).unwrap();
            writer
                .set_payload_source(Some(Box::new(move |ctx| {
                    Some(format!("{prefix}{}", ctx.position).into_bytes())
                })))
                .unwrap();
            writer
                .add_document(doc_with_body("a", "quick fox jumps"))
                .unwrap();
            writer.commit().unwrap();
            let sis = writer.segment_infos().clone();
            let field_infos = fi::FieldInfos {
                fields: vec![
                    fi::FieldInfo {
                        index_options: IndexOptions::None,
                        ..stored_only_field("id", 0)
                    },
                    body,
                ],
            };
            read_occurrences(&dir, &sis.segments[0], &field_infos, "body", b"fox", 1)[0]
                .iter()
                .map(|p| p.payload.clone())
                .collect()
        };

        let a = read_with("payload-source-a", "a");
        let b = read_with("payload-source-b", "bb");
        assert_eq!(a, vec![b"a1".to_vec()]);
        assert_eq!(b, vec![b"bb1".to_vec()]);
        assert_ne!(a, b);
    }

    /// Java's `FieldInfo.checkConsistency`: "indexed field cannot have
    /// payloads without positions". Caught at `open` since c40 -- the writer
    /// can no longer *hold* the field, so `set_postings_field` never sees it.
    #[test]
    fn open_rejects_payloads_on_a_field_without_positions() {
        let tmp = tempdir("payloads-without-positions");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            fi::FieldInfo {
                index_options: IndexOptions::DocsAndFreqs,
                store_payloads: true,
                ..stored_only_field("body", 1)
            },
        ];
        let Err(err) = IndexWriter::open(&dir, fields, "Lucene104", version()) else {
            panic!("a field with payloads but no positions must not open a writer");
        };
        assert!(
            matches!(err, Error::FieldInfos(fi::Error::Inconsistent(ref name, msg))
                if name == "body" && msg.contains("payloads without positions")),
            "{err:?}"
        );
    }

    /// Installing a payload source when no declared field stores payloads
    /// would discard every payload it produced, silently. That is a caller
    /// mistake worth naming rather than a no-op.
    #[test]
    fn set_payload_source_rejects_a_writer_with_no_payload_field() {
        let tmp = tempdir("payload-source-no-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        assert!(matches!(
            writer
                .set_payload_source(Some(Box::new(|_| None)))
                .unwrap_err(),
            Error::NoPayloadFields
        ));
        // Clearing is always allowed, whatever the field list.
        writer.set_payload_source(None).unwrap();
    }

    /// A `store_payloads` field with no source installed still has to carry
    /// the payload-length stream its `.fnm` promises: real Lucene frames
    /// `.pay` off `FieldInfo.hasPayloads()` alone, so "declared but never
    /// used" must be an all-zero-length stream, not an absent one.
    #[test]
    fn a_payloads_field_with_no_source_still_writes_the_payload_length_stream() {
        let tmp = tempdir("payloads-no-source");
        let dir = FsDirectory::open(&tmp);
        let body = fi::FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            store_payloads: true,
            ..stored_only_field("body", 1)
        };
        let fields = vec![stored_only_field("id", 0), body.clone()];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer
            .add_document(doc_with_body("a", "quick fox jumps"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert!(si.files.iter().any(|f| f.ends_with(".pay")));

        let field_infos = fi::FieldInfos {
            fields: vec![
                fi::FieldInfo {
                    index_options: IndexOptions::None,
                    ..stored_only_field("id", 0)
                },
                body,
            ],
        };
        let fox = read_occurrences(&dir, sci, &field_infos, "body", b"fox", 1);
        assert!(fox[0][0].payload.is_empty());
    }

    /// Term vectors must carry the same axes the field's postings do. Before
    /// this batch `build_term_vectors_output` recorded positions only,
    /// whatever the field's options -- which made
    /// `CheckIndex.testTermVectors`' offset and payload cross-checks vacuous
    /// rather than failing, since it compares only the axes a vector declares.
    #[test]
    fn term_vectors_carry_the_offsets_and_payloads_the_field_declares() {
        let tmp = tempdir("tv-axes");
        let dir = FsDirectory::open(&tmp);
        let body = fi::FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            store_payloads: true,
            store_term_vectors: true,
            ..stored_only_field("body", 1)
        };
        // The control: same segment, a term-vector field that declares
        // neither axis, so "the vector has offsets" cannot be something the
        // writer does unconditionally.
        let title = fi::FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            store_term_vectors: true,
            ..stored_only_field("title", 2)
        };
        let fields = vec![stored_only_field("id", 0), body, title];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();
        writer.add_term_vector_field("title").unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer
            .set_payload_source(Some(Box::new(|ctx| {
                Some(vec![ctx.position as u8, ctx.term.len() as u8])
            })))
            .unwrap();
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String("quick fox".to_string()),
                    },
                    StoredField {
                        field_number: 2,
                        value: FieldValue::String("quick".to_string()),
                    },
                ],
            })
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let reader =
            lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
        let doc0 = reader.document(0).unwrap().unwrap();

        let body_vec = doc0
            .fields
            .iter()
            .find(|f| f.field_number == 1)
            .expect("body vector");
        assert!(body_vec.has_positions);
        assert!(body_vec.has_offsets, "body indexes offsets");
        assert!(body_vec.has_payloads, "body stores payloads");
        let fox = body_vec
            .terms
            .iter()
            .find(|t| t.term == b"fox")
            .expect("fox in the vector");
        assert_eq!(fox.positions.as_deref(), Some(&[1i32][..]));
        assert_eq!(fox.start_offsets.as_deref(), Some(&[6i32][..]));
        assert_eq!(fox.end_offsets.as_deref(), Some(&[9i32][..]));
        assert_eq!(fox.payloads.as_deref(), Some(&[vec![1u8, 3]][..]));

        let title_vec = doc0
            .fields
            .iter()
            .find(|f| f.field_number == 2)
            .expect("title vector");
        assert!(title_vec.has_positions);
        assert!(!title_vec.has_offsets, "title indexes no offsets");
        assert!(!title_vec.has_payloads, "title stores no payloads");
        assert!(title_vec.terms[0].start_offsets.is_none());
        assert!(title_vec.terms[0].payloads.is_none());
    }

    /// A document's term-vector fields must come out in ascending field-**name**
    /// order, whatever order the caller configured them in.
    ///
    /// Real Lucene's `CheckIndex.checkTermVectors` iterates `TVFields`, which
    /// yields the order the fields were written, and `checkFields` throws
    /// unless that order is sorted by name. The wire format carries field
    /// *numbers*, and this writer's numbers come from the caller's field list,
    /// so a caller who declares `zeta` as field 1 and `alpha` as field 2 --
    /// and calls `set_term_vector_field("zeta")` before
    /// `add_term_vector_field("alpha")` -- used to get a segment written in
    /// call order, i.e. `zeta` then `alpha`, which real `CheckIndex` rejects.
    ///
    /// The negative control is the field *numbers*: they must stay 1 and 2
    /// (the caller's own numbering), so this is a reordering of the per-doc
    /// field list, not a renumbering of the schema.
    #[test]
    fn term_vector_fields_are_written_in_ascending_field_name_order() {
        let tmp = tempdir("tv-field-order");
        let dir = FsDirectory::open(&tmp);
        // Name order and number order deliberately disagree: `zeta` is field
        // 1, `alpha` is field 2.
        let zeta = fi::FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            store_term_vectors: true,
            ..stored_only_field("zeta", 1)
        };
        let alpha = fi::FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            store_term_vectors: true,
            ..stored_only_field("alpha", 2)
        };
        let fields = vec![stored_only_field("id", 0), zeta, alpha];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        // Configured in the *wrong* order on purpose.
        writer.set_term_vector_field(Some("zeta")).unwrap();
        writer.add_term_vector_field("alpha").unwrap();
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String("quick".to_string()),
                    },
                    StoredField {
                        field_number: 2,
                        value: FieldValue::String("fox".to_string()),
                    },
                ],
            })
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let reader =
            lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
        let doc0 = reader.document(0).unwrap().unwrap();

        let numbers: Vec<i32> = doc0.fields.iter().map(|f| f.field_number).collect();
        assert_eq!(
            numbers,
            vec![2, 1],
            "the document's vectors must be ordered by field name (alpha=2, zeta=1), \
             not by configuration order"
        );
        assert_eq!(doc0.fields[0].terms[0].term, b"fox", "alpha's only term");
        assert_eq!(doc0.fields[1].terms[0].term, b"quick", "zeta's only term");
    }

    /// A `.fnm` this port writes must be one this port can re-open. Java makes
    /// the failing combination unrepresentable -- `FieldInfo`'s constructor
    /// coerces `storeTermVectors`/`storePayloads`/`omitNorms` to `false` for a
    /// non-indexed field before `checkConsistency` ever looks -- but a Rust
    /// `FieldInfo` is a plain struct, so a caller can hand `IndexWriter::open`
    /// a stored-only field with `omit_norms` set. Before c23 those bits went
    /// straight onto the wire: real Lucene coerced them away again on read
    /// (which is why every cross-engine verifier stayed green), and this
    /// port's own `field_infos::parse` rejected the file outright, so
    /// `check_index` could not open the `.fnm` of a segment the writer had
    /// just produced and every postings check was silently skipped.
    ///
    /// The negative control is the second half: the same three bits on an
    /// *indexed* field must survive, so the fix cannot be "always clear them".
    #[test]
    fn a_non_indexed_field_with_indexed_only_flags_still_writes_a_reopenable_fnm() {
        let tmp = tempdir("fnm-coercion");
        let dir = FsDirectory::open(&tmp);
        let stored_only_but_flagged = fi::FieldInfo {
            index_options: IndexOptions::None,
            omit_norms: true,
            ..stored_only_field("id", 0)
        };
        let indexed = fi::FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            omit_norms: true,
            store_payloads: true,
            store_term_vectors: true,
            ..stored_only_field("body", 1)
        };
        let fields = vec![stored_only_but_flagged, indexed];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();
        writer
            .add_document(doc_with_body("a", "quick fox"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();

        let fnm = dir
            .open(&format!("{}.fnm", sis.segments[0].segment_name))
            .unwrap();
        let parsed = fi::parse(&fnm, &sis.segments[0].segment_id, "")
            .expect("this port must be able to re-open the .fnm it just wrote");
        let id = parsed.fields.iter().find(|f| f.name == "id").unwrap();
        assert!(!id.omit_norms, "coerced away for a non-indexed field");
        assert!(!id.store_payloads);
        assert!(!id.store_term_vectors);
        let body = parsed.fields.iter().find(|f| f.name == "body").unwrap();
        assert!(body.omit_norms, "an indexed field keeps its flags");
        assert!(body.store_payloads);
        assert!(body.store_term_vectors);

        // And the whole segment passes this port's own CheckIndex, which is
        // what the bug was hiding: `fnm.open` failing skipped every postings
        // check in the segment.
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    #[test]
    fn commit_with_no_postings_field_configured_stays_stored_only() {
        // Backward compatibility: a writer that never calls
        // `set_postings_field` must produce exactly the same on-disk shape
        // as before this feature existed -- no `.doc`/`.tim`/`.tip`/`.tmd`
        // files at all.
        let tmp = tempdir("no-postings-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer
            .add_document(doc_with_body("a", "the quick fox"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        for ext in ["doc", "tim", "tip", "tmd"] {
            assert!(!files.contains(&format!("{}.{ext}", sci.segment_name)));
        }
    }

    fn custom_freq_field(number: i32) -> FieldInfo {
        FieldInfo {
            index_options: IndexOptions::DocsAndCustomFreqs,
            ..stored_only_field("score", number)
        }
    }

    #[test]
    fn set_custom_freq_postings_field_rejects_an_unknown_field_name() {
        let tmp = tempdir("unknown-custom-freq-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer
            .set_custom_freq_postings_field(Some("nonexistent"))
            .unwrap_err();
        assert!(
            matches!(err, Error::UnknownCustomFreqPostingsField(name) if name == "nonexistent")
        );
    }

    #[test]
    fn set_custom_freq_postings_field_rejects_a_field_with_wrong_index_options() {
        let tmp = tempdir("custom-freq-field-wrong-options");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer
            .set_custom_freq_postings_field(Some("body"))
            .unwrap_err();
        assert!(matches!(
            err,
            Error::UnsupportedCustomFreqPostingsIndexOptions(name, IndexOptions::DocsAndFreqs)
                if name == "body"
        ));
    }

    #[test]
    fn set_custom_freq_postings_field_accepts_a_docs_and_custom_freqs_field() {
        let tmp = tempdir("custom-freq-field-accepted");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer
            .set_custom_freq_postings_field(Some("score"))
            .unwrap();
        let cfg = writer.custom_freq_postings_field.as_ref().unwrap();
        assert_eq!(cfg.field_number, 1);

        writer.set_custom_freq_postings_field(None).unwrap();
        assert!(writer.custom_freq_postings_field.is_none());
    }

    #[test]
    fn set_custom_freq_postings_field_rejects_when_a_text_postings_field_is_already_set() {
        let tmp = tempdir("mutual-exclusion-a");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            body_field(1),
            custom_freq_field(2),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        let err = writer
            .set_custom_freq_postings_field(Some("score"))
            .unwrap_err();
        assert!(matches!(
            err,
            Error::PostingsAndCustomFreqPostingsMutuallyExclusive("set_custom_freq_postings_field")
        ));
    }

    #[test]
    fn set_postings_field_rejects_when_a_custom_freq_postings_field_is_already_set() {
        let tmp = tempdir("mutual-exclusion-b");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            body_field(1),
            custom_freq_field(2),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer
            .set_custom_freq_postings_field(Some("score"))
            .unwrap();

        let err = writer.set_postings_field(Some("body")).unwrap_err();
        assert!(matches!(
            err,
            Error::PostingsAndCustomFreqPostingsMutuallyExclusive("set_postings_field")
        ));

        let err = writer.add_postings_field("body").unwrap_err();
        assert!(matches!(
            err,
            Error::PostingsAndCustomFreqPostingsMutuallyExclusive("add_postings_field")
        ));
    }

    #[test]
    fn commit_with_custom_freq_postings_field_writes_readable_postings_with_explicit_freqs() {
        // The whole point of `DocsAndCustomFreqs`: the freq value written is
        // exactly the caller's explicit `custom_freq`, not a derived
        // occurrence count -- proven here by supplying a `custom_freq` that
        // differs from "1 occurrence" for every doc/term pair and reading it
        // back byte-for-byte through the existing, unmodified
        // `blocktree`/`postings::DocInput` read side.
        let tmp = tempdir("custom-freq-postings-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer
            .set_custom_freq_postings_field(Some("score"))
            .unwrap();

        writer
            .add_document_with_custom_freq_terms(
                doc("a"),
                vec![("alpha".to_string(), 7), ("beta".to_string(), 3)],
            )
            .unwrap();
        writer
            .add_document_with_custom_freq_terms(doc("b"), vec![("alpha".to_string(), 42)])
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];

        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for ext in ["doc", "tim", "tip", "tmd"] {
            let name = format!(
                "{}.{ext}",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            );
            assert!(si.files.contains(&name), "missing {name} in .si files");
        }

        let tim = dir
            .open(&format!(
                "{}.tim",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let tip = dir
            .open(&format!(
                "{}.tip",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let tmd = dir
            .open(&format!(
                "{}.tmd",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let doc_bytes = dir
            .open(&format!(
                "{}.doc",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![
                fi::FieldInfo {
                    index_options: IndexOptions::None,
                    ..stored_only_field("id", 0)
                },
                custom_freq_field(1),
            ],
        };
        let block_fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            2,
        )
        .expect("blocktree::open on custom-freq .tim/.tip/.tmd");
        let doc_in = DocInput::open(
            &doc_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .expect("open .doc");
        let field = block_fields.field("score").unwrap();

        let postings = field.postings(b"alpha", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0, 1]);
        assert_eq!(postings.freqs, vec![7, 42]);

        let postings = field.postings(b"beta", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0]);
        assert_eq!(postings.freqs, vec![3]);

        assert!(field.seek_exact(b"missing").is_none());
    }

    #[test]
    fn commit_with_custom_freq_postings_field_but_no_pending_docs_writes_no_postings_files() {
        let tmp = tempdir("custom-freq-postings-empty-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer
            .set_custom_freq_postings_field(Some("score"))
            .unwrap();

        let sis = writer.commit().unwrap().clone();
        assert!(sis.segments.is_empty());
    }

    #[test]
    fn commit_with_custom_freq_postings_field_and_no_doc_supplying_terms_skips_postings() {
        // A doc added via plain `add_document` alongside custom-freq-terms
        // docs contributes an empty term list (see `add_document`'s own doc
        // comment on keeping `pending_custom_freq_terms` aligned) -- not an
        // error, just nothing to index for that doc.
        let tmp = tempdir("custom-freq-postings-no-terms");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer
            .set_custom_freq_postings_field(Some("score"))
            .unwrap();

        writer.add_document(doc("a")).unwrap(); // no custom-freq terms at all
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!(
            "{}.tim",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        )));
    }

    #[test]
    fn commit_with_no_custom_freq_postings_field_configured_stays_stored_only() {
        let tmp = tempdir("no-custom-freq-postings-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer
            .add_document_with_custom_freq_terms(doc("a"), vec![("alpha".to_string(), 5)])
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        for ext in ["doc", "tim", "tip", "tmd"] {
            assert!(!files.contains(&format!("{}.{ext}", sci.segment_name)));
        }
    }

    #[test]
    fn commit_rejects_a_custom_freq_below_one_and_leaves_the_writer_unchanged() {
        // The codec layer's own "freq < 1" validation
        // (`postings_writer::write_fields`) applies unchanged to
        // `DocsAndCustomFreqs` -- a caller-supplied `custom_freq` of `0` (or
        // negative) surfaces as an `Err`, not a silently-clamped value.
        let tmp = tempdir("custom-freq-below-one");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer
            .set_custom_freq_postings_field(Some("score"))
            .unwrap();

        writer
            .add_document_with_custom_freq_terms(doc("a"), vec![("alpha".to_string(), 0)])
            .unwrap();
        let err = writer.commit().unwrap_err();
        assert!(matches!(err, Error::PostingsWriter(_)));
        // Atomic failure: nothing committed, pending state untouched.
        assert!(writer.segment_infos.segments.is_empty());
        assert_eq!(writer.pending_doc_count(), 1);
    }

    #[test]
    fn rollback_discards_pending_custom_freq_terms_too() {
        let tmp = tempdir("custom-freq-rollback");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer
            .set_custom_freq_postings_field(Some("score"))
            .unwrap();

        writer
            .add_document_with_custom_freq_terms(doc("a"), vec![("alpha".to_string(), 5)])
            .unwrap();
        assert_eq!(writer.pending_custom_freq_terms.len(), 1);
        writer.rollback();
        assert!(writer.pending_custom_freq_terms.is_empty());

        // Next commit sees nothing from the rolled-back doc.
        let sis = writer.commit().unwrap().clone();
        assert!(sis.segments.is_empty());
    }

    /// `postings_writer` now emits real full `ForUtil`/`PForUtil` blocks for
    /// `docFreq >= BLOCK_SIZE (256)` (see `crates/lucene-codecs/src/
    /// postings_writer.rs`'s `write_full_block` and its own
    /// `docfreq_exactly_one_full_block_no_tail`/`docfreq_spans_multiple_full_blocks_plus_tail`
    /// unit tests for the byte-level round-trip proof), so `docFreq == 256`
    /// alone no longer rejects a `commit()` the way it used to. A later task
    /// added level-1 skip-entry emission
    /// (`postings_writer::write_level1_span`), so `docFreq >= LEVEL1_NUM_DOCS`
    /// (8192) is no longer a hard ceiling either -- see that module's doc
    /// comment for the current state (no further per-term docFreq ceiling
    /// remains). There is deliberately no end-to-end `IndexWriter`-level test
    /// of the 8192 boundary here: reaching it requires >=8192 pending docs in
    /// one flush, which trips a wholly unrelated, pre-existing cap in
    /// the postings layer is ever reached. The
    /// `LEVEL1_NUM_DOCS` boundary itself is exercised directly at the
    /// `postings_writer` unit level instead
    /// (`docfreq_at_level1_boundaries_round_trips`).
    /// A term just under the 256 boundary must still commit successfully.
    #[test]
    fn commit_succeeds_below_the_doc_freq_boundary() {
        let tmp = tempdir("postings-docfreq-just-under");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        for i in 0..255 {
            writer
                .add_document(doc_with_body(&i.to_string(), "shared"))
                .unwrap();
        }
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
    }

    #[test]
    fn commit_with_postings_field_but_no_pending_docs_writes_no_postings_files() {
        let tmp = tempdir("postings-empty-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        let sis = writer.commit().unwrap().clone();
        assert!(sis.segments.is_empty());
    }

    #[test]
    fn commit_with_postings_field_but_no_doc_has_that_fields_text_skips_postings() {
        // A document that omits the opted-in postings field entirely (no
        // `StoredField` for its `field_number`) contributes no postings --
        // this must not be an error, just "nothing to index this commit".
        let tmp = tempdir("postings-no-text");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        writer.add_document(doc("a")).unwrap(); // only field_number 0 ("id"), no "body"
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!(
            "{}.tim",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        )));
    }

    #[test]
    fn commit_with_postings_field_holding_a_non_string_value_skips_that_doc() {
        // A doc whose stored value for the opted-in postings field isn't a
        // `FieldValue::String` (e.g. `Int`) contributes no indexable text --
        // matches `set_postings_field`'s own doc comment.
        let tmp = tempdir("postings-non-string-value");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Int(42), // not a String -- must be skipped
                    },
                ],
            })
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!(
            "{}.tim",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        )));
    }

    #[test]
    fn commit_with_postings_field_text_that_tokenizes_to_nothing_skips_postings() {
        // The opted-in field has a `String` value on every doc, but that
        // text tokenizes to zero terms (e.g. only whitespace) -- still not
        // an error, just nothing to index this commit, distinct from the
        // "field missing/non-String" case above.
        let tmp = tempdir("postings-empty-tokenization");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        writer.add_document(doc_with_body("a", "   ")).unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!(
            "{}.tim",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        )));
    }

    #[test]
    fn setting_postings_field_back_to_none_restores_stored_only_behavior() {
        let tmp = tempdir("postings-field-reset");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_postings_field(None).unwrap();

        writer
            .add_document(doc_with_body("a", "the quick fox"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!(
            "{}.tim",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        )));
    }

    #[test]
    fn segments_with_postings_are_automatically_merged_with_postings_preserved() {
        // Enabling both set_postings_field and set_merge_policy at once now
        // merges postings-carrying segments for real: execute_merge opens
        // each source's .tim/.tip/.tmd/.doc (when present) and feeds them
        // through crate::merge::merge_postings via MergeSource::postings, so
        // segment_stats() no longer needs to exclude postings-bearing
        // segments from find_merges' candidate pool to avoid silent data
        // loss.
        let tmp = tempdir("postings-and-merge-policy");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        // tight_merge_policy's segments_per_tier is 2 -- three one-doc
        // commits, each producing a segment with real postings, must cross
        // that threshold and trigger a merge down to one segment.
        for (id, text) in [
            ("a", "shared apple"),
            ("b", "shared banana"),
            ("c", "shared cherry"),
        ] {
            writer.add_document(doc_with_body(id, text)).unwrap();
            writer.commit().unwrap();
        }

        let segments = writer.segment_infos().segments.clone();
        assert_eq!(
            segments.len(),
            1,
            "postings-carrying segments must now merge down like any other"
        );
        let sci = &segments[0];

        let files = dir.list_all().unwrap();
        assert!(files.contains(&format!(
            "{}.tim",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        )));
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert!(si.files.iter().any(|f| f.ends_with(".tim")));

        // The merged segment's postings must still resolve every term from
        // every source doc -- open the merged .tim/.tip/.tmd/.doc for real
        // and confirm "shared" (present in all three docs) and "banana"
        // (present in exactly one) both round-trip correctly.
        let tim = dir
            .open(&format!(
                "{}.tim",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let tip = dir
            .open(&format!(
                "{}.tip",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let tmd = dir
            .open(&format!(
                "{}.tmd",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let doc_bytes = dir
            .open(&format!(
                "{}.doc",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = lucene_codecs::field_infos::FieldInfos {
            fields: vec![body_field(1)],
        };
        let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
        let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
        let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
        let stored = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        let max_doc = stored.max_doc();
        let block_tree = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            max_doc,
        )
        .unwrap();
        let doc_in = DocInput::open(
            &doc_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let body_terms = block_tree.field("body").unwrap();

        let shared_stats = body_terms.seek_exact(b"shared").unwrap();
        assert_eq!(
            shared_stats.doc_freq, 3,
            "\"shared\" appears in all 3 merged docs"
        );
        let shared_postings = body_terms
            .postings(b"shared", Some(&doc_in))
            .unwrap()
            .unwrap();
        assert_eq!(shared_postings.docs.len(), 3);

        let banana_stats = body_terms.seek_exact(b"banana").unwrap();
        assert_eq!(banana_stats.doc_freq, 1);
    }

    // --- set_term_vector_field / commit()'s term-vector-writing path ---

    fn tv_body_field(number: i32) -> FieldInfo {
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            store_term_vectors: true,
            ..stored_only_field("body", number)
        }
    }

    #[test]
    fn set_term_vector_field_rejects_an_unknown_field_name() {
        let tmp = tempdir("unknown-tv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer
            .set_term_vector_field(Some("nonexistent"))
            .unwrap_err();
        assert!(matches!(err, Error::UnknownTermVectorField(name) if name == "nonexistent"));
    }

    #[test]
    fn set_term_vector_field_rejects_a_field_without_store_term_vectors() {
        let tmp = tempdir("unflagged-tv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)]; // body_field: store_term_vectors == false
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer.set_term_vector_field(Some("body")).unwrap_err();
        assert!(matches!(err, Error::UnsupportedTermVectorField(name) if name == "body"));
    }

    #[test]
    fn add_term_vector_field_rejects_an_unknown_field_name() {
        let tmp = tempdir("unknown-add-tv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        let err = writer.add_term_vector_field("nonexistent").unwrap_err();
        assert!(matches!(err, Error::UnknownTermVectorField(name) if name == "nonexistent"));
    }

    #[test]
    fn add_term_vector_field_rejects_a_duplicate_field() {
        let tmp = tempdir("duplicate-add-tv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        let err = writer.add_term_vector_field("body").unwrap_err();
        assert!(matches!(err, Error::DuplicateTermVectorField(name) if name == "body"));
    }

    #[test]
    fn commit_with_term_vector_field_writes_readable_term_vectors_for_multiple_docs_and_terms() {
        let tmp = tempdir("tv-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        writer
            .add_document(doc_with_body("a", "the quick fox"))
            .unwrap();
        writer
            .add_document(doc_with_body("b", "the lazy fox"))
            .unwrap();
        writer
            .add_document(doc_with_body("c", "the fox runs"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];

        // Stored fields are still intact (backward-compatible).
        assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b", "c"]);

        // The term-vector files exist and are listed in `.si`.
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for ext in ["tvd", "tvx", "tvm"] {
            let name = format!("{}.{ext}", sci.segment_name);
            assert!(si.files.contains(&name), "missing {name} in .si files");
            assert!(
                dir.list_all().unwrap().contains(&name),
                "missing {name} on disk"
            );
        }

        // Readable via the existing, unmodified read side.
        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let reader = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "")
            .expect("term_vectors::open on IndexWriter-produced .tvd/.tvx/.tvm");
        assert_eq!(reader.max_doc(), 3);

        let doc0 = reader.document(0).unwrap().unwrap();
        assert_eq!(doc0.fields.len(), 1);
        assert_eq!(doc0.fields[0].field_number, 1);
        assert!(doc0.fields[0].has_positions);
        let mut terms0: Vec<String> = doc0.fields[0]
            .terms
            .iter()
            .map(|t| String::from_utf8(t.term.clone()).unwrap())
            .collect();
        terms0.sort();
        assert_eq!(terms0, vec!["fox", "quick", "the"]);

        let doc2 = reader.document(2).unwrap().unwrap();
        let mut terms2: Vec<String> = doc2.fields[0]
            .terms
            .iter()
            .map(|t| String::from_utf8(t.term.clone()).unwrap())
            .collect();
        terms2.sort();
        assert_eq!(terms2, vec!["fox", "runs", "the"]);
    }

    #[test]
    fn commit_with_no_term_vector_field_configured_stays_stored_only() {
        // Backward compatibility: a writer that never calls
        // `set_term_vector_field` must produce exactly the same on-disk
        // shape as before this feature existed -- no `.tvd`/`.tvx`/`.tvm`
        // files at all.
        let tmp = tempdir("no-tv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer
            .add_document(doc_with_body("a", "the quick fox"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        for ext in ["tvd", "tvx", "tvm"] {
            assert!(!files.contains(&format!("{}.{ext}", sci.segment_name)));
        }
    }

    #[test]
    fn commit_with_term_vector_field_but_no_pending_docs_writes_no_term_vector_files() {
        let tmp = tempdir("tv-empty-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        let sis = writer.commit().unwrap().clone();
        assert!(sis.segments.is_empty());
    }

    #[test]
    fn commit_with_term_vector_field_but_no_doc_has_that_fields_text_skips_term_vectors() {
        let tmp = tempdir("tv-no-text");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        writer.add_document(doc("a")).unwrap(); // only field_number 0 ("id"), no "body"
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!("{}.tvd", sci.segment_name)));
    }

    #[test]
    fn commit_with_term_vector_field_holding_a_non_string_value_skips_that_doc() {
        let tmp = tempdir("tv-non-string-value");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Int(42), // not a String -- must be skipped
                    },
                ],
            })
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!("{}.tvd", sci.segment_name)));
    }

    #[test]
    fn commit_with_term_vector_field_text_that_tokenizes_to_nothing_skips_term_vectors() {
        let tmp = tempdir("tv-empty-tokenization");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        writer.add_document(doc_with_body("a", "   ")).unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!("{}.tvd", sci.segment_name)));
    }

    #[test]
    fn commit_with_term_vector_field_where_only_some_docs_have_text_still_writes_an_entry_per_doc()
    {
        // A doc with no indexable text for the opted-in field still needs a
        // `TermVectorsDocument` entry (empty `fields`) so doc IDs stay
        // aligned with the segment's real doc count --
        // `write_best_speed` derives `max_doc` from `docs.len()`.
        let tmp = tempdir("tv-partial-docs");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        writer.add_document(doc_with_body("a", "alpha")).unwrap();
        writer.add_document(doc("b")).unwrap(); // no "body" field at all
        writer.add_document(doc_with_body("c", "gamma")).unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let reader =
            lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
        assert_eq!(reader.max_doc(), 3);
        assert_eq!(reader.document(0).unwrap().unwrap().fields.len(), 1);
        // A doc contributing zero fields to this chunk decodes as `None`,
        // not `Some(fields: vec![])` -- see `TermVectorsReader::document`'s
        // own `doc_num_fields == 0 => Ok(None)` branch.
        assert!(reader.document(1).unwrap().is_none());
        assert_eq!(reader.document(2).unwrap().unwrap().fields.len(), 1);
    }

    #[test]
    fn setting_term_vector_field_back_to_none_restores_stored_only_behavior() {
        let tmp = tempdir("tv-field-reset");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();
        writer.set_term_vector_field(None).unwrap();

        writer
            .add_document(doc_with_body("a", "the quick fox"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!("{}.tvd", sci.segment_name)));
    }

    #[test]
    fn segments_with_term_vectors_are_automatically_merged_with_term_vectors_preserved() {
        // Same class of fix as `segments_with_postings_are_automatically_
        // merged_with_postings_preserved`, for term vectors instead of
        // postings: enabling both `set_term_vector_field` and
        // `set_merge_policy` at once now merges term-vector-carrying
        // segments for real -- `execute_merge` opens each source's real
        // `.tvd`/`.tvx`/`.tvm` (when present) and feeds them through
        // `crate::merge::write_merged_term_vectors` via `MergeSource::term_vectors`,
        // so `segment_stats()` no longer needs to exclude term-vector-
        // bearing segments from `find_merges`' candidate pool to avoid
        // silent data loss.
        let tmp = tempdir("tv-and-merge-policy");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        // tight_merge_policy's segments_per_tier is 2 -- three one-doc
        // commits, each producing a segment with real term vectors, must
        // cross that threshold and trigger a merge down to one segment.
        for (id, text) in [("a", "alpha one"), ("b", "beta two"), ("c", "gamma three")] {
            writer.add_document(doc_with_body(id, text)).unwrap();
            writer.commit().unwrap();
        }

        let segments = writer.segment_infos().segments.clone();
        assert_eq!(
            segments.len(),
            1,
            "term-vector-carrying segments must now merge down like any other"
        );
        let sci = &segments[0];

        let files = dir.list_all().unwrap();
        assert!(files.contains(&format!("{}.tvd", sci.segment_name)));
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert!(si.files.iter().any(|f| f.ends_with(".tvd")));

        // The merged segment's term vectors must still hold every source
        // doc's own distinct terms -- open the merged .tvd/.tvx/.tvm for real
        // and check each doc's fields/terms. Automatic merging doesn't
        // guarantee source-segment order in the merged segment, so this
        // checks the *set* of per-doc term-vector contents survived intact
        // rather than assuming a fixed doc-id assignment.
        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let reader =
            lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
        assert_eq!(reader.max_doc(), 3);

        let mut actual_docs: Vec<Vec<String>> = (0..3)
            .map(|doc_id| {
                let doc = reader.document(doc_id).unwrap().unwrap();
                assert_eq!(doc.fields.len(), 1);
                doc.fields[0]
                    .terms
                    .iter()
                    .map(|t| std::str::from_utf8(&t.term).unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        actual_docs.sort();

        let mut expected_docs = vec![
            vec!["alpha".to_string(), "one".to_string()],
            vec!["beta".to_string(), "two".to_string()],
            vec!["gamma".to_string(), "three".to_string()],
        ];
        expected_docs.sort();

        assert_eq!(
            actual_docs, expected_docs,
            "every source doc's own term-vector content must survive the merge intact"
        );
    }

    #[test]
    fn a_field_with_both_postings_and_term_vectors_configured_at_once_writes_both_correctly() {
        // Real Lucene's ordinary case: a field is both indexed (postings)
        // and has term vectors stored, in the same commit. Both write-side
        // passes must coexist: the segment's `.si` lists all seven files,
        // and both remain independently readable.
        let tmp = tempdir("postings-and-tv-together");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();

        writer
            .add_document(doc_with_body("a", "the quick fox"))
            .unwrap();
        writer
            .add_document(doc_with_body("b", "the lazy fox"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];

        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for ext in ["doc", "tim", "tip", "tmd"] {
            let name = format!(
                "{}.{ext}",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            );
            assert!(si.files.contains(&name), "missing {name} in .si files");
        }
        // Term vectors are not a per-field format, so they keep plain names.
        for ext in ["tvd", "tvx", "tvm"] {
            let name = format!("{}.{ext}", sci.segment_name);
            assert!(si.files.contains(&name), "missing {name} in .si files");
        }

        // Postings side still readable.
        let tim = dir
            .open(&format!(
                "{}.tim",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let tip = dir
            .open(&format!(
                "{}.tip",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let tmd = dir
            .open(&format!(
                "{}.tmd",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![
                fi::FieldInfo {
                    index_options: IndexOptions::None,
                    ..stored_only_field("id", 0)
                },
                tv_body_field(1),
            ],
        };
        let block_fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            2,
        )
        .expect("blocktree::open on IndexWriter-produced .tim/.tip/.tmd");
        let field = block_fields.field("body").unwrap();
        let doc_bytes = dir
            .open(&format!(
                "{}.doc",
                per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
            ))
            .unwrap();
        let doc_in = DocInput::open(
            &doc_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .expect("open .doc");
        let postings = field.postings(b"fox", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0, 1]);

        // Term-vector side also readable.
        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let reader =
            lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
        let doc0 = reader.document(0).unwrap().unwrap();
        let mut terms0: Vec<String> = doc0.fields[0]
            .terms
            .iter()
            .map(|t| String::from_utf8(t.term.clone()).unwrap())
            .collect();
        terms0.sort();
        assert_eq!(terms0, vec!["fox", "quick", "the"]);
    }

    fn numeric_field(name: &str, number: i32) -> FieldInfo {
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..stored_only_field(name, number)
        }
    }

    fn sorted_field(name: &str, number: i32) -> FieldInfo {
        FieldInfo {
            doc_values_type: DocValuesType::Sorted,
            ..stored_only_field(name, number)
        }
    }

    fn sorted_set_field(name: &str, number: i32) -> FieldInfo {
        FieldInfo {
            doc_values_type: DocValuesType::SortedSet,
            ..stored_only_field(name, number)
        }
    }

    fn binary_field(name: &str, number: i32) -> FieldInfo {
        FieldInfo {
            doc_values_type: DocValuesType::Binary,
            ..stored_only_field(name, number)
        }
    }

    fn sorted_numeric_field(name: &str, number: i32) -> FieldInfo {
        FieldInfo {
            doc_values_type: DocValuesType::SortedNumeric,
            ..stored_only_field(name, number)
        }
    }

    fn doc_with_score(id: &str, score: i64) -> Document {
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(id.to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(score),
                },
            ],
        }
    }

    #[test]
    fn set_doc_values_field_rejects_an_unknown_field_name() {
        let tmp = tempdir("unknown-dv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer
            .set_doc_values_field(Some("nonexistent"))
            .unwrap_err();
        assert!(matches!(err, Error::UnknownDocValuesField(name) if name == "nonexistent"));
    }

    #[test]
    fn set_doc_values_field_rejects_a_field_with_no_doc_values_type() {
        let tmp = tempdir("untyped-dv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer.set_doc_values_field(Some("id")).unwrap_err();
        assert!(matches!(
            err,
            Error::UnsupportedDocValuesType(name, DocValuesType::None) if name == "id"
        ));
    }

    #[test]
    fn commit_with_doc_values_field_writes_readable_numeric_values_for_multiple_docs() {
        let tmp = tempdir("dv-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        writer.add_document(doc_with_score("a", 5)).unwrap();
        writer.add_document(doc_with_score("b", 250)).unwrap();
        writer.add_document(doc_with_score("c", -7)).unwrap();
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];

        // Stored fields are still intact (backward-compatible).
        assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b", "c"]);

        // The doc-values files exist and are listed in `.si`.
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for ext in ["dvm", "dvd", "dvs"] {
            let name = format!(
                "{}.{ext}",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            );
            assert!(si.files.contains(&name), "missing {name} in .si files");
            assert!(
                dir.list_all().unwrap().contains(&name),
                "missing {name} on disk"
            );
        }

        // Readable via the existing, unmodified read side.
        let dvm = dir
            .open(&format!(
                "{}.dvm",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let dvd = dir
            .open(&format!(
                "{}.dvd",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![stored_only_field("id", 0), numeric_field("score", 1)],
        };
        let (_, meta) = lucene_codecs::doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .expect("parse_meta on IndexWriter-produced .dvm");
        let entry = meta.numeric_entry(1).unwrap();
        assert!(entry.is_dense());
        for (doc, want) in [(0, 5i64), (1, 250), (2, -7)] {
            assert_eq!(
                lucene_codecs::doc_values::numeric_value(&dvd, entry, doc).unwrap(),
                Some(want)
            );
        }
    }

    #[test]
    fn commit_with_doc_values_field_writes_readable_sorted_values_for_multiple_docs() {
        let tmp = tempdir("dv-commit-sorted");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), sorted_field("category", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("category")).unwrap();

        let categories = ["fruit", "vegetable", "fruit"];
        for (i, cat) in categories.iter().enumerate() {
            writer
                .add_document(Document {
                    fields: vec![
                        StoredField {
                            field_number: 0,
                            value: FieldValue::String(format!("doc{i}")),
                        },
                        StoredField {
                            field_number: 1,
                            value: FieldValue::String(cat.to_string()),
                        },
                    ],
                })
                .unwrap();
        }
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let dvm = dir
            .open(&format!(
                "{}.dvm",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let dvd = dir
            .open(&format!(
                "{}.dvd",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![stored_only_field("id", 0), sorted_field("category", 1)],
        };
        let (_, meta) = lucene_codecs::doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .expect("parse_meta on IndexWriter-produced SORTED .dvm");
        let entry = meta.sorted_entry(1).unwrap();
        let dict = lucene_codecs::terms_dict::decode_all_terms(&dvd, &entry.terms).unwrap();
        for (doc, want) in categories.iter().enumerate() {
            let ord = lucene_codecs::doc_values::sorted_ord(&dvd, entry, doc as i32)
                .unwrap()
                .unwrap();
            assert_eq!(dict[ord as usize], want.as_bytes());
        }
    }

    #[test]
    fn commit_with_doc_values_field_writes_readable_sorted_set_values_for_multiple_docs() {
        let tmp = tempdir("dv-commit-sorted-set");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), sorted_set_field("tags", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("tags")).unwrap();

        // A doc opts into multiple SORTED_SET values by repeating the field.
        let doc_tags: Vec<Vec<&str>> = vec![vec!["fruit", "red"], vec!["vegetable"], vec!["red"]];
        for (i, tags) in doc_tags.iter().enumerate() {
            let mut doc_fields = vec![StoredField {
                field_number: 0,
                value: FieldValue::String(format!("doc{i}")),
            }];
            for tag in tags {
                doc_fields.push(StoredField {
                    field_number: 1,
                    value: FieldValue::String(tag.to_string()),
                });
            }
            writer
                .add_document(Document { fields: doc_fields })
                .unwrap();
        }
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let dvm = dir
            .open(&format!(
                "{}.dvm",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let dvd = dir
            .open(&format!(
                "{}.dvd",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![stored_only_field("id", 0), sorted_set_field("tags", 1)],
        };
        let (_, meta) = lucene_codecs::doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .expect("parse_meta on IndexWriter-produced SORTED_SET .dvm");
        let entry = meta.sorted_set_entry(1).unwrap();
        let (ords_entry, terms_entry) = match &entry.kind {
            lucene_codecs::doc_values::SortedSetKind::Multi { ords, terms } => (ords, terms),
            lucene_codecs::doc_values::SortedSetKind::Single(_) => {
                panic!("expected Multi (some doc has 2 values)")
            }
        };
        let dict = lucene_codecs::terms_dict::decode_all_terms(&dvd, terms_entry).unwrap();
        for (doc, want) in doc_tags.iter().enumerate() {
            let ords =
                lucene_codecs::doc_values::sorted_numeric_values(&dvd, ords_entry, doc as i32)
                    .unwrap();
            let mut got: Vec<&str> = ords
                .iter()
                .map(|&ord| std::str::from_utf8(&dict[ord as usize]).unwrap())
                .collect();
            got.sort_unstable();
            let mut want_sorted = want.clone();
            want_sorted.sort_unstable();
            assert_eq!(got, want_sorted);
        }
    }

    #[test]
    fn commit_with_doc_values_field_writes_readable_binary_values_for_multiple_docs() {
        let tmp = tempdir("dv-commit-binary");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), binary_field("payload", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("payload")).unwrap();

        let payloads: [&[u8]; 3] = [b"abc", b"", b"hello world"];
        for (i, payload) in payloads.iter().enumerate() {
            writer
                .add_document(Document {
                    fields: vec![
                        StoredField {
                            field_number: 0,
                            value: FieldValue::String(format!("doc{i}")),
                        },
                        StoredField {
                            field_number: 1,
                            value: FieldValue::Binary(payload.to_vec()),
                        },
                    ],
                })
                .unwrap();
        }
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let dvm = dir
            .open(&format!(
                "{}.dvm",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let dvd = dir
            .open(&format!(
                "{}.dvd",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![stored_only_field("id", 0), binary_field("payload", 1)],
        };
        let (_, meta) = lucene_codecs::doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .expect("parse_meta on IndexWriter-produced BINARY .dvm");
        let entry = meta.binary_entry(1).unwrap();
        for (doc, want) in payloads.iter().enumerate() {
            let got = lucene_codecs::doc_values::binary_value(&dvd, entry, doc as i32)
                .unwrap()
                .unwrap();
            assert_eq!(got, *want);
        }
    }

    #[test]
    fn commit_with_doc_values_field_rejects_non_binary_binary_value() {
        let tmp = tempdir("dv-commit-binary-nonbinary");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), binary_field("payload", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("payload")).unwrap();

        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(1),
                    },
                ],
            })
            .unwrap();
        let err = writer.commit().unwrap_err();
        assert!(matches!(
            err,
            Error::NonBinaryDocValue(name, 0, "Long") if name == "payload"
        ));
    }

    #[test]
    fn commit_with_doc_values_field_writes_readable_sorted_numeric_values_for_multiple_docs() {
        let tmp = tempdir("dv-commit-sorted-numeric");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), sorted_numeric_field("nums", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("nums")).unwrap();

        // Doc with multiple values exercises the address-array path; the
        // other two docs (one value each) alone would collapse it away.
        let doc_nums: Vec<Vec<i64>> = vec![vec![10, 20], vec![-5], vec![0, 1, 2]];
        for (i, nums) in doc_nums.iter().enumerate() {
            let mut doc_fields = vec![StoredField {
                field_number: 0,
                value: FieldValue::String(format!("doc{i}")),
            }];
            for n in nums {
                doc_fields.push(StoredField {
                    field_number: 1,
                    value: FieldValue::Long(*n),
                });
            }
            writer
                .add_document(Document { fields: doc_fields })
                .unwrap();
        }
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let dvm = dir
            .open(&format!(
                "{}.dvm",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let dvd = dir
            .open(&format!(
                "{}.dvd",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![stored_only_field("id", 0), sorted_numeric_field("nums", 1)],
        };
        let (_, meta) = lucene_codecs::doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .expect("parse_meta on IndexWriter-produced SORTED_NUMERIC .dvm");
        let entry = meta.sorted_numeric_entry(1).unwrap();
        for (doc, want) in doc_nums.iter().enumerate() {
            let got =
                lucene_codecs::doc_values::sorted_numeric_values(&dvd, entry, doc as i32).unwrap();
            assert_eq!(&got, want);
        }
    }

    #[test]
    fn commit_with_doc_values_field_rejects_non_numeric_sorted_numeric_value() {
        let tmp = tempdir("dv-commit-sorted-numeric-nonnumeric");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), sorted_numeric_field("nums", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("nums")).unwrap();

        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String("not a number".to_string()),
                    },
                ],
            })
            .unwrap();
        let err = writer.commit().unwrap_err();
        assert!(matches!(
            err,
            Error::NonNumericDocValue(name, 0, "String") if name == "nums"
        ));
    }

    #[test]
    fn commit_with_doc_values_field_rejects_doc_with_no_sorted_numeric_values() {
        let tmp = tempdir("dv-commit-sorted-numeric-missing");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), sorted_numeric_field("nums", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("nums")).unwrap();

        writer
            .add_document(Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                }],
            })
            .unwrap();
        let err = writer.commit().unwrap_err();
        assert!(matches!(
            err,
            Error::MissingDenseDocValue(name, 0) if name == "nums"
        ));
    }

    #[test]
    fn commit_with_doc_values_field_rejects_non_binary_sorted_value() {
        let tmp = tempdir("dv-commit-sorted-nonbinary");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), sorted_field("category", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("category")).unwrap();

        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(1),
                    },
                ],
            })
            .unwrap();
        let err = writer.commit().unwrap_err();
        assert!(matches!(
            err,
            Error::NonBinaryDocValue(name, 0, "Long") if name == "category"
        ));
    }

    #[test]
    fn commit_with_doc_values_field_accepts_int_values_not_just_long() {
        // build_doc_values_output accepts both FieldValue::Int and
        // FieldValue::Long -- the multi-doc test above only exercises Long,
        // so this covers the Int arm's i64 sign-extension explicitly.
        let tmp = tempdir("dv-commit-int");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Int(-42),
                    },
                ],
            })
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let dvm = dir
            .open(&format!(
                "{}.dvm",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let dvd = dir
            .open(&format!(
                "{}.dvd",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![stored_only_field("id", 0), numeric_field("score", 1)],
        };
        let (_, meta) = lucene_codecs::doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .expect("parse_meta on IndexWriter-produced .dvm");
        let entry = meta.numeric_entry(1).unwrap();
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(&dvd, entry, 0).unwrap(),
            Some(-42i64)
        );
    }

    #[test]
    fn commit_with_no_doc_values_field_configured_stays_stored_only() {
        // Backward compatibility: a writer that never calls
        // `set_doc_values_field` must produce exactly the same on-disk shape
        // as before this feature existed -- no `.dvm`/`.dvd`/`.dvs` files at
        // all.
        let tmp = tempdir("no-dv-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc_with_score("a", 5)).unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        for ext in ["dvm", "dvd", "dvs"] {
            assert!(!files.contains(&format!("{}.{ext}", sci.segment_name)));
        }
    }

    #[test]
    fn commit_with_doc_values_field_but_no_pending_docs_writes_no_doc_values_files() {
        let tmp = tempdir("dv-empty-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        let sis = writer.commit().unwrap().clone();
        assert!(sis.segments.is_empty());
    }

    #[test]
    fn commit_writes_sparse_numeric_doc_values_when_a_pending_doc_has_no_value() {
        // A doc missing the opted-in NUMERIC field entirely no longer
        // rejects the whole commit: it routes the present docs' values
        // through `write_single_sparse_numeric_field` instead of the dense
        // writer, and the missing doc reads back as absent (`None`), not as
        // a wrong/zero value.
        let tmp = tempdir("dv-sparse-numeric");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        writer.add_document(doc_with_score("a", 5)).unwrap();
        writer.add_document(doc("b")).unwrap(); // only field_number 0 ("id"), no "score"
        writer.add_document(doc_with_score("c", 7)).unwrap();

        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let dvm = dir
            .open(&format!(
                "{}.dvm",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let dvd = dir
            .open(&format!(
                "{}.dvd",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![stored_only_field("id", 0), numeric_field("score", 1)],
        };
        let (_, meta) = lucene_codecs::doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .expect("parse_meta on IndexWriter-produced sparse .dvm");
        let entry = meta.numeric_entry(1).unwrap();

        assert_eq!(
            lucene_codecs::doc_values::numeric_value(&dvd, entry, 0).unwrap(),
            Some(5)
        );
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(&dvd, entry, 1).unwrap(),
            None
        );
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(&dvd, entry, 2).unwrap(),
            Some(7)
        );
    }

    #[test]
    fn commit_writes_sparse_sorted_doc_values_when_a_pending_doc_has_no_value() {
        // Same sparse contract as the NUMERIC test above, but for SORTED,
        // whose sparse writer builds a terms dictionary of only the present
        // docs' values and writes per-doc ordinals only for those docs.
        let tmp = tempdir("dv-sparse-sorted");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), sorted_field("category", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("category")).unwrap();

        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String("red".to_string()),
                    },
                ],
            })
            .unwrap();
        writer.add_document(doc("b")).unwrap(); // no "category" value
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("c".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String("blue".to_string()),
                    },
                ],
            })
            .unwrap();

        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let dvm = dir
            .open(&format!(
                "{}.dvm",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let dvd = dir
            .open(&format!(
                "{}.dvd",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![stored_only_field("id", 0), sorted_field("category", 1)],
        };
        let (_, meta) = lucene_codecs::doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .expect("parse_meta on IndexWriter-produced sparse SORTED .dvm");
        let entry = meta.sorted_entry(1).unwrap();

        assert!(lucene_codecs::doc_values::sorted_ord(&dvd, entry, 0)
            .unwrap()
            .is_some());
        assert_eq!(
            lucene_codecs::doc_values::sorted_ord(&dvd, entry, 1).unwrap(),
            None
        );
        assert!(lucene_codecs::doc_values::sorted_ord(&dvd, entry, 2)
            .unwrap()
            .is_some());
    }

    #[test]
    fn commit_writes_sparse_binary_doc_values_when_a_pending_doc_has_no_value() {
        let tmp = tempdir("dv-sparse-binary");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), binary_field("payload", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("payload")).unwrap();

        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Binary(b"abc".to_vec()),
                    },
                ],
            })
            .unwrap();
        writer.add_document(doc("b")).unwrap(); // no "payload" value
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("c".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Binary(b"xyz".to_vec()),
                    },
                ],
            })
            .unwrap();

        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let dvm = dir
            .open(&format!(
                "{}.dvm",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let dvd = dir
            .open(&format!(
                "{}.dvd",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![stored_only_field("id", 0), binary_field("payload", 1)],
        };
        let (_, meta) = lucene_codecs::doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .expect("parse_meta on IndexWriter-produced sparse BINARY .dvm");
        let entry = meta.binary_entry(1).unwrap();

        assert_eq!(
            lucene_codecs::doc_values::binary_value(&dvd, entry, 0).unwrap(),
            Some(b"abc".as_slice())
        );
        assert_eq!(
            lucene_codecs::doc_values::binary_value(&dvd, entry, 1).unwrap(),
            None
        );
        assert_eq!(
            lucene_codecs::doc_values::binary_value(&dvd, entry, 2).unwrap(),
            Some(b"xyz".as_slice())
        );
    }

    #[test]
    fn commit_writes_sparse_sorted_numeric_doc_values_when_a_pending_doc_has_no_value() {
        let tmp = tempdir("dv-sparse-sorted-numeric");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), sorted_numeric_field("nums", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("nums")).unwrap();

        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(10),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(20),
                    },
                ],
            })
            .unwrap();
        writer.add_document(doc("b")).unwrap(); // no "nums" value at all
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("c".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(-5),
                    },
                ],
            })
            .unwrap();

        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let dvm = dir
            .open(&format!(
                "{}.dvm",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let dvd = dir
            .open(&format!(
                "{}.dvd",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![stored_only_field("id", 0), sorted_numeric_field("nums", 1)],
        };
        let (_, meta) = lucene_codecs::doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .expect("parse_meta on IndexWriter-produced sparse SORTED_NUMERIC .dvm");
        let entry = meta.sorted_numeric_entry(1).unwrap();

        assert_eq!(
            lucene_codecs::doc_values::sorted_numeric_values(&dvd, entry, 0).unwrap(),
            vec![10, 20]
        );
        assert_eq!(
            lucene_codecs::doc_values::sorted_numeric_values(&dvd, entry, 1).unwrap(),
            Vec::<i64>::new()
        );
        assert_eq!(
            lucene_codecs::doc_values::sorted_numeric_values(&dvd, entry, 2).unwrap(),
            vec![-5]
        );
    }

    #[test]
    fn commit_writes_sparse_sorted_set_doc_values_when_a_pending_doc_has_no_value() {
        let tmp = tempdir("dv-sparse-sorted-set");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), sorted_set_field("tags", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("tags")).unwrap();

        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String("fruit".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String("red".to_string()),
                    },
                ],
            })
            .unwrap();
        writer.add_document(doc("b")).unwrap(); // no "tags" value at all
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("c".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String("vegetable".to_string()),
                    },
                ],
            })
            .unwrap();

        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let dvm = dir
            .open(&format!(
                "{}.dvm",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let dvd = dir
            .open(&format!(
                "{}.dvd",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![stored_only_field("id", 0), sorted_set_field("tags", 1)],
        };
        let (_, meta) = lucene_codecs::doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .expect("parse_meta on IndexWriter-produced sparse SORTED_SET .dvm");
        let entry = meta.sorted_set_entry(1).unwrap();
        let (ords_entry, terms_entry) = match &entry.kind {
            lucene_codecs::doc_values::SortedSetKind::Multi { ords, terms } => (ords, terms),
            lucene_codecs::doc_values::SortedSetKind::Single(_) => {
                panic!("expected Multi (some doc has 2 values)")
            }
        };
        let dict = lucene_codecs::terms_dict::decode_all_terms(&dvd, terms_entry).unwrap();

        let mut doc0: Vec<&str> =
            lucene_codecs::doc_values::sorted_numeric_values(&dvd, ords_entry, 0)
                .unwrap()
                .iter()
                .map(|&ord| std::str::from_utf8(&dict[ord as usize]).unwrap())
                .collect();
        doc0.sort_unstable();
        assert_eq!(doc0, vec!["fruit", "red"]);

        assert_eq!(
            lucene_codecs::doc_values::sorted_numeric_values(&dvd, ords_entry, 1).unwrap(),
            Vec::<i64>::new(),
            "doc without a \"tags\" value must report zero ordinals, not a stale/wrong one"
        );

        let doc2: Vec<&str> = lucene_codecs::doc_values::sorted_numeric_values(&dvd, ords_entry, 2)
            .unwrap()
            .iter()
            .map(|&ord| std::str::from_utf8(&dict[ord as usize]).unwrap())
            .collect();
        assert_eq!(doc2, vec!["vegetable"]);
    }

    // Note: a "one field dense, another field on the same docs sparse"
    // mixed-commit test is not expressible against the current API --
    // `IndexWriter` supports only one `doc_values_field` configured at a
    // time (`doc_values_field: Option<DocValuesFieldConfig>`, not a list;
    // see `set_doc_values_field`'s doc comment), a pre-existing limitation
    // unrelated to sparse support and out of this task's scope. The dense
    // vs. sparse build paths for a *single* field are independent match
    // arms in `build_doc_values_output` (dense/sparse is decided per field,
    // per call), so there is no shared mutable state between fields for
    // multiple fields to interfere through even once that limitation is
    // lifted.

    #[test]
    fn commit_rejects_when_a_pending_docs_value_is_not_numeric() {
        let tmp = tempdir("dv-non-numeric-value");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String("not-a-number".to_string()), // wrong type
                    },
                ],
            })
            .unwrap();

        let err = writer.commit().unwrap_err();
        assert!(matches!(
            err,
            Error::NonNumericDocValue(name, 0, "String") if name == "score"
        ));
    }

    #[test]
    fn setting_doc_values_field_back_to_none_restores_stored_only_behavior() {
        let tmp = tempdir("dv-field-reset");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();
        writer.set_doc_values_field(None).unwrap();

        writer.add_document(doc_with_score("a", 5)).unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let files = dir.list_all().unwrap();
        assert!(!files.contains(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        )));
    }

    #[test]
    fn an_automatic_merge_carries_doc_values_through_instead_of_dropping_them() {
        // Doc-values-bearing segments used to be withheld from the merge
        // policy entirely, because `execute_merge` opened no doc values and
        // merging would have dropped the column silently. It opens them now,
        // so the exclusion is gone -- and the thing to assert is not that the
        // merge happened but that **every document still has its own value**
        // afterwards, which is the property a lost or mis-mapped column
        // breaks while leaving a perfectly valid segment behind.
        let tmp = tempdir("dv-and-merge-policy");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        for (id, score) in [("a", 1i64), ("b", 2), ("c", 3)] {
            writer.add_document(doc_with_score(id, score)).unwrap();
            writer.commit().unwrap();
        }

        let infos = writer.segment_infos().clone();
        assert_eq!(
            infos.segments.len(),
            1,
            "a tight merge policy must now fold all three doc-values segments into one"
        );
        let sci = &infos.segments[0];
        let seg = per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME);
        let dvm = dir.open(&format!("{seg}.dvm")).unwrap().to_vec();
        let dvd = dir.open(&format!("{seg}.dvd")).unwrap().to_vec();
        let (_v, meta) = doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &lucene_codecs::field_infos::FieldInfos {
                fields: vec![stored_only_field("id", 0), numeric_field("score", 1)],
            },
        )
        .unwrap();
        let entry = meta.numeric_entry(1).unwrap();
        // Stored `id` and doc-values `score` were written from the same
        // document, so they must still describe the same document at the same
        // merged doc id.
        let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
        let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
        let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        assert_eq!(reader.max_doc(), 3);
        for d in 0..3 {
            let id = match &reader.document(d).unwrap().fields[0].value {
                FieldValue::String(v) => v.clone(),
                other => panic!("unexpected stored value {other:?}"),
            };
            let expected = match id.as_str() {
                "a" => 1,
                "b" => 2,
                "c" => 3,
                other => panic!("unexpected id {other}"),
            };
            assert_eq!(
                doc_values::numeric_value(&dvd, entry, d).unwrap(),
                Some(expected),
                "doc {d} (id={id})"
            );
        }
    }

    #[test]
    fn postings_term_vectors_and_doc_values_configured_together_all_write_correctly() {
        // Three independent write-side passes (postings, term vectors, doc
        // values) each patching the same `.si` after the fact -- proves
        // there's no ordering bug where a later writer's read-modify-write
        // clobbers an earlier one's file-list additions.
        let tmp = tempdir("postings-tv-and-dv-together");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            tv_body_field(1),
            numeric_field("score", 2),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String("the quick fox".to_string()),
                    },
                    StoredField {
                        field_number: 2,
                        value: FieldValue::Long(10),
                    },
                ],
            })
            .unwrap();
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("b".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String("the lazy fox".to_string()),
                    },
                    StoredField {
                        field_number: 2,
                        value: FieldValue::Long(20),
                    },
                ],
            })
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];

        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for (exts, format) in [
            (
                ["doc", "tim", "tip", "tmd"].as_slice(),
                Some(POSTINGS_FORMAT_NAME),
            ),
            (
                ["dvm", "dvd", "dvs"].as_slice(),
                Some(DOC_VALUES_FORMAT_NAME),
            ),
            // Term vectors are not a per-field format: plain segment name.
            (["tvd", "tvx", "tvm"].as_slice(), None),
        ] {
            for ext in exts {
                let name = match format {
                    Some(f) => format!("{}.{ext}", per_field_segment(&sci.segment_name, f)),
                    None => format!("{}.{ext}", sci.segment_name),
                };
                assert!(si.files.contains(&name), "missing {name} in .si files");
            }
        }

        // Doc-values side readable.
        let dvm = dir
            .open(&format!(
                "{}.dvm",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let dvd = dir
            .open(&format!(
                "{}.dvd",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![
                stored_only_field("id", 0),
                tv_body_field(1),
                numeric_field("score", 2),
            ],
        };
        let (_, meta) = lucene_codecs::doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .unwrap();
        let entry = meta.numeric_entry(2).unwrap();
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(&dvd, entry, 0).unwrap(),
            Some(10)
        );
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(&dvd, entry, 1).unwrap(),
            Some(20)
        );

        // Term-vector side still readable too.
        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let reader =
            lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
        assert!(reader.document(0).unwrap().is_some());
    }

    #[test]
    fn omit_norms_field_rejects_an_unknown_field_name() {
        let tmp = tempdir("unknown-norms-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer.omit_norms_field("nonexistent").unwrap_err();
        assert!(matches!(err, Error::UnknownNormsField(name) if name == "nonexistent"));
    }

    #[test]
    fn omit_norms_field_rejects_a_field_with_no_index_options() {
        let tmp = tempdir("unindexed-norms-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        let err = writer.omit_norms_field("id").unwrap_err();
        assert!(matches!(err, Error::UnsupportedNormsField(name) if name == "id"));
    }

    /// Naming the same field twice is a no-op, and a field that already
    /// declares `omit_norms` is accepted (Java's `setOmitNorms(true)` is
    /// idempotent), so neither leaves a second column or an error behind.
    #[test]
    fn omit_norms_field_is_idempotent() {
        let tmp = tempdir("omit-norms-idempotent");
        let dir = FsDirectory::open(&tmp);
        let omitted = FieldInfo {
            omit_norms: true,
            ..body_field(1)
        };
        let fields = vec![stored_only_field("id", 0), omitted];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        assert!(writer.norms_field_configs().is_empty());
        writer.omit_norms_field("body").unwrap();
        writer.omit_norms_field("body").unwrap();
        assert!(writer.norms_field_configs().is_empty());
    }

    /// **This is the c35 fix.** An indexed field the caller never named gets
    /// norms, because that is what Lucene does
    /// (`IndexingChain.writeNorms`: every `FieldInfo` with
    /// `omitsNorms() == false && indexOptions != NONE`). Before c35 the
    /// writer forced `omit_norms = true` into the `.fnm` for it and wrote no
    /// column, so BM25 scored the field against a constant length instead of
    /// each document's own -- a wrong score reachable by indexing a text
    /// field and searching it.
    #[test]
    fn an_indexed_field_gets_norms_with_no_opt_in_at_all() {
        let tmp = tempdir("norms-by-default");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.add_document(doc_with_body("a", "fox")).unwrap();
        writer
            .add_document(doc_with_body("b", "the quick brown fox"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
        let fis = lucene_codecs::field_infos::parse(&fnm, &sci.segment_id, "").unwrap();
        let body = fis.fields.iter().find(|f| f.name == "body").unwrap();
        assert!(
            !body.omit_norms,
            "an indexed field keeps the norms its caller asked for"
        );

        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert!(
            si.files.iter().any(|f| f.ends_with(".nvm")),
            "norms files must have been written: {:?}",
            si.files
        );

        // And the values are the documents' own lengths, not a constant.
        let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
        let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
        let (_v, meta) = norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
        let entry = meta.entry(1).unwrap();
        let d0 = norms::norm_value(&nvd, entry, 0).unwrap().unwrap();
        let d1 = norms::norm_value(&nvd, entry, 1).unwrap().unwrap();
        assert_ne!(d0, d1, "norms must vary with field length");
        assert_eq!(small_float::byte4_to_int(d0 as u8), 1);
        assert_eq!(small_float::byte4_to_int(d1 as u8), 4);
    }

    /// `omit_norms_field` is the opt-out, and it is the only norms knob.
    #[test]
    fn omit_norms_field_removes_the_column_and_says_so_in_the_fnm() {
        let tmp = tempdir("norms-opted-out");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.omit_norms_field("body").unwrap();
        writer.add_document(doc_with_body("a", "fox")).unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
        let fis = lucene_codecs::field_infos::parse(&fnm, &sci.segment_id, "").unwrap();
        let body = fis.fields.iter().find(|f| f.name == "body").unwrap();
        assert!(body.omit_norms);

        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert!(
            !si.files.iter().any(|f| f.ends_with(".nvm")),
            "no norms files should have been written"
        );
    }

    /// A document that does **not** carry an indexed field gets no norm at
    /// all, and one that carries it but tokenizes to nothing gets an
    /// explicit `0` -- Java's `PerField.finish`, which runs only for a field
    /// the document actually has and writes `0` for `invertState.length == 0`.
    /// Before c35 both were a dense `0`, so an absent field and an empty one
    /// were indistinguishable in the `.nvd`.
    #[test]
    fn a_document_without_the_field_gets_no_norm_and_an_empty_one_gets_zero() {
        let tmp = tempdir("norms-sparse");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        // doc 0 has `body`; doc 1 has no `body` field at all; doc 2 has an
        // empty `body`.
        writer
            .add_document(doc_with_body("a", "fox jumps"))
            .unwrap();
        writer.add_document(doc("b")).unwrap();
        writer.add_document(doc_with_body("c", "")).unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
        let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
        let (_v, meta) = norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
        let entry = meta.entry(1).unwrap();
        assert_eq!(
            norms::norm_value(&nvd, entry, 0).unwrap().map(|v| v as u8),
            Some(small_float::int_to_byte4(2))
        );
        assert_eq!(
            norms::norm_value(&nvd, entry, 1).unwrap(),
            None,
            "a doc that does not carry the field has no norm"
        );
        assert_eq!(
            norms::norm_value(&nvd, entry, 2).unwrap(),
            Some(0),
            "a doc that carries an empty field has an explicit zero norm"
        );
    }

    #[test]
    fn commit_writes_readable_length_dependent_norms_for_multiple_docs() {
        let tmp = tempdir("norms-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        // doc 0: "fox" -> length 1; doc 1: "the quick brown fox jumps" ->
        // length 5; doc 2: "fox fox fox" -> length 3 (repeated term still
        // counts every occurrence, not just distinct terms).
        writer.add_document(doc_with_body("a", "fox")).unwrap();
        writer
            .add_document(doc_with_body("b", "the quick brown fox jumps"))
            .unwrap();
        writer
            .add_document(doc_with_body("c", "fox fox fox"))
            .unwrap();
        let sis = writer.commit().unwrap().clone();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];

        // Stored fields are still intact (backward-compatible).
        assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b", "c"]);

        // The norms files exist and are listed in `.si`.
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for ext in ["nvm", "nvd"] {
            let name = format!("{}.{ext}", sci.segment_name);
            assert!(si.files.contains(&name), "missing {name} in .si files");
            assert!(
                dir.list_all().unwrap().contains(&name),
                "missing {name} on disk"
            );
        }

        // Readable via the existing, unmodified norms read side.
        let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
        let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
        let (_version, parsed) = lucene_codecs::norms::parse_meta(&nvm, &sci.segment_id, "")
            .expect("parse_meta on IndexWriter-produced .nvm");
        let entry = parsed.entry(1).expect("field 1 (body) has a norms entry");
        assert!(entry.is_dense());

        let norm_a = lucene_codecs::norms::norm_value(&nvd, entry, 0)
            .unwrap()
            .unwrap();
        let norm_b = lucene_codecs::norms::norm_value(&nvd, entry, 1)
            .unwrap()
            .unwrap();
        let norm_c = lucene_codecs::norms::norm_value(&nvd, entry, 2)
            .unwrap()
            .unwrap();

        // Decode back to lengths via this crate's own BM25 read-side
        // decoder -- these are exact for small lengths (below the 24-value
        // subnormal threshold), so this also pins down the exact expected
        // byte, not just "some nonzero value".
        assert_eq!(lucene_util::small_float::byte4_to_int(norm_a as u8), 1);
        assert_eq!(lucene_util::small_float::byte4_to_int(norm_b as u8), 5);
        assert_eq!(lucene_util::small_float::byte4_to_int(norm_c as u8), 3);

        // The core proof this is real, length-dependent computation and not
        // a constant: three different lengths produce three different norm
        // bytes.
        assert_ne!(norm_a, norm_b);
        assert_ne!(norm_b, norm_c);
        assert_ne!(norm_a, norm_c);
    }

    /// The inverse of `an_indexed_field_gets_norms_with_no_opt_in_at_all`:
    /// once every indexed field has opted out, the commit is stored-only
    /// again and writes no `.nvm`/`.nvd` at all.
    #[test]
    fn commit_with_every_field_opted_out_of_norms_stays_stored_only() {
        let tmp = tempdir("no-norms-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.omit_norms_field("body").unwrap();

        writer.add_document(doc_with_body("a", "fox")).unwrap();
        let sis = writer.commit().unwrap().clone();
        let sci = &sis.segments[0];

        assert!(!dir
            .list_all()
            .unwrap()
            .contains(&format!("{}.nvm", sci.segment_name)));
        assert!(!dir
            .list_all()
            .unwrap()
            .contains(&format!("{}.nvd", sci.segment_name)));
    }

    #[test]
    fn commit_with_norms_but_no_pending_docs_writes_no_norms_files() {
        let tmp = tempdir("norms-commit-no-docs");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.commit().unwrap();
        assert!(dir
            .list_all()
            .unwrap()
            .iter()
            .all(|f| !f.ends_with(".nvm") && !f.ends_with(".nvd")));
    }

    // ---------------------------------------------------------------------
    // File lifecycle: `IndexFileDeleter`.
    //
    // Every assertion below is on the *directory listing*, not on the writer's
    // in-memory state. An orphan file is by definition something no in-memory
    // structure points at, so only `list_all()` can prove it is gone.
    // ---------------------------------------------------------------------

    /// `dir.list_all()` filtered to Lucene index files, sorted -- the shape
    /// every lifecycle assertion below compares.
    fn index_files(dir: &FsDirectory) -> Vec<String> {
        let mut files: Vec<String> = dir
            .list_all()
            .unwrap()
            .into_iter()
            .filter(|f| crate::index_file_deleter::is_index_file_name(f))
            .collect();
        files.sort();
        files
    }

    /// A prepared commit has already flushed its segment to disk and written a
    /// `pending_segments_N`. Rolling it back must leave the directory exactly as
    /// it was before the prepare -- previously every one of those files stayed
    /// forever.
    #[test]
    fn rollback_after_a_prepared_commit_deletes_every_file_that_prepare_wrote() {
        let tmp = tempdir("deleter-rollback-prepare");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        writer.add_document(doc_with_body("a", "kept")).unwrap();
        writer.commit().unwrap();
        let after_first_commit = index_files(&dir);

        writer
            .add_document(doc_with_body("b", "rolled back"))
            .unwrap();
        writer.prepare_commit().unwrap();
        let after_prepare = index_files(&dir);
        assert!(
            after_prepare.len() > after_first_commit.len(),
            "prepare must have written the segment's files and a pending commit: {after_prepare:?}"
        );
        assert!(after_prepare
            .iter()
            .any(|f| f.starts_with("pending_segments")));

        writer.rollback();

        assert_eq!(
            index_files(&dir),
            after_first_commit,
            "rollback must leave exactly the files the last commit references"
        );
        // And the surviving commit is still readable.
        let sis = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(read_all_docs(&dir, &sis), vec!["a"]);
    }

    /// Each commit supersedes the previous `segments_N`; the default
    /// `KeepOnlyLastCommitDeletionPolicy` deletes it.
    #[test]
    fn each_commit_deletes_the_segments_file_it_supersedes() {
        let tmp = tempdir("deleter-superseded-commit");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("a")).unwrap();
        writer.commit().unwrap();
        assert!(index_files(&dir).contains(&"segments_1".to_string()));

        writer.add_document(doc("b")).unwrap();
        writer.commit().unwrap();

        let files = index_files(&dir);
        assert!(files.contains(&"segments_2".to_string()));
        assert!(
            !files.contains(&"segments_1".to_string()),
            "the superseded commit generation must be deleted: {files:?}"
        );
        // Exactly one commit file at any time under the default policy.
        assert_eq!(
            files.iter().filter(|f| f.starts_with("segments")).count(),
            1,
            "{files:?}"
        );
        // Both segments' data files survive -- only the commit file died.
        let sis = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(sis.segments.len(), 2);
        assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b"]);
    }

    /// `NoDeletionPolicy`: every generation stays, which is the property a
    /// replication setup relies on.
    #[test]
    fn the_keep_all_deletion_policy_retains_every_commit_generation() {
        let tmp = tempdir("deleter-keep-all");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer
            .set_deletion_policy(crate::index_file_deleter::DeletionPolicy::KeepAll)
            .unwrap();

        for id in ["a", "b", "c"] {
            writer.add_document(doc(id)).unwrap();
            writer.commit().unwrap();
        }

        let files = index_files(&dir);
        for name in ["segments_1", "segments_2", "segments_3"] {
            assert!(
                files.contains(&name.to_string()),
                "{name} missing: {files:?}"
            );
        }

        // Switching back to the default reclaims them immediately, which is
        // `IndexFileDeleter.revisitPolicy()`.
        writer
            .set_deletion_policy(crate::index_file_deleter::DeletionPolicy::KeepOnlyLastCommit)
            .unwrap();
        let files = index_files(&dir);
        assert!(!files.contains(&"segments_1".to_string()), "{files:?}");
        assert!(!files.contains(&"segments_2".to_string()), "{files:?}");
        assert!(files.contains(&"segments_3".to_string()), "{files:?}");
    }

    /// A merge's source segments are unreferenced the instant the commit that
    /// replaced them supersedes the previous one -- their files must go.
    #[test]
    fn an_automatic_merge_deletes_its_source_segments_files() {
        let tmp = tempdir("deleter-merge-sources");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        for (id, text) in [
            ("a", "shared apple"),
            ("b", "shared banana"),
            ("c", "shared cherry"),
        ] {
            writer.add_document(doc_with_body(id, text)).unwrap();
            writer.commit().unwrap();
        }

        let segments = writer.segment_infos().segments.clone();
        assert_eq!(segments.len(), 1, "the tight policy must have merged");
        let merged = &segments[0].segment_name;

        let files = index_files(&dir);
        for source in ["_0", "_1", "_2"] {
            assert!(
                !files.iter().any(|f| f.starts_with(&format!("{source}."))
                    || f.starts_with(&format!("{source}_"))),
                "merge source {source}'s files must be reclaimed: {files:?}"
            );
        }
        assert!(
            files.iter().any(|f| f.starts_with(&format!("{merged}."))),
            "the merged segment's own files must survive: {files:?}"
        );
        // The merged segment is still fully readable (merge order is the
        // merge policy's, not insertion order, so compare as a set).
        let sis = segment_infos::read_latest(&dir).unwrap();
        let mut docs = read_all_docs(&dir, &sis);
        docs.sort();
        assert_eq!(docs, vec!["a", "b", "c"]);
    }

    /// A second delete round writes `_0_2.liv` and abandons `_0_1.liv`. The
    /// superseded generation must be deleted, not accumulated.
    #[test]
    fn a_second_delete_round_deletes_the_live_docs_generation_it_supersedes() {
        let fx = open_fixture();
        let doc_in = fx.doc_in();
        let tmp = tempdir("deleter-liv-generations");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);

        let first_round = [SegmentDeleteSource {
            segment_name: "_0",
            fields: &fx.fields,
            doc_in: Some(&doc_in),
            live_docs: None,
            max_doc: fx.max_doc,
        }];

        let sis = writer
            .delete_documents_with_sources(&first_round, "body", b"cat")
            .unwrap()
            .clone();
        let files = index_files(&dir);
        assert!(files.contains(&"_0_1.liv".to_string()), "{files:?}");

        // The second round has to build on the first round's live docs, exactly
        // as a reader-pool-backed caller would -- otherwise `apply_deletes`
        // rejects the round for a del_count that does not match the bitset.
        let liv = dir.open("_0_1.liv").unwrap().to_vec();
        let live_docs = lucene_codecs::live_docs::parse(
            &liv,
            &fx.segment_id,
            1,
            fx.max_doc,
            sis.segments[0].del_count as usize,
        )
        .unwrap();
        let sources = [SegmentDeleteSource {
            segment_name: "_0",
            fields: &fx.fields,
            doc_in: Some(&doc_in),
            live_docs: Some(&live_docs),
            max_doc: fx.max_doc,
        }];

        writer
            .delete_documents_with_sources(&sources, "body", b"dog")
            .unwrap();
        let files = index_files(&dir);
        assert!(files.contains(&"_0_2.liv".to_string()), "{files:?}");
        assert!(
            !files.contains(&"_0_1.liv".to_string()),
            "the superseded .liv generation must be deleted: {files:?}"
        );
    }

    /// Opening a writer over a directory a crashed session left behind reclaims
    /// its orphans: a `pending_segments_N` from a prepare that never finished,
    /// and the segment files of a flush no commit ever named. Files that do not
    /// look like Lucene index files are never touched.
    #[test]
    fn opening_a_writer_reclaims_the_orphans_a_crashed_session_left() {
        let tmp = tempdir("deleter-open-sweep");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        {
            let mut writer =
                IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
            writer.add_document(doc("a")).unwrap();
            writer.commit().unwrap();
        }
        let committed = index_files(&dir);

        // Simulate a crash: a half-finished prepare and an uncommitted flush's
        // files, plus one file that is none of this port's business.
        write_file(&dir, "pending_segments_7", b"garbage").unwrap();
        write_file(&dir, "_9.fdt", b"orphaned flush").unwrap();
        write_file(&dir, "_9_Lucene104_0.tim", b"orphaned flush").unwrap();
        std::fs::write(tmp.join("README.txt"), b"not ours").unwrap();

        let writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        assert_eq!(
            index_files(&dir),
            committed,
            "every orphan must be reclaimed at open, and nothing else touched"
        );
        assert!(
            tmp.join("README.txt").exists(),
            "a non-index file is not ours to delete"
        );

        // `inflateGens`: the reclaimed names must never be handed out again.
        assert!(
            writer.segment_infos().counter > 9,
            "counter must be inflated past the orphaned segment name, got {}",
            writer.segment_infos().counter
        );
        assert!(
            writer.segment_infos().generation >= 7,
            "generation must be inflated past the orphaned pending commit, got {}",
            writer.segment_infos().generation
        );
    }

    // ---------------------------------------------------------------------
    // RAM accounting and the automatic flush trigger.
    // ---------------------------------------------------------------------

    #[test]
    fn ram_bytes_used_tracks_the_buffered_documents_and_resets_on_flush() {
        let tmp = tempdir("ram-accounting");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        assert_eq!(writer.ram_bytes_used(), 0);
        writer.add_document(doc("a")).unwrap();
        let one = writer.ram_bytes_used();
        assert!(one > 0, "a buffered document must cost something");

        writer
            .add_document(doc("a much longer identifier value than the first one"))
            .unwrap();
        let two = writer.ram_bytes_used();
        assert!(
            two > 2 * one,
            "the counter must reflect the actual string bytes, not a per-doc constant: \
             {one} then {two}"
        );

        writer.commit().unwrap();
        assert_eq!(writer.ram_bytes_used(), 0, "a flush resets the counter");
    }

    #[test]
    fn max_buffered_docs_flushes_a_segment_without_committing_it() {
        let tmp = tempdir("max-buffered-docs");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_max_buffered_docs(2).unwrap();

        writer.add_document(doc("a")).unwrap();
        assert_eq!(writer.pending_doc_count(), 1);
        writer.add_document(doc("b")).unwrap();
        assert_eq!(
            writer.pending_doc_count(),
            0,
            "the second document must have tripped the flush"
        );

        // Flushed, but not committed: a fresh reader sees nothing yet.
        let files = index_files(&dir);
        assert!(files.iter().any(|f| f.ends_with(".fdt")), "{files:?}");
        assert!(
            !files.iter().any(|f| f.starts_with("segments")),
            "an automatic flush must not publish a commit: {files:?}"
        );

        writer.add_document(doc("c")).unwrap();
        let sis = writer.commit().unwrap().clone();
        assert_eq!(
            sis.segments.len(),
            2,
            "the auto-flushed segment and the final one must both be in the commit"
        );
        assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b", "c"]);
    }

    /// The whole point of the flush trigger: peak memory stops growing with the
    /// number of documents added between commits.
    #[test]
    fn the_ram_buffer_bounds_the_buffer_no_matter_how_many_documents_arrive() {
        let tmp = tempdir("ram-buffer-bound");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        // Small enough that a few hundred short documents cross it.
        writer.set_ram_buffer_size_mb(0.01).unwrap();

        let mut peak = 0usize;
        for i in 0..2_000 {
            writer.add_document(doc(&format!("doc-{i}"))).unwrap();
            peak = peak.max(writer.ram_bytes_used());
        }
        assert!(
            peak < 2 * (0.01 * 1024.0 * 1024.0) as usize,
            "peak buffered bytes must stay near the configured limit, got {peak}"
        );

        let sis = writer.commit().unwrap().clone();
        assert!(
            sis.segments.len() > 1,
            "the run must have auto-flushed more than once, got {} segment(s)",
            sis.segments.len()
        );
        let docs = read_all_docs(&dir, &sis);
        assert_eq!(docs.len(), 2_000, "no document may be lost across flushes");
        assert_eq!(docs[0], "doc-0");
        assert_eq!(docs[1_999], "doc-1999");
    }

    /// A rollback must discard automatically flushed segments too -- and delete
    /// their files, since no commit ever named them.
    #[test]
    fn rollback_discards_auto_flushed_segments_and_deletes_their_files() {
        let tmp = tempdir("rollback-auto-flush");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        writer.add_document(doc("kept")).unwrap();
        writer.commit().unwrap();
        let after_commit = index_files(&dir);

        writer.set_max_buffered_docs(2).unwrap();
        writer.add_document(doc("x")).unwrap();
        writer.add_document(doc("y")).unwrap();
        assert!(
            index_files(&dir).len() > after_commit.len(),
            "the auto-flush must have written a segment"
        );

        writer.rollback();
        assert_eq!(
            index_files(&dir),
            after_commit,
            "rollback must reclaim an auto-flushed but uncommitted segment"
        );

        let sis = writer.commit().unwrap().clone();
        assert_eq!(read_all_docs(&dir, &sis), vec!["kept"]);
    }

    #[test]
    fn the_auto_flush_setters_port_javas_validation_exactly() {
        let tmp = tempdir("auto-flush-validation");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        assert_eq!(writer.ram_buffer_size_mb(), DEFAULT_RAM_BUFFER_SIZE_MB);
        assert_eq!(writer.max_buffered_docs(), DEFAULT_MAX_BUFFERED_DOCS);
        assert_eq!(DEFAULT_MAX_BUFFERED_DOCS, DISABLE_AUTO_FLUSH);

        // "ramBufferSize should be > 0.0 MB when enabled"
        assert!(matches!(
            writer.set_ram_buffer_size_mb(0.0).unwrap_err(),
            Error::InvalidRamBufferSize(_)
        ));
        assert!(matches!(
            writer.set_ram_buffer_size_mb(-2.0).unwrap_err(),
            Error::InvalidRamBufferSize(_)
        ));
        // "maxBufferedDocs must at least be 2 when enabled"
        assert!(matches!(
            writer.set_max_buffered_docs(1).unwrap_err(),
            Error::InvalidMaxBufferedDocs(1)
        ));
        assert!(matches!(
            writer.set_max_buffered_docs(0).unwrap_err(),
            Error::InvalidMaxBufferedDocs(0)
        ));

        // "at least one of ramBufferSize and maxBufferedDocs must be enabled"
        assert!(matches!(
            writer
                .set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB)
                .unwrap_err(),
            Error::BothAutoFlushTriggersDisabled
        ));
        writer.set_max_buffered_docs(10).unwrap();
        // ...now that doc-count flushing is on, disabling RAM flushing is legal.
        writer
            .set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB)
            .unwrap();
        assert!(matches!(
            writer
                .set_max_buffered_docs(DISABLE_AUTO_FLUSH)
                .unwrap_err(),
            Error::BothAutoFlushTriggersDisabled
        ));
    }

    /// `IndexWriter.deleteUnusedFiles()` is the fallible way to force the sweep
    /// `rollback()` performs silently.
    #[test]
    fn delete_unused_files_reclaims_an_orphan_written_behind_the_writers_back() {
        let tmp = tempdir("delete-unused-files");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.add_document(doc("a")).unwrap();
        writer.commit().unwrap();
        let committed = index_files(&dir);

        write_file(&dir, "_ff.fdt", b"orphan").unwrap();
        assert!(index_files(&dir).contains(&"_ff.fdt".to_string()));

        writer.delete_unused_files().unwrap();
        assert_eq!(index_files(&dir), committed);
    }

    // ---------------------------------------------------------------
    // Sequence numbers and buffered deletes (`c7-delete-queue`).
    //
    // These tests exist to prove the three properties
    // `crate::buffered_updates` claims, not merely that the methods run:
    //   1. every mutating call returns a strictly larger seqNo,
    //   2. a delete reaches every document buffered before it and none
    //      buffered after it, and
    //   3. a delete reaches every segment flushed before it and none
    //      flushed after it.
    // ---------------------------------------------------------------

    /// A `body` field with real postings, so the writer can resolve a delete
    /// term against segments it wrote itself -- which is the whole point:
    /// without postings there is no term dictionary to resolve against and
    /// every delete is a silent no-op.
    fn seq_fields() -> Vec<FieldInfo> {
        vec![stored_only_field("id", 0), body_field(1)]
    }

    fn seq_writer<'d>(dir: &'d FsDirectory) -> IndexWriter<'d> {
        let mut writer = IndexWriter::open(dir, seq_fields(), "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer
    }

    fn body_term(text: &str) -> Term {
        Term::new("body", text.as_bytes())
    }

    /// Every live document's `id`, in commit order.
    fn visible_ids(dir: &FsDirectory, infos: &SegmentInfos) -> Vec<String> {
        read_all_docs(dir, infos)
    }

    /// **`IndexWriter` drops a 100%-deleted segment when the deletes are
    /// applied, and real Lucene says so** (ledger item 11b).
    ///
    /// `IndexWriter.finishApply` removes from the in-memory `SegmentInfos`
    /// every segment `closeSegmentStates` found fully deleted
    /// (`rld.isFullyDeleted() && mergePolicy.keepFullyDeletedSegment(..) ==
    /// false`, where `PendingDeletes.isFullyDeleted` is
    /// `getDelCount() == maxDoc()` -- hard deletes only). This port kept such
    /// a segment in the commit forever: a segment nothing can ever match,
    /// carried by every later open, merge and `CheckIndex`.
    ///
    /// Ground truth is `fixtures/src/GenFullyDeletedDrop.java`, which runs the
    /// same four scripts through a real `IndexWriter` and records the
    /// committed segment count, each segment's `(maxDoc, delCount)`, and the
    /// visible ids. No index is committed: the bytes of an index with a
    /// dropped segment are indistinguishable from those of one that never had
    /// it, so the *behaviour* is what there is to record.
    ///
    /// `partial` is the control -- a segment with one of two documents deleted
    /// survives -- so this cannot pass by dropping every segment with deletes.
    #[test]
    fn a_fully_deleted_segment_is_dropped_exactly_where_real_lucene_drops_it() {
        let manifest = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/fully_deleted_drop/manifest.properties"
        ))
        .expect("run scripts/gen-fixtures.sh --only GenFullyDeletedDrop first");
        let get = |key: &str| -> String {
            manifest
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
                .to_string()
        };

        let scenarios: Vec<String> = get("scenarios").split(',').map(str::to_string).collect();
        assert_eq!(scenarios, ["drop", "partial", "all", "block"]);

        for name in &scenarios {
            let tmp = tempdir(&format!("fully-deleted-{name}"));
            let dir = FsDirectory::open(&tmp);
            let mut writer = seq_writer(&dir);
            match name.as_str() {
                "drop" => {
                    writer.add_document(doc_with_body("a", "shared")).unwrap();
                    writer.add_document(doc_with_body("b", "shared")).unwrap();
                    writer.flush().unwrap();
                    writer
                        .delete_documents_by_term(&[body_term("shared")])
                        .unwrap();
                    writer.add_document(doc_with_body("c", "other")).unwrap();
                    writer.add_document(doc_with_body("d", "other")).unwrap();
                }
                "partial" => {
                    writer.add_document(doc_with_body("a", "shared")).unwrap();
                    writer.add_document(doc_with_body("b", "kept")).unwrap();
                    writer.flush().unwrap();
                    writer
                        .delete_documents_by_term(&[body_term("shared")])
                        .unwrap();
                    writer.add_document(doc_with_body("c", "other")).unwrap();
                    writer.add_document(doc_with_body("d", "other")).unwrap();
                }
                "all" => {
                    writer.add_document(doc_with_body("a", "shared")).unwrap();
                    writer.add_document(doc_with_body("b", "shared")).unwrap();
                    writer.flush().unwrap();
                    writer.add_document(doc_with_body("c", "shared")).unwrap();
                    writer.add_document(doc_with_body("d", "shared")).unwrap();
                    writer.flush().unwrap();
                    writer
                        .delete_documents_by_term(&[body_term("shared")])
                        .unwrap();
                }
                "block" => {
                    writer
                        .add_documents(vec![doc_with_body("p1", "key"), doc_with_body("c1", "key")])
                        .unwrap();
                    writer.commit().unwrap();
                    writer
                        .update_documents(
                            body_term("key"),
                            vec![doc_with_body("p2", "key"), doc_with_body("c2", "key")],
                        )
                        .unwrap();
                }
                other => panic!("unknown scenario {other}"),
            }
            let infos = writer.commit().unwrap().clone();

            let expected_count: usize = get(&format!("{name}.segment_count")).parse().unwrap();
            assert_eq!(
                infos.segments.len(),
                expected_count,
                "{name}: real Lucene commits {expected_count} segment(s), this port \
                 {}",
                infos.segments.len()
            );

            let expected_shape = get(&format!("{name}.segment_shape"));
            let got_shape = infos
                .segments
                .iter()
                .map(|sci| {
                    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
                    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
                    format!("{}:{}", si.doc_count, sci.del_count)
                })
                .collect::<Vec<_>>()
                .join(",");
            assert_eq!(
                got_shape, expected_shape,
                "{name}: per-segment (maxDoc, delCount)"
            );

            let mut expected_ids: Vec<String> = get(&format!("{name}.visible_ids"))
                .split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            expected_ids.sort();
            let mut got_ids = visible_ids(&dir, &infos);
            got_ids.sort();
            assert_eq!(got_ids, expected_ids, "{name}: visible documents");
        }
    }

    /// [`MergePolicyConfig::keep_fully_deleted_segments`] is
    /// `MergePolicy.keepFullyDeletedSegment`, and it is the whole reason the
    /// drop is not unconditional: `SoftDeletesRetentionMergePolicy` returns
    /// `true` from it, so a writer retaining soft-deleted documents must be
    /// able to keep a segment the *hard* delete count has written off.
    #[test]
    fn keep_fully_deleted_segments_suppresses_the_drop() {
        let tmp = tempdir("keep-fully-deleted");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer.set_merge_policy(Some(MergePolicyConfig {
            keep_fully_deleted_segments: true,
            // A merge would drop the emptied segment by the other route
            // (`execute_merge`'s zero-live-document check), so keep the policy
            // from proposing one.
            max_merged_segment_size: 0,
            ..MergePolicyConfig::default()
        }));
        writer.add_document(doc_with_body("a", "shared")).unwrap();
        writer.add_document(doc_with_body("b", "shared")).unwrap();
        writer.flush().unwrap();
        writer
            .delete_documents_by_term(&[body_term("shared")])
            .unwrap();
        writer.add_document(doc_with_body("c", "other")).unwrap();
        let infos = writer.commit().unwrap().clone();

        assert_eq!(
            infos.segments.len(),
            2,
            "the emptied segment must be kept when the policy says so"
        );
        assert_eq!(infos.segments[0].del_count, 2);
        assert_eq!(visible_ids(&dir, &infos), vec!["c"]);
    }

    #[test]
    fn every_mutating_method_returns_a_strictly_increasing_sequence_number() {
        let tmp = tempdir("seqno-monotonic");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);

        let mut seqs = vec![writer.add_document(doc_with_body("a", "alpha")).unwrap()];
        seqs.push(
            writer
                .add_documents(vec![doc_with_body("b", "beta"), doc_with_body("c", "beta")])
                .unwrap(),
        );
        seqs.push(
            writer
                .update_document(body_term("alpha"), doc_with_body("a2", "alpha"))
                .unwrap(),
        );
        seqs.push(
            writer
                .update_documents(body_term("beta"), vec![doc_with_body("b2", "beta")])
                .unwrap(),
        );
        seqs.push(
            writer
                .delete_documents_by_term(&[body_term("gamma")])
                .unwrap(),
        );
        seqs.push(
            writer
                .delete_documents_by_query(&[DeleteQuery::Term(body_term("delta"))])
                .unwrap(),
        );
        seqs.push(
            writer
                .add_document_with_custom_freq_terms(doc_with_body("d", "delta"), Vec::new())
                .unwrap(),
        );

        // Java: "seqNo must start at 1 because some APIs negate this to also
        // return a boolean".
        assert_eq!(seqs[0], 1);
        assert!(
            seqs.windows(2).all(|w| w[0] < w[1]),
            "sequence numbers must be strictly increasing: {seqs:?}"
        );

        // The doc-values half of the surface needs a writer with doc-values
        // fields, so it gets its own writer -- but the same assertion, because
        // the property is about the whole mutating surface, not about one
        // configuration of it.
        let tmp2 = tempdir("seqno-monotonic-dv");
        let dir2 = FsDirectory::open(&tmp2);
        let mut dvw = dv_writer(&dir2);
        let mut dv_seqs = vec![dvw.add_document(doc_with_body("a", "alpha")).unwrap()];
        dv_seqs.push(
            dvw.update_numeric_doc_value(body_term("alpha"), "soft", 1)
                .unwrap(),
        );
        dv_seqs.push(
            dvw.update_binary_doc_value(body_term("alpha"), "payload", b"v")
                .unwrap(),
        );
        dv_seqs.push(
            dvw.update_doc_values(
                body_term("alpha"),
                &[DocValuesUpdate::Numeric {
                    term: body_term("alpha"),
                    field: "soft".into(),
                    value: None,
                }],
            )
            .unwrap(),
        );
        dv_seqs.push(
            dvw.soft_update_document(
                body_term("alpha"),
                doc_with_body("b", "alpha"),
                &[DocValuesUpdate::Numeric {
                    term: body_term("alpha"),
                    field: "soft".into(),
                    value: Some(1),
                }],
            )
            .unwrap(),
        );
        dv_seqs.push(
            dvw.soft_update_documents(
                body_term("alpha"),
                vec![doc_with_body("c", "alpha"), doc_with_body("d", "alpha")],
                &[DocValuesUpdate::Numeric {
                    term: body_term("alpha"),
                    field: "soft".into(),
                    value: Some(1),
                }],
            )
            .unwrap(),
        );
        assert_eq!(dv_seqs[0], 1);
        assert!(
            dv_seqs.windows(2).all(|w| w[0] < w[1]),
            "sequence numbers must be strictly increasing: {dv_seqs:?}"
        );

        // `deleteDocuments(MatchAllDocsQuery)` short-circuits to `deleteAll()`
        // (LUCENE-6379) and still consumes a number, so it belongs here too.
        let tmp3 = tempdir("seqno-monotonic-matchall");
        let dir3 = FsDirectory::open(&tmp3);
        let mut w3 = seq_writer(&dir3);
        let first = w3.add_document(doc_with_body("a", "x")).unwrap();
        let match_all = w3
            .delete_documents_by_query(&[DeleteQuery::MatchAll])
            .unwrap();
        let after = w3.add_document(doc_with_body("b", "x")).unwrap();
        assert!(first < match_all && match_all < after);
    }

    #[test]
    fn a_document_block_takes_exactly_one_sequence_number() {
        // Java's `updateDocuments(Node, docs)` returns one seqNo for the whole
        // block -- it is one operation, seen whole or not at all.
        let tmp = tempdir("seqno-block");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);

        let first = writer.add_document(doc_with_body("a", "x")).unwrap();
        let block = writer
            .add_documents(vec![
                doc_with_body("b", "x"),
                doc_with_body("c", "x"),
                doc_with_body("d", "x"),
            ])
            .unwrap();
        let after = writer.add_document(doc_with_body("e", "x")).unwrap();

        assert_eq!(first, 1);
        assert_eq!(block, 2, "a three-document block is one operation");
        assert_eq!(after, 3);
    }

    #[test]
    fn a_rollback_never_reissues_a_sequence_number() {
        // Java's `rollbackInternal` builds a *fresh* delete queue whose seqNos
        // continue past the aborted ones.
        let tmp = tempdir("seqno-rollback");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);

        let before = writer.add_document(doc_with_body("a", "x")).unwrap();
        writer.rollback();
        let after = writer.add_document(doc_with_body("b", "x")).unwrap();
        assert!(after > before, "{after} must be past {before}");
    }

    #[test]
    fn interleaved_adds_updates_and_deletes_produce_the_expected_visible_set() {
        // The ordering under test, all inside one segment:
        //   #1 add   a (doc 0)
        //   #2 add   b (doc 1)
        //   #3 update body:alpha -> a2   (deletes doc 0, adds doc 2)
        //   #4 add   c (doc 3)
        //   #5 delete body:beta          (deletes doc 1, limit 4)
        // Both deletes carry the buffer position they were issued at, so
        // neither can reach a document added after it.
        let tmp = tempdir("seqno-interleaved");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);

        writer.add_document(doc_with_body("a", "alpha")).unwrap();
        writer.add_document(doc_with_body("b", "beta")).unwrap();
        writer
            .update_document(body_term("alpha"), doc_with_body("a2", "alpha"))
            .unwrap();
        writer.add_document(doc_with_body("c", "gamma")).unwrap();
        writer
            .delete_documents_by_term(&[body_term("beta")])
            .unwrap();

        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["a2", "c"]);
    }

    #[test]
    fn a_delete_does_not_reach_a_document_added_after_it_in_the_same_segment() {
        // `docIDUpto`: the delete is issued when one document is buffered, so
        // it may clear doc 0 and nothing above it -- even though the document
        // added afterwards carries the very same term.
        let tmp = tempdir("seqno-docidupto");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);

        writer
            .add_document(doc_with_body("first", "shared"))
            .unwrap();
        writer
            .delete_documents_by_term(&[body_term("shared")])
            .unwrap();
        writer
            .add_document(doc_with_body("second", "shared"))
            .unwrap();

        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["second"]);
    }

    #[test]
    fn a_delete_issued_before_any_document_exists_deletes_nothing() {
        let tmp = tempdir("seqno-empty-delete");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);

        writer
            .delete_documents_by_term(&[body_term("shared")])
            .unwrap();
        writer.add_document(doc_with_body("a", "shared")).unwrap();

        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["a"]);
    }

    #[test]
    fn a_delete_applies_to_the_segments_flushed_before_it_and_not_to_the_ones_after() {
        // The cross-segment half of the contract, and the one that is easy to
        // get subtly wrong: `_0` is already on disk when the delete is issued,
        // so it must be cleared; `_1` is flushed afterwards from documents
        // carrying the *same* term, so it must survive untouched.
        let tmp = tempdir("seqno-across-flush");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer.set_max_buffered_docs(2).unwrap();

        writer.add_document(doc_with_body("a", "shared")).unwrap();
        writer.add_document(doc_with_body("b", "shared")).unwrap();
        // The threshold tripped: `_0` exists on disk now.
        assert_eq!(writer.pending_doc_count(), 0);

        writer
            .delete_documents_by_term(&[body_term("shared")])
            .unwrap();

        writer.add_document(doc_with_body("c", "shared")).unwrap();
        writer.add_document(doc_with_body("d", "shared")).unwrap();

        let infos = writer.commit().unwrap().clone();
        // `_0` ends up 100% deleted, so `IndexWriter.finishApply` drops it
        // (`drop_fully_deleted_segments`) -- and its absence is itself the
        // proof that the delete reached *both* of its documents. `_1` must
        // still be here, untouched: a delete that wrongly reached it too would
        // leave no segments at all.
        assert_eq!(
            infos
                .segments
                .iter()
                .map(|s| s.segment_name.as_str())
                .collect::<Vec<_>>(),
            vec!["_1"],
            "`_0` fully deleted and dropped, `_1` survives"
        );
        assert_eq!(visible_ids(&dir, &infos), vec!["c", "d"]);
        assert_eq!(infos.segments[0].del_count, 0);
        assert!(
            !index_files(&dir).iter().any(|f| f.starts_with("_0.")),
            "a dropped segment's files are reclaimed at the commit: {:?}",
            index_files(&dir)
        );
    }

    #[test]
    fn a_delete_issued_before_a_flush_still_reaches_the_segment_that_flush_produces() {
        // The mirror image: the delete is buffered while the documents are
        // still in RAM, so the segment the *next* flush produces must come out
        // with those documents already dead.
        let tmp = tempdir("seqno-before-flush");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);

        writer.add_document(doc_with_body("a", "shared")).unwrap();
        writer.add_document(doc_with_body("b", "keep")).unwrap();
        writer
            .delete_documents_by_term(&[body_term("shared")])
            .unwrap();

        let infos = writer.commit().unwrap().clone();
        assert_eq!(infos.segments.len(), 1);
        assert_eq!(infos.segments[0].del_count, 1);
        assert_eq!(visible_ids(&dir, &infos), vec!["b"]);
    }

    #[test]
    fn a_delete_issued_after_a_commit_reaches_the_committed_segment() {
        let tmp = tempdir("seqno-after-commit");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);

        writer.add_document(doc_with_body("a", "shared")).unwrap();
        writer.add_document(doc_with_body("b", "keep")).unwrap();
        writer.commit().unwrap();

        writer
            .delete_documents_by_term(&[body_term("shared")])
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["b"]);
    }

    #[test]
    fn update_document_replaces_across_a_flush_boundary() {
        // `updateDocument` is a delete of everything that came before plus an
        // add: the replacement lands in the new segment, the original dies in
        // the committed one.
        let tmp = tempdir("seqno-update-across");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);

        writer.add_document(doc_with_body("v1", "key")).unwrap();
        writer.commit().unwrap();

        writer
            .update_document(body_term("key"), doc_with_body("v2", "key"))
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["v2"]);
    }

    #[test]
    fn two_updates_of_the_same_term_in_one_buffer_leave_only_the_newest() {
        // `BufferedUpdates.addTerm` keeps the *higher* `docIDUpto`, which is
        // exactly what makes the second update delete the first update's
        // replacement as well as the original.
        let tmp = tempdir("seqno-double-update");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);

        writer.add_document(doc_with_body("v1", "key")).unwrap();
        writer
            .update_document(body_term("key"), doc_with_body("v2", "key"))
            .unwrap();
        writer
            .update_document(body_term("key"), doc_with_body("v3", "key"))
            .unwrap();

        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["v3"]);
    }

    #[test]
    fn a_rollback_after_a_buffered_delete_was_applied_restores_the_committed_segment_list() {
        // The failure this pins: applying a buffered delete bumps a *committed*
        // segment's `del_gen` in the writer's in-memory view and writes a new
        // `.liv` that no commit names yet. `rollback()` then runs the deleter's
        // `refresh()`, which reclaims exactly such an unreferenced file. Without
        // `SegmentInfos.rollbackSegmentInfos(rollbackSegments)` the in-memory
        // view would survive the rollback pointing at a deleted file, and the
        // next commit would write a `segments_N` naming it.
        let tmp = tempdir("seqno-rollback-restores");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer.add_document(doc_with_body("a", "shared")).unwrap();
        // A second document the delete does not match, so the segment is not
        // left 100% deleted: `finishApply`'s drop would otherwise remove it
        // before the rollback, and this test is about the rollback restoring a
        // segment whose `.liv` generation advanced, not about the drop.
        writer.add_document(doc_with_body("keep", "other")).unwrap();
        writer.commit().unwrap();
        let committed_del_gen = writer.segment_infos().segments[0].del_gen;
        assert_eq!(committed_del_gen, -1);

        // Force the delete to be resolved and written *before* the rollback.
        writer
            .delete_documents_by_term(&[body_term("shared")])
            .unwrap();
        writer.flush().unwrap();
        assert_eq!(writer.segment_infos().segments[0].del_gen, 1);
        assert!(index_files(&dir).contains(&"_0_1.liv".to_string()));

        writer.rollback();

        // The in-memory view is back to the commit...
        assert_eq!(writer.segment_infos().segments[0].del_gen, -1);
        assert_eq!(writer.segment_infos().segments[0].del_count, 0);
        // ...the orphaned `.liv` is gone...
        assert!(!index_files(&dir).contains(&"_0_1.liv".to_string()));
        // ...and the next commit names only files that exist.
        let infos = writer.commit().unwrap().clone();
        for sci in &infos.segments {
            let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
            let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
            for f in sci.files(&si.files) {
                assert!(index_files(&dir).contains(&f), "commit names missing {f}");
            }
        }
        assert_eq!(visible_ids(&dir, &infos), vec!["a", "keep"]);
    }

    /// End to end for `SegmentCommitInfo.generationAdvanced()`: the id a
    /// commit records for a segment must change when that segment changes.
    /// Read back out of `segments_N`, not off the in-memory writer, because
    /// the commit file is what a replication or NRT consumer actually
    /// compares.
    #[test]
    fn a_delete_changes_the_segment_commit_id_the_next_commit_records() {
        let tmp = tempdir("sci-id-changes-across-commits");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer.add_document(doc_with_body("a", "keep")).unwrap();
        writer.add_document(doc_with_body("b", "doomed")).unwrap();
        writer.commit().unwrap();

        let before = segment_infos::read_latest(&dir).unwrap().segments[0].sci_id;
        assert!(before.is_some());

        writer
            .delete_documents_by_term(&[body_term("doomed")])
            .unwrap();
        writer.commit().unwrap();

        let after = segment_infos::read_latest(&dir).unwrap().segments[0].sci_id;
        assert!(after.is_some());
        assert_ne!(
            after, before,
            "a deleted document left the segment-commit id unchanged"
        );
    }

    #[test]
    fn a_rollback_discards_buffered_deletes_along_with_buffered_documents() {
        let tmp = tempdir("seqno-rollback-deletes");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);

        writer.add_document(doc_with_body("a", "shared")).unwrap();
        writer.commit().unwrap();

        writer
            .delete_documents_by_term(&[body_term("shared")])
            .unwrap();
        writer.rollback();

        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["a"]);
    }

    // --- delete-by-query ---

    #[test]
    fn delete_documents_by_query_resolves_a_term_query() {
        let tmp = tempdir("query-delete-term");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer.add_document(doc_with_body("a", "alpha")).unwrap();
        writer.add_document(doc_with_body("b", "beta")).unwrap();
        writer.commit().unwrap();

        writer
            .delete_documents_by_query(&[DeleteQuery::Term(body_term("alpha"))])
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["b"]);
    }

    #[test]
    fn delete_documents_by_query_resolves_a_prefix_and_stops_at_the_span_end() {
        let tmp = tempdir("query-delete-prefix");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer.add_document(doc_with_body("a", "apple")).unwrap();
        writer.add_document(doc_with_body("b", "apricot")).unwrap();
        writer.add_document(doc_with_body("c", "banana")).unwrap();
        writer.commit().unwrap();

        writer
            .delete_documents_by_query(&[DeleteQuery::Prefix {
                field: "body".into(),
                prefix: b"ap".to_vec(),
            }])
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["c"]);
    }

    #[test]
    fn delete_documents_by_query_resolves_an_inclusive_and_an_exclusive_range() {
        let tmp = tempdir("query-delete-range");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        for (id, body) in [("a", "aaa"), ("b", "bbb"), ("c", "ccc"), ("d", "ddd")] {
            writer.add_document(doc_with_body(id, body)).unwrap();
        }
        writer.commit().unwrap();

        // [bbb, ddd) -> bbb and ccc.
        writer
            .delete_documents_by_query(&[DeleteQuery::TermRange {
                field: "body".into(),
                lower: Some(b"bbb".to_vec()),
                upper: Some(b"ddd".to_vec()),
                include_lower: true,
                include_upper: false,
            }])
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["a", "d"]);
    }

    #[test]
    fn delete_documents_by_query_resolves_boolean_composition() {
        let tmp = tempdir("query-delete-boolean");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer
            .add_document(doc_with_body("a", "red round"))
            .unwrap();
        writer
            .add_document(doc_with_body("b", "red square"))
            .unwrap();
        writer
            .add_document(doc_with_body("c", "blue round"))
            .unwrap();
        writer.commit().unwrap();

        // red AND round -> only "a".
        writer
            .delete_documents_by_query(&[DeleteQuery::All(vec![
                DeleteQuery::Term(body_term("red")),
                DeleteQuery::Term(body_term("round")),
            ])])
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["b", "c"]);
    }

    #[test]
    fn delete_documents_by_query_resolves_a_negation_over_live_docs_only() {
        let tmp = tempdir("query-delete-not");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer.add_document(doc_with_body("a", "keep")).unwrap();
        writer.add_document(doc_with_body("b", "drop")).unwrap();
        writer.add_document(doc_with_body("c", "drop")).unwrap();
        writer.commit().unwrap();

        // NOT keep -> b and c.
        writer
            .delete_documents_by_query(&[DeleteQuery::Not(Box::new(DeleteQuery::Term(body_term(
                "keep",
            ))))])
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["a"]);
    }

    #[test]
    fn delete_documents_by_query_resolves_a_union() {
        let tmp = tempdir("query-delete-any");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer.add_document(doc_with_body("a", "alpha")).unwrap();
        writer.add_document(doc_with_body("b", "beta")).unwrap();
        writer.add_document(doc_with_body("c", "gamma")).unwrap();
        writer.commit().unwrap();

        writer
            .delete_documents_by_query(&[DeleteQuery::Any(vec![
                DeleteQuery::Term(body_term("alpha")),
                DeleteQuery::Term(body_term("gamma")),
            ])])
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["b"]);
    }

    #[test]
    fn delete_documents_by_query_specialises_match_all_into_delete_all() {
        // LUCENE-6379: a `MatchAllDocsQuery` short-circuits to `deleteAll()`,
        // which drops whole segments instead of writing an all-zero `.liv`.
        let tmp = tempdir("query-delete-matchall");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer.add_document(doc_with_body("a", "alpha")).unwrap();
        writer.commit().unwrap();

        writer
            .delete_documents_by_query(&[DeleteQuery::MatchAll])
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        assert!(infos.segments.is_empty(), "deleteAll drops the segments");
        assert!(visible_ids(&dir, &infos).is_empty());
    }

    #[test]
    fn a_query_delete_honours_the_doc_id_upto_limit_within_its_own_segment() {
        let tmp = tempdir("query-delete-limit");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);

        writer
            .add_document(doc_with_body("first", "shared"))
            .unwrap();
        writer
            .delete_documents_by_query(&[DeleteQuery::Term(body_term("shared"))])
            .unwrap();
        writer
            .add_document(doc_with_body("second", "shared"))
            .unwrap();

        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["second"]);
    }

    // --- document blocks / `hasBlocks` ---

    fn parsed_si(dir: &FsDirectory, sci: &SegmentCommitInfo) -> segment_info::SegmentInfo {
        let bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        segment_info::parse(&bytes, &sci.segment_id).unwrap()
    }

    #[test]
    fn a_block_add_sets_has_blocks_on_the_flushed_segment() {
        let tmp = tempdir("block-has-blocks");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer
            .add_documents(vec![
                doc_with_body("parent", "p"),
                doc_with_body("child1", "c"),
                doc_with_body("child2", "c"),
            ])
            .unwrap();
        let infos = writer.commit().unwrap().clone();

        assert!(parsed_si(&dir, &infos.segments[0]).has_blocks);
        // Contiguous, ascending doc IDs in the order they were supplied.
        assert_eq!(
            visible_ids(&dir, &infos),
            vec!["parent", "child1", "child2"]
        );
    }

    #[test]
    fn single_document_adds_leave_has_blocks_unset() {
        let tmp = tempdir("block-no-blocks");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer.add_document(doc_with_body("a", "x")).unwrap();
        writer.add_document(doc_with_body("b", "x")).unwrap();
        // A one-document "block" is not a block, matching Java's `numDocs > 1`.
        writer.add_documents(vec![doc_with_body("c", "x")]).unwrap();
        let infos = writer.commit().unwrap().clone();

        assert!(!parsed_si(&dir, &infos.segments[0]).has_blocks);
    }

    #[test]
    fn has_blocks_does_not_leak_from_one_flush_into_the_next() {
        let tmp = tempdir("block-not-leaked");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer
            .add_documents(vec![doc_with_body("p", "p"), doc_with_body("c", "c")])
            .unwrap();
        writer.commit().unwrap();
        writer.add_document(doc_with_body("plain", "x")).unwrap();
        let infos = writer.commit().unwrap().clone();

        assert!(parsed_si(&dir, &infos.segments[0]).has_blocks);
        assert!(!parsed_si(&dir, &infos.segments[1]).has_blocks);
    }

    #[test]
    fn an_automatic_flush_never_splits_a_document_block() {
        // The contiguity guarantee: the threshold is consulted once per
        // `add_documents` call, so a block that crosses it still lands whole.
        let tmp = tempdir("block-not-split");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer.set_max_buffered_docs(2).unwrap();

        writer
            .add_documents(vec![
                doc_with_body("p", "p"),
                doc_with_body("c1", "c"),
                doc_with_body("c2", "c"),
                doc_with_body("c3", "c"),
            ])
            .unwrap();
        let infos = writer.commit().unwrap().clone();

        assert_eq!(infos.segments.len(), 1, "the block must not be split");
        assert_eq!(visible_ids(&dir, &infos), vec!["p", "c1", "c2", "c3"]);
    }

    #[test]
    fn update_documents_deletes_the_old_block_and_adds_the_new_one_atomically() {
        let tmp = tempdir("block-update");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer
            .add_documents(vec![doc_with_body("p1", "key"), doc_with_body("c1", "key")])
            .unwrap();
        writer.commit().unwrap();

        writer
            .update_documents(
                body_term("key"),
                vec![doc_with_body("p2", "key"), doc_with_body("c2", "key")],
            )
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["p2", "c2"]);
        // The old block was the whole of its segment, so applying the delete
        // leaves it 100% deleted and `finishApply` drops it -- the replacement
        // is the only segment left.
        assert_eq!(infos.segments.len(), 1, "the emptied segment is dropped");
        assert!(parsed_si(&dir, &infos.segments[0]).has_blocks);
    }

    // --- doc-values updates and soft deletes ---

    fn numeric_dv_field(name: &str, number: i32) -> FieldInfo {
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..stored_only_field(name, number)
        }
    }

    fn binary_dv_field(name: &str, number: i32) -> FieldInfo {
        FieldInfo {
            doc_values_type: DocValuesType::Binary,
            ..stored_only_field(name, number)
        }
    }

    fn dv_writer<'d>(dir: &'d FsDirectory) -> IndexWriter<'d> {
        let fields = vec![
            stored_only_field("id", 0),
            body_field(1),
            numeric_dv_field("soft", 2),
            binary_dv_field("payload", 3),
        ];
        let mut writer = IndexWriter::open(dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer
    }

    /// A writer whose `soft` field actually has a base doc-values column, so
    /// an update has something to supersede. `dv_writer` deliberately does
    /// not: a field that never had a column in the segment is Java's
    /// `FieldInfos.FieldNumbers.constructFieldInfo` case, and both are worth
    /// exercising.
    fn dv_base_writer<'d>(dir: &'d FsDirectory) -> IndexWriter<'d> {
        let mut writer = dv_writer(dir);
        writer.set_doc_values_field(Some("soft")).unwrap();
        writer
    }

    fn doc_with_dv(id: &str, body: &str, soft: i64) -> Document {
        let mut doc = doc_with_body(id, body);
        doc.fields.push(StoredField {
            field_number: 2,
            value: FieldValue::Long(soft),
        });
        doc
    }

    /// Opens `field_number`'s **current** doc-values column exactly the way
    /// `SegmentDocValuesProducer` does: the generation
    /// `SegmentCommitInfo.dv_update_files` names for the field, read with the
    /// generation-suffixed index header those files carry, against the
    /// generational `.fnm`'s one-field `FieldInfos`. Returns `None` when the
    /// field has no update generation at all.
    fn open_dv_generation(
        dir: &FsDirectory,
        sci: &SegmentCommitInfo,
        field_number: i32,
    ) -> Option<(lucene_codecs::doc_values::DocValuesMeta, Vec<u8>, i32)> {
        let (_, files) = sci
            .dv_update_files
            .iter()
            .find(|(n, _)| *n == field_number)?;
        let dvm_name = files.iter().find(|f| f.ends_with(".dvm")).unwrap().clone();
        let dvd_name = files.iter().find(|f| f.ends_with(".dvd")).unwrap().clone();
        let suffix = dvm_name
            .strip_prefix(&format!("{}_", sci.segment_name))
            .and_then(|s| s.strip_suffix(".dvm"))
            .unwrap()
            .to_string();

        let fnm_name = sci
            .field_infos_files
            .first()
            .expect("a doc-values update always writes a FieldInfos generation");
        let fnm = dir.open(fnm_name).unwrap();
        let infos = lucene_codecs::field_infos::parse(
            &fnm,
            &sci.segment_id,
            &lucene_util::base36::to_base36(sci.field_infos_gen),
        )
        .unwrap();
        let field = infos
            .fields
            .iter()
            .find(|f| f.number == field_number)
            .unwrap()
            .clone();
        assert_ne!(
            field.doc_values_gen, -1,
            "the generational .fnm must record the field's docValuesGen"
        );
        let only = lucene_codecs::field_infos::FieldInfos {
            fields: vec![field],
        };

        let dvm = dir.open(&dvm_name).unwrap();
        let dvd = dir.open(&dvd_name).unwrap();
        let (_, meta) =
            lucene_codecs::doc_values::parse_meta(&dvm, &sci.segment_id, &suffix, &only).unwrap();
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        Some((meta, dvd.to_vec(), si.doc_count))
    }

    /// The whole NUMERIC column a doc-values update generation left behind,
    /// one entry per doc. `None` means the doc has no value -- a `reset`, or a
    /// doc the base never had a value for.
    fn read_numeric_column(
        dir: &FsDirectory,
        sci: &SegmentCommitInfo,
        field_number: i32,
    ) -> Vec<Option<i64>> {
        let Some((meta, data, max_doc)) = open_dv_generation(dir, sci, field_number) else {
            return Vec::new();
        };
        let entry = meta.numeric_entry(field_number).unwrap();
        (0..max_doc)
            .map(|doc| lucene_codecs::doc_values::numeric_value(&data, entry, doc).unwrap())
            .collect()
    }

    /// [`read_numeric_column`] for BINARY doc values.
    fn read_binary_column(
        dir: &FsDirectory,
        sci: &SegmentCommitInfo,
        field_number: i32,
    ) -> Vec<Option<Vec<u8>>> {
        let Some((meta, data, max_doc)) = open_dv_generation(dir, sci, field_number) else {
            return Vec::new();
        };
        let entry = meta.binary_entry(field_number).unwrap();
        (0..max_doc)
            .map(|doc| {
                lucene_codecs::doc_values::binary_value(&data, entry, doc)
                    .unwrap()
                    .map(<[u8]>::to_vec)
            })
            .collect()
    }

    /// **`has_blocks` survives an automatic merge.** It is one byte in the
    /// `.si` that nothing else in the write path sets, and a merged segment
    /// holding blocks but reporting `hasBlocks = false` reads back perfectly
    /// while silently invalidating every parent/child join query against it.
    /// Java's `IndexWriter.mergeMiddle` ORs it across `merge.segments`.
    /// Caught by the Tier-2 review of `c22-sorted-merge`.
    #[test]
    fn an_automatic_merge_keeps_has_blocks() {
        let tmp = tempdir("has-blocks-merge");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));
        // Two block-bearing commits and one plain one, so the merge has to
        // OR the flag rather than copy the first source's.
        writer.add_documents(vec![doc("a1"), doc("a2")]).unwrap();
        writer.commit().unwrap();
        writer.add_document(doc("b")).unwrap();
        writer.commit().unwrap();
        writer.add_documents(vec![doc("c1"), doc("c2")]).unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(
            infos.segments.len(),
            1,
            "the three segments must have merged"
        );
        let sci = &infos.segments[0];
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert!(
            si.has_blocks,
            "the merged .si must still report that it holds document blocks"
        );
        // Each block is still contiguous and in order, which is what the flag
        // promises. (The *segments* may be merged in whatever order the
        // policy proposed them, so the block positions are not fixed.)
        let docs = read_all_docs(&dir, &infos);
        assert_eq!(docs.len(), 5);
        for (first, second) in [("a1", "a2"), ("c1", "c2")] {
            let i = docs.iter().position(|d| d == first).unwrap();
            assert_eq!(
                docs.get(i + 1).map(String::as_str),
                Some(second),
                "block {first}/{second} was split by the merge: {docs:?}"
            );
        }
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    /// **Positional postings survive an automatic merge.** `execute_merge`
    /// filtered its merge-time postings fields to `Docs`/`DocsAndFreqs` back
    /// when that was all this writer could flush; once positional flush
    /// landed, that filter meant a merged segment carried no postings for the
    /// field at all while its `.fnm` still declared
    /// `DocsAndFreqsAndPositions` -- an indexed field with no registered
    /// postings producer, which reads back as having no terms and raises
    /// nothing. Caught by the Tier-2 review of `c22-sorted-merge`.
    #[test]
    fn an_automatic_merge_carries_positional_postings_through() {
        let tmp = tempdir("positional-postings-merge");
        let dir = FsDirectory::open(&tmp);
        let positions_body_field = fi::FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            ..stored_only_field("body", 1)
        };
        let fields = vec![stored_only_field("id", 0), positions_body_field.clone()];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        for (id, body) in [
            ("a", "quick fox jumps"),
            ("b", "the fox sleeps"),
            ("c", "a fox again"),
        ] {
            writer.add_document(doc_with_body(id, body)).unwrap();
            writer.commit().unwrap();
        }
        let infos = writer.segment_infos().clone();
        assert_eq!(
            infos.segments.len(),
            1,
            "the three segments must have merged"
        );
        let sci = &infos.segments[0];

        let seg = per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert!(
            si.files.contains(&format!("{seg}.pos")),
            "the merged segment must carry a .pos"
        );
        let tim = dir.open(&format!("{seg}.tim")).unwrap();
        let tip = dir.open(&format!("{seg}.tip")).unwrap();
        let tmd = dir.open(&format!("{seg}.tmd")).unwrap();
        let doc_bytes = dir.open(&format!("{seg}.doc")).unwrap();
        let pos_bytes = dir.open(&format!("{seg}.pos")).unwrap();
        let field_infos = fi::FieldInfos {
            fields: vec![
                fi::FieldInfo {
                    index_options: IndexOptions::None,
                    ..stored_only_field("id", 0)
                },
                positions_body_field,
            ],
        };
        let block_fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            3,
        )
        .unwrap();
        let doc_in = DocInput::open(
            &doc_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let pos_in = lucene_codecs::postings::PosInput::open(
            &pos_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let field = block_fields.field("body").unwrap();
        // "fox" is at position 1 in every one of the three documents, which
        // are now three merged doc ids of one segment.
        let fox = field
            .positions(b"fox", Some(&doc_in), &pos_in, None)
            .unwrap()
            .expect("the merged dictionary must still have \"fox\"");
        assert_eq!(fox.len(), 3);
        for doc_positions in &fox {
            assert_eq!(doc_positions.len(), 1);
            assert_eq!(doc_positions[0].position, 1);
        }
        // ...and a document-unique term still lands on its own document.
        let quick = field
            .positions(b"quick", Some(&doc_in), &pos_in, None)
            .unwrap()
            .unwrap();
        assert_eq!(quick.len(), 1);
        assert_eq!(quick[0][0].position, 0);
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    /// **The other four doc-values types survive an automatic merge too.**
    /// `execute_merge` used to build only NUMERIC merge sources while
    /// `segment_stats` offered every `.dvd`-bearing segment, so a BINARY (or
    /// SORTED / SORTED_NUMERIC / SORTED_SET) column was dropped by the merge
    /// and `describe_written_files` then zeroed its `DocValuesType` to keep
    /// the `.fnm` honest -- a valid, `CheckIndex`-clean segment with the data
    /// gone. Caught by the Tier-2 review of `c22-sorted-merge`, which is why
    /// this asserts on the *values*, per document, rather than on the merge
    /// having happened.
    #[test]
    fn an_automatic_merge_carries_every_doc_values_type_through() {
        let tmp = tempdir("dv-types-and-merge-policy");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            numeric_dv_field("num", 1),
            binary_dv_field("bin", 2),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("num")).unwrap();
        writer.add_doc_values_field("bin").unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        for (id, num) in [("a", 1i64), ("b", 2), ("c", 3)] {
            writer
                .add_document(Document {
                    fields: vec![
                        StoredField {
                            field_number: 0,
                            value: FieldValue::String(id.to_string()),
                        },
                        StoredField {
                            field_number: 1,
                            value: FieldValue::Long(num),
                        },
                        StoredField {
                            field_number: 2,
                            value: FieldValue::String(format!("payload-{id}")),
                        },
                    ],
                })
                .unwrap();
            writer.commit().unwrap();
        }

        let infos = writer.segment_infos().clone();
        assert_eq!(
            infos.segments.len(),
            1,
            "the three segments must have merged"
        );
        let sci = &infos.segments[0];
        let seg = per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME);
        let dvm = dir.open(&format!("{seg}.dvm")).unwrap().to_vec();
        let dvd = dir.open(&format!("{seg}.dvd")).unwrap().to_vec();
        let (_v, meta) = doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &lucene_codecs::field_infos::FieldInfos {
                fields: vec![
                    stored_only_field("id", 0),
                    numeric_dv_field("num", 1),
                    binary_dv_field("bin", 2),
                ],
            },
        )
        .unwrap();
        let numeric = meta.numeric_entry(1).expect("the NUMERIC column survived");
        let binary = meta.binary_entry(2).expect("the BINARY column survived");

        let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
        let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
        let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        for d in 0..3 {
            let id = match &reader.document(d).unwrap().fields[0].value {
                FieldValue::String(v) => v.clone(),
                other => panic!("unexpected stored value {other:?}"),
            };
            let expected_num = match id.as_str() {
                "a" => 1,
                "b" => 2,
                "c" => 3,
                other => panic!("unexpected id {other}"),
            };
            assert_eq!(
                doc_values::numeric_value(&dvd, numeric, d).unwrap(),
                Some(expected_num),
                "num at doc {d} (id={id})"
            );
            assert_eq!(
                doc_values::binary_value(&dvd, binary, d).unwrap(),
                Some(format!("payload-{id}").as_bytes()),
                "bin at doc {d} (id={id})"
            );
        }
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    /// **A sort tier none of the merged documents has a value for.** The
    /// merged column is all-missing, which
    /// `Lucene90DocValuesConsumer.writeValues` writes as
    /// `docsWithFieldOffset = -2` rather than omitting the field. Omitting it
    /// would zero the field's `DocValuesType` in the merged `.fnm` while the
    /// merged `.si` still declares the sort over it: real Lucene's
    /// `DocValues.getNumeric` *throws* for a field whose `FieldInfo` exists
    /// but declares no doc values, so `CheckIndex.testSort` would fail rather
    /// than degrade -- and this port could never merge that segment again,
    /// because `read_sort_keys` would return `MergeSortColumnMissing` out of
    /// every subsequent `commit`. Caught by the Tier-2 review of
    /// `c22-sorted-merge`.
    #[test]
    fn a_sort_tier_no_merged_document_has_a_value_for_is_still_written() {
        let tmp = tempdir("sorted-merge-empty-tier");
        let dir = FsDirectory::open(&tmp);
        let mut writer = sorted_merge_writer(
            &dir,
            &[
                sort_field("rank", false, SortMissingValue::Last),
                sort_field("tie", false, SortMissingValue::Last),
            ],
        );
        writer.set_merge_policy(None);
        // `tie` is never set on any document, so the second tier's merged
        // column has no values at all.
        for batch in [vec![("b", 20i64)], vec![("a", 10)], vec![("c", 30)]] {
            for (id, rank) in batch {
                let body: String = std::iter::repeat_n(format!("t{id}"), rank as usize / 10)
                    .collect::<Vec<_>>()
                    .join(" ");
                let mut document = sortable_doc(id, rank, 0, &body);
                // Drop the `tie` stored field, so no document has a value for
                // the second sort tier.
                document.fields.retain(|f| f.field_number != 2);
                writer
                    .add_document_with_vectors(
                        document,
                        vec![DocumentVector::float32(
                            "v",
                            vec![rank as f32, 1.0, 2.0, 3.0],
                        )],
                    )
                    .unwrap();
            }
            writer.commit().unwrap();
        }

        let names: Vec<String> = writer
            .segment_infos()
            .segments
            .iter()
            .map(|s| s.segment_name.clone())
            .collect();
        writer.execute_merge(&names).unwrap();
        let infos = writer.segment_infos().clone();
        assert_eq!(infos.segments.len(), 1);
        let sci = &infos.segments[0];

        // The merged `.fnm` still declares `tie` as NUMERIC doc values, and
        // the merged `.dvm` has an entry for it with no values.
        let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
        let merged_fields = fi::parse(&fnm, &sci.segment_id, "").unwrap();
        let tie = merged_fields.field_by_number(2).unwrap();
        assert_eq!(tie.doc_values_type, DocValuesType::Numeric);
        assert_eq!(read_base_numeric_column(&dir, sci, 2, 3), vec![None; 3]);
        assert_eq!(
            read_base_numeric_column(&dir, sci, 1, 3),
            vec![Some(10), Some(20), Some(30)]
        );
        assert_eq!(read_all_docs(&dir, &infos), vec!["a", "b", "c"]);
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
        // ...and the segment can be merged again, which is the half that
        // would have failed with `MergeSortColumnMissing`.
        writer
            .add_document_with_vectors(
                {
                    let mut d = sortable_doc("d", 40, 0, "td td td td");
                    d.fields.retain(|f| f.field_number != 2);
                    d
                },
                vec![DocumentVector::float32("v", vec![40.0, 1.0, 2.0, 3.0])],
            )
            .unwrap();
        writer.commit().unwrap();
        let names: Vec<String> = writer
            .segment_infos()
            .segments
            .iter()
            .map(|s| s.segment_name.clone())
            .collect();
        writer.execute_merge(&names).unwrap();
        assert_eq!(
            read_all_docs(&dir, writer.segment_infos()),
            vec!["a", "b", "c", "d"]
        );
    }

    /// A doc-values update against a segment produced by a **merge**, not by
    /// a flush. Worth its own case because the merged `.fnm` is written by
    /// `merge::describe_written_files`, which this batch taught to zero
    /// `doc_values_type` for a field the merge wrote no column for -- the
    /// exact state `field_updates::check_updatable` used to reject outright
    /// (c17 finding 14/15). If the two disagreed, the update would fail on a
    /// merged segment while working on a flushed one.
    #[test]
    fn a_doc_values_update_still_works_against_a_merged_segment() {
        let tmp = tempdir("dv-update-after-merge");
        let dir = FsDirectory::open(&tmp);
        let mut writer = dv_base_writer(&dir);
        writer.set_merge_policy(Some(tight_merge_policy()));
        for (id, body, soft) in [("a", "alpha", 1i64), ("b", "beta", 2), ("c", "gamma", 3)] {
            let mut document = doc_with_body(id, body);
            document.fields.push(StoredField {
                field_number: 2,
                value: FieldValue::Long(soft),
            });
            writer.add_document(document).unwrap();
            writer.commit().unwrap();
        }
        let infos = writer.segment_infos().clone();
        assert_eq!(
            infos.segments.len(),
            1,
            "the three segments must have merged"
        );
        assert_eq!(infos.segments[0].doc_values_gen, -1, "no generation yet");

        writer
            .update_numeric_doc_value(body_term("alpha"), "soft", 42)
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(infos.segments[0].doc_values_gen, 1);
        // The generation supersedes only the updated document; the merged
        // base column still answers for the other two.
        let sci = &infos.segments[0];
        let seg = per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME);
        let base_dvm = dir.open(&format!("{seg}.dvm")).unwrap().to_vec();
        assert!(!base_dvm.is_empty(), "the merge wrote a base column");
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    #[test]
    fn update_numeric_doc_value_writes_a_generation_the_reader_can_replay() {
        let tmp = tempdir("dv-numeric-update");
        let dir = FsDirectory::open(&tmp);
        let mut writer = dv_writer(&dir);
        writer.add_document(doc_with_body("a", "alpha")).unwrap();
        writer.add_document(doc_with_body("b", "beta")).unwrap();
        writer.commit().unwrap();

        writer
            .update_numeric_doc_value(body_term("alpha"), "soft", 42)
            .unwrap();
        let infos = writer.commit().unwrap().clone();

        let sci = &infos.segments[0];
        assert_eq!(sci.doc_values_gen, 1, "one generation was written");
        assert_eq!(
            sci.field_infos_gen, 1,
            "a doc-values update also writes a FieldInfos generation"
        );
        assert_eq!(sci.del_count, 0, "a doc-values update never deletes");
        // Lucene's own file names, base-36 generation and per-field suffix.
        let files: Vec<&String> = sci
            .dv_update_files
            .iter()
            .find(|(n, _)| *n == 2)
            .map(|(_, f)| f.iter().collect())
            .expect("field 2 has an update generation");
        assert_eq!(
            files.iter().map(|f| f.as_str()).collect::<Vec<_>>(),
            vec![
                "_0_1_Lucene90_0.dvm",
                "_0_1_Lucene90_0.dvd",
                "_0_1_Lucene90_0.dvs"
            ]
        );
        assert_eq!(sci.field_infos_files, vec!["_0_1.fnm".to_string()]);
        assert_eq!(read_numeric_column(&dir, sci, 2), vec![Some(42), None]);
        // The documents themselves are untouched.
        assert_eq!(visible_ids(&dir, &infos), vec!["a", "b"]);
    }

    #[test]
    fn update_doc_values_with_a_null_value_records_a_removal_not_a_zero() {
        // Java: "If a doc values fields data is null the existing value is
        // removed from all documents matching the term" -- which reaches
        // `DocValuesFieldUpdates.reset(doc)`, not `add(doc, 0)`.
        let tmp = tempdir("dv-reset");
        let dir = FsDirectory::open(&tmp);
        // A *base* column, so "removed" and "never had one" are actually
        // distinguishable in the rewritten generation.
        let mut writer = dv_base_writer(&dir);
        writer.add_document(doc_with_dv("a", "alpha", 7)).unwrap();
        writer.add_document(doc_with_dv("b", "beta", 8)).unwrap();
        writer.commit().unwrap();

        writer
            .update_doc_values(
                body_term("alpha"),
                &[DocValuesUpdate::Numeric {
                    term: body_term("alpha"),
                    field: "soft".into(),
                    value: None,
                }],
            )
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(
            read_numeric_column(&dir, &infos.segments[0], 2),
            vec![None, Some(8)],
            "doc 0's value is removed (not zeroed), doc 1's base value survives"
        );
    }

    #[test]
    fn update_binary_doc_value_writes_a_binary_generation() {
        let tmp = tempdir("dv-binary-update");
        let dir = FsDirectory::open(&tmp);
        let mut writer = dv_writer(&dir);
        writer.add_document(doc_with_body("a", "alpha")).unwrap();
        writer.commit().unwrap();

        writer
            .update_binary_doc_value(body_term("alpha"), "payload", b"hello")
            .unwrap();
        let infos = writer.commit().unwrap().clone();

        let sci = &infos.segments[0];
        assert_eq!(
            read_binary_column(&dir, sci, 3),
            vec![Some(b"hello".to_vec())]
        );
    }

    #[test]
    fn successive_doc_values_updates_supersede_each_other_at_a_new_generation() {
        // Lucene's format is a *full rewrite* per generation, not a stack of
        // deltas: generation 2 is the field's complete column, so generation
        // 1's files stop being referenced (and are reclaimed) the moment it
        // lands. What must survive is only the semantic contract -- the newest
        // write wins.
        let tmp = tempdir("dv-generations");
        let dir = FsDirectory::open(&tmp);
        let mut writer = dv_writer(&dir);
        writer.add_document(doc_with_body("a", "alpha")).unwrap();
        writer.commit().unwrap();

        writer
            .update_numeric_doc_value(body_term("alpha"), "soft", 1)
            .unwrap();
        writer.commit().unwrap();
        writer
            .update_numeric_doc_value(body_term("alpha"), "soft", 2)
            .unwrap();
        let infos = writer.commit().unwrap().clone();

        let sci = &infos.segments[0];
        assert_eq!(sci.doc_values_gen, 2);
        assert_eq!(sci.field_infos_gen, 2);
        let (_, files) = sci.dv_update_files.iter().find(|(n, _)| *n == 2).unwrap();
        assert!(
            files.iter().all(|f| f.starts_with("_0_2_")),
            "only the newest generation stays referenced: {files:?}"
        );
        assert_eq!(read_numeric_column(&dir, sci, 2), vec![Some(2)]);
        assert!(
            !index_files(&dir).iter().any(|f| f.starts_with("_0_1_")),
            "the superseded generation must be reclaimed: {:?}",
            index_files(&dir)
        );
    }

    #[test]
    fn update_doc_values_rejects_an_unknown_or_wrongly_typed_field() {
        let tmp = tempdir("dv-validation");
        let dir = FsDirectory::open(&tmp);
        let mut writer = dv_writer(&dir);

        let unknown = writer
            .update_numeric_doc_value(body_term("alpha"), "nope", 1)
            .unwrap_err();
        assert!(matches!(
            unknown,
            Error::UnknownDocValuesUpdateField(name) if name == "nope"
        ));

        // `payload` is BINARY, so a numeric update against it is refused --
        // Java's `verifyOrCreateDvOnlyField` throws for the same mismatch.
        let wrong_type = writer
            .update_numeric_doc_value(body_term("alpha"), "payload", 1)
            .unwrap_err();
        assert!(matches!(
            wrong_type,
            Error::WrongDocValuesUpdateType { ref field, .. } if field == "payload"
        ));

        let empty = writer
            .update_doc_values(body_term("alpha"), &[])
            .unwrap_err();
        assert!(matches!(empty, Error::NoDocValuesUpdatesSupplied));
    }

    #[test]
    fn soft_update_document_adds_the_new_doc_and_marks_the_old_one_without_deleting_it() {
        // The whole point of a soft delete: the previous version keeps its
        // live bit and its postings, and is instead marked through a
        // doc-values field a retention policy can consult.
        let tmp = tempdir("soft-update");
        let dir = FsDirectory::open(&tmp);
        let mut writer = dv_writer(&dir);
        writer.add_document(doc_with_body("v1", "key")).unwrap();
        writer.commit().unwrap();

        writer
            .soft_update_document(
                body_term("key"),
                doc_with_body("v2", "key"),
                &[DocValuesUpdate::Numeric {
                    term: body_term("key"),
                    field: "soft".into(),
                    value: Some(1),
                }],
            )
            .unwrap();
        let infos = writer.commit().unwrap().clone();

        // Nothing was hard-deleted anywhere.
        assert!(infos.segments.iter().all(|s| s.del_count == 0));
        assert_eq!(visible_ids(&dir, &infos), vec!["v1", "v2"]);
        // ...but the original is marked.
        assert_eq!(
            read_numeric_column(&dir, &infos.segments[0], 2),
            vec![Some(1)]
        );
        // ...and the replacement is not: the marking carries the buffer
        // position it was issued at, exactly like a hard delete would.
        assert!(
            infos.segments[1].dv_update_files.is_empty(),
            "the replacement must not mark itself: {:?}",
            infos.segments[1].dv_update_files
        );
    }

    #[test]
    fn a_delete_after_a_rollback_still_reaches_every_committed_segment() {
        // Two committed segments carry `buffered_deletes_gen` 1 and 2. A
        // rollback used to rewind `BufferedUpdatesStream`'s counter to 1 (as
        // Java's `clear()` does -- safely, because Java then *closes* the
        // writer), so the next packet was stamped gen 1 and `applies_to`
        // rejected it for `_1`: the delete silently reached only the oldest
        // segment. One segment is not enough to see it -- gen 1 == gen 1 passes
        // by luck -- which is why this test uses two.
        let tmp = tempdir("rollback-then-delete");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);

        // Each segment also carries a document the delete does not match, so
        // neither is left 100% deleted and `finishApply`'s drop does not
        // remove the very evidence this test reads (`del_count` per segment).
        writer.add_document(doc_with_body("a", "shared")).unwrap();
        writer
            .add_document(doc_with_body("a-keep", "other"))
            .unwrap();
        writer.commit().unwrap();
        writer.add_document(doc_with_body("b", "shared")).unwrap();
        writer
            .add_document(doc_with_body("b-keep", "other"))
            .unwrap();
        writer.commit().unwrap();
        assert_eq!(writer.segment_infos().segments.len(), 2);

        writer.rollback();

        writer
            .delete_documents_by_term(&[body_term("shared")])
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(
            visible_ids(&dir, &infos),
            vec!["a-keep", "b-keep"],
            "the delete must reach both segments, not just the oldest"
        );
        assert_eq!(
            infos
                .segments
                .iter()
                .map(|s| s.del_count)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
    }

    #[test]
    fn documents_and_deletes_buffered_during_a_prepared_commit_survive_into_the_next_one() {
        // The loss this pins: `finish_commit` installs the `SegmentInfos`
        // snapshot `prepare_commit` took and clears `flushed_segments`, so a
        // flush in between would have its segment *and* every buffered delete
        // it resolved thrown away. With `max_buffered_docs = 2`, the two adds
        // below used to trip the automatic flush inside the window.
        let tmp = tempdir("prepared-window");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer.set_max_buffered_docs(2).unwrap();

        writer.add_document(doc_with_body("a", "shared")).unwrap();
        writer.commit().unwrap();

        writer.prepare_commit().unwrap();
        writer
            .delete_documents_by_term(&[body_term("shared")])
            .unwrap();
        writer.add_document(doc_with_body("b", "kept")).unwrap();
        writer.add_document(doc_with_body("c", "kept")).unwrap();
        // Deferred, not flushed: nothing may be published behind the prepare.
        assert_eq!(writer.pending_doc_count(), 2);

        // The prepared commit publishes exactly what it snapshotted...
        let after_finish = writer.finish_commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &after_finish), vec!["a"]);

        // ...and the next one carries everything buffered during the window.
        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["b", "c"]);
    }

    #[test]
    fn an_explicit_flush_is_refused_while_a_commit_is_prepared() {
        let tmp = tempdir("prepared-flush-refused");
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        writer.add_document(doc_with_body("a", "x")).unwrap();
        writer.prepare_commit().unwrap();
        writer.add_document(doc_with_body("b", "x")).unwrap();

        let err = writer.flush().unwrap_err();
        assert!(matches!(err, Error::PreparedCommitPending("flush")));
        // Nothing was lost by the refusal.
        assert_eq!(writer.pending_doc_count(), 1);
        writer.finish_commit().unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(visible_ids(&dir, &infos), vec!["a", "b"]);
    }

    #[test]
    fn every_generational_file_name_round_trips_through_parse_generation() {
        // The property that ties every generational name this writer produces
        // -- `field_updates`' `.dvm`/`.dvd`/`.dvs` and `.fnm`, and
        // `deletes::liv_file_name` -- to `IndexFileNames.parseGeneration`,
        // which is what `inflateGens` relies on after a crash. Checked across
        // the base-36 boundary (10, 35, 36) rather than at one value, because
        // decimal and base 36 agree below 10 -- which is exactly why writing
        // the generation in decimal went unnoticed until it was checked here.
        let per_field = per_field_codec_suffix(DOC_VALUES_FORMAT_NAME);
        for gen in 1..=100i64 {
            for ext in ["dvm", "dvd", "dvs"] {
                let dv = crate::field_updates::generation_file_name("_0", gen, &per_field, ext);
                assert_eq!(
                    crate::index_file_deleter::parse_generation_for_test(&dv),
                    gen,
                    "doc-values generation name {dv} must read back as generation {gen}"
                );
            }
            let fnm = crate::field_updates::field_infos_gen_file_name("_0", gen);
            assert_eq!(
                crate::index_file_deleter::parse_generation_for_test(&fnm),
                gen,
                "field-infos generation name {fnm} must read back as generation {gen}"
            );
            let liv = deletes::liv_file_name("_0", gen);
            assert_eq!(
                crate::index_file_deleter::parse_generation_for_test(&liv),
                gen,
                "live-docs name {liv} must read back as generation {gen}"
            );
        }
    }

    #[test]
    fn a_doc_values_update_skips_a_document_the_same_packet_just_deleted() {
        // Java runs `applyTermDeletes` -> `applyQueryDeletes` ->
        // `applyDocValuesUpdates` inside one `FrozenBufferedUpdates.apply`, and
        // the last of the three reads `segState.rld.getLiveDocs()` -- which by
        // then reflects the first two. So a document this packet just killed
        // takes no doc-values update. Here `a` is hard-deleted and separately
        // marked in the same buffer; only `b`'s mark may survive.
        let tmp = tempdir("dv-skips-deleted");
        let dir = FsDirectory::open(&tmp);
        let mut writer = dv_writer(&dir);
        writer
            .add_document(doc_with_body("a", "alpha shared"))
            .unwrap();
        writer
            .add_document(doc_with_body("b", "beta shared"))
            .unwrap();
        writer.commit().unwrap();

        writer
            .delete_documents_by_term(&[body_term("alpha")])
            .unwrap();
        writer
            .update_numeric_doc_value(body_term("shared"), "soft", 9)
            .unwrap();
        let infos = writer.commit().unwrap().clone();

        let sci = &infos.segments[0];
        assert_eq!(sci.del_count, 1, "only `a` is deleted");
        assert_eq!(
            read_numeric_column(&dir, sci, 2),
            vec![None, Some(9)],
            "doc 0 was deleted by this packet, so it takes no update"
        );
        assert_eq!(visible_ids(&dir, &infos), vec!["b"]);
    }

    #[test]
    fn soft_update_document_requires_at_least_one_soft_delete_field() {
        let tmp = tempdir("soft-update-empty");
        let dir = FsDirectory::open(&tmp);
        let mut writer = dv_writer(&dir);
        let err = writer
            .soft_update_document(body_term("key"), doc_with_body("v", "key"), &[])
            .unwrap_err();
        assert!(matches!(err, Error::NoSoftDeletesSupplied));
    }

    /// **The last un-mergeable case, closed (c22's carry-over).** A segment
    /// carrying a doc-values *update* used to be withheld from the merge
    /// policy outright: its newest column lives in generational
    /// `.dvm`/`.dvd` files that no `.si` lists, and `execute_merge` read the
    /// base pair -- so merging it would have silently resurrected the
    /// pre-update values into a valid, `CheckIndex`-clean segment.
    ///
    /// `execute_merge` now resolves every field to its **current**
    /// generation, through the same two functions the update path reads its
    /// own base from. This asserts on the merged *values*, not on the merge
    /// having happened: the failure mode is a merge that succeeds with the
    /// old numbers.
    ///
    /// Both shapes an update can take are exercised in one segment: `soft`
    /// has **no base column at all** (the flush wrote no `.dvd`, so the
    /// generation is the field's only column -- Java's
    /// `FieldInfos.FieldNumbers.constructFieldInfo` case), while `payload`
    /// has a base BINARY column this merge must carry forward untouched from
    /// the base pair the same segment also has.
    #[test]
    fn a_segment_with_a_doc_values_update_merges_at_its_newest_generation() {
        let tmp = tempdir("dv-update-merges");
        let dir = FsDirectory::open(&tmp);
        let mut writer = dv_writer(&dir);
        writer.set_doc_values_field(Some("payload")).unwrap();
        writer.set_merge_policy(None);
        writer
            .add_document({
                let mut doc = doc_with_body("a", "alpha");
                doc.fields.push(StoredField {
                    field_number: 3,
                    value: FieldValue::String("base-a".to_string()),
                });
                doc
            })
            .unwrap();
        writer.commit().unwrap();
        writer
            .update_numeric_doc_value(body_term("alpha"), "soft", 5)
            .unwrap();
        writer.commit().unwrap();

        // `soft` lives only in generation 1; `payload` only in the base pair.
        let updated = writer.segment_infos().segments[0].clone();
        assert_eq!(updated.doc_values_gen, 1);
        let si_bytes = dir.open(&format!("{}.si", updated.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &updated.segment_id).unwrap();
        assert!(
            !si.files
                .iter()
                .any(|f| f.ends_with(".dvd") && f.contains("_1_")),
            "the generational `.dvd` is not in the `.si`, which is the whole difficulty"
        );

        // A second segment, then a merge over both.
        writer.set_merge_policy(Some(tight_merge_policy()));
        for i in 0..2 {
            writer
                .add_document({
                    let mut doc = doc_with_body(&format!("x{i}"), "beta");
                    doc.fields.push(StoredField {
                        field_number: 3,
                        value: FieldValue::String(format!("base-x{i}")),
                    });
                    doc
                })
                .unwrap();
            writer.commit().unwrap();
        }

        let infos = writer.segment_infos().clone();
        assert_eq!(infos.segments.len(), 1, "the sources must have merged");
        let merged = &infos.segments[0];
        assert_eq!(
            merged.doc_values_gen, -1,
            "a merge folds every generation back into one base column"
        );

        // The merged columns live in the merged segment's *base*
        // `.dvm`/`.dvd`, not in a generation -- which is the claim.
        let seg = per_field_segment(&merged.segment_name, DOC_VALUES_FORMAT_NAME);
        let dvm = dir.open(&format!("{seg}.dvm")).unwrap();
        let dvd = dir.open(&format!("{seg}.dvd")).unwrap();
        let (_v, meta) = doc_values::parse_meta(
            &dvm,
            &merged.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &fi::FieldInfos {
                fields: vec![
                    stored_only_field("id", 0),
                    body_field(1),
                    numeric_dv_field("soft", 2),
                    binary_dv_field("payload", 3),
                ],
            },
        )
        .unwrap();
        let soft = meta
            .numeric_entry(2)
            .expect("`soft`'s generational column reached the merged base column");
        let payload = meta.binary_entry(3).expect("`payload`'s base column too");

        // Resolve each merged doc id through its stored `id`, so this asserts
        // the pairing rather than an assumed merged order.
        let fdt = dir.open(&format!("{}.fdt", merged.segment_name)).unwrap();
        let fdx = dir.open(&format!("{}.fdx", merged.segment_name)).unwrap();
        let fdm = dir.open(&format!("{}.fdm", merged.segment_name)).unwrap();
        let stored = stored_fields::open(&fdt, &fdx, &fdm, &merged.segment_id, "").unwrap();
        assert_eq!(stored.max_doc(), 3);
        for doc_id in 0..3 {
            let id = doc_value(&stored.document(doc_id).unwrap());
            let expected_soft = if id == "a" { Some(5) } else { None };
            assert_eq!(
                doc_values::numeric_value(&dvd, soft, doc_id).unwrap(),
                expected_soft,
                "`soft` at doc {doc_id} (id={id}): the update's value, not the base's absence"
            );
            assert_eq!(
                doc_values::binary_value(&dvd, payload, doc_id).unwrap(),
                Some(format!("base-{id}").as_bytes()),
                "`payload` at doc {doc_id} (id={id}): carried from the base pair"
            );
        }
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    #[test]
    fn a_doc_values_update_generation_is_referenced_and_survives_the_deleters_sweep() {
        // The generation files are named in `segments_N` through
        // `dv_update_files` and `field_infos_files`, so
        // `SegmentCommitInfo::files()` -- and therefore the deleter -- must
        // see all four. None of them is listed in the `.si`: they did not
        // exist when it was written.
        let tmp = tempdir("dv-refcount");
        let dir = FsDirectory::open(&tmp);
        let mut writer = dv_writer(&dir);
        writer.add_document(doc_with_body("a", "alpha")).unwrap();
        writer.commit().unwrap();
        writer
            .update_numeric_doc_value(body_term("alpha"), "soft", 7)
            .unwrap();
        writer.commit().unwrap();

        let before = index_files(&dir);
        writer.delete_unused_files().unwrap();
        assert_eq!(index_files(&dir), before);
        for expected in [
            "_0_1_Lucene90_0.dvm",
            "_0_1_Lucene90_0.dvd",
            "_0_1_Lucene90_0.dvs",
            "_0_1.fnm",
        ] {
            assert!(
                before.iter().any(|f| f == expected),
                "{expected} must still be there: {before:?}"
            );
        }
    }

    // -----------------------------------------------------------------
    // Vector fields (`set_vector_field` / `add_document_with_vectors`)
    // -----------------------------------------------------------------

    fn vector_field(
        name: &str,
        number: i32,
        dim: i32,
        encoding: VectorEncoding,
        similarity: VectorSimilarityFunction,
    ) -> FieldInfo {
        FieldInfo {
            vector_dimension: dim,
            vector_encoding: encoding,
            vector_similarity_function: similarity,
            ..stored_only_field(name, number)
        }
    }

    /// A deterministic, well-spread vector so a graph over these has real
    /// structure rather than every node being equidistant.
    fn test_vector(dim: usize, seed: i64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761);
        (0..dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s as u64) >> 40) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn open_vector_files(
        dir: &FsDirectory,
        sci: &SegmentCommitInfo,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let seg = per_field_segment(&sci.segment_name, KNN_VECTORS_FORMAT_NAME);
        (
            dir.open(&format!("{seg}.vec")).unwrap().to_vec(),
            dir.open(&format!("{seg}.vemf")).unwrap().to_vec(),
            dir.open(&format!("{seg}.vem")).unwrap().to_vec(),
            dir.open(&format!("{seg}.vex")).unwrap().to_vec(),
        )
    }

    /// The end-to-end shape: documents added with vectors produce a segment
    /// whose four vector files decode back to exactly the vectors that went
    /// in, dense and sparse, float and byte -- and whose graph is a real
    /// graph, not an empty one.
    #[test]
    fn add_document_with_vectors_writes_a_readable_vector_segment() {
        let tmp = tempdir("vectors-round-trip");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            vector_field(
                "dense",
                1,
                8,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
            vector_field(
                "sparse_bytes",
                2,
                4,
                VectorEncoding::Byte,
                VectorSimilarityFunction::DotProduct,
            ),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_vector_field(Some("dense")).unwrap();
        writer.add_vector_field("sparse_bytes").unwrap();

        let n = 800;
        for i in 0..n {
            let mut vectors = vec![DocumentVector::float32("dense", test_vector(8, i as i64))];
            if i % 4 == 0 {
                vectors.push(DocumentVector::byte(
                    "sparse_bytes",
                    vec![(i % 251) as u8, 7, 200, (i / 3 % 251) as u8],
                ));
            }
            writer
                .add_document_with_vectors(doc(&format!("doc{i}")), vectors)
                .unwrap();
        }
        let infos = writer.commit().unwrap().clone();
        assert_eq!(infos.segments.len(), 1);
        let sci = &infos.segments[0];
        let (vec_bytes, vemf, vem, vex) = open_vector_files(&dir, sci);

        let suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
        let flat =
            vectors::FlatVectorsReader::open(&vemf, &vec_bytes, &sci.segment_id, &suffix).unwrap();
        let dense = flat.float_vector_values(1).unwrap();
        assert_eq!(dense.size(), n);
        assert_eq!(dense.dimension(), 8);
        assert_eq!(
            dense.similarity(),
            VectorSimilarityFunction::Euclidean,
            "the similarity must come from the FieldInfo, not a default"
        );
        for ord in 0..n {
            assert_eq!(dense.ord_to_doc(ord).unwrap(), ord, "dense ord == doc");
            assert_eq!(dense.vector(ord).unwrap(), test_vector(8, ord as i64));
        }

        let sparse = flat.byte_vector_values(2).unwrap();
        assert_eq!(sparse.size(), n / 4);
        for ord in 0..sparse.size() {
            let doc_id = ord * 4;
            assert_eq!(sparse.ord_to_doc(ord).unwrap(), doc_id);
            assert_eq!(
                sparse.vector(ord).unwrap(),
                &[(doc_id % 251) as u8, 7, 200, (doc_id / 3 % 251) as u8][..]
            );
        }

        // The graph: 800 vectors is well past HNSW_GRAPH_THRESHOLD, so the
        // dense field must carry a real one and every node must be on level 0.
        let graphs =
            hnsw_vectors::HnswVectorsReader::open(&vem, &vex, &sci.segment_id, &suffix).unwrap();
        let graph = graphs
            .graph(1)
            .unwrap()
            .expect("dense field must have a graph");
        assert!(graph.num_levels() >= 1);
        assert_eq!(
            graph.sorted_nodes_on_level(0).unwrap().len(),
            n as usize,
            "level 0 must contain every vector"
        );
        // 200 byte vectors is below the threshold: Lucene builds no graph, and
        // neither may this writer.
        assert!(
            graphs.graph(2).unwrap().is_none(),
            "a sub-threshold field must have numLevels = 0"
        );
    }

    /// The graph the flush built has to be *usable*, not merely present: an
    /// approximate search over it must land on the same documents an
    /// exhaustive scan over the same `.vec` does. (This is the port's own
    /// end of the check; `VerifyVectorSegment` runs real Lucene's
    /// `KnnFloatVectorQuery` over the same bytes.)
    #[test]
    fn the_flushed_graph_finds_the_same_documents_an_exhaustive_scan_does() {
        let tmp = tempdir("vectors-search");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            vector_field(
                "v",
                1,
                12,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_vector_field(Some("v")).unwrap();
        for i in 0..1000 {
            writer
                .add_document_with_vectors(
                    doc(&format!("doc{i}")),
                    vec![DocumentVector::float32("v", test_vector(12, 5000 + i))],
                )
                .unwrap();
        }
        let infos = writer.commit().unwrap().clone();
        let sci = &infos.segments[0];
        let (vec_bytes, vemf, vem, vex) = open_vector_files(&dir, sci);
        let suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
        let flat =
            vectors::FlatVectorsReader::open(&vemf, &vec_bytes, &sci.segment_id, &suffix).unwrap();
        let graphs =
            hnsw_vectors::HnswVectorsReader::open(&vem, &vex, &sci.segment_id, &suffix).unwrap();
        let values = flat.float_vector_values(1).unwrap();
        let graph = graphs.graph(1).unwrap().unwrap();

        let mut hits = 0usize;
        let mut total = 0usize;
        for q in 0..8 {
            let query = test_vector(12, 90_000 + q);
            let exact: Vec<i32> = values
                .exhaustive_search(&query, 10)
                .unwrap()
                .into_iter()
                .map(|(d, _)| d)
                .collect();
            let mut scorer = values.scorer(&query).unwrap();
            // `SearchOptions::default()` is Java's unfiltered, unseeded
            // `KnnFloatVectorQuery`; the second half of the return is the
            // collector's early-termination flag, which this recall check
            // does not consult.
            let (approx, _early_terminated) = hnsw_vectors::search(
                &mut scorer,
                Some(&graph),
                10,
                u64::MAX,
                hnsw_vectors::SearchOptions::default(),
            )
            .unwrap();
            let approx: Vec<i32> = approx
                .into_iter()
                .map(|(ord, _)| values.ord_to_doc(ord).unwrap())
                .collect();
            total += exact.len();
            hits += approx.iter().filter(|d| exact.contains(d)).count();
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.8,
            "graph search over a flushed segment recalled {recall} of the exact top-10"
        );
    }

    /// A field declared with a dimension that no document ever carried must
    /// not be claimed by the `.fnm`, and must get no per-field format
    /// attributes -- see `fields_with_per_field_attributes`.
    #[test]
    fn a_vector_field_no_document_carried_is_not_claimed_in_the_fnm() {
        let tmp = tempdir("vectors-unused-field");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            vector_field(
                "used",
                1,
                4,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Cosine,
            ),
            vector_field(
                "unused",
                2,
                6,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_vector_field(Some("used")).unwrap();
        writer.add_vector_field("unused").unwrap();
        for i in 0..3 {
            writer
                .add_document_with_vectors(
                    doc(&format!("doc{i}")),
                    vec![DocumentVector::float32("used", test_vector(4, i))],
                )
                .unwrap();
        }
        let infos = writer.commit().unwrap().clone();
        let sci = &infos.segments[0];
        let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
        let parsed = fi::parse(&fnm, &sci.segment_id, "").unwrap();

        let used = parsed.fields.iter().find(|f| f.name == "used").unwrap();
        assert_eq!(used.vector_dimension, 4);
        assert!(used
            .attributes
            .iter()
            .any(|(k, v)| k == "PerFieldKnnVectorsFormat.format" && v == KNN_VECTORS_FORMAT_NAME));
        assert!(used
            .attributes
            .iter()
            .any(|(k, v)| k == "PerFieldKnnVectorsFormat.suffix" && v == PER_FIELD_SUFFIX));

        let unused = parsed.fields.iter().find(|f| f.name == "unused").unwrap();
        assert_eq!(
            unused.vector_dimension, 0,
            "a field no document carried must record dimension 0, or a reader \
             sees a vector-capable field with no vectors"
        );
        assert!(!unused
            .attributes
            .iter()
            .any(|(k, _)| k.starts_with("PerFieldKnnVectorsFormat")));
        // And the `.vemf` must carry only the field that got values.
        let (vec_bytes, vemf, _, _) = open_vector_files(&dir, sci);
        let suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
        let flat =
            vectors::FlatVectorsReader::open(&vemf, &vec_bytes, &sci.segment_id, &suffix).unwrap();
        assert_eq!(flat.fields().len(), 1);
        assert_eq!(flat.fields()[0].field_number, 1);
    }

    /// No opted-in field with any value at all: no vector files, and nothing
    /// in the `.si` naming them.
    #[test]
    fn a_flush_with_no_vectors_writes_no_vector_files() {
        let tmp = tempdir("vectors-none");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            vector_field(
                "v",
                1,
                4,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_vector_field(Some("v")).unwrap();
        writer.add_document(doc("only-stored")).unwrap();
        let infos = writer.commit().unwrap().clone();
        let sci = &infos.segments[0];
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert!(
            !si.files.iter().any(|f| f.ends_with(".vec")),
            "no vector files may be named: {:?}",
            si.files
        );
        assert!(dir
            .list_all()
            .unwrap()
            .iter()
            .all(|f| !f.ends_with(".vec") && !f.ends_with(".vem")));
    }

    /// The four files must be listed in `SegmentInfo.files`; `IndexFileDeleter`,
    /// `CheckIndex` and this port's own `checksum_verify` all walk that list,
    /// so a file missing from it is a file nothing knows about.
    #[test]
    fn the_segment_info_lists_the_four_vector_files() {
        let tmp = tempdir("vectors-si-files");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            vector_field(
                "v",
                1,
                4,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_vector_field(Some("v")).unwrap();
        for i in 0..4 {
            writer
                .add_document_with_vectors(
                    doc(&format!("doc{i}")),
                    vec![DocumentVector::float32("v", test_vector(4, i))],
                )
                .unwrap();
        }
        let infos = writer.commit().unwrap().clone();
        let sci = &infos.segments[0];
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        let seg = per_field_segment(&sci.segment_name, KNN_VECTORS_FORMAT_NAME);
        for ext in ["vec", "vemf", "vem", "vex"] {
            let name = format!("{seg}.{ext}");
            assert!(
                si.files.contains(&name),
                "{name} missing from {:?}",
                si.files
            );
            assert!(
                dir.list_all().unwrap().contains(&name),
                "{name} not written"
            );
        }
    }

    /// Every per-document validation `add_document_with_vectors` runs, one
    /// case each -- and each of them must leave the buffer untouched, since a
    /// half-buffered document would desynchronise `pending_vectors` from
    /// `pending_docs`.
    #[test]
    fn add_document_with_vectors_rejects_malformed_input() {
        let tmp = tempdir("vectors-validation");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            vector_field(
                "f32",
                1,
                4,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
            vector_field(
                "bytes",
                2,
                4,
                VectorEncoding::Byte,
                VectorSimilarityFunction::DotProduct,
            ),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_vector_field(Some("f32")).unwrap();
        writer.add_vector_field("bytes").unwrap();

        // Wrong dimension.
        assert!(matches!(
            writer.add_document_with_vectors(
                doc("a"),
                vec![DocumentVector::float32("f32", vec![1.0, 2.0])]
            ),
            Err(Error::VectorDimensionMismatch(name, 0, 4, 2)) if name == "f32"
        ));
        // Float value for a BYTE field.
        assert!(matches!(
            writer.add_document_with_vectors(
                doc("a"),
                vec![DocumentVector::float32("bytes", vec![1.0, 2.0, 3.0, 4.0])]
            ),
            Err(Error::VectorEncodingMismatch(
                _,
                0,
                VectorEncoding::Byte,
                VectorEncoding::Float32
            ))
        ));
        // Byte value for a FLOAT32 field.
        assert!(matches!(
            writer.add_document_with_vectors(
                doc("a"),
                vec![DocumentVector::byte("f32", vec![1, 2, 3, 4])]
            ),
            Err(Error::VectorEncodingMismatch(
                _,
                0,
                VectorEncoding::Float32,
                VectorEncoding::Byte
            ))
        ));
        // A field that is not opted in.
        assert!(matches!(
            writer.add_document_with_vectors(
                doc("a"),
                vec![DocumentVector::float32("nope", vec![1.0, 2.0, 3.0, 4.0])]
            ),
            Err(Error::UnknownVectorField(name)) if name == "nope"
        ));
        // The same field twice on one document.
        assert!(matches!(
            writer.add_document_with_vectors(
                doc("a"),
                vec![
                    DocumentVector::float32("f32", vec![1.0, 2.0, 3.0, 4.0]),
                    DocumentVector::float32("f32", vec![5.0, 6.0, 7.0, 8.0]),
                ]
            ),
            Err(Error::DuplicateVectorField(name)) if name == "f32"
        ));
        assert_eq!(
            writer.pending_doc_count(),
            0,
            "a rejected document must not be buffered"
        );

        // A non-finite component is caught at flush, by `VectorUtil.checkFinite`
        // in the codec -- the last point that can see the whole field.
        writer
            .add_document_with_vectors(
                doc("a"),
                vec![DocumentVector::float32(
                    "f32",
                    vec![1.0, f32::NAN, 3.0, 4.0],
                )],
            )
            .unwrap();
        assert!(matches!(
            writer.commit(),
            Err(Error::Vectors(vectors::Error::NonFiniteValue(1, 1, _)))
        ));
    }

    #[test]
    fn set_vector_field_validates_its_field() {
        let tmp = tempdir("vectors-config");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("plain", 0),
            vector_field(
                "v",
                1,
                4,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        assert!(matches!(
            writer.set_vector_field(Some("missing")),
            Err(Error::UnknownVectorField(_))
        ));
        assert!(matches!(
            writer.set_vector_field(Some("plain")),
            Err(Error::UnsupportedVectorField(_, 0))
        ));
        writer.set_vector_field(Some("v")).unwrap();
        assert!(matches!(
            writer.add_vector_field("v"),
            Err(Error::DuplicateVectorField(_))
        ));
        // `None` clears the whole list, the same "reassign, don't accumulate"
        // shape `set_postings_field` has.
        writer.set_vector_field(None).unwrap();
        writer.add_vector_field("v").unwrap();

        assert!(matches!(
            writer.set_hnsw_parameters(0, 100),
            Err(Error::Vectors(vectors::Error::InvalidGraphParameter(_)))
        ));
        assert!(matches!(
            writer.set_hnsw_parameters(513, 100),
            Err(Error::Vectors(vectors::Error::InvalidGraphParameter(_)))
        ));
        assert!(matches!(
            writer.set_hnsw_parameters(16, 0),
            Err(Error::Vectors(vectors::Error::InvalidGraphParameter(_)))
        ));
        assert!(matches!(
            writer.set_hnsw_parameters(16, 3201),
            Err(Error::Vectors(vectors::Error::InvalidGraphParameter(_)))
        ));
        writer.set_hnsw_parameters(8, 40).unwrap();
    }

    /// `pending_vectors` is indexed by doc id, so it must stay aligned 1:1
    /// with `pending_docs` no matter which entry point buffered a document and
    /// no matter what discarded the buffer. A drift here does not fail loudly:
    /// it shifts every vector to the wrong document.
    #[test]
    fn the_vector_buffer_stays_aligned_with_the_document_buffer() {
        let tmp = tempdir("vectors-alignment");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            vector_field(
                "v",
                1,
                2,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_vector_field(Some("v")).unwrap();

        // A plain add, a block add, a vector add, a rollback, then the same
        // again -- every entry point that touches the buffer.
        writer.add_document(doc("plain")).unwrap();
        writer
            .add_documents(vec![doc("block-a"), doc("block-b")])
            .unwrap();
        writer
            .add_document_with_vectors(
                doc("with-vector"),
                vec![DocumentVector::float32("v", vec![1.0, 2.0])],
            )
            .unwrap();
        writer.rollback();

        writer.add_document(doc("d0")).unwrap();
        writer
            .add_document_with_vectors(
                doc("d1"),
                vec![DocumentVector::float32("v", vec![3.0, 4.0])],
            )
            .unwrap();
        writer.add_documents(vec![doc("d2"), doc("d3")]).unwrap();
        writer
            .add_document_with_vectors(
                doc("d4"),
                vec![DocumentVector::float32("v", vec![5.0, 6.0])],
            )
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        let sci = &infos.segments[0];
        let (vec_bytes, vemf, _, _) = open_vector_files(&dir, sci);
        let suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
        let flat =
            vectors::FlatVectorsReader::open(&vemf, &vec_bytes, &sci.segment_id, &suffix).unwrap();
        let values = flat.float_vector_values(1).unwrap();
        assert_eq!(values.size(), 2);
        assert_eq!(values.ord_to_doc(0).unwrap(), 1, "d1 is doc 1");
        assert_eq!(values.ord_to_doc(1).unwrap(), 4, "d4 is doc 4");
        assert_eq!(values.vector(0).unwrap(), vec![3.0, 4.0]);
        assert_eq!(values.vector(1).unwrap(), vec![5.0, 6.0]);
        assert_eq!(read_all_docs(&dir, &infos).len(), 5);
    }

    /// Two segments, each with its own vector files: the second flush must not
    /// inherit the first's ordinals or overwrite its files.
    #[test]
    fn each_flushed_segment_gets_its_own_vector_files() {
        let tmp = tempdir("vectors-two-segments");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            vector_field(
                "v",
                1,
                3,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_vector_field(Some("v")).unwrap();
        for i in 0..3 {
            writer
                .add_document_with_vectors(
                    doc(&format!("a{i}")),
                    vec![DocumentVector::float32("v", vec![i as f32, 0.0, 1.0])],
                )
                .unwrap();
        }
        writer.flush().unwrap();
        for i in 0..2 {
            writer
                .add_document_with_vectors(
                    doc(&format!("b{i}")),
                    vec![DocumentVector::float32("v", vec![9.0, i as f32, 2.0])],
                )
                .unwrap();
        }
        let infos = writer.commit().unwrap().clone();
        assert_eq!(infos.segments.len(), 2);
        let suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
        let sizes: Vec<i32> = infos
            .segments
            .iter()
            .map(|sci| {
                let (vec_bytes, vemf, _, _) = open_vector_files(&dir, sci);
                vectors::FlatVectorsReader::open(&vemf, &vec_bytes, &sci.segment_id, &suffix)
                    .unwrap()
                    .float_vector_values(1)
                    .unwrap()
                    .size()
            })
            .collect();
        assert_eq!(sizes, vec![3, 2]);
    }

    /// This port's own `CheckIndex` port (`crate::check_index`) has never had
    /// a producer for its `testVectors`/`testHnswGraphs` blocks -- c9 ported
    /// them against Java-written fixtures only, because nothing here could
    /// write a vector segment. Run them over one now.
    #[test]
    fn our_own_check_index_passes_over_a_writer_produced_vector_segment() {
        let tmp = tempdir("vectors-check-index");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            vector_field(
                "dense",
                1,
                8,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
            vector_field(
                "sparse",
                2,
                4,
                VectorEncoding::Byte,
                VectorSimilarityFunction::DotProduct,
            ),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_vector_field(Some("dense")).unwrap();
        writer.add_vector_field("sparse").unwrap();
        for i in 0..900 {
            let mut vectors = vec![DocumentVector::float32("dense", test_vector(8, i))];
            if i % 2 == 0 {
                vectors.push(DocumentVector::byte(
                    "sparse",
                    vec![(i % 251) as u8, 3, 9, (i / 5 % 251) as u8],
                ));
            }
            writer
                .add_document_with_vectors(doc(&format!("doc{i}")), vectors)
                .unwrap();
        }
        writer.commit().unwrap();

        let results = crate::check_index::check_directory(&dir).unwrap();
        for result in &results {
            assert!(
                result.all_passed(),
                "check_index failed on {}: {:?}",
                result.segment_name,
                result.failures()
            );
        }
        // ...and it must actually have *looked* at the vectors, not skipped
        // them: two fields, 900 + 450 values.
        let stats = &results
            .iter()
            .find(|r| r.max_doc.is_some())
            .expect("one segment")
            .stats;
        assert_eq!(stats.vector_fields, 2);
        assert_eq!(stats.vector_values, 900 + 450);
    }

    /// `set_hnsw_parameters` must actually reach the graph: `M` is recorded in
    /// the `.vem` and bounds every node's neighbour count.
    #[test]
    fn set_hnsw_parameters_reaches_the_written_graph() {
        let tmp = tempdir("vectors-hnsw-params");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            vector_field(
                "v",
                1,
                6,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_vector_field(Some("v")).unwrap();
        writer.set_hnsw_parameters(6, 32).unwrap();
        for i in 0..800 {
            writer
                .add_document_with_vectors(
                    doc(&format!("doc{i}")),
                    vec![DocumentVector::float32("v", test_vector(6, i))],
                )
                .unwrap();
        }
        let infos = writer.commit().unwrap().clone();
        let sci = &infos.segments[0];
        let (_, _, vem, vex) = open_vector_files(&dir, sci);
        let suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
        let graphs =
            hnsw_vectors::HnswVectorsReader::open(&vem, &vex, &sci.segment_id, &suffix).unwrap();
        let graph = graphs.graph(1).unwrap().unwrap();
        assert_eq!(graph.max_conn(), 6);
        let mut neighbors = Vec::new();
        for node in graph.sorted_nodes_on_level(0).unwrap() {
            graph.neighbors_into(0, node, &mut neighbors).unwrap();
            assert!(
                neighbors.len() <= 12,
                "level 0 allows at most 2*M = 12 arcs, got {}",
                neighbors.len()
            );
        }
    }

    // ---------------------------------------------------------------
    // Index sorting (`IndexWriterConfig.setIndexSort`)
    // ---------------------------------------------------------------

    /// The two `Long.MIN_VALUE`/`Long.MAX_VALUE` sentinels this port's own
    /// sorted flush and sorted merge write, as a test-local shorthand.
    /// `IndexSortField` models Java's whole missing-value space now (any
    /// numeric sentinel, or none), so this names the two the tests below
    /// care about rather than pretending they are all there is.
    #[derive(Debug, Clone, Copy)]
    enum SortMissingValue {
        /// `SortField.setMissingValue(Long.MIN_VALUE)`.
        First,
        /// `SortField.setMissingValue(Long.MAX_VALUE)`.
        Last,
    }

    fn sort_field(name: &str, reverse: bool, missing: SortMissingValue) -> IndexSortField {
        IndexSortField::long(
            name,
            reverse,
            Some(match missing {
                SortMissingValue::First => i64::MIN,
                SortMissingValue::Last => i64::MAX,
            }),
        )
    }

    /// A writer over `id` (stored), `rank`/`tie` (NUMERIC doc values, also
    /// stored so the doc carries the value the flush reads its sort key
    /// from), `body` (postings + norms + term vectors) and `v` (vectors).
    /// One helper, because every assertion below wants a *different* format
    /// to be the one that gets the sort wrong.
    fn sortable_fields_with(postings: bool, term_vectors: bool) -> Vec<FieldInfo> {
        vec![
            stored_only_field("id", 0),
            numeric_field("rank", 1),
            numeric_field("tie", 2),
            FieldInfo {
                index_options: if postings {
                    IndexOptions::DocsAndFreqs
                } else {
                    IndexOptions::None
                },
                store_term_vectors: term_vectors,
                ..stored_only_field("body", 3)
            },
            vector_field(
                "v",
                4,
                4,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
        ]
    }

    /// The doc-values-only shape: same field numbers, but `body` is plain
    /// stored, so a flush that opts into neither postings nor term vectors
    /// still writes a `.fnm` that claims nothing the segment lacks.
    fn sortable_fields() -> Vec<FieldInfo> {
        sortable_fields_with(false, false)
    }

    fn sortable_doc(id: &str, rank: i64, tie: i64, body: &str) -> Document {
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(id.to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(rank),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::Long(tie),
                },
                StoredField {
                    field_number: 3,
                    value: FieldValue::String(body.to_string()),
                },
            ],
        }
    }

    fn read_index_sort(dir: &FsDirectory, sci: &SegmentCommitInfo) -> Option<Vec<IndexSortField>> {
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        segment_info::parse(&si_bytes, &sci.segment_id)
            .unwrap()
            .index_sort
    }

    /// Reads back one NUMERIC doc-values column of a *freshly flushed*
    /// segment (no doc-values generation), in doc order.
    fn read_base_numeric_column(
        dir: &FsDirectory,
        sci: &SegmentCommitInfo,
        field_number: i32,
        max_doc: i32,
    ) -> Vec<Option<i64>> {
        let seg = per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME);
        let dvm = dir.open(&format!("{seg}.dvm")).unwrap();
        let dvd = dir.open(&format!("{seg}.dvd")).unwrap();
        let field_infos = lucene_codecs::field_infos::FieldInfos {
            fields: sortable_fields(),
        };
        let (_, meta) = doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .unwrap();
        let entry = meta.numeric_entry(field_number).unwrap();
        (0..max_doc)
            .map(|d| doc_values::numeric_value(&dvd, entry, d).unwrap())
            .collect()
    }

    #[test]
    fn set_index_sort_rejects_an_empty_sort() {
        let tmp = tempdir("sort-empty");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        assert!(matches!(
            writer.set_index_sort(Some(&[])).unwrap_err(),
            Error::EmptyIndexSort
        ));
    }

    #[test]
    fn set_index_sort_rejects_an_unknown_field() {
        let tmp = tempdir("sort-unknown");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        let err = writer
            .set_index_sort(Some(&[sort_field("nope", false, SortMissingValue::Last)]))
            .unwrap_err();
        assert!(matches!(err, Error::UnknownIndexSortField(f) if f == "nope"));
    }

    /// `IndexingChain.validateIndexSortDVType`: a sort field that is not
    /// NUMERIC doc values cannot be read by the `LONG` `SortField` this
    /// port's `.si` encodes, so it is refused rather than written as a sort
    /// no reader can evaluate.
    #[test]
    fn set_index_sort_rejects_a_field_that_is_not_numeric_doc_values() {
        let tmp = tempdir("sort-wrong-dv-type");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0), sorted_field("name", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("name")).unwrap();
        let err = writer
            .set_index_sort(Some(&[sort_field("name", false, SortMissingValue::Last)]))
            .unwrap_err();
        assert!(matches!(
            err,
            Error::UnsupportedIndexSortField(f, DocValuesType::Sorted) if f == "name"
        ));
        // The `id` field has no doc values at all -- the other half of the
        // same check.
        let err = writer
            .set_index_sort(Some(&[sort_field("id", false, SortMissingValue::Last)]))
            .unwrap_err();
        assert!(matches!(
            err,
            Error::UnsupportedIndexSortField(f, DocValuesType::None) if f == "id"
        ));
    }

    /// A sort over a column nothing writes is a sort no reader can check:
    /// real Lucene's `DocValues.getNumeric` substitutes an all-missing
    /// instance rather than failing, so `CheckIndex.testSort` would compare
    /// `maxDoc` equal keys and pass. Refused at configuration time instead.
    #[test]
    fn set_index_sort_rejects_a_sort_field_no_doc_values_are_written_for() {
        let tmp = tempdir("sort-no-dv-optin");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        let err = writer
            .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
            .unwrap_err();
        assert!(matches!(err, Error::IndexSortFieldWithoutDocValues(f) if f == "rank"));
        // Opting in makes the same call succeed.
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer
            .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
            .unwrap();
    }

    /// The sort is read at flush, so changing it with documents already
    /// buffered would order one part of the batch by one key and the rest by
    /// another. Java snapshots `IndexWriterConfig` at construction and cannot
    /// reach the state at all.
    #[test]
    fn set_index_sort_is_refused_once_documents_are_buffered() {
        let tmp = tempdir("sort-mid-buffer");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_document(sortable_doc("a", 1, 0, "x")).unwrap();
        assert!(matches!(
            writer
                .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
                .unwrap_err(),
            Error::IndexSortChangedMidBuffer(1)
        ));
        // ...including clearing it.
        assert!(matches!(
            writer.set_index_sort(None).unwrap_err(),
            Error::IndexSortChangedMidBuffer(1)
        ));
        writer.commit().unwrap();
        writer.set_index_sort(None).unwrap();
    }

    /// `IndexWriter.validateIndexSort` + `isCongruentSort`: an existing
    /// segment's sort must have the incoming one as a **prefix**. An
    /// unsorted segment fails just as hard as a differently-sorted one.
    #[test]
    fn set_index_sort_must_be_congruent_with_the_segments_already_in_the_index() {
        let tmp = tempdir("sort-congruence");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_doc_values_field("tie").unwrap();
        writer.add_document(sortable_doc("a", 1, 0, "x")).unwrap();
        writer.commit().unwrap();

        // The committed segment is unsorted, so *any* sort is incongruent.
        let err = writer
            .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
            .unwrap_err();
        assert!(
            matches!(&err, Error::IncongruentIndexSort { segment, existing, .. }
            if segment == "_0" && existing == "<none>")
        );

        // A fresh index sorted by (rank, tie) accepts the prefix (rank), the
        // whole sort, and rejects a longer or different one.
        let tmp2 = tempdir("sort-congruence-2");
        let dir2 = FsDirectory::open(&tmp2);
        let mut w2 = IndexWriter::open(&dir2, sortable_fields(), "Lucene104", version()).unwrap();
        w2.set_doc_values_field(Some("rank")).unwrap();
        w2.add_doc_values_field("tie").unwrap();
        w2.set_index_sort(Some(&[
            sort_field("rank", false, SortMissingValue::Last),
            sort_field("tie", false, SortMissingValue::Last),
        ]))
        .unwrap();
        w2.add_document(sortable_doc("a", 1, 0, "x")).unwrap();
        w2.commit().unwrap();

        w2.set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
            .expect("a prefix of the existing sort is congruent");
        w2.set_index_sort(Some(&[
            sort_field("rank", false, SortMissingValue::Last),
            sort_field("tie", false, SortMissingValue::Last),
        ]))
        .expect("the identical sort is congruent");
        // Same fields, opposite direction: not a prefix.
        let err = w2
            .set_index_sort(Some(&[sort_field("rank", true, SortMissingValue::Last)]))
            .unwrap_err();
        assert!(matches!(err, Error::IncongruentIndexSort { .. }));
        // Longer than the existing sort: also not a prefix.
        let err = w2
            .set_index_sort(Some(&[
                sort_field("rank", false, SortMissingValue::Last),
                sort_field("tie", false, SortMissingValue::Last),
                sort_field("rank", false, SortMissingValue::Last),
            ]))
            .unwrap_err();
        assert!(matches!(err, Error::IncongruentIndexSort { .. }));
    }

    /// The one hole a mutable configuration leaves that Java's immutable one
    /// does not: `set_doc_values_field` *shrinks* the list, so after
    /// `set_index_sort` it could strand the sort over a column nothing
    /// writes.
    #[test]
    fn narrowing_the_doc_values_fields_cannot_strand_a_configured_index_sort() {
        let tmp = tempdir("sort-strand-dv");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_doc_values_field("tie").unwrap();
        writer
            .set_index_sort(Some(&[
                sort_field("rank", true, SortMissingValue::Last),
                sort_field("tie", false, SortMissingValue::First),
            ]))
            .unwrap();

        // Both of these would leave a tier with no column.
        assert!(matches!(
            writer.set_doc_values_field(Some("rank")).unwrap_err(),
            Error::IndexSortFieldWithoutDocValues(f) if f == "tie"
        ));
        assert!(matches!(
            writer.set_doc_values_field(None).unwrap_err(),
            Error::IndexSortFieldWithoutDocValues(f) if f == "rank"
        ));
        // The rejected calls left the list intact, so a flush still works.
        writer.add_document(sortable_doc("a", 1, 2, "x")).unwrap();
        writer.commit().unwrap();
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }

        // Clearing the sort first makes the same call legal.
        // Fresh directories, since the committed segment above pins the sort
        // every later `set_index_sort` on `dir` must be congruent with.
        let tmp2 = tempdir("sort-strand-dv-cleared");
        let dir2 = FsDirectory::open(&tmp2);
        let mut writer =
            IndexWriter::open(&dir2, sortable_fields(), "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_doc_values_field("tie").unwrap();
        writer
            .set_index_sort(Some(&[sort_field("rank", true, SortMissingValue::Last)]))
            .unwrap();
        writer.set_index_sort(None).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();

        // ...and narrowing to exactly the single tier's own column is fine.
        let tmp3 = tempdir("sort-strand-dv-single");
        let dir3 = FsDirectory::open(&tmp3);
        let mut writer =
            IndexWriter::open(&dir3, sortable_fields(), "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_doc_values_field("tie").unwrap();
        writer
            .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
            .unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
    }

    /// The heart of the batch: **one** permutation reaches every format.
    /// Stored fields, the doc-values column, postings doc ids, norms, term
    /// vectors and vectors must all describe the *same* document at the same
    /// doc id after a sorted flush -- the failure mode this guards against is
    /// silent (each file is well-formed and checksums cleanly; only the
    /// association between them is wrong).
    #[test]
    fn a_sorted_flush_reorders_every_format_together() {
        let tmp = tempdir("sort-all-formats");
        let dir = FsDirectory::open(&tmp);
        let mut writer = IndexWriter::open(
            &dir,
            sortable_fields_with(true, true),
            "Lucene104",
            version(),
        )
        .unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();
        writer.set_vector_field(Some("v")).unwrap();
        writer
            .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
            .unwrap();

        // Inserted in an order that has nothing to do with the sort, with a
        // per-document body whose token count (and therefore norm) and whose
        // unique term both identify the document.
        let inserted: Vec<(&str, i64)> = vec![("d", 40), ("a", 10), ("c", 30), ("b", 20)];
        for (id, rank) in &inserted {
            let body: String = std::iter::repeat_n(format!("t{id}"), *rank as usize / 10)
                .collect::<Vec<_>>()
                .join(" ");
            writer
                .add_document_with_vectors(
                    sortable_doc(id, *rank, 0, &body),
                    vec![DocumentVector::float32(
                        "v",
                        vec![*rank as f32, 1.0, 2.0, 3.0],
                    )],
                )
                .unwrap();
        }
        let infos = writer.commit().unwrap().clone();
        let sci = &infos.segments[0];

        // Stored fields: ascending by rank.
        assert_eq!(read_all_docs(&dir, &infos), vec!["a", "b", "c", "d"]);
        // The `.si` says so too.
        let sort = read_index_sort(&dir, sci).unwrap();
        assert_eq!(
            sort,
            vec![sort_field("rank", false, SortMissingValue::Last)]
        );
        // Doc values: the sort key column, in the new doc order.
        assert_eq!(
            read_base_numeric_column(&dir, sci, 1, 4),
            vec![Some(10), Some(20), Some(30), Some(40)]
        );

        // Postings: each document's unique term must resolve to its *new*
        // doc id. A permutation applied to stored fields but not to the
        // invert pass lands every term on the wrong document.
        let seg = per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
        let tim = dir.open(&format!("{seg}.tim")).unwrap();
        let tip = dir.open(&format!("{seg}.tip")).unwrap();
        let tmd = dir.open(&format!("{seg}.tmd")).unwrap();
        let doc_bytes = dir.open(&format!("{seg}.doc")).unwrap();
        let field_infos = fi::FieldInfos {
            fields: sortable_fields_with(true, true),
        };
        let block_fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            4,
        )
        .unwrap();
        let doc_in = DocInput::open(
            &doc_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let body_terms = block_fields.field("body").unwrap();
        for (new_doc, id) in ["a", "b", "c", "d"].iter().enumerate() {
            let postings = body_terms
                .postings(format!("t{id}").as_bytes(), Some(&doc_in))
                .unwrap()
                .unwrap();
            assert_eq!(
                postings.docs,
                vec![new_doc as i32],
                "term t{id} must land on the sorted doc id"
            );
        }

        // Norms: doc `i`'s norm encodes its field length, which is
        // `rank / 10` tokens -- strictly increasing in the sorted order.
        let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
        let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
        let (_, norms_meta) = norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
        let entry = norms_meta.entry(3).unwrap();
        let lengths: Vec<i64> = (0..4)
            .map(|d| {
                lucene_util::small_float::byte4_to_int(
                    norms::norm_value(&nvd, entry, d).unwrap().unwrap() as u8,
                ) as i64
            })
            .collect();
        assert_eq!(lengths, vec![1, 2, 3, 4]);

        // Term vectors: doc `i` must carry only its own term.
        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let tv = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
        for (new_doc, id) in ["a", "b", "c", "d"].iter().enumerate() {
            let d = tv.document(new_doc as i32).unwrap().unwrap();
            let terms: Vec<String> = d.fields[0]
                .terms
                .iter()
                .map(|t| String::from_utf8(t.term.clone()).unwrap())
                .collect();
            assert_eq!(terms, vec![format!("t{id}")]);
        }

        // Vectors: the ordinal -> doc mapping and the value itself. Every
        // vector's first component is its document's rank, so a vector that
        // drifted onto another document is immediately visible. This is the
        // "ordinals must be remapped" hazard, discharged by the buffer
        // permutation rather than by a per-format remap.
        let (vec_bytes, vemf, _, _) = open_vector_files(&dir, sci);
        let flat = vectors::FlatVectorsReader::open(
            &vemf,
            &vec_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME),
        )
        .unwrap();
        let values = flat.float_vector_values(4).unwrap();
        assert_eq!(values.size(), 4);
        for ord in 0..4 {
            let doc_id = values.ord_to_doc(ord).unwrap();
            assert_eq!(doc_id, ord, "dense field: ordinal == doc id");
            let v = values.vector(ord).unwrap();
            assert_eq!(
                v[0],
                ((ord + 1) * 10) as f32,
                "vector on doc {doc_id} belongs to another document"
            );
        }

        // ...and this port's own CheckIndex, which re-derives the sort's
        // comparators from the `.si` and the doc values and walks adjacent
        // doc ids.
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(
                result.all_passed(),
                "check_index failed: {:?}",
                result.failures()
            );
        }
    }

    /// Reads one segment back and asserts that **every** format describes the
    /// same document at the same doc id, given the ids the segment is
    /// expected to hold in physical order.
    ///
    /// The corpus convention (shared with
    /// `a_sorted_flush_reorders_every_format_together`): document `id` has
    /// `rank` doc-values, a body of `rank / 10` copies of the unique term
    /// `t{id}`, so its norm encodes `rank / 10`, and a vector whose first
    /// component is `rank`. Every one of those is a different file, so a
    /// doc map applied to one and not another shows up as a specific
    /// mismatch rather than as "something is off".
    fn assert_every_format_agrees(
        dir: &FsDirectory,
        sci: &SegmentCommitInfo,
        expected: &[(&str, i64)],
    ) {
        let max_doc = expected.len() as i32;
        let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
        let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
        let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
        let stored = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        assert_eq!(stored.max_doc(), max_doc);
        for (doc_id, (id, _)) in expected.iter().enumerate() {
            assert_eq!(
                doc_value(&stored.document(doc_id as i32).unwrap()),
                *id,
                "stored id at doc {doc_id}"
            );
        }

        let ranks: Vec<Option<i64>> = expected.iter().map(|(_, rank)| Some(*rank)).collect();
        let seg = per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME);
        let dvm = dir.open(&format!("{seg}.dvm")).unwrap();
        let dvd = dir.open(&format!("{seg}.dvd")).unwrap();
        let field_infos = fi::FieldInfos {
            fields: sortable_fields_with(true, true),
        };
        let (_, dv_meta) = doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .unwrap();
        let dv_entry = dv_meta.numeric_entry(1).unwrap();
        let read_ranks: Vec<Option<i64>> = (0..max_doc)
            .map(|d| doc_values::numeric_value(&dvd, dv_entry, d).unwrap())
            .collect();
        assert_eq!(read_ranks, ranks, "doc-values rank column");

        let seg = per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
        let tim = dir.open(&format!("{seg}.tim")).unwrap();
        let tip = dir.open(&format!("{seg}.tip")).unwrap();
        let tmd = dir.open(&format!("{seg}.tmd")).unwrap();
        let doc_bytes = dir.open(&format!("{seg}.doc")).unwrap();
        let block_fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            max_doc,
        )
        .unwrap();
        let doc_in = DocInput::open(
            &doc_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let body_terms = block_fields.field("body").unwrap();
        for (doc_id, (id, _)) in expected.iter().enumerate() {
            let postings = body_terms
                .postings(format!("t{id}").as_bytes(), Some(&doc_in))
                .unwrap()
                .unwrap_or_else(|| panic!("term t{id} is missing from the merged dictionary"));
            assert_eq!(
                postings.docs,
                vec![doc_id as i32],
                "term t{id} must land on doc {doc_id}"
            );
        }

        let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
        let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
        let (_, norms_meta) = norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
        let norms_entry = norms_meta.entry(3).unwrap();
        let lengths: Vec<i64> = (0..max_doc)
            .map(|d| {
                lucene_util::small_float::byte4_to_int(
                    norms::norm_value(&nvd, norms_entry, d).unwrap().unwrap() as u8,
                ) as i64
            })
            .collect();
        assert_eq!(
            lengths,
            expected.iter().map(|(_, r)| r / 10).collect::<Vec<_>>(),
            "norms"
        );

        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let tv = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
        for (doc_id, (id, _)) in expected.iter().enumerate() {
            let d = tv.document(doc_id as i32).unwrap().unwrap();
            let terms: Vec<String> = d.fields[0]
                .terms
                .iter()
                .map(|t| String::from_utf8(t.term.clone()).unwrap())
                .collect();
            assert_eq!(
                terms,
                vec![format!("t{id}")],
                "term vectors at doc {doc_id}"
            );
        }

        let (vec_bytes, vemf, vem, vex) = open_vector_files(dir, sci);
        let flat = vectors::FlatVectorsReader::open(
            &vemf,
            &vec_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME),
        )
        .unwrap();
        let values = flat.float_vector_values(4).unwrap();
        assert_eq!(values.size(), max_doc);
        for ord in 0..max_doc {
            let doc_id = values.ord_to_doc(ord).unwrap();
            assert_eq!(doc_id, ord, "dense vector field: ordinal == doc id");
            assert_eq!(
                values.vector(ord).unwrap()[0],
                expected[ord as usize].1 as f32,
                "vector on doc {doc_id} belongs to another document"
            );
        }
        // The graph must describe the merged ordinal space, not a source's.
        let graph_reader = hnsw_vectors::HnswVectorsReader::open(
            &vem,
            &vex,
            &sci.segment_id,
            &per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME),
        )
        .unwrap();
        assert_eq!(graph_reader.field(4).unwrap().size, max_doc);
    }

    /// Adds one document of the corpus `assert_every_format_agrees` reads
    /// back: unique term, length-encoding body, and a vector that names its
    /// own rank.
    fn add_sortable_doc(writer: &mut IndexWriter, id: &str, rank: i64, tie: i64) {
        let body: String = std::iter::repeat_n(format!("t{id}"), rank as usize / 10)
            .collect::<Vec<_>>()
            .join(" ");
        writer
            .add_document_with_vectors(
                sortable_doc(id, rank, tie, &body),
                vec![DocumentVector::float32(
                    "v",
                    vec![rank as f32, 1.0, 2.0, 3.0],
                )],
            )
            .unwrap();
    }

    /// **The merge-completeness gate, end to end.** The union of formats the
    /// flush wrote across the source segments must reappear in the merged
    /// segment's own `.si` -- not "the merge succeeded", which it does just
    /// as happily with a format dropped.
    ///
    /// This is the observable half of `merge::check_format_coverage`: the
    /// gate refuses a merge that never *opened* a format, and this asserts
    /// that every opened format was also *written*. Together they close
    /// c22's carry-over ("nothing mechanically checks that `execute_merge`
    /// opens every format the flush can write"), whose four instances --
    /// norms (finding 14, wrong BM25 scores since c4), non-NUMERIC doc
    /// values (22), positional postings (23) and `has_blocks` (24) -- each
    /// produced a well-formed, checksummed, `CheckIndex`-clean segment with
    /// the data gone.
    ///
    /// The writer is configured with every format the flush path has an
    /// opt-in for: stored fields (always), postings **with positions and
    /// offsets**, term vectors, doc values (two NUMERIC columns), norms on
    /// two fields, and KNN vectors. Points are the one format with no flush
    /// path at all (`docs/parity.md`), so no segment can carry them; the
    /// gate covers them anyway, and would fire the day one can.
    #[test]
    fn every_format_the_flush_writes_reappears_in_the_merged_si() {
        let tmp = tempdir("merge-format-completeness");
        let dir = FsDirectory::open(&tmp);
        let mut fields = sortable_fields_with(true, true);
        // Positions and offsets, so the postings half is the widest shape
        // `set_postings_field` accepts rather than the `DocsAndFreqs` one.
        fields[3].index_options = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
        // A second norms field, which was inexpressible before c26.
        fields[0].index_options = IndexOptions::DocsAndFreqs;
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.add_postings_field("id").unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_doc_values_field("tie").unwrap();
        writer.set_vector_field(Some("v")).unwrap();
        // No merging for the first three commits, so every source segment is
        // observable before anything folds it away.
        writer.set_merge_policy(None);

        for batch in [
            vec![("d", 40i64), ("a", 10)],
            vec![("e", 50), ("b", 20)],
            vec![("f", 60), ("c", 30)],
        ] {
            for (id, rank) in batch {
                add_sortable_doc(&mut writer, id, rank, 0);
            }
            writer.commit().unwrap();
        }

        let sources = writer.segment_infos().segments.clone();
        assert_eq!(sources.len(), 3, "three unmerged sources");
        let source_names: Vec<String> = sources.iter().map(|s| s.segment_name.clone()).collect();
        let source_formats: std::collections::BTreeSet<merge::SegmentFormat> = sources
            .iter()
            .flat_map(|sci| segment_formats(&dir, sci))
            .collect();

        // Now let the merge policy see them.
        writer.set_merge_policy(Some(tight_merge_policy()));
        writer.commit().unwrap();

        // Everything but points, which no flush path can write.
        let expected: std::collections::BTreeSet<merge::SegmentFormat> = merge::SegmentFormat::ALL
            .into_iter()
            .filter(|f| *f != merge::SegmentFormat::Points)
            .collect();
        assert_eq!(
            source_formats, expected,
            "the sources must exercise every format the flush can write, or this test proves \
             nothing about the ones it skipped"
        );

        let infos = writer.segment_infos().clone();
        assert_eq!(infos.segments.len(), 1, "the sources must have merged");
        let merged = &infos.segments[0];
        assert!(
            !source_names.contains(&merged.segment_name),
            "the merged segment must be a new one, not one of the sources"
        );
        assert_eq!(
            segment_formats(&dir, merged),
            source_formats,
            "the merged `.si` must carry every format its sources' `.si`s did"
        );
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    /// The set of `merge::SegmentFormat`s a segment's own `.si` file list
    /// says it has -- the same classification the gate uses, so this test
    /// and `check_format_coverage` cannot disagree about what a format is.
    fn segment_formats(
        dir: &FsDirectory,
        sci: &SegmentCommitInfo,
    ) -> std::collections::BTreeSet<merge::SegmentFormat> {
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        si.files
            .iter()
            .filter_map(|f| f.rsplit_once('.'))
            .filter_map(|(_, ext)| merge::SegmentFormat::for_extension(ext))
            .collect()
    }

    /// **Multi-field norms, flush and merge.** c22's finding 13 / carry-over:
    /// `merge_norms` took one field per merge (`Error::TooManyNormsFields`)
    /// because `norms::write_single_dense_field` writes a whole `.nvm`/`.nvd`
    /// pair, so two of them in one merge overwrote each other -- exactly the
    /// limitation `doc_values::write_dense_fields` had already removed for
    /// doc values. `norms::write_fields` is the norms analogue, and this is
    /// both sides taking it up: the flush's per-field column loop, and the
    /// widened `merge_norms` on the merge. Since c35 neither side needs an
    /// opt-in at all -- both `id` and `body` are indexed, so both get norms.
    ///
    /// Asserted on the *values*, per document per field, rather than on the
    /// merge having happened: a merge that carried only the first field
    /// would still produce a valid segment.
    #[test]
    fn an_automatic_merge_carries_norms_for_every_field() {
        let tmp = tempdir("multi-field-norms-merge");
        let dir = FsDirectory::open(&tmp);
        let mut fields = sortable_fields_with(true, false);
        fields[0].index_options = IndexOptions::DocsAndFreqs;
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.add_postings_field("id").unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        // `id` is one token, `body` is `rank / 10` of them -- so the two
        // fields' norms differ per document and a column written twice, or
        // one field's column read for the other, is visible.
        let corpus = [("aa", 10i64), ("bb", 20), ("cc", 30)];
        for (id, rank) in corpus {
            let body: String = std::iter::repeat_n(format!("t{id}"), rank as usize / 10)
                .collect::<Vec<_>>()
                .join(" ");
            writer
                .add_document(sortable_doc(id, rank, 0, &body))
                .unwrap();
            writer.commit().unwrap();
        }

        let infos = writer.segment_infos().clone();
        assert_eq!(infos.segments.len(), 1, "the sources must have merged");
        let sci = &infos.segments[0];
        let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
        let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
        let (_v, meta) = norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
        assert_eq!(
            meta.entries.len(),
            2,
            "both norms fields must share the one merged `.nvm`"
        );
        let id_entry = *meta
            .entries
            .iter()
            .find(|e| e.field_number == 0)
            .expect("`id`'s norms survived the merge");
        let body_entry = *meta
            .entries
            .iter()
            .find(|e| e.field_number == 3)
            .expect("`body`'s norms survived the merge");

        // Which document landed at which merged doc id is the merge's
        // business; resolve it through the stored `id` the way
        // `assert_every_format_agrees` does, so this asserts the *pairing*
        // rather than an assumed order.
        let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
        let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
        let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
        let stored = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        assert_eq!(stored.max_doc(), corpus.len() as i32);
        for doc_id in 0..stored.max_doc() {
            let id = doc_value(&stored.document(doc_id).unwrap());
            let rank = corpus
                .iter()
                .find(|(i, _)| *i == id)
                .map(|(_, r)| *r)
                .unwrap_or_else(|| panic!("unexpected stored id {id}"));
            assert_eq!(
                norms::norm_value(&nvd, &id_entry, doc_id).unwrap(),
                Some(small_float::int_to_byte4(1) as i8 as i64),
                "`id` norm at doc {doc_id} (id={id}): one token"
            );
            assert_eq!(
                norms::norm_value(&nvd, &body_entry, doc_id).unwrap(),
                Some(small_float::int_to_byte4(rank as u32 / 10) as i8 as i64),
                "`body` norm at doc {doc_id} (id={id})"
            );
        }
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    /// The norms field set is derived from the schema, not accumulated: every
    /// indexed field that has not opted out gets exactly one column, and
    /// `omit_norms_field` takes one away. Before c35 this was an
    /// accumulating opt-in whose duplicate-entry and clear-the-list rules had
    /// to be tested; the rule now is simply "the `.fnm` and the `.nvm` say the
    /// same thing".
    #[test]
    fn the_norms_columns_are_exactly_the_indexed_non_omitting_fields() {
        let tmp = tempdir("norms-column-set");
        let dir = FsDirectory::open(&tmp);
        let mut fields = sortable_fields_with(true, false);
        // `id` and `body` are indexed; `tie` is indexed but opts out; `rank`
        // is a plain numeric field (`index_options == None`).
        fields[0].index_options = IndexOptions::DocsAndFreqs;
        fields[2].omit_norms = true;
        fields[2].index_options = IndexOptions::DocsAndFreqs;
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

        assert!(matches!(
            writer.omit_norms_field("nonexistent"),
            Err(Error::UnknownNormsField(_))
        ));
        // `rank` is a plain numeric field: `index_options == None`, so there
        // are no norms to omit.
        assert!(matches!(
            writer.omit_norms_field("rank"),
            Err(Error::UnsupportedNormsField(_))
        ));

        writer.set_postings_field(Some("body")).unwrap();
        writer.add_postings_field("id").unwrap();
        writer
            .add_document(sortable_doc("a", 10, 0, "ta ta"))
            .unwrap();
        writer.commit().unwrap();

        let sci = writer.segment_infos().segments[0].clone();
        let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
        let (_v, meta) = norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
        assert_eq!(
            meta.entries.len(),
            2,
            "`id` and `body` each get exactly one column; `tie` opted out and \
             `rank` is not indexed"
        );

        // Opting the remaining two out writes no norms at all for the next
        // segment.
        writer.omit_norms_field("body").unwrap();
        writer.omit_norms_field("id").unwrap();
        writer
            .add_document(sortable_doc("b", 20, 0, "tb tb"))
            .unwrap();
        writer.commit().unwrap();
        let names: Vec<String> = writer
            .segment_infos()
            .segments
            .iter()
            .map(|s| s.segment_name.clone())
            .collect();
        assert!(
            !dir.list_all()
                .unwrap()
                .contains(&format!("{}.nvm", names[1])),
            "opting every field out must write no norms at all"
        );
    }

    /// **The c35 widening, write side.** A `SortedNumericSortField` with the
    /// `MAX` selector and **no missing value** was doubly inexpressible
    /// before: the old `IndexSortField` had no selector at all (so `parse`
    /// refused anything but `MIN`) and no way to say "no missing value" (Java
    /// compares such a document as `0`).
    ///
    /// Asserted through this port's own `CheckIndex`, which re-derives the
    /// comparator from the written `.si` and re-reads the keys out of the
    /// segment's SORTED_NUMERIC column -- so a flush that picked the wrong
    /// value out of the multi-valued column, or that ignored the selector,
    /// fails here rather than producing a plausible segment.
    #[test]
    fn a_sorted_numeric_max_selector_sort_orders_the_flush_and_survives_a_merge() {
        let tmp = tempdir("sorted-numeric-max-sort");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            FieldInfo {
                doc_values_type: DocValuesType::SortedNumeric,
                ..stored_only_field("multi", 1)
            },
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("multi")).unwrap();
        let sort = IndexSortField {
            field: "multi".to_string(),
            reverse: false,
            kind: crate::segment_info::IndexSortKind::SortedNumeric {
                key: crate::segment_info::NumericSortKey::Int(None),
                selector: crate::segment_info::SortedNumericSelector::Max,
            },
        };
        writer
            .set_index_sort(Some(std::slice::from_ref(&sort)))
            .unwrap();
        // No merging while the flushes are being observed.
        writer.set_merge_policy(None);

        // `(id, values)`. The MIN of each document is deliberately in the
        // *opposite* order from the MAX, so a flush that took the wrong end
        // of the column produces a different physical order.
        // `f`'s values are supplied *descending* on purpose: a document's
        // SORTED_NUMERIC values must be stored ascending
        // (`SortedNumericDocValuesWriter.finishCurrentDoc`'s `Arrays.sort`,
        // which real `CheckIndex.checkSortedNumericDocValues` enforces), and
        // MAX is then the *last* stored value. A writer that kept the
        // caller's order would both write a segment real Lucene rejects and
        // make the selector pick `3` instead of `6`.
        let corpus: [(&str, &[i64]); 6] = [
            ("a", &[1, 9]),
            ("b", &[8, 8]),
            ("c", &[0, 5]),
            ("d", &[7, 7]),
            ("e", &[2, 2]),
            ("f", &[6, 3]),
        ];
        for batch in [&corpus[..2], &corpus[2..4], &corpus[4..]] {
            for (id, values) in batch {
                let mut doc_fields = vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String(id.to_string()),
                }];
                for v in *values {
                    doc_fields.push(StoredField {
                        field_number: 1,
                        value: FieldValue::Long(*v),
                    });
                }
                writer
                    .add_document(Document { fields: doc_fields })
                    .unwrap();
            }
            writer.commit().unwrap();
        }

        // Ascending by MAX: b(8), a(9) in the first flush and c(5), d(7) in
        // the second; ascending by MIN it would have been a(1), b(8) and
        // c(0), d(7) -- so the first flush's order alone discriminates.
        let sis = writer.segment_infos().clone();
        assert_eq!(
            read_all_docs(&dir, &sis),
            vec!["b", "a", "c", "d", "e", "f"],
            "each flush is MAX-ordered on its own: b(8) before a(9), c(5) before \
             d(7), e(2) before f(6) -- by MIN it would have been a(1) before b(8)"
        );

        writer.set_merge_policy(Some(tight_merge_policy()));
        writer.commit().unwrap();
        let sis = writer.segment_infos().clone();
        assert_eq!(sis.segments.len(), 1, "the two flushes merged into one");
        assert_eq!(
            read_all_docs(&dir, &sis),
            vec!["e", "c", "f", "d", "b", "a"],
            "the sort-preserving merge is MAX-ordered across sources too: \
             e(2), c(5), f(6), d(7), b(8), a(9)"
        );

        // The merged `.si` still describes the widened sort, verbatim.
        let sci = &sis.segments[0];
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert_eq!(si.index_sort, Some(vec![sort]));

        // The column real Lucene requires: every document's values ascending,
        // so `f` came back as `[3, 6]` and not the `[6, 3]` it was given.
        let dvm = dir
            .open(&format!(
                "{}.dvm",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let dvd = dir
            .open(&format!(
                "{}.dvd",
                per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
            ))
            .unwrap();
        let infos = lucene_codecs::field_infos::parse(
            &dir.open(&format!("{}.fnm", sci.segment_name)).unwrap(),
            &sci.segment_id,
            "",
        )
        .unwrap();
        let (_v, dv_meta) = doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &infos,
        )
        .unwrap();
        let entry = dv_meta.sorted_numeric_entry(1).unwrap();
        for doc in 0..6 {
            let values = doc_values::sorted_numeric_values(&dvd, entry, doc).unwrap();
            assert!(
                values.windows(2).all(|w| w[0] <= w[1]),
                "doc {doc} values are not ascending: {values:?}"
            );
        }
        assert_eq!(
            doc_values::sorted_numeric_values(&dvd, entry, 2).unwrap(),
            vec![3, 6],
            "`f` is the third document in the merged order and its values were sorted"
        );

        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    /// `set_index_sort` accepts every kind whose key this writer can produce
    /// and names the ones it cannot, rather than mis-ordering them. Reading
    /// them is a separate question -- `segment_info::parse` handles all four
    /// providers, which is what lets this port open an index Lucene wrote.
    #[test]
    fn set_index_sort_refuses_the_ordinal_and_byte_sort_kinds() {
        use crate::segment_info::{IndexSortKind, SortedSetSelector, StringMissingValue};
        let tmp = tempdir("sort-kind-gate");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            numeric_field("rank", 1),
            sorted_field("name", 2),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("name")).unwrap();
        for kind in [
            IndexSortKind::String(StringMissingValue::First),
            IndexSortKind::SortedSet {
                selector: SortedSetSelector::Min,
                missing: StringMissingValue::None,
            },
            IndexSortKind::Binary(StringMissingValue::Last),
        ] {
            let sf = IndexSortField {
                field: "name".to_string(),
                reverse: false,
                kind: kind.clone(),
            };
            assert!(
                matches!(
                    writer.set_index_sort(Some(&[sf])),
                    Err(Error::UnsupportedIndexSortKind(f)) if f == "name"
                ),
                "{kind:?}"
            );
        }
        // And the doc-values type still has to match the kind: a
        // SortedNumeric sort over a NUMERIC column is refused by
        // `validateIndexSortDVType`'s rule, not silently read as NUMERIC.
        writer.set_doc_values_field(Some("rank")).unwrap();
        let sf = IndexSortField {
            field: "rank".to_string(),
            reverse: false,
            kind: IndexSortKind::SortedNumeric {
                key: crate::segment_info::NumericSortKey::Long(None),
                selector: crate::segment_info::SortedNumericSelector::Min,
            },
        };
        assert!(matches!(
            writer.set_index_sort(Some(&[sf])),
            Err(Error::UnsupportedIndexSortField(f, DocValuesType::Numeric)) if f == "rank"
        ));
    }

    /// A writer configured exactly the way the merge tests below need it.
    fn sorted_merge_writer<'a>(dir: &'a FsDirectory, sort: &[IndexSortField]) -> IndexWriter<'a> {
        let mut writer = IndexWriter::open(
            dir,
            sortable_fields_with(true, true),
            "Lucene104",
            version(),
        )
        .unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_doc_values_field("tie").unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();
        writer.set_vector_field(Some("v")).unwrap();
        writer.set_index_sort(Some(sort)).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));
        writer
    }

    /// **The headline of this batch.** Three index-sorted flushes are folded
    /// into one segment by the merge policy, and the merged segment is still
    /// in sort order -- globally, across sources, not per source -- with
    /// every format still describing the same document at the same doc id.
    ///
    /// Before this, the merge concatenated: the merged segment was valid,
    /// `CheckIndex`-clean and no longer in the order its inputs were in, and
    /// the only reason nothing broke was that `segment_stats` refused to
    /// offer a sorted segment to the merge policy at all.
    #[test]
    fn an_automatic_merge_preserves_the_index_sort_across_every_format() {
        let tmp = tempdir("sorted-merge-all-formats");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            sorted_merge_writer(&dir, &[sort_field("rank", false, SortMissingValue::Last)]);

        // Three flushes, each internally sorted, whose ranges interleave --
        // so a concatenating merge cannot accidentally come out ordered.
        for batch in [
            vec![("d", 40i64), ("a", 10)],
            vec![("e", 50), ("b", 20)],
            vec![("f", 60), ("c", 30)],
        ] {
            for (id, rank) in batch {
                add_sortable_doc(&mut writer, id, rank, 0);
            }
            writer.commit().unwrap();
        }

        let infos = writer.segment_infos().clone();
        assert_eq!(
            infos.segments.len(),
            1,
            "the three sorted segments must have merged into one"
        );
        let sci = &infos.segments[0];
        assert_eq!(
            read_index_sort(&dir, sci).unwrap(),
            vec![sort_field("rank", false, SortMissingValue::Last)],
            "the merged `.si` must still declare the sort"
        );
        assert_every_format_agrees(
            &dir,
            sci,
            &[
                ("a", 10),
                ("b", 20),
                ("c", 30),
                ("d", 40),
                ("e", 50),
                ("f", 60),
            ],
        );
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(
                result.all_passed(),
                "check_index failed: {:?}",
                result.failures()
            );
        }
    }

    /// **Negative control for the sort itself.** The same three segments
    /// merged with the sort *removed from their `.si` files* -- i.e. exactly
    /// what this code did before -- concatenate. The merged segment is
    /// entirely valid and passes this port's `CheckIndex`, because it makes
    /// no claim about its order; only comparing it against the sorted result
    /// shows the loss. Then stamping the sort onto the merged `.si` (which is
    /// what a merge that kept the declaration while concatenating would have
    /// produced) makes `CheckIndex.testSort` fail.
    #[test]
    fn a_concatenating_merge_of_the_same_segments_loses_the_order_silently() {
        let tmp = tempdir("sorted-merge-negative-control");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            sorted_merge_writer(&dir, &[sort_field("rank", false, SortMissingValue::Last)]);
        writer.set_merge_policy(None);
        for batch in [
            vec![("d", 40i64), ("a", 10)],
            vec![("e", 50), ("b", 20)],
            vec![("f", 60), ("c", 30)],
        ] {
            for (id, rank) in batch {
                add_sortable_doc(&mut writer, id, rank, 0);
            }
            writer.commit().unwrap();
        }
        assert_eq!(writer.segment_infos().segments.len(), 3);

        // Strip the sort declaration from every source, so `execute_merge`
        // takes the concatenating path over the very same bytes.
        for sci in &writer.segment_infos().segments.clone() {
            let si_name = format!("{}.si", sci.segment_name);
            let si_bytes = dir.open(&si_name).unwrap().to_vec();
            let mut si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
            si.index_sort = None;
            write_file(&dir, &si_name, &segment_info::write(&si, "")).unwrap();
        }
        let names: Vec<String> = writer
            .segment_infos()
            .segments
            .iter()
            .map(|s| s.segment_name.clone())
            .collect();
        writer.execute_merge(&names).unwrap();

        let infos = writer.segment_infos().clone();
        assert_eq!(infos.segments.len(), 1);
        let sci = &infos.segments[0];
        assert!(
            read_index_sort(&dir, sci).is_none(),
            "a concatenating merge must not claim a sort"
        );
        // Every format is still self-consistent -- the data is not corrupt,
        // it is merely in source order. Each source is internally sorted (its
        // flush sorted it), so the concatenation is sorted *within* a source
        // and jumps back down at every source boundary: the order looks
        // almost right, which is why nothing downstream notices.
        assert_every_format_agrees(
            &dir,
            sci,
            &[
                ("a", 10),
                ("d", 40),
                ("b", 20),
                ("e", 50),
                ("c", 30),
                ("f", 60),
            ],
        );
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(
                result.all_passed(),
                "a concatenated segment is perfectly valid: {:?}",
                result.failures()
            );
        }

        // ...and this is what makes it a *defect* rather than a choice: claim
        // the sort the inputs had, and the order check fails.
        let si_name = format!("{}.si", sci.segment_name);
        let si_bytes = dir.open(&si_name).unwrap().to_vec();
        let mut si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        si.index_sort = Some(vec![sort_field("rank", false, SortMissingValue::Last)]);
        write_file(&dir, &si_name, &segment_info::write(&si, "")).unwrap();
        let results = crate::check_index::check_directory(&dir).unwrap();
        let failures: Vec<String> = results
            .iter()
            .flat_map(|r| r.failures())
            .map(|c| format!("{}: {}", c.name, c.message))
            .collect();
        assert!(
            failures.iter().any(|f| f.contains("sort")),
            "expected the sort check to fail, got {failures:?}"
        );
    }

    /// A **multi-tier, reverse-first** sort across a merge, with the second
    /// tier doing most of the ordering: one ascending key with distinct
    /// values is the case most likely to come out right by accident.
    #[test]
    fn a_merge_honours_every_tier_of_a_multi_field_reverse_sort() {
        let tmp = tempdir("sorted-merge-multi-tier");
        let dir = FsDirectory::open(&tmp);
        let mut writer = sorted_merge_writer(
            &dir,
            &[
                sort_field("rank", true, SortMissingValue::Last),
                sort_field("tie", false, SortMissingValue::Last),
            ],
        );
        // Two rank groups of three, split across three flushes so that every
        // group spans every source and the tie tier decides within them.
        for batch in [
            vec![("a", 20i64, 1i64), ("d", 10, 1)],
            vec![("b", 20, 2), ("e", 10, 2)],
            vec![("c", 20, 3), ("f", 10, 3)],
        ] {
            for (id, rank, tie) in batch {
                add_sortable_doc(&mut writer, id, rank, tie);
            }
            writer.commit().unwrap();
        }
        let infos = writer.segment_infos().clone();
        assert_eq!(infos.segments.len(), 1);
        let sci = &infos.segments[0];
        // rank descending, then tie ascending.
        assert_every_format_agrees(
            &dir,
            sci,
            &[
                ("a", 20),
                ("b", 20),
                ("c", 20),
                ("d", 10),
                ("e", 10),
                ("f", 10),
            ],
        );
        assert_eq!(
            read_index_sort(&dir, sci).unwrap(),
            vec![
                sort_field("rank", true, SortMissingValue::Last),
                sort_field("tie", false, SortMissingValue::Last),
            ]
        );
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    /// A sorted merge whose sources have deletions: the survivors must come
    /// out in sort order and the deleted documents must be gone from *every*
    /// format, not just from the stored fields. Deletions also disable the
    /// stored-fields and term-vector bulk-copy paths, so this exercises the
    /// per-document ones.
    #[test]
    fn a_sorted_merge_drops_deleted_documents_from_every_format() {
        let tmp = tempdir("sorted-merge-deletions");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            sorted_merge_writer(&dir, &[sort_field("rank", false, SortMissingValue::Last)]);
        writer.set_merge_policy(None);
        for batch in [
            vec![("d", 40i64), ("a", 10)],
            vec![("e", 50), ("b", 20)],
            vec![("f", 60), ("c", 30)],
        ] {
            for (id, rank) in batch {
                add_sortable_doc(&mut writer, id, rank, 0);
            }
            writer.commit().unwrap();
        }
        // Delete one document from each segment, by its unique body term.
        for id in ["a", "e", "f"] {
            writer
                .delete_documents_by_term(&[Term::new("body", format!("t{id}").into_bytes())])
                .unwrap();
        }
        writer.commit().unwrap();
        assert_eq!(
            writer
                .segment_infos()
                .segments
                .iter()
                .map(|s| s.del_count)
                .sum::<i32>(),
            3
        );

        let names: Vec<String> = writer
            .segment_infos()
            .segments
            .iter()
            .map(|s| s.segment_name.clone())
            .collect();
        writer.execute_merge(&names).unwrap();
        let infos = writer.segment_infos().clone();
        assert_eq!(infos.segments.len(), 1);
        let sci = &infos.segments[0];
        assert_eq!(sci.del_count, 0, "the merge drops the deleted documents");
        assert_every_format_agrees(&dir, sci, &[("b", 20), ("c", 30), ("d", 40)]);
        // The deleted documents' terms must be gone from the dictionary too,
        // not merely unreachable.
        let seg = per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
        let tim = dir.open(&format!("{seg}.tim")).unwrap();
        let tip = dir.open(&format!("{seg}.tip")).unwrap();
        let tmd = dir.open(&format!("{seg}.tmd")).unwrap();
        let doc_bytes = dir.open(&format!("{seg}.doc")).unwrap();
        let block_fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            &fi::FieldInfos {
                fields: sortable_fields_with(true, true),
            },
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            3,
        )
        .unwrap();
        let doc_in = DocInput::open(
            &doc_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .unwrap();
        let body_terms = block_fields.field("body").unwrap();
        for id in ["a", "e", "f"] {
            assert!(
                body_terms
                    .postings(format!("t{id}").as_bytes(), Some(&doc_in))
                    .unwrap()
                    .is_none(),
                "deleted document {id}'s term survived the merge"
            );
        }
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    /// A multi-field sort where the **second** tier is what actually orders
    /// most of the batch, plus a reverse first tier -- a single ascending key
    /// is the case most likely to come out right by accident.
    #[test]
    fn a_multi_field_sort_breaks_ties_with_its_second_tier() {
        let tmp = tempdir("sort-multi-field");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_doc_values_field("tie").unwrap();
        writer
            .set_index_sort(Some(&[
                sort_field("rank", true, SortMissingValue::Last),
                sort_field("tie", false, SortMissingValue::Last),
            ]))
            .unwrap();

        // Two rank groups; within each, `tie` decides. Inserted in an order
        // that is wrong on both tiers.
        for (id, rank, tie) in [
            ("a", 1, 30),
            ("b", 2, 20),
            ("c", 1, 10),
            ("d", 2, 10),
            ("e", 1, 20),
        ] {
            writer
                .add_document(sortable_doc(id, rank, tie, "x"))
                .unwrap();
        }
        let infos = writer.commit().unwrap().clone();
        let sci = &infos.segments[0];

        // rank descending: the 2s first (d then b, by tie ascending), then
        // the 1s (c, e, a).
        assert_eq!(read_all_docs(&dir, &infos), vec!["d", "b", "c", "e", "a"]);
        assert_eq!(
            read_base_numeric_column(&dir, sci, 1, 5),
            vec![Some(2), Some(2), Some(1), Some(1), Some(1)]
        );
        assert_eq!(
            read_base_numeric_column(&dir, sci, 2, 5),
            vec![Some(10), Some(20), Some(10), Some(20), Some(30)]
        );
        assert_eq!(read_index_sort(&dir, sci).unwrap().len(), 2);
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    /// A buffer that is already in sort order still gets its sort recorded.
    /// Java's `Sorter.sortAndLeaveUnpacked` returns `null` (no `DocMap`) in
    /// this case, but `SegmentInfo.setIndexSort` was already called.
    #[test]
    fn an_already_ordered_buffer_is_still_recorded_as_sorted() {
        let tmp = tempdir("sort-identity");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer
            .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
            .unwrap();
        for (id, rank) in [("a", 1), ("b", 2), ("c", 3)] {
            writer.add_document(sortable_doc(id, rank, 0, "x")).unwrap();
        }
        let infos = writer.commit().unwrap().clone();
        assert_eq!(read_all_docs(&dir, &infos), vec!["a", "b", "c"]);
        assert_eq!(read_index_sort(&dir, &infos.segments[0]).unwrap().len(), 1);
    }

    /// A document with no value for the sort field takes that field's
    /// sentinel (`Long.MIN_VALUE` for `First`, `Long.MAX_VALUE` for `Last`)
    /// and is then compared like any other value -- so `reverse` moves it to
    /// the *other* end. This is what the `.si` this flush writes actually
    /// says, and what `CheckIndex.testSort` checks.
    #[test]
    fn a_missing_sort_key_takes_its_sentinel_and_reverses_with_it() {
        for (reverse, expected) in [
            (false, vec!["a", "c", "b"]),
            // `Last` == Long.MAX_VALUE; reversed, the largest comes first.
            (true, vec!["b", "c", "a"]),
        ] {
            let tmp = tempdir(&format!("sort-missing-{reverse}"));
            let dir = FsDirectory::open(&tmp);
            let mut writer =
                IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
            writer.set_doc_values_field(Some("rank")).unwrap();
            writer
                .set_index_sort(Some(&[sort_field("rank", reverse, SortMissingValue::Last)]))
                .unwrap();
            writer.add_document(sortable_doc("a", 10, 0, "x")).unwrap();
            // `b` has no `rank` field at all.
            writer
                .add_document(Document {
                    fields: vec![StoredField {
                        field_number: 0,
                        value: FieldValue::String("b".to_string()),
                    }],
                })
                .unwrap();
            writer.add_document(sortable_doc("c", 20, 0, "x")).unwrap();
            let infos = writer.commit().unwrap().clone();
            assert_eq!(read_all_docs(&dir, &infos), expected, "reverse={reverse}");
            // The doc-values column is sparse (b has no value) and the
            // segment still passes our own sort check, which re-derives the
            // comparator from the `.si`.
            for result in crate::check_index::check_directory(&dir).unwrap() {
                assert!(result.all_passed(), "{:?}", result.failures());
            }
        }
    }

    /// A segment-private delete's `docIDUpto` is a **pre-sort** buffer
    /// position. `update_document` buffers the delete at the position of the
    /// document it replaces, so on a sorted flush the limit has to be
    /// compared against `newToOld(doc)` -- Java keeps the `Sorter.DocMap` on
    /// the pooled `ReadersAndUpdates` for exactly this. Without the mapping
    /// the sort silently changes which documents an update deletes.
    #[test]
    fn a_private_delete_limit_is_mapped_back_through_the_sort() {
        let tmp = tempdir("sort-private-delete");
        let dir = FsDirectory::open(&tmp);
        let mut writer = IndexWriter::open(
            &dir,
            sortable_fields_with(true, false),
            "Lucene104",
            version(),
        )
        .unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        // Descending, so the buffer order and the sorted order are reversed
        // and an unmapped limit cuts exactly the wrong end off.
        writer
            .set_index_sort(Some(&[sort_field("rank", true, SortMissingValue::Last)]))
            .unwrap();

        // Buffer positions 0..2 all carry body term "dup"; position 3 is an
        // `update_document` whose delete has `docIDUpto = 3`, so it must
        // delete exactly the first three and not itself.
        for (id, rank) in [("p0", 10), ("p1", 20), ("p2", 30)] {
            writer
                .add_document(sortable_doc(id, rank, 0, "dup"))
                .unwrap();
        }
        writer
            .update_document(
                Term {
                    field: "body".to_string(),
                    bytes: b"dup".to_vec(),
                },
                sortable_doc("p3", 40, 0, "dup"),
            )
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        let sci = &infos.segments[0];

        // Sorted descending, doc 0 is p3 (rank 40) -- the replacement, which
        // in the *pre-sort* space sat at position 3 and is therefore NOT
        // below the limit. The other three are deleted.
        assert_eq!(sci.del_count, 3);
        let liv = dir
            .open(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))
            .unwrap();
        let live =
            lucene_codecs::live_docs::parse(&liv, &sci.segment_id, sci.del_gen, 4, 3).unwrap();
        assert!(live.get(0), "p3 (sorted first) must survive");
        for d in 1..4 {
            assert!(!live.get(d), "doc {d} must be deleted");
        }
    }

    /// A flush that fails **after** the sort must leave the buffer in
    /// insertion order, because every buffered delete's `docIDUpto` is a
    /// position in that order. Without the restore the retry sees an
    /// already-sorted buffer, short-circuits to "no sort map", and then
    /// compares pre-sort limits against sorted doc ids -- deleting the exact
    /// complement of the right documents, with a valid `.liv` and a clean
    /// `CheckIndex`.
    ///
    /// The failure is injected the way a caller would hit it: a second
    /// doc-values field with a non-numeric value on one document, which
    /// `build_doc_values_output` rejects *after* `sort_pending_buffer` has
    /// run.
    #[test]
    fn a_flush_that_fails_after_the_sort_leaves_the_buffer_in_insertion_order() {
        let tmp = tempdir("sort-failed-flush-retry");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            numeric_field("rank", 1),
            numeric_field("tie", 2),
            FieldInfo {
                index_options: IndexOptions::DocsAndFreqs,
                ..stored_only_field("body", 3)
            },
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_doc_values_field("tie").unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer
            .set_index_sort(Some(&[sort_field("rank", true, SortMissingValue::Last)]))
            .unwrap();

        for (id, rank) in [("p0", 10), ("p1", 20), ("p2", 30)] {
            writer
                .add_document(sortable_doc(id, rank, 0, "dup"))
                .unwrap();
        }
        // Position 3: the replacement, whose delete carries `docIDUpto = 3`.
        // Its `tie` is a *string*, so the doc-values build fails -- after the
        // sort has already permuted the buffer.
        writer
            .update_document(
                Term {
                    field: "body".to_string(),
                    bytes: b"dup".to_vec(),
                },
                Document {
                    fields: vec![
                        StoredField {
                            field_number: 0,
                            value: FieldValue::String("p3".to_string()),
                        },
                        StoredField {
                            field_number: 1,
                            value: FieldValue::Long(40),
                        },
                        StoredField {
                            field_number: 2,
                            value: FieldValue::String("not a number".to_string()),
                        },
                        StoredField {
                            field_number: 3,
                            value: FieldValue::String("dup".to_string()),
                        },
                    ],
                },
            )
            .unwrap();
        // The document index in the message is the doc id in the *segment
        // being built* (0 -- p3 sorts first under a descending rank), not the
        // caller's buffer position. That is the only observable trace the
        // sort leaves on an error, and it is the honest one: the value that
        // was rejected is at that doc id in the segment this flush was
        // writing.
        assert!(matches!(
            writer.flush().unwrap_err(),
            Error::NonNumericDocValue(ref f, 0, _) if f == "tie"
        ));

        // The buffer is back in insertion order -- checked directly, because
        // that is the invariant, not just its consequence.
        assert_eq!(
            writer
                .pending_docs
                .iter()
                .map(doc_value)
                .collect::<Vec<_>>(),
            vec!["p0", "p1", "p2", "p3"]
        );

        // Repair the offending document and retry. The delete must still
        // reach exactly the three documents that preceded it.
        writer.pending_docs[3].fields[2].value = FieldValue::Long(0);
        writer.commit().unwrap();
        let infos = writer.segment_infos().clone();
        let sci = &infos.segments[0];
        assert_eq!(sci.del_count, 3);
        let liv = dir
            .open(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))
            .unwrap();
        let live =
            lucene_codecs::live_docs::parse(&liv, &sci.segment_id, sci.del_gen, 4, 3).unwrap();
        // Descending by rank, so p3 (rank 40) is doc 0 -- and in the pre-sort
        // space it sat at position 3, which is not below the limit.
        assert!(live.get(0), "p3 must survive the retried flush");
        for d in 1..4 {
            assert!(!live.get(d), "doc {d} must be deleted");
        }
    }

    /// `IndexingChain.maybeSortSegment`'s `CorruptIndexException`: a block
    /// of documents must stay contiguous and in order, which an index sort
    /// would shred. Java allows the pair only with a parent field marking
    /// each block's last document; this port has no parent-field write path.
    #[test]
    fn document_blocks_and_an_index_sort_are_refused_together() {
        let tmp = tempdir("sort-blocks");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer
            .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
            .unwrap();
        writer
            .add_documents(vec![
                sortable_doc("a", 2, 0, "x"),
                sortable_doc("b", 1, 0, "x"),
            ])
            .unwrap();
        assert!(matches!(
            writer.flush().unwrap_err(),
            Error::IndexSortWithBlocksAndNoParentField
        ));
    }

    /// `IndexWriter.updateNumericDocValue`: rewriting the column the
    /// segment's physical order is defined over would leave every existing
    /// segment claiming a sort it no longer satisfies.
    #[test]
    fn a_doc_values_update_on_a_sort_field_is_refused() {
        let tmp = tempdir("sort-dv-update");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_doc_values_field("tie").unwrap();
        writer
            .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
            .unwrap();
        let term = Term {
            field: "body".to_string(),
            bytes: b"x".to_vec(),
        };
        let err = writer
            .update_numeric_doc_value(term.clone(), "rank", 7)
            .unwrap_err();
        assert!(matches!(
            err,
            Error::DocValuesUpdateOnIndexSortField { field, .. } if field == "rank"
        ));
        // A doc-values field that is *not* part of the sort is still fine.
        writer.update_numeric_doc_value(term, "tie", 7).unwrap();
    }

    /// Two doc-values fields land in one `.dvm`/`.dvd` pair, which is what
    /// makes a multi-field index sort expressible at all.
    #[test]
    fn two_doc_values_fields_share_one_dvm() {
        let tmp = tempdir("multi-dv");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_doc_values_field("tie").unwrap();
        assert!(matches!(
            writer.add_doc_values_field("tie").unwrap_err(),
            Error::DuplicateDocValuesField(f) if f == "tie"
        ));
        writer.add_document(sortable_doc("a", 1, 100, "x")).unwrap();
        writer.add_document(sortable_doc("b", 2, 200, "x")).unwrap();
        let infos = writer.commit().unwrap().clone();
        let sci = &infos.segments[0];
        assert_eq!(
            read_base_numeric_column(&dir, sci, 1, 2),
            vec![Some(1), Some(2)]
        );
        assert_eq!(
            read_base_numeric_column(&dir, sci, 2, 2),
            vec![Some(100), Some(200)]
        );
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    /// `doc_values::write_dense_fields` accepts a sparse column only for
    /// NUMERIC, so a multi-field configuration cannot silently drop another
    /// type's sparse docs -- and a sparse NUMERIC tier, which is what an
    /// index sort with missing keys needs, goes through.
    #[test]
    fn a_multi_field_doc_values_flush_allows_a_sparse_numeric_but_not_a_sparse_sorted() {
        // Sparse NUMERIC: accepted, and readable as sparse.
        let tmp = tempdir("multi-dv-sparse-numeric");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_doc_values_field("tie").unwrap();
        writer.add_document(sortable_doc("a", 1, 100, "x")).unwrap();
        // No `tie` value on this one.
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("b".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(2),
                    },
                ],
            })
            .unwrap();
        let infos = writer.commit().unwrap().clone();
        let sci = &infos.segments[0];
        assert_eq!(
            read_base_numeric_column(&dir, sci, 2, 2),
            vec![Some(100), None]
        );
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }

        // Sparse SORTED alongside it: rejected, naming the field and the
        // number of documents that have no value.
        let tmp = tempdir("multi-dv-sparse-sorted");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![
            stored_only_field("id", 0),
            numeric_field("rank", 1),
            sorted_field("label", 2),
        ];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_doc_values_field("label").unwrap();
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(1),
                    },
                    StoredField {
                        field_number: 2,
                        value: FieldValue::String("x".to_string()),
                    },
                ],
            })
            .unwrap();
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("b".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(2),
                    },
                ],
            })
            .unwrap();
        let err = writer.flush().unwrap_err();
        assert!(matches!(
            err,
            Error::SparseFieldInMultiFieldDocValues { ref field, missing: 1, max_doc: 2 }
                if field == "label"
        ));
    }

    /// An index-sorted segment *is* offered to the merge policy now, because
    /// the merge preserves the sort. What must not happen is a merge whose
    /// sources **disagree** about the sort: concatenating them would produce
    /// a segment in one order whose `.si` describes another, which is valid,
    /// `CheckIndex`-clean and wrong. Java raises
    /// `IllegalArgumentException("cannot change index sort ...")`
    /// (`IndexWriter.validateIndexSort`); this refuses the merge.
    ///
    /// The `.si` is patched after the fact rather than written by a sorted
    /// flush, so the sort declaration is the *only* difference between the
    /// three segments.
    /// Java's `SegmentMerger` takes `maxDoc` from `SegmentReader.maxDoc()`,
    /// i.e. from the `.si`. This port's `execute_merge` read it off the
    /// `.fdm` instead, where `stored_fields::open` checks only that it is
    /// non-negative -- and `stored_fields.rs` says so at the site ("This port
    /// has no `SegmentInfo` to hand, so the `.fdm` copy *is* the document
    /// count").
    ///
    /// `merge_segments` then sizes two `Vec`s from it, per source: the live
    /// doc-id list and the dense doc-id map. A four-byte edit claiming
    /// `maxDoc = i32::MAX` makes each of those ~8.6 GB -- an allocation
    /// abort, which is the one failure `catch_unwind` at the FFI boundary
    /// cannot intercept (`docs/arithmetic-gate.md`'s fourth row).
    ///
    /// Without the cross-check this test does not fail politely: it takes the
    /// container's whole memory cap and is killed.
    #[test]
    fn a_segment_whose_fdm_disagrees_with_its_si_about_max_doc_is_not_merged() {
        /// Overwrites the `.fdm`'s own `maxDoc`, leaving every other file
        /// alone, and re-signs the footer so only the semantic disagreement
        /// can fire.
        ///
        /// Layout: `CodecUtil.writeIndexHeader` (magic, codec name, version,
        /// id, suffix), a vint `chunkSize`, then the big-endian `maxDoc`.
        fn patch_fdm_max_doc(fdm: &mut [u8], expected_now: i32, max_doc: i32) {
            // The field is *found* rather than computed: the vint `chunkSize`
            // between the index header and `maxDoc` is a variable width. The
            // search asserts the match is unique in the window, so a wrong
            // offset cannot pass silently and patch some other field.
            let start =
                lucene_store::codec_util::index_header_length("Lucene90FieldsIndexMeta", "");
            // Little-endian: Lucene 9 switched `DataInput.readInt` to LE, and
            // this is `read_i32`, not `read_be_i32`. (The codec *footer*
            // below is still big-endian, which is why the two differ here.)
            let want = expected_now.to_le_bytes();
            let hits: Vec<usize> = (start..start + 8)
                .filter(|&p| fdm[p..p + 4] == want)
                .collect();
            assert_eq!(hits.len(), 1, "maxDoc offset is ambiguous: {hits:?}");
            let at = hits[0];
            fdm[at..at + 4].copy_from_slice(&max_doc.to_le_bytes());
            let n = fdm.len();
            let crc = crc32fast::hash(&fdm[..n - 8]) as u64;
            fdm[n - 8..].copy_from_slice(&crc.to_be_bytes());
        }

        let tmp = tempdir("fdm-vs-si-max-doc");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        // Three documents in the first segment and two in the second, so the
        // `maxDoc` this test patches is `3` -- a value that cannot be
        // confused with the run of zero bytes around it.
        for d in 0..3 {
            writer.add_document(doc(&format!("a{d}"))).unwrap();
        }
        writer.commit().unwrap();
        for d in 0..2 {
            writer.add_document(doc(&format!("b{d}"))).unwrap();
        }
        writer.commit().unwrap();
        let names: Vec<String> = writer
            .segment_infos()
            .segments
            .iter()
            .map(|s| s.segment_name.clone())
            .collect();
        assert_eq!(names.len(), 2);

        // The control first: these two segments merge cleanly, so the failure
        // below is the edit and not the fixture.
        {
            let tmp_ok = tempdir("fdm-vs-si-max-doc-control");
            let dir_ok = FsDirectory::open(&tmp_ok);
            let mut ok = IndexWriter::open(
                &dir_ok,
                vec![stored_only_field("id", 0)],
                "Lucene104",
                version(),
            )
            .unwrap();
            for d in 0..3 {
                ok.add_document(doc(&format!("a{d}"))).unwrap();
            }
            ok.commit().unwrap();
            for d in 0..2 {
                ok.add_document(doc(&format!("b{d}"))).unwrap();
            }
            ok.commit().unwrap();
            let ok_names: Vec<String> = ok
                .segment_infos()
                .segments
                .iter()
                .map(|s| s.segment_name.clone())
                .collect();
            ok.execute_merge(&ok_names).unwrap();
            assert_eq!(ok.segment_infos().segments.len(), 1);
        }

        let fdm_name = format!("{}.fdm", names[0]);
        let mut fdm = dir.open(&fdm_name).unwrap().to_vec();
        patch_fdm_max_doc(&mut fdm, 3, i32::MAX);
        write_file(&dir, &fdm_name, &fdm).unwrap();

        assert!(
            matches!(
                writer.execute_merge(&names),
                Err(Error::SegmentDocCountMismatch {
                    si_doc_count: 3,
                    stored_fields_max_doc: i32::MAX,
                    ..
                })
            ),
            "a .fdm claiming 2 billion documents must be reported, not reserved for"
        );
    }

    #[test]
    fn a_field_length_past_integer_max_value_saturates_instead_of_wrapping() {
        // Java steps `FieldInvertState.length` with
        // `Math.addExact(invertState.length, 1)` -- it throws rather than
        // wrapping past `Integer.MAX_VALUE`. The `+=` this replaced wrapped a
        // `u32` instead: the longest document in the index would encode to a
        // *small* norm and score as one of the shortest, silently, in every
        // BM25 query over the field. It also trips `int_to_byte4`'s own
        // `debug_assert` on the way past `i32::MAX`.
        const MAX: u32 = i32::MAX as u32;
        assert_eq!(accumulate_field_length(0, 0), 0);
        assert_eq!(accumulate_field_length(3, 4), 7);
        assert_eq!(accumulate_field_length(MAX, 1), MAX, "clamped, not wrapped");
        assert_eq!(accumulate_field_length(MAX - 1, 5), MAX);
        assert_eq!(accumulate_field_length(u32::MAX, 1), MAX);
        // A negative frequency can only come from an occurrence count above
        // `i32::MAX`, i.e. "longer than the longest", never "shorter".
        assert_eq!(accumulate_field_length(0, -1), MAX);
        // And the whole reachable domain still encodes monotonically.
        assert!(
            small_float::int_to_byte4(accumulate_field_length(MAX, 1))
                >= small_float::int_to_byte4(accumulate_field_length(1_000, 1))
        );
    }

    #[test]
    fn a_merge_of_segments_that_disagree_about_the_index_sort_is_refused() {
        let tmp = tempdir("sort-merge-disagreement");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        for batch in 0..3 {
            writer.add_document(doc(&format!("d{batch}"))).unwrap();
            writer.commit().unwrap();
        }
        assert_eq!(writer.segment_infos().segments.len(), 3);
        assert_eq!(writer.segment_stats().unwrap().len(), 3);

        // Declare a sort on the middle segment's `.si` only.
        let sci = writer.segment_infos().segments[1].clone();
        let si_name = format!("{}.si", sci.segment_name);
        let si_bytes = dir.open(&si_name).unwrap().to_vec();
        let mut si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        si.index_sort = Some(vec![sort_field("rank", false, SortMissingValue::Last)]);
        write_file(&dir, &si_name, &segment_info::write(&si, "")).unwrap();

        // Still a merge candidate -- the exclusion is gone.
        let stats = writer.segment_stats().unwrap();
        assert_eq!(stats.len(), 3);
        assert!(stats.iter().any(|s| s.name == sci.segment_name));

        let names: Vec<String> = writer
            .segment_infos()
            .segments
            .iter()
            .map(|s| s.segment_name.clone())
            .collect();
        assert!(matches!(
            writer.execute_merge(&names),
            Err(Error::MergeSortDisagreement { .. })
        ));
    }

    /// Two sorted flushes from one writer produce two segments that each
    /// claim the sort and are each internally ordered -- the precondition
    /// a sort-preserving merge is built on.
    #[test]
    fn every_sorted_flush_gets_its_own_ordered_segment() {
        let tmp = tempdir("sort-two-segments");
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer
            .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
            .unwrap();
        writer.add_document(sortable_doc("b", 20, 0, "x")).unwrap();
        writer.add_document(sortable_doc("a", 10, 0, "x")).unwrap();
        writer.commit().unwrap();
        writer.add_document(sortable_doc("d", 40, 0, "x")).unwrap();
        writer.add_document(sortable_doc("c", 30, 0, "x")).unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(infos.segments.len(), 2);
        for sci in &infos.segments {
            assert_eq!(read_index_sort(&dir, sci).unwrap().len(), 1);
        }
        assert_eq!(read_all_docs(&dir, &infos), vec!["a", "b", "c", "d"]);
    }
}
