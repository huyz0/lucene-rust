//! `ffi_open_writer`/`ffi_writer_add_document`/`ffi_writer_commit`/
//! `ffi_writer_prepare_commit`/`ffi_writer_finish_commit`/`ffi_writer_rollback`/
//! `ffi_writer_set_merge_policy`/`ffi_writer_update_document`/
//! `ffi_writer_delete_documents`/`ffi_close_writer` (IndexWriter commit/
//! merge-policy/update/delete FFI exposure): wraps
//! [`lucene_index::index_writer::IndexWriter`]'s open/add_document/commit/
//! prepare_commit/finish_commit/rollback/set_merge_policy/update_document/
//! delete_documents lifecycle -- no write-side logic reimplemented here, only
//! the FFI plumbing (handle lifecycle, wire decoding, error mapping) this
//! crate's other modules already follow.
//!
//! **In scope**: opening a writer over a filesystem path with a caller-supplied
//! field list, buffering stored-fields-only documents, the full
//! commit/two-phase-commit/rollback/auto-merge lifecycle, and atomic
//! delete-by-term/update-by-term, exactly as
//! `lucene_index::index_writer::IndexWriter` already implements them.
//!
//! `ffi_writer_set_postings_field`/`ffi_writer_set_term_vector_field`/
//! `ffi_writer_set_doc_values_field` wrap
//! [`IndexWriter::set_postings_field`]/[`IndexWriter::set_term_vector_field`]/
//! [`IndexWriter::set_doc_values_field`] the same way `ffi_writer_set_merge_policy`
//! wraps `set_merge_policy`: an `enabled` flag picks `None` (clears the
//! setting) vs `Some(field_name)`, mirroring these three Rust-side methods'
//! own `Option<&str>` parameter -- no new config surface invented, just the
//! FFI plumbing.
//!
//! `ffi_writer_update_document`/`ffi_writer_delete_documents` wrap
//! [`IndexWriter::update_document`]/[`IndexWriter::delete_documents_by_term`]. Both
//! identify their delete term as raw, already-analyzed `(field_name, term)`
//! bytes -- no analysis happens at this FFI boundary, same stance this
//! crate's `query.rs` already takes for its own raw-bytes terms.
//! `ffi_writer_update_document`'s replacement document reuses
//! `ffi_writer_add_document`'s exact parallel-array field encoding.
//!
//! **Both are buffered**, matching Java: `IndexWriter.updateDocument` and
//! `deleteDocuments(Term...)` put the operation in the delete queue and return
//! a sequence number, and the change becomes visible at the next
//! `ffi_writer_commit`. Until `c7-delete-queue` this crate instead reopened
//! every committed segment itself and drove an eager, immediately-committed
//! delete, which meant one `segments_N` generation *per updated document* --
//! a divergence both in cost and in what a JVM caller observes between the
//! call and the commit. The wrapped [`IndexWriter`] now owns segment
//! reopening, so that machinery is gone from this crate. A segment with no
//! `.tim` file on disk (flushed without [`ffi_writer_set_postings_field`]) is
//! still skipped rather than errored: there is no term dictionary to resolve
//! the delete against, so it contributes no matches.
//!
//! `ffi_writer_segment_infos_len`/`ffi_writer_segment_info_name`/
//! `ffi_writer_pending_doc_count` wrap [`IndexWriter::segment_infos`]/
//! [`IndexWriter::pending_doc_count`] read-only, so a caller can introspect
//! this writer's current committed segment list and buffered-doc count
//! without a separate directory-scan handle -- same "length first, then
//! per-index accessor" shape `results_fragments.rs`'s
//! `ffi_fragment_results_len`/`ffi_fragment_result_text` already established.
//!
//! **Deliberately out of scope, tracked in `docs/parity.md`**: this module
//! does not wrap `IndexWriter::apply_merge` -- folding an already-executed
//! [`lucene_index::merge::merge_stored_only_segments`] result back into a
//! writer only makes sense once that merge has actually been *run*, and this
//! crate exposes no FFI surface to run it (no `ffi_merge_stored_only_segments`
//! or equivalent exists anywhere in this crate today -- merging only ever
//! happens automatically, inside `commit()`, via `set_merge_policy`). Wrapping
//! `apply_merge` alone, with no way to produce the `SegmentCommitInfo` it
//! needs from the JVM side, would be a half-working surface a caller could
//! never actually drive; exposing manual merge execution is a separate,
//! larger task. `set_merge_policy` exposes **all eight**
//! [`lucene_index::merge_policy::MergePolicyConfig`] knobs as of the M2 sweep
//! (`max_merge_at_once`, `segments_per_tier`, `max_merged_segment_size`,
//! `reclaim_weight`, `floor_segment_size`, `force_merge_deletes_pct_allowed`,
//! `deletes_pct_allowed`, `target_search_concurrency`) -- the last three were
//! silently defaulted before, and `force_merge_deletes_pct_allowed` now has a
//! real `find_forced_delete_merges` to configure (b10 rewrote `merge_policy`
//! as a faithful `TieredMergePolicy`; see that module's doc comment).
//!
//! `ffi_writer_delete_all` wraps [`IndexWriter::delete_all`]
//! (`IndexWriter.deleteAll()`), and `ffi_writer_set_live_commit_data`/
//! `ffi_writer_live_commit_data_len`/`ffi_writer_live_commit_data_entry` wrap
//! [`IndexWriter::set_live_commit_data`]/[`IndexWriter::live_commit_data`]
//! (`setLiveCommitData`/`getLiveCommitData`) -- the three capabilities the
//! writer gained after this module was first written.

use std::os::raw::c_char;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::{DeleteQuery, DocValuesUpdate, Term};
use lucene_index::index_writer::{self, IndexWriter, MergePolicyConfig};
use lucene_index::segment_info::LuceneVersion;
use lucene_store::directory::{Directory, FsDirectory};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::{bytes_from_raw, str_from_raw, try_with_capacity};
use crate::registry::{lock_recovering, read_recovering, writers, WriterHandle};

/// Decodes the `(enabled, field_name, field_name_len)` triple
/// [`ffi_writer_set_postings_field`]/[`ffi_writer_set_term_vector_field`]/
/// [`ffi_writer_set_doc_values_field`] all share into the `Option<&str>`
/// their wrapped `IndexWriter` setter expects: `enabled == 0` is `None`
/// (`field_name`/`field_name_len` ignored, same as
/// [`ffi_writer_set_merge_policy`]'s own "ignored but not required" `enabled
/// == 0` convention); otherwise `field_name` is decoded via
/// [`str_from_raw`] (null pointer only valid when `field_name_len == 0`).
///
/// # Safety
/// `field_name` must be valid for reads of `field_name_len` bytes (or null
/// iff `field_name_len == 0`), same contract as [`str_from_raw`].
unsafe fn decode_optional_field_name<'a>(
    enabled: u8,
    field_name: *const c_char,
    field_name_len: usize,
) -> Result<Option<&'a str>, FfiStatus> {
    if enabled == 0 {
        return Ok(None);
    }
    // SAFETY: forwarded from this function's own caller contract.
    let name = unsafe { str_from_raw(field_name, field_name_len)? };
    Ok(Some(name))
}

/// Builds a [`WriterHandle`] over a brand-new (heap-boxed) [`FsDirectory`]
/// rooted at `path`.
///
/// # Safety (why the `unsafe` transmute below is sound)
/// [`IndexWriter`] borrows `&'d dyn Directory` -- there is no owned,
/// `'static` `IndexWriter` type to store in a handle directly. `dir` is
/// heap-allocated (`Box<FsDirectory>`) so its address is stable even if this
/// function's local `dir`/the eventual [`WriterHandle`] value is later moved
/// (only the `Box` pointer moves, never its heap allocation). The borrow
/// handed to `IndexWriter::open` is therefore valid for as long as `dir`
/// itself lives -- which, once both are packed into one [`WriterHandle`], is
/// guaranteed by that struct's field declaration order (`writer` before
/// `dir`; Rust drops fields in declaration order, so the borrow is always
/// dropped before its referent). See [`WriterHandle`]'s own doc comment for
/// the complete argument.
fn open_writer_handle(
    path: &str,
    fields: Vec<FieldInfo>,
    codec_name: String,
    version: LuceneVersion,
) -> index_writer::Result<WriterHandle> {
    let dir = Box::new(FsDirectory::open(path));
    let dir_ref: &dyn Directory = &*dir;
    // SAFETY: see this function's own doc comment and `WriterHandle`'s.
    let dir_ref: &'static dyn Directory = unsafe { std::mem::transmute(dir_ref) };
    let writer = IndexWriter::open(dir_ref, fields, codec_name, version)?;
    Ok(WriterHandle { writer, dir })
}

fn index_options_from_i32(v: i32) -> Result<IndexOptions, FfiStatus> {
    match v {
        0 => Ok(IndexOptions::None),
        1 => Ok(IndexOptions::Docs),
        2 => Ok(IndexOptions::DocsAndFreqs),
        3 => Ok(IndexOptions::DocsAndFreqsAndPositions),
        4 => Ok(IndexOptions::DocsAndFreqsAndPositionsAndOffsets),
        5 => Ok(IndexOptions::DocsAndCustomFreqs),
        _ => Err(FfiStatus::InvalidArgument),
    }
}

fn doc_values_type_from_i32(v: i32) -> Result<DocValuesType, FfiStatus> {
    match v {
        0 => Ok(DocValuesType::None),
        1 => Ok(DocValuesType::Numeric),
        2 => Ok(DocValuesType::Binary),
        3 => Ok(DocValuesType::Sorted),
        4 => Ok(DocValuesType::SortedSet),
        5 => Ok(DocValuesType::SortedNumeric),
        _ => Err(FfiStatus::InvalidArgument),
    }
}

/// Decodes one field's raw bytes into a [`FieldValue`] per `kind`:
/// `0` = UTF-8 string, `1` = raw binary, `2` = `i32` (4 bytes, little-endian),
/// `3` = `i64` (8 bytes, little-endian), `4` = `f32` (4 bytes, little-endian
/// bit pattern), `5` = `f64` (8 bytes, little-endian bit pattern) -- the same
/// six [`FieldValue`] variants `stored_fields.rs` already defines, just a
/// wire encoding this FFI boundary needs since a raw pointer/length pair
/// carries no type tag of its own.
fn decode_field_value(kind: u8, bytes: &[u8]) -> Result<FieldValue, FfiStatus> {
    match kind {
        0 => {
            let s = std::str::from_utf8(bytes).map_err(|_| FfiStatus::InvalidUtf8)?;
            Ok(FieldValue::String(s.to_string()))
        }
        1 => Ok(FieldValue::Binary(bytes.to_vec())),
        2 => {
            let arr: [u8; 4] = bytes.try_into().map_err(|_| FfiStatus::InvalidArgument)?;
            Ok(FieldValue::Int(i32::from_le_bytes(arr)))
        }
        3 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| FfiStatus::InvalidArgument)?;
            Ok(FieldValue::Long(i64::from_le_bytes(arr)))
        }
        4 => {
            let arr: [u8; 4] = bytes.try_into().map_err(|_| FfiStatus::InvalidArgument)?;
            Ok(FieldValue::Float(f32::from_le_bytes(arr)))
        }
        5 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| FfiStatus::InvalidArgument)?;
            Ok(FieldValue::Double(f64::from_le_bytes(arr)))
        }
        _ => Err(FfiStatus::InvalidArgument),
    }
}

/// Maps every [`index_writer::Error`] variant this module's functions can
/// actually produce to a stable [`FfiStatus`], recording the formatted error
/// as the last-error message first (same "set message, then return a status
/// code" convention every other module in this crate already follows).
/// Caller-input validation problems -- an unopened prepared commit, an
/// unknown/unsupported field passed to
/// [`ffi_writer_set_postings_field`]/[`ffi_writer_set_term_vector_field`]/
/// [`ffi_writer_set_doc_values_field`], or a doc-values commit missing/
/// mistyping the opted-in field's value -- become
/// [`FfiStatus::InvalidArgument`]; everything else (I/O, decode, or
/// downstream write-side errors) becomes [`FfiStatus::Io`].
fn map_writer_error(context: &str, e: index_writer::Error) -> FfiStatus {
    let status = match &e {
        index_writer::Error::NoPreparedCommit
        // Caller-misuse, exactly like NoPreparedCommit: a commit is already
        // prepared, so this call would have silently reverted it.
        | index_writer::Error::PrepareCommitAlreadyCalled
        | index_writer::Error::PreparedCommitPending(_)
        | index_writer::Error::UnknownPostingsField(_)
        | index_writer::Error::UnsupportedPostingsIndexOptions(_, _)
        // c23's two payload-configuration errors. Both are what Java raises
        // as `IllegalArgumentException`: `FieldInfo.checkConsistency`'s
        // "indexed field cannot have payloads without positions", and a
        // payload supplier installed against a field list where nothing
        // declares payloads (so every payload would be silently discarded).
        // c40: `IndexWriter::open`'s field-list validation -- Java's
        // `FieldInfo`/`FieldInfos` constructors, which throw
        // `IllegalArgumentException` (this is where "indexed field cannot have
        // payloads without positions" now surfaces).
        | index_writer::Error::FieldInfos(_)
        | index_writer::Error::NoPayloadFields
        | index_writer::Error::UnknownTermVectorField(_)
        | index_writer::Error::UnsupportedTermVectorField(_)
        | index_writer::Error::UnknownDocValuesField(_)
        | index_writer::Error::UnsupportedDocValuesType(_, _)
        | index_writer::Error::MissingDenseDocValue(_, _)
        | index_writer::Error::NonNumericDocValue(_, _, _)
        | index_writer::Error::NonBinaryDocValue(_, _, _)
        // Everything below was reaching the `_ => Io` arm, which is wrong for
        // the same reason the variants above are listed: each one is a
        // caller-misuse error real Lucene raises as `IllegalArgumentException`,
        // and a JNI caller branching on `Io` would retry, log a disk problem,
        // or fail a shard for what is actually a bad argument. The list grew
        // (vector fields, custom-freq postings, norms, the auto-flush knobs,
        // and `c7-delete-queue`'s soft-delete/doc-values-update errors) while
        // this mapping did not; every variant is enumerated explicitly rather
        // than left to a catch-all so the next one added to `lucene-index`
        // fails to compile here instead of silently becoming an I/O error.
        | index_writer::Error::UnknownVectorField(_)
        | index_writer::Error::UnsupportedVectorField(_, _)
        | index_writer::Error::DuplicateVectorField(_)
        | index_writer::Error::VectorDimensionMismatch(_, _, _, _)
        | index_writer::Error::VectorEncodingMismatch(_, _, _, _)
        | index_writer::Error::DuplicatePostingsField(_)
        | index_writer::Error::UnknownCustomFreqPostingsField(_)
        | index_writer::Error::UnsupportedCustomFreqPostingsIndexOptions(_, _)
        | index_writer::Error::PostingsAndCustomFreqPostingsMutuallyExclusive(_)
        | index_writer::Error::DuplicateTermVectorField(_)
        | index_writer::Error::UnknownNormsField(_)
        | index_writer::Error::UnsupportedNormsField(_)
        | index_writer::Error::InvalidRamBufferSize(_)
        | index_writer::Error::InvalidMaxBufferedDocs(_)
        | index_writer::Error::BothAutoFlushTriggersDisabled
        | index_writer::Error::NoSoftDeletesSupplied
        | index_writer::Error::NoDocValuesUpdatesSupplied
        | index_writer::Error::UnknownDocValuesUpdateField(_)
        | index_writer::Error::WrongDocValuesUpdateType { .. }
        // `c17-index-sort`'s configuration surface. Every one of these is an
        // `IllegalArgumentException` (or, for the blocks case, a
        // `CorruptIndexException` raised on a caller-chosen combination) in
        // real Lucene: a sort field that is not a NUMERIC doc-values field,
        // a sort that disagrees with the segments already in the index, a
        // doc-values update aimed at the column the sort is defined over.
        | index_writer::Error::EmptyIndexSort
        | index_writer::Error::UnknownIndexSortField(_)
        | index_writer::Error::UnsupportedIndexSortField(_, _)
        | index_writer::Error::UnsupportedIndexSortKind(_)
        | index_writer::Error::IndexSortFieldWithoutDocValues(_)
        | index_writer::Error::IncongruentIndexSort { .. }
        | index_writer::Error::IndexSortChangedMidBuffer(_)
        | index_writer::Error::IndexSortWithBlocksAndNoParentField
        | index_writer::Error::DocValuesUpdateOnIndexSortField { .. }
        | index_writer::Error::SparseFieldInMultiFieldDocValues { .. }
        | index_writer::Error::DuplicateDocValuesField(_)
        // `c22-sorted-merge`'s two configuration errors. Both describe a
        // writer configured inconsistently with the index it was opened on,
        // which is what real Lucene's `validateIndexSort` raises as
        // `IllegalArgumentException`: segments in one merge declaring
        // different sorts, and a sort field this writer's field list does not
        // contain. Retrying either is futile, so `Io` would be actively
        // misleading to a JNI caller.
        | index_writer::Error::MergeSortDisagreement { .. }
        | index_writer::Error::UnknownSortField(_) => FfiStatus::InvalidArgument,
        // Everything left is a genuine I/O or decode failure of the index
        // itself. Enumerated rather than left to a `_` arm so that the next
        // variant added to `lucene_index::index_writer::Error` fails to
        // compile here and has to be *classified*, instead of silently
        // becoming an `Io` a JNI caller would read as "the disk or the index
        // is broken" -- which is exactly how the whole list above came to be
        // misclassified in the first place.
        index_writer::Error::Store(_)
        | index_writer::Error::SegmentWriter(_)
        | index_writer::Error::SegmentInfos(_)
        | index_writer::Error::UpdateDocument(_)
        | index_writer::Error::TermDelete(_)
        | index_writer::Error::Deletes(_)
        | index_writer::Error::FieldUpdates(_)
        | index_writer::Error::Deleter(_)
        | index_writer::Error::Merge(_)
        | index_writer::Error::SegmentInfo(_)
        | index_writer::Error::StoredFields(_)
        | index_writer::Error::LiveDocs(_)
        | index_writer::Error::PostingsWriter(_)
        | index_writer::Error::Blocktree(_)
        | index_writer::Error::Postings(_)
        | index_writer::Error::TermVectors(_)
        | index_writer::Error::DocValues(_)
        | index_writer::Error::Norms(_)
        | index_writer::Error::Vectors(_)
        // The read-side halves of a merge's inputs, exactly like `DocValues`
        // and `Norms` above: a source segment's column could not be decoded.
        | index_writer::Error::DocValuesRead(_)
        | index_writer::Error::NormsRead(_)
        // Not caller misuse: a segment's own `.si` declares an index sort it
        // has no doc-values column to satisfy, so the segment contradicts
        // itself. That is a corrupt index, the same class as
        // `UnreadableSegmentPostings`, and no argument the caller could pass
        // would make it succeed.
        | index_writer::Error::MergeSortColumnMissing(_)
        // Same class again: a segment's `.si` and its stored-fields metadata
        // disagree about how many documents it has, so the segment
        // contradicts itself and no caller argument would make the merge
        // succeed.
        | index_writer::Error::SegmentDocCountMismatch { .. }
        | index_writer::Error::UnreadableSegmentPostings { .. } => FfiStatus::Io,
    };
    set_last_error(format!("{context}: {e}"));
    status
}

/// Opens a writer over a filesystem directory at `path`, describing every
/// field a later [`ffi_writer_add_document`] call may use via five parallel
/// arrays (`field_names`/`field_name_lens`, `field_numbers`,
/// `field_index_options`, `field_doc_values_types`,
/// `field_store_term_vectors`), each `field_count` elements long -- same
/// "parallel arrays describe a list of like-shaped things" convention
/// `segment.rs`'s file-name parameters and `query.rs`'s clause arrays
/// already use in this crate.
///
/// - `field_index_options`/`field_doc_values_types`: the wire encoding of
///   [`IndexOptions`]/[`DocValuesType`]'s declaration order (`0..=5`/`0..=5`
///   respectively) -- an out-of-range value is
///   [`FfiStatus::InvalidArgument`].
/// - `field_store_term_vectors`: `0`/non-`0` per field.
/// - Every other [`FieldInfo`] flag (`omit_norms`, `store_payloads`,
///   `soft_deletes_field`, `parent_field`, points/vector dimensions) is fixed
///   at its default/off value -- this task's scope is commit/merge-policy
///   FFI exposure, not a full field-schema wire format; a caller needing
///   those flags has no way to set them through this entry point yet (see
///   module doc comment / `docs/parity.md`).
///
/// Writes the new writer handle to `*out_handle` on success.
///
/// # Safety
/// `path` must be valid for reads of `path_len` bytes. Every
/// `(*const u8, len)` array pointer must be valid for reads of
/// `field_count` elements (or, when `field_count == 0`, may be null).
/// `codec_name` must be valid for reads of `codec_name_len` bytes.
/// `out_handle` must be valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_open_writer(
    path: *const c_char,
    path_len: usize,
    field_names: *const *const u8,
    field_name_lens: *const usize,
    field_numbers: *const i32,
    field_index_options: *const i32,
    field_doc_values_types: *const i32,
    field_store_term_vectors: *const u8,
    field_count: usize,
    codec_name: *const c_char,
    codec_name_len: usize,
    lucene_version_major: i32,
    lucene_version_minor: i32,
    lucene_version_bugfix: i32,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `path`/`codec_name` are valid for
        // their paired lengths.
        let (path_str, codec_name_str) = unsafe {
            (
                str_from_raw(path, path_len)?,
                str_from_raw(codec_name, codec_name_len)?,
            )
        };

        let fields = if field_count == 0 {
            Vec::new()
        } else {
            if field_names.is_null()
                || field_name_lens.is_null()
                || field_numbers.is_null()
                || field_index_options.is_null()
                || field_doc_values_types.is_null()
                || field_store_term_vectors.is_null()
            {
                return Err(FfiStatus::NullPointer);
            }
            // SAFETY: caller contract guarantees each array is valid for
            // `field_count` elements.
            let (names, name_lens, numbers, index_options, doc_values_types, store_tvs) = unsafe {
                (
                    std::slice::from_raw_parts(field_names, field_count),
                    std::slice::from_raw_parts(field_name_lens, field_count),
                    std::slice::from_raw_parts(field_numbers, field_count),
                    std::slice::from_raw_parts(field_index_options, field_count),
                    std::slice::from_raw_parts(field_doc_values_types, field_count),
                    std::slice::from_raw_parts(field_store_term_vectors, field_count),
                )
            };

            let mut fields = try_with_capacity(field_count)?;
            for i in 0..field_count {
                // SAFETY: caller contract guarantees `names[i]` is valid for
                // `name_lens[i]` bytes.
                // `field_names` is declared `*const *const u8` rather than
                // `*const *const c_char` like every other C-string parameter in
                // this crate; `.cast()` bridges that without an `as` expression,
                // which would be target-dependent (see `str_from_raw`).
                let name = unsafe { str_from_raw(names[i].cast::<c_char>(), name_lens[i])? };
                fields.push(FieldInfo {
                    name: name.to_string(),
                    number: numbers[i],
                    store_term_vectors: store_tvs[i] != 0,
                    omit_norms: false,
                    store_payloads: false,
                    soft_deletes_field: false,
                    parent_field: false,
                    index_options: index_options_from_i32(index_options[i])?,
                    doc_values_type: doc_values_type_from_i32(doc_values_types[i])?,
                    doc_values_skip_index_type: DocValuesSkipIndexType::None,
                    doc_values_gen: -1,
                    attributes: vec![],
                    point_dimension_count: 0,
                    point_index_dimension_count: 0,
                    point_num_bytes: 0,
                    vector_dimension: 0,
                    vector_encoding: VectorEncoding::Float32,
                    vector_similarity_function: VectorSimilarityFunction::Euclidean,
                });
            }
            fields
        };

        let version = LuceneVersion {
            major: lucene_version_major,
            minor: lucene_version_minor,
            bugfix: lucene_version_bugfix,
        };

        let handle = open_writer_handle(path_str, fields, codec_name_str.to_string(), version)
            .map_err(|e| map_writer_error("ffi_open_writer", e))?;
        let handle = lock_recovering(writers()).insert_checked(handle)?;
        // SAFETY: caller contract guarantees `out_handle` is valid for one write.
        unsafe {
            *out_handle = handle;
        }
        Ok(())
    })
}

/// Buffers one document for the writer identified by `writer_handle` (see
/// [`IndexWriter::add_document`]) -- nothing is written to disk until a
/// later [`ffi_writer_commit`]/[`ffi_writer_prepare_commit`] call.
///
/// The document's fields are described by four parallel arrays
/// (`field_numbers`, `field_kinds`, `field_value_ptrs`/`field_value_lens`),
/// each `field_count` elements long -- `field_kinds[i]` selects how
/// `field_value_ptrs[i]`/`field_value_lens[i]`'s bytes are decoded (see
/// [`decode_field_value`]'s doc comment for the six kind values).
///
/// # Safety
/// `field_numbers`/`field_kinds`/`field_value_ptrs`/`field_value_lens` must
/// each be valid for reads of `field_count` elements (or, when
/// `field_count == 0`, may be null); every `field_value_ptrs[i]` must be
/// valid for reads of `field_value_lens[i]` bytes (or null iff that length is
/// `0`).
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_add_document(
    writer_handle: u64,
    field_numbers: *const i32,
    field_kinds: *const u8,
    field_value_ptrs: *const *const u8,
    field_value_lens: *const usize,
    field_count: usize,
    out_seq_no: *mut i64,
) -> i32 {
    guard(|| {
        let mut fields = try_with_capacity(field_count)?;
        if field_count > 0 {
            if field_numbers.is_null()
                || field_kinds.is_null()
                || field_value_ptrs.is_null()
                || field_value_lens.is_null()
            {
                return Err(FfiStatus::NullPointer);
            }
            // SAFETY: caller contract guarantees each array is valid for
            // `field_count` elements.
            let (numbers, kinds, value_ptrs, value_lens) = unsafe {
                (
                    std::slice::from_raw_parts(field_numbers, field_count),
                    std::slice::from_raw_parts(field_kinds, field_count),
                    std::slice::from_raw_parts(field_value_ptrs, field_count),
                    std::slice::from_raw_parts(field_value_lens, field_count),
                )
            };
            for i in 0..field_count {
                // SAFETY: caller contract guarantees `value_ptrs[i]` is valid
                // for `value_lens[i]` bytes.
                let bytes = unsafe { bytes_from_raw(value_ptrs[i], value_lens[i])? };
                let value = decode_field_value(kinds[i], bytes)?;
                fields.push(StoredField {
                    field_number: numbers[i],
                    value,
                });
            }
        }

        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_add_document: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let seq_no = handle
            .writer
            .add_document(Document { fields })
            .map_err(|e| map_writer_error("ffi_writer_add_document", e))?;
        // SAFETY: caller contract: `out_seq_no` is valid for one `i64` write
        // or null.
        unsafe { write_seq_no(out_seq_no, seq_no) };
        Ok(())
    })
}

/// Flushes any buffered documents and writes the next `segments_N`
/// generation -- see [`IndexWriter::commit`].
#[no_mangle]
pub extern "C" fn ffi_writer_commit(writer_handle: u64) -> i32 {
    guard(|| {
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_commit: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        handle
            .writer
            .commit()
            .map(|_| ())
            .map_err(|e| map_writer_error("ffi_writer_commit", e))
    })
}

/// The file-writing half of a two-phase commit -- see
/// [`IndexWriter::prepare_commit`].
#[no_mangle]
pub extern "C" fn ffi_writer_prepare_commit(writer_handle: u64) -> i32 {
    guard(|| {
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_prepare_commit: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        handle
            .writer
            .prepare_commit()
            .map_err(|e| map_writer_error("ffi_writer_prepare_commit", e))
    })
}

/// The activation half of a two-phase commit -- see
/// [`IndexWriter::finish_commit`]. Returns [`FfiStatus::InvalidArgument`]
/// (via [`index_writer::Error::NoPreparedCommit`]) if no
/// [`ffi_writer_prepare_commit`] call is currently pending.
#[no_mangle]
pub extern "C" fn ffi_writer_finish_commit(writer_handle: u64) -> i32 {
    guard(|| {
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_finish_commit: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        handle
            .writer
            .finish_commit()
            .map(|_| ())
            .map_err(|e| map_writer_error("ffi_writer_finish_commit", e))
    })
}

/// Discards every document buffered since the last commit -- see
/// [`IndexWriter::rollback`]. Infallible (matches `IndexWriter::rollback`'s
/// own `fn rollback(&mut self)` signature), so the only failure mode here is
/// an invalid handle.
#[no_mangle]
pub extern "C" fn ffi_writer_rollback(writer_handle: u64) -> i32 {
    guard(|| {
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_rollback: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        handle.writer.rollback();
        Ok(())
    })
}

/// Opts (`enabled != 0`) or opts out (`enabled == 0`) this writer into
/// automatic merge triggering -- see [`IndexWriter::set_merge_policy`].
/// `max_merge_at_once`/`segments_per_tier`/`max_merged_segment_size`/
/// `reclaim_weight`/`floor_segment_size` map straight onto
/// [`lucene_index::merge_policy::MergePolicyConfig`]'s corresponding fields.
/// `floor_segment_size` is real Lucene's `floorSegmentBytes`
/// (`setFloorSegmentMB`, default `16 * 1024 * 1024`): segments smaller than
/// this score as if they were exactly this size, so a large pile of
/// genuinely tiny segments doesn't get scored as disproportionately cheap
/// relative to each other.
///
/// **Every `MergePolicyConfig` knob is now a parameter** (M2 sweep b15,
/// closing the three that used to be silently defaulted here):
/// `force_merge_deletes_pct_allowed` (real Lucene's
/// `setForceMergeDeletesPctAllowed`, default `10.0`,
/// `find_forced_delete_merges`' threshold), `deletes_pct_allowed`
/// (`setDeletesPctAllowed`, default `20.0`) and `target_search_concurrency`
/// (`setTargetSearchConcurrency`, default `1`). Pass `10.0`, `20.0` and `1`
/// to reproduce Lucene's defaults exactly.
///
/// **Every knob is range-checked against `TieredMergePolicy`'s own setter,
/// and those ranges are not uniform** -- copying one to another is how this
/// function got it wrong before:
/// - `segments_per_tier` >= 2 (`setSegmentsPerTier`: `v < 2.0` throws)
/// - `force_merge_deletes_pct_allowed` in `0.0..=100.0`
///   (`setForceMergeDeletesPctAllowed`)
/// - `deletes_pct_allowed` in `(0.0, 50.0]` -- **not** `0..=100`:
///   `setDeletesPctAllowed` throws on `v <= 0 || v > 50`
/// - `target_search_concurrency` >= 1 (`setTargetSearchConcurrency`)
///
/// Each is [`FfiStatus::InvalidArgument`] carrying Java's own message text,
/// the C-ABI equivalent of the `IllegalArgumentException` Java throws.
///
/// **`reclaim_weight` changed meaning** in the M2 sweep: it used to scale an
/// invented linear `size * (1 - w * delRatio)` score; it is now the exponent
/// real Lucene hardcodes as `2` in `score()`'s `Math.pow(nonDelRatio, 2)`.
/// A JVM caller still passing its old value (commonly `1.0`) gets a weaker
/// delete-reclaim preference than Lucene's, not a broken one; pass `2.0` for
/// Lucene-identical scoring.
///
/// Ignored (but still validated as present) when `enabled == 0`.
///
/// Signature note: `floor_segment_size` was added as a new trailing
/// parameter (breaking this function's C signature) rather than kept
/// backward-compatible via a defaulted overload -- this crate has no
/// existing convention for versioned/overloaded FFI exports, and the only
/// callers of this function are in-repo tests, so a direct signature break
/// with updated call sites was simpler than introducing that pattern for a
/// single knob.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn ffi_writer_set_merge_policy(
    writer_handle: u64,
    enabled: u8,
    max_merge_at_once: u64,
    segments_per_tier: u64,
    max_merged_segment_size: u64,
    reclaim_weight: f64,
    floor_segment_size: u64,
    force_merge_deletes_pct_allowed: f64,
    deletes_pct_allowed: f64,
    target_search_concurrency: u64,
) -> i32 {
    guard(|| {
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_set_merge_policy: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let config = if enabled == 0 {
            None
        } else {
            // Each bound below is `TieredMergePolicy`'s own setter, verbatim
            // (`TieredMergePolicy.java`), message shape included -- they are
            // deliberately *not* a uniform "0..=100 percentage" rule, because
            // Java's three ranges genuinely differ.
            //
            // `setSegmentsPerTier`: `v < 2.0` -> throw.
            if segments_per_tier < 2 {
                set_last_error(format!(
                    "ffi_writer_set_merge_policy: segmentsPerTier must be >= 2.0 (got \
                     {segments_per_tier})"
                ));
                return Err(FfiStatus::InvalidArgument);
            }
            // `setForceMergeDeletesPctAllowed`: `v < 0.0 || v > 100.0` -> throw.
            if !(0.0..=100.0).contains(&force_merge_deletes_pct_allowed) {
                set_last_error(format!(
                    "ffi_writer_set_merge_policy: forceMergeDeletesPctAllowed must be between \
                     0.0 and 100.0 inclusive (got {force_merge_deletes_pct_allowed})"
                ));
                return Err(FfiStatus::InvalidArgument);
            }
            // `setDeletesPctAllowed`: `v <= 0 || v > 50` -> throw. Note the
            // asymmetry with the knob above: 0 is rejected and the ceiling is
            // 50, not 100.
            if !(deletes_pct_allowed > 0.0 && deletes_pct_allowed <= 50.0) {
                set_last_error(format!(
                    "ffi_writer_set_merge_policy: indexPctDeletedTarget must be > 0 and <= 50 \
                     (got {deletes_pct_allowed})"
                ));
                return Err(FfiStatus::InvalidArgument);
            }
            // `setTargetSearchConcurrency`: `< 1` -> throw.
            if target_search_concurrency == 0 {
                set_last_error(
                    "ffi_writer_set_merge_policy: targetSearchConcurrency must be >= 1 (got 0)",
                );
                return Err(FfiStatus::InvalidArgument);
            }
            Some(MergePolicyConfig {
                max_merge_at_once: max_merge_at_once as usize,
                segments_per_tier: segments_per_tier as usize,
                max_merged_segment_size,
                reclaim_weight,
                floor_segment_size,
                force_merge_deletes_pct_allowed,
                deletes_pct_allowed,
                target_search_concurrency: target_search_concurrency as usize,
                // `MergePolicy.keepFullyDeletedSegment`'s default, which
                // `TieredMergePolicy` inherits. It is not one of
                // `setTieredMergePolicy`'s knobs on the OpenSearch side and
                // there is no wire field for it; a caller that needs
                // `SoftDeletesRetentionMergePolicy`'s behaviour needs a new
                // entry point, not a silent default flip.
                keep_fully_deleted_segments: false,
            })
        };
        handle.writer.set_merge_policy(config);
        Ok(())
    })
}

/// Opts (`enabled != 0`) or opts out (`enabled == 0`) this writer into
/// building and writing real postings for one field -- see
/// [`IndexWriter::set_postings_field`]. `field_name`/`field_name_len` are
/// ignored when `enabled == 0`.
///
/// # Safety
/// `field_name` must be valid for reads of `field_name_len` bytes (or null
/// iff `field_name_len == 0`), same contract as [`str_from_raw`]. Ignored
/// entirely when `enabled == 0`.
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_set_postings_field(
    writer_handle: u64,
    enabled: u8,
    field_name: *const c_char,
    field_name_len: usize,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own caller contract.
        let name = unsafe { decode_optional_field_name(enabled, field_name, field_name_len)? };
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_set_postings_field: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        handle
            .writer
            .set_postings_field(name)
            .map_err(|e| map_writer_error("ffi_writer_set_postings_field", e))
    })
}

/// Opts (`enabled != 0`) or opts out (`enabled == 0`) this writer into
/// building and writing real term vectors for one field -- see
/// [`IndexWriter::set_term_vector_field`]. `field_name`/`field_name_len` are
/// ignored when `enabled == 0`.
///
/// # Safety
/// `field_name` must be valid for reads of `field_name_len` bytes (or null
/// iff `field_name_len == 0`), same contract as [`str_from_raw`]. Ignored
/// entirely when `enabled == 0`.
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_set_term_vector_field(
    writer_handle: u64,
    enabled: u8,
    field_name: *const c_char,
    field_name_len: usize,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own caller contract.
        let name = unsafe { decode_optional_field_name(enabled, field_name, field_name_len)? };
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_set_term_vector_field: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        handle
            .writer
            .set_term_vector_field(name)
            .map_err(|e| map_writer_error("ffi_writer_set_term_vector_field", e))
    })
}

/// Opts (`enabled != 0`) or opts out (`enabled == 0`) this writer into
/// building and writing real doc values for one field -- see
/// [`IndexWriter::set_doc_values_field`]. `field_name`/`field_name_len` are
/// ignored when `enabled == 0`.
///
/// # Safety
/// `field_name` must be valid for reads of `field_name_len` bytes (or null
/// iff `field_name_len == 0`), same contract as [`str_from_raw`]. Ignored
/// entirely when `enabled == 0`.
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_set_doc_values_field(
    writer_handle: u64,
    enabled: u8,
    field_name: *const c_char,
    field_name_len: usize,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own caller contract.
        let name = unsafe { decode_optional_field_name(enabled, field_name, field_name_len)? };
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_set_doc_values_field: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        handle
            .writer
            .set_doc_values_field(name)
            .map_err(|e| map_writer_error("ffi_writer_set_doc_values_field", e))
    })
}

/// The atomic delete-by-term + add-document real Lucene calls
/// `updateDocument` -- see [`IndexWriter::update_document`]. `field_name`/
/// `field_name_len` and `term_ptr`/`term_len` identify the term to delete:
/// raw, already-analyzed bytes (e.g. lowercase) exactly as this writer's own
/// postings would have indexed them -- this FFI boundary performs no
/// analysis of its own, same as every other raw-bytes term this crate's
/// `query.rs` already accepts. The replacement document's fields are
/// described by the same four parallel arrays [`ffi_writer_add_document`]
/// uses (see its own doc comment, and [`decode_field_value`]'s, for the six
/// `field_kinds` values understood).
///
/// Delete resolution only reaches segments that already have a `.tim` file on
/// disk: a segment this writer flushed with no
/// [`ffi_writer_set_postings_field`] enabled at commit time has no term
/// dictionary to search, so it contributes no matches.
///
/// **Buffered**, exactly like [`ffi_writer_add_document`] and exactly like
/// Java's `IndexWriter.updateDocument`: the delete and the add both take
/// effect at the next [`ffi_writer_commit`], not at this call. The delete
/// carries the buffer position it was issued at, so it reaches every document
/// added before it and none added after -- including the replacement.
///
/// # Safety
/// `field_name` must be valid for reads of `field_name_len` bytes. `term_ptr`
/// must be valid for reads of `term_len` bytes (or null iff `term_len == 0`).
/// The four document-field arrays follow [`ffi_writer_add_document`]'s exact
/// same safety contract.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_writer_update_document(
    writer_handle: u64,
    field_name: *const c_char,
    field_name_len: usize,
    term_ptr: *const u8,
    term_len: usize,
    field_numbers: *const i32,
    field_kinds: *const u8,
    field_value_ptrs: *const *const u8,
    field_value_lens: *const usize,
    field_count: usize,
    out_seq_no: *mut i64,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract guarantees `field_name` is valid for
        // `field_name_len` bytes.
        let field = unsafe { str_from_raw(field_name, field_name_len)? };
        // SAFETY: caller contract guarantees `term_ptr` is valid for
        // `term_len` bytes (or null iff `term_len == 0`).
        let term = unsafe { bytes_from_raw(term_ptr, term_len)? };

        let mut new_fields = try_with_capacity(field_count)?;
        if field_count > 0 {
            if field_numbers.is_null()
                || field_kinds.is_null()
                || field_value_ptrs.is_null()
                || field_value_lens.is_null()
            {
                return Err(FfiStatus::NullPointer);
            }
            // SAFETY: caller contract guarantees each array is valid for
            // `field_count` elements.
            let (numbers, kinds, value_ptrs, value_lens) = unsafe {
                (
                    std::slice::from_raw_parts(field_numbers, field_count),
                    std::slice::from_raw_parts(field_kinds, field_count),
                    std::slice::from_raw_parts(field_value_ptrs, field_count),
                    std::slice::from_raw_parts(field_value_lens, field_count),
                )
            };
            for i in 0..field_count {
                // SAFETY: caller contract guarantees `value_ptrs[i]` is
                // valid for `value_lens[i]` bytes.
                let bytes = unsafe { bytes_from_raw(value_ptrs[i], value_lens[i])? };
                let value = decode_field_value(kinds[i], bytes)?;
                new_fields.push(StoredField {
                    field_number: numbers[i],
                    value,
                });
            }
        }

        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_update_document: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;

        let seq_no = handle
            .writer
            .update_document(Term::new(field, term), Document { fields: new_fields })
            .map_err(|e| map_writer_error("ffi_writer_update_document", e))?;
        // SAFETY: caller contract: `out_seq_no` is valid for one `i64` write
        // or null.
        unsafe { write_seq_no(out_seq_no, seq_no) };
        Ok(())
    })
}

/// Buffers a delete for every live doc matching `(field_name, term)` -- see
/// [`IndexWriter::delete_documents_by_term`]. `field_name`/`field_name_len`
/// and `term_ptr`/`term_len` use the exact same raw-bytes term convention as
/// [`ffi_writer_update_document`] (see its own doc comment); delete resolution
/// likewise only reaches segments with a `.tim` file already on disk.
///
/// **Buffered**, same as [`ffi_writer_update_document`]: the deletion becomes
/// visible at the next [`ffi_writer_commit`], not at this call.
///
/// # Safety
/// `field_name` must be valid for reads of `field_name_len` bytes. `term_ptr`
/// must be valid for reads of `term_len` bytes (or null iff `term_len == 0`).
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_delete_documents(
    writer_handle: u64,
    field_name: *const c_char,
    field_name_len: usize,
    term_ptr: *const u8,
    term_len: usize,
    out_seq_no: *mut i64,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract guarantees `field_name` is valid for
        // `field_name_len` bytes.
        let field = unsafe { str_from_raw(field_name, field_name_len)? };
        // SAFETY: caller contract guarantees `term_ptr` is valid for
        // `term_len` bytes (or null iff `term_len == 0`).
        let term = unsafe { bytes_from_raw(term_ptr, term_len)? };

        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_delete_documents: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;

        let seq_no = handle
            .writer
            .delete_documents_by_term(&[Term::new(field, term)])
            .map_err(|e| map_writer_error("ffi_writer_delete_documents", e))?;
        // SAFETY: caller contract: `out_seq_no` is valid for one `i64` write
        // or null.
        unsafe { write_seq_no(out_seq_no, seq_no) };
        Ok(())
    })
}

/// Writes the number of segments in `writer_handle`'s current committed
/// [`IndexWriter::segment_infos`] to `*out_len` -- call before looping
/// [`ffi_writer_segment_info_name`] over `0..len`, the same "length first"
/// shape [`crate::results_fragments::ffi_fragment_results_len`] establishes
/// for its own per-index accessor. Reflects only already-`commit()`ed
/// segments -- not-yet-flushed [`ffi_writer_add_document`] calls are not
/// counted here (see [`ffi_writer_pending_doc_count`] for those).
///
/// # Safety
/// `out_len` must be valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_segment_infos_len(
    writer_handle: u64,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = read_recovering(writers());
        let handle = registry.get(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_segment_infos_len: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        // SAFETY: caller contract guarantees `out_len` is valid for one write.
        unsafe {
            *out_len = handle.writer.segment_infos().segments.len();
        }
        Ok(())
    })
}

/// Copies segment index `index`'s name (e.g. `"_0"`) from `writer_handle`'s
/// current committed [`IndexWriter::segment_infos`] into `buf`
/// (caller-allocated, `buf_len` bytes), NUL-terminated, writing the number of
/// bytes written (excluding the NUL) to `*out_written` -- same
/// `buf`/`buf_len`/`out_written`/`BufferTooSmall` contract as
/// [`crate::ffi_get_last_error_message`]. Returns
/// [`FfiStatus::IndexOutOfBounds`] for `index >= ` [`ffi_writer_segment_infos_len`].
///
/// # Safety
/// `buf` must be valid for writes of `buf_len` bytes; `out_written` must be
/// valid for one `usize` write, or null.
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_segment_info_name(
    writer_handle: u64,
    index: usize,
    buf: *mut c_char,
    buf_len: usize,
    out_written: *mut usize,
) -> i32 {
    guard(|| {
        let registry = read_recovering(writers());
        let handle = registry.get(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_segment_info_name: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let segments = &handle.writer.segment_infos().segments;
        let segment = segments.get(index).ok_or_else(|| {
            set_last_error(format!(
                "ffi_writer_segment_info_name: index {index} out of bounds (len {})",
                segments.len()
            ));
            FfiStatus::IndexOutOfBounds
        })?;
        let bytes = segment.segment_name.as_bytes();
        if bytes.len() + 1 > buf_len {
            return Err(FfiStatus::BufferTooSmall);
        }
        if buf.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `buf` is valid for `buf_len`
        // bytes, and `bytes.len() + 1 <= buf_len` was just checked above.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, bytes.len());
            *buf.add(bytes.len()) = 0;
        }
        if !out_written.is_null() {
            // SAFETY: caller contract guarantees `out_written` is valid for
            // one write.
            unsafe {
                *out_written = bytes.len();
            }
        }
        Ok(())
    })
}

/// Writes the number of documents buffered by [`ffi_writer_add_document`] but
/// not yet written to disk by [`ffi_writer_commit`] to `*out_len` -- see
/// [`IndexWriter::pending_doc_count`].
///
/// # Safety
/// `out_len` must be valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_pending_doc_count(
    writer_handle: u64,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = read_recovering(writers());
        let handle = registry.get(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_pending_doc_count: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        // SAFETY: caller contract guarantees `out_len` is valid for one write.
        unsafe {
            *out_len = handle.writer.pending_doc_count();
        }
        Ok(())
    })
}

/// Writes the total number of documents actually committed to disk right
/// now (summed across every segment in [`IndexWriter::segment_infos`]) to
/// `*out_len` -- see [`IndexWriter::committed_doc_count`] for the exact
/// **total, not live** semantics (a deleted-but-not-yet-merged document is
/// still counted). Distinct from [`ffi_writer_pending_doc_count`], which
/// counts buffered-but-not-yet-`commit()`ed documents instead.
///
/// # Safety
/// `out_len` must be valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_committed_doc_count(
    writer_handle: u64,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = read_recovering(writers());
        let handle = registry.get(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_committed_doc_count: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let count = handle
            .writer
            .committed_doc_count()
            .map_err(|e| map_writer_error("ffi_writer_committed_doc_count", e))?;
        // SAFETY: caller contract guarantees `out_len` is valid for one write.
        unsafe {
            *out_len = count;
        }
        Ok(())
    })
}

/// Drops every buffered and every committed document from this writer --
/// [`IndexWriter::delete_all`], the port of Java's `IndexWriter.deleteAll()`.
/// Like Java's, the change is in-memory until the next
/// [`ffi_writer_commit`], and refused ([`FfiStatus::InvalidArgument`] via
/// `map_writer_error`) while a [`ffi_writer_prepare_commit`] is outstanding.
#[no_mangle]
pub extern "C" fn ffi_writer_delete_all(writer_handle: u64) -> i32 {
    guard(|| {
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_delete_all: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        handle
            .writer
            .delete_all()
            .map_err(|e| map_writer_error("ffi_writer_delete_all", e))
    })
}

/// Sets this writer's live commit data ([`IndexWriter::set_live_commit_data`],
/// the port of `IndexWriter.setLiveCommitData`) -- opaque caller metadata
/// carried verbatim into every `segments_N` this writer writes from now on.
///
/// `count` `(key, value)` pairs arrive as four parallel flat arrays
/// (`keys`/`key_lens`/`values`/`value_lens`), the same wire convention
/// [`crate::query::ffi_search_boolean_query`]'s clause lists use: entry `i`
/// is `(keys[i][..key_lens[i]], values[i][..value_lens[i]])`, each UTF-8.
/// `count == 0` clears the data (the four array pointers may then be null),
/// matching `setLiveCommitData(Collections.emptyList())`.
///
/// Order is preserved as given; duplicate keys are the caller's business,
/// exactly as in Java (`SegmentInfos.userData` is written as supplied).
///
/// # Safety
/// When `count > 0`, `keys`/`key_lens`/`values`/`value_lens` must each be
/// valid for reads of `count` elements, with every `keys[i]` valid for
/// `key_lens[i]` bytes and every `values[i]` valid for `value_lens[i]`
/// bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_set_live_commit_data(
    writer_handle: u64,
    keys: *const *const c_char,
    key_lens: *const usize,
    values: *const *const c_char,
    value_lens: *const usize,
    count: usize,
) -> i32 {
    guard(|| {
        let mut data: Vec<(String, String)> = try_with_capacity(count)?;
        if count > 0 {
            if keys.is_null() || key_lens.is_null() || values.is_null() || value_lens.is_null() {
                set_last_error(
                    "ffi_writer_set_live_commit_data: null key/value array with count > 0",
                );
                return Err(FfiStatus::NullPointer);
            }
            for i in 0..count {
                // SAFETY: caller contract guarantees each array is valid for
                // `count` elements and each element pair for its own length.
                let (key, value) = unsafe {
                    (
                        str_from_raw(*keys.add(i), *key_lens.add(i))?,
                        str_from_raw(*values.add(i), *value_lens.add(i))?,
                    )
                };
                data.push((key.to_string(), value.to_string()));
            }
        }
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_set_live_commit_data: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        handle.writer.set_live_commit_data(data);
        Ok(())
    })
}

/// Writes the number of live-commit-data entries this writer currently
/// carries to `*out_len` ([`IndexWriter::live_commit_data`], the port of
/// `IndexWriter.getLiveCommitData()`) -- call before looping
/// [`ffi_writer_live_commit_data_entry`] over `0..len`, the same "length
/// first" shape [`ffi_writer_segment_infos_len`] uses.
///
/// Reflects both what [`ffi_writer_set_live_commit_data`] last set *and*
/// what the commit resumed by [`ffi_open_writer`] carried, exactly as
/// `getLiveCommitData()` does.
///
/// # Safety
/// `out_len` must be valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_live_commit_data_len(
    writer_handle: u64,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = read_recovering(writers());
        let handle = registry.get(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_live_commit_data_len: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        // SAFETY: caller contract guarantees `out_len` is valid for one write.
        unsafe {
            *out_len = handle.writer.live_commit_data().len();
        }
        Ok(())
    })
}

/// Copies live-commit-data entry `index`'s key and value into two
/// caller-allocated buffers, NUL-terminated, writing each length (excluding
/// the NUL) to `*out_key_written`/`*out_value_written` -- same
/// `buf`/`buf_len`/`out_written`/[`FfiStatus::BufferTooSmall`] contract as
/// [`ffi_writer_segment_info_name`], applied to both halves at once so one
/// entry needs one call rather than two.
///
/// Neither buffer is written unless *both* are large enough, so a
/// `BufferTooSmall` never leaves the caller with a half-copied entry.
/// Returns [`FfiStatus::IndexOutOfBounds`] for
/// `index >= ` [`ffi_writer_live_commit_data_len`].
///
/// # Safety
/// `key_buf` must be valid for writes of `key_buf_len` bytes and
/// `value_buf` for `value_buf_len` bytes; `out_key_written`/
/// `out_value_written` must each be valid for one `usize` write, or null.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_writer_live_commit_data_entry(
    writer_handle: u64,
    index: usize,
    key_buf: *mut c_char,
    key_buf_len: usize,
    out_key_written: *mut usize,
    value_buf: *mut c_char,
    value_buf_len: usize,
    out_value_written: *mut usize,
) -> i32 {
    guard(|| {
        let registry = read_recovering(writers());
        let handle = registry.get(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_live_commit_data_entry: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let data = handle.writer.live_commit_data();
        let (key, value) = data.get(index).ok_or_else(|| {
            set_last_error(format!(
                "ffi_writer_live_commit_data_entry: index {index} out of bounds (len {})",
                data.len()
            ));
            FfiStatus::IndexOutOfBounds
        })?;
        let (key, value) = (key.as_bytes(), value.as_bytes());
        // Both checked before either is written -- see this function's doc
        // comment.
        if key.len() + 1 > key_buf_len || value.len() + 1 > value_buf_len {
            set_last_error(format!(
                "ffi_writer_live_commit_data_entry: need {}/{} bytes for key/value, got {}/{}",
                key.len() + 1,
                value.len() + 1,
                key_buf_len,
                value_buf_len
            ));
            return Err(FfiStatus::BufferTooSmall);
        }
        if key_buf.is_null() || value_buf.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees both buffers are valid for their
        // paired lengths, and both fit (checked immediately above).
        unsafe {
            std::ptr::copy_nonoverlapping(key.as_ptr().cast::<c_char>(), key_buf, key.len());
            *key_buf.add(key.len()) = 0;
            std::ptr::copy_nonoverlapping(value.as_ptr().cast::<c_char>(), value_buf, value.len());
            *value_buf.add(value.len()) = 0;
        }
        if !out_key_written.is_null() {
            // SAFETY: caller contract guarantees one valid `usize` write.
            unsafe {
                *out_key_written = key.len();
            }
        }
        if !out_value_written.is_null() {
            // SAFETY: caller contract guarantees one valid `usize` write.
            unsafe {
                *out_value_written = value.len();
            }
        }
        Ok(())
    })
}

/// `IndexWriter.addPostingsField`-equivalent: opts **one more** field into
/// real postings, alongside whatever [`ffi_writer_set_postings_field`]
/// already selected, instead of replacing it. Wraps
/// [`IndexWriter::add_postings_field`].
///
/// Recorded as unexposed by `b15-ffi-core`: without it a writer could index
/// exactly one postings field, so a JVM caller could not build a
/// multi-searchable-field index at all.
///
/// # Safety
/// `field_name` must be valid for reads of `field_name_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_add_postings_field(
    writer_handle: u64,
    field_name: *const c_char,
    field_name_len: usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract guarantees `field_name` is valid for
        // `field_name_len` bytes.
        let name = unsafe { str_from_raw(field_name, field_name_len)? };
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_add_postings_field: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        handle
            .writer
            .add_postings_field(name)
            .map_err(|e| map_writer_error("ffi_writer_add_postings_field", e))
    })
}

/// [`ffi_writer_add_postings_field`]'s term-vector twin, wrapping
/// [`IndexWriter::add_term_vector_field`]: opts one more field into term
/// vectors alongside whatever [`ffi_writer_set_term_vector_field`] selected.
///
/// # Safety
/// `field_name` must be valid for reads of `field_name_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_add_term_vector_field(
    writer_handle: u64,
    field_name: *const c_char,
    field_name_len: usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract guarantees `field_name` is valid for
        // `field_name_len` bytes.
        let name = unsafe { str_from_raw(field_name, field_name_len)? };
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_add_term_vector_field: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        handle
            .writer
            .add_term_vector_field(name)
            .map_err(|e| map_writer_error("ffi_writer_add_term_vector_field", e))
    })
}

/// Opts (`enabled != 0`) or out (`enabled == 0`) of writing this field's
/// postings with `IndexOptions.DOCS_AND_FREQS` where the freq is the opaque
/// per-`(doc, term)` value a custom similarity interprets -- Lucene's
/// `DocsAndCustomFreqs`. Wraps
/// [`IndexWriter::set_custom_freq_postings_field`]; the per-document freq
/// values are supplied by
/// [`ffi_writer_add_document_with_custom_freq_terms`].
///
/// Mutually exclusive with [`ffi_writer_set_postings_field`]/
/// [`ffi_writer_add_postings_field`] -- the writer refuses both at once, and
/// that refusal surfaces here as [`FfiStatus::InvalidArgument`].
///
/// # Safety
/// `field_name` must be valid for reads of `field_name_len` bytes (or null
/// iff `field_name_len == 0`). Ignored entirely when `enabled == 0`.
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_set_custom_freq_postings_field(
    writer_handle: u64,
    enabled: u8,
    field_name: *const c_char,
    field_name_len: usize,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own caller contract.
        let name = unsafe { decode_optional_field_name(enabled, field_name, field_name_len)? };
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error(
                "ffi_writer_set_custom_freq_postings_field: unknown or already-closed handle",
            );
            FfiStatus::InvalidHandle
        })?;
        handle
            .writer
            .set_custom_freq_postings_field(name)
            .map_err(|e| map_writer_error("ffi_writer_set_custom_freq_postings_field", e))
    })
}

/// `FieldType.setOmitNorms(true)` for one field: no segment this writer
/// flushes from here on carries a norm column for it, and its `.fnm` says so.
/// Wraps [`IndexWriter::omit_norms_field`].
///
/// This replaced `ffi_writer_set_norms_field` in c35. That call was an
/// opt-*in*, which is not a knob Lucene has: Java writes norms for every
/// indexed field whose `omitNorms` is false, so a caller that did not name a
/// field got length-unnormalised BM25 for it. Norms are now on by default and
/// this is the only norms call, in the direction Lucene actually offers.
///
/// # Safety
/// `field_name` must be valid for reads of `field_name_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_writer_omit_norms_field(
    writer_handle: u64,
    field_name: *const c_char,
    field_name_len: usize,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own caller contract.
        let name = unsafe { decode_optional_field_name(1, field_name, field_name_len)? };
        let name = name.ok_or_else(|| {
            set_last_error("ffi_writer_omit_norms_field: field_name must not be null");
            FfiStatus::InvalidArgument
        })?;
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_omit_norms_field: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        handle
            .writer
            .omit_norms_field(name)
            .map_err(|e| map_writer_error("ffi_writer_omit_norms_field", e))
    })
}

/// Writes `seq_no` to `out` unless `out` is null.
///
/// **Why every mutating writer entry point has one now** (M2 sweep batch
/// `c13-ffi-surface`): real Lucene's `IndexWriter` returns a `long`
/// sequence number from `addDocument`/`updateDocument`/`deleteDocuments`/
/// `softUpdateDocument`/`updateDocValues`, and callers use it -- it is how a
/// caller knows whether a `DirectoryReader` it holds already reflects an
/// operation (`DirectoryReader.openIfChanged` + `IndexWriter.getMaxCompletedSequenceNumber`)
/// and how OpenSearch's translog orders replicated operations. `c7-delete-queue`
/// landed the real `DocumentsWriterDeleteQueue` and the seqNo it produces, and
/// recorded (its finding A7) that the number was still being dropped on the
/// floor at this boundary. It is a `*mut i64` out-parameter rather than a
/// return value because every exported function in this crate returns its
/// [`FfiStatus`] code; null means "the caller does not want it", which keeps
/// the seqNo free for callers that do not track it.
///
/// # Safety
/// `out` must be valid for one `i64` write, or null.
unsafe fn write_seq_no(out: *mut i64, seq_no: i64) {
    if !out.is_null() {
        // SAFETY: this function's own contract.
        unsafe {
            *out = seq_no;
        }
    }
}

/// Decodes one document's fields from the parallel-array encoding
/// [`ffi_writer_add_document`] documents, reading `field_count` elements
/// starting at `offset` in each array.
///
/// # Safety
/// Each array must be valid for reads of `offset + field_count` elements (or
/// null when `field_count == 0`); each `field_value_ptrs[i]` must be valid
/// for `field_value_lens[i]` bytes.
unsafe fn decode_document_at(
    field_numbers: *const i32,
    field_kinds: *const u8,
    field_value_ptrs: *const *const u8,
    field_value_lens: *const usize,
    offset: usize,
    field_count: usize,
) -> Result<Document, FfiStatus> {
    let mut fields = try_with_capacity(field_count)?;
    if field_count > 0 {
        if field_numbers.is_null()
            || field_kinds.is_null()
            || field_value_ptrs.is_null()
            || field_value_lens.is_null()
        {
            return Err(FfiStatus::NullPointer);
        }
        for i in offset..offset + field_count {
            // SAFETY: this function's own contract.
            let (number, kind, value_ptr, value_len) = unsafe {
                (
                    *field_numbers.add(i),
                    *field_kinds.add(i),
                    *field_value_ptrs.add(i),
                    *field_value_lens.add(i),
                )
            };
            // SAFETY: this function's own contract.
            let bytes = unsafe { bytes_from_raw(value_ptr, value_len)? };
            fields.push(StoredField {
                field_number: number,
                value: decode_field_value(kind, bytes)?,
            });
        }
    }
    Ok(Document { fields })
}

/// Decodes a whole *block* of documents: `doc_field_counts[d]` fields for
/// document `d`, laid end to end in the same four parallel arrays
/// [`ffi_writer_add_document`] uses for one document.
///
/// # Safety
/// `doc_field_counts` must be valid for `doc_count` reads; the four field
/// arrays must be valid for `sum(doc_field_counts)` reads each.
unsafe fn decode_document_block(
    doc_field_counts: *const usize,
    doc_count: usize,
    field_numbers: *const i32,
    field_kinds: *const u8,
    field_value_ptrs: *const *const u8,
    field_value_lens: *const usize,
) -> Result<Vec<Document>, FfiStatus> {
    if doc_count == 0 {
        return Ok(Vec::new());
    }
    if doc_field_counts.is_null() {
        return Err(FfiStatus::NullPointer);
    }
    // SAFETY: caller contract guarantees `doc_count` readable elements.
    let counts = unsafe { std::slice::from_raw_parts(doc_field_counts, doc_count) };
    let mut docs = try_with_capacity(doc_count)?;
    let mut offset = 0usize;
    for count in counts {
        // SAFETY: forwarded from this function's contract; `offset` is the
        // sum of every preceding count, which the `checked_add` below keeps
        // from wrapping -- a wrapped offset in release would make the reads
        // in `decode_document_at` land outside the four field arrays.
        docs.push(unsafe {
            decode_document_at(
                field_numbers,
                field_kinds,
                field_value_ptrs,
                field_value_lens,
                offset,
                *count,
            )?
        });
        offset = offset.checked_add(*count).ok_or_else(|| {
            set_last_error("document block field counts overflow usize");
            FfiStatus::InvalidArgument
        })?;
    }
    Ok(docs)
}

/// Adds a whole document **block** atomically -- Java's
/// `IndexWriter.addDocuments(Iterable<Iterable<IndexableField>>)`.
///
/// A block is Lucene's parent/child (nested-document) primitive: the
/// documents land contiguously in one segment, in this order, with the last
/// one the parent, and no flush ever splits them. Unblocked by
/// `c7-delete-queue`, which gave `IndexWriter` the buffer-position bookkeeping
/// (`pending_has_blocks`) a block needs.
///
/// `doc_field_counts` has `doc_count` entries; document `d` owns
/// `doc_field_counts[d]` consecutive entries of the same four parallel field
/// arrays [`ffi_writer_add_document`] takes, laid end to end. One sequence
/// number is written to `*out_seq_no` for the whole block, as Java returns
/// one for the whole call.
///
/// # Safety
/// `doc_field_counts` must be valid for `doc_count` reads; the four field
/// arrays must each be valid for `sum(doc_field_counts)` reads, with every
/// `field_value_ptrs[i]` valid for `field_value_lens[i]` bytes; `out_seq_no`
/// must be valid for one `i64` write, or null.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_writer_add_documents(
    writer_handle: u64,
    doc_field_counts: *const usize,
    doc_count: usize,
    field_numbers: *const i32,
    field_kinds: *const u8,
    field_value_ptrs: *const *const u8,
    field_value_lens: *const usize,
    out_seq_no: *mut i64,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own caller contract.
        let docs = unsafe {
            decode_document_block(
                doc_field_counts,
                doc_count,
                field_numbers,
                field_kinds,
                field_value_ptrs,
                field_value_lens,
            )?
        };
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_add_documents: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let seq_no = handle
            .writer
            .add_documents(docs)
            .map_err(|e| map_writer_error("ffi_writer_add_documents", e))?;
        // SAFETY: caller contract.
        unsafe { write_seq_no(out_seq_no, seq_no) };
        Ok(())
    })
}

/// `IndexWriter.updateDocuments(Term, Iterable<Iterable<IndexableField>>)`:
/// [`ffi_writer_add_documents`] plus an atomic delete-by-term, sharing one
/// sequence number so the delete and the block can never be observed apart.
/// The term is raw, already-analyzed bytes, same convention as
/// [`ffi_writer_update_document`].
///
/// # Safety
/// `field_name`/`term_ptr` must be valid for their paired lengths; the block
/// arrays follow [`ffi_writer_add_documents`]'s contract exactly; `out_seq_no`
/// must be valid for one `i64` write, or null.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_writer_update_documents(
    writer_handle: u64,
    field_name: *const c_char,
    field_name_len: usize,
    term_ptr: *const u8,
    term_len: usize,
    doc_field_counts: *const usize,
    doc_count: usize,
    field_numbers: *const i32,
    field_kinds: *const u8,
    field_value_ptrs: *const *const u8,
    field_value_lens: *const usize,
    out_seq_no: *mut i64,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own caller contract.
        let (field, term, docs) = unsafe {
            (
                str_from_raw(field_name, field_name_len)?,
                bytes_from_raw(term_ptr, term_len)?,
                decode_document_block(
                    doc_field_counts,
                    doc_count,
                    field_numbers,
                    field_kinds,
                    field_value_ptrs,
                    field_value_lens,
                )?,
            )
        };
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_update_documents: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let seq_no = handle
            .writer
            .update_documents(Term::new(field, term), docs)
            .map_err(|e| map_writer_error("ffi_writer_update_documents", e))?;
        // SAFETY: caller contract.
        unsafe { write_seq_no(out_seq_no, seq_no) };
        Ok(())
    })
}

/// `IndexWriter.softUpdateDocument(Term, doc, Field... softDeletes)`: adds
/// `doc` and, instead of *deleting* the documents matching the term, marks
/// them through a numeric doc-values field -- the whole soft-delete
/// mechanism, unblocked by `c7-delete-queue`.
///
/// `soft_delete_field`/`soft_delete_value` is the single
/// `NumericDocValuesField` Java's callers pass (`new
/// NumericDocValuesField(softDeletesField, 1)`); Java's *"at least one soft
/// delete must be present"* check is what refuses a null/empty field name
/// here, surfaced as [`FfiStatus::InvalidArgument`].
///
/// # Safety
/// `term_field_name`/`term_ptr`/`soft_delete_field` must be valid for their
/// paired lengths; the document field arrays follow
/// [`ffi_writer_add_document`]'s contract; `out_seq_no` must be valid for one
/// `i64` write, or null.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_writer_soft_update_document(
    writer_handle: u64,
    term_field_name: *const c_char,
    term_field_name_len: usize,
    term_ptr: *const u8,
    term_len: usize,
    soft_delete_field: *const c_char,
    soft_delete_field_len: usize,
    soft_delete_value: i64,
    field_numbers: *const i32,
    field_kinds: *const u8,
    field_value_ptrs: *const *const u8,
    field_value_lens: *const usize,
    field_count: usize,
    out_seq_no: *mut i64,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own caller contract.
        let (term_field, term, soft_field, doc) = unsafe {
            (
                str_from_raw(term_field_name, term_field_name_len)?,
                bytes_from_raw(term_ptr, term_len)?,
                str_from_raw(soft_delete_field, soft_delete_field_len)?,
                decode_document_at(
                    field_numbers,
                    field_kinds,
                    field_value_ptrs,
                    field_value_lens,
                    0,
                    field_count,
                )?,
            )
        };
        if soft_field.is_empty() {
            set_last_error("at least one soft delete must be present");
            return Err(FfiStatus::InvalidArgument);
        }
        let term = Term::new(term_field, term);
        let update = DocValuesUpdate::Numeric {
            term: term.clone(),
            field: soft_field.to_string(),
            value: Some(soft_delete_value),
        };
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_soft_update_document: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let seq_no = handle
            .writer
            .soft_update_document(term, doc, &[update])
            .map_err(|e| map_writer_error("ffi_writer_soft_update_document", e))?;
        // SAFETY: caller contract.
        unsafe { write_seq_no(out_seq_no, seq_no) };
        Ok(())
    })
}

/// `IndexWriter.updateNumericDocValue(Term, String field, long value)`:
/// sets `field`'s NUMERIC doc-values value to `value` on every document
/// matching the term, without reindexing them. Unblocked by
/// `c7-delete-queue`'s `DocValuesUpdate` plumbing.
///
/// # Safety
/// `term_field_name`/`term_ptr`/`dv_field_name` must be valid for their
/// paired lengths; `out_seq_no` must be valid for one `i64` write, or null.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_writer_update_numeric_doc_value(
    writer_handle: u64,
    term_field_name: *const c_char,
    term_field_name_len: usize,
    term_ptr: *const u8,
    term_len: usize,
    dv_field_name: *const c_char,
    dv_field_name_len: usize,
    value: i64,
    out_seq_no: *mut i64,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own caller contract.
        let (term_field, term, dv_field) = unsafe {
            (
                str_from_raw(term_field_name, term_field_name_len)?,
                bytes_from_raw(term_ptr, term_len)?,
                str_from_raw(dv_field_name, dv_field_name_len)?,
            )
        };
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_update_numeric_doc_value: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let seq_no = handle
            .writer
            .update_numeric_doc_value(Term::new(term_field, term), dv_field, value)
            .map_err(|e| map_writer_error("ffi_writer_update_numeric_doc_value", e))?;
        // SAFETY: caller contract.
        unsafe { write_seq_no(out_seq_no, seq_no) };
        Ok(())
    })
}

/// `IndexWriter.updateBinaryDocValue(Term, String field, BytesRef value)`.
/// See [`ffi_writer_update_numeric_doc_value`].
///
/// # Safety
/// `term_field_name`/`term_ptr`/`dv_field_name`/`value_ptr` must be valid for
/// their paired lengths; `out_seq_no` must be valid for one `i64` write, or
/// null.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_writer_update_binary_doc_value(
    writer_handle: u64,
    term_field_name: *const c_char,
    term_field_name_len: usize,
    term_ptr: *const u8,
    term_len: usize,
    dv_field_name: *const c_char,
    dv_field_name_len: usize,
    value_ptr: *const u8,
    value_len: usize,
    out_seq_no: *mut i64,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own caller contract.
        let (term_field, term, dv_field, value) = unsafe {
            (
                str_from_raw(term_field_name, term_field_name_len)?,
                bytes_from_raw(term_ptr, term_len)?,
                str_from_raw(dv_field_name, dv_field_name_len)?,
                bytes_from_raw(value_ptr, value_len)?,
            )
        };
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_update_binary_doc_value: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let seq_no = handle
            .writer
            .update_binary_doc_value(Term::new(term_field, term), dv_field, value)
            .map_err(|e| map_writer_error("ffi_writer_update_binary_doc_value", e))?;
        // SAFETY: caller contract.
        unsafe { write_seq_no(out_seq_no, seq_no) };
        Ok(())
    })
}

/// [`ffi_writer_add_document`] with an accompanying list of `(term, freq)`
/// pairs for the field [`ffi_writer_set_custom_freq_postings_field`] opted
/// in -- wraps [`IndexWriter::add_document_with_custom_freq_terms`].
///
/// `custom_freq_terms`/`custom_freq_term_lens`/`custom_freqs` are three
/// parallel arrays of `custom_freq_term_count` elements: element `i` is the
/// term `custom_freq_terms[i][..custom_freq_term_lens[i]]` with freq
/// `custom_freqs[i]`. The term bytes are raw and already analyzed, the
/// convention every term in this crate follows. Each freq must be `>= 1`
/// (the postings wire format has no encoding for zero or negative), which
/// the codec layer enforces at flush time and this boundary enforces up
/// front so a bad value is an `InvalidArgument` at the call that caused it
/// rather than a flush error many documents later.
///
/// # Safety
/// The four document field arrays follow [`ffi_writer_add_document`]'s
/// contract; the three custom-freq arrays must each be valid for
/// `custom_freq_term_count` reads, with every `custom_freq_terms[i]` valid
/// for `custom_freq_term_lens[i]` bytes; `out_seq_no` must be valid for one
/// `i64` write, or null.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_writer_add_document_with_custom_freq_terms(
    writer_handle: u64,
    field_numbers: *const i32,
    field_kinds: *const u8,
    field_value_ptrs: *const *const u8,
    field_value_lens: *const usize,
    field_count: usize,
    custom_freq_terms: *const *const u8,
    custom_freq_term_lens: *const usize,
    custom_freqs: *const i32,
    custom_freq_term_count: usize,
    out_seq_no: *mut i64,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own caller contract.
        let doc = unsafe {
            decode_document_at(
                field_numbers,
                field_kinds,
                field_value_ptrs,
                field_value_lens,
                0,
                field_count,
            )?
        };
        let mut terms: Vec<(String, i32)> = try_with_capacity(custom_freq_term_count)?;
        if custom_freq_term_count > 0 {
            if custom_freq_terms.is_null()
                || custom_freq_term_lens.is_null()
                || custom_freqs.is_null()
            {
                return Err(FfiStatus::NullPointer);
            }
            for i in 0..custom_freq_term_count {
                // SAFETY: caller contract guarantees each array is valid for
                // `custom_freq_term_count` elements, and each term pointer for
                // its paired length.
                let (term, freq) = unsafe {
                    (
                        str_from_raw(
                            (*custom_freq_terms.add(i)).cast::<c_char>(),
                            *custom_freq_term_lens.add(i),
                        )?,
                        *custom_freqs.add(i),
                    )
                };
                if freq < 1 {
                    set_last_error(format!(
                        "custom freq {freq} for term {term:?} is below 1 (the postings wire \
                         format has no encoding for a zero or negative freq)"
                    ));
                    return Err(FfiStatus::InvalidArgument);
                }
                terms.push((term.to_string(), freq));
            }
        }
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error(
                "ffi_writer_add_document_with_custom_freq_terms: unknown or already-closed handle",
            );
            FfiStatus::InvalidHandle
        })?;
        let seq_no = handle
            .writer
            .add_document_with_custom_freq_terms(doc, terms)
            .map_err(|e| map_writer_error("ffi_writer_add_document_with_custom_freq_terms", e))?;
        // SAFETY: caller contract.
        unsafe { write_seq_no(out_seq_no, seq_no) };
        Ok(())
    })
}

/// A [`DeleteQuery::Term`] node.
pub(crate) const DELETE_QUERY_TERM: u8 = 0;
/// A [`DeleteQuery::Prefix`] node: `lower` is the prefix, `upper` unused.
pub(crate) const DELETE_QUERY_PREFIX: u8 = 1;
/// A [`DeleteQuery::TermRange`] node: `lower`/`upper` are the bounds, with
/// [`DELETE_QUERY_FLAG_INCLUDE_LOWER`]/[`DELETE_QUERY_FLAG_INCLUDE_UPPER`]/
/// [`DELETE_QUERY_FLAG_OPEN_LOWER`]/[`DELETE_QUERY_FLAG_OPEN_UPPER`].
pub(crate) const DELETE_QUERY_TERM_RANGE: u8 = 2;
/// A [`DeleteQuery::MatchAll`] node: no field, no bounds, no children.
pub(crate) const DELETE_QUERY_MATCH_ALL: u8 = 3;
/// A [`DeleteQuery::Any`] node (`BooleanQuery`, every clause `SHOULD`).
pub(crate) const DELETE_QUERY_ANY: u8 = 4;
/// A [`DeleteQuery::All`] node (`BooleanQuery`, every clause `MUST`).
pub(crate) const DELETE_QUERY_ALL: u8 = 5;
/// A [`DeleteQuery::Not`] node: exactly one child.
pub(crate) const DELETE_QUERY_NOT: u8 = 6;

pub(crate) const DELETE_QUERY_FLAG_INCLUDE_LOWER: i32 = 1;
pub(crate) const DELETE_QUERY_FLAG_INCLUDE_UPPER: i32 = 2;
pub(crate) const DELETE_QUERY_FLAG_OPEN_LOWER: i32 = 4;
pub(crate) const DELETE_QUERY_FLAG_OPEN_UPPER: i32 = 8;

/// The deepest `query_parents` chain [`ffi_writer_delete_documents_by_query`]
/// accepts, for the same caller-controlled-stack-depth reason
/// [`crate::query::MAX_CLAUSE_DEPTH`] gives: `DeleteQuery` is a recursive
/// enum, resolved recursively and dropped recursively.
pub(crate) const MAX_DELETE_QUERY_DEPTH: usize = 32;

/// The most nodes one `deleteDocuments(Query...)` call may carry -- the same
/// denial-of-service guard, and the same number, as
/// [`crate::query::MAX_CLAUSE_COUNT`].
pub(crate) const MAX_DELETE_QUERY_NODES: usize = 1024;

/// Decodes a `DeleteQuery` forest from the parent-indexed node-array wire
/// format described on [`ffi_writer_delete_documents_by_query`], returning
/// the top-level (`parent == -1`) queries in caller order.
///
/// The shape deliberately mirrors [`crate::query::read_boolean_query`]'s: a
/// flat, kind-tagged, parent-indexed array, decoded in one reverse pass with
/// no recursion, with `parent < i` enforced so a cycle is unrepresentable
/// rather than merely rejected. A new `DeleteQuery` variant is a new tag
/// value in `query_kinds`, not another ABI break.
///
/// # Safety
/// Every array must be valid for `node_count` reads (`query_parents`/
/// `query_flags` may be null, meaning all-top-level / all-zero); each
/// field/bound pointer must be valid for its paired length.
#[allow(clippy::too_many_arguments)]
unsafe fn read_delete_queries(
    query_kinds: *const u8,
    query_fields: *const *const c_char,
    query_field_lens: *const usize,
    query_lowers: *const *const u8,
    query_lower_lens: *const usize,
    query_uppers: *const *const u8,
    query_upper_lens: *const usize,
    query_parents: *const i32,
    query_flags: *const i32,
    node_count: usize,
) -> Result<Vec<DeleteQuery>, FfiStatus> {
    if node_count == 0 {
        return Ok(Vec::new());
    }
    if node_count > MAX_DELETE_QUERY_NODES {
        set_last_error(format!(
            "a delete query may carry at most {MAX_DELETE_QUERY_NODES} nodes, got {node_count}"
        ));
        return Err(FfiStatus::InvalidArgument);
    }
    if query_kinds.is_null() {
        return Err(FfiStatus::NullPointer);
    }
    // SAFETY: caller contract guarantees `node_count` readable elements.
    let kinds = unsafe { std::slice::from_raw_parts(query_kinds, node_count) };
    let parents: Option<&[i32]> = if query_parents.is_null() {
        None
    } else {
        // SAFETY: caller contract.
        Some(unsafe { std::slice::from_raw_parts(query_parents, node_count) })
    };
    let flags: Option<&[i32]> = if query_flags.is_null() {
        None
    } else {
        // SAFETY: caller contract.
        Some(unsafe { std::slice::from_raw_parts(query_flags, node_count) })
    };

    let mut children: Vec<Vec<DeleteQuery>> = try_with_capacity(node_count)?;
    let mut depth: Vec<usize> = try_with_capacity(node_count)?;
    let mut parent_of: Vec<i32> = try_with_capacity(node_count)?;
    for _ in 0..node_count {
        children.push(Vec::new());
    }

    // Forward pass: validate, and decode every node's own scalars.
    struct Node<'a> {
        kind: u8,
        field: &'a str,
        lower: &'a [u8],
        upper: &'a [u8],
        flags: i32,
    }
    let mut nodes: Vec<Node<'_>> = try_with_capacity(node_count)?;
    for i in 0..node_count {
        let kind = kinds[i];
        if kind > DELETE_QUERY_NOT {
            set_last_error(format!(
                "delete query node {i}: unknown kind {kind} (0=TERM, 1=PREFIX, 2=TERM_RANGE, \
                 3=MATCH_ALL, 4=ANY, 5=ALL, 6=NOT)"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        let parent = parents.map_or(-1, |p| p[i]);
        if parent < -1 || parent >= i as i32 {
            set_last_error(format!(
                "delete query node {i}: parent index {parent} must be -1 or an earlier node's index"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        if parent >= 0 {
            let pk = kinds[parent as usize];
            if pk != DELETE_QUERY_ANY && pk != DELETE_QUERY_ALL && pk != DELETE_QUERY_NOT {
                set_last_error(format!(
                    "delete query node {i}: parent node {parent} is a leaf and cannot contain \
                     other nodes"
                ));
                return Err(FfiStatus::InvalidArgument);
            }
        }
        let my_depth = if parent < 0 {
            0
        } else {
            depth[parent as usize] + 1
        };
        if my_depth >= MAX_DELETE_QUERY_DEPTH {
            set_last_error(format!(
                "delete query node {i}: nesting depth {} exceeds the maximum of \
                 {MAX_DELETE_QUERY_DEPTH}",
                my_depth + 1
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        depth.push(my_depth);
        parent_of.push(parent);

        let leaf = matches!(
            kind,
            DELETE_QUERY_TERM | DELETE_QUERY_PREFIX | DELETE_QUERY_TERM_RANGE
        );
        let (field, lower, upper) = if leaf {
            if query_fields.is_null()
                || query_field_lens.is_null()
                || query_lowers.is_null()
                || query_lower_lens.is_null()
                || query_uppers.is_null()
                || query_upper_lens.is_null()
            {
                return Err(FfiStatus::NullPointer);
            }
            // SAFETY: caller contract guarantees each array is valid for
            // `node_count` elements and each pointer for its paired length.
            unsafe {
                (
                    str_from_raw(*query_fields.add(i), *query_field_lens.add(i))?,
                    bytes_from_raw(*query_lowers.add(i), *query_lower_lens.add(i))?,
                    bytes_from_raw(*query_uppers.add(i), *query_upper_lens.add(i))?,
                )
            }
        } else {
            ("", &[][..], &[][..])
        };
        nodes.push(Node {
            kind,
            field,
            lower,
            upper,
            flags: flags.map_or(0, |f| f[i]),
        });
    }

    // Reverse pass: a parent's index is always smaller, so every child is
    // finished before its parent is built. No recursion.
    let mut roots: Vec<DeleteQuery> = Vec::new();
    for i in (0..node_count).rev() {
        let n = &nodes[i];
        let mut kids = std::mem::take(&mut children[i]);
        kids.reverse();
        let query = match n.kind {
            DELETE_QUERY_TERM => DeleteQuery::Term(Term::new(n.field, n.lower)),
            DELETE_QUERY_PREFIX => DeleteQuery::Prefix {
                field: n.field.to_string(),
                prefix: n.lower.to_vec(),
            },
            DELETE_QUERY_TERM_RANGE => DeleteQuery::TermRange {
                field: n.field.to_string(),
                lower: (n.flags & DELETE_QUERY_FLAG_OPEN_LOWER == 0).then(|| n.lower.to_vec()),
                upper: (n.flags & DELETE_QUERY_FLAG_OPEN_UPPER == 0).then(|| n.upper.to_vec()),
                include_lower: n.flags & DELETE_QUERY_FLAG_INCLUDE_LOWER != 0,
                include_upper: n.flags & DELETE_QUERY_FLAG_INCLUDE_UPPER != 0,
            },
            DELETE_QUERY_MATCH_ALL => DeleteQuery::MatchAll,
            DELETE_QUERY_ANY => DeleteQuery::Any(kids),
            DELETE_QUERY_ALL => DeleteQuery::All(kids),
            _ => {
                if kids.len() != 1 {
                    set_last_error(format!(
                        "delete query node {i}: a NOT node needs exactly one child, got {}",
                        kids.len()
                    ));
                    return Err(FfiStatus::InvalidArgument);
                }
                DeleteQuery::Not(Box::new(kids.pop().expect("checked len == 1")))
            }
        };
        match parent_of[i] {
            -1 => roots.push(query),
            p => children[p as usize].push(query),
        }
    }
    roots.reverse();
    Ok(roots)
}

/// `IndexWriter.deleteDocuments(Query...)`: buffers a delete for every live
/// document matching any of the supplied queries. Wraps
/// [`IndexWriter::delete_documents_by_query`], unblocked by
/// `c7-delete-queue`'s [`DeleteQuery`] resolution.
///
/// # Wire format
///
/// One flat array of `node_count` nodes forming a forest, described by nine
/// parallel arrays -- the same kind-tagged, parent-indexed shape
/// [`crate::query::read_boolean_query`] uses, chosen for the same reason (a
/// new query shape is a new tag value, not a new C signature):
///
/// | array | type | meaning |
/// |---|---|---|
/// | `query_kinds` | `u8` | [`DELETE_QUERY_TERM`]..=[`DELETE_QUERY_NOT`] |
/// | `query_fields`/`query_field_lens` | `(*const c_char, usize)` | the field, for a leaf node |
/// | `query_lowers`/`query_lower_lens` | `(*const u8, usize)` | the term, the prefix, or a range's lower bound |
/// | `query_uppers`/`query_upper_lens` | `(*const u8, usize)` | a range's upper bound |
/// | `query_parents` | `i32` | index of the enclosing ANY/ALL/NOT node, or `-1` for a top-level query. Must be `< i`. May be null, meaning "all top-level" |
/// | `query_flags` | `i32` | a range's [`DELETE_QUERY_FLAG_INCLUDE_LOWER`]/[`DELETE_QUERY_FLAG_INCLUDE_UPPER`]/[`DELETE_QUERY_FLAG_OPEN_LOWER`]/[`DELETE_QUERY_FLAG_OPEN_UPPER`] bits. May be null, meaning all zero |
///
/// **Buffered**, like every other delete here: it takes effect at the next
/// [`ffi_writer_commit`], carrying the buffer position it was issued at.
///
/// # Safety
/// Every array must be valid for `node_count` reads (with the two nullable
/// exceptions above); each field/bound pointer must be valid for its paired
/// length; `out_seq_no` must be valid for one `i64` write, or null.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_writer_delete_documents_by_query(
    writer_handle: u64,
    query_kinds: *const u8,
    query_fields: *const *const c_char,
    query_field_lens: *const usize,
    query_lowers: *const *const u8,
    query_lower_lens: *const usize,
    query_uppers: *const *const u8,
    query_upper_lens: *const usize,
    query_parents: *const i32,
    query_flags: *const i32,
    node_count: usize,
    out_seq_no: *mut i64,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own caller contract.
        let queries = unsafe {
            read_delete_queries(
                query_kinds,
                query_fields,
                query_field_lens,
                query_lowers,
                query_lower_lens,
                query_uppers,
                query_upper_lens,
                query_parents,
                query_flags,
                node_count,
            )?
        };
        if queries.is_empty() {
            set_last_error("deleteDocuments(Query...) needs at least one query");
            return Err(FfiStatus::InvalidArgument);
        }
        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error(
                "ffi_writer_delete_documents_by_query: unknown or already-closed handle",
            );
            FfiStatus::InvalidHandle
        })?;
        let seq_no = handle
            .writer
            .delete_documents_by_query(&queries)
            .map_err(|e| map_writer_error("ffi_writer_delete_documents_by_query", e))?;
        // SAFETY: caller contract.
        unsafe { write_seq_no(out_seq_no, seq_no) };
        Ok(())
    })
}

/// Closes a writer handle opened by [`ffi_open_writer`]. Returns
/// [`FfiStatus::InvalidHandle`] for an unknown/already-closed handle.
#[no_mangle]
pub extern "C" fn ffi_close_writer(writer_handle: u64) -> i32 {
    guard(|| {
        lock_recovering(writers())
            .remove(writer_handle)
            .map(|_| ())
            .ok_or_else(|| {
                set_last_error("ffi_close_writer: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_index::index_writer::{
        per_field_codec_suffix, per_field_segment, DOC_VALUES_FORMAT_NAME, POSTINGS_FORMAT_NAME,
    };
    use lucene_index::segment_infos;

    use lucene_util::test_support::TempDir;

    /// A scratch directory that removes itself when the test ends -- unless
    /// the test is panicking, in which case its bytes stay for inspection.
    fn tempdir(tag: &str) -> TempDir {
        TempDir::new(&format!("ffi-writer-{tag}"))
    }

    /// Opens a writer with a single stored-only field named `id` (field
    /// number `0`).
    fn open_test_writer(path: &std::path::Path) -> (i32, u64) {
        let path_str = path.to_str().unwrap();
        let codec = "Lucene104";
        let name = "id";
        let mut handle: u64 = 0;
        let name_ptr = name.as_ptr();
        let name_lens = [name.len()];
        let name_ptrs = [name_ptr];
        let numbers = [0i32];
        let index_options = [0i32]; // None
        let doc_values_types = [0i32]; // None
        let store_tvs = [0u8];
        let rc = unsafe {
            ffi_open_writer(
                path_str.as_ptr() as *const c_char,
                path_str.len(),
                name_ptrs.as_ptr(),
                name_lens.as_ptr(),
                numbers.as_ptr(),
                index_options.as_ptr(),
                doc_values_types.as_ptr(),
                store_tvs.as_ptr(),
                1,
                codec.as_ptr() as *const c_char,
                codec.len(),
                10,
                0,
                0,
                &mut handle as *mut _,
            )
        };
        (rc, handle)
    }

    fn add_doc(writer_handle: u64, value: &str) -> i32 {
        let numbers = [0i32];
        let kinds = [0u8]; // String
        let ptrs = [value.as_ptr()];
        let lens = [value.len()];
        unsafe {
            ffi_writer_add_document(
                writer_handle,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                1,
                std::ptr::null_mut(),
            )
        }
    }

    fn read_live_commit_data(handle: u64) -> Vec<(String, String)> {
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_writer_live_commit_data_len(handle, &mut len) },
            FfiStatus::Ok.code()
        );
        (0..len)
            .map(|i| {
                let mut key = [0 as c_char; 128];
                let mut value = [0 as c_char; 128];
                let (mut kw, mut vw) = (0usize, 0usize);
                assert_eq!(
                    unsafe {
                        ffi_writer_live_commit_data_entry(
                            handle,
                            i,
                            key.as_mut_ptr(),
                            key.len(),
                            &mut kw,
                            value.as_mut_ptr(),
                            value.len(),
                            &mut vw,
                        )
                    },
                    FfiStatus::Ok.code()
                );
                let k = unsafe { std::ffi::CStr::from_ptr(key.as_ptr()) }
                    .to_str()
                    .unwrap()
                    .to_string();
                let v = unsafe { std::ffi::CStr::from_ptr(value.as_ptr()) }
                    .to_str()
                    .unwrap()
                    .to_string();
                assert_eq!(kw, k.len());
                assert_eq!(vw, v.len());
                (k, v)
            })
            .collect()
    }

    fn set_live_commit_data(handle: u64, pairs: &[(&str, &str)]) -> i32 {
        let keys: Vec<*const c_char> = pairs.iter().map(|(k, _)| k.as_ptr().cast()).collect();
        let key_lens: Vec<usize> = pairs.iter().map(|(k, _)| k.len()).collect();
        let values: Vec<*const c_char> = pairs.iter().map(|(_, v)| v.as_ptr().cast()).collect();
        let value_lens: Vec<usize> = pairs.iter().map(|(_, v)| v.len()).collect();
        unsafe {
            ffi_writer_set_live_commit_data(
                handle,
                keys.as_ptr(),
                key_lens.as_ptr(),
                values.as_ptr(),
                value_lens.as_ptr(),
                pairs.len(),
            )
        }
    }

    /// `IndexWriter.deleteAll()` over the C ABI: every buffered *and* every
    /// committed document goes, and the emptiness is durable once committed.
    #[test]
    fn delete_all_drops_buffered_and_committed_documents() {
        let tmp = tempdir("delete-all");
        let (rc, handle) = open_test_writer(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());

        assert_eq!(add_doc(handle, "a"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "b"), FfiStatus::Ok.code());

        let mut segments: usize = 0;
        unsafe { ffi_writer_segment_infos_len(handle, &mut segments) };
        assert_eq!(segments, 1);
        let mut pending: usize = 0;
        unsafe { ffi_writer_pending_doc_count(handle, &mut pending) };
        assert_eq!(pending, 1);

        assert_eq!(ffi_writer_delete_all(handle), FfiStatus::Ok.code());

        unsafe { ffi_writer_segment_infos_len(handle, &mut segments) };
        assert_eq!(segments, 0, "committed segments dropped");
        unsafe { ffi_writer_pending_doc_count(handle, &mut pending) };
        assert_eq!(pending, 0, "buffered docs dropped");

        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        let dir = FsDirectory::open(&tmp);
        let sis = segment_infos::read_latest(&dir).unwrap();
        assert!(sis.segments.is_empty(), "the empty state is durable");

        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
    }

    #[test]
    fn delete_all_is_refused_while_a_prepared_commit_is_outstanding() {
        let tmp = tempdir("delete-all-prepared");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(add_doc(handle, "a"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_prepare_commit(handle), FfiStatus::Ok.code());
        assert_ne!(
            ffi_writer_delete_all(handle),
            FfiStatus::Ok.code(),
            "deleteAll must not run mid-two-phase-commit"
        );
        assert_eq!(ffi_writer_finish_commit(handle), FfiStatus::Ok.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
    }

    #[test]
    fn delete_all_unknown_handle_is_invalid_handle() {
        assert_eq!(
            ffi_writer_delete_all(0xDEAD_BEEF),
            FfiStatus::InvalidHandle.code()
        );
    }

    /// `setLiveCommitData`/`getLiveCommitData` round-trip, and the data
    /// really lands in the `segments_N` the commit writes.
    #[test]
    fn live_commit_data_roundtrips_and_persists_into_the_commit() {
        let tmp = tempdir("live-commit-data");
        let (rc, handle) = open_test_writer(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(read_live_commit_data(handle).is_empty());

        assert_eq!(
            set_live_commit_data(handle, &[("userData", "v1"), ("translog", "abc")]),
            FfiStatus::Ok.code()
        );
        assert_eq!(
            read_live_commit_data(handle),
            vec![
                ("userData".to_string(), "v1".to_string()),
                ("translog".to_string(), "abc".to_string()),
            ],
            "order preserved as supplied, exactly as Java writes SegmentInfos.userData"
        );

        assert_eq!(add_doc(handle, "a"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let dir = FsDirectory::open(&tmp);
        let sis = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(
            sis.user_data,
            vec![
                ("userData".to_string(), "v1".to_string()),
                ("translog".to_string(), "abc".to_string()),
            ]
        );

        // An empty list clears it, matching setLiveCommitData(emptyList()).
        assert_eq!(
            unsafe {
                ffi_writer_set_live_commit_data(
                    handle,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                )
            },
            FfiStatus::Ok.code()
        );
        assert!(read_live_commit_data(handle).is_empty());

        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
    }

    #[test]
    fn live_commit_data_entry_reports_a_too_small_buffer_without_partial_writes() {
        let tmp = tempdir("live-commit-data-small");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(
            set_live_commit_data(handle, &[("a-long-key", "a-long-value")]),
            FfiStatus::Ok.code()
        );

        let mut key = [0x7f as c_char; 4];
        let mut value = [0x7f as c_char; 64];
        let rc = unsafe {
            ffi_writer_live_commit_data_entry(
                handle,
                0,
                key.as_mut_ptr(),
                key.len(),
                std::ptr::null_mut(),
                value.as_mut_ptr(),
                value.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        // Neither buffer touched: a too-small key must not leave a
        // half-copied value behind either.
        assert!(key.iter().all(|&b| b == 0x7f));
        assert!(value.iter().all(|&b| b == 0x7f));

        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
    }

    #[test]
    fn live_commit_data_entry_out_of_bounds_and_bad_handles() {
        let tmp = tempdir("live-commit-data-bounds");
        let (_, handle) = open_test_writer(&tmp);
        let mut buf = [0 as c_char; 32];
        let rc = unsafe {
            ffi_writer_live_commit_data_entry(
                handle,
                0,
                buf.as_mut_ptr(),
                buf.len(),
                std::ptr::null_mut(),
                buf.as_mut_ptr(),
                buf.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::IndexOutOfBounds.code());
        assert_eq!(
            unsafe { ffi_writer_live_commit_data_len(0xDEAD_BEEF, &mut 0usize) },
            FfiStatus::InvalidHandle.code()
        );
        assert_eq!(
            unsafe { ffi_writer_live_commit_data_len(handle, std::ptr::null_mut()) },
            FfiStatus::NullPointer.code()
        );
        assert_eq!(
            set_live_commit_data(0xDEAD_BEEF, &[("k", "v")]),
            FfiStatus::InvalidHandle.code()
        );
        assert_eq!(
            unsafe {
                ffi_writer_set_live_commit_data(
                    handle,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    1,
                )
            },
            FfiStatus::NullPointer.code()
        );
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
    }

    /// The three merge-policy knobs this entry point gained in the M2 sweep
    /// are validated the way Java's own setters are, not passed through to
    /// produce nonsense merge scores.
    #[test]
    fn set_merge_policy_rejects_out_of_range_new_knobs() {
        let tmp = tempdir("merge-policy-range");
        let (_, handle) = open_test_writer(&tmp);
        let base = |fmdpa: f64, dpa: f64, tsc: u64| {
            ffi_writer_set_merge_policy(handle, 1, 2, 2, 1024, 2.0, 0, fmdpa, dpa, tsc)
        };
        assert_eq!(base(10.0, 20.0, 1), FfiStatus::Ok.code(), "Lucene defaults");
        // setForceMergeDeletesPctAllowed: v < 0.0 || v > 100.0
        assert_eq!(base(-0.1, 20.0, 1), FfiStatus::InvalidArgument.code());
        assert_eq!(base(100.1, 20.0, 1), FfiStatus::InvalidArgument.code());
        assert_eq!(
            base(0.0, 20.0, 1),
            FfiStatus::Ok.code(),
            "0.0 is legal here"
        );
        assert_eq!(
            base(100.0, 20.0, 1),
            FfiStatus::Ok.code(),
            "100.0 is legal here"
        );
        // setDeletesPctAllowed: v <= 0 || v > 50 -- a *different* range, and
        // the one this function used to get wrong by copying the one above.
        assert_eq!(base(10.0, -1.0, 1), FfiStatus::InvalidArgument.code());
        assert_eq!(
            base(10.0, 0.0, 1),
            FfiStatus::InvalidArgument.code(),
            "Java rejects 0 for deletesPctAllowed"
        );
        assert_eq!(
            base(10.0, 50.1, 1),
            FfiStatus::InvalidArgument.code(),
            "Java's ceiling is 50, not 100"
        );
        assert_eq!(
            base(10.0, 101.0, 1),
            FfiStatus::InvalidArgument.code(),
            "regression guard: this used to be accepted"
        );
        assert_eq!(base(10.0, 50.0, 1), FfiStatus::Ok.code(), "50 is the edge");
        // setTargetSearchConcurrency: < 1
        assert_eq!(base(10.0, 20.0, 0), FfiStatus::InvalidArgument.code());
        // setSegmentsPerTier: v < 2.0
        assert_eq!(
            ffi_writer_set_merge_policy(handle, 1, 2, 1, 1024, 2.0, 0, 10.0, 20.0, 1),
            FfiStatus::InvalidArgument.code(),
            "segmentsPerTier must be >= 2"
        );
        assert_eq!(
            ffi_writer_set_merge_policy(handle, 1, 2, 2, 1024, 2.0, 0, 10.0, 20.0, 1),
            FfiStatus::Ok.code()
        );
        // Disabling never validates its (ignored) knobs.
        assert_eq!(
            ffi_writer_set_merge_policy(handle, 0, 0, 0, 0, 0.0, 0, -5.0, 999.0, 0),
            FfiStatus::Ok.code()
        );
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
    }

    #[test]
    fn open_add_commit_end_to_end_produces_a_readable_segment() {
        let tmp = tempdir("e2e");
        let (rc, handle) = open_test_writer(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_ne!(handle, 0);

        assert_eq!(add_doc(handle, "a"), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "b"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        // Real end-to-end read-back: reopen the directory Rust-side (not
        // through this handle) and read every document out of every segment
        // `segments_N` now lists -- proves the FFI-driven commit produced a
        // genuinely valid, queryable on-disk segment, not just an `Ok`
        // status code.
        let dir = FsDirectory::open(&tmp);
        let sis = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];
        let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
        let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
        let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
        let reader =
            lucene_codecs::stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        assert_eq!(reader.max_doc(), 2);
        let mut values = Vec::new();
        for doc_id in 0..reader.max_doc() {
            let doc = reader.document(doc_id).unwrap();
            match &doc.fields[0].value {
                FieldValue::String(s) => values.push(s.clone()),
                other => panic!("unexpected value: {other:?}"),
            }
        }
        assert_eq!(values, vec!["a", "b"]);

        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
    }

    #[test]
    fn prepare_commit_then_finish_commit_round_trips_through_ffi() {
        let tmp = tempdir("two-phase");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(add_doc(handle, "x"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_prepare_commit(handle), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_finish_commit(handle), FfiStatus::Ok.code());

        let dir = FsDirectory::open(&tmp);
        let sis = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(sis.segments.len(), 1);

        ffi_close_writer(handle);
    }

    #[test]
    fn finish_commit_without_prepare_is_invalid_argument() {
        let tmp = tempdir("finish-without-prepare");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(
            ffi_writer_finish_commit(handle),
            FfiStatus::InvalidArgument.code()
        );
        ffi_close_writer(handle);
    }

    #[test]
    fn rollback_discards_pending_docs() {
        let tmp = tempdir("rollback");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(add_doc(handle, "a"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_rollback(handle), FfiStatus::Ok.code());
        // Committing now must produce zero segments (rollback discarded the
        // only buffered doc).
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        let dir = FsDirectory::open(&tmp);
        let sis = segment_infos::read_latest(&dir).unwrap();
        assert!(sis.segments.is_empty());
        ffi_close_writer(handle);
    }

    #[test]
    fn add_document_covers_binary_i64_f32_f64_kinds_round_trip() {
        // Kinds 0 (string) and 2 (i32) already have happy-path coverage
        // elsewhere; this closes the remaining four (1=binary, 3=i64,
        // 4=f32, 5=f64) that were previously only exercised by the
        // wrong-length/invalid-kind rejection tests, never a real value.
        let tmp = tempdir("add-doc-kinds");
        let (_, handle) = open_test_writer(&tmp);

        let binary_val: [u8; 3] = [1, 2, 3];
        let i64_val: i64 = -123_456_789_012;
        let f32_val: f32 = 2.5;
        let f64_val: f64 = -9.5;

        let cases: [(u8, &[u8]); 4] = [
            (1, &binary_val),
            (3, &i64_val.to_le_bytes()),
            (4, &f32_val.to_le_bytes()),
            (5, &f64_val.to_le_bytes()),
        ];

        for (kind, bytes) in cases {
            let numbers = [0i32];
            let kinds = [kind];
            let ptrs = [bytes.as_ptr()];
            let lens = [bytes.len()];
            let rc = unsafe {
                ffi_writer_add_document(
                    handle,
                    numbers.as_ptr(),
                    kinds.as_ptr(),
                    ptrs.as_ptr(),
                    lens.as_ptr(),
                    1,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(rc, FfiStatus::Ok.code(), "kind {kind} failed to add");
        }
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let dir = FsDirectory::open(&tmp);
        let sis = segment_infos::read_latest(&dir).unwrap();
        let sci = &sis.segments[0];
        let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
        let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
        let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
        let reader =
            lucene_codecs::stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        assert_eq!(reader.max_doc(), 4);
        match &reader.document(0).unwrap().fields[0].value {
            FieldValue::Binary(b) => assert_eq!(b, &binary_val),
            other => panic!("unexpected value: {other:?}"),
        }
        match &reader.document(1).unwrap().fields[0].value {
            FieldValue::Long(v) => assert_eq!(*v, i64_val),
            other => panic!("unexpected value: {other:?}"),
        }
        match &reader.document(2).unwrap().fields[0].value {
            FieldValue::Float(v) => assert_eq!(*v, f32_val),
            other => panic!("unexpected value: {other:?}"),
        }
        match &reader.document(3).unwrap().fields[0].value {
            FieldValue::Double(v) => assert_eq!(*v, f64_val),
            other => panic!("unexpected value: {other:?}"),
        }

        ffi_close_writer(handle);
    }

    #[test]
    fn open_writer_at_nonexistent_parent_path_is_io_error() {
        // FsDirectory::open itself is infallible; the failure surfaces from
        // IndexWriter::open's dir.list_all() call. Every other path-based
        // test in this module uses a real tempdir, so this closes the one
        // reachable-but-untested error branch through ffi_open_writer.
        let bogus = std::path::Path::new("/nonexistent/definitely/not/a/real/path/xyz123");
        let path_str = bogus.to_str().unwrap();
        let name = "id";
        let name_lens = [name.len()];
        let name_ptrs = [name.as_ptr()];
        let numbers = [0i32];
        let index_options = [0i32];
        let doc_values_types = [0i32];
        let store_tvs = [0u8];
        let codec = "Lucene104";
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_writer(
                path_str.as_ptr() as *const c_char,
                path_str.len(),
                name_ptrs.as_ptr(),
                name_lens.as_ptr(),
                numbers.as_ptr(),
                index_options.as_ptr(),
                doc_values_types.as_ptr(),
                store_tvs.as_ptr(),
                1,
                codec.as_ptr() as *const c_char,
                codec.len(),
                10,
                0,
                0,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Io.code());
        assert_eq!(handle, 0);
    }

    #[test]
    fn rollback_after_prepare_commit_discards_prepared_state_too() {
        // Found in review: rollback() previously only cleared pending docs,
        // leaving a prior prepare_commit()'s stashed state intact -- so
        // prepare_commit() -> rollback() -> finish_commit() would silently
        // activate the segment the caller just rolled back. Fixed at the
        // lucene-index level; this proves the fix is reachable and correct
        // through the FFI surface too.
        let tmp = tempdir("rollback-after-prepare");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(add_doc(handle, "a"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_prepare_commit(handle), FfiStatus::Ok.code());

        assert_eq!(ffi_writer_rollback(handle), FfiStatus::Ok.code());

        assert_eq!(
            ffi_writer_finish_commit(handle),
            FfiStatus::InvalidArgument.code()
        );

        // Nothing was ever written to disk.
        let dir = FsDirectory::open(&tmp);
        assert!(segment_infos::read_latest(&dir).is_err());

        // The writer is still fully usable afterward.
        assert_eq!(add_doc(handle, "b"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        let sis = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(sis.segments.len(), 1);

        ffi_close_writer(handle);
    }

    /// Opens a writer with the fixed `id` (number `0`, stored-only) field
    /// plus one caller-supplied extra field -- used by the
    /// `set_postings_field`/`set_term_vector_field`/`set_doc_values_field`
    /// end-to-end tests, each of which needs a second field with different
    /// `index_options`/`doc_values_type`/`store_term_vectors` than
    /// [`open_test_writer`]'s single stored-only field allows.
    #[allow(clippy::too_many_arguments)]
    fn open_test_writer_with_extra_field(
        path: &std::path::Path,
        extra_name: &str,
        index_options: i32,
        doc_values_type: i32,
        store_term_vectors: u8,
    ) -> (i32, u64) {
        let path_str = path.to_str().unwrap();
        let codec = "Lucene104";
        let names = ["id", extra_name];
        let name_lens = [names[0].len(), names[1].len()];
        let name_ptrs = [names[0].as_ptr(), names[1].as_ptr()];
        let numbers = [0i32, 1i32];
        let index_options_arr = [0i32, index_options];
        let doc_values_types_arr = [0i32, doc_values_type];
        let store_tvs = [0u8, store_term_vectors];
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_writer(
                path_str.as_ptr() as *const c_char,
                path_str.len(),
                name_ptrs.as_ptr(),
                name_lens.as_ptr(),
                numbers.as_ptr(),
                index_options_arr.as_ptr(),
                doc_values_types_arr.as_ptr(),
                store_tvs.as_ptr(),
                2,
                codec.as_ptr() as *const c_char,
                codec.len(),
                10,
                0,
                0,
                &mut handle as *mut _,
            )
        };
        (rc, handle)
    }

    fn add_doc_id_and_extra(writer_handle: u64, id: &str, extra: &str) -> i32 {
        let numbers = [0i32, 1i32];
        let kinds = [0u8, 0u8]; // both String
        let ptrs = [id.as_ptr(), extra.as_ptr()];
        let lens = [id.len(), extra.len()];
        unsafe {
            ffi_writer_add_document(
                writer_handle,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                2,
                std::ptr::null_mut(),
            )
        }
    }

    #[test]
    fn set_postings_field_end_to_end_writes_readable_postings() {
        let tmp = tempdir("postings-ffi");
        // index_options 2 == DocsAndFreqs (see index_options_from_i32).
        let (rc, handle) = open_test_writer_with_extra_field(&tmp, "body", 2, 0, 0);
        assert_eq!(rc, FfiStatus::Ok.code());

        let field_name = "body";
        assert_eq!(
            unsafe {
                ffi_writer_set_postings_field(
                    handle,
                    1,
                    field_name.as_ptr() as *const c_char,
                    field_name.len(),
                )
            },
            FfiStatus::Ok.code()
        );

        assert_eq!(
            add_doc_id_and_extra(handle, "a", "the quick fox"),
            FfiStatus::Ok.code()
        );
        assert_eq!(
            add_doc_id_and_extra(handle, "b", "the lazy fox"),
            FfiStatus::Ok.code()
        );
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        // Real end-to-end read-back through this crate's own unmodified
        // read-side (`lucene_codecs::blocktree`/`postings`), not through the
        // FFI writer handle -- proves the postings field was genuinely
        // written, not just that the FFI calls returned Ok.
        let dir = FsDirectory::open(&tmp);
        let sis = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(sis.segments.len(), 1);
        let sci = &sis.segments[0];
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
            fields: vec![
                FieldInfo {
                    name: "id".to_string(),
                    number: 0,
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
                },
                FieldInfo {
                    name: "body".to_string(),
                    number: 1,
                    store_term_vectors: false,
                    omit_norms: false,
                    store_payloads: false,
                    soft_deletes_field: false,
                    parent_field: false,
                    index_options: IndexOptions::DocsAndFreqs,
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
                },
            ],
        };
        let block_fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            2,
        )
        .expect("blocktree::open on FFI-produced .tim/.tip/.tmd");
        let doc_in = lucene_codecs::postings::DocInput::open(
            &doc_bytes,
            &sci.segment_id,
            &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        )
        .expect("open .doc");
        let field = block_fields.field("body").unwrap();
        let postings = field.postings(b"fox", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0, 1]);
        let postings = field.postings(b"quick", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0]);
        let postings = field.postings(b"lazy", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![1]);

        ffi_close_writer(handle);
    }

    #[test]
    fn set_term_vector_field_end_to_end_writes_readable_term_vectors() {
        let tmp = tempdir("tv-ffi");
        // index_options 2 == DocsAndFreqs (term vectors require an indexed field).
        let (rc, handle) = open_test_writer_with_extra_field(&tmp, "body", 2, 0, 1);
        assert_eq!(rc, FfiStatus::Ok.code());

        let field_name = "body";
        assert_eq!(
            unsafe {
                ffi_writer_set_term_vector_field(
                    handle,
                    1,
                    field_name.as_ptr() as *const c_char,
                    field_name.len(),
                )
            },
            FfiStatus::Ok.code()
        );

        assert_eq!(
            add_doc_id_and_extra(handle, "a", "the quick fox"),
            FfiStatus::Ok.code()
        );
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        // Real end-to-end read-back through this crate's own unmodified
        // `lucene_codecs::term_vectors::open` read side.
        let dir = FsDirectory::open(&tmp);
        let sis = segment_infos::read_latest(&dir).unwrap();
        let sci = &sis.segments[0];
        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let reader = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "")
            .expect("term_vectors::open on FFI-produced .tvd/.tvx/.tvm");
        assert_eq!(reader.max_doc(), 1);
        let doc0 = reader.document(0).unwrap().unwrap();
        assert_eq!(doc0.fields.len(), 1);
        assert_eq!(doc0.fields[0].field_number, 1);
        let mut terms0: Vec<String> = doc0.fields[0]
            .terms
            .iter()
            .map(|t| String::from_utf8(t.term.clone()).unwrap())
            .collect();
        terms0.sort();
        assert_eq!(terms0, vec!["fox", "quick", "the"]);

        ffi_close_writer(handle);
    }

    #[test]
    fn set_doc_values_field_end_to_end_writes_readable_numeric_values() {
        let tmp = tempdir("dv-ffi");
        // doc_values_type 1 == Numeric (see doc_values_type_from_i32).
        let (rc, handle) = open_test_writer_with_extra_field(&tmp, "score", 0, 1, 0);
        assert_eq!(rc, FfiStatus::Ok.code());

        let field_name = "score";
        assert_eq!(
            unsafe {
                ffi_writer_set_doc_values_field(
                    handle,
                    1,
                    field_name.as_ptr() as *const c_char,
                    field_name.len(),
                )
            },
            FfiStatus::Ok.code()
        );

        // Doc-values are dense-only: every pending doc must carry a value
        // for the opted-in field (see `IndexWriter::set_doc_values_field`'s
        // doc comment), so use kind 3 (i64) for "score" here.
        let ids = ["a", "b"];
        let scores: [i64; 2] = [5, -7];
        for (id, score) in ids.iter().zip(scores.iter()) {
            let numbers = [0i32, 1i32];
            let kinds = [0u8, 3u8]; // String, Long
            let score_bytes = score.to_le_bytes();
            let ptrs = [id.as_ptr(), score_bytes.as_ptr()];
            let lens = [id.len(), score_bytes.len()];
            let rc = unsafe {
                ffi_writer_add_document(
                    handle,
                    numbers.as_ptr(),
                    kinds.as_ptr(),
                    ptrs.as_ptr(),
                    lens.as_ptr(),
                    2,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(rc, FfiStatus::Ok.code());
        }
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        // Real end-to-end read-back through this crate's own unmodified
        // `lucene_codecs::doc_values` read side.
        let dir = FsDirectory::open(&tmp);
        let sis = segment_infos::read_latest(&dir).unwrap();
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
        let field_infos = lucene_codecs::field_infos::FieldInfos {
            fields: vec![
                FieldInfo {
                    name: "id".to_string(),
                    number: 0,
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
                },
                FieldInfo {
                    name: "score".to_string(),
                    number: 1,
                    store_term_vectors: false,
                    omit_norms: false,
                    store_payloads: false,
                    soft_deletes_field: false,
                    parent_field: false,
                    index_options: IndexOptions::None,
                    doc_values_type: DocValuesType::Numeric,
                    doc_values_skip_index_type: DocValuesSkipIndexType::None,
                    doc_values_gen: -1,
                    attributes: vec![],
                    point_dimension_count: 0,
                    point_index_dimension_count: 0,
                    point_num_bytes: 0,
                    vector_dimension: 0,
                    vector_encoding: VectorEncoding::Float32,
                    vector_similarity_function: VectorSimilarityFunction::Euclidean,
                },
            ],
        };
        let (_, meta) = lucene_codecs::doc_values::parse_meta(
            &dvm,
            &sci.segment_id,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            &field_infos,
        )
        .expect("parse_meta on FFI-produced .dvm");
        let entry = meta.numeric_entry(1).unwrap();
        for (doc, want) in [(0, 5i64), (1, -7)] {
            assert_eq!(
                lucene_codecs::doc_values::numeric_value(&dvd, entry, doc).unwrap(),
                Some(want)
            );
        }

        ffi_close_writer(handle);
    }

    #[test]
    fn set_postings_field_unknown_writer_handle_is_invalid_handle() {
        let name = "body";
        assert_eq!(
            unsafe {
                ffi_writer_set_postings_field(
                    0xDEAD_BEEF,
                    1,
                    name.as_ptr() as *const c_char,
                    name.len(),
                )
            },
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn set_postings_field_disabled_is_a_no_op_and_ok() {
        let tmp = tempdir("postings-disabled");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(
            unsafe { ffi_writer_set_postings_field(handle, 0, std::ptr::null(), 0) },
            FfiStatus::Ok.code()
        );
        ffi_close_writer(handle);
    }

    #[test]
    fn set_postings_field_unknown_field_name_is_invalid_argument() {
        let tmp = tempdir("postings-unknown-field");
        let (_, handle) = open_test_writer(&tmp);
        let name = "nonexistent";
        assert_eq!(
            unsafe {
                ffi_writer_set_postings_field(handle, 1, name.as_ptr() as *const c_char, name.len())
            },
            FfiStatus::InvalidArgument.code()
        );
        ffi_close_writer(handle);
    }

    #[test]
    fn set_term_vector_field_unknown_writer_handle_is_invalid_handle() {
        let name = "body";
        assert_eq!(
            unsafe {
                ffi_writer_set_term_vector_field(
                    0xDEAD_BEEF,
                    1,
                    name.as_ptr() as *const c_char,
                    name.len(),
                )
            },
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn set_term_vector_field_disabled_is_a_no_op_and_ok() {
        let tmp = tempdir("tv-disabled");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(
            unsafe { ffi_writer_set_term_vector_field(handle, 0, std::ptr::null(), 0) },
            FfiStatus::Ok.code()
        );
        ffi_close_writer(handle);
    }

    #[test]
    fn set_term_vector_field_unknown_field_name_is_invalid_argument() {
        let tmp = tempdir("tv-unknown-field");
        let (_, handle) = open_test_writer(&tmp);
        let name = "nonexistent";
        assert_eq!(
            unsafe {
                ffi_writer_set_term_vector_field(
                    handle,
                    1,
                    name.as_ptr() as *const c_char,
                    name.len(),
                )
            },
            FfiStatus::InvalidArgument.code()
        );
        ffi_close_writer(handle);
    }

    #[test]
    fn set_doc_values_field_unknown_writer_handle_is_invalid_handle() {
        let name = "score";
        assert_eq!(
            unsafe {
                ffi_writer_set_doc_values_field(
                    0xDEAD_BEEF,
                    1,
                    name.as_ptr() as *const c_char,
                    name.len(),
                )
            },
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn set_doc_values_field_disabled_is_a_no_op_and_ok() {
        let tmp = tempdir("dv-disabled");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(
            unsafe { ffi_writer_set_doc_values_field(handle, 0, std::ptr::null(), 0) },
            FfiStatus::Ok.code()
        );
        ffi_close_writer(handle);
    }

    #[test]
    fn set_doc_values_field_unknown_field_name_is_invalid_argument() {
        let tmp = tempdir("dv-unknown-field");
        let (_, handle) = open_test_writer(&tmp);
        let name = "nonexistent";
        assert_eq!(
            unsafe {
                ffi_writer_set_doc_values_field(
                    handle,
                    1,
                    name.as_ptr() as *const c_char,
                    name.len(),
                )
            },
            FfiStatus::InvalidArgument.code()
        );
        ffi_close_writer(handle);
    }

    #[test]
    fn set_postings_field_rejects_a_field_with_no_index_options() {
        // open_test_writer's "id" field is stored-only (index_options=0/
        // None), so it's a real field but not a valid postings target --
        // exercises Error::UnsupportedPostingsIndexOptions via
        // map_writer_error, distinct from the "unknown field name" path
        // already tested above.
        let tmp = tempdir("postings-unsupported-index-options");
        let (_, handle) = open_test_writer(&tmp);
        let name = "id";
        assert_eq!(
            unsafe {
                ffi_writer_set_postings_field(handle, 1, name.as_ptr() as *const c_char, name.len())
            },
            FfiStatus::InvalidArgument.code()
        );
        ffi_close_writer(handle);
    }

    #[test]
    fn set_doc_values_field_rejects_a_field_with_no_doc_values_type() {
        let tmp = tempdir("dv-unsupported-type");
        let (_, handle) = open_test_writer(&tmp);
        let name = "id";
        assert_eq!(
            unsafe {
                ffi_writer_set_doc_values_field(
                    handle,
                    1,
                    name.as_ptr() as *const c_char,
                    name.len(),
                )
            },
            FfiStatus::InvalidArgument.code()
        );
        ffi_close_writer(handle);
    }

    #[test]
    fn set_term_vector_field_rejects_a_field_without_store_term_vectors() {
        // "body" is a real field with real index_options, but
        // open_test_writer_with_extra_field's store_term_vectors=0 here
        // means it was never configured to store term vectors --
        // exercises Error::UnsupportedTermVectorField.
        let tmp = tempdir("tv-unsupported-field");
        let (_, handle) = open_test_writer_with_extra_field(&tmp, "body", 2, 0, 0);
        let name = "body";
        assert_eq!(
            unsafe {
                ffi_writer_set_term_vector_field(
                    handle,
                    1,
                    name.as_ptr() as *const c_char,
                    name.len(),
                )
            },
            FfiStatus::InvalidArgument.code()
        );
        ffi_close_writer(handle);
    }

    #[test]
    fn set_postings_field_can_be_switched_to_a_different_field_and_disabled() {
        // Configuring a postings field twice (to two different field
        // names) must fully replace the prior config, not append/conflict;
        // disabling afterward must also succeed cleanly -- proves
        // set_postings_field's assignment is a real reassignment, not a
        // merge, at the FFI boundary (the Rust-side guarantee was already
        // unit-tested in lucene-index; this proves it's reachable the same
        // way through FFI).
        let tmp = tempdir("postings-switch-field");
        let path_str = tmp.to_str().unwrap();
        let codec = "Lucene104";
        let names = ["id", "body", "extra"];
        let name_lens: Vec<usize> = names.iter().map(|n| n.len()).collect();
        let name_ptrs: Vec<*const u8> = names.iter().map(|n| n.as_ptr()).collect();
        let numbers = [0i32, 1i32, 2i32];
        let index_options_arr = [0i32, 2i32, 2i32]; // id=None, body/extra=DocsAndFreqs
        let doc_values_types_arr = [0i32, 0i32, 0i32];
        let store_tvs = [0u8, 0u8, 0u8];
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_writer(
                path_str.as_ptr() as *const c_char,
                path_str.len(),
                name_ptrs.as_ptr(),
                name_lens.as_ptr(),
                numbers.as_ptr(),
                index_options_arr.as_ptr(),
                doc_values_types_arr.as_ptr(),
                store_tvs.as_ptr(),
                3,
                codec.as_ptr() as *const c_char,
                codec.len(),
                10,
                0,
                0,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let body = "body";
        assert_eq!(
            unsafe {
                ffi_writer_set_postings_field(handle, 1, body.as_ptr() as *const c_char, body.len())
            },
            FfiStatus::Ok.code()
        );
        let extra = "extra";
        assert_eq!(
            unsafe {
                ffi_writer_set_postings_field(
                    handle,
                    1,
                    extra.as_ptr() as *const c_char,
                    extra.len(),
                )
            },
            FfiStatus::Ok.code()
        );
        assert_eq!(
            unsafe { ffi_writer_set_postings_field(handle, 0, std::ptr::null(), 0) },
            FfiStatus::Ok.code()
        );

        ffi_close_writer(handle);
    }

    #[test]
    fn set_merge_policy_then_many_commits_converge_to_fewer_segments() {
        let tmp = tempdir("merge-policy");
        let (_, handle) = open_test_writer(&tmp);
        // A tight policy: merge as soon as 2 segments exist.
        assert_eq!(
            ffi_writer_set_merge_policy(
                handle,
                1,
                2,
                2,
                5_000 * 1024 * 1024,
                1.0,
                0,
                10.0,
                20.0,
                1
            ),
            FfiStatus::Ok.code()
        );

        for i in 0..6 {
            assert_eq!(add_doc(handle, &format!("doc{i}")), FfiStatus::Ok.code());
            assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        }

        let dir = FsDirectory::open(&tmp);
        let sis = segment_infos::read_latest(&dir).unwrap();
        // Never more segments than commits, and the tight policy should
        // have merged at least once (fewer segments than 6 commits).
        assert!(sis.segments.len() < 6);
        assert!(!sis.segments.is_empty());

        ffi_close_writer(handle);
    }

    #[test]
    fn set_merge_policy_disabled_is_a_no_op_and_ok() {
        let tmp = tempdir("merge-policy-disabled");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(
            ffi_writer_set_merge_policy(handle, 0, 0, 0, 0, 0.0, 0, 0.0, 0.0, 0),
            FfiStatus::Ok.code()
        );
        ffi_close_writer(handle);
    }

    #[test]
    fn open_writer_null_out_handle_is_null_pointer_error() {
        let tmp = tempdir("null-out-handle");
        let path_str = tmp.to_str().unwrap();
        let codec = "Lucene104";
        let rc = unsafe {
            ffi_open_writer(
                path_str.as_ptr() as *const c_char,
                path_str.len(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                codec.as_ptr() as *const c_char,
                codec.len(),
                10,
                0,
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn open_writer_invalid_utf8_path_is_invalid_utf8_error() {
        let bytes = [0xFFu8, 0xFE];
        let codec = "Lucene104";
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_writer(
                bytes.as_ptr() as *const c_char,
                bytes.len(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                codec.as_ptr() as *const c_char,
                codec.len(),
                10,
                0,
                0,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidUtf8.code());
    }

    #[test]
    fn open_writer_null_field_array_with_nonzero_count_is_null_pointer_error() {
        let tmp = tempdir("null-field-array");
        let path_str = tmp.to_str().unwrap();
        let codec = "Lucene104";
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_writer(
                path_str.as_ptr() as *const c_char,
                path_str.len(),
                std::ptr::null(), // field_names: null, but field_count == 1
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                1,
                codec.as_ptr() as *const c_char,
                codec.len(),
                10,
                0,
                0,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn open_writer_out_of_range_index_options_is_invalid_argument() {
        let tmp = tempdir("bad-index-options");
        let path_str = tmp.to_str().unwrap();
        let codec = "Lucene104";
        let name = "id";
        let name_lens = [name.len()];
        let name_ptrs = [name.as_ptr()];
        let numbers = [0i32];
        let index_options = [99i32]; // out of range
        let doc_values_types = [0i32];
        let store_tvs = [0u8];
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_writer(
                path_str.as_ptr() as *const c_char,
                path_str.len(),
                name_ptrs.as_ptr(),
                name_lens.as_ptr(),
                numbers.as_ptr(),
                index_options.as_ptr(),
                doc_values_types.as_ptr(),
                store_tvs.as_ptr(),
                1,
                codec.as_ptr() as *const c_char,
                codec.len(),
                10,
                0,
                0,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
    }

    #[test]
    fn add_document_unknown_writer_handle_is_invalid_handle() {
        let numbers = [0i32];
        let kinds = [0u8];
        let value = "x";
        let ptrs = [value.as_ptr()];
        let lens = [value.len()];
        let rc = unsafe {
            ffi_writer_add_document(
                0xDEAD_BEEF,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                1,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn add_document_null_array_with_nonzero_count_is_null_pointer_error() {
        let tmp = tempdir("add-doc-null-array");
        let (_, handle) = open_test_writer(&tmp);
        let rc = unsafe {
            ffi_writer_add_document(
                handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                1,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_writer(handle);
    }

    #[test]
    fn add_document_unknown_kind_is_invalid_argument() {
        let tmp = tempdir("add-doc-bad-kind");
        let (_, handle) = open_test_writer(&tmp);
        let numbers = [0i32];
        let kinds = [200u8]; // unknown kind
        let value = "x";
        let ptrs = [value.as_ptr()];
        let lens = [value.len()];
        let rc = unsafe {
            ffi_writer_add_document(
                handle,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                1,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        ffi_close_writer(handle);
    }

    #[test]
    fn add_document_wrong_length_int_value_is_invalid_argument() {
        let tmp = tempdir("add-doc-bad-int-len");
        let (_, handle) = open_test_writer(&tmp);
        let numbers = [0i32];
        let kinds = [2u8]; // Int, expects 4 bytes
        let value = [0u8, 1, 2]; // only 3 bytes
        let ptrs = [value.as_ptr()];
        let lens = [value.len()];
        let rc = unsafe {
            ffi_writer_add_document(
                handle,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                1,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        ffi_close_writer(handle);
    }

    #[test]
    fn commit_unknown_writer_handle_is_invalid_handle() {
        assert_eq!(
            ffi_writer_commit(0xDEAD_BEEF),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn prepare_commit_unknown_writer_handle_is_invalid_handle() {
        assert_eq!(
            ffi_writer_prepare_commit(0xDEAD_BEEF),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn finish_commit_unknown_writer_handle_is_invalid_handle() {
        assert_eq!(
            ffi_writer_finish_commit(0xDEAD_BEEF),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn rollback_unknown_writer_handle_is_invalid_handle() {
        assert_eq!(
            ffi_writer_rollback(0xDEAD_BEEF),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn set_merge_policy_unknown_writer_handle_is_invalid_handle() {
        assert_eq!(
            ffi_writer_set_merge_policy(0xDEAD_BEEF, 1, 2, 2, 1024, 1.0, 0, 10.0, 20.0, 1),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn close_unknown_writer_handle_is_invalid_handle() {
        assert_eq!(
            ffi_close_writer(0xDEAD_BEEF),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn double_close_writer_is_invalid_handle_not_a_crash() {
        let tmp = tempdir("double-close");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::InvalidHandle.code());
    }

    /// A directory handle-shaped value must never be silently accepted by
    /// this module's functions -- the registry-tag check in `handle.rs`
    /// rejects it before any index/generation lookup happens. Exercised here
    /// via a segment/directory registry handle passed to a writer function.
    /// Opens a writer with a single field named `id` (field number `0`),
    /// stored and indexed with `DocsAndFreqs` -- callers then opt it into
    /// [`ffi_writer_set_postings_field`] so `ffi_writer_update_document`/
    /// `ffi_writer_delete_documents` have real postings to resolve their
    /// delete term against.
    fn open_test_writer_with_postings_id_field(path: &std::path::Path) -> (i32, u64) {
        let path_str = path.to_str().unwrap();
        let codec = "Lucene104";
        let name = "id";
        let mut handle: u64 = 0;
        let name_ptr = name.as_ptr();
        let name_lens = [name.len()];
        let name_ptrs = [name_ptr];
        let numbers = [0i32];
        let index_options = [2i32]; // DocsAndFreqs
        let doc_values_types = [0i32]; // None
        let store_tvs = [0u8];
        let rc = unsafe {
            ffi_open_writer(
                path_str.as_ptr() as *const c_char,
                path_str.len(),
                name_ptrs.as_ptr(),
                name_lens.as_ptr(),
                numbers.as_ptr(),
                index_options.as_ptr(),
                doc_values_types.as_ptr(),
                store_tvs.as_ptr(),
                1,
                codec.as_ptr() as *const c_char,
                codec.len(),
                10,
                0,
                0,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let id_field = "id";
        assert_eq!(
            unsafe {
                ffi_writer_set_postings_field(
                    handle,
                    1,
                    id_field.as_ptr() as *const c_char,
                    id_field.len(),
                )
            },
            FfiStatus::Ok.code()
        );
        (rc, handle)
    }

    /// Reads every field-0 (`id`) string value still live across every
    /// segment `segments_N` currently lists, filtering out docs a `.liv` file
    /// marks dead -- the "real end-to-end read-back through this crate's own
    /// unmodified read side" this module's other end-to-end tests already
    /// use, extended to skip deleted docs the way a real reader would.
    fn read_all_live_ids(dir: &FsDirectory) -> Vec<String> {
        let sis = segment_infos::read_latest(dir).unwrap();
        let mut values = Vec::new();
        for sci in &sis.segments {
            let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
            let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
            let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
            let reader =
                lucene_codecs::stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
            let max_doc = reader.max_doc() as usize;
            let live_docs = if sci.del_gen >= 0 {
                let liv = dir
                    .open(&lucene_index::deletes::liv_file_name(
                        &sci.segment_name,
                        sci.del_gen,
                    ))
                    .unwrap();
                Some(
                    lucene_codecs::live_docs::parse(
                        &liv,
                        &sci.segment_id,
                        sci.del_gen,
                        max_doc,
                        sci.del_count as usize,
                    )
                    .unwrap(),
                )
            } else {
                None
            };
            for doc_id in 0..reader.max_doc() {
                if let Some(bits) = &live_docs {
                    if !bits.get(doc_id as usize) {
                        continue;
                    }
                }
                let doc = reader.document(doc_id).unwrap();
                match &doc.fields[0].value {
                    FieldValue::String(s) => values.push(s.clone()),
                    other => panic!("unexpected value: {other:?}"),
                }
            }
        }
        values.sort();
        values
    }

    /// [`read_all_live_ids`] without the sort -- segment order, then doc
    /// order within each segment. What a block add's contiguity claim has to
    /// be checked against, since the sorted view cannot see ordering at all.
    fn read_all_live_ids_in_order(dir: &FsDirectory) -> Vec<String> {
        let sis = segment_infos::read_latest(dir).unwrap();
        let mut values = Vec::new();
        for sci in &sis.segments {
            let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
            let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
            let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
            let reader =
                lucene_codecs::stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
            for doc_id in 0..reader.max_doc() {
                let doc = reader.document(doc_id).unwrap();
                match &doc.fields[0].value {
                    FieldValue::String(v) => values.push(v.clone()),
                    other => panic!("unexpected value: {other:?}"),
                }
            }
        }
        values
    }

    #[test]
    fn update_document_end_to_end_replaces_the_old_doc_with_the_new_one() {
        let tmp = tempdir("update-doc-ffi");
        let (_, handle) = open_test_writer_with_postings_id_field(&tmp);

        assert_eq!(add_doc(handle, "docaaa"), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "docbbb"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let field_name = "id";
        let term = b"docaaa";
        let new_numbers = [0i32];
        let new_kinds = [0u8];
        let new_value = "docccc";
        let new_ptrs = [new_value.as_ptr()];
        let new_lens = [new_value.len()];
        let rc = unsafe {
            ffi_writer_update_document(
                handle,
                field_name.as_ptr() as *const c_char,
                field_name.len(),
                term.as_ptr(),
                term.len(),
                new_numbers.as_ptr(),
                new_kinds.as_ptr(),
                new_ptrs.as_ptr(),
                new_lens.as_ptr(),
                1,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        // Visibility timing, pinned: `ffi_writer_update_document` **buffers**,
        // exactly as `IndexWriter.updateDocument` does in Java. Nothing on
        // disk has changed yet.
        let dir = FsDirectory::open(&tmp);
        assert_eq!(
            read_all_live_ids(&dir),
            vec!["docaaa".to_string(), "docbbb".to_string()],
            "the update must not be visible before the next commit"
        );

        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        let values = read_all_live_ids(&dir);
        assert_eq!(values, vec!["docbbb".to_string(), "docccc".to_string()]);

        ffi_close_writer(handle);
    }

    #[test]
    fn delete_documents_end_to_end_removes_only_the_matching_doc() {
        let tmp = tempdir("delete-doc-ffi");
        let (_, handle) = open_test_writer_with_postings_id_field(&tmp);

        assert_eq!(add_doc(handle, "doc1"), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "doc2"), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "doc3"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let field_name = "id";
        let term = b"doc2";
        let rc = unsafe {
            ffi_writer_delete_documents(
                handle,
                field_name.as_ptr() as *const c_char,
                field_name.len(),
                term.as_ptr(),
                term.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        // Same buffered timing as `ffi_writer_update_document`.
        let dir = FsDirectory::open(&tmp);
        assert_eq!(
            read_all_live_ids(&dir),
            vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()],
            "the delete must not be visible before the next commit"
        );

        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        let values = read_all_live_ids(&dir);
        assert_eq!(values, vec!["doc1".to_string(), "doc3".to_string()]);

        ffi_close_writer(handle);
    }

    #[test]
    fn update_document_unknown_writer_handle_is_invalid_handle() {
        let field_name = "id";
        let term = b"x";
        let numbers = [0i32];
        let kinds = [0u8];
        let value = "y";
        let ptrs = [value.as_ptr()];
        let lens = [value.len()];
        let rc = unsafe {
            ffi_writer_update_document(
                0xDEAD_BEEF,
                field_name.as_ptr() as *const c_char,
                field_name.len(),
                term.as_ptr(),
                term.len(),
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                1,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn update_document_null_field_name_is_null_pointer_error() {
        let term = b"x";
        let rc = unsafe {
            ffi_writer_update_document(
                0xDEAD_BEEF,
                std::ptr::null(),
                1,
                term.as_ptr(),
                term.len(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn delete_documents_unknown_writer_handle_is_invalid_handle() {
        let field_name = "id";
        let term = b"x";
        let rc = unsafe {
            ffi_writer_delete_documents(
                0xDEAD_BEEF,
                field_name.as_ptr() as *const c_char,
                field_name.len(),
                term.as_ptr(),
                term.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn delete_documents_null_field_name_is_null_pointer_error() {
        let term = b"x";
        let rc = unsafe {
            ffi_writer_delete_documents(
                0xDEAD_BEEF,
                std::ptr::null(),
                1,
                term.as_ptr(),
                term.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn a_buffered_delete_does_not_reach_a_document_added_after_it() {
        // The observable consequence of `c7-delete-queue` at the JVM boundary:
        // `ffi_writer_delete_documents` no longer resolves eagerly against the
        // committed segments, it buffers against the delete queue -- so it
        // reaches every document that already existed and none added after,
        // exactly as `IndexWriter.deleteDocuments(Term...)` does in Java. A
        // document added after the delete and carrying the very same term must
        // survive.
        let tmp = tempdir("ffi-delete-ordering");
        let (_, handle) = open_test_writer_with_postings_id_field(&tmp);
        assert_eq!(add_doc(handle, "shared"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let field_name = "id";
        let term = b"shared";
        let rc = unsafe {
            ffi_writer_delete_documents(
                handle,
                field_name.as_ptr() as *const c_char,
                field_name.len(),
                term.as_ptr(),
                term.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        // Added *after* the delete was issued.
        assert_eq!(add_doc(handle, "shared"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let dir = FsDirectory::open(&tmp);
        // Exactly one survivor: the committed one died, the later one lived.
        assert_eq!(read_all_live_ids(&dir), vec!["shared".to_string()]);

        ffi_close_writer(handle);
    }

    #[test]
    fn delete_documents_on_a_writer_with_no_postings_segments_is_a_no_op() {
        // A writer whose only committed segment was flushed stored-only (no
        // `set_postings_field`) has no `.tim` file at all --
        // so the buffered delete opens it with an empty term dictionary and
        // resolves to zero documents rather than erroring -- exactly the
        // "nothing to resolve against" no-op path.
        let tmp = tempdir("delete-doc-no-postings");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(add_doc(handle, "a"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let field_name = "id";
        let term = b"a";
        let rc = unsafe {
            ffi_writer_delete_documents(
                handle,
                field_name.as_ptr() as *const c_char,
                field_name.len(),
                term.as_ptr(),
                term.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let dir = FsDirectory::open(&tmp);
        let sis = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(sis.segments.len(), 1);
        assert_eq!(sis.segments[0].del_count, 0);

        ffi_close_writer(handle);
    }

    #[test]
    fn delete_documents_with_a_term_matching_zero_docs_in_a_postings_segment_is_a_no_op() {
        // Distinct from the "no postings segments at all" no-op above: here
        // a real `.tim` file exists and is opened as a delete source, but
        // the term itself matches nothing -- `resolve_term_doc_ids`
        // returning empty, not the segment open skipping the
        // segment entirely.
        let tmp = tempdir("delete-doc-zero-match");
        let (_, handle) = open_test_writer_with_postings_id_field(&tmp);
        assert_eq!(add_doc(handle, "doc1"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let field_name = "id";
        let term = b"nonexistent";
        let rc = unsafe {
            ffi_writer_delete_documents(
                handle,
                field_name.as_ptr() as *const c_char,
                field_name.len(),
                term.as_ptr(),
                term.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let dir = FsDirectory::open(&tmp);
        let values = read_all_live_ids(&dir);
        assert_eq!(values, vec!["doc1".to_string()]);

        ffi_close_writer(handle);
    }

    #[test]
    fn delete_documents_matches_docs_across_multiple_committed_segments() {
        // Two separate commits produce two segments; the delete term
        // matches a doc in each. Proves the buffered delete's frozen packet
        // reaches *every* segment that predates it, not just the most recent
        // one.
        let tmp = tempdir("delete-doc-cross-segment");
        let (_, handle) = open_test_writer_with_postings_id_field(&tmp);
        assert_eq!(add_doc(handle, "target"), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "keep1"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "target"), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "keep2"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let dir = FsDirectory::open(&tmp);
        let sis = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(sis.segments.len(), 2, "expected two separate segments");

        let field_name = "id";
        let term = b"target";
        let rc = unsafe {
            ffi_writer_delete_documents(
                handle,
                field_name.as_ptr() as *const c_char,
                field_name.len(),
                term.as_ptr(),
                term.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let values = read_all_live_ids(&dir);
        assert_eq!(values, vec!["keep1".to_string(), "keep2".to_string()]);

        ffi_close_writer(handle);
    }

    #[test]
    fn update_document_where_term_matches_multiple_docs_replaces_all_of_them() {
        // Real Lucene's updateDocument semantics: every doc matching the
        // term is deleted, then exactly one new doc is added -- not "delete
        // the first match" or "error on ambiguous match."
        let tmp = tempdir("update-doc-multi-match");
        let (_, handle) = open_test_writer_with_postings_id_field(&tmp);
        assert_eq!(add_doc(handle, "dup"), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "dup"), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "keep"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let field_name = "id";
        let term = b"dup";
        let new_numbers = [0i32];
        let new_kinds = [0u8];
        let new_value = "replaced";
        let new_ptrs = [new_value.as_ptr()];
        let new_lens = [new_value.len()];
        let rc = unsafe {
            ffi_writer_update_document(
                handle,
                field_name.as_ptr() as *const c_char,
                field_name.len(),
                term.as_ptr(),
                term.len(),
                new_numbers.as_ptr(),
                new_kinds.as_ptr(),
                new_ptrs.as_ptr(),
                new_lens.as_ptr(),
                1,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let dir = FsDirectory::open(&tmp);
        let values = read_all_live_ids(&dir);
        assert_eq!(values, vec!["keep".to_string(), "replaced".to_string()]);

        ffi_close_writer(handle);
    }

    #[test]
    fn delete_documents_does_not_affect_pending_uncommitted_documents() {
        // A doc added but not yet committed lives only in pending_docs,
        // which delete_documents/update_document never touch (they operate
        // on already-committed segment_infos) -- it must survive an
        // interleaved delete untouched and still appear after the next
        // commit. Uses single-word values deliberately (no underscore/
        // punctuation): the "id" field goes through the real tokenizer via
        // set_postings_field, so a value like "committed_target" would
        // split into two tokens and never match an exact-term delete for
        // the whole string -- caught during review by dumping segment_infos
        // and seeing del_count stay 0 despite del_gen bumping to 1.
        let tmp = tempdir("delete-doc-pending-survives");
        let (_, handle) = open_test_writer_with_postings_id_field(&tmp);
        assert_eq!(add_doc(handle, "target"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        // Buffered but not yet committed.
        assert_eq!(add_doc(handle, "pending"), FfiStatus::Ok.code());

        let field_name = "id";
        let term = b"target";
        let rc = unsafe {
            ffi_writer_delete_documents(
                handle,
                field_name.as_ptr() as *const c_char,
                field_name.len(),
                term.as_ptr(),
                term.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        // Flush the still-pending doc.
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let dir = FsDirectory::open(&tmp);
        let values = read_all_live_ids(&dir);
        assert_eq!(values, vec!["pending".to_string()]);

        ffi_close_writer(handle);
    }

    #[test]
    fn directory_handle_passed_to_writer_function_is_invalid_handle() {
        use crate::directory::ffi_open_directory;
        let tmp = tempdir("cross-registry");
        let path_str = tmp.to_str().unwrap();
        let mut dir_handle: u64 = 0;
        unsafe {
            ffi_open_directory(
                path_str.as_ptr() as *const c_char,
                path_str.len(),
                &mut dir_handle as *mut _,
            );
        }
        assert_ne!(dir_handle, 0);
        assert_eq!(
            ffi_writer_commit(dir_handle),
            FfiStatus::InvalidHandle.code()
        );
        crate::directory::ffi_close_directory(dir_handle);
    }

    #[test]
    fn segment_infos_len_unknown_handle_is_invalid_handle() {
        let mut out_len: usize = 0;
        let rc = unsafe { ffi_writer_segment_infos_len(0xDEAD_BEEF, &mut out_len as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn segment_infos_len_null_out_len_is_null_pointer_error() {
        let tmp = tempdir("segment-infos-len-null");
        let (_, handle) = open_test_writer(&tmp);
        let rc = unsafe { ffi_writer_segment_infos_len(handle, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_writer(handle);
    }

    #[test]
    fn segment_info_name_unknown_handle_is_invalid_handle() {
        let mut buf = [0 as c_char; 64];
        let mut written: usize = 0;
        let rc = unsafe {
            ffi_writer_segment_info_name(
                0xDEAD_BEEF,
                0,
                buf.as_mut_ptr(),
                buf.len(),
                &mut written as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn segment_info_name_out_of_bounds_index_is_index_out_of_bounds() {
        let tmp = tempdir("segment-info-name-oob");
        let (_, handle) = open_test_writer(&tmp);
        let mut buf = [0 as c_char; 64];
        let mut written: usize = 0;
        let rc = unsafe {
            ffi_writer_segment_info_name(
                handle,
                0,
                buf.as_mut_ptr(),
                buf.len(),
                &mut written as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::IndexOutOfBounds.code());
        ffi_close_writer(handle);
    }

    #[test]
    fn segment_info_name_buffer_too_small_leaves_buffer_untouched() {
        let tmp = tempdir("segment-info-name-small-buf");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(add_doc(handle, "a"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let mut buf = [0 as c_char; 1]; // segment name "_0" needs 3 bytes (2 + NUL)
        let mut written: usize = 0;
        let rc = unsafe {
            ffi_writer_segment_info_name(
                handle,
                0,
                buf.as_mut_ptr(),
                buf.len(),
                &mut written as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        ffi_close_writer(handle);
    }

    #[test]
    fn segment_info_name_exact_size_buffer_succeeds() {
        // A buffer of exactly `name_bytes.len() + 1` (room for the NUL
        // terminator, no more) is the boundary between the too-small case
        // above and the generously-large case the end-to-end test uses --
        // must succeed, not be rejected as one byte short.
        let tmp = tempdir("segment-info-name-exact-buf");
        let (_, handle) = open_test_writer(&tmp);
        assert_eq!(add_doc(handle, "a"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let mut buf = [0 as c_char; 3]; // "_0" is 2 bytes + 1 for NUL == 3
        let mut written: usize = 0;
        let rc = unsafe {
            ffi_writer_segment_info_name(
                handle,
                0,
                buf.as_mut_ptr(),
                buf.len(),
                &mut written as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(written, 2);
        let name = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_str()
            .unwrap();
        assert_eq!(name, "_0");
        ffi_close_writer(handle);
    }

    #[test]
    fn pending_doc_count_unknown_handle_is_invalid_handle() {
        let mut out_len: usize = 0;
        let rc = unsafe { ffi_writer_pending_doc_count(0xDEAD_BEEF, &mut out_len as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn pending_doc_count_null_out_len_is_null_pointer_error() {
        let tmp = tempdir("pending-doc-count-null");
        let (_, handle) = open_test_writer(&tmp);
        let rc = unsafe { ffi_writer_pending_doc_count(handle, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_writer(handle);
    }

    #[test]
    fn committed_doc_count_unknown_handle_is_invalid_handle() {
        let mut out_len: usize = 0;
        let rc = unsafe { ffi_writer_committed_doc_count(0xDEAD_BEEF, &mut out_len as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn committed_doc_count_null_out_len_is_null_pointer_error() {
        let tmp = tempdir("committed-doc-count-null");
        let (_, handle) = open_test_writer(&tmp);
        let rc = unsafe { ffi_writer_committed_doc_count(handle, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_writer(handle);
    }

    #[test]
    fn committed_doc_count_is_distinct_from_pending_doc_count_across_commits() {
        let tmp = tempdir("committed-doc-count-e2e");
        let (rc, handle) = open_test_writer(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());

        let mut committed: usize = 0;
        let mut pending: usize = 0;

        // Fresh writer: nothing committed, nothing pending.
        assert_eq!(
            unsafe { ffi_writer_committed_doc_count(handle, &mut committed as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(committed, 0);

        // Commit 3 docs.
        assert_eq!(add_doc(handle, "a"), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "b"), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "c"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(
            unsafe { ffi_writer_committed_doc_count(handle, &mut committed as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(committed, 3);
        assert_eq!(
            unsafe { ffi_writer_pending_doc_count(handle, &mut pending as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(pending, 0);

        // Buffer 2 more docs without committing: committed_doc_count must
        // stay at 3 (not conflated with the buffer) while
        // pending_doc_count reports the 2 buffered docs.
        assert_eq!(add_doc(handle, "d"), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "e"), FfiStatus::Ok.code());
        assert_eq!(
            unsafe { ffi_writer_committed_doc_count(handle, &mut committed as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(committed, 3);
        assert_eq!(
            unsafe { ffi_writer_pending_doc_count(handle, &mut pending as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(pending, 2);

        // Commit the buffered pair: committed_doc_count now reflects both
        // segments' worth of docs.
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(
            unsafe { ffi_writer_committed_doc_count(handle, &mut committed as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(committed, 5);
        assert_eq!(
            unsafe { ffi_writer_pending_doc_count(handle, &mut pending as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(pending, 0);

        ffi_close_writer(handle);
    }

    #[test]
    fn segment_infos_and_pending_doc_count_reflect_writer_state_across_commits() {
        let tmp = tempdir("segment-infos-pending-e2e");
        let (rc, handle) = open_test_writer(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());

        // Fresh writer: no segments yet, nothing pending.
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_writer_segment_infos_len(handle, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 0);
        assert_eq!(
            unsafe { ffi_writer_pending_doc_count(handle, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 0);

        // Buffer two documents: pending_doc_count reflects them, but no
        // segment exists until commit().
        assert_eq!(add_doc(handle, "a"), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "b"), FfiStatus::Ok.code());
        let mut pending: usize = 0;
        assert_eq!(
            unsafe { ffi_writer_pending_doc_count(handle, &mut pending as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(pending, 2);
        assert_eq!(
            unsafe { ffi_writer_segment_infos_len(handle, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 0);

        // First commit: one segment, pending count back to 0.
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(
            unsafe { ffi_writer_pending_doc_count(handle, &mut pending as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(pending, 0);
        assert_eq!(
            unsafe { ffi_writer_segment_infos_len(handle, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 1);

        let mut buf = [0 as c_char; 64];
        let mut written: usize = 0;
        assert_eq!(
            unsafe {
                ffi_writer_segment_info_name(
                    handle,
                    0,
                    buf.as_mut_ptr(),
                    buf.len(),
                    &mut written as *mut _,
                )
            },
            FfiStatus::Ok.code()
        );
        let name = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_str()
            .unwrap();
        assert_eq!(name, "_0");
        assert_eq!(written, 2);

        // Second commit (no default auto-merge policy set, so no merging
        // happens here): two segments.
        assert_eq!(add_doc(handle, "c"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(
            unsafe { ffi_writer_segment_infos_len(handle, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 2);

        ffi_close_writer(handle);
    }
    // ------------------------------------------------------------------
    // c13-ffi-surface: sequence numbers, the four unwrapped field setters,
    // and c7's delete-queue-backed APIs
    // ------------------------------------------------------------------

    fn add_doc_seq(writer_handle: u64, value: &str, out: &mut i64) -> i32 {
        let numbers = [0i32];
        let kinds = [0u8];
        let ptrs = [value.as_ptr()];
        let lens = [value.len()];
        unsafe {
            ffi_writer_add_document(
                writer_handle,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                1,
                out as *mut i64,
            )
        }
    }

    /// Java's `IndexWriter` returns a `long` seqNo from every mutating
    /// method, and `c7-delete-queue` (finding A7) recorded that the boundary
    /// dropped it. It must now come back, start at 1, and strictly increase
    /// across add/update/delete.
    #[test]
    fn every_mutating_call_returns_a_strictly_increasing_sequence_number() {
        let tmp = tempdir("seqno");
        let (rc, handle) = open_test_writer_with_postings_id_field(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());

        let mut first = 0i64;
        assert_eq!(add_doc_seq(handle, "a", &mut first), FfiStatus::Ok.code());
        assert_eq!(first, 1, "DocumentsWriterDeleteQueue starts seqNos at 1");

        let mut second = 0i64;
        assert_eq!(add_doc_seq(handle, "b", &mut second), FfiStatus::Ok.code());
        assert!(second > first);

        let field = "id";
        let term = b"a";
        let numbers = [0i32];
        let kinds = [0u8];
        let value = "c";
        let ptrs = [value.as_ptr()];
        let lens = [value.len()];
        let mut updated = 0i64;
        let rc = unsafe {
            ffi_writer_update_document(
                handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                1,
                &mut updated as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(updated > second);

        let mut deleted = 0i64;
        let rc = unsafe {
            ffi_writer_delete_documents(
                handle,
                field.as_ptr() as *const c_char,
                field.len(),
                b"b".as_ptr(),
                1,
                &mut deleted as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(deleted > updated);

        // A null out pointer is the documented "the caller does not want it".
        assert_eq!(add_doc(handle, "d"), FfiStatus::Ok.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `addDocuments` (block add): the documents must land contiguously in
    /// one segment, in order, under one sequence number.
    #[test]
    fn add_documents_writes_the_whole_block_contiguously_under_one_seq_no() {
        let tmp = tempdir("block-add");
        let (rc, handle) = open_test_writer(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());

        let values = ["p1", "c1", "c2"];
        let counts = [1usize, 1, 1];
        let numbers = [0i32, 0, 0];
        let kinds = [0u8, 0, 0];
        let ptrs: Vec<*const u8> = values.iter().map(|v| v.as_ptr()).collect();
        let lens: Vec<usize> = values.iter().map(|v| v.len()).collect();
        let mut seq = 0i64;
        let rc = unsafe {
            ffi_writer_add_documents(
                handle,
                counts.as_ptr(),
                3,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                &mut seq as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(seq, 1, "one sequence number for the whole block");
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());

        let dir = FsDirectory::open(tmp.to_str().unwrap());
        assert_eq!(
            read_all_live_ids_in_order(&dir),
            vec!["p1", "c1", "c2"],
            "a block must land contiguously, in the caller's order"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A document block whose per-document field counts differ: document `d`
    /// must get exactly its own slice of the flat field arrays.
    #[test]
    fn add_documents_slices_the_flat_field_arrays_per_document() {
        let tmp = tempdir("block-add-ragged");
        let (rc, handle) = open_test_writer_with_extra_field(&tmp, "body", 0, 0, 0);
        assert_eq!(rc, FfiStatus::Ok.code());

        // doc 0: (id="x", body="B"); doc 1: (id="y")
        let vals = ["x", "B", "y"];
        let counts = [2usize, 1];
        let numbers = [0i32, 1, 0];
        let kinds = [0u8, 0, 0];
        let ptrs: Vec<*const u8> = vals.iter().map(|v| v.as_ptr()).collect();
        let lens: Vec<usize> = vals.iter().map(|v| v.len()).collect();
        let rc = unsafe {
            ffi_writer_add_documents(
                handle,
                counts.as_ptr(),
                2,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
        let dir = FsDirectory::open(tmp.to_str().unwrap());
        assert_eq!(read_all_live_ids(&dir), vec!["x", "y"]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn add_documents_rejects_a_null_count_array_and_an_unknown_handle() {
        let tmp = tempdir("block-add-errors");
        let (rc, handle) = open_test_writer(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());
        let rc = unsafe {
            ffi_writer_add_documents(
                handle,
                std::ptr::null(),
                2,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let counts = [1usize];
        let rc = unsafe {
            ffi_writer_add_documents(
                0xDEAD_BEEF,
                counts.as_ptr(),
                1,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        // Overflowing per-document field counts must be a status, not a
        // wrapped total that later indexes past the field arrays.
        let counts = [usize::MAX, 2usize];
        let rc = unsafe {
            ffi_writer_add_documents(
                handle,
                counts.as_ptr(),
                2,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `updateDocuments(Term, block)`: the delete and the whole block share
    /// one sequence number and become visible together.
    #[test]
    fn update_documents_replaces_the_matched_docs_with_the_whole_block() {
        let tmp = tempdir("block-update");
        let (rc, handle) = open_test_writer_with_postings_id_field(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "old"), FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "keep"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let field = "id";
        let term = b"old";
        let vals = ["new1", "new2"];
        let counts = [1usize, 1];
        let numbers = [0i32, 0];
        let kinds = [0u8, 0];
        let ptrs: Vec<*const u8> = vals.iter().map(|v| v.as_ptr()).collect();
        let lens: Vec<usize> = vals.iter().map(|v| v.len()).collect();
        let mut seq = 0i64;
        let rc = unsafe {
            ffi_writer_update_documents(
                handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                counts.as_ptr(),
                2,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                &mut seq as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(seq > 0);
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());

        let dir = FsDirectory::open(tmp.to_str().unwrap());
        let mut ids = read_all_live_ids(&dir);
        ids.sort();
        assert_eq!(ids, vec!["keep", "new1", "new2"]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `softUpdateDocument`: the previous version keeps its live bit -- that
    /// is the whole point of a soft delete -- while the new one is added.
    #[test]
    fn soft_update_document_adds_without_hard_deleting_the_previous_version() {
        let tmp = tempdir("soft-update");
        let (rc, handle) = open_test_writer_with_postings_id_field(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(add_doc(handle, "v1"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let field = "id";
        let term = b"v1";
        let soft = "__soft_deletes";
        let value = "v2";
        let numbers = [0i32];
        let kinds = [0u8];
        let ptrs = [value.as_ptr()];
        let lens = [value.len()];
        let mut seq = 0i64;
        let rc = unsafe {
            ffi_writer_soft_update_document(
                handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                soft.as_ptr() as *const c_char,
                soft.len(),
                1,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                1,
                &mut seq as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(seq > 0);
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());

        let dir = FsDirectory::open(tmp.to_str().unwrap());
        let mut ids = read_all_live_ids(&dir);
        ids.sort();
        assert_eq!(
            ids,
            vec!["v1", "v2"],
            "a soft delete must not remove the previous version's live bit"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Java's *"at least one soft delete must be present"* check.
    #[test]
    fn soft_update_document_without_a_soft_delete_field_is_rejected() {
        let tmp = tempdir("soft-update-empty");
        let (rc, handle) = open_test_writer(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());
        let field = "id";
        let rc = unsafe {
            ffi_writer_soft_update_document(
                handle,
                field.as_ptr() as *const c_char,
                field.len(),
                b"x".as_ptr(),
                1,
                std::ptr::null(),
                0,
                1,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `updateNumericDocValue`/`updateBinaryDocValue` must be accepted,
    /// buffered, and return an increasing seqNo -- the doc-values-update
    /// surface `c7-delete-queue` unblocked.
    #[test]
    fn doc_values_updates_are_accepted_and_buffered_with_sequence_numbers() {
        let tmp = tempdir("dv-update");
        // `rank` declared NUMERIC (doc_values_type 1) so the update targets a
        // field the writer knows; `id` carries the postings the delete term
        // resolves against.
        let (rc, handle) = open_test_writer_with_extra_field(&tmp, "rank", 0, 1, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        let id = "id";
        assert_eq!(
            unsafe {
                ffi_writer_set_postings_field(handle, 1, id.as_ptr() as *const c_char, id.len())
            },
            FfiStatus::InvalidArgument.code(),
            "`id` is stored-only here, so postings are not available for it"
        );
        assert_eq!(add_doc_id_and_extra(handle, "a", "1"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let term_field = "id";
        let dv_field = "rank";
        let mut n_seq = 0i64;
        let rc = unsafe {
            ffi_writer_update_numeric_doc_value(
                handle,
                term_field.as_ptr() as *const c_char,
                term_field.len(),
                b"a".as_ptr(),
                1,
                dv_field.as_ptr() as *const c_char,
                dv_field.len(),
                42,
                &mut n_seq as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(n_seq > 0);

        // A *binary* update against the same NUMERIC-declared field is
        // Java's `IllegalArgumentException`, and must arrive as
        // `InvalidArgument` rather than the `Io` this boundary used to map
        // every unenumerated writer error to.
        let mut b_seq = 0i64;
        let payload = b"hello";
        let rc = unsafe {
            ffi_writer_update_binary_doc_value(
                handle,
                term_field.as_ptr() as *const c_char,
                term_field.len(),
                b"a".as_ptr(),
                1,
                dv_field.as_ptr() as *const c_char,
                dv_field.len(),
                payload.as_ptr(),
                payload.len(),
                &mut b_seq as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert_eq!(b_seq, 0, "the out-parameter is untouched on failure");
        // An update naming a field the writer has never heard of is likewise
        // an argument error, not an I/O one.
        let rc = unsafe {
            ffi_writer_update_numeric_doc_value(
                handle,
                term_field.as_ptr() as *const c_char,
                term_field.len(),
                b"a".as_ptr(),
                1,
                b"no_such_field".as_ptr().cast::<c_char>(),
                13,
                1,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        // Neither refusal may consume a sequence number: the next accepted
        // update must be exactly one past the last accepted one. (Asserting
        // on the out-parameter alone would be tautological -- it starts at 0
        // and the failing path never writes it.)
        let mut next_seq = 0i64;
        let rc = unsafe {
            ffi_writer_update_numeric_doc_value(
                handle,
                term_field.as_ptr() as *const c_char,
                term_field.len(),
                b"a".as_ptr(),
                1,
                dv_field.as_ptr() as *const c_char,
                dv_field.len(),
                43,
                &mut next_seq as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(
            next_seq,
            n_seq + 1,
            "two refused calls between them must not have advanced the delete queue"
        );

        // Unknown handles are rejected before anything is buffered.
        let rc = unsafe {
            ffi_writer_update_numeric_doc_value(
                0xDEAD_BEEF,
                term_field.as_ptr() as *const c_char,
                term_field.len(),
                b"a".as_ptr(),
                1,
                dv_field.as_ptr() as *const c_char,
                dv_field.len(),
                1,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Builds the nine `ffi_writer_delete_documents_by_query` arrays from a
    /// declarative node list.
    /// `(kind, field, lower, upper, parent, flags)` -- one
    /// `ffi_writer_delete_documents_by_query` node, declaratively.
    type DeleteNodeSpec = (u8, &'static str, &'static [u8], &'static [u8], i32, i32);

    struct DeleteNodes {
        kinds: Vec<u8>,
        fields: Vec<*const c_char>,
        field_lens: Vec<usize>,
        lowers: Vec<*const u8>,
        lower_lens: Vec<usize>,
        uppers: Vec<*const u8>,
        upper_lens: Vec<usize>,
        parents: Vec<i32>,
        flags: Vec<i32>,
    }

    impl DeleteNodes {
        /// `(kind, field, lower, upper, parent, flags)`.
        fn new(specs: &[DeleteNodeSpec]) -> Self {
            let mut n = DeleteNodes {
                kinds: Vec::new(),
                fields: Vec::new(),
                field_lens: Vec::new(),
                lowers: Vec::new(),
                lower_lens: Vec::new(),
                uppers: Vec::new(),
                upper_lens: Vec::new(),
                parents: Vec::new(),
                flags: Vec::new(),
            };
            for (kind, field, lower, upper, parent, flags) in specs {
                n.kinds.push(*kind);
                n.fields.push(field.as_ptr() as *const c_char);
                n.field_lens.push(field.len());
                n.lowers.push(lower.as_ptr());
                n.lower_lens.push(lower.len());
                n.uppers.push(upper.as_ptr());
                n.upper_lens.push(upper.len());
                n.parents.push(*parent);
                n.flags.push(*flags);
            }
            n
        }

        fn run(&self, handle: u64) -> (i32, i64) {
            let mut seq = 0i64;
            let rc = unsafe {
                ffi_writer_delete_documents_by_query(
                    handle,
                    self.kinds.as_ptr(),
                    self.fields.as_ptr(),
                    self.field_lens.as_ptr(),
                    self.lowers.as_ptr(),
                    self.lower_lens.as_ptr(),
                    self.uppers.as_ptr(),
                    self.upper_lens.as_ptr(),
                    self.parents.as_ptr(),
                    self.flags.as_ptr(),
                    self.kinds.len(),
                    &mut seq as *mut _,
                )
            };
            (rc, seq)
        }
    }

    /// `deleteDocuments(Query...)` end to end: a prefix query removes exactly
    /// the documents whose term starts with the prefix.
    #[test]
    fn delete_by_prefix_query_removes_exactly_the_matching_documents() {
        let tmp = tempdir("delete-by-query");
        let (rc, handle) = open_test_writer_with_postings_id_field(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());
        for id in ["aa1", "aa2", "bb1"] {
            assert_eq!(add_doc(handle, id), FfiStatus::Ok.code());
        }
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let nodes = DeleteNodes::new(&[(DELETE_QUERY_PREFIX, "id", b"aa", b"", -1, 0)]);
        let (rc, seq) = nodes.run(handle);
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(seq > 0);
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());

        let dir = FsDirectory::open(tmp.to_str().unwrap());
        assert_eq!(read_all_live_ids(&dir), vec!["bb1"]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A composed query: `ANY(term aa1, term bb1)`, proving the
    /// parent-indexed node array actually nests.
    #[test]
    fn delete_by_a_composed_any_query_removes_every_branch_s_matches() {
        let tmp = tempdir("delete-by-any");
        let (rc, handle) = open_test_writer_with_postings_id_field(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());
        for id in ["aa1", "aa2", "bb1"] {
            assert_eq!(add_doc(handle, id), FfiStatus::Ok.code());
        }
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let nodes = DeleteNodes::new(&[
            (DELETE_QUERY_ANY, "", b"", b"", -1, 0),
            (DELETE_QUERY_TERM, "id", b"aa1", b"", 0, 0),
            (DELETE_QUERY_TERM, "id", b"bb1", b"", 0, 0),
        ]);
        let (rc, _) = nodes.run(handle);
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());

        let dir = FsDirectory::open(tmp.to_str().unwrap());
        assert_eq!(read_all_live_ids(&dir), vec!["aa2"]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A term-range query, exercising the flag bits (inclusive bounds and an
    /// open upper bound).
    #[test]
    fn delete_by_term_range_query_honours_the_bound_flags() {
        let tmp = tempdir("delete-by-range");
        let (rc, handle) = open_test_writer_with_postings_id_field(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());
        for id in ["a", "b", "c", "d"] {
            assert_eq!(add_doc(handle, id), FfiStatus::Ok.code());
        }
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        // [b, TO *]  -- inclusive lower, open upper.
        let nodes = DeleteNodes::new(&[(
            DELETE_QUERY_TERM_RANGE,
            "id",
            b"b",
            b"",
            -1,
            DELETE_QUERY_FLAG_INCLUDE_LOWER | DELETE_QUERY_FLAG_OPEN_UPPER,
        )]);
        let (rc, _) = nodes.run(handle);
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());

        let dir = FsDirectory::open(tmp.to_str().unwrap());
        assert_eq!(read_all_live_ids(&dir), vec!["a"]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The success path of `ffi_writer_update_binary_doc_value`: without it
    /// the symbol's only exercise is an error case, and a transposed
    /// `dv_field_name`/`value_ptr` pair in a ten-parameter C signature would
    /// go unnoticed.
    #[test]
    fn a_binary_doc_values_update_against_a_binary_field_succeeds() {
        let tmp = tempdir("dv-update-binary");
        // doc_values_type 2 == Binary (see doc_values_type_from_i32).
        let (rc, handle) = open_test_writer_with_extra_field(&tmp, "blob", 0, 2, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(add_doc_id_and_extra(handle, "a", "x"), FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());

        let term_field = "id";
        let dv_field = "blob";
        let payload = b"hello";
        let mut seq = 0i64;
        let rc = unsafe {
            ffi_writer_update_binary_doc_value(
                handle,
                term_field.as_ptr() as *const c_char,
                term_field.len(),
                b"a".as_ptr(),
                1,
                dv_field.as_ptr() as *const c_char,
                dv_field.len(),
                payload.as_ptr(),
                payload.len(),
                &mut seq as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(
            seq > 0,
            "an accepted update must return its sequence number"
        );
        // A *numeric* update against the same BINARY field is the mirror-image
        // type error.
        let rc = unsafe {
            ffi_writer_update_numeric_doc_value(
                handle,
                term_field.as_ptr() as *const c_char,
                term_field.len(),
                b"a".as_ptr(),
                1,
                dv_field.as_ptr() as *const c_char,
                dv_field.len(),
                1,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn every_malformed_delete_query_node_array_is_rejected() {
        let tmp = tempdir("delete-by-query-errors");
        let (rc, handle) = open_test_writer_with_postings_id_field(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());

        // No nodes at all.
        let (rc, _) = DeleteNodes::new(&[]).run(handle);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        // Unknown kind.
        let (rc, _) = DeleteNodes::new(&[(9, "id", b"a", b"", -1, 0)]).run(handle);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        // A forward parent reference.
        let (rc, _) = DeleteNodes::new(&[
            (DELETE_QUERY_TERM, "id", b"a", b"", 1, 0),
            (DELETE_QUERY_ANY, "", b"", b"", -1, 0),
        ])
        .run(handle);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        // A leaf used as a parent.
        let (rc, _) = DeleteNodes::new(&[
            (DELETE_QUERY_TERM, "id", b"a", b"", -1, 0),
            (DELETE_QUERY_TERM, "id", b"b", b"", 0, 0),
        ])
        .run(handle);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        // A NOT node with the wrong number of children.
        let (rc, _) = DeleteNodes::new(&[
            (DELETE_QUERY_NOT, "", b"", b"", -1, 0),
            (DELETE_QUERY_TERM, "id", b"a", b"", 0, 0),
            (DELETE_QUERY_TERM, "id", b"b", b"", 0, 0),
        ])
        .run(handle);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        // Too deep.
        let mut deep: Vec<DeleteNodeSpec> = Vec::new();
        for i in 0..MAX_DELETE_QUERY_DEPTH + 1 {
            deep.push((DELETE_QUERY_NOT, "", b"", b"", i as i32 - 1, 0));
        }
        let (rc, _) = DeleteNodes::new(&deep).run(handle);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        // Too many nodes.
        let many: Vec<DeleteNodeSpec> = (0..MAX_DELETE_QUERY_NODES + 1)
            .map(|_| {
                (
                    DELETE_QUERY_TERM,
                    "id",
                    b"a".as_slice(),
                    b"".as_slice(),
                    -1,
                    0,
                )
            })
            .collect();
        let (rc, _) = DeleteNodes::new(&many).run(handle);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        // Null kinds with a non-zero count.
        let rc = unsafe {
            ffi_writer_delete_documents_by_query(
                handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                2,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        // Unknown handle.
        let (rc, _) =
            DeleteNodes::new(&[(DELETE_QUERY_MATCH_ALL, "", b"", b"", -1, 0)]).run(0xDEAD_BEEF);
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `deleteDocuments(new MatchAllDocsQuery())` -- Java specialises it into
    /// `deleteAll()`, and so does this port.
    #[test]
    fn delete_by_match_all_query_empties_the_index() {
        let tmp = tempdir("delete-by-match-all");
        let (rc, handle) = open_test_writer_with_postings_id_field(&tmp);
        assert_eq!(rc, FfiStatus::Ok.code());
        for id in ["a", "b"] {
            assert_eq!(add_doc(handle, id), FfiStatus::Ok.code());
        }
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        let (rc, _) =
            DeleteNodes::new(&[(DELETE_QUERY_MATCH_ALL, "", b"", b"", -1, 0)]).run(handle);
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
        let dir = FsDirectory::open(tmp.to_str().unwrap());
        assert!(read_all_live_ids(&dir).is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `add_postings_field` opts in a *second* field alongside
    /// `set_postings_field`'s -- without it a writer could index exactly one
    /// searchable field.
    #[test]
    fn add_postings_field_indexes_a_second_field_alongside_the_first() {
        let tmp = tempdir("add-postings-field");
        let (rc, handle) = open_test_writer_with_extra_field(&tmp, "body", 2, 0, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        let id = "id";
        let body = "body";
        assert_eq!(
            unsafe {
                ffi_writer_set_postings_field(handle, 1, id.as_ptr() as *const c_char, id.len())
            },
            FfiStatus::InvalidArgument.code(),
            "field `id` has IndexOptions::None, so it cannot carry postings"
        );
        assert_eq!(
            unsafe {
                ffi_writer_set_postings_field(handle, 1, body.as_ptr() as *const c_char, body.len())
            },
            FfiStatus::Ok.code()
        );
        // Adding the same field twice is refused by the writer.
        assert_eq!(
            unsafe {
                ffi_writer_add_postings_field(handle, body.as_ptr() as *const c_char, body.len())
            },
            FfiStatus::InvalidArgument.code()
        );
        assert_eq!(
            unsafe {
                ffi_writer_add_postings_field(handle, id.as_ptr() as *const c_char, id.len())
            },
            FfiStatus::InvalidArgument.code()
        );
        assert_eq!(
            unsafe {
                ffi_writer_add_postings_field(
                    0xDEAD_BEEF,
                    body.as_ptr() as *const c_char,
                    body.len(),
                )
            },
            FfiStatus::InvalidHandle.code()
        );
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn add_term_vector_field_and_omit_norms_field_are_wired_through() {
        let tmp = tempdir("add-tv-norms");
        // `body` with DocsAndFreqs + stored term vectors.
        let (rc, handle) = open_test_writer_with_extra_field(&tmp, "body", 2, 0, 1);
        assert_eq!(rc, FfiStatus::Ok.code());
        let body = "body";
        assert_eq!(
            unsafe {
                ffi_writer_add_term_vector_field(handle, body.as_ptr() as *const c_char, body.len())
            },
            FfiStatus::Ok.code()
        );
        assert_eq!(
            unsafe {
                ffi_writer_omit_norms_field(handle, body.as_ptr() as *const c_char, body.len())
            },
            FfiStatus::Ok.code()
        );
        // A null field name is a caller error, not "all fields".
        assert_eq!(
            unsafe { ffi_writer_omit_norms_field(handle, std::ptr::null(), 0) },
            FfiStatus::InvalidArgument.code()
        );
        // A field with no index options has no norms to omit.
        let id = "id";
        assert_eq!(
            unsafe { ffi_writer_omit_norms_field(handle, id.as_ptr() as *const c_char, id.len()) },
            FfiStatus::InvalidArgument.code()
        );
        // Unknown handles.
        assert_eq!(
            unsafe {
                ffi_writer_add_term_vector_field(
                    0xDEAD_BEEF,
                    body.as_ptr() as *const c_char,
                    body.len(),
                )
            },
            FfiStatus::InvalidHandle.code()
        );
        assert_eq!(
            unsafe {
                ffi_writer_omit_norms_field(0xDEAD_BEEF, body.as_ptr() as *const c_char, body.len())
            },
            FfiStatus::InvalidHandle.code()
        );
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Norms are written for an indexed field with no norms call at all --
    /// Lucene's default -- and `ffi_writer_omit_norms_field` is what takes
    /// them away again. Before c35 the FFI had only the opt-*in*, so a caller
    /// that never made the call got a segment whose `body` field scored
    /// against a constant length.
    #[test]
    fn norms_are_written_by_default_and_omit_norms_field_removes_them() {
        for omit in [false, true] {
            let tmp = tempdir(if omit { "norms-omit" } else { "norms-default" });
            let (rc, handle) = open_test_writer_with_extra_field(&tmp, "body", 2, 0, 0);
            assert_eq!(rc, FfiStatus::Ok.code());
            let body = "body";
            assert_eq!(
                unsafe {
                    ffi_writer_set_postings_field(
                        handle,
                        1,
                        body.as_ptr() as *const c_char,
                        body.len(),
                    )
                },
                FfiStatus::Ok.code()
            );
            if omit {
                assert_eq!(
                    unsafe {
                        ffi_writer_omit_norms_field(
                            handle,
                            body.as_ptr() as *const c_char,
                            body.len(),
                        )
                    },
                    FfiStatus::Ok.code()
                );
            }
            for (id, text) in [("1", "a b c"), ("2", "a")] {
                assert_eq!(add_doc_id_and_extra(handle, id, text), FfiStatus::Ok.code());
            }
            assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
            assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());

            let dir = FsDirectory::open(tmp.to_str().unwrap());
            let sis = segment_infos::read_latest(&dir).unwrap();
            let sci = &sis.segments[0];
            let nvm = dir.open(&format!("{}.nvm", sci.segment_name));
            if omit {
                assert!(nvm.is_err(), "an omitted field writes no norms at all");
            } else {
                let nvm = nvm.unwrap();
                let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
                let (_version, norms) =
                    lucene_codecs::norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
                assert!(
                    norms.entry(1).is_some(),
                    "field 1 (`body`) must have a norms entry with no norms call at all"
                );
                assert!(!nvd.is_empty());
            }
            let _ = std::fs::remove_dir_all(&tmp);
        }
    }

    /// The custom-freq postings surface: opting a field in and supplying
    /// per-term freqs, plus the boundary's own `freq >= 1` check.
    #[test]
    fn custom_freq_postings_field_and_per_document_terms_round_trip() {
        let tmp = tempdir("custom-freq");
        // index_options 5 == DocsAndCustomFreqs, the only options
        // `set_custom_freq_postings_field` accepts.
        let (rc, handle) = open_test_writer_with_extra_field(&tmp, "body", 5, 0, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        let body = "body";
        assert_eq!(
            unsafe {
                ffi_writer_set_custom_freq_postings_field(
                    handle,
                    1,
                    body.as_ptr() as *const c_char,
                    body.len(),
                )
            },
            FfiStatus::Ok.code()
        );

        let value = "1";
        let numbers = [0i32];
        let kinds = [0u8];
        let ptrs = [value.as_ptr()];
        let lens = [value.len()];
        let terms = ["alpha", "beta"];
        let term_ptrs: Vec<*const u8> = terms.iter().map(|t| t.as_ptr()).collect();
        let term_lens: Vec<usize> = terms.iter().map(|t| t.len()).collect();
        let freqs = [3i32, 1];
        let mut seq = 0i64;
        let rc = unsafe {
            ffi_writer_add_document_with_custom_freq_terms(
                handle,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                1,
                term_ptrs.as_ptr(),
                term_lens.as_ptr(),
                freqs.as_ptr(),
                2,
                &mut seq as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(seq > 0);

        // A zero freq has no wire encoding and must be refused up front.
        let bad_freqs = [0i32, 1];
        let rc = unsafe {
            ffi_writer_add_document_with_custom_freq_terms(
                handle,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                1,
                term_ptrs.as_ptr(),
                term_lens.as_ptr(),
                bad_freqs.as_ptr(),
                2,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        // Null term arrays with a non-zero count.
        let rc = unsafe {
            ffi_writer_add_document_with_custom_freq_terms(
                handle,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                1,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                1,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        // Unknown handle.
        let rc = unsafe {
            ffi_writer_add_document_with_custom_freq_terms(
                0xDEAD_BEEF,
                numbers.as_ptr(),
                kinds.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                1,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());

        assert_eq!(ffi_writer_commit(handle), FfiStatus::Ok.code());
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
        let dir = FsDirectory::open(tmp.to_str().unwrap());
        assert_eq!(read_all_live_ids(&dir), vec!["1"]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn custom_freq_and_plain_postings_fields_are_mutually_exclusive() {
        let tmp = tempdir("custom-freq-exclusive");
        let (rc, handle) = open_test_writer_with_extra_field(&tmp, "body", 5, 0, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        let body = "body";
        assert_eq!(
            unsafe {
                ffi_writer_set_custom_freq_postings_field(
                    handle,
                    1,
                    body.as_ptr() as *const c_char,
                    body.len(),
                )
            },
            FfiStatus::Ok.code()
        );
        // Both kinds of postings on one writer is exactly what the writer
        // refuses -- and it must arrive as an argument error.
        assert_eq!(
            unsafe {
                ffi_writer_add_postings_field(handle, body.as_ptr() as *const c_char, body.len())
            },
            FfiStatus::InvalidArgument.code()
        );
        assert_eq!(
            unsafe {
                ffi_writer_set_postings_field(handle, 1, body.as_ptr() as *const c_char, body.len())
            },
            FfiStatus::InvalidArgument.code()
        );
        assert_eq!(
            unsafe {
                ffi_writer_set_custom_freq_postings_field(0xDEAD_BEEF, 0, std::ptr::null(), 0)
            },
            FfiStatus::InvalidHandle.code()
        );
        assert_eq!(ffi_close_writer(handle), FfiStatus::Ok.code());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
