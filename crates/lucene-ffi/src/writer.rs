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
//! [`IndexWriter::update_document`]/[`IndexWriter::delete_documents`]. Both
//! identify their delete term as raw, already-analyzed `(field_name, term)`
//! bytes -- no analysis happens at this FFI boundary, same stance this
//! crate's `query.rs` already takes for its own raw-bytes terms.
//! `ffi_writer_update_document`'s replacement document reuses
//! `ffi_writer_add_document`'s exact parallel-array field encoding. Because
//! [`IndexWriter::update_document`]/[`IndexWriter::delete_documents`] need a
//! [`lucene_index::update_document::SegmentDeleteSource`] per segment the
//! caller wants the term resolved against, this module builds those itself
//! via [`open_all_segment_sources`]/[`build_delete_sources`]: every
//! currently-committed segment with a `.tim` file on disk (i.e. flushed with
//! [`ffi_writer_set_postings_field`] enabled at that commit) is reopened
//! fresh from the writer's own directory; a segment with no `.tim` file is
//! skipped, not errored, matching `SegmentDeleteSource`'s own "unlisted
//! segment left untouched" contract. Neither call is buffered -- both commit
//! immediately on success, matching the wrapped methods' own atomicity.
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
//! larger task. `set_merge_policy` itself only exposes the four
//! knobs [`lucene_index::merge_policy::MergePolicyConfig`] actually has today
//! (`max_merge_at_once`, `segments_per_tier`, `max_merged_segment_size`,
//! `reclaim_weight`) -- no additional `TieredMergePolicy` knobs (e.g.
//! `forceMergeDeletesPctAllowed`, `floorSegmentMB`) are invented, since none
//! exist in this port's `merge_policy.rs` to expose.

use std::os::raw::c_char;

use lucene_codecs::blocktree::{self, BlockTreeFields};
use lucene_codecs::field_infos::FieldInfos;
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::postings::DocInput;
use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_index::deletes;
use lucene_index::index_writer::{self, IndexWriter, MergePolicyConfig};
use lucene_index::segment_info::LuceneVersion;
use lucene_index::update_document::SegmentDeleteSource;
use lucene_store::directory::{Directory, FsDirectory};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::{bytes_from_raw, str_from_raw};
use crate::registry::{lock_recovering, writers, WriterHandle};

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
    let name = unsafe { str_from_raw(field_name as *const u8, field_name_len)? };
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
        | index_writer::Error::UnknownPostingsField(_)
        | index_writer::Error::UnsupportedPostingsIndexOptions(_, _)
        | index_writer::Error::UnknownTermVectorField(_)
        | index_writer::Error::UnsupportedTermVectorField(_)
        | index_writer::Error::UnknownDocValuesField(_)
        | index_writer::Error::UnsupportedDocValuesType(_, _)
        | index_writer::Error::MissingDenseDocValue(_, _)
        | index_writer::Error::NonNumericDocValue(_, _, _)
        | index_writer::Error::NonBinaryDocValue(_, _, _) => FfiStatus::InvalidArgument,
        _ => FfiStatus::Io,
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
                str_from_raw(path as *const u8, path_len)?,
                str_from_raw(codec_name as *const u8, codec_name_len)?,
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

            let mut fields = Vec::with_capacity(field_count);
            for i in 0..field_count {
                // SAFETY: caller contract guarantees `names[i]` is valid for
                // `name_lens[i]` bytes.
                let name = unsafe { str_from_raw(names[i], name_lens[i])? };
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
        let handle = lock_recovering(writers()).insert(handle);
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
) -> i32 {
    guard(|| {
        let mut fields = Vec::with_capacity(field_count);
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
        handle.writer.add_document(Document { fields });
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
/// `reclaim_weight` map straight onto
/// [`lucene_index::merge_policy::MergePolicyConfig`]'s four fields -- the
/// only merge-policy knobs this port's `merge_policy.rs` actually
/// implements today (no `floorSegmentMB`/`forceMergeDeletesPctAllowed`/etc,
/// since real `TieredMergePolicy` has those but this port's
/// `MergePolicyConfig` does not -- see this module's doc comment). Ignored
/// (but still validated as present) when `enabled == 0`.
#[no_mangle]
pub extern "C" fn ffi_writer_set_merge_policy(
    writer_handle: u64,
    enabled: u8,
    max_merge_at_once: u64,
    segments_per_tier: u64,
    max_merged_segment_size: u64,
    reclaim_weight: f64,
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
            Some(MergePolicyConfig {
                max_merge_at_once: max_merge_at_once as usize,
                segments_per_tier: segments_per_tier as usize,
                max_merged_segment_size,
                reclaim_weight,
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

/// One committed segment's postings/live-docs bytes, reopened fresh from
/// `dir` -- the owned backing storage a [`SegmentDeleteSource`] built by
/// [`open_all_segment_sources`] borrows from for the lifetime of one
/// [`ffi_writer_update_document`]/[`ffi_writer_delete_documents`] call.
struct OpenedSegmentSource {
    segment_name: String,
    segment_id: [u8; lucene_store::codec_util::ID_LENGTH],
    tim: lucene_store::directory::Input,
    tip: lucene_store::directory::Input,
    tmd: lucene_store::directory::Input,
    doc: lucene_store::directory::Input,
    live_docs: Option<FixedBitSet>,
    max_doc: usize,
}

/// Reopens every currently-committed segment of `writer` that has a `.tim`
/// file -- i.e. was flushed by a commit with
/// [`ffi_writer_set_postings_field`] enabled -- so
/// [`ffi_writer_update_document`]/[`ffi_writer_delete_documents`] can build a
/// [`SegmentDeleteSource`] per segment. A segment with no `.tim` file is
/// skipped entirely (not an error): [`IndexWriter::update_document`]/
/// [`IndexWriter::delete_documents`] already treat a segment absent from
/// `delete_sources` as "left untouched" (see those methods' own doc
/// comments), so a stored-fields-only segment simply never has its docs
/// considered for this delete/update -- matching real Lucene's own
/// `BufferedUpdates` timing limitation this port documents (a delete only
/// ever resolves against segments whose postings are actually available to
/// resolve it against).
fn open_all_segment_sources(writer: &IndexWriter) -> Result<Vec<OpenedSegmentSource>, FfiStatus> {
    let dir = writer.dir();
    let all_files = dir.list_all().map_err(|e| {
        set_last_error(format!("ffi_writer: listing directory: {e}"));
        FfiStatus::Io
    })?;

    let read = |name: &str| -> Result<lucene_store::directory::Input, FfiStatus> {
        dir.open(name).map_err(|e| {
            set_last_error(format!("ffi_writer: opening {name}: {e}"));
            FfiStatus::Io
        })
    };

    let mut out = Vec::new();
    for sci in &writer.segment_infos().segments {
        let tim_name = format!("{}.tim", sci.segment_name);
        if !all_files.iter().any(|f| f == &tim_name) {
            continue;
        }
        let tim = read(&tim_name)?;
        let tip = read(&format!("{}.tip", sci.segment_name))?;
        let tmd = read(&format!("{}.tmd", sci.segment_name))?;
        let doc = read(&format!("{}.doc", sci.segment_name))?;

        let fdt = read(&format!("{}.fdt", sci.segment_name))?;
        let fdx = read(&format!("{}.fdx", sci.segment_name))?;
        let fdm = read(&format!("{}.fdm", sci.segment_name))?;
        let max_doc = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "")
            .map_err(|e| {
                set_last_error(format!(
                    "ffi_writer: opening stored fields for {}: {e}",
                    sci.segment_name
                ));
                FfiStatus::Io
            })?
            .max_doc() as usize;

        let live_docs = if sci.del_gen >= 0 {
            let liv = read(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))?;
            let parsed = lucene_codecs::live_docs::parse(
                &liv,
                &sci.segment_id,
                sci.del_gen,
                max_doc,
                sci.del_count as usize,
            )
            .map_err(|e| {
                set_last_error(format!(
                    "ffi_writer: parsing live docs for {}: {e}",
                    sci.segment_name
                ));
                FfiStatus::Io
            })?;
            Some(parsed)
        } else {
            None
        };

        out.push(OpenedSegmentSource {
            segment_name: sci.segment_name.clone(),
            segment_id: sci.segment_id,
            tim,
            tip,
            tmd,
            doc,
            live_docs,
            max_doc,
        });
    }
    Ok(out)
}

/// Builds one [`SegmentDeleteSource`] per [`OpenedSegmentSource`], resolving
/// each segment's [`BlockTreeFields`]/[`DocInput`] against `field_infos` --
/// the shared setup [`ffi_writer_update_document`]/
/// [`ffi_writer_delete_documents`] both need before calling into
/// [`IndexWriter::update_document`]/[`IndexWriter::delete_documents`].
fn build_delete_sources<'a>(
    opened: &'a [OpenedSegmentSource],
    field_infos: &FieldInfos,
    blocktree_fields: &'a mut Vec<BlockTreeFields>,
    doc_inputs: &'a mut Vec<DocInput<'a>>,
) -> Result<Vec<SegmentDeleteSource<'a>>, FfiStatus> {
    for seg in opened {
        let fields = blocktree::open(
            &seg.tim,
            &seg.tip,
            &seg.tmd,
            field_infos,
            &seg.segment_id,
            "",
            seg.max_doc as i32,
        )
        .map_err(|e| {
            set_last_error(format!(
                "ffi_writer: opening blocktree fields for {}: {e}",
                seg.segment_name
            ));
            FfiStatus::Io
        })?;
        blocktree_fields.push(fields);

        let doc_in = DocInput::open(&seg.doc, &seg.segment_id, "").map_err(|e| {
            set_last_error(format!(
                "ffi_writer: opening doc input for {}: {e}",
                seg.segment_name
            ));
            FfiStatus::Io
        })?;
        doc_inputs.push(doc_in);
    }

    Ok(opened
        .iter()
        .zip(blocktree_fields.iter())
        .zip(doc_inputs.iter())
        .map(|((seg, fields), doc_in)| SegmentDeleteSource {
            segment_name: &seg.segment_name,
            fields,
            doc_in: Some(doc_in),
            live_docs: seg.live_docs.as_ref(),
            max_doc: seg.max_doc,
        })
        .collect())
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
/// Delete resolution only reaches segments that already have a `.tim` file
/// on disk -- see [`open_all_segment_sources`]'s doc comment for exactly why
/// (a segment this writer flushed with no [`ffi_writer_set_postings_field`]
/// enabled at commit time has no postings to search, so it is left
/// untouched, matching [`IndexWriter::update_document`]'s own "no matching
/// source" contract). Unlike [`ffi_writer_add_document`], this commits
/// immediately on success -- it is **not** buffered (see
/// [`IndexWriter::update_document`]'s own doc comment for why).
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
) -> i32 {
    guard(|| {
        // SAFETY: caller contract guarantees `field_name` is valid for
        // `field_name_len` bytes.
        let field = unsafe { str_from_raw(field_name as *const u8, field_name_len)? };
        // SAFETY: caller contract guarantees `term_ptr` is valid for
        // `term_len` bytes (or null iff `term_len == 0`).
        let term = unsafe { bytes_from_raw(term_ptr, term_len)? };

        let mut new_fields = Vec::with_capacity(field_count);
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

        let opened = open_all_segment_sources(&handle.writer)?;
        let field_infos = FieldInfos {
            fields: handle.writer.fields().to_vec(),
        };
        let mut blocktree_fields = Vec::new();
        let mut doc_inputs = Vec::new();
        let sources = build_delete_sources(
            &opened,
            &field_infos,
            &mut blocktree_fields,
            &mut doc_inputs,
        )?;

        handle
            .writer
            .update_document(&sources, field, term, Document { fields: new_fields })
            .map(|_| ())
            .map_err(|e| map_writer_error("ffi_writer_update_document", e))
    })
}

/// Deletes every live doc matching `(field_name, term)` -- see
/// [`IndexWriter::delete_documents`]. `field_name`/`field_name_len` and
/// `term_ptr`/`term_len` use the exact same raw-bytes term convention as
/// [`ffi_writer_update_document`] (see its own doc comment); delete
/// resolution likewise only reaches segments with a `.tim` file already on
/// disk (see [`open_all_segment_sources`]). Commits immediately on success,
/// same as [`ffi_writer_update_document`].
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
) -> i32 {
    guard(|| {
        // SAFETY: caller contract guarantees `field_name` is valid for
        // `field_name_len` bytes.
        let field = unsafe { str_from_raw(field_name as *const u8, field_name_len)? };
        // SAFETY: caller contract guarantees `term_ptr` is valid for
        // `term_len` bytes (or null iff `term_len == 0`).
        let term = unsafe { bytes_from_raw(term_ptr, term_len)? };

        let mut registry = lock_recovering(writers());
        let handle = registry.get_mut(writer_handle).ok_or_else(|| {
            set_last_error("ffi_writer_delete_documents: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;

        let opened = open_all_segment_sources(&handle.writer)?;
        let field_infos = FieldInfos {
            fields: handle.writer.fields().to_vec(),
        };
        let mut blocktree_fields = Vec::new();
        let mut doc_inputs = Vec::new();
        let sources = build_delete_sources(
            &opened,
            &field_infos,
            &mut blocktree_fields,
            &mut doc_inputs,
        )?;

        handle
            .writer
            .delete_documents(&sources, field, term)
            .map(|_| ())
            .map_err(|e| map_writer_error("ffi_writer_delete_documents", e))
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
        let registry = lock_recovering(writers());
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
        let registry = lock_recovering(writers());
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
        let registry = lock_recovering(writers());
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
    use lucene_index::segment_infos;

    fn tempdir(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lucene-ffi-writer-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
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
            )
        }
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
        let tim = dir.open(&format!("{}.tim", sci.segment_name)).unwrap();
        let tip = dir.open(&format!("{}.tip", sci.segment_name)).unwrap();
        let tmd = dir.open(&format!("{}.tmd", sci.segment_name)).unwrap();
        let doc_bytes = dir.open(&format!("{}.doc", sci.segment_name)).unwrap();
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
        let block_fields =
            lucene_codecs::blocktree::open(&tim, &tip, &tmd, &field_infos, &sci.segment_id, "", 2)
                .expect("blocktree::open on FFI-produced .tim/.tip/.tmd");
        let doc_in = lucene_codecs::postings::DocInput::open(&doc_bytes, &sci.segment_id, "")
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
        let dvm = dir.open(&format!("{}.dvm", sci.segment_name)).unwrap();
        let dvd = dir.open(&format!("{}.dvd", sci.segment_name)).unwrap();
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
        let (_, meta) =
            lucene_codecs::doc_values::parse_meta(&dvm, &sci.segment_id, "", &field_infos)
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
            ffi_writer_set_merge_policy(handle, 1, 2, 2, 5_000 * 1024 * 1024, 1.0),
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
            ffi_writer_set_merge_policy(handle, 0, 0, 0, 0, 0.0),
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
            ffi_writer_set_merge_policy(0xDEAD_BEEF, 1, 2, 2, 1024, 1.0),
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
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        // `update_document` commits immediately -- no separate
        // `ffi_writer_commit` call needed before the on-disk state reflects
        // the replacement, but this module's own read-back must go through
        // a fresh `segments_N` read, which is what `read_all_live_ids` does.

        let dir = FsDirectory::open(&tmp);
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
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let dir = FsDirectory::open(&tmp);
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
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn delete_documents_null_field_name_is_null_pointer_error() {
        let term = b"x";
        let rc = unsafe {
            ffi_writer_delete_documents(0xDEAD_BEEF, std::ptr::null(), 1, term.as_ptr(), term.len())
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn delete_documents_on_a_writer_with_no_postings_segments_is_a_no_op() {
        // A writer whose only committed segment was flushed stored-only (no
        // `set_postings_field`) has no `.tim` file at all --
        // `open_all_segment_sources` must skip it (not error), leaving
        // `delete_documents` with an empty `delete_sources` list -- exactly
        // the "no matching source" no-op path.
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
        // returning empty, not `open_all_segment_sources` skipping the
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
        // matches a doc in each. Proves `open_all_segment_sources`/
        // `build_delete_sources` resolve every committed segment, not just
        // the most recent one.
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
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

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
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

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
}
