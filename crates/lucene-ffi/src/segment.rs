//! `ffi_open_segment`/`ffi_close_segment`: opens one segment's term
//! dictionary (`.tim`/`.tip`/`.tmd` via `blocktree::open`) plus whichever
//! postings files (`.doc`, and optionally `.pos`/`.pay` for phrase queries)
//! the caller names, all read from an already-open [`crate::directory`]
//! handle.
//!
//! **Why explicit file names instead of re-deriving them from a `.si`**:
//! this port has no unified "open every file a `.si` names" reader yet (see
//! `lucene-search/src/lib.rs`'s module doc, "no `SegmentReader`/
//! `DirectoryReader` abstraction exists yet") — building one is its own
//! task. A real caller (the JNI wrapper, out of this repo's scope) already
//! has to parse `segments_N`/`.si` to discover segment names, field
//! numbers, and the segment ID/suffix before it can call anything in this
//! crate; passing those already-known values straight through keeps this
//! entry point composable with whatever read-path abstraction lands later,
//! rather than baking a `.si` parse into the FFI boundary itself.
use std::os::raw::c_char;

use lucene_codecs::blocktree::{self, BlockTreeFields};
use lucene_codecs::doc_values;
use lucene_codecs::field_infos;
use lucene_codecs::live_docs;
use lucene_codecs::norms;
use lucene_codecs::points;
use lucene_codecs::postings::{DocInput, PayInput, PosInput};

use crate::directory::read_whole_file;
use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::str_from_raw;
use crate::registry::{
    lock_recovering, read_recovering, segments, DocValuesGenerationColumn, SegmentHandle,
};

/// Opens one segment's term dictionary and postings files.
///
/// - `dir_handle`: an [`crate::directory::ffi_open_directory`] handle.
/// - `fnm_name`/`tim_name`/`tip_name`/`tmd_name` (each a `(*const u8, len)`
///   pair): the segment's `.fnm`/`.tim`/`.tip`/`.tmd` file names, required.
/// - `nvm_name`/`nvd_name` (task #30): the segment's `.nvm`/`.nvd` (norms
///   metadata/data) file names, or a null pointer (any `len`) to open
///   neither -- required for a scored query (`ffi_search_*_query_scored`) to
///   use real per-doc/avg field lengths rather than falling back to
///   `lucene_search::similarity::UNNORMED_FIELD_LENGTH`; a scored query
///   against a segment opened without norms still succeeds, just with that
///   constant-length approximation, same as passing `norms: None` directly
///   to `lucene_search`'s scored functions. Unlike `.doc`/`.pos`/`.tim`/etc,
///   real Lucene's norms files are named from the segment name alone (no
///   codec-suffix component), so `nvm_name`/`nvd_name` are ordinary caller-
///   supplied names like every other file name here, not derived from
///   `segment_suffix`.
/// - `dvm_name`/`dvd_name`/`dv_suffix` (task #40): the segment's `.dvm`/`.dvd`
///   (doc-values metadata/data) file names, or a null pointer (any `len`) to
///   open neither -- required for `ffi_sort_by_doc_value`/
///   `ffi_sort_by_multi_valued_doc_value` (`sort.rs`) to have any values to
///   sort by; a segment opened without doc values simply can't serve those
///   two calls ([`crate::error::FfiStatus::InvalidArgument`], since unlike
///   norms there is no meaningful "sort by a constant" fallback). Doc-values
///   files *do* carry a per-field codec-suffix component in their index
///   header (like `.tim`/`.doc`, unlike `.nvm`/`.nvd`), so `dv_suffix` is a
///   separate parameter from `segment_suffix` -- real Lucene's doc-values
///   codec suffix and postings codec suffix are independent strings, not
///   guaranteed equal, even though this port's own fixtures happen to reuse
///   the same value for both.
/// - `kdm_name`/`kdi_name`/`kdd_name` (Points range query FFI exposure): the
///   segment's `.kdm`/`.kdi`/`.kdd` (BKD points meta/index/data) file names,
///   or a null pointer (any `len`) to open none -- required for
///   `ffi_search_points_range` (`points_query.rs`) to have any points data to
///   search; a segment opened without them simply can't serve that call
///   ([`FfiStatus::InvalidArgument`], same as `.dvm`/`.dvd`'s "no sensible
///   fallback" story, unlike norms). Like `.nvm`/`.nvd`, real Lucene's points
///   format has no per-field codec-suffix component in its index header, so
///   these are always validated against the empty suffix (`""`), not this
///   segment's postings `suffix` or `dv_suffix` -- matching
///   `lucene_codecs::points`'s own tests, which always write/open with `""`.
///   All three names must be non-null together, or all null -- one without
///   the others leaves points data unavailable for this segment, same
///   "opened together or not at all" convention as `.nvm`/`.nvd`/`.dvm`/`.dvd`.
/// - `segment_id`: the segment's 16-byte ID (`SegmentInfo.getId()`).
/// - `segment_suffix`: the codec suffix string used in every file's index
///   header (often empty).
/// - `max_doc`: the segment's `SegmentInfo.maxDoc()`.
///
/// Writes the new segment handle to `*out_handle` on success.
///
/// # Safety
/// Every `(*const u8, len)` pointer pair must be valid for reads of `len`
/// bytes (or null when explicitly allowed above); `out_handle` must be
/// valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_open_segment(
    dir_handle: u64,
    fnm_name: *const c_char,
    fnm_name_len: usize,
    tim_name: *const c_char,
    tim_name_len: usize,
    tip_name: *const c_char,
    tip_name_len: usize,
    tmd_name: *const c_char,
    tmd_name_len: usize,
    doc_name: *const c_char,
    doc_name_len: usize,
    pos_name: *const c_char,
    pos_name_len: usize,
    pay_name: *const c_char,
    pay_name_len: usize,
    nvm_name: *const c_char,
    nvm_name_len: usize,
    nvd_name: *const c_char,
    nvd_name_len: usize,
    dvm_name: *const c_char,
    dvm_name_len: usize,
    dvd_name: *const c_char,
    dvd_name_len: usize,
    dv_suffix: *const c_char,
    dv_suffix_len: usize,
    kdm_name: *const c_char,
    kdm_name_len: usize,
    kdi_name: *const c_char,
    kdi_name_len: usize,
    kdd_name: *const c_char,
    kdd_name_len: usize,
    segment_id: *const u8,
    segment_suffix: *const c_char,
    segment_suffix_len: usize,
    max_doc: i32,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() || segment_id.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // `max_doc` is `SegmentInfo.maxDoc()`, a non-negative count. It is
        // threaded straight into `blocktree::open` and into every later
        // `0..max_doc` doc-ID range check on this handle, so a negative value
        // from a caller bug would make those ranges empty (a segment that
        // silently matches nothing) rather than failing loudly.
        if max_doc < 0 {
            set_last_error(format!(
                "ffi_open_segment: max_doc {max_doc} is negative (SegmentInfo.maxDoc() is a count)"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        // SAFETY: caller contract guarantees each name pointer is valid for its
        // paired length, and `segment_id` is valid for 16 bytes.
        let (fnm_name, tim_name, tip_name, tmd_name, suffix) = unsafe {
            (
                str_from_raw(fnm_name, fnm_name_len)?,
                str_from_raw(tim_name, tim_name_len)?,
                str_from_raw(tip_name, tip_name_len)?,
                str_from_raw(tmd_name, tmd_name_len)?,
                str_from_raw(segment_suffix, segment_suffix_len)?,
            )
        };
        let mut id = [0u8; 16];
        // SAFETY: caller contract guarantees `segment_id` is valid for 16 bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(segment_id, id.as_mut_ptr(), 16);
        }

        let fnm = read_whole_file(dir_handle, fnm_name)?;
        let field_infos = field_infos::parse(&fnm, &id, "").map_err(|e| {
            set_last_error(format!("parsing .fnm: {e}"));
            FfiStatus::Decode
        })?;

        let tim = read_whole_file(dir_handle, tim_name)?;
        let tip = read_whole_file(dir_handle, tip_name)?;
        let tmd = read_whole_file(dir_handle, tmd_name)?;
        let fields: BlockTreeFields =
            blocktree::open(&tim, &tip, &tmd, &field_infos, &id, suffix, max_doc).map_err(|e| {
                set_last_error(format!("opening term dictionary: {e}"));
                FfiStatus::Decode
            })?;

        let doc_bytes = if doc_name.is_null() {
            None
        } else {
            // SAFETY: caller contract guarantees `doc_name` is valid for `doc_name_len`.
            let name = unsafe { str_from_raw(doc_name, doc_name_len)? };
            let bytes = read_whole_file(dir_handle, name)?;
            DocInput::open(&bytes, &id, suffix).map_err(|e| {
                set_last_error(format!("opening .doc: {e}"));
                FfiStatus::Decode
            })?;
            Some(bytes)
        };

        let pos_bytes = if pos_name.is_null() {
            None
        } else {
            // SAFETY: caller contract guarantees `pos_name` is valid for `pos_name_len`.
            let name = unsafe { str_from_raw(pos_name, pos_name_len)? };
            let bytes = read_whole_file(dir_handle, name)?;
            PosInput::open(&bytes, &id, suffix).map_err(|e| {
                set_last_error(format!("opening .pos: {e}"));
                FfiStatus::Decode
            })?;
            Some(bytes)
        };

        let pay_bytes = if pay_name.is_null() {
            None
        } else {
            // SAFETY: caller contract guarantees `pay_name` is valid for `pay_name_len`.
            let name = unsafe { str_from_raw(pay_name, pay_name_len)? };
            let bytes = read_whole_file(dir_handle, name)?;
            PayInput::open(&bytes, &id, suffix).map_err(|e| {
                set_last_error(format!("opening .pay: {e}"));
                FfiStatus::Decode
            })?;
            Some(bytes)
        };

        // Task #30: `.nvm`/`.nvd` are opened together or not at all -- a
        // caller that passes one but not the other gets the same "null means
        // none" behavior as `doc_name`/`pos_name` for whichever one is null,
        // but `norms`/`norms_data` are only ever both `Some` or both `None`
        // (see `registry.rs`'s `SegmentHandle` doc comment), so a lone
        // `nvm_name` with a null `nvd_name` (or vice versa) parses/validates
        // only its own file and leaves norms unavailable for this segment,
        // same as passing neither.
        let (norms, norms_data) = if nvm_name.is_null() || nvd_name.is_null() {
            (None, None)
        } else {
            // SAFETY: caller contract guarantees `nvm_name`/`nvd_name` are valid
            // for their paired lengths.
            let (nvm, nvd) = unsafe {
                (
                    str_from_raw(nvm_name, nvm_name_len)?,
                    str_from_raw(nvd_name, nvd_name_len)?,
                )
            };
            // Real Lucene's norms format (`Lucene90NormsFormat`) has no
            // per-field codec-suffix component in its index header, unlike
            // `.tim`/`.doc`/etc -- always validate against the empty suffix,
            // not this segment's postings `suffix` (see
            // `ffi_open_segment`'s doc comment and
            // `lucene-search/tests/scoring_fixtures.rs`'s `open_body_norms`,
            // which does the same `""` for its differential norms test).
            let meta_bytes = read_whole_file(dir_handle, nvm)?;
            let (_version, parsed) = norms::parse_meta(&meta_bytes, &id, "").map_err(|e| {
                set_last_error(format!("parsing .nvm: {e}"));
                FfiStatus::Decode
            })?;
            let data_bytes = read_whole_file(dir_handle, nvd)?;
            norms::check_data_header_footer(&data_bytes, &id, "").map_err(|e| {
                set_last_error(format!("opening .nvd: {e}"));
                FfiStatus::Decode
            })?;
            (Some(parsed), Some(data_bytes))
        };

        // Task #40: `.dvm`/`.dvd` are opened together or not at all -- same
        // "null means none, one without the other leaves it unavailable"
        // convention as `.nvm`/`.nvd` above.
        let (dv_meta, dv_data) = if dvm_name.is_null() || dvd_name.is_null() {
            (None, None)
        } else {
            // SAFETY: caller contract guarantees `dvm_name`/`dvd_name`/`dv_suffix`
            // are valid for their paired lengths.
            let (dvm, dvd, dv_suffix) = unsafe {
                (
                    str_from_raw(dvm_name, dvm_name_len)?,
                    str_from_raw(dvd_name, dvd_name_len)?,
                    str_from_raw(dv_suffix, dv_suffix_len)?,
                )
            };
            let meta_bytes = read_whole_file(dir_handle, dvm)?;
            let (_version, parsed) =
                doc_values::parse_meta(&meta_bytes, &id, dv_suffix, &field_infos).map_err(|e| {
                    set_last_error(format!("parsing .dvm: {e}"));
                    FfiStatus::Decode
                })?;
            let data_bytes = read_whole_file(dir_handle, dvd)?;
            doc_values::check_data_header_footer(&data_bytes, &id, dv_suffix).map_err(|e| {
                set_last_error(format!("opening .dvd: {e}"));
                FfiStatus::Decode
            })?;
            (Some(parsed), Some(data_bytes))
        };

        // Points range query FFI exposure: `.kdm`/`.kdi`/`.kdd` are opened
        // together or not at all -- same "null means none, one without the
        // others leaves it unavailable" convention as `.nvm`/`.nvd` and
        // `.dvm`/`.dvd` above. Validated once here (via `points::open`, then
        // discarded -- see `SegmentHandle::points_data`'s doc comment for why
        // a fresh reader is reconstructed per query call instead) so a
        // corrupt file surfaces as `FfiStatus::Decode` at open time, same as
        // every other file this function opens.
        let points_data = if kdm_name.is_null() || kdi_name.is_null() || kdd_name.is_null() {
            None
        } else {
            // SAFETY: caller contract guarantees `kdm_name`/`kdi_name`/`kdd_name`
            // are valid for their paired lengths.
            let (kdm, kdi, kdd) = unsafe {
                (
                    str_from_raw(kdm_name, kdm_name_len)?,
                    str_from_raw(kdi_name, kdi_name_len)?,
                    str_from_raw(kdd_name, kdd_name_len)?,
                )
            };
            let kdm_bytes = read_whole_file(dir_handle, kdm)?;
            let kdi_bytes = read_whole_file(dir_handle, kdi)?;
            let kdd_bytes = read_whole_file(dir_handle, kdd)?;
            // Empty suffix -- real Lucene's points format has no per-field
            // codec-suffix component in its index header, same as `.nvm`/`.nvd`
            // above (see this function's doc comment).
            points::open(&kdm_bytes, &kdi_bytes, &kdd_bytes, &id, "").map_err(|e| {
                set_last_error(format!("opening points data: {e}"));
                FfiStatus::Decode
            })?;
            Some((kdm_bytes, kdi_bytes, kdd_bytes))
        };

        let handle = lock_recovering(segments()).insert_checked(SegmentHandle {
            fields,
            doc_bytes,
            pos_bytes,
            pay_bytes,
            segment_id: id,
            segment_suffix: suffix.to_string(),
            max_doc,
            field_infos,
            norms_data,
            norms,
            dv_data,
            dv_meta,
            points_data,
            // Deletions are attached separately -- see
            // `ffi_segment_set_live_docs` and `SegmentHandle::live_docs`'s
            // doc comment for why this is not another `ffi_open_segment`
            // parameter.
            live_docs: None,
            // Doc-values update generations are attached separately, for the
            // same reason `.liv` is -- see
            // `ffi_segment_add_doc_values_generation`.
            dv_generations: Vec::new(),
        })?;
        // SAFETY: caller contract guarantees `out_handle` is valid for one write.
        unsafe {
            *out_handle = handle;
        }
        Ok(())
    })
}

/// Reads and decodes one `.liv` (live docs / deletions) file, or `None` when
/// `liv_name` is null ("this segment has no deletions").
///
/// Shared by [`ffi_segment_set_live_docs`] and
/// [`crate::vectors::ffi_vectors_set_live_docs`] so the two handle kinds
/// cannot drift apart on validation: `del_gen >= 1`, `del_count >= 0`,
/// `max_doc >= 0`, and then `live_docs::parse`'s own cross-check of the
/// decoded bitset's cardinality against `del_count` (so a `.liv` that
/// disagrees with the commit is a decode error, not silently wrong results).
///
/// `what` is the calling entry point's name, for the last-error message.
///
/// Does its I/O *without* holding either handle registry's guard -- the
/// caller copies out `segment_id`/`max_doc` under a scoped read guard, drops
/// it, calls this, and only then takes the write guard. `std`'s `RwLock`
/// neither re-enters nor upgrades, so overlapping the two would deadlock the
/// calling thread.
///
/// # Safety
/// `liv_name` must be valid for reads of `liv_name_len` bytes, or null.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn decode_live_docs(
    what: &str,
    dir_handle: u64,
    liv_name: *const c_char,
    liv_name_len: usize,
    del_gen: i64,
    del_count: i32,
    segment_id: &[u8; 16],
    max_doc: i32,
) -> Result<Option<lucene_util::fixed_bit_set::FixedBitSet>, FfiStatus> {
    if liv_name.is_null() {
        return Ok(None);
    }
    // SAFETY: caller contract guarantees `liv_name` is valid for
    // `liv_name_len` bytes.
    let name = unsafe { str_from_raw(liv_name, liv_name_len)? };
    if del_gen < 1 {
        set_last_error(format!(
            "{what}: del_gen {del_gen} is not a live-docs generation (a segment with no \
             deletions has del_gen -1 and no .liv file -- pass a null liv_name to clear)"
        ));
        return Err(FfiStatus::InvalidArgument);
    }
    if del_count < 0 {
        set_last_error(format!("{what}: del_count {del_count} is negative"));
        return Err(FfiStatus::InvalidArgument);
    }
    if max_doc < 0 {
        set_last_error(format!("{what}: segment max_doc {max_doc} is negative"));
        return Err(FfiStatus::InvalidArgument);
    }
    let bytes = read_whole_file(dir_handle, name)?;
    let bits = live_docs::parse(
        &bytes,
        segment_id,
        del_gen,
        max_doc as usize,
        del_count as usize,
    )
    .map_err(|e| {
        set_last_error(format!("parsing {name}: {e}"));
        FfiStatus::Decode
    })?;
    Ok(Some(bits))
}

/// Attaches (or clears) this segment's `.liv` live-docs bitset, so every
/// query run against `segment_handle` afterwards skips deleted documents
/// instead of reporting them as matches.
///
/// - `dir_handle`: the [`crate::directory::ffi_open_directory`] handle the
///   `.liv` file is read from (the same directory `ffi_open_segment` read
///   this segment's other files from -- not remembered by the segment
///   handle, which owns bytes rather than a directory, so it is passed
///   again here).
/// - `liv_name`/`liv_name_len`: the generation-suffixed live-docs file name
///   (`SegmentCommitInfo.files()`'s `.liv` entry, e.g. `"_0_1.liv"`). A
///   **null** `liv_name` (any `len`) *clears* any previously attached
///   bitset, restoring the "no deletions, every doc live" behavior -- the
///   right call for a segment whose `del_gen` is `-1`.
/// - `del_gen`: `SegmentCommitInfo.getDelGen()`. The `.liv` codec suffix is
///   this generation in base-36, so a wrong value is a header mismatch, not
///   a silently different bitset. Must be `>= 1` (a segment with no
///   deletions has `del_gen == -1` and no `.liv` file at all, which is the
///   null-`liv_name` case above).
/// - `del_count`: `SegmentCommitInfo.getDelCount()`, cross-checked against
///   the bitset's actual cardinality by
///   [`lucene_codecs::live_docs::parse`], so a `.liv` that disagrees with
///   the commit is [`FfiStatus::Decode`], not silently wrong results.
///
/// The bitset is sized by this segment's own `max_doc` (as passed to
/// [`ffi_open_segment`]); bit `d` set means doc `d` is live.
///
/// **Which calls honour it**: every query entry point that *produces* a doc
/// set from this segment -- `query.rs`'s term/boolean/phrase searches
/// (scored, unscored and MAXSCORE-pruned), `points_query.rs`'s range
/// search, `explain.rs`'s three explain calls, and `range_sort.rs`'s
/// range-then-sort. `sort.rs`'s and `facets.rs`'s entry points take a
/// caller-supplied *candidate list* rather than producing one, so filtering
/// deleted docs out of that list is the caller's decision, not this
/// segment's (pass a candidate list from a query that already honoured
/// deletions).
///
/// # Safety
/// `liv_name` must be valid for reads of `liv_name_len` bytes, or null.
#[no_mangle]
pub unsafe extern "C" fn ffi_segment_set_live_docs(
    segment_handle: u64,
    dir_handle: u64,
    liv_name: *const c_char,
    liv_name_len: usize,
    del_gen: i64,
    del_count: i32,
) -> i32 {
    guard(|| {
        // Scoped block, deliberately: this read guard MUST be dropped before
        // the write guard taken below. `std`'s `RwLock` is not reentrant and
        // does not upgrade -- holding a read guard while asking for the write
        // guard on the same lock deadlocks the calling thread. Copy out the
        // two scalars needed, then let go. The file I/O in `decode_live_docs`
        // then happens with no segment-registry guard held at all, so it
        // cannot block a concurrent query either.
        let (segment_id, max_doc) = {
            let registry = read_recovering(segments());
            let segment = registry.get(segment_handle).ok_or_else(|| {
                set_last_error(
                    "ffi_segment_set_live_docs: unknown or already-closed segment handle",
                );
                FfiStatus::InvalidHandle
            })?;
            (segment.segment_id, segment.max_doc)
        };
        // SAFETY: forwarded from this function's own caller contract.
        let parsed = unsafe {
            decode_live_docs(
                "ffi_segment_set_live_docs",
                dir_handle,
                liv_name,
                liv_name_len,
                del_gen,
                del_count,
                &segment_id,
                max_doc,
            )?
        };

        let mut registry = lock_recovering(segments());
        let segment = registry.get_mut(segment_handle).ok_or_else(|| {
            set_last_error("ffi_segment_set_live_docs: unknown or already-closed segment handle");
            FfiStatus::InvalidHandle
        })?;
        segment.live_docs = parsed;
        Ok(())
    })
}

/// Attaches one field's doc-values **update generation** to an already-open
/// segment: the generation-suffixed `.dvm`/`.dvd` pair that
/// `IndexWriter.updateNumericDocValue`/`updateBinaryDocValue` rewrote for
/// `field`, which supersedes that field's entry in the base pair
/// [`ffi_open_segment`] opened.
///
/// **Why this is not optional for a correct answer.** A doc-values update
/// rewrites the whole column for one field into
/// `_<segment>_<gen>_<perFieldSuffix>.dvd` and leaves the base `.dvd`
/// untouched, so the base column still holds that field's *pre-update*
/// values. Java never reads it: `SegmentDocValuesProducer` keeps one producer
/// per distinct `FieldInfo.docValuesGen` and routes each field to its own.
/// Without this call a sort, a facet count or a per-doc doc-values lookup on
/// an updated field returns the superseded values -- a wrong answer, not a
/// missing feature, and one nothing else in the stack can notice.
///
/// - `segment_handle`: an [`ffi_open_segment`] handle, opened as usual with
///   the segment's **base** `.fnm`/`.dvm`/`.dvd`. Nothing about that call
///   changes.
/// - `dir_handle`: the [`crate::directory::ffi_open_directory`] handle the
///   files are read from (a segment handle owns bytes, not a directory, so it
///   is passed again here -- same as [`ffi_segment_set_live_docs`]).
/// - `fnm_name`/`fnm_suffix`: the **generational** field-infos file
///   (`IndexFileNames.fileNameFromGeneration(segment, "fnm",
///   SegmentCommitInfo.getFieldInfosGen())`, e.g. `"_0_4.fnm"`) and the codec
///   suffix in its index header, which is that generation in base 36 (`"4"`).
///   This file, not the base `.fnm`, is the only one recording each field's
///   `FieldInfo.docValuesGen`, which is why Java's own `SegmentReader` reads
///   it. It is a parameter here rather than of [`ffi_open_segment`] because
///   it belongs to the update generation, like the `.dvm`/`.dvd` beside it,
///   and because it changes without the segment itself being rewritten.
/// - `field`/`field_len`: the field name. It must exist in *both* the
///   generational `.fnm` and the segment's own, with the same field number --
///   a disagreement is [`FfiStatus::InvalidArgument`] rather than a
///   generation silently attached to the wrong column.
/// - `dvm_name`/`dvd_name`: the generation's doc-values file names
///   (`SegmentCommitInfo.files()`, e.g. `"_0_4_Lucene90_0.dvm"`).
/// - `dv_suffix`: the codec suffix in those two files' index headers,
///   `PerFieldDocValuesFormat.getFullSegmentSuffix(base36(gen), perFieldSuffix)`
///   -- e.g. `"4_Lucene90_0"` for generation 4. A wrong value is a header
///   mismatch ([`FfiStatus::Decode`]), not a silently different column.
///
/// The generation's `.dvm` describes exactly the one updated field, so it is
/// parsed against a **one-field** `FieldInfos` (Java does the same: one
/// producer per generation, over that generation's own field list). A `.dvm`
/// naming some other field therefore fails to parse rather than being
/// accepted and mapped onto the wrong column.
///
/// Calling this twice for the same field replaces the previous generation,
/// which is what reopening onto a newer commit means. A failed call attaches
/// nothing and leaves the handle exactly as it was.
///
/// # Safety
/// `fnm_name`/`fnm_suffix`/`field`/`dvm_name`/`dvd_name`/`dv_suffix` must
/// each be valid for reads of their paired lengths.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_segment_add_doc_values_generation(
    segment_handle: u64,
    dir_handle: u64,
    fnm_name: *const c_char,
    fnm_name_len: usize,
    fnm_suffix: *const c_char,
    fnm_suffix_len: usize,
    field: *const c_char,
    field_len: usize,
    dvm_name: *const c_char,
    dvm_name_len: usize,
    dvd_name: *const c_char,
    dvd_name_len: usize,
    dv_suffix: *const c_char,
    dv_suffix_len: usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract guarantees each pointer is valid for its
        // paired length.
        let (fnm, fnm_suffix, field, dvm, dvd, suffix) = unsafe {
            (
                str_from_raw(fnm_name, fnm_name_len)?,
                str_from_raw(fnm_suffix, fnm_suffix_len)?,
                str_from_raw(field, field_len)?,
                str_from_raw(dvm_name, dvm_name_len)?,
                str_from_raw(dvd_name, dvd_name_len)?,
                str_from_raw(dv_suffix, dv_suffix_len)?,
            )
        };

        // Scoped, deliberately: this read guard MUST be dropped before the
        // write guard below -- `std`'s `RwLock` neither re-enters nor
        // upgrades, so holding both deadlocks the calling thread. Copy out
        // what is needed (the segment id and the base `.fnm`'s field number),
        // then let go; every file read below then happens with no registry
        // guard held, so it cannot block a concurrent query either.
        let (segment_id, base_field_number) = {
            let registry = read_recovering(segments());
            let segment = registry.get(segment_handle).ok_or_else(|| {
                set_last_error(
                    "ffi_segment_add_doc_values_generation: unknown or already-closed segment \
                     handle",
                );
                FfiStatus::InvalidHandle
            })?;
            let number = segment
                .field_infos
                .fields
                .iter()
                .find(|f| f.name == field)
                .map(|f| f.number)
                .ok_or_else(|| {
                    set_last_error(format!(
                        "ffi_segment_add_doc_values_generation: unknown field {field}"
                    ));
                    FfiStatus::InvalidArgument
                })?;
            (segment.segment_id, number)
        };

        let fnm_bytes = read_whole_file(dir_handle, fnm)?;
        let generational =
            field_infos::parse(&fnm_bytes, &segment_id, fnm_suffix).map_err(|e| {
                set_last_error(format!("parsing generational field infos {fnm}: {e}"));
                FfiStatus::Decode
            })?;
        let field_info = generational
            .fields
            .iter()
            .find(|f| f.name == field)
            .cloned()
            .ok_or_else(|| {
                set_last_error(format!(
                    "ffi_segment_add_doc_values_generation: field {field} is absent from {fnm}"
                ));
                FfiStatus::InvalidArgument
            })?;
        // Field numbers are stable across generations (`FieldInfos` is only
        // ever appended to), so a disagreement means the two files describe
        // different segments -- attaching anyway would key the generation to
        // a number the rest of this handle resolves to a different field.
        if field_info.number != base_field_number {
            set_last_error(format!(
                "ffi_segment_add_doc_values_generation: field {field} is number {} in {fnm} but \
                 number {base_field_number} in the segment's own .fnm",
                field_info.number
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        // `FieldInfo.docValuesGen == -1` means "this field has never been
        // updated", which is exactly the case the base column is correct for.
        // Accepting a generation here would be accepting a column this
        // segment's own metadata says does not exist -- and the likeliest
        // cause is the caller passing the *base* `.fnm`, where every field
        // reads -1, in which case the silent outcome would be no generation
        // attached at all and every read left on the superseded column.
        if field_info.doc_values_gen == -1 {
            set_last_error(format!(
                "ffi_segment_add_doc_values_generation: field {field} has docValuesGen -1 in \
                 {fnm} (no doc-values update); pass the generational .fnm named by \
                 SegmentCommitInfo.getFieldInfosGen(), not the base one"
            ));
            return Err(FfiStatus::InvalidArgument);
        }

        let meta_bytes = read_whole_file(dir_handle, dvm)?;
        // One-field `FieldInfos`, matching `SegmentDocValuesProducer`'s
        // per-generation producer: handing over the whole list would accept a
        // `.dvm` describing a field this generation is not for.
        let only = field_infos::FieldInfos {
            fields: vec![field_info],
        };
        let (_version, meta) = doc_values::parse_meta(&meta_bytes, &segment_id, suffix, &only)
            .map_err(|e| {
                set_last_error(format!("parsing doc-values generation {dvm}: {e}"));
                FfiStatus::Decode
            })?;
        let data = read_whole_file(dir_handle, dvd)?;
        doc_values::check_data_header_footer(&data, &segment_id, suffix).map_err(|e| {
            set_last_error(format!("opening doc-values generation {dvd}: {e}"));
            FfiStatus::Decode
        })?;

        let mut registry = lock_recovering(segments());
        let segment = registry.get_mut(segment_handle).ok_or_else(|| {
            set_last_error(
                "ffi_segment_add_doc_values_generation: unknown or already-closed segment handle",
            );
            FfiStatus::InvalidHandle
        })?;
        segment
            .dv_generations
            .retain(|g| g.field_number != base_field_number);
        segment.dv_generations.push(DocValuesGenerationColumn {
            field_number: base_field_number,
            meta,
            data,
        });
        Ok(())
    })
}

/// Closes a segment handle opened by [`ffi_open_segment`]. Returns
/// [`FfiStatus::InvalidHandle`] for an unknown/already-closed handle.
#[no_mangle]
pub extern "C" fn ffi_close_segment(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(segments())
            .remove(handle)
            .map(|_| ())
            .ok_or_else(|| {
                set_last_error("ffi_close_segment: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::{ffi_close_directory, ffi_open_directory};

    fn fixture_dir_path() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        )
        .to_string()
    }

    fn open_dir() -> u64 {
        let path = fixture_dir_path();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_directory(
                path.as_ptr() as *const c_char,
                path.len(),
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        handle
    }

    fn segment_id_bytes() -> [u8; 16] {
        let hex = "bea914ffd84e035aaac43aca30240b47";
        let mut id = [0u8; 16];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        id
    }

    #[allow(clippy::too_many_arguments)]
    fn open_segment_with(
        dir_handle: u64,
        doc_name: Option<&str>,
        pos_name: Option<&str>,
    ) -> (i32, u64) {
        open_segment_with_norms(dir_handle, doc_name, pos_name, None, None)
    }

    /// [`open_segment_with`] plus this fixture's real Java-written
    /// `_0_Lucene104_0.pay` -- the file `c13-ffi-surface` added a parameter
    /// for, closing b15's `.pay` deferral.
    fn open_segment_with_pay(
        dir_handle: u64,
        doc_name: Option<&str>,
        pos_name: Option<&str>,
        pay_name: Option<&str>,
    ) -> (i32, u64) {
        open_segment_full(dir_handle, doc_name, pos_name, pay_name, None, None)
    }

    /// Same as [`open_segment_with`], plus optional `.nvm`/`.nvd` names
    /// (task #30) for tests that need a segment opened with real norms.
    #[allow(clippy::too_many_arguments)]
    fn open_segment_with_norms(
        dir_handle: u64,
        doc_name: Option<&str>,
        pos_name: Option<&str>,
        nvm_name: Option<&str>,
        nvd_name: Option<&str>,
    ) -> (i32, u64) {
        open_segment_full(dir_handle, doc_name, pos_name, None, nvm_name, nvd_name)
    }

    #[allow(clippy::too_many_arguments)]
    fn open_segment_full(
        dir_handle: u64,
        doc_name: Option<&str>,
        pos_name: Option<&str>,
        pay_name: Option<&str>,
        nvm_name: Option<&str>,
        nvd_name: Option<&str>,
    ) -> (i32, u64) {
        let fnm = "_0.fnm";
        let tim = "_0_Lucene104_0.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let suffix = "Lucene104_0";
        let id = segment_id_bytes();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                tim.as_ptr() as *const c_char,
                tim.len(),
                tip.as_ptr() as *const c_char,
                tip.len(),
                tmd.as_ptr() as *const c_char,
                tmd.len(),
                doc_name.map_or(std::ptr::null(), |s| s.as_ptr()) as *const c_char,
                doc_name.map_or(0, |s| s.len()),
                pos_name.map_or(std::ptr::null(), |s| s.as_ptr()) as *const c_char,
                pos_name.map_or(0, |s| s.len()),
                pay_name.map_or(std::ptr::null(), |s| s.as_ptr()) as *const c_char,
                pay_name.map_or(0, |s| s.len()),
                nvm_name.map_or(std::ptr::null(), |s| s.as_ptr()) as *const c_char,
                nvm_name.map_or(0, |s| s.len()),
                nvd_name.map_or(std::ptr::null(), |s| s.as_ptr()) as *const c_char,
                nvd_name.map_or(0, |s| s.len()),
                std::ptr::null(), // dvm_name: this fixture has no doc-values files
                0,
                std::ptr::null(), // dvd_name
                0,
                std::ptr::null(), // dv_suffix
                0,
                std::ptr::null(), // kdm_name: no points data needed by this test/call
                0,
                std::ptr::null(), // kdi_name
                0,
                std::ptr::null(), // kdd_name
                0,
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                8959,
                &mut handle as *mut _,
            )
        };
        (rc, handle)
    }

    /// `max_doc` is `SegmentInfo.maxDoc()`, a count: a negative one is a
    /// caller error, not a segment that matches nothing.
    #[test]
    fn open_segment_rejects_a_negative_max_doc() {
        let dir_handle = open_dir();
        let fnm = "_0.fnm";
        let tim = "_0_Lucene104_0.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let suffix = "Lucene104_0";
        let id = segment_id_bytes();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                tim.as_ptr() as *const c_char,
                tim.len(),
                tip.as_ptr() as *const c_char,
                tip.len(),
                tmd.as_ptr() as *const c_char,
                tmd.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                -1,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert_eq!(handle, 0);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn open_segment_with_doc_file_then_close_roundtrips() {
        let dir_handle = open_dir();
        let (rc, seg_handle) = open_segment_with(dir_handle, Some("_0_Lucene104_0.doc"), None);
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_ne!(seg_handle, 0);

        assert!(read_recovering(segments()).get(seg_handle).is_some());
        assert_eq!(ffi_close_segment(seg_handle), FfiStatus::Ok.code());
        assert!(read_recovering(segments()).get(seg_handle).is_none());

        ffi_close_directory(dir_handle);
    }

    #[test]
    fn open_segment_without_doc_file_succeeds_for_singleton_only_fields() {
        let dir_handle = open_dir();
        let (rc, seg_handle) = open_segment_with(dir_handle, None, None);
        assert_eq!(rc, FfiStatus::Ok.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn close_unknown_segment_handle_is_invalid_handle() {
        assert_eq!(ffi_close_segment(0x1234), FfiStatus::InvalidHandle.code());
    }

    /// A directory handle passed where a segment handle is expected must be
    /// rejected by the registry-tag check, not accidentally succeed against
    /// (or corrupt) the segment registry -- see `handle.rs`'s `RegistryTag`.
    #[test]
    fn directory_handle_passed_to_close_segment_is_invalid_handle() {
        let dir_handle = open_dir();
        let (rc, seg_handle) = open_segment_with(dir_handle, Some("_0_Lucene104_0.doc"), None);
        assert_eq!(rc, FfiStatus::Ok.code());

        // The directory handle must not be accepted by `ffi_close_segment`,
        // and the real segment handle must remain untouched afterwards.
        assert_eq!(
            ffi_close_segment(dir_handle),
            FfiStatus::InvalidHandle.code()
        );
        assert!(read_recovering(segments()).get(seg_handle).is_some());

        assert_eq!(ffi_close_segment(seg_handle), FfiStatus::Ok.code());
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn double_close_segment_is_invalid_handle_not_a_crash() {
        let dir_handle = open_dir();
        let (_, seg_handle) = open_segment_with(dir_handle, Some("_0_Lucene104_0.doc"), None);
        assert_eq!(ffi_close_segment(seg_handle), FfiStatus::Ok.code());
        assert_eq!(
            ffi_close_segment(seg_handle),
            FfiStatus::InvalidHandle.code()
        );
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn open_segment_unknown_directory_handle_is_invalid_handle() {
        let (rc, _) = open_segment_with(0xFFFF_FFFF, Some("_0_Lucene104_0.doc"), None);
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn open_segment_missing_file_is_io_error() {
        let dir_handle = open_dir();
        let fnm = "does-not-exist.fnm";
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(), // kdm_name: no points data needed by this test/call
                0,
                std::ptr::null(), // kdi_name
                0,
                std::ptr::null(), // kdd_name
                0,
                segment_id_bytes().as_ptr(),
                std::ptr::null(),
                0,
                8959,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Io.code());
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn open_segment_null_out_handle_is_null_pointer_error() {
        let dir_handle = open_dir();
        let fnm = "_0.fnm";
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(), // kdm_name: no points data needed by this test/call
                0,
                std::ptr::null(), // kdi_name
                0,
                std::ptr::null(), // kdd_name
                0,
                segment_id_bytes().as_ptr(),
                std::ptr::null(),
                0,
                8959,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn open_segment_null_segment_id_is_null_pointer_error() {
        let dir_handle = open_dir();
        let fnm = "_0.fnm";
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(), // kdm_name: no points data needed by this test/call
                0,
                std::ptr::null(), // kdi_name
                0,
                std::ptr::null(), // kdd_name
                0,
                std::ptr::null(), // segment_id: null -- the point of this test
                std::ptr::null(),
                0,
                8959,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_directory(dir_handle);
    }

    /// A scratch directory containing copies of the fixture segment's real
    /// `.fnm`/`.tim`/`.tip`/`.tmd` files (so a caller can selectively swap
    /// one of them for garbage bytes without disturbing the shared fixture
    /// data other tests in this crate also read from).
    fn scratch_dir_with_fixture_copies() -> lucene_util::test_support::TempDir {
        let src = fixture_dir_path();
        let dst = lucene_util::test_support::TempDir::new("ffi-segment");
        for name in [
            "_0.fnm",
            "_0_Lucene104_0.tim",
            "_0_Lucene104_0.tip",
            "_0_Lucene104_0.tmd",
            "_0.nvm",
            "_0.nvd",
        ] {
            std::fs::copy(format!("{src}{name}"), dst.join(name)).unwrap();
        }
        dst
    }

    fn open_dir_at(path: &std::path::Path) -> u64 {
        let path_str = path.to_str().unwrap();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_directory(
                path_str.as_ptr() as *const c_char,
                path_str.len(),
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        handle
    }

    #[test]
    fn open_segment_garbage_fnm_bytes_is_decode_error() {
        let dir = scratch_dir_with_fixture_copies();
        std::fs::write(dir.join("garbage.fnm"), [0u8; 8]).unwrap();
        let dir_handle = open_dir_at(&dir);

        let fnm = "garbage.fnm";
        let tim = "_0_Lucene104_0.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let suffix = "Lucene104_0";
        let id = segment_id_bytes();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                tim.as_ptr() as *const c_char,
                tim.len(),
                tip.as_ptr() as *const c_char,
                tip.len(),
                tmd.as_ptr() as *const c_char,
                tmd.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(), // kdm_name: no points data needed by this test/call
                0,
                std::ptr::null(), // kdi_name
                0,
                std::ptr::null(), // kdd_name
                0,
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                8959,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn open_segment_garbage_tim_bytes_is_decode_error() {
        let dir = scratch_dir_with_fixture_copies();
        std::fs::write(dir.join("garbage.tim"), [0u8; 8]).unwrap();
        let dir_handle = open_dir_at(&dir);

        let fnm = "_0.fnm";
        let tim = "garbage.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let suffix = "Lucene104_0";
        let id = segment_id_bytes();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                tim.as_ptr() as *const c_char,
                tim.len(),
                tip.as_ptr() as *const c_char,
                tip.len(),
                tmd.as_ptr() as *const c_char,
                tmd.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(), // kdm_name: no points data needed by this test/call
                0,
                std::ptr::null(), // kdi_name
                0,
                std::ptr::null(), // kdd_name
                0,
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                8959,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn open_segment_garbage_doc_bytes_is_decode_error() {
        let dir = scratch_dir_with_fixture_copies();
        std::fs::write(dir.join("garbage.doc"), [0u8; 8]).unwrap();
        let dir_handle = open_dir_at(&dir);

        let fnm = "_0.fnm";
        let tim = "_0_Lucene104_0.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let doc = "garbage.doc";
        let suffix = "Lucene104_0";
        let id = segment_id_bytes();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                tim.as_ptr() as *const c_char,
                tim.len(),
                tip.as_ptr() as *const c_char,
                tip.len(),
                tmd.as_ptr() as *const c_char,
                tmd.len(),
                doc.as_ptr() as *const c_char,
                doc.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(), // kdm_name: no points data needed by this test/call
                0,
                std::ptr::null(), // kdi_name
                0,
                std::ptr::null(), // kdd_name
                0,
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                8959,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn open_segment_garbage_pos_bytes_is_decode_error() {
        let dir = scratch_dir_with_fixture_copies();
        std::fs::write(dir.join("garbage.pos"), [0u8; 8]).unwrap();
        let dir_handle = open_dir_at(&dir);

        let fnm = "_0.fnm";
        let tim = "_0_Lucene104_0.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let pos = "garbage.pos";
        let suffix = "Lucene104_0";
        let id = segment_id_bytes();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                tim.as_ptr() as *const c_char,
                tim.len(),
                tip.as_ptr() as *const c_char,
                tip.len(),
                tmd.as_ptr() as *const c_char,
                tmd.len(),
                std::ptr::null(),
                0,
                pos.as_ptr() as *const c_char,
                pos.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(), // kdm_name: no points data needed by this test/call
                0,
                std::ptr::null(), // kdi_name
                0,
                std::ptr::null(), // kdd_name
                0,
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                8959,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Task #30: a garbage `.nvm` (norms metadata) file must surface as
    /// [`FfiStatus::Decode`], same as every other garbage-bytes test above.
    #[test]
    fn open_segment_garbage_nvm_bytes_is_decode_error() {
        let dir = scratch_dir_with_fixture_copies();
        std::fs::write(dir.join("garbage.nvm"), [0u8; 8]).unwrap();
        let dir_handle = open_dir_at(&dir);

        let fnm = "_0.fnm";
        let tim = "_0_Lucene104_0.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let nvm = "garbage.nvm";
        let nvd = "_0.nvd";
        let suffix = "Lucene104_0";
        let id = segment_id_bytes();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                tim.as_ptr() as *const c_char,
                tim.len(),
                tip.as_ptr() as *const c_char,
                tip.len(),
                tmd.as_ptr() as *const c_char,
                tmd.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                nvm.as_ptr() as *const c_char,
                nvm.len(),
                nvd.as_ptr() as *const c_char,
                nvd.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(), // kdm_name: no points data needed by this test/call
                0,
                std::ptr::null(), // kdi_name
                0,
                std::ptr::null(), // kdd_name
                0,
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                8959,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Task #30: a garbage `.nvd` (norms data) file must surface as
    /// [`FfiStatus::Decode`] -- the `.nvm` parses fine, but `.nvd`'s own
    /// header/footer check fails.
    #[test]
    fn open_segment_garbage_nvd_bytes_is_decode_error() {
        let dir = scratch_dir_with_fixture_copies();
        std::fs::write(dir.join("garbage.nvd"), [0u8; 8]).unwrap();
        let dir_handle = open_dir_at(&dir);

        let fnm = "_0.fnm";
        let tim = "_0_Lucene104_0.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let nvm = "_0.nvm";
        let nvd = "garbage.nvd";
        let suffix = "Lucene104_0";
        let id = segment_id_bytes();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                tim.as_ptr() as *const c_char,
                tim.len(),
                tip.as_ptr() as *const c_char,
                tip.len(),
                tmd.as_ptr() as *const c_char,
                tmd.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                nvm.as_ptr() as *const c_char,
                nvm.len(),
                nvd.as_ptr() as *const c_char,
                nvd.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(), // kdm_name: no points data needed by this test/call
                0,
                std::ptr::null(), // kdi_name
                0,
                std::ptr::null(), // kdd_name
                0,
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                8959,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Opening a segment with real `.nvm`/`.nvd` names succeeds and the
    /// resulting handle carries `Some` norms data -- the "happy path" for
    /// task #30's `nvm_name`/`nvd_name` parameters (the norms-aware scored-query
    /// tests in `query.rs` exercise the rest of this path end-to-end).
    #[test]
    fn open_segment_with_norms_files_succeeds() {
        let dir_handle = open_dir();
        let (rc, seg_handle) = open_segment_with_norms(
            dir_handle,
            Some("_0_Lucene104_0.doc"),
            None,
            Some("_0.nvm"),
            Some("_0.nvd"),
        );
        assert_eq!(rc, FfiStatus::Ok.code());
        {
            let segments = read_recovering(segments());
            let segment = segments.get(seg_handle).expect("segment handle");
            assert!(segment.norms.is_some());
            assert!(segment.norms_data.is_some());
        }
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }
    // ------------------------------------------------------------------
    // c13-ffi-surface: `.pay` (payloads/offsets)
    // ------------------------------------------------------------------

    /// The fixture's real Java-written `.pay` must open and validate, and the
    /// segment handle must actually carry it -- b15's documented deferral,
    /// closed now that `lucene_search`'s `DirectoryReader` opens `.pay` per
    /// segment and the single-segment path was the only one that could not.
    #[test]
    fn opening_a_segment_with_its_pay_file_attaches_it() {
        let dir_handle = open_dir();
        let (rc, handle) = open_segment_with_pay(
            dir_handle,
            Some("_0_Lucene104_0.doc"),
            Some("_0_Lucene104_0.pos"),
            Some("_0_Lucene104_0.pay"),
        );
        assert_eq!(rc, FfiStatus::Ok.code());
        {
            let registry = read_recovering(segments());
            let segment = registry.get(handle).expect("segment handle");
            assert!(segment.pay_bytes.is_some(), ".pay must be attached");
            assert!(segment.pos_bytes.is_some());
        }
        assert_eq!(ffi_close_segment(handle), FfiStatus::Ok.code());

        // A null `pay_name` still means "no payloads", the pre-c13 behaviour.
        let (rc, handle) = open_segment_with_pay(
            dir_handle,
            Some("_0_Lucene104_0.doc"),
            Some("_0_Lucene104_0.pos"),
            None,
        );
        assert_eq!(rc, FfiStatus::Ok.code());
        {
            let registry = read_recovering(segments());
            assert!(registry.get(handle).unwrap().pay_bytes.is_none());
        }
        assert_eq!(ffi_close_segment(handle), FfiStatus::Ok.code());
        ffi_close_directory(dir_handle);
    }

    /// A file that is not a `.pay` must fail the codec header check at open
    /// time, exactly as `.doc`/`.pos` already do -- not at the first query.
    #[test]
    fn a_pay_name_pointing_at_another_file_is_a_decode_error() {
        let dir_handle = open_dir();
        let (rc, _) = open_segment_with_pay(
            dir_handle,
            Some("_0_Lucene104_0.doc"),
            Some("_0_Lucene104_0.pos"),
            Some("_0_Lucene104_0.pos"),
        );
        assert_eq!(rc, FfiStatus::Decode.code());
        // A missing file is still an I/O error, not a decode one.
        let (rc, _) = open_segment_with_pay(
            dir_handle,
            Some("_0_Lucene104_0.doc"),
            Some("_0_Lucene104_0.pos"),
            Some("_0_nope.pay"),
        );
        assert_eq!(rc, FfiStatus::Io.code());
        ffi_close_directory(dir_handle);
    }
}

/// End-to-end tests for [`ffi_segment_set_live_docs`] against the real,
/// Java-written `fixtures/data/live_docs_index/` fixture (5 docs,
/// `id:0..4`, docs 1 and 3 deleted at `del_gen == 1` -- see
/// `fixtures/src/GenLiveDocs.java` and that fixture's
/// `manifest.properties`). Kept in its own module rather than folded into
/// `tests` above because every helper here names a *different* fixture
/// directory, segment id, suffix and `max_doc`.
#[cfg(test)]
mod live_docs_tests {
    use super::*;
    use crate::directory::{ffi_close_directory, ffi_open_directory};
    use crate::query::ffi_search_term_query;
    use crate::results::{ffi_close_results, ffi_results_copy, ffi_results_len};

    fn open_dir() -> u64 {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/live_docs_index/"
        );
        let mut handle: u64 = 0;
        let rc =
            unsafe { ffi_open_directory(path.as_ptr().cast::<c_char>(), path.len(), &mut handle) };
        assert_eq!(rc, FfiStatus::Ok.code());
        handle
    }

    /// `id_hex` from the fixture's `manifest.properties`.
    fn segment_id_bytes() -> [u8; 16] {
        let hex = "e0811e4220a8e70d1ad3e053cc6f8ee7";
        let mut id = [0u8; 16];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        id
    }

    fn open_segment(dir_handle: u64) -> u64 {
        let (fnm, tim, tip, tmd, doc) = (
            "_0.fnm",
            "_0_Lucene104_0.tim",
            "_0_Lucene104_0.tip",
            "_0_Lucene104_0.tmd",
            "_0_Lucene104_0.doc",
        );
        let suffix = "Lucene104_0";
        let id = segment_id_bytes();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr().cast::<c_char>(),
                fnm.len(),
                tim.as_ptr().cast::<c_char>(),
                tim.len(),
                tip.as_ptr().cast::<c_char>(),
                tip.len(),
                tmd.as_ptr().cast::<c_char>(),
                tmd.len(),
                doc.as_ptr().cast::<c_char>(),
                doc.len(),
                std::ptr::null(), // pos
                0,
                std::ptr::null(),
                0,
                std::ptr::null(), // nvm
                0,
                std::ptr::null(), // nvd
                0,
                std::ptr::null(), // dvm
                0,
                std::ptr::null(), // dvd
                0,
                std::ptr::null(), // dv_suffix
                0,
                std::ptr::null(), // kdm
                0,
                std::ptr::null(), // kdi
                0,
                std::ptr::null(), // kdd
                0,
                id.as_ptr(),
                suffix.as_ptr().cast::<c_char>(),
                suffix.len(),
                5,
                &mut handle,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        handle
    }

    fn attach_live_docs(seg: u64, dir: u64) -> i32 {
        let liv = "_0_1.liv";
        unsafe {
            ffi_segment_set_live_docs(seg, dir, liv.as_ptr().cast::<c_char>(), liv.len(), 1, 2)
        }
    }

    fn term_query_docs(seg: u64, term: &str) -> Vec<i32> {
        let field = "id";
        let mut results: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query(
                seg,
                field.as_ptr().cast::<c_char>(),
                field.len(),
                term.as_ptr(),
                term.len(),
                &mut results,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_results_len(results, &mut len) },
            FfiStatus::Ok.code()
        );
        let mut buf = vec![0i32; len];
        assert_eq!(
            unsafe { ffi_results_copy(results, buf.as_mut_ptr(), buf.len()) },
            FfiStatus::Ok.code()
        );
        assert_eq!(ffi_close_results(results), FfiStatus::Ok.code());
        buf
    }

    /// The correctness fix this call exists for: before live docs are
    /// attached, a query still matches a *deleted* document; after, it does
    /// not -- while a live document keeps matching either way.
    #[test]
    fn attaching_live_docs_hides_deleted_documents_from_a_term_query() {
        let dir = open_dir();
        let seg = open_segment(dir);

        // Docs 1 and 3 are deleted in this commit, docs 0/2/4 are live.
        assert_eq!(term_query_docs(seg, "1"), vec![1], "no live docs attached");
        assert_eq!(term_query_docs(seg, "3"), vec![3], "no live docs attached");
        assert_eq!(term_query_docs(seg, "2"), vec![2]);

        assert_eq!(attach_live_docs(seg, dir), FfiStatus::Ok.code());

        assert!(
            term_query_docs(seg, "1").is_empty(),
            "deleted doc 1 must not match once .liv is attached"
        );
        assert!(
            term_query_docs(seg, "3").is_empty(),
            "deleted doc 3 must not match once .liv is attached"
        );
        assert_eq!(term_query_docs(seg, "0"), vec![0]);
        assert_eq!(term_query_docs(seg, "2"), vec![2]);
        assert_eq!(term_query_docs(seg, "4"), vec![4]);

        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    /// The decoded bitset must be exactly what `GenLiveDocs.java` recorded in
    /// the fixture manifest -- proving this reads the real `.liv` rather than
    /// merely "some" bitset of the right length.
    #[test]
    fn attached_bitset_matches_the_fixture_manifest() {
        let dir = open_dir();
        let seg = open_segment(dir);
        assert_eq!(attach_live_docs(seg, dir), FfiStatus::Ok.code());

        let registry = read_recovering(segments());
        let handle = registry.get(seg).unwrap();
        let live = handle.live_docs.as_ref().expect("live docs attached");
        assert_eq!(live.cardinality(), 3);
        for (doc, expected_live) in [(0, true), (1, false), (2, true), (3, false), (4, true)] {
            assert_eq!(live.get(doc), expected_live, "doc {doc}");
        }
        drop(registry);

        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    /// A null `liv_name` clears an attached bitset, restoring the "no
    /// deletions" behavior a `del_gen == -1` segment needs.
    #[test]
    fn null_liv_name_clears_previously_attached_live_docs() {
        let dir = open_dir();
        let seg = open_segment(dir);
        assert_eq!(attach_live_docs(seg, dir), FfiStatus::Ok.code());
        assert!(term_query_docs(seg, "1").is_empty());

        let rc = unsafe { ffi_segment_set_live_docs(seg, dir, std::ptr::null(), 0, -1, 0) };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(read_recovering(segments())
            .get(seg)
            .unwrap()
            .live_docs
            .is_none());
        assert_eq!(term_query_docs(seg, "1"), vec![1]);

        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    #[test]
    fn a_del_count_that_disagrees_with_the_file_is_a_decode_error() {
        let dir = open_dir();
        let seg = open_segment(dir);
        let liv = "_0_1.liv";
        // The fixture really has 2 deletions; claiming 1 must be rejected,
        // not silently accepted with a bitset that contradicts the commit.
        let rc = unsafe {
            ffi_segment_set_live_docs(seg, dir, liv.as_ptr().cast::<c_char>(), liv.len(), 1, 1)
        };
        assert_eq!(rc, FfiStatus::Decode.code());
        assert!(read_recovering(segments())
            .get(seg)
            .unwrap()
            .live_docs
            .is_none());

        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    /// A *failed* re-attach must leave the previously attached bitset in
    /// place: the parse happens before the write guard is taken, so there is
    /// no window in which a segment carries half-updated deletion state. The
    /// sibling test above only ever fails from a segment with nothing
    /// attached, so it would pass under the wrong behavior too.
    #[test]
    fn a_failed_reattach_leaves_the_previous_bitset_intact() {
        let dir = open_dir();
        let seg = open_segment(dir);
        assert_eq!(attach_live_docs(seg, dir), FfiStatus::Ok.code());
        assert!(term_query_docs(seg, "1").is_empty());

        let liv = "_0_1.liv";
        let rc = unsafe {
            ffi_segment_set_live_docs(seg, dir, liv.as_ptr().cast::<c_char>(), liv.len(), 1, 1)
        };
        assert_eq!(rc, FfiStatus::Decode.code());

        // Still the original bitset, and queries still hide docs 1 and 3.
        {
            let registry = read_recovering(segments());
            let live = registry
                .get(seg)
                .unwrap()
                .live_docs
                .as_ref()
                .expect("previous bitset must survive a failed re-attach");
            assert_eq!(live.cardinality(), 3);
        }
        assert!(term_query_docs(seg, "1").is_empty());
        assert!(term_query_docs(seg, "3").is_empty());
        assert_eq!(term_query_docs(seg, "2"), vec![2]);

        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    #[test]
    fn a_wrong_del_gen_is_a_decode_error_not_a_different_bitset() {
        let dir = open_dir();
        let seg = open_segment(dir);
        let liv = "_0_1.liv";
        // `.liv`'s codec suffix is the base-36 del_gen, so gen 2 fails the
        // index-header check rather than decoding gen 1's bytes.
        let rc = unsafe {
            ffi_segment_set_live_docs(seg, dir, liv.as_ptr().cast::<c_char>(), liv.len(), 2, 2)
        };
        assert_eq!(rc, FfiStatus::Decode.code());

        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    #[test]
    fn a_non_positive_del_gen_with_a_real_name_is_invalid_argument() {
        let dir = open_dir();
        let seg = open_segment(dir);
        let liv = "_0_1.liv";
        let rc = unsafe {
            ffi_segment_set_live_docs(seg, dir, liv.as_ptr().cast::<c_char>(), liv.len(), -1, 2)
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        let rc = unsafe {
            ffi_segment_set_live_docs(seg, dir, liv.as_ptr().cast::<c_char>(), liv.len(), 1, -1)
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    #[test]
    fn unknown_segment_handle_is_invalid_handle() {
        let dir = open_dir();
        let liv = "_0_1.liv";
        let rc = unsafe {
            ffi_segment_set_live_docs(
                0xDEAD_BEEF,
                dir,
                liv.as_ptr().cast::<c_char>(),
                liv.len(),
                1,
                2,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        ffi_close_directory(dir);
    }

    #[test]
    fn unknown_directory_handle_is_invalid_handle() {
        let dir = open_dir();
        let seg = open_segment(dir);
        let liv = "_0_1.liv";
        let rc = unsafe {
            ffi_segment_set_live_docs(
                seg,
                0xDEAD_BEEF,
                liv.as_ptr().cast::<c_char>(),
                liv.len(),
                1,
                2,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    /// Clearing live docs on a handle that never had any is a no-op success,
    /// not an error -- the natural call for a `del_gen == -1` segment.
    #[test]
    fn clearing_when_none_attached_is_ok() {
        let dir = open_dir();
        let seg = open_segment(dir);
        let rc = unsafe { ffi_segment_set_live_docs(seg, dir, std::ptr::null(), 0, -1, 0) };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    // -----------------------------------------------------------------
    // Doc-values update generations (c14's A1)
    // -----------------------------------------------------------------
    //
    // `fixtures/data/doc_values_updates_index/` is written by a real
    // `IndexWriter` (`fixtures/src/GenDocValuesUpdates.java`) that calls
    // `updateNumericDocValue("val", ...)` and `updateBinaryDocValue("tag",
    // ...)`, so the segment carries a base `.dvm`/`.dvd` holding the
    // *original* values plus one generation-suffixed pair per updated field
    // holding the current ones. Its manifest records both, which is what lets
    // these tests assert that the FFI reads the current column and, in the
    // negative control, that the base column really is different (a test that
    // could pass either way proves nothing).

    fn updates_dir_path() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/doc_values_updates_index/"
        )
        .to_string()
    }

    fn updates_manifest() -> Vec<(String, String)> {
        let text = std::fs::read_to_string(format!("{}manifest.properties", updates_dir_path()))
            .expect("run fixtures/src/GenDocValuesUpdates.java via scripts/gen-fixtures.sh");
        text.lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn manifest_get<'m>(manifest: &'m [(String, String)], key: &str) -> &'m str {
        manifest
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }

    /// `expected_val`'s per-doc column: `None` where the last update round
    /// reset the document back to having no value.
    fn expected_column(manifest: &[(String, String)], key: &str) -> Vec<Option<i64>> {
        manifest_get(manifest, key)
            .split(',')
            .map(|c| (!c.is_empty()).then(|| c.parse().unwrap()))
            .collect()
    }

    fn open_updates_dir() -> u64 {
        let path = updates_dir_path();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_directory(
                path.as_ptr() as *const c_char,
                path.len(),
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        handle
    }

    /// The calling thread's last-error message, read back through the real
    /// exported accessor rather than the thread-local directly, so these
    /// tests also prove the message reaches a JNI caller.
    fn last_error_message() -> String {
        let mut buf = [0 as c_char; 512];
        let rc = unsafe {
            crate::ffi_get_last_error_message(buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    fn hex_id(hex: &str) -> [u8; 16] {
        let mut id = [0u8; 16];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        id
    }

    /// Opens the updates fixture's segment exactly the way any caller does:
    /// base `.fnm`, base `.dvm`/`.dvd`. That is the state the FFI could only
    /// ever be in before this batch, and it is what every negative control
    /// below reads from.
    fn open_updates_segment(dir_handle: u64, manifest: &[(String, String)]) -> u64 {
        let fnm = "_0.fnm";
        let tim = "_0_Lucene104_0.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let dvm = "_0_Lucene90_0.dvm";
        let dvd = "_0_Lucene90_0.dvd";
        let suffix = "Lucene104_0";
        let dv_suffix = "Lucene90_0";
        let id = hex_id(manifest_get(manifest, "id_hex"));
        let max_doc: i32 = manifest_get(manifest, "max_doc").parse().unwrap();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                tim.as_ptr() as *const c_char,
                tim.len(),
                tip.as_ptr() as *const c_char,
                tip.len(),
                tmd.as_ptr() as *const c_char,
                tmd.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                dvm.as_ptr() as *const c_char,
                dvm.len(),
                dvd.as_ptr() as *const c_char,
                dvd.len(),
                dv_suffix.as_ptr() as *const c_char,
                dv_suffix.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                max_doc,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        handle
    }

    /// Attaches `field`'s generation, always through the fixture's real
    /// generational field-infos file (`_0_4.fnm`, codec suffix `"4"`).
    fn add_generation(
        seg: u64,
        dir: u64,
        field: &str,
        dvm: &str,
        dvd: &str,
        dv_suffix: &str,
    ) -> i32 {
        add_generation_with_fnm(seg, dir, "_0_4.fnm", "4", field, dvm, dvd, dv_suffix)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_generation_with_fnm(
        seg: u64,
        dir: u64,
        fnm: &str,
        fnm_suffix: &str,
        field: &str,
        dvm: &str,
        dvd: &str,
        dv_suffix: &str,
    ) -> i32 {
        unsafe {
            ffi_segment_add_doc_values_generation(
                seg,
                dir,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                fnm_suffix.as_ptr() as *const c_char,
                fnm_suffix.len(),
                field.as_ptr() as *const c_char,
                field.len(),
                dvm.as_ptr() as *const c_char,
                dvm.len(),
                dvd.as_ptr() as *const c_char,
                dvd.len(),
                dv_suffix.as_ptr() as *const c_char,
                dv_suffix.len(),
            )
        }
    }

    /// Every document's `val`, read one at a time through
    /// [`crate::sort::ffi_numeric_doc_value_for_doc`] -- the per-doc FFI a
    /// sort or a facet count is built out of.
    fn read_val_column(seg: u64, max_doc: i32) -> Vec<Option<i64>> {
        let field = "val";
        (0..max_doc)
            .map(|doc| {
                let mut has = false;
                let mut value = 0i64;
                let rc = unsafe {
                    crate::sort::ffi_numeric_doc_value_for_doc(
                        seg,
                        field.as_ptr() as *const c_char,
                        field.len(),
                        doc,
                        &mut has as *mut _,
                        &mut value as *mut _,
                    )
                };
                assert_eq!(rc, FfiStatus::Ok.code());
                has.then_some(value)
            })
            .collect()
    }

    /// The whole of c14's A1: after `updateNumericDocValue`, the base column
    /// still holds the pre-update values, so a segment handle that only knows
    /// about the base pair answers every doc-values call with them. Attaching
    /// the field's generation is what makes the FFI agree with what real
    /// Lucene reads back.
    #[test]
    fn an_updated_field_reads_from_its_generation_not_the_superseded_base_column() {
        let manifest = updates_manifest();
        let max_doc: i32 = manifest_get(&manifest, "max_doc").parse().unwrap();
        let expected = expected_column(&manifest, "expected_val");
        assert_eq!(expected.len(), max_doc as usize);

        let dir = open_updates_dir();
        let seg = open_updates_segment(dir, &manifest);

        // Negative control, and the bug as it stood: with only the base pair
        // attached, `val` reads back its *original* per-doc value (the
        // generator writes doc `d`'s original value as `d`), not the updated
        // one.
        let base = read_val_column(seg, max_doc);
        assert_eq!(
            base,
            (0..max_doc).map(|d| Some(d as i64)).collect::<Vec<_>>(),
            "the base column holds the pre-update values -- if this ever stops \
             differing from `expected_val`, the test below proves nothing"
        );
        assert_ne!(base, expected);

        // `val` was updated in generation 4 (manifest `field_dv_gen.val`), so
        // its column is `_0_4_Lucene90_0.dv{m,d}` with codec suffix
        // `4_Lucene90_0` (`PerFieldDocValuesFormat.getFullSegmentSuffix`).
        assert_eq!(manifest_get(&manifest, "field_dv_gen.val"), "4");
        let rc = add_generation(
            seg,
            dir,
            "val",
            "_0_4_Lucene90_0.dvm",
            "_0_4_Lucene90_0.dvd",
            "4_Lucene90_0",
        );
        assert_eq!(rc, FfiStatus::Ok.code());

        assert_eq!(
            read_val_column(seg, max_doc),
            expected,
            "after attaching the generation, every document reads the value \
             real Lucene reads (including the documents the last round reset \
             back to having no value at all)"
        );

        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    /// A field the update never touched keeps coming out of the base column
    /// while a sibling field is served from a generation --
    /// `SegmentDocValuesProducer`'s per-field routing, not a whole-segment
    /// switch.
    #[test]
    fn an_untouched_field_still_reads_the_base_column_alongside_an_updated_one() {
        let manifest = updates_manifest();
        let max_doc: i32 = manifest_get(&manifest, "max_doc").parse().unwrap();
        let dir = open_updates_dir();
        let seg = open_updates_segment(dir, &manifest);
        assert_eq!(
            add_generation(
                seg,
                dir,
                "val",
                "_0_4_Lucene90_0.dvm",
                "_0_4_Lucene90_0.dvd",
                "4_Lucene90_0",
            ),
            FfiStatus::Ok.code()
        );

        let expected_keep = expected_column(&manifest, "expected_keep");
        let field = "keep";
        for doc in 0..max_doc {
            let mut has = false;
            let mut value = 0i64;
            let rc = unsafe {
                crate::sort::ffi_numeric_doc_value_for_doc(
                    seg,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    doc,
                    &mut has as *mut _,
                    &mut value as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::Ok.code());
            assert_eq!(
                has.then_some(value),
                expected_keep[doc as usize],
                "keep doc {doc}"
            );
        }

        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    /// A sort is the consumer c14's A1 named first: `ffi_sort_by_doc_value`
    /// must rank by the updated values, and the two rankings genuinely differ
    /// (the base column is `0..maxDoc` ascending, the generation is not).
    #[test]
    fn sorting_by_an_updated_field_ranks_by_the_updated_values() {
        use crate::results_sorted::{
            ffi_close_sorted_results, ffi_sorted_results_copy, ffi_sorted_results_len,
        };

        let manifest = updates_manifest();
        let max_doc: i32 = manifest_get(&manifest, "max_doc").parse().unwrap();
        let expected = expected_column(&manifest, "expected_val");
        let dir = open_updates_dir();
        let seg = open_updates_segment(dir, &manifest);
        assert_eq!(
            add_generation(
                seg,
                dir,
                "val",
                "_0_4_Lucene90_0.dvm",
                "_0_4_Lucene90_0.dvd",
                "4_Lucene90_0",
            ),
            FfiStatus::Ok.code()
        );

        let candidates: Vec<i32> = (0..max_doc).collect();
        let field = "val";
        let mut results: u64 = 0;
        let rc = unsafe {
            crate::sort::ffi_sort_by_doc_value(
                seg,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                false, // MissingValue::Exclude -- reset docs drop out
                0,
                &mut results as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_sorted_results_len(results, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        let mut docs = vec![0i32; len];
        let mut values = vec![0i64; len];
        assert_eq!(
            unsafe {
                ffi_sorted_results_copy(results, docs.as_mut_ptr(), values.as_mut_ptr(), len)
            },
            FfiStatus::Ok.code()
        );

        // Ascending by value, ties by ascending doc id -- computed from the
        // manifest's own column rather than restated by hand.
        let mut want: Vec<(i32, i64)> = expected
            .iter()
            .enumerate()
            .filter_map(|(doc, v)| v.map(|v| (doc as i32, v)))
            .collect();
        want.sort_by_key(|&(doc, v)| (v, doc));
        assert_eq!(
            docs.iter()
                .copied()
                .zip(values.iter().copied())
                .collect::<Vec<_>>(),
            want
        );
        // The base column would have sorted every document, ascending by doc
        // id, with no document missing -- a visibly different answer.
        assert!(len < max_doc as usize);

        assert_eq!(ffi_close_sorted_results(results), FfiStatus::Ok.code());
        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    /// A range facet count over an updated field is the other consumer c14
    /// named. `7000` is the value the generator sets on every even document
    /// it did not reset, and it appears nowhere in the base column.
    #[test]
    fn range_facet_counts_over_an_updated_field_count_the_updated_values() {
        let manifest = updates_manifest();
        let max_doc: i32 = manifest_get(&manifest, "max_doc").parse().unwrap();
        let expected = expected_column(&manifest, "expected_val");
        let dir = open_updates_dir();
        let seg = open_updates_segment(dir, &manifest);

        let candidates: Vec<i32> = (0..max_doc).collect();
        let field = "val";
        let label = "high";
        let mins = [7000i64];
        let maxs = [7000i64];
        let inclusive = [1u8];
        let label_lens = [label.len()];

        let count_high = |seg: u64| -> u64 {
            let mut counts = [0u64; 1];
            let rc = unsafe {
                crate::facets::ffi_range_facet_counts(
                    seg,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    candidates.as_ptr(),
                    candidates.len(),
                    1,
                    mins.as_ptr(),
                    inclusive.as_ptr(),
                    maxs.as_ptr(),
                    inclusive.as_ptr(),
                    label.as_ptr(),
                    label_lens.as_ptr(),
                    counts.as_mut_ptr(),
                )
            };
            assert_eq!(rc, FfiStatus::Ok.code());
            counts[0]
        };

        // Negative control: no document in the *base* column holds 7000.
        assert_eq!(count_high(seg), 0);

        assert_eq!(
            add_generation(
                seg,
                dir,
                "val",
                "_0_4_Lucene90_0.dvm",
                "_0_4_Lucene90_0.dvd",
                "4_Lucene90_0",
            ),
            FfiStatus::Ok.code()
        );
        let want = expected.iter().filter(|v| **v == Some(7000)).count() as u64;
        assert!(want > 0);
        assert_eq!(count_high(seg), want);

        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    /// Attaching a generation twice for the same field replaces it rather
    /// than shadowing it -- what reopening onto a newer commit means.
    #[test]
    fn attaching_the_same_field_twice_replaces_the_generation() {
        let manifest = updates_manifest();
        let max_doc: i32 = manifest_get(&manifest, "max_doc").parse().unwrap();
        let expected = expected_column(&manifest, "expected_val");
        let dir = open_updates_dir();
        let seg = open_updates_segment(dir, &manifest);
        for _ in 0..2 {
            assert_eq!(
                add_generation(
                    seg,
                    dir,
                    "val",
                    "_0_4_Lucene90_0.dvm",
                    "_0_4_Lucene90_0.dvd",
                    "4_Lucene90_0",
                ),
                FfiStatus::Ok.code()
            );
        }
        {
            let registry = read_recovering(segments());
            let handle = registry.get(seg).unwrap();
            assert_eq!(handle.dv_generations.len(), 1);
        }
        assert_eq!(read_val_column(seg, max_doc), expected);
        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    /// A field whose `.fnm` says `docValuesGen == -1` has no generation to
    /// attach. The likeliest cause is the caller passing the *base* `.fnm`,
    /// where every field reads -1 -- so this must be a loud error, not a
    /// silent no-op that leaves every read on the superseded column.
    #[test]
    fn attaching_a_generation_to_a_never_updated_field_is_rejected() {
        let manifest = updates_manifest();
        let dir = open_updates_dir();
        let seg = open_updates_segment(dir, &manifest);
        let rc = add_generation(
            seg,
            dir,
            "keep",
            "_0_4_Lucene90_0.dvm",
            "_0_4_Lucene90_0.dvd",
            "4_Lucene90_0",
        );
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(last_error_message().contains("docValuesGen -1"));
        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    #[test]
    fn attaching_a_generation_for_an_unknown_field_is_rejected() {
        let manifest = updates_manifest();
        let dir = open_updates_dir();
        let seg = open_updates_segment(dir, &manifest);
        let rc = add_generation(
            seg,
            dir,
            "nope",
            "_0_4_Lucene90_0.dvm",
            "_0_4_Lucene90_0.dvd",
            "4_Lucene90_0",
        );
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(last_error_message().contains("unknown field nope"));
        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    #[test]
    fn attaching_a_generation_to_an_unknown_segment_handle_is_rejected() {
        let dir = open_updates_dir();
        let rc = add_generation(
            u64::MAX,
            dir,
            "val",
            "_0_4_Lucene90_0.dvm",
            "_0_4_Lucene90_0.dvd",
            "4_Lucene90_0",
        );
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        ffi_close_directory(dir);
    }

    /// A wrong codec suffix is a header mismatch, which must surface as a
    /// decode error rather than being accepted and silently mapped onto the
    /// wrong column.
    #[test]
    fn a_generation_with_the_wrong_codec_suffix_is_a_decode_error() {
        let manifest = updates_manifest();
        let dir = open_updates_dir();
        let seg = open_updates_segment(dir, &manifest);
        let rc = add_generation(
            seg,
            dir,
            "val",
            "_0_4_Lucene90_0.dvm",
            "_0_4_Lucene90_0.dvd",
            "2_Lucene90_0", // tag's generation suffix, not val's
        );
        assert_eq!(rc, FfiStatus::Decode.code());
        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    /// The generation's `.dvm` is parsed against a **one-field**
    /// `FieldInfos`, so a `.dvm` that describes some other field fails to
    /// parse instead of being accepted and read as this field's column.
    #[test]
    fn a_generation_dvm_describing_a_different_field_is_rejected() {
        let manifest = updates_manifest();
        let dir = open_updates_dir();
        let seg = open_updates_segment(dir, &manifest);
        // `tag`'s generation files, offered as `val`'s -- right segment,
        // right codec, wrong field.
        let rc = add_generation(
            seg,
            dir,
            "val",
            "_0_2_Lucene90_0.dvm",
            "_0_2_Lucene90_0.dvd",
            "2_Lucene90_0",
        );
        assert_ne!(rc, FfiStatus::Ok.code());
        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }

    #[test]
    fn a_generation_naming_a_missing_file_is_an_error_and_leaves_the_handle_usable() {
        let manifest = updates_manifest();
        let max_doc: i32 = manifest_get(&manifest, "max_doc").parse().unwrap();
        let dir = open_updates_dir();
        let seg = open_updates_segment(dir, &manifest);
        let rc = add_generation(
            seg,
            dir,
            "val",
            "_0_9_Lucene90_0.dvm",
            "_0_9_Lucene90_0.dvd",
            "9_Lucene90_0",
        );
        assert_ne!(rc, FfiStatus::Ok.code());
        // The failed attach left nothing behind: the handle still answers
        // from the base column.
        assert_eq!(
            read_val_column(seg, max_doc),
            (0..max_doc).map(|d| Some(d as i64)).collect::<Vec<_>>()
        );
        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }
}
