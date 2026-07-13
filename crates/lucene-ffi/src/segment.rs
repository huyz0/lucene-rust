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
use lucene_codecs::norms;
use lucene_codecs::postings::{DocInput, PosInput};

use crate::directory::read_whole_file;
use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::str_from_raw;
use crate::registry::{lock_recovering, segments, SegmentHandle};

/// Opens one segment's term dictionary and postings files.
///
/// - `dir_handle`: an [`crate::directory::ffi_open_directory`] handle.
/// - `fnm_name`/`tim_name`/`tip_name`/`tmd_name` (each a `(*const u8, len)`
///   pair): the segment's `.fnm`/`.tim`/`.tip`/`.tmd` file names, required.
/// - `doc_name`/`pos_name`: the segment's `.doc`/`.pos` file names, or a
///   null pointer (any `len`) to open none (matches `search_term_query`'s
///   own `doc_in: Option<&DocInput>` contract — some fields never need a
///   `.doc` file, see that function's doc comment).
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
        // SAFETY: caller contract guarantees each name pointer is valid for its
        // paired length, and `segment_id` is valid for 16 bytes.
        let (fnm_name, tim_name, tip_name, tmd_name, suffix) = unsafe {
            (
                str_from_raw(fnm_name as *const u8, fnm_name_len)?,
                str_from_raw(tim_name as *const u8, tim_name_len)?,
                str_from_raw(tip_name as *const u8, tip_name_len)?,
                str_from_raw(tmd_name as *const u8, tmd_name_len)?,
                str_from_raw(segment_suffix as *const u8, segment_suffix_len)?,
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
            let name = unsafe { str_from_raw(doc_name as *const u8, doc_name_len)? };
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
            let name = unsafe { str_from_raw(pos_name as *const u8, pos_name_len)? };
            let bytes = read_whole_file(dir_handle, name)?;
            PosInput::open(&bytes, &id, suffix).map_err(|e| {
                set_last_error(format!("opening .pos: {e}"));
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
                    str_from_raw(nvm_name as *const u8, nvm_name_len)?,
                    str_from_raw(nvd_name as *const u8, nvd_name_len)?,
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
                    str_from_raw(dvm_name as *const u8, dvm_name_len)?,
                    str_from_raw(dvd_name as *const u8, dvd_name_len)?,
                    str_from_raw(dv_suffix as *const u8, dv_suffix_len)?,
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

        let handle = lock_recovering(segments()).insert(SegmentHandle {
            fields,
            doc_bytes,
            pos_bytes,
            segment_id: id,
            segment_suffix: suffix.to_string(),
            max_doc,
            field_infos,
            norms_data,
            norms,
            dv_data,
            dv_meta,
        });
        // SAFETY: caller contract guarantees `out_handle` is valid for one write.
        unsafe {
            *out_handle = handle;
        }
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
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                8959,
                &mut handle as *mut _,
            )
        };
        (rc, handle)
    }

    #[test]
    fn open_segment_with_doc_file_then_close_roundtrips() {
        let dir_handle = open_dir();
        let (rc, seg_handle) = open_segment_with(dir_handle, Some("_0_Lucene104_0.doc"), None);
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_ne!(seg_handle, 0);

        assert!(lock_recovering(segments()).get(seg_handle).is_some());
        assert_eq!(ffi_close_segment(seg_handle), FfiStatus::Ok.code());
        assert!(lock_recovering(segments()).get(seg_handle).is_none());

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
        assert!(lock_recovering(segments()).get(seg_handle).is_some());

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
    fn scratch_dir_with_fixture_copies() -> std::path::PathBuf {
        let src = fixture_dir_path();
        let dst = std::env::temp_dir().join(format!(
            "lucene-ffi-segment-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dst).unwrap();
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
            let segments = lock_recovering(segments());
            let segment = segments.get(seg_handle).expect("segment handle");
            assert!(segment.norms.is_some());
            assert!(segment.norms_data.is_some());
        }
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }
}
