//! `ffi_search_points_range` (Points range query FFI exposure): wraps
//! `lucene_search::points_query::search_points_range` -- an already-existing,
//! already-tested Rust function; this module reimplements none of its BKD
//! read/traversal or packed-value comparison logic, only the C-ABI
//! marshaling around it (looking a field *name* up to a field *number* via
//! the segment's `field_infos`, reconstructing a `PointsReader` from the
//! segment's stored `.kdm`/`.kdi`/`.kdd` bytes, feeding the result into a
//! `VecCollector`).
//!
//! **Result-handle shape: reuses [`crate::registry::ResultsHandle`], no new
//! handle type.** `search_points_range` collects through this crate's plain
//! [`Collector`](lucene_search::Collector) trait into a flat, ascending
//! `Vec<i32>` of matched doc IDs -- exactly the same "unscored, unsorted
//! doc-ID list" shape `query.rs`'s `ffi_search_term_query`/
//! `ffi_search_boolean_query`/`ffi_search_phrase_query` already collect into
//! `ResultsHandle` (also built from a `VecCollector`, see that module's doc
//! comment). A `PointRangeQuery`-shaped match has no score and no sort key of
//! its own (real Lucene's own `PointRangeQuery` is `ConstantScoreQuery`-shaped,
//! see `search_points_range`'s own module doc for why there is no scored
//! sibling), so neither `ScoredResultsHandle` (a `(doc_id, f32 score)` pair)
//! nor `SortedResultsHandle` (a `(doc_id, i64 value)` pair) fit this result at
//! all -- `ResultsHandle` is the one existing handle whose element type
//! (`i32` alone) already matches. Read back via the existing `results.rs`'s
//! `ffi_results_len`/`ffi_results_copy`, released via the existing
//! `ffi_close_results` -- no new accessor trio or registry needed.
//!
//! **Points data plumbing**: `ffi_open_segment` (`segment.rs`) gained three
//! new optional parameters, `kdm_name`/`kdi_name`/`kdd_name`, for this task --
//! see that function's doc comment and [`crate::registry::SegmentHandle::points_data`]'s
//! doc comment for why the raw `.kdm`/`.kdi`/`.kdd` bytes are stored (not an
//! already-open `PointsReader`) and reconstructed fresh per query call, the
//! same self-referential-borrow reasoning `doc_bytes`/`pos_bytes` already
//! follow. A segment opened without them can't serve this call
//! ([`FfiStatus::InvalidArgument`] -- there is nothing to search, same "no
//! sensible fallback" reasoning `dv_data`'s absence already gets from
//! `sort.rs`).
//!
//! **Field-name lookup**: `min_packed`/`max_packed` are keyed to a field
//! *number* in `lucene_codecs::points::PointsReader`, but a caller only has a
//! field *name* over this C ABI -- looked up via the segment's `field_infos`,
//! same `find(|f| f.name == field)` pattern `sort.rs`'s `numeric_entry_for`
//! and `query.rs`'s `open_field_norms` already use. An unknown field name is
//! [`FfiStatus::InvalidArgument`] (same precedent as `numeric_entry_for`) --
//! unlike `search_points_range`'s own "unknown field *number* matches
//! nothing" convention (which only applies once a valid field number is
//! already in hand), a caller-supplied field *name* this segment's
//! `field_infos` has never heard of is a caller-side mistake, not "zero
//! points happen to exist for this field yet."
//!
//! **Explicit packed-length validation, not a caught panic**:
//! `search_points_range`'s own doc comment documents that a wrong
//! `min_packed`/`max_packed` length panics via a slice index (same contract
//! `resolve_points_range_doc_ids`/`PointsField::min_packed_value` document) --
//! reachable here from adversarial-but-otherwise-well-formed caller bytes
//! (unlike, e.g., `range_sort.rs`'s array-length check, which that module's
//! own doc comment notes is unreachable by construction). This module checks
//! `min_packed.len() == max_packed.len() == num_dims * bytes_per_dim` itself
//! first and returns [`FfiStatus::InvalidArgument`] instead of relying on
//! `guard`'s `catch_unwind` to turn a length mismatch into
//! [`FfiStatus::Panic`] -- a clearer, more specific status for an error shape
//! this boundary can predict in advance, without adding any new `unsafe`.
//!
//! **`live_docs` is always `None`**: no `.liv`/deletions FFI surface exists
//! yet anywhere in this crate (see `lib.rs`'s module doc) -- every doc is
//! treated as live, the same documented behavior every other query entry
//! point here already has for a bare `None`.

use std::os::raw::c_char;

use lucene_codecs::points::{self, PointsReader};
use lucene_search::points_query::search_points_range;
use lucene_search::VecCollector;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::{bytes_from_raw, str_from_raw};
use crate::registry::{lock_recovering, results, segments, ResultsHandle, SegmentHandle};

/// Looks `field`'s number up from `segment`'s `field_infos` -- shared helper,
/// same "unknown field name is a caller error" precedent as `sort.rs`'s
/// `numeric_entry_for`/`query.rs`'s `open_field_norms`.
fn field_number_for(segment: &SegmentHandle, field: &str) -> Result<i32, FfiStatus> {
    segment
        .field_infos
        .fields
        .iter()
        .find(|f| f.name == field)
        .map(|f| f.number)
        .ok_or_else(|| {
            set_last_error(format!("ffi_search_points_range: unknown field {field}"));
            FfiStatus::InvalidArgument
        })
}

/// Reconstructs a fresh [`PointsReader`] from `segment`'s stored
/// `.kdm`/`.kdi`/`.kdd` bytes (see [`crate::registry::SegmentHandle::points_data`]'s
/// doc comment for why this is reconstructed per call rather than cached),
/// or [`FfiStatus::InvalidArgument`] when the segment was opened without
/// points data.
fn open_points_reader(segment: &SegmentHandle) -> Result<PointsReader<'_>, FfiStatus> {
    let (kdm, kdi, kdd) = segment.points_data.as_ref().ok_or_else(|| {
        set_last_error("ffi_search_points_range: segment was opened without points data");
        FfiStatus::InvalidArgument
    })?;
    points::open(kdm, kdi, kdd, &segment.segment_id, "").map_err(|e| {
        set_last_error(format!("reopening points data: {e}"));
        FfiStatus::Decode
    })
}

/// Runs `search_points_range` for `field` against `segment_handle`: every
/// live doc whose packed BKD value (across every dimension) falls within the
/// inclusive `[min_packed, max_packed]` range, collected into a new
/// [`ResultsHandle`] written to `*out_results_handle` on success.
/// `min_packed`/`max_packed` must each be exactly `num_dims * bytes_per_dim`
/// bytes for `field` (see this module's doc comment) -- a mismatch is
/// [`FfiStatus::InvalidArgument`], not a panic. `live_docs` is always `None`
/// (see this module's doc comment).
///
/// # Safety
/// `field` must be valid for `field_len` bytes; `min_packed`/`max_packed`
/// must each be valid for their paired lengths; `out_results_handle` must be
/// valid for one `u64` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_search_points_range(
    segment_handle: u64,
    field: *const c_char,
    field_len: usize,
    min_packed: *const u8,
    min_packed_len: usize,
    max_packed: *const u8,
    max_packed_len: usize,
    out_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `field`/`min_packed`/`max_packed`
        // are valid for their paired lengths.
        let (field, min_packed, max_packed) = unsafe {
            (
                str_from_raw(field as *const u8, field_len)?,
                bytes_from_raw(min_packed, min_packed_len)?,
                bytes_from_raw(max_packed, max_packed_len)?,
            )
        };

        let segments_registry = lock_recovering(segments());
        let segment = segments_registry.get(segment_handle).ok_or_else(|| {
            set_last_error("ffi_search_points_range: unknown or already-closed segment handle");
            FfiStatus::InvalidHandle
        })?;

        let field_number = field_number_for(segment, field)?;
        let reader = open_points_reader(segment)?;

        // Explicit length check -- see this module's doc comment for why this
        // is a predictable `InvalidArgument` here rather than a caught panic.
        // A field with no points entry at all (`reader.field` returns `None`)
        // has no expected length to check against; `search_points_range`
        // itself already documents "unknown field number matches nothing, not
        // an error" for that case, so this check is simply skipped and the
        // call below runs (and returns empty).
        if let Some(points_field) = reader.field(field_number) {
            let expected_len = (points_field.num_dims * points_field.bytes_per_dim) as usize;
            if min_packed.len() != expected_len || max_packed.len() != expected_len {
                set_last_error(format!(
                    "ffi_search_points_range: min_packed/max_packed must be {expected_len} bytes for field {field}, got {} / {}",
                    min_packed.len(),
                    max_packed.len()
                ));
                return Err(FfiStatus::InvalidArgument);
            }
        }

        let mut collector = VecCollector::default();
        search_points_range(
            &reader,
            None,
            field_number,
            min_packed,
            max_packed,
            &mut collector,
        )
        .map_err(|e| {
            set_last_error(format!("search failed: {e}"));
            FfiStatus::Search
        })?;

        let handle = lock_recovering(results()).insert(ResultsHandle {
            docs: collector.docs,
        });
        // SAFETY: caller contract guarantees `out_results_handle` is valid for
        // one write.
        unsafe {
            *out_results_handle = handle;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::{ffi_close_directory, ffi_open_directory};
    use crate::registry::segments as segments_registry;
    use crate::results::{ffi_close_results, ffi_results_copy, ffi_results_len};
    use crate::segment::{ffi_close_segment, ffi_open_segment};

    use lucene_codecs::points::{self as points_codec, WritePointsField};
    use lucene_store::codec_util::ID_LENGTH;

    fn long_bytes(v: i64) -> [u8; 8] {
        ((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes()
    }

    /// Same segment ID as the real `doc_values_index` fixture's `_0.fnm`
    /// (`manifest.properties`'s `id_hex`) -- these tests reuse that fixture's
    /// `.fnm` (for real `field_infos`, so a field *name* -> *number* lookup
    /// has something real to resolve) alongside hand-built `.kdm`/`.kdi`/`.kdd`
    /// files, and every index-header check in this crate validates the
    /// segment ID it was opened with matches the ID baked into each file's
    /// header, so every file here must agree on this same ID.
    fn segment_id_bytes() -> [u8; ID_LENGTH] {
        let hex = "b13327c83170075fccfd10d7f050f83b";
        let mut id = [0u8; ID_LENGTH];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        id
    }

    /// Same single-dimension `LongPoint`-shaped fixture
    /// `points_query.rs`'s own Rust-level tests use: doc 0 -> 10, doc 1 -> 20,
    /// doc 2 -> 30, doc 3 -> 40, doc 4 -> 50, on field number 2 (the real
    /// `doc_values_index` fixture's "gcd" field, per its
    /// `manifest.properties`'s `field_numbers`).
    fn write_single_dim_kd_files(dir: &std::path::Path) {
        let points: Vec<(i32, Vec<u8>)> = vec![
            (0, long_bytes(10).to_vec()),
            (1, long_bytes(20).to_vec()),
            (2, long_bytes(30).to_vec()),
            (3, long_bytes(40).to_vec()),
            (4, long_bytes(50).to_vec()),
        ];
        let field = WritePointsField {
            field_number: 2,
            num_dims: 1,
            bytes_per_dim: 8,
            points,
        };
        let (kdm, kdi, kdd) = points_codec::write(&[field], 512, &segment_id_bytes(), "").unwrap();
        std::fs::write(dir.join("_0.kdm"), kdm).unwrap();
        std::fs::write(dir.join("_0.kdi"), kdi).unwrap();
        std::fs::write(dir.join("_0.kdd"), kdd).unwrap();
    }

    /// 2D fixture: doc 0 -> (0, 0), doc 1 -> (10, 10), doc 2 -> (20, 20),
    /// doc 3 -> (10, 100) -- exercises the multi-dimension path end to end
    /// through the FFI (not just Rust-level, see `points_query.rs`'s own test
    /// of the same shape).
    fn write_two_dim_kd_files(dir: &std::path::Path) {
        let pack = |a: i64, b: i64| -> Vec<u8> {
            let mut v = long_bytes(a).to_vec();
            v.extend_from_slice(&long_bytes(b));
            v
        };
        let points: Vec<(i32, Vec<u8>)> = vec![
            (0, pack(0, 0)),
            (1, pack(10, 10)),
            (2, pack(20, 20)),
            (3, pack(10, 100)),
        ];
        let field = WritePointsField {
            field_number: 2,
            num_dims: 2,
            bytes_per_dim: 8,
            points,
        };
        let (kdm, kdi, kdd) = points_codec::write(&[field], 512, &segment_id_bytes(), "").unwrap();
        std::fs::write(dir.join("_0.kdm"), kdm).unwrap();
        std::fs::write(dir.join("_0.kdi"), kdi).unwrap();
        std::fs::write(dir.join("_0.kdd"), kdd).unwrap();
    }

    /// A scratch directory holding the real fixture's `.fnm` (so `field_infos`
    /// has a "gcd" field at number 1, matching the points fixtures above)
    /// plus freshly written `.kdm`/`.kdi`/`.kdd` files -- this crate has no
    /// pre-generated fixture combining real field infos with these
    /// hand-built single/two-dimension point sets, so each test builds its
    /// own scratch directory (mirroring `segment.rs`'s
    /// `scratch_dir_with_fixture_copies` pattern).
    fn scratch_dir(write_points: impl FnOnce(&std::path::Path)) -> std::path::PathBuf {
        let dst = std::env::temp_dir().join(format!(
            "lucene-ffi-points-query-test-{}-{:?}-{}",
            std::process::id(),
            std::thread::current().id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dst).unwrap();
        let src = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/doc_values_index"
        );
        for name in [
            "_0.fnm",
            "_0_Lucene104_0.tim",
            "_0_Lucene104_0.tip",
            "_0_Lucene104_0.tmd",
        ] {
            std::fs::copy(format!("{src}/{name}"), dst.join(name)).unwrap();
        }
        write_points(&dst);
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

    #[allow(clippy::too_many_arguments)]
    fn open_segment_with_points(dir_handle: u64, with_points: bool) -> (i32, u64) {
        let fnm = "_0.fnm";
        let tim = "_0_Lucene104_0.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let suffix = "Lucene104_0";
        let kdm = "_0.kdm";
        let kdi = "_0.kdi";
        let kdd = "_0.kdd";
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
                std::ptr::null(), // doc_name
                0,
                std::ptr::null(), // pos_name
                0,
                std::ptr::null(), // nvm_name
                0,
                std::ptr::null(), // nvd_name
                0,
                std::ptr::null(), // dvm_name
                0,
                std::ptr::null(), // dvd_name
                0,
                std::ptr::null(), // dv_suffix
                0,
                if with_points {
                    kdm.as_ptr() as *const c_char
                } else {
                    std::ptr::null()
                },
                if with_points { kdm.len() } else { 0 },
                if with_points {
                    kdi.as_ptr() as *const c_char
                } else {
                    std::ptr::null()
                },
                if with_points { kdi.len() } else { 0 },
                if with_points {
                    kdd.as_ptr() as *const c_char
                } else {
                    std::ptr::null()
                },
                if with_points { kdd.len() } else { 0 },
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                5,
                &mut handle as *mut _,
            )
        };
        (rc, handle)
    }

    fn read_results(results_handle: u64) -> Vec<i32> {
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_results_len(results_handle, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        let mut buf = vec![0i32; len];
        assert_eq!(
            unsafe { ffi_results_copy(results_handle, buf.as_mut_ptr(), buf.len()) },
            FfiStatus::Ok.code()
        );
        buf
    }

    #[test]
    fn points_range_single_dim_matches_expected_docs_real_fixture() {
        let dir = scratch_dir(write_single_dim_kd_files);
        let dir_handle = open_dir_at(&dir);
        let (rc, seg_handle) = open_segment_with_points(dir_handle, true);
        assert_eq!(rc, FfiStatus::Ok.code());

        let field = "gcd";
        let min = long_bytes(15);
        let max = long_bytes(35);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_points_range(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                min.as_ptr(),
                min.len(),
                max.as_ptr(),
                max.len(),
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        // Differential: matches calling the underlying Rust function directly.
        let expected = {
            let kdm = std::fs::read(dir.join("_0.kdm")).unwrap();
            let kdi = std::fs::read(dir.join("_0.kdi")).unwrap();
            let kdd = std::fs::read(dir.join("_0.kdd")).unwrap();
            let reader = points::open(&kdm, &kdi, &kdd, &segment_id_bytes(), "").unwrap();
            let mut collector = VecCollector::default();
            search_points_range(&reader, None, 2, &min, &max, &mut collector).unwrap();
            collector.docs
        };
        assert_eq!(read_results(out), expected);
        assert_eq!(read_results(out), vec![1, 2]);

        ffi_close_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn points_range_boundary_values_are_inclusive_on_both_ends() {
        let dir = scratch_dir(write_single_dim_kd_files);
        let dir_handle = open_dir_at(&dir);
        let (rc, seg_handle) = open_segment_with_points(dir_handle, true);
        assert_eq!(rc, FfiStatus::Ok.code());

        let field = "gcd";
        let min = long_bytes(10);
        let max = long_bytes(30);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_points_range(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                min.as_ptr(),
                min.len(),
                max.as_ptr(),
                max.len(),
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(read_results(out), vec![0, 1, 2]);

        ffi_close_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn points_range_two_dimension_checks_every_dimension_independently() {
        let dir = scratch_dir(write_two_dim_kd_files);
        let dir_handle = open_dir_at(&dir);
        let (rc, seg_handle) = open_segment_with_points(dir_handle, true);
        assert_eq!(rc, FfiStatus::Ok.code());

        let pack = |a: i64, b: i64| -> Vec<u8> {
            let mut v = long_bytes(a).to_vec();
            v.extend_from_slice(&long_bytes(b));
            v
        };
        let field = "gcd";
        let min = pack(0, 0);
        let max = pack(20, 20);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_points_range(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                min.as_ptr(),
                min.len(),
                max.as_ptr(),
                max.len(),
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        // doc 3's dim-0 value (10) is in range but dim-1 (100) is not.
        assert_eq!(read_results(out), vec![0, 1, 2]);

        ffi_close_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn points_range_empty_range_matches_no_docs() {
        let dir = scratch_dir(write_single_dim_kd_files);
        let dir_handle = open_dir_at(&dir);
        let (rc, seg_handle) = open_segment_with_points(dir_handle, true);
        assert_eq!(rc, FfiStatus::Ok.code());

        let field = "gcd";
        let min = long_bytes(1000);
        let max = long_bytes(2000);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_points_range(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                min.as_ptr(),
                min.len(),
                max.as_ptr(),
                max.len(),
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(read_results(out).is_empty());

        ffi_close_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn points_range_unknown_field_name_is_invalid_argument() {
        let dir = scratch_dir(write_single_dim_kd_files);
        let dir_handle = open_dir_at(&dir);
        let (rc, seg_handle) = open_segment_with_points(dir_handle, true);
        assert_eq!(rc, FfiStatus::Ok.code());

        let field = "no-such-field";
        let min = long_bytes(0);
        let max = long_bytes(100);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_points_range(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                min.as_ptr(),
                min.len(),
                max.as_ptr(),
                max.len(),
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn points_range_segment_without_points_data_is_invalid_argument() {
        let dir = scratch_dir(write_single_dim_kd_files);
        let dir_handle = open_dir_at(&dir);
        let (rc, seg_handle) = open_segment_with_points(dir_handle, false); // no .kdm/.kdi/.kdd opened
        assert_eq!(rc, FfiStatus::Ok.code());

        let field = "gcd";
        let min = long_bytes(0);
        let max = long_bytes(100);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_points_range(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                min.as_ptr(),
                min.len(),
                max.as_ptr(),
                max.len(),
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn points_range_wrong_packed_length_is_invalid_argument_not_a_panic() {
        let dir = scratch_dir(write_single_dim_kd_files);
        let dir_handle = open_dir_at(&dir);
        let (rc, seg_handle) = open_segment_with_points(dir_handle, true);
        assert_eq!(rc, FfiStatus::Ok.code());

        let field = "gcd";
        let min = [0u8; 4]; // wrong length: field is 8 bytes/dim
        let max = [0u8; 4];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_points_range(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                min.as_ptr(),
                min.len(),
                max.as_ptr(),
                max.len(),
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn points_range_unknown_segment_handle_is_invalid_handle() {
        let field = "gcd";
        let min = long_bytes(0);
        let max = long_bytes(100);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_points_range(
                0xFFFF,
                field.as_ptr() as *const c_char,
                field.len(),
                min.as_ptr(),
                min.len(),
                max.as_ptr(),
                max.len(),
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    /// A directory handle passed where a segment handle is expected must be
    /// rejected by the registry-tag check, not accidentally accepted --
    /// see `handle.rs`'s `RegistryTag`.
    #[test]
    fn points_range_directory_handle_passed_as_segment_handle_is_invalid_handle() {
        let dir = scratch_dir(write_single_dim_kd_files);
        let dir_handle = open_dir_at(&dir);
        let field = "gcd";
        let min = long_bytes(0);
        let max = long_bytes(100);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_points_range(
                dir_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                min.as_ptr(),
                min.len(),
                max.as_ptr(),
                max.len(),
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn points_range_null_out_handle_is_null_pointer_error() {
        let dir = scratch_dir(write_single_dim_kd_files);
        let dir_handle = open_dir_at(&dir);
        let (rc, seg_handle) = open_segment_with_points(dir_handle, true);
        assert_eq!(rc, FfiStatus::Ok.code());

        let field = "gcd";
        let min = long_bytes(0);
        let max = long_bytes(100);
        let rc = unsafe {
            ffi_search_points_range(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                min.as_ptr(),
                min.len(),
                max.as_ptr(),
                max.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Swaps a live segment handle's `.kdd` bytes for garbage that fails
    /// `points::open`'s header check, so the "reopen points data for this
    /// query" `map_err` branch in `ffi_search_points_range` is reachable --
    /// this can't happen through the public API alone since `ffi_open_segment`
    /// already validates the `.kdd` bytes once at open time. Mirrors
    /// `query.rs`'s `corrupt_doc_bytes`/`range_sort.rs`'s `corrupt_dv_data`.
    fn corrupt_kdd_bytes(seg_handle: u64) {
        let mut segs = lock_recovering(segments_registry());
        let segment = segs.get_mut(seg_handle).expect("segment handle");
        segment.points_data = Some((vec![0u8; 4], vec![0u8; 4], vec![0u8; 4]));
    }

    #[test]
    fn points_range_decode_error_propagates_as_decode_status() {
        let dir = scratch_dir(write_single_dim_kd_files);
        let dir_handle = open_dir_at(&dir);
        let (rc, seg_handle) = open_segment_with_points(dir_handle, true);
        assert_eq!(rc, FfiStatus::Ok.code());
        corrupt_kdd_bytes(seg_handle);

        let field = "gcd";
        let min = long_bytes(0);
        let max = long_bytes(100);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_points_range(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                min.as_ptr(),
                min.len(),
                max.as_ptr(),
                max.len(),
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
        let _ = std::fs::remove_dir_all(dir);
    }
}
