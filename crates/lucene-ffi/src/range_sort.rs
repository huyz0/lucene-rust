//! `ffi_search_numeric_range_sorted_by_field`/
//! `ffi_search_numeric_range_sorted_by_field_multi_segment` (TopFieldCollector
//! FFI exposure): wraps `lucene_search::doc_value_query::search_numeric_range_sorted_by_field`
//! (single segment) and `lucene_search::multi_segment::search_numeric_range_sorted_by_field_multi_segment`
//! (multi-segment fan-out/merge) -- both already-existing, already-tested
//! Rust functions; this module reimplements none of their range-matching,
//! sorting, or doc-ID-translation logic, only the C-ABI marshaling around
//! them.
//!
//! **Result-handle shape: reuses [`crate::registry::SortedResultsHandle`],
//! no new handle type.** Both wrapped functions return
//! `Vec<lucene_search::collector::FieldValueDoc>`, i.e. a `(doc_id: i32,
//! value: i64)` pair per hit -- exactly the wire shape
//! `SortedResultsHandle` already carries for `sort.rs`'s
//! `ffi_sort_by_doc_value`/`ffi_sort_by_multi_valued_doc_value` (also a
//! `Vec<(i32, i64)>` of doc-value-ranked hits). Unlike `ScoredResultsHandle`
//! (an `f32` BM25 score, a different wire type and meaning -- see that
//! struct's own doc comment), a `FieldValueDoc`'s `value` is the identical
//! "arbitrary `i64` doc-value used for ordering" `SortedResultsHandle`'s doc
//! comment already describes; inventing a second, structurally identical
//! handle/registry here would be exactly the pointless duplication this
//! crate's other handle doc comments warn against (see e.g.
//! `directory_reader.rs`'s module doc reusing `ScoredResultsHandle` for the
//! same reason). Read back via the existing
//! `results_sorted.rs`'s `ffi_sorted_results_len`/`ffi_sorted_results_copy`,
//! released via the existing `ffi_close_sorted_results` -- no new accessor
//! trio needed either.
//!
//! **Multi-segment wire format: an array of already-open segment handles
//! plus a parallel array of caller-supplied `doc_base`s, not a
//! [`crate::registry::DirectoryReaderHandle`].** `DirectoryReaderHandle`
//! (task #51) carries no `.dvm`/`.dvd` doc-values data per segment at all
//! (see that handle's own doc comment) -- there is nothing for this
//! function to read a doc-value from if it took a reader handle. Each
//! already-opened [`crate::segment::ffi_open_segment`] handle already
//! carries its own `dv_data`/`dv_meta` (task #40's doc-values plumbing), so
//! the multi-segment entry point instead takes a flat array of segment
//! handles it looks up in the existing `segments()` registry, one at a
//! time, building the exact `lucene_search::multi_segment::DocValueSegment`
//! list that function already expects -- `doc_base` has no derivable value
//! from a bare segment handle (a segment doesn't know its own position in a
//! larger commit), so the caller supplies it explicitly, the same
//! responsibility `DocValueSegment::doc_base`'s own doc comment already
//! assigns to whoever builds the list.
//!
//! **Direction wire encoding**: `direction: i32`, `0` =
//! [`SortDirection::Ascending`], `1` = [`SortDirection::Descending`], any
//! other value is [`FfiStatus::InvalidArgument`] -- the same
//! `0`/`1`-selector convention `sort.rs`'s `ffi_sort_by_multi_valued_doc_value`
//! already established for [`ValueSelector`].
//!
//! **Missing-value policy and field lookup**: identical wire shape and
//! semantics to `sort.rs`'s existing `missing_is_default`/`missing_default`
//! pair and `numeric_entry_for` field-name -> `NumericEntry` lookup -- both
//! reused directly (`pub(crate)` in `sort.rs`) rather than duplicated.
//!
//! **`live_docs` is always `None`**: no `.liv`/deletions FFI surface exists
//! yet anywhere in this crate (see `lib.rs`'s module doc) -- every doc in
//! `0..max_doc` is treated as live, the same documented behavior every other
//! query entry point here already has for a bare `None`.

use std::os::raw::c_char;

use lucene_search::doc_value_query;
#[cfg(test)]
use lucene_search::doc_value_query::MissingValue;
use lucene_search::multi_segment::{self, DocValueSegment};
use lucene_search::SortDirection;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::str_from_raw;
use crate::registry::{lock_recovering, segments, sorted_results, SortedResultsHandle};
use crate::sort::{map_sort_error, missing_value, numeric_entry_for};

/// Parses `direction`'s `0`/`1` wire encoding into a [`SortDirection`], or
/// `Err(InvalidArgument)` for any other value -- see this module's doc
/// comment.
fn sort_direction(direction: i32) -> Result<SortDirection, FfiStatus> {
    match direction {
        0 => Ok(SortDirection::Ascending),
        1 => Ok(SortDirection::Descending),
        _ => {
            set_last_error(format!(
                "ffi_search_numeric_range_sorted_by_field: invalid direction {direction} (expected 0=Ascending or 1=Descending)"
            ));
            Err(FfiStatus::InvalidArgument)
        }
    }
}

/// Reads `len` `u64`s from `ptr` into an owned `Vec<u64>` -- same
/// null-only-when-`len==0` convention as [`crate::raw::bytes_from_raw`].
///
/// # Safety
/// `ptr` must be valid for reads of `len` `u64`s (or null when `len == 0`).
unsafe fn u64_slice_from_raw(ptr: *const u64, len: usize) -> Result<Vec<u64>, FfiStatus> {
    if ptr.is_null() {
        return if len == 0 {
            Ok(Vec::new())
        } else {
            Err(FfiStatus::NullPointer)
        };
    }
    // SAFETY: caller contract guarantees `ptr` is valid for `len` `u64`s.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
}

/// Reads `len` `i32`s from `ptr` into an owned `Vec<i32>` -- same convention
/// as [`u64_slice_from_raw`].
///
/// # Safety
/// `ptr` must be valid for reads of `len` `i32`s (or null when `len == 0`).
unsafe fn i32_slice_from_raw(ptr: *const i32, len: usize) -> Result<Vec<i32>, FfiStatus> {
    if ptr.is_null() {
        return if len == 0 {
            Ok(Vec::new())
        } else {
            Err(FfiStatus::NullPointer)
        };
    }
    // SAFETY: caller contract guarantees `ptr` is valid for `len` `i32`s.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
}

/// Runs `search_numeric_range_sorted_by_field` against `segment_handle`:
/// matches every live doc whose `range_field` NUMERIC doc-value falls in
/// `[min, max]`, ranks the matches by `sort_field`'s NUMERIC doc-value per
/// `direction`, and keeps the best `top_n` hits (ties broken by ascending
/// doc ID) in a new [`SortedResultsHandle`] written to
/// `*out_sorted_results_handle`. `range_field` and `sort_field` may name the
/// same field or different ones (see the wrapped function's own doc
/// comment). `missing_is_default`/`missing_default` select
/// [`MissingValue::Exclude`] (`false`) or [`MissingValue::Default`] (`true`)
/// for a candidate with no value for `sort_field` -- same convention as
/// `sort.rs`'s functions.
///
/// # Safety
/// `range_field` must be valid for `range_field_len` bytes; `sort_field`
/// must be valid for `sort_field_len` bytes; `out_sorted_results_handle`
/// must be valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_search_numeric_range_sorted_by_field(
    segment_handle: u64,
    range_field: *const c_char,
    range_field_len: usize,
    min: i64,
    max: i64,
    sort_field: *const c_char,
    sort_field_len: usize,
    direction: i32,
    missing_is_default: bool,
    missing_default: i64,
    top_n: usize,
    out_sorted_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_sorted_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let direction = sort_direction(direction)?;
        // SAFETY: caller contract guarantees `range_field`/`sort_field` are
        // valid for their paired lengths.
        let (range_field, sort_field) = unsafe {
            (
                str_from_raw(range_field as *const u8, range_field_len)?,
                str_from_raw(sort_field as *const u8, sort_field_len)?,
            )
        };

        let segments_registry = lock_recovering(segments());
        let segment = segments_registry.get(segment_handle).ok_or_else(|| {
            set_last_error(
                "ffi_search_numeric_range_sorted_by_field: unknown or already-closed segment handle",
            );
            FfiStatus::InvalidHandle
        })?;

        let range_entry = numeric_entry_for(segment, range_field)?;
        let sort_entry = numeric_entry_for(segment, sort_field)?;
        // `dv_meta` being `Some` (checked by `numeric_entry_for` above)
        // implies `dv_data` is also `Some` -- see `SegmentHandle`'s doc
        // comment.
        let dv_data = segment.dv_data.as_deref().unwrap_or(&[]);

        let hits = doc_value_query::search_numeric_range_sorted_by_field(
            dv_data,
            range_entry,
            None,
            segment.max_doc,
            min,
            max,
            sort_entry,
            direction,
            missing_value(missing_is_default, missing_default),
            top_n,
        )
        .map_err(map_sort_error)?;

        let handle = lock_recovering(sorted_results()).insert(SortedResultsHandle {
            pairs: hits.into_iter().map(|h| (h.doc_id, h.value)).collect(),
        });
        // SAFETY: caller contract guarantees `out_sorted_results_handle` is
        // valid for one write.
        unsafe {
            *out_sorted_results_handle = handle;
        }
        Ok(())
    })
}

/// Multi-segment sibling of [`ffi_search_numeric_range_sorted_by_field`]:
/// runs the same range-match-then-sort against every segment named in
/// `segment_handles`/`doc_bases` (parallel arrays of length
/// `segment_count` -- element `i` is `(segment_handles[i], doc_bases[i])`,
/// see this module's doc comment for why a flat handle array rather than a
/// [`crate::registry::DirectoryReaderHandle`]), translates each segment's
/// local doc IDs to global via its `doc_base`, and merges into the globally
/// top-`top_n` hits (via
/// `lucene_search::multi_segment::search_numeric_range_sorted_by_field_multi_segment`)
/// in a new [`SortedResultsHandle`].
///
/// # Safety
/// `range_field`/`sort_field` must be valid for their paired lengths;
/// `segment_handles` must be valid for `segment_count` `u64`s (or null when
/// `segment_count == 0`); `doc_bases` must be valid for `segment_count`
/// `i32`s (or null when `segment_count == 0`); `out_sorted_results_handle`
/// must be valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_search_numeric_range_sorted_by_field_multi_segment(
    segment_handles: *const u64,
    doc_bases: *const i32,
    segment_count: usize,
    range_field: *const c_char,
    range_field_len: usize,
    min: i64,
    max: i64,
    sort_field: *const c_char,
    sort_field_len: usize,
    direction: i32,
    missing_is_default: bool,
    missing_default: i64,
    top_n: usize,
    out_sorted_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_sorted_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let direction = sort_direction(direction)?;
        // SAFETY: caller contract guarantees `range_field`/`sort_field` are
        // valid for their paired lengths, and (per this function's `#
        // Safety` section) `segment_handles`/`doc_bases` are valid for
        // `segment_count` elements each.
        let (range_field, sort_field, segment_handles, doc_bases) = unsafe {
            let range_field = str_from_raw(range_field as *const u8, range_field_len)?;
            let sort_field = str_from_raw(sort_field as *const u8, sort_field_len)?;
            let segment_handles = u64_slice_from_raw(segment_handles, segment_count)?;
            let doc_bases = i32_slice_from_raw(doc_bases, segment_count)?;
            (range_field, sort_field, segment_handles, doc_bases)
        };
        // `segment_handles`/`doc_bases` are both read via `_slice_from_raw`
        // with the same `segment_count`, so they're always the same length
        // here (or one already errored above) -- no length-mismatch check
        // needed, unlike a caller-supplied-length-per-array API would need.

        let segments_registry = lock_recovering(segments());
        let mut opened = Vec::with_capacity(segment_handles.len());
        for &handle in &segment_handles {
            let segment = segments_registry.get(handle).ok_or_else(|| {
                set_last_error(
                    "ffi_search_numeric_range_sorted_by_field_multi_segment: unknown or already-closed segment handle",
                );
                FfiStatus::InvalidHandle
            })?;
            opened.push(segment);
        }

        let mut doc_value_segments = Vec::with_capacity(opened.len());
        for (segment, &doc_base) in opened.iter().zip(&doc_bases) {
            let range_entry = numeric_entry_for(segment, range_field)?;
            let sort_entry = numeric_entry_for(segment, sort_field)?;
            let dv_data = segment.dv_data.as_deref().unwrap_or(&[]);
            doc_value_segments.push(DocValueSegment {
                doc_values_data: dv_data,
                range_entry,
                sort_entry,
                live_docs: None,
                max_doc: segment.max_doc,
                doc_base,
            });
        }

        let hits = multi_segment::search_numeric_range_sorted_by_field_multi_segment(
            &doc_value_segments,
            min,
            max,
            direction,
            missing_value(missing_is_default, missing_default),
            top_n,
        )
        .map_err(map_sort_error)?;

        let handle = lock_recovering(sorted_results()).insert(SortedResultsHandle {
            pairs: hits.into_iter().map(|h| (h.doc_id, h.value)).collect(),
        });
        // SAFETY: caller contract guarantees `out_sorted_results_handle` is
        // valid for one write.
        unsafe {
            *out_sorted_results_handle = handle;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::{ffi_close_directory, ffi_open_directory};
    use crate::registry::segments as segments_registry;
    use crate::results_sorted::{
        ffi_close_sorted_results, ffi_sorted_results_copy, ffi_sorted_results_len,
    };
    use crate::segment::{ffi_close_segment, ffi_open_segment};

    fn id_from_hex(hex: &str) -> [u8; 16] {
        let mut id = [0u8; 16];
        for i in 0..16 {
            id[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        id
    }

    struct Manifest {
        kv: Vec<(String, String)>,
    }

    impl Manifest {
        fn load(dir: &str) -> Self {
            let text = std::fs::read_to_string(format!("{dir}manifest.properties"))
                .expect("run fixtures generator first");
            let kv = text
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            Manifest { kv }
        }

        fn get(&self, key: &str) -> &str {
            self.kv
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
        }
    }

    fn open_dir(path: &str) -> u64 {
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

    fn open_segment(dir_handle: u64, manifest: &Manifest) -> u64 {
        let fnm = "_0.fnm";
        let tim = "_0_Lucene104_0.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let dvm = "_0_Lucene90_0.dvm";
        let dvd = "_0_Lucene90_0.dvd";
        let suffix = "Lucene104_0";
        let dv_suffix = "Lucene90_0";
        let id = id_from_hex(manifest.get("id_hex"));
        let max_doc: i32 = manifest.get("max_doc").parse().unwrap();
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
                dvm.as_ptr() as *const c_char,
                dvm.len(),
                dvd.as_ptr() as *const c_char,
                dvd.len(),
                dv_suffix.as_ptr() as *const c_char,
                dv_suffix.len(),
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

    fn read_sorted(handle: u64) -> Vec<(i32, i64)> {
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_sorted_results_len(handle, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        let mut doc_ids = vec![0i32; len];
        let mut values = vec![0i64; len];
        assert_eq!(
            unsafe {
                ffi_sorted_results_copy(
                    handle,
                    doc_ids.as_mut_ptr(),
                    values.as_mut_ptr(),
                    doc_ids.len(),
                )
            },
            FfiStatus::Ok.code()
        );
        doc_ids.into_iter().zip(values).collect()
    }

    fn dv_dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/doc_values_index/"
        )
        .to_string()
    }

    // --- ffi_search_numeric_range_sorted_by_field: real-Lucene fixture --
    // "gcd" in [1000, 1100] matches docs 0,1,2,4 (values 1000,1025,1075,1050;
    // doc 3's 1200 is out of range); sorted by "varying"
    // (-100,7,42,1000,-3 for docs 0..4) -- same fixture/values
    // `doc_value_query.rs`'s own
    // `search_numeric_range_sorted_by_field_end_to_end_real_fixture` test uses.

    #[test]
    fn range_sorted_by_field_ascending_real_fixture() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let range_field = "gcd";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field(
                seg_handle,
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0, // Ascending
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        // Differential: matches calling the underlying Rust function directly.
        let expected = {
            let bytes = std::fs::read(format!("{dir}_0_Lucene90_0.dvd")).unwrap();
            let dvm_bytes = std::fs::read(format!("{dir}_0_Lucene90_0.dvm")).unwrap();
            let fnm_bytes = std::fs::read(format!("{dir}_0.fnm")).unwrap();
            let id = id_from_hex(manifest.get("id_hex"));
            let field_infos = lucene_codecs::field_infos::parse(&fnm_bytes, &id, "").unwrap();
            let (_version, dv_meta) =
                lucene_codecs::doc_values::parse_meta(&dvm_bytes, &id, "Lucene90_0", &field_infos)
                    .unwrap();
            let field_number = |name: &str| -> i32 {
                field_infos
                    .fields
                    .iter()
                    .find(|f| f.name == name)
                    .unwrap()
                    .number
            };
            let range_entry = dv_meta.numeric_entry(field_number(range_field)).unwrap();
            let sort_entry = dv_meta.numeric_entry(field_number(sort_field)).unwrap();
            let max_doc: i32 = manifest.get("max_doc").parse().unwrap();
            doc_value_query::search_numeric_range_sorted_by_field(
                &bytes,
                range_entry,
                None,
                max_doc,
                1000,
                1100,
                sort_entry,
                SortDirection::Ascending,
                MissingValue::Exclude,
                10,
            )
            .unwrap()
            .into_iter()
            .map(|h| (h.doc_id, h.value))
            .collect::<Vec<_>>()
        };
        assert_eq!(read_sorted(out), expected);
        assert_eq!(read_sorted(out), vec![(0, -100), (4, -3), (1, 7), (2, 42)]);

        ffi_close_sorted_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_sorted_by_field_descending_and_top_n_real_fixture() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let range_field = "gcd";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field(
                seg_handle,
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                1, // Descending
                false,
                0,
                2,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(read_sorted(out), vec![(2, 42), (1, 7)]);

        ffi_close_sorted_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_sorted_by_field_missing_default_policy_real_fixture() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        // Widen the range to also match doc 3 (gcd=1200), then sort by
        // "sparse" (5, NONE, 15, NONE, 25 -- doc 3 has no value).
        let range_field = "gcd";
        let sort_field = "sparse";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field(
                seg_handle,
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1200,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                true,
                1_000_000,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(
            read_sorted(out),
            vec![(0, 5), (2, 15), (4, 25), (1, 1_000_000), (3, 1_000_000)]
        );

        ffi_close_sorted_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_sorted_by_field_invalid_direction_is_invalid_argument() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let range_field = "gcd";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field(
                seg_handle,
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                2, // invalid
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_sorted_by_field_unknown_range_field_is_invalid_argument() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let range_field = "no-such-field";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field(
                seg_handle,
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_sorted_by_field_unknown_sort_field_is_invalid_argument() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let range_field = "gcd";
        let sort_field = "no-such-field";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field(
                seg_handle,
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_sorted_by_field_segment_without_doc_values_is_invalid_argument() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        );
        let dir_handle = open_dir(path);
        let fnm = "_0.fnm";
        let tim = "_0_Lucene104_0.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let suffix = "Lucene104_0";
        let hex = "bea914ffd84e035aaac43aca30240b47";
        let id = id_from_hex(hex);
        let mut seg_handle: u64 = 0;
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
                &mut seg_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let range_field = "body";
        let sort_field = "body";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field(
                seg_handle,
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                0,
                0,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_sorted_by_field_unknown_segment_handle_is_invalid_handle() {
        let range_field = "gcd";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field(
                0xFFFF,
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn range_sorted_by_field_null_out_handle_is_null_pointer_error() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let range_field = "gcd";
        let sort_field = "varying";
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field(
                seg_handle,
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// Swaps a live segment handle's `.dvd` bytes for garbage that fails
    /// mid-decode -- mirrors `sort.rs`'s `corrupt_dv_data` helper.
    fn corrupt_dv_data(seg_handle: u64) {
        let mut segs = lock_recovering(segments_registry());
        let segment = segs.get_mut(seg_handle).expect("segment handle");
        segment.dv_data = Some(vec![]);
    }

    #[test]
    fn range_sorted_by_field_decode_error_propagates_as_decode_status() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);
        corrupt_dv_data(seg_handle);

        let range_field = "gcd";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field(
                seg_handle,
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    // --- ffi_search_numeric_range_sorted_by_field_multi_segment: two
    //     "segments" built from the same fixture, second at doc_base=5 --
    //     mirrors `multi_segment.rs`'s own doc-ID-translation tests. ---

    #[test]
    fn range_sorted_by_field_multi_segment_translates_doc_ids_real_fixture() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg0 = open_segment(dir_handle, &manifest);
        let seg1 = open_segment(dir_handle, &manifest);

        let segment_handles = [seg0, seg1];
        let doc_bases = [0i32, 5];
        let range_field = "gcd";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field_multi_segment(
                segment_handles.as_ptr(),
                doc_bases.as_ptr(),
                segment_handles.len(),
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        // Each segment independently matches docs 0,1,2,4 (values -100,7,42,-3);
        // segment 1's are translated by +5 -- merged ascending: -100 ties at
        // (0,5), -3 ties at (4,9), 7 ties at (1,6), 42 ties at (2,7).
        assert_eq!(
            read_sorted(out),
            vec![
                (0, -100),
                (5, -100),
                (4, -3),
                (9, -3),
                (1, 7),
                (6, 7),
                (2, 42),
                (7, 42)
            ]
        );

        ffi_close_sorted_results(out);
        ffi_close_segment(seg0);
        ffi_close_segment(seg1);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_sorted_by_field_multi_segment_descending_and_top_n_real_fixture() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg0 = open_segment(dir_handle, &manifest);
        let seg1 = open_segment(dir_handle, &manifest);

        let segment_handles = [seg0, seg1];
        let doc_bases = [0i32, 5];
        let range_field = "gcd";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field_multi_segment(
                segment_handles.as_ptr(),
                doc_bases.as_ptr(),
                segment_handles.len(),
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                1, // Descending
                false,
                0,
                3,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(read_sorted(out), vec![(2, 42), (7, 42), (1, 7)]);

        ffi_close_sorted_results(out);
        ffi_close_segment(seg0);
        ffi_close_segment(seg1);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_sorted_by_field_multi_segment_no_segments_returns_empty() {
        let range_field = "gcd";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field_multi_segment(
                std::ptr::null(),
                std::ptr::null(),
                0,
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(read_sorted(out).is_empty());
        ffi_close_sorted_results(out);
    }

    #[test]
    fn range_sorted_by_field_multi_segment_invalid_direction_is_invalid_argument() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg0 = open_segment(dir_handle, &manifest);

        let segment_handles = [seg0];
        let doc_bases = [0i32];
        let range_field = "gcd";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field_multi_segment(
                segment_handles.as_ptr(),
                doc_bases.as_ptr(),
                segment_handles.len(),
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                7, // invalid
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg0);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_sorted_by_field_multi_segment_unknown_segment_handle_is_invalid_handle() {
        let segment_handles = [0xFFFFu64];
        let doc_bases = [0i32];
        let range_field = "gcd";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field_multi_segment(
                segment_handles.as_ptr(),
                doc_bases.as_ptr(),
                segment_handles.len(),
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn range_sorted_by_field_multi_segment_unknown_field_is_invalid_argument() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg0 = open_segment(dir_handle, &manifest);

        let segment_handles = [seg0];
        let doc_bases = [0i32];
        let range_field = "no-such-field";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field_multi_segment(
                segment_handles.as_ptr(),
                doc_bases.as_ptr(),
                segment_handles.len(),
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg0);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_sorted_by_field_multi_segment_null_out_handle_is_null_pointer_error() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg0 = open_segment(dir_handle, &manifest);

        let segment_handles = [seg0];
        let doc_bases = [0i32];
        let range_field = "gcd";
        let sort_field = "varying";
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field_multi_segment(
                segment_handles.as_ptr(),
                doc_bases.as_ptr(),
                segment_handles.len(),
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());

        ffi_close_segment(seg0);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_sorted_by_field_multi_segment_null_segment_handles_with_nonzero_count_is_null_pointer_error(
    ) {
        let doc_bases = [0i32];
        let range_field = "gcd";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field_multi_segment(
                std::ptr::null(),
                doc_bases.as_ptr(),
                1,
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn range_sorted_by_field_multi_segment_null_doc_bases_with_nonzero_count_is_null_pointer_error()
    {
        let segment_handles = [0u64];
        let range_field = "gcd";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field_multi_segment(
                segment_handles.as_ptr(),
                std::ptr::null(),
                1,
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    /// Same idea as `range_sorted_by_field_decode_error_propagates_as_decode_status`,
    /// exercised through the multi-segment entry point.
    #[test]
    fn range_sorted_by_field_multi_segment_decode_error_propagates_as_decode_status() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg0 = open_segment(dir_handle, &manifest);
        corrupt_dv_data(seg0);

        let segment_handles = [seg0];
        let doc_bases = [0i32];
        let range_field = "gcd";
        let sort_field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_search_numeric_range_sorted_by_field_multi_segment(
                segment_handles.as_ptr(),
                doc_bases.as_ptr(),
                segment_handles.len(),
                range_field.as_ptr() as *const c_char,
                range_field.len(),
                1000,
                1100,
                sort_field.as_ptr() as *const c_char,
                sort_field.len(),
                0,
                false,
                0,
                10,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());

        ffi_close_segment(seg0);
        ffi_close_directory(dir_handle);
    }
}
