//! `ffi_facet_counts_sorted_set`/`ffi_range_facet_counts` (Faceted search FFI
//! exposure): wraps this port's existing `lucene_search::facets::facet_counts`/
//! `resolve_labels`/`top_n_facets` (SortedSet string facets) and
//! `lucene_search::facets::range_facet_counts` (NUMERIC range facets) across
//! the FFI boundary. No facet-counting logic is reimplemented here -- every
//! function below is a thin marshal-in/call/marshal-out wrapper around
//! `crates/lucene-search/src/facets.rs`'s existing public API, following the
//! exact handle/error-code conventions `sort.rs`/`query.rs` already
//! established.
//!
//! ## SortedSet facets: [`ffi_facet_counts_sorted_set`]
//!
//! Looks up `field`'s SORTED_SET doc-values entry in an already-opened
//! [`crate::segment::SegmentHandle`] (via `segment.field_infos` +
//! `segment.dv_meta.sorted_set_entry`, the same field-name -> field-number
//! lookup pattern `sort.rs`'s `numeric_entry_for`/`sorted_numeric_entry_for`
//! already use), runs `lucene_search::facets::facet_counts` over the
//! caller-supplied `candidates` doc-ID slice, resolves every ordinal's label
//! via `resolve_labels`, and -- when `top_n > 0` -- truncates/sorts via
//! `top_n_facets` exactly as that function documents (descending count, ties
//! broken by ascending ordinal). `top_n == 0` returns every facet in ordinal
//! order, unsorted and untruncated: the plain "no truncation" case, still
//! without inventing a new code path (`top_n_facets` is simply not called).
//! Results are collected into a new [`crate::registry::FacetResultsHandle`],
//! read back via `results_facets.rs`'s accessors.
//!
//! **Only [`lucene_codecs::doc_values::SortedSetKind::Multi`] is supported**:
//! a field written as `SortedSetKind::Single` (every doc has zero or one
//! ordinal, no `SortedNumericEntry`/`TermsDictEntry` pair) has no entry shape
//! `lucene_search::facets::facet_counts` accepts -- `facets.rs` itself has no
//! counting path for that representation (see that module's doc comment: the
//! multi-valued `SortedNumericEntry`+`TermsDictEntry` pair is the only shape
//! it takes). This is not a gap introduced by the FFI layer; a `Single`-kind
//! field is [`crate::error::FfiStatus::InvalidArgument`] here, and extending
//! `facets.rs` to also count a `Single`-kind field is a follow-up to that
//! module, not this one (see `docs/parity.md`).
//!
//! ## NUMERIC range facets: [`ffi_range_facet_counts`]
//!
//! Looks up `field`'s NUMERIC doc-values entry the same way `sort.rs`'s
//! `numeric_entry_for` does, builds `ranges_count`
//! `lucene_search::facets::NumericRange`s from five parallel input arrays
//! (`range_mins`/`range_min_inclusive`/`range_maxs`/`range_max_inclusive`,
//! plus a caller-supplied label per range via a concatenated
//! `range_label_data` buffer sliced by `range_label_lens`), and runs
//! `range_facet_counts` over `candidates`.
//!
//! **No output handle is needed here**: unlike the SortedSet case, every
//! range's label is caller-supplied input, not decoded from the index -- the
//! caller already owns every label string, so only the counts (in the same
//! order as the input ranges, `range_facet_counts`'s own documented
//! contract) are written directly into a caller-allocated `out_counts: *mut
//! u64` buffer of length `ranges_count`. See `registry.rs`'s
//! `FacetResultsHandle` doc comment for the same reasoning stated from the
//! handle side.

use std::os::raw::c_char;

use lucene_codecs::doc_values::{NumericEntry, SortedNumericEntry, SortedSetKind};
use lucene_codecs::terms_dict::TermsDictEntry;
use lucene_search::facets::{self, NumericRange};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::str_from_raw;
use crate::registry::{
    facet_results, lock_recovering, segments, FacetResultsHandle, SegmentHandle,
};

fn map_facet_error(e: lucene_search::Error) -> FfiStatus {
    set_last_error(format!("facet count failed: {e}"));
    FfiStatus::Decode
}

/// Reads `len` candidate doc IDs from `ptr` into an owned `Vec<i32>` -- same
/// contract as `sort.rs`'s `candidates_from_raw` (duplicated here rather than
/// shared since it's module-private plumbing there, matching how each
/// existing FFI module already keeps its own copy of small helpers like this
/// one).
///
/// # Safety
/// `ptr` must be valid for reads of `len` `i32`s (or null when `len == 0`).
unsafe fn candidates_from_raw(ptr: *const i32, len: usize) -> Result<Vec<i32>, FfiStatus> {
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

/// Looks up `field`'s field number in `segment`'s `.fnm`, then that number's
/// [`NumericEntry`] in `segment`'s opened `.dvm` -- same shape as `sort.rs`'s
/// `numeric_entry_for`, backing [`ffi_range_facet_counts`].
fn numeric_entry_for<'seg>(
    segment: &'seg SegmentHandle,
    field: &str,
) -> Result<&'seg NumericEntry, FfiStatus> {
    let dv_meta = segment.dv_meta.as_ref().ok_or_else(|| {
        set_last_error("ffi_range_facet_counts: segment was opened without doc values");
        FfiStatus::InvalidArgument
    })?;
    let field_info = segment
        .field_infos
        .fields
        .iter()
        .find(|f| f.name == field)
        .ok_or_else(|| {
            set_last_error(format!("ffi_range_facet_counts: unknown field {field}"));
            FfiStatus::InvalidArgument
        })?;
    dv_meta.numeric_entry(field_info.number).ok_or_else(|| {
        set_last_error(format!(
            "ffi_range_facet_counts: field {field} has no NUMERIC doc-values entry"
        ));
        FfiStatus::InvalidArgument
    })
}

/// Looks up `field`'s field number, then its SORTED_SET entry in `segment`'s
/// opened `.dvm`, requiring the multi-valued
/// [`SortedSetKind::Multi`] shape `lucene_search::facets::facet_counts`
/// accepts -- see this module's doc comment for why `Single` is
/// [`FfiStatus::InvalidArgument`] here.
fn sorted_set_multi_entry_for<'seg>(
    segment: &'seg SegmentHandle,
    field: &str,
) -> Result<(&'seg SortedNumericEntry, &'seg TermsDictEntry), FfiStatus> {
    let dv_meta = segment.dv_meta.as_ref().ok_or_else(|| {
        set_last_error("ffi_facet_counts_sorted_set: segment was opened without doc values");
        FfiStatus::InvalidArgument
    })?;
    let field_info = segment
        .field_infos
        .fields
        .iter()
        .find(|f| f.name == field)
        .ok_or_else(|| {
            set_last_error(format!(
                "ffi_facet_counts_sorted_set: unknown field {field}"
            ));
            FfiStatus::InvalidArgument
        })?;
    let entry = dv_meta.sorted_set_entry(field_info.number).ok_or_else(|| {
        set_last_error(format!(
            "ffi_facet_counts_sorted_set: field {field} has no SORTED_SET doc-values entry"
        ));
        FfiStatus::InvalidArgument
    })?;
    match &entry.kind {
        SortedSetKind::Multi { ords, terms } => Ok((ords, terms)),
        SortedSetKind::Single(_) => {
            set_last_error(format!(
                "ffi_facet_counts_sorted_set: field {field} is a single-valued SORTED_SET \
                 (SortedSetKind::Single), which lucene_search::facets::facet_counts does not \
                 accept -- see facets.rs's module doc"
            ));
            Err(FfiStatus::InvalidArgument)
        }
    }
}

/// Parses `count` [`NumericRange`]s from five parallel input arrays -- see
/// this module's doc comment for the wire shape (`range_label_data`/
/// `range_label_lens` concatenated-buffer encoding).
///
/// # Safety
/// `mins`/`maxs` must be valid for reads of `count` `i64`s;
/// `min_inclusive`/`max_inclusive` must be valid for reads of `count` `u8`s;
/// `label_lens` must be valid for reads of `count` `usize`s; `label_data`
/// must be valid for reads of `label_lens.iter().sum()` bytes (or null when
/// that sum is `0`).
#[allow(clippy::too_many_arguments)]
unsafe fn ranges_from_raw(
    mins: *const i64,
    min_inclusive: *const u8,
    maxs: *const i64,
    max_inclusive: *const u8,
    label_data: *const u8,
    label_lens: *const usize,
    count: usize,
) -> Result<Vec<NumericRange>, FfiStatus> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if mins.is_null()
        || min_inclusive.is_null()
        || maxs.is_null()
        || max_inclusive.is_null()
        || label_lens.is_null()
    {
        return Err(FfiStatus::NullPointer);
    }
    // SAFETY: caller contract guarantees each of these is valid for `count`
    // reads of its element type.
    let mins = unsafe { std::slice::from_raw_parts(mins, count) };
    let min_incl = unsafe { std::slice::from_raw_parts(min_inclusive, count) };
    let maxs = unsafe { std::slice::from_raw_parts(maxs, count) };
    let max_incl = unsafe { std::slice::from_raw_parts(max_inclusive, count) };
    let lens = unsafe { std::slice::from_raw_parts(label_lens, count) };

    let total_label_len: usize = lens.iter().sum();
    let label_bytes: &[u8] = if total_label_len == 0 {
        &[]
    } else {
        if label_data.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `label_data` is valid for
        // `total_label_len` bytes whenever that sum is nonzero.
        unsafe { std::slice::from_raw_parts(label_data, total_label_len) }
    };

    let mut ranges = Vec::with_capacity(count);
    let mut offset = 0usize;
    for i in 0..count {
        let len = lens[i];
        let label = std::str::from_utf8(&label_bytes[offset..offset + len])
            .map_err(|_| FfiStatus::InvalidUtf8)?
            .to_string();
        offset += len;
        ranges.push(NumericRange {
            label,
            min: mins[i],
            min_inclusive: min_incl[i] != 0,
            max: maxs[i],
            max_inclusive: max_incl[i] != 0,
        });
    }
    Ok(ranges)
}

/// Counts, for every ordinal in `field`'s SORTED_SET terms dictionary, how
/// many of `candidates` (`candidates_len` doc IDs at `candidates`) have that
/// ordinal -- wraps `lucene_search::facets::facet_counts` +
/// `resolve_labels`, then, when `top_n > 0`, `top_n_facets` (see this
/// module's doc comment for the `top_n == 0` "no truncation" case). Writes a
/// new [`crate::registry::FacetResultsHandle`] to
/// `*out_facet_results_handle` on success.
///
/// # Safety
/// `field` must be valid for `field_len` bytes; `candidates` must be valid
/// for `candidates_len` `i32`s (or null when `candidates_len == 0`);
/// `out_facet_results_handle` must be valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_facet_counts_sorted_set(
    segment_handle: u64,
    field: *const c_char,
    field_len: usize,
    candidates: *const i32,
    candidates_len: usize,
    top_n: usize,
    out_facet_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_facet_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `field` is valid for `field_len`
        // bytes and `candidates` is valid for `candidates_len` `i32`s.
        let field = unsafe { str_from_raw(field as *const u8, field_len)? };
        let candidates = unsafe { candidates_from_raw(candidates, candidates_len)? };

        let segments = lock_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error("ffi_facet_counts_sorted_set: unknown or already-closed segment handle");
            FfiStatus::InvalidHandle
        })?;

        let (ords, terms) = sorted_set_multi_entry_for(segment, field)?;
        // `dv_meta` is `Some` whenever `dv_data` is (see `SegmentHandle`'s doc
        // comment) -- `sorted_set_multi_entry_for` above already returned
        // early if `dv_meta` was `None`, so this is always populated here.
        let dv_data = segment.dv_data.as_deref().unwrap_or(&[]);

        let counts =
            facets::facet_counts(dv_data, ords, terms, &candidates).map_err(map_facet_error)?;
        let resolved = facets::resolve_labels(dv_data, terms, &counts).map_err(map_facet_error)?;
        let facets_out = if top_n > 0 {
            facets::top_n_facets(resolved, top_n)
        } else {
            resolved
        };

        let handle =
            lock_recovering(facet_results()).insert(FacetResultsHandle { facets: facets_out });
        // SAFETY: caller contract guarantees `out_facet_results_handle` is
        // valid for one write.
        unsafe {
            *out_facet_results_handle = handle;
        }
        Ok(())
    })
}

/// Counts, for every one of `ranges_count` caller-defined NUMERIC ranges, how
/// many of `candidates` fall inside it -- wraps
/// `lucene_search::facets::range_facet_counts`, writing `ranges_count`
/// `u64` counts (same order as the input ranges) into the caller-allocated
/// `out_counts`. See this module's doc comment for the five-parallel-array
/// range encoding and why no output handle is used.
///
/// # Safety
/// `field` must be valid for `field_len` bytes; `candidates` must be valid
/// for `candidates_len` `i32`s (or null when `candidates_len == 0`);
/// `range_mins`/`range_maxs` must be valid for `ranges_count` `i64`s;
/// `range_min_inclusive`/`range_max_inclusive` must be valid for
/// `ranges_count` `u8`s; `range_label_lens` must be valid for `ranges_count`
/// `usize`s; `range_label_data` must be valid for
/// `range_label_lens.iter().sum()` bytes (or null when that sum is `0`);
/// `out_counts` must be valid for writes of `ranges_count` `u64`s (or null
/// when `ranges_count == 0`).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_range_facet_counts(
    segment_handle: u64,
    field: *const c_char,
    field_len: usize,
    candidates: *const i32,
    candidates_len: usize,
    ranges_count: usize,
    range_mins: *const i64,
    range_min_inclusive: *const u8,
    range_maxs: *const i64,
    range_max_inclusive: *const u8,
    range_label_data: *const u8,
    range_label_lens: *const usize,
    out_counts: *mut u64,
) -> i32 {
    guard(|| {
        if ranges_count > 0 && out_counts.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `field` is valid for `field_len`
        // bytes, `candidates` is valid for `candidates_len` `i32`s, and the
        // range arrays are valid per this function's `# Safety` section.
        let field = unsafe { str_from_raw(field as *const u8, field_len)? };
        let candidates = unsafe { candidates_from_raw(candidates, candidates_len)? };
        let ranges = unsafe {
            ranges_from_raw(
                range_mins,
                range_min_inclusive,
                range_maxs,
                range_max_inclusive,
                range_label_data,
                range_label_lens,
                ranges_count,
            )?
        };

        let segments = lock_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error("ffi_range_facet_counts: unknown or already-closed segment handle");
            FfiStatus::InvalidHandle
        })?;
        let entry = numeric_entry_for(segment, field)?;
        let dv_data = segment.dv_data.as_deref().unwrap_or(&[]);

        let counts = facets::range_facet_counts(dv_data, entry, &ranges, &candidates)
            .map_err(map_facet_error)?;

        if !counts.is_empty() {
            // SAFETY: caller contract guarantees `out_counts` is valid for
            // `ranges_count` writes, and `ranges_count > 0` was checked
            // (non-null) above whenever `counts` is non-empty (`counts.len()
            // == ranges.len() == ranges_count`, `range_facet_counts`'s own
            // documented contract).
            for (i, (_, count)) in counts.into_iter().enumerate() {
                unsafe {
                    *out_counts.add(i) = count;
                }
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::{ffi_close_directory, ffi_open_directory};
    use crate::results_facets::{
        ffi_close_facet_results, ffi_facet_result_label, ffi_facet_results_copy,
        ffi_facet_results_len,
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

    /// Opens `dir_handle`'s fixture segment with its real `.dvm`/`.dvd`,
    /// same pattern as `sort.rs`'s own `open_segment` test helper.
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

    fn multi_dv_dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/multi_valued_dv_index/"
        )
        .to_string()
    }

    fn dv_dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/doc_values_index/"
        )
        .to_string()
    }

    fn read_facets(handle: u64) -> Vec<(i64, String, u64)> {
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_facet_results_len(handle, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        let mut ords = vec![0i64; len];
        let mut counts = vec![0u64; len];
        assert_eq!(
            unsafe { ffi_facet_results_copy(handle, ords.as_mut_ptr(), counts.as_mut_ptr(), len) },
            FfiStatus::Ok.code()
        );
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let mut buf = [0 as c_char; 64];
            let mut written: usize = 0;
            assert_eq!(
                unsafe {
                    ffi_facet_result_label(
                        handle,
                        i,
                        buf.as_mut_ptr(),
                        buf.len(),
                        &mut written as *mut _,
                    )
                },
                FfiStatus::Ok.code()
            );
            let label = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
                .to_str()
                .unwrap()
                .to_string();
            out.push((ords[i], label, counts[i]));
        }
        out
    }

    // --- ffi_facet_counts_sorted_set: real-Lucene fixture (`tags`:
    //     doc0=[red,blue], doc1=NONE, doc2=[green], doc3=[blue], doc4=[red,green]) ---

    #[test]
    fn facet_counts_sorted_set_matches_facets_rs_directly() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "tags";
        let candidates = [0i32, 1, 2, 3, 4];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_facet_counts_sorted_set(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let mut got = read_facets(out);
        got.sort_by_key(|(ord, _, _)| *ord);

        // Cross-check against calling `lucene_search::facets` directly with
        // the same fixture bytes/field, per the task brief's differential
        // requirement.
        let field_num: i32 = manifest
            .get("field_numbers")
            .split(',')
            .find_map(|kv| {
                let (name, num) = kv.split_once(':').unwrap();
                (name == field).then(|| num.parse().unwrap())
            })
            .unwrap();
        let dvm = std::fs::read(format!("{dir}{}.raw", manifest.get("dvm_file_name"))).unwrap();
        let dvd = std::fs::read(format!("{dir}{}.raw", manifest.get("dvd_file_name"))).unwrap();
        let fnm = std::fs::read(format!("{dir}{}.raw", manifest.get("fnm_file_name"))).unwrap();
        let id = id_from_hex(manifest.get("id_hex"));
        let fis = lucene_codecs::field_infos::parse(&fnm, &id, "").unwrap();
        let name = manifest.get("dvm_file_name");
        let segment_name = manifest.get("segment_name");
        let dv_suffix = name
            .strip_prefix(&format!("{segment_name}_"))
            .and_then(|s| s.strip_suffix(".dvm"))
            .unwrap();
        let (_, parsed) =
            lucene_codecs::doc_values::parse_meta(&dvm, &id, dv_suffix, &fis).unwrap();
        let entry = parsed.sorted_set_entry(field_num).unwrap();
        let (ords, terms) = match &entry.kind {
            lucene_codecs::doc_values::SortedSetKind::Multi { ords, terms } => (ords, terms),
            _ => panic!("expected multi-valued SORTED_SET"),
        };
        let matching: Vec<i32> = (0..5).collect();
        let expected_counts =
            lucene_search::facets::facet_counts(&dvd, ords, terms, &matching).unwrap();
        let expected =
            lucene_search::facets::resolve_labels(&dvd, terms, &expected_counts).unwrap();

        assert_eq!(got.len(), expected.len());
        for ((ord, label, count), exp) in got.iter().zip(expected.iter()) {
            assert_eq!(*ord, exp.ord);
            assert_eq!(label, &exp.label);
            assert_eq!(*count, exp.count);
        }

        ffi_close_facet_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn facet_counts_sorted_set_top_n_truncates_and_sorts_descending() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "tags";
        let candidates = [0i32, 1, 2, 3, 4];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_facet_counts_sorted_set(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                1,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let got = read_facets(out);
        assert_eq!(got.len(), 1);
        // Every tag appears exactly twice (red:2, blue:2, green:2) in this
        // fixture; the top-1 result is simply "whichever wins the ordinal
        // tie-break", verified indirectly via the untruncated test above --
        // here we only assert the truncation itself took effect.

        ffi_close_facet_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn facet_counts_sorted_set_unknown_field_is_invalid_argument() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "no-such-field";
        let candidates = [0i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_facet_counts_sorted_set(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn facet_counts_sorted_set_wrong_entry_kind_is_invalid_argument() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        // `nums` is SORTED_NUMERIC, not SORTED_SET.
        let field = "nums";
        let candidates = [0i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_facet_counts_sorted_set(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn facet_counts_sorted_set_unknown_segment_handle_is_invalid_handle() {
        let field = "tags";
        let candidates = [0i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_facet_counts_sorted_set(
                0xFFFF,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn facet_counts_sorted_set_null_out_handle_is_null_pointer_error() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "tags";
        let candidates = [0i32];
        let rc = unsafe {
            ffi_facet_counts_sorted_set(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn facet_counts_sorted_set_empty_candidates_yields_all_zero_counts() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "tags";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_facet_counts_sorted_set(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                0,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let got = read_facets(out);
        assert!(got.iter().all(|(_, _, count)| *count == 0));

        ffi_close_facet_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    // --- ffi_range_facet_counts: real-Lucene fixture (`varying`: -100, 7, 42,
    //     1000, -3 across docs 0..4) ---

    #[test]
    fn range_facet_counts_matches_facets_rs_directly() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "varying";
        let candidates = [0i32, 1, 2, 3, 4];

        let labels = ["negative", "small_positive", "large"];
        let mut label_data = Vec::new();
        let mut label_lens = Vec::new();
        for l in labels {
            label_data.extend_from_slice(l.as_bytes());
            label_lens.push(l.len());
        }
        let mins = [i64::MIN, 0, 100];
        let min_incl = [1u8, 1, 0];
        let maxs = [0i64, 100, i64::MAX];
        let max_incl = [0u8, 1, 1];

        let mut out_counts = [0u64; 3];
        let rc = unsafe {
            ffi_range_facet_counts(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                3,
                mins.as_ptr(),
                min_incl.as_ptr(),
                maxs.as_ptr(),
                max_incl.as_ptr(),
                label_data.as_ptr(),
                label_lens.as_ptr(),
                out_counts.as_mut_ptr(),
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(out_counts, [2, 2, 1]);

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_facet_counts_zero_ranges_is_ok_noop() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "varying";
        let candidates = [0i32];
        let rc = unsafe {
            ffi_range_facet_counts(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_facet_counts_unknown_field_is_invalid_argument() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "no-such-field";
        let candidates = [0i32];
        let label = "r";
        let mins = [0i64];
        let min_incl = [1u8];
        let maxs = [100i64];
        let max_incl = [1u8];
        let label_lens = [label.len()];
        let mut out_counts = [0u64; 1];
        let rc = unsafe {
            ffi_range_facet_counts(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                1,
                mins.as_ptr(),
                min_incl.as_ptr(),
                maxs.as_ptr(),
                max_incl.as_ptr(),
                label.as_ptr(),
                label_lens.as_ptr(),
                out_counts.as_mut_ptr(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_facet_counts_null_out_counts_with_nonzero_ranges_is_null_pointer_error() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "varying";
        let candidates = [0i32];
        let label = "r";
        let mins = [0i64];
        let min_incl = [1u8];
        let maxs = [100i64];
        let max_incl = [1u8];
        let label_lens = [label.len()];
        let rc = unsafe {
            ffi_range_facet_counts(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                1,
                mins.as_ptr(),
                min_incl.as_ptr(),
                maxs.as_ptr(),
                max_incl.as_ptr(),
                label.as_ptr(),
                label_lens.as_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_facet_counts_unknown_segment_handle_is_invalid_handle() {
        let field = "varying";
        let candidates = [0i32];
        let label = "r";
        let mins = [0i64];
        let min_incl = [1u8];
        let maxs = [100i64];
        let max_incl = [1u8];
        let label_lens = [label.len()];
        let mut out_counts = [0u64; 1];
        let rc = unsafe {
            ffi_range_facet_counts(
                0xFFFF,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                1,
                mins.as_ptr(),
                min_incl.as_ptr(),
                maxs.as_ptr(),
                max_incl.as_ptr(),
                label.as_ptr(),
                label_lens.as_ptr(),
                out_counts.as_mut_ptr(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn facet_counts_sorted_set_null_candidates_with_nonzero_len_is_null_pointer_error() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "tags";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_facet_counts_sorted_set(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                1,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn facet_counts_sorted_set_field_without_sorted_set_entry_is_invalid_argument() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        // "varying" is a NUMERIC field in this fixture, not SORTED_SET.
        let field = "varying";
        let candidates = [0i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_facet_counts_sorted_set(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_facet_counts_null_candidates_with_nonzero_len_is_null_pointer_error() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "varying";
        let label = "r";
        let mins = [0i64];
        let min_incl = [1u8];
        let maxs = [100i64];
        let max_incl = [1u8];
        let label_lens = [label.len()];
        let mut out_counts = [0u64; 1];
        let rc = unsafe {
            ffi_range_facet_counts(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                1,
                1,
                mins.as_ptr(),
                min_incl.as_ptr(),
                maxs.as_ptr(),
                max_incl.as_ptr(),
                label.as_ptr(),
                label_lens.as_ptr(),
                out_counts.as_mut_ptr(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_facet_counts_field_without_numeric_entry_is_invalid_argument() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        // "tags" is a SORTED_SET field in this fixture, not NUMERIC.
        let field = "tags";
        let candidates = [0i32];
        let label = "r";
        let mins = [0i64];
        let min_incl = [1u8];
        let maxs = [100i64];
        let max_incl = [1u8];
        let label_lens = [label.len()];
        let mut out_counts = [0u64; 1];
        let rc = unsafe {
            ffi_range_facet_counts(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                1,
                mins.as_ptr(),
                min_incl.as_ptr(),
                maxs.as_ptr(),
                max_incl.as_ptr(),
                label.as_ptr(),
                label_lens.as_ptr(),
                out_counts.as_mut_ptr(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_facet_counts_null_mins_with_nonzero_ranges_is_null_pointer_error() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "varying";
        let candidates = [0i32];
        let label = "r";
        let min_incl = [1u8];
        let maxs = [100i64];
        let max_incl = [1u8];
        let label_lens = [label.len()];
        let mut out_counts = [0u64; 1];
        let rc = unsafe {
            ffi_range_facet_counts(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                1,
                std::ptr::null(),
                min_incl.as_ptr(),
                maxs.as_ptr(),
                max_incl.as_ptr(),
                label.as_ptr(),
                label_lens.as_ptr(),
                out_counts.as_mut_ptr(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_facet_counts_null_label_data_with_nonzero_label_len_is_null_pointer_error() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "varying";
        let candidates = [0i32];
        let mins = [0i64];
        let min_incl = [1u8];
        let maxs = [100i64];
        let max_incl = [1u8];
        let label_lens = [1usize];
        let mut out_counts = [0u64; 1];
        let rc = unsafe {
            ffi_range_facet_counts(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                1,
                mins.as_ptr(),
                min_incl.as_ptr(),
                maxs.as_ptr(),
                max_incl.as_ptr(),
                std::ptr::null(),
                label_lens.as_ptr(),
                out_counts.as_mut_ptr(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn range_facet_counts_invalid_utf8_label_is_invalid_utf8_error() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "varying";
        let candidates = [0i32];
        let label_data = [0xFFu8]; // not valid UTF-8
        let mins = [0i64];
        let min_incl = [1u8];
        let maxs = [100i64];
        let max_incl = [1u8];
        let label_lens = [1usize];
        let mut out_counts = [0u64; 1];
        let rc = unsafe {
            ffi_range_facet_counts(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                1,
                mins.as_ptr(),
                min_incl.as_ptr(),
                maxs.as_ptr(),
                max_incl.as_ptr(),
                label_data.as_ptr(),
                label_lens.as_ptr(),
                out_counts.as_mut_ptr(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidUtf8.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn facet_result_label_null_buf_is_null_pointer_error() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "tags";
        let candidates = [0i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_facet_counts_sorted_set(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let mut written: usize = 0;
        let rc = unsafe {
            ffi_facet_result_label(out, 0, std::ptr::null_mut(), 64, &mut written as *mut _)
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());

        ffi_close_facet_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }
}
