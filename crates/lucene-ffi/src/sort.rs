//! `ffi_sort_by_doc_value`/`ffi_sort_by_multi_valued_doc_value` (task #40):
//! runs this port's existing `lucene_search::sort_by_numeric_doc_value`/
//! `lucene_search::doc_value_query::sort_by_multi_valued_doc_value` against
//! an already-opened [`crate::segment::SegmentHandle`]'s doc-values data
//! (opened by [`crate::segment::ffi_open_segment`]'s `dvm_name`/`dvd_name`/
//! `dv_suffix` parameters), collecting the resulting ascending
//! `(doc_id, value)` pairs into a new [`crate::registry::SortedResultsHandle`]
//! -- the same "run the existing pure-Rust function, park its output behind
//! a fresh opaque handle" shape `query.rs`'s scored-query functions (task
//! #30) already established.
//!
//! **Ascending only, single sort key only**: mirrors
//! `lucene_search::doc_value_query`'s own documented scope -- that module's
//! doc comment lists "descending sort / multiple sort fields" as
//! deliberately out of scope (no `Sort`/`SortField` composition exists in
//! this port yet), so there is nothing for this FFI surface to expose beyond
//! what the wrapped functions already do. A `descending` flag here would
//! either silently lie about a capability the Rust layer doesn't have, or
//! require a reverse-then-relabel hack that isn't real Lucene sort
//! semantics (a real descending sort needs its own tie-break rule, not just
//! a reversed ascending list) -- deferred alongside the underlying
//! functions, tracked in `docs/parity.md`.
//!
//! **Missing-value policy over the wire**: [`lucene_search::MissingValue`]
//! is a two-variant enum (`Exclude` / `Default(i64)`); rather than invent a
//! tagged-union wire encoding, both entry points take a plain
//! `missing_is_default: bool` plus a `missing_default: i64` that's simply
//! ignored when the bool is `false` -- the smallest wire shape that can
//! express both variants without a caller ever needing to pack/unpack a
//! discriminant by hand.
//!
//! **Field lookup and entry-kind validation**: same field-name -> field-number
//! lookup `query.rs`'s [`crate::query`]`::open_field_norms` already does via
//! `segment.field_infos`, then a [`lucene_codecs::doc_values::DocValuesMeta::numeric_entry`]/
//! `sorted_numeric_entry` lookup against the segment's opened `.dvm`. A field
//! with no matching entry (wrong doc-values type, or a segment opened
//! without doc values at all) is [`crate::error::FfiStatus::InvalidArgument`]
//! -- unlike norms, there is no meaningful "fall back to a constant" story
//! for "nothing to sort by".

use std::os::raw::c_char;

use lucene_codecs::doc_values::{NumericEntry, SortedNumericEntry};
use lucene_search::doc_value_query::{self, ValueSelector};
use lucene_search::MissingValue;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::str_from_raw;
use crate::registry::{
    lock_recovering, segments, sorted_results, SegmentHandle, SortedResultsHandle,
};

pub(crate) fn map_sort_error(e: lucene_search::Error) -> FfiStatus {
    set_last_error(format!("doc-value sort failed: {e}"));
    FfiStatus::Decode
}

pub(crate) fn missing_value(missing_is_default: bool, missing_default: i64) -> MissingValue {
    if missing_is_default {
        MissingValue::Default(missing_default)
    } else {
        MissingValue::Exclude
    }
}

/// Reads `len` candidate doc IDs from `ptr` into an owned `Vec<i32>`. `ptr`
/// may be null only when `len == 0` (an empty candidate set), same
/// convention as [`crate::raw::bytes_from_raw`].
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
/// [`NumericEntry`] in `segment`'s opened `.dvm`. `Err(InvalidArgument)` when
/// the segment has no doc values open, the field is unknown, or the field
/// has no NUMERIC doc-values entry.
///
/// `pub(crate)` (rather than private) so [`crate::range_sort`]'s
/// range-then-sort-by-field entry points (TopFieldCollector FFI exposure)
/// can reuse the exact same field-name -> `NumericEntry` lookup instead of
/// duplicating it -- both `range_entry`/`sort_entry` there are NUMERIC
/// entries looked up the same way `ffi_sort_by_doc_value`'s single sort key
/// already is.
pub(crate) fn numeric_entry_for<'seg>(
    segment: &'seg SegmentHandle,
    field: &str,
) -> Result<&'seg NumericEntry, FfiStatus> {
    let dv_meta = segment.dv_meta.as_ref().ok_or_else(|| {
        set_last_error("ffi_sort_by_doc_value: segment was opened without doc values");
        FfiStatus::InvalidArgument
    })?;
    let field_info = segment
        .field_infos
        .fields
        .iter()
        .find(|f| f.name == field)
        .ok_or_else(|| {
            set_last_error(format!("ffi_sort_by_doc_value: unknown field {field}"));
            FfiStatus::InvalidArgument
        })?;
    dv_meta.numeric_entry(field_info.number).ok_or_else(|| {
        set_last_error(format!(
            "ffi_sort_by_doc_value: field {field} has no NUMERIC doc-values entry"
        ));
        FfiStatus::InvalidArgument
    })
}

/// Same as [`numeric_entry_for`], but for a SORTED_NUMERIC entry (also used
/// for a SORTED_SET field's ordinal array, see `lucene_search::doc_value_query`'s
/// module doc) -- backs [`ffi_sort_by_multi_valued_doc_value`].
fn sorted_numeric_entry_for<'seg>(
    segment: &'seg SegmentHandle,
    field: &str,
) -> Result<&'seg SortedNumericEntry, FfiStatus> {
    let dv_meta = segment.dv_meta.as_ref().ok_or_else(|| {
        set_last_error("ffi_sort_by_multi_valued_doc_value: segment was opened without doc values");
        FfiStatus::InvalidArgument
    })?;
    let field_info = segment
        .field_infos
        .fields
        .iter()
        .find(|f| f.name == field)
        .ok_or_else(|| {
            set_last_error(format!(
                "ffi_sort_by_multi_valued_doc_value: unknown field {field}"
            ));
            FfiStatus::InvalidArgument
        })?;
    dv_meta
        .sorted_numeric_entry(field_info.number)
        .ok_or_else(|| {
            set_last_error(format!(
                "ffi_sort_by_multi_valued_doc_value: field {field} has no SORTED_NUMERIC doc-values entry"
            ));
            FfiStatus::InvalidArgument
        })
}

/// Sorts `candidates` (`candidates_len` doc IDs at `candidates`) ascending by
/// `field`'s NUMERIC doc-value (`lucene_search::sort_by_numeric_doc_value`),
/// writing a new [`crate::registry::SortedResultsHandle`] to
/// `*out_sorted_results_handle` on success. `missing_is_default`/
/// `missing_default` select [`MissingValue::Exclude`] (`false`) or
/// [`MissingValue::Default`] (`true`, using `missing_default`) for a
/// candidate with no value for `field` -- see this module's doc comment.
///
/// # Safety
/// `field` must be valid for `field_len` bytes; `candidates` must be valid
/// for `candidates_len` `i32`s (or null when `candidates_len == 0`);
/// `out_sorted_results_handle` must be valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_sort_by_doc_value(
    segment_handle: u64,
    field: *const c_char,
    field_len: usize,
    candidates: *const i32,
    candidates_len: usize,
    missing_is_default: bool,
    missing_default: i64,
    out_sorted_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_sorted_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `field` is valid for `field_len`
        // bytes and `candidates` is valid for `candidates_len` `i32`s.
        let field = unsafe { str_from_raw(field as *const u8, field_len)? };
        let candidates = unsafe { candidates_from_raw(candidates, candidates_len)? };

        let segments = lock_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error("ffi_sort_by_doc_value: unknown or already-closed segment handle");
            FfiStatus::InvalidHandle
        })?;

        // Test-only: see `PANIC_ON_NEXT_SORT`'s doc comment. Fires while
        // `segments` (the `MutexGuard` above) is still held, exactly like a
        // real decode panic reached through `sort_by_numeric_doc_value` below
        // would.
        #[cfg(test)]
        if PANIC_ON_NEXT_SORT.with(|c| c.replace(false)) {
            panic!(
                "test-only simulated panic while the segments registry lock is held (sort path)"
            );
        }

        let entry = numeric_entry_for(segment, field)?;
        // `dv_meta` is `Some` whenever `dv_data` is (see `SegmentHandle`'s doc
        // comment) -- `numeric_entry_for` above already returned early if
        // `dv_meta` was `None`, so this is always populated here.
        let dv_data = segment.dv_data.as_deref().unwrap_or(&[]);

        let pairs = doc_value_query::sort_by_numeric_doc_value(
            dv_data,
            entry,
            &candidates,
            missing_value(missing_is_default, missing_default),
        )
        .map_err(map_sort_error)?;

        let handle = lock_recovering(sorted_results()).insert(SortedResultsHandle { pairs });
        // SAFETY: caller contract guarantees `out_sorted_results_handle` is
        // valid for one write.
        unsafe {
            *out_sorted_results_handle = handle;
        }
        Ok(())
    })
}

/// Multi-valued sibling of [`ffi_sort_by_doc_value`]: sorts `candidates`
/// ascending by `field`'s SORTED_NUMERIC doc-value, reduced to one value per
/// doc via `selector` (`0` = [`ValueSelector::Min`], `1` = [`ValueSelector::Max`],
/// any other value is [`FfiStatus::InvalidArgument`]) -- see
/// `lucene_search::doc_value_query::sort_by_multi_valued_doc_value`'s doc
/// comment for the reduction/tie-break rules, and this module's doc comment
/// for why the same function also serves a SORTED_SET field's ordinal array.
///
/// # Safety
/// Same contract as [`ffi_sort_by_doc_value`]'s, plus
/// `out_sorted_results_handle` must be valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_sort_by_multi_valued_doc_value(
    segment_handle: u64,
    field: *const c_char,
    field_len: usize,
    candidates: *const i32,
    candidates_len: usize,
    selector: i32,
    missing_is_default: bool,
    missing_default: i64,
    out_sorted_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_sorted_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let selector = match selector {
            0 => ValueSelector::Min,
            1 => ValueSelector::Max,
            _ => {
                set_last_error(format!(
                    "ffi_sort_by_multi_valued_doc_value: invalid selector {selector} (expected 0=Min or 1=Max)"
                ));
                return Err(FfiStatus::InvalidArgument);
            }
        };
        // SAFETY: caller contract guarantees `field` is valid for `field_len`
        // bytes and `candidates` is valid for `candidates_len` `i32`s.
        let field = unsafe { str_from_raw(field as *const u8, field_len)? };
        let candidates = unsafe { candidates_from_raw(candidates, candidates_len)? };

        let segments = lock_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error(
                "ffi_sort_by_multi_valued_doc_value: unknown or already-closed segment handle",
            );
            FfiStatus::InvalidHandle
        })?;
        let entry = sorted_numeric_entry_for(segment, field)?;
        let dv_data = segment.dv_data.as_deref().unwrap_or(&[]);

        let pairs = doc_value_query::sort_by_multi_valued_doc_value(
            dv_data,
            entry,
            selector,
            &candidates,
            missing_value(missing_is_default, missing_default),
        )
        .map_err(map_sort_error)?;

        let handle = lock_recovering(sorted_results()).insert(SortedResultsHandle { pairs });
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

    /// Opens `dir_handle`'s fixture segment with its real `.dvm`/`.dvd`
    /// (both named `_0_Lucene90_0.dv{m,d}`, real-Lucene fixture convention --
    /// see `doc_value_query.rs`'s own fixture tests) so the sort FFI path can
    /// exercise real doc-values bytes end-to-end, same as `query.rs`'s
    /// norms-aware tests do for scored queries.
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

    fn field_number(manifest: &Manifest, field: &str) -> i32 {
        manifest
            .get("field_numbers")
            .split(',')
            .find_map(|kv| {
                let (name, num) = kv.split_once(':').unwrap();
                (name == field).then(|| num.parse().unwrap())
            })
            .unwrap_or_else(|| panic!("field {field} missing from field_numbers"))
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

    fn multi_dv_dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/multi_valued_dv_index/"
        )
        .to_string()
    }

    // --- ffi_sort_by_doc_value: real-Lucene fixture (`varying`: -100, 7, 42,
    //     1000, -3 across docs 0..4 -- same fixture/values `doc_value_query.rs`'s
    //     own `sort_by_value_ascending_order_real_fixture` test uses) ---

    #[test]
    fn sort_by_doc_value_ascending_order_real_fixture() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "varying";
        let candidates = [0i32, 1, 2, 3, 4];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(
            read_sorted(out),
            vec![(0, -100), (4, -3), (1, 7), (2, 42), (3, 1000)]
        );

        ffi_close_sorted_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn sort_by_doc_value_missing_default_policy_real_fixture() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        // sparse: 5, NONE, 15, NONE, 25 -- docs 1 and 3 have no value.
        let field = "sparse";
        let candidates = [0i32, 1, 2, 3, 4];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                true,
                1_000_000,
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
    fn sort_by_doc_value_exclude_policy_drops_missing_docs() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "sparse";
        let candidates = [0i32, 1, 2, 3, 4];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(read_sorted(out), vec![(0, 5), (2, 15), (4, 25)]);

        ffi_close_sorted_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn sort_by_doc_value_empty_candidates_yields_empty_result() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                0,
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(read_sorted(out).is_empty());

        ffi_close_sorted_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn sort_by_doc_value_unknown_field_is_invalid_argument() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "no-such-field";
        let candidates = [0i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn sort_by_doc_value_wrong_entry_kind_is_invalid_argument() {
        // `field_number` exists (via field_infos) but a BINARY field like
        // `bin_fixed` has no NUMERIC entry.
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);
        let _ = field_number(&manifest, "bin_fixed");

        let field = "bin_fixed";
        let candidates = [0i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn sort_by_doc_value_segment_without_doc_values_is_invalid_argument() {
        // The blocktree_index fixture's segment has no `.dvm`/`.dvd` opened.
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

        let field = "body";
        let candidates = [0i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn sort_by_doc_value_unknown_segment_handle_is_invalid_handle() {
        let field = "varying";
        let candidates = [0i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_doc_value(
                0xFFFF,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn sort_by_doc_value_null_out_handle_is_null_pointer_error() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "varying";
        let candidates = [0i32];
        let rc = unsafe {
            ffi_sort_by_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                false,
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn sort_by_doc_value_null_candidates_with_nonzero_len_is_null_pointer_error() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "varying";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                3,
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// Swaps a live segment handle's `.dvd` bytes for garbage that fails
    /// mid-decode, reaching `map_sort_error`'s `FfiStatus::Decode` branch --
    /// mirrors `query.rs`'s `corrupt_doc_bytes` helper for the same purpose.
    fn corrupt_dv_data(seg_handle: u64) {
        let mut segs = lock_recovering(segments_registry());
        let segment = segs.get_mut(seg_handle).expect("segment handle");
        // Truncated to zero bytes: any real decode of a NUMERIC entry needing
        // to read actual values bytes fails.
        segment.dv_data = Some(vec![]);
    }

    #[test]
    fn sort_by_doc_value_decode_error_propagates_as_decode_status() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);
        corrupt_dv_data(seg_handle);

        // `varying` is a dense NUMERIC field with `bits_per_value != 0` --
        // decoding doc 0's value needs real bytes, which are now empty.
        let field = "varying";
        let candidates = [0i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    // --- ffi_sort_by_multi_valued_doc_value: real-Lucene fixture (`nums`:
    //     [5,10], NONE, [7], [1,2,3], NONE across docs 0..4) ---

    #[test]
    fn multi_sort_min_selector_ascending_order_real_fixture() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "nums";
        let candidates = [0i32, 2, 3];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_multi_valued_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                0, // Min
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(read_sorted(out), vec![(3, 1), (0, 5), (2, 7)]);

        ffi_close_sorted_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn multi_sort_max_selector_ascending_order_real_fixture() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "nums";
        let candidates = [0i32, 2, 3];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_multi_valued_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                1, // Max
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(read_sorted(out), vec![(3, 3), (2, 7), (0, 10)]);

        ffi_close_sorted_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn multi_sort_missing_default_policy_real_fixture() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "nums";
        let candidates = [0i32, 1, 2, 3, 4];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_multi_valued_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                0, // Min
                true,
                1_000_000,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(
            read_sorted(out),
            vec![(3, 1), (0, 5), (2, 7), (1, 1_000_000), (4, 1_000_000)]
        );

        ffi_close_sorted_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn multi_sort_invalid_selector_is_invalid_argument() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "nums";
        let candidates = [0i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_multi_valued_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                2, // invalid
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn multi_sort_unknown_field_is_invalid_argument() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "no-such-field";
        let candidates = [0i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_multi_valued_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                0,
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn multi_sort_unknown_segment_handle_is_invalid_handle() {
        let field = "nums";
        let candidates = [0i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_multi_valued_doc_value(
                0xFFFF,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                0,
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn multi_sort_null_out_handle_is_null_pointer_error() {
        let dir = multi_dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "nums";
        let candidates = [0i32];
        let rc = unsafe {
            ffi_sort_by_multi_valued_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                0,
                false,
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// Regression test for the same poison-recovery contract task #30's
    /// `registry_mutex_recovers_from_poisoning_after_a_panic_mid_scored_query`
    /// covers, applied to this new sort path -- a **thread-local** switch (see
    /// this file's top-of-module note below and the `ffi-safety`/session
    /// history note: a process-wide `AtomicBool` here would be exposed to the
    /// exact cross-test flakiness a prior task already hit, since `cargo
    /// test` runs this crate's tests in parallel by default).
    #[test]
    fn registry_mutex_recovers_from_poisoning_after_a_panic_mid_sort() {
        let dir = dv_dir();
        let manifest = Manifest::load(&dir);
        let dir_handle = open_dir(&dir);
        let seg_handle = open_segment(dir_handle, &manifest);

        let field = "varying";
        let candidates = [0i32, 1, 2, 3, 4];

        arm_panic_on_next_sort();
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_sort_by_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                false,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Panic.code());

        // The `segments()` mutex must have recovered from the panic-induced
        // poison -- a second, ordinary call right after must still succeed,
        // proving `lock_recovering` (not a wedged-forever poisoned mutex) is
        // what every subsequent call goes through.
        let mut out2: u64 = 0;
        let rc2 = unsafe {
            ffi_sort_by_doc_value(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                candidates.as_ptr(),
                candidates.len(),
                false,
                0,
                &mut out2 as *mut _,
            )
        };
        assert_eq!(rc2, FfiStatus::Ok.code());
        assert_eq!(
            read_sorted(out2),
            vec![(0, -100), (4, -3), (1, 7), (2, 42), (3, 1000)]
        );

        ffi_close_sorted_results(out2);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }
}

// Test-only panic-injection switch for
// `registry_mutex_recovers_from_poisoning_after_a_panic_mid_sort` above --
// same purpose and shape as `query.rs`'s `PANIC_ON_NEXT_SCORED_TERM_QUERY`
// (see that file's doc comment for the full rationale): fires while
// `ffi_sort_by_doc_value` still holds the `segments()` registry's
// `MutexGuard`, the exact condition that poisons the mutex, and is
// **thread-local rather than a process-wide `static`** so no other,
// concurrently-running test on a different thread can ever observe or
// trigger it -- avoiding the exact cross-test-flakiness pattern a
// process-wide `AtomicBool` bears (see task #29's history and `query.rs`'s
// own `PANIC_ON_NEXT_SCORED_TERM_QUERY` doc comment for the full story).
#[cfg(test)]
thread_local! {
    static PANIC_ON_NEXT_SORT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn arm_panic_on_next_sort() {
    PANIC_ON_NEXT_SORT.with(|c| c.set(true));
}
