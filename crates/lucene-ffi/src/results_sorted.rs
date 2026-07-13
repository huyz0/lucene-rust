//! Sorted results handles (task #40): `ffi_sort_by_doc_value`/
//! `ffi_sort_by_multi_valued_doc_value` (`sort.rs`) collect the ascending
//! `(doc_id, value)` pairs from `lucene_search::sort_by_numeric_doc_value`/
//! `sort_by_multi_valued_doc_value` into a new
//! [`crate::registry::SortedResultsHandle`]; the caller reads it back via
//! [`ffi_sorted_results_len`]/[`ffi_sorted_results_copy`] before releasing it
//! with [`ffi_close_sorted_results`] -- the exact same shape
//! `results_scored.rs` already established for `(doc_id, score)` pairs, see
//! that module's doc comment for the parallel-buffers rationale (this module
//! reuses it verbatim: `out_doc_ids: *mut i32` / `out_values: *mut i64`
//! rather than one interleaved buffer).

use crate::error::{guard, set_last_error, FfiStatus};
use crate::registry::{lock_recovering, sorted_results};

/// Writes the number of `(doc_id, value)` pairs held by
/// `sorted_results_handle` to `*out_len`.
///
/// # Safety
/// `out_len` must be valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_sorted_results_len(
    sorted_results_handle: u64,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = lock_recovering(sorted_results());
        let handle = registry.get(sorted_results_handle).ok_or_else(|| {
            set_last_error("ffi_sorted_results_len: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        // SAFETY: caller contract guarantees `out_len` is valid for one write.
        unsafe {
            *out_len = handle.pairs.len();
        }
        Ok(())
    })
}

/// Bulk-copies up to `buf_len` `(doc_id, value)` pairs from
/// `sorted_results_handle` into the caller-allocated `out_doc_ids`/
/// `out_values` (parallel buffers -- see this module's doc comment). Returns
/// [`FfiStatus::BufferTooSmall`] (with nothing written) if `buf_len` is
/// smaller than the results' actual length -- call [`ffi_sorted_results_len`]
/// first to size the buffers.
///
/// # Safety
/// `out_doc_ids` must be valid for writes of `buf_len` `i32`s, `out_values`
/// for writes of `buf_len` `i64`s.
#[no_mangle]
pub unsafe extern "C" fn ffi_sorted_results_copy(
    sorted_results_handle: u64,
    out_doc_ids: *mut i32,
    out_values: *mut i64,
    buf_len: usize,
) -> i32 {
    guard(|| {
        let registry = lock_recovering(sorted_results());
        let handle = registry.get(sorted_results_handle).ok_or_else(|| {
            set_last_error("ffi_sorted_results_copy: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        if handle.pairs.len() > buf_len {
            return Err(FfiStatus::BufferTooSmall);
        }
        if !handle.pairs.is_empty() {
            if out_doc_ids.is_null() || out_values.is_null() {
                return Err(FfiStatus::NullPointer);
            }
            // SAFETY: caller contract guarantees `out_doc_ids`/`out_values` are
            // each valid for `buf_len` writes of their element type, and
            // `handle.pairs.len() <= buf_len` was just checked. Written one
            // element at a time (rather than `copy_nonoverlapping` from a
            // temporary contiguous buffer) since a `(i32, i64)` tuple isn't
            // laid out as two parallel arrays in memory -- same reasoning as
            // `results_scored.rs`'s `ffi_scored_results_copy`.
            for (i, &(doc_id, value)) in handle.pairs.iter().enumerate() {
                unsafe {
                    *out_doc_ids.add(i) = doc_id;
                    *out_values.add(i) = value;
                }
            }
        }
        Ok(())
    })
}

/// Closes a sorted results handle. Returns [`FfiStatus::InvalidHandle`] for
/// an unknown/already-closed handle.
#[no_mangle]
pub extern "C" fn ffi_close_sorted_results(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(sorted_results())
            .remove(handle)
            .map(|_| ())
            .ok_or_else(|| {
                set_last_error("ffi_close_sorted_results: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::SortedResultsHandle;

    fn insert(pairs: Vec<(i32, i64)>) -> u64 {
        lock_recovering(sorted_results()).insert(SortedResultsHandle { pairs })
    }

    #[test]
    fn len_and_copy_roundtrip() {
        let h = insert(vec![(1, 5), (2, 3), (3, 1)]);
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_sorted_results_len(h, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 3);

        let mut doc_ids = [0i32; 3];
        let mut values = [0i64; 3];
        assert_eq!(
            unsafe {
                ffi_sorted_results_copy(h, doc_ids.as_mut_ptr(), values.as_mut_ptr(), doc_ids.len())
            },
            FfiStatus::Ok.code()
        );
        assert_eq!(doc_ids, [1, 2, 3]);
        assert_eq!(values, [5, 3, 1]);
        assert_eq!(ffi_close_sorted_results(h), FfiStatus::Ok.code());
    }

    #[test]
    fn copy_with_buffer_too_small_leaves_error_and_writes_nothing_observable() {
        let h = insert(vec![(1, 5), (2, 3), (3, 1)]);
        let mut doc_ids = [9i32; 2];
        let mut values = [9i64; 2];
        let rc = unsafe {
            ffi_sorted_results_copy(h, doc_ids.as_mut_ptr(), values.as_mut_ptr(), doc_ids.len())
        };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        assert_eq!(doc_ids, [9, 9]);
        assert_eq!(values, [9, 9]);
        ffi_close_sorted_results(h);
    }

    #[test]
    fn copy_empty_results_with_null_bufs_and_zero_len_is_ok() {
        let h = insert(vec![]);
        let rc =
            unsafe { ffi_sorted_results_copy(h, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        assert_eq!(rc, FfiStatus::Ok.code());
        ffi_close_sorted_results(h);
    }

    #[test]
    fn copy_nonempty_with_null_doc_ids_is_null_pointer_error() {
        let h = insert(vec![(1, 5)]);
        let mut values = [0i64; 1];
        let rc =
            unsafe { ffi_sorted_results_copy(h, std::ptr::null_mut(), values.as_mut_ptr(), 1) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_sorted_results(h);
    }

    #[test]
    fn copy_nonempty_with_null_values_is_null_pointer_error() {
        let h = insert(vec![(1, 5)]);
        let mut doc_ids = [0i32; 1];
        let rc =
            unsafe { ffi_sorted_results_copy(h, doc_ids.as_mut_ptr(), std::ptr::null_mut(), 1) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_sorted_results(h);
    }

    #[test]
    fn len_unknown_handle_is_invalid_handle() {
        let mut len: usize = 0;
        let rc = unsafe { ffi_sorted_results_len(0xABCD, &mut len as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn len_null_out_len_is_null_pointer_error() {
        let h = insert(vec![(1, 1)]);
        let rc = unsafe { ffi_sorted_results_len(h, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_sorted_results(h);
    }

    #[test]
    fn copy_unknown_handle_is_invalid_handle() {
        let mut doc_ids = [0i32; 1];
        let mut values = [0i64; 1];
        let rc = unsafe {
            ffi_sorted_results_copy(0xABCD, doc_ids.as_mut_ptr(), values.as_mut_ptr(), 1)
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn close_unknown_sorted_results_handle_is_invalid_handle() {
        assert_eq!(
            ffi_close_sorted_results(0xABCD),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn double_close_sorted_results_is_invalid_handle_not_a_crash() {
        let h = insert(vec![(1, 1)]);
        assert_eq!(ffi_close_sorted_results(h), FfiStatus::Ok.code());
        assert_eq!(ffi_close_sorted_results(h), FfiStatus::InvalidHandle.code());
    }
}
