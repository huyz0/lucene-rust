//! Facet results handles (Faceted search FFI exposure):
//! `ffi_facet_counts_sorted_set` (`facets.rs`) collects resolved
//! `(ord, label, count)` triples from `lucene_search::facets::facet_counts`/
//! `resolve_labels`/`top_n_facets` into a new
//! [`crate::registry::FacetResultsHandle`]; the caller reads it back via
//! [`ffi_facet_results_len`]/[`ffi_facet_results_copy`]/[`ffi_facet_result_label`]
//! before releasing it with [`ffi_close_facet_results`].
//!
//! **Two-part readback, unlike `results_sorted.rs`'s single `_copy` call**:
//! `ffi_facet_results_copy` only returns the fixed-size `(ord, count)` half
//! into parallel `i64`/`u64` buffers (the same parallel-buffers shape
//! `results_sorted.rs` already established). Labels are variable-length
//! strings, so they're fetched one at a time via [`ffi_facet_result_label`],
//! which reuses this crate's existing `buf`/`buf_len`/`out_written`/
//! `BufferTooSmall` contract from [`crate::ffi_get_last_error_message`]
//! rather than inventing a new offset-array wire encoding for an output a
//! caller can simply loop over `0..len` to fetch (this crate's first
//! string-array *output*; the `ffi_range_facet_counts` *input* side of
//! `facets.rs` uses a concatenated-buffer encoding instead, since there the
//! caller already knows every label's length up front and building one
//! contiguous buffer is cheap on the C side -- an output side has no such
//! natural place to compute a total buffer size before its first call, since
//! label lengths aren't known until the counts are).

use std::os::raw::c_char;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::registry::{facet_results, lock_recovering};

/// Writes the number of `(ord, label, count)` triples held by
/// `facet_results_handle` to `*out_len`.
///
/// # Safety
/// `out_len` must be valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_facet_results_len(
    facet_results_handle: u64,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = lock_recovering(facet_results());
        let handle = registry.get(facet_results_handle).ok_or_else(|| {
            set_last_error("ffi_facet_results_len: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        // SAFETY: caller contract guarantees `out_len` is valid for one write.
        unsafe {
            *out_len = handle.facets.len();
        }
        Ok(())
    })
}

/// Bulk-copies up to `buf_len` `(ord, count)` pairs from `facet_results_handle`
/// into the caller-allocated `out_ords`/`out_counts` (parallel buffers, same
/// index order [`ffi_facet_result_label`] uses to look up each entry's
/// label). Returns [`FfiStatus::BufferTooSmall`] (with nothing written) if
/// `buf_len` is smaller than the results' actual length -- call
/// [`ffi_facet_results_len`] first to size the buffers.
///
/// # Safety
/// `out_ords` must be valid for writes of `buf_len` `i64`s, `out_counts` for
/// writes of `buf_len` `u64`s.
#[no_mangle]
pub unsafe extern "C" fn ffi_facet_results_copy(
    facet_results_handle: u64,
    out_ords: *mut i64,
    out_counts: *mut u64,
    buf_len: usize,
) -> i32 {
    guard(|| {
        let registry = lock_recovering(facet_results());
        let handle = registry.get(facet_results_handle).ok_or_else(|| {
            set_last_error("ffi_facet_results_copy: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        if handle.facets.len() > buf_len {
            return Err(FfiStatus::BufferTooSmall);
        }
        if !handle.facets.is_empty() {
            if out_ords.is_null() || out_counts.is_null() {
                return Err(FfiStatus::NullPointer);
            }
            // SAFETY: caller contract guarantees `out_ords`/`out_counts` are
            // each valid for `buf_len` writes of their element type, and
            // `handle.facets.len() <= buf_len` was just checked.
            for (i, f) in handle.facets.iter().enumerate() {
                unsafe {
                    *out_ords.add(i) = f.ord;
                    *out_counts.add(i) = f.count;
                }
            }
        }
        Ok(())
    })
}

/// Copies facet-results index `index`'s resolved label into `buf`
/// (caller-allocated, `buf_len` bytes), NUL-terminated, writing the number
/// of bytes written (excluding the NUL) to `*out_written` -- same
/// `buf`/`buf_len`/`out_written`/`BufferTooSmall` contract as
/// [`crate::ffi_get_last_error_message`]. Returns
/// [`FfiStatus::IndexOutOfBounds`] for `index >= ` the handle's length (read
/// via [`ffi_facet_results_len`]).
///
/// # Safety
/// `buf` must be valid for writes of `buf_len` bytes; `out_written` must be
/// valid for one `usize` write, or null.
#[no_mangle]
pub unsafe extern "C" fn ffi_facet_result_label(
    facet_results_handle: u64,
    index: usize,
    buf: *mut c_char,
    buf_len: usize,
    out_written: *mut usize,
) -> i32 {
    guard(|| {
        let registry = lock_recovering(facet_results());
        let handle = registry.get(facet_results_handle).ok_or_else(|| {
            set_last_error("ffi_facet_result_label: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let facet = handle.facets.get(index).ok_or_else(|| {
            set_last_error(format!(
                "ffi_facet_result_label: index {index} out of bounds (len {})",
                handle.facets.len()
            ));
            FfiStatus::IndexOutOfBounds
        })?;
        let bytes = facet.label.as_bytes();
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

/// Closes a facet results handle. Returns [`FfiStatus::InvalidHandle`] for an
/// unknown/already-closed handle.
#[no_mangle]
pub extern "C" fn ffi_close_facet_results(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(facet_results())
            .remove(handle)
            .map(|_| ())
            .ok_or_else(|| {
                set_last_error("ffi_close_facet_results: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::FacetResultsHandle;
    use lucene_search::facets::FacetCount;

    fn insert(facets: Vec<FacetCount>) -> u64 {
        lock_recovering(facet_results()).insert(FacetResultsHandle { facets })
    }

    fn sample() -> Vec<FacetCount> {
        vec![
            FacetCount {
                ord: 0,
                label: "red".into(),
                count: 2,
            },
            FacetCount {
                ord: 1,
                label: "blue".into(),
                count: 3,
            },
        ]
    }

    #[test]
    fn len_and_copy_and_label_roundtrip() {
        let h = insert(sample());
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_facet_results_len(h, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 2);

        let mut ords = [0i64; 2];
        let mut counts = [0u64; 2];
        assert_eq!(
            unsafe { ffi_facet_results_copy(h, ords.as_mut_ptr(), counts.as_mut_ptr(), 2) },
            FfiStatus::Ok.code()
        );
        assert_eq!(ords, [0, 1]);
        assert_eq!(counts, [2, 3]);

        for (i, expected) in ["red", "blue"].into_iter().enumerate() {
            let mut buf = [0 as c_char; 16];
            let mut written: usize = 0;
            assert_eq!(
                unsafe {
                    ffi_facet_result_label(
                        h,
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
                .unwrap();
            assert_eq!(label, expected);
            assert_eq!(written, expected.len());
        }

        assert_eq!(ffi_close_facet_results(h), FfiStatus::Ok.code());
    }

    #[test]
    fn copy_with_buffer_too_small_leaves_error_and_writes_nothing_observable() {
        let h = insert(sample());
        let mut ords = [9i64; 1];
        let mut counts = [9u64; 1];
        let rc = unsafe { ffi_facet_results_copy(h, ords.as_mut_ptr(), counts.as_mut_ptr(), 1) };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        assert_eq!(ords, [9]);
        assert_eq!(counts, [9]);
        ffi_close_facet_results(h);
    }

    #[test]
    fn copy_empty_results_with_null_bufs_and_zero_len_is_ok() {
        let h = insert(vec![]);
        let rc =
            unsafe { ffi_facet_results_copy(h, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        assert_eq!(rc, FfiStatus::Ok.code());
        ffi_close_facet_results(h);
    }

    #[test]
    fn copy_nonempty_with_null_ords_is_null_pointer_error() {
        let h = insert(sample());
        let mut counts = [0u64; 2];
        let rc = unsafe { ffi_facet_results_copy(h, std::ptr::null_mut(), counts.as_mut_ptr(), 2) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_facet_results(h);
    }

    #[test]
    fn copy_nonempty_with_null_counts_is_null_pointer_error() {
        let h = insert(sample());
        let mut ords = [0i64; 2];
        let rc = unsafe { ffi_facet_results_copy(h, ords.as_mut_ptr(), std::ptr::null_mut(), 2) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_facet_results(h);
    }

    #[test]
    fn label_out_of_bounds_index_is_index_out_of_bounds() {
        let h = insert(sample());
        let mut buf = [0 as c_char; 16];
        let rc = unsafe {
            ffi_facet_result_label(h, 5, buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
        };
        assert_eq!(rc, FfiStatus::IndexOutOfBounds.code());
        ffi_close_facet_results(h);
    }

    #[test]
    fn label_buffer_too_small_is_buffer_too_small() {
        let h = insert(sample());
        let mut buf = [0 as c_char; 2];
        let rc = unsafe {
            ffi_facet_result_label(h, 1, buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
        };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        ffi_close_facet_results(h);
    }

    #[test]
    fn len_unknown_handle_is_invalid_handle() {
        let mut len: usize = 0;
        let rc = unsafe { ffi_facet_results_len(0xABCD, &mut len as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn len_null_out_len_is_null_pointer_error() {
        let h = insert(sample());
        let rc = unsafe { ffi_facet_results_len(h, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_facet_results(h);
    }

    #[test]
    fn copy_unknown_handle_is_invalid_handle() {
        let mut ords = [0i64; 1];
        let mut counts = [0u64; 1];
        let rc =
            unsafe { ffi_facet_results_copy(0xABCD, ords.as_mut_ptr(), counts.as_mut_ptr(), 1) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn label_unknown_handle_is_invalid_handle() {
        let mut buf = [0 as c_char; 16];
        let rc = unsafe {
            ffi_facet_result_label(0xABCD, 0, buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn close_unknown_facet_results_handle_is_invalid_handle() {
        assert_eq!(
            ffi_close_facet_results(0xABCD),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn double_close_facet_results_is_invalid_handle_not_a_crash() {
        let h = insert(sample());
        assert_eq!(ffi_close_facet_results(h), FfiStatus::Ok.code());
        assert_eq!(ffi_close_facet_results(h), FfiStatus::InvalidHandle.code());
    }
}
