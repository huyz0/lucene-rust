//! Results handles: every query call below collects matching doc IDs into a
//! plain Rust `Vec<i32>` (via `lucene_search::VecCollector`, entirely
//! Rust-side — see the `ffi-safety` skill's "no callbacks from Rust into
//! Java" rule), stores it under a new handle, and the caller reads it back
//! via [`ffi_results_len`]/[`ffi_results_copy`] before releasing it with
//! [`ffi_close_results`].
//!
//! **Why a bulk `ffi_results_copy` instead of a per-index `ffi_results_get`**:
//! a JNI caller almost always wants the whole doc-ID array at once (to
//! build a Java `int[]`/`IntBuffer`) rather than one JNI round-trip per
//! doc; `ffi_results_copy` matches that access pattern with one call. Both
//! are cheap to add, but `ffi_results_len` + `ffi_results_copy` is the
//! documented, supported bulk path.

use crate::error::{guard, set_last_error, FfiStatus};
use crate::registry::{lock_recovering, results};

/// Writes the number of doc IDs held by `results_handle` to `*out_len`.
///
/// # Safety
/// `out_len` must be valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_results_len(results_handle: u64, out_len: *mut usize) -> i32 {
    guard(|| {
        if out_len.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = lock_recovering(results());
        let handle = registry.get(results_handle).ok_or_else(|| {
            set_last_error("ffi_results_len: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        // SAFETY: caller contract guarantees `out_len` is valid for one write.
        unsafe {
            *out_len = handle.docs.len();
        }
        Ok(())
    })
}

/// Bulk-copies up to `buf_len` doc IDs from `results_handle` into the
/// caller-allocated `out_buf`. Returns [`FfiStatus::BufferTooSmall`] (with
/// nothing written) if `buf_len` is smaller than the results' actual
/// length -- call [`ffi_results_len`] first to size the buffer.
///
/// # Safety
/// `out_buf` must be valid for writes of `buf_len` `i32`s.
#[no_mangle]
pub unsafe extern "C" fn ffi_results_copy(
    results_handle: u64,
    out_buf: *mut i32,
    buf_len: usize,
) -> i32 {
    guard(|| {
        let registry = lock_recovering(results());
        let handle = registry.get(results_handle).ok_or_else(|| {
            set_last_error("ffi_results_copy: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        if handle.docs.len() > buf_len {
            return Err(FfiStatus::BufferTooSmall);
        }
        if !handle.docs.is_empty() {
            if out_buf.is_null() {
                return Err(FfiStatus::NullPointer);
            }
            // SAFETY: caller contract guarantees `out_buf` is valid for `buf_len`
            // i32 writes, and `handle.docs.len() <= buf_len` was just checked.
            unsafe {
                std::ptr::copy_nonoverlapping(handle.docs.as_ptr(), out_buf, handle.docs.len());
            }
        }
        Ok(())
    })
}

/// Closes a results handle. Returns [`FfiStatus::InvalidHandle`] for an
/// unknown/already-closed handle.
#[no_mangle]
pub extern "C" fn ffi_close_results(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(results())
            .remove(handle)
            .map(|_| ())
            .ok_or_else(|| {
                set_last_error("ffi_close_results: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ResultsHandle;

    fn insert(docs: Vec<i32>) -> u64 {
        lock_recovering(results()).insert(ResultsHandle { docs })
    }

    #[test]
    fn len_and_copy_roundtrip() {
        let h = insert(vec![1, 2, 3]);
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_results_len(h, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 3);

        let mut buf = [0i32; 3];
        assert_eq!(
            unsafe { ffi_results_copy(h, buf.as_mut_ptr(), buf.len()) },
            FfiStatus::Ok.code()
        );
        assert_eq!(buf, [1, 2, 3]);
        assert_eq!(ffi_close_results(h), FfiStatus::Ok.code());
    }

    #[test]
    fn copy_with_buffer_too_small_leaves_error_and_writes_nothing_observable() {
        let h = insert(vec![1, 2, 3]);
        let mut buf = [9i32; 2];
        let rc = unsafe { ffi_results_copy(h, buf.as_mut_ptr(), buf.len()) };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        assert_eq!(buf, [9, 9]);
        ffi_close_results(h);
    }

    #[test]
    fn copy_empty_results_with_null_buf_and_zero_len_is_ok() {
        let h = insert(vec![]);
        let rc = unsafe { ffi_results_copy(h, std::ptr::null_mut(), 0) };
        assert_eq!(rc, FfiStatus::Ok.code());
        ffi_close_results(h);
    }

    #[test]
    fn copy_nonempty_with_null_buf_is_null_pointer_error() {
        let h = insert(vec![1]);
        let rc = unsafe { ffi_results_copy(h, std::ptr::null_mut(), 1) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_results(h);
    }

    #[test]
    fn len_unknown_handle_is_invalid_handle() {
        let mut len: usize = 0;
        let rc = unsafe { ffi_results_len(0xABCD, &mut len as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn len_null_out_len_is_null_pointer_error() {
        let h = insert(vec![1]);
        let rc = unsafe { ffi_results_len(h, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_results(h);
    }

    #[test]
    fn copy_unknown_handle_is_invalid_handle() {
        let mut buf = [0i32; 1];
        let rc = unsafe { ffi_results_copy(0xABCD, buf.as_mut_ptr(), buf.len()) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn close_unknown_results_handle_is_invalid_handle() {
        assert_eq!(ffi_close_results(0xABCD), FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn double_close_results_is_invalid_handle_not_a_crash() {
        let h = insert(vec![1]);
        assert_eq!(ffi_close_results(h), FfiStatus::Ok.code());
        assert_eq!(ffi_close_results(h), FfiStatus::InvalidHandle.code());
    }
}
