//! Fragment results handles (Highlighter FFI exposure):
//! `ffi_assemble_fragments` (`highlighter.rs`) collects assembled
//! `lucene_search::highlighter::Fragment`s into a new
//! [`crate::registry::FragmentResultsHandle`]; the caller reads it back via
//! [`ffi_fragment_results_len`]/[`ffi_fragment_result_text`]/
//! [`ffi_fragment_result_matched_terms_len`]/[`ffi_fragment_result_matched_term`]
//! before releasing it with [`ffi_close_fragment_results`].
//!
//! **Two variable-length fields per element, unlike `results_facets.rs`'s
//! one**: a [`lucene_search::highlighter::Fragment`] has no fixed-size half
//! at all -- both `text` and `matched_terms` are variable-length, so there is
//! no `_copy`-style bulk call here (contrast `results_facets.rs`'s
//! `ffi_facet_results_copy`, which bulk-copies the fixed-size `(ord, count)`
//! half of each facet). Every accessor here is per-index/per-term, reusing
//! this crate's existing `buf`/`buf_len`/`out_written`/`BufferTooSmall`
//! contract from [`crate::ffi_get_last_error_message`] for each string --
//! `ffi_fragment_result_matched_terms_len` exists precisely so a caller can
//! size its loop over `matched_terms` before fetching them one at a time via
//! `ffi_fragment_result_matched_term`, the same "length first, then
//! per-index string accessor" shape `results_facets.rs`'s
//! `ffi_facet_result_label` already established for a single string field.

use std::os::raw::c_char;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::registry::{fragment_results, lock_recovering};

/// Writes the number of assembled fragments held by
/// `fragment_results_handle` to `*out_len`.
///
/// # Safety
/// `out_len` must be valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_fragment_results_len(
    fragment_results_handle: u64,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = lock_recovering(fragment_results());
        let handle = registry.get(fragment_results_handle).ok_or_else(|| {
            set_last_error("ffi_fragment_results_len: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        // SAFETY: caller contract guarantees `out_len` is valid for one write.
        unsafe {
            *out_len = handle.fragments.len();
        }
        Ok(())
    })
}

/// Copies fragment index `index`'s highlighted `text` into `buf`
/// (caller-allocated, `buf_len` bytes), NUL-terminated, writing the number of
/// bytes written (excluding the NUL) to `*out_written` -- same
/// `buf`/`buf_len`/`out_written`/`BufferTooSmall` contract as
/// [`crate::ffi_get_last_error_message`]. Returns
/// [`FfiStatus::IndexOutOfBounds`] for `index >= ` the handle's length (read
/// via [`ffi_fragment_results_len`]).
///
/// # Safety
/// `buf` must be valid for writes of `buf_len` bytes; `out_written` must be
/// valid for one `usize` write, or null.
#[no_mangle]
pub unsafe extern "C" fn ffi_fragment_result_text(
    fragment_results_handle: u64,
    index: usize,
    buf: *mut c_char,
    buf_len: usize,
    out_written: *mut usize,
) -> i32 {
    guard(|| {
        let registry = lock_recovering(fragment_results());
        let handle = registry.get(fragment_results_handle).ok_or_else(|| {
            set_last_error("ffi_fragment_result_text: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let fragment = handle.fragments.get(index).ok_or_else(|| {
            set_last_error(format!(
                "ffi_fragment_result_text: index {index} out of bounds (len {})",
                handle.fragments.len()
            ));
            FfiStatus::IndexOutOfBounds
        })?;
        let bytes = fragment.text.as_bytes();
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

/// Writes the number of distinct matched terms held by fragment index
/// `index` to `*out_len` -- call before looping
/// [`ffi_fragment_result_matched_term`] over `0..len`, the same "length
/// first" shape [`ffi_fragment_results_len`] establishes for the fragment
/// list itself.
///
/// # Safety
/// `out_len` must be valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_fragment_result_matched_terms_len(
    fragment_results_handle: u64,
    index: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = lock_recovering(fragment_results());
        let handle = registry.get(fragment_results_handle).ok_or_else(|| {
            set_last_error(
                "ffi_fragment_result_matched_terms_len: unknown or already-closed handle",
            );
            FfiStatus::InvalidHandle
        })?;
        let fragment = handle.fragments.get(index).ok_or_else(|| {
            set_last_error(format!(
                "ffi_fragment_result_matched_terms_len: index {index} out of bounds (len {})",
                handle.fragments.len()
            ));
            FfiStatus::IndexOutOfBounds
        })?;
        // SAFETY: caller contract guarantees `out_len` is valid for one write.
        unsafe {
            *out_len = fragment.matched_terms.len();
        }
        Ok(())
    })
}

/// Copies fragment index `index`'s `term_index`'th matched term into `buf`,
/// same `buf`/`buf_len`/`out_written`/`BufferTooSmall` contract as
/// [`ffi_fragment_result_text`]. Returns [`FfiStatus::IndexOutOfBounds`] for
/// either an out-of-range `index` or an out-of-range `term_index` (read via
/// [`ffi_fragment_result_matched_terms_len`]).
///
/// # Safety
/// `buf` must be valid for writes of `buf_len` bytes; `out_written` must be
/// valid for one `usize` write, or null.
#[no_mangle]
pub unsafe extern "C" fn ffi_fragment_result_matched_term(
    fragment_results_handle: u64,
    index: usize,
    term_index: usize,
    buf: *mut c_char,
    buf_len: usize,
    out_written: *mut usize,
) -> i32 {
    guard(|| {
        let registry = lock_recovering(fragment_results());
        let handle = registry.get(fragment_results_handle).ok_or_else(|| {
            set_last_error("ffi_fragment_result_matched_term: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let fragment = handle.fragments.get(index).ok_or_else(|| {
            set_last_error(format!(
                "ffi_fragment_result_matched_term: index {index} out of bounds (len {})",
                handle.fragments.len()
            ));
            FfiStatus::IndexOutOfBounds
        })?;
        let term = fragment.matched_terms.get(term_index).ok_or_else(|| {
            set_last_error(format!(
                "ffi_fragment_result_matched_term: term_index {term_index} out of bounds (len {})",
                fragment.matched_terms.len()
            ));
            FfiStatus::IndexOutOfBounds
        })?;
        let bytes = term.as_bytes();
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

/// Closes a fragment results handle. Returns [`FfiStatus::InvalidHandle`] for
/// an unknown/already-closed handle.
#[no_mangle]
pub extern "C" fn ffi_close_fragment_results(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(fragment_results())
            .remove(handle)
            .map(|_| ())
            .ok_or_else(|| {
                set_last_error("ffi_close_fragment_results: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::FragmentResultsHandle;
    use lucene_search::highlighter::Fragment;

    fn insert(fragments: Vec<Fragment>) -> u64 {
        lock_recovering(fragment_results()).insert(FragmentResultsHandle { fragments })
    }

    fn sample() -> Vec<Fragment> {
        vec![
            Fragment {
                text: "<b>cat</b> runs".to_string(),
                matched_terms: vec!["cat".to_string()],
            },
            Fragment {
                text: "<b>car</b> and <b>cat</b>".to_string(),
                matched_terms: vec!["car".to_string(), "cat".to_string()],
            },
        ]
    }

    #[test]
    fn len_text_and_matched_terms_roundtrip() {
        let h = insert(sample());
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_fragment_results_len(h, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 2);

        for (i, expected_text) in ["<b>cat</b> runs", "<b>car</b> and <b>cat</b>"]
            .into_iter()
            .enumerate()
        {
            let mut buf = [0 as c_char; 64];
            let mut written: usize = 0;
            assert_eq!(
                unsafe {
                    ffi_fragment_result_text(
                        h,
                        i,
                        buf.as_mut_ptr(),
                        buf.len(),
                        &mut written as *mut _,
                    )
                },
                FfiStatus::Ok.code()
            );
            let text = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
                .to_str()
                .unwrap();
            assert_eq!(text, expected_text);
            assert_eq!(written, expected_text.len());
        }

        let mut terms_len: usize = 0;
        assert_eq!(
            unsafe { ffi_fragment_result_matched_terms_len(h, 1, &mut terms_len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(terms_len, 2);

        for (j, expected_term) in ["car", "cat"].into_iter().enumerate() {
            let mut buf = [0 as c_char; 16];
            let mut written: usize = 0;
            assert_eq!(
                unsafe {
                    ffi_fragment_result_matched_term(
                        h,
                        1,
                        j,
                        buf.as_mut_ptr(),
                        buf.len(),
                        &mut written as *mut _,
                    )
                },
                FfiStatus::Ok.code()
            );
            let term = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
                .to_str()
                .unwrap();
            assert_eq!(term, expected_term);
            assert_eq!(written, expected_term.len());
        }

        assert_eq!(ffi_close_fragment_results(h), FfiStatus::Ok.code());
    }

    #[test]
    fn text_out_of_bounds_index_is_index_out_of_bounds() {
        let h = insert(sample());
        let mut buf = [0 as c_char; 16];
        let rc = unsafe {
            ffi_fragment_result_text(h, 5, buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
        };
        assert_eq!(rc, FfiStatus::IndexOutOfBounds.code());
        ffi_close_fragment_results(h);
    }

    #[test]
    fn text_buffer_too_small_is_buffer_too_small() {
        let h = insert(sample());
        let mut buf = [0 as c_char; 2];
        let rc = unsafe {
            ffi_fragment_result_text(h, 0, buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
        };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        ffi_close_fragment_results(h);
    }

    #[test]
    fn text_null_buf_is_null_pointer_error() {
        let h = insert(sample());
        let mut written: usize = 0;
        let rc = unsafe {
            ffi_fragment_result_text(h, 0, std::ptr::null_mut(), 64, &mut written as *mut _)
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_fragment_results(h);
    }

    #[test]
    fn text_unknown_handle_is_invalid_handle() {
        let mut buf = [0 as c_char; 16];
        let rc = unsafe {
            ffi_fragment_result_text(0xABCD, 0, buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn matched_terms_len_unknown_handle_is_invalid_handle() {
        let mut len: usize = 0;
        let rc = unsafe { ffi_fragment_result_matched_terms_len(0xABCD, 0, &mut len as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn matched_terms_len_null_out_len_is_null_pointer_error() {
        let h = insert(sample());
        let rc = unsafe { ffi_fragment_result_matched_terms_len(h, 0, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_fragment_results(h);
    }

    #[test]
    fn matched_terms_len_out_of_bounds_fragment_index_is_index_out_of_bounds() {
        let h = insert(sample());
        let mut len: usize = 0;
        let rc = unsafe { ffi_fragment_result_matched_terms_len(h, 9, &mut len as *mut _) };
        assert_eq!(rc, FfiStatus::IndexOutOfBounds.code());
        ffi_close_fragment_results(h);
    }

    #[test]
    fn matched_term_out_of_bounds_fragment_index_is_index_out_of_bounds() {
        let h = insert(sample());
        let mut buf = [0 as c_char; 16];
        let rc = unsafe {
            ffi_fragment_result_matched_term(
                h,
                9,
                0,
                buf.as_mut_ptr(),
                buf.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::IndexOutOfBounds.code());
        ffi_close_fragment_results(h);
    }

    #[test]
    fn matched_term_out_of_bounds_term_index_is_index_out_of_bounds() {
        let h = insert(sample());
        let mut buf = [0 as c_char; 16];
        let rc = unsafe {
            ffi_fragment_result_matched_term(
                h,
                0,
                9,
                buf.as_mut_ptr(),
                buf.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::IndexOutOfBounds.code());
        ffi_close_fragment_results(h);
    }

    #[test]
    fn matched_term_buffer_too_small_is_buffer_too_small() {
        let h = insert(sample());
        let mut buf = [0 as c_char; 1];
        let rc = unsafe {
            ffi_fragment_result_matched_term(
                h,
                1,
                0,
                buf.as_mut_ptr(),
                buf.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        ffi_close_fragment_results(h);
    }

    #[test]
    fn matched_term_null_buf_is_null_pointer_error() {
        let h = insert(sample());
        let rc = unsafe {
            ffi_fragment_result_matched_term(
                h,
                0,
                0,
                std::ptr::null_mut(),
                64,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_fragment_results(h);
    }

    #[test]
    fn matched_term_unknown_handle_is_invalid_handle() {
        let mut buf = [0 as c_char; 16];
        let rc = unsafe {
            ffi_fragment_result_matched_term(
                0xABCD,
                0,
                0,
                buf.as_mut_ptr(),
                buf.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn close_unknown_fragment_results_handle_is_invalid_handle() {
        assert_eq!(
            ffi_close_fragment_results(0xABCD),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn double_close_fragment_results_is_invalid_handle_not_a_crash() {
        let h = insert(sample());
        assert_eq!(ffi_close_fragment_results(h), FfiStatus::Ok.code());
        assert_eq!(
            ffi_close_fragment_results(h),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn len_null_out_len_is_null_pointer_error() {
        let h = insert(sample());
        let rc = unsafe { ffi_fragment_results_len(h, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_fragment_results(h);
    }

    #[test]
    fn len_unknown_handle_is_invalid_handle() {
        let mut len: usize = 0;
        let rc = unsafe { ffi_fragment_results_len(0xABCD, &mut len as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn text_null_out_written_is_ok_and_ignored() {
        let h = insert(sample());
        let mut buf = [0 as c_char; 64];
        let rc = unsafe {
            ffi_fragment_result_text(h, 0, buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        ffi_close_fragment_results(h);
    }

    #[test]
    fn matched_term_null_out_written_is_ok_and_ignored() {
        let h = insert(sample());
        let mut buf = [0 as c_char; 16];
        let rc = unsafe {
            ffi_fragment_result_matched_term(
                h,
                1,
                0,
                buf.as_mut_ptr(),
                buf.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        ffi_close_fragment_results(h);
    }
}
