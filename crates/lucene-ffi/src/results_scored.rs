//! Scored results handles: `ffi_search_*_query_scored` (`query.rs`) collect
//! each matched, live doc's `(doc_id, score)` pair (via a
//! `lucene_search::TopDocsCollector`, entirely Rust-side -- see the
//! `ffi-safety` skill's "no callbacks from Rust into Java" rule) into a new
//! [`crate::registry::ScoredResultsHandle`]; the caller reads it back via
//! [`ffi_scored_results_len`]/[`ffi_scored_results_copy`] before releasing it
//! with [`ffi_close_scored_results`].
//!
//! **Wire shape: two parallel out-buffers, not one interleaved buffer**:
//! `ffi_scored_results_copy` writes doc IDs into a caller-allocated `*mut i32`
//! buffer and scores into a separate caller-allocated `*mut f32` buffer (both
//! of length `buf_len`, index `i` in one corresponding to index `i` in the
//! other) rather than one `[i32, f32, i32, f32, ...]`-interleaved buffer. A
//! JNI caller almost always wants to build two separate Java arrays (`int[]`
//! doc IDs, `float[]` scores) or feed a doc-ID array plus a score array to two
//! different downstream APIs -- parallel buffers hand that back directly,
//! while an interleaved buffer would force the caller to de-interleave one
//! step-4-bytes-at-a-time (`i32`) / step-4-bytes-at-a-time (`f32`) buffer
//! itself on the other side of the boundary for no benefit. This mirrors
//! `results.rs`'s own "bulk copy matches the caller's real access pattern"
//! reasoning for choosing one shape over another.

use crate::error::{guard, set_last_error, FfiStatus};
use crate::registry::{lock_recovering, scored_results};

/// Writes the number of `(doc_id, score)` hits held by `scored_results_handle`
/// to `*out_len`.
///
/// # Safety
/// `out_len` must be valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_scored_results_len(
    scored_results_handle: u64,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = lock_recovering(scored_results());
        let handle = registry.get(scored_results_handle).ok_or_else(|| {
            set_last_error("ffi_scored_results_len: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        // SAFETY: caller contract guarantees `out_len` is valid for one write.
        unsafe {
            *out_len = handle.hits.len();
        }
        Ok(())
    })
}

/// Bulk-copies up to `buf_len` `(doc_id, score)` hits from `scored_results_handle`
/// into the caller-allocated `out_doc_ids`/`out_scores` (parallel buffers -- see
/// this module's doc comment for why not one interleaved buffer). Returns
/// [`FfiStatus::BufferTooSmall`] (with nothing written) if `buf_len` is smaller
/// than the results' actual length -- call [`ffi_scored_results_len`] first to
/// size the buffers.
///
/// # Safety
/// `out_doc_ids` must be valid for writes of `buf_len` `i32`s, `out_scores` for
/// writes of `buf_len` `f32`s.
#[no_mangle]
pub unsafe extern "C" fn ffi_scored_results_copy(
    scored_results_handle: u64,
    out_doc_ids: *mut i32,
    out_scores: *mut f32,
    buf_len: usize,
) -> i32 {
    guard(|| {
        let registry = lock_recovering(scored_results());
        let handle = registry.get(scored_results_handle).ok_or_else(|| {
            set_last_error("ffi_scored_results_copy: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        if handle.hits.len() > buf_len {
            return Err(FfiStatus::BufferTooSmall);
        }
        if !handle.hits.is_empty() {
            if out_doc_ids.is_null() || out_scores.is_null() {
                return Err(FfiStatus::NullPointer);
            }
            // SAFETY: caller contract guarantees `out_doc_ids`/`out_scores` are
            // each valid for `buf_len` writes of their element type, and
            // `handle.hits.len() <= buf_len` was just checked. Written one
            // element at a time (rather than `copy_nonoverlapping` from a
            // temporary contiguous buffer) since `ScoreDoc`'s two fields aren't
            // laid out as two parallel arrays in memory.
            for (i, hit) in handle.hits.iter().enumerate() {
                unsafe {
                    *out_doc_ids.add(i) = hit.doc_id;
                    *out_scores.add(i) = hit.score;
                }
            }
        }
        Ok(())
    })
}

/// Closes a scored results handle. Returns [`FfiStatus::InvalidHandle`] for an
/// unknown/already-closed handle.
#[no_mangle]
pub extern "C" fn ffi_close_scored_results(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(scored_results())
            .remove(handle)
            .map(|_| ())
            .ok_or_else(|| {
                set_last_error("ffi_close_scored_results: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ScoredResultsHandle;
    use lucene_search::ScoreDoc;

    fn insert(hits: Vec<(i32, f32)>) -> u64 {
        lock_recovering(scored_results()).insert(ScoredResultsHandle {
            hits: hits
                .into_iter()
                .map(|(doc_id, score)| ScoreDoc { doc_id, score })
                .collect(),
        })
    }

    #[test]
    fn len_and_copy_roundtrip() {
        let h = insert(vec![(1, 5.0), (2, 3.0), (3, 1.0)]);
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_scored_results_len(h, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 3);

        let mut doc_ids = [0i32; 3];
        let mut scores = [0.0f32; 3];
        assert_eq!(
            unsafe {
                ffi_scored_results_copy(h, doc_ids.as_mut_ptr(), scores.as_mut_ptr(), doc_ids.len())
            },
            FfiStatus::Ok.code()
        );
        assert_eq!(doc_ids, [1, 2, 3]);
        assert_eq!(scores, [5.0, 3.0, 1.0]);
        assert_eq!(ffi_close_scored_results(h), FfiStatus::Ok.code());
    }

    #[test]
    fn copy_with_buffer_too_small_leaves_error_and_writes_nothing_observable() {
        let h = insert(vec![(1, 5.0), (2, 3.0), (3, 1.0)]);
        let mut doc_ids = [9i32; 2];
        let mut scores = [9.0f32; 2];
        let rc = unsafe {
            ffi_scored_results_copy(h, doc_ids.as_mut_ptr(), scores.as_mut_ptr(), doc_ids.len())
        };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        assert_eq!(doc_ids, [9, 9]);
        assert_eq!(scores, [9.0, 9.0]);
        ffi_close_scored_results(h);
    }

    #[test]
    fn copy_empty_results_with_null_bufs_and_zero_len_is_ok() {
        let h = insert(vec![]);
        let rc =
            unsafe { ffi_scored_results_copy(h, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        assert_eq!(rc, FfiStatus::Ok.code());
        ffi_close_scored_results(h);
    }

    #[test]
    fn copy_nonempty_with_null_doc_ids_is_null_pointer_error() {
        let h = insert(vec![(1, 5.0)]);
        let mut scores = [0.0f32; 1];
        let rc =
            unsafe { ffi_scored_results_copy(h, std::ptr::null_mut(), scores.as_mut_ptr(), 1) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_scored_results(h);
    }

    #[test]
    fn copy_nonempty_with_null_scores_is_null_pointer_error() {
        let h = insert(vec![(1, 5.0)]);
        let mut doc_ids = [0i32; 1];
        let rc =
            unsafe { ffi_scored_results_copy(h, doc_ids.as_mut_ptr(), std::ptr::null_mut(), 1) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_scored_results(h);
    }

    #[test]
    fn len_unknown_handle_is_invalid_handle() {
        let mut len: usize = 0;
        let rc = unsafe { ffi_scored_results_len(0xABCD, &mut len as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn len_null_out_len_is_null_pointer_error() {
        let h = insert(vec![(1, 1.0)]);
        let rc = unsafe { ffi_scored_results_len(h, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_scored_results(h);
    }

    #[test]
    fn copy_unknown_handle_is_invalid_handle() {
        let mut doc_ids = [0i32; 1];
        let mut scores = [0.0f32; 1];
        let rc = unsafe {
            ffi_scored_results_copy(0xABCD, doc_ids.as_mut_ptr(), scores.as_mut_ptr(), 1)
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn close_unknown_scored_results_handle_is_invalid_handle() {
        assert_eq!(
            ffi_close_scored_results(0xABCD),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn double_close_scored_results_is_invalid_handle_not_a_crash() {
        let h = insert(vec![(1, 1.0)]);
        assert_eq!(ffi_close_scored_results(h), FfiStatus::Ok.code());
        assert_eq!(ffi_close_scored_results(h), FfiStatus::InvalidHandle.code());
    }
}
