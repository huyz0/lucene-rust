//! Explain results handles (Query explain FFI exposure): `explain.rs`'s
//! `ffi_explain_term_query`/`ffi_explain_phrase_query`/`ffi_explain_boolean_query`
//! flatten a `lucene_search::explain::Explanation` tree into a new
//! [`crate::registry::ExplainResultsHandle`]; the caller reads it back via
//! [`ffi_explain_node_value`]/[`ffi_explain_node_matched`]/
//! [`ffi_explain_node_description`]/[`ffi_explain_node_child_count`]/
//! [`ffi_explain_node_child_at`] before releasing it with
//! [`ffi_close_explain_results`].
//!
//! **Per-node accessors over a node index, not a bulk `_copy` call**: unlike
//! `results_facets.rs`'s fixed-size `(ord, count)` half, an explain node has
//! no fixed-size element at all that every node shares in a way a single
//! parallel-buffer call could return -- and, being a *tree*, there is no flat
//! "one call per element in order" walk either (contrast
//! `results_fragments.rs`'s per-fragment accessors, which only need a
//! fragment *index*, never a fragment's own nested structure). Every accessor
//! here therefore takes a `node_index` (into `explain.rs`'s flattened `Vec`,
//! see that module's doc comment -- root is always `0`), the same
//! `buf`/`buf_len`/`out_written`/`BufferTooSmall` string contract as
//! [`crate::ffi_get_last_error_message`] for `description`, plus
//! [`ffi_explain_node_child_count`]/[`ffi_explain_node_child_at`] for a caller
//! to walk from a node to each of its children's own indices -- exactly the
//! "length first, then per-index accessor" shape
//! `results_fragments.rs`'s `ffi_fragment_result_matched_terms_len`/
//! `ffi_fragment_result_matched_term` pair already established for a
//! variable-length list, applied here to a node's *children* instead of a
//! fragment's *matched terms*.

use std::os::raw::c_char;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::registry::{explain_results, lock_recovering};

/// Writes the total number of flattened nodes held by
/// `explain_results_handle` to `*out_len` -- every valid `node_index` passed
/// to the other accessors below is `< ` this value.
///
/// # Safety
/// `out_len` must be valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_explain_results_len(
    explain_results_handle: u64,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = lock_recovering(explain_results());
        let handle = registry.get(explain_results_handle).ok_or_else(|| {
            set_last_error("ffi_explain_results_len: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        // SAFETY: caller contract guarantees `out_len` is valid for one write.
        unsafe {
            *out_len = handle.nodes.len();
        }
        Ok(())
    })
}

/// Writes node `node_index`'s `value` (real Lucene `Explanation.getValue()`
/// equivalent) to `*out_value`.
///
/// # Safety
/// `out_value` must be valid for one `f32` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_explain_node_value(
    explain_results_handle: u64,
    node_index: usize,
    out_value: *mut f32,
) -> i32 {
    guard(|| {
        if out_value.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = lock_recovering(explain_results());
        let handle = registry.get(explain_results_handle).ok_or_else(|| {
            set_last_error("ffi_explain_node_value: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let node = handle.nodes.get(node_index).ok_or_else(|| {
            set_last_error(format!(
                "ffi_explain_node_value: node_index {node_index} out of bounds (len {})",
                handle.nodes.len()
            ));
            FfiStatus::IndexOutOfBounds
        })?;
        // SAFETY: caller contract guarantees `out_value` is valid for one write.
        unsafe {
            *out_value = node.value;
        }
        Ok(())
    })
}

/// Writes node `node_index`'s `matched` (real Lucene `Explanation.isMatch()`
/// equivalent) to `*out_matched` as `0`/`1`.
///
/// # Safety
/// `out_matched` must be valid for one `u8` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_explain_node_matched(
    explain_results_handle: u64,
    node_index: usize,
    out_matched: *mut u8,
) -> i32 {
    guard(|| {
        if out_matched.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = lock_recovering(explain_results());
        let handle = registry.get(explain_results_handle).ok_or_else(|| {
            set_last_error("ffi_explain_node_matched: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let node = handle.nodes.get(node_index).ok_or_else(|| {
            set_last_error(format!(
                "ffi_explain_node_matched: node_index {node_index} out of bounds (len {})",
                handle.nodes.len()
            ));
            FfiStatus::IndexOutOfBounds
        })?;
        // SAFETY: caller contract guarantees `out_matched` is valid for one write.
        unsafe {
            *out_matched = node.matched as u8;
        }
        Ok(())
    })
}

/// Copies node `node_index`'s `description` into `buf` (caller-allocated,
/// `buf_len` bytes), NUL-terminated, writing the number of bytes written
/// (excluding the NUL) to `*out_written` -- same
/// `buf`/`buf_len`/`out_written`/`BufferTooSmall` contract as
/// [`crate::ffi_get_last_error_message`].
///
/// # Safety
/// `buf` must be valid for writes of `buf_len` bytes; `out_written` must be
/// valid for one `usize` write, or null.
#[no_mangle]
pub unsafe extern "C" fn ffi_explain_node_description(
    explain_results_handle: u64,
    node_index: usize,
    buf: *mut c_char,
    buf_len: usize,
    out_written: *mut usize,
) -> i32 {
    guard(|| {
        let registry = lock_recovering(explain_results());
        let handle = registry.get(explain_results_handle).ok_or_else(|| {
            set_last_error("ffi_explain_node_description: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let node = handle.nodes.get(node_index).ok_or_else(|| {
            set_last_error(format!(
                "ffi_explain_node_description: node_index {node_index} out of bounds (len {})",
                handle.nodes.len()
            ));
            FfiStatus::IndexOutOfBounds
        })?;
        let bytes = node.description.as_bytes();
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

/// Writes the number of children held by node `node_index` to `*out_len` --
/// call before looping [`ffi_explain_node_child_at`] over `0..len`, the same
/// "length first" shape [`crate::results_fragments::ffi_fragment_result_matched_terms_len`]
/// establishes for a fragment's matched-terms list.
///
/// # Safety
/// `out_len` must be valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_explain_node_child_count(
    explain_results_handle: u64,
    node_index: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = lock_recovering(explain_results());
        let handle = registry.get(explain_results_handle).ok_or_else(|| {
            set_last_error("ffi_explain_node_child_count: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let node = handle.nodes.get(node_index).ok_or_else(|| {
            set_last_error(format!(
                "ffi_explain_node_child_count: node_index {node_index} out of bounds (len {})",
                handle.nodes.len()
            ));
            FfiStatus::IndexOutOfBounds
        })?;
        // SAFETY: caller contract guarantees `out_len` is valid for one write.
        unsafe {
            *out_len = node.children.len();
        }
        Ok(())
    })
}

/// Writes node `node_index`'s `child_index`'th child's own node index (into
/// this same handle's flattened tree -- pass it straight back into
/// [`ffi_explain_node_value`]/[`ffi_explain_node_matched`]/
/// [`ffi_explain_node_description`]/[`ffi_explain_node_child_count`] to
/// descend the tree) to `*out_child_node_index`. Returns
/// [`FfiStatus::IndexOutOfBounds`] for either an out-of-range `node_index` or
/// an out-of-range `child_index` (read via [`ffi_explain_node_child_count`]).
///
/// # Safety
/// `out_child_node_index` must be valid for one `usize` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_explain_node_child_at(
    explain_results_handle: u64,
    node_index: usize,
    child_index: usize,
    out_child_node_index: *mut usize,
) -> i32 {
    guard(|| {
        if out_child_node_index.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let registry = lock_recovering(explain_results());
        let handle = registry.get(explain_results_handle).ok_or_else(|| {
            set_last_error("ffi_explain_node_child_at: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let node = handle.nodes.get(node_index).ok_or_else(|| {
            set_last_error(format!(
                "ffi_explain_node_child_at: node_index {node_index} out of bounds (len {})",
                handle.nodes.len()
            ));
            FfiStatus::IndexOutOfBounds
        })?;
        let child = node.children.get(child_index).ok_or_else(|| {
            set_last_error(format!(
                "ffi_explain_node_child_at: child_index {child_index} out of bounds (len {})",
                node.children.len()
            ));
            FfiStatus::IndexOutOfBounds
        })?;
        // SAFETY: caller contract guarantees `out_child_node_index` is valid
        // for one write.
        unsafe {
            *out_child_node_index = *child;
        }
        Ok(())
    })
}

/// Closes an explain results handle. Returns [`FfiStatus::InvalidHandle`] for
/// an unknown/already-closed handle.
#[no_mangle]
pub extern "C" fn ffi_close_explain_results(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(explain_results())
            .remove(handle)
            .map(|_| ())
            .ok_or_else(|| {
                set_last_error("ffi_close_explain_results: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{ExplainNode, ExplainResultsHandle};

    fn insert(nodes: Vec<ExplainNode>) -> u64 {
        lock_recovering(explain_results()).insert(ExplainResultsHandle { nodes })
    }

    /// A tiny two-level tree: root (value 3.0, matched, "sum of:") with two
    /// leaf children (1.0 "a", 2.0 "b").
    fn sample() -> Vec<ExplainNode> {
        vec![
            ExplainNode {
                matched: true,
                value: 3.0,
                description: "sum of:".to_string(),
                children: vec![1, 2],
            },
            ExplainNode {
                matched: true,
                value: 1.0,
                description: "a".to_string(),
                children: vec![],
            },
            ExplainNode {
                matched: true,
                value: 2.0,
                description: "b".to_string(),
                children: vec![],
            },
        ]
    }

    #[test]
    fn full_tree_walk_roundtrip() {
        let h = insert(sample());

        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_explain_results_len(h, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 3);

        let mut value: f32 = 0.0;
        assert_eq!(
            unsafe { ffi_explain_node_value(h, 0, &mut value as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(value, 3.0);

        let mut matched: u8 = 0;
        assert_eq!(
            unsafe { ffi_explain_node_matched(h, 0, &mut matched as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(matched, 1);

        let mut buf = [0 as c_char; 32];
        let mut written: usize = 0;
        assert_eq!(
            unsafe {
                ffi_explain_node_description(
                    h,
                    0,
                    buf.as_mut_ptr(),
                    buf.len(),
                    &mut written as *mut _,
                )
            },
            FfiStatus::Ok.code()
        );
        let desc = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_str()
            .unwrap();
        assert_eq!(desc, "sum of:");

        let mut child_count: usize = 0;
        assert_eq!(
            unsafe { ffi_explain_node_child_count(h, 0, &mut child_count as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(child_count, 2);

        let mut child0: usize = 0;
        assert_eq!(
            unsafe { ffi_explain_node_child_at(h, 0, 0, &mut child0 as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(child0, 1);
        let mut child1: usize = 0;
        assert_eq!(
            unsafe { ffi_explain_node_child_at(h, 0, 1, &mut child1 as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(child1, 2);

        for (idx, expected_value, expected_desc) in [(1usize, 1.0f32, "a"), (2, 2.0, "b")] {
            let mut v: f32 = 0.0;
            assert_eq!(
                unsafe { ffi_explain_node_value(h, idx, &mut v as *mut _) },
                FfiStatus::Ok.code()
            );
            assert_eq!(v, expected_value);
            let mut cc: usize = 0;
            assert_eq!(
                unsafe { ffi_explain_node_child_count(h, idx, &mut cc as *mut _) },
                FfiStatus::Ok.code()
            );
            assert_eq!(cc, 0);
            let mut buf = [0 as c_char; 16];
            assert_eq!(
                unsafe {
                    ffi_explain_node_description(
                        h,
                        idx,
                        buf.as_mut_ptr(),
                        buf.len(),
                        std::ptr::null_mut(),
                    )
                },
                FfiStatus::Ok.code()
            );
            let desc = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
                .to_str()
                .unwrap();
            assert_eq!(desc, expected_desc);
        }

        assert_eq!(ffi_close_explain_results(h), FfiStatus::Ok.code());
    }

    #[test]
    fn results_len_unknown_handle_is_invalid_handle() {
        let mut len: usize = 0;
        let rc = unsafe { ffi_explain_results_len(0xABCD, &mut len as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn results_len_null_out_len_is_null_pointer_error() {
        let h = insert(sample());
        let rc = unsafe { ffi_explain_results_len(h, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_explain_results(h);
    }

    #[test]
    fn value_unknown_handle_is_invalid_handle() {
        let mut value: f32 = 0.0;
        let rc = unsafe { ffi_explain_node_value(0xABCD, 0, &mut value as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn value_out_of_range_node_index_is_index_out_of_bounds() {
        let h = insert(sample());
        let mut value: f32 = 0.0;
        let rc = unsafe { ffi_explain_node_value(h, 9, &mut value as *mut _) };
        assert_eq!(rc, FfiStatus::IndexOutOfBounds.code());
        ffi_close_explain_results(h);
    }

    #[test]
    fn value_null_out_value_is_null_pointer_error() {
        let h = insert(sample());
        let rc = unsafe { ffi_explain_node_value(h, 0, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_explain_results(h);
    }

    #[test]
    fn matched_unknown_handle_is_invalid_handle() {
        let mut matched: u8 = 0;
        let rc = unsafe { ffi_explain_node_matched(0xABCD, 0, &mut matched as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn matched_out_of_range_node_index_is_index_out_of_bounds() {
        let h = insert(sample());
        let mut matched: u8 = 0;
        let rc = unsafe { ffi_explain_node_matched(h, 9, &mut matched as *mut _) };
        assert_eq!(rc, FfiStatus::IndexOutOfBounds.code());
        ffi_close_explain_results(h);
    }

    #[test]
    fn matched_null_out_matched_is_null_pointer_error() {
        let h = insert(sample());
        let rc = unsafe { ffi_explain_node_matched(h, 0, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_explain_results(h);
    }

    #[test]
    fn description_unknown_handle_is_invalid_handle() {
        let mut buf = [0 as c_char; 16];
        let rc = unsafe {
            ffi_explain_node_description(
                0xABCD,
                0,
                buf.as_mut_ptr(),
                buf.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn description_out_of_range_node_index_is_index_out_of_bounds() {
        let h = insert(sample());
        let mut buf = [0 as c_char; 16];
        let rc = unsafe {
            ffi_explain_node_description(h, 9, buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
        };
        assert_eq!(rc, FfiStatus::IndexOutOfBounds.code());
        ffi_close_explain_results(h);
    }

    #[test]
    fn description_buffer_too_small_is_buffer_too_small() {
        let h = insert(sample());
        let mut buf = [0 as c_char; 2];
        let rc = unsafe {
            ffi_explain_node_description(h, 0, buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
        };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        ffi_close_explain_results(h);
    }

    #[test]
    fn description_null_buf_is_null_pointer_error() {
        let h = insert(sample());
        let mut written: usize = 0;
        let rc = unsafe {
            ffi_explain_node_description(h, 0, std::ptr::null_mut(), 64, &mut written as *mut _)
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_explain_results(h);
    }

    #[test]
    fn child_count_unknown_handle_is_invalid_handle() {
        let mut len: usize = 0;
        let rc = unsafe { ffi_explain_node_child_count(0xABCD, 0, &mut len as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn child_count_out_of_range_node_index_is_index_out_of_bounds() {
        let h = insert(sample());
        let mut len: usize = 0;
        let rc = unsafe { ffi_explain_node_child_count(h, 9, &mut len as *mut _) };
        assert_eq!(rc, FfiStatus::IndexOutOfBounds.code());
        ffi_close_explain_results(h);
    }

    #[test]
    fn child_count_null_out_len_is_null_pointer_error() {
        let h = insert(sample());
        let rc = unsafe { ffi_explain_node_child_count(h, 0, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_explain_results(h);
    }

    #[test]
    fn child_at_unknown_handle_is_invalid_handle() {
        let mut out: usize = 0;
        let rc = unsafe { ffi_explain_node_child_at(0xABCD, 0, 0, &mut out as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn child_at_out_of_range_node_index_is_index_out_of_bounds() {
        let h = insert(sample());
        let mut out: usize = 0;
        let rc = unsafe { ffi_explain_node_child_at(h, 9, 0, &mut out as *mut _) };
        assert_eq!(rc, FfiStatus::IndexOutOfBounds.code());
        ffi_close_explain_results(h);
    }

    #[test]
    fn child_at_out_of_range_child_index_is_index_out_of_bounds() {
        let h = insert(sample());
        let mut out: usize = 0;
        let rc = unsafe { ffi_explain_node_child_at(h, 0, 9, &mut out as *mut _) };
        assert_eq!(rc, FfiStatus::IndexOutOfBounds.code());
        ffi_close_explain_results(h);
    }

    #[test]
    fn child_at_null_out_is_null_pointer_error() {
        let h = insert(sample());
        let rc = unsafe { ffi_explain_node_child_at(h, 0, 0, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_explain_results(h);
    }

    #[test]
    fn leaf_node_zero_children_is_ok() {
        let h = insert(sample());
        let mut cc: usize = 0;
        assert_eq!(
            unsafe { ffi_explain_node_child_count(h, 1, &mut cc as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(cc, 0);
        ffi_close_explain_results(h);
    }

    #[test]
    fn close_unknown_explain_results_handle_is_invalid_handle() {
        assert_eq!(
            ffi_close_explain_results(0xABCD),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn double_close_explain_results_is_invalid_handle_not_a_crash() {
        let h = insert(sample());
        assert_eq!(ffi_close_explain_results(h), FfiStatus::Ok.code());
        assert_eq!(
            ffi_close_explain_results(h),
            FfiStatus::InvalidHandle.code()
        );
    }
}
