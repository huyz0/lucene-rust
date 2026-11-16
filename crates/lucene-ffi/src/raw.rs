//! Raw-pointer-to-Rust-value conversions shared by every exported function.
//! Centralized here so every `unsafe` pointer dereference in this crate's
//! public surface goes through exactly these two helpers (see the
//! `ffi-safety` skill: `unsafe` is scoped, not sprinkled).

use crate::error::FfiStatus;
use std::ffi::c_char;

/// Reads `len` bytes at `ptr` as a UTF-8 `&str` borrowing from the caller's
/// buffer -- no copy, no ownership transfer. `ptr` may be null only when
/// `len == 0` (an empty string), matching a common C-ABI convention for
/// "empty and possibly not backed by a real allocation".
///
/// Takes `*const c_char` because every caller is converting a C string. The
/// widening to `*const u8` happens here, once, via `.cast()` rather than an
/// `as` expression: `c_char` is `i8` on x86_64 but `u8` on aarch64, so an
/// `as *const u8` at each call site is a real cast on one target and a no-op
/// the linter rejects on the other.
///
/// # Safety
/// `ptr` must be valid for reads of `len` bytes for the duration of the
/// borrow returned.
pub unsafe fn str_from_raw<'a>(ptr: *const c_char, len: usize) -> Result<&'a str, FfiStatus> {
    if ptr.is_null() {
        return if len == 0 {
            Ok("")
        } else {
            Err(FfiStatus::NullPointer)
        };
    }
    // SAFETY: caller contract guarantees `ptr` is valid for `len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) };
    std::str::from_utf8(bytes).map_err(|_| FfiStatus::InvalidUtf8)
}

/// Reads `len` bytes at `ptr` as a byte slice borrowing from the caller's
/// buffer. Same null/zero-length convention as [`str_from_raw`].
///
/// # Safety
/// `ptr` must be valid for reads of `len` bytes for the duration of the
/// borrow returned.
pub unsafe fn bytes_from_raw<'a>(ptr: *const u8, len: usize) -> Result<&'a [u8], FfiStatus> {
    if ptr.is_null() {
        return if len == 0 {
            Ok(&[])
        } else {
            Err(FfiStatus::NullPointer)
        };
    }
    // SAFETY: caller contract guarantees `ptr` is valid for `len` bytes.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

pub use lucene_engine::alloc::{try_to_vec, try_with_capacity};

#[cfg(test)]
mod tests {
    // Tests index and count freely: the gate is about values read off disk.
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    #[test]
    fn str_from_raw_reads_valid_utf8() {
        let s = "hello";
        let got = unsafe { str_from_raw(s.as_ptr().cast::<c_char>(), s.len()) }.unwrap();
        assert_eq!(got, "hello");
    }

    #[test]
    fn str_from_raw_rejects_invalid_utf8() {
        let bytes = [0xFFu8, 0xFE];
        let got = unsafe { str_from_raw(bytes.as_ptr().cast::<c_char>(), bytes.len()) };
        assert_eq!(got, Err(FfiStatus::InvalidUtf8));
    }

    #[test]
    fn str_from_raw_null_with_zero_len_is_empty_string() {
        let got = unsafe { str_from_raw(std::ptr::null(), 0) }.unwrap();
        assert_eq!(got, "");
    }

    #[test]
    fn str_from_raw_null_with_nonzero_len_is_null_pointer_error() {
        let got = unsafe { str_from_raw(std::ptr::null(), 3) };
        assert_eq!(got, Err(FfiStatus::NullPointer));
    }

    #[test]
    fn bytes_from_raw_reads_bytes() {
        let b = [1u8, 2, 3];
        let got = unsafe { bytes_from_raw(b.as_ptr(), b.len()) }.unwrap();
        assert_eq!(got, &[1, 2, 3]);
    }

    #[test]
    fn bytes_from_raw_null_with_zero_len_is_empty_slice() {
        let got = unsafe { bytes_from_raw(std::ptr::null(), 0) }.unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn bytes_from_raw_null_with_nonzero_len_is_null_pointer_error() {
        let got = unsafe { bytes_from_raw(std::ptr::null(), 5) };
        assert_eq!(got, Err(FfiStatus::NullPointer));
    }
}

/// Tests that the whole crate really is free of the "allocate a
/// caller-supplied length before validating it" abort hazard -- a
/// source-scanning invariant test, since the failure mode
/// (`handle_alloc_error` aborting the JVM) cannot be reproduced from inside
/// a test.
///
/// **Why not a `clippy.toml` `disallowed-methods` entry**, which would be a
/// real Tier-1 gate: `clippy.toml` is per-*workspace*, not per-crate, and
/// `Vec::with_capacity` is legitimate everywhere else in this workspace (a
/// decoder sizing a buffer from an already-validated header count is fine).
/// Banning it workspace-wide would produce dozens of false positives in
/// crates that never see a caller-supplied length. Scoped lint config would
/// be the right tool the day cargo grows it.
#[cfg(test)]
mod allocation_hazard_tests;
