//! Raw-pointer-to-Rust-value conversions shared by every exported function.
//! Centralized here so every `unsafe` pointer dereference in this crate's
//! public surface goes through exactly these two helpers (see the
//! `ffi-safety` skill: `unsafe` is scoped, not sprinkled).

use crate::error::FfiStatus;

/// Reads `len` bytes at `ptr` as a UTF-8 `&str` borrowing from the caller's
/// buffer -- no copy, no ownership transfer. `ptr` may be null only when
/// `len == 0` (an empty string), matching a common C-ABI convention for
/// "empty and possibly not backed by a real allocation".
///
/// # Safety
/// `ptr` must be valid for reads of `len` bytes for the duration of the
/// borrow returned.
pub unsafe fn str_from_raw<'a>(ptr: *const u8, len: usize) -> Result<&'a str, FfiStatus> {
    if ptr.is_null() {
        return if len == 0 {
            Ok("")
        } else {
            Err(FfiStatus::NullPointer)
        };
    }
    // SAFETY: caller contract guarantees `ptr` is valid for `len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn str_from_raw_reads_valid_utf8() {
        let s = "hello";
        let got = unsafe { str_from_raw(s.as_ptr(), s.len()) }.unwrap();
        assert_eq!(got, "hello");
    }

    #[test]
    fn str_from_raw_rejects_invalid_utf8() {
        let bytes = [0xFFu8, 0xFE];
        let got = unsafe { str_from_raw(bytes.as_ptr(), bytes.len()) };
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
