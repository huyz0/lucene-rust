//! The alignment contract between formats and page-based readers.
//!
//! A format whose blocks start on fixed boundaries (stored-field blocks
//! compressed to whole pages, say) opens its files with
//! [`Directory::open_aligned`](crate::Directory::open_aligned) and its block
//! size as the hint. A reader that caches whole pages then only uses pages
//! of [`page_size_for`] that hint, so a block never straddles two pages.
//! Every hint is [`ALIGN_BASE`] times a power of two.

use crate::error::{Error, Result};

/// The unit every alignment hint is a power-of-two multiple of: 4 KiB, the
/// OS page and the logical block of every device in use.
pub const ALIGN_BASE: usize = 4096;

/// Whether `bytes` is a valid alignment hint: [`ALIGN_BASE`] times a power
/// of two.
pub const fn is_valid_hint(bytes: usize) -> bool {
    bytes >= ALIGN_BASE && bytes.is_power_of_two()
}

/// Checks a format's alignment hint ([`is_valid_hint`]).
pub fn check_hint(bytes: usize) -> Result<usize> {
    if is_valid_hint(bytes) {
        Ok(bytes)
    } else {
        Err(Error::IllegalArgument(format!(
            "alignment of {bytes} bytes is not {ALIGN_BASE} times a power of two"
        )))
    }
}

/// The page size to use for a file with alignment `hint` when `wanted`
/// bytes would be chosen otherwise: the smallest `hint * 2^k` at least
/// `wanted`, never below the hint. An invalid hint counts as
/// [`ALIGN_BASE`].
pub fn page_size_for(hint: usize, wanted: usize) -> usize {
    let hint = if is_valid_hint(hint) {
        hint
    } else {
        ALIGN_BASE
    };
    wanted.max(hint).checked_next_power_of_two().unwrap_or(hint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hints_are_base_times_powers_of_two() {
        assert!(is_valid_hint(4096));
        assert!(is_valid_hint(16 << 10));
        assert!(!is_valid_hint(2048));
        assert!(!is_valid_hint(12 << 10));
        assert!(check_hint(0).is_err());
    }

    #[test]
    fn page_sizes_never_go_below_the_hint() {
        assert_eq!(page_size_for(16 << 10, 4 << 10), 16 << 10);
        assert_eq!(page_size_for(16 << 10, 64 << 10), 64 << 10);
        assert_eq!(page_size_for(16 << 10, 40 << 10), 64 << 10);
        assert_eq!(page_size_for(ALIGN_BASE, 4 << 10), 4 << 10);
        assert_eq!(page_size_for(3000, 4 << 10), 4 << 10);
    }
}
