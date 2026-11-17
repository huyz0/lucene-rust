//! The dictionary-handle plumbing the two zstd FFI surfaces share
//! (`zstd_dictionary` for stored fields, `zstd_terms` for terms blocks): a
//! registry of prepared decoder dictionaries by handle, and the argument
//! checking around one decode call. Each surface keeps its own registry (its
//! own handle tag) and its own exported symbols.

use std::panic::UnwindSafe;
use std::sync::{Arc, RwLock};

use lucene_engine::handle::{RegistryTag, SlotMap};
use lucene_zstd::DecoderDictionary;

use crate::error::{guard, FfiStatus};
use crate::raw::bytes_from_raw;
use crate::registry::{lock_recovering, read_recovering};

/// Prepared dictionaries by handle.
pub(crate) type Registry = RwLock<SlotMap<Arc<DecoderDictionary>>>;

/// An empty registry whose handles carry `tag`.
pub(crate) fn new_registry(tag: RegistryTag) -> Registry {
    RwLock::new(SlotMap::new(tag))
}

/// Prepares the `len` dictionary bytes at `dict` and writes its handle to
/// `*out_handle`.
///
/// # Safety
/// `dict` must be valid for `len` bytes and `out_handle` for one write.
pub(crate) unsafe fn open(
    registry: &Registry,
    dict: *const u8,
    len: usize,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let dict = unsafe { bytes_from_raw(dict, len) }?;
        let prepared = Arc::new(DecoderDictionary::new(dict));
        let handle = lock_recovering(registry).insert_checked(prepared)?;
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
}

/// Releases a handle from [`open`].
pub(crate) fn close(registry: &Registry, handle: u64) -> i32 {
    guard(|| {
        lock_recovering(registry)
            .remove(handle)
            .map(drop)
            .ok_or(FfiStatus::InvalidHandle)
    })
}

/// Resolves `handle` (0: no dictionary), views the caller's buffers, and runs
/// `decode(src, dictionary, dst)`.
///
/// # Safety
/// `src` must be valid for `src_len` bytes and `dst` for `dst_len` writes.
pub(crate) unsafe fn decompress(
    registry: &Registry,
    handle: u64,
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_len: usize,
    decode: impl FnOnce(&[u8], Option<&DecoderDictionary>, &mut [u8]) -> Result<(), FfiStatus>
        + UnwindSafe,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let src = unsafe { bytes_from_raw(src, src_len) }?;
        if dst.is_null() && dst_len > 0 {
            return Err(FfiStatus::NullPointer);
        }
        let dst: &mut [u8] = if dst_len == 0 {
            &mut []
        } else {
            // SAFETY: caller contract; non-null checked above.
            unsafe { std::slice::from_raw_parts_mut(dst, dst_len) }
        };
        let dict = if handle == 0 {
            None
        } else {
            Some(
                read_recovering(registry)
                    .get(handle)
                    .cloned()
                    .ok_or(FfiStatus::InvalidHandle)?,
            )
        };
        decode(src, dict.as_deref(), dst)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> Registry {
        new_registry(RegistryTag::ZstdDictionary)
    }

    #[test]
    fn open_refuses_a_null_out_pointer_and_a_null_dictionary_with_length() {
        let r = registry();
        let mut h = 0u64;
        // SAFETY: the null `out_handle` is refused before any write.
        let st = unsafe { open(&r, b"x".as_ptr(), 1, std::ptr::null_mut()) };
        assert_eq!(st, FfiStatus::NullPointer.code());
        // SAFETY: `h` is valid for one write; null with a length is refused.
        let st = unsafe { open(&r, std::ptr::null(), 4, &mut h) };
        assert_eq!(st, FfiStatus::NullPointer.code());
        assert_eq!(h, 0);
    }

    #[test]
    fn decompress_checks_handles_and_buffers_before_decoding() {
        let r = registry();
        let called = |_: &[u8], _: Option<&DecoderDictionary>, _: &mut [u8]| Ok(());
        let mut out = [0u8; 4];
        // SAFETY: valid buffers; the handle is unknown.
        let st = unsafe { decompress(&r, 99, [].as_ptr(), 0, out.as_mut_ptr(), 4, called) };
        assert_eq!(st, FfiStatus::InvalidHandle.code());
        // SAFETY: a null destination with a length is refused first.
        let st = unsafe { decompress(&r, 0, [].as_ptr(), 0, std::ptr::null_mut(), 4, called) };
        assert_eq!(st, FfiStatus::NullPointer.code());
        // SAFETY: an empty destination needs no pointer; handle 0 is "none".
        let st = unsafe { decompress(&r, 0, [].as_ptr(), 0, std::ptr::null_mut(), 0, called) };
        assert_eq!(st, 0);
        assert_eq!(close(&r, 5), FfiStatus::InvalidHandle.code());
    }
}
