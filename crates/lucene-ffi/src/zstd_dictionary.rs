//! Zstd decoding for the plugin's Java stored-fields reader of
//! [`lucene_codecs::stored_fields::Mode::ZstdPaged`] segments: a segment's
//! dictionary is prepared once behind a handle, and each chunk's frame is
//! decoded against it.

use std::sync::{Arc, OnceLock, RwLock};

use lucene_zstd::{DecoderDictionary, Dictionary};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::bytes_from_raw;
use crate::registry::{lock_recovering, read_recovering};
use lucene_engine::handle::{RegistryTag, SlotMap};

/// The prepared dictionaries, by handle.
fn dictionaries() -> &'static RwLock<SlotMap<Arc<DecoderDictionary>>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<Arc<DecoderDictionary>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::ZstdDictionary)))
}

/// Prepares the `len` dictionary bytes at `dict` and writes its handle to
/// `*out_handle`; release it with [`ffi_zstd_dictionary_close`].
///
/// # Safety
/// `dict` must be valid for `len` bytes and `out_handle` for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_zstd_dictionary_open(
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
        let handle = lock_recovering(dictionaries()).insert_checked(prepared)?;
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
}

/// Releases a handle from [`ffi_zstd_dictionary_open`].
#[no_mangle]
pub extern "C" fn ffi_zstd_dictionary_close(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(dictionaries())
            .remove(handle)
            .map(drop)
            .ok_or(FfiStatus::InvalidHandle)
    })
}

/// Decodes the zstd frame `src` (`src_len` bytes), compressed against the
/// dictionary `handle` (0: none), into exactly `dst_len` bytes at `dst`.
///
/// # Safety
/// `src` must be valid for `src_len` bytes and `dst` for `dst_len` writes.
#[no_mangle]
pub unsafe extern "C" fn ffi_zstd_paged_decompress(
    handle: u64,
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_len: usize,
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
                read_recovering(dictionaries())
                    .get(handle)
                    .cloned()
                    .ok_or(FfiStatus::InvalidHandle)?,
            )
        };
        let d = dict
            .as_deref()
            .map_or(Dictionary::None, Dictionary::Prepared);
        match lucene_zstd::with_decompressor(|z| z.decompress_to_slice(src, d, dst)) {
            Ok(n) if n == dst_len => Ok(()),
            Ok(n) => {
                set_last_error(format!("zstd frame holds {n} bytes, expected {dst_len}"));
                Err(FfiStatus::Decode)
            }
            Err(e) => {
                set_last_error(format!("zstd stored fields: {e}"));
                Err(FfiStatus::Decode)
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_decodes_against_its_dictionary_through_a_handle() {
        let dict = b"the quick brown fox jumps over the lazy dog ".repeat(40);
        let plain = b"the lazy dog and the quick brown fox ".repeat(30);
        let mut z = lucene_zstd::Compressor::new(3).unwrap();
        let prepared = lucene_zstd::EncoderDictionary::new(&dict, 3).unwrap();
        let mut frame = Vec::new();
        z.compress_with_prepared(&prepared, &plain, &mut frame)
            .unwrap();

        let mut handle = 0u64;
        // SAFETY: valid buffers.
        let status = unsafe { ffi_zstd_dictionary_open(dict.as_ptr(), dict.len(), &mut handle) };
        assert_eq!(status, 0);
        let mut out = vec![0u8; plain.len()];
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_zstd_paged_decompress(
                handle,
                frame.as_ptr(),
                frame.len(),
                out.as_mut_ptr(),
                out.len(),
            )
        };
        assert_eq!(status, 0);
        assert_eq!(out, plain);
        // A wrong length is a decode error, not a short read.
        let mut short = vec![0u8; plain.len() - 1];
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_zstd_paged_decompress(
                handle,
                frame.as_ptr(),
                frame.len(),
                short.as_mut_ptr(),
                short.len(),
            )
        };
        assert_eq!(status, FfiStatus::Decode.code());
        assert_eq!(ffi_zstd_dictionary_close(handle), 0);
        assert_eq!(
            ffi_zstd_dictionary_close(handle),
            FfiStatus::InvalidHandle.code()
        );
    }
}
