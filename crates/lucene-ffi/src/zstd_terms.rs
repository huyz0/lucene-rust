//! Zstd decoding for the plugin's Java reader of `ZstdTerms104` terms
//! dictionaries ([`lucene_codecs::per_field_postings::Lucene104PostingsFormat::rust_zstd`]):
//! a field's dictionary is prepared once behind a handle, and each zstd term
//! block's bare frame (`lucene_zstd::BareDecompressor`) is decoded against it.

use std::sync::OnceLock;

use lucene_engine::handle::RegistryTag;

use crate::error::{set_last_error, FfiStatus};
use crate::zstd_shared::{self as shared, Registry};

/// The prepared dictionaries, by handle.
fn dictionaries() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| shared::new_registry(RegistryTag::ZstdTermsDictionary))
}

/// Prepares the `len` dictionary bytes at `dict` and writes its handle to
/// `*out_handle`; release it with [`ffi_zstd_terms_dictionary_close`].
///
/// # Safety
/// `dict` must be valid for `len` bytes and `out_handle` for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_zstd_terms_dictionary_open(
    dict: *const u8,
    len: usize,
    out_handle: *mut u64,
) -> i32 {
    // SAFETY: caller contract.
    unsafe { shared::open(dictionaries(), dict, len, out_handle) }
}

/// Releases a handle from [`ffi_zstd_terms_dictionary_open`].
#[no_mangle]
pub extern "C" fn ffi_zstd_terms_dictionary_close(handle: u64) -> i32 {
    shared::close(dictionaries(), handle)
}

/// Decodes the bare zstd frame `src` (`src_len` bytes) of a zstd term
/// block, compressed against the dictionary `handle` (0: none), into exactly
/// `dst_len` bytes at `dst`.
///
/// # Safety
/// `src` must be valid for `src_len` bytes and `dst` for `dst_len` writes.
#[no_mangle]
pub unsafe extern "C" fn ffi_zstd_terms_decompress(
    handle: u64,
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_len: usize,
) -> i32 {
    // SAFETY: caller contract.
    unsafe {
        shared::decompress(
            dictionaries(),
            handle,
            src,
            src_len,
            dst,
            dst_len,
            |src, dict, dst| {
                lucene_zstd::with_bare_decompressor(|z| z.decompress_exact(src, dict, dst)).map_err(
                    |e| {
                        set_last_error(format!("zstd terms block: {e}"));
                        FfiStatus::Decode
                    },
                )
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_term_block_decodes_against_its_dictionary_through_a_handle() {
        let dict = b"/api/v1/items/ /api/v2/users/ ".repeat(60);
        let plain = b"/api/v1/items/17 /api/v2/users/9".repeat(20);
        let prepared = lucene_zstd::EncoderDictionary::new(&dict, 3).unwrap();
        let mut frame = Vec::new();
        lucene_zstd::BareCompressor::new(3, Some(&prepared))
            .unwrap()
            .compress(&plain, &mut frame)
            .unwrap();

        let mut handle = 0u64;
        // SAFETY: valid buffers.
        let status =
            unsafe { ffi_zstd_terms_dictionary_open(dict.as_ptr(), dict.len(), &mut handle) };
        assert_eq!(status, 0);
        let mut out = vec![0u8; plain.len()];
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_zstd_terms_decompress(
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
        let mut long = vec![0u8; plain.len() + 1];
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_zstd_terms_decompress(
                handle,
                frame.as_ptr(),
                frame.len(),
                long.as_mut_ptr(),
                long.len(),
            )
        };
        assert_eq!(status, FfiStatus::Decode.code());
        assert_eq!(ffi_zstd_terms_dictionary_close(handle), 0);
        assert_eq!(
            ffi_zstd_terms_dictionary_close(handle),
            FfiStatus::InvalidHandle.code()
        );
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_zstd_terms_decompress(
                handle,
                frame.as_ptr(),
                frame.len(),
                out.as_mut_ptr(),
                out.len(),
            )
        };
        assert_eq!(status, FfiStatus::InvalidHandle.code());
    }
}
