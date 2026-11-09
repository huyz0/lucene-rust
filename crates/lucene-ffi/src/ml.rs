//! Local model inference for the plugin's ml-commons port
//! (`org.lucenerust.opensearch.ml`), over [`lucene_ml`].
//!
//! The plugin loads the ONNX Runtime library it ships once
//! ([`ffi_ml_runtime_init`]), opens each deployed model behind a handle
//! ([`ffi_ml_text_embedding_open`]) and asks it for vectors
//! ([`ffi_ml_text_embedding_embed_alloc`]).
//!
//! Texts go in as a sequence of `u32` little-endian byte length + UTF-8
//! bytes. Vectors come back, one per text in order, as `u32` rank, `rank`
//! `u32` dimensions, then the `f32` values, all little endian.

use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};

use lucene_ml::text_embedding::{Config, Pooling, Tensor, TextEmbeddingModel};
use lucene_ml::MlError;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::ffm_bridge::{clear_out, give};
use crate::handle::{RegistryTag, SlotMap};
use crate::raw::{bytes_from_raw, str_from_raw};
use crate::registry::{lock_recovering, read_recovering};

fn models() -> &'static RwLock<SlotMap<Arc<TextEmbeddingModel>>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<Arc<TextEmbeddingModel>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::MlModel)))
}

fn ml_failure(e: MlError) -> FfiStatus {
    let status = match e {
        MlError::InvalidArgument(_) => FfiStatus::InvalidArgument,
        MlError::Runtime(_) | MlError::Tokenizer(_) | MlError::Model(_) => FfiStatus::Search,
    };
    set_last_error(e.to_string());
    status
}

/// Loads the ONNX Runtime library at `path` (`path_len` UTF-8 bytes), with
/// `threads` intra-op threads shared by every model (0: one per core). Later
/// calls keep the first library.
///
/// # Safety
/// `path` must be valid for `path_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_runtime_init(path: *const u8, path_len: usize, threads: u32) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let path = unsafe { str_from_raw(path.cast(), path_len) }?;
        let threads = usize::try_from(threads).map_err(|_| FfiStatus::InvalidArgument)?;
        lucene_ml::runtime::init(Path::new(path), threads).map_err(ml_failure)
    })
}

/// Opens the ONNX text-embedding model at `model` with the `tokenizer.json`
/// bytes `tokenizer`, the ml-commons `model_type` and `pooling_mode` names,
/// and `normalize` (0 or 1); writes its handle to `*out_handle`. Close it
/// with [`ffi_ml_model_close`].
///
/// # Safety
/// Each pointer must be valid for its length; `out_handle` for one write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_ml_text_embedding_open(
    model: *const u8,
    model_len: usize,
    tokenizer: *const u8,
    tokenizer_len: usize,
    model_type: *const u8,
    model_type_len: usize,
    pooling: *const u8,
    pooling_len: usize,
    normalize: u8,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract, for each of the four.
        let (model, tokenizer, model_type, pooling) = unsafe {
            (
                str_from_raw(model.cast(), model_len)?,
                bytes_from_raw(tokenizer, tokenizer_len)?,
                str_from_raw(model_type.cast(), model_type_len)?,
                str_from_raw(pooling.cast(), pooling_len)?,
            )
        };
        let config = Config {
            model_type: model_type.to_string(),
            pooling: Pooling::from_name(pooling).map_err(ml_failure)?,
            normalize: normalize != 0,
        };
        let loaded =
            TextEmbeddingModel::load(Path::new(model), tokenizer, &config).map_err(ml_failure)?;
        let handle = lock_recovering(models()).insert_checked(Arc::new(loaded))?;
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
}

/// Releases a model handle. Embeddings running on it finish first: they hold
/// their own reference.
#[no_mangle]
pub extern "C" fn ffi_ml_model_close(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(models())
            .remove(handle)
            .map(drop)
            .ok_or(FfiStatus::InvalidHandle)
    })
}

/// Embeds the texts in `texts` (`len` bytes, see the module doc) with the
/// model `handle`; the vectors go to `*out_ptr`/`*out_len`, a buffer to free
/// with [`crate::ffm_bridge::ffi_jvm_free_bytes`].
///
/// # Safety
/// `texts` must be valid for `len` bytes; `out_ptr` and `out_len` for one
/// write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_text_embedding_embed_alloc(
    handle: u64,
    texts: *const u8,
    len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len) }?;
        // SAFETY: caller contract.
        let bytes = unsafe { bytes_from_raw(texts, len) }?;
        let texts = decode_texts(bytes)?;
        let model = read_recovering(models())
            .get(handle)
            .cloned()
            .ok_or(FfiStatus::InvalidHandle)?;
        let vectors = model.embed(&texts).map_err(ml_failure)?;
        let encoded = encode_tensors(&vectors)?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

fn decode_texts(mut bytes: &[u8]) -> Result<Vec<&str>, FfiStatus> {
    let mut texts = Vec::new();
    while !bytes.is_empty() {
        let (len, rest) = bytes.split_first_chunk::<4>().ok_or(FfiStatus::Decode)?;
        let len = usize::try_from(u32::from_le_bytes(*len)).map_err(|_| FfiStatus::Decode)?;
        if rest.len() < len {
            return Err(FfiStatus::Decode);
        }
        let (text, rest) = rest.split_at(len);
        texts.push(std::str::from_utf8(text).map_err(|_| FfiStatus::InvalidUtf8)?);
        bytes = rest;
    }
    Ok(texts)
}

fn encode_tensors(tensors: &[Tensor]) -> Result<Vec<u8>, FfiStatus> {
    let mut out = Vec::new();
    let u32_of = |n: usize| u32::try_from(n).map_err(|_| FfiStatus::InvalidArgument);
    for t in tensors {
        out.extend_from_slice(&u32_of(t.shape.len())?.to_le_bytes());
        for &d in &t.shape {
            out.extend_from_slice(&u32_of(d)?.to_le_bytes());
        }
        for v in &t.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(items: &[&str]) -> Vec<u8> {
        let mut out = Vec::new();
        for t in items {
            out.extend_from_slice(&(t.len() as u32).to_le_bytes());
            out.extend_from_slice(t.as_bytes());
        }
        out
    }

    #[test]
    fn decodes_texts_and_refuses_truncated_ones() {
        let bytes = texts(&["a", "", "caf\u{e9}"]);
        assert_eq!(decode_texts(&bytes).unwrap(), vec!["a", "", "caf\u{e9}"]);
        assert_eq!(decode_texts(&bytes[..2]), Err(FfiStatus::Decode));
        assert_eq!(decode_texts(&bytes[..4]), Err(FfiStatus::Decode));
        let bad = [1, 0, 0, 0, 0xff];
        assert_eq!(decode_texts(&bad), Err(FfiStatus::InvalidUtf8));
        assert!(decode_texts(&[]).unwrap().is_empty());
    }

    #[test]
    fn encodes_rank_dims_and_values() {
        let t = Tensor {
            shape: vec![1, 2],
            data: vec![0.5, -1.0],
        };
        let bytes = encode_tensors(&[t]).unwrap();
        let mut want = Vec::new();
        for n in [2u32, 1, 2] {
            want.extend_from_slice(&n.to_le_bytes());
        }
        want.extend_from_slice(&0.5f32.to_le_bytes());
        want.extend_from_slice(&(-1.0f32).to_le_bytes());
        assert_eq!(bytes, want);
    }

    #[test]
    fn refuses_bad_handles_and_arguments() {
        assert_eq!(ffi_ml_model_close(0), FfiStatus::InvalidHandle.code());
        let mut ptr = std::ptr::null_mut();
        let mut len = 0usize;
        let input = texts(&["x"]);
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_ml_text_embedding_embed_alloc(0, input.as_ptr(), input.len(), &mut ptr, &mut len)
        };
        assert_eq!(status, FfiStatus::InvalidHandle.code());
        let mut handle = 0u64;
        let s = "x";
        // SAFETY: valid buffers; `pooling` is not a mode.
        let status = unsafe {
            ffi_ml_text_embedding_open(
                s.as_ptr(),
                1,
                s.as_ptr(),
                1,
                s.as_ptr(),
                1,
                s.as_ptr(),
                1,
                1,
                &mut handle,
            )
        };
        assert_eq!(status, FfiStatus::InvalidArgument.code());
        // SAFETY: a null out-pointer is refused before anything is read.
        let status = unsafe {
            ffi_ml_text_embedding_open(
                s.as_ptr(),
                1,
                s.as_ptr(),
                1,
                s.as_ptr(),
                1,
                s.as_ptr(),
                1,
                1,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, FfiStatus::NullPointer.code());
    }

    /// The whole path, when the ONNX Runtime library is available
    /// (`ORT_DYLIB_PATH`).
    #[test]
    fn embeds_through_a_handle() {
        let Some(lib) = std::env::var_os("ORT_DYLIB_PATH") else {
            eprintln!("ORT_DYLIB_PATH is not set: skipping");
            return;
        };
        let lib = lib.into_string().unwrap();
        // SAFETY: valid buffers.
        assert_eq!(unsafe { ffi_ml_runtime_init(lib.as_ptr(), lib.len(), 1) }, 0);
        let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/ml");
        let model = format!("{fixtures}/tiny-bert.onnx");
        let tokenizer = std::fs::read(format!("{fixtures}/tokenizer.json")).unwrap();
        let (model_type, pooling) = ("bert", "mean");
        let mut handle = 0u64;
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_ml_text_embedding_open(
                model.as_ptr(),
                model.len(),
                tokenizer.as_ptr(),
                tokenizer.len(),
                model_type.as_ptr(),
                model_type.len(),
                pooling.as_ptr(),
                pooling.len(),
                1,
                &mut handle,
            )
        };
        assert_eq!(status, 0);
        let input = texts(&["hello world", "neural search"]);
        let (mut ptr, mut len) = (std::ptr::null_mut(), 0usize);
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_ml_text_embedding_embed_alloc(handle, input.as_ptr(), input.len(), &mut ptr, &mut len)
        };
        assert_eq!(status, 0);
        // Two vectors of [32]: (4 + 4 + 32 * 4) bytes each.
        assert_eq!(len, 2 * (8 + 128));
        // SAFETY: the pair this module wrote.
        unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
        // An empty text has no tokens, which the model refuses, as on stock.
        let empty = texts(&[""]);
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_ml_text_embedding_embed_alloc(handle, empty.as_ptr(), empty.len(), &mut ptr, &mut len)
        };
        assert_eq!(status, FfiStatus::Search.code());
        assert_eq!(ffi_ml_model_close(handle), 0);
    }
}
