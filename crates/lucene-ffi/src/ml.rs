//! Local model inference for the plugin's ml-commons port
//! (`org.lucenerust.opensearch.ml`), over [`lucene_ml`].
//!
//! The plugin loads the ONNX Runtime library it ships once
//! ([`ffi_ml_runtime_init`]), opens each deployed model behind a handle
//! ([`ffi_ml_text_embedding_open`]) and asks it for vectors
//! ([`ffi_ml_text_embedding_embed_alloc`]); sparse tokenize models
//! ([`ffi_ml_sparse_tokenize_open`]) answer with token weights
//! ([`ffi_ml_sparse_tokenize_alloc`]), as do sparse encoding models
//! ([`ffi_ml_sparse_encoding_open`], [`ffi_ml_sparse_encoding_alloc`]);
//! cross-encoders ([`ffi_ml_text_similarity_open`]) score documents against
//! a query ([`ffi_ml_text_similarity_alloc`]).
//!
//! TorchScript models run on libtorch, loaded once ([`ffi_ml_torch_init`]);
//! [`ffi_ml_torch_open`] opens one for any of those functions. Sparse
//! encoding and similarity handles answer through the same calls whatever
//! the format; a TorchScript text-embedding model answers with typed outputs
//! ([`ffi_ml_torch_text_embedding_alloc`]).
//!
//! Texts go in as a sequence of `u32` little-endian byte length + UTF-8
//! bytes. Vectors come back, one per text in order, as `u32` rank, `rank`
//! `u32` dimensions, then the `f32` values; token weights, one list per text
//! in order, as `u32` count, then per token `u32` id, `u32` byte length, the
//! UTF-8 token and its `f32` weight; all little endian. Sparse encoding and
//! similarity results carry each model output: per text a `u32` output
//! count, then per output its weights (after a `u32` byte length and the
//! UTF-8 output name) or its tensor. Typed outputs are, per text, a `u32`
//! output count, then per output a `u32` byte length and the UTF-8 name, the
//! `u32` dict number (0 outside a dict), a `u8` data type (0 `FLOAT32`, 1
//! `INT64`), the `u32` rank and dimensions, and the values (`f32` or `i64`).

use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};

use lucene_ml::sparse_encoding::{SparseEncodingModel, SparseOutput};
use lucene_ml::sparse_tokenize::{SparseTokenizer, Weight};
use lucene_ml::text_embedding::{
    Config, Pooling, Tensor, TextEmbeddingModel, TorchTextEmbeddingModel,
};
use lucene_ml::text_similarity::TextSimilarityModel;
use lucene_ml::torchscript::{Data, TorchOutput};
use lucene_ml::MlError;

use crate::ml_torch;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::ffm_bridge::{clear_out, give};
use crate::handle::{RegistryTag, SlotMap};
use crate::raw::{bytes_from_raw, str_from_raw};
use crate::registry::{lock_recovering, read_recovering};

/// A deployed model behind a handle.
enum Model {
    TextEmbedding(TextEmbeddingModel),
    SparseTokenize(SparseTokenizer),
    SparseEncoding(SparseEncodingModel),
    TextSimilarity(TextSimilarityModel),
    TorchTextEmbedding(TorchTextEmbeddingModel),
}

fn models() -> &'static RwLock<SlotMap<Arc<Model>>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<Arc<Model>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::MlModel)))
}

fn model(handle: u64) -> Result<Arc<Model>, FfiStatus> {
    read_recovering(models())
        .get(handle)
        .cloned()
        .ok_or(FfiStatus::InvalidHandle)
}

fn wrong_kind(what: &str) -> FfiStatus {
    set_last_error(format!("the model is not a {what} model"));
    FfiStatus::InvalidArgument
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
pub unsafe extern "C" fn ffi_ml_runtime_init(
    path: *const u8,
    path_len: usize,
    threads: u32,
) -> i32 {
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
        let handle =
            lock_recovering(models()).insert_checked(Arc::new(Model::TextEmbedding(loaded)))?;
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
        let model = model(handle)?;
        let Model::TextEmbedding(model) = &*model else {
            return Err(wrong_kind("text embedding"));
        };
        let vectors = model.embed(&texts).map_err(ml_failure)?;
        let encoded = encode_tensors(&vectors)?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

/// Opens a sparse tokenize model from its `tokenizer.json` bytes and, when
/// `idf` is not null, its `idf.json` bytes; writes its handle to
/// `*out_handle`. Close it with [`ffi_ml_model_close`].
///
/// # Safety
/// Each non-null pointer must be valid for its length; `out_handle` for one
/// write.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_sparse_tokenize_open(
    tokenizer: *const u8,
    tokenizer_len: usize,
    idf: *const u8,
    idf_len: usize,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let tokenizer = unsafe { bytes_from_raw(tokenizer, tokenizer_len) }?;
        let idf = if idf.is_null() {
            None
        } else {
            // SAFETY: caller contract.
            Some(unsafe { bytes_from_raw(idf, idf_len) }?)
        };
        let loaded = SparseTokenizer::load(tokenizer, idf).map_err(ml_failure)?;
        let handle =
            lock_recovering(models()).insert_checked(Arc::new(Model::SparseTokenize(loaded)))?;
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
}

/// The token weights of the texts in `texts` (see the module doc) from the
/// sparse tokenize model `handle`, to `*out_ptr`/`*out_len`, a buffer to
/// free with [`crate::ffm_bridge::ffi_jvm_free_bytes`].
///
/// # Safety
/// `texts` must be valid for `len` bytes; `out_ptr` and `out_len` for one
/// write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_sparse_tokenize_alloc(
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
        let model = model(handle)?;
        let Model::SparseTokenize(model) = &*model else {
            return Err(wrong_kind("sparse tokenize"));
        };
        let mut weights = Vec::new();
        for t in texts {
            weights.push(model.encode(t).map_err(ml_failure)?);
        }
        let encoded = encode_weights(&weights)?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

/// Opens the ONNX model at `model` with the `tokenizer.json` bytes
/// `tokenizer` as `open` builds it, and registers it as `wrap` names it.
///
/// # Safety
/// Each pointer must be valid for its length; `out_handle` for one write.
unsafe fn open_onnx<M>(
    model: *const u8,
    model_len: usize,
    tokenizer: *const u8,
    tokenizer_len: usize,
    out_handle: *mut u64,
    open: impl FnOnce(&Path, &[u8]) -> lucene_ml::Result<M>,
    wrap: impl FnOnce(M) -> Model,
) -> Result<(), FfiStatus> {
    if out_handle.is_null() {
        return Err(FfiStatus::NullPointer);
    }
    // SAFETY: caller contract, for both.
    let (model, tokenizer) = unsafe {
        (
            str_from_raw(model.cast(), model_len)?,
            bytes_from_raw(tokenizer, tokenizer_len)?,
        )
    };
    let loaded = open(Path::new(model), tokenizer).map_err(ml_failure)?;
    let handle = lock_recovering(models()).insert_checked(Arc::new(wrap(loaded)))?;
    // SAFETY: caller contract.
    unsafe { *out_handle = handle };
    Ok(())
}

/// Opens the ONNX sparse encoding model at `model` with the `tokenizer.json`
/// bytes `tokenizer`; writes its handle to `*out_handle`. Close it with
/// [`ffi_ml_model_close`].
///
/// # Safety
/// Each pointer must be valid for its length; `out_handle` for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_sparse_encoding_open(
    model: *const u8,
    model_len: usize,
    tokenizer: *const u8,
    tokenizer_len: usize,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe {
            open_onnx(
                model,
                model_len,
                tokenizer,
                tokenizer_len,
                out_handle,
                SparseEncodingModel::load,
                Model::SparseEncoding,
            )
        }
    })
}

/// The sparse vectors of the texts in `texts` (see the module doc) from the
/// sparse encoding model `handle`, keyed by token id when `by_id` is 1, to
/// `*out_ptr`/`*out_len`, a buffer to free with
/// [`crate::ffm_bridge::ffi_jvm_free_bytes`].
///
/// # Safety
/// `texts` must be valid for `len` bytes; `out_ptr` and `out_len` for one
/// write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_sparse_encoding_alloc(
    handle: u64,
    texts: *const u8,
    len: usize,
    by_id: u8,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len) }?;
        // SAFETY: caller contract.
        let bytes = unsafe { bytes_from_raw(texts, len) }?;
        let texts = decode_texts(bytes)?;
        let model = model(handle)?;
        let Model::SparseEncoding(model) = &*model else {
            return Err(wrong_kind("sparse encoding"));
        };
        let mut out = Vec::new();
        for t in texts {
            let outputs = model.encode(t, by_id != 0).map_err(ml_failure)?;
            encode_sparse_outputs(&outputs, &mut out)?;
        }
        // SAFETY: caller contract.
        unsafe { give(out, out_ptr, out_len) };
        Ok(())
    })
}

/// Opens the ONNX cross-encoder at `model` with the `tokenizer.json` bytes
/// `tokenizer`; writes its handle to `*out_handle`. Close it with
/// [`ffi_ml_model_close`].
///
/// # Safety
/// Each pointer must be valid for its length; `out_handle` for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_text_similarity_open(
    model: *const u8,
    model_len: usize,
    tokenizer: *const u8,
    tokenizer_len: usize,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe {
            open_onnx(
                model,
                model_len,
                tokenizer,
                tokenizer_len,
                out_handle,
                TextSimilarityModel::load,
                Model::TextSimilarity,
            )
        }
    })
}

/// Scores each document in `texts` (see the module doc) against `query`
/// (`query_len` UTF-8 bytes) with the cross-encoder `handle`; per document
/// its output tensors go to `*out_ptr`/`*out_len`, a buffer to free with
/// [`crate::ffm_bridge::ffi_jvm_free_bytes`].
///
/// # Safety
/// `query` and `texts` must be valid for their lengths; `out_ptr` and
/// `out_len` for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_text_similarity_alloc(
    handle: u64,
    query: *const u8,
    query_len: usize,
    texts: *const u8,
    len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len) }?;
        // SAFETY: caller contract, for both.
        let (query, bytes) = unsafe {
            (
                str_from_raw(query.cast(), query_len)?,
                bytes_from_raw(texts, len)?,
            )
        };
        let texts = decode_texts(bytes)?;
        let model = model(handle)?;
        let Model::TextSimilarity(model) = &*model else {
            return Err(wrong_kind("text similarity"));
        };
        let mut out = Vec::new();
        for t in texts {
            let tensors = model.score(query, t).map_err(ml_failure)?;
            out.extend_from_slice(&u32_of(tensors.len())?.to_le_bytes());
            out.extend_from_slice(&encode_tensors(&tensors)?);
        }
        // SAFETY: caller contract.
        unsafe { give(out, out_ptr, out_len) };
        Ok(())
    })
}

/// Loads libtorch from the directory `dir` (`dir_len` UTF-8 bytes) and the
/// first TorchScript shim in `shims` (paths in the text encoding of the
/// module doc) that runs against it, with `threads` intra-op threads shared
/// by every model (0: one per physical core). Later calls keep the first.
///
/// # Safety
/// `dir` and `shims` must be valid for their lengths.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_torch_init(
    dir: *const u8,
    dir_len: usize,
    shims: *const u8,
    shims_len: usize,
    threads: u32,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract, for both.
        let (dir, shims) = unsafe {
            (
                str_from_raw(dir.cast(), dir_len)?,
                bytes_from_raw(shims, shims_len)?,
            )
        };
        let shims: Vec<&Path> = decode_texts(shims)?.into_iter().map(Path::new).collect();
        ml_torch::init(Path::new(dir), &shims, threads).map_err(ml_failure)
    })
}

/// The ml-commons function a TorchScript model is opened for.
const TORCH_SENTENCE_TRANSFORMERS: u8 = 0;
const TORCH_HUGGING_FACE: u8 = 1;
const TORCH_SPARSE_ENCODING: u8 = 2;
const TORCH_TEXT_SIMILARITY: u8 = 3;

/// Opens the TorchScript model at `model` with the `tokenizer.json` bytes
/// `tokenizer` for `function`: 0 a `sentence_transformers` text embedding
/// model, 1 a `huggingface_transformers` one (pooled by the `pooling` mode
/// name and normalized when `normalize` is 1), 2 sparse encoding, 3 a
/// cross-encoder. Writes its handle to `*out_handle`; close it with
/// [`ffi_ml_model_close`]. [`ffi_ml_torch_init`] must have run.
///
/// # Safety
/// Each pointer must be valid for its length; `out_handle` for one write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_ml_torch_open(
    function: u8,
    model: *const u8,
    model_len: usize,
    tokenizer: *const u8,
    tokenizer_len: usize,
    pooling: *const u8,
    pooling_len: usize,
    normalize: u8,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract, for each of the three.
        let (model, tokenizer, pooling) = unsafe {
            (
                str_from_raw(model.cast(), model_len)?,
                bytes_from_raw(tokenizer, tokenizer_len)?,
                str_from_raw(pooling.cast(), pooling_len)?,
            )
        };
        let opened = match function {
            TORCH_SENTENCE_TRANSFORMERS..=TORCH_TEXT_SIMILARITY => {
                Box::new(ml_torch::Module::load(Path::new(model)).map_err(ml_failure)?)
            }
            _ => {
                set_last_error(format!("unknown TorchScript model function {function}"));
                return Err(FfiStatus::InvalidArgument);
            }
        };
        let loaded = match function {
            TORCH_SENTENCE_TRANSFORMERS => {
                TorchTextEmbeddingModel::sentence_transformers(opened, tokenizer)
                    .map(Model::TorchTextEmbedding)
            }
            TORCH_HUGGING_FACE => {
                let pooling = Pooling::from_name(pooling).map_err(ml_failure)?;
                TorchTextEmbeddingModel::hugging_face(opened, tokenizer, pooling, normalize != 0)
                    .map(Model::TorchTextEmbedding)
            }
            TORCH_SPARSE_ENCODING => {
                SparseEncodingModel::torchscript(opened, tokenizer).map(Model::SparseEncoding)
            }
            _ => TextSimilarityModel::torchscript(opened, tokenizer).map(Model::TextSimilarity),
        }
        .map_err(ml_failure)?;
        let handle = lock_recovering(models()).insert_checked(Arc::new(loaded))?;
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
}

/// The typed outputs (see the module doc) of the TorchScript text-embedding
/// model `handle` for each text in `texts`, to `*out_ptr`/`*out_len`, a
/// buffer to free with [`crate::ffm_bridge::ffi_jvm_free_bytes`].
///
/// # Safety
/// `texts` must be valid for `len` bytes; `out_ptr` and `out_len` for one
/// write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_torch_text_embedding_alloc(
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
        let model = model(handle)?;
        let Model::TorchTextEmbedding(model) = &*model else {
            return Err(wrong_kind("TorchScript text embedding"));
        };
        let mut out = Vec::new();
        for t in texts {
            let outputs = model.embed(t).map_err(ml_failure)?;
            encode_typed_outputs(&outputs, &mut out)?;
        }
        // SAFETY: caller contract.
        unsafe { give(out, out_ptr, out_len) };
        Ok(())
    })
}

fn encode_typed_outputs(outputs: &[TorchOutput], out: &mut Vec<u8>) -> Result<(), FfiStatus> {
    out.extend_from_slice(&u32_of(outputs.len())?.to_le_bytes());
    for o in outputs {
        out.extend_from_slice(&u32_of(o.name.len())?.to_le_bytes());
        out.extend_from_slice(o.name.as_bytes());
        out.extend_from_slice(&o.group.to_le_bytes());
        out.push(match o.data {
            Data::F32(_) => 0,
            Data::I64(_) => 1,
        });
        out.extend_from_slice(&u32_of(o.shape.len())?.to_le_bytes());
        for &d in &o.shape {
            out.extend_from_slice(&u32_of(d)?.to_le_bytes());
        }
        match &o.data {
            Data::F32(v) => v
                .iter()
                .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
            Data::I64(v) => v
                .iter()
                .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        }
    }
    Ok(())
}

fn u32_of(n: usize) -> Result<u32, FfiStatus> {
    u32::try_from(n).map_err(|_| FfiStatus::InvalidArgument)
}

fn encode_weights(texts: &[Vec<Weight>]) -> Result<Vec<u8>, FfiStatus> {
    let mut out = Vec::new();
    for weights in texts {
        encode_weight_list(weights, &mut out)?;
    }
    Ok(out)
}

fn encode_weight_list(weights: &[Weight], out: &mut Vec<u8>) -> Result<(), FfiStatus> {
    out.extend_from_slice(&u32_of(weights.len())?.to_le_bytes());
    for w in weights {
        out.extend_from_slice(&w.id.to_le_bytes());
        out.extend_from_slice(&u32_of(w.token.len())?.to_le_bytes());
        out.extend_from_slice(w.token.as_bytes());
        out.extend_from_slice(&w.weight.to_le_bytes());
    }
    Ok(())
}

fn encode_sparse_outputs(outputs: &[SparseOutput], out: &mut Vec<u8>) -> Result<(), FfiStatus> {
    out.extend_from_slice(&u32_of(outputs.len())?.to_le_bytes());
    for o in outputs {
        out.extend_from_slice(&u32_of(o.name.len())?.to_le_bytes());
        out.extend_from_slice(o.name.as_bytes());
        encode_weight_list(&o.weights, out)?;
    }
    Ok(())
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
    /// (`LUCENE_ML_ONNXRUNTIME`).
    #[test]
    fn embeds_through_a_handle() {
        let Some(lib) = std::env::var_os("LUCENE_ML_ONNXRUNTIME") else {
            eprintln!("LUCENE_ML_ONNXRUNTIME is not set: skipping");
            return;
        };
        let lib = lib.into_string().unwrap();
        // SAFETY: valid buffers.
        assert_eq!(
            unsafe { ffi_ml_runtime_init(lib.as_ptr(), lib.len(), 1) },
            0
        );
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
            ffi_ml_text_embedding_embed_alloc(
                handle,
                input.as_ptr(),
                input.len(),
                &mut ptr,
                &mut len,
            )
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
            ffi_ml_text_embedding_embed_alloc(
                handle,
                empty.as_ptr(),
                empty.len(),
                &mut ptr,
                &mut len,
            )
        };
        assert_eq!(status, FfiStatus::Search.code());
        assert_eq!(ffi_ml_model_close(handle), 0);
    }

    #[test]
    fn sparse_tokenize_weighs_tokens_and_refuses_the_wrong_kind() {
        let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/ml");
        let tokenizer = std::fs::read(format!("{fixtures}/tokenizer.json")).unwrap();
        let idf = br#"{"search": 3.0}"#;
        let mut handle = 0u64;
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_ml_sparse_tokenize_open(
                tokenizer.as_ptr(),
                tokenizer.len(),
                idf.as_ptr(),
                idf.len(),
                &mut handle,
            )
        };
        assert_eq!(status, 0);
        let input = texts(&["search search", ""]);
        let (mut ptr, mut len) = (std::ptr::null_mut(), 0usize);
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_ml_sparse_tokenize_alloc(handle, input.as_ptr(), input.len(), &mut ptr, &mut len)
        };
        assert_eq!(status, 0);
        // SAFETY: the pair this module wrote.
        let out = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
        // SAFETY: the pair this module wrote.
        unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
        let u32_at = |i: usize| u32::from_le_bytes(out[i..i + 4].try_into().unwrap());
        assert_eq!(u32_at(0), 1, "one distinct token besides the special ones");
        let token_len = u32_at(8) as usize;
        assert_eq!(&out[12..12 + token_len], b"search");
        let weight = f32::from_le_bytes(out[12 + token_len..16 + token_len].try_into().unwrap());
        assert_eq!(weight, 3.0);
        assert_eq!(u32_at(16 + token_len), 0, "the empty text has no tokens");
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_ml_text_embedding_embed_alloc(
                handle,
                input.as_ptr(),
                input.len(),
                &mut ptr,
                &mut len,
            )
        };
        assert_eq!(status, FfiStatus::InvalidArgument.code());
        assert_eq!(ffi_ml_model_close(handle), 0);
        // SAFETY: a null idf means none; a null out-pointer is refused.
        let status = unsafe {
            ffi_ml_sparse_tokenize_open(
                tokenizer.as_ptr(),
                tokenizer.len(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, FfiStatus::NullPointer.code());
    }

    /// Sparse encoding and cross-encoder handles, when the ONNX Runtime
    /// library is available (`LUCENE_ML_ONNXRUNTIME`).
    #[test]
    fn sparse_encoding_and_similarity_through_handles() {
        let Some(lib) = std::env::var_os("LUCENE_ML_ONNXRUNTIME") else {
            eprintln!("LUCENE_ML_ONNXRUNTIME is not set: skipping");
            return;
        };
        let lib = lib.into_string().unwrap();
        // SAFETY: valid buffers.
        assert_eq!(
            unsafe { ffi_ml_runtime_init(lib.as_ptr(), lib.len(), 1) },
            0
        );
        let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/ml");
        let tokenizer = std::fs::read(format!("{fixtures}/tokenizer.json")).unwrap();
        let open = |name: &str,
                    f: unsafe extern "C" fn(
            *const u8,
            usize,
            *const u8,
            usize,
            *mut u64,
        ) -> i32| {
            let model = format!("{fixtures}/{name}.onnx");
            let mut handle = 0u64;
            // SAFETY: valid buffers.
            let status = unsafe {
                f(
                    model.as_ptr(),
                    model.len(),
                    tokenizer.as_ptr(),
                    tokenizer.len(),
                    &mut handle,
                )
            };
            assert_eq!(status, 0, "{name}");
            handle
        };
        let take = |ptr: *mut u8, len: usize| {
            // SAFETY: the pair this module wrote.
            let out = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
            // SAFETY: the pair this module wrote.
            unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
            out
        };
        let input = texts(&["hello world"]);
        let (mut ptr, mut len) = (std::ptr::null_mut(), 0usize);

        let sparse = open("tiny-sparse", ffi_ml_sparse_encoding_open);
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_ml_sparse_encoding_alloc(sparse, input.as_ptr(), input.len(), 1, &mut ptr, &mut len)
        };
        assert_eq!(status, 0);
        let out = take(ptr, len);
        let u32_at = |i: usize| u32::from_le_bytes(out[i..i + 4].try_into().unwrap());
        assert_eq!(u32_at(0), 1, "one output");
        assert_eq!(&out[8..8 + u32_at(4) as usize], b"output");
        assert!(u32_at(14) > 0, "some non-zero weights");
        assert_eq!(u32_at(22), 0, "by id: no token text");
        let empty = texts(&[""]);
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_ml_sparse_encoding_alloc(sparse, empty.as_ptr(), empty.len(), 0, &mut ptr, &mut len)
        };
        assert_eq!(status, FfiStatus::Search.code(), "no tokens, as on stock");

        let cross = open("tiny-cross-encoder", ffi_ml_text_similarity_open);
        let query = "a query";
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_ml_text_similarity_alloc(
                cross,
                query.as_ptr(),
                query.len(),
                input.as_ptr(),
                input.len(),
                &mut ptr,
                &mut len,
            )
        };
        assert_eq!(status, 0);
        let out = take(ptr, len);
        // One output of shape [1]: count, rank, dimension, value.
        assert_eq!(out.len(), 16);
        assert_eq!(&out[..12], &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
        // SAFETY: valid buffers; the handle is the wrong kind.
        let status = unsafe {
            ffi_ml_sparse_encoding_alloc(cross, input.as_ptr(), input.len(), 0, &mut ptr, &mut len)
        };
        assert_eq!(status, FfiStatus::InvalidArgument.code());
        // SAFETY: valid buffers; the handle is the wrong kind.
        let status = unsafe {
            ffi_ml_text_similarity_alloc(
                sparse,
                query.as_ptr(),
                query.len(),
                input.as_ptr(),
                input.len(),
                &mut ptr,
                &mut len,
            )
        };
        assert_eq!(status, FfiStatus::InvalidArgument.code());
        assert_eq!(ffi_ml_model_close(sparse), 0);
        assert_eq!(ffi_ml_model_close(cross), 0);
        // SAFETY: a null out-pointer is refused before anything is read.
        let status = unsafe {
            ffi_ml_text_similarity_open(query.as_ptr(), 1, query.as_ptr(), 1, std::ptr::null_mut())
        };
        assert_eq!(status, FfiStatus::NullPointer.code());
    }

    /// TorchScript models through handles, when libtorch and the shims are
    /// available (`LUCENE_ML_LIBTORCH`: a libtorch directory;
    /// `LUCENE_ML_TORCH_SHIMS`: the shim builds, `:`-separated).
    #[test]
    fn torchscript_models_through_handles() {
        let (Some(dir), Some(shims)) = (
            std::env::var_os("LUCENE_ML_LIBTORCH"),
            std::env::var_os("LUCENE_ML_TORCH_SHIMS"),
        ) else {
            eprintln!("LUCENE_ML_LIBTORCH or LUCENE_ML_TORCH_SHIMS is not set: skipping");
            return;
        };
        let dir = dir.into_string().unwrap();
        let shims = shims.into_string().unwrap();
        let shims = texts(&shims.split(':').collect::<Vec<_>>());
        // SAFETY: valid buffers.
        let status =
            unsafe { ffi_ml_torch_init(dir.as_ptr(), dir.len(), shims.as_ptr(), shims.len(), 2) };
        assert_eq!(status, 0, "{:?}", crate::error::last_error());
        let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/ml");
        let tokenizer = std::fs::read(format!("{fixtures}/tokenizer.json")).unwrap();
        let open = |name: &str, function: u8, pooling: &str| {
            let model = format!("{fixtures}/{name}.pt");
            let mut handle = 0u64;
            // SAFETY: valid buffers.
            let status = unsafe {
                ffi_ml_torch_open(
                    function,
                    model.as_ptr(),
                    model.len(),
                    tokenizer.as_ptr(),
                    tokenizer.len(),
                    pooling.as_ptr(),
                    pooling.len(),
                    1,
                    &mut handle,
                )
            };
            assert_eq!(status, 0, "{name}: {:?}", crate::error::last_error());
            handle
        };
        let take = |ptr: *mut u8, len: usize| {
            // SAFETY: the pair this module wrote.
            let out = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
            // SAFETY: the pair this module wrote.
            unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
            out
        };
        let input = texts(&["hello world"]);
        let (mut ptr, mut len) = (std::ptr::null_mut(), 0usize);
        let mut embed = |handle: u64| {
            // SAFETY: valid buffers.
            let status = unsafe {
                ffi_ml_torch_text_embedding_alloc(
                    handle,
                    input.as_ptr(),
                    input.len(),
                    &mut ptr,
                    &mut len,
                )
            };
            assert_eq!(status, 0, "{:?}", crate::error::last_error());
            take(ptr, len)
        };
        let st = embed(open("tiny-st", TORCH_SENTENCE_TRANSFORMERS, ""));
        let u32_at = |out: &[u8], i: usize| u32::from_le_bytes(out[i..i + 4].try_into().unwrap());
        assert_eq!(u32_at(&st, 0), 4, "the features dict");
        let hf = embed(open("tiny-hf", TORCH_HUGGING_FACE, "mean"));
        assert_eq!(u32_at(&hf, 0), 1);
        assert_eq!(&hf[8..26], b"sentence_embedding");
        // group 0, FLOAT32, rank 2: [1, 32].
        assert_eq!(&hf[26..31], &[0, 0, 0, 0, 0]);
        assert_eq!(
            (u32_at(&hf, 31), u32_at(&hf, 35), u32_at(&hf, 39)),
            (2, 1, 32)
        );
        assert_eq!(hf.len(), 43 + 32 * 4);

        let sparse = open("tiny-sparse-ts", TORCH_SPARSE_ENCODING, "");
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_ml_sparse_encoding_alloc(sparse, input.as_ptr(), input.len(), 0, &mut ptr, &mut len)
        };
        assert_eq!(status, 0);
        let out = take(ptr, len);
        assert_eq!(&out[8..14], b"output");
        let cross = open("tiny-cross-ts", TORCH_TEXT_SIMILARITY, "");
        let query = "a query";
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_ml_text_similarity_alloc(
                cross,
                query.as_ptr(),
                query.len(),
                input.as_ptr(),
                input.len(),
                &mut ptr,
                &mut len,
            )
        };
        assert_eq!(status, 0);
        assert_eq!(take(ptr, len).len(), 16);
        // The wrong kind, an unknown function and a missing file are refused.
        // SAFETY: valid buffers.
        let status = unsafe {
            ffi_ml_torch_text_embedding_alloc(
                sparse,
                input.as_ptr(),
                input.len(),
                &mut ptr,
                &mut len,
            )
        };
        assert_eq!(status, FfiStatus::InvalidArgument.code());
        let mut handle = 0u64;
        for (function, model) in [(9u8, "x.pt"), (TORCH_SPARSE_ENCODING, "missing.pt")] {
            // SAFETY: valid buffers.
            let status = unsafe {
                ffi_ml_torch_open(
                    function,
                    model.as_ptr(),
                    model.len(),
                    tokenizer.as_ptr(),
                    tokenizer.len(),
                    model.as_ptr(),
                    0,
                    0,
                    &mut handle,
                )
            };
            assert_ne!(status, 0);
        }
        assert_eq!(ffi_ml_model_close(sparse), 0);
        assert_eq!(ffi_ml_model_close(cross), 0);
    }
}
