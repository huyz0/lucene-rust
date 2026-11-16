#![cfg(test)]
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
        ffi_ml_text_embedding_embed_alloc(handle, input.as_ptr(), input.len(), &mut ptr, &mut len)
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

#[test]
fn hf_analyzer_streams_tokens_with_offsets_and_refuses_the_wrong_kind() {
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/ml");
    let tokenizer = std::fs::read(format!("{fixtures}/tokenizer.json")).unwrap();
    let idf = br#"{"search": 3.0}"#;
    let mut handle = 0u64;
    // SAFETY: valid buffers.
    let status = unsafe {
        ffi_ml_hf_analyzer_open(
            tokenizer.as_ptr(),
            tokenizer.len(),
            idf.as_ptr(),
            idf.len(),
            &mut handle,
        )
    };
    assert_eq!(status, 0);
    let text = "search search";
    let (mut ptr, mut len) = (std::ptr::null_mut(), 0usize);
    // SAFETY: valid buffers.
    let status =
        unsafe { ffi_ml_hf_analyze_alloc(handle, text.as_ptr(), text.len(), &mut ptr, &mut len) };
    assert_eq!(status, 0);
    // SAFETY: the pair this module wrote.
    let out = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
    // SAFETY: the pair this module wrote.
    unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
    let u32_at = |i: usize| u32::from_le_bytes(out[i..i + 4].try_into().unwrap());
    assert_eq!(u32_at(0), 2, "both occurrences, no special tokens");
    assert_eq!((u32_at(8), u32_at(12)), (0, 6));
    let token_len = u32_at(16) as usize;
    assert_eq!(&out[20..20 + token_len], b"search");
    let weight = f32::from_le_bytes(out[20 + token_len..24 + token_len].try_into().unwrap());
    assert_eq!(weight, 3.0);
    let second = 24 + token_len;
    assert_eq!((u32_at(second + 4), u32_at(second + 8)), (7, 13));
    // SAFETY: invalid UTF-8 is refused.
    let status =
        unsafe { ffi_ml_hf_analyze_alloc(handle, [0xffu8].as_ptr(), 1, &mut ptr, &mut len) };
    assert_eq!(status, FfiStatus::Decode.code());
    let input = texts(&["search"]);
    // SAFETY: valid buffers.
    let status = unsafe {
        ffi_ml_sparse_tokenize_alloc(handle, input.as_ptr(), input.len(), &mut ptr, &mut len)
    };
    assert_eq!(status, FfiStatus::InvalidArgument.code());
    assert_eq!(ffi_ml_model_close(handle), 0);
    // SAFETY: a null out-pointer is refused.
    let status = unsafe {
        ffi_ml_hf_analyzer_open(
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
    let open =
        |name: &str,
         f: unsafe extern "C" fn(*const u8, usize, *const u8, usize, *mut u64) -> i32| {
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
        ffi_ml_torch_text_embedding_alloc(sparse, input.as_ptr(), input.len(), &mut ptr, &mut len)
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
