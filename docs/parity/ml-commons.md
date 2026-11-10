# lucene-ml -- ml-commons' local model inference

[Index](../parity.md). Not a Lucene module: OpenSearch's ml-commons plugin
(3.9.0.0) runs deployed local models through DJL, with ONNX Runtime and
Hugging Face tokenizers underneath. `lucene-ml` runs the same models on the
same tokenizer library and ONNX Runtime. The plugin's `ml` package serves the
REST API (`scripts/check-ml-plugin.py` compares it with a stock node).

| Java | Rust | Status |
|---|---|---|
| -- (DJL `HuggingFaceTokenizer.builder().optPadding(true)`) | `lucene-ml/src/tokenizer.rs::{load, encode, Encoded, MODEL_MAX_LENGTH}` | **rust-only** (port of ml-commons' tokenizer setup). Special tokens, `LONGEST_FIRST` truncation at the file's length capped at 512, the file's stride, batch-longest padding. Tests: unit tests; `tests/stock_vectors.rs`. |
| -- (DJL `OrtEngine` environment) | `lucene-ml/src/runtime.rs::{init, is_loaded}` | **rust-only**. One ONNX Runtime library and environment per process, one intra-op pool shared by every model (ONNX Runtime's default spinning). Tests: `a_second_init_keeps_the_first_runtime`. |
| -- (ml-commons `TextEmbeddingDenseModel`, `ONNXSentenceTransformerTextEmbeddingTranslator`) | `lucene-ml/src/text_embedding.rs::{TextEmbeddingModel, Config, Pooling, Tensor, pool, normalize_first_axis}` | **rust-only** (port of ml-commons). `input_ids`, `attention_mask`, and `token_type_ids` for `bert`/`albert`; MEAN, MEAN_SQRT_LEN, MAX, WEIGHTED_MEAN, CLS, LAST_TOKEN, NONE pooling; L2 normalization along the first axis. One text per run, as stock. Tests: `tests/stock_vectors.rs` (vectors recorded from a stock node, 27 model/pooling cases, relative 2e-5). |
| -- (ml-commons `SparseTokenizerModel`) | `lucene-ml/src/sparse_tokenize.rs::{SparseTokenizer, Weight}` | **rust-only** (port of ml-commons). Distinct token ids, decoded with special tokens skipped, weighted by `idf.json` (1.0 when absent). Tests: unit tests; `scripts/check-ml-plugin.py` (same weights by token and by id as stock). |
| -- (the plugin's FFM bridge) | `lucene-ffi/src/ml.rs::{ffi_ml_runtime_init, ffi_ml_text_embedding_open, ffi_ml_text_embedding_embed_alloc, ffi_ml_sparse_tokenize_open, ffi_ml_sparse_tokenize_alloc, ffi_ml_model_close}` | **rust-only** (no Java counterpart: the plugin's native boundary). Model handles and the length-prefixed text and result encodings. Tests: unit tests. |
