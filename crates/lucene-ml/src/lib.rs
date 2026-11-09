#![forbid(unsafe_code)]
//! Local model inference for the plugin's port of OpenSearch ml-commons.
//!
//! ml-commons runs a deployed local model through DJL: a Hugging Face
//! tokenizer, an ONNX Runtime session, then pooling and normalization done
//! on DJL arrays. This crate does the same steps natively, with the same
//! tokenizer library and the same runtime, so a model zip registered on a
//! stock cluster gives the same vectors here.
//!
//! - [`runtime`] loads the ONNX Runtime shared library once per process.
//! - [`tokenizer`] builds the tokenizer the way DJL's `HuggingFaceTokenizer`
//!   configures it (truncation, padding).
//! - [`text_embedding`] is the `TEXT_EMBEDDING` function: one text at a time,
//!   as ml-commons' `TextEmbeddingModel` runs it, pooled by
//!   [`text_embedding::Pooling`].

pub mod runtime;
pub mod text_embedding;
pub mod tokenizer;

/// A failure loading or running a model.
#[derive(Debug, thiserror::Error)]
pub enum MlError {
    /// The ONNX Runtime library could not be loaded, or was not loaded yet.
    #[error("onnx runtime: {0}")]
    Runtime(String),
    /// The tokenizer file could not be parsed, or a text could not be encoded.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    /// The model file was refused, or running it failed.
    #[error("model: {0}")]
    Model(String),
    /// A configuration value the model cannot use.
    #[error("{0}")]
    InvalidArgument(String),
}

/// This crate's result type.
pub type Result<T> = std::result::Result<T, MlError>;
