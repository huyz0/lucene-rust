//! lucene-engine: what the OpenSearch plugin's native library does, as safe
//! Rust with no pointers in sight.
//!
//! `lucene-ffi` is the boundary: handles, the panic guard, and raw reads
//! from the caller's memory. Everything that decides what a request means
//! lives here, so it is compiled under `forbid(unsafe_code)` and tested
//! without a handle.
//!
//! - [`status`] is the code every entry point returns and the message that
//!   explains a failure.
//! - [`query_blob`] decodes a request's query blob; [`nodes`], [`queries`],
//!   [`clauses`], [`function_score`], [`knn_score`], [`knn_painless`],
//!   [`max_sim_query`], [`painless_score`], [`pattern_phrase`] and [`similarity`]
//!   decode the node kinds it names.
//! - [`analysis`] builds analyzers from the plugin's specs, over the
//!   modules [`analysis_modules`] registers.

#![forbid(unsafe_code)]

pub mod alloc;
pub mod analysis;
pub mod analysis_modules;
pub mod clauses;
pub mod fetch;
pub mod function_score;
pub mod invert;
pub mod knn_painless;
pub mod knn_score;
pub mod max_sim_query;
pub mod memory;
pub mod nodes;
pub mod painless_score;
pub mod pattern_phrase;
pub mod queries;
pub mod query_blob;
pub mod reader_handle;
pub mod similarity;
pub mod status;
pub mod writer_blob;
pub mod writer_error;

pub use status::Status;
