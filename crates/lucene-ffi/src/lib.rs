//! lucene-ffi: the native library the OpenSearch plugin calls through Java's
//! Foreign Function & Memory API.
//!
//! - [`jvm_reader`] opens a shard's commit and runs query trees, sorts,
//!   collapse and rescore; [`jvm_aggs`] collects aggregations; [`jvm_fetch`]
//!   reads stored documents; [`jvm_analysis`] builds analyzers from the
//!   plugin's specs.
//! - [`engine_writer`] is the shard's index writer, fed operation blobs.
//! - [`store`] binds encrypted shard directories to their key upcall.
//! - [`ffm_bridge`] holds the allocation and error helpers only FFM needs.
//!
//! Every exported function runs inside [`error::guard`], so a panic becomes
//! [`FfiStatus::Panic`] and never unwinds into the JVM. Native objects cross
//! the boundary as generation-tagged `u64` handles ([`handle::SlotMap`]),
//! validated on every call; failures return a status code and leave a
//! thread-local message.

mod analysis_modules;
mod clauses;
pub(crate) mod engine_writer;
mod error;
mod ffm_bridge;
mod handle;
mod jvm_aggs;
mod jvm_analysis;
mod jvm_fetch;
mod jvm_function_score;
mod jvm_invert;
mod jvm_nodes;
mod jvm_queries;
mod jvm_reader;
mod jvm_similarity;
mod raw;
mod registry;
mod store;

pub use error::FfiStatus;

/// mimalloc rather than the system allocator: indexing threads free buffers
/// that other threads allocated (a flushed segment's postings, a merge's
/// readers), and glibc's per-arena locks serialise those frees. The
/// `local_dynamic_tls` feature lets the library load through `dlopen`.
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;
