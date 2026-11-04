#![forbid(unsafe_code)]
//! Approximate search over learned sparse vectors in the segment format of
//! OpenSearch's neural-search plugin (its Seismic `sparse_vector` field,
//! `lucene` engine).
//!
//! A segment of an `index.sparse` index records the codec
//! [`CODEC_NAME`], which keeps its delegate's formats and adds, for each
//! `sparse_vector` field of a segment at or over the field's
//! `approximate_threshold` documents:
//!
//! - clustered postings in `.sit`/`.sip` ([`postings`]): per token, clusters
//!   of documents with a quantized weight each and a summary vector per
//!   cluster;
//! - a forward index, which is the field's BINARY doc values: each document's
//!   `(token, weight)` pairs ([`vector::decode_forward`]).
//!
//! [`search()`] runs the query the way neural-search's
//! `OrderedPostingWithClustersScorer` and `ExactMatchScorer` do and returns the
//! same documents with the same scores. Segments under the threshold hold plain
//! `FeatureField` postings and are searched as an ordinary query instead.
//!
//! [`SparseCodec`] writes the same files: an index writer given it clusters
//! each flushed and merged segment's postings ([`writer`]).

pub mod postings;
pub mod search;
pub mod vector;
pub mod writer;

pub use postings::{Cluster, ClusteredPostings};
pub use search::{search, SeismicField, SeismicQuery};
pub use vector::SparseVector;
pub use writer::SparseCodec;

/// The codec name neural-search records for a segment of an `index.sparse` index.
pub const CODEC_NAME: &str = "Sparse10010Codec";

/// Makes [`CODEC_NAME`] segments readable. The codec resolves to Lucene 10.1's
/// formats on read, as neural-search's own `SparseCodec()` does; the per-field
/// postings and doc-values formats come from each field's attributes.
pub fn register_codec() {
    if let Some(base) = lucene_codecs::codecs::for_name("Lucene101") {
        lucene_codecs::codecs::register(lucene_codecs::codecs::renamed(base, CODEC_NAME));
    }
}

/// Errors reading a Seismic segment.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("corrupted sparse index: {0}")]
    Corrupted(String),
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    DocValues(#[from] lucene_codecs::doc_values::Error),
    #[error("{0}")]
    IllegalArgument(String),
}

/// This crate's result.
pub type Result<T> = std::result::Result<T, Error>;
