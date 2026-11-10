#![forbid(unsafe_code)]
//! An IVF (inverted file) approximate nearest-neighbour index over float
//! vectors, with a pluggable quantizer.
//!
//! A field's vectors are partitioned into clusters of a few hundred vectors
//! ([`kmeans`]). Each cluster keeps a **posting list**: its documents and one
//! quantized code per vector, encoded relative to the cluster's centroid by
//! any [`lucene_vector_quant::Quantizer`] (Lucene's BBQ / OSQ, Lucene99-style
//! scalar quantization, or one registered later). A search scores the query
//! against every centroid, scans the codes of the `nprobe` best clusters, and
//! reranks the best estimates with the full-precision vectors.
//!
//! The index is written once per segment ([`writer`]) and never updated; a
//! merge writes the merged segment's vectors afresh, clustered from scratch
//! or, when one input holds most of them, from that input's clusters
//! rebalanced locally ([`merge::merge_field`], [`MergeClustering`]).
//!
//! # Search depth
//!
//! [`SearchParams`] fixes the clusters probed and the candidates reranked
//! ([`Probes::Fixed`], [`Rerank::Fixed`]), or leaves them to the field
//! ([`Probes::Auto`], [`Rerank::Auto`]). Writing a field calibrates its
//! depth for a target recall on a sample of its own vectors ([`calibrate`])
//! and records it as a share of its clusters and a multiple of `k`. An
//! automatic rerank also stops early: candidates go in estimate order, and
//! once the `k`-th exact score beats every remaining candidate's estimate
//! plus the quantizer's error margin
//! (`lucene_vector_quant::QueryScorer::error_bounds`), the rest are
//! skipped ([`Rerank::Bounded`] does the same up to a fixed cap).
//!
//! A search over several segments can share its running `k`-th exact score
//! ([`SharedThreshold`], [`IvfReader::search_with`]): each probed cluster's
//! radius bounds its entries' exact scores, so a segment skips the clusters
//! (and candidates) that cannot enter the shared top `k`; and
//! [`segment_nprobe`] splits a probe budget by the segments' sizes.
//!
//! # Files
//!
//! Three files, each with a Lucene index header (segment id and suffix) and
//! a CRC footer:
//!
//! | file | holds | access |
//! |---|---|---|
//! | `.ivm` meta | per field: similarity, rerank vector encoding, dimension, the quantizer's name and state, every centroid, cluster size and radius, the calibrated search depth | read whole at open; small enough to keep in memory |
//! | `.ivp` postings | per cluster, contiguous: its slots (`u32` LE, ascending) then its codes as one block in the quantizer's layout (`Quantizer::write_block`: interleaved 32 at a time for fast scan with BBQ) | one range read per probed cluster |
//! | `.ivv` vectors | full-precision vectors in cluster order, in the field's [`VectorEncoding`] (`f32`, `f16`, `bf16`, or 8-bit residuals from the cluster centroid with a per-vector scale; [`IvfParams::rerank_encoding`]), then a document-to-slot table | one small range read per reranked candidate |
//!
//! The layout is chosen for both local block storage and blob storage behind
//! a local cache: probing a cluster is one contiguous read, the codes file
//! is a fraction of the vectors' size (1 bit per dimension plus 16 bytes per
//! vector for BBQ) so it can be pinned in memory or on local NVMe, and the
//! full-precision vectors are only touched for the few candidates reranked.
//! Storing those in `f16`, `bf16` or 8 bits halves or quarters the bytes
//! each reranked candidate reads.
//! Readers take the postings and vectors files as a [`RangeSource`], which a
//! caching blob reader can implement directly.

pub mod calibrate;
pub mod encoding;
pub mod kmeans;
pub mod merge;
pub mod reader;
pub mod routing;
pub mod writer;

pub use calibrate::{Calibration, DEFAULT_CALIBRATION_RECALL};
pub use encoding::VectorEncoding;
pub use merge::{merge_field, merge_field_with_vectors, MergeClustering, MergeSource};
pub use reader::{
    segment_nprobe, Grouping, IvfField, IvfReader, Probes, RangeSource, Rerank, ScoredDoc,
    SearchOutput, SearchParams, SearchStats, SharedThreshold, DEFAULT_PROBE_FRACTION,
    DEFAULT_RERANK_MULTIPLE,
};
pub use routing::Routing;
pub use writer::{FieldVectors, IvfFiles, IvfParams, IvfWriter};

pub use lucene_vector_quant::Similarity;

/// Codec name of the `.ivm` header.
pub const META_CODEC: &str = "LuceneRustIvfMeta";
/// Codec name of the `.ivp` header.
pub const POSTINGS_CODEC: &str = "LuceneRustIvfPostings";
/// Codec name of the `.ivv` header.
pub const VECTORS_CODEC: &str = "LuceneRustIvfVectors";
/// The meta file's extension.
pub const META_EXTENSION: &str = "ivm";
/// The postings file's extension.
pub const POSTINGS_EXTENSION: &str = "ivp";
/// The vectors file's extension.
pub const VECTORS_EXTENSION: &str = "ivv";
/// The first format version.
pub const VERSION_START: i32 = 0;
/// Cluster codes in the quantizer's block layout (`Quantizer::write_block`)
/// rather than end to end. Files of earlier versions are not read.
pub const VERSION_BLOCK_LAYOUT: i32 = 1;
/// A per-field rerank vector encoding byte in `.ivm`
/// ([`VectorEncoding`]). Version 1 fields are read as `f32`.
pub const VERSION_VECTOR_ENCODING: i32 = 2;
/// Per-field cluster radii and calibrated search depth in `.ivm`
/// ([`IvfField::radius`], [`IvfField::calibration`]). Version 1 and 2
/// fields are read with neither.
pub const VERSION_CALIBRATION: i32 = 3;
/// A per-field centroid routing tree in `.ivm` (`routing::tree_parents`).
/// Version 1 to 3 fields are read without one.
pub const VERSION_ROUTING_TREE: i32 = 4;
/// A per-field byte in `.ivm` saying whether the field was clustered per
/// partition (`IvfWriter::add_partitioned_field`). Version 1 to 4 fields
/// were not.
pub const VERSION_PARTITIONS: i32 = 5;
/// The current format version.
pub const VERSION_CURRENT: i32 = VERSION_PARTITIONS;

/// Errors building or reading an IVF index.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The files do not decode.
    #[error("corrupted IVF index: {0}")]
    Corrupted(String),
    /// A store-level read or checksum failure.
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    /// The quantizer could not be built or read.
    #[error(transparent)]
    Quantizer(#[from] lucene_vector_quant::Error),
    /// Bad input from the caller.
    #[error("{0}")]
    IllegalArgument(String),
}

/// This crate's result.
pub type Result<T> = std::result::Result<T, Error>;

/// `VectorSimilarityFunction`'s ordinal, as Lucene records it.
pub fn similarity_ordinal(s: Similarity) -> u8 {
    match s {
        Similarity::Euclidean => 0,
        Similarity::DotProduct => 1,
        Similarity::Cosine => 2,
        Similarity::MaximumInnerProduct => 3,
    }
}

/// The similarity of a recorded ordinal.
pub fn similarity_from_ordinal(o: u8) -> Option<Similarity> {
    Some(match o {
        0 => Similarity::Euclidean,
        1 => Similarity::DotProduct,
        2 => Similarity::Cosine,
        3 => Similarity::MaximumInnerProduct,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_ordinals_round_trip() {
        for s in [
            Similarity::Euclidean,
            Similarity::DotProduct,
            Similarity::Cosine,
            Similarity::MaximumInnerProduct,
        ] {
            assert_eq!(similarity_from_ordinal(similarity_ordinal(s)), Some(s));
        }
        assert_eq!(similarity_from_ordinal(4), None);
    }
}
