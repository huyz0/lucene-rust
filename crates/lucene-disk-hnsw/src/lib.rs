#![forbid(unsafe_code)]
//! An on-disk layout for HNSW graphs in which one read serves one hop.
//!
//! Stock Lucene HNSW (`Lucene99HnswVectorsFormat`) keeps a node's
//! neighbours in `.vex` and its vector in `.vec` (or its quantized code in
//! `.veq`), each file ordered by ordinal. Expanding a node reads its
//! neighbour list and then scores every neighbour, so a search touches one
//! graph page and up to `2 * M` vector or code pages per hop, at random. That
//! costs nothing while the files are in memory and a page miss per score
//! once they are not.
//!
//! This layout follows DiskANN (Subramanya et al., NeurIPS 2019) and
//! LM-DiskANN (Pan et al., 2023). It is opt-in and sits next to stock HNSW:
//! the graph is the stock one, built by `lucene_codecs::hnsw` or read from a
//! stock segment ([`DiskHnswWriter::add_field_with_graph`] takes any
//! `HnswGraphView`), and a field can be written both ways.
//!
//! - Each level-0 node is one fixed-size **record**: its document, its
//!   neighbour list and its full-precision vector, and in the
//!   memory-lean mode ([`DiskHnswParams::inline_codes`], LM-DiskANN) its
//!   neighbours' quantized codes as well. Records never straddle a page,
//!   so expanding a node is one aligned page read.
//! - A compact quantized code per node ([`lucene_vector_quant::Quantizer`],
//!   1-bit BBQ for one) stays in memory and scores the neighbours, so the
//!   search only reads the pages of the nodes it expands. In the
//!   memory-lean mode the codes are read from the expanded record instead
//!   and nothing per node stays in memory.
//! - The upper levels (about one node in `M`) and their codes are small
//!   and are read whole at open.
//! - Nodes are renumbered in breadth-first order from the entry node
//!   ([`DiskHnswParams::reorder`]), so neighbours tend to share pages and a
//!   page cache serves more hops from one fetch.
//!
//! A search ([`DiskHnswReader::search`]) descends the upper levels greedily
//! on the codes, then runs DiskANN's beam search on level 0: candidates are
//! ranked by their estimated score, the best `beam_width` unexpanded ones
//! are fetched together (one round trip: their ranges are announced with
//! [`RangeSource::prefetch`] before they are read), each expanded node is
//! scored exactly from the vector in its record, and its neighbours join
//! the candidate list on their estimates. The search ends when the best
//! `list_len` candidates are all expanded and returns the best exact
//! scores, so no separate rerank pass is needed.
//!
//! # Files
//!
//! Three files, each with a Lucene index header (segment id and suffix) and
//! a CRC footer:
//!
//! | file | holds | access |
//! |---|---|---|
//! | `.dhm` meta | per field: similarity, dimension, vector encoding, the quantizer's name and state, the field's mean vector, the record layout, the entry node, the upper levels' adjacency and codes | read whole at open |
//! | `.dhc` codes | per field, one quantized code per node in node order | read whole at open, unless the field has inline codes and the reader leaves them out ([`OpenOptions::resident_codes`]) |
//! | `.dhn` nodes | per field, page-aligned: the level-0 records ([`layout::NodeLayout`]) | one record read per expanded node |
//!
//! Readers take the nodes file as a [`RangeSource`], which a page cache over
//! local disk or over a blob store implements directly.

pub mod graph;
pub mod layout;
pub mod reader;
pub mod writer;

pub use graph::{build_graph, GraphParams};
pub use layout::NodeLayout;
pub use reader::{
    DiskHnswField, DiskHnswReader, OpenOptions, ScoredDoc, SearchOutput, SearchParams, SearchStats,
};
pub use writer::{DiskHnswFiles, DiskHnswParams, DiskHnswWriter, FieldVectors};

pub use lucene_ivf::{RangeSource, VectorEncoding};
pub use lucene_vector_quant::Similarity;

/// Codec name of the `.dhm` header.
pub const META_CODEC: &str = "DiskHnswMeta";
/// Codec name of the `.dhc` header.
pub const CODES_CODEC: &str = "DiskHnswCodes";
/// Codec name of the `.dhn` header.
pub const NODES_CODEC: &str = "DiskHnswNodes";
/// The meta file's extension.
pub const META_EXTENSION: &str = "dhm";
/// The codes file's extension.
pub const CODES_EXTENSION: &str = "dhc";
/// The nodes file's extension.
pub const NODES_EXTENSION: &str = "dhn";
/// The first format version.
pub const VERSION_START: i32 = 0;
/// The current format version.
pub const VERSION_CURRENT: i32 = VERSION_START;
/// The most levels, level 0 included, a field may have.
pub(crate) const MAX_LEVELS: usize = 64;

/// Errors building or reading a disk HNSW index.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The files do not decode.
    #[error("corrupted disk HNSW index: {0}")]
    Corrupted(String),
    /// A store-level read or checksum failure.
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    /// A range read failure.
    #[error(transparent)]
    Range(#[from] lucene_ivf::Error),
    /// The quantizer could not be built or read.
    #[error(transparent)]
    Quantizer(#[from] lucene_vector_quant::Error),
    /// The graph could not be built or read.
    #[error("HNSW graph: {0}")]
    Graph(String),
    /// Bad input from the caller.
    #[error("{0}")]
    IllegalArgument(String),
}

impl From<lucene_codecs::vectors::Error> for Error {
    fn from(e: lucene_codecs::vectors::Error) -> Self {
        Error::Graph(e.to_string())
    }
}

/// This crate's result.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) use lucene_ivf::{similarity_from_ordinal, similarity_ordinal};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_convert() {
        let e: Error = lucene_codecs::vectors::Error::InvalidGraphParameter("x".into()).into();
        assert!(e.to_string().contains("HNSW graph"));
    }
}
