//! The disk layout of a [`KnnVectorsFormat::HnswDisk`] instance (`.dhm`,
//! `.dhc`, `.dhn`; `lucene-disk-hnsw`), written beside the instance's raw
//! vectors and HNSW graph at flush and at merge. The graph is the plain one
//! built on the raw vectors, so a field is readable as stock HNSW too; the
//! disk files copy each node's neighbours and vector into one page-aligned
//! record and add the quantized code every candidate is ranked on. A merge
//! writes the files afresh over the merged graph.
//!
//! Only `FLOAT32` fields with a graph get disk files: a segment too small for
//! a graph (`Lucene99HnswVectorsWriter.shouldCreateGraph`) is searched
//! exhaustively, as for every graph format.

use std::sync::Arc;

use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_codecs::hnsw::OnHeapHnswGraph;
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::scalar_quantized_vectors::util_similarity;
use lucene_disk_hnsw::{
    DiskHnswParams, DiskHnswReader, DiskHnswWriter, FieldVectors, OpenOptions, RangeSource,
};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::Input;
use lucene_vector_quant::QuantizerRegistry;

use crate::ivf_vectors::quantizer_factory;

type Result<T> = std::result::Result<T, lucene_disk_hnsw::Error>;

/// The disk files' extensions: metadata, codes, nodes.
pub const EXTENSIONS: [&str; 3] = [
    lucene_disk_hnsw::META_EXTENSION,
    lucene_disk_hnsw::CODES_EXTENSION,
    lucene_disk_hnsw::NODES_EXTENSION,
];

/// An instance's disk files by extension.
pub(crate) type Files = Vec<(&'static str, Vec<u8>)>;

/// One float field of a disk HNSW instance with the graph built on its
/// vectors: ordinal `i` of the graph is the field's `i`-th vector.
pub(crate) struct DiskField<'a> {
    pub number: i32,
    pub similarity: VectorSimilarityFunction,
    pub dimension: usize,
    pub docs: &'a [i32],
    pub vectors: &'a [f32],
    pub graph: &'a OnHeapHnswGraph,
}

/// Writes the disk files of a [`KnnVectorsFormat::HnswDisk`] instance's
/// fields: `None` for another format, or when no field has a graph.
///
/// # Errors
/// An unknown quantizer, vectors the quantizer refuses, or a graph that
/// does not match its vectors.
pub(crate) fn write(
    format: KnnVectorsFormat,
    fields: &[DiskField<'_>],
    segment_id: &[u8; ID_LENGTH],
    suffix: &str,
) -> Result<Option<Files>> {
    let Some(quantizer) = format.hnsw_disk_quantizer() else {
        return Ok(None);
    };
    let factory = quantizer_factory(quantizer)
        .filter(|f| f.candidates().is_none())
        .ok_or_else(|| {
            lucene_disk_hnsw::Error::IllegalArgument(format!(
                "unknown disk HNSW quantizer {quantizer}"
            ))
        })?;
    let params = DiskHnswParams::default();
    let mut writer = DiskHnswWriter::new(segment_id, suffix);
    let mut any = false;
    for field in fields
        .iter()
        .filter(|f| f.dimension > 0 && !f.docs.is_empty())
    {
        writer.add_field_with_graph(
            &FieldVectors {
                number: field.number,
                similarity: util_similarity(field.similarity),
                dimension: field.dimension,
                docs: field.docs,
                vectors: field.vectors,
            },
            field.graph,
            factory.as_ref(),
            &params,
        )?;
        any = true;
    }
    if !any {
        return Ok(None);
    }
    let files = writer.finish();
    Ok(Some(vec![
        (EXTENSIONS[0], files.meta),
        (EXTENSIONS[1], files.codes),
        (EXTENSIONS[2], files.nodes),
    ]))
}

/// Opens an instance's disk files.
///
/// # Errors
/// A header, footer or checksum mismatch, or a quantizer this release does
/// not know.
pub fn open(
    meta: &Input,
    codes: Arc<Input>,
    nodes: Arc<Input>,
    segment_id: &[u8; ID_LENGTH],
    suffix: &str,
) -> Result<DiskHnswReader> {
    let codes: Arc<dyn RangeSource> = codes;
    let nodes: Arc<dyn RangeSource> = nodes;
    DiskHnswReader::open(
        meta,
        codes,
        nodes,
        segment_id,
        suffix,
        &QuantizerRegistry::with_builtins(),
        OpenOptions::default(),
    )
}
