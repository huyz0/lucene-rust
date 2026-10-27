//! Direct component port of Java `tests/codecs/vector/ConfigurableMCodec`.
//!
//! One stable Lucene99 HNSW format overrides the default Lucene104 delegate,
//! with M=128 (or an explicit M), beam width 100, one merge worker, no merge
//! executor, tiny-segment threshold 100 and writer version 1. Other codec
//! accessors forward on every invocation through [`CodecDelegate`], retaining
//! the delegate's returned identity and error type. The native default delegate
//! is a routing descriptor, without Java's component factories or SPI loading.
//!
//! [`ConfigurableMHnswFormat::flush`] writes the four actual native vector files
//! in caller field order. Java buffers values and replays them when the graph
//! threshold is crossed; this owned batch adapter replays the same ordinals at
//! flush, using an explicit seed instead of Java's mutable global seed. It uses
//! the existing flat writer, scorer and HNSW builder without changing their
//! representations. Validation/build failures therefore occur at flush rather
//! than `addValue`, and no streaming writer lifecycle is claimed. Directory,
//! IOContext, sorted flush and configured-M merge graph reuse remain with
//! this adapter's limits; installed in `IndexWriter` (its `Lucene104Codec`
//! impl) the writer's own flush and merge write the direct format. The borrowed native reader
//! checks all four CRCs on open; Java defers data CRCs to `checkIntegrity`.
//! Inherited codec display, format display and the format's fixed
//! `getMaxDimensions(fieldName) == 1024` accessor are not exposed by this API.

use std::collections::HashSet;
use std::io;

use lucene_codecs::codec::DefaultLucene104Codec;
use lucene_codecs::field_infos::VectorEncoding;
use lucene_codecs::hnsw::{should_create_graph, HnswGraphBuilder};
use lucene_codecs::hnsw_vectors::{write_hnsw_vectors, HnswVectorsField};
use lucene_codecs::per_field_knn_vectors::{InvalidFormat, KnnVectorsFormat};
use lucene_codecs::segment_state::SegmentWriteState;
use lucene_codecs::vectors::{write_flat_vectors, FlatVectorsField, FlatVectorsReader};

pub use super::asserting_knn_vectors::{Files, Lucene99KnnVectorsReader, ReadState};

/// Java's `FilterCodec` with a stable direct KNN format override.
pub struct ConfigurableMCodec<D = DefaultLucene104Codec> {
    delegate: D,
    vectors: ConfigurableMHnswFormat,
}

impl Default for ConfigurableMCodec {
    fn default() -> Self {
        Self::new()
    }
}
impl ConfigurableMCodec {
    pub fn new() -> Self {
        Self::with_max_conn(128).expect("Java's fixed default M is valid")
    }
    pub fn with_max_conn(max_conn: i32) -> Result<Self, InvalidFormat> {
        Self::with_delegate(max_conn, DefaultLucene104Codec)
    }
}
impl<D> ConfigurableMCodec<D> {
    pub fn with_delegate(max_conn: i32, delegate: D) -> Result<Self, InvalidFormat> {
        KnnVectorsFormat::hnsw(max_conn, 100)?;
        Ok(Self {
            delegate,
            vectors: ConfigurableMHnswFormat { max_conn },
        })
    }
    pub fn name(&self) -> &'static str {
        "ConfigurableMCodec"
    }
    pub fn delegate(&self) -> &D {
        &self.delegate
    }
    pub fn knn_vectors_format(&self) -> &ConfigurableMHnswFormat {
        &self.vectors
    }
}

impl<D> std::fmt::Debug for ConfigurableMCodec<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigurableMCodec")
            .field("max_conn", &self.vectors.max_conn)
            .finish()
    }
}

/// Installation in `IndexWriter::set_codec`: Java's codec as `IndexWriter`
/// runs it. Every field's vectors go to the one direct
/// `Lucene99HnswVectorsFormat(M, 100)` (unsuffixed `_0.vec`, no per-field
/// attributes) and every other format is `Lucene104`'s; segments record
/// `ConfigurableMCodec`, which [`super::register_codecs`] makes readable.
impl lucene_codecs::codec::Lucene104Codec for ConfigurableMCodec<DefaultLucene104Codec> {
    fn name(&self) -> &str {
        ConfigurableMCodec::name(self)
    }

    fn knn_vectors_format_for_field(&self, _field: &str) -> KnnVectorsFormat {
        KnnVectorsFormat::hnsw(self.vectors.max_conn, 100).expect("validated at construction")
    }

    fn knn_vectors_format_is_direct(&self) -> bool {
        true
    }
}

/// The ten FilterCodec accessors that ConfigurableMCodec does not override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentKind {
    DocValues,
    FieldInfos,
    LiveDocs,
    Norms,
    Postings,
    SegmentInfo,
    StoredFields,
    TermVectors,
    Compound,
    Points,
}

/// Per-invocation callback seam for Java's otherwise absent codec factories.
pub trait CodecDelegate {
    type Component;
    type Error;
    fn component(&self, kind: ComponentKind) -> Result<Self::Component, Self::Error>;
}
impl<D: CodecDelegate> ConfigurableMCodec<D> {
    pub fn component(&self, kind: ComponentKind) -> Result<D::Component, D::Error> {
        self.delegate.component(kind)
    }
}

/// Fixed two-argument Lucene99 format settings selected by the codec.
#[derive(Debug, PartialEq, Eq)]
pub struct ConfigurableMHnswFormat {
    max_conn: i32,
}
impl ConfigurableMHnswFormat {
    pub fn name(&self) -> &'static str {
        "Lucene99HnswVectorsFormat"
    }
    pub fn max_conn(&self) -> i32 {
        self.max_conn
    }
    pub fn beam_width(&self) -> i32 {
        100
    }
    pub fn num_merge_workers(&self) -> i32 {
        1
    }
    pub fn tiny_segments_threshold(&self) -> i32 {
        100
    }
    pub fn write_version(&self) -> i32 {
        1
    }
    pub fn merge_executor(&self) -> Option<()> {
        None
    }

    /// Unsorted, buffered flush in Java's field insertion order. Each field
    /// starts a separate builder with the same seed, as Java's FieldWriter does.
    pub fn flush(
        &self,
        state: SegmentWriteState<'_>,
        fields: &[FlatVectorsField],
        seed: u64,
    ) -> io::Result<Output> {
        if state.max_doc < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "negative maxDoc",
            ));
        }
        let mut seen = HashSet::new();
        let mut counts = Vec::with_capacity(fields.len());
        for field in fields {
            let info = state.field_info(field.field_number).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "unrecognized vector field")
            })?;
            if !seen.insert(field.field_number)
                || info.vector_dimension != field.dimension
                || info.vector_encoding != field.values.encoding()
                || info.vector_similarity_function != field.similarity
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "duplicate or inconsistent vector field metadata",
                ));
            }
            counts.push(i32::try_from(field.docs.len()).map_err(io::Error::other)?);
        }
        let (vec, vemf) = write_flat_vectors(
            fields,
            state.max_doc,
            state.segment_id,
            state.segment_suffix,
        )
        .map_err(io::Error::other)?;
        let flat = FlatVectorsReader::open(&vemf, &vec, state.segment_id, state.segment_suffix)
            .map_err(io::Error::other)?;
        let mut graphs = Vec::with_capacity(fields.len());
        for (field, &count) in fields.iter().zip(&counts) {
            let graph = if should_create_graph(self.tiny_segments_threshold(), count) {
                Some(
                    match field.values.encoding() {
                        VectorEncoding::Float32 => HnswGraphBuilder::new(
                            flat.float_vector_values(field.field_number)
                                .map_err(io::Error::other)?
                                .ord_scorer(),
                            self.max_conn,
                            self.beam_width(),
                            seed,
                        )
                        .and_then(|builder| builder.build(count)),
                        VectorEncoding::Byte => HnswGraphBuilder::new(
                            flat.byte_vector_values(field.field_number)
                                .map_err(io::Error::other)?
                                .ord_scorer(),
                            self.max_conn,
                            self.beam_width(),
                            seed,
                        )
                        .and_then(|builder| builder.build(count)),
                    }
                    .map_err(io::Error::other)?,
                )
            } else {
                None
            };
            graphs.push(graph);
        }
        let metadata: Vec<_> = fields
            .iter()
            .zip(&counts)
            .zip(&graphs)
            .map(|((field, &count), graph)| HnswVectorsField {
                field_number: field.field_number,
                encoding: field.values.encoding(),
                similarity: field.similarity,
                dimension: field.dimension,
                count,
                graph: graph.as_ref(),
                m: self.max_conn,
            })
            .collect();
        let (vex, vem) = write_hnsw_vectors(&metadata, state.segment_id, state.segment_suffix)
            .map_err(io::Error::other)?;
        Ok(Output {
            vec,
            vemf,
            vex,
            vem,
        })
    }

    pub fn fields_reader<'a>(
        &self,
        state: ReadState<'a>,
    ) -> io::Result<Lucene99KnnVectorsReader<'a>> {
        Lucene99KnnVectorsReader::open(state)
    }
}

/// Four owned files; readers borrow their bytes instead of a Java Directory.
#[derive(Debug)]
pub struct Output {
    pub vec: Vec<u8>,
    pub vemf: Vec<u8>,
    pub vex: Vec<u8>,
    pub vem: Vec<u8>,
}
impl Output {
    pub fn files(&self) -> Files<'_> {
        Files {
            vec: &self.vec,
            vemf: &self.vemf,
            vex: &self.vex,
            vem: &self.vem,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::field_infos::{FieldInfo, FieldInfos, VectorSimilarityFunction};
    use lucene_codecs::hnsw_vectors::HnswVectorsReader;
    use lucene_codecs::vectors::FieldVectorData;

    #[test]
    fn boundary_connection_settings_survive_empty_and_singleton_metadata() {
        let id = [7; 16];
        let infos = FieldInfos::new(vec![FieldInfo::new("f", 5).with_vectors(
            1,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        )])
        .unwrap();
        let state = SegmentWriteState {
            max_doc: 1,
            segment_id: &id,
            segment_suffix: "s",
            field_infos: &infos,
        };
        for m in [1, 512] {
            for singleton in [false, true] {
                let field = FlatVectorsField {
                    field_number: 5,
                    similarity: VectorSimilarityFunction::Euclidean,
                    dimension: 1,
                    docs: if singleton { vec![0] } else { Vec::new() },
                    values: FieldVectorData::Float32(if singleton {
                        vec![3.0]
                    } else {
                        Vec::new()
                    }),
                };
                let codec = ConfigurableMCodec::with_max_conn(m).unwrap();
                let output = codec
                    .knn_vectors_format()
                    .flush(state, &[field], 42)
                    .unwrap();
                let graph = HnswVectorsReader::open(&output.vem, &output.vex, &id, "s").unwrap();
                let entry = graph.field(5).unwrap();
                assert_eq!(entry.m, m);
                assert_eq!(entry.size, i32::from(singleton));
                assert!(graph.graph(5).unwrap().is_none());
            }
        }
    }
}
