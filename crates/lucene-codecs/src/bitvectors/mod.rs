//! Port of lucene-codecs' `org.apache.lucene.codecs.bitvectors`: byte
//! vectors read as packed bit vectors and compared by Hamming distance.
//!
//! [`HnswBitVectorsFormat`] writes exactly `Lucene99HnswVectorsFormat`'s
//! files (`.vec`/`.vemf` from [`crate::vectors`], `.vex`/`.vem` from
//! [`crate::hnsw_vectors`]); only the scorer the graph is built and searched
//! with differs. [`FlatBitVectorsScorer`] scores a candidate as
//!
//! ```text
//! (bitDimensions - xorBitCount(query, candidate)) / (float) bitDimensions
//! ```
//!
//! with `bitDimensions = dimension * 8`, whatever similarity function the
//! field declares, so the recorded similarity only rides along in `.vemf`.
//!
//! # What differs from Java
//!
//! - Module-level, like `IDVersionPostingsFormat`, the bloom filter and the
//!   `memory` formats: the files go in and come out as bytes, this port's
//!   `IndexWriter` routes no field here and `DirectoryReader` does not open
//!   these files.
//! - A merge is not ported: [`HnswBitVectorsFormat::write`] builds every
//!   graph from scratch (a flush), where Java's `mergeOneField` seeds the
//!   merged graph from the largest incoming one. The merged segment's
//!   vectors and the searches over a Java-merged graph are what the
//!   differential tests compare; a Rust-built merged graph is not.
//! - `numMergeWorkers`/`mergeExec` (merge-time concurrency, and with them
//!   the constructor's `numMergeWorkers == 1 && mergeExec != null` check)
//!   and `ramBytesUsed` are not ported, nor is a sorted segment
//!   (`Sorter.DocMap`).
//! - `search(field, float[] target, ...)` is not ported: Java's reader
//!   refuses it as an encoding mismatch (every field here is BYTE);
//!   [`FlatBitVectorsScorer::float_scorer`] keeps the scorer's refusal.

#[cfg(test)]
mod tests;

use std::fmt;

use lucene_store::codec_util::ID_LENGTH;
use lucene_util::vector_util::xor_bit_count;

use crate::field_infos::VectorEncoding;
use crate::hnsw::{self, UpdateableVectorScorer, VectorScorer};
use crate::hnsw_vectors::{self, HnswVectorsField, HnswVectorsReader, SearchOptions};
use crate::vectors::{
    self, ByteVectorValues, FieldVectorData, FlatVectorsField, FlatVectorsReader,
};

/// What reading or writing these formats fails with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The flat and HNSW vector files, and their argument checks.
    #[error(transparent)]
    Vectors(#[from] vectors::Error),
    /// Java's `IllegalArgumentException`s that are not about the files.
    #[error("{0}")]
    Unsupported(&'static str),
}

/// `Result` of this module.
pub type Result<T> = std::result::Result<T, Error>;

/// Port of `FlatBitVectorsScorer`: hands out [`BitVectorScorer`]s.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FlatBitVectorsScorer;

impl fmt::Display for FlatBitVectorsScorer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FlatBitVectorsScorer()")
    }
}

impl FlatBitVectorsScorer {
    /// `getRandomVectorScorer(similarity, values, byte[] target)`.
    ///
    /// # Errors
    /// A `target` whose length is not the field's dimension. This mirrors
    /// the up-front check of `CodecReader.searchNearestVectors`, so it fails
    /// even where Java's scorer would never score (an empty field, `k == 0`)
    /// and `VectorUtil.xorBitCount` would never see the mismatch.
    pub fn scorer<'a>(
        &self,
        values: &ByteVectorValues<'a>,
        target: &[u8],
    ) -> Result<BitVectorScorer<'a>> {
        if target.len() != values.dimension() {
            return Err(vectors::Error::QueryDimensionMismatch(
                target.len() as i32,
                values.dimension() as i32,
            )
            .into());
        }
        Ok(BitVectorScorer::new(values, target.to_vec()))
    }

    /// `getRandomVectorScorer(similarity, values, float[] target)`.
    ///
    /// # Errors
    /// Always, as Java's.
    pub fn float_scorer(&self, _target: &[f32]) -> Result<BitVectorScorer<'static>> {
        Err(Error::Unsupported(
            "bit vectors do not support float[] targets",
        ))
    }

    /// `getRandomVectorScorerSupplier(similarity, values).scorer()`: the
    /// query is set later with [`UpdateableVectorScorer::set_scoring_ordinal`].
    pub fn ord_scorer<'a>(&self, values: &ByteVectorValues<'a>) -> BitVectorScorer<'a> {
        BitVectorScorer::new(values, vec![0; values.dimension()])
    }
}

/// Port of `FlatBitVectorsScorer.BitRandomVectorScorer`.
#[derive(Debug, Clone)]
pub struct BitVectorScorer<'a> {
    values: ByteVectorValues<'a>,
    bit_dimensions: i32,
    query: Vec<u8>,
}

impl<'a> BitVectorScorer<'a> {
    fn new(values: &ByteVectorValues<'a>, query: Vec<u8>) -> Self {
        // `saturating_mul`: a dimension past `i32::MAX / 8` (which no `.vemf`
        // the flat reader accepts carries) cannot overflow.
        let bit_dimensions = (values.dimension() as i32).saturating_mul(8);
        BitVectorScorer {
            values: values.clone(),
            bit_dimensions,
            query,
        }
    }

    /// `ordToDoc(ord)`.
    ///
    /// # Errors
    /// An ordinal out of range.
    pub fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        Ok(self.values.ord_to_doc(ord)?)
    }
}

impl VectorScorer for BitVectorScorer<'_> {
    fn score(&mut self, node: i32) -> vectors::Result<f32> {
        let candidate = self.values.vector(node)?;
        let distance = xor_bit_count(&self.query, candidate);
        // ARITH: `distance` counts differing bits of two `dimension`-byte
        // vectors, so it is at most `bit_dimensions`.
        #[allow(clippy::arithmetic_side_effects)]
        let same = self.bit_dimensions - distance;
        Ok(same as f32 / self.bit_dimensions as f32)
    }

    fn max_ord(&self) -> i32 {
        self.values.size()
    }
}

impl UpdateableVectorScorer for BitVectorScorer<'_> {
    fn set_scoring_ordinal(&mut self, ord: i32) -> vectors::Result<()> {
        let v = self.values.vector(ord)?;
        self.query.copy_from_slice(v);
        Ok(())
    }
}

/// The four files [`HnswBitVectorsFormat::write`] produces.
#[derive(Debug, Clone, Default)]
pub struct BitVectorsFiles {
    /// `.vec`: the vectors.
    pub vec: Vec<u8>,
    /// `.vemf`: the flat vectors' metadata.
    pub vemf: Vec<u8>,
    /// `.vex`: the graphs.
    pub vex: Vec<u8>,
    /// `.vem`: the graphs' metadata.
    pub vem: Vec<u8>,
}

/// Port of `HnswBitVectorsFormat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswBitVectorsFormat {
    max_conn: i32,
    beam_width: i32,
}

impl Default for HnswBitVectorsFormat {
    fn default() -> Self {
        HnswBitVectorsFormat {
            max_conn: hnsw::DEFAULT_MAX_CONN,
            beam_width: hnsw::DEFAULT_BEAM_WIDTH,
        }
    }
}

impl fmt::Display for HnswBitVectorsFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HnswBitVectorsFormat(name={}, maxConn={}, beamWidth={}, \
             flatVectorFormat=Lucene99FlatVectorsFormat(vectorsScorer={}))",
            Self::NAME,
            self.max_conn,
            self.beam_width,
            FlatBitVectorsScorer
        )
    }
}

impl HnswBitVectorsFormat {
    /// `NAME`.
    pub const NAME: &'static str = "HnswBitVectorsFormat";

    /// `getMaxDimensions(field)`, the same for every field.
    pub const MAX_DIMENSIONS: i32 = 1024;

    /// `new HnswBitVectorsFormat(maxConn, beamWidth)`.
    ///
    /// # Errors
    /// `maxConn` outside `1..=512` or `beamWidth` outside `1..=3200`, with
    /// Java's messages.
    pub fn new(max_conn: i32, beam_width: i32) -> Result<Self> {
        if max_conn <= 0 || max_conn > hnsw::MAXIMUM_MAX_CONN {
            return Err(vectors::Error::InvalidGraphParameter(format!(
                "maxConn must be positive and less than or equal to {}; maxConn={max_conn}",
                hnsw::MAXIMUM_MAX_CONN
            ))
            .into());
        }
        if beam_width <= 0 || beam_width > hnsw::MAXIMUM_BEAM_WIDTH {
            return Err(vectors::Error::InvalidGraphParameter(format!(
                "beamWidth must be positive and less than or equal to {}; beamWidth={beam_width}",
                hnsw::MAXIMUM_BEAM_WIDTH
            ))
            .into());
        }
        Ok(HnswBitVectorsFormat {
            max_conn,
            beam_width,
        })
    }

    /// `maxConn`.
    pub fn max_conn(&self) -> i32 {
        self.max_conn
    }

    /// `beamWidth`.
    pub fn beam_width(&self) -> i32 {
        self.beam_width
    }

    /// `getMaxDimensions(field)`.
    pub fn max_dimensions(&self, _field: &str) -> i32 {
        Self::MAX_DIMENSIONS
    }

    /// `fieldsWriter(state)`: every field's `addField`, its vectors, then
    /// `flush(maxDoc, null)` and `finish()`.
    ///
    /// Each graph is built over the `.vec` bytes just written, with the
    /// bit scorer and `HnswGraphBuilder`'s fixed seed, as Java builds it
    /// while the vectors arrive; a field too small for a graph
    /// (`shouldCreateGraph`) gets none.
    ///
    /// # Errors
    /// A field that is not `BYTE`-encoded (`"HnswBitVectorsFormat only
    /// supports BYTE encoding"`), and whatever the flat writer refuses.
    pub fn write(
        &self,
        fields: &[FlatVectorsField],
        max_doc: i32,
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
    ) -> Result<BitVectorsFiles> {
        if fields
            .iter()
            .any(|f| !matches!(f.values, FieldVectorData::Byte(_)))
        {
            return Err(Error::Unsupported(
                "HnswBitVectorsFormat only supports BYTE encoding",
            ));
        }
        let (vec, vemf) = vectors::write_flat_vectors(fields, max_doc, segment_id, segment_suffix)?;
        let graphs = {
            let flat = FlatVectorsReader::open(&vemf, &vec, segment_id, segment_suffix)?;
            let mut graphs = Vec::with_capacity(fields.len());
            for field in fields {
                let count = field.docs.len() as i32;
                if !hnsw::should_create_graph(hnsw::HNSW_GRAPH_THRESHOLD, count) {
                    graphs.push(None);
                    continue;
                }
                let values = flat.byte_vector_values(field.field_number)?;
                let graph = hnsw::HnswGraphBuilder::new(
                    FlatBitVectorsScorer.ord_scorer(&values),
                    self.max_conn,
                    self.beam_width,
                    hnsw::DEFAULT_RAND_SEED,
                )?
                .build(count)?;
                graphs.push(Some(graph));
            }
            graphs
        };
        let hnsw_fields: Vec<HnswVectorsField<'_>> = fields
            .iter()
            .zip(&graphs)
            .map(|(field, graph)| HnswVectorsField {
                field_number: field.field_number,
                encoding: VectorEncoding::Byte,
                similarity: field.similarity,
                dimension: field.dimension,
                count: field.docs.len() as i32,
                graph: graph.as_ref(),
                m: self.max_conn,
            })
            .collect();
        let (vex, vem) =
            hnsw_vectors::write_hnsw_vectors(&hnsw_fields, segment_id, segment_suffix)?;
        Ok(BitVectorsFiles {
            vec,
            vemf,
            vex,
            vem,
        })
    }

    /// `fieldsReader(state)`.
    ///
    /// # Errors
    /// Headers, footers and metadata the flat or HNSW reader refuses.
    pub fn open<'a>(
        &self,
        files: &'a BitVectorsFiles,
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
    ) -> Result<BitVectorsReader<'a>> {
        BitVectorsReader::open(files, segment_id, segment_suffix)
    }
}

/// `Lucene99HnswVectorsReader` over a `Lucene99FlatVectorsReader`, both
/// scoring with [`FlatBitVectorsScorer`].
#[derive(Debug)]
pub struct BitVectorsReader<'a> {
    flat: FlatVectorsReader<'a>,
    hnsw: HnswVectorsReader<'a>,
}

impl<'a> BitVectorsReader<'a> {
    /// Opens the four files.
    ///
    /// # Errors
    /// See [`HnswBitVectorsFormat::open`].
    pub fn open(
        files: &'a BitVectorsFiles,
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
    ) -> Result<Self> {
        let flat = FlatVectorsReader::open(&files.vemf, &files.vec, segment_id, segment_suffix)?;
        let hnsw = HnswVectorsReader::open(&files.vem, &files.vex, segment_id, segment_suffix)?;
        Ok(BitVectorsReader { flat, hnsw })
    }

    /// `getByteVectorValues(field)`.
    ///
    /// # Errors
    /// An unknown field.
    pub fn byte_vector_values(&self, field_number: i32) -> Result<ByteVectorValues<'a>> {
        Ok(self.flat.byte_vector_values(field_number)?)
    }

    /// `search(field, byte[] target, knnCollector, acceptDocs)`: the `k`
    /// best `(doc, score)` pairs, best first, and whether the visit limit
    /// cut the search short.
    ///
    /// # Errors
    /// An unknown field, a `target` of the wrong length, and graph or
    /// vector bytes that do not decode.
    pub fn search(
        &self,
        field_number: i32,
        target: &[u8],
        k: usize,
        visit_limit: u64,
        options: SearchOptions<'_>,
    ) -> Result<(Vec<(i32, f32)>, bool)> {
        let values = self.byte_vector_values(field_number)?;
        let mut scorer = FlatBitVectorsScorer.scorer(&values, target)?;
        let graph = self.hnsw.graph(field_number)?;
        let (hits, early) =
            hnsw_vectors::search(&mut scorer, graph.as_ref(), k, visit_limit, options)?;
        let hits = hits
            .into_iter()
            .map(|(ord, score)| Ok((values.ord_to_doc(ord)?, score)))
            .collect::<Result<Vec<_>>>()?;
        Ok((hits, early))
    }
}
