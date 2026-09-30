//! `Lucene104ScalarQuantizedVectorsFormat` and
//! `Lucene104HnswScalarQuantizedVectorsFormat`: per-vector optimized scalar
//! quantization of `FLOAT32` vector fields, stored beside the raw vectors.
//!
//! Port of `org.apache.lucene.codecs.lucene104.{Lucene104ScalarQuantizedVectorsFormat,
//! Lucene104ScalarQuantizedVectorsReader, Lucene104ScalarQuantizedVectorsWriter,
//! Lucene104ScalarQuantizedVectorScorer, OffHeapScalarQuantizedVectorValues,
//! OffHeapScalarQuantizedFloatVectorValues, Lucene104HnswScalarQuantizedVectorsFormat}`.
//! The quantizer itself is [`lucene_util::quantization`].
//!
//! A segment written with the HNSW format holds three file pairs: the raw
//! vectors in `Lucene99FlatVectorsFormat`'s `.vec`/`.vemf` ([`crate::vectors`]),
//! the quantized vectors here in `.veq`/`.vemq`, and the graph in
//! `Lucene99HnswVectorsFormat`'s `.vem`/`.vex` ([`crate::hnsw_vectors`]).
//! Only `FLOAT32` fields are quantized; a `BYTE` field has no `.vemq` entry.
//!
//! `.veq`:
//! ```text
//! IndexHeader(codec="Lucene104ScalarQuantizedVectorsFormatData", version=0, id, suffix)
//! for each field:
//!   zero padding to a multiple of 4
//!   for each ordinal: packed codes (ScalarEncoding.getDocPackedLength(dim) bytes),
//!                     lowerInterval, upperInterval, additionalCorrection (f32 bits, int32 LE),
//!                     quantizedComponentSum (int32 LE)
//!   if sparse: IndexedDISI + DirectMonotonic ord->doc (OrdToDocDISIReaderConfiguration)
//! Footer
//! ```
//!
//! `.vemq`:
//! ```text
//! IndexHeader(codec="Lucene104ScalarQuantizedVectorsFormatMeta", version=0, id, suffix)
//! for each field:
//!   FieldNumber int32, VectorEncoding int32, VectorSimilarityFunction int32,
//!   Dimension vint, VectorDataOffset vlong, VectorDataLength vlong, Count vint,
//!   if Count > 0: ScalarEncoding wire number vint, centroid (dim x f32 LE), centroidDP (f32 bits)
//!   OrdToDocDISIReaderConfiguration stored meta
//! -1 int32
//! Footer
//! ```
//!
//! # What a flush quantizes, and what a merge does
//!
//! A flushed field's graph is built on the **raw** float vectors (Java's
//! `Lucene99HnswVectorsWriter` hands the quantized scorer plain float values
//! while indexing, and it delegates those to the float scorer), so a flushed
//! graph is the plain `Lucene99HnswVectorsFormat` graph. A merge rebuilds the
//! graph with the quantized scorer ([`merge_scorer_supplier`]).

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;
use lucene_util::quantization::{
    self, OptimizedScalarQuantizer, QuantizationResult, ScalarEncoding,
};
use lucene_util::vector_util;

use crate::direct_monotonic;
use crate::field_infos::{VectorEncoding, VectorSimilarityFunction};
use crate::indexed_disi::DisiCursor;
use crate::vectors::{
    encoding_ordinal, file_region, read_similarity_function, read_vector_encoding,
    similarity_ordinal, write_stored_meta, DocToOrdCursor, Error, OrdToDoc, Result,
};

/// `Lucene104ScalarQuantizedVectorsFormat.NAME`.
pub const NAME: &str = "Lucene104ScalarQuantizedVectorsFormat";
/// `Lucene104HnswScalarQuantizedVectorsFormat.NAME` -- the SPI name recorded
/// in `PerFieldKnnVectorsFormat.format` (note: "Binary", as in 10.5.0).
pub const HNSW_NAME: &str = "Lucene104HnswBinaryQuantizedVectorsFormat";
/// `META_CODEC_NAME`.
pub const META_CODEC: &str = "Lucene104ScalarQuantizedVectorsFormatMeta";
/// `VECTOR_DATA_CODEC_NAME`.
pub const DATA_CODEC: &str = "Lucene104ScalarQuantizedVectorsFormatData";
/// `META_EXTENSION`.
pub const META_EXTENSION: &str = "vemq";
/// `VECTOR_DATA_EXTENSION`.
pub const DATA_EXTENSION: &str = "veq";
/// `VERSION_START`.
pub const VERSION_START: i32 = 0;
/// `VERSION_CURRENT`.
pub const VERSION_CURRENT: i32 = VERSION_START;
/// `DIRECT_MONOTONIC_BLOCK_SHIFT`.
pub const DIRECT_MONOTONIC_BLOCK_SHIFT: u32 = 16;
/// `getMaxDimensions`.
pub const MAX_DIMENSIONS: i32 = 1024;
/// Bytes of corrective terms after each vector's codes: three floats and an int.
const CORRECTIONS_BYTES: usize = 16;

fn corrupt<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::CorruptMeta(msg.into()))
}

/// The quantizer's mirror of [`VectorSimilarityFunction`].
pub fn util_similarity(sim: VectorSimilarityFunction) -> quantization::VectorSimilarityFunction {
    match sim {
        VectorSimilarityFunction::Euclidean => quantization::VectorSimilarityFunction::Euclidean,
        VectorSimilarityFunction::DotProduct => quantization::VectorSimilarityFunction::DotProduct,
        VectorSimilarityFunction::Cosine => quantization::VectorSimilarityFunction::Cosine,
        VectorSimilarityFunction::MaximumInnerProduct => {
            quantization::VectorSimilarityFunction::MaximumInnerProduct
        }
    }
}

/// Packs `scratch` (one code per discretized dimension) into `packed` per
/// `encoding` -- the `switch` in `writeVectors` / `QuantizedFloatVectorValues.quantize`.
fn pack_codes(encoding: ScalarEncoding, scratch: &[u8], packed: &mut Vec<u8>) {
    packed.clear();
    match encoding {
        ScalarEncoding::UnsignedByte | ScalarEncoding::SevenBit => {
            packed.extend_from_slice(scratch)
        }
        ScalarEncoding::PackedNibble => {
            packed.resize(encoding.doc_packed_length(scratch.len()), 0);
            quantization::pack_nibbles(scratch, packed);
        }
        ScalarEncoding::SingleBitQueryNibble => {
            packed.resize(encoding.doc_packed_length(scratch.len()), 0);
            quantization::pack_as_binary(scratch, packed);
        }
        ScalarEncoding::DibitQueryNibble => {
            packed.resize(encoding.doc_packed_length(scratch.len()), 0);
            quantization::transpose_dibit(scratch, packed);
        }
    }
}

fn write_corrections(out: &mut Vec<u8>, c: &QuantizationResult) {
    out.write_i32(c.lower_interval.to_bits() as i32);
    out.write_i32(c.upper_interval.to_bits() as i32);
    out.write_i32(c.additional_correction.to_bits() as i32);
    out.write_i32(c.quantized_component_sum);
}

/// `IndexOutput.alignFilePointer(alignment)`: zero-pads `out` to a multiple
/// of `alignment` and returns the new length.
fn align(out: &mut Vec<u8>, alignment: usize) -> usize {
    let aligned = out.len().next_multiple_of(alignment);
    out.resize(aligned, 0);
    aligned
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// One `FLOAT32` field to flush: its raw vectors in document order.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedVectorsField<'a> {
    pub field_number: i32,
    pub similarity: VectorSimilarityFunction,
    pub dimension: i32,
    /// Strictly ascending document ids, one per vector.
    pub docs: &'a [i32],
    /// `docs.len() * dimension` components, the vectors as indexed (before
    /// any COSINE normalization -- this writer normalizes, as Java's does).
    pub vectors: &'a [f32],
}

/// One segment being merged, for one field.
#[derive(Debug, Clone)]
pub struct QuantizedMergeSource<'a> {
    /// The source segment's quantized centroid as Java's `getCentroid` sees
    /// it. `None` when the source has no quantized entry for the field -- and
    /// **always `None` under `Lucene104HnswScalarQuantizedVectorsFormat`**:
    /// `getCentroid` unwraps the per-field reader to the field's reader, a
    /// `Lucene99HnswVectorsReader` wrapping the quantized one, and only
    /// recognises a bare `Lucene104ScalarQuantizedVectorsReader`, so the
    /// merged centroid is recomputed from every source vector
    /// (`tests/scalar_quantized_fixtures.rs` pins this against Lucene).
    pub centroid: Option<&'a [f32]>,
    /// Every vector the source holds for the field, deleted ones included,
    /// in its ordinal order (Java's `calculateCentroid` iterates them all).
    pub vectors: &'a [f32],
    /// Whether the source has deletions (Java's `mergeState.liveDocs[i] != null`),
    /// which forces the centroid to be recomputed from the live vectors.
    pub has_deletions: bool,
}

/// Port of `Lucene104ScalarQuantizedVectorsWriter` (the quantized half; the
/// raw vectors are the caller's [`crate::vectors::FlatVectorsWriter`]).
#[derive(Debug)]
pub struct ScalarQuantizedVectorsWriter {
    encoding: ScalarEncoding,
    max_doc: i32,
    meta: Vec<u8>,
    data: Vec<u8>,
}

impl ScalarQuantizedVectorsWriter {
    /// The constructor: both headers written.
    pub fn new(
        encoding: ScalarEncoding,
        max_doc: i32,
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
    ) -> Self {
        let mut meta = Vec::new();
        codec_util::write_index_header(
            &mut meta,
            META_CODEC,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        );
        let mut data = Vec::new();
        codec_util::write_index_header(
            &mut data,
            DATA_CODEC,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        );
        ScalarQuantizedVectorsWriter {
            encoding,
            max_doc,
            meta,
            data,
        }
    }

    fn check_field(
        field_number: i32,
        dimension: i32,
        docs: &[i32],
        components: usize,
        max_doc: i32,
    ) -> Result<usize> {
        if dimension <= 0 {
            return Err(Error::DimensionMismatch(field_number, 1, dimension));
        }
        let dim = dimension as usize;
        if docs.len().checked_mul(dim) != Some(components) {
            return Err(Error::DimensionMismatch(field_number, dimension, -1));
        }
        if docs.windows(2).any(|w| w[0] >= w[1]) || docs.iter().any(|&d| d < 0 || d >= max_doc) {
            return Err(Error::InvalidGraphParameter(format!(
                "field {field_number}: documents must be strictly ascending in 0..{max_doc}"
            )));
        }
        Ok(dim)
    }

    /// `flush` + `writeField` for one field (no index sort).
    pub fn write_field(&mut self, field: &QuantizedVectorsField<'_>) -> Result<()> {
        let dim = Self::check_field(
            field.field_number,
            field.dimension,
            field.docs,
            field.vectors.len(),
            self.max_doc,
        )?;
        let count = field.docs.len();
        let cosine = field.similarity == VectorSimilarityFunction::Cosine;
        // FieldWriter.addValue: the running per-dimension sums (of the
        // normalized vector for COSINE) and each COSINE vector's magnitude.
        let mut dimension_sums = vec![0f32; dim];
        let mut vectors: Vec<f32> = field.vectors.to_vec();
        for v in vectors.chunks_exact_mut(dim) {
            if cosine {
                let dp = vector_util_dot(v, v);
                let divisor = (dp as f64).sqrt() as f32;
                for (s, &x) in dimension_sums.iter_mut().zip(v.iter()) {
                    *s += x / divisor;
                }
                // FieldWriter.normalizeVectors, applied at flush.
                for x in v.iter_mut() {
                    *x /= divisor;
                }
            } else {
                for (s, &x) in dimension_sums.iter_mut().zip(v.iter()) {
                    *s += x;
                }
            }
        }
        let mut centroid = vec![0f32; dim];
        if count > 0 {
            for (c, &s) in centroid.iter_mut().zip(&dimension_sums) {
                *c = s / count as f32;
            }
            if cosine {
                // Java's l2normalize throws on a zero centroid.
                vector_util::l2normalize(&mut centroid, true)
                    .map_err(|e| Error::CorruptMeta(e.to_string()))?;
            }
        }
        let quantizer = OptimizedScalarQuantizer::new(util_similarity(field.similarity));
        let offset = align(&mut self.data, 4);
        let mut scratch = vec![0u8; self.encoding.discrete_dimensions(dim)];
        let mut packed = Vec::new();
        for v in vectors.chunks_exact_mut(dim) {
            let corrections =
                quantizer.scalar_quantize(v, &mut scratch, self.encoding.bits(), &centroid);
            pack_codes(self.encoding, &scratch, &mut packed);
            self.data.extend_from_slice(&packed);
            write_corrections(&mut self.data, &corrections);
        }
        // ARITH: `offset` was `data.len()` when the loop began and `data` only
        // grows.
        #[allow(clippy::arithmetic_side_effects)]
        let length = self.data.len() - offset;
        let centroid_dp = if count > 0 {
            vector_util_dot(&centroid, &centroid)
        } else {
            0.0
        };
        self.write_meta(
            field.field_number,
            field.similarity,
            dim,
            offset,
            length,
            &centroid,
            centroid_dp,
            field.docs,
        );
        Ok(())
    }

    /// `mergeOneFlatVectorField`'s quantized half: `merged` is the merged
    /// field's live vectors in merged-ordinal order (raw, not normalized) and
    /// `docs` their merged document ids; `sources` describe the segments
    /// they came from, for the centroid.
    pub fn merge_field(
        &mut self,
        field_number: i32,
        similarity: VectorSimilarityFunction,
        dimension: i32,
        sources: &[QuantizedMergeSource<'_>],
        docs: &[i32],
        merged: &[f32],
    ) -> Result<()> {
        let dim = Self::check_field(field_number, dimension, docs, merged.len(), self.max_doc)?;
        let cosine = similarity == VectorSimilarityFunction::Cosine;
        let mut centroid = vec![0f32; dim];
        merge_and_recalculate_centroids(sources, cosine, dim, &mut centroid)?;
        let quantizer = OptimizedScalarQuantizer::new(util_similarity(similarity));
        let offset = align(&mut self.data, 4);
        let mut scratch = vec![0u8; self.encoding.discrete_dimensions(dim)];
        let mut packed = Vec::new();
        let mut v = vec![0f32; dim];
        for raw in merged.chunks_exact(dim) {
            v.copy_from_slice(raw);
            if cosine {
                // NormalizedFloatVectorValues.vectorValue
                vector_util::l2normalize(&mut v, true)
                    .map_err(|e| Error::CorruptMeta(e.to_string()))?;
            }
            let corrections =
                quantizer.scalar_quantize(&mut v, &mut scratch, self.encoding.bits(), &centroid);
            pack_codes(self.encoding, &scratch, &mut packed);
            self.data.extend_from_slice(&packed);
            write_corrections(&mut self.data, &corrections);
        }
        // ARITH: as in `write_field`.
        #[allow(clippy::arithmetic_side_effects)]
        let length = self.data.len() - offset;
        let centroid_dp = if docs.is_empty() {
            0.0
        } else {
            vector_util_dot(&centroid, &centroid)
        };
        self.write_meta(
            field_number,
            similarity,
            dim,
            offset,
            length,
            &centroid,
            centroid_dp,
            docs,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_meta(
        &mut self,
        field_number: i32,
        similarity: VectorSimilarityFunction,
        dim: usize,
        offset: usize,
        length: usize,
        centroid: &[f32],
        centroid_dp: f32,
        docs: &[i32],
    ) {
        let meta = &mut self.meta;
        meta.write_i32(field_number);
        meta.write_i32(encoding_ordinal(VectorEncoding::Float32));
        meta.write_i32(similarity_ordinal(similarity));
        meta.write_vint(dim as i32);
        meta.write_vlong(offset as i64);
        meta.write_vlong(length as i64);
        meta.write_vint(docs.len() as i32);
        if !docs.is_empty() {
            meta.write_vint(self.encoding.wire_number());
            for &c in centroid {
                meta.write_i32(c.to_bits() as i32);
            }
            meta.write_i32(centroid_dp.to_bits() as i32);
        }
        write_stored_meta(meta, &mut self.data, docs, self.max_doc);
    }

    /// `finish()`: the end-of-fields marker and both footers. Returns
    /// `(veq, vemq)`.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
        self.meta.write_i32(-1);
        codec_util::write_footer(&mut self.meta);
        codec_util::write_footer(&mut self.data);
        (self.data, self.meta)
    }
}

/// `VectorUtil.dotProduct(float[], float[])`.
fn vector_util_dot(a: &[f32], b: &[f32]) -> f32 {
    lucene_util::simd::dot_f32(a, b)
}

/// `Lucene104ScalarQuantizedVectorsWriter.mergeAndRecalculateCentroids` (+
/// `calculateCentroid`): the vector-count-weighted mean of the sources'
/// centroids, or -- when a source has deletions or no quantized centroid --
/// the mean of every source vector (deleted ones included, as Java's
/// `calculateCentroid` iterates the readers' values without live docs).
fn merge_and_recalculate_centroids(
    sources: &[QuantizedMergeSource<'_>],
    cosine: bool,
    dim: usize,
    centroid: &mut [f32],
) -> Result<usize> {
    let mut recalculate = false;
    let mut total = 0usize;
    for s in sources {
        let (Some(0), Some(vector_count)) = (
            s.vectors.len().checked_rem(dim),
            s.vectors.len().checked_div(dim),
        ) else {
            return Err(Error::DimensionMismatch(-1, dim as i32, -1));
        };
        if vector_count == 0 {
            continue;
        }
        total = total.saturating_add(vector_count);
        let Some(c) = s.centroid.filter(|_| !s.has_deletions) else {
            recalculate = true;
            break;
        };
        if c.len() != dim {
            return Err(Error::DimensionMismatch(-1, dim as i32, c.len() as i32));
        }
        for (m, &x) in centroid.iter_mut().zip(c) {
            *m += x * vector_count as f32;
        }
    }
    if total == 0 {
        return Ok(0);
    }
    if recalculate {
        // calculateCentroid: over every source vector, reader by reader.
        centroid.fill(0.0);
        let mut count = 0usize;
        for s in sources {
            for v in s.vectors.chunks_exact(dim) {
                count = count.saturating_add(1);
                for (c, &x) in centroid.iter_mut().zip(v) {
                    *c += x;
                }
            }
        }
        if count == 0 {
            return Ok(0);
        }
        for c in centroid.iter_mut() {
            *c /= count as f32;
        }
        if cosine {
            vector_util::l2normalize(centroid, true)
                .map_err(|e| Error::CorruptMeta(e.to_string()))?;
        }
        return Ok(count);
    }
    for c in centroid.iter_mut() {
        *c /= total as f32;
    }
    if cosine {
        vector_util::l2normalize(centroid, true).map_err(|e| Error::CorruptMeta(e.to_string()))?;
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// One field's `.vemq` entry (Java's `FieldEntry`).
#[derive(Debug, Clone)]
pub struct QuantizedFieldEntry {
    pub field_number: i32,
    pub similarity: VectorSimilarityFunction,
    pub vector_encoding: VectorEncoding,
    pub dimension: i32,
    pub vector_data_offset: i64,
    pub vector_data_length: i64,
    pub size: i32,
    pub scalar_encoding: ScalarEncoding,
    /// `None` when the field has no vectors.
    pub centroid: Option<Vec<f32>>,
    pub centroid_dp: f32,
    pub ord_to_doc: OrdToDoc,
}

/// Port of `Lucene104ScalarQuantizedVectorsReader` (the quantized half; the
/// raw vectors are a [`crate::vectors::FlatVectorsReader`] over `.vec`).
#[derive(Debug, Clone)]
pub struct ScalarQuantizedVectorsReader<'a> {
    data: &'a [u8],
    fields: Vec<QuantizedFieldEntry>,
}

impl<'a> ScalarQuantizedVectorsReader<'a> {
    /// The constructor: `meta_buf` is `.vemq`, `data_buf` is `.veq`.
    pub fn open(
        meta_buf: &[u8],
        data_buf: &'a [u8],
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
    ) -> Result<Self> {
        let mut meta = SliceInput::new(meta_buf);
        let meta_version = codec_util::check_index_header(
            &mut meta,
            META_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?
        .version;
        let mut data_in = SliceInput::new(data_buf);
        let data_version = codec_util::check_index_header(
            &mut data_in,
            DATA_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?
        .version;
        if meta_version != data_version {
            return corrupt(format!(
                "Format versions mismatch: meta={meta_version}, {DATA_CODEC}={data_version}"
            ));
        }
        let (Some(meta_footer), Some(data_footer)) = (
            meta_buf.len().checked_sub(codec_util::FOOTER_LENGTH),
            data_buf.len().checked_sub(codec_util::FOOTER_LENGTH),
        ) else {
            return corrupt("quantized vector files shorter than a codec footer");
        };
        codec_util::check_whole_file_footer(meta_buf, meta_footer)?;
        codec_util::check_whole_file_footer(data_buf, data_footer)?;
        let mut fields = Vec::new();
        loop {
            let field_number = meta.read_i32()?;
            if field_number == -1 {
                break;
            }
            if field_number < 0 {
                return corrupt(format!("Invalid field number: {field_number}"));
            }
            fields.push(read_field(&mut meta, field_number, data_buf.len())?);
        }
        Ok(ScalarQuantizedVectorsReader {
            data: data_buf,
            fields,
        })
    }

    /// Every field's entry.
    pub fn fields(&self) -> &[QuantizedFieldEntry] {
        &self.fields
    }

    /// One field's entry.
    pub fn field(&self, field_number: i32) -> Option<&QuantizedFieldEntry> {
        self.fields.iter().find(|f| f.field_number == field_number)
    }

    /// `getCentroid(field)`.
    pub fn centroid(&self, field_number: i32) -> Option<&[f32]> {
        self.field(field_number).and_then(|f| f.centroid.as_deref())
    }

    /// `getQuantizedVectorValues(field)` -> `OffHeapScalarQuantizedVectorValues.load`.
    pub fn quantized_vector_values(&self, field_number: i32) -> Result<QuantizedVectorValues<'a>> {
        let entry = self
            .field(field_number)
            .ok_or(Error::UnknownField(field_number))?;
        if entry.vector_encoding != VectorEncoding::Float32 {
            return Err(Error::EncodingMismatch(
                field_number,
                entry.vector_encoding,
                VectorEncoding::Float32,
            ));
        }
        let dim = entry.dimension as usize;
        let packed_len = entry.scalar_encoding.doc_packed_length(dim);
        // ARITH: `read_field` proved `(packed_len + 16) * size ==
        // vectorDataLength`, which is inside the file.
        #[allow(clippy::arithmetic_side_effects)]
        let byte_size = packed_len + CORRECTIONS_BYTES;
        let slice = if entry.size == 0 {
            &[][..]
        } else {
            file_region(
                self.data,
                entry.vector_data_offset,
                entry.vector_data_length,
            )
            .ok_or_else(|| Error::CorruptMeta("quantized vector data out of bounds".into()))?
        };
        Ok(QuantizedVectorValues {
            slice,
            file: self.data,
            dimension: dim,
            size: entry.size,
            encoding: entry.scalar_encoding,
            similarity: entry.similarity,
            centroid: entry.centroid.clone().unwrap_or_default(),
            centroid_dp: entry.centroid_dp,
            ord_to_doc: entry.ord_to_doc.clone(),
            packed_len,
            byte_size,
        })
    }
}

/// `readField` + `FieldEntry.create` + `validateFieldEntry`'s length identity.
fn read_field(
    meta: &mut SliceInput<'_>,
    field_number: i32,
    data_len: usize,
) -> Result<QuantizedFieldEntry> {
    let vector_encoding = read_vector_encoding(meta)?;
    let similarity = read_similarity_function(meta)?;
    let dimension = meta.read_vint()?;
    let vector_data_offset = meta.read_vlong()?;
    let vector_data_length = meta.read_vlong()?;
    let size = meta.read_vint()?;
    if dimension <= 0 {
        return corrupt(format!("illegal vector dimension {dimension}"));
    }
    if size < 0 || vector_data_offset < 0 || vector_data_length < 0 {
        return corrupt(format!(
            "illegal quantized vector region: size={size} [{vector_data_offset}, +{vector_data_length})"
        ));
    }
    let mut scalar_encoding = ScalarEncoding::UnsignedByte;
    let mut centroid = None;
    let mut centroid_dp = 0f32;
    if size > 0 {
        let wire = meta.read_vint()?;
        scalar_encoding = ScalarEncoding::from_wire_number(wire).ok_or_else(|| {
            Error::CorruptMeta(format!(
                "Could not get ScalarEncoding from wire number: {wire}"
            ))
        })?;
        // Four bytes per component must be there before they are reserved.
        let dim = dimension as usize;
        if dim > meta.remaining() / 4 {
            return Err(Error::Store(lucene_store::Error::Eof {
                offset: meta.position(),
            }));
        }
        let mut c = Vec::with_capacity(dim);
        for _ in 0..dim {
            c.push(f32::from_bits(meta.read_i32()? as u32));
        }
        centroid = Some(c);
        centroid_dp = f32::from_bits(meta.read_i32()? as u32);
    }
    let ord_to_doc = OrdToDoc::from_stored_meta(meta, size)?;
    // validateFieldEntry: Math.multiplyExact((packedLength + 16), size).
    let per_vector = (scalar_encoding.doc_packed_length(dimension as usize) as i64)
        .checked_add(CORRECTIONS_BYTES as i64);
    let expected = per_vector.and_then(|p| p.checked_mul(size as i64));
    if expected != Some(vector_data_length) {
        return corrupt(format!(
            "vector data length {vector_data_length} not matching size = {size} * (dims={dimension} + 16)"
        ));
    }
    if file_region_fits(vector_data_offset, vector_data_length, data_len).is_none() {
        return corrupt(format!(
            "quantized vector data [{vector_data_offset}, +{vector_data_length}) past the end of a {data_len} byte .veq"
        ));
    }
    Ok(QuantizedFieldEntry {
        field_number,
        similarity,
        vector_encoding,
        dimension,
        vector_data_offset,
        vector_data_length,
        size,
        scalar_encoding,
        centroid,
        centroid_dp,
        ord_to_doc,
    })
}

fn file_region_fits(offset: i64, length: i64, len: usize) -> Option<()> {
    let start = u64::try_from(offset).ok()?;
    let end = start.checked_add(u64::try_from(length).ok()?)?;
    (end <= len as u64).then_some(())
}

/// Port of `OffHeapScalarQuantizedVectorValues`: a field's quantized vectors,
/// each `(codes, corrective terms)`, addressed by ordinal.
#[derive(Debug, Clone)]
pub struct QuantizedVectorValues<'a> {
    slice: &'a [u8],
    file: &'a [u8],
    dimension: usize,
    size: i32,
    encoding: ScalarEncoding,
    similarity: VectorSimilarityFunction,
    centroid: Vec<f32>,
    centroid_dp: f32,
    ord_to_doc: OrdToDoc,
    packed_len: usize,
    byte_size: usize,
}

impl<'a> QuantizedVectorValues<'a> {
    /// `dimension()`.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// `size()`.
    pub fn size(&self) -> i32 {
        self.size
    }

    /// `getScalarEncoding()`.
    pub fn encoding(&self) -> ScalarEncoding {
        self.encoding
    }

    /// The field's similarity.
    pub fn similarity(&self) -> VectorSimilarityFunction {
        self.similarity
    }

    /// `getCentroid()` (empty for a field with no vectors).
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    /// `getCentroidDP()`.
    pub fn centroid_dp(&self) -> f32 {
        self.centroid_dp
    }

    fn record(&self, ord: i32) -> Result<&'a [u8]> {
        if ord < 0 || ord >= self.size {
            return Err(Error::OrdOutOfRange(ord, self.size));
        }
        // ARITH: `0 <= ord < size` and `slice.len() == size * byte_size`
        // (checked in `read_field`), so neither product nor sum overflows.
        #[allow(clippy::arithmetic_side_effects)]
        let start = ord as usize * self.byte_size;
        // ARITH: as above.
        #[allow(clippy::arithmetic_side_effects)]
        let end = start + self.byte_size;
        self.slice
            .get(start..end)
            .ok_or(Error::OrdOutOfRange(ord, self.size))
    }

    /// `vectorValue(ord)`: the packed codes.
    pub fn vector(&self, ord: i32) -> Result<&'a [u8]> {
        Ok(&self.record(ord)?[..self.packed_len])
    }

    /// `getCorrectiveTerms(ord)`.
    pub fn corrective_terms(&self, ord: i32) -> Result<QuantizationResult> {
        let rec = self.record(ord)?;
        let word = |i: usize| {
            // ARITH: `i <= 3` and the record is `packed_len + 16` bytes, so
            // `at + 4 <= rec.len()`.
            #[allow(clippy::arithmetic_side_effects)]
            let (at, end) = (self.packed_len + 4 * i, self.packed_len + 4 * i + 4);
            u32::from_le_bytes(rec[at..end].try_into().expect("4 bytes"))
        };
        Ok(QuantizationResult {
            lower_interval: f32::from_bits(word(0)),
            upper_interval: f32::from_bits(word(1)),
            additional_correction: f32::from_bits(word(2)),
            quantized_component_sum: word(3) as i32,
        })
    }

    /// `ordToDoc(ord)`.
    pub fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        if ord < 0 || ord >= self.size {
            return Err(Error::OrdOutOfRange(ord, self.size));
        }
        match &self.ord_to_doc {
            OrdToDoc::Empty | OrdToDoc::Dense => Ok(ord),
            // `ord < size` above; `get`, so a table shorter than `size` is an
            // error rather than a panic.
            OrdToDoc::Explicit(docs) => match docs.get(ord as usize) {
                Some(&doc) => Ok(doc),
                None => Err(Error::OrdOutOfRange(ord, self.size)),
            },
            OrdToDoc::Sparse {
                addresses_offset,
                addresses_length,
                meta,
                ..
            } => {
                let region = file_region(self.file, *addresses_offset, *addresses_length)
                    .ok_or_else(|| {
                        Error::CorruptMeta("ordToDoc addresses region out of bounds".into())
                    })?;
                Ok(direct_monotonic::get(region, meta, ord as i64)? as i32)
            }
        }
    }

    /// The doc -> ordinal direction (`iterator()` / `getAcceptOrds`).
    pub fn doc_to_ord(&self) -> Result<DocToOrdCursor<'a>> {
        match &self.ord_to_doc {
            OrdToDoc::Empty => Ok(DocToOrdCursor::Empty),
            OrdToDoc::Dense => Ok(DocToOrdCursor::Dense { size: self.size }),
            // Only the retired Lucene90/91 HNSW readers build an explicit
            // table; no scalar-quantized field entry carries one.
            OrdToDoc::Explicit(_) => Err(Error::CorruptMeta(
                "explicit ordToDoc table in a scalar-quantized field".into(),
            )),
            OrdToDoc::Sparse {
                docs_with_field_offset,
                docs_with_field_length,
                jump_table_entry_count,
                dense_rank_power,
                ..
            } => {
                let region =
                    file_region(self.file, *docs_with_field_offset, *docs_with_field_length)
                        .ok_or_else(|| {
                            Error::CorruptMeta("docsWithField region out of bounds".into())
                        })?;
                Ok(DocToOrdCursor::Sparse(Box::new(DisiCursor::new(
                    region,
                    *dense_rank_power,
                    *jump_table_entry_count,
                ))))
            }
        }
    }

    /// `OffHeapScalarQuantizedFloatVectorValues.vectorValue(ord)`: the
    /// dequantized vector (centroid re-added) into `out`.
    pub fn dequantized(&self, ord: i32, out: &mut [f32]) -> Result<()> {
        let codes = self.vector(ord)?;
        let c = self.corrective_terms(ord)?;
        let bits = self.encoding.bits();
        let mut unpacked;
        let values: &[u8] = match self.encoding {
            ScalarEncoding::UnsignedByte | ScalarEncoding::SevenBit => codes,
            ScalarEncoding::PackedNibble => {
                unpacked = vec![0u8; codes.len().saturating_mul(2)];
                quantization::unpack_nibbles(codes, &mut unpacked);
                &unpacked
            }
            ScalarEncoding::SingleBitQueryNibble => {
                unpacked = vec![0u8; self.dimension];
                quantization::unpack_binary(codes, &mut unpacked);
                &unpacked
            }
            ScalarEncoding::DibitQueryNibble => {
                unpacked = vec![0u8; self.dimension];
                quantization::untranspose_dibit(codes, &mut unpacked);
                &unpacked
            }
        };
        let n = self.dimension.min(values.len()).min(out.len());
        quantization::dequantize(
            &values[..n],
            &mut out[..n],
            bits,
            c.lower_interval,
            c.upper_interval,
            &self.centroid,
        );
        Ok(())
    }
}

/// `Lucene104ScalarQuantizedVectorScorer.SCALE_LUT`: `1 / (2^bits - 1)`.
const SCALE_LUT: [f32; 8] = [
    1.0,
    1.0 / 3.0,
    1.0 / 7.0,
    1.0 / 15.0,
    1.0 / 31.0,
    1.0 / 63.0,
    1.0 / 127.0,
    1.0 / 255.0,
];

/// `Lucene104ScalarQuantizedVectorScorer.quantizedScore`.
pub fn quantized_score(
    query: &[u8],
    query_corrections: &QuantizationResult,
    targets: &QuantizedVectorValues<'_>,
    target_ord: i32,
    similarity: VectorSimilarityFunction,
) -> Result<f32> {
    let doc = targets.vector(target_ord)?;
    let qc_dist = match targets.encoding {
        ScalarEncoding::UnsignedByte => vector_util::uint8_dot_product(query, doc) as f32,
        ScalarEncoding::SevenBit => vector_util::dot_product_i8(query, doc) as f32,
        ScalarEncoding::PackedNibble => {
            vector_util::int4_dot_product_single_packed(query, doc) as f32
        }
        ScalarEncoding::SingleBitQueryNibble => {
            vector_util::int4_bit_dot_product(query, doc) as f32
        }
        ScalarEncoding::DibitQueryNibble => vector_util::int4_dibit_dot_product(query, doc) as f32,
    };
    let index = targets.corrective_terms(target_ord)?;
    Ok(score_from_parts(
        qc_dist,
        query_corrections,
        &index,
        targets,
        similarity,
    ))
}

#[inline]
fn score_from_parts(
    qc_dist: f32,
    q: &QuantizationResult,
    index: &QuantizationResult,
    targets: &QuantizedVectorValues<'_>,
    similarity: VectorSimilarityFunction,
) -> f32 {
    // `bits` is 1..=8 for every encoding, so the LUT lookups are in range.
    let query_scale = SCALE_LUT[(targets.encoding.query_bits() as usize).saturating_sub(1)];
    let scale = SCALE_LUT[(targets.encoding.bits() as usize).saturating_sub(1)];
    let x1 = index.quantized_component_sum as f32;
    let ax = index.lower_interval;
    let lx = (index.upper_interval - ax) * scale;
    let ay = q.lower_interval;
    let ly = (q.upper_interval - ay) * query_scale;
    let y1 = q.quantized_component_sum as f32;
    let mut score =
        ax * ay * targets.dimension as i32 as f32 + ay * lx * x1 + ax * ly * y1 + lx * ly * qc_dist;
    if similarity == VectorSimilarityFunction::Euclidean {
        score = q.additional_correction + index.additional_correction - 2.0 * score;
        return 1.0 / (1.0 + java_max(score, 0.0));
    }
    score += q.additional_correction + index.additional_correction - targets.centroid_dp;
    if similarity == VectorSimilarityFunction::MaximumInnerProduct {
        return crate::vectors::scale_max_inner_product_score(score);
    }
    // Math.clamp(score, -1, 1)
    let score = if score.is_nan() {
        score
    } else {
        score.clamp(-1.0, 1.0)
    };
    (1.0 + score) / 2.0
}

/// `Math.max(float, float)`: a NaN operand wins.
fn java_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else {
        a.max(b)
    }
}

/// `Lucene104ScalarQuantizedVectorScorer.getRandomVectorScorer(sim, qv, float[] target)`:
/// the query quantized once (at the encoding's query width), scored against
/// every stored vector.
#[derive(Debug, Clone)]
pub struct QuantizedQueryScorer<'v, 'a> {
    values: &'v QuantizedVectorValues<'a>,
    query: Vec<u8>,
    corrections: QuantizationResult,
    similarity: VectorSimilarityFunction,
}

impl<'v, 'a> QuantizedQueryScorer<'v, 'a> {
    /// Quantizes `target` against the field's centroid.
    pub fn new(
        values: &'v QuantizedVectorValues<'a>,
        similarity: VectorSimilarityFunction,
        target: &[f32],
    ) -> Result<Self> {
        if target.len() != values.dimension {
            return Err(Error::QueryDimensionMismatch(
                target.len() as i32,
                values.dimension as i32,
            ));
        }
        let encoding = values.encoding;
        let mut scratch = vec![0u8; encoding.discrete_dimensions(values.dimension)];
        let mut copy = target.to_vec();
        if similarity == VectorSimilarityFunction::Cosine {
            vector_util::l2normalize(&mut copy, true)
                .map_err(|e| Error::CorruptMeta(e.to_string()))?;
        }
        let quantizer = OptimizedScalarQuantizer::new(util_similarity(similarity));
        let corrections = quantizer.scalar_quantize(
            &mut copy,
            &mut scratch,
            encoding.query_bits(),
            &values.centroid,
        );
        let query = if encoding == ScalarEncoding::SingleBitQueryNibble
            || encoding == ScalarEncoding::DibitQueryNibble
        {
            let mut t = vec![0u8; encoding.query_packed_length(scratch.len())];
            quantization::transpose_half_byte(&scratch, &mut t);
            t
        } else {
            scratch
        };
        Ok(QuantizedQueryScorer {
            values,
            query,
            corrections,
            similarity,
        })
    }

    /// The quantized query (for tests and diagnostics).
    pub fn query(&self) -> (&[u8], &QuantizationResult) {
        (&self.query, &self.corrections)
    }
}

impl crate::hnsw::VectorScorer for QuantizedQueryScorer<'_, '_> {
    fn score(&mut self, node: i32) -> Result<f32> {
        quantized_score(
            &self.query,
            &self.corrections,
            self.values,
            node,
            self.similarity,
        )
    }

    fn max_ord(&self) -> i32 {
        self.values.size
    }
}

/// The merge-time graph scorer: `ScalarQuantizedVectorScorerSupplier` for a
/// symmetric encoding (a stored vector scores the others), or
/// `AsymmetricQuantizedRandomVectorScorerSupplier` over the field's vectors
/// re-quantized at the 4-bit query width (`writeBinarizedQueryData`).
#[derive(Debug, Clone)]
pub struct QuantizedOrdScorer<'v, 'a> {
    values: &'v QuantizedVectorValues<'a>,
    similarity: VectorSimilarityFunction,
    /// Asymmetric only: each ordinal's query-side codes and corrections.
    query_side: Option<&'v [(Vec<u8>, QuantizationResult)]>,
    target: Vec<u8>,
    target_corrections: Option<QuantizationResult>,
}

/// Builds the query-side vectors an asymmetric merge scores with:
/// `writeBinarizedQueryData` over the merged raw vectors (`vectors`, one per
/// ordinal).
pub fn binarized_query_data(
    values: &QuantizedVectorValues<'_>,
    similarity: VectorSimilarityFunction,
    vectors: &[f32],
) -> Result<Vec<(Vec<u8>, QuantizationResult)>> {
    let encoding = values.encoding;
    if !encoding.is_asymmetric() {
        return Err(Error::InvalidGraphParameter(
            "encoding and queryEncoding must be different".into(),
        ));
    }
    let dim = values.dimension;
    let quantizer = OptimizedScalarQuantizer::new(util_similarity(similarity));
    let discrete = encoding.discrete_dimensions(dim);
    let mut scratch = vec![0u8; discrete];
    let mut out = Vec::new();
    let mut v = vec![0f32; dim];
    for raw in vectors.chunks_exact(dim) {
        v.copy_from_slice(raw);
        let r = quantizer.scalar_quantize(
            &mut v,
            &mut scratch,
            encoding.query_bits(),
            &values.centroid,
        );
        let mut to_query = vec![0u8; encoding.query_packed_length(discrete)];
        quantization::transpose_half_byte(&scratch, &mut to_query);
        out.push((to_query, r));
    }
    Ok(out)
}

impl<'v, 'a> QuantizedOrdScorer<'v, 'a> {
    /// A symmetric-encoding merge scorer.
    pub fn symmetric(
        values: &'v QuantizedVectorValues<'a>,
        similarity: VectorSimilarityFunction,
    ) -> Result<Self> {
        if values.encoding.is_asymmetric() {
            return Err(Error::InvalidGraphParameter(format!(
                "{:?} encoding is not supported for symmetric quantization",
                values.encoding
            )));
        }
        Ok(QuantizedOrdScorer {
            values,
            similarity,
            query_side: None,
            target: Vec::new(),
            target_corrections: None,
        })
    }

    /// An asymmetric-encoding merge scorer over [`binarized_query_data`].
    pub fn asymmetric(
        values: &'v QuantizedVectorValues<'a>,
        similarity: VectorSimilarityFunction,
        query_side: &'v [(Vec<u8>, QuantizationResult)],
    ) -> Self {
        QuantizedOrdScorer {
            values,
            similarity,
            query_side: Some(query_side),
            target: Vec::new(),
            target_corrections: None,
        }
    }
}

impl crate::hnsw::VectorScorer for QuantizedOrdScorer<'_, '_> {
    fn score(&mut self, node: i32) -> Result<f32> {
        let Some(c) = self.target_corrections else {
            return Err(Error::InvalidGraphParameter(
                "setScoringOrdinal was not called".into(),
            ));
        };
        quantized_score(&self.target, &c, self.values, node, self.similarity)
    }

    fn max_ord(&self) -> i32 {
        self.values.size
    }
}

impl crate::hnsw::UpdateableVectorScorer for QuantizedOrdScorer<'_, '_> {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
        match self.query_side {
            Some(q) => {
                let (codes, c) = usize::try_from(ord)
                    .ok()
                    .and_then(|o| q.get(o))
                    .ok_or(Error::OrdOutOfRange(ord, q.len() as i32))?;
                self.target.clear();
                self.target.extend_from_slice(codes);
                self.target_corrections = Some(*c);
            }
            None => {
                let raw = self.values.vector(ord)?;
                self.target.clear();
                match self.values.encoding {
                    ScalarEncoding::PackedNibble => {
                        self.target
                            .resize(quantization::discretize(self.values.dimension, 2), 0);
                        quantization::unpack_nibbles(raw, &mut self.target);
                    }
                    _ => self.target.extend_from_slice(raw),
                }
                self.target_corrections = Some(self.values.corrective_terms(ord)?);
            }
        }
        Ok(())
    }
}

/// Convenience: the merge scorer for `values`, choosing the symmetric or
/// asymmetric supplier as `getRandomVectorScorerSupplierForMerge` does.
/// `query_side` must be [`binarized_query_data`]'s output for an asymmetric
/// encoding.
pub fn merge_scorer_supplier<'v, 'a>(
    values: &'v QuantizedVectorValues<'a>,
    similarity: VectorSimilarityFunction,
    query_side: Option<&'v [(Vec<u8>, QuantizationResult)]>,
) -> Result<QuantizedOrdScorer<'v, 'a>> {
    match (values.encoding.is_asymmetric(), query_side) {
        (false, _) => QuantizedOrdScorer::symmetric(values, similarity),
        (true, Some(q)) => Ok(QuantizedOrdScorer::asymmetric(values, similarity, q)),
        (true, None) => Err(Error::InvalidGraphParameter(
            "an asymmetric encoding needs its query-side vectors".into(),
        )),
    }
}

/// `Lucene104ScalarQuantizedVectorsReader.search(field, float[] target, ...)`'s
/// exhaustive branch over one field: every accepted ordinal scored with the
/// quantized query, bulk-scored 64 at a time. Returns `(ord, score)` best
/// first. The graph walk is [`crate::hnsw_vectors::search`] with a
/// [`QuantizedQueryScorer`].
pub fn exhaustive_search(
    values: &QuantizedVectorValues<'_>,
    similarity: VectorSimilarityFunction,
    target: &[f32],
    k: usize,
    accept_ords: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
) -> Result<Vec<(i32, f32)>> {
    let mut scorer = QuantizedQueryScorer::new(values, similarity, target)?;
    let options = crate::hnsw_vectors::SearchOptions {
        accept_ords,
        ..Default::default()
    };
    let (hits, _) = crate::hnsw_vectors::search::<crate::hnsw::OnHeapHnswGraph, _>(
        &mut scorer,
        None,
        k,
        u64::MAX,
        options,
    )?;
    Ok(hits)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::hnsw::{UpdateableVectorScorer, VectorScorer};

    const ID: [u8; 16] = *b"quantizedvec0001";

    fn vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n * dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn write(
        encoding: ScalarEncoding,
        sim: VectorSimilarityFunction,
        docs: &[i32],
        v: &[f32],
        dim: i32,
        max_doc: i32,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut w = ScalarQuantizedVectorsWriter::new(encoding, max_doc, &ID, "suf");
        w.write_field(&QuantizedVectorsField {
            field_number: 3,
            similarity: sim,
            dimension: dim,
            docs,
            vectors: v,
        })
        .unwrap();
        w.finish()
    }

    #[test]
    fn round_trip_every_encoding_and_similarity() {
        let dim = 13usize;
        let n = 40usize;
        let v = vectors(n, dim, 9);
        let dense: Vec<i32> = (0..n as i32).collect();
        let sparse: Vec<i32> = (0..n as i32).map(|d| d * 3 + 1).collect();
        for encoding in [
            ScalarEncoding::UnsignedByte,
            ScalarEncoding::PackedNibble,
            ScalarEncoding::SevenBit,
            ScalarEncoding::SingleBitQueryNibble,
            ScalarEncoding::DibitQueryNibble,
        ] {
            for sim in [
                VectorSimilarityFunction::Euclidean,
                VectorSimilarityFunction::DotProduct,
                VectorSimilarityFunction::Cosine,
                VectorSimilarityFunction::MaximumInnerProduct,
            ] {
                for (docs, max_doc) in [(&dense, n as i32), (&sparse, 3 * n as i32 + 1)] {
                    let (veq, vemq) = write(encoding, sim, docs, &v, dim as i32, max_doc);
                    let r = ScalarQuantizedVectorsReader::open(&vemq, &veq, &ID, "suf").unwrap();
                    let e = r.field(3).unwrap();
                    assert_eq!(e.scalar_encoding, encoding);
                    assert_eq!(e.size, n as i32);
                    assert!(r.centroid(3).is_some());
                    let values = r.quantized_vector_values(3).unwrap();
                    assert_eq!(values.dimension(), dim);
                    assert_eq!(values.similarity(), sim);
                    assert_eq!(values.encoding(), encoding);
                    assert_eq!(values.centroid().len(), dim);
                    assert!(values.centroid_dp().is_finite());
                    for ord in 0..n as i32 {
                        assert_eq!(values.ord_to_doc(ord).unwrap(), docs[ord as usize]);
                        assert_eq!(
                            values.vector(ord).unwrap().len(),
                            encoding.doc_packed_length(dim)
                        );
                        let c = values.corrective_terms(ord).unwrap();
                        assert!(c.lower_interval <= c.upper_interval);
                    }
                    let mut cursor = values.doc_to_ord().unwrap();
                    assert_eq!(cursor.ordinal(docs[5]).unwrap(), Some(5));
                    // Querying with a stored vector finds it first (or close).
                    let target = &v[7 * dim..8 * dim];
                    let hits = exhaustive_search(&values, sim, target, 5, None).unwrap();
                    assert_eq!(hits.len(), 5);
                    if encoding == ScalarEncoding::UnsignedByte
                        && sim != VectorSimilarityFunction::MaximumInnerProduct
                    {
                        assert_eq!(hits[0].0, 7, "{encoding:?} {sim:?}");
                    }
                    let mut out = vec![0f32; dim];
                    values.dequantized(7, &mut out).unwrap();
                    if encoding == ScalarEncoding::UnsignedByte
                        && sim == VectorSimilarityFunction::Euclidean
                    {
                        for (a, b) in out.iter().zip(target) {
                            assert!((a - b).abs() < 0.02, "{a} vs {b}");
                        }
                    }
                    // Merge scorer.
                    let q = if encoding.is_asymmetric() {
                        Some(binarized_query_data(&values, sim, &v).unwrap())
                    } else {
                        assert!(binarized_query_data(&values, sim, &v).is_err());
                        None
                    };
                    let mut s = merge_scorer_supplier(&values, sim, q.as_deref()).unwrap();
                    assert!(s.score(0).is_err());
                    s.set_scoring_ordinal(2).unwrap();
                    assert!(s.score(3).unwrap().is_finite());
                    assert_eq!(s.max_ord(), n as i32);
                    assert!(s.set_scoring_ordinal(n as i32).is_err());
                }
            }
        }
    }

    #[test]
    fn empty_field_and_bad_inputs() {
        let (veq, vemq) = write(
            ScalarEncoding::UnsignedByte,
            VectorSimilarityFunction::Euclidean,
            &[],
            &[],
            4,
            10,
        );
        let r = ScalarQuantizedVectorsReader::open(&vemq, &veq, &ID, "suf").unwrap();
        let values = r.quantized_vector_values(3).unwrap();
        assert_eq!(values.size(), 0);
        assert!(r.centroid(3).is_none());
        assert!(matches!(
            r.quantized_vector_values(4),
            Err(Error::UnknownField(4))
        ));
        assert!(values.vector(0).is_err());
        assert!(values.ord_to_doc(0).is_err());
        let q = QuantizedQueryScorer::new(&values, VectorSimilarityFunction::Euclidean, &[0.0; 3]);
        assert!(q.is_err());
        assert!(merge_scorer_supplier(&values, VectorSimilarityFunction::Euclidean, None).is_ok());

        let mut w = ScalarQuantizedVectorsWriter::new(ScalarEncoding::UnsignedByte, 10, &ID, "");
        let bad = QuantizedVectorsField {
            field_number: 0,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 2,
            docs: &[1, 1],
            vectors: &[0.0; 4],
        };
        assert!(w.write_field(&bad).is_err());
        let bad_len = QuantizedVectorsField { docs: &[1], ..bad };
        assert!(w.write_field(&bad_len).is_err());
        let bad_dim = QuantizedVectorsField {
            dimension: 0,
            ..bad
        };
        assert!(w.write_field(&bad_dim).is_err());

        // Corruption: flip a byte of the meta and re-sign nothing -> footer fails.
        let (veq, mut vemq) = write(
            ScalarEncoding::PackedNibble,
            VectorSimilarityFunction::DotProduct,
            &[0, 1],
            &[0.5, 0.1, -0.3, 0.2],
            2,
            2,
        );
        let n = vemq.len();
        vemq[n - 20] ^= 1;
        assert!(ScalarQuantizedVectorsReader::open(&vemq, &veq, &ID, "suf").is_err());
        assert!(ScalarQuantizedVectorsReader::open(&[], &veq, &ID, "suf").is_err());
    }

    #[test]
    fn merge_recomputes_or_weights_centroids() {
        let dim = 4usize;
        let a = vectors(10, dim, 1);
        let b = vectors(6, dim, 2);
        let mut merged = a.clone();
        merged.extend_from_slice(&b);
        let docs: Vec<i32> = (0..16).collect();
        let centroid = |v: &[f32]| -> Vec<f32> {
            let n = v.len() / dim;
            let mut c = vec![0f32; dim];
            for x in v.chunks(dim) {
                for (s, &y) in c.iter_mut().zip(x) {
                    *s += y;
                }
            }
            c.iter().map(|s| s / n as f32).collect()
        };
        let (ca, cb) = (centroid(&a), centroid(&b));
        for (has_del, missing) in [(false, false), (true, false), (false, true)] {
            let sources = [
                QuantizedMergeSource {
                    centroid: if missing { None } else { Some(&ca) },
                    vectors: &a,
                    has_deletions: has_del,
                },
                QuantizedMergeSource {
                    centroid: Some(&cb),
                    vectors: &b,
                    has_deletions: false,
                },
            ];
            for sim in [
                VectorSimilarityFunction::Euclidean,
                VectorSimilarityFunction::Cosine,
            ] {
                let mut w =
                    ScalarQuantizedVectorsWriter::new(ScalarEncoding::SevenBit, 16, &ID, "");
                w.merge_field(1, sim, dim as i32, &sources, &docs, &merged)
                    .unwrap();
                let (veq, vemq) = w.finish();
                let r = ScalarQuantizedVectorsReader::open(&vemq, &veq, &ID, "").unwrap();
                let got = r.centroid(1).unwrap();
                if sim == VectorSimilarityFunction::Euclidean {
                    let want = if has_del || missing {
                        centroid(&merged)
                    } else {
                        (0..dim)
                            .map(|j| (ca[j] * 10.0 + cb[j] * 6.0) / 16.0)
                            .collect()
                    };
                    assert_eq!(got, &want[..]);
                } else {
                    assert!(vector_util::is_unit_vector(got));
                }
            }
        }
        let mut c = vec![0f32; dim];
        assert_eq!(
            merge_and_recalculate_centroids(&[], false, dim, &mut c).unwrap(),
            0
        );
    }

    #[test]
    fn score_parts_follow_java() {
        assert_eq!(java_max(-0.0, 0.0).to_bits(), 0f32.to_bits());
        assert!(java_max(f32::NAN, 0.0).is_nan());
        assert_eq!(java_max(0.0, -0.0).to_bits(), 0f32.to_bits());
        assert_eq!(
            util_similarity(VectorSimilarityFunction::MaximumInnerProduct),
            quantization::VectorSimilarityFunction::MaximumInnerProduct
        );
        let mut v = vec![1u8, 2, 3];
        assert_eq!(align(&mut v, 4), 4);
        assert_eq!(v, [1, 2, 3, 0]);
    }
}
