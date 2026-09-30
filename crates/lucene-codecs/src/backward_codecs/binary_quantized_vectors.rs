//! `Lucene102BinaryQuantizedVectorsFormat` (Lucene 10.2-10.3): one bit per
//! dimension of the centroid-centred vector, optimized-scalar-quantized, with
//! the query at four bits. Ports of `backward_codecs.lucene102.{
//! Lucene102BinaryQuantizedVectorsFormat, Lucene102BinaryQuantizedVectorsReader,
//! BinarizedByteVectorValues, OffHeapBinarizedVectorValues,
//! Lucene102BinaryFlatVectorsScorer}`. The quantizer is
//! [`lucene_util::quantization::OptimizedScalarQuantizer`], shared with the
//! current `Lucene104ScalarQuantizedVectorsFormat`
//! ([`crate::scalar_quantized_vectors`]), whose single-bit encoding is this
//! format's successor -- the two differ in the corrective-term width (a
//! `short` component sum here, an `int` there), in the metadata (no encoding
//! number here) and in how a score is clamped.
//!
//! `.vemb`:
//! ```text
//! IndexHeader(codec="Lucene102BinaryQuantizedVectorsFormatMeta", version=0, id, suffix)
//! for each field:
//!   FieldNumber int32, VectorEncoding int32, VectorSimilarityFunction int32,
//!   Dimension vint, VectorDataOffset vlong, VectorDataLength vlong, Count vint,
//!   if Count > 0: centroid (dim x f32 LE), centroidDP (f32 bits)
//!   OrdToDocDISIReaderConfiguration stored meta (its structures live in .veb)
//! -1 int32
//! Footer
//! ```
//!
//! `.veb`: `IndexHeader(codec="Lucene102BinaryQuantizedVectorsFormatData")`,
//! then per field and ordinal `discretize(dim, 64) / 8` code bytes,
//! `lowerInterval`, `upperInterval`, `additionalCorrection` (f32 LE) and the
//! quantized component sum (`short`, unsigned); then the sparse `ordToDoc`
//! structures; footer.
//!
//! Rust-only differences: as [`super::scalar_quantized_vectors`] -- no
//! `FieldInfos` in the constructor ([`Lucene102BinaryQuantizedVectorsReader::check_field_infos`]),
//! and the data file's whole checksum is [`Lucene102BinaryQuantizedVectorsReader::check_integrity`]'s.
//! The merge-time supplier (`getRandomVectorScorerSupplierForMerge`) keeps
//! its query-side codes in memory rather than in a temporary file.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_util::quantization::{self, OptimizedScalarQuantizer, QuantizationResult};
use lucene_util::vector_util;

use super::quantized_vectors::{doc_to_ord, ord_to_doc};
use crate::field_infos::{FieldInfos, VectorEncoding, VectorSimilarityFunction};
use crate::hnsw::{UpdateableVectorScorer, VectorScorer};
use crate::scalar_quantized_vectors::util_similarity;
use crate::vectors::{
    file_region, read_similarity_function, read_vector_encoding, DocToOrdCursor, Error, OrdToDoc,
    Result,
};

/// `Lucene102BinaryQuantizedVectorsFormat.NAME`.
pub const NAME: &str = "Lucene102BinaryQuantizedVectorsFormat";
/// `Lucene102HnswBinaryQuantizedVectorsFormat.NAME`.
pub const HNSW_NAME: &str = "Lucene102HnswBinaryQuantizedVectorsFormat";
/// `META_CODEC_NAME`.
pub const META_CODEC: &str = "Lucene102BinaryQuantizedVectorsFormatMeta";
/// `VECTOR_DATA_CODEC_NAME`.
pub const DATA_CODEC: &str = "Lucene102BinaryQuantizedVectorsFormatData";
/// `META_EXTENSION`.
pub const META_EXTENSION: &str = "vemb";
/// `VECTOR_DATA_EXTENSION`.
pub const DATA_EXTENSION: &str = "veb";
/// `VERSION_START` (= `VERSION_CURRENT`).
pub const VERSION_START: i32 = 0;
/// `VERSION_CURRENT`.
pub const VERSION_CURRENT: i32 = VERSION_START;
/// `QUERY_BITS`.
pub const QUERY_BITS: u8 = 4;
/// `INDEX_BITS`.
pub const INDEX_BITS: u8 = 1;
/// `getMaxDimensions`.
pub const MAX_DIMENSIONS: i32 = 1024;
/// `Lucene102BinaryFlatVectorsScorer.FOUR_BIT_SCALE`.
const FOUR_BIT_SCALE: f32 = 1.0 / ((1 << 4) - 1) as f32;
/// Three floats and a short after each vector's codes.
const CORRECTIONS_BYTES: usize = 14;

fn corrupt<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::CorruptMeta(msg.into()))
}

/// `discretize(dimension, 64) / 8`: the code bytes of one vector.
fn binary_bytes(dimension: usize) -> usize {
    quantization::discretize(dimension, 64) / 8
}

/// One field's `.vemb` entry: Java's `FieldEntry` record.
#[derive(Debug, Clone)]
pub struct BinaryQuantizedFieldEntry {
    pub field_number: i32,
    pub similarity: VectorSimilarityFunction,
    pub vector_encoding: VectorEncoding,
    pub dimension: i32,
    pub vector_data_offset: i64,
    pub vector_data_length: i64,
    pub size: i32,
    /// `None` for a field with no vectors.
    pub centroid: Option<Vec<f32>>,
    pub centroid_dp: f32,
    pub ord_to_doc: OrdToDoc,
}

/// Port of `Lucene102BinaryQuantizedVectorsReader` (the quantized half; the
/// raw vectors are a [`crate::vectors::FlatVectorsReader`] over the same
/// segment suffix's `.vec`/`.vemf`).
#[derive(Debug, Clone)]
pub struct Lucene102BinaryQuantizedVectorsReader<'a> {
    data: &'a [u8],
    fields: Vec<BinaryQuantizedFieldEntry>,
}

impl<'a> Lucene102BinaryQuantizedVectorsReader<'a> {
    /// The constructor: `meta_buf` is `.vemb`, `data_buf` is `.veb`.
    pub fn open(
        meta_buf: &[u8],
        data_buf: &'a [u8],
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
    ) -> Result<Self> {
        let mut meta = SliceInput::new(meta_buf);
        let version = codec_util::check_index_header(
            &mut meta,
            META_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?
        .version;
        let Some(meta_footer) = meta_buf.len().checked_sub(codec_util::FOOTER_LENGTH) else {
            return corrupt(".vemb is shorter than its footer");
        };
        codec_util::check_whole_file_footer(meta_buf, meta_footer)?;
        let mut fields: Vec<BinaryQuantizedFieldEntry> = Vec::new();
        loop {
            let field_number = meta.read_i32()?;
            if field_number == -1 {
                break;
            }
            if field_number < 0 || fields.iter().any(|f| f.field_number == field_number) {
                return corrupt(format!("Invalid field number: {field_number}"));
            }
            fields.push(read_field(&mut meta, field_number, data_buf.len())?);
        }
        let data_version = codec_util::check_index_header(
            &mut SliceInput::new(data_buf),
            DATA_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?
        .version;
        if data_version != version {
            return corrupt(format!(
                "Format versions mismatch: meta={version}, {DATA_CODEC}={data_version}"
            ));
        }
        codec_util::retrieve_checksum(data_buf)?;
        Ok(Lucene102BinaryQuantizedVectorsReader {
            data: data_buf,
            fields,
        })
    }

    /// Every field's entry.
    pub fn fields(&self) -> &[BinaryQuantizedFieldEntry] {
        &self.fields
    }

    /// One field's entry.
    pub fn field(&self, field_number: i32) -> Option<&BinaryQuantizedFieldEntry> {
        self.fields.iter().find(|f| f.field_number == field_number)
    }

    /// The `FieldInfos` half of `readFields`/`validateFieldEntry`.
    pub fn check_field_infos(&self, infos: &FieldInfos) -> Result<()> {
        for entry in &self.fields {
            let Some(info) = infos.field_by_number(entry.field_number) else {
                return corrupt(format!("Invalid field number: {}", entry.field_number));
            };
            if info.vector_similarity_function != entry.similarity {
                return corrupt(format!(
                    "Inconsistent vector similarity function for field=\"{}\"; {:?} != {:?}",
                    info.name, entry.similarity, info.vector_similarity_function
                ));
            }
            if info.vector_dimension != entry.dimension {
                return corrupt(format!(
                    "Inconsistent vector dimension for field=\"{}\"; {} != {}",
                    info.name, info.vector_dimension, entry.dimension
                ));
            }
        }
        Ok(())
    }

    /// `checkIntegrity`'s quantized half: `.veb`'s whole-file checksum.
    pub fn check_integrity(&self) -> Result<()> {
        let Some(end) = self.data.len().checked_sub(codec_util::FOOTER_LENGTH) else {
            return corrupt(".veb is shorter than its footer");
        };
        codec_util::check_whole_file_footer(self.data, end)?;
        Ok(())
    }

    /// `getCentroid(field)`.
    pub fn centroid(&self, field_number: i32) -> Option<&[f32]> {
        self.field(field_number).and_then(|f| f.centroid.as_deref())
    }

    /// `OffHeapBinarizedVectorValues.load` for `field` (the quantized half of
    /// `getFloatVectorValues`), which must be `FLOAT32`.
    pub fn binarized_vector_values(&self, field_number: i32) -> Result<BinarizedVectorValues<'a>> {
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
        let empty = entry.ord_to_doc.is_empty();
        let slice = if empty {
            &[][..]
        } else {
            file_region(
                self.data,
                entry.vector_data_offset,
                entry.vector_data_length,
            )
            .ok_or_else(|| Error::CorruptMeta("binarized vector data out of bounds".into()))?
        };
        let num_bytes = binary_bytes(entry.dimension as usize);
        Ok(BinarizedVectorValues {
            slice,
            file: self.data,
            dimension: entry.dimension as usize,
            size: if empty { 0 } else { entry.size },
            similarity: entry.similarity,
            centroid: entry.centroid.clone().unwrap_or_default(),
            centroid_dp: entry.centroid_dp,
            ord_to_doc: entry.ord_to_doc.clone(),
            num_bytes,
            // ARITH: `num_bytes` is at most `dimension / 8 + 8`.
            #[allow(clippy::arithmetic_side_effects)]
            byte_size: num_bytes + CORRECTIONS_BYTES,
        })
    }

    /// `getRandomVectorScorer(field, float[] target)`:
    /// `Lucene102BinaryFlatVectorsScorer.getRandomVectorScorer`.
    pub fn scorer(
        &self,
        field_number: i32,
        target: &[f32],
    ) -> Result<Lucene102BinaryQuantizedScorer<'a>> {
        Lucene102BinaryQuantizedScorer::new(self.binarized_vector_values(field_number)?, target)
    }
}

/// `readField` + `FieldEntry.create` + `validateFieldEntry`'s length identity.
fn read_field(
    meta: &mut SliceInput<'_>,
    field_number: i32,
    data_len: usize,
) -> Result<BinaryQuantizedFieldEntry> {
    let vector_encoding = read_vector_encoding(meta)?;
    let similarity = read_similarity_function(meta)?;
    let dimension = meta.read_vint()?;
    let vector_data_offset = meta.read_vlong()?;
    let vector_data_length = meta.read_vlong()?;
    let size = meta.read_vint()?;
    if dimension <= 0 || size < 0 || vector_data_offset < 0 || vector_data_length < 0 {
        return corrupt(format!(
            "illegal binarized vector entry: dimension={dimension} size={size} \
             [{vector_data_offset}, +{vector_data_length})"
        ));
    }
    let mut centroid = None;
    let mut centroid_dp = 0f32;
    if size > 0 {
        let dim = dimension as usize;
        // Four bytes per component must be there before they are reserved.
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
    // `validateFieldEntry`: `Math.multiplyExact(binaryDims + 14, size)`.
    let per_vector =
        (binary_bytes(dimension as usize) as i64).checked_add(CORRECTIONS_BYTES as i64);
    let expected = per_vector.and_then(|p| p.checked_mul(i64::from(size)));
    if expected != Some(vector_data_length) {
        return corrupt(format!(
            "Binarized vector data length {vector_data_length} not matching size = {size} * \
             (binaryBytes={} + 14) = {}",
            binary_bytes(dimension as usize),
            per_vector.map_or(i64::MAX, |p| p.saturating_mul(i64::from(size)))
        ));
    }
    let end = vector_data_offset.checked_add(vector_data_length);
    if end.is_none_or(|e| e > data_len as i64) {
        return corrupt(format!(
            "binarized vector data [{vector_data_offset}, +{vector_data_length}) past the end \
             of a {data_len} byte .veb"
        ));
    }
    Ok(BinaryQuantizedFieldEntry {
        field_number,
        similarity,
        vector_encoding,
        dimension,
        vector_data_offset,
        vector_data_length,
        size,
        centroid,
        centroid_dp,
        ord_to_doc,
    })
}

/// Port of `BinarizedByteVectorValues` as `OffHeapBinarizedVectorValues`
/// implements it: one-bit codes and their corrective terms, by ordinal.
#[derive(Debug, Clone)]
pub struct BinarizedVectorValues<'a> {
    slice: &'a [u8],
    file: &'a [u8],
    dimension: usize,
    size: i32,
    similarity: VectorSimilarityFunction,
    centroid: Vec<f32>,
    centroid_dp: f32,
    ord_to_doc: OrdToDoc,
    num_bytes: usize,
    byte_size: usize,
}

impl<'a> BinarizedVectorValues<'a> {
    /// `dimension()`.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// `size()`.
    pub fn size(&self) -> i32 {
        self.size
    }

    /// `discretizedDimensions()`: the dimension rounded up to 64.
    pub fn discretized_dimensions(&self) -> usize {
        quantization::discretize(self.dimension, 64)
    }

    /// The field's similarity.
    pub fn similarity(&self) -> VectorSimilarityFunction {
        self.similarity
    }

    /// `getCentroid()` (empty for a field with no vectors).
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    /// `getCentroidDP()`: the stored dot product of the centroid with itself.
    pub fn centroid_dp(&self) -> f32 {
        self.centroid_dp
    }

    /// `getVectorByteLength()`.
    pub fn vector_byte_length(&self) -> usize {
        self.num_bytes
    }

    fn record(&self, ord: i32) -> Result<&'a [u8]> {
        if ord < 0 || ord >= self.size {
            return Err(Error::OrdOutOfRange(ord, self.size));
        }
        let start = (ord as usize).checked_mul(self.byte_size);
        let end = start.and_then(|s| s.checked_add(self.byte_size));
        match (start, end) {
            (Some(s), Some(e)) => self.slice.get(s..e),
            _ => None,
        }
        .ok_or(Error::OrdOutOfRange(ord, self.size))
    }

    /// `vectorValue(ord)`: the packed one-bit codes.
    pub fn vector(&self, ord: i32) -> Result<&'a [u8]> {
        Ok(&self.record(ord)?[..self.num_bytes])
    }

    /// `getCorrectiveTerms(ord)`: three floats, then the component sum as an
    /// unsigned `short`.
    pub fn corrective_terms(&self, ord: i32) -> Result<QuantizationResult> {
        // `record` is exactly `num_bytes + 14` bytes long.
        let rec = &self.record(ord)?[self.num_bytes..];
        let f = |b: &[u8]| f32::from_le_bytes(b.try_into().expect("4 bytes"));
        let (lower, rest) = rec.split_at(4);
        let (upper, rest) = rest.split_at(4);
        let (additional, sum) = rest.split_at(4);
        Ok(QuantizationResult {
            lower_interval: f(lower),
            upper_interval: f(upper),
            additional_correction: f(additional),
            quantized_component_sum: i32::from(u16::from_le_bytes([sum[0], sum[1]])),
        })
    }

    /// `ordToDoc(ord)`.
    pub fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        ord_to_doc(&self.ord_to_doc, self.file, self.size, ord)
    }

    /// The doc -> ordinal direction (`iterator()` / `getAcceptOrds`).
    pub fn doc_to_ord(&self) -> Result<DocToOrdCursor<'a>> {
        doc_to_ord(&self.ord_to_doc, self.file, self.size)
    }
}

/// `Lucene102BinaryFlatVectorsScorer.quantizedScore`.
pub fn quantized_score(
    quantized_query: &[u8],
    query_corrections: &QuantizationResult,
    targets: &BinarizedVectorValues<'_>,
    target_ord: i32,
    similarity: VectorSimilarityFunction,
) -> Result<f32> {
    let binary_code = targets.vector(target_ord)?;
    let qc_dist = vector_util::int4_bit_dot_product(quantized_query, binary_code) as f32;
    let index = targets.corrective_terms(target_ord)?;
    let x1 = index.quantized_component_sum as f32;
    let ax = index.lower_interval;
    // "Here we assume `lx` is simply bit vectors, so the scaling isn't necessary"
    let lx = index.upper_interval - ax;
    let ay = query_corrections.lower_interval;
    let ly = (query_corrections.upper_interval - ay) * FOUR_BIT_SCALE;
    let y1 = query_corrections.quantized_component_sum as f32;
    let mut score =
        ax * ay * targets.dimension as i32 as f32 + ay * lx * x1 + ax * ly * y1 + lx * ly * qc_dist;
    if similarity == VectorSimilarityFunction::Euclidean {
        score = query_corrections.additional_correction + index.additional_correction - 2.0 * score;
        return Ok(java_max(1.0 / (1.0 + score), 0.0));
    }
    score +=
        query_corrections.additional_correction + index.additional_correction - targets.centroid_dp;
    if similarity == VectorSimilarityFunction::MaximumInnerProduct {
        return Ok(crate::vectors::scale_max_inner_product_score(score));
    }
    Ok(java_max((1.0 + score) / 2.0, 0.0))
}

/// `Math.max(float, float)`: a NaN operand wins; `-0.0 < 0.0`.
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

/// The query side of one vector: `scalarQuantize(vector, initial, 4,
/// centroid)` (on a copy, normalized first for `COSINE`) and
/// `transposeHalfByte`. Shared by the query scorer and
/// `writeBinarizedQueryData`.
fn quantize_query_side(
    values: &BinarizedVectorValues<'_>,
    vector: &[f32],
) -> Result<(Vec<u8>, QuantizationResult)> {
    let mut copy = vector.to_vec();
    if values.similarity == VectorSimilarityFunction::Cosine {
        vector_util::l2normalize(&mut copy, true)
            .map_err(|e| Error::InvalidGraphParameter(e.to_string()))?;
    }
    let quantizer = OptimizedScalarQuantizer::new(util_similarity(values.similarity));
    let mut initial = vec![0u8; copy.len()];
    let corrections =
        quantizer.scalar_quantize(&mut copy, &mut initial, QUERY_BITS, &values.centroid);
    // ARITH: `discretize(dim, 64)` is a multiple of 8, and times 4 is far
    // below `usize::MAX` for a dimension that fits an `i32`.
    #[allow(clippy::arithmetic_side_effects)]
    let mut quantized = vec![0u8; usize::from(QUERY_BITS) * values.discretized_dimensions() / 8];
    quantization::transpose_half_byte(&initial, &mut quantized);
    Ok((quantized, corrections))
}

/// `Lucene102BinaryFlatVectorsScorer.getRandomVectorScorer(sim, values,
/// float[] target)`.
#[derive(Debug, Clone)]
pub struct Lucene102BinaryQuantizedScorer<'a> {
    values: BinarizedVectorValues<'a>,
    query: Vec<u8>,
    corrections: QuantizationResult,
}

impl<'a> Lucene102BinaryQuantizedScorer<'a> {
    /// Quantizes `target` against the field's centroid.
    pub fn new(values: BinarizedVectorValues<'a>, target: &[f32]) -> Result<Self> {
        if target.len() != values.dimension {
            return Err(Error::QueryDimensionMismatch(
                target.len() as i32,
                values.dimension as i32,
            ));
        }
        let (query, corrections) = quantize_query_side(&values, target)?;
        Ok(Lucene102BinaryQuantizedScorer {
            values,
            query,
            corrections,
        })
    }

    /// The quantized query and its corrections (for tests and diagnostics).
    pub fn query(&self) -> (&[u8], &QuantizationResult) {
        (&self.query, &self.corrections)
    }

    /// `ordToDoc(ord)`.
    pub fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        self.values.ord_to_doc(ord)
    }
}

impl VectorScorer for Lucene102BinaryQuantizedScorer<'_> {
    fn score(&mut self, node: i32) -> Result<f32> {
        quantized_score(
            &self.query,
            &self.corrections,
            &self.values,
            node,
            self.values.similarity,
        )
    }

    fn max_ord(&self) -> i32 {
        self.values.size
    }
}

/// `writeBinarizedQueryData`: every raw vector (one per ordinal, `vectors`
/// flattened) quantized to its four-bit query side -- the
/// `OffHeapBinarizedQueryVectorValues` a merge scores with, kept in memory.
pub fn binarized_query_data(
    values: &BinarizedVectorValues<'_>,
    vectors: &[f32],
) -> Result<Vec<(Vec<u8>, QuantizationResult)>> {
    if vectors.len().checked_rem(values.dimension) != Some(0) {
        return Err(Error::QueryDimensionMismatch(
            vectors.len() as i32,
            values.dimension as i32,
        ));
    }
    vectors
        .chunks_exact(values.dimension)
        .map(|v| {
            let (q, mut c) = quantize_query_side(values, v)?;
            // `writeShort((short) r.quantizedComponentSum())`, read back
            // with `Short.toUnsignedInt`.
            c.quantized_component_sum = i32::from(c.quantized_component_sum as u16);
            Ok((q, c))
        })
        .collect()
}

/// `BinarizedRandomVectorScorerSupplier.scorer()`: an ordinal's query side
/// (set with `set_scoring_ordinal`) scored against every stored vector.
#[derive(Debug, Clone)]
pub struct Lucene102BinaryQuantizedOrdScorer<'v, 'a> {
    values: BinarizedVectorValues<'a>,
    query_side: &'v [(Vec<u8>, QuantizationResult)],
    current: Option<usize>,
}

impl<'v, 'a> Lucene102BinaryQuantizedOrdScorer<'v, 'a> {
    /// Over [`binarized_query_data`]'s output.
    pub fn new(
        values: BinarizedVectorValues<'a>,
        query_side: &'v [(Vec<u8>, QuantizationResult)],
    ) -> Self {
        Lucene102BinaryQuantizedOrdScorer {
            values,
            query_side,
            current: None,
        }
    }
}

impl VectorScorer for Lucene102BinaryQuantizedOrdScorer<'_, '_> {
    fn score(&mut self, node: i32) -> Result<f32> {
        let Some((q, c)) = self.current.and_then(|i| self.query_side.get(i)) else {
            return Err(Error::InvalidGraphParameter(
                "setScoringOrdinal was not called".into(),
            ));
        };
        quantized_score(q, c, &self.values, node, self.values.similarity)
    }

    fn max_ord(&self) -> i32 {
        self.values.size
    }
}

impl UpdateableVectorScorer for Lucene102BinaryQuantizedOrdScorer<'_, '_> {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
        match usize::try_from(ord)
            .ok()
            .filter(|&o| o < self.query_side.len())
        {
            Some(o) => {
                self.current = Some(o);
                Ok(())
            }
            None => Err(Error::OrdOutOfRange(ord, self.query_side.len() as i32)),
        }
    }
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::backward_codecs::quantized_vectors::test_support::fixture_group;
    use crate::field_infos::FieldInfo;
    use crate::vectors::FlatVectorsReader;
    use lucene_store::data_output::DataOutput;

    const ID: [u8; ID_LENGTH] = *b"binary-quant-001";

    struct TestField {
        number: i32,
        encoding: VectorEncoding,
        sim: VectorSimilarityFunction,
        dim: usize,
        docs: Vec<i32>,
        max_doc: i32,
        vectors: Vec<f32>,
        dim_override: Option<i32>,
        offset_delta: i64,
        length_delta: i64,
    }

    fn field(number: i32, sim: VectorSimilarityFunction, dim: usize) -> TestField {
        let n = 25usize;
        let mut s = 91u64 + number as u64;
        let vectors = (0..n * dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        TestField {
            number,
            encoding: VectorEncoding::Float32,
            sim,
            dim,
            docs: (0..n as i32).map(|d| d * 3).collect(),
            max_doc: 3 * n as i32,
            vectors,
            dim_override: None,
            offset_delta: 0,
            length_delta: 0,
        }
    }

    /// A minimal `Lucene102BinaryQuantizedVectorsWriter`: the centroid is the
    /// mean, each vector one-bit quantized against it and packed.
    fn write(fields: &[TestField]) -> (Vec<u8>, Vec<u8>) {
        let mut meta = Vec::new();
        let mut data = Vec::new();
        codec_util::write_index_header(&mut meta, META_CODEC, VERSION_CURRENT, &ID, "b");
        codec_util::write_index_header(&mut data, DATA_CODEC, VERSION_CURRENT, &ID, "b");
        for f in fields {
            let n = f.docs.len();
            let mut centroid = vec![0f32; f.dim];
            let normalized: Vec<Vec<f32>> = f
                .vectors
                .chunks(f.dim.max(1))
                .take(n)
                .map(|v| {
                    let mut v = v.to_vec();
                    if f.sim == VectorSimilarityFunction::Cosine {
                        vector_util::l2normalize(&mut v, true).unwrap();
                    }
                    v
                })
                .collect();
            for v in &normalized {
                for (c, x) in centroid.iter_mut().zip(v) {
                    *c += x / n as f32;
                }
            }
            let quantizer = OptimizedScalarQuantizer::new(util_similarity(f.sim));
            let offset = data.len() as i64;
            for v in &normalized {
                let mut v = v.clone();
                let mut scratch = vec![0u8; quantization::discretize(f.dim, 64)];
                let r = quantizer.scalar_quantize(&mut v, &mut scratch, INDEX_BITS, &centroid);
                let mut packed = vec![0u8; binary_bytes(f.dim)];
                quantization::pack_as_binary(&scratch, &mut packed);
                data.extend_from_slice(&packed);
                data.write_i32(r.lower_interval.to_bits() as i32);
                data.write_i32(r.upper_interval.to_bits() as i32);
                data.write_i32(r.additional_correction.to_bits() as i32);
                data.write_i16(r.quantized_component_sum as i16);
            }
            let length = data.len() as i64 - offset;
            meta.write_i32(f.number);
            meta.write_i32(match f.encoding {
                VectorEncoding::Byte => 0,
                VectorEncoding::Float32 => 1,
            });
            meta.write_i32(match f.sim {
                VectorSimilarityFunction::Euclidean => 0,
                VectorSimilarityFunction::DotProduct => 1,
                VectorSimilarityFunction::Cosine => 2,
                VectorSimilarityFunction::MaximumInnerProduct => 3,
            });
            meta.write_vint(f.dim_override.unwrap_or(f.dim as i32));
            meta.write_vlong(offset + f.offset_delta);
            meta.write_vlong(length + f.length_delta);
            meta.write_vint(n as i32);
            if n > 0 {
                for c in &centroid {
                    meta.write_i32(c.to_bits() as i32);
                }
                let dp: f32 = centroid.iter().map(|c| c * c).sum();
                meta.write_i32(dp.to_bits() as i32);
            }
            crate::vectors::write_stored_meta(&mut meta, &mut data, &f.docs, f.max_doc);
        }
        meta.write_i32(-1);
        codec_util::write_footer(&mut meta);
        codec_util::write_footer(&mut data);
        (meta, data)
    }

    fn infos(fields: &[TestField]) -> FieldInfos {
        FieldInfos::new(
            fields
                .iter()
                .map(|f| {
                    let mut fi = FieldInfo::new(format!("f{}", f.number), f.number);
                    fi.vector_dimension = f.dim as i32;
                    fi.vector_encoding = f.encoding;
                    fi.vector_similarity_function = f.sim;
                    fi
                })
                .collect(),
        )
        .unwrap()
    }

    fn open<'a>(meta: &[u8], data: &'a [u8]) -> Result<Lucene102BinaryQuantizedVectorsReader<'a>> {
        Lucene102BinaryQuantizedVectorsReader::open(meta, data, &ID, "b")
    }

    fn err_of<T: std::fmt::Debug>(r: Result<T>) -> String {
        r.unwrap_err().to_string()
    }

    /// Every fixture group: the merge-time supplier over the raw vectors'
    /// query side scores an ordinal pair exactly as the query scorer does
    /// with that raw vector as its target -- the two quantize the same way.
    #[test]
    fn fixture_groups_open_and_the_merge_supplier_agrees_with_the_query_scorer() {
        let mut checked = 0;
        for format in [NAME, HNSW_NAME] {
            for n in 0..6 {
                for seg in ["_0", "_1"] {
                    let Some(g) = fixture_group("10.2.2", seg, format, n) else {
                        continue;
                    };
                    let r = Lucene102BinaryQuantizedVectorsReader::open(
                        g.file("vemb"),
                        g.file("veb"),
                        &g.id,
                        &g.suffix,
                    )
                    .unwrap();
                    r.check_integrity().unwrap();
                    let flat =
                        FlatVectorsReader::open(g.file("vemf"), g.file("vec"), &g.id, &g.suffix)
                            .unwrap();
                    for e in r.fields().to_vec() {
                        let n = e.field_number;
                        assert_eq!(r.centroid(n).map(<[f32]>::len), Some(e.dimension as usize));
                        let values = r.binarized_vector_values(n).unwrap();
                        assert_eq!(values.similarity(), e.similarity);
                        assert_eq!(values.centroid().len(), values.dimension());
                        assert_eq!(values.discretized_dimensions() % 64, 0);
                        assert_eq!(
                            values.vector_byte_length() * 8,
                            values.discretized_dimensions()
                        );
                        assert!(values.centroid_dp() >= 0.0);
                        let raw = flat.float_vector_values(n).unwrap();
                        let dim = values.dimension();
                        let mut all = Vec::new();
                        for ord in 0..raw.size() {
                            all.extend(raw.vector(ord).unwrap());
                        }
                        let query_side = binarized_query_data(&values, &all).unwrap();
                        let mut supplier =
                            Lucene102BinaryQuantizedOrdScorer::new(values.clone(), &query_side);
                        let mut cursor = values.doc_to_ord().unwrap();
                        for ord in (0..values.size()).step_by(7) {
                            let target = &all[ord as usize * dim..(ord as usize + 1) * dim];
                            let mut q = r.scorer(n, target).unwrap();
                            assert_eq!(q.max_ord(), values.size());
                            assert_eq!(supplier.max_ord(), values.size());
                            supplier.set_scoring_ordinal(ord).unwrap();
                            for other in [0, ord, values.size() - 1] {
                                assert_eq!(
                                    q.score(other).unwrap().to_bits(),
                                    supplier.score(other).unwrap().to_bits()
                                );
                            }
                            let doc = values.ord_to_doc(ord).unwrap();
                            assert_eq!(q.ord_to_doc(ord).unwrap(), doc);
                            assert_eq!(doc, raw.ord_to_doc(ord).unwrap());
                            assert_eq!(cursor.ordinal(doc).unwrap(), Some(ord));
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked >= 12, "{checked} fields");
    }

    #[test]
    fn hand_built_fields_round_trip() {
        let fields = [
            field(0, VectorSimilarityFunction::Euclidean, 40),
            field(1, VectorSimilarityFunction::DotProduct, 64),
            field(2, VectorSimilarityFunction::Cosine, 70),
            field(3, VectorSimilarityFunction::MaximumInnerProduct, 9),
        ];
        let (meta, data) = write(&fields);
        let r = open(&meta, &data).unwrap();
        r.check_field_infos(&infos(&fields)).unwrap();
        for f in &fields {
            let values = r.binarized_vector_values(f.number).unwrap();
            assert_eq!(values.size(), 25);
            assert_eq!(values.ord_to_doc(4).unwrap(), 12);
            // Scores are in range, and the stored vector itself outranks the
            // average of the others.
            let target = &f.vectors[..f.dim];
            let mut s = r.scorer(f.number, target).unwrap();
            let own = s.score(0).unwrap();
            let mean: f32 = (1..25).map(|o| s.score(o).unwrap()).sum::<f32>() / 24.0;
            assert!(own > mean, "{:?}: {own} vs {mean}", f.sim);
            assert!((0..25).all(|o| s.score(o).unwrap() >= 0.0));
            assert_eq!(
                s.query().0.len(),
                usize::from(QUERY_BITS) * values.discretized_dimensions() / 8
            );
            assert!(values.vector(25).is_err());
            assert!(values.corrective_terms(-1).is_err());
        }
    }

    #[test]
    fn empty_and_byte_fields() {
        let mut empty = field(5, VectorSimilarityFunction::Euclidean, 16);
        empty.docs.clear();
        let mut byte = field(6, VectorSimilarityFunction::Euclidean, 16);
        byte.encoding = VectorEncoding::Byte;
        byte.docs.clear();
        let (meta, data) = write(&[empty, byte]);
        let r = open(&meta, &data).unwrap();
        assert!(r.centroid(5).is_none());
        let values = r.binarized_vector_values(5).unwrap();
        assert_eq!(values.size(), 0);
        assert!(matches!(
            values.doc_to_ord().unwrap(),
            DocToOrdCursor::Empty
        ));
        assert!(matches!(
            r.binarized_vector_values(6),
            Err(Error::EncodingMismatch(
                6,
                VectorEncoding::Byte,
                VectorEncoding::Float32
            ))
        ));
        assert!(matches!(
            r.binarized_vector_values(7),
            Err(Error::UnknownField(7))
        ));
    }

    #[test]
    fn caller_mistakes_are_errors() {
        let fields = [field(1, VectorSimilarityFunction::Cosine, 16)];
        let (meta, data) = write(&fields);
        let r = open(&meta, &data).unwrap();
        assert!(matches!(
            r.scorer(1, &[0.1; 3]),
            Err(Error::QueryDimensionMismatch(3, 16))
        ));
        assert!(r.scorer(1, &[0.0; 16]).is_err());
        let values = r.binarized_vector_values(1).unwrap();
        assert!(binarized_query_data(&values, &[0.5; 17]).is_err());
        let side = binarized_query_data(&values, &fields[0].vectors).unwrap();
        let mut s = Lucene102BinaryQuantizedOrdScorer::new(values, &side);
        assert!(s.score(0).is_err());
        assert!(s.set_scoring_ordinal(25).is_err());
        assert!(s.set_scoring_ordinal(-1).is_err());

        let mut other = infos(&fields);
        other.fields[0].vector_similarity_function = VectorSimilarityFunction::Euclidean;
        assert!(err_of(r.check_field_infos(&other)).contains("similarity"));
        let mut other = infos(&fields);
        other.fields[0].vector_dimension = 15;
        assert!(err_of(r.check_field_infos(&other)).contains("dimension"));
        assert!(
            err_of(r.check_field_infos(&FieldInfos::new(Vec::new()).unwrap()))
                .contains("Invalid field number")
        );
    }

    #[test]
    fn corrupt_metadata_is_rejected() {
        let good = || field(1, VectorSimilarityFunction::DotProduct, 16);
        let mut f = good();
        f.length_delta = 1;
        let (meta, data) = write(&[f]);
        assert!(err_of(open(&meta, &data)).contains("not matching"));
        let mut f = good();
        f.offset_delta = 1 << 20;
        let (meta, data) = write(&[f]);
        assert!(err_of(open(&meta, &data)).contains("past the end"));
        let mut f = good();
        f.dim_override = Some(0);
        let (meta, data) = write(&[f]);
        assert!(err_of(open(&meta, &data)).contains("illegal binarized vector entry"));
        // A centroid longer than what is left of the file.
        let mut f = good();
        f.dim_override = Some(1 << 20);
        let (meta, data) = write(&[f]);
        assert!(open(&meta, &data).is_err());
        let mut f = good();
        f.number = -3;
        let (meta, data) = write(&[f]);
        assert!(err_of(open(&meta, &data)).contains("Invalid field number"));
        let (meta, data) = write(&[good(), good()]);
        assert!(err_of(open(&meta, &data)).contains("Invalid field number"));

        let (meta, data) = write(&[good()]);
        assert!(open(&meta[..8], &data).is_err());
        assert!(open(&meta, &data[..data.len() - 2]).is_err());
        // The data file at another version than the metadata.
        let mut data1 = Vec::new();
        codec_util::write_index_header(&mut data1, DATA_CODEC, 1, &ID, "b");
        let header = data1.len();
        data1.extend_from_slice(&data[header..data.len() - codec_util::FOOTER_LENGTH]);
        codec_util::write_footer(&mut data1);
        assert!(open(&meta, &data1).is_err());
        let mut flipped = data.clone();
        let mid = flipped.len() / 2;
        flipped[mid] ^= 0x10;
        assert!(open(&meta, &flipped).unwrap().check_integrity().is_err());
    }

    #[test]
    fn scores_clamp_as_java_does() {
        assert!(java_max(f32::NAN, 0.0).is_nan());
        assert!(java_max(0.0, f32::NAN).is_nan());
        assert!(java_max(-0.0, 0.0).is_sign_positive());
        assert!(java_max(0.0, -0.0).is_sign_positive());
        assert_eq!(java_max(-1.0, 0.0), 0.0);
    }
}
