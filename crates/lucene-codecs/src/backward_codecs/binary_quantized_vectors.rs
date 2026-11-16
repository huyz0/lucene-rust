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

    /// `getVectorCount(fieldInfo)` (Lucene 10.6, `branch_10_6`, #16647): the
    /// quantized entry's size, or for a field without one (a `BYTE` field,
    /// which is not quantized) the raw reader's count -- `raw` being the
    /// `rawVectorsReader` this port opens beside this reader.
    ///
    /// # Errors
    /// [`Error::UnknownField`] for a field neither reader holds.
    pub fn vector_count(
        &self,
        field_number: i32,
        raw: &crate::vectors::FlatVectorsReader<'_>,
    ) -> Result<i32> {
        match self.field(field_number) {
            Some(f) => Ok(f.size),
            None => raw.vector_count(field_number),
        }
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
mod tests;
