//! `Lucene99ScalarQuantizedVectorsFormat` (Lucene 9.9-10.3): the legacy,
//! globally quantized vectors a `Lucene99HnswScalarQuantizedVectorsFormat`
//! (or the flat format alone) keeps beside the raw floats. Ports of
//! `backward_codecs.lucene99.{Lucene99ScalarQuantizedVectorsFormat,
//! Lucene99ScalarQuantizedVectorsReader, OffHeapQuantizedByteVectorValues,
//! OffHeapQuantizedFloatVectorValues}`, `util.quantization.LegacyQuantizedByteVectorValues`,
//! and the two scorers core still ships for it,
//! `codecs.lucene99.Lucene99ScalarQuantizedVectorScorer` (what the format
//! uses) and `codecs.hnsw.ScalarQuantizedVectorScorer` (the generic one,
//! scoring through `ScalarQuantizedVectorSimilarity`). The quantizer is
//! [`lucene_util::quantization::ScalarQuantizer`].
//!
//! `.vemq`:
//! ```text
//! IndexHeader(codec="Lucene99ScalarQuantizedVectorsFormatMeta", version 0..=1, id, suffix)
//! for each field:
//!   FieldNumber int32, VectorEncoding int32, VectorSimilarityFunction int32,
//!   VectorDataOffset vlong, VectorDataLength vlong, Dimension vint, Count int32,
//!   if Count > 0:
//!     version 0: confidenceInterval (f32 bits; -1 = null, 0 = dynamic: both corrupt),
//!                lowerQuantile, upperQuantile (f32 bits) -- 7 bits, uncompressed
//!     version 1: confidenceInterval (unused), bits byte, compress byte (1 = true),
//!                lowerQuantile, upperQuantile (f32 bits)
//!   OrdToDocDISIReaderConfiguration stored meta (its structures live in .veq)
//! -1 int32
//! Footer
//! ```
//!
//! `.veq`: `IndexHeader(codec="Lucene99ScalarQuantizedVectorsFormatData")`,
//! then per field and ordinal the codes (`dimension` bytes, or `(dimension +
//! 1) / 2` when 4-bit codes are compressed two to a byte) and one `f32`
//! score-correction constant, then the sparse `ordToDoc` structures; footer.
//!
//! Only `FLOAT32` fields are quantized: a `BYTE` field of this format has no
//! `.vemq` entry and is served by the raw [`crate::vectors::FlatVectorsReader`].
//!
//! Rust-only differences: the constructor does not take the segment's
//! `FieldInfos`; [`Lucene99ScalarQuantizedVectorsReader::check_field_infos`]
//! is the half of `readFields`/`validateFieldEntry` that needs them. The data
//! file's footer is checked for shape at open (`CodecUtil.retrieveChecksum`)
//! as Java does; its whole-file checksum is `checkIntegrity`'s, which this
//! reader leaves to [`Lucene99ScalarQuantizedVectorsReader::check_integrity`].

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_util::quantization::{ScalarQuantizedVectorSimilarity, ScalarQuantizer};
use lucene_util::vector_util;

use super::quantized_vectors::{doc_to_ord, ord_to_doc};
use crate::field_infos::{FieldInfos, VectorEncoding, VectorSimilarityFunction};
use crate::hnsw::{UpdateableVectorScorer, VectorScorer};
use crate::scalar_quantized_vectors::util_similarity;
use crate::vectors::{
    file_region, read_similarity_function, read_vector_encoding, DocToOrdCursor, Error, OrdToDoc,
    Result,
};

/// `Lucene99ScalarQuantizedVectorsFormat.NAME`.
pub const NAME: &str = "Lucene99ScalarQuantizedVectorsFormat";
/// `Lucene99HnswScalarQuantizedVectorsFormat.NAME`.
pub const HNSW_NAME: &str = "Lucene99HnswScalarQuantizedVectorsFormat";
/// `META_CODEC_NAME`.
pub const META_CODEC: &str = "Lucene99ScalarQuantizedVectorsFormatMeta";
/// `VECTOR_DATA_CODEC_NAME`.
pub const DATA_CODEC: &str = "Lucene99ScalarQuantizedVectorsFormatData";
/// `META_EXTENSION`.
pub const META_EXTENSION: &str = "vemq";
/// `VECTOR_DATA_EXTENSION`.
pub const DATA_EXTENSION: &str = "veq";
/// `VERSION_START`.
pub const VERSION_START: i32 = 0;
/// `VERSION_ADD_BITS`: the bits and compress bytes.
pub const VERSION_ADD_BITS: i32 = 1;
/// `VERSION_CURRENT`.
pub const VERSION_CURRENT: i32 = VERSION_ADD_BITS;
/// `DYNAMIC_CONFIDENCE_INTERVAL`.
pub const DYNAMIC_CONFIDENCE_INTERVAL: f32 = 0.0;

fn corrupt<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::CorruptMeta(msg.into()))
}

/// One field's `.vemq` entry: Java's `FieldEntry` record.
#[derive(Debug, Clone)]
pub struct ScalarQuantizedFieldEntry {
    pub field_number: i32,
    pub similarity: VectorSimilarityFunction,
    pub vector_encoding: VectorEncoding,
    pub dimension: i32,
    pub vector_data_offset: i64,
    pub vector_data_length: i64,
    /// `None` for a field with no vectors.
    pub scalar_quantizer: Option<ScalarQuantizer>,
    pub size: i32,
    pub bits: u8,
    pub compress: bool,
    pub ord_to_doc: OrdToDoc,
}

impl ScalarQuantizedFieldEntry {
    /// `numBytes`: the codes of one vector as stored.
    fn num_bytes(&self) -> usize {
        packed_len(self.dimension as usize, self.bits, self.compress)
    }
}

/// `OffHeapQuantizedByteVectorValues`' `numBytes`.
fn packed_len(dimension: usize, bits: u8, compress: bool) -> usize {
    if bits <= 4 && compress {
        dimension.div_ceil(2)
    } else {
        dimension
    }
}

/// Port of `Lucene99ScalarQuantizedVectorsReader` (the quantized half; the
/// raw vectors are a [`crate::vectors::FlatVectorsReader`] over the same
/// segment suffix's `.vec`/`.vemf`).
#[derive(Debug, Clone)]
pub struct Lucene99ScalarQuantizedVectorsReader<'a> {
    data: &'a [u8],
    version: i32,
    fields: Vec<ScalarQuantizedFieldEntry>,
}

impl<'a> Lucene99ScalarQuantizedVectorsReader<'a> {
    /// The constructor: `meta_buf` is `.vemq`, `data_buf` is `.veq`.
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
            return corrupt(".vemq is shorter than its footer");
        };
        codec_util::check_whole_file_footer(meta_buf, meta_footer)?;
        let mut fields: Vec<ScalarQuantizedFieldEntry> = Vec::new();
        loop {
            let field_number = meta.read_i32()?;
            if field_number == -1 {
                break;
            }
            if field_number < 0 || fields.iter().any(|f| f.field_number == field_number) {
                return corrupt(format!("Invalid field number: {field_number}"));
            }
            fields.push(read_field(
                &mut meta,
                version,
                field_number,
                data_buf.len(),
            )?);
        }
        // `openDataInput`: header, version agreement, footer shape.
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
        Ok(Lucene99ScalarQuantizedVectorsReader {
            data: data_buf,
            version,
            fields,
        })
    }

    /// The `.vemq` format version (0, or 1 from Lucene 9.10 on).
    pub fn version(&self) -> i32 {
        self.version
    }

    /// Every field's entry.
    pub fn fields(&self) -> &[ScalarQuantizedFieldEntry] {
        &self.fields
    }

    /// One field's entry.
    pub fn field(&self, field_number: i32) -> Option<&ScalarQuantizedFieldEntry> {
        self.fields.iter().find(|f| f.field_number == field_number)
    }

    /// The `FieldInfos` half of `readFields`: every entry's field must exist
    /// ("Invalid field number"), with the entry's similarity
    /// (`readField`'s "Inconsistent vector similarity function") and
    /// dimension (`validateFieldEntry`'s "Inconsistent vector dimension").
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

    /// `checkIntegrity`'s quantized half: `.veq`'s whole-file checksum.
    pub fn check_integrity(&self) -> Result<()> {
        let Some(end) = self.data.len().checked_sub(codec_util::FOOTER_LENGTH) else {
            return corrupt(".veq is shorter than its footer");
        };
        codec_util::check_whole_file_footer(self.data, end)?;
        Ok(())
    }

    /// `getQuantizationState(field)`.
    pub fn quantization_state(&self, field_number: i32) -> Result<Option<ScalarQuantizer>> {
        Ok(self.float_entry(field_number)?.scalar_quantizer)
    }

    /// `getFieldEntry(field)`: the entry, which must be `FLOAT32`.
    fn float_entry(&self, field_number: i32) -> Result<&ScalarQuantizedFieldEntry> {
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
        Ok(entry)
    }

    /// `getQuantizedVectorValues(field)` -> `OffHeapQuantizedByteVectorValues.load`.
    pub fn quantized_vector_values(
        &self,
        field_number: i32,
    ) -> Result<LegacyQuantizedVectorValues<'a>> {
        let entry = self.float_entry(field_number)?;
        let slice = if entry.ord_to_doc.is_empty() {
            &[][..]
        } else {
            file_region(
                self.data,
                entry.vector_data_offset,
                entry.vector_data_length,
            )
            .ok_or_else(|| Error::CorruptMeta("quantized vector data out of bounds".into()))?
        };
        let num_bytes = entry.num_bytes();
        Ok(LegacyQuantizedVectorValues {
            slice,
            file: self.data,
            dimension: entry.dimension as usize,
            size: if entry.ord_to_doc.is_empty() {
                0
            } else {
                entry.size
            },
            quantizer: entry.scalar_quantizer,
            similarity: entry.similarity,
            ord_to_doc: entry.ord_to_doc.clone(),
            num_bytes,
            // ARITH: `num_bytes <= dimension`, a positive `i32`.
            #[allow(clippy::arithmetic_side_effects)]
            byte_size: num_bytes + 4,
        })
    }

    /// `getRandomVectorScorer(field, float[] target)` with a quantizer:
    /// `Lucene99ScalarQuantizedVectorScorer.getRandomVectorScorer`. `None`
    /// when the field has no quantizer (no vectors), where Java falls back
    /// to the raw reader.
    pub fn scorer(
        &self,
        field_number: i32,
        target: &[f32],
    ) -> Result<Option<Lucene99ScalarQuantizedScorer<'a>>> {
        let values = self.quantized_vector_values(field_number)?;
        if values.quantizer.is_none() {
            return Ok(None);
        }
        Lucene99ScalarQuantizedScorer::new(values, target).map(Some)
    }
}

/// `readField` + `FieldEntry.create` + `validateFieldEntry`'s length identity.
fn read_field(
    meta: &mut SliceInput<'_>,
    version: i32,
    field_number: i32,
    data_len: usize,
) -> Result<ScalarQuantizedFieldEntry> {
    let vector_encoding = read_vector_encoding(meta)?;
    let similarity = read_similarity_function(meta)?;
    let vector_data_offset = meta.read_vlong()?;
    let vector_data_length = meta.read_vlong()?;
    let dimension = meta.read_vint()?;
    let size = meta.read_i32()?;
    if dimension <= 0 || size < 0 || vector_data_offset < 0 || vector_data_length < 0 {
        return corrupt(format!(
            "illegal quantized vector entry: dimension={dimension} size={size} \
             [{vector_data_offset}, +{vector_data_length})"
        ));
    }
    let (scalar_quantizer, bits, compress) = if size > 0 {
        let (bits, compress) = if version < VERSION_ADD_BITS {
            let confidence_bits = meta.read_i32()?;
            if confidence_bits == -1 {
                return corrupt("Missing confidence interval for scalar quantizer");
            }
            let confidence = f32::from_bits(confidence_bits as u32);
            if confidence == DYNAMIC_CONFIDENCE_INTERVAL {
                return corrupt(format!(
                    "Invalid confidence interval for scalar quantizer: {confidence}"
                ));
            }
            (7u8, false)
        } else {
            meta.read_i32()?; // confidenceInterval, unused
            let bits = meta.read_byte()?;
            let compress = meta.read_byte()? == 1;
            (bits, compress)
        };
        if !(1..=8).contains(&bits) {
            return corrupt(format!("illegal quantization bits {bits}"));
        }
        let min_quantile = f32::from_bits(meta.read_i32()? as u32);
        let max_quantile = f32::from_bits(meta.read_i32()? as u32);
        let quantizer = ScalarQuantizer::new(min_quantile, max_quantile, bits)
            .map_err(|e| Error::CorruptMeta(e.to_string()))?;
        (Some(quantizer), bits, compress)
    } else {
        (None, 7, false)
    };
    let ord_to_doc = OrdToDoc::from_stored_meta(meta, size)?;
    // `validateFieldEntry`: `Math.multiplyExact(quantizedVectorBytes, size)`.
    let per_vector = (packed_len(dimension as usize, bits, compress) as i64).checked_add(4);
    let expected = per_vector.and_then(|p| p.checked_mul(i64::from(size)));
    if expected != Some(vector_data_length) {
        return corrupt(format!(
            "Quantized vector data length {vector_data_length} not matching size={size} * \
             (dim={dimension} + 4) = {}",
            per_vector.map_or(i64::MAX, |p| p.saturating_mul(i64::from(size)))
        ));
    }
    let end = vector_data_offset.checked_add(vector_data_length);
    if end.is_none_or(|e| e > data_len as i64) {
        return corrupt(format!(
            "quantized vector data [{vector_data_offset}, +{vector_data_length}) past the end \
             of a {data_len} byte .veq"
        ));
    }
    Ok(ScalarQuantizedFieldEntry {
        field_number,
        similarity,
        vector_encoding,
        dimension,
        vector_data_offset,
        vector_data_length,
        scalar_quantizer,
        size,
        bits,
        compress,
        ord_to_doc,
    })
}

/// `OffHeapQuantizedByteVectorValues.decompressBytes`: `packed[i]`'s high
/// nibble is code `i`, its low nibble code `numBytes + i`. `out` must be
/// exactly twice `packed` -- Java throws otherwise, which is what an odd
/// dimension with compression runs into.
fn decompress_bytes(packed: &[u8], out: &mut [u8]) -> Result<()> {
    if packed.len().checked_mul(2) != Some(out.len()) {
        return Err(Error::InvalidGraphParameter(format!(
            "numBytes: {} does not match compressed length: {}",
            packed.len(),
            out.len()
        )));
    }
    let (hi, lo) = out.split_at_mut(packed.len());
    for ((h, l), &p) in hi.iter_mut().zip(lo.iter_mut()).zip(packed) {
        *h = p >> 4;
        *l = p & 0x0F;
    }
    Ok(())
}

/// Port of `LegacyQuantizedByteVectorValues` as
/// `OffHeapQuantizedByteVectorValues` implements it: a field's codes (as
/// stored: possibly two per byte) and their score-correction constants,
/// addressed by ordinal.
#[derive(Debug, Clone)]
pub struct LegacyQuantizedVectorValues<'a> {
    slice: &'a [u8],
    file: &'a [u8],
    dimension: usize,
    size: i32,
    quantizer: Option<ScalarQuantizer>,
    similarity: VectorSimilarityFunction,
    ord_to_doc: OrdToDoc,
    num_bytes: usize,
    byte_size: usize,
}

impl<'a> LegacyQuantizedVectorValues<'a> {
    /// `dimension()`.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// `size()`.
    pub fn size(&self) -> i32 {
        self.size
    }

    /// `getScalarQuantizer()`.
    pub fn scalar_quantizer(&self) -> Option<&ScalarQuantizer> {
        self.quantizer.as_ref()
    }

    /// The field's similarity.
    pub fn similarity(&self) -> VectorSimilarityFunction {
        self.similarity
    }

    /// `getVectorByteLength()`: the stored codes' length.
    pub fn vector_byte_length(&self) -> usize {
        self.num_bytes
    }

    /// Whether the codes are packed two to a byte.
    pub fn is_compressed(&self) -> bool {
        self.num_bytes != self.dimension
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

    /// The codes of `ord` as stored (`getSlice()` at the vector's offset).
    pub fn packed_vector(&self, ord: i32) -> Result<&'a [u8]> {
        Ok(&self.record(ord)?[..self.num_bytes])
    }

    /// `vectorValue(ord)`: the codes, one per dimension (decompressed).
    pub fn vector_into(&self, ord: i32, out: &mut Vec<u8>) -> Result<()> {
        let packed = self.packed_vector(ord)?;
        out.clear();
        if self.is_compressed() {
            out.resize(self.dimension, 0);
            decompress_bytes(packed, out)
        } else {
            out.extend_from_slice(packed);
            Ok(())
        }
    }

    /// `getScoreCorrectionConstant(ord)`.
    pub fn score_correction_constant(&self, ord: i32) -> Result<f32> {
        let rec = self.record(ord)?;
        let bytes: [u8; 4] = rec[self.num_bytes..]
            .try_into()
            .map_err(|_| Error::OrdOutOfRange(ord, self.size))?;
        Ok(f32::from_le_bytes(bytes))
    }

    /// `ordToDoc(ord)`.
    pub fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        ord_to_doc(&self.ord_to_doc, self.file, self.size, ord)
    }

    /// The doc -> ordinal direction (`iterator()` / `getAcceptOrds`).
    pub fn doc_to_ord(&self) -> Result<DocToOrdCursor<'a>> {
        doc_to_ord(&self.ord_to_doc, self.file, self.size)
    }

    /// `OffHeapQuantizedFloatVectorValues.vectorValue(ord)`: the codes
    /// dequantized (`ScalarQuantizer.deQuantize`) -- the float view a field
    /// whose raw vectors are empty is served through.
    pub fn dequantized(&self, ord: i32, out: &mut [f32]) -> Result<()> {
        let quantizer = self
            .quantizer
            .ok_or_else(|| Error::CorruptMeta("no quantizer for an empty field".into()))?;
        let mut codes = Vec::with_capacity(self.dimension);
        self.vector_into(ord, &mut codes)?;
        quantizer.dequantize(&codes, out);
        Ok(())
    }
}

/// `ScalarQuantizedVectorScorer.quantizeQuery`: the query (normalized first
/// for `COSINE`) quantized with the field's quantizer; returns its
/// corrective offset.
pub fn quantize_query(
    query: &[f32],
    quantized: &mut [u8],
    similarity: VectorSimilarityFunction,
    quantizer: &ScalarQuantizer,
) -> Result<f32> {
    let mut processed = query.to_vec();
    if similarity == VectorSimilarityFunction::Cosine {
        vector_util::l2normalize(&mut processed, true)
            .map_err(|e| Error::InvalidGraphParameter(e.to_string()))?;
    }
    Ok(quantizer.quantize(&processed, quantized, util_similarity(similarity)))
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

/// `Lucene99ScalarQuantizedVectorScorer.fromVectorSimilarity`'s scorers
/// (`Euclidean`, `DotProduct`, `Int4DotProduct`, `CompressedInt4DotProduct`)
/// as one: the target's codes and offset against every stored vector.
#[derive(Debug, Clone)]
struct Lucene99Kernel<'a> {
    values: LegacyQuantizedVectorValues<'a>,
    similarity: VectorSimilarityFunction,
    const_multiplier: f32,
    target: Vec<u8>,
    offset_correction: f32,
    scratch: Vec<u8>,
}

impl Lucene99Kernel<'_> {
    fn score(&mut self, node: i32) -> Result<f32> {
        let bits = self.values.quantizer.map_or(7, |q| q.bits());
        let compressed = self.values.is_compressed();
        if self.similarity == VectorSimilarityFunction::Euclidean {
            // `Euclidean.score`: always on the decompressed codes.
            self.values.vector_into(node, &mut self.scratch)?;
            let square_distance = vector_util::uint8_square_distance(&self.scratch, &self.target);
            let adjusted = square_distance as f32 * self.const_multiplier;
            return Ok(1.0 / (1.0 + adjusted));
        }
        let dot = if bits <= 4 && compressed {
            // `CompressedInt4DotProduct`: the stored bytes as they are.
            vector_util::int4_dot_product_single_packed(
                &self.target,
                self.values.packed_vector(node)?,
            )
        } else {
            self.values.vector_into(node, &mut self.scratch)?;
            if bits <= 4 {
                vector_util::int4_dot_product(&self.scratch, &self.target)
            } else {
                vector_util::uint8_dot_product(&self.scratch, &self.target)
            }
        };
        let vector_offset = self.values.score_correction_constant(node)?;
        let adjusted = dot as f32 * self.const_multiplier + self.offset_correction + vector_offset;
        Ok(match self.similarity {
            VectorSimilarityFunction::MaximumInnerProduct => {
                crate::vectors::scale_max_inner_product_score(adjusted)
            }
            _ => java_max((1.0 + adjusted) / 2.0, 0.0),
        })
    }

    /// `setScoringOrdinal(node)`: the target becomes a stored vector.
    fn set_scoring_ordinal(&mut self, node: i32) -> Result<()> {
        let mut codes = std::mem::take(&mut self.target);
        self.values.vector_into(node, &mut codes)?;
        self.target = codes;
        self.offset_correction = self.values.score_correction_constant(node)?;
        Ok(())
    }
}

/// `Lucene99ScalarQuantizedVectorScorer.getRandomVectorScorer(sim, values,
/// float[] target)`: the query quantized once, scored against every stored
/// vector without dequantizing.
#[derive(Debug, Clone)]
pub struct Lucene99ScalarQuantizedScorer<'a> {
    kernel: Lucene99Kernel<'a>,
}

impl<'a> Lucene99ScalarQuantizedScorer<'a> {
    /// Quantizes `target` with the field's quantizer.
    pub fn new(values: LegacyQuantizedVectorValues<'a>, target: &[f32]) -> Result<Self> {
        let quantizer = values
            .quantizer
            .ok_or_else(|| Error::CorruptMeta("no scalar quantizer for this field".into()))?;
        // `FlatVectorsScorer.checkDimensions`.
        if target.len() != values.dimension {
            return Err(Error::QueryDimensionMismatch(
                target.len() as i32,
                values.dimension as i32,
            ));
        }
        let mut target_bytes = vec![0u8; target.len()];
        let offset_correction =
            quantize_query(target, &mut target_bytes, values.similarity, &quantizer)?;
        Ok(Lucene99ScalarQuantizedScorer {
            kernel: Lucene99Kernel {
                similarity: values.similarity,
                const_multiplier: quantizer.constant_multiplier(),
                values,
                target: target_bytes,
                offset_correction,
                scratch: Vec::new(),
            },
        })
    }

    /// The quantized query and its offset (for tests and diagnostics).
    pub fn query(&self) -> (&[u8], f32) {
        (&self.kernel.target, self.kernel.offset_correction)
    }

    /// `ordToDoc(ord)`.
    pub fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        self.kernel.values.ord_to_doc(ord)
    }
}

impl VectorScorer for Lucene99ScalarQuantizedScorer<'_> {
    fn score(&mut self, node: i32) -> Result<f32> {
        self.kernel.score(node)
    }

    fn max_ord(&self) -> i32 {
        self.kernel.values.size
    }
}

/// `Lucene99ScalarQuantizedVectorScorer.ScalarQuantizedRandomVectorScorerSupplier.scorer()`:
/// a stored vector (set with `set_scoring_ordinal`) scored against the others
/// -- what a merge into this format builds its graph with.
#[derive(Debug, Clone)]
pub struct Lucene99ScalarQuantizedOrdScorer<'a> {
    kernel: Lucene99Kernel<'a>,
}

impl<'a> Lucene99ScalarQuantizedOrdScorer<'a> {
    /// `new byte[dimension]`, offset 0, until an ordinal is set.
    pub fn new(values: LegacyQuantizedVectorValues<'a>) -> Result<Self> {
        let quantizer = values
            .quantizer
            .ok_or_else(|| Error::CorruptMeta("no scalar quantizer for this field".into()))?;
        Ok(Lucene99ScalarQuantizedOrdScorer {
            kernel: Lucene99Kernel {
                similarity: values.similarity,
                const_multiplier: quantizer.constant_multiplier(),
                target: vec![0u8; values.dimension],
                values,
                offset_correction: 0.0,
                scratch: Vec::new(),
            },
        })
    }
}

impl VectorScorer for Lucene99ScalarQuantizedOrdScorer<'_> {
    fn score(&mut self, node: i32) -> Result<f32> {
        self.kernel.score(node)
    }

    fn max_ord(&self) -> i32 {
        self.kernel.values.size
    }
}

impl UpdateableVectorScorer for Lucene99ScalarQuantizedOrdScorer<'_> {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
        self.kernel.set_scoring_ordinal(ord)
    }
}

/// `ScalarQuantizedVectorScorer` (the generic `codecs.hnsw` one):
/// `getRandomVectorScorer`'s scorer, through
/// [`ScalarQuantizedVectorSimilarity`] on decompressed codes, and its
/// supplier's scorer when built with [`Self::for_ordinals`].
#[derive(Debug, Clone)]
pub struct ScalarQuantizedVectorScorer<'a> {
    values: LegacyQuantizedVectorValues<'a>,
    similarity: ScalarQuantizedVectorSimilarity,
    query: Vec<u8>,
    query_offset: f32,
    scratch: Vec<u8>,
}

impl<'a> ScalarQuantizedVectorScorer<'a> {
    /// `getRandomVectorScorer(sim, values, target)`.
    pub fn new(values: LegacyQuantizedVectorValues<'a>, target: &[f32]) -> Result<Self> {
        let quantizer = values
            .quantizer
            .ok_or_else(|| Error::CorruptMeta("no scalar quantizer for this field".into()))?;
        let mut query = vec![0u8; target.len()];
        let query_offset = quantize_query(target, &mut query, values.similarity, &quantizer)?;
        Ok(ScalarQuantizedVectorScorer {
            similarity: ScalarQuantizedVectorSimilarity::from_vector_similarity(
                util_similarity(values.similarity),
                quantizer.constant_multiplier(),
                quantizer.bits(),
            ),
            values,
            query,
            query_offset,
            scratch: Vec::new(),
        })
    }

    /// `ScalarQuantizedRandomVectorScorerSupplier.scorer()`: an empty query
    /// until [`UpdateableVectorScorer::set_scoring_ordinal`].
    pub fn for_ordinals(values: LegacyQuantizedVectorValues<'a>) -> Result<Self> {
        let quantizer = values
            .quantizer
            .ok_or_else(|| Error::CorruptMeta("no scalar quantizer for this field".into()))?;
        Ok(ScalarQuantizedVectorScorer {
            similarity: ScalarQuantizedVectorSimilarity::from_vector_similarity(
                util_similarity(values.similarity),
                quantizer.constant_multiplier(),
                quantizer.bits(),
            ),
            query: vec![0u8; values.dimension],
            values,
            query_offset: 0.0,
            scratch: Vec::new(),
        })
    }
}

impl VectorScorer for ScalarQuantizedVectorScorer<'_> {
    fn score(&mut self, node: i32) -> Result<f32> {
        self.values.vector_into(node, &mut self.scratch)?;
        let node_offset = self.values.score_correction_constant(node)?;
        Ok(self
            .similarity
            .score(&self.query, self.query_offset, &self.scratch, node_offset))
    }

    fn max_ord(&self) -> i32 {
        self.values.size
    }
}

impl UpdateableVectorScorer for ScalarQuantizedVectorScorer<'_> {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
        let mut q = std::mem::take(&mut self.query);
        self.values.vector_into(ord, &mut q)?;
        self.query = q;
        self.query_offset = self.values.score_correction_constant(ord)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::backward_codecs::quantized_vectors::test_support::{fixture_group, Group};
    use crate::field_infos::FieldInfo;
    use lucene_store::data_output::DataOutput;

    const ID: [u8; ID_LENGTH] = *b"legacy-sq-000001";

    /// `compressBytes`: the writer's half of [`decompress_bytes`].
    fn compress_bytes(raw: &[u8], compressed: &mut [u8]) {
        let n = compressed.len();
        for (i, c) in compressed.iter_mut().enumerate() {
            *c = (raw[i] << 4) | raw[n + i];
        }
    }

    /// One field for [`write`], every vector quantized with `quantizer`.
    struct TestField {
        number: i32,
        encoding: VectorEncoding,
        sim: VectorSimilarityFunction,
        dim: usize,
        bits: u8,
        compress: bool,
        docs: Vec<i32>,
        max_doc: i32,
        vectors: Vec<f32>,
        /// `confidenceInterval` as written (version 0 insists on a real one).
        confidence_bits: i32,
        /// Added to the recorded data offset and length.
        offset_delta: i64,
        length_delta: i64,
    }

    fn field(
        number: i32,
        sim: VectorSimilarityFunction,
        dim: usize,
        bits: u8,
        compress: bool,
    ) -> TestField {
        let n = 30usize;
        let mut s = 17u64 + number as u64;
        let vectors = (0..n * dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u32 << 24) as f32) * 1.8 - 0.9
            })
            .collect();
        TestField {
            number,
            encoding: VectorEncoding::Float32,
            sim,
            dim,
            bits,
            compress,
            docs: (0..n as i32).map(|d| d * 2 + 1).collect(),
            max_doc: 2 * n as i32 + 1,
            vectors,
            confidence_bits: 0.95f32.to_bits() as i32,
            offset_delta: 0,
            length_delta: 0,
        }
    }

    /// A minimal `Lucene99ScalarQuantizedVectorsWriter`: the quantizer's
    /// `quantize` per vector, codes compressed as `compressBytes` does.
    fn write(version: i32, fields: &[TestField]) -> (Vec<u8>, Vec<u8>) {
        let mut meta = Vec::new();
        let mut data = Vec::new();
        codec_util::write_index_header(&mut meta, META_CODEC, version, &ID, "s");
        codec_util::write_index_header(&mut data, DATA_CODEC, version, &ID, "s");
        for f in fields {
            let quantizer = ScalarQuantizer::new(-1.0, 1.0, f.bits).unwrap();
            let offset = data.len() as i64;
            let n = f.docs.len();
            for v in f.vectors.chunks(f.dim.max(1)).take(n) {
                let mut codes = vec![0u8; f.dim];
                let correction = quantize_query(v, &mut codes, f.sim, &quantizer).unwrap();
                if f.compress {
                    // An odd dimension is padded, which Java's writer cannot
                    // even do -- only to build codes its reader rejects.
                    codes.resize(f.dim.next_multiple_of(2), 0);
                    let mut packed = vec![0u8; f.dim.div_ceil(2)];
                    compress_bytes(&codes, &mut packed);
                    data.extend_from_slice(&packed);
                } else {
                    data.extend_from_slice(&codes);
                }
                data.extend_from_slice(&correction.to_le_bytes());
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
            meta.write_vlong(offset + f.offset_delta);
            meta.write_vlong(length + f.length_delta);
            meta.write_vint(f.dim as i32);
            meta.write_i32(n as i32);
            if n > 0 {
                meta.write_i32(f.confidence_bits);
                if version >= VERSION_ADD_BITS {
                    meta.write_byte(f.bits);
                    meta.write_byte(u8::from(f.compress));
                }
                meta.write_i32(quantizer.lower_quantile().to_bits() as i32);
                meta.write_i32(quantizer.upper_quantile().to_bits() as i32);
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

    fn open<'a>(meta: &[u8], data: &'a [u8]) -> Result<Lucene99ScalarQuantizedVectorsReader<'a>> {
        Lucene99ScalarQuantizedVectorsReader::open(meta, data, &ID, "s")
    }

    fn err_of<T: std::fmt::Debug>(r: Result<T>) -> String {
        r.unwrap_err().to_string()
    }

    /// The fixture's `Lucene99HnswScalarQuantizedVectorsFormat` groups.
    fn groups(version: &str) -> Vec<Group> {
        (0..10)
            .filter_map(|i| fixture_group(version, "_0", HNSW_NAME, i))
            .chain(fixture_group(version, "_0", NAME, 0))
            .collect()
    }

    #[test]
    fn every_fixture_group_opens_and_both_scorers_agree_bit_for_bit() {
        for (version, want_version) in [("9.9.2", 0), ("9.12.2", 1), ("10.2.2", 1)] {
            let groups = groups(version);
            assert!(groups.len() >= 5, "{version}: {} groups", groups.len());
            let mut saw_compressed = false;
            for g in &groups {
                let r = Lucene99ScalarQuantizedVectorsReader::open(
                    g.file("vemq"),
                    g.file("veq"),
                    &g.id,
                    &g.suffix,
                )
                .unwrap();
                assert_eq!(r.version(), want_version);
                r.check_integrity().unwrap();
                for entry in r.fields().to_vec() {
                    if entry.vector_encoding == VectorEncoding::Byte {
                        // A BYTE field has no `.vemq` entry in Java; if one
                        // were there, the float API would refuse it.
                        assert!(r.quantized_vector_values(entry.field_number).is_err());
                        continue;
                    }
                    let n = entry.field_number;
                    let values = r.quantized_vector_values(n).unwrap();
                    saw_compressed |= values.is_compressed();
                    assert_eq!(values.similarity(), entry.similarity);
                    assert_eq!(values.scalar_quantizer(), entry.scalar_quantizer.as_ref());
                    assert_eq!(r.quantization_state(n).unwrap(), entry.scalar_quantizer);
                    let target: Vec<f32> = (0..values.dimension())
                        .map(|k| ((k + 1) as f64).sin() as f32)
                        .collect();
                    let mut fast = r.scorer(n, &target).unwrap().unwrap();
                    let mut generic =
                        ScalarQuantizedVectorScorer::new(values.clone(), &target).unwrap();
                    let mut ords = Lucene99ScalarQuantizedOrdScorer::new(values.clone()).unwrap();
                    let mut generic_ords =
                        ScalarQuantizedVectorScorer::for_ordinals(values.clone()).unwrap();
                    assert_eq!(fast.max_ord(), values.size());
                    assert_eq!(generic.max_ord(), values.size());
                    assert_eq!(ords.max_ord(), values.size());
                    assert_eq!(generic_ords.max_ord(), values.size());
                    assert_eq!(fast.query().0.len(), values.dimension());
                    let mut cursor = values.doc_to_ord().unwrap();
                    let mut dq = vec![0f32; values.dimension()];
                    for ord in 0..values.size() {
                        let a = fast.score(ord).unwrap();
                        assert_eq!(a.to_bits(), generic.score(ord).unwrap().to_bits());
                        let doc = values.ord_to_doc(ord).unwrap();
                        assert_eq!(fast.ord_to_doc(ord).unwrap(), doc);
                        assert_eq!(cursor.ordinal(doc).unwrap(), Some(ord));
                        values.dequantized(ord, &mut dq).unwrap();
                        assert!(dq.iter().all(|x| x.is_finite()));
                        // The merge-time scorers: ordinal `ord` against its
                        // neighbour, both classes, the same bits.
                        let other = (ord + 1) % values.size();
                        ords.set_scoring_ordinal(ord).unwrap();
                        generic_ords.set_scoring_ordinal(ord).unwrap();
                        assert_eq!(
                            ords.score(other).unwrap().to_bits(),
                            generic_ords.score(other).unwrap().to_bits()
                        );
                    }
                    assert!(values.ord_to_doc(values.size()).is_err());
                    assert!(values.packed_vector(-1).is_err());
                }
            }
            if version != "9.9.2" {
                assert!(saw_compressed, "{version} has a compressed field");
            }
        }
    }

    #[test]
    fn hand_built_segments_round_trip_at_both_versions() {
        let sims = [
            VectorSimilarityFunction::Euclidean,
            VectorSimilarityFunction::DotProduct,
            VectorSimilarityFunction::Cosine,
            VectorSimilarityFunction::MaximumInnerProduct,
        ];
        for version in [VERSION_START, VERSION_ADD_BITS] {
            let mut fields = Vec::new();
            for (i, sim) in sims.into_iter().enumerate() {
                fields.push(field(i as i32, sim, 8, 7, false));
                if version == VERSION_ADD_BITS {
                    fields.push(field(10 + i as i32, sim, 8, 4, true));
                    fields.push(field(20 + i as i32, sim, 8, 4, false));
                }
            }
            let (meta, data) = write(version, &fields);
            let r = open(&meta, &data).unwrap();
            r.check_field_infos(&infos(&fields)).unwrap();
            for f in &fields {
                let e = r.field(f.number).unwrap();
                assert_eq!((e.bits, e.compress), (f.bits, f.compress));
                let values = r.quantized_vector_values(f.number).unwrap();
                assert_eq!(values.vector_byte_length(), if f.compress { 4 } else { 8 });
                let q = &f.vectors[..f.dim];
                let mut s = r.scorer(f.number, q).unwrap().unwrap();
                // The first stored vector is the query: the best score, as
                // far as quantization allows, and a finite one.
                let first = s.score(0).unwrap();
                assert!(first.is_finite() && first >= 0.0, "{:?} {first}", f.sim);
                let mut codes = Vec::new();
                values.vector_into(0, &mut codes).unwrap();
                assert_eq!(codes, s.query().0);
                assert_eq!(values.ord_to_doc(3).unwrap(), 7);
            }
        }
    }

    #[test]
    fn empty_fields_have_no_quantizer_and_no_scorer() {
        let mut f = field(4, VectorSimilarityFunction::Euclidean, 8, 7, false);
        f.docs.clear();
        let (meta, data) = write(VERSION_ADD_BITS, &[f]);
        let r = open(&meta, &data).unwrap();
        let e = r.field(4).unwrap();
        assert!(e.scalar_quantizer.is_none());
        assert!(e.ord_to_doc.is_empty());
        assert!(r.scorer(4, &[0.0; 8]).unwrap().is_none());
        let values = r.quantized_vector_values(4).unwrap();
        assert_eq!(values.size(), 0);
        assert!(values.dequantized(0, &mut [0.0; 8]).is_err());
        assert!(Lucene99ScalarQuantizedScorer::new(values.clone(), &[0.0; 8]).is_err());
        assert!(Lucene99ScalarQuantizedOrdScorer::new(values.clone()).is_err());
        assert!(ScalarQuantizedVectorScorer::new(values.clone(), &[0.0; 8]).is_err());
        assert!(ScalarQuantizedVectorScorer::for_ordinals(values).is_err());
    }

    #[test]
    fn caller_mistakes_are_errors() {
        let fields = [field(1, VectorSimilarityFunction::Cosine, 8, 7, false)];
        let (meta, data) = write(VERSION_ADD_BITS, &fields);
        let r = open(&meta, &data).unwrap();
        assert!(matches!(
            r.quantized_vector_values(9),
            Err(Error::UnknownField(9))
        ));
        assert!(matches!(
            r.scorer(1, &[0.5; 3]),
            Err(Error::QueryDimensionMismatch(3, 8))
        ));
        // COSINE normalizes the query, and a zero vector cannot be.
        assert!(r.scorer(1, &[0.0; 8]).is_err());

        let mut other = infos(&fields);
        other.fields[0].vector_similarity_function = VectorSimilarityFunction::Euclidean;
        assert!(err_of(r.check_field_infos(&other)).contains("similarity"));
        let mut other = infos(&fields);
        other.fields[0].vector_dimension = 9;
        assert!(err_of(r.check_field_infos(&other)).contains("dimension"));
        let empty = FieldInfos::new(Vec::new()).unwrap();
        assert!(err_of(r.check_field_infos(&empty)).contains("Invalid field number"));

        let mut byte = field(2, VectorSimilarityFunction::Euclidean, 8, 7, false);
        byte.encoding = VectorEncoding::Byte;
        let (meta, data) = write(VERSION_ADD_BITS, &[byte]);
        let r = open(&meta, &data).unwrap();
        assert!(matches!(
            r.quantized_vector_values(2),
            Err(Error::EncodingMismatch(
                2,
                VectorEncoding::Byte,
                VectorEncoding::Float32
            ))
        ));
        assert!(r.quantization_state(2).is_err());
    }

    #[test]
    fn corrupt_metadata_is_rejected() {
        let good = || field(1, VectorSimilarityFunction::DotProduct, 8, 7, false);
        // Version 0: a null (-1) or dynamic (0) confidence interval.
        for bits in [-1, 0] {
            let mut f = good();
            f.confidence_bits = bits;
            let (meta, data) = write(VERSION_START, &[f]);
            assert!(err_of(open(&meta, &data)).contains("confidence interval"));
        }
        // Version 1 ignores it.
        let mut f = good();
        f.confidence_bits = -1;
        let (meta, data) = write(VERSION_ADD_BITS, &[f]);
        open(&meta, &data).unwrap();

        // Illegal bits: rewrite the bits byte of an otherwise good entry.
        let (meta, data) = write(VERSION_ADD_BITS, &[good()]);
        let mut entry = Vec::new();
        entry.write_i32(1);
        entry.write_i32(1);
        entry.write_i32(1);
        let at = meta.windows(entry.len()).position(|w| w == entry).unwrap();
        // encoding, similarity, offset/length vlongs, dim vint, size int,
        // confidence int, then bits.
        let mut probe = SliceInput::new(&meta[at + 12..]);
        probe.read_vlong().unwrap();
        probe.read_vlong().unwrap();
        probe.read_vint().unwrap();
        probe.read_i32().unwrap();
        probe.read_i32().unwrap();
        let bits_at = at + 12 + probe.position();
        let mut bad = meta.clone();
        bad[bits_at] = 9;
        let payload = bad.len() - codec_util::FOOTER_LENGTH;
        bad.truncate(payload);
        codec_util::write_footer(&mut bad);
        assert!(err_of(open(&bad, &data)).contains("illegal quantization bits"));

        // A length that is not size * (dim + 4), and a region past the end.
        let mut f = good();
        f.length_delta = 1;
        let (meta, data) = write(VERSION_ADD_BITS, &[f]);
        assert!(err_of(open(&meta, &data)).contains("not matching"));
        let mut f = good();
        f.offset_delta = 1 << 20;
        let (meta, data) = write(VERSION_ADD_BITS, &[f]);
        assert!(err_of(open(&meta, &data)).contains("past the end"));

        // Negative and duplicate field numbers, and dimension 0.
        let mut f = good();
        f.number = -5;
        let (meta, data) = write(VERSION_ADD_BITS, &[f]);
        assert!(err_of(open(&meta, &data)).contains("Invalid field number"));
        let (meta, data) = write(VERSION_ADD_BITS, &[good(), good()]);
        assert!(err_of(open(&meta, &data)).contains("Invalid field number"));
        let mut f = good();
        f.dim = 0;
        f.docs.clear();
        let (meta, data) = write(VERSION_ADD_BITS, &[f]);
        assert!(err_of(open(&meta, &data)).contains("illegal quantized vector entry"));

        // Data and meta disagreeing on the version; a truncated data footer;
        // a meta shorter than a footer.
        let (meta, _) = write(VERSION_ADD_BITS, &[good()]);
        let (_, data0) = write(VERSION_START, &[good()]);
        assert!(err_of(open(&meta, &data0)).contains("Format versions mismatch"));
        let (meta, data) = write(VERSION_ADD_BITS, &[good()]);
        assert!(open(&meta, &data[..data.len() - 3]).is_err());
        assert!(open(&meta[..10], &data).is_err());
        // `checkIntegrity` catches a flipped payload byte `retrieveChecksum`
        // does not look at.
        let mut flipped = data.clone();
        let mid = flipped.len() / 2;
        flipped[mid] ^= 0x40;
        let r = open(&meta, &flipped).unwrap();
        assert!(r.check_integrity().is_err());
    }

    #[test]
    fn decompress_is_compress_reversed_and_rejects_odd_lengths() {
        let raw: Vec<u8> = (0..10u8).map(|i| i % 16).collect();
        let mut packed = [0u8; 5];
        compress_bytes(&raw, &mut packed);
        assert_eq!(packed[0], raw[0] << 4 | raw[5]);
        let mut back = [0u8; 10];
        decompress_bytes(&packed, &mut back).unwrap();
        assert_eq!(&back[..], &raw[..]);
        assert!(decompress_bytes(&packed, &mut [0u8; 9]).is_err());
        // An odd dimension with compression: Java's `decompressBytes` throws.
        let mut f = field(1, VectorSimilarityFunction::Euclidean, 7, 4, true);
        f.vectors = f.vectors.iter().copied().chain([0.0; 30]).collect();
        let (meta, data) = write(VERSION_ADD_BITS, &[f]);
        let r = open(&meta, &data).unwrap();
        let values = r.quantized_vector_values(1).unwrap();
        assert!(values.vector_into(0, &mut Vec::new()).is_err());
    }

    #[test]
    fn java_max_is_javas() {
        assert!(java_max(f32::NAN, 0.0).is_nan());
        assert!(java_max(0.0, f32::NAN).is_nan());
        assert!(java_max(-0.0, 0.0).is_sign_positive());
        assert!(java_max(0.0, -0.0).is_sign_positive());
        assert_eq!(java_max(-2.0, 0.0), 0.0);
        assert_eq!(java_max(3.0, 0.0), 3.0);
    }
}
