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
mod tests;
