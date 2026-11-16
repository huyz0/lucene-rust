//! The quantized codes of a [`KnnVectorsFormat::HnswQuantized`] instance
//! (`.vqm` metadata, `.vqd` codes), written beside the instance's raw vectors
//! and HNSW graph at flush and at merge. The graph is built on the raw
//! vectors, as for [`KnnVectorsFormat::Hnsw`], so its files are the plain
//! HNSW ones; a search walks it scoring the codes and reranks the candidates
//! on the raw vectors. Only `FLOAT32` fields get codes.
//!
//! Each field's codes are one quantizer's ([`quantizer_factory`]), encoded
//! relative to the mean of the field's vectors in the segment (unit-normalized
//! first for `Cosine`), in vector ordinal order. A merge encodes the merged
//! vectors afresh.
//!
//! `.vqm`: an index header ([`META_CODEC`]), the field count (vint), then per
//! field its number (vint), similarity ordinal (byte), dimension (vint),
//! quantizer name (string), quantizer state (vint length, bytes), centroid
//! (`dimension` little-endian `f32`), vector count (vint), code length
//! (vint) and the codes' offset in `.vqd` (vlong); a footer.
//!
//! `.vqd`: an index header ([`DATA_CODEC`]), each field's codes end to end
//! from an offset aligned to [`CODES_ALIGNMENT`], a footer.
//!
//! [`KnnVectorsFormat::HnswQuantized`]: lucene_codecs::per_field_knn_vectors::KnnVectorsFormat::HnswQuantized
//! [`KnnVectorsFormat::Hnsw`]: lucene_codecs::per_field_knn_vectors::KnnVectorsFormat::Hnsw

use std::sync::Arc;

use rayon::prelude::*;

use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::scalar_quantized_vectors::util_similarity;
use lucene_store::codec_util::{self, FOOTER_LENGTH, ID_LENGTH};
use lucene_store::directory::Input;
use lucene_store::{DataInput, DataOutput, SliceInput};
use lucene_vector_quant::{Quantizer, QuantizerRegistry, QueryScorer};

use crate::ivf_vectors::quantizer_factory;

type Result<T> = std::result::Result<T, lucene_ivf::Error>;

/// The codes' extensions: metadata, then codes.
pub const EXTENSIONS: [&str; 2] = ["vqm", "vqd"];

/// `.vqm`'s codec name.
pub const META_CODEC: &str = "HnswQuantizedMeta";
/// `.vqd`'s codec name.
pub const DATA_CODEC: &str = "HnswQuantizedData";
/// The format version.
pub const VERSION_CURRENT: i32 = 0;

/// Each field's codes start at a multiple of this many bytes of `.vqd`.
pub const CODES_ALIGNMENT: usize = 64;

/// The most vectors a quantizer is trained on.
const TRAINING_SAMPLE: usize = 65_536;

/// An instance's code files by extension.
pub(crate) type Files = Vec<(&'static str, Vec<u8>)>;

/// One float field of a quantized HNSW instance: its number, similarity,
/// dimension and vectors end to end, in ordinal order.
pub(crate) struct CodesField<'a> {
    pub number: i32,
    pub similarity: VectorSimilarityFunction,
    pub dimension: usize,
    pub vectors: &'a [f32],
}

fn corrupt<T>(msg: impl Into<String>) -> Result<T> {
    Err(lucene_store::Error::Corrupted(msg.into()).into())
}

/// The mean of `vectors` (`dimension` floats each), each unit-normalized
/// first for `Cosine`, summed in ordinal order in `f64`.
fn centroid(vectors: &[f32], dimension: usize, cosine: bool) -> Vec<f32> {
    let mut sum = vec![0f64; dimension];
    let mut n = 0usize;
    for v in vectors.chunks_exact(dimension) {
        let scale = if cosine {
            let norm = lucene_util::simd::dot_f32(v, v).sqrt();
            if norm > 0.0 {
                1.0 / f64::from(norm)
            } else {
                0.0
            }
        } else {
            1.0
        };
        for (s, &x) in sum.iter_mut().zip(v) {
            *s += f64::from(x) * scale;
        }
        n = n.saturating_add(1);
    }
    let inv = if n > 0 { 1.0 / n as f64 } else { 0.0 };
    sum.into_iter().map(|s| (s * inv) as f32).collect()
}

/// Writes the codes of a [`KnnVectorsFormat::HnswQuantized`] instance's
/// float fields: `None` for another format, or when no field has vectors.
///
/// # Errors
/// An unknown quantizer, or vectors the quantizer refuses.
pub(crate) fn write(
    format: KnnVectorsFormat,
    fields: &[CodesField<'_>],
    segment_id: &[u8; ID_LENGTH],
    suffix: &str,
) -> Result<Option<Files>> {
    let Some(quantizer) = format.hnsw_quantizer() else {
        return Ok(None);
    };
    let factory = quantizer_factory(quantizer)
        .filter(|f| f.candidates().is_none())
        .ok_or_else(|| {
            lucene_ivf::Error::IllegalArgument(format!("unknown HNSW quantizer {quantizer}"))
        })?;
    let mut meta = Vec::new();
    let mut data = Vec::new();
    codec_util::write_index_header(&mut meta, META_CODEC, VERSION_CURRENT, segment_id, suffix);
    codec_util::write_index_header(&mut data, DATA_CODEC, VERSION_CURRENT, segment_id, suffix);
    let written: Vec<&CodesField<'_>> = fields
        .iter()
        .filter(|f| f.dimension > 0 && !f.vectors.is_empty())
        .collect();
    if written.is_empty() {
        return Ok(None);
    }
    meta.write_vint(i32::try_from(written.len()).unwrap_or(i32::MAX));
    for field in written {
        let dim = field.dimension;
        if field.vectors.len().checked_rem(dim) != Some(0) {
            return Err(lucene_ivf::Error::IllegalArgument(format!(
                "{} floats are not whole vectors of dimension {dim}",
                field.vectors.len()
            )));
        }
        let count = field.vectors.len().checked_div(dim).unwrap_or(0);
        let similarity = util_similarity(field.similarity);
        let sample = &field.vectors[..count.min(TRAINING_SAMPLE).saturating_mul(dim)];
        let q = factory.train(dim, similarity, sample)?;
        let center = centroid(
            field.vectors,
            dim,
            similarity == lucene_vector_quant::Similarity::Cosine,
        );
        let prepared = q
            .prepare_centroid(&center)
            .unwrap_or_else(|| center.clone());
        let code_len = q.code_len();
        let aligned = data.len().next_multiple_of(CODES_ALIGNMENT);
        data.resize(aligned, 0);
        let offset = data.len();
        data.resize(offset.saturating_add(count.saturating_mul(code_len)), 0);
        if code_len > 0 {
            data[offset..]
                .par_chunks_mut(code_len.saturating_mul(1024))
                .zip(field.vectors.par_chunks(dim.saturating_mul(1024)))
                .for_each(|(out, vectors)| {
                    for (code, v) in out
                        .chunks_exact_mut(code_len)
                        .zip(vectors.chunks_exact(dim))
                    {
                        q.encode_prepared(v, &prepared, code);
                    }
                });
        }
        meta.write_vint(field.number);
        meta.write_byte(lucene_ivf::similarity_ordinal(similarity));
        meta.write_vint(i32::try_from(dim).unwrap_or(i32::MAX));
        meta.write_string(q.name());
        let state = q.state();
        meta.write_vint(i32::try_from(state.len()).unwrap_or(i32::MAX));
        meta.write_bytes(&state);
        for c in &center {
            meta.write_bytes(&c.to_le_bytes());
        }
        meta.write_vint(i32::try_from(count).unwrap_or(i32::MAX));
        meta.write_vint(i32::try_from(code_len).unwrap_or(i32::MAX));
        meta.write_vlong(i64::try_from(offset).unwrap_or(i64::MAX));
    }
    codec_util::write_footer(&mut meta);
    codec_util::write_footer(&mut data);
    Ok(Some(vec![(EXTENSIONS[0], meta), (EXTENSIONS[1], data)]))
}

/// One field's codes and the quantizer that made them.
#[derive(Debug)]
pub struct CodesFieldReader {
    pub number: i32,
    similarity: lucene_vector_quant::Similarity,
    dimension: usize,
    quantizer: Box<dyn Quantizer>,
    /// The centroid as the quantizer scores against it
    /// ([`Quantizer::prepare_centroid`]).
    prepared: Vec<f32>,
    count: usize,
    code_len: usize,
    offset: usize,
}

impl CodesFieldReader {
    /// The field's quantizer.
    pub fn quantizer(&self) -> &dyn Quantizer {
        self.quantizer.as_ref()
    }

    /// The number of codes: the field's vector count.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Bytes of one code.
    pub fn code_len(&self) -> usize {
        self.code_len
    }

    /// Checks the codes against the field's raw vectors: the same
    /// dimension, similarity and vector count.
    ///
    /// # Errors
    /// [`lucene_store::Error::Corrupted`] on any mismatch.
    pub fn check_matches(
        &self,
        dimension: usize,
        similarity: VectorSimilarityFunction,
        count: usize,
    ) -> Result<()> {
        if self.dimension != dimension
            || self.similarity != util_similarity(similarity)
            || self.count != count
        {
            return corrupt(format!(
                "field {}: {} codes of dimension {} ({:?}) for {count} vectors of dimension \
                 {dimension} ({similarity:?})",
                self.number, self.count, self.dimension, self.similarity
            ));
        }
        Ok(())
    }

    /// `query` prepared against the field's codes.
    pub fn query_scorer(&self, query: &[f32]) -> Box<dyn QueryScorer + '_> {
        let mut scorer = self.quantizer.query_scorer(query);
        scorer.set_prepared_centroid(&self.prepared);
        scorer
    }
}

/// An instance's codes, opened from its `.vqm` and `.vqd`.
#[derive(Debug)]
pub struct HnswCodesReader {
    fields: Vec<CodesFieldReader>,
    data: Arc<Input>,
}

impl HnswCodesReader {
    /// Opens an instance's codes.
    ///
    /// # Errors
    /// A header, footer or checksum mismatch, codes that do not fit `.vqd`,
    /// or a quantizer this release does not know.
    pub fn open(
        meta: &[u8],
        data: Arc<Input>,
        segment_id: &[u8; ID_LENGTH],
        suffix: &str,
    ) -> Result<Self> {
        let registry = QuantizerRegistry::with_builtins();
        let header_len = codec_util::index_header_length(DATA_CODEC, suffix);
        let Some(data_end) = data.len().checked_sub(FOOTER_LENGTH) else {
            return corrupt(format!("{DATA_CODEC}: {} bytes have no footer", data.len()));
        };
        if data_end < header_len {
            return corrupt(format!("{DATA_CODEC}: {} bytes are truncated", data.len()));
        }
        let data_version = codec_util::check_index_header(
            &mut SliceInput::new(&data[..header_len]),
            DATA_CODEC,
            VERSION_CURRENT,
            VERSION_CURRENT,
            segment_id,
            suffix,
        )?
        .version;
        codec_util::retrieve_checksum(&data[data_end..])?;
        let mut input = SliceInput::new(meta);
        let version = codec_util::check_index_header(
            &mut input,
            META_CODEC,
            VERSION_CURRENT,
            VERSION_CURRENT,
            segment_id,
            suffix,
        )?
        .version;
        if version != data_version {
            return corrupt(format!(
                "format versions differ: meta {version}, data {data_version}"
            ));
        }
        let count = |input: &mut SliceInput<'_>, what: &str| -> Result<usize> {
            let v = input.read_vint()?;
            usize::try_from(v).or_else(|_| corrupt(format!("negative {what}: {v}")))
        };
        let field_count = count(&mut input, "field count")?;
        let mut fields: Vec<CodesFieldReader> = Vec::new();
        for _ in 0..field_count {
            let number = input.read_vint()?;
            let Some(similarity) = lucene_ivf::similarity_from_ordinal(input.read_byte()?) else {
                return corrupt(format!("field {number}: unknown similarity"));
            };
            let dimension = count(&mut input, "dimension")?;
            let name = input.read_string()?;
            let state_len = count(&mut input, "state length")?;
            if state_len > meta.len() {
                return corrupt(format!("field {number}: state of {state_len} bytes"));
            }
            let mut state = vec![0u8; state_len];
            input.read_bytes(&mut state)?;
            if dimension.saturating_mul(4) > meta.len() {
                return corrupt(format!("field {number}: dimension {dimension}"));
            }
            let mut center = vec![0f32; dimension];
            for c in &mut center {
                let mut b = [0u8; 4];
                input.read_bytes(&mut b)?;
                *c = f32::from_le_bytes(b);
            }
            let vectors = count(&mut input, "vector count")?;
            let code_len = count(&mut input, "code length")?;
            let offset = usize::try_from(input.read_vlong()?)
                .or_else(|_| corrupt(format!("field {number}: negative offset")))?;
            let quantizer = registry.get(&name)?.read(dimension, similarity, &state)?;
            if quantizer.code_len() != code_len {
                return corrupt(format!(
                    "field {number}: code length {code_len}, quantizer's {}",
                    quantizer.code_len()
                ));
            }
            let end = vectors
                .checked_mul(code_len)
                .and_then(|n| n.checked_add(offset));
            if offset < header_len || end.is_none_or(|e| e > data_end) {
                return corrupt(format!(
                    "field {number}: {vectors} codes of {code_len} bytes at {offset} do not fit \
                     {data_end} bytes"
                ));
            }
            if fields.iter().any(|f| f.number == number) {
                return corrupt(format!("field {number} recorded twice"));
            }
            let prepared = quantizer.prepare_centroid(&center).unwrap_or(center);
            fields.push(CodesFieldReader {
                number,
                similarity,
                dimension,
                quantizer,
                prepared,
                count: vectors,
                code_len,
                offset,
            });
        }
        codec_util::check_footer(&mut input, meta.len())?;
        Ok(HnswCodesReader { fields, data })
    }

    /// The codes of field `number`, if the instance holds any.
    pub fn field(&self, number: i32) -> Option<&CodesFieldReader> {
        self.fields.iter().find(|f| f.number == number)
    }

    /// Field `field`'s codes, end to end in ordinal order.
    pub fn codes(&self, field: &CodesFieldReader) -> &[u8] {
        let end = field
            .offset
            .saturating_add(field.count.saturating_mul(field.code_len));
        self.data.get(field.offset..end).unwrap_or(&[])
    }

    /// Verifies `.vqd`'s checksum, reading it whole.
    ///
    /// # Errors
    /// A checksum mismatch.
    pub fn check_integrity(&self) -> Result<()> {
        let end = self.data.len().saturating_sub(FOOTER_LENGTH);
        codec_util::check_whole_file_footer(&self.data, end)?;
        Ok(())
    }

    /// A float `query`'s scorer over field `field`'s codes, by ordinal.
    pub fn scorer<'a>(&'a self, field: &'a CodesFieldReader, query: &[f32]) -> CodeScorer<'a> {
        CodeScorer {
            scorer: field.query_scorer(query),
            codes: self.codes(field),
            code_len: field.code_len,
            count: i32::try_from(field.count).unwrap_or(i32::MAX),
        }
    }
}

/// Scores one query against a field's codes by vector ordinal: what a
/// graph walk over a [`KnnVectorsFormat::HnswQuantized`] field scores with.
///
/// [`KnnVectorsFormat::HnswQuantized`]: lucene_codecs::per_field_knn_vectors::KnnVectorsFormat::HnswQuantized
pub struct CodeScorer<'a> {
    scorer: Box<dyn QueryScorer + 'a>,
    codes: &'a [u8],
    code_len: usize,
    count: i32,
}

impl std::fmt::Debug for CodeScorer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeScorer")
            .field("code_len", &self.code_len)
            .field("count", &self.count)
            .finish()
    }
}

impl CodeScorer<'_> {
    fn code(&self, node: i32) -> std::result::Result<&[u8], lucene_codecs::vectors::Error> {
        usize::try_from(node)
            .ok()
            .and_then(|n| n.checked_mul(self.code_len))
            .and_then(|at| self.codes.get(at..at.checked_add(self.code_len)?))
            .ok_or(lucene_codecs::vectors::Error::OrdOutOfRange(
                node, self.count,
            ))
    }
}

impl lucene_codecs::hnsw::VectorScorer for CodeScorer<'_> {
    fn read_ahead_span(&self) -> usize {
        lucene_codecs::hnsw::read_ahead_span(self.code_len)
    }

    fn read_ahead(&self, from: i32, to: i32) {
        lucene_codecs::hnsw::read_ahead_rows(self.codes, self.count, self.code_len, from, to);
    }

    fn score(&mut self, node: i32) -> std::result::Result<f32, lucene_codecs::vectors::Error> {
        Ok(self.scorer.score(self.code(node)?))
    }

    fn max_ord(&self) -> i32 {
        self.count
    }

    fn prefetch(&self, nodes: &[i32]) {
        for &n in nodes {
            if let Ok(code) = self.code(n) {
                lucene_util::simd::prefetch_bytes(code);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests;
