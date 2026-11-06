//! Optimized scalar quantization (OSQ) relative to a partition centroid: the
//! quantizer of Lucene's `Lucene104ScalarQuantizedVectorsFormat`, whose
//! 1-bit encoding is Lucene's BBQ.
//!
//! Each vector is centred on the centroid it is encoded against, its
//! quantization interval optimized for that vector alone
//! ([`OptimizedScalarQuantizer`]), and stored as Lucene stores it:
//!
//! ```text
//! packed codes (ScalarEncoding::doc_packed_length(padded dim) bytes)
//! lowerInterval f32, upperInterval f32, additionalCorrection f32, quantizedComponentSum i32 (LE)
//! ```
//!
//! At 4 bits and under the dimensions are padded with zeros to a multiple of
//! 64, so the packed codes are whole 64-bit words.
//!
//! Queries are quantized against the same centroid at the encoding's query
//! width (4 bits for the 1- and 2-bit encodings) and scored with Lucene's
//! corrected integer dot product (`Lucene104ScalarQuantizedVectorScorer`).
//! Lucene centres a whole segment on one centroid; an index that passes a
//! partition's own centroid gets smaller residuals and so tighter estimates.

use lucene_util::quantization::{
    self, OptimizedScalarQuantizer, QuantizationResult, ScalarEncoding,
};
use lucene_util::{simd, vector_util};

use crate::{
    check_sample, f32_at, i32_at, prepare_vector, Error, Quantizer, QueryScorer, Result, Similarity,
};

/// The registry name of [`OsqFactory`].
pub const NAME: &str = "osq";

/// `OptimizedScalarQuantizer.MINIMUM_MSE_GRID`: the MSE-optimal interval,
/// in standard deviations around the mean, of a Gaussian quantized to 1..=8
/// bits.
const MINIMUM_MSE_GRID: [[f32; 2]; 8] = [
    [-0.798, 0.798],
    [-1.493, 1.493],
    [-2.051, 2.051],
    [-2.514, 2.514],
    [-2.916, 2.916],
    [-3.278, 3.278],
    [-3.611, 3.611],
    [-3.922, 3.922],
];

/// Quantizes `query - centroid` to `bits` per dimension into `out` (whose
/// tail past the dimension stays zero) and returns its corrective terms.
///
/// This is `OptimizedScalarQuantizer.scalarQuantize` without the interval
/// refinement: Lucene refines a query's interval as it does a document's,
/// once per segment, but an IVF search quantizes the query once per probed
/// cluster, where five refinement rounds cost more than scanning the
/// cluster's codes. The starting interval, the MSE-optimal one for a
/// Gaussian at the query's width, is kept. Statistics are plain sums rather
/// than Java's running (Welford) mean, which needs a division per dimension.
// ARITH: `bits` is the encoding's query width, 1..=8.
#[allow(clippy::arithmetic_side_effects)]
fn quantize_query(
    query: &[f32],
    centroid: &[f32],
    bits: u8,
    similarity: Similarity,
    out: &mut [u8],
) -> QuantizationResult {
    let dims = query.len() as f64;
    let (mut sum, mut sum_sq) = (0f64, 0f64);
    let (mut min, mut max) = (f32::MAX, -f32::MAX);
    let (mut norm2, mut centroid_dot) = (0f32, 0f32);
    for (&q, &c) in query.iter().zip(centroid) {
        let r = q - c;
        centroid_dot += q * c;
        norm2 += r * r;
        min = min.min(r);
        max = max.max(r);
        sum += f64::from(r);
        sum_sq += f64::from(r) * f64::from(r);
    }
    let mean = sum / dims;
    let std = (sum_sq / dims - mean * mean).max(0.0).sqrt();
    let grid = MINIMUM_MSE_GRID[usize::from(bits) - 1];
    let clamp = |x: f64| x.clamp(f64::from(min), f64::from(max)) as f32;
    let a = clamp(f64::from(grid[0]) * std + mean);
    let b = clamp(f64::from(grid[1]) * std + mean);
    let steps = ((1u32 << bits) - 1) as f32;
    let step_inv = if b > a { steps / (b - a) } else { 0.0 };
    let mut component_sum = 0i32;
    for ((o, &q), &c) in out.iter_mut().zip(query).zip(centroid) {
        let x = (q - c).clamp(a, b);
        let level = ((x - a) * step_inv + 0.5).floor();
        // `level` is in 0..=steps (at most 255).
        let level = level as u8;
        component_sum += i32::from(level);
        *o = level;
    }
    QuantizationResult {
        lower_interval: a,
        upper_interval: b,
        additional_correction: if similarity == Similarity::Euclidean {
            norm2
        } else {
            centroid_dot
        },
        quantized_component_sum: component_sum,
    }
}

/// Bytes of corrective terms after each code: three floats and an int.
const CORRECTIONS_BYTES: usize = 16;

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

/// Builds [`OsqQuantizer`]s of one [`ScalarEncoding`]. OSQ learns nothing
/// up front: training only checks its input, and the state is the
/// encoding's wire number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OsqFactory {
    encoding: ScalarEncoding,
}

impl OsqFactory {
    /// Quantizers of `encoding`.
    pub fn new(encoding: ScalarEncoding) -> Self {
        OsqFactory { encoding }
    }

    /// Lucene's BBQ: 1-bit documents, 4-bit queries.
    pub fn bbq() -> Self {
        Self::new(ScalarEncoding::SingleBitQueryNibble)
    }

    /// The encoding new quantizers use.
    pub fn encoding(&self) -> ScalarEncoding {
        self.encoding
    }
}

impl crate::QuantizerFactory for OsqFactory {
    fn name(&self) -> &str {
        NAME
    }

    fn train(
        &self,
        dimension: usize,
        similarity: Similarity,
        sample: &[f32],
    ) -> Result<Box<dyn Quantizer>> {
        check_sample(dimension, sample)?;
        Ok(Box::new(OsqQuantizer::new(
            dimension,
            similarity,
            self.encoding,
        )))
    }

    fn read(
        &self,
        dimension: usize,
        similarity: Similarity,
        state: &[u8],
    ) -> Result<Box<dyn Quantizer>> {
        let [wire] = state else {
            return Err(Error::CorruptState(format!(
                "osq state is one byte, got {}",
                state.len()
            )));
        };
        let encoding = ScalarEncoding::from_wire_number(i32::from(*wire))
            .ok_or_else(|| Error::CorruptState(format!("unknown osq encoding {wire}")))?;
        check_sample(dimension, &[])?;
        Ok(Box::new(OsqQuantizer::new(dimension, similarity, encoding)))
    }
}

/// Optimized scalar quantization of one field's vectors.
#[derive(Debug, Clone)]
pub struct OsqQuantizer {
    dimension: usize,
    similarity: Similarity,
    encoding: ScalarEncoding,
    quantizer: OptimizedScalarQuantizer,
    /// Codes before packing: one per discretized dimension.
    discrete: usize,
    packed: usize,
}

impl OsqQuantizer {
    /// A quantizer of `dimension`-dimensional vectors.
    pub fn new(dimension: usize, similarity: Similarity, encoding: ScalarEncoding) -> Self {
        OsqQuantizer {
            dimension,
            similarity,
            encoding,
            quantizer: OptimizedScalarQuantizer::new(similarity),
            discrete: discrete_dimensions(encoding, dimension),
            packed: encoding.doc_packed_length(discrete_dimensions(encoding, dimension)),
        }
    }

    /// The encoding.
    pub fn encoding(&self) -> ScalarEncoding {
        self.encoding
    }
}

/// Dimensions after padding: for the packed encodings (4 bits and under), a
/// multiple of 64, so every bit plane and nibble row is whole 64-bit words
/// and the kernels need no tail loop; otherwise Lucene's discretization. The
/// padding is zero in documents and queries alike and adds nothing to a
/// score.
fn discrete_dimensions(encoding: ScalarEncoding, dimension: usize) -> usize {
    if encoding.bits() <= 4 {
        dimension.next_multiple_of(64)
    } else {
        encoding.discrete_dimensions(dimension)
    }
}

/// The 4-bit query's four bit planes (`transpose_half_byte`, each
/// `d.len()` bytes) against a 1-bit document stripe, both whole 64-bit
/// words: `sum(q[i] * d[i])`.
#[inline]
fn bit_planes_dot(q: &[u8], d: &[u8]) -> u32 {
    let n = d.len();
    let (q0, rest) = q.split_at(n);
    let (q1, rest) = rest.split_at(n);
    let (q2, q3) = rest.split_at(n);
    let word = |b: &[u8]| {
        let mut w = [0u8; 8];
        w.copy_from_slice(b);
        u64::from_le_bytes(w)
    };
    let (mut s0, mut s1, mut s2, mut s3) = (0u32, 0u32, 0u32, 0u32);
    for ((((dw, a), b), c), e) in d
        .chunks_exact(8)
        .zip(q0.chunks_exact(8))
        .zip(q1.chunks_exact(8))
        .zip(q2.chunks_exact(8))
        .zip(q3.chunks_exact(8))
    {
        let w = word(dw);
        s0 = s0.wrapping_add((word(a) & w).count_ones());
        s1 = s1.wrapping_add((word(b) & w).count_ones());
        s2 = s2.wrapping_add((word(c) & w).count_ones());
        s3 = s3.wrapping_add((word(e) & w).count_ones());
    }
    s0.wrapping_add(s1 << 1)
        .wrapping_add(s2 << 2)
        .wrapping_add(s3 << 3)
}

/// Packs one code per discretized dimension (`unpacked`) into `packed` as
/// `encoding` stores documents.
fn pack_doc(encoding: ScalarEncoding, unpacked: &[u8], packed: &mut [u8]) {
    match encoding {
        ScalarEncoding::UnsignedByte | ScalarEncoding::SevenBit => packed.copy_from_slice(unpacked),
        ScalarEncoding::PackedNibble => quantization::pack_nibbles(unpacked, packed),
        ScalarEncoding::SingleBitQueryNibble => quantization::pack_as_binary(unpacked, packed),
        ScalarEncoding::DibitQueryNibble => quantization::transpose_dibit(unpacked, packed),
    }
}

impl Quantizer for OsqQuantizer {
    fn name(&self) -> &str {
        NAME
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn similarity(&self) -> Similarity {
        self.similarity
    }

    fn code_len(&self) -> usize {
        // ARITH: a packed length is at most a byte per dimension plus one.
        #[allow(clippy::arithmetic_side_effects)]
        let len = self.packed + CORRECTIONS_BYTES;
        len
    }

    fn uses_centroid(&self) -> bool {
        true
    }

    fn encode(&self, vector: &[f32], centroid: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dimension, "vector dimension");
        assert_eq!(centroid.len(), self.dimension, "centroid dimension");
        assert_eq!(out.len(), self.code_len(), "code length");
        let mut work = vec![0f32; self.dimension];
        prepare_vector(self.similarity, vector, &mut work);
        let mut unpacked = vec![0u8; self.discrete];
        let c = self.quantizer.scalar_quantize(
            &mut work,
            &mut unpacked,
            self.encoding.bits(),
            centroid,
        );
        let (codes, tail) = out.split_at_mut(self.packed);
        pack_doc(self.encoding, &unpacked, codes);
        tail[..4].copy_from_slice(&c.lower_interval.to_le_bytes());
        tail[4..8].copy_from_slice(&c.upper_interval.to_le_bytes());
        tail[8..12].copy_from_slice(&c.additional_correction.to_le_bytes());
        tail[12..16].copy_from_slice(&c.quantized_component_sum.to_le_bytes());
    }

    fn query_scorer(&self, query: &[f32]) -> Box<dyn QueryScorer + '_> {
        assert_eq!(query.len(), self.dimension, "query dimension");
        let mut normalized = vec![0f32; self.dimension];
        prepare_vector(self.similarity, query, &mut normalized);
        Box::new(OsqQueryScorer {
            q: self,
            query: normalized,
            unpacked: vec![0u8; self.discrete],
            codes: vec![
                0u8;
                if self.encoding.is_asymmetric() {
                    self.encoding.query_packed_length(self.discrete)
                } else {
                    self.discrete
                }
            ],
            corrections: QuantizationResult {
                lower_interval: 0.0,
                upper_interval: 0.0,
                additional_correction: 0.0,
                quantized_component_sum: 0,
            },
            centroid_dp: 0.0,
        })
    }

    fn state(&self) -> Vec<u8> {
        // Wire numbers are 0..=4.
        vec![u8::try_from(self.encoding.wire_number()).unwrap_or(u8::MAX)]
    }
}

/// A query quantized against one centroid at a time.
#[derive(Debug)]
struct OsqQueryScorer<'q> {
    q: &'q OsqQuantizer,
    query: Vec<f32>,
    unpacked: Vec<u8>,
    /// The query codes as the kernel takes them: unpacked, or transposed
    /// into four bit planes for the asymmetric encodings.
    codes: Vec<u8>,
    corrections: QuantizationResult,
    centroid_dp: f32,
}

impl QueryScorer for OsqQueryScorer<'_> {
    fn set_centroid(&mut self, centroid: &[f32]) {
        assert_eq!(centroid.len(), self.q.dimension, "centroid dimension");
        let encoding = self.q.encoding;
        self.corrections = quantize_query(
            &self.query,
            centroid,
            encoding.query_bits(),
            self.q.similarity,
            &mut self.unpacked,
        );
        if encoding.is_asymmetric() {
            quantization::transpose_half_byte(&self.unpacked, &mut self.codes);
        } else {
            self.codes.copy_from_slice(&self.unpacked);
        }
        self.centroid_dp = if self.q.similarity == Similarity::Euclidean {
            0.0
        } else {
            simd::dot_f32(centroid, centroid)
        };
    }

    fn score(&self, code: &[u8]) -> f32 {
        self.score_code(code, &self.terms())
    }

    fn score_block(&self, codes: &[u8], scores: &mut [f32]) {
        let code_len = self.q.code_len();
        assert_eq!(
            Some(codes.len()),
            code_len.checked_mul(scores.len()),
            "codes must hold one code per score"
        );
        let terms = self.terms();
        for (s, code) in scores.iter_mut().zip(codes.chunks_exact(code_len)) {
            *s = self.score_code(code, &terms);
        }
    }
}

/// The query's half of the corrected score, fixed for one centroid.
#[derive(Debug, Clone, Copy)]
struct QueryTerms {
    ay: f32,
    ly: f32,
    /// `ay * dimension`.
    ay_dims: f32,
    /// `ly * quantizedComponentSum`.
    ly_y1: f32,
    scale: f32,
    /// The constant part of the corrections: the query's additional
    /// correction, less the centroid's squared norm for inner products.
    additional: f32,
}

impl OsqQueryScorer<'_> {
    fn terms(&self) -> QueryTerms {
        let encoding = self.q.encoding;
        // `bits` is 1..=8 for every encoding, so the LUT lookups are in range.
        let query_scale = SCALE_LUT[usize::from(encoding.query_bits()).saturating_sub(1)];
        let q = &self.corrections;
        let ay = q.lower_interval;
        let ly = (q.upper_interval - ay) * query_scale;
        QueryTerms {
            ay,
            ly,
            ay_dims: ay * self.q.dimension as f32,
            ly_y1: ly * q.quantized_component_sum as f32,
            scale: SCALE_LUT[usize::from(encoding.bits()).saturating_sub(1)],
            additional: if self.q.similarity == Similarity::Euclidean {
                q.additional_correction
            } else {
                q.additional_correction - self.centroid_dp
            },
        }
    }

    /// `Lucene104ScalarQuantizedVectorScorer.quantizedScore`: the integer dot
    /// product of the codes, with both quantizations undone through their
    /// corrective terms.
    #[inline]
    fn score_code(&self, code: &[u8], t: &QueryTerms) -> f32 {
        let (doc, tail) = code.split_at(self.q.packed);
        let qc_dist = match self.q.encoding {
            ScalarEncoding::UnsignedByte => vector_util::uint8_dot_product(&self.codes, doc) as f32,
            ScalarEncoding::SevenBit => vector_util::dot_product_i8(&self.codes, doc) as f32,
            ScalarEncoding::PackedNibble => {
                vector_util::int4_dot_product_single_packed(&self.codes, doc) as f32
            }
            ScalarEncoding::SingleBitQueryNibble => bit_planes_dot(&self.codes, doc) as f32,
            ScalarEncoding::DibitQueryNibble => {
                let (lower, upper) = doc.split_at(doc.len() / 2);
                let low = bit_planes_dot(&self.codes, lower);
                let high = bit_planes_dot(&self.codes, upper);
                (low as f32) + 2.0 * (high as f32)
            }
        };
        let ax = f32_at(tail, 0);
        let lx = (f32_at(tail, 4) - ax) * t.scale;
        let x1 = i32_at(tail, 12) as f32;
        let score = ax * t.ay_dims + t.ay * lx * x1 + ax * t.ly_y1 + lx * t.ly * qc_dist;
        let corrections = t.additional + f32_at(tail, 8);
        match self.q.similarity {
            Similarity::Euclidean => 1.0 / (1.0 + (corrections - 2.0 * score).max(0.0)),
            Similarity::MaximumInnerProduct => {
                quantization::scale_max_inner_product_score(score + corrections)
            }
            Similarity::DotProduct | Similarity::Cosine => {
                ((1.0 + score + corrections) / 2.0).max(0.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::QuantizerFactory;

    const ENCODINGS: [ScalarEncoding; 5] = [
        ScalarEncoding::UnsignedByte,
        ScalarEncoding::PackedNibble,
        ScalarEncoding::SevenBit,
        ScalarEncoding::SingleBitQueryNibble,
        ScalarEncoding::DibitQueryNibble,
    ];
    const SIMS: [Similarity; 4] = [
        Similarity::Euclidean,
        Similarity::DotProduct,
        Similarity::Cosine,
        Similarity::MaximumInnerProduct,
    ];

    fn vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        s = s
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                    })
                    .collect()
            })
            .collect()
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    /// Estimated scores track exact ones: high correlation and small error
    /// for every encoding and similarity, tighter with more bits.
    #[test]
    fn estimates_track_exact_scores() {
        let dim = 64;
        let mut data = vectors(200, dim, 7);
        let queries = vectors(5, dim, 99);
        for sim in SIMS {
            if sim == Similarity::DotProduct {
                data = data.iter().map(|v| normalize(v)).collect();
            }
            let queries: Vec<Vec<f32>> = if sim == Similarity::DotProduct {
                queries.iter().map(|v| normalize(v)).collect()
            } else {
                queries.clone()
            };
            let mut centroid = vec![0f32; dim];
            for v in &data {
                for (c, x) in centroid.iter_mut().zip(v) {
                    *c += x / data.len() as f32;
                }
            }
            for enc in ENCODINGS {
                let q = OsqFactory::new(enc).train(dim, sim, &[]).unwrap();
                assert_eq!(q.name(), NAME);
                assert_eq!(q.dimension(), dim);
                assert_eq!(q.similarity(), sim);
                assert!(q.uses_centroid());
                let mut codes = vec![0u8; q.code_len() * data.len()];
                for (v, out) in data.iter().zip(codes.chunks_exact_mut(q.code_len())) {
                    q.encode(v, &centroid, out);
                }
                let mut err = 0f64;
                for query in &queries {
                    let mut s = q.query_scorer(query);
                    s.set_centroid(&centroid);
                    let mut est = vec![0f32; data.len()];
                    s.score_block(&codes, &mut est);
                    for (v, e) in data.iter().zip(&est) {
                        let exact = sim.score(query, v);
                        err += ((exact - e) / exact.max(1e-3)).abs() as f64;
                    }
                }
                let mean = err / (queries.len() * data.len()) as f64;
                let bound = if enc.bits() <= 2 { 0.25 } else { 0.05 };
                assert!(mean < bound, "{sim:?} {enc:?}: mean relative error {mean}");
            }
        }
    }

    #[test]
    fn state_round_trips_and_rejects_garbage() {
        let f = OsqFactory::bbq();
        assert_eq!(f.encoding(), ScalarEncoding::SingleBitQueryNibble);
        for enc in ENCODINGS {
            let q = OsqFactory::new(enc)
                .train(16, Similarity::Euclidean, &[0.0; 32])
                .unwrap();
            let back = f.read(16, Similarity::Euclidean, &q.state()).unwrap();
            assert_eq!(back.code_len(), q.code_len());
            assert_eq!(back.state(), q.state());
        }
        assert!(matches!(
            f.read(16, Similarity::Euclidean, &[]),
            Err(Error::CorruptState(_))
        ));
        assert!(matches!(
            f.read(16, Similarity::Euclidean, &[9]),
            Err(Error::CorruptState(_))
        ));
        assert!(f.read(0, Similarity::Euclidean, &[3]).is_err());
        assert!(f.train(3, Similarity::Euclidean, &[0.0; 4]).is_err());
    }

    #[test]
    fn bbq_code_is_one_bit_per_dimension_plus_corrections() {
        let q = OsqQuantizer::new(
            100,
            Similarity::Euclidean,
            ScalarEncoding::SingleBitQueryNibble,
        );
        assert_eq!(q.encoding(), ScalarEncoding::SingleBitQueryNibble);
        // 100 dims pad to 128 bits (16 bytes); 16 bytes of corrections.
        assert_eq!(q.code_len(), 16 + 16);
    }

    #[test]
    fn euclidean_self_score_is_near_one_and_mip_scales() {
        let dim = 32;
        let v = vectors(1, dim, 3).remove(0);
        let centroid = vec![0.1f32; dim];
        for (sim, check) in [
            (Similarity::Euclidean, 0.9f32),
            (Similarity::MaximumInnerProduct, 1.0),
        ] {
            let q = OsqQuantizer::new(dim, sim, ScalarEncoding::UnsignedByte);
            let mut code = vec![0u8; q.code_len()];
            q.encode(&v, &centroid, &mut code);
            let mut s = q.query_scorer(&v);
            s.set_centroid(&centroid);
            assert!(s.score(&code) >= check, "{sim:?} {}", s.score(&code));
        }
    }

    #[test]
    #[should_panic(expected = "code length")]
    fn encode_checks_lengths() {
        let q = OsqQuantizer::new(8, Similarity::Euclidean, ScalarEncoding::UnsignedByte);
        q.encode(&[0.0; 8], &[0.0; 8], &mut [0u8; 3]);
    }

    #[test]
    #[should_panic(expected = "centroid dimension")]
    fn set_centroid_checks_length() {
        let q = OsqQuantizer::new(8, Similarity::Euclidean, ScalarEncoding::UnsignedByte);
        q.query_scorer(&[0.0; 8]).set_centroid(&[0.0; 4]);
    }
}
