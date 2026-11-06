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
    // Eight independent lanes, so the loop vectorizes; sums in `f64`, as
    // the variance below subtracts two of them.
    let (mut sum, mut sq) = ([0f64; 8], [0f64; 8]);
    let (mut lo, mut hi) = ([f32::MAX; 8], [-f32::MAX; 8]);
    let mut lane = |j: usize, r: f32| {
        sum[j] += f64::from(r);
        sq[j] += f64::from(r) * f64::from(r);
        lo[j] = lo[j].min(r);
        hi[j] = hi[j].max(r);
    };
    let (qs, qt) = query.split_at(query.len() - query.len() % 8);
    let (cs, ct) = centroid.split_at(qs.len());
    for (q8, c8) in qs.chunks_exact(8).zip(cs.chunks_exact(8)) {
        for j in 0..8 {
            lane(j, q8[j] - c8[j]);
        }
    }
    for (j, (&q, &c)) in qt.iter().zip(ct).enumerate() {
        lane(j, q - c);
    }
    let norm2: f64 = sq.iter().sum();
    let min = lo.iter().fold(f32::MAX, |m, &x| m.min(x));
    let max = hi.iter().fold(-f32::MAX, |m, &x| m.max(x));
    let mean = sum.iter().sum::<f64>() / dims;
    let std = (norm2 / dims - mean * mean).max(0.0).sqrt();
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
            norm2 as f32
        } else {
            simd::dot_f32(query, centroid)
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

    /// The discretized dimension: a multiple of 64 at 4 bits and under.
    fn code_dimension(&self, dimension: usize) -> usize {
        discrete_dimensions(self.encoding, dimension)
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

/// The integer dot product of a query's codes (`query`, one per
/// discretized dimension) with a document's bit planes (`doc`, plane `p`
/// weighted `2^p`), each plane packed as Lucene packs binary codes:
/// dimension `d` in bit `7 - d % 8` of byte `d / 8`.
// ARITH: plane weights are at most 2 and sums at most 2 * 15 * 3 * dims.
#[allow(clippy::arithmetic_side_effects)]
fn bit_planes_dot(query: &[u8], doc: &[u8]) -> u32 {
    let plane_len = query.len() / 8;
    let mut sum = 0u32;
    for (p, plane) in doc.chunks_exact(plane_len.max(1)).enumerate() {
        let mut s = 0u32;
        for (q8, &bits) in query.chunks_exact(8).zip(plane) {
            for (k, &q) in q8.iter().enumerate() {
                s += u32::from(q) * u32::from((bits >> (7 - k)) & 1);
            }
        }
        sum += s << p;
    }
    sum
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
            corrections: QuantizationResult {
                lower_interval: 0.0,
                upper_interval: 0.0,
                additional_correction: 0.0,
                quantized_component_sum: 0,
            },
            centroid_dp: 0.0,
            luts: vec![0u8; self.fast_scan_groups().saturating_mul(16)],
        })
    }

    fn state(&self) -> Vec<u8> {
        // Wire numbers are 0..=4.
        vec![u8::try_from(self.encoding.wire_number()).unwrap_or(u8::MAX)]
    }

    fn block_len(&self, n: usize) -> usize {
        if self.fast_scan_planes() == 0 {
            return self.code_len().saturating_mul(n);
        }
        n.div_ceil(FAST_SCAN_BLOCK)
            .saturating_mul(self.fast_scan_block_bytes())
    }

    // ARITH: offsets are below the code length, groups below the discretized
    // dimension; `codes` is checked to be whole codes.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_block(&self, codes: &[u8], out: &mut Vec<u8>) {
        let code_len = self.code_len();
        assert_eq!(codes.len() % code_len, 0, "codes must be whole codes");
        let planes = self.fast_scan_planes();
        if planes == 0 {
            out.extend_from_slice(codes);
            return;
        }
        let plane_len = self.discrete / 8;
        let groups = self.discrete / 4;
        for block in codes.chunks(FAST_SCAN_BLOCK * code_len) {
            let code = |j: usize| block.get(j * code_len..(j + 1) * code_len);
            // Sub-code `g` of plane `p`: the bits of dimensions 4g..4g+4,
            // dimension 4g + k in bit k. Packed planes hold dimension `d` in
            // bit `7 - d % 8` of byte `d / 8`.
            let nibble = |j: usize, p: usize, g: usize| -> u8 {
                code(j).map_or(0, |c| {
                    let byte = c[p * plane_len + g / 2];
                    let half = if g.is_multiple_of(2) {
                        byte >> 4
                    } else {
                        byte & 15
                    };
                    REVERSE_NIBBLE[usize::from(half)]
                })
            };
            for p in 0..planes {
                for g in 0..groups {
                    for j in 0..16 {
                        out.push(nibble(j, p, g) | (nibble(j + 16, p, g) << 4));
                    }
                }
            }
            // The corrective terms, one array of 32 per term.
            for at in [0, 4, 8, 12] {
                for j in 0..FAST_SCAN_BLOCK {
                    match code(j) {
                        Some(c) => {
                            out.extend_from_slice(&c[self.packed + at..self.packed + at + 4])
                        }
                        None => out.extend_from_slice(&[0; 4]),
                    }
                }
            }
        }
    }
}

/// Codes per fast-scan block.
const FAST_SCAN_BLOCK: usize = simd::FAST_SCAN_BLOCK;

/// `REVERSE_NIBBLE[x]` is `x` with its four bits in reverse order.
const REVERSE_NIBBLE: [u8; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];

/// Fills one 16-entry table per bit plane and group of four dimensions:
/// entry `v` of group `g` is the sum of `query[4g + k]` over the bits `k`
/// set in `v`, shifted left by the plane. `query` holds one code of at most
/// 4 bits per discretized dimension, so an entry is at most `2 * 60`.
// ARITH: table indices are below `luts.len()`; entries are sums of at most
// four 4-bit values, doubled at most once.
#[allow(clippy::arithmetic_side_effects)]
fn fill_luts(query: &[u8], luts: &mut [u8]) {
    let groups = query.len() / 4;
    for (i, lut) in luts.chunks_exact_mut(16).enumerate() {
        let (plane, g) = (i / groups, i % groups);
        let q = &query[4 * g..4 * g + 4];
        lut[0] = 0;
        for v in 1..16usize {
            let low = v.trailing_zeros() as usize;
            lut[v] = lut[v & (v - 1)] + (q[low] << plane);
        }
    }
}

impl OsqQuantizer {
    /// Bit planes of a document code scored by fast scan: 1 for BBQ, 2 for
    /// the 2-bit encoding, 0 for the encodings scored code by code.
    fn fast_scan_planes(&self) -> usize {
        match self.encoding {
            ScalarEncoding::SingleBitQueryNibble => 1,
            ScalarEncoding::DibitQueryNibble => 2,
            _ => 0,
        }
    }

    /// Fast-scan table groups: four dimensions per group, per plane.
    fn fast_scan_groups(&self) -> usize {
        self.fast_scan_planes().saturating_mul(self.discrete / 4)
    }

    /// Bytes of one fast-scan block: 16 per group, then four corrective
    /// terms for each of its 32 codes.
    // ARITH: groups are bounded by the dimension.
    #[allow(clippy::arithmetic_side_effects)]
    fn fast_scan_block_bytes(&self) -> usize {
        self.fast_scan_groups() * 16 + FAST_SCAN_BLOCK * CORRECTIONS_BYTES
    }
}

/// A query quantized against one centroid at a time.
#[derive(Debug)]
struct OsqQueryScorer<'q> {
    q: &'q OsqQuantizer,
    query: Vec<f32>,
    unpacked: Vec<u8>,
    corrections: QuantizationResult,
    centroid_dp: f32,
    /// For the fast-scan layout, one 16-entry table per bit plane and group
    /// of four dimensions: the query's dot product with every 4-bit pattern
    /// of document bits, weighted by the plane.
    luts: Vec<u8>,
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
        self.centroid_dp = if self.q.similarity == Similarity::Euclidean {
            0.0
        } else {
            simd::dot_f32(centroid, centroid)
        };
        if self.q.fast_scan_planes() > 0 {
            fill_luts(&self.unpacked, &mut self.luts);
        }
    }

    fn score(&self, code: &[u8]) -> f32 {
        self.score_code(code, &self.terms())
    }

    fn score_block(&self, codes: &[u8], scores: &mut [f32]) {
        if self.q.fast_scan_planes() > 0 {
            self.fast_scan(codes, scores);
            return;
        }
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

    /// Scores a fast-scan block layout: per block of 32 codes, the integer
    /// dot products come from table lookups, then each code's corrective
    /// terms finish its score exactly as [`OsqQueryScorer::score_code`] does.
    // ARITH: block offsets are below `codes.len()`, checked against the
    // layout's length up front.
    #[allow(clippy::arithmetic_side_effects)]
    fn fast_scan(&self, codes: &[u8], scores: &mut [f32]) {
        assert_eq!(
            codes.len(),
            self.q.block_len(scores.len()),
            "codes must hold one code per score"
        );
        let terms = self.terms();
        let nibbles = self.luts.len();
        let block_bytes = self.q.fast_scan_block_bytes();
        for (block, out) in codes
            .chunks_exact(block_bytes)
            .zip(scores.chunks_mut(FAST_SCAN_BLOCK))
        {
            let (nibble_bytes, term_bytes) = block.split_at(nibbles);
            let mut dots = [0u32; FAST_SCAN_BLOCK];
            simd::lut16_accumulate32(&self.luts, nibble_bytes, &mut dots);
            // The four corrective terms, one array of 32 each.
            let mut t = [[0u32; FAST_SCAN_BLOCK]; 4];
            for (row, bytes) in t
                .iter_mut()
                .zip(term_bytes.chunks_exact(4 * FAST_SCAN_BLOCK))
            {
                for (x, b) in row.iter_mut().zip(bytes.chunks_exact(4)) {
                    *x = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
            // `finish`, one step at a time over all 32 codes so each step
            // vectorizes; the operations and their order are `finish`'s.
            let mut score = [0f32; FAST_SCAN_BLOCK];
            let mut corrections = [0f32; FAST_SCAN_BLOCK];
            for j in 0..FAST_SCAN_BLOCK {
                let ax = f32::from_bits(t[0][j]);
                let lx = (f32::from_bits(t[1][j]) - ax) * terms.scale;
                let x1 = t[3][j] as i32 as f32;
                score[j] = ax * terms.ay_dims
                    + terms.ay * lx * x1
                    + ax * terms.ly_y1
                    + lx * terms.ly * dots[j] as f32;
                corrections[j] = terms.additional + f32::from_bits(t[2][j]);
            }
            let k = out.len();
            match self.q.similarity {
                Similarity::Euclidean => {
                    for ((s, &x), &c) in out.iter_mut().zip(&score[..k]).zip(&corrections[..k]) {
                        *s = 1.0 / (1.0 + (c - 2.0 * x).max(0.0));
                    }
                }
                Similarity::MaximumInnerProduct => {
                    for ((s, &x), &c) in out.iter_mut().zip(&score[..k]).zip(&corrections[..k]) {
                        *s = quantization::scale_max_inner_product_score(x + c);
                    }
                }
                Similarity::DotProduct | Similarity::Cosine => {
                    for ((s, &x), &c) in out.iter_mut().zip(&score[..k]).zip(&corrections[..k]) {
                        *s = ((1.0 + x + c) / 2.0).max(0.0);
                    }
                }
            }
        }
    }

    /// `Lucene104ScalarQuantizedVectorScorer.quantizedScore`: the integer dot
    /// product of the codes, with both quantizations undone through their
    /// corrective terms.
    #[inline]
    fn score_code(&self, code: &[u8], t: &QueryTerms) -> f32 {
        let (doc, tail) = code.split_at(self.q.packed);
        let qc_dist = match self.q.encoding {
            ScalarEncoding::UnsignedByte => {
                vector_util::uint8_dot_product(&self.unpacked, doc) as f32
            }
            ScalarEncoding::SevenBit => vector_util::dot_product_i8(&self.unpacked, doc) as f32,
            ScalarEncoding::PackedNibble => {
                vector_util::int4_dot_product_single_packed(&self.unpacked, doc) as f32
            }
            ScalarEncoding::SingleBitQueryNibble | ScalarEncoding::DibitQueryNibble => {
                bit_planes_dot(&self.unpacked, doc) as f32
            }
        };
        self.finish(
            [f32_at(tail, 0), f32_at(tail, 4), f32_at(tail, 8)],
            i32_at(tail, 12),
            qc_dist,
            t,
        )
    }

    /// The corrected score from a document's corrective terms (lower and
    /// upper interval, additional correction; component sum) and the integer
    /// dot product of the codes.
    #[inline]
    fn finish(
        &self,
        [ax, upper, additional]: [f32; 3],
        x1: i32,
        qc_dist: f32,
        t: &QueryTerms,
    ) -> f32 {
        let lx = (upper - ax) * t.scale;
        let x1 = x1 as f32;
        let score = ax * t.ay_dims + t.ay * lx * x1 + ax * t.ly_y1 + lx * t.ly * qc_dist;
        let corrections = t.additional + additional;
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
                let mut block = Vec::new();
                q.write_block(&codes, &mut block);
                let codes = block;
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

    /// The fast-scan block layout scores every code exactly as scoring it
    /// alone does, for partial and whole blocks.
    #[test]
    fn fast_scan_blocks_score_like_single_codes() {
        for encoding in [
            ScalarEncoding::SingleBitQueryNibble,
            ScalarEncoding::DibitQueryNibble,
            ScalarEncoding::PackedNibble,
        ] {
            for sim in SIMS {
                for dim in [5, 100, 300] {
                    let q = OsqQuantizer::new(dim, sim, encoding);
                    let centroid = vectors(1, dim, 99).remove(0);
                    for n in [0, 1, 31, 32, 33, 70] {
                        let docs = vectors(n, dim, 7 + n as u64);
                        let mut codes = vec![0u8; n * q.code_len()];
                        for (d, c) in docs.iter().zip(codes.chunks_exact_mut(q.code_len())) {
                            q.encode(d, &centroid, c);
                        }
                        let mut block = Vec::new();
                        q.write_block(&codes, &mut block);
                        assert_eq!(block.len(), q.block_len(n));
                        let query = vectors(1, dim, 1234).remove(0);
                        let mut scorer = q.query_scorer(&query);
                        scorer.set_centroid(&centroid);
                        let mut scores = vec![0f32; n];
                        scorer.score_block(&block, &mut scores);
                        for (i, c) in codes.chunks_exact(q.code_len()).enumerate() {
                            assert_eq!(
                                scores[i].to_bits(),
                                scorer.score(c).to_bits(),
                                "{encoding:?} {sim:?} dim {dim} n {n} code {i}"
                            );
                        }
                    }
                }
            }
        }
    }
}
