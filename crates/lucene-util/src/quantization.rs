//! Port of `org.apache.lucene.util.quantization`: the scalar quantizers the
//! quantized vector formats are built on.
//!
//! | Java | Rust |
//! |---|---|
//! | `OptimizedScalarQuantizer` (per-vector, centroid-centred, anisotropic loss) | [`OptimizedScalarQuantizer`] |
//! | `QuantizedByteVectorValues.ScalarEncoding` | [`ScalarEncoding`] |
//! | `ScalarQuantizer` (legacy global min/max, `Lucene99ScalarQuantized*`) | [`ScalarQuantizer`] |
//! | `ScalarQuantizedVectorSimilarity` | [`ScalarQuantizedVectorSimilarity`] |
//! | `QuantizedByteVectorValues`, `BaseQuantizedByteVectorValues`, `LegacyQuantizedByteVectorValues`, `QuantizedVectorsReader` | the `Lucene104` reader's value types, `lucene_codecs::scalar_quantized_vectors` |
//!
//! Java's entry points take an `index.VectorSimilarityFunction`; this crate
//! sits below the one that defines it, so [`VectorSimilarityFunction`] here
//! is a mirror of its four constants (`lucene-codecs` converts), with the
//! float `compare` the legacy quantizer's neighbour search needs.
//!
//! # Floating point
//!
//! Every expression keeps Java's types and order: `float` arithmetic stays
//! `f32`, `double` stays `f64`, a `float` widened into a `double` expression
//! is widened at the same point, and `Math.round` is Java's
//! ([`java_round_f32`]/[`java_round_f64`]). That is what makes the quantized
//! bytes and corrective terms bit-identical to Lucene's
//! (`tests/scalar_quantized_fixtures.rs`).

use crate::simd;
use crate::vector_util::{self, java_round_f32, java_round_f64};

/// `index.VectorSimilarityFunction`'s four constants, in ordinal order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorSimilarityFunction {
    Euclidean,
    DotProduct,
    Cosine,
    MaximumInnerProduct,
}

/// `VectorUtil.scaleMaxInnerProductScore`.
pub fn scale_max_inner_product_score(dot: f32) -> f32 {
    if dot < 0.0 {
        1.0 / (1.0 + -dot)
    } else {
        dot + 1.0
    }
}

impl VectorSimilarityFunction {
    /// `VectorSimilarityFunction.compare(float[], float[])`, with the float
    /// kernels of [`crate::simd`] (the ones `lucene-codecs` scores with).
    pub fn score(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            VectorSimilarityFunction::Euclidean => 1.0 / (1.0 + simd::square_distance_f32(a, b)),
            VectorSimilarityFunction::DotProduct => {
                java_max_f32((1.0 + simd::dot_f32(a, b)) / 2.0, 0.0)
            }
            VectorSimilarityFunction::Cosine => {
                let (sum, n1, n2) = simd::cosine_parts_f32(a, b);
                let cos = if n1 == 0.0 || n2 == 0.0 {
                    0.0
                } else {
                    (sum as f64 / ((n1 as f64) * (n2 as f64)).sqrt()) as f32
                };
                java_max_f32((1.0 + cos) / 2.0, 0.0)
            }
            VectorSimilarityFunction::MaximumInnerProduct => {
                scale_max_inner_product_score(simd::dot_f32(a, b))
            }
        }
    }
}

/// `OptimizedScalarQuantizer.MINIMUM_MSE_GRID`: the initial interval, in
/// standard deviations, for each bit count 1..=8.
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
const DEFAULT_LAMBDA: f32 = 0.1;
const DEFAULT_ITERS: i32 = 5;

/// `OptimizedScalarQuantizer.QuantizationResult`: a vector's corrective terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizationResult {
    pub lower_interval: f32,
    pub upper_interval: f32,
    pub additional_correction: f32,
    pub quantized_component_sum: i32,
}

/// `QuantizedByteVectorValues.ScalarEncoding`: how many bits a document
/// (and a query) dimension is quantized to, and how they are packed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarEncoding {
    /// 8 bits per dimension, one per byte.
    UnsignedByte,
    /// 4 bits per dimension, two per byte (`packNibbles`).
    PackedNibble,
    /// 7 bits per dimension, one per byte (scored as signed bytes).
    SevenBit,
    /// 1-bit documents (`packAsBinary`), 4-bit transposed queries.
    SingleBitQueryNibble,
    /// 2-bit documents (`transposeDibit`), 4-bit transposed queries.
    DibitQueryNibble,
}

impl ScalarEncoding {
    const ALL: [ScalarEncoding; 5] = [
        ScalarEncoding::UnsignedByte,
        ScalarEncoding::PackedNibble,
        ScalarEncoding::SevenBit,
        ScalarEncoding::SingleBitQueryNibble,
        ScalarEncoding::DibitQueryNibble,
    ];

    /// `getWireNumber()`.
    pub fn wire_number(self) -> i32 {
        match self {
            ScalarEncoding::UnsignedByte => 0,
            ScalarEncoding::PackedNibble => 1,
            ScalarEncoding::SevenBit => 2,
            ScalarEncoding::SingleBitQueryNibble => 3,
            ScalarEncoding::DibitQueryNibble => 4,
        }
    }

    /// `fromWireNumber`.
    pub fn from_wire_number(wire: i32) -> Option<ScalarEncoding> {
        Self::ALL.into_iter().find(|e| e.wire_number() == wire)
    }

    /// `fromNumBits`: the first encoding whose document width is `bits`.
    pub fn from_num_bits(bits: u8) -> Option<ScalarEncoding> {
        Self::ALL.into_iter().find(|e| e.bits() == bits)
    }

    /// `getBits()`: document bits per dimension.
    pub fn bits(self) -> u8 {
        match self {
            ScalarEncoding::UnsignedByte => 8,
            ScalarEncoding::PackedNibble => 4,
            ScalarEncoding::SevenBit => 7,
            ScalarEncoding::SingleBitQueryNibble => 1,
            ScalarEncoding::DibitQueryNibble => 2,
        }
    }

    /// `getQueryBits()`.
    pub fn query_bits(self) -> u8 {
        match self {
            ScalarEncoding::SingleBitQueryNibble | ScalarEncoding::DibitQueryNibble => 4,
            other => other.bits(),
        }
    }

    /// `getDocBitsPerDim()`: storage bits per document dimension.
    pub fn doc_bits_per_dim(self) -> usize {
        match self {
            ScalarEncoding::UnsignedByte | ScalarEncoding::SevenBit => 8,
            ScalarEncoding::PackedNibble => 4,
            ScalarEncoding::SingleBitQueryNibble => 1,
            ScalarEncoding::DibitQueryNibble => 2,
        }
    }

    /// `getQueryBitsPerDim()`.
    pub fn query_bits_per_dim(self) -> usize {
        match self {
            ScalarEncoding::SingleBitQueryNibble | ScalarEncoding::DibitQueryNibble => 4,
            other => other.doc_bits_per_dim(),
        }
    }

    /// `isAsymmetric()`: queries and documents use different widths.
    pub fn is_asymmetric(self) -> bool {
        self.bits() != self.query_bits()
    }

    /// `getDiscreteDimensions(dimensions)`: the dimension count rounded up so
    /// both packed forms fill whole bytes.
    pub fn discrete_dimensions(self, dimensions: usize) -> usize {
        if self == ScalarEncoding::DibitQueryNibble {
            let query = (dimensions * 4).div_ceil(8) * 8 / 4;
            let doc = dimensions.div_ceil(8) * 8;
            return query.max(doc);
        }
        let bits = self.doc_bits_per_dim();
        let qbits = self.query_bits_per_dim();
        if bits == qbits {
            return (dimensions * bits).div_ceil(8) * 8 / bits;
        }
        let query = (dimensions * qbits).div_ceil(8) * 8 / qbits;
        let doc = (dimensions * bits).div_ceil(8) * 8 / bits;
        query.max(doc)
    }

    /// `getDocPackedLength(dimensions)`: bytes of one stored document vector.
    pub fn doc_packed_length(self, dimensions: usize) -> usize {
        let discretized = self.discrete_dimensions(dimensions);
        if self == ScalarEncoding::DibitQueryNibble {
            return 2 * discretized.div_ceil(8);
        }
        (discretized * self.doc_bits_per_dim()).div_ceil(8)
    }

    /// `getQueryPackedLength(dimensions)`: bytes of one packed query vector.
    pub fn query_packed_length(self, dimensions: usize) -> usize {
        let discretized = self.discrete_dimensions(dimensions);
        (discretized * self.query_bits_per_dim()).div_ceil(8)
    }
}

/// `Math.min(Math.max(x, a), b)` in `double`, Java's `clamp` helper.
#[inline]
fn clamp(x: f64, a: f64, b: f64) -> f64 {
    java_min_f64(java_max_f64(x, a), b)
}

/// `Math.max(double, double)`: NaN wins, and `max(-0.0, 0.0)` is `0.0`. On
/// equal operands the sign bits are ANDed, which picks `+0.0` for a
/// mixed-sign zero pair and is the identity otherwise -- one compare pair and
/// no data-dependent branch on the common path.
#[inline]
fn java_max_f64(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else if b > a {
        b
    } else if a == b {
        f64::from_bits(a.to_bits() & b.to_bits())
    } else {
        f64::NAN
    }
}

/// `Math.min(double, double)`: NaN wins, and `min(-0.0, 0.0)` is `-0.0`
/// (the sign bits ORed on a tie).
#[inline]
fn java_min_f64(a: f64, b: f64) -> f64 {
    if a < b {
        a
    } else if b < a {
        b
    } else if a == b {
        f64::from_bits(a.to_bits() | b.to_bits())
    } else {
        f64::NAN
    }
}

/// `Math.max(float, float)`.
#[inline]
fn java_max_f32(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else if b > a {
        b
    } else if a == b {
        f32::from_bits(a.to_bits() & b.to_bits())
    } else {
        f32::NAN
    }
}

/// `Math.min(float, float)`.
#[inline]
fn java_min_f32(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else if b < a {
        b
    } else if a == b {
        f32::from_bits(a.to_bits() | b.to_bits())
    } else {
        f32::NAN
    }
}

/// `OptimizedScalarQuantizer`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizedScalarQuantizer {
    similarity: VectorSimilarityFunction,
    lambda: f32,
    iters: i32,
}

impl OptimizedScalarQuantizer {
    /// `new OptimizedScalarQuantizer(similarityFunction)`.
    pub fn new(similarity: VectorSimilarityFunction) -> Self {
        Self::with_params(similarity, DEFAULT_LAMBDA, DEFAULT_ITERS)
    }

    /// `new OptimizedScalarQuantizer(similarityFunction, lambda, iters)`.
    pub fn with_params(similarity: VectorSimilarityFunction, lambda: f32, iters: i32) -> Self {
        OptimizedScalarQuantizer {
            similarity,
            lambda,
            iters,
        }
    }

    /// The first loop of `scalarQuantize`/`multiScalarQuantize`: centre
    /// `vector` on `centroid` in place and gather its statistics.
    fn center(&self, vector: &mut [f32], centroid: &[f32]) -> Stats {
        let mut vec_mean = 0f64;
        let mut vec_var = 0f64;
        let mut norm2 = 0f32;
        let mut centroid_dot = 0f32;
        let mut min = f32::MAX;
        let mut max = -f32::MAX;
        for (i, (v, &c)) in vector.iter_mut().zip(centroid).enumerate() {
            if self.similarity != VectorSimilarityFunction::Euclidean {
                centroid_dot += *v * c;
            }
            *v -= c;
            min = java_min_f32(min, *v);
            max = java_max_f32(max, *v);
            norm2 += *v * *v;
            let delta = *v as f64 - vec_mean;
            vec_mean += delta / (i + 1) as f64;
            vec_var += delta * (*v as f64 - vec_mean);
        }
        vec_var /= vector.len() as f64;
        Stats {
            vec_mean,
            vec_std: vec_var.sqrt(),
            norm2,
            centroid_dot,
            min,
            max,
        }
    }

    /// One bit count's half of `scalarQuantize`: optimize the interval and
    /// write the codes.
    fn quantize_with(
        &self,
        vector: &[f32],
        stats: &Stats,
        destination: &mut [u8],
        bits: u8,
    ) -> QuantizationResult {
        assert!((1..=8).contains(&bits), "bits must be in 1..=8, got {bits}");
        let points = 1i32 << bits;
        let grid = MINIMUM_MSE_GRID[bits as usize - 1];
        let mut interval = [
            clamp(
                grid[0] as f64 * stats.vec_std + stats.vec_mean,
                stats.min as f64,
                stats.max as f64,
            ) as f32,
            clamp(
                grid[1] as f64 * stats.vec_std + stats.vec_mean,
                stats.min as f64,
                stats.max as f64,
            ) as f32,
        ];
        self.optimize_intervals(&mut interval, vector, stats.norm2, points);
        let n_steps = ((1i32 << bits) - 1) as f32;
        let (a, b) = (interval[0], interval[1]);
        let step = (b - a) / n_steps;
        let mut sum_query = 0i32;
        for (h, &v) in vector.iter().enumerate() {
            let xi = clamp(v as f64, a as f64, b as f64) as f32;
            let assignment = java_round_f32((xi - a) / step);
            sum_query = sum_query.wrapping_add(assignment);
            destination[h] = assignment as u8;
        }
        QuantizationResult {
            lower_interval: interval[0],
            upper_interval: interval[1],
            additional_correction: if self.similarity == VectorSimilarityFunction::Euclidean {
                stats.norm2
            } else {
                stats.centroid_dot
            },
            quantized_component_sum: sum_query,
        }
    }

    /// `scalarQuantize(vector, destination, bits, centroid)`. **Centres
    /// `vector` in place**, as Java does. `destination` must hold at least
    /// `vector.len()` bytes; the rest is left as is.
    pub fn scalar_quantize(
        &self,
        vector: &mut [f32],
        destination: &mut [u8],
        bits: u8,
        centroid: &[f32],
    ) -> QuantizationResult {
        assert!(vector.len() <= destination.len());
        let stats = self.center(vector, centroid);
        self.quantize_with(vector, &stats, destination, bits)
    }

    /// `multiScalarQuantize(vector, destinations, bits, centroid)`: one
    /// centring, one quantization per bit count.
    pub fn multi_scalar_quantize(
        &self,
        vector: &mut [f32],
        destinations: &mut [Vec<u8>],
        bits: &[u8],
        centroid: &[f32],
    ) -> Vec<QuantizationResult> {
        assert_eq!(bits.len(), destinations.len());
        let stats = self.center(vector, centroid);
        bits.iter()
            .zip(destinations.iter_mut())
            .map(|(&b, dest)| self.quantize_with(vector, &stats, dest, b))
            .collect()
    }

    /// `loss`.
    fn loss(&self, vector: &[f32], interval: [f32; 2], points: i32, norm2: f32) -> f64 {
        let a = interval[0] as f64;
        let b = interval[1] as f64;
        let step = (b - a) / (points as f32 - 1.0f32) as f64;
        let step_inv = 1.0 / step;
        let mut xe = 0.0f64;
        let mut e = 0.0f64;
        for &xi in vector {
            let xi = xi as f64;
            let xiq = a + step * java_round_f64((clamp(xi, a, b) - a) * step_inv) as f64;
            xe += xi * (xi - xiq);
            e += (xi - xiq) * (xi - xiq);
        }
        (1.0 - self.lambda as f64) * xe * xe / norm2 as f64 + self.lambda as f64 * e
    }

    /// `optimizeIntervals`: a few rounds of least-squares refinement of the
    /// interval, kept only while the loss improves.
    fn optimize_intervals(&self, init: &mut [f32; 2], vector: &[f32], norm2: f32, points: i32) {
        let mut initial_loss = self.loss(vector, *init, points, norm2);
        let scale = (1.0f32 - self.lambda) / norm2;
        if !scale.is_finite() {
            return;
        }
        let lambda = self.lambda as f64;
        for _ in 0..self.iters {
            let a = init[0];
            let b = init[1];
            let step_inv = (points as f32 - 1.0f32) / (b - a);
            let (mut daa, mut dab, mut dbb, mut dax, mut dbx) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for &xi in vector {
                let k = java_round_f64(
                    (clamp(xi as f64, a as f64, b as f64) - a as f64) * step_inv as f64,
                ) as f32;
                let s = k / (points - 1) as f32;
                let s64 = s as f64;
                daa += (1.0 - s64) * (1.0 - s64);
                dab += (1.0 - s64) * s64;
                dbb += (s * s) as f64;
                dax += xi as f64 * (1.0 - s64);
                dbx += (xi * s) as f64;
            }
            let scale64 = scale as f64;
            let m0 = scale64 * dax * dax + lambda * daa;
            let m1 = scale64 * dax * dbx + lambda * dab;
            let m2 = scale64 * dbx * dbx + lambda * dbb;
            let det = m0 * m2 - m1 * m1;
            if det == 0.0 {
                return;
            }
            let a_opt = ((m2 * dax - m1 * dbx) / det) as f32;
            let b_opt = ((m0 * dbx - m1 * dax) / det) as f32;
            if ((init[0] - a_opt).abs() as f64) < 1e-8 && ((init[1] - b_opt).abs() as f64) < 1e-8 {
                return;
            }
            let new_loss = self.loss(vector, [a_opt, b_opt], points, norm2);
            if new_loss > initial_loss {
                return;
            }
            init[0] = a_opt;
            init[1] = b_opt;
            initial_loss = new_loss;
        }
    }
}

/// The centring loop's statistics.
struct Stats {
    vec_mean: f64,
    vec_std: f64,
    norm2: f32,
    centroid_dot: f32,
    min: f32,
    max: f32,
}

/// `OptimizedScalarQuantizer.deQuantize`: back to floats, re-adding the
/// centroid.
pub fn dequantize(
    quantized: &[u8],
    dequantized: &mut [f32],
    bits: u8,
    lower_interval: f32,
    upper_interval: f32,
    centroid: &[f32],
) {
    let n_steps = (1i32 << bits) - 1;
    // Java: `double step = (upperInterval - lowerInterval) / nSteps;` -- a
    // `float` division widened afterwards.
    let step = ((upper_interval - lower_interval) / n_steps as f32) as f64;
    for (h, &q) in quantized.iter().enumerate() {
        let xi = q as f64 * step + lower_interval as f64;
        dequantized[h] = (xi + centroid[h] as f64) as f32;
    }
}

/// `OptimizedScalarQuantizer.discretize`: `value` rounded up to a multiple of
/// `bucket`.
pub fn discretize(value: usize, bucket: usize) -> usize {
    value.div_ceil(bucket) * bucket
}

/// `OptimizedScalarQuantizer.transposeHalfByte`: split 4-bit values into
/// four bit planes, each `q.len() / 8` bytes (bit 7 of a plane byte is the
/// first of its eight values).
pub fn transpose_half_byte(q: &[u8], out: &mut [u8]) {
    let quarter = out.len() / 4;
    let mut i = 0;
    while i < q.len() {
        let (mut lower, mut lower_middle, mut upper_middle, mut upper) = (0u8, 0u8, 0u8, 0u8);
        let mut j = 7i32;
        while j >= 0 && i < q.len() {
            debug_assert!(q[i] <= 15);
            lower |= (q[i] & 1) << j;
            lower_middle |= ((q[i] >> 1) & 1) << j;
            upper_middle |= ((q[i] >> 2) & 1) << j;
            upper |= ((q[i] >> 3) & 1) << j;
            i += 1;
            j -= 1;
        }
        let index = i.div_ceil(8) - 1;
        out[index] = lower;
        out[index + quarter] = lower_middle;
        out[index + out.len() / 2] = upper_middle;
        out[index + 3 * out.len() / 4] = upper;
    }
}

/// `OptimizedScalarQuantizer.packAsBinary`: eight 0/1 values per byte,
/// first value in bit 7.
pub fn pack_as_binary(vector: &[u8], packed: &mut [u8]) {
    for (chunk, p) in vector.chunks(8).zip(packed.iter_mut()) {
        let mut result = 0u8;
        for (j, &v) in chunk.iter().enumerate() {
            debug_assert!(v <= 1);
            result |= (v & 1) << (7 - j);
        }
        *p = result;
    }
}

/// `OptimizedScalarQuantizer.unpackBinary`.
pub fn unpack_binary(packed: &[u8], vector: &mut [u8]) {
    for (chunk, &p) in vector.chunks_mut(8).zip(packed) {
        for (j, v) in chunk.iter_mut().enumerate() {
            *v = (p >> (7 - j)) & 1;
        }
    }
}

/// `OptimizedScalarQuantizer.transposeDibit`: 2-bit values as two bit
/// stripes, low bits first, each `packed.len() / 2` bytes.
pub fn transpose_dibit(vector: &[u8], packed: &mut [u8]) {
    let half = packed.len() / 2;
    for (index, chunk) in vector.chunks(8).enumerate() {
        let (mut lower, mut upper) = (0u8, 0u8);
        for (j, &v) in chunk.iter().enumerate() {
            debug_assert!(v <= 3);
            lower |= (v & 1) << (7 - j);
            upper |= ((v >> 1) & 1) << (7 - j);
        }
        packed[index] = lower;
        packed[index + half] = upper;
    }
}

/// `OptimizedScalarQuantizer.untransposeDibit`.
pub fn untranspose_dibit(packed: &[u8], vector: &mut [u8]) {
    let stripe = packed.len() / 2;
    for (index, chunk) in vector.chunks_mut(8).enumerate() {
        let (lower, upper) = (packed[index], packed[index + stripe]);
        for (j, v) in chunk.iter_mut().enumerate() {
            let s = 7 - j;
            *v = ((lower >> s) & 1) | (((upper >> s) & 1) << 1);
        }
    }
}

/// `OffHeapScalarQuantizedVectorValues.packNibbles`: value `i` into the high
/// nibble of byte `i`, value `i + n` into its low nibble.
pub fn pack_nibbles(unpacked: &[u8], packed: &mut [u8]) {
    let n = packed.len();
    debug_assert_eq!(unpacked.len(), n * 2);
    for i in 0..n {
        packed[i] = (unpacked[i] << 4) | unpacked[n + i];
    }
}

/// `OffHeapScalarQuantizedVectorValues.unpackNibbles`.
pub fn unpack_nibbles(packed: &[u8], unpacked: &mut [u8]) {
    let n = packed.len();
    for i in 0..n {
        unpacked[i] = (packed[i] >> 4) & 0x0F;
        unpacked[n + i] = packed[i] & 0x0F;
    }
}

// ---------------------------------------------------------------------------
// ScalarQuantizedVectorSimilarity
// ---------------------------------------------------------------------------

/// `ScalarQuantizedVectorSimilarity`: scores two legacy-quantized vectors
/// with their corrective offsets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarQuantizedVectorSimilarity {
    Euclidean { const_multiplier: f32 },
    DotProduct { const_multiplier: f32, int4: bool },
    MaximumInnerProduct { const_multiplier: f32, int4: bool },
}

impl ScalarQuantizedVectorSimilarity {
    /// `fromVectorSimilarity(sim, constMultiplier, bits)`.
    pub fn from_vector_similarity(
        sim: VectorSimilarityFunction,
        const_multiplier: f32,
        bits: u8,
    ) -> Self {
        let int4 = bits <= 4;
        match sim {
            VectorSimilarityFunction::Euclidean => Self::Euclidean { const_multiplier },
            VectorSimilarityFunction::Cosine | VectorSimilarityFunction::DotProduct => {
                Self::DotProduct {
                    const_multiplier,
                    int4,
                }
            }
            VectorSimilarityFunction::MaximumInnerProduct => Self::MaximumInnerProduct {
                const_multiplier,
                int4,
            },
        }
    }

    /// `score(queryVector, queryVectorOffset, storedVector, vectorOffset)`.
    pub fn score(&self, query: &[u8], query_offset: f32, stored: &[u8], vector_offset: f32) -> f32 {
        let dot = |int4: bool| {
            if int4 {
                vector_util::int4_dot_product(stored, query)
            } else {
                vector_util::uint8_dot_product(stored, query)
            }
        };
        match *self {
            Self::Euclidean { const_multiplier } => {
                let square_distance = vector_util::uint8_square_distance(stored, query);
                let adjusted = square_distance as f32 * const_multiplier;
                1.0 / (1.0 + adjusted)
            }
            Self::DotProduct {
                const_multiplier,
                int4,
            } => {
                let adjusted = dot(int4) as f32 * const_multiplier + query_offset + vector_offset;
                java_max_f32((1.0 + adjusted) / 2.0, 0.0)
            }
            Self::MaximumInnerProduct {
                const_multiplier,
                int4,
            } => {
                let adjusted = dot(int4) as f32 * const_multiplier + query_offset + vector_offset;
                scale_max_inner_product_score(adjusted)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ScalarQuantizer (legacy, global quantiles)
// ---------------------------------------------------------------------------

/// `ScalarQuantizer.SCALAR_QUANTIZATION_SAMPLE_SIZE`.
pub const SCALAR_QUANTIZATION_SAMPLE_SIZE: usize = 25_000;
/// `ScalarQuantizer.SCRATCH_SIZE`.
const SCRATCH_SIZE: usize = 20;

/// Java's `private static final Random random = new Random(42)`: one stream
/// shared by every reservoir sample in the process, as in the JVM.
static RESERVOIR_RANDOM: std::sync::Mutex<Option<crate::java_random::JavaRandom>> =
    std::sync::Mutex::new(None);

/// Errors of the legacy quantizer (Java's `IllegalStateException`s).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QuantizerError {
    #[error("Scalar quantizer does not support infinite or NaN values")]
    NonFinite,
    #[error("Quantile calculation resulted in NaN or infinite values")]
    NonFiniteQuantiles,
}

/// `ScalarQuantizer`: the legacy global min/max quantizer of
/// `Lucene99ScalarQuantizedVectorsFormat`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarQuantizer {
    alpha: f32,
    scale: f32,
    bits: u8,
    min_quantile: f32,
    max_quantile: f32,
}

impl ScalarQuantizer {
    /// `new ScalarQuantizer(minQuantile, maxQuantile, bits)`.
    pub fn new(min_quantile: f32, max_quantile: f32, bits: u8) -> Result<Self, QuantizerError> {
        if !min_quantile.is_finite() || !max_quantile.is_finite() {
            return Err(QuantizerError::NonFinite);
        }
        debug_assert!(max_quantile >= min_quantile);
        debug_assert!((1..=8).contains(&bits));
        let divisor = ((1i32 << bits) - 1) as f32;
        Ok(ScalarQuantizer {
            scale: divisor / (max_quantile - min_quantile),
            alpha: (max_quantile - min_quantile) / divisor,
            bits,
            min_quantile,
            max_quantile,
        })
    }

    /// `quantize(src, dest, similarityFunction)`: the corrective offset (0
    /// for Euclidean).
    pub fn quantize(&self, src: &[f32], dest: &mut [u8], sim: VectorSimilarityFunction) -> f32 {
        let correction = vector_util::min_max_scalar_quantize(
            src,
            dest,
            self.scale,
            self.alpha,
            self.min_quantile,
            self.max_quantile,
        );
        if sim == VectorSimilarityFunction::Euclidean {
            0.0
        } else {
            correction
        }
    }

    /// `recalculateCorrectiveOffset`: a vector quantized by `old` re-offset
    /// for this quantizer.
    pub fn recalculate_corrective_offset(
        &self,
        quantized: &[u8],
        old: &ScalarQuantizer,
        sim: VectorSimilarityFunction,
    ) -> f32 {
        if sim == VectorSimilarityFunction::Euclidean {
            return 0.0;
        }
        vector_util::recalculate_offset(
            quantized,
            old.alpha,
            old.min_quantile,
            self.scale,
            self.alpha,
            self.min_quantile,
            self.max_quantile,
        )
    }

    /// `deQuantize(src, dest)`.
    pub fn dequantize(&self, src: &[u8], dest: &mut [f32]) {
        for (d, &s) in dest.iter_mut().zip(src) {
            *d = (self.alpha * s as f32) + self.min_quantile;
        }
    }

    /// `getLowerQuantile()`.
    pub fn lower_quantile(&self) -> f32 {
        self.min_quantile
    }

    /// `getUpperQuantile()`.
    pub fn upper_quantile(&self) -> f32 {
        self.max_quantile
    }

    /// `getConstantMultiplier()`: `alpha^2`.
    pub fn constant_multiplier(&self) -> f32 {
        self.alpha * self.alpha
    }

    /// `getBits()`.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// `reservoirSampleIndices`, drawing from the process-wide stream.
    fn reservoir_sample_indices(num_float_vecs: usize, sample_size: usize) -> Vec<usize> {
        let mut guard = RESERVOIR_RANDOM.lock().unwrap_or_else(|p| p.into_inner());
        let random = guard.get_or_insert_with(|| crate::java_random::JavaRandom::new(42));
        let mut take: Vec<usize> = (0..sample_size).collect();
        for i in sample_size..num_float_vecs {
            let j = random.next_int_bounded((i + 1) as i32) as usize;
            if j < sample_size {
                take[j] = i;
            }
        }
        take.sort_unstable();
        take
    }

    /// `fromVectors(floatVectorValues, confidenceInterval, totalVectorCount, bits)`
    /// over the vectors in iteration order.
    pub fn from_vectors<V: AsRef<[f32]>>(
        vectors: &[V],
        confidence_interval: f32,
        total_vector_count: usize,
        bits: u8,
    ) -> Result<Self, QuantizerError> {
        Self::from_vectors_with_sample_size(
            vectors,
            confidence_interval,
            total_vector_count,
            bits,
            SCALAR_QUANTIZATION_SAMPLE_SIZE,
        )
    }

    /// `fromVectors` with Java's package-private sample-size parameter.
    pub fn from_vectors_with_sample_size<V: AsRef<[f32]>>(
        vectors: &[V],
        confidence_interval: f32,
        total_vector_count: usize,
        bits: u8,
        sample_size: usize,
    ) -> Result<Self, QuantizerError> {
        debug_assert!((0.9..=1.0).contains(&confidence_interval));
        debug_assert!(sample_size > SCRATCH_SIZE);
        if total_vector_count == 0 {
            return ScalarQuantizer::new(0.0, 0.0, bits);
        }
        if confidence_interval == 1.0 {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for v in vectors {
                for &x in v.as_ref() {
                    min = java_min_f32(min, x);
                    max = java_max_f32(max, x);
                }
            }
            return ScalarQuantizer::new(min, max, bits);
        }
        let dim = vectors.first().map_or(0, |v| v.as_ref().len());
        let mut scratch = vec![0f32; dim * SCRATCH_SIZE.min(total_vector_count)];
        let mut count = 0usize;
        let mut upper_sum = [0f64; 1];
        let mut lower_sum = [0f64; 1];
        let cis = [confidence_interval];
        if total_vector_count <= sample_size {
            let scratch_size = SCRATCH_SIZE.min(total_vector_count);
            let mut i = 0;
            for v in vectors {
                scratch[i * dim..(i + 1) * dim].copy_from_slice(v.as_ref());
                i += 1;
                if i == scratch_size {
                    extract_quantiles(&cis, &mut scratch, &mut upper_sum, &mut lower_sum);
                    i = 0;
                    count += 1;
                }
            }
            return ScalarQuantizer::new(
                lower_sum[0] as f32 / count as f32,
                upper_sum[0] as f32 / count as f32,
                bits,
            );
        }
        let take = Self::reservoir_sample_indices(total_vector_count, sample_size);
        let mut idx = 0;
        for &i in &take {
            scratch[idx * dim..(idx + 1) * dim].copy_from_slice(vectors[i].as_ref());
            idx += 1;
            if idx == SCRATCH_SIZE {
                extract_quantiles(&cis, &mut scratch, &mut upper_sum, &mut lower_sum);
                count += 1;
                idx = 0;
            }
        }
        ScalarQuantizer::new(
            lower_sum[0] as f32 / count as f32,
            upper_sum[0] as f32 / count as f32,
            bits,
        )
    }

    /// `fromVectorsAutoInterval(floatVectorValues, function, totalVectorCount, bits)`:
    /// grid-search the quantiles that best preserve nearest-neighbour score
    /// order on a sample.
    pub fn from_vectors_auto_interval<V: AsRef<[f32]>>(
        vectors: &[V],
        function: VectorSimilarityFunction,
        total_vector_count: usize,
        bits: u8,
    ) -> Result<Self, QuantizerError> {
        debug_assert!(function != VectorSimilarityFunction::Cosine);
        if total_vector_count == 0 {
            return ScalarQuantizer::new(0.0, 0.0, bits);
        }
        let dim = vectors.first().map_or(0, |v| v.as_ref().len());
        let sample_size = total_vector_count.min(1000);
        let mut scratch = vec![0f32; dim * SCRATCH_SIZE.min(total_vector_count)];
        let mut count = 0usize;
        let mut upper_sum = [0f64; 2];
        let mut lower_sum = [0f64; 2];
        let mut sampled: Vec<Vec<f32>> = Vec::with_capacity(sample_size);
        let dim_f = dim as f32;
        let cis = [
            1.0 - java_min_f32(32.0, dim_f / 10.0) / (dim_f + 1.0),
            1.0 - 1.0 / (dim_f + 1.0),
        ];
        let mut gather = |v: &[f32], scratch: &mut [f32], i: usize| {
            sampled.push(v.to_vec());
            scratch[i * dim..(i + 1) * dim].copy_from_slice(v);
        };
        if total_vector_count <= sample_size {
            let scratch_size = SCRATCH_SIZE.min(total_vector_count);
            let mut i = 0;
            for v in vectors {
                gather(v.as_ref(), &mut scratch, i);
                i += 1;
                if i == scratch_size {
                    extract_quantiles(&cis, &mut scratch, &mut upper_sum, &mut lower_sum);
                    i = 0;
                    count += 1;
                }
            }
        } else {
            let take = Self::reservoir_sample_indices(total_vector_count, 1000);
            let mut idx = 0;
            for &i in &take {
                gather(vectors[i].as_ref(), &mut scratch, idx);
                idx += 1;
                if idx == SCRATCH_SIZE {
                    extract_quantiles(&cis, &mut scratch, &mut upper_sum, &mut lower_sum);
                    count += 1;
                    idx = 0;
                }
            }
        }
        let al = lower_sum[1] as f32 / count as f32;
        let bu = upper_sum[1] as f32 / count as f32;
        let au = lower_sum[0] as f32 / count as f32;
        let bl = upper_sum[0] as f32 / count as f32;
        if ![al, au, bl, bu].iter().all(|x| x.is_finite()) {
            return Err(QuantizerError::NonFiniteQuantiles);
        }
        let mut lower_candidates = [0f32; 16];
        let mut upper_candidates = [0f32; 16];
        let mut i = 0f32;
        let mut idx = 0;
        while i < 32.0 {
            lower_candidates[idx] = al + i * (au - al) / 32.0;
            upper_candidates[idx] = bl + i * (bu - bl) / 32.0;
            idx += 1;
            i += 2.0;
        }
        let neighbors = find_nearest_neighbors(&sampled, function);
        let (lower, upper) = candidate_grid_search(
            &neighbors,
            &sampled,
            &lower_candidates,
            &upper_candidates,
            function,
            bits,
        )?;
        ScalarQuantizer::new(lower, upper, bits)
    }
}

impl std::fmt::Display for ScalarQuantizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ScalarQuantizer{{minQuantile={}, maxQuantile={}, bits={}}}",
            self.min_quantile, self.max_quantile, self.bits
        )
    }
}

/// `ScalarQuantizer.extractQuantiles`.
fn extract_quantiles(
    cis: &[f32],
    scratch: &mut [f32],
    upper_sum: &mut [f64],
    lower_sum: &mut [f64],
) {
    for (i, &ci) in cis.iter().enumerate() {
        let (lo, hi) = upper_and_lower_quantile(scratch, ci);
        upper_sum[i] += hi as f64;
        lower_sum[i] += lo as f64;
    }
}

/// `ScalarQuantizer.getUpperAndLowerQuantile`: the min and max of `arr`
/// after dropping `selectorIndex` values at each end. Java selects with an
/// `IntroSelector`; any correct selection leaves the same *set* in the kept
/// range, and only its min and max are read, so the result does not depend
/// on the selection algorithm.
pub fn upper_and_lower_quantile(arr: &mut [f32], confidence_interval: f32) -> (f32, f32) {
    assert!(!arr.is_empty());
    let cmp = |a: &f32, b: &f32| java_float_compare(*a, *b);
    if arr.len() <= 2 {
        arr.sort_by(cmp);
        return (arr[0], arr[arr.len() - 1]);
    }
    let n = arr.len();
    let selector_index = (n as f32 * (1.0 - confidence_interval) / 2.0 + 0.5) as usize;
    if selector_index > 0 {
        arr.select_nth_unstable_by(n - selector_index, cmp);
        arr[..n - selector_index].select_nth_unstable_by(selector_index, cmp);
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in &arr[selector_index..n - selector_index] {
        min = java_min_f32(v, min);
        max = java_max_f32(v, max);
    }
    (min, max)
}

/// `Float.compare`: `-0.0 < 0.0`, every NaN equal and above everything.
fn java_float_compare(a: f32, b: f32) -> std::cmp::Ordering {
    let canon = |x: f32| if x.is_nan() { f32::NAN } else { x };
    canon(a).total_cmp(&canon(b))
}

/// One sampled vector's ten nearest neighbours and their score variance.
struct ScoreDocsAndScoreVariance {
    /// `(doc, score)`, best first.
    score_docs: Vec<(usize, f32)>,
    score_variance: f32,
}

/// `ScalarQuantizer.OnlineMeanAndVar`.
#[derive(Default)]
struct OnlineMeanAndVar {
    mean: f64,
    var: f64,
    n: i32,
}

impl OnlineMeanAndVar {
    fn reset(&mut self) {
        *self = Self::default();
    }
    fn add(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.var += delta * (x - self.mean);
    }
    fn var(&self) -> f32 {
        (self.var / (self.n - 1) as f64) as f32
    }
}

/// `HitQueue(10, false)` ordering: a lower score is "less"; on a tie the
/// greater doc is "less".
fn hit_less(a: (usize, f32), b: (usize, f32)) -> bool {
    if a.1 == b.1 {
        a.0 > b.0
    } else {
        a.1 < b.1
    }
}

/// A 10-entry `HitQueue`'s `insertWithOverflow`, kept as a sorted list
/// (least first) -- the retained set and the pop order are what the
/// ordering defines, whatever the heap's layout.
fn insert_top10(q: &mut Vec<(usize, f32)>, e: (usize, f32)) {
    if q.len() < 10 {
        let pos = q.iter().position(|&x| hit_less(e, x)).unwrap_or(q.len());
        q.insert(pos, e);
    } else if hit_less(q[0], e) {
        q.remove(0);
        let pos = q.iter().position(|&x| hit_less(e, x)).unwrap_or(q.len());
        q.insert(pos, e);
    }
}

/// `ScalarQuantizer.findNearestNeighbors`.
fn find_nearest_neighbors(
    vectors: &[Vec<f32>],
    sim: VectorSimilarityFunction,
) -> Vec<ScoreDocsAndScoreVariance> {
    let mut queues: Vec<Vec<(usize, f32)>> = vec![Vec::new(); vectors.len().max(1)];
    for i in 0..vectors.len() {
        for j in i + 1..vectors.len() {
            let score = sim.score(&vectors[i], &vectors[j]);
            insert_top10(&mut queues[i], (j, score));
            insert_top10(&mut queues[j], (i, score));
        }
    }
    let mut mv = OnlineMeanAndVar::default();
    let mut result = Vec::with_capacity(vectors.len());
    for q in queues.iter().take(vectors.len()) {
        // Java pops least first into scoreDocs[size-1..=0] and adds each
        // popped score to the running variance in pop order.
        for &(_, s) in q {
            mv.add(s as f64);
        }
        let score_docs: Vec<(usize, f32)> = q.iter().rev().copied().collect();
        result.push(ScoreDocsAndScoreVariance {
            score_docs,
            score_variance: mv.var(),
        });
        mv.reset();
    }
    result
}

/// `ScalarQuantizer.candidateGridSearch`.
fn candidate_grid_search(
    neighbors: &[ScoreDocsAndScoreVariance],
    vectors: &[Vec<f32>],
    lower_candidates: &[f32; 16],
    upper_candidates: &[f32; 16],
    function: VectorSimilarityFunction,
    bits: u8,
) -> Result<(f32, f32), QuantizerError> {
    let mut max_corr = f64::NEG_INFINITY;
    let (mut best_lower, mut best_upper) = (0f32, 0f32);
    let mut correlator = ScoreErrorCorrelator::new(function, neighbors, vectors, bits);
    let (mut best_q_lower, mut best_q_upper) = (0usize, 0usize);
    for i in (0..16).step_by(4) {
        let lower = lower_candidates[i];
        if !lower.is_finite() {
            continue;
        }
        for j in (0..16).step_by(4) {
            let upper = upper_candidates[j];
            if !upper.is_finite() || upper <= lower {
                continue;
            }
            let mean = correlator.score_error_correlation(lower, upper)?;
            if mean > max_corr {
                max_corr = mean;
                best_lower = lower;
                best_upper = upper;
                best_q_lower = i;
                best_q_upper = j;
            }
        }
    }
    for &lower in &lower_candidates[best_q_lower + 1..best_q_lower + 4] {
        for &upper in &upper_candidates[best_q_upper + 1..best_q_upper + 4] {
            if !lower.is_finite() || !upper.is_finite() || upper <= lower {
                continue;
            }
            let mean = correlator.score_error_correlation(lower, upper)?;
            if mean > max_corr {
                max_corr = mean;
                best_lower = lower;
                best_upper = upper;
            }
        }
    }
    Ok((best_lower, best_upper))
}

/// `ScalarQuantizer.ScoreErrorCorrelator`.
struct ScoreErrorCorrelator<'a> {
    corr: OnlineMeanAndVar,
    errors: OnlineMeanAndVar,
    function: VectorSimilarityFunction,
    neighbors: &'a [ScoreDocsAndScoreVariance],
    vectors: &'a [Vec<f32>],
    query: Vec<u8>,
    vector: Vec<u8>,
    bits: u8,
}

impl<'a> ScoreErrorCorrelator<'a> {
    fn new(
        function: VectorSimilarityFunction,
        neighbors: &'a [ScoreDocsAndScoreVariance],
        vectors: &'a [Vec<f32>],
        bits: u8,
    ) -> Self {
        let dim = vectors[0].len();
        ScoreErrorCorrelator {
            corr: OnlineMeanAndVar::default(),
            errors: OnlineMeanAndVar::default(),
            function,
            neighbors,
            vectors,
            query: vec![0; dim],
            vector: vec![0; dim],
            bits,
        }
    }

    fn score_error_correlation(&mut self, lower: f32, upper: f32) -> Result<f64, QuantizerError> {
        self.corr.reset();
        let quantizer = ScalarQuantizer::new(lower, upper, self.bits)?;
        let sim = ScalarQuantizedVectorSimilarity::from_vector_similarity(
            self.function,
            quantizer.constant_multiplier(),
            quantizer.bits,
        );
        for (i, nn) in self.neighbors.iter().enumerate() {
            let query_correction =
                quantizer.quantize(&self.vectors[i], &mut self.query, self.function);
            self.errors.reset();
            for &(doc, score) in &nn.score_docs {
                let vector_correction =
                    quantizer.quantize(&self.vectors[doc], &mut self.vector, self.function);
                let q_score = sim.score(
                    &self.query,
                    query_correction,
                    &self.vector,
                    vector_correction,
                );
                self.errors.add((q_score - score) as f64);
            }
            self.corr
                .add((1.0 - self.errors.var() / nn.score_variance) as f64);
        }
        Ok(if self.corr.mean.is_nan() {
            0.0
        } else {
            self.corr.mean
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_similarities() {
        let a = [0.6f32, 0.8];
        let b = [1.0f32, 0.0];
        assert_eq!(
            VectorSimilarityFunction::Euclidean.score(&a, &b),
            1.0 / (1.0 + (0.16 + 0.64))
        );
        assert_eq!(VectorSimilarityFunction::DotProduct.score(&a, &b), 0.8);
        assert_eq!(VectorSimilarityFunction::Cosine.score(&a, &b), 0.8);
        assert_eq!(VectorSimilarityFunction::Cosine.score(&a, &[0.0, 0.0]), 0.5);
        assert_eq!(
            VectorSimilarityFunction::MaximumInnerProduct.score(&a, &b),
            1.6
        );
        assert_eq!(scale_max_inner_product_score(-1.0), 0.5);
    }

    #[test]
    fn encodings_geometry_matches_java() {
        use ScalarEncoding::*;
        for e in ScalarEncoding::ALL {
            assert_eq!(ScalarEncoding::from_wire_number(e.wire_number()), Some(e));
        }
        assert_eq!(ScalarEncoding::from_wire_number(9), None);
        assert_eq!(ScalarEncoding::from_num_bits(4), Some(PackedNibble));
        assert_eq!(ScalarEncoding::from_num_bits(3), None);
        assert!(!UnsignedByte.is_asymmetric());
        assert!(SingleBitQueryNibble.is_asymmetric());
        // dims 13
        assert_eq!(UnsignedByte.discrete_dimensions(13), 13);
        assert_eq!(UnsignedByte.doc_packed_length(13), 13);
        assert_eq!(PackedNibble.discrete_dimensions(13), 14);
        assert_eq!(PackedNibble.doc_packed_length(13), 7);
        assert_eq!(SevenBit.doc_packed_length(13), 13);
        assert_eq!(SingleBitQueryNibble.discrete_dimensions(13), 16);
        assert_eq!(SingleBitQueryNibble.doc_packed_length(13), 2);
        assert_eq!(SingleBitQueryNibble.query_packed_length(13), 8);
        assert_eq!(DibitQueryNibble.discrete_dimensions(13), 16);
        assert_eq!(DibitQueryNibble.doc_packed_length(13), 4);
        assert_eq!(DibitQueryNibble.query_packed_length(13), 8);
        assert_eq!(UnsignedByte.query_packed_length(13), 13);
        assert_eq!(UnsignedByte.query_bits_per_dim(), 8);
    }

    #[test]
    fn pack_unpack_round_trips() {
        let v: Vec<u8> = (0..24).map(|i| (i * 7 % 16) as u8).collect();
        let mut packed = vec![0u8; 12];
        pack_nibbles(&v, &mut packed);
        let mut back = vec![0u8; 24];
        unpack_nibbles(&packed, &mut back);
        assert_eq!(back, v);

        let bits: Vec<u8> = (0..21).map(|i| (i % 3 == 0) as u8).collect();
        let mut packed = vec![0u8; 3];
        pack_as_binary(&bits, &mut packed);
        let mut back = vec![0u8; 21];
        unpack_binary(&packed, &mut back);
        assert_eq!(back, bits);

        let dibits: Vec<u8> = (0..21).map(|i| (i % 4) as u8).collect();
        let mut packed = vec![0u8; 6];
        transpose_dibit(&dibits, &mut packed);
        let mut back = vec![0u8; 21];
        untranspose_dibit(&packed, &mut back);
        assert_eq!(back, dibits);

        let q: Vec<u8> = (0..16).map(|i| (i % 16) as u8).collect();
        let mut t = vec![0u8; 8];
        transpose_half_byte(&q, &mut t);
        // plane 0 (low bits) of 0..16: odd values -> 0b01010101 twice.
        assert_eq!(&t[..2], &[0x55, 0x55]);
        assert_eq!(discretize(13, 8), 16);
        assert_eq!(discretize(16, 8), 16);
    }

    #[test]
    fn quantize_dequantize_is_close() {
        let q = OptimizedScalarQuantizer::new(VectorSimilarityFunction::Euclidean);
        let orig: Vec<f32> = (0..32)
            .map(|i| ((i * 37 % 17) as f32 - 8.0) / 3.0)
            .collect();
        let centroid = vec![0.1f32; 32];
        for bits in [1u8, 2, 4, 7, 8] {
            let mut v = orig.clone();
            let mut dest = vec![0u8; 32];
            let r = q.scalar_quantize(&mut v, &mut dest, bits, &centroid);
            assert!(dest.iter().all(|&d| (d as u32) < (1 << bits)));
            assert_eq!(
                r.quantized_component_sum,
                dest.iter().map(|&d| d as i32).sum::<i32>()
            );
            let mut back = vec![0f32; 32];
            dequantize(
                &dest,
                &mut back,
                bits,
                r.lower_interval,
                r.upper_interval,
                &centroid,
            );
            if bits == 8 {
                for (a, b) in orig.iter().zip(&back) {
                    assert!((a - b).abs() < 0.1, "{a} vs {b}");
                }
            }
        }
        let mut v = orig.clone();
        let mut dests = vec![vec![0u8; 32], vec![0u8; 32]];
        let rs = q.multi_scalar_quantize(&mut v, &mut dests, &[4, 8], &centroid);
        let mut v4 = orig.clone();
        let mut d4 = vec![0u8; 32];
        assert_eq!(rs[0], q.scalar_quantize(&mut v4, &mut d4, 4, &centroid));
        assert_eq!(dests[0], d4);
        // A constant vector (zero variance, zero norm) takes the early exit.
        let mut flat = vec![0.1f32; 8];
        let mut d = vec![0u8; 8];
        let r = q.scalar_quantize(&mut flat, &mut d, 4, &[0.1; 8]);
        assert_eq!(r.additional_correction, 0.0);
    }

    #[test]
    fn legacy_quantizer_basics() {
        assert_eq!(
            ScalarQuantizer::new(f32::NAN, 1.0, 7),
            Err(QuantizerError::NonFinite)
        );
        let sq = ScalarQuantizer::new(-1.0, 1.0, 7).unwrap();
        assert_eq!(sq.bits(), 7);
        assert_eq!(sq.lower_quantile(), -1.0);
        assert_eq!(sq.upper_quantile(), 1.0);
        let alpha = 2.0f32 / 127.0;
        assert_eq!(sq.constant_multiplier(), alpha * alpha);
        let mut dest = [0u8; 3];
        let c = sq.quantize(
            &[-1.0, 0.0, 1.0],
            &mut dest,
            VectorSimilarityFunction::DotProduct,
        );
        assert_eq!(dest, [0, 64, 127]);
        assert!(c.is_finite());
        assert_eq!(
            sq.quantize(
                &[0.5, 0.0, 1.0],
                &mut dest,
                VectorSimilarityFunction::Euclidean
            ),
            0.0
        );
        let mut back = [0f32; 3];
        sq.dequantize(&dest, &mut back);
        assert!((back[2] - 1.0).abs() < 1e-6);
        let other = ScalarQuantizer::new(-2.0, 2.0, 7).unwrap();
        assert_eq!(
            other.recalculate_corrective_offset(&dest, &sq, VectorSimilarityFunction::Euclidean),
            0.0
        );
        assert!(other
            .recalculate_corrective_offset(&dest, &sq, VectorSimilarityFunction::DotProduct)
            .is_finite());
        assert_eq!(
            sq.to_string(),
            "ScalarQuantizer{minQuantile=-1, maxQuantile=1, bits=7}"
        );

        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32 / 50.0, -(i as f32) / 25.0])
            .collect();
        let full = ScalarQuantizer::from_vectors(&vectors, 1.0, 50, 7).unwrap();
        assert_eq!(
            (full.lower_quantile(), full.upper_quantile()),
            (-49.0 / 25.0, 49.0 / 50.0)
        );
        let empty = ScalarQuantizer::from_vectors(&vectors[..0], 0.99, 0, 7).unwrap();
        assert_eq!(empty.lower_quantile(), 0.0);
        let ci = ScalarQuantizer::from_vectors(&vectors, 0.9, 50, 7).unwrap();
        assert!(ci.lower_quantile() > -49.0 / 25.0 && ci.upper_quantile() < 49.0 / 50.0);
        let auto = ScalarQuantizer::from_vectors_auto_interval(
            &vectors,
            VectorSimilarityFunction::DotProduct,
            50,
            4,
        )
        .unwrap();
        assert!(auto.lower_quantile() < auto.upper_quantile());
        assert_eq!(
            ScalarQuantizer::from_vectors_auto_interval(
                &vectors[..0],
                VectorSimilarityFunction::Euclidean,
                0,
                4
            )
            .unwrap()
            .upper_quantile(),
            0.0
        );
    }

    #[test]
    fn quantile_selection() {
        let mut two = [3.0f32, 1.0];
        assert_eq!(upper_and_lower_quantile(&mut two, 0.9), (1.0, 3.0));
        let mut arr: Vec<f32> = (0..100).map(|i| ((i * 37) % 100) as f32).collect();
        // (100 * 0.1 / 2 + 0.5) = 5 dropped at each end.
        assert_eq!(upper_and_lower_quantile(&mut arr, 0.9), (5.0, 94.0));
        assert_eq!(java_float_compare(-0.0, 0.0), std::cmp::Ordering::Less);
        assert_eq!(
            java_float_compare(f32::NAN, f32::INFINITY),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn quantized_similarities() {
        let e = ScalarQuantizedVectorSimilarity::from_vector_similarity(
            VectorSimilarityFunction::Euclidean,
            0.5,
            7,
        );
        assert_eq!(e.score(&[1, 2], 0.0, &[3, 2], 0.0), 1.0 / (1.0 + 2.0));
        let d = ScalarQuantizedVectorSimilarity::from_vector_similarity(
            VectorSimilarityFunction::Cosine,
            0.5,
            4,
        );
        assert_eq!(
            d.score(&[1, 2], 0.25, &[3, 2], 0.25),
            (1.0 + 3.5 + 0.5) / 2.0
        );
        let d8 = ScalarQuantizedVectorSimilarity::from_vector_similarity(
            VectorSimilarityFunction::DotProduct,
            0.5,
            8,
        );
        assert_eq!(d8.score(&[200], 0.0, &[2], 0.0), (1.0 + 200.0) / 2.0);
        let m = ScalarQuantizedVectorSimilarity::from_vector_similarity(
            VectorSimilarityFunction::MaximumInnerProduct,
            1.0,
            8,
        );
        assert_eq!(m.score(&[1], 0.0, &[2], 0.0), 3.0);
    }

    #[test]
    fn top10_queue_keeps_the_best_and_pops_least_first() {
        let mut q = Vec::new();
        for i in 0..20usize {
            insert_top10(&mut q, (i, (i % 7) as f32));
        }
        assert_eq!(q.len(), 10);
        // Least first: lowest score, and among ties the highest doc.
        assert!(q.windows(2).all(|w| hit_less(w[0], w[1])));
        assert!(q.iter().all(|&(_, s)| s >= 3.0));
    }
}
