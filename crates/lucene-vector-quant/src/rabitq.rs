//! Extended RaBitQ: multi-bit RaBitQ quantization relative to a partition
//! centroid (Gao, Gou, Xu, Yang, Long, Wong, "Practical and Asymptotically
//! Optimal Quantization of High-Dimensional Vectors in Euclidean Space for
//! Approximate Nearest Neighbor Search", SIGMOD 2025).
//!
//! Vectors and centroids are rotated by a random orthogonal matrix `P` (the
//! seeded randomized Hadamard transform of [`crate::rotation::Rotation`],
//! zero-padded to `n` dimensions: the dimension rounded up to a multiple of
//! 32, or up to 32 to a power of two, at least 8); a
//! centroid is rotated once, by [`Quantizer::prepare_centroid`]. A
//! vector's rotated residual `P r = P o - P c` is snapped to the codebook of
//! `B`-bit vectors `y = u - (2^B - 1) / 2`, `u` in `0..2^B` per dimension:
//! the code is the `y` of largest cosine with `P r` among the codebook points
//! nearest to `t * P r` over rescaling factors `t` (the paper's
//! critical-value search; from 5 bits, only factors near the one that
//! saturates the largest coordinate). The inner product of `r` with any
//! vector `x` is estimated, without bias, as
//!
//! ```text
//! <r, x> ~ |r|^2 * <y, P x> / <y, P r>
//! ```
//!
//! A code is `u`, then three little-endian `f32` corrective terms:
//!
//! ```text
//! u: up to 6 bits, B bit planes of n / 8 bytes (dimension d in bit d % 8 of byte d / 8);
//!    7 or 8 bits, one byte per dimension
//! f = |r|^2 / <y, P r>,  g = f * sum(u),  extra (|r|^2 for Euclidean, <r, c> otherwise)
//! ```
//!
//! The query residual `z = P q - P c` is quantized per centroid over its
//! `[min, max]` range to integers `v`, so `<u, z>` comes from the integer
//! dot product `<u, v>`; the codebook offset is applied to the quantized
//! query, so only `<y, z - z_quantized>` is left as query error. Up to 6
//! bits, `v` has 6 bits and codes are stored 32 to a block, each bit plane
//! as 4-bit groups of dimensions, scored by fast scan
//! (`lucene_util::simd::lut16_accumulate32`): one pass per bit plane over
//! 16-entry tables of sums of four query values, highest plane first,
//! doubling between planes. At 7 and 8 bits `v` has 8 bits and codes are
//! scored with a byte dot product per code. The 6-bit query adds about 10%
//! to the error at 5 bits and 35% at 6.

use std::cell::RefCell;

use lucene_util::{quantization, simd, vector_util};

use crate::rotation::Rotation;
use crate::{
    check_sample, f32_at, prepare_vector, Error, Quantizer, QueryScorer, Result, Similarity,
};

/// The registry name of [`ExRabitqFactory`].
pub const NAME: &str = "erq";

/// The rotation seed of [`ExRabitqFactory::new`].
pub const DEFAULT_SEED: u64 = 0x5eed_e5ab_17c0_0001;

/// Sign-flip and Walsh-Hadamard rounds of the rotation.
const ROUNDS: u8 = 3;

/// Bytes of corrective terms after `u`: three floats.
const CORRECTIONS_BYTES: usize = 12;

/// The largest code dimension: an integer dot product of `n` byte pairs
/// stays below `2^31`.
const MAX_PADDED_DIMENSION: usize = 1 << 15;

/// Above this many dimensions the code dimension is a multiple of it.
const CODE_ALIGN: usize = 32;

/// The code dimension of `dimension`-dimensional vectors: the dimension
/// rounded up to a multiple of [`CODE_ALIGN`], or up to it the next power
/// of two, at least 8. Either way bit planes are whole bytes and fast-scan
/// groups of four dimensions whole, and the padding the rotation spreads
/// energy into is small.
fn code_dimension(dimension: usize) -> usize {
    if dimension <= CODE_ALIGN {
        dimension.next_power_of_two().max(8)
    } else {
        dimension
            .checked_next_multiple_of(CODE_ALIGN)
            .unwrap_or(dimension)
    }
}

/// The widest codes stored as bit planes and scored by fast scan.
const MAX_PLANE_BITS: u8 = 6;

/// The largest query value for fast scan: a table entry sums four, at most
/// `4 * 63`, which fits the tables' bytes.
const QUERY_TOP_FAST_SCAN: u8 = 63;

/// Bit planes of a fast-scan query value (at most [`QUERY_TOP_FAST_SCAN`]).
const QUERY_PLANES: usize = 6;

/// The number of bits set in both `a` and `b`, over their common length.
fn and_popcount(a: &[u8], b: &[u8]) -> u32 {
    let (mut wa, mut wb) = (a.chunks_exact(8), b.chunks_exact(8));
    let mut count = 0u32;
    for (x, y) in (&mut wa).zip(&mut wb) {
        let x = u64::from_le_bytes(x.try_into().unwrap_or([0; 8]));
        let y = u64::from_le_bytes(y.try_into().unwrap_or([0; 8]));
        count = count.saturating_add((x & y).count_ones());
    }
    for (&x, &y) in wa.remainder().iter().zip(wb.remainder()) {
        count = count.saturating_add((x & y).count_ones());
    }
    count
}

/// Codes per fast-scan block.
const BLOCK: usize = simd::FAST_SCAN_BLOCK;

/// Builds [`ExRabitqQuantizer`]s of one bit width. Nothing is learned from
/// the sample; the state is the bit width and the rotation seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExRabitqFactory {
    bits: u8,
    seed: u64,
}

impl ExRabitqFactory {
    /// Quantizers of `bits` bits per dimension with the default rotation.
    ///
    /// # Panics
    ///
    /// When `bits` is not in `2..=8`.
    pub fn new(bits: u8) -> Self {
        Self::with_seed(bits, DEFAULT_SEED)
    }

    /// Quantizers of `bits` bits per dimension whose rotation draws its signs
    /// from `seed`.
    ///
    /// # Panics
    ///
    /// When `bits` is not in `2..=8`.
    pub fn with_seed(bits: u8, seed: u64) -> Self {
        assert!((2..=8).contains(&bits), "bits must be in 2..=8, got {bits}");
        ExRabitqFactory { bits, seed }
    }

    /// Bits per dimension.
    pub fn bits(&self) -> u8 {
        self.bits
    }
}

impl crate::QuantizerFactory for ExRabitqFactory {
    fn name(&self) -> &str {
        NAME
    }

    /// The dimension rounded up to a multiple of 32, or to a power of two
    /// (at least 8) up to 32.
    fn code_dimension(&self, dimension: usize) -> usize {
        code_dimension(dimension)
    }

    fn train(
        &self,
        dimension: usize,
        similarity: Similarity,
        sample: &[f32],
    ) -> Result<Box<dyn Quantizer>> {
        check_sample(dimension, sample)?;
        Ok(Box::new(ExRabitqQuantizer::new(
            dimension, similarity, self.bits, self.seed,
        )?))
    }

    fn read(
        &self,
        dimension: usize,
        similarity: Similarity,
        state: &[u8],
    ) -> Result<Box<dyn Quantizer>> {
        let [bits, seed @ ..] = state else {
            return Err(Error::CorruptState("empty erq state".into()));
        };
        let seed: [u8; 8] = seed.try_into().map_err(|_| {
            Error::CorruptState(format!("erq state is 9 bytes, got {}", state.len()))
        })?;
        if !(2..=8).contains(bits) {
            return Err(Error::CorruptState(format!("erq bits {bits}")));
        }
        check_sample(dimension, &[])?;
        Ok(Box::new(ExRabitqQuantizer::new(
            dimension,
            similarity,
            *bits,
            u64::from_le_bytes(seed),
        )?))
    }
}

/// Extended RaBitQ quantization of one field's vectors.
#[derive(Debug, Clone)]
pub struct ExRabitqQuantizer {
    dimension: usize,
    similarity: Similarity,
    bits: u8,
    rotation: Rotation,
    /// The code dimension `n`, the rotation's width: [`code_dimension`].
    dims: usize,
}

impl ExRabitqQuantizer {
    /// A quantizer of `dimension`-dimensional vectors at `bits` bits per
    /// dimension, rotating with signs drawn from `seed`.
    ///
    /// # Errors
    ///
    /// When `bits` is not in `2..=8`, or `dimension` is zero or pads past
    /// `2^15`.
    pub fn new(dimension: usize, similarity: Similarity, bits: u8, seed: u64) -> Result<Self> {
        if !(2..=8).contains(&bits) {
            return Err(Error::IllegalArgument(format!(
                "erq bits must be in 2..=8, got {bits}"
            )));
        }
        if dimension > MAX_PADDED_DIMENSION {
            return Err(Error::IllegalArgument(format!(
                "erq dimension must be at most {MAX_PADDED_DIMENSION}, got {dimension}"
            )));
        }
        let dims = code_dimension(dimension);
        Ok(ExRabitqQuantizer {
            dimension,
            similarity,
            bits,
            rotation: Rotation::new(dimension, dims, seed, ROUNDS)?,
            dims,
        })
    }

    /// Bits per dimension.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// The rotation seed.
    pub fn seed(&self) -> u64 {
        self.rotation.seed()
    }

    /// Whether codes are bit planes scored by fast scan.
    fn planes(&self) -> bool {
        self.bits <= MAX_PLANE_BITS
    }

    /// Bytes of `u` in a code.
    fn packed(&self) -> usize {
        if self.planes() {
            // `dims` is a multiple of 8 below 2^16.
            usize::from(self.bits).saturating_mul(self.dims / 8)
        } else {
            self.dims
        }
    }

    /// The largest quantized query value: 6 bits for fast scan, else 8.
    fn query_top(&self) -> u8 {
        if self.planes() {
            QUERY_TOP_FAST_SCAN
        } else {
            u8::MAX
        }
    }

    /// Fast-scan table groups: four dimensions each.
    fn groups(&self) -> usize {
        self.dims / 4
    }

    /// Bytes of one fast-scan block: a 16-byte row of 32 nibbles per plane
    /// and group, then each corrective term for the 32 codes.
    // ARITH: groups and bits are bounded by the code dimension and 6.
    #[allow(clippy::arithmetic_side_effects)]
    fn block_bytes(&self) -> usize {
        usize::from(self.bits) * self.groups() * 16 + BLOCK * CORRECTIONS_BYTES
    }

    /// `x` (of the field's dimension) rotated into `out` (of the code
    /// dimension).
    fn rotate(&self, x: &[f32], out: &mut [f32]) {
        self.rotation.rotate(x, out);
    }
}

/// Reused buffers for encoding.
#[derive(Debug, Default)]
struct Scratch {
    rotated: Vec<f32>,
    abs: Vec<f32>,
    levels: Vec<u8>,
    search: Search,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::default();
}

/// Reused buffers of the critical-value search.
#[derive(Debug, Default)]
struct Search {
    /// The critical values in the window, as [`event_key`]s.
    keys: Vec<u64>,
    /// Bucket offsets of the critical values.
    buckets: Vec<u32>,
    /// The critical values in increasing order.
    sorted: Vec<u64>,
}

/// The window of rescaling factors the critical-value search visits, as
/// fractions of the factor at which the largest coordinate reaches the top
/// level, per bit width. For rotated (near-Gaussian) coordinates of 64 to
/// 1024 dimensions the best factor falls in 1.7..4.3 at 2 bits, 1.2..2.7 at
/// 3, 1.0..1.8 at 4, 0.95..1.45 at 5, 0.85..1.15 at 6, 0.89..1.05 at 7 and
/// 0.83..1.02 at 8; each window keeps a margin around that and skips the
/// critical values outside it.
const SEARCH_WINDOW: [[f32; 2]; 9] = [
    FULL_WINDOW,
    FULL_WINDOW,
    [1.2, 6.0],
    [0.9, 3.5],
    [0.85, 2.2],
    [0.7, 1.8],
    [0.75, 1.3],
    [0.8, 1.15],
    [0.75, 1.1],
];

/// A ceiling on `tan` of the angle between a code `y` and the rotated
/// residual `P r` it encodes, per bit width. After the rotation a residual's
/// coordinates are near Gaussian whatever the data, so the angle depends on
/// little but the bit width: over isotropic and skewed residuals of 25 to
/// 1024 dimensions the largest values seen were 0.48, 0.32, 0.16, 0.085,
/// 0.045, 0.025 and 0.011 at 2 to 8 bits.
const TAN_CEILING: [f32; 9] = [1.0, 1.0, 0.5, 0.33, 0.17, 0.09, 0.05, 0.027, 0.012];

/// Every rescaling factor.
const FULL_WINDOW: [f32; 2] = [0.0, f32::INFINITY];

/// A sort key ordering by decreasing `value` (a positive float, whose bits
/// order as it does), then by `index`.
#[inline]
fn event_key(value: f32, index: u32) -> u64 {
    (u64::from(!value.to_bits()) << 32) | u64::from(index)
}

/// `1 / k` for the levels `k`.
const RECIPROCALS: [f32; 128] = {
    let mut r = [0f32; 128];
    let mut k = 1;
    while k < 128 {
        r[k] = 1.0 / k as f32;
        k += 1;
    }
    r
};

/// The magnitude levels `k` (the code value `|y| = k + 1/2`, `k` at most
/// `max_level`) of the rounding of `t * a` with the largest cosine to `a`
/// over the rescaling factors `t` in `window` (fractions of
/// `max_level / max(a)`). Returns the best `t` as such a fraction.
///
/// The rounding changes only where `t * a_i` crosses an integer `k`, at the
/// critical value `t = k / a_i`. Those in the window are put in increasing
/// order by a bucket sort over their range and visited in turn while
/// `<y, a>` and `|y|^2` are updated one coordinate at a time.
// ARITH: levels are at most `max_level` (at most 127); bucket indices are
// below the bucket count; critical values number at most
// `a.len() * max_level`, below 2^22.
#[allow(clippy::arithmetic_side_effects)]
fn best_levels(
    a: &[f32],
    max_level: u8,
    [start, end]: [f32; 2],
    levels: &mut [u8],
    s: &mut Search,
) -> f32 {
    let a_max = a.iter().fold(0f32, |m, &x| m.max(x));
    levels.fill(0);
    if a_max <= 0.0 {
        return 0.0;
    }
    let t_top = f32::from(max_level) / a_max;
    let (t_start, t_end) = (start * t_top, end * t_top);
    let (mut dot, mut norm2) = (0f64, 0f64);
    for (l, &x) in levels.iter_mut().zip(a) {
        // The float-to-int cast saturates at 255; the level is capped below.
        *l = ((t_start * x).floor() as u8).min(max_level);
        let y = f64::from(*l) + 0.5;
        dot += y * f64::from(x);
        norm2 += y * y;
    }
    // Each coordinate's critical values above its starting level, up to
    // the window's end, as their inverses `u = a_i / k`: bounded by `max(a)`
    // where `t` grows without bound for small `a_i`, so they spread evenly
    // over buckets.
    let u_end = 1.0 / t_end;
    s.keys.clear();
    let (mut u_min, mut u_max) = (f32::INFINITY, 0f32);
    for (i, (&x, &l)) in a.iter().zip(levels.iter()).enumerate() {
        for k in l + 1..=max_level {
            let u = x * RECIPROCALS[usize::from(k)];
            // A zero magnitude never reaches a level.
            if !(u >= u_end && u > 0.0) {
                break;
            }
            u_min = u_min.min(u);
            u_max = u_max.max(u);
            s.keys.push(event_key(u, i as u32));
        }
    }
    // Bucket sort, one bucket per critical value on average: the bucket
    // grows as `u` falls, so buckets in turn, each sorted, visit the
    // critical values in increasing order of `t`.
    let buckets = s.keys.len().max(1);
    let scale = if u_max > u_min {
        buckets as f32 / (u_max - u_min)
    } else {
        0.0
    };
    let bucket = |key: u64| -> usize {
        let u = f32::from_bits(!((key >> 32) as u32));
        // Non-negative and saturating; capped at the last bucket.
        (((u_max - u) * scale) as usize).min(buckets - 1)
    };
    s.buckets.clear();
    s.buckets.resize(buckets + 1, 0);
    for &key in &s.keys {
        s.buckets[bucket(key) + 1] += 1;
    }
    for b in 1..=buckets {
        s.buckets[b] += s.buckets[b - 1];
    }
    s.sorted.clear();
    s.sorted.resize(s.keys.len(), 0);
    for &key in &s.keys {
        let b = bucket(key);
        s.sorted[s.buckets[b] as usize] = key;
        s.buckets[b] += 1;
    }
    // Insertion sort within each bucket: a few values each.
    let mut lo = 0usize;
    for &hi in &s.buckets[..buckets] {
        let run = &mut s.sorted[lo..hi as usize];
        for j in 1..run.len() {
            let key = run[j];
            let mut at = j;
            while at > 0 && run[at - 1] > key {
                run[at] = run[at - 1];
                at -= 1;
            }
            run[at] = key;
        }
        lo = hi as usize;
    }
    let (mut best_dot2, mut best_norm2) = (dot * dot, norm2);
    let (mut best_t, mut best_events) = (t_start, 0usize);
    for (e, &key) in s.sorted.iter().enumerate() {
        let i = key as u32 as usize;
        let k = levels[i] + 1;
        levels[i] = k;
        dot += f64::from(a[i]);
        // (k + 1/2)^2 - (k - 1/2)^2
        norm2 += 2.0 * f64::from(k);
        // A larger cosine: dot^2 / norm2 > best_dot2 / best_norm2.
        if dot * dot * best_norm2 > best_dot2 * norm2 {
            (best_dot2, best_norm2) = (dot * dot, norm2);
            best_t = 1.0 / f32::from_bits(!((key >> 32) as u32));
            best_events = e + 1;
        }
    }
    for &key in &s.sorted[best_events..] {
        levels[key as u32 as usize] -= 1;
    }
    best_t / t_top
}

impl Quantizer for ExRabitqQuantizer {
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
        self.packed().saturating_add(CORRECTIONS_BYTES)
    }

    fn uses_centroid(&self) -> bool {
        true
    }

    fn encode(&self, vector: &[f32], centroid: &[f32], out: &mut [u8]) {
        assert_eq!(centroid.len(), self.dimension, "centroid dimension");
        let mut prepared = vec![0f32; self.dims];
        self.rotate(centroid, &mut prepared);
        self.encode_prepared(vector, &prepared, out);
    }

    fn prepare_centroid(&self, centroid: &[f32]) -> Option<Vec<f32>> {
        assert_eq!(centroid.len(), self.dimension, "centroid dimension");
        let mut prepared = vec![0f32; self.dims];
        self.rotate(centroid, &mut prepared);
        Some(prepared)
    }

    // ARITH: levels are at most 127 and codes below 256, their sum below
    // 2^23; byte and plane indices are below the packed length.
    #[allow(clippy::arithmetic_side_effects)]
    fn encode_prepared(&self, vector: &[f32], prepared: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dimension, "vector dimension");
        assert_eq!(prepared.len(), self.dims, "centroid dimension");
        assert_eq!(out.len(), self.code_len(), "code length");
        SCRATCH.with_borrow_mut(|s| {
            let n = self.dims;
            s.rotated.resize(n, 0.0);
            s.abs.resize(n, 0.0);
            s.levels.resize(n, 0);
            // The prepared vector, before the rotation, lands in `abs`.
            let normalized = &mut s.abs[..self.dimension];
            prepare_vector(self.similarity, vector, normalized);
            self.rotation.rotate(normalized, &mut s.rotated);
            let extra = if self.similarity == Similarity::Euclidean {
                0.0
            } else {
                simd::dot_f32(&s.rotated, prepared) - simd::dot_f32(prepared, prepared)
            };
            for (x, &c) in s.rotated.iter_mut().zip(prepared) {
                *x -= c;
            }
            let norm2 = simd::dot_f32(&s.rotated, &s.rotated);
            let extra = if self.similarity == Similarity::Euclidean {
                norm2
            } else {
                extra
            };
            for (a, &x) in s.abs.iter_mut().zip(&s.rotated) {
                *a = x.abs();
            }
            let half = 1u8 << (self.bits - 1);
            best_levels(
                &s.abs,
                half - 1,
                SEARCH_WINDOW[usize::from(self.bits)],
                &mut s.levels,
                &mut s.search,
            );
            // `u = half + k` for a non-negative coordinate, `half - 1 - k`
            // otherwise, so `y = u - (2^B - 1) / 2 = +-(k + 1/2)`.
            let mut y_dot_r = 0f64;
            let mut sum_u = 0u32;
            for (u, &x) in s.levels.iter_mut().zip(&s.rotated) {
                let k = *u;
                y_dot_r += (f64::from(k) + 0.5) * f64::from(x.abs());
                *u = if x >= 0.0 { half + k } else { half - 1 - k };
                sum_u += u32::from(*u);
            }
            let f = if y_dot_r > 0.0 {
                (f64::from(norm2) / y_dot_r) as f32
            } else {
                0.0
            };
            let (packed, tail) = out.split_at_mut(self.packed());
            if self.planes() {
                packed.fill(0);
                for (p, plane) in packed.chunks_exact_mut(n / 8).enumerate() {
                    for (byte, u8s) in plane.iter_mut().zip(s.levels.chunks_exact(8)) {
                        for (b, &u) in u8s.iter().enumerate() {
                            *byte |= ((u >> p) & 1) << b;
                        }
                    }
                }
            } else {
                packed.copy_from_slice(&s.levels);
            }
            tail[..4].copy_from_slice(&f.to_le_bytes());
            tail[4..8].copy_from_slice(&(f * sum_u as f32).to_le_bytes());
            tail[8..12].copy_from_slice(&extra.to_le_bytes());
        });
    }

    fn query_scorer(&self, query: &[f32]) -> Box<dyn QueryScorer + '_> {
        assert_eq!(query.len(), self.dimension, "query dimension");
        let mut normalized = vec![0f32; self.dimension];
        prepare_vector(self.similarity, query, &mut normalized);
        let mut rotated = vec![0f32; self.dims];
        self.rotate(&normalized, &mut rotated);
        Box::new(ExRabitqQueryScorer {
            q: self,
            rotated,
            values: vec![0u8; self.dims],
            luts: if self.planes() {
                vec![0u8; self.groups().saturating_mul(16)]
            } else {
                Vec::new()
            },
            query_planes: if self.planes() {
                vec![0u8; QUERY_PLANES.saturating_mul(self.dims / 8)]
            } else {
                Vec::new()
            },
            lower: 0.0,
            step: 0.0,
            offset_sum: 0.0,
            query_extra: 0.0,
            z_norm: 0.0,
            centroid_norm2: 0.0,
        })
    }

    fn state(&self) -> Vec<u8> {
        let mut state = vec![self.bits];
        state.extend_from_slice(&self.seed().to_le_bytes());
        state
    }

    fn block_len(&self, n: usize) -> usize {
        if self.planes() {
            n.div_ceil(BLOCK).saturating_mul(self.block_bytes())
        } else {
            self.code_len().saturating_mul(n)
        }
    }

    // ARITH: offsets are below the code length and groups below the code
    // dimension; `codes` is checked to be whole codes.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_block(&self, codes: &[u8], out: &mut Vec<u8>) {
        let code_len = self.code_len();
        assert_eq!(codes.len() % code_len, 0, "codes must be whole codes");
        if !self.planes() {
            out.extend_from_slice(codes);
            return;
        }
        let plane_len = self.dims / 8;
        let packed = self.packed();
        for block in codes.chunks(BLOCK * code_len) {
            let code = |j: usize| block.get(j * code_len..(j + 1) * code_len);
            // Group `g` of plane `p`: dimensions 4g..4g+4, dimension
            // 4g + k in bit k.
            let nibble = |j: usize, p: usize, g: usize| -> u8 {
                code(j).map_or(0, |c| (c[p * plane_len + g / 2] >> (4 * (g % 2))) & 15)
            };
            for p in 0..usize::from(self.bits) {
                for g in 0..self.groups() {
                    for j in 0..16 {
                        out.push(nibble(j, p, g) | (nibble(j + 16, p, g) << 4));
                    }
                }
            }
            for at in [0, 4, 8] {
                for j in 0..BLOCK {
                    match code(j) {
                        Some(c) => out.extend_from_slice(&c[packed + at..packed + at + 4]),
                        None => out.extend_from_slice(&[0; 4]),
                    }
                }
            }
        }
    }
}

/// A query rotated once and quantized against one centroid at a time.
#[derive(Debug)]
struct ExRabitqQueryScorer<'q> {
    q: &'q ExRabitqQuantizer,
    /// `P q`, of the code dimension.
    rotated: Vec<f32>,
    /// `z` quantized to 6 bits for fast scan, else 8: `z_i ~ lower + step *
    /// v_i`.
    values: Vec<u8>,
    /// For fast scan, one 16-entry table per group of four dimensions: entry
    /// `m` sums `v` over the dimensions whose bits are set in `m`.
    luts: Vec<u8>,
    /// For fast scan, `v` as [`QUERY_PLANES`] bit planes of `n / 8` bytes
    /// (bit `j` of `v_d` in bit `d % 8` of byte `d / 8` of plane `j`), so one
    /// code scores by AND popcounts without its block.
    query_planes: Vec<u8>,
    lower: f32,
    step: f32,
    /// `(2^B - 1) / 2 * sum(lower + step * v)`: the codebook's offset applied
    /// to the quantized query.
    offset_sum: f32,
    /// `|z|^2` for Euclidean, `<P c, P q>` otherwise.
    query_extra: f32,
    /// `|z|`.
    z_norm: f32,
    /// `|c|^2`.
    centroid_norm2: f32,
}

impl QueryScorer for ExRabitqQueryScorer<'_> {
    fn set_centroid(&mut self, centroid: &[f32]) {
        let prepared = self.q.prepare_centroid(centroid).unwrap_or_default();
        self.set_prepared_centroid(&prepared);
    }

    // ARITH: tables are built only for fast scan, where values are at most
    // 63 and a table entry (four values) at most 252; values sum to below
    // 2^8 * 2^15.
    #[allow(clippy::arithmetic_side_effects)]
    fn set_prepared_centroid(&mut self, prepared: &[f32]) {
        assert_eq!(prepared.len(), self.q.dims, "centroid dimension");
        // `z = P q - P c`, its range and norm in one pass, then its levels.
        let stats = simd::residual_stats(&self.rotated, prepared);
        let (lo, hi, norm2) = (stats.min, stats.max, stats.sum_sq as f32);
        let top = self.q.query_top();
        let inv = if hi > lo {
            f32::from(top) / (hi - lo)
        } else {
            0.0
        };
        let step = if hi > lo {
            (hi - lo) / f32::from(top)
        } else {
            0.0
        };
        // Every `z` is in `[lo, hi]`, so each level is in `0..=top`.
        let (sum, _) = simd::quantize_residual(
            &self.rotated,
            prepared,
            [lo, hi],
            inv,
            step,
            &mut self.values,
        );
        let sum = sum as u32;
        self.lower = lo;
        self.step = step;
        let center = f64::from((1u16 << self.q.bits) - 1) / 2.0;
        let n = self.q.dims as f64;
        self.offset_sum =
            (center * (f64::from(lo) * n + f64::from(self.step) * f64::from(sum))) as f32;
        self.centroid_norm2 = simd::dot_f32(prepared, prepared);
        self.query_extra = if self.q.similarity == Similarity::Euclidean {
            norm2
        } else {
            simd::dot_f32(&self.rotated, prepared)
        };
        self.z_norm = norm2.sqrt();
        if !self.luts.is_empty() {
            simd::nibble_luts(&self.values, &mut self.luts);
            // The 6-bit query values as bit planes, as codes store theirs.
            let plane_len = self.q.dims / 8;
            self.query_planes.fill(0);
            for (j, plane) in self.query_planes.chunks_exact_mut(plane_len).enumerate() {
                for (byte, v8) in plane.iter_mut().zip(self.values.chunks_exact(8)) {
                    for (b, &v) in v8.iter().enumerate() {
                        *byte |= ((v >> j) & 1) << b;
                    }
                }
            }
        }
    }

    // ARITH: the sum over planes is below `2^6 * 63 * 2^15`, under 2^31;
    // bit positions are below 8.
    #[allow(clippy::arithmetic_side_effects)]
    fn score(&self, code: &[u8]) -> f32 {
        let (packed, tail) = code.split_at(self.q.packed());
        let dot = if self.q.planes() {
            // `<u, v>` as sums of AND popcounts of the code's bit planes with
            // the query's: plane `p` of `u` against plane `j` of `v` counts
            // `2^(p + j)` per common bit.
            let plane_len = self.q.dims / 8;
            let mut acc = 0u32;
            for plane in packed.chunks_exact(plane_len).rev() {
                let mut s = 0u32;
                for (j, query) in self.query_planes.chunks_exact(plane_len).enumerate() {
                    s += and_popcount(query, plane) << j;
                }
                acc = acc * 2 + s;
            }
            acc as f32
        } else {
            vector_util::uint8_dot_product(&self.values, packed) as f32
        };
        let ip = self.inner_product(dot, f32_at(tail, 0), f32_at(tail, 4));
        let mut s = [0f32];
        self.finish(&[ip], &[f32_at(tail, 8)], &mut s);
        s[0]
    }

    fn score_block(&self, codes: &[u8], scores: &mut [f32]) {
        if !self.q.planes() {
            let code_len = self.q.code_len();
            assert_eq!(
                Some(codes.len()),
                code_len.checked_mul(scores.len()),
                "codes must hold one code per score"
            );
            let packed = self.q.packed();
            // Integer dot products and estimates first, then the scores of up
            // to 32 codes in one vectorized pass.
            for (codes, out) in codes
                .chunks(code_len.saturating_mul(BLOCK))
                .zip(scores.chunks_mut(BLOCK))
            {
                let (mut ip, mut extra) = ([0f32; BLOCK], [0f32; BLOCK]);
                for ((code, ip), extra) in codes.chunks_exact(code_len).zip(&mut ip).zip(&mut extra)
                {
                    let (u, tail) = code.split_at(packed);
                    let dot = vector_util::uint8_dot_product(&self.values, u) as f32;
                    *ip = self.inner_product(dot, f32_at(tail, 0), f32_at(tail, 4));
                    *extra = f32_at(tail, 8);
                }
                let k = out.len();
                self.finish(&ip[..k], &extra[..k], out);
            }
            return;
        }
        self.fast_scan(codes, scores);
    }

    /// RaBitQ's bound. The estimate of `<r, z>` is off by `<e, z>` for the
    /// code's error `e = f y - P r`, `|e| = |r| tan(theta)`, plus `f <y, d>`
    /// for the query's rounding `d` (at most half a step per dimension);
    /// with the rotation making `e` a random direction to `z`, their
    /// deviation is `|r| sqrt((tan(theta) |z|)^2 / n + step^2 / (12
    /// cos(theta)^2))`, with `tan(theta)` at the bit width's ceiling. `|r|^2`
    /// is the stored `extra` for `Euclidean` and `1 - 2 extra - |c|^2` for
    /// `Cosine` and `DotProduct`, whose vectors are unit length; with no way
    /// to recover `|r|` for `MaximumInnerProduct`, its error is unknown. The
    /// margin is [`crate::BOUND_CONFIDENCE`] deviations (twice that in
    /// squared distance for `Euclidean`).
    // ARITH: `j` is below the block's code count, so every offset is inside
    // `codes`, whose length the layout check fixes.
    #[allow(clippy::arithmetic_side_effects)]
    fn error_bounds(&self, codes: &[u8], scores: &[f32], bounds: &mut [f32]) -> bool {
        let similarity = self.q.similarity;
        if similarity == Similarity::MaximumInnerProduct {
            return false;
        }
        assert_eq!(scores.len(), bounds.len(), "one bound per score");
        assert_eq!(
            codes.len(),
            self.q.block_len(scores.len()),
            "codes must hold one code per score"
        );
        let per_r = self.margin_per_r();
        for (j, (b, &s)) in bounds.iter_mut().zip(scores).enumerate() {
            *b = self.margin(codes, j, s, per_r);
        }
        true
    }

    fn error_bound(&self, codes: &[u8], index: usize, score: f32) -> Option<f32> {
        (self.q.similarity != Similarity::MaximumInnerProduct)
            .then(|| self.margin(codes, index, score, self.margin_per_r()))
    }
}

impl ExRabitqQueryScorer<'_> {
    /// The raw error margin per unit of `|r|` (see
    /// [`QueryScorer::error_bounds`]).
    fn margin_per_r(&self) -> f32 {
        let tan = TAN_CEILING[usize::from(self.q.bits)];
        let doc = tan * self.z_norm / (self.q.dims as f32).sqrt();
        let query = self.step * (1.0 + tan * tan).sqrt() / 12f32.sqrt();
        let deviation = (doc * doc + query * query).sqrt();
        if self.q.similarity == Similarity::Euclidean {
            2.0 * crate::BOUND_CONFIDENCE * deviation
        } else {
            crate::BOUND_CONFIDENCE * deviation
        }
    }

    /// The error margin of code `j` of the block `codes`, estimated `score`.
    // ARITH: callers pass a `j` inside `codes`, so every offset is too.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn margin(&self, codes: &[u8], j: usize, score: f32, per_r: f32) -> f32 {
        let at = if self.q.planes() {
            let terms = usize::from(self.q.bits) * self.q.groups() * 16;
            (j / BLOCK) * self.q.block_bytes() + terms + (2 * BLOCK + j % BLOCK) * 4
        } else {
            j * self.q.code_len() + self.q.packed() + 8
        };
        let extra = f32_at(codes, at);
        let r2 = if self.q.similarity == Similarity::Euclidean {
            extra
        } else {
            1.0 - 2.0 * extra - self.centroid_norm2
        };
        crate::score_margin(self.q.similarity, score, per_r * r2.max(0.0).sqrt())
    }

    /// The estimate of `<r, z>` from the integer dot product `<u, v>` and a
    /// code's `f` and `g`.
    #[inline]
    fn inner_product(&self, dot: f32, f: f32, g: f32) -> f32 {
        // <r, z> ~ f * <y, z> = f * (<u, z> - offset_sum), with
        // <u, z> ~ lower * sum(u) + step * <u, v>.
        f * (self.step * dot - self.offset_sum) + self.lower * g
    }

    /// The scores of codes with inner-product estimates `ip` and extra
    /// corrective terms `extra`, into `out`, one similarity loop at a time
    /// so each vectorizes.
    #[inline]
    fn finish(&self, ip: &[f32], extra: &[f32], out: &mut [f32]) {
        let q = self.query_extra;
        let codes = out.iter_mut().zip(ip).zip(extra);
        match self.q.similarity {
            Similarity::Euclidean => {
                for ((s, &ip), &e) in codes {
                    *s = 1.0 / (1.0 + (e + q - 2.0 * ip).max(0.0));
                }
            }
            // <o, q> = <r, q - c> + <r, c> + <c, q>
            Similarity::MaximumInnerProduct => {
                for ((s, &ip), &e) in codes {
                    *s = quantization::scale_max_inner_product_score(ip + e + q);
                }
            }
            Similarity::DotProduct | Similarity::Cosine => {
                for ((s, &ip), &e) in codes {
                    *s = ((1.0 + ip + e + q) / 2.0).max(0.0);
                }
            }
        }
    }

    /// Scores the fast-scan block layout: per block of 32 codes and query
    /// row, one table pass per bit plane, highest first, doubling the sums
    /// between planes; then the corrective terms finish all 32 scores as
    /// [`QueryScorer::score`] does.
    // ARITH: block offsets are below `codes.len()`, checked against the
    // layout's length up front; sums are bounded as in `score`.
    #[allow(clippy::arithmetic_side_effects)]
    fn fast_scan(&self, codes: &[u8], scores: &mut [f32]) {
        assert_eq!(
            codes.len(),
            self.q.block_len(scores.len()),
            "codes must hold one code per score"
        );
        let plane_bytes = self.q.groups() * 16;
        let wide = self.q.groups() >= simd::FAST_SCAN_WIDE_GROUPS;
        for (block, out) in codes
            .chunks_exact(self.q.block_bytes())
            .zip(scores.chunks_mut(BLOCK))
        {
            let (planes, terms) = block.split_at(plane_bytes * usize::from(self.q.bits));
            let mut acc = [0u32; BLOCK];
            for plane in planes.chunks_exact(plane_bytes).rev() {
                for x in &mut acc {
                    *x *= 2;
                }
                if wide {
                    simd::lut16_accumulate32_wide(&self.luts, plane, &mut acc);
                } else {
                    simd::lut16_accumulate32(&self.luts, plane, &mut acc);
                }
            }
            let mut t = [[0f32; BLOCK]; 3];
            for (row, bytes) in t.iter_mut().zip(terms.chunks_exact(4 * BLOCK)) {
                for (x, b) in row.iter_mut().zip(bytes.chunks_exact(4)) {
                    *x = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
            let mut ip = [0f32; BLOCK];
            for (((ip, &d), &f), &g) in ip.iter_mut().zip(&acc).zip(&t[0]).zip(&t[1]) {
                *ip = self.inner_product(d as f32, f, g);
            }
            let k = out.len();
            self.finish(&ip[..k], &t[2][..k], out);
        }
    }
}

#[cfg(test)]
mod tests;
