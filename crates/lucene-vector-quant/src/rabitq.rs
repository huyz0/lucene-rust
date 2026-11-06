//! Extended RaBitQ: multi-bit RaBitQ quantization relative to a partition
//! centroid (Gao, Gou, Xu, Yang, Long, Wong, "Practical and Asymptotically
//! Optimal Quantization of High-Dimensional Vectors in Euclidean Space for
//! Approximate Nearest Neighbor Search", SIGMOD 2025).
//!
//! A vector's residual `r = o - c` against its centroid is rotated by a
//! random orthogonal matrix `P` (sign flips and fast Walsh-Hadamard
//! transforms over the dimension padded to a power of two `n`). Its
//! direction is then snapped to the codebook of `B`-bit vectors
//! `y = u - (2^B - 1) / 2`, `u` in `0..2^B` per dimension: the code is the
//! `y` of largest cosine with `P r` among the codebook points nearest to
//! `t * P r` over rescaling factors `t` (the paper's critical-value search;
//! from 5 bits, only factors near the one that saturates the largest
//! coordinate). The inner
//! product of `r` with any vector `x` is estimated, without bias, as
//!
//! ```text
//! <r, x> ~ |r|^2 * <y, P x> / <y, P r>
//! ```
//!
//! A code holds `u` packed `B` bits per dimension, then three
//! little-endian `f32` corrective terms:
//!
//! ```text
//! u, split into parts of 8, 4, 2 or 1 bits (n * B / 8 bytes)
//! f = |r|^2 / <y, P r>,  g = f * sum(u),  extra (|r|^2 for Euclidean, <r, c> otherwise)
//! ```
//!
//! A `w`-bit part takes `n * w / 8` bytes `L`; byte `j` holds dimensions
//! `j + k * L` for `k` in `0..8 / w`, the first in the high bits. Part widths
//! are the binary digits of `B` (5 is 4 + 1, 7 is 4 + 2 + 1), widest first in
//! the code; a narrower part holds lower bits of `u`.
//!
//! The query residual `z = P (q - c)` is quantized per centroid over its
//! `[min, max]` range to 8 bits (16 bits, as two byte rows, for 7- and 8-bit
//! codes), so `<u, z>` is a sum of integer dot products of byte rows with
//! each part, shifted by the part's bit offset.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use lucene_util::{quantization, simd, vector_util};

use crate::{
    check_sample, f32_at, prepare_vector, Error, Quantizer, QueryScorer, Result, Similarity,
};

/// The registry name of [`ExRabitqFactory`].
pub const NAME: &str = "erq";

/// The rotation seed of [`ExRabitqFactory::new`].
pub const DEFAULT_SEED: u64 = 0x5eed_e5ab_17c0_0001;

/// Sign-flip and Walsh-Hadamard rounds of the rotation.
const ROUNDS: usize = 3;

/// Bytes of corrective terms after the packed code: three floats.
const CORRECTIONS_BYTES: usize = 12;

/// The largest padded dimension: an integer dot product of `n` byte pairs
/// stays below `2^31`.
const MAX_PADDED_DIMENSION: usize = 1 << 15;

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

/// One packed part of a code: `width` bits of `u` starting at bit `shift`,
/// at byte `offset` of the code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Part {
    width: u32,
    shift: u32,
    offset: usize,
    len: usize,
}

/// Extended RaBitQ quantization of one field's vectors.
#[derive(Debug, Clone)]
pub struct ExRabitqQuantizer {
    dimension: usize,
    similarity: Similarity,
    bits: u8,
    seed: u64,
    /// The padded dimension: a power of two, at least 8.
    padded: usize,
    /// `ROUNDS` rows of `padded` signs, each `+-1 / sqrt(padded)`.
    signs: Vec<f32>,
    parts: Vec<Part>,
    /// Bytes of the packed `u`.
    packed: usize,
}

/// `splitmix64`: the next value of the sign generator.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
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
        let padded = dimension
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
            .max(8);
        if dimension == 0 || padded > MAX_PADDED_DIMENSION {
            return Err(Error::IllegalArgument(format!(
                "erq dimension must be in 1..={MAX_PADDED_DIMENSION}, got {dimension}"
            )));
        }
        let scale = (padded as f32).sqrt().recip();
        let mut state = seed;
        let signs = (0..ROUNDS.saturating_mul(padded))
            .map(|_| {
                if splitmix64(&mut state) >> 63 == 0 {
                    scale
                } else {
                    -scale
                }
            })
            .collect();
        let mut parts = Vec::new();
        let mut offset = 0usize;
        for width in [8u32, 4, 2, 1] {
            let shift = u32::from(bits) & (width.wrapping_sub(1));
            if u32::from(bits) & width != 0 {
                // ARITH: `padded` is a multiple of 8 below 2^15 and `width` at
                // most 8, so a part is at most `padded` bytes.
                #[allow(clippy::arithmetic_side_effects)]
                let len = padded * width as usize / 8;
                parts.push(Part {
                    width,
                    shift,
                    offset,
                    len,
                });
                offset = offset.saturating_add(len);
            }
        }
        Ok(ExRabitqQuantizer {
            dimension,
            similarity,
            bits,
            seed,
            padded,
            signs,
            parts,
            packed: offset,
        })
    }

    /// Bits per dimension.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Bytes per dimension of a quantized query: one up to 6 bits, where the
    /// query's rounding error is well below the code's, and two from 7 bits,
    /// where one byte would dominate it.
    fn query_bytes(&self) -> usize {
        if self.bits <= 6 {
            1
        } else {
            2
        }
    }

    /// The rotation seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Packs `codes` (one `u` per padded dimension) into `packed` in the
    /// part layout of the module docs.
    // ARITH: part offsets and lengths lie within the packed length; shifts
    // are below 8 since a part's slots fill one byte.
    #[allow(clippy::arithmetic_side_effects)]
    fn pack(&self, codes: &[u8], packed: &mut [u8]) {
        packed.fill(0);
        for p in &self.parts {
            let bytes = &mut packed[p.offset..p.offset + p.len];
            let mask = ((1u16 << p.width) - 1) as u8;
            for (k, chunk) in codes.chunks_exact(p.len).enumerate() {
                let shift = 8 - p.width * (k as u32 + 1);
                for (b, &u) in bytes.iter_mut().zip(chunk) {
                    *b |= ((u >> p.shift) & mask) << shift;
                }
            }
        }
    }

    /// `<u, query>` for a packed code and one byte per padded dimension of
    /// query: each part's integer dot product, shifted to its bits.
    fn packed_dot(&self, query: &[u8], packed: &[u8]) -> u64 {
        let mut dot = 0u64;
        for p in &self.parts {
            let bytes = &packed[p.offset..][..p.len];
            let d = if p.width == 8 {
                // At most 255 * 255 * 2^15, below `i32::MAX`.
                vector_util::uint8_dot_product(query, bytes).unsigned_abs()
            } else {
                part_dot(query, bytes, p.width)
            };
            dot = dot.saturating_add(u64::from(d) << p.shift);
        }
        dot
    }

    /// `x` (of the field's dimension) zero-padded and rotated into `out` (of
    /// the padded dimension).
    fn rotate(&self, x: &[f32], out: &mut [f32]) {
        let (head, tail) = out.split_at_mut(x.len());
        head.copy_from_slice(x);
        tail.fill(0.0);
        for signs in self.signs.chunks_exact(self.padded) {
            for (o, s) in out.iter_mut().zip(signs) {
                *o *= s;
            }
            fwht(out);
        }
    }
}

/// The unnormalized fast Walsh-Hadamard transform of `x`, whose length is a
/// power of two.
fn fwht(x: &mut [f32]) {
    let mut h = 1usize;
    while h < x.len() {
        for block in x.chunks_exact_mut(h.saturating_mul(2)) {
            let (a, b) = block.split_at_mut(h);
            for (a, b) in a.iter_mut().zip(b) {
                let (s, d) = (*a + *b, *a - *b);
                *a = s;
                *b = d;
            }
        }
        h = h.saturating_mul(2);
    }
}

/// The window of rescaling factors the critical-value search visits, as
/// fractions of the factor at which the largest coordinate reaches the top
/// level, per bit width. Up to 4 bits every critical value is visited. From 5
/// bits, for rotated (near-Gaussian) coordinates of 64 to 1024 dimensions the
/// best factor falls in 0.83..1.45; the window keeps a wide margin around
/// that and skips most of the `n * (2^(B-1) - 1)` critical values.
const SEARCH_WINDOW: [[f32; 2]; 9] = [
    FULL_WINDOW,
    FULL_WINDOW,
    FULL_WINDOW,
    FULL_WINDOW,
    FULL_WINDOW,
    [0.5, 2.0],
    [0.5, 2.0],
    [0.5, 2.0],
    [0.5, 2.0],
];

/// Every rescaling factor.
const FULL_WINDOW: [f32; 2] = [0.0, f32::INFINITY];

/// The magnitude levels `k` (the code value `|y| = k + 1/2`, `k` at most
/// `max_level`) of the rounding of `t * a` with the largest cosine to `a`
/// over the rescaling factors `t` in `window` (fractions of
/// `max_level / max(a)`). Returns the best `t` as such a fraction.
///
/// The rounding changes only where `t * a_i` crosses an integer, so the
/// search visits those critical values in increasing order, updating `<y, a>`
/// and `|y|^2` one coordinate at a time.
// ARITH: levels are below `max_level` (at most 127) before an increment and
// coordinates index `a`; `events` holds at most `a.len() * max_level` indices.
#[allow(clippy::arithmetic_side_effects)]
fn best_levels(a: &[f32], max_level: u8, [start, end]: [f32; 2], levels: &mut [u8]) -> f32 {
    let a_max = a.iter().fold(0f32, |m, &x| m.max(x));
    levels.fill(0);
    if a_max <= 0.0 {
        return 0.0;
    }
    let t_top = f32::from(max_level) / a_max;
    let (t_start, t_end) = (start * t_top, end * t_top);
    let (mut dot, mut norm2) = (0f64, 0f64);
    let mut heap = BinaryHeap::new();
    let push = |heap: &mut BinaryHeap<Reverse<u64>>, i: usize, level: u8, x: f32| {
        if level < max_level && x > 0.0 {
            let t = f32::from(level + 1) / x;
            if t <= t_end {
                // Positive floats order as their bits do; ties break by index.
                heap.push(Reverse((u64::from(t.to_bits()) << 32) | i as u64));
            }
        }
    };
    for (i, (l, &x)) in levels.iter_mut().zip(a).enumerate() {
        // The float-to-int cast saturates at 255; the level is capped below.
        *l = ((t_start * x).floor() as u8).min(max_level);
        let y = f64::from(*l) + 0.5;
        dot += y * f64::from(x);
        norm2 += y * y;
        push(&mut heap, i, *l, x);
    }
    let mut best = dot * dot / norm2;
    let mut best_t = t_start;
    let mut events = Vec::new();
    let mut best_events = 0usize;
    while let Some(Reverse(key)) = heap.pop() {
        let i = (key & u64::from(u32::MAX)) as usize;
        let l = levels[i];
        dot += f64::from(a[i]);
        // (l + 1.5)^2 - (l + 0.5)^2
        norm2 += 2.0 * f64::from(l) + 2.0;
        levels[i] = l + 1;
        events.push(i as u32);
        let cos2 = dot * dot / norm2;
        if cos2 > best {
            best = cos2;
            best_events = events.len();
            best_t = f32::from_bits((key >> 32) as u32);
        }
        push(&mut heap, i, l + 1, a[i]);
    }
    for &i in &events[best_events..] {
        levels[i as usize] -= 1;
    }
    best_t / t_top
}

/// The integer dot product of `query` (one byte per padded dimension) with
/// one `width`-bit part (`width` < 8) of a code, as laid out in the module
/// docs.
// ARITH: `width` is 1, 2 or 4, so shifts are below 8; products of two bytes
// summed over at most 2^15 dimensions stay below 2^31.
#[allow(clippy::arithmetic_side_effects)]
fn part_dot(query: &[u8], part: &[u8], width: u32) -> u32 {
    let mask = (1u8 << width) - 1;
    let mut total = 0u32;
    for (k, q) in query.chunks_exact(part.len()).enumerate() {
        let shift = 8 - width * (k as u32 + 1);
        total += q
            .iter()
            .zip(part)
            .map(|(&q, &p)| u32::from(q) * u32::from((p >> shift) & mask))
            .sum::<u32>();
    }
    total
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
        self.packed.saturating_add(CORRECTIONS_BYTES)
    }

    fn uses_centroid(&self) -> bool {
        true
    }

    // ARITH: levels are at most 127 and codes below 256; part offsets and
    // byte indices are below the packed length, dimension indices below the
    // padded dimension.
    #[allow(clippy::arithmetic_side_effects)]
    fn encode(&self, vector: &[f32], centroid: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dimension, "vector dimension");
        assert_eq!(centroid.len(), self.dimension, "centroid dimension");
        assert_eq!(out.len(), self.code_len(), "code length");
        let mut v = vec![0f32; self.dimension];
        prepare_vector(self.similarity, vector, &mut v);
        let norm2 = simd::square_distance_f32(&v, centroid);
        let extra = if self.similarity == Similarity::Euclidean {
            norm2
        } else {
            simd::dot_f32(&v, centroid) - simd::dot_f32(centroid, centroid)
        };
        for (x, &c) in v.iter_mut().zip(centroid) {
            *x -= c;
        }
        let mut rotated = vec![0f32; self.padded];
        self.rotate(&v, &mut rotated);
        let abs: Vec<f32> = rotated.iter().map(|x| x.abs()).collect();
        let half = 1u8 << (self.bits - 1);
        let mut levels = vec![0u8; self.padded];
        best_levels(
            &abs,
            half - 1,
            SEARCH_WINDOW[usize::from(self.bits)],
            &mut levels,
        );
        // `u = half + k` for a non-negative coordinate, `half - 1 - k`
        // otherwise, so `y = u - (2^B - 1) / 2 = +-(k + 1/2)`.
        let mut y_dot_r = 0f64;
        let mut sum_u = 0u32;
        let mut codes = levels;
        for (u, &x) in codes.iter_mut().zip(&rotated) {
            let k = *u;
            let y = f64::from(k) + 0.5;
            if x >= 0.0 {
                *u = half + k;
                y_dot_r += y * f64::from(x);
            } else {
                *u = half - 1 - k;
                y_dot_r -= y * f64::from(x);
            }
            sum_u += u32::from(*u);
        }
        let f = if y_dot_r > 0.0 {
            (f64::from(norm2) / y_dot_r) as f32
        } else {
            0.0
        };
        let (packed, tail) = out.split_at_mut(self.packed);
        self.pack(&codes, packed);
        tail[..4].copy_from_slice(&f.to_le_bytes());
        tail[4..8].copy_from_slice(&(f * sum_u as f32).to_le_bytes());
        tail[8..12].copy_from_slice(&extra.to_le_bytes());
    }

    fn query_scorer(&self, query: &[f32]) -> Box<dyn QueryScorer + '_> {
        assert_eq!(query.len(), self.dimension, "query dimension");
        let mut normalized = vec![0f32; self.dimension];
        prepare_vector(self.similarity, query, &mut normalized);
        let mut rotated = vec![0f32; self.padded];
        self.rotate(&normalized, &mut rotated);
        Box::new(ExRabitqQueryScorer {
            q: self,
            query: normalized,
            rotated,
            scratch: vec![0f32; self.padded],
            bytes: vec![0u8; self.padded.saturating_mul(self.query_bytes())],
            lower: 0.0,
            step: 0.0,
            offset_sum: 0.0,
            query_extra: 0.0,
        })
    }

    fn state(&self) -> Vec<u8> {
        let mut state = vec![self.bits];
        state.extend_from_slice(&self.seed.to_le_bytes());
        state
    }
}

/// A query rotated once and quantized against one centroid at a time.
#[derive(Debug)]
struct ExRabitqQueryScorer<'q> {
    q: &'q ExRabitqQuantizer,
    /// The query, normalized for `Cosine`.
    query: Vec<f32>,
    /// `P q`.
    rotated: Vec<f32>,
    scratch: Vec<f32>,
    /// `z = P (q - c)` quantized to `v_i` of 8 or 16 bits, `z_i ~ lower +
    /// step * v_i`, as rows of bytes, most significant first.
    bytes: Vec<u8>,
    lower: f32,
    step: f32,
    /// `(2^B - 1) / 2 * sum(z)`: the codebook's offset applied to the query.
    offset_sum: f32,
    /// `|q - c|^2` for Euclidean, `<c, q>` otherwise.
    query_extra: f32,
}

impl QueryScorer for ExRabitqQueryScorer<'_> {
    fn set_centroid(&mut self, centroid: &[f32]) {
        assert_eq!(centroid.len(), self.q.dimension, "centroid dimension");
        self.q.rotate(centroid, &mut self.scratch);
        let (mut lo, mut hi, mut sum) = (f32::MAX, -f32::MAX, 0f64);
        for (z, &q) in self.scratch.iter_mut().zip(&self.rotated) {
            *z = q - *z;
            lo = lo.min(*z);
            hi = hi.max(*z);
            sum += f64::from(*z);
        }
        let top = if self.q.query_bytes() == 1 {
            f32::from(u8::MAX)
        } else {
            f32::from(u16::MAX)
        };
        let inv = if hi > lo { top / (hi - lo) } else { 0.0 };
        let (high, low) = self.bytes.split_at_mut(self.q.padded);
        for (i, &z) in self.scratch.iter().enumerate() {
            // In 0..=top by construction; the cast saturates rounding noise.
            let [h, l] = (((z - lo) * inv + 0.5) as u16).to_be_bytes();
            if low.is_empty() {
                high[i] = l;
            } else {
                (high[i], low[i]) = (h, l);
            }
        }
        self.lower = lo;
        self.step = if hi > lo { (hi - lo) / top } else { 0.0 };
        let center = f64::from((1u16 << self.q.bits).saturating_sub(1)) / 2.0;
        self.offset_sum = (center * sum) as f32;
        self.query_extra = if self.q.similarity == Similarity::Euclidean {
            simd::square_distance_f32(&self.query, centroid)
        } else {
            simd::dot_f32(&self.query, centroid)
        };
    }

    fn score(&self, code: &[u8]) -> f32 {
        let (packed, tail) = code.split_at(self.q.packed);
        let mut dot = 0u64;
        for row in self.bytes.chunks_exact(self.q.padded) {
            // Rows are a query value's bytes, most significant first.
            dot = (dot << 8).saturating_add(self.q.packed_dot(row, packed));
        }
        let (f, g, extra) = (f32_at(tail, 0), f32_at(tail, 4), f32_at(tail, 8));
        // <r, q - c> ~ f * <y, z> = f * (<u, z> - offset_sum), with
        // <u, z> ~ lower * sum(u) + step * <u, bytes>.
        let ip = f * (self.step * dot as f32 - self.offset_sum) + self.lower * g;
        match self.q.similarity {
            Similarity::Euclidean => 1.0 / (1.0 + (extra + self.query_extra - 2.0 * ip).max(0.0)),
            // <o, q> = <r, q - c> + <r, c> + <c, q>
            Similarity::MaximumInnerProduct => {
                quantization::scale_max_inner_product_score(ip + extra + self.query_extra)
            }
            Similarity::DotProduct | Similarity::Cosine => {
                ((1.0 + ip + extra + self.query_extra) / 2.0).max(0.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::osq::OsqFactory;
    use crate::QuantizerFactory;
    use lucene_util::quantization::ScalarEncoding;

    const SIMS: [Similarity; 4] = [
        Similarity::Euclidean,
        Similarity::DotProduct,
        Similarity::Cosine,
        Similarity::MaximumInnerProduct,
    ];

    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f32 {
            (splitmix64(&mut self.0) >> 40) as f32 / (1u64 << 24) as f32
        }
        fn gaussian(&mut self) -> f32 {
            let (u, v) = (self.uniform().max(1e-7), self.uniform());
            (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
        }
    }

    /// Gaussian vectors around a common offset; when `skewed`, with
    /// per-dimension scales decaying geometrically and every 17th dimension
    /// ten times larger.
    fn gaussians(n: usize, dim: usize, seed: u64, skewed: bool) -> Vec<Vec<f32>> {
        let mut rng = Rng(seed);
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|d| {
                        let scale = if !skewed {
                            1.0
                        } else if d % 17 == 3 {
                            10.0 * 0.97f32.powi(d as i32)
                        } else {
                            0.97f32.powi(d as i32)
                        };
                        0.5 + scale * rng.gaussian()
                    })
                    .collect()
            })
            .collect()
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    fn mean(data: &[Vec<f32>]) -> Vec<f32> {
        let mut c = vec![0f32; data[0].len()];
        for v in data {
            for (c, x) in c.iter_mut().zip(v) {
                *c += x / data.len() as f32;
            }
        }
        c
    }

    /// The raw similarity (squared distance or inner product) a score maps
    /// back to.
    fn raw(sim: Similarity, score: f32) -> f64 {
        let s = f64::from(score);
        match sim {
            Similarity::Euclidean => 1.0 / s - 1.0,
            Similarity::DotProduct | Similarity::Cosine => 2.0 * s - 1.0,
            Similarity::MaximumInnerProduct => {
                if s < 1.0 {
                    1.0 - 1.0 / s
                } else {
                    s - 1.0
                }
            }
        }
    }

    /// Mean absolute error of the raw estimates, relative to the mean
    /// absolute raw value, of `factory` over `data` and `queries`.
    fn error(
        factory: &dyn QuantizerFactory,
        sim: Similarity,
        data: &[Vec<f32>],
        queries: &[Vec<f32>],
    ) -> f64 {
        let dim = data[0].len();
        let q = factory.train(dim, sim, &[]).unwrap();
        let centroid = mean(data);
        let mut codes = vec![0u8; q.code_len() * data.len()];
        for (v, out) in data.iter().zip(codes.chunks_exact_mut(q.code_len())) {
            q.encode(v, &centroid, out);
        }
        let mut block = Vec::new();
        q.write_block(&codes, &mut block);
        let (mut err, mut scale) = (0f64, 0f64);
        for query in queries {
            let mut s = q.query_scorer(query);
            s.set_centroid(&centroid);
            let mut est = vec![0f32; data.len()];
            s.score_block(&block, &mut est);
            for (v, &e) in data.iter().zip(&est) {
                let exact = raw(sim, sim.score(query, v));
                err += (raw(sim, e) - exact).abs();
                scale += exact.abs();
            }
        }
        err / scale
    }

    fn dataset(sim: Similarity, n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        skewed_or_not(sim, n, dim, seed, true)
    }

    fn skewed_or_not(
        sim: Similarity,
        n: usize,
        dim: usize,
        seed: u64,
        skewed: bool,
    ) -> Vec<Vec<f32>> {
        let data = gaussians(n, dim, seed, skewed);
        if matches!(sim, Similarity::DotProduct | Similarity::Cosine) {
            data.iter().map(|v| normalize(v)).collect()
        } else {
            data
        }
    }

    /// Error falls with every added bit, for every similarity, and is at or
    /// below OSQ's at the bit widths both have, on skewed and isotropic data.
    #[test]
    fn error_falls_with_bits_and_beats_osq() {
        let dim = 128;
        for (sim, skewed) in SIMS.into_iter().flat_map(|s| [(s, true), (s, false)]) {
            let data = skewed_or_not(sim, 300, dim, 7, skewed);
            let queries = skewed_or_not(sim, 8, dim, 99, skewed);
            let errs: Vec<f64> = (2..=8)
                .map(|b| error(&ExRabitqFactory::new(b), sim, &data, &queries))
                .collect();
            for w in errs.windows(2) {
                assert!(w[1] < w[0], "{sim:?} skewed {skewed}: {errs:?}");
            }
            for (bits, enc) in [
                (2, ScalarEncoding::DibitQueryNibble),
                (4, ScalarEncoding::PackedNibble),
                (7, ScalarEncoding::SevenBit),
                (8, ScalarEncoding::UnsignedByte),
            ] {
                let osq = error(&OsqFactory::new(enc), sim, &data, &queries);
                let erq = errs[bits - 2];
                println!(
                    "{sim:?} skewed {skewed} {bits} bits: erq {erq:.5} osq {osq:.5} all {errs:.5?}"
                );
                // Isotropic data at 2 bits ties within a fraction of a percent.
                assert!(erq < osq * 1.01, "{sim:?} {bits} bits: erq {erq} osq {osq}");
            }
        }
    }

    /// Averaged over rotations, the inner-product estimate's error is
    /// centred on zero: its mean is a small fraction of its spread.
    #[test]
    fn estimator_is_unbiased() {
        let dim = 64;
        let mut rng = Rng(5);
        let centroid = vec![0f32; dim];
        let doc: Vec<f32> = (0..dim).map(|_| rng.gaussian()).collect();
        let query: Vec<f32> = (0..dim).map(|_| rng.gaussian()).collect();
        let exact = f64::from(simd::dot_f32(&doc, &query));
        for bits in [2, 3] {
            let (mut sum, mut sq) = (0f64, 0f64);
            let trials = 2000;
            for seed in 0..trials {
                let q = ExRabitqQuantizer::new(dim, Similarity::MaximumInnerProduct, bits, seed)
                    .unwrap();
                let mut code = vec![0u8; q.code_len()];
                q.encode(&doc, &centroid, &mut code);
                let mut s = q.query_scorer(&query);
                s.set_centroid(&centroid);
                let e = raw(Similarity::MaximumInnerProduct, s.score(&code)) - exact;
                sum += e;
                sq += e * e;
            }
            let mean = sum / trials as f64;
            let std = (sq / trials as f64 - mean * mean).sqrt();
            let stderr = std / (trials as f64).sqrt();
            assert!(std > 0.0);
            assert!(
                mean.abs() < 4.0 * stderr,
                "{bits} bits: bias {mean} stderr {stderr}"
            );
        }
    }

    /// Every similarity and odd dimension scores close to the exact score,
    /// including a vector against itself.
    #[test]
    fn all_similarities_and_dimensions() {
        for dim in [1, 7, 64, 100, 300] {
            for sim in SIMS {
                let data = dataset(sim, 40, dim, 3 + dim as u64);
                let queries = dataset(sim, 3, dim, 11);
                let e = error(&ExRabitqFactory::new(6), sim, &data, &queries);
                assert!(e < 0.02, "{sim:?} dim {dim}: {e}");
                let q = ExRabitqFactory::new(4).train(dim, sim, &[]).unwrap();
                assert_eq!((q.name(), q.dimension(), q.similarity()), (NAME, dim, sim));
                assert!(q.uses_centroid());
                let padded = dim.next_power_of_two().max(8);
                assert_eq!(q.code_len(), padded / 2 + CORRECTIONS_BYTES);
            }
        }
    }

    #[test]
    fn dimension_one_is_exact_up_to_query_rounding() {
        let q = ExRabitqQuantizer::new(1, Similarity::Euclidean, 2, 1).unwrap();
        let mut code = vec![0u8; q.code_len()];
        q.encode(&[3.0], &[1.0], &mut code);
        let mut s = q.query_scorer(&[-1.0]);
        s.set_centroid(&[1.0]);
        let d2 = raw(Similarity::Euclidean, s.score(&code));
        assert!((d2 - 16.0).abs() < 0.1, "{d2}");
    }

    #[test]
    fn state_round_trips_and_rejects_garbage() {
        let f = ExRabitqFactory::with_seed(5, 42);
        assert_eq!(f.bits(), 5);
        let q = f.train(10, Similarity::Cosine, &[0.0; 20]).unwrap();
        assert_eq!(q.state().len(), 9);
        let back = ExRabitqFactory::new(3)
            .read(10, Similarity::Cosine, &q.state())
            .unwrap();
        assert_eq!(back.state(), q.state());
        assert_eq!(back.code_len(), q.code_len());
        let v: Vec<f32> = (0..10).map(|i| i as f32 - 4.0).collect();
        let c = vec![0.1f32; 10];
        let (mut a, mut b) = (vec![0u8; q.code_len()], vec![0u8; q.code_len()]);
        q.encode(&v, &c, &mut a);
        back.encode(&v, &c, &mut b);
        assert_eq!(a, b);
        let typed = ExRabitqQuantizer::new(10, Similarity::Cosine, 5, 42).unwrap();
        assert_eq!((typed.bits(), typed.seed()), (5, 42));
        for bad in [
            &[][..],
            &[4, 0, 0][..],
            &[1, 0, 0, 0, 0, 0, 0, 0, 0][..],
            &[9; 9][..],
        ] {
            assert!(matches!(
                f.read(10, Similarity::Cosine, bad),
                Err(Error::CorruptState(_))
            ));
        }
        assert!(f.read(0, Similarity::Cosine, &q.state()).is_err());
        assert!(matches!(
            f.read(MAX_PADDED_DIMENSION + 1, Similarity::Cosine, &q.state()),
            Err(Error::IllegalArgument(_))
        ));
        assert!(f.train(3, Similarity::Cosine, &[0.0; 4]).is_err());
        assert!(f
            .train(MAX_PADDED_DIMENSION + 1, Similarity::Cosine, &[])
            .is_err());
        assert!(ExRabitqQuantizer::new(8, Similarity::Cosine, 1, 0).is_err());
        assert!(ExRabitqQuantizer::new(0, Similarity::Cosine, 4, 0).is_err());
        assert!(ExRabitqQuantizer::new(usize::MAX, Similarity::Cosine, 4, 0).is_err());
    }

    #[test]
    #[should_panic(expected = "bits must be in 2..=8")]
    fn factory_rejects_bits() {
        ExRabitqFactory::new(9);
    }

    /// The rotation keeps norms and inner products.
    #[test]
    fn rotation_is_orthogonal() {
        let q = ExRabitqQuantizer::new(100, Similarity::Euclidean, 4, 3).unwrap();
        let mut rng = Rng(8);
        let a: Vec<f32> = (0..100).map(|_| rng.gaussian()).collect();
        let b: Vec<f32> = (0..100).map(|_| rng.gaussian()).collect();
        let (mut ra, mut rb) = (vec![0f32; 128], vec![0f32; 128]);
        q.rotate(&a, &mut ra);
        q.rotate(&b, &mut rb);
        let (d, rd) = (simd::dot_f32(&a, &b), simd::dot_f32(&ra, &rb));
        assert!((d - rd).abs() < 1e-3, "{d} {rd}");
        let (n, rn) = (simd::dot_f32(&a, &a), simd::dot_f32(&ra, &ra));
        assert!((n - rn).abs() < 1e-3 * n);
    }

    /// The windowed level search finds the codes the full search does, and
    /// their cosine to the vector grows with the bits as the paper reports
    /// (about 0.94 at 2 bits, 0.98 at 3, 0.995 at 4).
    #[test]
    fn search_window_loses_nothing() {
        let mut rng = Rng(21);
        let min_cos = [0.0, 0.0, 0.92, 0.97, 0.99, 0.997, 0.999, 0.9997, 0.9999];
        for bits in 2..=8u8 {
            let max_level = (1u8 << (bits - 1)) - 1;
            for n in [64, 512] {
                for _ in 0..5 {
                    let a: Vec<f32> = (0..n).map(|_| rng.gaussian().abs()).collect();
                    let cos = |levels: &[u8]| {
                        let y: Vec<f32> = levels.iter().map(|&l| l as f32 + 0.5).collect();
                        simd::dot_f32(&y, &a)
                            / (simd::dot_f32(&y, &y) * simd::dot_f32(&a, &a)).sqrt()
                    };
                    let (mut full, mut fast) = (vec![0u8; n], vec![0u8; n]);
                    best_levels(&a, max_level, FULL_WINDOW, &mut full);
                    let t = best_levels(&a, max_level, SEARCH_WINDOW[usize::from(bits)], &mut fast);
                    assert_eq!(full, fast, "{bits} bits n {n} t {t}");
                    assert!(full.iter().all(|&l| l <= max_level));
                    assert!(cos(&full) > min_cos[usize::from(bits)], "{bits} bits n {n}");
                }
            }
        }
        let mut zero = [9u8; 4];
        assert_eq!(best_levels(&[0.0; 4], 3, FULL_WINDOW, &mut zero), 0.0);
        assert_eq!(zero, [0; 4]);
    }

    #[test]
    fn zero_residual_scores_the_centroid() {
        let q = ExRabitqQuantizer::new(16, Similarity::Euclidean, 3, 0).unwrap();
        let c = vec![0.5f32; 16];
        let mut code = vec![0u8; q.code_len()];
        q.encode(&c, &c, &mut code);
        let mut s = q.query_scorer(&c);
        s.set_centroid(&c);
        assert_eq!(s.score(&code), 1.0);
    }

    /// Packed codes of every width score the plain dot product of the
    /// unpacked values.
    #[test]
    fn packed_dot_matches_unpacked() {
        let mut rng = Rng(4);
        for bits in 2..=8u8 {
            let q = ExRabitqQuantizer::new(64, Similarity::Euclidean, bits, 0).unwrap();
            let byte = |rng: &mut Rng| (rng.uniform() * 256.0) as u8;
            let codes: Vec<u8> = (0..64).map(|_| byte(&mut rng) >> (8 - bits)).collect();
            let query: Vec<u8> = (0..64).map(|_| byte(&mut rng)).collect();
            let mut packed = vec![0xffu8; q.packed];
            q.pack(&codes, &mut packed);
            assert_eq!(packed.len(), 64 * usize::from(bits) / 8);
            let want: u64 = codes
                .iter()
                .zip(&query)
                .map(|(&u, &x)| u64::from(u) * u64::from(x))
                .sum();
            assert_eq!(q.packed_dot(&query, &packed), want, "{bits} bits");
        }
        // Width 2: byte j holds dimensions j, j+4, j+8, j+12, high bits first.
        let query: Vec<u8> = (0..16).map(|i| i * 13).collect();
        let part = [0b00_01_10_11u8, 0, 0, 0b11_00_00_00];
        let want = [(4, 1), (8, 2), (12, 3), (3, 3)]
            .iter()
            .map(|&(d, w)| w * u32::from(query[d]))
            .sum::<u32>();
        assert_eq!(part_dot(&query, &part, 2), want);
    }

    #[test]
    #[should_panic(expected = "code length")]
    fn encode_checks_lengths() {
        let q = ExRabitqQuantizer::new(8, Similarity::Euclidean, 4, 0).unwrap();
        q.encode(&[0.0; 8], &[0.0; 8], &mut [0u8; 3]);
    }

    #[test]
    #[should_panic(expected = "centroid dimension")]
    fn set_centroid_checks_length() {
        let q = ExRabitqQuantizer::new(8, Similarity::Euclidean, 4, 0).unwrap();
        q.query_scorer(&[0.0; 8]).set_centroid(&[0.0; 4]);
    }
}
