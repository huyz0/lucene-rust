//! A random orthogonal rotation in front of any quantizer, the step that
//! gives RaBitQ its accuracy: rotated, a vector's energy is spread evenly
//! over its dimensions, so a few bits per dimension lose less of it.
//!
//! [`RotatedFactory`] wraps another [`QuantizerFactory`] and registers as
//! `rot-<inner name>` (`rot-osq`, `rot-sq`). Its quantizer zero-pads every
//! vector it is given (training sample, document, centroid, query) to the
//! inner quantizer's code width ([`QuantizerFactory::code_dimension`]: for
//! OSQ at 4 bits and under, the next multiple of 64) and applies a
//! randomized Hadamard [`Rotation`] at that width. The padding costs no code
//! bytes, and the rotation spreads a vector's energy into it. Every step is
//! orthogonal, so dot products, norms and Euclidean distances, and so every
//! similarity's score, are unchanged. The inner quantizer is trained on,
//! encodes and scores the rotated vectors at the padded dimension; codes,
//! blocks and scores are its own. [`Quantizer::prepare_centroid`] returns
//! the rotated centroid, so an index that keeps it rotates each centroid
//! once rather than per vector encoded or per partition probed.
//!
//! The state is the seed (`u64` LE), the round count (one byte), the
//! rotation width (`u32` LE), then the inner quantizer's state.

use std::sync::Arc;

use crate::{check_sample, Error, Quantizer, QuantizerFactory, QueryScorer, Result, Similarity};

/// The prefix of a rotated factory's registry name: `rot-` then the inner
/// factory's name.
pub const PREFIX: &str = "rot-";

/// The seed new rotations use unless [`RotatedFactory::with_seed`] sets one.
pub const DEFAULT_SEED: u64 = 0x5241_4249_5451_0001;

/// Sign-flip and Hadamard rounds new rotations use unless
/// [`RotatedFactory::with_rounds`] sets a count.
pub const DEFAULT_ROUNDS: u8 = 3;

/// The most rounds a rotation may have.
pub const MAX_ROUNDS: u8 = 8;

/// Bytes of the state before the inner quantizer's.
const HEADER_BYTES: usize = 13;

/// In-place unnormalized fast Walsh-Hadamard transform: `x` becomes `H x`,
/// `H` the Sylvester Hadamard matrix of order `x.len()`, so applying it
/// twice multiplies `x` by its length.
///
/// # Panics
///
/// When the length of `x` is not a power of two.
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    assert!(n.is_power_of_two(), "length {n} is not a power of two");
    if n >= 2 {
        butterflies_of::<1>(x);
    }
    if n >= 4 {
        butterflies_of::<2>(x);
    }
    if n >= 8 {
        butterflies_of::<4>(x);
    }
    let mut h = 8;
    while h < n {
        butterflies(x, h);
        h = h.saturating_mul(2);
    }
}

/// One butterfly stage of half-width `H`, known at compile time so the
/// short stages unroll.
#[inline]
fn butterflies_of<const H: usize>(x: &mut [f32]) {
    butterflies(x, H);
}

/// One butterfly stage: within each run of `2h`, `(a, b)` becomes
/// `(a + b, a - b)` for the pairs `h` apart.
#[inline(always)]
fn butterflies(x: &mut [f32], h: usize) {
    for run in x.chunks_exact_mut(h.saturating_mul(2)) {
        let (a, b) = run.split_at_mut(h);
        for (u, v) in a.iter_mut().zip(b.iter_mut()) {
            let (p, q) = (*u, *v);
            *u = p + q;
            *v = p - q;
        }
    }
}

/// The next value of a SplitMix64 generator.
// ARITH: SplitMix64 is defined with wrapping arithmetic.
#[allow(clippy::arithmetic_side_effects)]
fn split_mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The widest rotation of `dimension`-dimensional vectors: twice the
/// dimension, or 64, whichever is larger. It holds the next power of two
/// and the next multiple of 64.
pub fn max_width(dimension: usize) -> usize {
    dimension.saturating_mul(2).max(64)
}

/// A seeded randomized Hadamard rotation of `dimension`-dimensional vectors,
/// zero-padded, into `width` dimensions.
///
/// With `b` the largest power of two at most `width`, a round flips the
/// signs of a seeded random subset of coordinates, then applies a fast
/// Walsh-Hadamard transform scaled by `1/sqrt(b)` to `[0, b)`. When `width`
/// is not a power of two, a round first permutes the coordinates at random
/// and ends with a second sign flip and scaled transform on the overlapping
/// block `[width - b, width)`, so the two blocks cover every coordinate and,
/// over rounds, every output mixes every input. Every step is orthogonal; a
/// round costs `O(width log width)`. At a power-of-two width a round is the
/// sign flip and one transform.
#[derive(Debug, Clone, PartialEq)]
pub struct Rotation {
    dimension: usize,
    width: usize,
    /// The Hadamard block: the largest power of two at most `width`.
    block: usize,
    seed: u64,
    rounds: u8,
    /// One row of `width` signs per round: `±1/sqrt(block)` on the first
    /// block, so the sign flip also scales the first transform, and `±1`
    /// past it.
    signs: Vec<f32>,
    /// When `width` is not a power of two, one row of `block` signs per
    /// round, each `±1/sqrt(block)`, for the second block; else empty.
    last_signs: Vec<f32>,
    /// When `width` is not a power of two, one permutation of `0..width`
    /// per round, else empty: coordinate `i` takes coordinate `perms[i]`.
    perms: Vec<u32>,
}

impl Rotation {
    /// The rotation of `dimension`-dimensional vectors into `width`
    /// dimensions drawn from `seed`, with `rounds` rounds.
    ///
    /// # Errors
    ///
    /// A zero dimension, a `width` below the dimension or above
    /// [`max_width`] or `u32::MAX`, or `rounds` outside `1..=MAX_ROUNDS`.
    pub fn new(dimension: usize, width: usize, seed: u64, rounds: u8) -> Result<Self> {
        if dimension == 0 {
            return Err(Error::IllegalArgument("dimension must be positive".into()));
        }
        if !(1..=MAX_ROUNDS).contains(&rounds) {
            return Err(Error::IllegalArgument(format!(
                "rotation rounds must be 1..={MAX_ROUNDS}, got {rounds}"
            )));
        }
        if width < dimension || width > max_width(dimension) || u32::try_from(width).is_err() {
            return Err(Error::IllegalArgument(format!(
                "rotation width {width} must be in {dimension}..={}",
                max_width(dimension)
            )));
        }
        // `width` is positive, so its highest set bit is a power of two.
        let block = 1usize << width.ilog2();
        let rounds_len = usize::from(rounds);
        let scale = inverse_sqrt(block);
        let mut state = seed;
        let signs = draw_signs(width.saturating_mul(rounds_len), &mut state, |i| {
            if i.checked_rem(width).unwrap_or(0) < block {
                scale
            } else {
                1.0
            }
        });
        let (mut last_signs, mut perms) = (Vec::new(), Vec::new());
        if block < width {
            last_signs = draw_signs(block.saturating_mul(rounds_len), &mut state, |_| scale);
            perms.reserve_exact(width.saturating_mul(rounds_len));
            for _ in 0..rounds {
                let at = perms.len();
                // `width` fits in `u32`, checked above.
                perms.extend((0..width).map(|i| i as u32));
                shuffle(&mut perms[at..], &mut state);
            }
        }
        Ok(Rotation {
            dimension,
            width,
            block,
            seed,
            rounds,
            signs,
            last_signs,
            perms,
        })
    }

    /// The dimension of the vectors rotated.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The dimension of a rotated vector.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The seed the signs and permutations were drawn from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Rounds of the rotation.
    pub fn rounds(&self) -> u8 {
        self.rounds
    }

    /// Rotates `x` (of [`Rotation::dimension`] floats), zero-padded, into
    /// `out` (of [`Rotation::width`] floats).
    ///
    /// # Panics
    ///
    /// On a wrong-length `x` or `out`.
    pub fn rotate(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.dimension, "vector dimension");
        assert_eq!(out.len(), self.width, "rotated dimension");
        let (head, tail) = out.split_at_mut(self.dimension);
        head.copy_from_slice(x);
        tail.fill(0.0);
        if self.perms.is_empty() {
            for signs in self.signs.chunks_exact(self.width) {
                for (o, s) in out.iter_mut().zip(signs) {
                    *o *= s;
                }
                fwht(out);
            }
            return;
        }
        let second = self.width.saturating_sub(self.block);
        let mut before = vec![0f32; self.width];
        for ((signs, last_signs), perm) in self
            .signs
            .chunks_exact(self.width)
            .zip(self.last_signs.chunks_exact(self.block))
            .zip(self.perms.chunks_exact(self.width))
        {
            before.copy_from_slice(out);
            for ((o, s), &p) in out.iter_mut().zip(signs).zip(perm) {
                *o = before[p as usize] * s;
            }
            fwht(&mut out[..self.block]);
            let last = &mut out[second..];
            for (o, s) in last.iter_mut().zip(last_signs) {
                *o *= s;
            }
            fwht(last);
        }
    }
}

/// `n` random signs drawn from `state`, sign `i` of magnitude
/// `magnitude(i)`.
fn draw_signs(n: usize, state: &mut u64, magnitude: impl Fn(usize) -> f32) -> Vec<f32> {
    let mut bits = 0u64;
    (0..n)
        .map(|i| {
            if i.is_multiple_of(64) {
                bits = split_mix(state);
            }
            let negative = bits & 1 == 1;
            bits >>= 1;
            if negative {
                -magnitude(i)
            } else {
                magnitude(i)
            }
        })
        .collect()
}

/// `1 / sqrt(n)` for a power of two `n`: exact, or the nearest float to an
/// exact power of two's root.
fn inverse_sqrt(n: usize) -> f32 {
    (1.0 / (n as f64).sqrt()) as f32
}

/// Shuffles `x` uniformly (Fisher-Yates) with values drawn from `state`.
fn shuffle(x: &mut [u32], state: &mut u64) {
    for i in (1..x.len()).rev() {
        // ARITH: `i + 1` is at most `x.len()`; a 64-bit draw times a length
        // below 2^33 fits in 128 bits, and its top 64 bits are below `i + 1`.
        #[allow(clippy::arithmetic_side_effects)]
        let j = ((u128::from(split_mix(state)) * (i as u128 + 1)) >> 64) as usize;
        x.swap(i, j);
    }
}

/// Builds [`RotatedQuantizer`]s: a [`Rotation`] in front of the quantizers
/// an inner factory builds. Registered as `rot-<inner name>`.
#[derive(Debug, Clone)]
pub struct RotatedFactory {
    inner: Arc<dyn QuantizerFactory>,
    name: String,
    seed: u64,
    rounds: u8,
}

impl RotatedFactory {
    /// Rotates the vectors of the quantizers `inner` builds, with
    /// [`DEFAULT_SEED`] and [`DEFAULT_ROUNDS`].
    pub fn new(inner: Arc<dyn QuantizerFactory>) -> Self {
        let name = format!("{PREFIX}{}", inner.name());
        RotatedFactory {
            inner,
            name,
            seed: DEFAULT_SEED,
            rounds: DEFAULT_ROUNDS,
        }
    }

    /// The seed newly trained quantizers draw their rotation from.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Rounds of newly trained quantizers' rotations, `1..=MAX_ROUNDS`
    /// (checked when training).
    pub fn with_rounds(mut self, rounds: u8) -> Self {
        self.rounds = rounds;
        self
    }

    /// The wrapped factory.
    pub fn inner(&self) -> &Arc<dyn QuantizerFactory> {
        &self.inner
    }
}

impl QuantizerFactory for RotatedFactory {
    fn name(&self) -> &str {
        &self.name
    }

    /// The inner factory's code dimension: rotated vectors are padded to it.
    fn code_dimension(&self, dimension: usize) -> usize {
        self.inner.code_dimension(dimension)
    }

    fn train(
        &self,
        dimension: usize,
        similarity: Similarity,
        sample: &[f32],
    ) -> Result<Box<dyn Quantizer>> {
        check_sample(dimension, sample)?;
        let rotation = Rotation::new(
            dimension,
            self.inner.code_dimension(dimension),
            self.seed,
            self.rounds,
        )?;
        let padded = rotation.width();
        let mut rotated = vec![
            0f32;
            sample
                .len()
                .checked_div(dimension)
                .unwrap_or(0)
                .saturating_mul(padded)
        ];
        for (v, out) in sample
            .chunks_exact(dimension)
            .zip(rotated.chunks_exact_mut(padded))
        {
            rotation.rotate(v, out);
        }
        let inner = self.inner.train(padded, similarity, &rotated)?;
        Ok(Box::new(RotatedQuantizer::new(
            self.name.clone(),
            rotation,
            inner,
        )))
    }

    fn read(
        &self,
        dimension: usize,
        similarity: Similarity,
        state: &[u8],
    ) -> Result<Box<dyn Quantizer>> {
        let Some((header, inner_state)) = state.split_first_chunk::<HEADER_BYTES>() else {
            return Err(Error::CorruptState(format!(
                "rotation state needs {HEADER_BYTES} bytes, got {}",
                state.len()
            )));
        };
        let (seed, rest) = header.split_at(8);
        let (rounds, width) = rest.split_at(1);
        let mut seed_bytes = [0u8; 8];
        seed_bytes.copy_from_slice(seed);
        let mut width_bytes = [0u8; 4];
        width_bytes.copy_from_slice(width);
        // A width that does not fit is out of range, and rejected below.
        let width = usize::try_from(u32::from_le_bytes(width_bytes)).unwrap_or(usize::MAX);
        let rotation = Rotation::new(dimension, width, u64::from_le_bytes(seed_bytes), rounds[0])
            .map_err(|e| Error::CorruptState(e.to_string()))?;
        let inner = self.inner.read(width, similarity, inner_state)?;
        Ok(Box::new(RotatedQuantizer::new(
            self.name.clone(),
            rotation,
            inner,
        )))
    }
}

/// A quantizer of rotated vectors: rotates what it is given and hands the
/// result to the inner quantizer.
#[derive(Debug)]
pub struct RotatedQuantizer {
    name: String,
    rotation: Rotation,
    inner: Box<dyn Quantizer>,
    /// The padded dimension's zero vector: the centroid handed to an inner
    /// quantizer that ignores centroids.
    zeros: Vec<f32>,
}

impl RotatedQuantizer {
    /// `inner`, a quantizer of [`Rotation::width`] floats, behind
    /// `rotation`; `name` is the factory's registry name.
    ///
    /// # Panics
    ///
    /// When `inner`'s dimension is not the rotation's padded dimension.
    pub fn new(name: String, rotation: Rotation, inner: Box<dyn Quantizer>) -> Self {
        assert_eq!(
            inner.dimension(),
            rotation.width(),
            "inner quantizer dimension"
        );
        RotatedQuantizer {
            name,
            zeros: vec![0f32; rotation.width()],
            rotation,
            inner,
        }
    }

    /// The rotation.
    pub fn rotation(&self) -> &Rotation {
        &self.rotation
    }

    /// The quantizer of the rotated vectors.
    pub fn inner(&self) -> &dyn Quantizer {
        self.inner.as_ref()
    }

    /// What the inner quantizer gets for a prepared centroid: `prepared`
    /// itself, or zeros when it ignores centroids.
    fn inner_centroid<'a>(&'a self, prepared: &'a [f32]) -> &'a [f32] {
        if self.inner.uses_centroid() {
            prepared
        } else {
            &self.zeros
        }
    }
}

impl Quantizer for RotatedQuantizer {
    fn name(&self) -> &str {
        &self.name
    }

    fn dimension(&self) -> usize {
        self.rotation.dimension()
    }

    fn similarity(&self) -> Similarity {
        self.inner.similarity()
    }

    fn code_len(&self) -> usize {
        self.inner.code_len()
    }

    fn uses_centroid(&self) -> bool {
        self.inner.uses_centroid()
    }

    /// The rotated centroid, prepared in turn by the inner quantizer; `None`
    /// when the inner quantizer ignores centroids.
    fn prepare_centroid(&self, centroid: &[f32]) -> Option<Vec<f32>> {
        assert_eq!(centroid.len(), self.dimension(), "centroid dimension");
        if !self.inner.uses_centroid() {
            return None;
        }
        let mut rotated = vec![0f32; self.rotation.width()];
        self.rotation.rotate(centroid, &mut rotated);
        Some(self.inner.prepare_centroid(&rotated).unwrap_or(rotated))
    }

    fn encode(&self, vector: &[f32], centroid: &[f32], out: &mut [u8]) {
        match self.prepare_centroid(centroid) {
            Some(prepared) => self.encode_prepared(vector, &prepared, out),
            None => self.encode_prepared(vector, centroid, out),
        }
    }

    fn encode_prepared(&self, vector: &[f32], prepared: &[f32], out: &mut [u8]) {
        let mut rotated = vec![0f32; self.rotation.width()];
        self.rotation.rotate(vector, &mut rotated);
        self.inner
            .encode_prepared(&rotated, self.inner_centroid(prepared), out);
    }

    fn query_scorer(&self, query: &[f32]) -> Box<dyn QueryScorer + '_> {
        let mut rotated = vec![0f32; self.rotation.width()];
        self.rotation.rotate(query, &mut rotated);
        Box::new(RotatedQueryScorer {
            q: self,
            inner: self.inner.query_scorer(&rotated),
        })
    }

    fn state(&self) -> Vec<u8> {
        let inner = self.inner.state();
        let mut state = Vec::with_capacity(HEADER_BYTES.saturating_add(inner.len()));
        state.extend_from_slice(&self.rotation.seed().to_le_bytes());
        state.push(self.rotation.rounds());
        // `Rotation::new` checks that the width fits in `u32`.
        state.extend_from_slice(&(self.rotation.width() as u32).to_le_bytes());
        state.extend_from_slice(&inner);
        state
    }

    fn block_len(&self, n: usize) -> usize {
        self.inner.block_len(n)
    }

    fn write_block(&self, codes: &[u8], out: &mut Vec<u8>) {
        self.inner.write_block(codes, out);
    }
}

/// A rotated query in front of the inner quantizer's scorer.
struct RotatedQueryScorer<'q> {
    q: &'q RotatedQuantizer,
    inner: Box<dyn QueryScorer + 'q>,
}

impl QueryScorer for RotatedQueryScorer<'_> {
    fn set_centroid(&mut self, centroid: &[f32]) {
        match self.q.prepare_centroid(centroid) {
            Some(prepared) => self.set_prepared_centroid(&prepared),
            None => self.set_prepared_centroid(centroid),
        }
    }

    fn set_prepared_centroid(&mut self, prepared: &[f32]) {
        self.inner
            .set_prepared_centroid(self.q.inner_centroid(prepared));
    }

    fn score(&self, code: &[u8]) -> f32 {
        self.inner.score(code)
    }

    fn score_block(&self, codes: &[u8], scores: &mut [f32]) {
        self.inner.score_block(codes, scores);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::osq::OsqFactory;
    use crate::scalar::ScalarFactory;
    use lucene_util::quantization::ScalarEncoding;

    const SIMS: [Similarity; 4] = [
        Similarity::Euclidean,
        Similarity::DotProduct,
        Similarity::Cosine,
        Similarity::MaximumInnerProduct,
    ];

    fn vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n * dim)
            .map(|_| ((split_mix(&mut s) >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
            .collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
    }

    fn l2(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
            .sum()
    }

    fn rot_bbq() -> RotatedFactory {
        RotatedFactory::new(Arc::new(OsqFactory::bbq()))
    }

    #[test]
    fn fwht_matches_the_hadamard_matrix() {
        let mut x = [1.0, 0.0, 0.0, 0.0];
        fwht(&mut x);
        assert_eq!(x, [1.0; 4]);
        let mut x = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        fwht(&mut x);
        assert_eq!(x, [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let mut one = [3.0];
        fwht(&mut one);
        assert_eq!(one, [3.0]);
        for n in [2, 16, 64, 256] {
            let orig = vectors(1, n, n as u64);
            let mut x = orig.clone();
            fwht(&mut x);
            fwht(&mut x);
            for (a, b) in x.iter().zip(&orig) {
                assert!((a / n as f32 - b).abs() < 1e-5, "n {n}");
            }
            // Row i of H is (-1)^popcount(i & j).
            let mut e = vec![0f32; n];
            e[3 % n] = 1.0;
            fwht(&mut e);
            for (j, v) in e.iter().enumerate() {
                let sign = if ((3 % n) & j).count_ones() % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                assert_eq!(*v, sign);
            }
        }
    }

    #[test]
    #[should_panic(expected = "not a power of two")]
    fn fwht_rejects_other_lengths() {
        fwht(&mut [0.0; 6]);
    }

    /// The `(dimension, width)` pairs the rotation tests cover: each
    /// dimension at itself, at the next multiple of 64 and at the next
    /// power of two.
    fn widths() -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for dim in [1usize, 7, 64, 100, 300, 768] {
            for width in [dim, dim.next_multiple_of(64), dim.next_power_of_two()] {
                if !pairs.contains(&(dim, width)) {
                    pairs.push((dim, width));
                }
            }
        }
        pairs
    }

    #[test]
    fn rotation_is_orthogonal() {
        for (dim, width) in widths() {
            for rounds in [1, 2, 3, MAX_ROUNDS] {
                let r = Rotation::new(dim, width, 42, rounds).unwrap();
                assert_eq!((r.dimension(), r.width()), (dim, width));
                let v = vectors(2, dim, dim as u64 + 1);
                let (a, b) = v.split_at(dim);
                let mut ra = vec![0f32; width];
                let mut rb = vec![0f32; width];
                r.rotate(a, &mut ra);
                r.rotate(b, &mut rb);
                let tol = |x: f64| 1e-5 * x.abs().max(1.0);
                let at = format!("{dim} -> {width}, {rounds} rounds");
                assert!((dot(&ra, &rb) - dot(a, b)).abs() < tol(dot(a, a)), "{at}");
                assert!((dot(&ra, &ra) - dot(a, a)).abs() < tol(dot(a, a)), "{at}");
                assert!((l2(&ra, &rb) - l2(a, b)).abs() < tol(l2(a, b)), "{at}");
            }
        }
    }

    /// The largest `|coordinate| * sqrt(width)` of the rotated unit axes of
    /// a three-round rotation: 1 when an axis is spread evenly over every
    /// coordinate, `sqrt(width)` when it is not moved.
    fn axis_peak(dim: usize, width: usize, seed: u64) -> f32 {
        let r = Rotation::new(dim, width, seed, 3).unwrap();
        let mut peak = 0f32;
        let mut e = vec![0f32; dim];
        let mut out = vec![0f32; width];
        for i in 0..dim {
            e.fill(0.0);
            e[i] = 1.0;
            r.rotate(&e, &mut out);
            peak = out.iter().fold(peak, |m, x| m.max(x.abs()));
        }
        peak * (width as f32).sqrt()
    }

    /// Three rounds spread every unit axis over all coordinates about as a
    /// random rotation does: the largest coordinate is a few times
    /// `1/sqrt(width)` (measured: at most 5.1 from 64 dimensions up, at
    /// power-of-two widths and others alike), not the `0.5` a transform
    /// over one block alone leaves.
    #[test]
    fn rotation_spreads_every_axis() {
        for (dim, width) in widths() {
            for seed in 0..3 {
                let peak = axis_peak(dim, width, seed);
                assert!(peak < 6.0, "{dim} -> {width}, seed {seed}: {peak}");
            }
        }
    }

    #[test]
    fn rotation_mixes_and_is_deterministic_per_seed() {
        for (dim, width) in [(100, 128), (300, 320)] {
            let a = Rotation::new(dim, width, 7, 3).unwrap();
            let b = Rotation::new(dim, width, 7, 3).unwrap();
            let c = Rotation::new(dim, width, 8, 3).unwrap();
            assert_eq!(a, b);
            assert_ne!(a, c);
            assert_eq!((a.seed(), a.rounds()), (7, 3));
            let mut e = vec![0f32; dim];
            e[0] = 1.0;
            let (mut ra, mut rb, mut rc) =
                (vec![0f32; width], vec![0f32; width], vec![0f32; width]);
            a.rotate(&e, &mut ra);
            b.rotate(&e, &mut rb);
            c.rotate(&e, &mut rc);
            assert_eq!(ra, rb);
            assert_ne!(ra, rc);
        }
    }

    #[test]
    fn rotation_rejects_bad_parameters() {
        assert!(Rotation::new(0, 1, 1, 3).is_err());
        assert!(Rotation::new(4, 4, 1, 0).is_err());
        assert!(Rotation::new(4, 4, 1, MAX_ROUNDS + 1).is_err());
        assert!(Rotation::new(4, 3, 1, 3).is_err());
        assert!(Rotation::new(4, 65, 1, 3).is_err());
        assert!(Rotation::new(100, 201, 1, 3).is_err());
        assert!(Rotation::new(100, 200, 1, 1).is_ok());
        assert_eq!(max_width(1), 64);
        assert_eq!(max_width(100), 200);
        assert!(Rotation::new(usize::MAX, usize::MAX, 1, 3).is_err());
        assert!(Rotation::new(1 << 33, 1 << 33, 1, 3).is_err());
    }

    #[test]
    #[should_panic(expected = "vector dimension")]
    fn rotate_checks_length() {
        Rotation::new(5, 8, 1, 1)
            .unwrap()
            .rotate(&[0.0; 4], &mut [0.0; 8]);
    }

    #[test]
    #[should_panic(expected = "rotated dimension")]
    fn rotate_checks_width() {
        Rotation::new(5, 6, 1, 1)
            .unwrap()
            .rotate(&[0.0; 5], &mut [0.0; 8]);
    }

    #[test]
    fn factory_names_and_forwarding() {
        let f = rot_bbq().with_seed(9).with_rounds(2);
        assert_eq!(f.name(), "rot-osq");
        assert_eq!(f.inner().name(), "osq");
        assert_eq!(
            RotatedFactory::new(Arc::new(ScalarFactory::int7())).name(),
            "rot-sq"
        );
        let dim = 100;
        let sample = vectors(50, dim, 3);
        let q = f.train(dim, Similarity::Euclidean, &sample).unwrap();
        assert_eq!(q.name(), "rot-osq");
        assert_eq!(q.dimension(), dim);
        assert_eq!(q.similarity(), Similarity::Euclidean);
        assert!(q.uses_centroid());
        let plain = OsqFactory::bbq()
            .train(128, Similarity::Euclidean, &[])
            .unwrap();
        assert_eq!(q.code_len(), plain.code_len());
        assert_eq!(q.block_len(40), plain.block_len(40));
        assert!(f.train(dim, Similarity::Euclidean, &sample[1..]).is_err());
        assert!(rot_bbq()
            .with_rounds(0)
            .train(dim, Similarity::Euclidean, &[])
            .is_err());
    }

    /// Rotating into the inner quantizer's code dimension costs no code
    /// bytes: rotated BBQ at 300 dimensions has BBQ's 320-bit codes, not
    /// 512 bits, and so for every inner quantizer.
    #[test]
    fn rotated_codes_are_as_long_as_the_inner_codes() {
        let inners: Vec<Arc<dyn QuantizerFactory>> = vec![
            Arc::new(OsqFactory::bbq()),
            Arc::new(OsqFactory::new(ScalarEncoding::DibitQueryNibble)),
            Arc::new(OsqFactory::new(ScalarEncoding::PackedNibble)),
            Arc::new(OsqFactory::new(ScalarEncoding::SevenBit)),
            Arc::new(OsqFactory::new(ScalarEncoding::UnsignedByte)),
            Arc::new(ScalarFactory::int4()),
            Arc::new(ScalarFactory::int7()),
            Arc::new(crate::rabitq::ExRabitqFactory::new(2)),
        ];
        for inner in inners {
            let rot = RotatedFactory::new(Arc::clone(&inner));
            for dim in [1, 7, 100, 300, 768] {
                let sample = vectors(20, dim, 51);
                let plain = inner.train(dim, Similarity::Euclidean, &sample).unwrap();
                let q = rot.train(dim, Similarity::Euclidean, &sample).unwrap();
                let at = format!("{} at {dim}", q.name());
                assert_eq!(rot.code_dimension(dim), inner.code_dimension(dim), "{at}");
                assert_eq!(q.code_len(), plain.code_len(), "{at}");
                assert_eq!(q.block_len(33), plain.block_len(33), "{at}");
                let state = q.state();
                assert_eq!(
                    state[9..13],
                    (inner.code_dimension(dim) as u32).to_le_bytes(),
                    "{at}"
                );
            }
        }
        let bbq = OsqFactory::bbq();
        assert_eq!(bbq.code_dimension(300), 320);
        assert_eq!(rot_bbq().code_dimension(300), 320);
        assert_eq!(
            rot_bbq()
                .train(300, Similarity::Euclidean, &[])
                .unwrap()
                .code_len(),
            320 / 8 + 16
        );
        assert_eq!(ScalarFactory::int4().code_dimension(7), 8);
        assert_eq!(ScalarFactory::int7().code_dimension(7), 7);
    }

    #[test]
    fn state_round_trips_and_rejects_garbage() {
        let dim = 50;
        let centroid = vectors(1, dim, 5);
        let docs = vectors(40, dim, 6);
        let query = vectors(1, dim, 7);
        let factories = [
            rot_bbq().with_seed(1234).with_rounds(2),
            RotatedFactory::new(Arc::new(ScalarFactory::int4())),
            RotatedFactory::new(Arc::new(OsqFactory::new(ScalarEncoding::DibitQueryNibble))),
        ];
        for f in &factories {
            let q = f.train(dim, Similarity::Cosine, &docs).unwrap();
            let state = q.state();
            let back = f.read(dim, Similarity::Cosine, &state).unwrap();
            assert_eq!(back.state(), state);
            assert_eq!(back.name(), q.name());
            let (mut a, mut b) = (vec![0u8; q.code_len()], vec![0u8; q.code_len()]);
            for d in docs.chunks_exact(dim) {
                q.encode(d, &centroid, &mut a);
                back.encode(d, &centroid, &mut b);
                assert_eq!(a, b);
                let mut s = q.query_scorer(&query);
                s.set_centroid(&centroid);
                let mut t = back.query_scorer(&query);
                t.set_centroid(&centroid);
                assert_eq!(s.score(&a).to_bits(), t.score(&b).to_bits());
            }
        }
        let f = &factories[0];
        let state = f.train(dim, Similarity::Cosine, &[]).unwrap().state();
        assert_eq!(&state[..8], &1234u64.to_le_bytes());
        assert_eq!(state[8], 2);
        // A factory's own seed does not matter when reading.
        let other = rot_bbq().with_seed(1);
        assert_eq!(
            other.read(dim, Similarity::Cosine, &state).unwrap().state(),
            state
        );
        assert_eq!(state[9..13], 64u32.to_le_bytes());
        for bad in [&state[..3], &state[..9], &state[..13]] {
            assert!(matches!(
                f.read(dim, Similarity::Cosine, bad),
                Err(Error::CorruptState(_))
            ));
        }
        let mut zero_rounds = state.clone();
        zero_rounds[8] = 0;
        assert!(matches!(
            f.read(dim, Similarity::Cosine, &zero_rounds),
            Err(Error::CorruptState(_))
        ));
        for width in [49u32, 101, u32::MAX] {
            let mut bad_width = state.clone();
            bad_width[9..13].copy_from_slice(&width.to_le_bytes());
            assert!(matches!(
                f.read(dim, Similarity::Cosine, &bad_width),
                Err(Error::CorruptState(_))
            ));
        }
        assert!(matches!(
            f.read(0, Similarity::Cosine, &state),
            Err(Error::CorruptState(_))
        ));
    }

    /// Rotating changes nothing a score depends on: the rotated quantizer's
    /// estimates of every similarity track exact scores of the original
    /// vectors, and blocks score like single codes.
    #[test]
    fn rotated_estimates_track_exact_scores() {
        let dim = 100;
        let n = 70;
        for sim in SIMS {
            let mut data = vectors(n, dim, 11);
            let mut queries = vectors(4, dim, 12);
            if sim == Similarity::DotProduct {
                for v in data
                    .chunks_exact_mut(dim)
                    .chain(queries.chunks_exact_mut(dim))
                {
                    lucene_util::vector_util::l2normalize(v, false).unwrap();
                }
            }
            let mut centroid = vec![0f32; dim];
            for v in data.chunks_exact(dim) {
                for (c, x) in centroid.iter_mut().zip(v) {
                    *c += x / n as f32;
                }
            }
            for inner in [
                Arc::new(OsqFactory::new(ScalarEncoding::PackedNibble))
                    as Arc<dyn QuantizerFactory>,
                Arc::new(OsqFactory::new(ScalarEncoding::DibitQueryNibble)),
                Arc::new(ScalarFactory::int7()),
            ] {
                let q = RotatedFactory::new(inner).train(dim, sim, &data).unwrap();
                let mut codes = vec![0u8; n * q.code_len()];
                for (v, c) in data
                    .chunks_exact(dim)
                    .zip(codes.chunks_exact_mut(q.code_len()))
                {
                    q.encode(v, &centroid, c);
                }
                let mut block = Vec::new();
                q.write_block(&codes, &mut block);
                assert_eq!(block.len(), q.block_len(n));
                let mut err = 0f64;
                for query in queries.chunks_exact(dim) {
                    let mut s = q.query_scorer(query);
                    s.set_centroid(&centroid);
                    let mut est = vec![0f32; n];
                    s.score_block(&block, &mut est);
                    for ((v, e), c) in data
                        .chunks_exact(dim)
                        .zip(&est)
                        .zip(codes.chunks_exact(q.code_len()))
                    {
                        assert_eq!(e.to_bits(), s.score(c).to_bits());
                        let exact = sim.score(query, v);
                        err += ((exact - e) / exact.max(1e-3)).abs() as f64;
                    }
                }
                let mean = err / (4 * n) as f64;
                assert!(mean < 0.25, "{sim:?} {}: {mean}", q.name());
            }
        }
    }

    /// Mean absolute estimate error of BBQ behind a rotation of `rounds`,
    /// over four seeds, on data whose variance sits in a few dimensions.
    fn skewed_bbq_error(dim: usize, sim: Similarity, rounds: u8) -> f64 {
        let n = 200;
        let mut total = 0f64;
        for seed in 0..4u64 {
            let mut data = vectors(n, dim, 100 + seed);
            let mut queries = vectors(5, dim, 200 + seed);
            for v in data
                .chunks_exact_mut(dim)
                .chain(queries.chunks_exact_mut(dim))
            {
                for (i, x) in v.iter_mut().enumerate() {
                    *x *= (-(i as f32) / 6.0).exp() * 4.0 + 0.05;
                }
            }
            let mut centroid = vec![0f32; dim];
            for v in data.chunks_exact(dim) {
                for (c, x) in centroid.iter_mut().zip(v) {
                    *c += x / n as f32;
                }
            }
            let q = rot_bbq()
                .with_seed(seed * 7 + 1)
                .with_rounds(rounds)
                .train(dim, sim, &data)
                .unwrap();
            let prepared = q.prepare_centroid(&centroid).unwrap();
            let mut code = vec![0u8; q.code_len()];
            for query in queries.chunks_exact(dim) {
                let mut s = q.query_scorer(query);
                s.set_prepared_centroid(&prepared);
                for v in data.chunks_exact(dim) {
                    q.encode_prepared(v, &prepared, &mut code);
                    total += (sim.score(query, v) - s.score(&code)).abs() as f64;
                }
            }
        }
        total / (4 * 5 * n) as f64
    }

    /// A second round clearly lowers the estimate error and a third lowers
    /// it a little further (a few percent), so three is the default: with
    /// centroids prepared once, a round costs one transform per vector
    /// encoded and per query.
    #[test]
    fn rounds_lower_the_estimate_error() {
        for sim in [Similarity::Euclidean, Similarity::MaximumInnerProduct] {
            let [one, two, three] = [1, 2, 3].map(|r| skewed_bbq_error(128, sim, r));
            assert!(two < 0.75 * one, "{sim:?}: one {one} two {two}");
            assert!(three < two, "{sim:?}: two {two} three {three}");
        }
    }

    /// On data whose variance sits in a few dimensions, rotated BBQ
    /// estimates scores with clearly less error than plain BBQ.
    #[test]
    fn rotation_cuts_bbq_error_on_skewed_data() {
        let n = 300;
        for dim in [64, 100, 300] {
            let mut data = vectors(n, dim, 21);
            let mut queries = vectors(10, dim, 22);
            for v in data
                .chunks_exact_mut(dim)
                .chain(queries.chunks_exact_mut(dim))
            {
                for (i, x) in v.iter_mut().enumerate() {
                    *x *= (-(i as f32) / 6.0).exp() * 4.0 + 0.05;
                }
            }
            let mut centroid = vec![0f32; dim];
            for v in data.chunks_exact(dim) {
                for (c, x) in centroid.iter_mut().zip(v) {
                    *c += x / n as f32;
                }
            }
            for sim in [Similarity::Euclidean, Similarity::MaximumInnerProduct] {
                let error = |q: &dyn Quantizer| {
                    let mut code = vec![0u8; q.code_len()];
                    let mut err = 0f64;
                    for query in queries.chunks_exact(dim) {
                        let mut s = q.query_scorer(query);
                        s.set_centroid(&centroid);
                        for v in data.chunks_exact(dim) {
                            q.encode(v, &centroid, &mut code);
                            err += (sim.score(query, v) - s.score(&code)).abs() as f64;
                        }
                    }
                    err / (queries.len() / dim * n) as f64
                };
                let plain = OsqFactory::bbq().train(dim, sim, &data).unwrap();
                let rotated = rot_bbq().train(dim, sim, &data).unwrap();
                let (e_plain, e_rot) = (error(plain.as_ref()), error(rotated.as_ref()));
                // Measured: about a third of plain BBQ's error.
                assert!(
                    e_rot < 0.6 * e_plain,
                    "dim {dim} {sim:?}: rotated {e_rot} plain {e_plain}"
                );
            }
        }
    }

    /// Encoding is the inner quantizer's encoding of the rotated vector
    /// against the rotated centroid, and a prepared centroid encodes and
    /// scores as the centroid does, nested rotations included.
    #[test]
    fn encoding_and_scoring_against_prepared_centroids() {
        let dim = 30;
        let docs = vectors(6, dim, 31);
        let centroids = vectors(2, dim, 32);
        let a = rot_bbq()
            .with_seed(1)
            .train(dim, Similarity::Euclidean, &[])
            .unwrap();
        let b = rot_bbq()
            .with_seed(2)
            .train(dim, Similarity::Euclidean, &[])
            .unwrap();
        let nested = RotatedFactory::new(Arc::new(rot_bbq()));
        assert_eq!(nested.name(), "rot-rot-osq");
        let c = nested.train(dim, Similarity::Euclidean, &docs).unwrap();
        let c_back = nested.read(dim, Similarity::Euclidean, &c.state()).unwrap();
        let expected = |q: &dyn Quantizer, d: &[f32], centroid: &[f32]| {
            let mut padded = vec![0f32; 64];
            let rq = q.state();
            assert_eq!(rq[9..13], 64u32.to_le_bytes());
            let rotation = Rotation::new(
                dim,
                64,
                u64::from_le_bytes(rq[..8].try_into().unwrap()),
                rq[8],
            )
            .unwrap();
            let inner = OsqFactory::bbq()
                .train(64, Similarity::Euclidean, &[])
                .unwrap();
            let mut pc = vec![0f32; 64];
            rotation.rotate(d, &mut padded);
            rotation.rotate(centroid, &mut pc);
            let mut code = vec![0u8; inner.code_len()];
            inner.encode(&padded, &pc, &mut code);
            code
        };
        let mut code = vec![0u8; a.code_len()];
        let mut code_back = vec![0u8; c.code_len()];
        for (i, d) in docs.chunks_exact(dim).enumerate() {
            let centroid = &centroids[(i % 2) * dim..][..dim];
            for q in [&a, &b, &c] {
                q.encode(d, centroid, &mut code);
                if !std::ptr::eq(q, &c) {
                    assert_eq!(code, expected(q.as_ref(), d, centroid));
                }
                let prepared = q.prepare_centroid(centroid).unwrap();
                assert_eq!(prepared.len(), 64);
                q.encode_prepared(d, &prepared, &mut code_back);
                assert_eq!(code, code_back);
                let mut s = q.query_scorer(&docs[..dim]);
                s.set_centroid(centroid);
                let plain = s.score(&code);
                s.set_prepared_centroid(&prepared);
                assert_eq!(plain.to_bits(), s.score(&code).to_bits());
            }
            let mut nested_code = vec![0u8; c.code_len()];
            c.encode(d, centroid, &mut nested_code);
            c_back.encode(d, centroid, &mut code_back);
            assert_eq!(nested_code, code_back);
        }
    }

    #[test]
    fn a_centroid_free_inner_quantizer_gets_zeros() {
        let dim = 20;
        let data = vectors(30, dim, 41);
        let q = RotatedFactory::new(Arc::new(ScalarFactory::int7()))
            .train(dim, Similarity::Euclidean, &data)
            .unwrap();
        assert!(!q.uses_centroid());
        assert_eq!(q.prepare_centroid(&data[dim..2 * dim]), None);
        let mut code = vec![0u8; q.code_len()];
        q.encode(&data[..dim], &data[dim..2 * dim], &mut code);
        let mut other = vec![0u8; q.code_len()];
        q.encode_prepared(&data[..dim], &data[2 * dim..3 * dim], &mut other);
        assert_eq!(code, other);
        let mut s = q.query_scorer(&data[..dim]);
        s.set_centroid(&data[dim..2 * dim]);
        let self_score = s.score(&code);
        assert!(self_score > 0.9, "{self_score}");
        s.set_prepared_centroid(&data[2 * dim..3 * dim]);
        assert_eq!(s.score(&code).to_bits(), self_score.to_bits());
    }

    #[test]
    #[should_panic(expected = "centroid dimension")]
    fn set_centroid_checks_length() {
        let q = rot_bbq().train(8, Similarity::Euclidean, &[]).unwrap();
        q.query_scorer(&[0.0; 8]).set_centroid(&[0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "centroid dimension")]
    fn encode_checks_centroid_length() {
        let q = rot_bbq().train(8, Similarity::Euclidean, &[]).unwrap();
        q.encode(&[0.0; 8], &[0.0; 4], &mut [0u8; 64]);
    }

    #[test]
    #[should_panic(expected = "inner quantizer dimension")]
    fn wrapping_checks_the_inner_dimension() {
        let inner = OsqFactory::bbq()
            .train(8, Similarity::Euclidean, &[])
            .unwrap();
        let _ = RotatedQuantizer::new("rot-osq".into(), Rotation::new(3, 4, 1, 1).unwrap(), inner);
    }

    #[test]
    fn accessors() {
        let inner = OsqFactory::bbq()
            .train(8, Similarity::Euclidean, &[])
            .unwrap();
        let q = RotatedQuantizer::new("rot-osq".into(), Rotation::new(5, 8, 1, 1).unwrap(), inner);
        assert_eq!(q.rotation().width(), 8);
        assert_eq!(q.inner().dimension(), 8);
    }
}
