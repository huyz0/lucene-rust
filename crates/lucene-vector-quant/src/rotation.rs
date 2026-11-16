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
    if n < 8 {
        if n >= 2 {
            butterflies(x, 1);
        }
        if n >= 4 {
            butterflies(x, 2);
        }
        return;
    }
    first_three_stages(x);
    // Two stages per pass over the data where two remain: the pass is
    // bound by loads and stores, and the sums are the same ones in the same
    // order as two single-stage passes, so the result is bit-identical.
    let mut h = 8usize;
    while h.saturating_mul(2) < n {
        butterflies_radix4(x, h);
        h = h.saturating_mul(4);
    }
    if h < n {
        butterflies(x, h);
    }
}

/// The stages of half-width 1, 2 and 4: an 8-point transform of each run of
/// eight, held in registers.
#[inline(always)]
fn first_three_stages(x: &mut [f32]) {
    for c in x.as_chunks_mut::<8>().0 {
        let a = [
            c[0] + c[1],
            c[0] - c[1],
            c[2] + c[3],
            c[2] - c[3],
            c[4] + c[5],
            c[4] - c[5],
            c[6] + c[7],
            c[6] - c[7],
        ];
        let b = [
            a[0] + a[2],
            a[1] + a[3],
            a[0] - a[2],
            a[1] - a[3],
            a[4] + a[6],
            a[5] + a[7],
            a[4] - a[6],
            a[5] - a[7],
        ];
        *c = [
            b[0] + b[4],
            b[1] + b[5],
            b[2] + b[6],
            b[3] + b[7],
            b[0] - b[4],
            b[1] - b[5],
            b[2] - b[6],
            b[3] - b[7],
        ];
    }
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

/// The stages of half-width `h` and `2h` in one pass: within each run of
/// `4h`, the quarters `(a, b, c, d)` become `(a + b + c + d, a - b + c - d,
/// a + b - c - d, a - b - c + d)`, summed as the two stages sum them.
#[inline(always)]
fn butterflies_radix4(x: &mut [f32], h: usize) {
    for run in x.chunks_exact_mut(h.saturating_mul(4)) {
        let (ab, cd) = run.split_at_mut(h.saturating_mul(2));
        let (a, b) = ab.split_at_mut(h);
        let (c, d) = cd.split_at_mut(h);
        for (((a, b), c), d) in a
            .iter_mut()
            .zip(b.iter_mut())
            .zip(c.iter_mut())
            .zip(d.iter_mut())
        {
            let (t0, t1) = (*a + *b, *a - *b);
            let (t2, t3) = (*c + *d, *c - *d);
            *a = t0 + t2;
            *b = t1 + t3;
            *c = t0 - t2;
            *d = t1 - t3;
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
        // A buffer per thread rather than per call: encoding rotates every
        // vector of a field.
        BEFORE.with_borrow_mut(|before| {
            before.resize(self.width, 0.0);
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
        });
    }
}

thread_local! {
    /// [`Rotation::rotate`]'s copy of a round's input, for the permutation.
    static BEFORE: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
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

    /// The inner scorer's bounds: the rotation keeps every norm and inner
    /// product, so they hold for the original vectors too.
    fn error_bounds(&self, codes: &[u8], scores: &[f32], bounds: &mut [f32]) -> bool {
        self.inner.error_bounds(codes, scores, bounds)
    }

    fn error_bound(&self, codes: &[u8], index: usize, score: f32) -> Option<f32> {
        self.inner.error_bound(codes, index, score)
    }
}

#[cfg(test)]
mod tests;
