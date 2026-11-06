//! A random orthogonal rotation in front of any quantizer, the step that
//! gives RaBitQ its accuracy: rotated, a vector's energy is spread evenly
//! over its dimensions, so a few bits per dimension lose less of it.
//!
//! [`RotatedFactory`] wraps another [`QuantizerFactory`] and registers as
//! `rot-<inner name>` (`rot-osq`, `rot-sq`). Its quantizer zero-pads every
//! vector it is given (training sample, document, centroid, query) to the
//! next power of two and applies a randomized Hadamard transform: `rounds`
//! times, flip the signs of a seeded random subset of coordinates, then a
//! fast Walsh-Hadamard transform scaled by `1/sqrt(n)`. Every step is
//! orthogonal, so dot products, norms and Euclidean distances, and so every
//! similarity's score, are unchanged. The inner quantizer is trained on,
//! encodes and scores the rotated vectors at the padded dimension; codes,
//! blocks and scores are its own. [`Quantizer::prepare_centroid`] returns
//! the rotated centroid, so an index that keeps it rotates each centroid
//! once rather than per vector encoded or per partition probed.
//!
//! The state is the seed (`u64` LE), the round count (one byte), then the
//! inner quantizer's state.

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
const HEADER_BYTES: usize = 9;

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

/// A seeded randomized Hadamard rotation of `dimension`-dimensional vectors
/// into the next power of two.
#[derive(Debug, Clone, PartialEq)]
pub struct Rotation {
    dimension: usize,
    padded: usize,
    seed: u64,
    rounds: u8,
    /// One row of `padded` signs per round, each `±1/sqrt(padded)`, so a
    /// round is a multiply then [`fwht`].
    signs: Vec<f32>,
}

impl Rotation {
    /// The rotation of `dimension`-dimensional vectors drawn from `seed`,
    /// with `rounds` sign-flip and Hadamard rounds.
    ///
    /// # Errors
    ///
    /// A zero dimension, one whose padding overflows, or `rounds` outside
    /// `1..=MAX_ROUNDS`.
    pub fn new(dimension: usize, seed: u64, rounds: u8) -> Result<Self> {
        if dimension == 0 {
            return Err(Error::IllegalArgument("dimension must be positive".into()));
        }
        if !(1..=MAX_ROUNDS).contains(&rounds) {
            return Err(Error::IllegalArgument(format!(
                "rotation rounds must be 1..={MAX_ROUNDS}, got {rounds}"
            )));
        }
        let padded = dimension
            .checked_next_power_of_two()
            .ok_or_else(|| Error::IllegalArgument(format!("dimension {dimension} is too large")))?;
        let len = padded
            .checked_mul(usize::from(rounds))
            .ok_or_else(|| Error::IllegalArgument(format!("dimension {dimension} is too large")))?;
        // `padded` is a power of two, so its square root is exact or the
        // nearest float to an exact power of two's root.
        let scale = (1.0 / (padded as f64).sqrt()) as f32;
        let mut state = seed;
        let mut bits = 0u64;
        let signs = (0..len)
            .map(|i| {
                if i.is_multiple_of(64) {
                    bits = split_mix(&mut state);
                }
                let negative = bits & 1 == 1;
                bits >>= 1;
                if negative {
                    -scale
                } else {
                    scale
                }
            })
            .collect();
        Ok(Rotation {
            dimension,
            padded,
            seed,
            rounds,
            signs,
        })
    }

    /// The dimension of the vectors rotated.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The dimension of a rotated vector: the next power of two.
    pub fn padded_dimension(&self) -> usize {
        self.padded
    }

    /// The seed the signs were drawn from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Sign-flip and Hadamard rounds.
    pub fn rounds(&self) -> u8 {
        self.rounds
    }

    /// Rotates `x` (of [`Rotation::dimension`] floats), zero-padded, into
    /// `out` (of [`Rotation::padded_dimension`] floats).
    ///
    /// # Panics
    ///
    /// On a wrong-length `x` or `out`.
    pub fn rotate(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.dimension, "vector dimension");
        assert_eq!(out.len(), self.padded, "rotated dimension");
        let (head, tail) = out.split_at_mut(self.dimension);
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

    fn train(
        &self,
        dimension: usize,
        similarity: Similarity,
        sample: &[f32],
    ) -> Result<Box<dyn Quantizer>> {
        check_sample(dimension, sample)?;
        let rotation = Rotation::new(dimension, self.seed, self.rounds)?;
        let padded = rotation.padded_dimension();
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
        let (seed, rounds) = header.split_at(8);
        let mut seed_bytes = [0u8; 8];
        seed_bytes.copy_from_slice(seed);
        let rotation = Rotation::new(dimension, u64::from_le_bytes(seed_bytes), rounds[0])
            .map_err(|e| Error::CorruptState(e.to_string()))?;
        let inner = self
            .inner
            .read(rotation.padded_dimension(), similarity, inner_state)?;
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
    /// `inner`, a quantizer of [`Rotation::padded_dimension`] floats, behind
    /// `rotation`; `name` is the factory's registry name.
    ///
    /// # Panics
    ///
    /// When `inner`'s dimension is not the rotation's padded dimension.
    pub fn new(name: String, rotation: Rotation, inner: Box<dyn Quantizer>) -> Self {
        assert_eq!(
            inner.dimension(),
            rotation.padded_dimension(),
            "inner quantizer dimension"
        );
        RotatedQuantizer {
            name,
            zeros: vec![0f32; rotation.padded_dimension()],
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
        let mut rotated = vec![0f32; self.rotation.padded_dimension()];
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
        let mut rotated = vec![0f32; self.rotation.padded_dimension()];
        self.rotation.rotate(vector, &mut rotated);
        self.inner
            .encode_prepared(&rotated, self.inner_centroid(prepared), out);
    }

    fn query_scorer(&self, query: &[f32]) -> Box<dyn QueryScorer + '_> {
        let mut rotated = vec![0f32; self.rotation.padded_dimension()];
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

    #[test]
    fn rotation_is_orthogonal() {
        for dim in [1, 7, 64, 100, 300] {
            for rounds in [1, 2, 3, MAX_ROUNDS] {
                let r = Rotation::new(dim, 42, rounds).unwrap();
                assert_eq!(r.dimension(), dim);
                assert_eq!(r.padded_dimension(), dim.next_power_of_two());
                let v = vectors(2, dim, dim as u64 + 1);
                let (a, b) = v.split_at(dim);
                let mut ra = vec![0f32; r.padded_dimension()];
                let mut rb = vec![0f32; r.padded_dimension()];
                r.rotate(a, &mut ra);
                r.rotate(b, &mut rb);
                let tol = |x: f64| 1e-5 * x.abs().max(1.0);
                assert!((dot(&ra, &rb) - dot(a, b)).abs() < tol(dot(a, a)), "{dim}");
                assert!((dot(&ra, &ra) - dot(a, a)).abs() < tol(dot(a, a)), "{dim}");
                assert!((l2(&ra, &rb) - l2(a, b)).abs() < tol(l2(a, b)), "{dim}");
            }
        }
    }

    #[test]
    fn rotation_mixes_and_is_deterministic_per_seed() {
        let dim = 100;
        let a = Rotation::new(dim, 7, 3).unwrap();
        let b = Rotation::new(dim, 7, 3).unwrap();
        let c = Rotation::new(dim, 8, 3).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!((a.seed(), a.rounds()), (7, 3));
        let mut e = vec![0f32; dim];
        e[0] = 1.0;
        let (mut ra, mut rb, mut rc) = (vec![0f32; 128], vec![0f32; 128], vec![0f32; 128]);
        a.rotate(&e, &mut ra);
        b.rotate(&e, &mut rb);
        c.rotate(&e, &mut rc);
        assert_eq!(ra, rb);
        assert_ne!(ra, rc);
        // A unit axis is spread over every coordinate.
        let max = ra.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(max < 0.5, "max coordinate {max}");
    }

    #[test]
    fn rotation_rejects_bad_parameters() {
        assert!(Rotation::new(0, 1, 3).is_err());
        assert!(Rotation::new(4, 1, 0).is_err());
        assert!(Rotation::new(4, 1, MAX_ROUNDS + 1).is_err());
        assert!(Rotation::new(usize::MAX, 1, 3).is_err());
        assert!(Rotation::new(1 << (usize::BITS - 1), 1, 3).is_err());
    }

    #[test]
    #[should_panic(expected = "vector dimension")]
    fn rotate_checks_length() {
        Rotation::new(5, 1, 1)
            .unwrap()
            .rotate(&[0.0; 4], &mut [0.0; 8]);
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
        for bad in [&state[..3], &state[..9]] {
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
        for dim in [64, 100] {
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
            let mut padded = vec![0f32; 32];
            let rq = q.state();
            let rotation =
                Rotation::new(dim, u64::from_le_bytes(rq[..8].try_into().unwrap()), rq[8]).unwrap();
            let inner = OsqFactory::bbq()
                .train(32, Similarity::Euclidean, &[])
                .unwrap();
            let mut pc = vec![0f32; 32];
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
                assert_eq!(prepared.len(), 32);
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
        let _ = RotatedQuantizer::new("rot-osq".into(), Rotation::new(3, 1, 1).unwrap(), inner);
    }

    #[test]
    fn accessors() {
        let inner = OsqFactory::bbq()
            .train(8, Similarity::Euclidean, &[])
            .unwrap();
        let q = RotatedQuantizer::new("rot-osq".into(), Rotation::new(5, 1, 1).unwrap(), inner);
        assert_eq!(q.rotation().padded_dimension(), 8);
        assert_eq!(q.inner().dimension(), 8);
    }
}
