//! Data-oblivious Lloyd-Max scalar quantization of partition residuals: the
//! scalar stage of TurboQuant (`TurboQuant_mse`) and IsoQuant.
//!
//! Each vector is centred on its partition centroid, its residual `r`
//! normalized to `u = r / |r|`, and every coordinate of `u * sqrt(d)` coded
//! to the nearest of the `2^bits` Lloyd-Max levels of a standard normal.
//! After a random rotation the coordinates of a unit vector are close to
//! `N(0, 1/d)`, so one fixed codebook fits every vector and nothing is
//! trained. The rotation is not done here: the caller rotates vectors and
//! queries once (it preserves distances), and this quantizer codes what it
//! is given.
//!
//! A code is the packed level indices, then two `f32`s: the scale `f` that
//! turns the reconstruction's dot product into the residual's, and `|r|^2`.
//! With `unbiased` the scale is `|r| / <u, x>` (the RaBitQ estimator, `x`
//! the reconstructed unit vector), otherwise `|r|` (TurboQuant's MSE
//! reconstruction, which shrinks inner products).
//!
//! Queries are scored with fast scan: 4-bit sub-codes (four 1-bit, two 2-bit
//! or one 4-bit coordinate) looked up in 16-entry `u8` tables, 32 codes at a
//! time.
#![allow(clippy::arithmetic_side_effects, clippy::cast_possible_truncation)]

use lucene_util::{quantization, simd};

use crate::{
    check_sample, f32_at, prepare_vector, Error, Quantizer, QueryScorer, Result, Similarity,
};

/// The registry name of [`LloydMaxFactory`].
pub const NAME: &str = "lloydmax";

const BLOCK: usize = simd::FAST_SCAN_BLOCK;
const TAIL: usize = 8;

/// The `2^bits` Lloyd-Max levels of a standard normal, ascending.
pub fn standard_normal_levels(bits: u32) -> Vec<f32> {
    let k = 1usize << bits;
    // Integrate on a fine grid: plenty for 16 levels.
    let (lo, hi, n) = (-8.0f64, 8.0f64, 160_001usize);
    let step = (hi - lo) / (n - 1) as f64;
    let xs: Vec<f64> = (0..n).map(|i| lo + i as f64 * step).collect();
    let pdf: Vec<f64> = xs.iter().map(|x| (-0.5 * x * x).exp()).collect();
    // Start from equal-probability quantiles' rough positions.
    let mut c: Vec<f64> = (0..k)
        .map(|i| -2.5 + 5.0 * (i as f64 + 0.5) / k as f64)
        .collect();
    for _ in 0..500 {
        let mut num = vec![0f64; k];
        let mut den = vec![0f64; k];
        let mut j = 0usize;
        for (x, p) in xs.iter().zip(&pdf) {
            while j + 1 < k && *x > (c[j] + c[j + 1]) / 2.0 {
                j += 1;
            }
            num[j] += x * p;
            den[j] += p;
        }
        for i in 0..k {
            if den[i] > 0.0 {
                c[i] = num[i] / den[i];
            }
        }
    }
    c.into_iter().map(|x| x as f32).collect()
}

/// Builds [`LloydMaxQuantizer`]s; nothing is trained.
#[derive(Debug, Clone, Copy)]
pub struct LloydMaxFactory {
    bits: u32,
    unbiased: bool,
}

impl LloydMaxFactory {
    /// Codes of `bits` (1, 2 or 4) per dimension.
    pub fn new(bits: u32, unbiased: bool) -> Self {
        assert!(matches!(bits, 1 | 2 | 4), "bits must be 1, 2 or 4");
        Self { bits, unbiased }
    }
}

impl crate::QuantizerFactory for LloydMaxFactory {
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
        Ok(Box::new(LloydMaxQuantizer::new(
            dimension,
            similarity,
            self.bits,
            self.unbiased,
        )))
    }

    fn read(
        &self,
        dimension: usize,
        similarity: Similarity,
        state: &[u8],
    ) -> Result<Box<dyn Quantizer>> {
        match state {
            [b @ (1 | 2 | 4), u @ (0 | 1)] if dimension > 0 => Ok(Box::new(
                LloydMaxQuantizer::new(dimension, similarity, u32::from(*b), *u == 1),
            )),
            _ => Err(Error::CorruptState(format!("lloydmax state {state:?}"))),
        }
    }
}

/// Lloyd-Max codes of partition residuals.
#[derive(Debug, Clone)]
pub struct LloydMaxQuantizer {
    dimension: usize,
    similarity: Similarity,
    bits: u32,
    unbiased: bool,
    /// Levels divided by `sqrt(dimension)`: the reconstruction of a unit
    /// vector's coordinate.
    levels: Vec<f32>,
    /// Midpoints between consecutive standard-normal levels.
    bounds: Vec<f32>,
    /// Coordinates per 4-bit sub-code.
    per_group: usize,
    groups: usize,
    packed: usize,
}

impl LloydMaxQuantizer {
    /// A quantizer of `bits` per dimension.
    pub fn new(dimension: usize, similarity: Similarity, bits: u32, unbiased: bool) -> Self {
        let std = standard_normal_levels(bits);
        let bounds = std.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
        let inv = 1.0 / (dimension as f32).sqrt();
        let per_group = 4 / bits as usize;
        let groups = dimension.div_ceil(per_group);
        Self {
            dimension,
            similarity,
            bits,
            unbiased,
            levels: std.iter().map(|l| l * inv).collect(),
            bounds,
            per_group,
            groups,
            packed: groups.div_ceil(2),
        }
    }

    fn nibble(code: &[u8], g: usize) -> u8 {
        let b = code[g / 2];
        if g % 2 == 0 {
            b & 15
        } else {
            b >> 4
        }
    }

    fn block_bytes(&self) -> usize {
        self.groups * 16 + BLOCK * TAIL
    }
}

impl Quantizer for LloydMaxQuantizer {
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
        self.packed + TAIL
    }

    fn uses_centroid(&self) -> bool {
        true
    }

    fn encode(&self, vector: &[f32], centroid: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dimension, "vector dimension");
        assert_eq!(centroid.len(), self.dimension, "centroid dimension");
        assert_eq!(out.len(), self.code_len(), "code length");
        let mut r = vec![0f32; self.dimension];
        prepare_vector(self.similarity, vector, &mut r);
        for (x, c) in r.iter_mut().zip(centroid) {
            *x -= c;
        }
        let n2 = simd::dot_f32(&r, &r);
        let n = n2.sqrt();
        let sqrt_d = (self.dimension as f32).sqrt();
        out.fill(0);
        let mut ux = 0f32;
        for (i, &x) in r.iter().enumerate() {
            let z = if n > 0.0 { x / n * sqrt_d } else { 0.0 };
            let idx = self.bounds.partition_point(|&b| b < z);
            ux += if n > 0.0 { x / n } else { 0.0 } * self.levels[idx];
            let g = i / self.per_group;
            let shift = (i % self.per_group) as u32 * self.bits;
            let sub = (idx as u8) << shift;
            out[g / 2] |= if g % 2 == 0 { sub } else { sub << 4 };
        }
        let f = if n == 0.0 {
            0.0
        } else if self.unbiased && ux > 0.0 {
            n / ux
        } else {
            n
        };
        out[self.packed..self.packed + 4].copy_from_slice(&f.to_le_bytes());
        out[self.packed + 4..self.packed + 8].copy_from_slice(&n2.to_le_bytes());
    }

    fn query_scorer(&self, query: &[f32]) -> Box<dyn QueryScorer + '_> {
        assert_eq!(query.len(), self.dimension, "query dimension");
        let mut q = vec![0f32; self.dimension];
        prepare_vector(self.similarity, query, &mut q);
        let mut s = LloydMaxScorer {
            q: self,
            query: q,
            target: vec![0f32; self.groups * self.per_group],
            luts: vec![0u8; self.groups * 16],
            lut_scale: 0.0,
            lut_bias: 0.0,
            qc: 0.0,
            qn2: 0.0,
        };
        if self.similarity != Similarity::Euclidean {
            // Inner products estimate <q, r>, which does not depend on the
            // centroid: one table for every partition.
            s.target[..self.dimension].copy_from_slice(&s.query);
            s.fill_luts();
        }
        Box::new(s)
    }

    fn state(&self) -> Vec<u8> {
        vec![self.bits as u8, u8::from(self.unbiased)]
    }

    fn block_len(&self, n: usize) -> usize {
        n.div_ceil(BLOCK) * self.block_bytes()
    }

    fn write_block(&self, codes: &[u8], out: &mut Vec<u8>) {
        let code_len = self.code_len();
        assert_eq!(codes.len() % code_len, 0, "codes must be whole codes");
        for block in codes.chunks(BLOCK * code_len) {
            let code = |j: usize| block.get(j * code_len..(j + 1) * code_len);
            for g in 0..self.groups {
                for j in 0..16 {
                    let lo = code(j).map_or(0, |c| Self::nibble(c, g));
                    let hi = code(j + 16).map_or(0, |c| Self::nibble(c, g));
                    out.push(lo | (hi << 4));
                }
            }
            for at in [0, 4] {
                for j in 0..BLOCK {
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

#[derive(Debug)]
struct LloydMaxScorer<'q> {
    q: &'q LloydMaxQuantizer,
    query: Vec<f32>,
    /// What the codes are dotted with, padded to whole groups: the query
    /// for inner products, the query less the centroid for Euclidean.
    target: Vec<f32>,
    luts: Vec<u8>,
    lut_scale: f32,
    lut_bias: f32,
    /// `<q, c>`.
    qc: f32,
    /// `|q - c|^2`.
    qn2: f32,
}

impl LloydMaxScorer<'_> {
    /// One float table per group, then quantized to `u8` with a common
    /// scale and a per-group offset (the offsets summed into one bias).
    fn fill_luts(&mut self) {
        let q = self.q;
        let mask = (1usize << q.bits) - 1;
        let lo = q.levels[0];
        let hi = q.levels[mask];
        // Levels ascend, so a group's smallest entry takes each coordinate's
        // smaller product and its range is the sum of |x| * (hi - lo).
        let mut range = 0f32;
        let mut bias = 0f32;
        for x in self.target.chunks_exact(q.per_group) {
            let mut r = 0f32;
            for &xj in x {
                bias += (xj * lo).min(xj * hi);
                r += xj.abs();
            }
            range = range.max(r * (hi - lo));
        }
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
        let inv = 1.0 / scale;
        let bits = q.bits as usize;
        for (x, lut) in self
            .target
            .chunks_exact(q.per_group)
            .zip(self.luts.chunks_exact_mut(16))
        {
            let mut t = [0f32; 16];
            let mut mn = 0f32;
            for (j, &xj) in x.iter().enumerate() {
                mn += (xj * lo).min(xj * hi);
                for (v, e) in t.iter_mut().enumerate() {
                    *e += xj * q.levels[(v >> (j * bits)) & mask];
                }
            }
            for (l, e) in lut.iter_mut().zip(t) {
                *l = ((e - mn) * inv + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
        self.lut_scale = scale;
        self.lut_bias = bias;
    }

    #[inline]
    fn finish(&self, ip: f32, n2: f32) -> f32 {
        match self.q.similarity {
            Similarity::Euclidean => 1.0 / (1.0 + (self.qn2 + n2 - 2.0 * ip).max(0.0)),
            Similarity::MaximumInnerProduct => {
                quantization::scale_max_inner_product_score(self.qc + ip)
            }
            Similarity::DotProduct | Similarity::Cosine => ((1.0 + self.qc + ip) / 2.0).max(0.0),
        }
    }
}

impl QueryScorer for LloydMaxScorer<'_> {
    fn set_centroid(&mut self, centroid: &[f32]) {
        assert_eq!(centroid.len(), self.q.dimension, "centroid dimension");
        if self.q.similarity == Similarity::Euclidean {
            for ((t, &x), &c) in self.target.iter_mut().zip(&self.query).zip(centroid) {
                *t = x - c;
            }
            let d = &self.target[..self.q.dimension];
            self.qn2 = simd::dot_f32(d, d);
            self.fill_luts();
        } else {
            self.qc = simd::dot_f32(&self.query, centroid);
        }
    }

    fn score(&self, code: &[u8]) -> f32 {
        let q = self.q;
        let mask = (1usize << q.bits) - 1;
        let mut est = 0f32;
        for (i, x) in self.target[..q.dimension].iter().enumerate() {
            let g = i / q.per_group;
            let shift = (i % q.per_group) * q.bits as usize;
            let idx = (usize::from(LloydMaxQuantizer::nibble(code, g)) >> shift) & mask;
            est += x * q.levels[idx];
        }
        let f = f32_at(code, q.packed);
        let n2 = f32_at(code, q.packed + 4);
        self.finish(f * est, n2)
    }

    fn score_block(&self, codes: &[u8], scores: &mut [f32]) {
        let q = self.q;
        assert_eq!(
            codes.len(),
            q.block_len(scores.len()),
            "codes must hold one code per score"
        );
        let nib = q.groups * 16;
        for (block, out) in codes
            .chunks_exact(q.block_bytes())
            .zip(scores.chunks_mut(BLOCK))
        {
            let (nibbles, tail) = block.split_at(nib);
            let mut dots = [0u32; BLOCK];
            simd::lut16_accumulate32(&self.luts, nibbles, &mut dots);
            for (j, s) in out.iter_mut().enumerate() {
                let f = f32_at(tail, 4 * j);
                let n2 = f32_at(tail, 4 * (BLOCK + j));
                let est = dots[j] as f32 * self.lut_scale + self.lut_bias;
                *s = self.finish(f * est, n2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QuantizerFactory;

    #[test]
    fn levels_match_known_values() {
        let l = standard_normal_levels(2);
        assert!(
            (l[3] - 1.5104).abs() < 1e-3 && (l[2] - 0.4528).abs() < 1e-3,
            "{l:?}"
        );
        let l = standard_normal_levels(4);
        assert!((l[15] - 2.7326).abs() < 2e-3, "{l:?}");
    }

    #[test]
    fn block_scores_track_single_scores_and_exact() {
        let dim = 64;
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let vecs: Vec<Vec<f32>> = (0..40).map(|_| (0..dim).map(|_| rnd()).collect()).collect();
        let centroid: Vec<f32> = (0..dim).map(|_| rnd() * 0.1).collect();
        let query: Vec<f32> = (0..dim).map(|_| rnd()).collect();
        for sim in [Similarity::Euclidean, Similarity::DotProduct] {
            for bits in [1, 2, 4] {
                let q = LloydMaxFactory::new(bits, true)
                    .train(dim, sim, &[])
                    .unwrap();
                let mut codes = Vec::new();
                for v in &vecs {
                    let mut c = vec![0u8; q.code_len()];
                    q.encode(v, &centroid, &mut c);
                    codes.extend_from_slice(&c);
                }
                let mut block = Vec::new();
                q.write_block(&codes, &mut block);
                let mut s = q.query_scorer(&query);
                s.set_centroid(&centroid);
                let mut scores = vec![0f32; vecs.len()];
                s.score_block(&block, &mut scores);
                for (i, v) in vecs.iter().enumerate() {
                    let single = s.score(&codes[i * q.code_len()..][..q.code_len()]);
                    assert!(
                        (single - scores[i]).abs() < 0.02,
                        "{sim:?} {bits}: {single} {}",
                        scores[i]
                    );
                    let exact = sim.score(&query, v);
                    let tol: f32 = [0.0, 0.25, 0.12, 0.0, 0.04][bits as usize];
                    assert!(
                        (single - exact).abs() < tol.max(0.04) * 2.0,
                        "{sim:?} {bits}: {single} vs {exact}"
                    );
                }
            }
        }
    }

    #[test]
    fn state_round_trips() {
        let f = LloydMaxFactory::new(2, true);
        let q = f.train(8, Similarity::Euclidean, &[]).unwrap();
        assert_eq!(
            f.read(8, Similarity::Euclidean, &q.state())
                .unwrap()
                .code_len(),
            q.code_len()
        );
        assert!(f.read(8, Similarity::Euclidean, &[3, 0]).is_err());
    }
}
