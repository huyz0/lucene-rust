//! Late-interaction (`MaxSim`) kernels: for every query token, the best dot
//! product with any document token, summed over the query tokens.
//!
//! The document side comes in a block-transposed layout, so a kernel scores
//! [`LANES`] document tokens per vector instruction without a horizontal
//! reduction: block `b` holds tokens `16 * b .. 16 * b + 16`, component-major,
//! `block[k * 16 + l]` = component `k` of token `16 * b + l`. A block past the
//! last token is padded with zeros, and the padded lanes never count.
//!
//! Every lane accumulates its own dot product sequentially over the
//! components with a fused multiply-add, so the AVX2, AVX-512 and NEON paths
//! produce the scalar path's result bit for bit; the scalar path is the
//! specification and the tests compare all of them. A token's max uses
//! `if best > v { best } else { v }` (what `maxps` does); `NaN` inputs give
//! an unspecified result.

/// Document tokens per block.
pub const LANES: usize = 16;

/// The number of `f32`s a block-transposed multi-vector of `tokens` tokens
/// with `dim` components takes, or `None` when that does not fit a `usize`.
#[inline]
pub fn blocked_len(tokens: usize, dim: usize) -> Option<usize> {
    tokens.div_ceil(LANES).checked_mul(dim)?.checked_mul(LANES)
}

/// Writes `tokens` (token-major, `dim` components each) into `out` in the
/// block-transposed layout, replacing what `out` held. Returns the number of
/// tokens.
///
/// # Panics
/// When `dim` is zero or `tokens.len()` is not a multiple of `dim`.
pub fn transpose_into_blocks(tokens: &[f32], dim: usize, out: &mut Vec<f32>) -> usize {
    assert!(
        dim > 0 && tokens.len().is_multiple_of(dim),
        "token data must be whole tokens"
    );
    let n = tokens.len() / dim;
    out.clear();
    // `n * dim` floats exist, so rounding `n` up to a block cannot overflow.
    out.resize(
        blocked_len(n, dim).expect("an allocated slice's length"),
        0.0,
    );
    for (i, token) in tokens.chunks_exact(dim).enumerate() {
        let base = (i / LANES) * dim * LANES + i % LANES;
        for (k, &x) in token.iter().enumerate() {
            out[base + k * LANES] = x;
        }
    }
    n
}

/// A document's tokens in the block layout: `f32`s, or IEEE half floats as
/// little-endian byte pairs (two bytes per entry), which the kernels widen
/// as they load a row, with no intermediate buffer.
#[derive(Debug, Clone, Copy)]
pub enum Blocks<'a> {
    /// `blocked_len(tokens, dim)` floats.
    F32(&'a [f32]),
    /// `2 * blocked_len(tokens, dim)` bytes of half floats.
    F16(&'a [u8]),
}

impl Blocks<'_> {
    fn entries(&self) -> Option<usize> {
        match self {
            Blocks::F32(b) => Some(b.len()),
            Blocks::F16(b) => b.len().is_multiple_of(2).then_some(b.len() / 2),
        }
    }

    /// Entry `i` as an `f32` (the specification of every row load).
    #[inline]
    fn get(&self, i: usize) -> f32 {
        match self {
            Blocks::F32(b) => b[i],
            Blocks::F16(b) => f16_to_f32(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])),
        }
    }
}

/// IEEE half float to `f32`, exact (what `vcvtph2ps` computes).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // subnormal: normalise the mantissa
            let shift = man.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((man << shift) & 0x3ff) << 13
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

fn check(query: &[f32], dim: usize, blocks: &Blocks<'_>, tokens: usize) {
    assert!(
        dim > 0 && query.len().is_multiple_of(dim),
        "query must be whole tokens"
    );
    // Checked arithmetic: the kernels index by `tokens` and `dim` alone, so a
    // wrapped length here would let them read past `blocks`.
    assert!(
        blocks.entries().is_some() && blocks.entries() == blocked_len(tokens, dim),
        "blocks do not hold `tokens` tokens"
    );
}

/// `sum over query tokens q of max over document tokens d of q . d`: `query`
/// token-major, `blocks` in the block layout ([`transpose_into_blocks`]).
/// Zero document tokens give `f32::NEG_INFINITY` per query token.
///
/// # Panics
/// When `query` is not whole tokens or `blocks` does not hold `tokens`
/// tokens of `dim` components.
#[inline]
pub fn max_sim_blocks(query: &[f32], dim: usize, blocks: Blocks<'_>, tokens: usize) -> f32 {
    check(query, dim, &blocks, tokens);
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu::level() == crate::cpu::Level::Avx512 {
            // SAFETY: the CPU has AVX-512F (`cpu::level`), and `check` proved
            // the slice lengths the kernel indexes within.
            return unsafe { avx512::max_sim(query, dim, blocks, tokens) };
        }
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "f16c"
        ))]
        {
            // SAFETY: AVX2, FMA and F16C are enabled for this whole build
            // (x86-64-v3); lengths as above.
            return unsafe { avx2::max_sim(query, dim, blocks, tokens) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is aarch64's baseline; lengths as above.
        return unsafe { neon::max_sim(query, dim, blocks, tokens) };
    }
    #[allow(unreachable_code)]
    max_sim_blocks_scalar(query, dim, blocks, tokens)
}

/// The specification [`max_sim_blocks`] must match exactly.
///
/// # Panics
/// As [`max_sim_blocks`].
pub fn max_sim_blocks_scalar(query: &[f32], dim: usize, blocks: Blocks<'_>, tokens: usize) -> f32 {
    check(query, dim, &blocks, tokens);
    let mut total = 0.0f32;
    for q in query.chunks_exact(dim) {
        let mut best = f32::NEG_INFINITY;
        for b in 0..tokens.div_ceil(LANES) {
            let base = b * dim * LANES;
            let valid = (tokens - b * LANES).min(LANES);
            for l in 0..valid {
                let mut acc = 0.0f32;
                for (k, &qk) in q.iter().enumerate() {
                    acc = qk.mul_add(blocks.get(base + k * LANES + l), acc);
                }
                best = if best > acc { best } else { acc };
            }
        }
        total += best;
    }
    total
}

/// Sums the per-query-token bests of `pass` over `ROWS`-token groups, the
/// remainder one token at a time (the order the scalar path adds them in).
macro_rules! drive {
    ($rows:expr, $query:ident, $dim:ident, $blocks:ident, $tokens:ident, $pass:ident) => {{
        let nq = $query.len() / $dim;
        let mut total = 0.0f32;
        let mut qi = 0;
        while qi < nq {
            let rows = (nq - qi).min($rows);
            let mut best = [f32::NEG_INFINITY; $rows];
            if rows == $rows {
                // SAFETY: `$rows` query tokens from qi are inside `query`.
                unsafe { $pass::<$rows>($query, qi, $dim, $blocks, $tokens, &mut best) };
            } else {
                for r in 0..rows {
                    // SAFETY: one query token at qi + r < nq.
                    unsafe { $pass::<1>($query, qi + r, $dim, $blocks, $tokens, &mut best[r..]) };
                }
            }
            for &b in &best[..rows] {
                total += b;
            }
            qi += rows;
        }
        total
    }};
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{Blocks, LANES};
    use std::arch::x86_64::*;

    /// # Safety
    /// The CPU must have AVX-512F; lengths as `max_sim_blocks` checks them.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn max_sim(
        query: &[f32],
        dim: usize,
        blocks: Blocks<'_>,
        tokens: usize,
    ) -> f32 {
        match blocks {
            Blocks::F32(b) => drive!(8, query, dim, b, tokens, pass_f32),
            Blocks::F16(b) => drive!(8, query, dim, b, tokens, pass_f16),
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn pass_f32<const R: usize>(
        q: &[f32],
        qi: usize,
        dim: usize,
        b: &[f32],
        tokens: usize,
        best: &mut [f32],
    ) {
        // SAFETY: row k of block j is floats (j * dim + k) * 16 .. + 16 of `b`.
        unsafe {
            pass::<R>(q, qi, dim, tokens, best, |blk, k| {
                _mm512_loadu_ps(b.as_ptr().add((blk * dim + k) * LANES))
            })
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn pass_f16<const R: usize>(
        q: &[f32],
        qi: usize,
        dim: usize,
        b: &[u8],
        tokens: usize,
        best: &mut [f32],
    ) {
        // SAFETY: row k of block j is bytes 2 (j * dim + k) * 16 .. + 32 of `b`.
        unsafe {
            pass::<R>(q, qi, dim, tokens, best, |blk, k| {
                _mm512_cvtph_ps(_mm256_loadu_si256(
                    b.as_ptr().add(2 * (blk * dim + k) * LANES) as *const __m256i,
                ))
            })
        }
    }

    /// `R` query tokens from `qi` against every block; `row(block, k)` loads
    /// component `k` of a block's 16 tokens.
    ///
    /// # Safety
    /// AVX-512F is enabled (the caller is a kernel with the feature on);
    /// `query` holds `qi + R` rows of `dim` floats; `row` is in bounds for every
    /// block below `tokens.div_ceil(16)` and `k < dim`.
    #[inline(always)]
    unsafe fn pass<const R: usize>(
        query: &[f32],
        qi: usize,
        dim: usize,
        tokens: usize,
        best: &mut [f32],
        row: impl Fn(usize, usize) -> __m512,
    ) {
        // SAFETY: only called from the `#[target_feature]` kernels above, which
        // are themselves only reached once AVX-512F is detected, and inlined
        // into them, so the intrinsics run with the feature on; `row` is one of
        // those kernels' loads, in bounds per their contracts.
        unsafe {
            let q = query.as_ptr();
            let mut maxes = [_mm512_set1_ps(f32::NEG_INFINITY); R];
            for blk in 0..tokens.div_ceil(LANES) {
                let mut acc = [_mm512_setzero_ps(); R];
                for k in 0..dim {
                    let r = row(blk, k);
                    for (j, a) in acc.iter_mut().enumerate() {
                        // (qi + j) * dim + k < query.len().
                        let qv = _mm512_set1_ps(*q.add((qi + j) * dim + k));
                        *a = _mm512_fmadd_ps(qv, r, *a);
                    }
                }
                let valid = (tokens - blk * LANES).min(LANES);
                let mask: __mmask16 = if valid == LANES {
                    0xFFFF
                } else {
                    ((1u32 << valid) - 1) as u16
                };
                for (m, a) in maxes.iter_mut().zip(acc) {
                    *m = _mm512_mask_max_ps(*m, mask, *m, a);
                }
            }
            for (dst, m) in best.iter_mut().zip(maxes) {
                let mut lanes = [0f32; LANES];
                // 16 floats into a 16-float array.
                _mm512_storeu_ps(lanes.as_mut_ptr(), m);
                *dst = super::lane_max(&lanes);
            }
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "f16c"
))]
mod avx2 {
    use super::{Blocks, LANES};
    use std::arch::x86_64::*;

    /// # Safety
    /// AVX2, FMA and F16C; lengths as `max_sim_blocks` checks them.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn max_sim(
        query: &[f32],
        dim: usize,
        blocks: Blocks<'_>,
        tokens: usize,
    ) -> f32 {
        match blocks {
            Blocks::F32(b) => drive!(4, query, dim, b, tokens, pass_f32),
            Blocks::F16(b) => drive!(4, query, dim, b, tokens, pass_f16),
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn pass_f32<const R: usize>(
        q: &[f32],
        qi: usize,
        dim: usize,
        b: &[f32],
        tokens: usize,
        best: &mut [f32],
    ) {
        // SAFETY: as in the AVX-512 kernel, two 8-float halves of the row.
        unsafe {
            pass::<R>(q, qi, dim, tokens, best, |blk, k| {
                let p = b.as_ptr().add((blk * dim + k) * LANES);
                (_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8)))
            })
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn pass_f16<const R: usize>(
        q: &[f32],
        qi: usize,
        dim: usize,
        b: &[u8],
        tokens: usize,
        best: &mut [f32],
    ) {
        // SAFETY: as in the AVX-512 kernel, two 16-byte halves of the row.
        unsafe {
            pass::<R>(q, qi, dim, tokens, best, |blk, k| {
                let p = b.as_ptr().add(2 * (blk * dim + k) * LANES);
                (
                    _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i)),
                    _mm256_cvtph_ps(_mm_loadu_si128(p.add(16) as *const __m128i)),
                )
            })
        }
    }

    /// # Safety
    /// AVX2, FMA and F16C are enabled; the rest as the AVX-512 `pass`.
    #[inline(always)]
    unsafe fn pass<const R: usize>(
        query: &[f32],
        qi: usize,
        dim: usize,
        tokens: usize,
        best: &mut [f32],
        row: impl Fn(usize, usize) -> (__m256, __m256),
    ) {
        // SAFETY: only called from the `#[target_feature]` kernels above, which
        // are themselves only reached once AVX2, FMA and F16C are detected, and
        // inlined into them, so the intrinsics run with the features on; `row`
        // is one of those kernels' loads, in bounds per their contracts.
        unsafe {
            let q = query.as_ptr();
            let mut maxes = [[f32::NEG_INFINITY; LANES]; R];
            for blk in 0..tokens.div_ceil(LANES) {
                let mut lo = [_mm256_setzero_ps(); R];
                let mut hi = [_mm256_setzero_ps(); R];
                for k in 0..dim {
                    let (r0, r1) = row(blk, k);
                    for j in 0..R {
                        // (qi + j) * dim + k < query.len().
                        let qv = _mm256_set1_ps(*q.add((qi + j) * dim + k));
                        lo[j] = _mm256_fmadd_ps(qv, r0, lo[j]);
                        hi[j] = _mm256_fmadd_ps(qv, r1, hi[j]);
                    }
                }
                let valid = (tokens - blk * LANES).min(LANES);
                for j in 0..R {
                    let mut lanes = [0f32; LANES];
                    // 2 x 8 floats into a 16-float array.
                    _mm256_storeu_ps(lanes.as_mut_ptr(), lo[j]);
                    _mm256_storeu_ps(lanes.as_mut_ptr().add(8), hi[j]);
                    for (m, &v) in maxes[j].iter_mut().zip(&lanes[..valid]) {
                        *m = if *m > v { *m } else { v };
                    }
                }
            }
            for (dst, m) in best.iter_mut().zip(&maxes) {
                *dst = super::lane_max(m);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{Blocks, LANES};
    use std::arch::aarch64::*;

    /// # Safety
    /// Lengths as `max_sim_blocks` checks them.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn max_sim(
        query: &[f32],
        dim: usize,
        blocks: Blocks<'_>,
        tokens: usize,
    ) -> f32 {
        match blocks {
            Blocks::F32(b) => drive!(4, query, dim, b, tokens, pass_f32),
            Blocks::F16(b) => drive!(4, query, dim, b, tokens, pass_f16),
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn pass_f32<const R: usize>(
        q: &[f32],
        qi: usize,
        dim: usize,
        b: &[f32],
        tokens: usize,
        best: &mut [f32],
    ) {
        // SAFETY: row k of block j is floats (j * dim + k) * 16 .. + 16 of `b`.
        unsafe {
            pass::<R>(q, qi, dim, tokens, best, |blk, k| {
                let p = b.as_ptr().add((blk * dim + k) * LANES);
                [
                    vld1q_f32(p),
                    vld1q_f32(p.add(4)),
                    vld1q_f32(p.add(8)),
                    vld1q_f32(p.add(12)),
                ]
            })
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn pass_f16<const R: usize>(
        q: &[f32],
        qi: usize,
        dim: usize,
        b: &[u8],
        tokens: usize,
        best: &mut [f32],
    ) {
        // Half-float NEON conversions are not stable in Rust: widen the row
        // with the exact scalar conversion, then load it.
        // SAFETY: the widened row is a 16-float local.
        unsafe {
            pass::<R>(q, qi, dim, tokens, best, |blk, k| {
                let base = 2 * (blk * dim + k) * LANES;
                let mut row = [0f32; LANES];
                for (l, r) in row.iter_mut().enumerate() {
                    *r = super::f16_to_f32(u16::from_le_bytes([
                        b[base + 2 * l],
                        b[base + 2 * l + 1],
                    ]));
                }
                let p = row.as_ptr();
                [
                    vld1q_f32(p),
                    vld1q_f32(p.add(4)),
                    vld1q_f32(p.add(8)),
                    vld1q_f32(p.add(12)),
                ]
            })
        }
    }

    #[inline(always)]
    unsafe fn pass<const R: usize>(
        query: &[f32],
        qi: usize,
        dim: usize,
        tokens: usize,
        best: &mut [f32],
        row: impl Fn(usize, usize) -> [float32x4_t; 4],
    ) {
        let q = query.as_ptr();
        let mut maxes = [[f32::NEG_INFINITY; LANES]; R];
        for blk in 0..tokens.div_ceil(LANES) {
            let mut acc = [[vdupq_n_f32(0.0); 4]; R];
            for k in 0..dim {
                let rows = row(blk, k);
                for (j, a) in acc.iter_mut().enumerate() {
                    // SAFETY: (qi + j) * dim + k < query.len().
                    let qv = vdupq_n_f32(unsafe { *q.add((qi + j) * dim + k) });
                    for (x, &r) in a.iter_mut().zip(&rows) {
                        *x = vfmaq_f32(*x, qv, r);
                    }
                }
            }
            let valid = (tokens - blk * LANES).min(LANES);
            for j in 0..R {
                let mut lanes = [0f32; LANES];
                for (c, &a) in acc[j].iter().enumerate() {
                    // SAFETY: 4 floats at offset 4c of a 16-float array.
                    unsafe { vst1q_f32(lanes.as_mut_ptr().add(4 * c), a) };
                }
                for (m, &v) in maxes[j].iter_mut().zip(&lanes[..valid]) {
                    *m = if *m > v { *m } else { v };
                }
            }
        }
        for (dst, m) in best.iter_mut().zip(&maxes) {
            *dst = super::lane_max(m);
        }
    }
}

/// The max of a lane array in lane order, as the scalar path folds it.
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code)
)]
fn lane_max(lanes: &[f32; LANES]) -> f32 {
    lanes
        .iter()
        .fold(f32::NEG_INFINITY, |m, &v| if m > v { m } else { v })
}

#[cfg(test)]
mod tests;
