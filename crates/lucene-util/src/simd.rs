//! Vector kernels for the postings hot loops, behind safe APIs.
//!
//! These are the Rust counterparts of what Lucene gets from its Panama
//! `VectorUtilSupport` (`findNextGEQ`) and `ForDeltaUtil`'s vectorized prefix
//! sum. LLVM does not auto-vectorize either: both are loops whose next step
//! depends on the last (a running sum, an early exit), which is exactly the
//! shape it gives up on.
//!
//! Each kernel has an AVX2 path, compiled in when the build targets a CPU with
//! AVX2 (this workspace's `.cargo/config.toml` builds Linux x86_64 for
//! `x86-64-v3`), and a scalar path for every other target. The unsafe code is
//! confined to this module -- `lucene-util` is one of the three crates
//! AGENTS.md invariant 4 allows it in -- and each `unsafe` block states the
//! bounds it relies on; the public functions take fixed-size arrays or check
//! lengths, so no caller can hand them an out-of-bounds read. The scalar path
//! is the specification: the tests run both and compare.

/// `out[i] = base + deltas[0] + ... + deltas[i]`, in wrapping 32-bit
/// arithmetic -- turning one decoded postings block of doc-id deltas into doc
/// ids (`ForDeltaUtil.decodeAndPrefixSum`'s second half).
///
/// Wrapping, not checked: the deltas come off disk, and a corrupt block must
/// produce garbage doc ids the caller's own invariant checks reject, never a
/// panic. On well-formed input nothing wraps.
#[inline]
pub fn prefix_sum_256(base: i32, deltas: &[u32; 256], out: &mut [i32; 256]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: AVX2 is enabled for this whole build (the `cfg` above), and
        // the kernel reads and writes exactly the 256 lanes of the two
        // fixed-size arrays it is handed.
        unsafe { prefix_sum_256_avx2(base, deltas, out) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    prefix_sum_256_scalar(base, deltas, out)
}

/// The specification [`prefix_sum_256`] must match.
#[inline]
pub fn prefix_sum_256_scalar(base: i32, deltas: &[u32; 256], out: &mut [i32; 256]) {
    let mut sum = base;
    for (o, &d) in out.iter_mut().zip(deltas) {
        sum = sum.wrapping_add(d as i32);
        *o = sum;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn prefix_sum_256_avx2(base: i32, deltas: &[u32; 256], out: &mut [i32; 256]) {
    use std::arch::x86_64::*;
    let mut carry = _mm256_set1_epi32(base);
    let last = _mm256_set1_epi32(7);
    let src = deltas.as_ptr() as *const __m256i;
    let dst = out.as_mut_ptr() as *mut __m256i;
    // 32 chunks of 8 lanes cover exactly the 256 elements of each array.
    for chunk in 0..32 {
        // SAFETY: `chunk < 32`, so `src.add(chunk)`/`dst.add(chunk)` address
        // bytes `32 * chunk .. 32 * chunk + 32` of 1024-byte arrays.
        let mut x = unsafe { _mm256_loadu_si256(src.add(chunk)) };
        // Inclusive scan within each 128-bit lane...
        x = _mm256_add_epi32(x, _mm256_slli_si256::<4>(x));
        x = _mm256_add_epi32(x, _mm256_slli_si256::<8>(x));
        // ...then the low lane's total into every element of the high lane:
        // broadcast lane element 3, move the low lane's copy up, zero below.
        let low_total = _mm256_shuffle_epi32::<0xFF>(x);
        x = _mm256_add_epi32(x, _mm256_permute2x128_si256::<0x08>(low_total, low_total));
        x = _mm256_add_epi32(x, carry);
        // SAFETY: as for the load.
        unsafe { _mm256_storeu_si256(dst.add(chunk), x) };
        carry = _mm256_permutevar8x32_epi32(x, last);
    }
}

/// `VectorUtil.findNextGEQ`: the index of the first element of `buf` that is
/// `>= target`, or `buf.len()` when there is none. `buf` must be ascending
/// for the answer to mean "where `target` would land", though the scan itself
/// is correct for any input.
#[inline]
pub fn find_next_geq(buf: &[i32], target: i32) -> usize {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: AVX2 is enabled for this whole build; the kernel only reads
        // whole 8-lane chunks that lie inside `buf` and finishes the tail
        // with safe indexing.
        unsafe { find_next_geq_avx2(buf, target) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    find_next_geq_scalar(buf, target)
}

/// `PanamaVectorUtilSupport.findNextGEQ`'s AVX2 shape -- the V1 intersection
/// step of Lemire, Boytsov and Kurz, *SIMD Compression and the Intersection of
/// Sorted Integers*: test the one element eight ahead with a plain branch,
/// and only once it reaches `target` count the eight before it that fall
/// short. Returns the same index as [`find_next_geq`] for an ascending `buf`
/// (for any other input it is still at most `buf.len()`).
///
/// Two such steps (eighteen entries), then [`count_less_than`] for the rest.
/// The steps beat a vector compare on every chunk when the answer is close --
/// the branch is predictable on a steady advance, so the CPU runs ahead
/// through it instead of waiting on a mask -- which is why the postings
/// cursor's in-block `advance` uses this. Past them Lucene keeps stepping nine
/// at a time, where the chunked scan is faster: a long in-block advance
/// measured 25% slower stepping all the way.
#[inline(always)]
pub fn find_next_geq_v1(buf: &[i32], target: i32) -> usize {
    let mut from = 0usize;
    for _ in 0..2 {
        if from + 8 >= buf.len() {
            break;
        }
        if buf[from + 8] >= target {
            let chunk: &[i32; 8] = buf[from..from + 8].try_into().expect("eight lanes");
            return from + 8 - geq_mask8(chunk, target).count_ones() as usize;
        }
        from += 9;
    }
    from + count_less_than(&buf[from..], target)
}

/// How many elements of `buf` are `< target` -- for an ascending `buf`, the
/// same index [`find_next_geq`] returns, found without an early exit.
///
/// That is the point of it: a scan that stops at the answer ends on a branch
/// whose outcome moves with the answer, and a landing a hundred entries in
/// pays the mispredict on top of the compares. Counting every lane costs a
/// fixed, pipelined compare-and-subtract per eight entries instead, which on a
/// 256-entry postings block is the cheaper of the two once the answer is more
/// than a couple of chunks away.
#[inline]
pub fn count_less_than(buf: &[i32], target: i32) -> usize {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: AVX2 is enabled for this whole build; the kernel only reads
        // whole 8-lane chunks inside `buf` and counts the tail safely.
        unsafe { count_less_than_avx2(buf, target) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    count_less_than_scalar(buf, target)
}

/// The specification [`count_less_than`] must match.
pub fn count_less_than_scalar(buf: &[i32], target: i32) -> usize {
    buf.iter().filter(|&&d| d < target).count()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn count_less_than_avx2(buf: &[i32], target: i32) -> usize {
    use std::arch::x86_64::*;
    // Each lane counts down by one per element below `target` (a true
    // compare is all ones, i.e. -1), in 32 bits. Below 2^31 elements there
    // are under 2^28 chunks, so no lane -- nor the sum of all eight -- can
    // wrap; anything longer is counted the slow way.
    if buf.len() >= 1 << 31 {
        return count_less_than_scalar(buf, target);
    }
    let bound = _mm256_set1_epi32(target);
    let chunks = buf.len() / 8;
    let ptr = buf.as_ptr() as *const __m256i;
    let mut acc = _mm256_setzero_si256();
    for chunk in 0..chunks {
        // SAFETY: `chunk < buf.len() / 8`, so the 8 lanes read lie inside
        // `buf`.
        let x = unsafe { _mm256_loadu_si256(ptr.add(chunk)) };
        acc = _mm256_add_epi32(acc, _mm256_cmpgt_epi32(bound, x));
    }
    let lo = _mm256_castsi256_si128(acc);
    let hi = _mm256_extracti128_si256::<1>(acc);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
    let vector = _mm_cvtsi128_si32(s).unsigned_abs() as usize;
    vector + count_less_than_scalar(&buf[chunks * 8..], target)
}

/// Bit `j` set when `chunk[j] >= target`: one 8-lane compare, for a caller
/// that knows the answer is usually within the next eight entries and wants
/// no loop setup in front of it (Lucene's `findNextGEQ` probes the same way).
#[inline(always)]
pub fn geq_mask8(chunk: &[i32; 8], target: i32) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: AVX2 is enabled for this whole build, and the load reads
        // exactly the eight `i32`s of the array it is handed.
        unsafe {
            use std::arch::x86_64::*;
            if target == i32::MIN {
                return 0xFF;
            }
            let x = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let gt = _mm256_cmpgt_epi32(x, _mm256_set1_epi32(target - 1));
            _mm256_movemask_ps(_mm256_castsi256_ps(gt)) as u32
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    geq_mask8_scalar(chunk, target)
}

/// The specification [`geq_mask8`] must match.
pub fn geq_mask8_scalar(chunk: &[i32; 8], target: i32) -> u32 {
    chunk
        .iter()
        .enumerate()
        .fold(0u32, |m, (j, &d)| m | (u32::from(d >= target) << j))
}

/// The specification [`find_next_geq`] must match.
#[inline]
pub fn find_next_geq_scalar(buf: &[i32], target: i32) -> usize {
    buf.iter().position(|&d| d >= target).unwrap_or(buf.len())
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn find_next_geq_avx2(buf: &[i32], target: i32) -> usize {
    use std::arch::x86_64::*;
    // `d >= target` is `d > target - 1`; `target == i32::MIN` makes every
    // element qualify, which the scalar path answers the same way.
    if target == i32::MIN {
        return 0;
    }
    let bound = _mm256_set1_epi32(target - 1);
    let chunks = buf.len() / 8;
    let ptr = buf.as_ptr() as *const __m256i;
    for chunk in 0..chunks {
        // SAFETY: `chunk < buf.len() / 8`, so the 8 lanes read lie inside
        // `buf`.
        let x = unsafe { _mm256_loadu_si256(ptr.add(chunk)) };
        let mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(x, bound)));
        if mask != 0 {
            return chunk * 8 + mask.trailing_zeros() as usize;
        }
    }
    let tail = chunks * 8;
    tail + find_next_geq_scalar(&buf[tail..], target)
}

// ---------------------------------------------------------------------------
// Float vector similarity: `VectorUtil.dotProduct`/`squareDistance`/`cosine`.
//
// The layout is `PanamaVectorUtilSupport`'s: four 8-lane accumulators over
// 32-element steps (two per sum for cosine, which has three sums in flight),
// one more 8-element step into the first accumulator, then a scalar tail.
// Every product is a fused multiply-add. The AVX2 path and the scalar
// specification share that layout *and* the order the lanes are folded in, so
// they agree bit for bit and the tests can demand exact equality.
//
// Explicit intrinsics rather than an autovectorized loop: a `[f32; 32]`
// accumulator array written lane by lane is what the scalar spec below is,
// and LLVM kept it scalar -- 0.17 ns an element, a third of Lucene's speed.
// ---------------------------------------------------------------------------

/// `a * b + c`, fused where the target has the instruction; see the section
/// comment. A software `fmaf` call would be far slower than either.
#[inline(always)]
fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(any(target_feature = "fma", target_arch = "aarch64")) {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// Four 8-lane groups folded into one (`acc1.add(acc2)...`), then the eight
/// lanes pairwise.
#[inline(always)]
fn fold32(acc: &[f32; 32]) -> f32 {
    let mut v = [0.0f32; 8];
    for (j, slot) in v.iter_mut().enumerate() {
        *slot = (acc[j] + acc[j + 8]) + (acc[j + 16] + acc[j + 24]);
    }
    fold8(&v)
}

#[inline(always)]
fn fold8(v: &[f32; 8]) -> f32 {
    ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]))
}

#[inline(always)]
fn fold16(acc: &[f32; 16]) -> f32 {
    let mut v = [0.0f32; 8];
    for (j, slot) in v.iter_mut().enumerate() {
        *slot = acc[j] + acc[j + 8];
    }
    fold8(&v)
}

/// Vectors this short (`<= 2 * FLOAT_SPECIES.length()`) are not vectorized by
/// `PanamaVectorUtilSupport` either: it runs a plain sequential `fma` loop,
/// and so do these kernels, so their results match Lucene's bit for bit
/// there.
const SHORT_VECTOR: usize = 16;

/// Java's scalar `dotProduct` loop: `res = fma(a[i], b[i], res)`.
fn dot_f32_sequential(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |s, (&x, &y)| fmadd(x, y, s))
}

/// Java's scalar `squareDistance` loop.
fn square_distance_f32_sequential(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |s, (&x, &y)| {
        let d = x - y;
        fmadd(d, d, s)
    })
}

/// Java's scalar `cosine` loop: three sequential `fma` sums.
fn cosine_parts_f32_sequential(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    a.iter()
        .zip(b)
        .fold((0.0, 0.0, 0.0), |(s, n1, n2), (&x, &y)| {
            (fmadd(x, y, s), fmadd(x, x, n1), fmadd(y, y, n2))
        })
}

/// Statistics of the residual `r = query - centroid` that scalar
/// quantization of a query against a centroid starts from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualStats {
    /// `sum(r)`.
    pub sum: f64,
    /// `sum(r^2)`.
    pub sum_sq: f64,
    /// The smallest `r_i` (`f32::MAX` when empty).
    pub min: f32,
    /// The largest `r_i` (`-f32::MAX` when empty).
    pub max: f32,
    /// `r . centroid`.
    pub dot_centroid: f32,
}

/// [`ResidualStats`] of `query - centroid` over their common prefix. Sums
/// run in eight lanes (`f64` for the two that a variance subtracts), so
/// results match [`residual_stats_scalar`] only to rounding.
#[inline]
pub fn residual_stats(query: &[f32], centroid: &[f32]) -> ResidualStats {
    let n = query.len().min(centroid.len());
    let (query, centroid) = (&query[..n], &centroid[..n]);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        // SAFETY: AVX2 and FMA are enabled for this whole build; both slices
        // are `n` long and the kernel reads whole 8-lane chunks below `n`.
        unsafe { residual_stats_avx2(query, centroid) }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    residual_stats_scalar(query, centroid)
}

/// The portable [`residual_stats`]: eight lanes, then their sum.
pub fn residual_stats_scalar(query: &[f32], centroid: &[f32]) -> ResidualStats {
    const L: usize = 8;
    let (mut sum, mut sq) = ([0f64; L], [0f64; L]);
    let (mut lo, mut hi) = ([f32::MAX; L], [-f32::MAX; L]);
    let mut rc = [0f32; L];
    for (i, (&q, &c)) in query.iter().zip(centroid).enumerate() {
        let (j, r) = (i % L, q - c);
        sum[j] += f64::from(r);
        sq[j] += f64::from(r) * f64::from(r);
        lo[j] = lo[j].min(r);
        hi[j] = hi[j].max(r);
        rc[j] += r * c;
    }
    ResidualStats {
        sum: sum.iter().sum(),
        sum_sq: sq.iter().sum(),
        min: lo.iter().fold(f32::MAX, |m, &x| m.min(x)),
        max: hi.iter().fold(-f32::MAX, |m, &x| m.max(x)),
        dot_centroid: rc.iter().sum(),
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn residual_stats_avx2(query: &[f32], centroid: &[f32]) -> ResidualStats {
    use std::arch::x86_64::*;
    let n = query.len();
    let whole = n - n % 8;
    let (pq, pc) = (query.as_ptr(), centroid.as_ptr());
    let (mut sum_lo, mut sum_hi) = (_mm256_setzero_pd(), _mm256_setzero_pd());
    let (mut sq_lo, mut sq_hi) = (_mm256_setzero_pd(), _mm256_setzero_pd());
    let mut lo = _mm256_set1_ps(f32::MAX);
    let mut hi = _mm256_set1_ps(-f32::MAX);
    let mut rc = _mm256_setzero_ps();
    let mut i = 0;
    while i < whole {
        // SAFETY: `i + 8 <= whole <= n`, the length of both slices.
        let (q, c) = unsafe { (_mm256_loadu_ps(pq.add(i)), _mm256_loadu_ps(pc.add(i))) };
        let r = _mm256_sub_ps(q, c);
        let r_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(r));
        let r_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(r, 1));
        sum_lo = _mm256_add_pd(sum_lo, r_lo);
        sum_hi = _mm256_add_pd(sum_hi, r_hi);
        sq_lo = _mm256_fmadd_pd(r_lo, r_lo, sq_lo);
        sq_hi = _mm256_fmadd_pd(r_hi, r_hi, sq_hi);
        lo = _mm256_min_ps(lo, r);
        hi = _mm256_max_ps(hi, r);
        rc = _mm256_fmadd_ps(r, c, rc);
        i += 8;
    }
    // SAFETY (each): the vectors are plain lanes with no padding.
    let sum = unsafe { std::mem::transmute::<[__m256d; 2], [f64; 8]>([sum_lo, sum_hi]) };
    let sq = unsafe { std::mem::transmute::<[__m256d; 2], [f64; 8]>([sq_lo, sq_hi]) };
    let lo = unsafe { std::mem::transmute::<__m256, [f32; 8]>(lo) };
    let hi = unsafe { std::mem::transmute::<__m256, [f32; 8]>(hi) };
    let rc = unsafe { std::mem::transmute::<__m256, [f32; 8]>(rc) };
    let tail = residual_stats_scalar(&query[whole..], &centroid[whole..]);
    ResidualStats {
        sum: sum.iter().sum::<f64>() + tail.sum,
        sum_sq: sq.iter().sum::<f64>() + tail.sum_sq,
        min: lo.iter().fold(tail.min, |m, &x| m.min(x)),
        max: hi.iter().fold(tail.max, |m, &x| m.max(x)),
        dot_centroid: rc.iter().sum::<f32>() + tail.dot_centroid,
    }
}

/// Scalar-quantizes `r = query - centroid` over their common prefix into
/// `out` (which must be at least that long): `r_i` clamped to
/// `[lower, upper]`, level `floor((r_i - lower) * step_inv + 0.5)` clamped
/// to `0..=255` (a NaN level is 0), and returns the levels' sum and the
/// squared norm of the error `r_i - (lower + level * step)`. The vector path
/// fuses the multiply-add, so a level exactly at a half step may round the
/// other way than in [`quantize_residual_scalar`], and its error sum runs in
/// another order.
///
/// # Panics
///
/// When `out` is shorter than the common prefix.
#[inline]
pub fn quantize_residual(
    query: &[f32],
    centroid: &[f32],
    interval: [f32; 2],
    step_inv: f32,
    step: f32,
    out: &mut [u8],
) -> (i32, f32) {
    let n = query.len().min(centroid.len());
    assert!(
        out.len() >= n,
        "quantize_residual: {} outputs for {n} dimensions",
        out.len()
    );
    let (query, centroid, out) = (&query[..n], &centroid[..n], &mut out[..n]);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        // SAFETY: AVX2 and FMA are enabled for this whole build; all three
        // slices are `n` long and the kernel touches whole 8-lane chunks
        // below `n`.
        unsafe { quantize_residual_avx2(query, centroid, interval, step_inv, step, out) }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    quantize_residual_scalar(query, centroid, interval, step_inv, step, out)
}

/// The portable [`quantize_residual`].
///
/// # Panics
///
/// As [`quantize_residual`].
// ARITH: lane indices are below 8; the level sum wraps.
#[allow(clippy::arithmetic_side_effects)]
pub fn quantize_residual_scalar(
    query: &[f32],
    centroid: &[f32],
    interval: [f32; 2],
    step_inv: f32,
    step: f32,
    out: &mut [u8],
) -> (i32, f32) {
    const L: usize = 8;
    let [lower, upper] = interval;
    let mut errors = [0f32; L];
    let mut levels = 0i32;
    for (i, ((o, &q), &c)) in out.iter_mut().zip(query).zip(centroid).enumerate() {
        let r = q - c;
        let x = r.max(lower).min(upper);
        // `max` first, so a NaN level becomes 0, as the vector path's
        // (`clamp` would keep the NaN).
        #[allow(clippy::manual_clamp)]
        let level = ((x - lower) * step_inv + 0.5).floor().max(0.0).min(255.0);
        let e = r - (lower + level * step);
        errors[i % L] += e * e;
        // In 0..=255: exact as an integer.
        let level = level as i32;
        levels = levels.wrapping_add(level);
        *o = level as u8;
    }
    (levels, errors.iter().sum())
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2,fma")]
// ARITH: `i` steps by 8 up to `whole <= n`, the slices' length; the level
// sums wrap.
#[allow(clippy::arithmetic_side_effects)]
unsafe fn quantize_residual_avx2(
    query: &[f32],
    centroid: &[f32],
    interval: [f32; 2],
    step_inv: f32,
    step: f32,
    out: &mut [u8],
) -> (i32, f32) {
    use std::arch::x86_64::*;
    let n = query.len();
    let whole = n - n % 8;
    let (pq, pc, po) = (query.as_ptr(), centroid.as_ptr(), out.as_mut_ptr());
    let [lower, upper] = interval;
    let (a, b) = (_mm256_set1_ps(lower), _mm256_set1_ps(upper));
    let (inv, st, half) = (
        _mm256_set1_ps(step_inv),
        _mm256_set1_ps(step),
        _mm256_set1_ps(0.5),
    );
    let (zero, top) = (_mm256_setzero_ps(), _mm256_set1_ps(255.0));
    let mut err = _mm256_setzero_ps();
    let mut sum = _mm256_setzero_si256();
    let mut i = 0;
    while i < whole {
        // SAFETY: `i + 8 <= whole <= n`, the length of all three slices.
        let (q, c) = unsafe { (_mm256_loadu_ps(pq.add(i)), _mm256_loadu_ps(pc.add(i))) };
        let r = _mm256_sub_ps(q, c);
        // `max` then `min`, as the scalar path, so a NaN residual becomes
        // `lower` in both.
        let x = _mm256_min_ps(_mm256_max_ps(r, a), b);
        // Clamped to 0..=255; `max` returns its second operand, 0, for NaN.
        let level = _mm256_floor_ps(_mm256_fmadd_ps(_mm256_sub_ps(x, a), inv, half));
        let level = _mm256_min_ps(_mm256_max_ps(level, zero), top);
        let e = _mm256_sub_ps(r, _mm256_fmadd_ps(level, st, a));
        err = _mm256_fmadd_ps(e, e, err);
        let li = _mm256_cvttps_epi32(level);
        sum = _mm256_add_epi32(sum, li);
        // Levels to bytes: 32 to 16 to 8 bits within each 128-bit half, so
        // half `k`'s four bytes land in its lowest 32 bits.
        let w = _mm256_packus_epi32(li, li);
        let bytes = _mm256_packus_epi16(w, w);
        let lo4 = _mm256_extract_epi32::<0>(bytes) as u32;
        let hi4 = _mm256_extract_epi32::<4>(bytes) as u32;
        let eight = u64::from(lo4) | (u64::from(hi4) << 32);
        // SAFETY: as above, the 8 bytes at `i` are inside `out`.
        unsafe { std::ptr::write_unaligned(po.add(i) as *mut [u8; 8], eight.to_le_bytes()) };
        i += 8;
    }
    // SAFETY: the vectors are plain lanes with no padding.
    let (err, sum): ([f32; 8], [i32; 8]) = unsafe {
        (
            std::mem::transmute::<__m256, [f32; 8]>(err),
            std::mem::transmute::<__m256i, [i32; 8]>(sum),
        )
    };
    let (tail_sum, tail_err) = quantize_residual_scalar(
        &query[whole..],
        &centroid[whole..],
        interval,
        step_inv,
        step,
        &mut out[whole..],
    );
    (
        sum.iter().fold(tail_sum, |s, &x| s.wrapping_add(x)),
        err.iter().sum::<f32>() + tail_err,
    )
}

/// Fast-scan tables for one bit plane: for each group of four values
/// `query[4g..4g + 4]`, 16 bytes at `out[16g..]` whose entry `v` sums the
/// values whose bit is set in `v`. Every sum must fit a byte (values below
/// 64 always do).
///
/// # Panics
///
/// When `query` is not whole groups of four or `out` is not 16 bytes per
/// group.
#[inline]
pub fn nibble_luts(query: &[u8], out: &mut [u8]) {
    assert!(
        query.len().is_multiple_of(4) && out.len() == query.len() * 4,
        "nibble_luts: {} values for {} table bytes",
        query.len(),
        out.len()
    );
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: AVX2 is enabled for this whole build; the lengths are
        // checked above and the kernel touches whole groups inside them.
        unsafe { nibble_luts_avx2(query, out) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    nibble_luts_scalar(query, out);
}

/// The specification [`nibble_luts`] must match.
///
/// # Panics
///
/// As [`nibble_luts`].
// ARITH: the caller's values keep every sum within a byte.
#[allow(clippy::arithmetic_side_effects)]
pub fn nibble_luts_scalar(query: &[u8], out: &mut [u8]) {
    assert!(query.len().is_multiple_of(4) && out.len() == query.len() * 4);
    for (q, lut) in query.chunks_exact(4).zip(out.chunks_exact_mut(16)) {
        for (v, e) in lut.iter_mut().enumerate() {
            *e = (0..4).filter(|k| v >> k & 1 == 1).map(|k| q[k]).sum();
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
// ARITH: `g` steps by 2 up to `groups`; offsets `4g` and `16g` are inside
// `query` and `out`, whose lengths the caller checked.
#[allow(clippy::arithmetic_side_effects)]
unsafe fn nibble_luts_avx2(query: &[u8], out: &mut [u8]) {
    use std::arch::x86_64::*;
    /// Shuffle indices broadcasting value `k` of a lane's group: lane 0
    /// holds group `g` (bytes 0..4), lane 1 group `g + 1` (bytes 4..8).
    const fn spread(k: i8) -> [i8; 32] {
        let mut x = [0i8; 32];
        let mut i = 0;
        while i < 32 {
            x[i] = if i < 16 { k } else { 4 + k };
            i += 1;
        }
        x
    }
    /// All ones in entry `v` of both lanes when bit `k` of `v` is set.
    const fn mask(k: usize) -> [i8; 32] {
        let mut x = [0i8; 32];
        let mut i = 0;
        while i < 32 {
            if (i % 16) >> k & 1 == 1 {
                x[i] = -1;
            }
            i += 1;
        }
        x
    }
    const SPREAD: [[i8; 32]; 4] = [spread(0), spread(1), spread(2), spread(3)];
    const MASK: [[i8; 32]; 4] = [mask(0), mask(1), mask(2), mask(3)];
    // SAFETY: the constants are 32 bytes each.
    let load = |x: &[i8; 32]| unsafe { _mm256_loadu_si256(x.as_ptr() as *const __m256i) };
    let (s, m) = (
        [
            load(&SPREAD[0]),
            load(&SPREAD[1]),
            load(&SPREAD[2]),
            load(&SPREAD[3]),
        ],
        [
            load(&MASK[0]),
            load(&MASK[1]),
            load(&MASK[2]),
            load(&MASK[3]),
        ],
    );
    let groups = query.len() / 4;
    let (pq, po) = (query.as_ptr(), out.as_mut_ptr());
    let mut g = 0;
    while g + 2 <= groups {
        // SAFETY: groups `g` and `g + 1` are the 8 bytes at `4g` of `query`
        // and the 32 at `16g` of `out`.
        let q = unsafe { _mm_loadl_epi64(pq.add(4 * g) as *const __m128i) };
        let q = _mm256_broadcastsi128_si256(q);
        let mut lut = _mm256_and_si256(_mm256_shuffle_epi8(q, s[0]), m[0]);
        for k in 1..4 {
            lut = _mm256_add_epi8(lut, _mm256_and_si256(_mm256_shuffle_epi8(q, s[k]), m[k]));
        }
        unsafe { _mm256_storeu_si256(po.add(16 * g) as *mut __m256i, lut) };
        g += 2;
    }
    if g < groups {
        nibble_luts_scalar(&query[4 * g..], &mut out[16 * g..]);
    }
}

/// Dot product of the common prefix of `a` and `b`.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len().min(b.len()) <= SHORT_VECTOR {
        return dot_f32_sequential(a, b);
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        // SAFETY: AVX2 and FMA are enabled for this whole build; the kernel
        // reads only whole 8-lane chunks inside both slices.
        unsafe { dot_f32_avx2(a, b) }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    dot_f32_scalar(a, b)
}

/// The specification [`dot_f32`] must match exactly.
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    if a.len().min(b.len()) <= SHORT_VECTOR {
        return dot_f32_sequential(a, b);
    }
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f32; 32];
    let mut i = 0;
    while i + 32 <= n {
        for j in 0..32 {
            acc[j] = fmadd(a[i + j], b[i + j], acc[j]);
        }
        i += 32;
    }
    while i + 8 <= n {
        for j in 0..8 {
            acc[j] = fmadd(a[i + j], b[i + j], acc[j]);
        }
        i += 8;
    }
    let mut sum = fold32(&acc);
    while i < n {
        sum = fmadd(a[i], b[i], sum);
        i += 1;
    }
    sum
}

/// Squared Euclidean distance over the common prefix of `a` and `b`.
#[inline]
pub fn square_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len().min(b.len()) <= SHORT_VECTOR {
        return square_distance_f32_sequential(a, b);
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        // SAFETY: as for `dot_f32`.
        unsafe { square_distance_f32_avx2(a, b) }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    square_distance_f32_scalar(a, b)
}

/// The specification [`square_distance_f32`] must match exactly.
pub fn square_distance_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    if a.len().min(b.len()) <= SHORT_VECTOR {
        return square_distance_f32_sequential(a, b);
    }
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f32; 32];
    let mut i = 0;
    while i + 32 <= n {
        for j in 0..32 {
            let d = a[i + j] - b[i + j];
            acc[j] = fmadd(d, d, acc[j]);
        }
        i += 32;
    }
    while i + 8 <= n {
        for j in 0..8 {
            let d = a[i + j] - b[i + j];
            acc[j] = fmadd(d, d, acc[j]);
        }
        i += 8;
    }
    let mut sum = fold32(&acc);
    while i < n {
        let d = a[i] - b[i];
        sum = fmadd(d, d, sum);
        i += 1;
    }
    sum
}

/// `(a . b, a . a, b . b)` over the common prefix -- the three sums
/// `VectorUtil.cosine` divides.
#[inline]
pub fn cosine_parts_f32(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    if a.len().min(b.len()) <= SHORT_VECTOR {
        return cosine_parts_f32_sequential(a, b);
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        // SAFETY: as for `dot_f32`.
        unsafe { cosine_parts_f32_avx2(a, b) }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    cosine_parts_f32_scalar(a, b)
}

/// The specification [`cosine_parts_f32`] must match exactly: two 8-lane
/// accumulators per sum over 16-element steps, then a scalar tail.
pub fn cosine_parts_f32_scalar(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    if a.len().min(b.len()) <= SHORT_VECTOR {
        return cosine_parts_f32_sequential(a, b);
    }
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (mut s, mut n1, mut n2) = ([0.0f32; 16], [0.0f32; 16], [0.0f32; 16]);
    let mut i = 0;
    while i + 16 <= n {
        for j in 0..16 {
            let (x, y) = (a[i + j], b[i + j]);
            s[j] = fmadd(x, y, s[j]);
            n1[j] = fmadd(x, x, n1[j]);
            n2[j] = fmadd(y, y, n2[j]);
        }
        i += 16;
    }
    let (mut sum, mut norm1, mut norm2) = (fold16(&s), fold16(&n1), fold16(&n2));
    while i < n {
        let (x, y) = (a[i], b[i]);
        sum = fmadd(x, y, sum);
        norm1 = fmadd(x, x, norm1);
        norm2 = fmadd(y, y, norm2);
        i += 1;
    }
    (sum, norm1, norm2)
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    // SAFETY (every load below): the loop guards keep `i + 8 * k + 8 <= n`,
    // and `n` is at most either slice's length.
    while i + 32 <= n {
        for (k, slot) in acc.iter_mut().enumerate() {
            let (x, y) = unsafe {
                (
                    _mm256_loadu_ps(pa.add(i + 8 * k)),
                    _mm256_loadu_ps(pb.add(i + 8 * k)),
                )
            };
            *slot = _mm256_fmadd_ps(x, y, *slot);
        }
        i += 32;
    }
    while i + 8 <= n {
        let (x, y) = unsafe { (_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))) };
        acc[0] = _mm256_fmadd_ps(x, y, acc[0]);
        i += 8;
    }
    let mut sum = fold32(&spill4(&acc));
    while i < n {
        sum = a[i].mul_add(b[i], sum);
        i += 1;
    }
    sum
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn square_distance_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    // SAFETY (every load below): as in `dot_f32_avx2`.
    while i + 32 <= n {
        for (k, slot) in acc.iter_mut().enumerate() {
            let d = unsafe {
                _mm256_sub_ps(
                    _mm256_loadu_ps(pa.add(i + 8 * k)),
                    _mm256_loadu_ps(pb.add(i + 8 * k)),
                )
            };
            *slot = _mm256_fmadd_ps(d, d, *slot);
        }
        i += 32;
    }
    while i + 8 <= n {
        let d = unsafe { _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))) };
        acc[0] = _mm256_fmadd_ps(d, d, acc[0]);
        i += 8;
    }
    let mut sum = fold32(&spill4(&acc));
    while i < n {
        let d = a[i] - b[i];
        sum = d.mul_add(d, sum);
        i += 1;
    }
    sum
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn cosine_parts_f32_avx2(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut s = [_mm256_setzero_ps(); 2];
    let mut n1 = [_mm256_setzero_ps(); 2];
    let mut n2 = [_mm256_setzero_ps(); 2];
    let mut i = 0;
    // SAFETY (every load below): the guard keeps `i + 16 <= n`.
    while i + 16 <= n {
        for k in 0..2 {
            let (x, y) = unsafe {
                (
                    _mm256_loadu_ps(pa.add(i + 8 * k)),
                    _mm256_loadu_ps(pb.add(i + 8 * k)),
                )
            };
            s[k] = _mm256_fmadd_ps(x, y, s[k]);
            n1[k] = _mm256_fmadd_ps(x, x, n1[k]);
            n2[k] = _mm256_fmadd_ps(y, y, n2[k]);
        }
        i += 16;
    }
    let (mut sum, mut norm1, mut norm2) = (
        fold16(&spill2(&s)),
        fold16(&spill2(&n1)),
        fold16(&spill2(&n2)),
    );
    while i < n {
        let (x, y) = (a[i], b[i]);
        sum = x.mul_add(y, sum);
        norm1 = x.mul_add(x, norm1);
        norm2 = y.mul_add(y, norm2);
        i += 1;
    }
    (sum, norm1, norm2)
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
fn spill4(acc: &[std::arch::x86_64::__m256; 4]) -> [f32; 32] {
    // `__m256` is eight `f32`s with no padding; four of them are 32.
    let mut out = [0.0f32; 32];
    for (k, v) in acc.iter().enumerate() {
        // SAFETY: `__m256` and `[f32; 8]` have the same size and every bit
        // pattern is a valid `f32`.
        let lanes: [f32; 8] = unsafe { std::mem::transmute(*v) };
        out[8 * k..8 * k + 8].copy_from_slice(&lanes);
    }
    out
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
fn spill2(acc: &[std::arch::x86_64::__m256; 2]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for (k, v) in acc.iter().enumerate() {
        // SAFETY: as in `spill4`.
        let lanes: [f32; 8] = unsafe { std::mem::transmute(*v) };
        out[8 * k..8 * k + 8].copy_from_slice(&lanes);
    }
    out
}

/// `dst[i] |= src[i]` over the common prefix -- `FixedBitSet.or`'s loop, which
/// C2 vectorizes and unrolls. Written out as four 256-bit registers per step,
/// because LLVM kept the chunked loop at one register and half Lucene's
/// throughput on L2-resident sets.
#[inline]
pub fn or_words(dst: &mut [u64], src: &[u64]) {
    let n = dst.len().min(src.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: AVX2 is enabled for this whole build; the kernel touches
        // only indices below `n`, which both slices have.
        unsafe { or_words_avx2(&mut dst[..n], &src[..n]) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    for (d, s) in dst[..n].iter_mut().zip(&src[..n]) {
        *d |= *s;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn or_words_avx2(dst: &mut [u64], src: &[u64]) {
    use std::arch::x86_64::*;
    let n = dst.len();
    debug_assert_eq!(n, src.len());
    // Scalar words until `dst` is 32-byte aligned: a heap `Vec<u64>` is only
    // guaranteed 16, and a 256-bit access at 16 mod 32 straddles a cache line
    // every other time. Both arrays of a bitset operation are usually
    // allocated alike, so this aligns `src` too; if not, its loads stay
    // unaligned and correct.
    let misaligned_words = (dst.as_ptr() as usize & 31) / 8;
    let mut i = if misaligned_words == 0 {
        0
    } else {
        (4 - misaligned_words).min(n)
    };
    for j in 0..i {
        dst[j] |= src[j];
    }
    let (d, s) = (dst.as_mut_ptr(), src.as_ptr());
    // SAFETY (every access): the guard keeps words `i .. i + 32` below `n`,
    // so each 4-word access at `i + 4 * j` (`j < 8`) lies inside both slices;
    // `d.add(i)` is 32-byte aligned by the peel above, and `i` only moves in
    // steps of 32 words, so every aligned load and store is aligned.
    while i + 32 <= n {
        for j in 0..8 {
            unsafe {
                let dp = d.add(i + 4 * j) as *mut __m256i;
                let v = _mm256_or_si256(
                    _mm256_load_si256(dp),
                    _mm256_loadu_si256(s.add(i + 4 * j) as *const __m256i),
                );
                _mm256_store_si256(dp, v);
            }
        }
        i += 32;
    }
    for j in i..n {
        dst[j] |= src[j];
    }
}

/// Codes [`lut16_accumulate32`] scores per call.
pub const FAST_SCAN_BLOCK: usize = 32;

/// Groups [`lut16_accumulate32`] sums in 16-bit lanes before widening: 256
/// table entries of at most 255 stay below `u16::MAX`.
const FAST_SCAN_FLUSH: usize = 256;

/// Fast-scan table lookup over a block of 32 codes of 4-bit sub-codes:
/// `out[i] += luts[g][code_i[g]]` summed over the groups `g`.
///
/// `luts` holds one 16-entry table per group (`16 * groups` bytes). `codes`
/// holds 16 bytes per group: byte `j` of group `g` carries code `j`'s
/// sub-code `g` in its low nibble and code `j + 16`'s in its high nibble.
/// The vector paths look up 32 entries per shuffle, one table held in a
/// register, which is what makes scanning short codes cheaper than decoding
/// them one at a time.
///
/// # Panics
///
/// When `luts` and `codes` are not the same whole number of 16-byte groups.
#[inline]
pub fn lut16_accumulate32(luts: &[u8], codes: &[u8], out: &mut [u32; FAST_SCAN_BLOCK]) {
    lut16_accumulate32_at(luts, codes, out, false);
}

/// The code length, in 16-byte groups, from which
/// [`lut16_accumulate32_wide`] is faster than [`lut16_accumulate32`]: 64
/// groups is 256 dimensions at four per group. Measured on Sapphire Rapids,
/// 300-dimension blocks scan 11-25% faster on AVX-512 and 100-dimension
/// ones up to 20% faster or not at all, which end to end is no gain.
pub const FAST_SCAN_WIDE_GROUPS: usize = 64;

/// [`lut16_accumulate32`], on AVX-512 where [`crate::cpu::level`] allows it:
/// four groups per shuffle instead of two. The sums are the same. It pays
/// only on long codes; the AVX2 kernel is as fast on short ones (about 25
/// groups at 100 dimensions), so the caller picks by code length.
///
/// # Panics
///
/// As [`lut16_accumulate32`].
#[inline]
pub fn lut16_accumulate32_wide(luts: &[u8], codes: &[u8], out: &mut [u32; FAST_SCAN_BLOCK]) {
    lut16_accumulate32_at(
        luts,
        codes,
        out,
        crate::cpu::level() == crate::cpu::Level::Avx512,
    );
}

#[inline]
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
fn lut16_accumulate32_at(
    luts: &[u8],
    codes: &[u8],
    out: &mut [u32; FAST_SCAN_BLOCK],
    avx512: bool,
) {
    assert!(
        luts.len() == codes.len() && luts.len().is_multiple_of(16),
        "lut16_accumulate32: {} table bytes for {} code bytes",
        luts.len(),
        codes.len()
    );
    for (l, c) in luts
        .chunks(16 * FAST_SCAN_FLUSH)
        .zip(codes.chunks(16 * FAST_SCAN_FLUSH))
    {
        #[cfg(target_arch = "x86_64")]
        if avx512 {
            // SAFETY: `crate::cpu::level` found AVX-512F and AVX-512BW on
            // this CPU; bounds as for the AVX2 kernel below.
            unsafe { lut16_accumulate32_avx512(l, c, out) };
            continue;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // SAFETY: AVX2 is enabled for this whole build; `l` and `c` are
            // the same whole number of 16-byte groups (asserted above), and
            // the kernel reads only whole groups inside them.
            unsafe { lut16_accumulate32_avx2(l, c, out) }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is part of the aarch64 baseline; as above for the
            // bounds.
            unsafe { lut16_accumulate32_neon(l, c, out) }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "avx2"),
            target_arch = "aarch64"
        )))]
        lut16_accumulate32_scalar(l, c, out);
    }
}

/// The specification [`lut16_accumulate32`] must match.
///
/// # Panics
///
/// As [`lut16_accumulate32`].
pub fn lut16_accumulate32_scalar(luts: &[u8], codes: &[u8], out: &mut [u32; FAST_SCAN_BLOCK]) {
    assert!(luts.len() == codes.len() && luts.len().is_multiple_of(16));
    for (lut, group) in luts.chunks_exact(16).zip(codes.chunks_exact(16)) {
        for (j, &b) in group.iter().enumerate() {
            out[j] += u32::from(lut[usize::from(b & 15)]);
            out[j + 16] += u32::from(lut[usize::from(b >> 4)]);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn lut16_accumulate32_avx2(luts: &[u8], codes: &[u8], out: &mut [u32; FAST_SCAN_BLOCK]) {
    use std::arch::x86_64::*;
    let groups = luts.len() / 16;
    debug_assert!(groups <= FAST_SCAN_FLUSH);
    let (pl, pc) = (luts.as_ptr(), codes.as_ptr());
    let nibble = _mm256_set1_epi8(0x0f);
    // Each looked-up byte pair is added as one 16-bit word, whose low byte
    // is an even code's entry and high byte the next code's; a second sum
    // of the high bytes alone separates them at the end. Each 16-bit lane
    // sums at most 128 groups of byte entries, so the high-byte sums and the
    // separated even sums stay below 2^16; the word sums wrap, but only
    // modulo 2^16, which the subtraction cancels.
    let (mut lo_words, mut lo_high) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    let (mut hi_words, mut hi_high) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    let mut g = 0;
    // SAFETY (every load): `g + 2 <= groups` keeps the 32 bytes at `16 * g`
    // inside both slices.
    while g + 2 <= groups {
        let (t, c) = unsafe {
            (
                _mm256_loadu_si256(pl.add(16 * g) as *const __m256i),
                _mm256_loadu_si256(pc.add(16 * g) as *const __m256i),
            )
        };
        // Lane k of `t` is group g + k's table; the shuffle looks up within
        // each 128-bit lane, so lane k of the results is group g + k's.
        let a = _mm256_shuffle_epi8(t, _mm256_and_si256(c, nibble));
        let b = _mm256_shuffle_epi8(t, _mm256_and_si256(_mm256_srli_epi16(c, 4), nibble));
        lo_words = _mm256_add_epi16(lo_words, a);
        lo_high = _mm256_add_epi16(lo_high, _mm256_srli_epi16(a, 8));
        hi_words = _mm256_add_epi16(hi_words, b);
        hi_high = _mm256_add_epi16(hi_high, _mm256_srli_epi16(b, 8));
        g += 2;
    }
    if g < groups {
        // SAFETY: group `g` is the last whole 16 bytes of both slices; the
        // upper lane of the table and codes is zero, and entry 0 of a zero
        // table is zero.
        let (t, c) = unsafe {
            (
                _mm256_zextsi128_si256(_mm_loadu_si128(pl.add(16 * g) as *const __m128i)),
                _mm256_zextsi128_si256(_mm_loadu_si128(pc.add(16 * g) as *const __m128i)),
            )
        };
        let a = _mm256_shuffle_epi8(t, _mm256_and_si256(c, nibble));
        let b = _mm256_shuffle_epi8(t, _mm256_and_si256(_mm256_srli_epi16(c, 4), nibble));
        lo_words = _mm256_add_epi16(lo_words, a);
        lo_high = _mm256_add_epi16(lo_high, _mm256_srli_epi16(a, 8));
        hi_words = _mm256_add_epi16(hi_words, b);
        hi_high = _mm256_add_epi16(hi_high, _mm256_srli_epi16(b, 8));
    }
    // Even codes' sums: the words less their high bytes' sums, shifted.
    let lo_even = _mm256_sub_epi16(lo_words, _mm256_slli_epi16(lo_high, 8));
    let hi_even = _mm256_sub_epi16(hi_words, _mm256_slli_epi16(hi_high, 8));
    // Word `m` of lane `k` holds codes `2m` (even) and `2m + 1` (odd) of
    // group `g + k`'s; the two lanes' groups add up, still within 16 bits
    // (at most 256 entries of a byte), then even and odd interleave into
    // code order and widen to 32 bits.
    let lanes =
        |x: __m256i| _mm_add_epi16(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    let po = out.as_mut_ptr() as *mut __m256i;
    for (half, (even, odd)) in [(lo_even, lo_high), (hi_even, hi_high)]
        .into_iter()
        .enumerate()
    {
        let (e, o) = (lanes(even), lanes(odd));
        let first = _mm256_cvtepu16_epi32(_mm_unpacklo_epi16(e, o));
        let second = _mm256_cvtepu16_epi32(_mm_unpackhi_epi16(e, o));
        // SAFETY: `out` is 32 `u32`s, four whole vectors; `half` is 0 or 1.
        unsafe {
            let at = po.add(2 * half);
            _mm256_storeu_si256(at, _mm256_add_epi32(_mm256_loadu_si256(at), first));
            let at = po.add(2 * half + 1);
            _mm256_storeu_si256(at, _mm256_add_epi32(_mm256_loadu_si256(at), second));
        }
    }
}

/// The AVX2 kernel (`lut16_accumulate32_avx2`) four groups at a time: lane `k` of the
/// 512-bit table and code registers is group `g + k`, and a masked load
/// zeroes the lanes past the last group (entry 0 of a zero table is zero).
/// Each 16-bit lane sums at most 64 of the 256 groups, and the four lanes
/// fold into at most 256 byte entries, so the separated sums fit in 16 bits
/// (the word sums wrap modulo 2^16, as in the AVX2 kernel, and the
/// subtraction cancels that).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn lut16_accumulate32_avx512(luts: &[u8], codes: &[u8], out: &mut [u32; FAST_SCAN_BLOCK]) {
    use std::arch::x86_64::*;
    let groups = luts.len() / 16;
    debug_assert!(groups <= FAST_SCAN_FLUSH);
    let (pl, pc) = (luts.as_ptr(), codes.as_ptr());
    let nibble = _mm512_set1_epi8(0x0f);
    let (mut lo_words, mut lo_high) = (_mm512_setzero_si512(), _mm512_setzero_si512());
    let (mut hi_words, mut hi_high) = (_mm512_setzero_si512(), _mm512_setzero_si512());
    let mut g = 0;
    while g < groups {
        let left = groups - g;
        // SAFETY (both branches): `g < groups`, and the full load reads 64
        // bytes only when four whole groups remain; the masked load reads
        // the `16 * left` bytes that remain and no others.
        let (t, c) = if left >= 4 {
            unsafe {
                (
                    _mm512_loadu_si512(pl.add(16 * g).cast()),
                    _mm512_loadu_si512(pc.add(16 * g).cast()),
                )
            }
        } else {
            let mask: __mmask64 = (1u64 << (16 * left)) - 1;
            unsafe {
                (
                    _mm512_maskz_loadu_epi8(mask, pl.add(16 * g).cast()),
                    _mm512_maskz_loadu_epi8(mask, pc.add(16 * g).cast()),
                )
            }
        };
        let a = _mm512_shuffle_epi8(t, _mm512_and_si512(c, nibble));
        let b = _mm512_shuffle_epi8(t, _mm512_and_si512(_mm512_srli_epi16(c, 4), nibble));
        lo_words = _mm512_add_epi16(lo_words, a);
        lo_high = _mm512_add_epi16(lo_high, _mm512_srli_epi16(a, 8));
        hi_words = _mm512_add_epi16(hi_words, b);
        hi_high = _mm512_add_epi16(hi_high, _mm512_srli_epi16(b, 8));
        g += 4;
    }
    let lo_even = _mm512_sub_epi16(lo_words, _mm512_slli_epi16(lo_high, 8));
    let hi_even = _mm512_sub_epi16(hi_words, _mm512_slli_epi16(hi_high, 8));
    let lanes = |x: __m512i| {
        let y = _mm256_add_epi16(_mm512_castsi512_si256(x), _mm512_extracti64x4_epi64(x, 1));
        _mm_add_epi16(_mm256_castsi256_si128(y), _mm256_extracti128_si256(y, 1))
    };
    let po = out.as_mut_ptr() as *mut __m256i;
    for (half, (even, odd)) in [(lo_even, lo_high), (hi_even, hi_high)]
        .into_iter()
        .enumerate()
    {
        let (e, o) = (lanes(even), lanes(odd));
        let first = _mm256_cvtepu16_epi32(_mm_unpacklo_epi16(e, o));
        let second = _mm256_cvtepu16_epi32(_mm_unpackhi_epi16(e, o));
        // SAFETY: `out` is 32 `u32`s, four whole vectors; `half` is 0 or 1.
        unsafe {
            let at = po.add(2 * half);
            _mm256_storeu_si256(at, _mm256_add_epi32(_mm256_loadu_si256(at), first));
            let at = po.add(2 * half + 1);
            _mm256_storeu_si256(at, _mm256_add_epi32(_mm256_loadu_si256(at), second));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn lut16_accumulate32_neon(luts: &[u8], codes: &[u8], out: &mut [u32; FAST_SCAN_BLOCK]) {
    use std::arch::aarch64::*;
    let groups = luts.len() / 16;
    debug_assert!(groups <= FAST_SCAN_FLUSH);
    let (pl, pc) = (luts.as_ptr(), codes.as_ptr());
    let nibble = vdupq_n_u8(0x0f);
    // 16-bit sums for codes 0..8, 8..16, 16..24 and 24..32.
    let mut acc = [vdupq_n_u16(0); 4];
    for g in 0..groups {
        // SAFETY: group `g < groups` is 16 bytes inside both slices.
        let (t, c) = unsafe { (vld1q_u8(pl.add(16 * g)), vld1q_u8(pc.add(16 * g))) };
        let a = vqtbl1q_u8(t, vandq_u8(c, nibble));
        let b = vqtbl1q_u8(t, vshrq_n_u8(c, 4));
        acc[0] = vaddw_u8(acc[0], vget_low_u8(a));
        acc[1] = vaddw_high_u8(acc[1], a);
        acc[2] = vaddw_u8(acc[2], vget_low_u8(b));
        acc[3] = vaddw_high_u8(acc[3], b);
    }
    for (k, v) in acc.iter().enumerate() {
        let mut lanes = [0u16; 8];
        // SAFETY: `lanes` is eight `u16`s.
        unsafe { vst1q_u16(lanes.as_mut_ptr(), *v) };
        for (j, x) in lanes.iter().enumerate() {
            out[8 * k + j] += u32::from(*x);
        }
    }
}

/// For every byte value, the positions of its set bits, ascending, padded
/// with zeros to eight.
static BYTE_BITS: [[u8; 8]; 256] = {
    let mut table = [[0u8; 8]; 256];
    let mut b = 0;
    while b < 256 {
        let mut n = 0;
        let mut bit = 0;
        while bit < 8 {
            if b & (1 << bit) != 0 {
                table[b][n] = bit as u8;
                n += 1;
            }
            bit += 1;
        }
        b += 1;
    }
    table
};

/// The doc ids of a dense postings block: `base + i` for every set bit `i` of
/// `words`, ascending, written to `out` until it is full (at most 256
/// entries). Returns how many were written -- `out.len()` unless `words` has
/// fewer set bits, which for a whole block means corruption and is the
/// caller's to report.
///
/// This is `Lucene104PostingsReader`'s unary doc encoding (`bitsPerValue <
/// 0`). Peeling one bit per iteration (`trailing_zeros`, then clear it) makes
/// one serial step per document; a word with more than a few bits set is
/// instead worked a byte at a time -- one table load gives up to eight
/// positions, written as a fixed-width 8-lane store (which LLVM vectorizes)
/// whose surplus lanes the next byte overwrites. Sparse words keep the bit
/// loop, which touches only their set bits.
pub fn expand_bitset(words: &[u64], base: i32, out: &mut [i32]) -> usize {
    let cap = out.len().min(256);
    // Every write below happens before its `found >= cap` test, so an empty
    // output must not reach them.
    if cap == 0 {
        return 0;
    }
    // Slack for the surplus of the last store: eight lanes for a byte, 64
    // for a full word.
    let mut buf = [0i32; 256 + 64];
    let mut found = 0usize;
    'words: for (w, &word) in words.iter().enumerate() {
        if word == 0 {
            continue;
        }
        // Wrapping, like every doc id derived from on-disk deltas.
        let word_base = base.wrapping_add((w * 64) as i32);
        if word == u64::MAX {
            // A full word -- all four of a consecutive (`bitsPerValue == 0`)
            // block, which the postings cursor keeps as a bit set the way
            // Lucene does -- is 64 consecutive ids: one run LLVM vectorizes,
            // instead of eight table lookups. `found < cap <= 256`, so the
            // run fits the slack.
            let dst: &mut [i32; 64] = (&mut buf[found..found + 64]).try_into().unwrap();
            for (j, d) in dst.iter_mut().enumerate() {
                *d = word_base.wrapping_add(j as i32);
            }
            found += 64;
            if found >= cap {
                found = cap;
                break 'words;
            }
            continue;
        }
        if word.count_ones() <= 4 {
            let mut bits = word;
            while bits != 0 {
                buf[found] = word_base.wrapping_add(bits.trailing_zeros() as i32);
                found += 1;
                if found >= cap {
                    break 'words;
                }
                bits &= bits - 1;
            }
            continue;
        }
        // No `byte == 0` test: at a postings block's typical density about one
        // byte in eight is empty, too often to predict, and an empty byte's
        // row writes nothing `found` keeps.
        for (k, byte) in word.to_le_bytes().into_iter().enumerate() {
            let at = word_base.wrapping_add((k * 8) as i32);
            let lanes = &BYTE_BITS[byte as usize];
            // `found < cap <= 256` here (checked below), so `found + 8 <= 263`.
            let dst: &mut [i32; 8] = (&mut buf[found..found + 8]).try_into().unwrap();
            for j in 0..8 {
                dst[j] = at.wrapping_add(lanes[j] as i32);
            }
            found += byte.count_ones() as usize;
            if found >= cap {
                found = cap;
                break 'words;
            }
        }
    }
    // Only what was found: `out[found..]` is left as the caller had it, as
    // the scalar specification leaves it.
    out[..found].copy_from_slice(&buf[..found]);
    found
}

/// The specification [`expand_bitset`] must match: one bit at a time.
pub fn expand_bitset_scalar(words: &[u64], base: i32, out: &mut [i32]) -> usize {
    let cap = out.len().min(256);
    let mut found = 0usize;
    for (w, &word) in words.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            if found == cap {
                return found;
            }
            out[found] = base.wrapping_add((w * 64) as i32 + bits.trailing_zeros() as i32);
            found += 1;
            bits &= bits - 1;
        }
    }
    found
}

/// Starts loading the cache line holding `slice[i]`, so a lookup that will
/// read it soon does not stall on the miss. A hint only: nothing is read,
/// an index out of bounds does nothing, and on targets without a prefetch
/// instruction the call is empty.
#[inline(always)]
pub fn prefetch<T>(slice: &[T], i: usize) {
    #[cfg(target_arch = "x86_64")]
    if let Some(r) = slice.get(i) {
        // SAFETY: `PREFETCHT0` is a hint that never faults and reads nothing
        // into the program; the address is an element of `slice` anyway.
        unsafe {
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                (r as *const T).cast::<i8>(),
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (slice, i);
}

/// `sum(q[i] * c[i])` over the common prefix of `q` and `c`, with `q` read
/// as unsigned bytes no greater than 127 and `c` as signed bytes: the
/// 8-bit dot product a coarse ranker scores many codes with. Keeping the
/// query to 7 bits lets AVX2 multiply 32 pairs per `maddubs` without
/// saturating (two products sum to at most `2 * 127 * 128 < 2^15`); NEON
/// widens pairs of products into 16 bits the same way.
///
/// Query bytes above 127 give unspecified (but memory-safe) results.
#[inline]
pub fn dot_u7_i8(q: &[u8], c: &[u8]) -> i32 {
    let n = q.len().min(c.len());
    let (q, c) = (&q[..n], &c[..n]);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: AVX2 is enabled for this whole build; both slices have
        // length `n`.
        unsafe { dot_u7_i8_avx2(q, c) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the aarch64 baseline; both slices have
        // length `n`.
        unsafe { dot_u7_i8_neon(q, c) }
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        target_arch = "aarch64"
    )))]
    dot_u7_i8_scalar(q, c)
}

/// The specification [`dot_u7_i8`] matches for query bytes up to 127.
pub fn dot_u7_i8_scalar(q: &[u8], c: &[u8]) -> i32 {
    q.iter().zip(c).fold(0i32, |s, (&x, &y)| {
        s.wrapping_add(i32::from(x) * i32::from(y as i8))
    })
}

/// # Safety
///
/// AVX2 must be available and `q.len() == c.len()`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_u7_i8_avx2(q: &[u8], c: &[u8]) -> i32 {
    use std::arch::x86_64::*;
    let n = q.len();
    let (pq, pc) = (q.as_ptr(), c.as_ptr());
    let ones = _mm256_set1_epi16(1);
    let (mut a0, mut a1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    let mut i = 0;
    // SAFETY (every load): `i + 32 <= n` (or `i + 64 <= n`) keeps the loads
    // inside both slices.
    while i + 64 <= n {
        let (x0, y0, x1, y1) = unsafe {
            (
                _mm256_loadu_si256(pq.add(i) as *const __m256i),
                _mm256_loadu_si256(pc.add(i) as *const __m256i),
                _mm256_loadu_si256(pq.add(i + 32) as *const __m256i),
                _mm256_loadu_si256(pc.add(i + 32) as *const __m256i),
            )
        };
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, y0), ones));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(x1, y1), ones));
        i += 64;
    }
    if i + 32 <= n {
        let (x, y) = unsafe {
            (
                _mm256_loadu_si256(pq.add(i) as *const __m256i),
                _mm256_loadu_si256(pc.add(i) as *const __m256i),
            )
        };
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(x, y), ones));
        i += 32;
    }
    let acc = _mm256_add_epi32(a0, a1);
    let s = _mm_add_epi32(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256(acc, 1),
    );
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    _mm_cvtsi128_si32(s).wrapping_add(dot_u7_i8_scalar(&q[i..], &c[i..]))
}

/// # Safety
///
/// `q.len() == c.len()`.
#[cfg(target_arch = "aarch64")]
unsafe fn dot_u7_i8_neon(q: &[u8], c: &[u8]) -> i32 {
    use std::arch::aarch64::*;
    let n = q.len();
    let (pq, pc) = (q.as_ptr(), c.as_ptr());
    let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
    let mut i = 0;
    // SAFETY (every load): `i + 32 <= n` (or `i + 16 <= n`) keeps the loads
    // inside both slices.
    while i + 32 <= n {
        let (x0, y0, x1, y1) = unsafe {
            (
                vreinterpretq_s8_u8(vld1q_u8(pq.add(i))),
                vld1q_s8(pc.add(i) as *const i8),
                vreinterpretq_s8_u8(vld1q_u8(pq.add(i + 16))),
                vld1q_s8(pc.add(i + 16) as *const i8),
            )
        };
        // Query bytes are at most 127, so they read the same as `i8`; two
        // products of at most 127 * 128 fit an `i16`.
        let p0 = vmlal_high_s8(vmull_s8(vget_low_s8(x0), vget_low_s8(y0)), x0, y0);
        let p1 = vmlal_high_s8(vmull_s8(vget_low_s8(x1), vget_low_s8(y1)), x1, y1);
        a0 = vpadalq_s16(a0, p0);
        a1 = vpadalq_s16(a1, p1);
        i += 32;
    }
    if i + 16 <= n {
        let (x, y) = unsafe {
            (
                vreinterpretq_s8_u8(vld1q_u8(pq.add(i))),
                vld1q_s8(pc.add(i) as *const i8),
            )
        };
        a0 = vpadalq_s16(
            a0,
            vmlal_high_s8(vmull_s8(vget_low_s8(x), vget_low_s8(y)), x, y),
        );
        i += 16;
    }
    vaddvq_s32(vaddq_s32(a0, a1)).wrapping_add(dot_u7_i8_scalar(&q[i..], &c[i..]))
}

/// [`dot_u7_i8`] of one query against four code rows at once, sharing the
/// query loads and the final horizontal sums. Each row uses its common
/// prefix with `q`.
#[inline]
pub fn dot_u7_i8_x4(q: &[u8], c: [&[u8]; 4]) -> [i32; 4] {
    let n = c.iter().fold(q.len(), |n, r| n.min(r.len()));
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: AVX2 is enabled for this whole build; the kernel reads
        // only the first `n` bytes of each slice, all at least `n` long.
        unsafe { dot_u7_i8_x4_avx2(q, c, n) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the aarch64 baseline; as above for the
        // bounds.
        unsafe { dot_u7_i8_x4_neon(q, c, n) }
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        target_arch = "aarch64"
    )))]
    c.map(|r| dot_u7_i8_scalar(&q[..n], &r[..n]))
}

/// # Safety
///
/// AVX2 must be available, and `q` and every row at least `n` bytes long.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_u7_i8_x4_avx2(q: &[u8], c: [&[u8]; 4], n: usize) -> [i32; 4] {
    use std::arch::x86_64::*;
    let pq = q.as_ptr();
    let p = c.map(<[u8]>::as_ptr);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_si256(); 4];
    let mut i = 0;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` keeps every load inside its slice.
        let x = unsafe { _mm256_loadu_si256(pq.add(i) as *const __m256i) };
        for (a, &pr) in acc.iter_mut().zip(&p) {
            // SAFETY: as above.
            let y = unsafe { _mm256_loadu_si256(pr.add(i) as *const __m256i) };
            *a = _mm256_add_epi32(*a, _mm256_madd_epi16(_mm256_maddubs_epi16(x, y), ones));
        }
        i += 32;
    }
    // Lanes 0..4 of `h` hold rows 0..4's sums over the low halves, lanes
    // 4..8 over the high halves.
    let h = _mm256_hadd_epi32(
        _mm256_hadd_epi32(acc[0], acc[1]),
        _mm256_hadd_epi32(acc[2], acc[3]),
    );
    let s = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
    let mut out = [0i32; 4];
    // SAFETY: `out` is 16 bytes.
    unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, s) };
    for (o, r) in out.iter_mut().zip(c) {
        *o = o.wrapping_add(dot_u7_i8_scalar(&q[i..n], &r[i..n]));
    }
    out
}

/// # Safety
///
/// `q` and every row at least `n` bytes long.
#[cfg(target_arch = "aarch64")]
unsafe fn dot_u7_i8_x4_neon(q: &[u8], c: [&[u8]; 4], n: usize) -> [i32; 4] {
    use std::arch::aarch64::*;
    let pq = q.as_ptr();
    let p = c.map(<[u8]>::as_ptr);
    let mut acc = [vdupq_n_s32(0); 4];
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n` keeps every load inside its slice.
        let x = unsafe { vreinterpretq_s8_u8(vld1q_u8(pq.add(i))) };
        let xl = vget_low_s8(x);
        for (a, &pr) in acc.iter_mut().zip(&p) {
            // SAFETY: as above.
            let y = unsafe { vld1q_s8(pr.add(i) as *const i8) };
            *a = vpadalq_s16(*a, vmlal_high_s8(vmull_s8(xl, vget_low_s8(y)), x, y));
        }
        i += 16;
    }
    let s = vpaddq_s32(vpaddq_s32(acc[0], acc[1]), vpaddq_s32(acc[2], acc[3]));
    let mut out = [0i32; 4];
    // SAFETY: `out` is 16 bytes.
    unsafe { vst1q_s32(out.as_mut_ptr(), s) };
    for (o, r) in out.iter_mut().zip(c) {
        *o = o.wrapping_add(dot_u7_i8_scalar(&q[i..n], &r[i..n]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The float kernels agree with their scalar specifications bit for bit
    /// at every length around the 8/16/32-lane steps, on unequal lengths
    /// (the common prefix), and with non-finite inputs.
    #[test]
    fn float_kernels_match_the_scalar_specification_exactly() {
        let mut s = 0xDEAD_BEEF_0BAD_F00Du64;
        let mut f = || (xorshift(&mut s) >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
        for len in (0..80).chain([127, 128, 129, 767, 768, 1024]) {
            let a: Vec<f32> = (0..len).map(|_| f()).collect();
            let b: Vec<f32> = (0..len + len % 3).map(|_| f()).collect();
            assert_eq!(
                dot_f32(&a, &b).to_bits(),
                dot_f32_scalar(&a, &b).to_bits(),
                "dot {len}"
            );
            assert_eq!(
                square_distance_f32(&a, &b).to_bits(),
                square_distance_f32_scalar(&a, &b).to_bits(),
                "l2 {len}"
            );
            let (x, y) = (cosine_parts_f32(&a, &b), cosine_parts_f32_scalar(&a, &b));
            assert_eq!(
                (x.0.to_bits(), x.1.to_bits(), x.2.to_bits()),
                (y.0.to_bits(), y.1.to_bits(), y.2.to_bits()),
                "cos {len}"
            );
        }
        let a = [1.0f32, f32::INFINITY, 2.0];
        let b = [f32::NAN, 1.0, 2.0];
        assert!(dot_f32(&a, &b).is_nan());
        assert!(square_distance_f32(&a, &b).is_nan());
        assert!(cosine_parts_f32(&a, &b).0.is_nan());
        assert_eq!(dot_f32(&[], &[]), 0.0);
    }

    #[test]
    fn geq_mask8_matches_the_scalar_specification() {
        let mut s = 0xFEED_FACE_CAFE_BEEFu64;
        for _ in 0..5000 {
            let chunk: [i32; 8] = std::array::from_fn(|_| (xorshift(&mut s) % 64) as i32 - 32);
            let target = (xorshift(&mut s) % 70) as i32 - 35;
            assert_eq!(geq_mask8(&chunk, target), geq_mask8_scalar(&chunk, target));
        }
        let chunk = [i32::MIN, -1, 0, 1, i32::MAX, 5, 6, 7];
        for target in [i32::MIN, i32::MAX, 0] {
            assert_eq!(geq_mask8(&chunk, target), geq_mask8_scalar(&chunk, target));
        }
    }

    #[test]
    fn find_next_geq_v1_matches_the_scalar_specification_on_ascending_input() {
        let mut s = 0x2468_ACE0_1357_9BDFu64;
        for len in [0usize, 1, 7, 8, 9, 10, 17, 18, 19, 64, 255, 256] {
            let mut doc = -5i32;
            let buf: Vec<i32> = (0..len)
                .map(|_| {
                    doc += 1 + (xorshift(&mut s) % 4) as i32;
                    doc
                })
                .collect();
            for target in -10..doc + 3 {
                assert_eq!(
                    find_next_geq_v1(&buf, target),
                    find_next_geq_scalar(&buf, target),
                    "len {len} target {target}"
                );
            }
        }
        assert_eq!(find_next_geq_v1(&[i32::MIN; 12], i32::MIN), 0);
    }

    #[test]
    fn count_less_than_matches_the_scalar_specification() {
        let mut s = 0x0BAD_F00D_DEAD_BEEFu64;
        for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 100, 256] {
            let buf: Vec<i32> = (0..len)
                .map(|_| (xorshift(&mut s) % 200) as i32 - 100)
                .collect();
            for target in [i32::MIN, -101, -100, -1, 0, 1, 50, 99, 100, i32::MAX] {
                assert_eq!(
                    count_less_than(&buf, target),
                    count_less_than_scalar(&buf, target),
                    "len {len} target {target}"
                );
            }
        }
        // Every lane below: the count is the whole length, not a wrapped one.
        assert_eq!(count_less_than(&[i32::MIN; 256], i32::MAX), 256);
    }

    #[test]
    fn or_words_matches_a_plain_loop_at_every_length() {
        let mut s = 0x0F0F_1234_5678_9ABCu64;
        for len in [0usize, 1, 3, 15, 16, 17, 31, 32, 33, 100, 1000] {
            let a: Vec<u64> = (0..len).map(|_| xorshift(&mut s)).collect();
            let b: Vec<u64> = (0..len + 5).map(|_| xorshift(&mut s)).collect();
            let mut got = a.clone();
            or_words(&mut got, &b);
            let want: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x | y).collect();
            assert_eq!(got, want, "len {len}");
            // A shorter source leaves the rest of `dst` alone.
            let mut got = b.clone();
            or_words(&mut got, &a);
            assert_eq!(got[..len], want[..], "len {len}");
            assert_eq!(got[len..], b[len..], "len {len}");
            // Every destination alignment the peel handles: start the slices
            // one to three words into their allocations.
            for off in 1..4usize.min(len.max(1)) {
                let mut got = a.clone();
                or_words(&mut got[off..], &b[off..len]);
                assert_eq!(got[..off], a[..off], "len {len} off {off}");
                assert_eq!(got[off..], want[off..], "len {len} off {off}");
            }
        }
    }

    #[test]
    fn expand_bitset_matches_the_scalar_specification() {
        let mut s = 0x1234_5678_9abc_def1u64;
        for round in 0..2000 {
            let n_words = 1 + (xorshift(&mut s) % 128) as usize;
            // Densities from sparse to all-ones, so blocks hit 256 early, late
            // or never.
            let density = xorshift(&mut s) % 5;
            let words: Vec<u64> = (0..n_words)
                .map(|_| match density {
                    0 => xorshift(&mut s) & xorshift(&mut s) & xorshift(&mut s),
                    1 => xorshift(&mut s),
                    2 => xorshift(&mut s) | xorshift(&mut s),
                    3 => u64::MAX,
                    _ => 0,
                })
                .collect();
            let base = (xorshift(&mut s) as i32) >> (round % 32);
            let (mut a, mut b) = ([7i32; 256], [7i32; 256]);
            let fa = expand_bitset(&words, base, &mut a);
            let fb = expand_bitset_scalar(&words, base, &mut b);
            assert_eq!(fa, fb, "round {round}");
            assert_eq!(a[..fa], b[..fb], "round {round}");
            // A partial output, as a decode that starts mid-block fills.
            let cap = (xorshift(&mut s) % 257) as usize;
            let (mut a, mut b) = ([7i32; 256], [7i32; 256]);
            let fa = expand_bitset(&words, base, &mut a[..cap]);
            let fb = expand_bitset_scalar(&words, base, &mut b[..cap]);
            assert_eq!(fa, fb, "round {round} cap {cap}");
            assert_eq!(a[..fa], b[..fb], "round {round} cap {cap}");
        }
        // Full words (the 64-at-once path) between sparse and dense ones, cut
        // off at every kind of output length, including inside a full word.
        // The second set has fewer bits than most outputs: what lies past the
        // found count must be left alone, as the specification leaves it.
        let dense = [
            u64::MAX,
            0x0F,
            u64::MAX,
            0xFFFF_0000_FFFF_0000,
            u64::MAX,
            u64::MAX,
        ];
        let short = [u64::MAX, 0x0F];
        for words in [&dense[..], &short[..]] {
            for cap in [0usize, 1, 63, 64, 65, 68, 100, 131, 132, 200, 255, 256] {
                let (mut a, mut b) = ([7i32; 256], [7i32; 256]);
                let fa = expand_bitset(words, -30, &mut a[..cap]);
                let fb = expand_bitset_scalar(words, -30, &mut b[..cap]);
                assert_eq!((fa, &a[..]), (fb, &b[..]), "cap {cap}");
            }
        }
        // Exactly 256 bits, ending on the last bit of the last word.
        let words = [u64::MAX; 4];
        let mut out = [0i32; 256];
        assert_eq!(expand_bitset(&words, 10, &mut out), 256);
        // An empty output takes nothing, whatever the words hold.
        assert_eq!(expand_bitset(&[0b1011], 0, &mut []), 0);
        assert_eq!(expand_bitset_scalar(&[0b1011], 0, &mut []), 0);
        assert_eq!(out[255], 10 + 255);
    }

    /// The fast-scan lookup matches its scalar specification for every
    /// group count around the two- and four-group steps and the 16-bit
    /// flush, on random tables and on all-0xFF ones, the largest sums a
    /// flush can hold. The AVX-512 kernel is checked directly where the CPU
    /// has it, whatever `LUCENE_RUST_SIMD` says.
    #[test]
    fn lut16_accumulate32_matches_the_scalar_specification() {
        #[cfg(target_arch = "x86_64")]
        let avx512 = std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw");
        #[cfg(not(target_arch = "x86_64"))]
        let avx512 = false;
        if !avx512 {
            eprintln!("lut16_accumulate32: no AVX-512 here, its kernel is not checked");
        }
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let random = (0..40).chain([255, 256, 257, 600]).map(|g| (g, false));
        for (groups, saturated) in random.chain([(64, true), (256, true), (257, true), (600, true)])
        {
            let luts: Vec<u8> = (0..16 * groups)
                .map(|_| {
                    if saturated {
                        0xFF
                    } else {
                        xorshift(&mut s) as u8
                    }
                })
                .collect();
            let codes: Vec<u8> = (0..16 * groups).map(|_| xorshift(&mut s) as u8).collect();
            let mut spec = [7u32; FAST_SCAN_BLOCK];
            lut16_accumulate32_scalar(&luts, &codes, &mut spec);
            let mut fast = [7u32; FAST_SCAN_BLOCK];
            lut16_accumulate32(&luts, &codes, &mut fast);
            assert_eq!(fast, spec, "groups {groups}");
            let mut wide = [7u32; FAST_SCAN_BLOCK];
            lut16_accumulate32_wide(&luts, &codes, &mut wide);
            assert_eq!(wide, spec, "groups {groups}, wide");
            if avx512 {
                let mut forced = [7u32; FAST_SCAN_BLOCK];
                lut16_accumulate32_at(&luts, &codes, &mut forced, true);
                assert_eq!(forced, spec, "groups {groups}, AVX-512");
            }
        }
    }

    #[test]
    #[should_panic(expected = "lut16_accumulate32")]
    fn lut16_accumulate32_rejects_mismatched_lengths() {
        lut16_accumulate32(&[0; 16], &[0; 32], &mut [0; FAST_SCAN_BLOCK]);
    }

    #[test]
    fn nibble_luts_match_the_scalar_specification() {
        let mut s = 0x0bad_cafe_f00d_1234u64;
        for groups in 0..40 {
            let query: Vec<u8> = (0..4 * groups)
                .map(|_| (xorshift(&mut s) % 64) as u8)
                .collect();
            let mut fast = vec![0u8; 16 * groups];
            let mut spec = vec![1u8; 16 * groups];
            nibble_luts(&query, &mut fast);
            nibble_luts_scalar(&query, &mut spec);
            assert_eq!(fast, spec, "groups {groups}");
        }
    }

    fn floats(s: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| ((xorshift(s) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale)
            .collect()
    }

    #[test]
    fn residual_kernels_match_the_scalar_specification() {
        let mut s = 0x5eed_1234_abcd_0001u64;
        for n in (0..40).chain([100, 300, 768, 1021]) {
            let q = floats(&mut s, n, 4.0);
            let c = floats(&mut s, n, 2.0);
            let (fast, spec) = (residual_stats(&q, &c), residual_stats_scalar(&q, &c));
            assert_eq!((fast.min, fast.max), (spec.min, spec.max), "n {n}");
            let close = |a: f64, b: f64| (a - b).abs() <= 1e-4 * (1.0 + b.abs());
            assert!(
                close(fast.sum, spec.sum) && close(fast.sum_sq, spec.sum_sq),
                "n {n}"
            );
            assert!(
                close(f64::from(fast.dot_centroid), f64::from(spec.dot_centroid)),
                "n {n}"
            );
            for bits in [1u32, 2, 4, 7, 8] {
                let (a, b) = (fast.min * 0.8, fast.max * 0.8);
                let steps = ((1 << bits) - 1) as f32;
                let (inv, step) = if b > a {
                    (steps / (b - a), (b - a) / steps)
                } else {
                    (0.0, 0.0)
                };
                let (mut fo, mut so) = (vec![0u8; n], vec![0u8; n]);
                let (fl, fe) = quantize_residual(&q, &c, [a, b], inv, step, &mut fo);
                let (sl, se) = quantize_residual_scalar(&q, &c, [a, b], inv, step, &mut so);
                // Fused and separate rounding can split a level at a half
                // step; anything else must agree.
                let differ = fo.iter().zip(&so).filter(|(x, y)| x != y).count();
                assert!(
                    differ <= n / 50,
                    "n {n} bits {bits}: {differ} levels differ"
                );
                assert!(fo.iter().zip(&so).all(|(x, y)| x.abs_diff(*y) <= 1));
                assert!(
                    (fl - sl).unsigned_abs() as usize <= differ,
                    "n {n} bits {bits}"
                );
                assert!(
                    close(f64::from(fe), f64::from(se)) || differ > 0,
                    "n {n} bits {bits}"
                );
            }
        }
    }

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    #[test]
    fn prefix_sum_matches_the_scalar_specification() {
        let mut s = 0x1234_5678_9abc_def0u64;
        for round in 0..200 {
            let mut deltas = [0u32; 256];
            for d in deltas.iter_mut() {
                let x = xorshift(&mut s);
                // Mostly small deltas, sometimes huge ones that wrap.
                *d = if round % 7 == 0 {
                    x as u32
                } else {
                    (x % 1000) as u32
                };
            }
            let base = xorshift(&mut s) as i32;
            let (mut fast, mut spec) = ([0i32; 256], [0i32; 256]);
            prefix_sum_256(base, &deltas, &mut fast);
            prefix_sum_256_scalar(base, &deltas, &mut spec);
            assert_eq!(fast, spec, "round {round}");
        }
    }

    #[test]
    fn find_next_geq_matches_the_scalar_specification() {
        let mut s = 0xfeed_beef_u64;
        for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 255, 256] {
            let mut buf: Vec<i32> = (0..len)
                .map(|_| (xorshift(&mut s) % 10_000) as i32)
                .collect();
            buf.sort_unstable();
            let targets = buf.iter().flat_map(|&d| [d - 1, d, d + 1]).chain([
                i32::MIN,
                i32::MAX,
                0,
                -5,
                20_000,
            ]);
            for t in targets {
                assert_eq!(
                    find_next_geq(&buf, t),
                    find_next_geq_scalar(&buf, t),
                    "len {len} target {t}"
                );
            }
        }
    }

    #[test]
    fn dot_u7_i8_matches_scalar() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut byte = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u8
        };
        for n in [
            0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 300, 1000,
        ] {
            let q: Vec<u8> = (0..n).map(|_| byte() & 0x7f).collect();
            let c: Vec<u8> = (0..n).map(|_| byte()).collect();
            assert_eq!(dot_u7_i8(&q, &c), dot_u7_i8_scalar(&q, &c), "{n}");
        }
        // Extremes: every product at its largest magnitude.
        let q = vec![127u8; 1000];
        for v in [0x80u8, 0x7f] {
            let c = vec![v; 1000];
            assert_eq!(dot_u7_i8(&q, &c), dot_u7_i8_scalar(&q, &c));
        }
        assert_eq!(dot_u7_i8(&q, &[0x80; 1000]), -127 * 128 * 1000);
        // Unequal lengths use the common prefix.
        assert_eq!(dot_u7_i8(&[2, 3, 5], &[1, 0xff]), 2 - 3);
        // Four rows at once match one at a time.
        for n in [0, 5, 16, 31, 32, 48, 100, 320] {
            let q: Vec<u8> = (0..n).map(|_| byte() & 0x7f).collect();
            let rows: Vec<Vec<u8>> = (0..4)
                .map(|_| (0..n + 3).map(|_| byte()).collect())
                .collect();
            let got = dot_u7_i8_x4(&q, [&rows[0], &rows[1], &rows[2], &rows[3]]);
            for (g, r) in got.iter().zip(&rows) {
                assert_eq!(*g, dot_u7_i8_scalar(&q, &r[..n]), "{n}");
            }
        }
        let short = [1u8, 0xff];
        assert_eq!(dot_u7_i8_x4(&[2, 3, 5], [&short; 4]), [-1; 4]);
    }
}
