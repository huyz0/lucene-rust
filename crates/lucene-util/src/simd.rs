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
        for (k, byte) in word.to_le_bytes().into_iter().enumerate() {
            if byte == 0 {
                continue;
            }
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
}
