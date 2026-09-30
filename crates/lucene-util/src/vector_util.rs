//! The rest of `org.apache.lucene.util.VectorUtil` and its
//! `internal.vectorization.VectorUtilSupport` kernels: the byte-vector and
//! quantized-vector distance kernels the scalar-quantized formats score with,
//! plus `l2normalize`, `isUnitVector`, `isZeroVector`, `add`, `xorBitCount`,
//! and the legacy min/max scalar quantizer's `minMaxScalarQuantize` /
//! `recalculateOffset`.
//!
//! The float kernels (`dotProduct`, `squareDistance`, `cosine` over `f32`)
//! are [`crate::simd`]'s; `l2normalize` and `isUnitVector` call them exactly
//! where Java calls `IMPL.dotProduct`.
//!
//! Every integer kernel returns exactly what `DefaultVectorUtilSupport`
//! returns -- integer sums do not depend on evaluation order -- so the
//! `*_scalar` functions here are both the specification and the Java port,
//! and the unsuffixed ones may use any summation order the compiler or an
//! intrinsic likes. Lengths are checked as `VectorUtil` checks them; a
//! mismatch is Java's `IllegalArgumentException`, a panic here (the caller's
//! bug, never a disk value).

use crate::simd;

/// `VectorUtil.EPSILON`.
const EPSILON: f32 = 1e-4;

#[cold]
#[inline(never)]
fn dims_differ(a: usize, b: usize) -> ! {
    panic!("vector dimensions differ: {a}!={b}");
}

#[inline]
fn check_same(a: usize, b: usize) {
    if a != b {
        dims_differ(a, b);
    }
}

/// `VectorUtil.dotProduct(byte[], byte[])`: signed bytes.
#[inline]
pub fn dot_product_i8(a: &[u8], b: &[u8]) -> i32 {
    check_same(a.len(), b.len());
    a.iter().zip(b).fold(0i32, |t, (&x, &y)| {
        t.wrapping_add(x as i8 as i32 * y as i8 as i32)
    })
}

/// `VectorUtil.uint8DotProduct`: unsigned bytes.
#[inline]
pub fn uint8_dot_product(a: &[u8], b: &[u8]) -> i32 {
    check_same(a.len(), b.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: AVX2 is enabled for this whole build; the kernel reads only
        // whole 32-byte chunks inside both (equal-length) slices.
        unsafe { uint8_dot_product_avx2(a, b) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    uint8_dot_product_scalar(a, b)
}

/// The specification [`uint8_dot_product`] matches: `DefaultVectorUtilSupport.uint8DotProduct`.
pub fn uint8_dot_product_scalar(a: &[u8], b: &[u8]) -> i32 {
    check_same(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0i32, |t, (&x, &y)| t.wrapping_add(x as i32 * y as i32))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn uint8_dot_product_avx2(a: &[u8], b: &[u8]) -> i32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n == a.len() == b.len()`.
        let (va, vb) = unsafe {
            (
                _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i),
                _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i),
            )
        };
        // Widen each half to 16 lanes of u16 (values <= 255, so the i16
        // multiply-add below cannot overflow: 2 * 255 * 255 < 2^31).
        let a_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
        let a_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(va));
        let b_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
        let b_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(vb));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_lo, b_lo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_hi, b_hi));
        i += 32;
    }
    let mut lanes = [0i32; 8];
    // SAFETY: `lanes` is 32 bytes.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc) };
    let mut total = lanes.iter().fold(0i32, |s, &v| s.wrapping_add(v));
    while i < n {
        total = total.wrapping_add(a[i] as i32 * b[i] as i32);
        i += 1;
    }
    total
}

/// `VectorUtil.int4DotProduct`: both vectors unpacked, one 4-bit value per
/// byte (so the signed byte product is the unsigned one).
#[inline]
pub fn int4_dot_product(a: &[u8], b: &[u8]) -> i32 {
    dot_product_i8(a, b)
}

/// `VectorUtil.int4DotProductSinglePacked`: `unpacked` holds one value per
/// byte; `packed` holds two per byte -- value `i` in the high nibble of byte
/// `i`, value `i + packed.len()` in the low nibble.
#[inline]
pub fn int4_dot_product_single_packed(unpacked: &[u8], packed: &[u8]) -> i32 {
    if packed.len() != (unpacked.len() + 1) >> 1 {
        panic!(
            "vector dimensions differ: {} != 2 * {}",
            unpacked.len(),
            packed.len()
        );
    }
    let n = packed.len();
    let (lo, hi) = unpacked.split_at(n);
    let mut total = 0i32;
    for i in 0..n {
        let p = packed[i] as i32;
        total = total.wrapping_add((p & 0x0F) * hi[i] as i8 as i32);
        total = total.wrapping_add((p >> 4) * lo[i] as i8 as i32);
    }
    total
}

/// `VectorUtil.int4DotProductBothPacked`.
#[inline]
pub fn int4_dot_product_both_packed(a: &[u8], b: &[u8]) -> i32 {
    check_same(a.len(), b.len());
    a.iter().zip(b).fold(0i32, |t, (&x, &y)| {
        let (x, y) = (x as i32, y as i32);
        t.wrapping_add((x & 0x0F) * (y & 0x0F))
            .wrapping_add((x >> 4) * (y >> 4))
    })
}

/// `VectorUtil.squareDistance(byte[], byte[])`: signed bytes.
#[inline]
pub fn square_distance_i8(a: &[u8], b: &[u8]) -> i32 {
    check_same(a.len(), b.len());
    a.iter().zip(b).fold(0i32, |t, (&x, &y)| {
        let d = x as i8 as i32 - y as i8 as i32;
        t.wrapping_add(d * d)
    })
}

/// `VectorUtil.int4SquareDistance`.
#[inline]
pub fn int4_square_distance(a: &[u8], b: &[u8]) -> i32 {
    square_distance_i8(a, b)
}

/// `VectorUtil.int4SquareDistanceSinglePacked`.
#[inline]
pub fn int4_square_distance_single_packed(unpacked: &[u8], packed: &[u8]) -> i32 {
    if packed.len() != (unpacked.len() + 1) >> 1 {
        panic!(
            "vector dimensions differ: {}!= 2 * {}",
            unpacked.len(),
            packed.len()
        );
    }
    let n = packed.len();
    let (lo, hi) = unpacked.split_at(n);
    let mut total = 0i32;
    for i in 0..n {
        let p = packed[i] as i32;
        let d1 = (p & 0x0F) - hi[i] as i8 as i32;
        let d2 = (p >> 4) - lo[i] as i8 as i32;
        total = total.wrapping_add(d1 * d1 + d2 * d2);
    }
    total
}

/// `VectorUtil.int4SquareDistanceBothPacked`.
#[inline]
pub fn int4_square_distance_both_packed(a: &[u8], b: &[u8]) -> i32 {
    check_same(a.len(), b.len());
    a.iter().zip(b).fold(0i32, |t, (&x, &y)| {
        let (x, y) = (x as i32, y as i32);
        let d1 = (x & 0x0F) - (y & 0x0F);
        let d2 = (x >> 4) - (y >> 4);
        t.wrapping_add(d1 * d1 + d2 * d2)
    })
}

/// `VectorUtil.uint8SquareDistance`.
#[inline]
pub fn uint8_square_distance(a: &[u8], b: &[u8]) -> i32 {
    check_same(a.len(), b.len());
    a.iter().zip(b).fold(0i32, |t, (&x, &y)| {
        let d = x as i32 - y as i32;
        t.wrapping_add(d * d)
    })
}

/// `DefaultVectorUtilSupport.int4BitDotProductImpl(q, d, dOffset, stripeSize)`:
/// the four bit planes of `q` (each `stripe` bytes) ANDed with `d[d_off..]`,
/// popcounts weighted by plane.
#[inline]
fn int4_bit_dot_stripe(q: &[u8], d: &[u8], stripe: usize) -> i64 {
    let d = &d[..stripe];
    let mut ret = 0i64;
    for plane in 0..4 {
        let qp = &q[plane * stripe..(plane + 1) * stripe];
        let mut sub = 0i64;
        let mut qc = qp.chunks_exact(8);
        let mut dc = d.chunks_exact(8);
        for (x, y) in (&mut qc).zip(&mut dc) {
            let x = u64::from_ne_bytes(x.try_into().expect("8 bytes"));
            let y = u64::from_ne_bytes(y.try_into().expect("8 bytes"));
            sub += (x & y).count_ones() as i64;
        }
        for (&x, &y) in qc.remainder().iter().zip(dc.remainder()) {
            sub += (x & y).count_ones() as i64;
        }
        ret += sub << plane;
    }
    ret
}

/// `VectorUtil.int4BitDotProduct`: a 4-bit query transposed into four bit
/// planes (`OptimizedScalarQuantizer.transposeHalfByte`) against a 1-bit
/// document vector (`packAsBinary`). `q.len()` must be `4 * d.len()`.
pub fn int4_bit_dot_product(q: &[u8], d: &[u8]) -> i64 {
    if q.len() != d.len() * 4 {
        panic!(
            "vector dimensions incompatible: {}!= 4 x {}",
            q.len(),
            d.len()
        );
    }
    int4_bit_dot_stripe(q, d, d.len())
}

/// `VectorUtil.int4DibitDotProduct`: a transposed 4-bit query against a
/// 2-bit document vector stored as two bit stripes (`transposeDibit`).
/// `q.len()` must be `2 * d.len()`.
pub fn int4_dibit_dot_product(q: &[u8], d: &[u8]) -> i64 {
    if q.len() != d.len() * 2 {
        panic!(
            "vector dimensions incompatible: {}!= 2 x {}",
            q.len(),
            d.len()
        );
    }
    let stripe = d.len() / 2;
    let ret0 = int4_bit_dot_stripe(q, d, stripe);
    let ret1 = int4_bit_dot_stripe(q, &d[stripe..], stripe);
    ret0 + (ret1 << 1)
}

/// `VectorUtil.xorBitCount`: Hamming distance. Java strides by `int` on
/// aarch64 and `long` elsewhere; the count is the same either way.
pub fn xor_bit_count(a: &[u8], b: &[u8]) -> i32 {
    check_same(a.len(), b.len());
    let mut distance = 0i32;
    let mut ac = a.chunks_exact(8);
    let mut bc = b.chunks_exact(8);
    for (x, y) in (&mut ac).zip(&mut bc) {
        let x = u64::from_ne_bytes(x.try_into().expect("8 bytes"));
        let y = u64::from_ne_bytes(y.try_into().expect("8 bytes"));
        distance += (x ^ y).count_ones() as i32;
    }
    for (&x, &y) in ac.remainder().iter().zip(bc.remainder()) {
        distance += (x ^ y).count_ones() as i32;
    }
    distance
}

/// `VectorUtil.isUnitVector`.
pub fn is_unit_vector(v: &[f32]) -> bool {
    let l1norm = simd::dot_f32(v, v) as f64;
    (l1norm - 1.0).abs() <= EPSILON as f64
}

/// `VectorUtil.l2normalize(v, throwOnZero)`: scales `v` to unit length.
/// `Err(())` for a zero vector when `throw_on_zero` (Java's
/// `IllegalArgumentException`); a zero vector is otherwise left alone, as is
/// a vector already within `EPSILON` of unit length.
pub fn l2normalize(v: &mut [f32], throw_on_zero: bool) -> Result<(), ZeroVector> {
    let l1norm = simd::dot_f32(v, v) as f64;
    if l1norm == 0.0 {
        return if throw_on_zero {
            Err(ZeroVector)
        } else {
            Ok(())
        };
    }
    if (l1norm - 1.0).abs() <= EPSILON as f64 {
        return Ok(());
    }
    let l2norm = l1norm.sqrt() as f32;
    for x in v.iter_mut() {
        *x /= l2norm;
    }
    Ok(())
}

/// The error of [`l2normalize`] on a zero vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("Cannot normalize a zero-length vector")]
pub struct ZeroVector;

/// `VectorUtil.add`: `u += v` componentwise.
pub fn add(u: &mut [f32], v: &[f32]) {
    for (x, &y) in u.iter_mut().zip(v) {
        *x += y;
    }
}

/// `VectorUtil.isZeroVector(float[])`.
pub fn is_zero_vector(v: &[f32]) -> bool {
    v.iter().all(|&x| x == 0.0)
}

/// `VectorUtil.isZeroVector(byte[])`.
pub fn is_zero_vector_bytes(v: &[u8]) -> bool {
    v.iter().all(|&x| x == 0)
}

/// Java's `Math.round(float)`: nearest integer, ties toward positive
/// infinity, NaN to 0, saturating at the `int` range.
#[inline]
pub fn java_round_f32(x: f32) -> i32 {
    if x.is_nan() {
        return 0;
    }
    let f = x.floor();
    // `x - floor(x)` is exact for every float (the fractional part of a
    // float is representable), so this is a true tie test, unlike
    // `floor(x + 0.5)`.
    let r = if x - f >= 0.5 { f + 1.0 } else { f };
    r as i32
}

/// Java's `Math.round(double)`: as [`java_round_f32`], saturating at the
/// `long` range.
#[inline]
pub fn java_round_f64(x: f64) -> i64 {
    if x.is_nan() {
        return 0;
    }
    let f = x.floor();
    let r = if x - f >= 0.5 { f + 1.0 } else { f };
    r as i64
}

/// `DefaultVectorUtilSupport.ScalarQuantizer.quantizeFloat`.
#[inline]
fn quantize_float(
    v: f32,
    scale: f32,
    alpha: f32,
    min_quantile: f32,
    max_quantile: f32,
) -> (f32, u8) {
    let dx = v - min_quantile;
    let dxc = java_max(min_quantile, java_min(max_quantile, v)) - min_quantile;
    let rounded = java_round_f32(scale * dxc);
    let dxq = rounded as f32 * alpha;
    (
        min_quantile * (v - min_quantile / 2.0) + (dx - dxq) * dxq,
        rounded as u8,
    )
}

/// `java.lang.Math.max(float, float)`/`min`'s NaN rule, which `f32::max`
/// does not share: a NaN operand wins.
#[inline]
fn java_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 {
        // Java: max(-0.0, 0.0) == 0.0
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else {
        a.max(b)
    }
}

#[inline]
fn java_min(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else {
        a.min(b)
    }
}

/// `VectorUtil.minMaxScalarQuantize`: quantizes `vector` into `dest` (the
/// legacy global min/max scalar quantizer) and returns the corrective offset.
pub fn min_max_scalar_quantize(
    vector: &[f32],
    dest: &mut [u8],
    scale: f32,
    alpha: f32,
    min_quantile: f32,
    max_quantile: f32,
) -> f32 {
    if vector.len() != dest.len() {
        panic!("source and destination arrays should be the same size");
    }
    let mut correction = 0f32;
    for (d, &v) in dest.iter_mut().zip(vector) {
        let (c, q) = quantize_float(v, scale, alpha, min_quantile, max_quantile);
        *d = q;
        correction += c;
    }
    correction
}

/// `VectorUtil.recalculateOffset`: the corrective offset of an already
/// quantized vector under a new quantizer.
pub fn recalculate_offset(
    vector: &[u8],
    old_alpha: f32,
    old_min_quantile: f32,
    scale: f32,
    alpha: f32,
    min_quantile: f32,
    max_quantile: f32,
) -> f32 {
    let mut correction = 0f32;
    for &q in vector {
        let v = (old_alpha * q as f32) + old_min_quantile;
        correction += quantize_float(v, scale, alpha, min_quantile, max_quantile).0;
    }
    correction
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(n: usize, seed: u32, mask: u8) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8 & mask
            })
            .collect()
    }

    #[test]
    fn uint8_dot_matches_the_scalar_specification() {
        for n in [0usize, 1, 7, 31, 32, 33, 64, 100, 1024, 1031] {
            let a = bytes(n, 1, 0xff);
            let b = bytes(n, 2, 0xff);
            assert_eq!(
                uint8_dot_product(&a, &b),
                uint8_dot_product_scalar(&a, &b),
                "n={n}"
            );
        }
        assert_eq!(uint8_dot_product(&[255, 1], &[255, 2]), 255 * 255 + 2);
    }

    #[test]
    fn signed_and_unsigned_kernels_differ_on_high_bytes() {
        assert_eq!(dot_product_i8(&[0xff, 2], &[0xff, 3]), 1 + 6);
        assert_eq!(uint8_dot_product(&[0xff, 2], &[0xff, 3]), 255 * 255 + 6);
        assert_eq!(square_distance_i8(&[0x80], &[0x7f]), 255 * 255);
        assert_eq!(uint8_square_distance(&[0x80], &[0x7f]), 1);
        assert_eq!(int4_dot_product(&[3, 4], &[5, 6]), 39);
        assert_eq!(int4_square_distance(&[3, 4], &[5, 6]), 8);
    }

    #[test]
    fn packed_int4_kernels_match_their_unpacked_forms() {
        for n in [2usize, 8, 64, 130] {
            let a = bytes(n, 3, 0x0f);
            let b = bytes(n, 4, 0x0f);
            let half = n / 2;
            let pack =
                |v: &[u8]| -> Vec<u8> { (0..half).map(|i| (v[i] << 4) | v[half + i]).collect() };
            let (pa, pb) = (pack(&a), pack(&b));
            assert_eq!(
                int4_dot_product_single_packed(&a, &pb),
                int4_dot_product(&a, &b)
            );
            assert_eq!(
                int4_square_distance_single_packed(&a, &pb),
                int4_square_distance(&a, &b)
            );
            // Both packed pairs the nibbles position-wise, which is again the
            // unpacked sum.
            assert_eq!(
                int4_dot_product_both_packed(&pa, &pb),
                int4_dot_product(&a, &b)
            );
            assert_eq!(
                int4_square_distance_both_packed(&pa, &pb),
                int4_square_distance(&a, &b)
            );
        }
    }

    #[test]
    #[should_panic(expected = "dimensions differ")]
    fn single_packed_length_check() {
        int4_dot_product_single_packed(&[0; 8], &[0; 3]);
    }

    #[test]
    #[should_panic(expected = "dimensions differ")]
    fn single_packed_square_length_check() {
        int4_square_distance_single_packed(&[0; 8], &[0; 3]);
    }

    #[test]
    #[should_panic(expected = "dimensions differ")]
    fn mismatched_lengths_panic() {
        uint8_dot_product(&[1, 2], &[1]);
    }

    /// Reference: the 4-bit query dot the 1-bit document, value by value.
    fn bit_reference(query: &[u8], doc_bits: &[u8]) -> i64 {
        query
            .iter()
            .zip(doc_bits)
            .map(|(&q, &d)| q as i64 * d as i64)
            .sum()
    }

    fn transpose_half_byte(q: &[u8]) -> Vec<u8> {
        // OptimizedScalarQuantizer.transposeHalfByte for a multiple of 8.
        let stripe = q.len() / 8;
        let mut out = vec![0u8; stripe * 4];
        for (chunk_idx, chunk) in q.chunks(8).enumerate() {
            for (j, &v) in chunk.iter().enumerate() {
                for plane in 0..4 {
                    out[plane * stripe + chunk_idx] |= ((v >> plane) & 1) << (7 - j);
                }
            }
        }
        out
    }

    fn pack_bits(v: &[u8], bit: u32) -> Vec<u8> {
        v.chunks(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u8, |acc, (j, &x)| acc | (((x >> bit) & 1) << (7 - j)))
            })
            .collect()
    }

    #[test]
    fn int4_bit_and_dibit_dot_products_match_value_by_value_sums() {
        for dims in [8usize, 64, 72, 520] {
            let query = bytes(dims, 5, 0x0f);
            let doc1 = bytes(dims, 6, 0x01);
            let q = transpose_half_byte(&query);
            let d = pack_bits(&doc1, 0);
            assert_eq!(
                int4_bit_dot_product(&q, &d),
                bit_reference(&query, &doc1),
                "dims={dims}"
            );

            let doc2 = bytes(dims, 7, 0x03);
            let mut dd = pack_bits(&doc2, 0);
            dd.extend(pack_bits(&doc2, 1));
            // transposeDibit's stripe is the query's plane width.
            assert_eq!(
                int4_dibit_dot_product(&q, &dd),
                bit_reference(&query, &doc2),
                "dims={dims}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "incompatible")]
    fn bit_dot_length_check() {
        int4_bit_dot_product(&[0; 7], &[0; 2]);
    }

    #[test]
    #[should_panic(expected = "incompatible")]
    fn dibit_dot_length_check() {
        int4_dibit_dot_product(&[0; 7], &[0; 2]);
    }

    #[test]
    fn xor_bit_count_is_hamming_distance() {
        let a = bytes(37, 8, 0xff);
        let b = bytes(37, 9, 0xff);
        let want: i32 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x ^ y).count_ones() as i32)
            .sum();
        assert_eq!(xor_bit_count(&a, &b), want);
    }

    #[test]
    fn normalize_unit_and_zero_vectors() {
        let mut v = [3.0f32, 4.0];
        l2normalize(&mut v, true).unwrap();
        assert_eq!(v, [0.6, 0.8]);
        assert!(is_unit_vector(&v));
        let mut z = [0.0f32; 3];
        assert_eq!(l2normalize(&mut z, true), Err(ZeroVector));
        assert!(l2normalize(&mut z, false).is_ok());
        assert!(is_zero_vector(&z));
        assert!(!is_zero_vector(&v));
        assert!(is_zero_vector_bytes(&[0, 0]));
        assert!(!is_zero_vector_bytes(&[0, 1]));
        let mut near = [1.00001f32, 0.0];
        l2normalize(&mut near, true).unwrap();
        assert_eq!(near[0], 1.00001); // within EPSILON: untouched
        let mut u = [1.0f32, 2.0];
        add(&mut u, &[0.5, -2.0]);
        assert_eq!(u, [1.5, 0.0]);
        assert_eq!(
            ZeroVector.to_string(),
            "Cannot normalize a zero-length vector"
        );
    }

    #[test]
    fn java_round_semantics() {
        assert_eq!(java_round_f32(0.5), 1);
        assert_eq!(java_round_f32(-0.5), 0);
        assert_eq!(java_round_f32(-2.5), -2);
        assert_eq!(java_round_f32(2.5), 3);
        assert_eq!(java_round_f32(0.49999997), 0);
        assert_eq!(java_round_f32(f32::NAN), 0);
        assert_eq!(java_round_f32(1e20), i32::MAX);
        assert_eq!(java_round_f32(-1e20), i32::MIN);
        assert_eq!(java_round_f64(-2.5), -2);
        assert_eq!(java_round_f64(0.49999999999999994), 0);
        assert_eq!(java_round_f64(f64::NAN), 0);
        assert_eq!(java_round_f64(1e300), i64::MAX);
    }

    #[test]
    fn min_max_quantize_and_recalculate() {
        let v = [0.0f32, 0.25, 0.5, 1.0, 2.0, -1.0];
        let (lo, hi) = (0.0f32, 1.0f32);
        let scale = 255.0 / (hi - lo);
        let alpha = (hi - lo) / 255.0;
        let mut dest = [0u8; 6];
        let c = min_max_scalar_quantize(&v, &mut dest, scale, alpha, lo, hi);
        assert_eq!(dest, [0, 64, 128, 255, 255, 0]);
        let mut sum = 0f32;
        for &x in &v {
            sum += quantize_float(x, scale, alpha, lo, hi).0;
        }
        assert_eq!(c, sum);
        let r = recalculate_offset(&dest, alpha, lo, scale, alpha, lo, hi);
        assert!(r.is_finite());
        assert_eq!(java_max(-0.0, 0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(java_min(-0.0, 0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(java_min(0.0, -0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(java_max(0.0, -0.0).to_bits(), 0.0f32.to_bits());
        assert!(java_max(f32::NAN, 1.0).is_nan());
        assert!(java_min(1.0, f32::NAN).is_nan());
    }

    #[test]
    #[should_panic(expected = "same size")]
    fn min_max_length_check() {
        min_max_scalar_quantize(&[1.0], &mut [0u8; 2], 1.0, 1.0, 0.0, 1.0);
    }
}
