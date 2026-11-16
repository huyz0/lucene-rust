#![cfg(test)]
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
fn le_f32s_views_aligned_bytes_and_refuses_the_rest() {
    let floats = [1.5f32, -0.0, f32::MAX, 3.25];
    let mut storage = [0u8; 16 + 4];
    // The first 4-byte-aligned offset into the buffer.
    let at = storage.as_ptr().align_offset(4);
    for (i, f) in floats.iter().enumerate() {
        storage[at + 4 * i..at + 4 * i + 4].copy_from_slice(&f.to_le_bytes());
    }
    let aligned = &storage[at..at + 16];
    if cfg!(target_endian = "little") {
        assert_eq!(le_f32s(aligned), Some(&floats[..]));
    } else {
        assert_eq!(le_f32s(aligned), None);
    }
    assert_eq!(le_f32s(&aligned[1..5]), None, "misaligned");
    assert_eq!(le_f32s(&aligned[..6]), None, "partial float");
}

proptest::proptest! {
    /// Every byte kernel against its scalar specification, at lengths
    /// around the 32-byte step (tails included) and over all byte values.
    #[test]
    fn byte_kernels_match_their_scalar_specifications(
        pairs in proptest::collection::vec((proptest::prelude::any::<u8>(), proptest::prelude::any::<u8>()), 0..700),
    ) {
        let (a, b): (Vec<u8>, Vec<u8>) = pairs.into_iter().unzip();
        proptest::prop_assert_eq!(dot_product_i8(&a, &b), dot_product_i8_scalar(&a, &b));
        proptest::prop_assert_eq!(uint8_dot_product(&a, &b), uint8_dot_product_scalar(&a, &b));
        proptest::prop_assert_eq!(square_distance_i8(&a, &b), square_distance_i8_scalar(&a, &b));
        proptest::prop_assert_eq!(uint8_square_distance(&a, &b), uint8_square_distance_scalar(&a, &b));
        proptest::prop_assert_eq!(cosine_sums_i8(&a, &b), cosine_sums_i8_scalar(&a, &b));
    }
}

/// The extremes, in a vector long enough for 32-bit lane sums to wrap.
#[test]
fn byte_kernels_wrap_like_the_specifications() {
    for (x, y) in [(0x80u8, 0x80u8), (0x80, 0x7f), (0xff, 0x00), (0xff, 0xff)] {
        let a = vec![x; 70_001];
        let b = vec![y; 70_001];
        // 140_001 * 2^14 passes `i32::MAX`, so the cosine sums wrap too.
        let (a2, b2) = (vec![x; 140_001], vec![y; 140_001]);
        assert_eq!(cosine_sums_i8(&a2, &b2), cosine_sums_i8_scalar(&a2, &b2));
        assert_eq!(dot_product_i8(&a, &b), dot_product_i8_scalar(&a, &b));
        assert_eq!(uint8_dot_product(&a, &b), uint8_dot_product_scalar(&a, &b));
        assert_eq!(
            square_distance_i8(&a, &b),
            square_distance_i8_scalar(&a, &b)
        );
        assert_eq!(
            uint8_square_distance(&a, &b),
            uint8_square_distance_scalar(&a, &b)
        );
    }
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
        let pack = |v: &[u8]| -> Vec<u8> { (0..half).map(|i| (v[i] << 4) | v[half + i]).collect() };
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
fn int4_single_packed_simd_matches_scalar_for_signed_bytes() {
    for n in [1usize, 31, 32, 33, 64, 100, 384] {
        let unpacked = bytes(2 * n, 17, 0xff);
        let packed = bytes(n, 18, 0xff);
        let (lo, hi) = unpacked.split_at(n);
        assert_eq!(
            int4_dot_product_single_packed(&unpacked, &packed),
            int4_single_packed_scalar(lo, hi, &packed),
            "n={n}"
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
