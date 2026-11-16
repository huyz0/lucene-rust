#![cfg(test)]
use super::*;

#[test]
fn prefetch_bytes_never_panics() {
    let data: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
    prefetch_bytes(&[]);
    prefetch_bytes(&data[..1]);
    prefetch_bytes(&data[3..403]);
    prefetch_bytes(&data);
}

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
        let d = dot_and_norm_f32(&a, &b);
        assert_eq!(
            (d.0.to_bits(), d.1.to_bits()),
            (y.0.to_bits(), y.2.to_bits()),
            "dot and norm {len}"
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
        // Both paths, whichever the dispatch picks on this CPU.
        let mut c = [7i32; 256];
        let fc = expand_bitset_portable(&words, base, &mut c);
        assert_eq!((fc, &c[..fc]), (fb, &b[..fb]), "round {round} portable");
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            let mut c = [7i32; 256];
            // SAFETY: the CPU has AVX-512F (checked just above).
            let fc = unsafe { expand_bitset_avx512(&words, base, &mut c) };
            assert_eq!((fc, &c[..fc]), (fb, &b[..fb]), "round {round} avx512");
        }
        // A partial output, as a decode that starts mid-block fills.
        let cap = (xorshift(&mut s) % 257) as usize;
        let (mut a, mut b) = ([7i32; 256], [7i32; 256]);
        let fa = expand_bitset(&words, base, &mut a[..cap]);
        let fb = expand_bitset_scalar(&words, base, &mut b[..cap]);
        assert_eq!(fa, fb, "round {round} cap {cap}");
        assert_eq!(a[..fa], b[..fb], "round {round} cap {cap}");
        let mut c = [7i32; 256];
        let fc = expand_bitset_portable(&words, base, &mut c[..cap]);
        assert_eq!(
            (fc, &c[..]),
            (fb, &b[..]),
            "round {round} cap {cap} portable"
        );
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            let mut c = [7i32; 256];
            // SAFETY: the CPU has AVX-512F (checked just above).
            let fc = unsafe { expand_bitset_avx512(&words, base, &mut c[..cap]) };
            assert_eq!((fc, &c[..]), (fb, &b[..]), "round {round} cap {cap} avx512");
        }
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
    for (groups, saturated) in random.chain([(64, true), (256, true), (257, true), (600, true)]) {
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

#[test]
fn residual_kernels_skip_nan_like_the_scalar_specification() {
    let mut q: Vec<f32> = (0..19).map(|i| i as f32 * 0.25 - 2.0).collect();
    q[3] = f32::NAN;
    q[12] = f32::NAN;
    let c = vec![0.1f32; q.len()];
    let (fast, spec) = (residual_stats(&q, &c), residual_stats_scalar(&q, &c));
    assert_eq!((fast.min, fast.max), (spec.min, spec.max));
    let (mut fo, mut so) = (vec![9u8; q.len()], vec![9u8; q.len()]);
    let (fl, fe) = quantize_residual(&q, &c, [-1.0, 1.0], 7.5, 1.0 / 7.5, &mut fo);
    let (sl, se) = quantize_residual_scalar(&q, &c, [-1.0, 1.0], 7.5, 1.0 / 7.5, &mut so);
    assert_eq!((fo[3], fo[12]), (0, 0));
    assert_eq!((fo, fl), (so, sl));
    assert!(fe.is_nan() && se.is_nan());
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
        // Each kernel the dispatch may pick, whichever it picks here.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let mut v = [0i32; 256];
            // SAFETY: AVX2 is enabled for this whole build.
            unsafe { prefix_sum_256_avx2(base, &deltas, &mut v) };
            assert_eq!(v, spec, "round {round} avx2");
        }
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
        let targets =
            buf.iter()
                .flat_map(|&d| [d - 1, d, d + 1])
                .chain([i32::MIN, i32::MAX, 0, -5, 20_000]);
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
        0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 255, 256, 257, 300, 383, 384, 1000,
        1537,
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
