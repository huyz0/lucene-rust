#![cfg(test)]
use super::*;

fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn naive(query: &[f32], doc: &[f32], dim: usize) -> f32 {
    query
        .chunks_exact(dim)
        .map(|q| {
            doc.chunks_exact(dim)
                .map(|d| q.iter().zip(d).fold(0.0f32, |a, (&x, &y)| x.mul_add(y, a)))
                .fold(f32::NEG_INFINITY, |m, v| if m > v { m } else { v })
        })
        .sum()
}

/// Every kernel this CPU and build can run, not only the dispatched one.
fn every_kernel(query: &[f32], dim: usize, blocks: Blocks<'_>, tokens: usize) -> Vec<f32> {
    // only x86 has more than the dispatched kernel to add
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
    let mut out = vec![max_sim_blocks(query, dim, blocks, tokens)];
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::level() == crate::cpu::Level::Avx512 {
        // SAFETY: AVX-512F is present; `max_sim_blocks` above checked the lengths.
        out.push(unsafe { avx512::max_sim(query, dim, blocks, tokens) });
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c"
    ))]
    // SAFETY: the build enables AVX2, FMA and F16C; lengths checked above.
    out.push(unsafe { avx2::max_sim(query, dim, blocks, tokens) });
    out
}

#[test]
fn every_path_matches_the_scalar_specification_bit_for_bit() {
    for &(nq, nd, dim) in &[
        (1, 1, 1),
        (3, 5, 7),
        (8, 16, 16),
        (9, 17, 96),
        (32, 100, 128),
        (23, 236, 96),
        (5, 33, 3),
    ] {
        let q = data(nq * dim, 7 + nq as u64);
        let d = data(nd * dim, 11 + nd as u64);
        let mut blocks = Vec::new();
        assert_eq!(transpose_into_blocks(&d, dim, &mut blocks), nd);
        let spec = max_sim_blocks_scalar(&q, dim, Blocks::F32(&blocks), nd);
        assert_eq!(
            spec.to_bits(),
            naive(&q, &d, dim).to_bits(),
            "{nq} {nd} {dim}"
        );
        for got in every_kernel(&q, dim, Blocks::F32(&blocks), nd) {
            assert_eq!(got.to_bits(), spec.to_bits(), "{nq} {nd} {dim}");
        }
        // the same tokens as half floats: every path widens them exactly
        let halves: Vec<u8> = blocks
            .iter()
            .flat_map(|&x| to_f16(x).to_le_bytes())
            .collect();
        let widened: Vec<f32> = halves
            .chunks(2)
            .map(|h| f16_to_f32(u16::from_le_bytes([h[0], h[1]])))
            .collect();
        let spec16 = max_sim_blocks_scalar(&q, dim, Blocks::F16(&halves), nd);
        assert_eq!(
            spec16.to_bits(),
            max_sim_blocks_scalar(&q, dim, Blocks::F32(&widened), nd).to_bits()
        );
        for got in every_kernel(&q, dim, Blocks::F16(&halves), nd) {
            assert_eq!(got.to_bits(), spec16.to_bits(), "f16 {nq} {nd} {dim}");
        }
    }
}

#[test]
fn padded_lanes_never_reach_the_score() {
    for &(nq, nd, dim) in &[(1, 1, 3), (9, 17, 96), (5, 33, 7)] {
        let q = data(nq * dim, 3);
        let d = data(nd * dim, 5);
        let mut clean = Vec::new();
        transpose_into_blocks(&d, dim, &mut clean);
        let mut dirty = clean.clone();
        for b in 0..nd.div_ceil(LANES) {
            for l in (nd - b * LANES).min(LANES)..LANES {
                for k in 0..dim {
                    let v = if (k + l) % 2 == 0 { f32::NAN } else { f32::MAX };
                    dirty[b * dim * LANES + k * LANES + l] = v;
                }
            }
        }
        let want = max_sim_blocks_scalar(&q, dim, Blocks::F32(&clean), nd).to_bits();
        for got in every_kernel(&q, dim, Blocks::F32(&dirty), nd) {
            assert_eq!(got.to_bits(), want, "f32 {nq} {nd} {dim}");
        }
        // half floats: NaN 0x7e00, infinity 0x7c00 in the padding
        let half = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|&x| {
                    let h = if x.is_nan() {
                        0x7e00
                    } else if x == f32::MAX {
                        0x7c00
                    } else {
                        to_f16(x)
                    };
                    h.to_le_bytes()
                })
                .collect()
        };
        let (clean16, dirty16) = (half(&clean), half(&dirty));
        let want16 = max_sim_blocks_scalar(&q, dim, Blocks::F16(&clean16), nd).to_bits();
        for got in every_kernel(&q, dim, Blocks::F16(&dirty16), nd) {
            assert_eq!(got.to_bits(), want16, "f16 {nq} {nd} {dim}");
        }
    }
}

#[test]
#[should_panic(expected = "blocks do not hold")]
fn a_token_count_whose_length_wraps_is_refused() {
    // div_ceil(16) * 1 * 16 wraps to 0 without checked arithmetic
    max_sim_blocks(&[1.0], 1, Blocks::F32(&[]), usize::MAX - 14);
}

#[test]
fn a_document_without_tokens_gives_negative_infinity_per_query_token() {
    let q = data(4, 3);
    assert_eq!(
        max_sim_blocks(&q, 2, Blocks::F32(&[]), 0),
        f32::NEG_INFINITY
    );
    assert_eq!(
        max_sim_blocks_scalar(&q, 2, Blocks::F16(&[]), 0),
        f32::NEG_INFINITY
    );
}

#[test]
#[should_panic(expected = "blocks do not hold")]
fn a_short_block_buffer_is_refused() {
    max_sim_blocks(&[1.0, 2.0], 2, Blocks::F32(&[0.0; 16]), 1);
}

#[test]
#[should_panic(expected = "blocks do not hold")]
fn an_odd_half_float_buffer_is_refused() {
    max_sim_blocks(&[1.0, 2.0], 2, Blocks::F16(&[0; 63]), 1);
}

/// Round to nearest even; test data only (normal range, no NaN).
fn to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    let man = b & 0x7f_ffff;
    let mut h = u32::from(sign) | ((exp as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    h as u16
}

#[test]
fn half_floats_widen_exactly() {
    assert_eq!(f16_to_f32(0x3c00), 1.0);
    assert_eq!(f16_to_f32(0xc000), -2.0);
    assert_eq!(f16_to_f32(0x7bff), 65504.0);
    assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2f32.powi(-24));
    assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
    assert!(f16_to_f32(0x7e00).is_nan());
    for h in 0..=u16::MAX {
        let e = (h >> 10) & 0x1f;
        if e != 0x1f {
            let v = f16_to_f32(h);
            let m = f64::from(h & 0x3ff);
            let want = if e == 0 {
                m * 2f64.powi(-24)
            } else {
                (1.0 + m / 1024.0) * 2f64.powi(i32::from(e) - 15)
            };
            let want = if h >> 15 == 1 { -want } else { want };
            assert_eq!(f64::from(v), want, "{h:#x}");
        }
    }
}

#[test]
fn transpose_lays_tokens_out_component_major_per_block() {
    let d: Vec<f32> = (0..17 * 2).map(|x| x as f32).collect();
    let mut b = Vec::new();
    transpose_into_blocks(&d, 2, &mut b);
    assert_eq!(b.len(), 2 * 2 * 16);
    assert_eq!((b[0], b[1], b[16], b[17]), (0.0, 2.0, 1.0, 3.0));
    assert_eq!((b[32], b[48], b[33]), (32.0, 33.0, 0.0));
}
