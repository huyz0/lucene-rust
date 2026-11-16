#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]
use super::*;

/// A half's value, from its fields.
fn half_value(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f64::from(h & 0x3ff);
    sign * match e {
        0 => m * 2f64.powi(-24),
        0x1f if m == 0.0 => f64::INFINITY,
        0x1f => f64::NAN,
        _ => (1024.0 + m) * 2f64.powi(e - 25),
    }
}

#[test]
fn f16_decodes_every_half_exactly_and_round_trips() {
    for h in 0..=u16::MAX {
        let v = f16_to_f32(h);
        let want = half_value(h);
        if want.is_nan() {
            assert!(v.is_nan(), "{h:#x}");
            assert!(f32_to_f16(v) & 0x7fff > 0x7c00, "{h:#x}");
            continue;
        }
        assert_eq!(f64::from(v), want, "{h:#x}");
        assert_eq!(f32_to_f16(v), h, "{h:#x}");
    }
}

#[test]
fn f16_rounds_to_nearest_even() {
    // Between each pair of adjacent non-negative finite halves: just
    // below, at and just above the midpoint.
    for h in 0..0x7bffu16 {
        let (a, b) = (half_value(h), half_value(h + 1));
        let mid = ((a + b) / 2.0) as f32;
        assert_eq!(f64::from(mid), (a + b) / 2.0, "{h:#x} midpoint exact");
        let even = if h & 1 == 0 { h } else { h + 1 };
        assert_eq!(f32_to_f16(mid), even, "{h:#x}");
        assert_eq!(f32_to_f16(mid.next_down()), h, "{h:#x}");
        assert_eq!(f32_to_f16(mid.next_up()), h + 1, "{h:#x}");
        assert_eq!(f32_to_f16(-mid.next_up()), 0x8000 | (h + 1), "{h:#x}");
    }
    // Overflow and underflow.
    assert_eq!(f32_to_f16(65504.0), 0x7bff);
    assert_eq!(f32_to_f16(65519.99), 0x7bff);
    assert_eq!(f32_to_f16(65520.0), 0x7c00);
    assert_eq!(f32_to_f16(1e10), 0x7c00);
    assert_eq!(f32_to_f16(-1e10), 0xfc00);
    assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
    assert_eq!(f32_to_f16(2f32.powi(-25)), 0);
    assert_eq!(f32_to_f16(2f32.powi(-25).next_up()), 1);
    assert_eq!(f32_to_f16(1e-30), 0);
    assert_eq!(f32_to_f16(-1e-30), 0x8000);
    assert_eq!(f32_to_f16(f32::from_bits(1)), 0);
    assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
}

#[test]
fn bf16_rounds_to_nearest_even() {
    for h in [0u16, 1, 0x3f80, 0x3f81, 0x7f7e, 0x8001, 0xbf80] {
        let v = bf16_to_f32(h);
        assert_eq!(f32_to_bf16(v), h);
        let next = bf16_to_f32(h + 1);
        let mid = f32::from_bits(((u64::from(v.to_bits()) + u64::from(next.to_bits())) / 2) as u32);
        let even = if h & 1 == 0 { h } else { h + 1 };
        assert_eq!(f32_to_bf16(mid), even, "{h:#x}");
        assert_eq!(f32_to_bf16(mid.next_toward(v)), h, "{h:#x}");
        assert_eq!(f32_to_bf16(mid.next_toward(next)), h + 1, "{h:#x}");
    }
    assert_eq!(f32_to_bf16(f32::MAX), 0x7f80);
    assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xff80);
    assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    assert!(bf16_to_f32(f32_to_bf16(f32::from_bits(0x7f80_0001))).is_nan());
}

trait NextToward {
    fn next_toward(self, to: f32) -> f32;
}
impl NextToward for f32 {
    fn next_toward(self, to: f32) -> f32 {
        if to > self {
            self.next_up()
        } else {
            self.next_down()
        }
    }
}

#[test]
fn names_and_ordinals() {
    for e in VectorEncoding::ALL {
        assert_eq!(VectorEncoding::from_ordinal(e.ordinal()), Some(e));
        assert_eq!(VectorEncoding::from_name(e.name()), Some(e));
    }
    assert_eq!(VectorEncoding::from_ordinal(4), None);
    assert_eq!(VectorEncoding::from_name("f64"), None);
    assert_eq!(VectorEncoding::default(), VectorEncoding::F32);
    let residual: Vec<_> = VectorEncoding::ALL
        .into_iter()
        .filter(|e| e.residual())
        .collect();
    assert_eq!(residual, [VectorEncoding::Int8]);
    assert_eq!(VectorEncoding::F32.vector_bytes(3), Some(12));
    assert_eq!(VectorEncoding::F16.vector_bytes(3), Some(6));
    assert_eq!(VectorEncoding::Bf16.vector_bytes(3), Some(6));
    assert_eq!(VectorEncoding::Int8.vector_bytes(3), Some(11));
    assert_eq!(VectorEncoding::F32.vector_bytes(usize::MAX), None);
    assert_eq!(VectorEncoding::Int8.vector_bytes(usize::MAX), None);
}

fn round_trip(e: VectorEncoding, v: &[f32]) -> Option<Vec<f32>> {
    round_trip_from(e, v, &vec![0.0; v.len()])
}

fn round_trip_from(e: VectorEncoding, v: &[f32], centroid: &[f32]) -> Option<Vec<f32>> {
    let mut bytes = vec![0u8; e.vector_bytes(v.len()).unwrap()];
    if !e.encode(v, centroid, &mut bytes) {
        return None;
    }
    let mut out = vec![0f32; v.len()];
    e.decode(&bytes, centroid, &mut out);
    Some(out)
}

#[test]
fn encodings_round_trip_within_their_precision() {
    let v: Vec<f32> = (0..37)
        .map(|i| ((i * 7919) % 101) as f32 / 37.0 - 1.3)
        .collect();
    let max = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let range =
        v.iter().copied().fold(f32::MIN, f32::max) - v.iter().copied().fold(f32::MAX, f32::min);
    for (e, tol) in [
        (VectorEncoding::F32, 0.0),
        (VectorEncoding::F16, max * 2f32.powi(-11)),
        (VectorEncoding::Bf16, max * 2f32.powi(-8)),
        (VectorEncoding::Int8, range / 255.0 / 2.0 * 1.0001),
    ] {
        let out = round_trip(e, &v).unwrap();
        for (a, b) in v.iter().zip(&out) {
            assert!((a - b).abs() <= tol, "{e:?}: {a} vs {b}");
        }
    }
    // The extremes of an int8 vector decode exactly or nearly so.
    let out = round_trip(VectorEncoding::Int8, &[-2.0, 0.5, 3.0]).unwrap();
    assert_eq!(out[0], -2.0);
    assert!((out[2] - 3.0).abs() < 1e-6);
    // A residual is quantized: the centroid's offset costs nothing.
    let far: Vec<f32> = v.iter().map(|x| x + 1000.0).collect();
    let centroid = vec![1000.0; v.len()];
    let out = round_trip_from(VectorEncoding::Int8, &far, &centroid).unwrap();
    for (a, b) in far.iter().zip(&out) {
        assert!((a - b).abs() <= range / 255.0 / 2.0 + 1e-3, "{a} vs {b}");
    }
    // A constant vector has no range.
    assert_eq!(
        round_trip(VectorEncoding::Int8, &[4.5; 5]).unwrap(),
        vec![4.5; 5]
    );
}

#[test]
fn values_out_of_range_are_refused() {
    assert!(round_trip(VectorEncoding::F16, &[1.0, 70000.0]).is_none());
    assert!(round_trip(VectorEncoding::Bf16, &[f32::MAX]).is_none());
    assert!(round_trip(VectorEncoding::F32, &[f32::MAX]).is_some());
    // A range wider than `f32::MAX`, and a residual that overflows.
    assert!(round_trip(VectorEncoding::Int8, &[3e38, -3e38]).is_none());
    assert!(round_trip_from(VectorEncoding::Int8, &[3e38], &[-3e38]).is_none());
    // Large values next to a large centroid.
    assert!(round_trip_from(VectorEncoding::Int8, &[f32::MAX, 0.0], &[f32::MAX, 0.0]).is_some());
    assert!(round_trip_from(VectorEncoding::Int8, &[f32::MAX, 0.0], &[3e38, 0.0]).is_some());
}

#[test]
fn unaligned_f32_bytes_decode() {
    let mut bytes = [0u8; 13];
    assert!(VectorEncoding::F32.encode(&[1.5, -2.0, 3.25], &[], &mut bytes[1..]));
    let mut out = [0f32; 3];
    VectorEncoding::F32.decode(&bytes[1..], &[], &mut out);
    assert_eq!(out, [1.5, -2.0, 3.25]);
}
