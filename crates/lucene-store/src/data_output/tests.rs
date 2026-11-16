#![cfg(test)]
use super::*;
use crate::data_input::{DataInput, SliceInput};
use proptest::prelude::*;

/// The branch-free spare-capacity encoder against the plain loop, at
/// every length boundary (each power of 128 and one below) and over the
/// whole range.
#[test]
fn push_varint_matches_the_byte_loop_at_every_length_boundary() {
    let mut cases = vec![0u64, 1, u64::MAX];
    for k in 1..=9u32 {
        let p = 1u64 << (7 * k);
        cases.extend([p - 1, p, p + 1]);
    }
    cases.extend([1 << 63, (1 << 63) - 1]);
    for v in cases {
        let mut want = [0u8; 10];
        let n = encode_varint(v, &mut want);
        let mut got = vec![0xAA];
        push_varint(&mut got, v);
        assert_eq!(&got[1..], &want[..n], "value {v:#x}");
    }
}

proptest! {
    #[test]
    fn push_varint_matches_the_byte_loop(v: u64, shift in 0u32..64) {
        let v = v >> shift;
        let mut want = [0u8; 10];
        let n = encode_varint(v, &mut want);
        let mut got = Vec::new();
        push_varint(&mut got, v);
        prop_assert_eq!(&got[..], &want[..n]);
    }

    #[test]
    fn spread7_matches_the_portable_lanes(v: u64) {
        prop_assert_eq!(spread7(v), spread7_portable(v));
    }

    #[test]
    fn default_and_vec_encoders_agree(v: i32, l in 0i64..=i64::MAX, z: i64, g in proptest::collection::vec(any::<u32>(), 0..11)) {
        // `HeaderCursor`-like sink: only the two required methods, so
        // every provided encoder takes the trait's default path.
        struct Plain(Vec<u8>);
        impl DataOutput for Plain {
            fn write_byte(&mut self, b: u8) { self.0.push(b); }
            fn write_bytes(&mut self, b: &[u8]) { self.0.extend_from_slice(b); }
        }
        let mut plain = Plain(Vec::new());
        let mut vec: Vec<u8> = Vec::new();
        let mut vdo = VecDataOutput::new();
        plain.write_vint(v); vec.write_vint(v); vdo.write_vint(v);
        plain.write_vlong(l); vec.write_vlong(l); vdo.write_vlong(l);
        plain.write_zlong(z); vec.write_zlong(z); vdo.write_zlong(z);
        plain.write_group_vints(&g); vec.write_group_vints(&g); vdo.write_group_vints(&g);
        prop_assert_eq!(&plain.0, &vec);
        prop_assert_eq!(&plain.0, &vdo.buf);
        let mut input = SliceInput::new(&vec);
        prop_assert_eq!(input.read_vint().unwrap(), v);
        prop_assert_eq!(input.read_vlong().unwrap(), l);
        prop_assert_eq!(input.read_zlong().unwrap(), z);
        let mut back = vec![0u64; g.len()];
        input.read_group_vints(&mut back).unwrap();
        prop_assert_eq!(back, g.iter().map(|&x| x as u64).collect::<Vec<_>>());
    }
}

#[test]
fn vint_round_trips_including_negative_and_boundary_values() {
    for v in [0i32, 1, 127, 128, 16384, -1, -2, i32::MIN, i32::MAX] {
        let mut out = VecDataOutput::new();
        out.write_vint(v);
        let mut input = SliceInput::new(&out.buf);
        assert_eq!(input.read_vint().unwrap(), v, "roundtrip {v}");
    }
}

#[test]
fn vlong_round_trips_boundary_values() {
    for v in [0i64, 1, 127, 128, i64::MAX] {
        let mut out = VecDataOutput::new();
        out.write_vlong(v);
        let mut input = SliceInput::new(&out.buf);
        assert_eq!(input.read_vlong().unwrap(), v, "roundtrip {v}");
    }
}

#[test]
fn zlong_round_trips_negative_and_positive() {
    for v in [0i64, 1, -1, i64::MIN, i64::MAX] {
        let mut out = VecDataOutput::new();
        out.write_zlong(v);
        let mut input = SliceInput::new(&out.buf);
        assert_eq!(input.read_zlong().unwrap(), v, "roundtrip {v}");
    }
}

#[test]
fn zint_round_trips_and_is_compact_for_small_negatives() {
    for v in [0i32, 1, -1, -64, 63, i32::MIN, i32::MAX] {
        let mut out = VecDataOutput::new();
        out.write_zint(v);
        let mut input = SliceInput::new(&out.buf);
        assert_eq!(input.read_zint().unwrap(), v, "roundtrip {v}");
    }
    // The point of zigzag: -1 is one byte here, five through write_vint.
    let mut z = VecDataOutput::new();
    z.write_zint(-1);
    let mut v = VecDataOutput::new();
    v.write_vint(-1);
    assert_eq!(z.len(), 1);
    assert_eq!(v.len(), 5);
}

#[test]
fn group_vints_round_trip_through_read_group_vints() {
    // Widths 1..4 in every slot, plus a 1..3-value tail that must fall
    // back to plain vints.
    for extra in 0..4usize {
        let mut values: Vec<u32> = vec![
            0,
            0xFF,
            0x0100,
            0xFFFF,
            0x01_0000,
            0xFF_FFFF,
            0x0100_0000,
            u32::MAX,
        ];
        values.extend((0..extra).map(|i| 1000 + i as u32));
        let mut out = VecDataOutput::new();
        out.write_group_vints(&values);

        let mut dst = vec![0u64; values.len()];
        let mut input = SliceInput::new(&out.buf);
        input.read_group_vints(&mut dst).unwrap();
        assert_eq!(
            dst,
            values.iter().map(|&v| v as u64).collect::<Vec<_>>(),
            "tail of {extra}"
        );
        assert_eq!(input.remaining(), 0, "no bytes left over (tail of {extra})");
    }
}

#[test]
fn group_vints_flag_byte_packs_widths_most_significant_first() {
    // Hand-checked against GroupVIntUtil.writeGroupVInts: widths 1,2,3,4
    // pack as (0<<6)|(1<<4)|(2<<2)|3 == 0x1B, then the values follow
    // little-endian, truncated to their width.
    let mut out = VecDataOutput::new();
    out.write_group_vints(&[0x01, 0x0203, 0x04_0506, 0x0708_090A]);
    assert_eq!(
        out.buf,
        vec![0x1B, 0x01, 0x03, 0x02, 0x06, 0x05, 0x04, 0x0A, 0x09, 0x08, 0x07]
    );
}

#[test]
fn group_vints_fewer_than_four_values_are_all_tail_vints() {
    let mut out = VecDataOutput::new();
    out.write_group_vints(&[1, 2, 3]);
    assert_eq!(out.buf, vec![1, 2, 3], "no flag byte for a partial group");
}

#[test]
#[should_panic(expected = "cannot write negative vLong")]
fn write_vlong_rejects_negative_in_debug() {
    // Java's writeVLong throws IllegalArgumentException for this.
    VecDataOutput::new().write_vlong(-1);
}

#[test]
fn string_round_trips() {
    let mut out = VecDataOutput::new();
    out.write_string("hello world");
    let mut input = SliceInput::new(&out.buf);
    assert_eq!(input.read_string().unwrap(), "hello world");
}

#[test]
fn fixed_width_round_trips() {
    let mut out = VecDataOutput::new();
    out.write_i16(-1234);
    out.write_i32(-123_456_789);
    out.write_i64(-123_456_789_012_345);
    out.write_be_u32(0xDEAD_BEEF);
    out.write_be_u64(0x0102_0304_0506_0708);
    let mut input = SliceInput::new(&out.buf);
    assert_eq!(input.read_i16().unwrap(), -1234);
    assert_eq!(input.read_i32().unwrap(), -123_456_789);
    assert_eq!(input.read_i64().unwrap(), -123_456_789_012_345);
    assert_eq!(input.read_be_u32().unwrap(), 0xDEAD_BEEF);
    assert_eq!(input.read_be_u64().unwrap(), 0x0102_0304_0506_0708);
}

#[test]
fn map_and_set_of_strings_round_trip() {
    let mut out = VecDataOutput::new();
    out.write_map_of_strings(&[
        ("a".to_string(), "1".to_string()),
        ("b".to_string(), "2".to_string()),
    ]);
    out.write_set_of_strings(&["x".to_string(), "y".to_string(), "z".to_string()]);
    let mut input = SliceInput::new(&out.buf);
    assert_eq!(
        input.read_map_of_strings().unwrap(),
        vec![
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "2".to_string())
        ]
    );
    assert_eq!(
        input.read_set_of_strings().unwrap(),
        vec!["x".to_string(), "y".to_string(), "z".to_string()]
    );
}

#[test]
fn vec_u8_implements_data_output_directly() {
    let mut buf: Vec<u8> = Vec::new();
    buf.write_vint(300);
    let mut input = SliceInput::new(&buf);
    assert_eq!(input.read_vint().unwrap(), 300);
}

#[test]
fn vec_data_output_len_is_empty_and_into_inner() {
    let mut out = VecDataOutput::new();
    assert!(out.is_empty());
    assert_eq!(out.len(), 0);
    out.write_byte(1);
    out.write_byte(2);
    assert!(!out.is_empty());
    assert_eq!(out.len(), 2);
    assert_eq!(out.into_inner(), vec![1, 2]);
}
