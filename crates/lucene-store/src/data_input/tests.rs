#![cfg(test)]
// Test code opts out of the arithmetic gate at the module boundary: the
// gate exists for values read off disk, and a test's `i + 1` is not one.
// See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use proptest::prelude::*;

/// A backend that implements only the trait's required methods, so every
/// provided method runs in its *default* form. `SliceInput` overrides
/// `read_u32_le`/`read_u32s_le`/`peek_u32_le`; without a second backend
/// those defaults would never execute, and a default that disagrees with
/// its override is exactly the bug that would then ship. Every test below
/// that decodes through this type asserts the two agree.
struct PlainInput<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl DataInput for PlainInput<'_> {
    fn read_byte(&mut self) -> Result<u8> {
        let b = *self
            .buf
            .get(self.pos)
            .ok_or(Error::Eof { offset: self.pos })?;
        self.pos += 1;
        Ok(b)
    }

    fn read_bytes(&mut self, out: &mut [u8]) -> Result<()> {
        let end = self.pos + out.len();
        let src = self
            .buf
            .get(self.pos..end)
            .ok_or(Error::Eof { offset: self.pos })?;
        out.copy_from_slice(src);
        self.pos = end;
        Ok(())
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Deliberately refuses: this is the documented way a backend that
    /// cannot peek forces `read_group_vints` down its safe path.
    fn peek_u32_le(&mut self) -> Result<u32> {
        Err(Error::Eof { offset: self.pos })
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        if self.remaining() < n {
            return Err(Error::Eof { offset: self.pos });
        }
        self.pos += n;
        Ok(())
    }
}

fn error_kind(r: &Result<impl std::fmt::Debug>) -> String {
    format!("{r:?}")
}

proptest! {
    /// The in-slice varint fast paths against the trait's byte
    /// loops on arbitrary bytes (biased toward continuation bits, so
    /// every length, the over-long malformed runs and the near-end
    /// fallbacks all occur): same value or same error, same position, for
    /// a whole stream decoded as vint, vlong, zlong and zint in turn.
    #[test]
    fn varint_fast_paths_match_the_byte_loops(
        raw in proptest::collection::vec((any::<u8>(), 0u8..4), 0..64),
    ) {
        // Three in four bytes get the continuation bit forced on.
        let bytes: Vec<u8> = raw.iter().map(|&(b, k)| if k == 0 { b & 0x7f } else { b | 0x80 }).collect();
        for which in 0..4 {
            let mut fast = SliceInput::new(&bytes);
            let mut plain = PlainInput { buf: &bytes, pos: 0 };
            loop {
                let (f, p) = match which {
                    0 => (error_kind(&fast.read_vint()), error_kind(&plain.read_vint())),
                    1 => (error_kind(&fast.read_vlong()), error_kind(&plain.read_vlong())),
                    2 => (error_kind(&fast.read_zlong()), error_kind(&plain.read_zlong())),
                    _ => (error_kind(&fast.read_zint()), error_kind(&plain.read_zint())),
                };
                prop_assert_eq!(&f, &p, "decoder {} at {}", which, plain.pos);
                prop_assert_eq!(fast.position(), plain.pos);
                if f.starts_with("Err") {
                    break;
                }
            }
        }
    }

    #[test]
    fn u32_i32_and_string_overrides_match_the_defaults(
        bytes in proptest::collection::vec(any::<u8>(), 0..24),
    ) {
        let mut fast = SliceInput::new(&bytes);
        let mut plain = PlainInput { buf: &bytes, pos: 0 };
        prop_assert_eq!(error_kind(&fast.read_u32_le()), error_kind(&plain.read_u32_le()));
        prop_assert_eq!(error_kind(&fast.read_i32()), error_kind(&plain.read_i32()));
        prop_assert_eq!(error_kind(&fast.read_string()), error_kind(&plain.read_string()));
        prop_assert_eq!(fast.position(), plain.pos);
    }
}

/// `SliceInput`'s group-varint fast path decodes exactly what the
/// trait's own path does, value for value and to the same position, for
/// every value width and with the last groups close enough to the end to
/// fall back -- and truncated input fails at the same offset.
#[test]
fn read_group_vints_matches_the_default_implementation() {
    let widths = [0u32, 0x7F, 0xFF, 0x1234, 0xFF_FF, 0x12_3456, 0xFFFF_FFFF, 1];
    let mut bytes = Vec::new();
    let mut want = Vec::new();
    for g in 0..9 {
        let vals: Vec<u32> = (0..4).map(|j| widths[(g * 3 + j) % widths.len()]).collect();
        let mut flag = 0u8;
        let mut body = Vec::new();
        for (j, &v) in vals.iter().enumerate() {
            let n = (4 - v.leading_zeros() / 8).max(1);
            flag |= ((n - 1) as u8) << (6 - 2 * j);
            body.extend_from_slice(&v.to_le_bytes()[..n as usize]);
        }
        bytes.push(flag);
        bytes.extend(body);
        want.extend(vals.iter().map(|&v| u64::from(v)));
    }
    // A plain-vint tail of three values.
    for v in [5u32, 300, 70_000] {
        let mut x = v;
        while x >= 0x80 {
            bytes.push((x as u8) | 0x80);
            x >>= 7;
        }
        bytes.push(x as u8);
        want.push(u64::from(v));
    }
    let n = want.len();
    let mut fast = vec![0u64; n];
    let mut slice = SliceInput::new(&bytes);
    slice.read_group_vints(&mut fast).unwrap();
    let mut slow = vec![0u64; n];
    let mut plain = PlainInput {
        buf: &bytes,
        pos: 0,
    };
    plain.read_group_vints(&mut slow).unwrap();
    assert_eq!(fast, want);
    assert_eq!(slow, want);
    assert_eq!(slice.position(), plain.pos);
    assert_eq!(slice.position(), bytes.len());
    let cut = &bytes[..bytes.len() - 2];
    let mut short = vec![0u64; n];
    let fast_err = SliceInput::new(cut).read_group_vints(&mut short);
    let slow_err = PlainInput { buf: cut, pos: 0 }.read_group_vints(&mut short);
    assert!(fast_err.is_err());
    assert_eq!(format!("{fast_err:?}"), format!("{slow_err:?}"));
}

#[test]
fn read_group_vints_i32_matches_the_u64_decoder() {
    // 40 groups then a 3-value vint tail: the fast path, and -- the last
    // group and the tail, closer to the end than 17 bytes -- the
    // near-end path.
    let widths = [0u32, 0x7F, 0xFF, 0x1234, 0xFF_FF, 0x12_3456, 0xFFFF_FFFF, 1];
    let mut bytes = Vec::new();
    let mut want = Vec::new();
    for g in 0..40 {
        let vals: Vec<u32> = (0..4).map(|j| widths[(g * 5 + j) % widths.len()]).collect();
        let mut flag = 0u8;
        let mut body = Vec::new();
        for (j, &v) in vals.iter().enumerate() {
            let n = (4 - v.leading_zeros() / 8).max(1);
            flag |= ((n - 1) as u8) << (6 - 2 * j);
            body.extend_from_slice(&v.to_le_bytes()[..n as usize]);
        }
        bytes.push(flag);
        bytes.extend(body);
        want.extend(vals);
    }
    for v in [5u32, 300, 70_000] {
        let mut x = v;
        while x >= 0x80 {
            bytes.push((x as u8) | 0x80);
            x >>= 7;
        }
        bytes.push(x as u8);
        want.push(v);
    }
    let n = want.len();
    let mut got = vec![0i32; n];
    let mut input = SliceInput::new(&bytes);
    input.read_group_vints_i32(&mut got).unwrap();
    assert_eq!(input.position(), bytes.len());
    assert!(got.iter().zip(&want).all(|(&g, &w)| g as u32 == w));
    // Only the near-end path: a buffer too short for any fast group.
    let last = &bytes[bytes.len() - 17..];
    let mut wide = vec![0u64; 3];
    let mut narrow = vec![0i32; 3];
    let tail_at = last.len() - 1 - 2 - 3;
    SliceInput::new(&last[tail_at..])
        .read_group_vints(&mut wide)
        .unwrap();
    SliceInput::new(&last[tail_at..])
        .read_group_vints_i32(&mut narrow)
        .unwrap();
    assert!(narrow
        .iter()
        .zip(&wide)
        .all(|(&a, &b)| u64::from(a as u32) == b));
    // Groups with no slack after them -- each its own bytes and nothing
    // more -- decode on the near-end path, and one cut short fails there
    // exactly as the `u64` decoder does.
    let mut at = 0;
    for _ in 0..8 {
        let flag = bytes[at];
        let need = 1
            + (0..4)
                .map(|j| usize::from((flag >> (6 - 2 * j)) & 3) + 1)
                .sum::<usize>();
        let group = &bytes[at..at + need];
        let mut wide = [0u64; 4];
        let mut narrow = [0i32; 4];
        SliceInput::new(group).read_group_vints(&mut wide).unwrap();
        let mut input = SliceInput::new(group);
        input.read_group_vints_i32(&mut narrow).unwrap();
        assert_eq!(input.position(), need);
        assert!(narrow
            .iter()
            .zip(&wide)
            .all(|(&a, &b)| u64::from(a as u32) == b));
        let short = &group[..need - 1];
        let wide_err = SliceInput::new(short).read_group_vints(&mut wide);
        let narrow_err = SliceInput::new(short).read_group_vints_i32(&mut narrow);
        assert!(narrow_err.is_err());
        assert_eq!(format!("{narrow_err:?}"), format!("{wide_err:?}"));
        at += need;
    }
    // A truncated buffer fails as the `u64` decoder does, at the same offset.
    let cut = &bytes[..bytes.len() - 2];
    let mut a = vec![0u64; n];
    let mut b = vec![0i32; n];
    let wide_err = SliceInput::new(cut).read_group_vints(&mut a);
    let narrow_err = SliceInput::new(cut).read_group_vints_i32(&mut b);
    assert!(narrow_err.is_err());
    assert_eq!(format!("{narrow_err:?}"), format!("{wide_err:?}"));
}

#[test]
fn read_u32s_le_matches_the_default_implementation_word_for_word() {
    let bytes: Vec<u8> = (0..64u8).collect();
    let mut fast = [0u32; 16];
    SliceInput::new(&bytes).read_u32s_le(&mut fast).unwrap();
    let mut slow = [0u32; 16];
    PlainInput {
        buf: &bytes,
        pos: 0,
    }
    .read_u32s_le(&mut slow)
    .unwrap();
    assert_eq!(fast, slow);
    assert_eq!(fast[0], u32::from_le_bytes([0, 1, 2, 3]));
    assert_eq!(fast[15], u32::from_le_bytes([60, 61, 62, 63]));
}

#[test]
fn read_u32s_le_advances_by_exactly_four_bytes_per_word() {
    let bytes: Vec<u8> = (0..12u8).collect();
    let mut input = SliceInput::new(&bytes);
    let mut words = [0u32; 2];
    input.read_u32s_le(&mut words).unwrap();
    assert_eq!(input.position(), 8);
    assert_eq!(
        input.read_u32_le().unwrap(),
        u32::from_le_bytes([8, 9, 10, 11])
    );
}

#[test]
fn read_u32s_le_past_the_end_is_eof_and_consumes_nothing() {
    let bytes = [0u8; 7]; // one word plus a partial one
    let mut input = SliceInput::new(&bytes);
    let mut words = [0u32; 2];
    assert!(matches!(
        input.read_u32s_le(&mut words),
        Err(Error::Eof { .. })
    ));
    // The whole run is rejected up front, so the cursor must not have moved
    // -- a partially-consumed input would desynchronize every later read.
    assert_eq!(input.position(), 0);
}

#[test]
fn read_u32s_le_of_nothing_is_a_no_op() {
    let bytes = [1u8, 2, 3, 4];
    let mut input = SliceInput::new(&bytes);
    input.read_u32s_le(&mut []).unwrap();
    assert_eq!(input.position(), 0);
}

/// Local test-only encoders mirroring Lucene's `DataOutput.writeVInt/VLong`
/// algorithm. These exist purely to generate inputs for round-tripping our
/// decoder's edge/error handling — differential correctness against *real*
/// Java-written bytes already lives in `tests/java_fixtures.rs`; this module
/// is about this decoder's own boundary and failure behavior.
fn encode_vint(mut v: i32) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut b = (v & 0x7f) as u8;
        v = ((v as u32) >> 7) as i32;
        if v != 0 {
            b |= 0x80;
            out.push(b);
        } else {
            out.push(b);
            break;
        }
    }
    out
}

fn encode_vlong(mut v: i64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut b = (v & 0x7f) as u8;
        v = ((v as u64) >> 7) as i64;
        if v != 0 {
            b |= 0x80;
            out.push(b);
        } else {
            out.push(b);
            break;
        }
    }
    out
}

fn encode_zlong(v: i64) -> Vec<u8> {
    encode_vlong(lucene_util::zigzag::encode(v) as i64)
}

// --- vint ---

#[test]
fn vint_known_boundary_values() {
    for &(v, expected_len) in &[(0i32, 1), (127, 1), (128, 2), (16383, 2), (16384, 3)] {
        let bytes = encode_vint(v);
        assert_eq!(bytes.len(), expected_len, "encoding length for {v}");
        let mut input = SliceInput::new(&bytes);
        assert_eq!(input.read_vint().unwrap(), v);
    }
}

#[test]
fn vint_negative_uses_five_bytes() {
    let bytes = encode_vint(-1);
    assert_eq!(bytes.len(), 5);
    let mut input = SliceInput::new(&bytes);
    assert_eq!(input.read_vint().unwrap(), -1);
}

#[test]
fn vint_extremes_roundtrip() {
    for v in [i32::MIN, i32::MAX, -1, 0, 1] {
        let bytes = encode_vint(v);
        let mut input = SliceInput::new(&bytes);
        assert_eq!(input.read_vint().unwrap(), v, "v={v}");
    }
}

#[test]
fn vint_malformed_too_many_continuation_bytes() {
    let bytes = [0xFFu8; 10];
    let mut input = SliceInput::new(&bytes);
    assert!(matches!(input.read_vint(), Err(Error::MalformedVarint)));
}

#[test]
fn vint_never_consumes_more_than_javas_five_bytes() {
    // A well-formed vInt is at most five bytes, so a sixth continuation
    // byte is corruption. Lucene 10.5.0 would keep consuming here (its
    // loop is unbounded); this port stops at five and errors. Consuming a
    // sixth *and returning a value* would desynchronize every field after
    // this one in the same file, which is far worse than erroring.
    let bytes = [0xFFu8, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x2A];
    let mut input = SliceInput::new(&bytes);
    assert!(matches!(input.read_vint(), Err(Error::MalformedVarint)));
    assert_eq!(input.position(), 5, "must not read past the fifth byte");
}

#[test]
fn vint_fifth_byte_keeps_only_its_low_four_bits_like_java() {
    // Java computes `i |= (b & 0x7F) << 28` on an int, so bits 4..6 of the
    // fifth byte shift out entirely. 0x7F and 0x0F must decode alike.
    let with_junk = [0x80u8, 0x80, 0x80, 0x80, 0x7F];
    let clean = [0x80u8, 0x80, 0x80, 0x80, 0x0F];
    assert_eq!(
        SliceInput::new(&with_junk).read_vint().unwrap(),
        SliceInput::new(&clean).read_vint().unwrap()
    );
    assert_eq!(
        SliceInput::new(&clean).read_vint().unwrap(),
        0xF000_0000u32 as i32
    );
}

// --- zint (32-bit zigzag vint) ---

#[test]
fn zint_small_negatives_cost_one_byte_and_round_trip() {
    for (v, want_len) in [
        (0i32, 1),
        (-1, 1),
        (1, 1),
        (-64, 1),
        (i32::MIN, 5),
        (i32::MAX, 5),
    ] {
        let bytes = encode_vint(lucene_util::zigzag::encode_i32(v) as i32);
        assert_eq!(bytes.len(), want_len, "encoded length for {v}");
        let mut input = SliceInput::new(&bytes);
        assert_eq!(input.read_zint().unwrap(), v, "v={v}");
    }
}

proptest! {
    #[test]
    fn zint_roundtrips_any_i32(v: i32) {
        let bytes = encode_vint(lucene_util::zigzag::encode_i32(v) as i32);
        let mut input = SliceInput::new(&bytes);
        prop_assert_eq!(input.read_zint().unwrap(), v);
    }
}

#[test]
fn vint_truncated_is_eof() {
    let bytes = [0x80u8]; // continuation bit set, no next byte
    let mut input = SliceInput::new(&bytes);
    assert!(matches!(input.read_vint(), Err(Error::Eof { .. })));
}

proptest! {
    #[test]
    fn vint_roundtrips_any_i32(v: i32) {
        let bytes = encode_vint(v);
        let mut input = SliceInput::new(&bytes);
        prop_assert_eq!(input.read_vint().unwrap(), v);
    }
}

// --- vlong ---

#[test]
fn vlong_known_boundary_values() {
    for &v in &[0i64, 127, 128, i64::MAX] {
        let bytes = encode_vlong(v);
        let mut input = SliceInput::new(&bytes);
        assert_eq!(input.read_vlong().unwrap(), v);
    }
}

#[test]
fn vlong_malformed_too_many_continuation_bytes() {
    let bytes = [0xFFu8; 11];
    let mut input = SliceInput::new(&bytes);
    assert!(matches!(input.read_vlong(), Err(Error::MalformedVarint)));
}

proptest! {
    #[test]
    fn vlong_roundtrips_non_negative_i64(v in 0i64..=i64::MAX) {
        let bytes = encode_vlong(v);
        let mut input = SliceInput::new(&bytes);
        prop_assert_eq!(input.read_vlong().unwrap(), v);
    }
}

// --- zlong ---

#[test]
fn zlong_full_range_boundaries() {
    for &v in &[0i64, -1, 1, i64::MIN, i64::MAX] {
        let bytes = encode_zlong(v);
        let mut input = SliceInput::new(&bytes);
        assert_eq!(input.read_zlong().unwrap(), v, "v={v}");
    }
}

#[test]
fn zlong_malformed_too_many_continuation_bytes() {
    let bytes = [0xFFu8; 12];
    let mut input = SliceInput::new(&bytes);
    assert!(matches!(input.read_zlong(), Err(Error::MalformedVarint)));
}

proptest! {
    #[test]
    fn zlong_roundtrips_any_i64(v: i64) {
        let bytes = encode_zlong(v);
        let mut input = SliceInput::new(&bytes);
        prop_assert_eq!(input.read_zlong().unwrap(), v);
    }
}

// --- group varint ---

#[test]
fn group_vints_single_full_group_various_widths() {
    // widths 1,2,3,4 bytes packed in one flag byte's four slots
    let values: [u32; 4] = [0x00, 0xFF, 0xFFFF, 0xFF_FFFF];
    let flag: u8 = (1 << 4) | (2 << 2) | 3; // lens-1 per value (slot 0 is width-1, so contributes 0)
    let mut bytes = vec![flag];
    bytes.extend_from_slice(&values[0].to_le_bytes()[..1]);
    bytes.extend_from_slice(&values[1].to_le_bytes()[..2]);
    bytes.extend_from_slice(&values[2].to_le_bytes()[..3]);
    bytes.extend_from_slice(&values[3].to_le_bytes()[..4]);

    let mut input = SliceInput::new(&bytes);
    let mut dst = [0u64; 4];
    input.read_group_vints(&mut dst).unwrap();
    assert_eq!(dst, values.map(|v| v as u64));
}

#[test]
fn group_vints_non_multiple_of_four_uses_vint_tail() {
    // 5 values: one full group (4) + a plain-vint tail (1)
    let flag = 0u8; // all 4 values are 1-byte wide
    let mut bytes = vec![flag, 1, 2, 3, 4];
    bytes.extend(encode_vint(42));
    let mut input = SliceInput::new(&bytes);
    let mut dst = [0u64; 5];
    input.read_group_vints(&mut dst).unwrap();
    assert_eq!(dst, [1, 2, 3, 4, 42]);
}

#[test]
fn group_vints_slow_path_when_remaining_lt_4() {
    // Force the `remaining() < 4` branch inside the group loop: the last
    // slot of the group sits exactly at the buffer's tail with <4 bytes left.
    let flag = 0u8; // all four 1-byte values
    let bytes = vec![flag, 9, 8, 7, 6];
    let mut input = SliceInput::new(&bytes);
    let mut dst = [0u64; 4];
    input.read_group_vints(&mut dst).unwrap();
    assert_eq!(dst, [9, 8, 7, 6]);
}

#[test]
fn group_vints_falls_back_when_the_backend_cannot_peek() {
    // `PlainInput::peek_u32_le` always fails, the documented way a
    // non-random-access backend opts out of the branchless path (Java's
    // `instanceof RandomAccessInput` check). The values must still decode,
    // and identically to the peeking backend.
    let flag: u8 = (1 << 4) | (2 << 2) | 3;
    let mut bytes = vec![flag];
    bytes.extend_from_slice(&0x11u32.to_le_bytes()[..1]);
    bytes.extend_from_slice(&0x2233u32.to_le_bytes()[..2]);
    bytes.extend_from_slice(&0x44_5566u32.to_le_bytes()[..3]);
    bytes.extend_from_slice(&0x7788_99AAu32.to_le_bytes()[..4]);
    bytes.extend_from_slice(&[0u8; 16]); // plenty remaining, so only the peek probe decides

    let mut fast = [0u64; 4];
    SliceInput::new(&bytes).read_group_vints(&mut fast).unwrap();
    let mut slow = [0u64; 4];
    PlainInput {
        buf: &bytes,
        pos: 0,
    }
    .read_group_vints(&mut slow)
    .unwrap();
    assert_eq!(fast, [0x11, 0x2233, 0x44_5566, 0x7788_99AA]);
    assert_eq!(fast, slow);
}

#[test]
fn group_vints_last_group_of_a_file_decodes_without_over_reading() {
    // The final group sits within 16 bytes of EOF, so the branchless path
    // is off (Java: `length - pos >= 4 * Integer.BYTES`). Widths here are
    // 4,4,4,1: over-reading the third value would run past the buffer.
    let flag: u8 = (3 << 6) | (3 << 4) | (3 << 2); // widths 4,4,4,1
    let mut bytes = vec![flag];
    bytes.extend_from_slice(&0xAABB_CCDDu32.to_le_bytes());
    bytes.extend_from_slice(&0x0102_0304u32.to_le_bytes());
    bytes.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    bytes.push(0x7F);
    let mut dst = [0u64; 4];
    let mut input = SliceInput::new(&bytes);
    input.read_group_vints(&mut dst).unwrap();
    assert_eq!(dst, [0xAABB_CCDD, 0x0102_0304, 0xDEAD_BEEF, 0x7F]);
    assert_eq!(input.remaining(), 0, "consumed exactly the group's bytes");
}

#[test]
fn group_vints_truncated_group_is_eof_not_a_silent_zero() {
    // Flag says the last value is 4 bytes wide but only 2 are present.
    let flag: u8 = 3;
    let bytes = vec![flag, 1, 2, 3, 4, 5];
    let mut dst = [0u64; 4];
    let mut input = SliceInput::new(&bytes);
    assert!(matches!(
        input.read_group_vints(&mut dst),
        Err(Error::Eof { .. })
    ));
}

// --- big-endian primitives ---

#[test]
fn be_u32_and_u64_assemble_most_significant_byte_first() {
    let bytes = [0x12, 0x34, 0x56, 0x78];
    let mut input = SliceInput::new(&bytes);
    assert_eq!(input.read_be_u32().unwrap(), 0x1234_5678);

    let bytes = [0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02];
    let mut input = SliceInput::new(&bytes);
    assert_eq!(input.read_be_u64().unwrap(), 0x0000_0001_0000_0002);
}

#[test]
fn be_i32_reinterprets_high_bit_as_sign() {
    let bytes = [0xFF, 0xFF, 0xFF, 0xFF];
    let mut input = SliceInput::new(&bytes);
    assert_eq!(input.read_be_i32().unwrap(), -1);
}

// --- strings / collections ---

#[test]
fn string_roundtrip_and_empty() {
    for s in ["", "hello", "héllo wörld", "segments_2"] {
        let mut bytes = encode_vint(s.len() as i32);
        bytes.extend_from_slice(s.as_bytes());
        let mut input = SliceInput::new(&bytes);
        assert_eq!(input.read_string().unwrap(), s);
    }
}

#[test]
fn string_invalid_utf8_is_corrupted_error() {
    let mut bytes = encode_vint(2);
    bytes.extend_from_slice(&[0xFF, 0xFE]); // not valid UTF-8
    let mut input = SliceInput::new(&bytes);
    assert!(matches!(input.read_string(), Err(Error::Corrupted(_))));
}

#[test]
fn string_with_a_negative_length_is_rejected_not_allocated() {
    // Java throws NegativeArraySizeException here; `len as usize` would be
    // ~2^64 and abort the process on allocation failure, which no
    // `catch_unwind` at the FFI boundary can intercept.
    let mut bytes = encode_vint(-1);
    bytes.extend_from_slice(b"whatever");
    let mut input = SliceInput::new(&bytes);
    assert!(matches!(input.read_string(), Err(Error::Corrupted(_))));
}

#[test]
fn string_longer_than_the_input_is_rejected_before_allocating() {
    let mut bytes = encode_vint(1 << 30); // 1GiB claimed, 3 bytes present
    bytes.extend_from_slice(b"abc");
    let mut input = SliceInput::new(&bytes);
    assert!(matches!(input.read_string(), Err(Error::Corrupted(_))));
}

#[test]
fn collection_counts_beyond_the_input_are_rejected_before_reserving() {
    // A three-byte file must not be able to ask for a 2^31-entry Vec.
    let mut bytes = encode_vint(i32::MAX);
    bytes.extend_from_slice(b"ab");
    assert!(matches!(
        SliceInput::new(&bytes).read_map_of_strings(),
        Err(Error::Corrupted(_))
    ));
    assert!(matches!(
        SliceInput::new(&bytes).read_set_of_strings(),
        Err(Error::Corrupted(_))
    ));

    let negative = encode_vint(-3);
    assert!(matches!(
        SliceInput::new(&negative).read_map_of_strings(),
        Err(Error::Corrupted(_))
    ));
    assert!(matches!(
        SliceInput::new(&negative).read_set_of_strings(),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn map_of_strings_zero_one_many() {
    for pairs in [
        vec![],
        vec![("a".to_string(), "1".to_string())],
        vec![
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "2".to_string()),
            ("c".to_string(), "3".to_string()),
        ],
    ] {
        let mut bytes = encode_vint(pairs.len() as i32);
        for (k, v) in &pairs {
            bytes.extend(encode_vint(k.len() as i32));
            bytes.extend_from_slice(k.as_bytes());
            bytes.extend(encode_vint(v.len() as i32));
            bytes.extend_from_slice(v.as_bytes());
        }
        let mut input = SliceInput::new(&bytes);
        assert_eq!(input.read_map_of_strings().unwrap(), pairs);
    }
}

#[test]
fn set_of_strings_zero_one_many() {
    for items in [
        vec![],
        vec!["x".to_string()],
        vec!["x".to_string(), "y".to_string(), "z".to_string()],
    ] {
        let mut bytes = encode_vint(items.len() as i32);
        for s in &items {
            bytes.extend(encode_vint(s.len() as i32));
            bytes.extend_from_slice(s.as_bytes());
        }
        let mut input = SliceInput::new(&bytes);
        assert_eq!(input.read_set_of_strings().unwrap(), items);
    }
}

// --- i16 ---

#[test]
fn i16_little_endian_and_negative() {
    let bytes = 0x0102i16.to_le_bytes();
    let mut input = SliceInput::new(&bytes);
    assert_eq!(input.read_i16().unwrap(), 0x0102);

    let bytes = (-1i16).to_le_bytes();
    let mut input = SliceInput::new(&bytes);
    assert_eq!(input.read_i16().unwrap(), -1);
}

#[test]
fn u16_little_endian_widens_unsigned() {
    // The bit pattern of -1i16 (0xFFFF) read as u16 is 65535, not -1.
    let bytes = (-1i16).to_le_bytes();
    let mut input = SliceInput::new(&bytes);
    assert_eq!(input.read_u16().unwrap(), 0xFFFF);
}

// --- i64 / i64s ---

#[test]
fn i64_and_i64s_little_endian() {
    let bytes = 0x0102_0304_0506_0708i64.to_le_bytes();
    let mut input = SliceInput::new(&bytes);
    assert_eq!(input.read_i64().unwrap(), 0x0102_0304_0506_0708);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1i64.to_le_bytes());
    bytes.extend_from_slice(&(-2i64).to_le_bytes());
    bytes.extend_from_slice(&3i64.to_le_bytes());
    let mut input = SliceInput::new(&bytes);
    let mut dst = [0i64; 3];
    input.read_i64s(&mut dst).unwrap();
    assert_eq!(dst, [1, -2, 3]);
}

#[test]
fn i64s_bulk_override_matches_the_default_implementation() {
    let mut bytes = Vec::new();
    for v in [1i64, -2, 3, i64::MIN, i64::MAX] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut fast = [0i64; 5];
    SliceInput::new(&bytes).read_i64s(&mut fast).unwrap();
    let mut slow = [0i64; 5];
    PlainInput {
        buf: &bytes,
        pos: 0,
    }
    .read_i64s(&mut slow)
    .unwrap();
    assert_eq!(fast, [1, -2, 3, i64::MIN, i64::MAX]);
    assert_eq!(fast, slow);
}

#[test]
fn i64s_past_the_end_is_eof_and_consumes_nothing() {
    let bytes = [0u8; 12]; // one word plus a partial one
    let mut input = SliceInput::new(&bytes);
    let mut words = [0i64; 2];
    assert!(matches!(
        input.read_i64s(&mut words),
        Err(Error::Eof { .. })
    ));
    assert_eq!(input.position(), 0);
}

#[test]
fn i64s_of_nothing_is_a_no_op() {
    let bytes = [1u8; 8];
    let mut input = SliceInput::new(&bytes);
    input.read_i64s(&mut []).unwrap();
    assert_eq!(input.position(), 0);
}

// --- SliceInput cursor mechanics ---

#[test]
fn slice_input_len_position_is_empty() {
    let bytes = [1, 2, 3];
    let mut input = SliceInput::new(&bytes);
    assert_eq!(input.len(), 3);
    assert!(!input.is_empty());
    assert_eq!(input.position(), 0);
    input.read_byte().unwrap();
    assert_eq!(input.position(), 1);

    let empty: [u8; 0] = [];
    assert!(SliceInput::new(&empty).is_empty());
}

#[test]
fn slice_input_seek_out_of_bounds_is_eof() {
    let bytes = [1, 2, 3];
    let mut input = SliceInput::new(&bytes);
    assert!(input.seek(10).is_err());
    assert!(input.seek(3).is_ok()); // exactly at end is valid (no more reads)
}

#[test]
fn slice_input_slice_out_of_bounds_is_eof() {
    let bytes = [1, 2, 3];
    let input = SliceInput::new(&bytes);
    assert!(input.slice(0, 4).is_err());
    assert_eq!(input.slice(1, 3).unwrap(), &[2, 3]);
}

#[test]
fn read_bytes_past_end_is_eof_and_does_not_partially_advance() {
    let bytes = [1, 2, 3];
    let mut input = SliceInput::new(&bytes);
    let mut out = [0u8; 5];
    assert!(matches!(input.read_bytes(&mut out), Err(Error::Eof { .. })));
    // position must be unchanged after a failed read (no torn reads)
    assert_eq!(input.position(), 0);
}

#[test]
fn peek_u32_le_past_end_is_eof() {
    let bytes = [1, 2];
    let mut input = SliceInput::new(&bytes);
    assert!(matches!(input.peek_u32_le(), Err(Error::Eof { .. })));
}

// --- slice_input / clone: IndexInput.slice()/clone() semantics ---

#[test]
fn slice_input_reads_the_addressed_range_zero_based() {
    let bytes = b"0123456789";
    let root = SliceInput::new(bytes);
    let mid = root.slice_input("mid", 3, 4).unwrap();
    assert_eq!(mid.as_slice(), b"3456");
    assert_eq!(mid.position(), 0, "slice starts its own cursor at 0");
}

#[test]
fn slice_input_bounds_respected_reading_past_end_is_error_not_parent_leak() {
    let bytes = b"0123456789";
    let root = SliceInput::new(bytes);
    let mut mid = root.slice_input("mid", 3, 4).unwrap(); // addresses "3456"
    let mut buf = [0u8; 4];
    mid.read_bytes(&mut buf).unwrap();
    assert_eq!(&buf, b"3456");
    // A 5th byte would be '7' in the parent buffer, but the slice's own
    // extent ends here -- must be Eof, not silently returning '7'.
    assert!(matches!(mid.read_byte(), Err(Error::Eof { .. })));
}

#[test]
fn slice_input_offset_or_length_out_of_range_is_error() {
    let bytes = b"01234";
    let root = SliceInput::new(bytes);
    assert!(root.slice_input("d", 3, 10).is_err()); // extends past end
    assert!(root.slice_input("d", 10, 1).is_err()); // offset past end
    assert!(root.slice_input("d", u64::MAX, 1).is_err()); // doesn't fit usize on any sane target
}

#[test]
fn slice_input_zero_length_is_empty_and_immediately_eof() {
    let bytes = b"01234";
    let root = SliceInput::new(bytes);
    let mut empty = root.slice_input("empty", 2, 0).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.as_slice(), b"");
    assert!(matches!(empty.read_byte(), Err(Error::Eof { .. })));
}

#[test]
fn slice_of_a_slice_is_supported_offsets_relative_to_the_slice() {
    let bytes = b"0123456789";
    let root = SliceInput::new(bytes);
    let mid = root.slice_input("mid", 2, 6).unwrap(); // "234567"
    assert_eq!(mid.as_slice(), b"234567");
    let inner = mid.slice_input("inner", 1, 3).unwrap(); // relative to mid -> "345"
    assert_eq!(inner.as_slice(), b"345");
    assert_eq!(inner.position(), 0);
}

#[test]
fn slice_input_has_independent_file_pointer_from_parent_and_siblings() {
    let bytes = b"abcdefghij";
    let mut root = SliceInput::new(bytes);
    root.read_byte().unwrap(); // advance parent's own cursor to 1 ('b')

    let mut slice_a = root.slice_input("a", 0, 5).unwrap(); // "abcde"
    let mut slice_b = root.slice_input("b", 5, 5).unwrap(); // "fghij"

    // Interleave reads through both slices and the still-live parent;
    // each must track its own position independently.
    assert_eq!(slice_a.read_byte().unwrap(), b'a');
    assert_eq!(slice_b.read_byte().unwrap(), b'f');
    assert_eq!(root.read_byte().unwrap(), b'b'); // parent continues from where it left off (pos 1)
    assert_eq!(slice_a.read_byte().unwrap(), b'b');
    assert_eq!(slice_b.read_byte().unwrap(), b'g');
    assert_eq!(slice_a.position(), 2);
    assert_eq!(slice_b.position(), 2);
    assert_eq!(root.position(), 2);
}

#[test]
fn clone_has_independent_file_pointer() {
    let bytes = b"clone-me!!";
    let mut original = SliceInput::new(bytes);
    original.read_byte().unwrap(); // advance to 1, matching Lucene's clone() copying the position too

    let mut cloned = original.clone();
    assert_eq!(cloned.position(), original.position());

    // Reading through the clone must not move the original's pointer, and
    // vice versa.
    assert_eq!(cloned.read_byte().unwrap(), b'l');
    assert_eq!(original.position(), 1, "original untouched by clone's read");
    assert_eq!(original.read_byte().unwrap(), b'l'); // original independently re-reads the same byte at pos 1
    assert_eq!(cloned.position(), 2, "clone untouched by original's read");
}
