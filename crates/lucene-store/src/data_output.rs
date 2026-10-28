//! Encoders for Lucene's `DataOutput` wire primitives — the write-side
//! counterpart of [`crate::data_input::DataInput`], format-compatible with
//! Lucene 10.x. First piece of the write path (PLAN.md Phase 5): an
//! in-memory byte-buffer sink only, no `Directory::createOutput`/fsync/
//! rename lifecycle yet — every codec writer built on this returns owned
//! `Vec<u8>`s that a caller can hand to any `Directory` once one exists.

/// Sequential writer over Lucene-encoded bytes. Mirrors [`crate::data_input::
/// DataInput`]'s method set exactly so every wire primitive round-trips
/// through the same pair of functions.
pub trait DataOutput {
    fn write_byte(&mut self, b: u8);
    fn write_bytes(&mut self, b: &[u8]);

    /// Lucene `writeVInt`. Negative Java ints occupy 5 bytes (the sign bits
    /// keep shifting out), matching `readVInt`'s wrapping-shift decode.
    #[inline]
    fn write_vint(&mut self, v: i32) {
        if v as u32 <= 0x7f {
            self.write_byte(v as u8);
            return;
        }
        let mut buf = [0u8; 10];
        let n = encode_varint(v as u32 as u64, &mut buf);
        self.write_bytes(&buf[..n]);
    }

    /// Lucene `writeVLong` (up to 9 bytes).
    ///
    /// Java throws `IllegalArgumentException` for a negative value — the
    /// format reserves the 10-byte encoding for `writeZLong`/internal use —
    /// so passing one is a caller bug, caught here in debug builds. Release
    /// builds encode it as the same 10 bytes Java's private
    /// `writeSignedVLong` would, which `readVLong` does decode back, rather
    /// than corrupting the stream.
    #[inline]
    fn write_vlong(&mut self, v: i64) {
        debug_assert!(v >= 0, "cannot write negative vLong (got: {v})");
        self.write_vlong_raw_u64(v as u64);
    }

    /// Lucene `writeZLong`: zigzag-encoded vlong; full i64 range.
    #[inline]
    fn write_zlong(&mut self, v: i64) {
        self.write_vlong_raw_u64(lucene_util::zigzag::encode(v));
    }

    /// Raw vlong encode over an already-unsigned 64-bit pattern (shared by
    /// `write_zlong`, which needs to write the zigzag-encoded bit pattern
    /// without reinterpreting it as a signed magnitude again).
    #[inline]
    fn write_vlong_raw_u64(&mut self, v: u64) {
        if v <= 0x7f {
            self.write_byte(v as u8);
            return;
        }
        let mut buf = [0u8; 10];
        let n = encode_varint(v, &mut buf);
        self.write_bytes(&buf[..n]);
    }

    /// Lucene `writeZInt`: zigzag-encoded vint, the 32-bit counterpart of
    /// [`Self::write_zlong`] and the exact inverse of
    /// [`crate::data_input::DataInput::read_zint`]. Small negative values
    /// cost one byte instead of the five `write_vint` would spend.
    #[inline]
    fn write_zint(&mut self, v: i32) {
        self.write_vint(lucene_util::zigzag::encode_i32(v) as i32);
    }

    /// Lucene `DataOutput.writeGroupVInts` / `GroupVIntUtil.writeGroupVInts`:
    /// values in groups of four, each group prefixed by a flag byte packing
    /// the four values' (byte length - 1) into 2 bits each, most significant
    /// pair first; the trailing 1..3 values that don't fill a group are
    /// written as plain vints. Exact inverse of
    /// [`crate::data_input::DataInput::read_group_vints`].
    ///
    /// Java's parameter is an `int[]` whose values are treated as unsigned
    /// when computing widths; this port takes `&[u32]` so that contract is in
    /// the type rather than in a comment.
    // ARITH: writer-side only -- `values` is this port's own in-memory data,
    // never a length read off disk. `i` steps by 4 under `i + 4 <=
    // values.len()`. The `| 1` is the invariant that makes the width
    // arithmetic safe: the operand is never zero, so `leading_zeros() <= 31`,
    // `/ 8` is 0..=3, `lens[k]` is 1..=4, and `lens[k] - 1` is 0..=3. The
    // shifts are by constants.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_group_vints(&mut self, values: &[u32]) {
        let mut i = 0;
        while i + 4 <= values.len() {
            let chunk = &values[i..i + 4];
            // `GroupVIntUtil.numBytes`: `| 1` so zero still occupies one byte.
            let lens = [
                4 - (chunk[0] | 1).leading_zeros() / 8,
                4 - (chunk[1] | 1).leading_zeros() / 8,
                4 - (chunk[2] | 1).leading_zeros() / 8,
                4 - (chunk[3] | 1).leading_zeros() / 8,
            ];
            let flag =
                ((lens[0] - 1) << 6) | ((lens[1] - 1) << 4) | ((lens[2] - 1) << 2) | (lens[3] - 1);
            // One write per group rather than five: the flag byte, then each
            // value's low `len` bytes, assembled on the stack.
            let mut group = [0u8; 17];
            group[0] = flag as u8;
            let mut n = 1;
            for (v, len) in chunk.iter().zip(lens) {
                group[n..n + 4].copy_from_slice(&v.to_le_bytes());
                n += len as usize;
            }
            self.write_bytes(&group[..n]);
            i += 4;
        }
        for &v in &values[i..] {
            self.write_vint(v as i32);
        }
    }

    /// Big-endian u32, as used by `CodecUtil.writeBEInt` (header/footer
    /// magics only -- everything else in the format is little-endian).
    #[inline]
    fn write_be_u32(&mut self, v: u32) {
        self.write_bytes(&v.to_be_bytes());
    }

    /// Big-endian u64, as used by the footer's checksum field.
    #[inline]
    fn write_be_u64(&mut self, v: u64) {
        self.write_bytes(&v.to_be_bytes());
    }

    /// Lucene `DataOutput.writeShort`: plain little-endian i16.
    #[inline]
    fn write_i16(&mut self, v: i16) {
        self.write_bytes(&v.to_le_bytes());
    }

    /// Lucene `DataOutput.writeInt`: plain little-endian i32.
    #[inline]
    fn write_i32(&mut self, v: i32) {
        self.write_bytes(&v.to_le_bytes());
    }

    /// Lucene `DataOutput.writeLong`: plain little-endian i64.
    #[inline]
    fn write_i64(&mut self, v: i64) {
        self.write_bytes(&v.to_le_bytes());
    }

    /// Lucene `DataOutput.writeString`: vint byte-length-prefixed UTF-8
    /// (Lucene uses standard UTF-8 for segment metadata strings, not
    /// modified-UTF-8 -- only a few legacy formats this port doesn't touch
    /// use that).
    #[inline]
    fn write_string(&mut self, s: &str) {
        self.write_vint(s.len() as i32);
        self.write_bytes(s.as_bytes());
    }

    /// Lucene `DataOutput.writeMapOfStrings`: vint count, then `count`
    /// (key, value) string pairs.
    fn write_map_of_strings(&mut self, map: &[(String, String)]) {
        self.write_vint(map.len() as i32);
        for (k, v) in map {
            self.write_string(k);
            self.write_string(v);
        }
    }

    /// Lucene `DataOutput.writeSetOfStrings`: vint count, then `count` strings.
    fn write_set_of_strings(&mut self, set: &[String]) {
        self.write_vint(set.len() as i32);
        for s in set {
            self.write_string(s);
        }
    }
}

/// Lucene's varint byte layout: 7 bits per byte, least significant group
/// first, continuation bit `0x80` on every byte but the last. Writes into
/// `buf` and returns the encoded length, so the callers above hand a whole
/// varint to `write_bytes` in one call (one capacity check, one copy) rather
/// than one `write_byte` per byte.
// ARITH: `n` counts the bytes written; a u64 has at most ten 7-bit groups,
// which is `buf`'s length, so `n` never exceeds 9 when used as an index.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
pub(crate) fn encode_varint(mut v: u64, buf: &mut [u8; 10]) -> usize {
    let mut n = 0;
    while v & !0x7f != 0 {
        buf[n] = (v & 0x7f) as u8 | 0x80;
        v >>= 7;
        n += 1;
    }
    buf[n] = v as u8;
    n + 1
}

/// A `DataOutput` backed by an owned, growable byte buffer -- the only sink
/// this port has today; see the module doc for why there's no on-disk
/// `IndexOutput` yet.
#[derive(Debug, Default, Clone)]
pub struct VecDataOutput {
    pub buf: Vec<u8>,
}

impl VecDataOutput {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.buf
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

impl DataOutput for VecDataOutput {
    #[inline]
    fn write_byte(&mut self, b: u8) {
        self.buf.push(b);
    }

    #[inline]
    fn write_bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    #[inline]
    fn write_vint(&mut self, v: i32) {
        push_varint(&mut self.buf, v as u32 as u64);
    }

    #[inline]
    fn write_vlong_raw_u64(&mut self, v: u64) {
        push_varint(&mut self.buf, v);
    }
}

/// Appends `v`'s varint encoding to `out` straight into its spare capacity,
/// branch-free: the 7-bit groups are spread into byte lanes (`pdep` where the
/// target has BMI2), continuation bits are OR-ed onto every byte but the
/// last, and all 16 lanes are stored at once with only the encoded length
/// claimed. One capacity check and no per-byte loop, so the encoded length's
/// unpredictability costs no branch mispredictions. Byte-for-byte the same as
/// [`encode_varint`] (a property test below holds the two together).
// ARITH: `bits` is 1..=64 (`v | 1` is never zero), so `n = ceil(bits / 7)`
// is 1..=10 and the shift `8 * (n - 1)` is 0..=72 < 128; `len + n` is at
// most `len + 10`, inside the reserved 16.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
pub(crate) fn push_varint(out: &mut Vec<u8>, v: u64) {
    let bits = 64 - (v | 1).leading_zeros();
    let n = bits.div_ceil(7);
    let lanes = spread7(v) as u128 | (((v >> 56) & 0x7f) as u128) << 64 | ((v >> 63) as u128) << 72;
    const CONT: u128 = 0x8080_8080_8080_8080_8080;
    let encoded = lanes | (CONT & ((1u128 << (8 * (n - 1))) - 1));
    out.reserve(16);
    let len = out.len();
    let spare = &mut out.spare_capacity_mut()[..16];
    for (dst, b) in spare.iter_mut().zip(encoded.to_le_bytes()) {
        dst.write(b);
    }
    // SAFETY: `reserve(16)` guarantees capacity for `len + 16` bytes, all 16
    // of which the loop above initialised; only the first `n <= 10` are
    // claimed, the rest stay spare capacity.
    unsafe { out.set_len(len + n as usize) };
}

/// The low 56 bits of `v` as eight 7-bit groups, one per byte lane, least
/// significant first: `pdep(v, 0x7f7f..7f)`.
#[inline(always)]
fn spread7(v: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        // SAFETY: `bmi2` is enabled for this compilation target (checked by
        // the `cfg` above), so the instruction exists on every CPU this
        // binary may run on.
        unsafe { std::arch::x86_64::_pdep_u64(v, 0x7f7f_7f7f_7f7f_7f7f) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        spread7_portable(v)
    }
}

/// [`spread7`] without BMI2: one shift and mask per lane.
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "bmi2", not(test)),
    allow(dead_code)
)]
#[inline(always)]
fn spread7_portable(v: u64) -> u64 {
    (v & 0x7f)
        | (v << 1) & 0x7f00
        | (v << 2) & 0x7f_0000
        | (v << 3) & 0x7f00_0000
        | (v << 4) & 0x7f_0000_0000
        | (v << 5) & 0x7f00_0000_0000
        | (v << 6) & 0x7f_0000_0000_0000
        | (v << 7) & 0x7f00_0000_0000_0000
}

impl DataOutput for Vec<u8> {
    #[inline]
    fn write_byte(&mut self, b: u8) {
        self.push(b);
    }

    #[inline]
    fn write_bytes(&mut self, b: &[u8]) {
        self.extend_from_slice(b);
    }

    #[inline]
    fn write_vint(&mut self, v: i32) {
        push_varint(self, v as u32 as u64);
    }

    #[inline]
    fn write_vlong_raw_u64(&mut self, v: u64) {
        push_varint(self, v);
    }
}

#[cfg(test)]
mod tests {
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
}
