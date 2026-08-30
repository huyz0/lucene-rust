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
        let mut v = v as u32;
        loop {
            if v & !0x7f == 0 {
                self.write_byte(v as u8);
                return;
            }
            self.write_byte((v & 0x7f) as u8 | 0x80);
            v >>= 7;
        }
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
        let mut v = v as u64;
        loop {
            if v & !0x7f == 0 {
                self.write_byte(v as u8);
                return;
            }
            self.write_byte((v & 0x7f) as u8 | 0x80);
            v >>= 7;
        }
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
    fn write_vlong_raw_u64(&mut self, mut v: u64) {
        loop {
            if v & !0x7f == 0 {
                self.write_byte(v as u8);
                return;
            }
            self.write_byte((v & 0x7f) as u8 | 0x80);
            v >>= 7;
        }
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
            self.write_byte(flag as u8);
            for (v, len) in chunk.iter().zip(lens) {
                self.write_bytes(&v.to_le_bytes()[..len as usize]);
            }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_input::{DataInput, SliceInput};

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
