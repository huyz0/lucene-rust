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

    /// Lucene `writeVLong` (caller must pass a non-negative value; up to 9 bytes).
    #[inline]
    fn write_vlong(&mut self, v: i64) {
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
