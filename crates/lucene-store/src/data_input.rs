//! Decoders for Lucene's `DataInput` wire primitives (vint, vlong, zigzag,
//! group-varint, little-endian fixed-width), format-compatible with Lucene 10.x.
//!
//! Design notes (see PLAN.md §3.5): this is a trait with provided methods over a
//! minimal byte-cursor core, and `SliceInput` is the monomorphized zero-copy
//! implementation used for mmap'd regions — decode loops inline fully, no dyn.

use crate::error::{Error, Result};

/// Sequential reader over Lucene-encoded bytes.
///
/// Implementors provide raw byte access; all wire-format decoding lives in the
/// provided methods so every backend decodes identically.
pub trait DataInput {
    fn read_byte(&mut self) -> Result<u8>;
    fn read_bytes(&mut self, out: &mut [u8]) -> Result<()>;

    /// Bytes remaining, used to pick the fast branchless path in group-varint.
    fn remaining(&self) -> usize;

    /// Absolute-position peek of 4 LE bytes without bounds concern beyond `remaining`.
    /// Default implementation goes through `read_bytes`; `SliceInput` overrides with
    /// an unaligned load.
    #[inline]
    fn read_u32_le(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.read_bytes(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// Lucene `readVInt`. Negative Java ints occupy 5 bytes.
    #[inline]
    fn read_vint(&mut self) -> Result<i32> {
        let mut b = self.read_byte()?;
        let mut v = (b & 0x7f) as i32;
        let mut shift = 7;
        while b & 0x80 != 0 {
            if shift > 28 + 7 {
                return Err(Error::MalformedVarint);
            }
            b = self.read_byte()?;
            // Final (5th) byte contributes its low 4 bits into the sign area,
            // matching Java's unchecked shift semantics.
            v |= ((b & 0x7f) as i32).wrapping_shl(shift);
            shift += 7;
        }
        Ok(v)
    }

    /// Lucene `readVLong` (non-negative on the wire; up to 9 bytes).
    #[inline]
    fn read_vlong(&mut self) -> Result<i64> {
        let mut b = self.read_byte()?;
        let mut v = (b & 0x7f) as i64;
        let mut shift = 7;
        while b & 0x80 != 0 {
            if shift >= 64 {
                return Err(Error::MalformedVarint);
            }
            b = self.read_byte()?;
            v |= ((b & 0x7f) as i64).wrapping_shl(shift);
            shift += 7;
        }
        Ok(v)
    }

    /// Lucene `readZLong`: zigzag-decoded vlong; full i64 range.
    #[inline]
    fn read_zlong(&mut self) -> Result<i64> {
        // Same varint framing as vlong but must accept the 10-byte encoding of
        // values with the top bit set.
        let mut v: u64 = 0;
        let mut shift = 0u32;
        loop {
            let b = self.read_byte()?;
            v |= ((b & 0x7f) as u64).wrapping_shl(shift);
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 70 {
                return Err(Error::MalformedVarint);
            }
        }
        Ok(lucene_util::zigzag::decode(v))
    }

    /// Lucene group-varint (`GroupVIntUtil.readGroupVInts`): full groups of 4
    /// values (1 flag byte + 1..4 LE bytes each), then a plain-vint tail.
    /// Values are unsigned 32-bit, widened to u64 like Lucene's `long[]` variant.
    fn read_group_vints(&mut self, dst: &mut [u64]) -> Result<()> {
        const MASKS: [u32; 4] = [0xFF, 0xFFFF, 0xFF_FFFF, u32::MAX];
        let limit = dst.len();
        let mut i = 0;
        while i + 4 <= limit {
            let flag = self.read_byte()? as usize;
            let lens = [(flag >> 6) & 3, (flag >> 4) & 3, (flag >> 2) & 3, flag & 3];
            for (j, &n_minus_1) in lens.iter().enumerate() {
                // Branchless fast path: over-read 4 LE bytes and mask, when safe.
                let v = if self.remaining() >= 4 {
                    let v = self.peek_u32_le()? & MASKS[n_minus_1];
                    self.skip(n_minus_1 + 1)?;
                    v
                } else {
                    let mut b = [0u8; 4];
                    self.read_bytes(&mut b[..n_minus_1 + 1])?;
                    u32::from_le_bytes(b)
                };
                dst[i + j] = v as u64;
            }
            i += 4;
        }
        while i < limit {
            dst[i] = self.read_vint()? as u32 as u64;
            i += 1;
        }
        Ok(())
    }

    /// Peek 4 LE bytes without advancing. Backends that can't peek may return Eof
    /// to force the safe path in `read_group_vints`.
    fn peek_u32_le(&mut self) -> Result<u32>;

    fn skip(&mut self, n: usize) -> Result<()>;

    /// Big-endian i32, as used by `CodecUtil.readBEInt` (header/footer magics only —
    /// everything else in the format is little-endian).
    #[inline]
    fn read_be_u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.read_bytes(&mut b)?;
        Ok(u32::from_be_bytes(b))
    }

    /// Big-endian i64, as used by `CodecUtil.readBELong` (footer checksum).
    #[inline]
    fn read_be_u64(&mut self) -> Result<u64> {
        let hi = self.read_be_u32()? as u64;
        let lo = self.read_be_u32()? as u64;
        Ok((hi << 32) | lo)
    }

    /// Signed convenience wrapper: many callers of `CodecUtil.readBEInt` treat the
    /// result as a signed count (numSegments, delCount, ...).
    #[inline]
    fn read_be_i32(&mut self) -> Result<i32> {
        Ok(self.read_be_u32()? as i32)
    }

    /// Lucene `DataInput.readString`: vint byte-length prefix, UTF-8 payload.
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_vint()? as usize;
        let mut buf = vec![0u8; len];
        self.read_bytes(&mut buf)?;
        String::from_utf8(buf).map_err(|_| Error::Corrupted("invalid UTF-8 string".into()))
    }

    /// Lucene `DataInput.readShort`: plain little-endian i16.
    #[inline]
    fn read_i16(&mut self) -> Result<i16> {
        let mut b = [0u8; 2];
        self.read_bytes(&mut b)?;
        Ok(i16::from_le_bytes(b))
    }

    /// `Short.toUnsignedInt(readShort())`: the same little-endian 2 bytes,
    /// widened as unsigned (used by block/doc-count headers, which are never
    /// negative even though Java stores them as `short`).
    #[inline]
    fn read_u16(&mut self) -> Result<u16> {
        let mut b = [0u8; 2];
        self.read_bytes(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    /// Lucene `DataInput.readInt`: plain little-endian i32 (distinct from the
    /// header/footer's big-endian magics).
    #[inline]
    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32_le()? as i32)
    }

    /// Lucene `DataInput.readLong`: plain little-endian i64.
    #[inline]
    fn read_i64(&mut self) -> Result<i64> {
        let mut b = [0u8; 8];
        self.read_bytes(&mut b)?;
        Ok(i64::from_le_bytes(b))
    }

    /// Lucene `DataInput.readLongs`: `count` consecutive little-endian i64s.
    fn read_i64s(&mut self, dst: &mut [i64]) -> Result<()> {
        for slot in dst.iter_mut() {
            *slot = self.read_i64()?;
        }
        Ok(())
    }

    /// Lucene `DataInput.readMapOfStrings`: vint count, then `count` (key, value) string pairs.
    fn read_map_of_strings(&mut self) -> Result<Vec<(String, String)>> {
        let count = self.read_vint()? as usize;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let k = self.read_string()?;
            let v = self.read_string()?;
            out.push((k, v));
        }
        Ok(out)
    }

    /// Lucene `DataInput.readSetOfStrings`: vint count, then `count` strings.
    fn read_set_of_strings(&mut self) -> Result<Vec<String>> {
        let count = self.read_vint()? as usize;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.read_string()?);
        }
        Ok(out)
    }
}

/// Zero-copy cursor over a byte slice (e.g. an mmap'd file region).
/// Cheap to `Clone` (a slice ref + a `usize`): cloning is how Lucene's
/// `IndexInput.clone()` maps to Rust.
#[derive(Clone, Debug)]
pub struct SliceInput<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> SliceInput<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.buf.len() {
            return Err(self.eof());
        }
        self.pos = pos;
        Ok(())
    }

    /// Zero-copy view of `[from..to)` of the underlying buffer, independent of the
    /// cursor position. Used by `codec_util` to compute the footer's CRC-32 over
    /// the exact byte range Lucene checksummed.
    pub fn slice(&self, from: usize, to: usize) -> Result<&'a [u8]> {
        self.buf.get(from..to).ok_or_else(|| self.eof())
    }

    /// Port of `IndexInput.slice(sliceDescription, offset, length)`: returns a
    /// new, independent-file-pointer cursor over `[offset, offset+length)` of
    /// this input's own buffer, addressed from 0 as if it were its own file —
    /// exactly what a merge reading one source segment's sub-range (a stored
    /// field/doc-values/points block inside a shared `.cfs`, say) needs: a
    /// reader it can seek/advance freely without disturbing the parent or any
    /// other slice/clone of it.
    ///
    /// `description` mirrors Lucene's signature (useful context for future
    /// error messages) but isn't otherwise interpreted.
    ///
    /// Slicing a slice is supported: the result is itself a `SliceInput`, so
    /// calling `.slice_input(..)` again narrows further, offsets always being
    /// relative to *this* input's own `[0, len())`, not the original root
    /// buffer's addressing.
    ///
    /// Bounds are enforced up front: an out-of-range `offset`/`length` is
    /// rejected here rather than silently granting access to bytes outside
    /// the intended range (e.g. a neighboring sub-file's data).
    pub fn slice_input(
        &self,
        description: &str,
        offset: u64,
        length: u64,
    ) -> Result<SliceInput<'a>> {
        let _ = description;
        let start = usize::try_from(offset).map_err(|_| self.eof())?;
        let len = usize::try_from(length).map_err(|_| self.eof())?;
        let end = start.checked_add(len).ok_or_else(|| self.eof())?;
        let buf = self.buf.get(start..end).ok_or_else(|| self.eof())?;
        Ok(SliceInput { buf, pos: 0 })
    }

    /// Remaining unread bytes `[pos, len())`, independent of `pos`'s effect on
    /// `read_bytes`/etc. Mostly a test/debug convenience (e.g. asserting a
    /// slice's full contents at once).
    pub fn as_slice(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    #[cold]
    fn eof(&self) -> Error {
        Error::Eof { offset: self.pos }
    }
}

impl DataInput for SliceInput<'_> {
    #[inline]
    fn read_byte(&mut self) -> Result<u8> {
        let b = *self.buf.get(self.pos).ok_or_else(|| self.eof())?;
        self.pos += 1;
        Ok(b)
    }

    #[inline]
    fn read_bytes(&mut self, out: &mut [u8]) -> Result<()> {
        let end = self.pos + out.len();
        let src = self.buf.get(self.pos..end).ok_or_else(|| self.eof())?;
        out.copy_from_slice(src);
        self.pos = end;
        Ok(())
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    #[inline]
    fn peek_u32_le(&mut self) -> Result<u32> {
        let src = self
            .buf
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.eof())?;
        Ok(u32::from_le_bytes(src.try_into().unwrap()))
    }

    #[inline]
    fn skip(&mut self, n: usize) -> Result<()> {
        if self.remaining() < n {
            return Err(self.eof());
        }
        self.pos += n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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
}
