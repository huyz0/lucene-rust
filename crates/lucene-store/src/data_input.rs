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

    /// Lucene `DataInput.readInt`: plain little-endian i32 (distinct from the
    /// header/footer's big-endian magics).
    #[inline]
    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32_le()? as i32)
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
/// `Copy`-cheap: cloning is how Lucene's `IndexInput.clone()` maps to Rust.
#[derive(Clone)]
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
