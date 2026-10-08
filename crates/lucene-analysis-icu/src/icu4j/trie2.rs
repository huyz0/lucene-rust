//! `com.ibm.icu.impl.Trie2_32` and `Trie2_16` (read side): the code point
//! tries ICU's collation data (32-bit CE32s) and case data (16-bit
//! properties, `ucase.icu`) are stored in.
//!
//! Serialized form (`Trie2.createFromSerialized`): a 16-byte header --
//! signature `Tri2` (either byte order), options (value width in the low
//! nibble; only 32-bit is read here), index length, shifted data length,
//! the null index-2 and data offsets, shifted high start -- then the index
//! (`u16`s) and the data (`i32`s, or for a 16-bit trie `u16`s following
//! the index in the same array). Lookups are Java's: the BMP through one
//! index level (lead surrogate code units through their own block), the
//! supplementary planes below `highStart` through two, and the single
//! high value above it. Rust-forced change: every index read is
//! bounds-checked and a corrupt trie reads its error value.

use crate::icu4j::binary::ByteReader;
use crate::IcuError;

const SHIFT_1: i32 = 6 + 5;
const SHIFT_2: i32 = 5;
const INDEX_SHIFT: i32 = 2;
const DATA_MASK: i32 = (1 << SHIFT_2) - 1;
const INDEX_2_MASK: i32 = (1 << (SHIFT_1 - SHIFT_2)) - 1;
const LSCP_INDEX_2_OFFSET: i32 = 0x10000 >> SHIFT_2;
const OMITTED_BMP_INDEX_1_LENGTH: i32 = 0x10000 >> SHIFT_1;
const INDEX_1_OFFSET: i32 = LSCP_INDEX_2_OFFSET + (0x400 >> SHIFT_2) + (0x800 >> 6);
const DATA_GRANULARITY: i32 = 1 << INDEX_SHIFT;
const BAD_UTF8_DATA_OFFSET: usize = 0x80;

/// `Trie2_32` (or, read with [`Trie2_32::from_serialized_16`], a
/// `Trie2_16` whose values are widened to `i32`).
#[derive(Debug, Clone)]
pub struct Trie2_32 {
    index: Vec<u16>,
    data32: Vec<i32>,
    index_length: usize,
    bits32: bool,
    data_offset: i32,
    high_start: i32,
    high_value_index: i32,
    error_value: i32,
}

impl Trie2_32 {
    /// `Trie2_32.createFromSerialized`: reads a 32-bit `Trie2` at the
    /// reader's position, leaving it after the trie.
    pub fn from_serialized(r: &mut ByteReader<'_>) -> Result<Trie2_32, IcuError> {
        Self::from_serialized_width(r, true)
    }

    /// `Trie2_16.createFromSerialized`: a 16-bit `Trie2` at the reader's
    /// position.
    pub fn from_serialized_16(r: &mut ByteReader<'_>) -> Result<Trie2_32, IcuError> {
        Self::from_serialized_width(r, false)
    }

    fn from_serialized_width(r: &mut ByteReader<'_>, bits32: bool) -> Result<Trie2_32, IcuError> {
        let outer = r.is_big_endian();
        let signature = r.i32()?;
        match signature {
            0x5472_6932 => {}
            0x3269_7254 => r.set_big_endian(!outer),
            _ => {
                return Err(IcuError::with_kind(
                    crate::IcuErrorKind::IllegalArgument,
                    "Buffer does not contain a serialized UTrie2",
                ))
            }
        }
        let result = Self::read_body(r, bits32);
        r.set_big_endian(outer);
        result
    }

    // ARITH: the header fields are u16s widened to i32/usize, shifted by at
    // most 11 bits.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_body(r: &mut ByteReader<'_>, bits32: bool) -> Result<Trie2_32, IcuError> {
        let options = r.u16()?;
        let index_length = usize::from(r.u16()?);
        let shifted_data_length = i32::from(r.u16()?);
        let _index2_null_offset = r.u16()?;
        let data_null_offset = usize::from(r.u16()?);
        let shifted_high_start = i32::from(r.u16()?);
        if options & 0xf > 1 {
            return Err(IcuError::with_kind(
                crate::IcuErrorKind::IllegalArgument,
                "UTrie2 serialized format error.",
            ));
        }
        if (options & 0xf == 1) != bits32 {
            return Err(IcuError::new("UTrie2: unexpected data width"));
        }
        let data_length = shifted_data_length << INDEX_SHIFT;
        let index = r.u16s(index_length)?;
        let n = usize::try_from(data_length).unwrap_or(0);
        let data32 = if bits32 {
            r.i32s(n)?
        } else {
            r.u16s(n)?.into_iter().map(i32::from).collect()
        };
        if data32.len() <= BAD_UTF8_DATA_OFFSET || data32.len() <= data_null_offset {
            return Err(IcuError::new("UTrie2: data too short"));
        }
        Ok(Trie2_32 {
            error_value: data32[BAD_UTF8_DATA_OFFSET],
            index,
            data32,
            index_length,
            bits32,
            high_start: shifted_high_start << SHIFT_1,
            // A 16-bit trie's data follow its index in one array; the index
            // entries and the high value index count from the index start.
            data_offset: if bits32 { 0 } else { index_length as i32 },
            high_value_index: data_length - DATA_GRANULARITY
                + if bits32 { 0 } else { index_length as i32 },
        })
    }

    /// `getSerializedLength()`.
    pub fn serialized_length(&self) -> usize {
        16usize
            .saturating_add(self.index_length.saturating_mul(2))
            .saturating_add(
                self.data32
                    .len()
                    .saturating_mul(if self.bits32 { 4 } else { 2 }),
            )
    }

    #[inline]
    fn idx(&self, i: i32) -> i32 {
        usize::try_from(i)
            .ok()
            .and_then(|i| self.index.get(i))
            .map_or(0, |&v| i32::from(v))
    }

    #[inline]
    fn data(&self, i: i32) -> i32 {
        usize::try_from(i.wrapping_sub(self.data_offset))
            .ok()
            .and_then(|i| self.data32.get(i))
            .copied()
            .unwrap_or(self.error_value)
    }

    /// `get(codePoint)`.
    // ARITH: index values are u16s (shifted left by 2), code points below
    // 0x110000.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn get(&self, cp: i32) -> i32 {
        if cp >= 0 {
            if cp < 0xd800 || (cp > 0xdbff && cp <= 0xffff) {
                let ix = (self.idx(cp >> SHIFT_2) << INDEX_SHIFT) + (cp & DATA_MASK);
                return self.data(ix);
            }
            if cp <= 0xffff {
                let ix = self.idx(LSCP_INDEX_2_OFFSET + ((cp - 0xd800) >> SHIFT_2));
                return self.data((ix << INDEX_SHIFT) + (cp & DATA_MASK));
            }
            if cp < self.high_start {
                let ix = (INDEX_1_OFFSET - OMITTED_BMP_INDEX_1_LENGTH) + (cp >> SHIFT_1);
                let ix = self.idx(ix) + ((cp >> SHIFT_2) & INDEX_2_MASK);
                let ix = self.idx(ix);
                return self.data((ix << INDEX_SHIFT) + (cp & DATA_MASK));
            }
            if cp <= 0x10ffff {
                return self.data(self.high_value_index);
            }
        }
        self.error_value
    }

    /// `getFromU16SingleLead(codeUnit)`: a BMP code unit, a lead surrogate
    /// reading its code-unit value rather than its code point's.
    // ARITH: as for `get`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    pub fn get_from_u16_single_lead(&self, c: u16) -> i32 {
        let c = i32::from(c);
        let ix = (self.idx(c >> SHIFT_2) << INDEX_SHIFT) + (c & DATA_MASK);
        self.data(ix)
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    /// A trie whose every value is `v` but the error value 7, built by hand:
    /// index entries all point at the first data block (for a 16-bit trie,
    /// right after the index, which it shares an array with).
    fn trie_width(v: i32, big_endian: bool, bits16: bool) -> Vec<u8> {
        let index_length = if bits16 {
            INDEX_1_OFFSET as usize + 4
        } else {
            INDEX_1_OFFSET as usize + 1
        };
        let data_length = 0x100usize;
        let mut out = Vec::new();
        let put16 = |out: &mut Vec<u8>, x: u16| {
            if big_endian {
                out.extend_from_slice(&x.to_be_bytes())
            } else {
                out.extend_from_slice(&x.to_le_bytes())
            }
        };
        let put32 = |out: &mut Vec<u8>, x: i32| {
            if big_endian {
                out.extend_from_slice(&x.to_be_bytes())
            } else {
                out.extend_from_slice(&x.to_le_bytes())
            }
        };
        put32(&mut out, 0x5472_6932);
        put16(&mut out, if bits16 { 0 } else { 1 });
        put16(&mut out, index_length as u16);
        put16(&mut out, (data_length >> 2) as u16);
        put16(&mut out, 0);
        put16(&mut out, 0);
        put16(&mut out, 0x21); // high start 0x10800
        let block = if bits16 {
            (index_length >> 2) as u16
        } else {
            0
        };
        for i in 0..index_length {
            // The index-1 entry points at the index-2 block at 0.
            put16(
                &mut out,
                if i == INDEX_1_OFFSET as usize {
                    0
                } else {
                    block
                },
            );
        }
        for i in 0..data_length {
            let x = if i == BAD_UTF8_DATA_OFFSET { 7 } else { v };
            if bits16 {
                put16(&mut out, x as u16);
            } else {
                put32(&mut out, x);
            }
        }
        out
    }

    fn trie(v: i32, big_endian: bool) -> Vec<u8> {
        trie_width(v, big_endian, false)
    }

    #[test]
    fn reads_16_bit() {
        for be in [true, false] {
            let bytes = trie_width(42, be, true);
            let mut r = ByteReader::new(&bytes);
            r.set_big_endian(be);
            let t = Trie2_32::from_serialized_16(&mut r).unwrap();
            assert_eq!(t.serialized_length(), bytes.len());
            assert_eq!(t.get(0x41), 42);
            assert_eq!(t.get(0xdc00), 42);
            assert_eq!(t.get(0x10000), 42);
            assert_eq!(t.get(0x10ffff), 42);
            assert_eq!(t.get(0x110000), 7);
            // A 32-bit reader refuses it, and the other way round.
            assert!(Trie2_32::from_serialized(&mut ByteReader::new(&bytes)).is_err());
        }
        let bytes = trie(1, true);
        assert!(Trie2_32::from_serialized_16(&mut ByteReader::new(&bytes)).is_err());
    }

    #[test]
    fn reads_both_orders() {
        for be in [true, false] {
            let bytes = trie(42, be);
            let mut r = ByteReader::new(&bytes);
            r.set_big_endian(be);
            let t = Trie2_32::from_serialized(&mut r).unwrap();
            assert_eq!(t.serialized_length(), bytes.len());
            assert_eq!(t.get(0x41), 42);
            assert_eq!(t.get(0xd800), 42);
            assert_eq!(t.get(0x10000), 42);
            assert_eq!(t.get(0x10ffff), 42);
            assert_eq!(t.get(0x110000), 7);
            assert_eq!(t.get(-1), 7);
            assert_eq!(t.get_from_u16_single_lead(0xd800), 42);
            // Reversed signature: read in the other order.
            let mut r = ByteReader::new(&bytes);
            r.set_big_endian(!be);
            let t = Trie2_32::from_serialized(&mut r).unwrap();
            assert_eq!(t.get(0x41), 42);
            assert_eq!(r.is_big_endian(), !be);
        }
    }

    #[test]
    fn refuses_bad_headers() {
        let mut bytes = trie(1, true);
        bytes[0] = 0;
        assert!(Trie2_32::from_serialized(&mut ByteReader::new(&bytes)).is_err());
        let mut bytes = trie(1, true);
        bytes[5] = 2;
        assert!(Trie2_32::from_serialized(&mut ByteReader::new(&bytes)).is_err());
        let mut bytes = trie(1, true);
        bytes[5] = 0;
        assert!(Trie2_32::from_serialized(&mut ByteReader::new(&bytes)).is_err());
        let mut bytes = trie(1, true);
        bytes[8] = 0;
        bytes[9] = 1; // 4 data entries: shorter than the error-value offset
        assert!(Trie2_32::from_serialized(&mut ByteReader::new(&bytes)).is_err());
        let bytes = trie(1, true);
        assert!(Trie2_32::from_serialized(&mut ByteReader::new(&bytes[..40])).is_err());
    }
}
