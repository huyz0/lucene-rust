//! `com.ibm.icu.util.CodePointTrie`: ICU's immutable code point trie
//! (`UCPTrie`, signature `Tri3`), the lookup structure of the `.nrm`
//! normalization data and the `.brk` break rules.
//!
//! Serialized form (`CodePointTrie.fromBinary`): a 16-byte header -- `int32
//! signature` (`Tri3`), `uint16 options` (bits 15..12 data length high
//! bits, 11..8 data-null-offset high bits, 7..6 type `FAST`/`SMALL`, 5..3
//! reserved zero, 2..0 value width 16/32/8), `uint16 indexLength`, `uint16
//! dataLength`, `uint16 index3NullOffset`, `uint16 dataNullOffset`, `uint16
//! shiftedHighStart` -- then `indexLength` UTF-16 index units and
//! `dataLength` values. A `FAST` trie indexes the whole BMP in 64-value
//! blocks; a `SMALL` trie only below U+1000; everything else goes through
//! the three-level index. The last two data values are the error value
//! (for out-of-range input) and the value of every code point at or above
//! `highStart`.
//!
//! Only reading is ported (`MutableCodePointTrie`, the builder, is not
//! needed: every trie this crate uses is precompiled ICU data). Rust-forced
//! change: every index read off the data is bounds-checked; a corrupt trie
//! reads the error value instead of throwing
//! `ArrayIndexOutOfBoundsException`.

use crate::icu4j::binary::ByteReader;
use crate::IcuError;

/// `CodePointTrie.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrieType {
    /// `FAST`: the BMP in fast-indexed 64-value blocks.
    Fast,
    /// `SMALL`: only U+0000..U+0FFF fast-indexed.
    Small,
}

/// `CodePointTrie.ValueWidth`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueWidth {
    /// `BITS_16`.
    Bits16,
    /// `BITS_32`.
    Bits32,
    /// `BITS_8`.
    Bits8,
}

const FAST_SHIFT: u32 = 6;
const FAST_DATA_MASK: u32 = (1 << FAST_SHIFT) - 1;
const SMALL_MAX: u32 = 0xfff;
const ERROR_VALUE_NEG_DATA_OFFSET: usize = 1;
const HIGH_VALUE_NEG_DATA_OFFSET: usize = 2;
const BMP_INDEX_LENGTH: u32 = 0x10000 >> FAST_SHIFT;
const SMALL_LIMIT: u32 = 0x1000;
const SMALL_INDEX_LENGTH: u32 = SMALL_LIMIT >> FAST_SHIFT;
const SHIFT_3: u32 = 4;
const SHIFT_2: u32 = 5 + SHIFT_3;
const SHIFT_1: u32 = 5 + SHIFT_2;
const SHIFT_2_3: u32 = SHIFT_2 - SHIFT_3;
const SHIFT_1_2: u32 = SHIFT_1 - SHIFT_2;
const OMITTED_BMP_INDEX_1_LENGTH: u32 = 0x10000 >> SHIFT_1;
const INDEX_2_MASK: u32 = (1 << SHIFT_1_2) - 1;
const INDEX_3_MASK: u32 = (1 << SHIFT_2_3) - 1;
const SMALL_DATA_MASK: u32 = (1 << SHIFT_3) - 1;
const OPTIONS_DATA_LENGTH_MASK: u16 = 0xf000;
const OPTIONS_DATA_NULL_OFFSET_MASK: u16 = 0xf00;
const OPTIONS_RESERVED_MASK: u16 = 0x38;
const OPTIONS_VALUE_BITS_MASK: u16 = 7;

/// A read-only `CodePointTrie`.
#[derive(Debug, Clone)]
pub struct CodePointTrie {
    trie_type: TrieType,
    index: Vec<u16>,
    /// The values, whatever their serialized width (8- and 16-bit values
    /// are unsigned).
    data: Vec<i32>,
    high_start: u32,
    /// `dataLength - ERROR_VALUE_NEG_DATA_OFFSET`, saturated at 0.
    error_index: usize,
    /// `dataLength - HIGH_VALUE_NEG_DATA_OFFSET`, saturated at 0.
    high_index: usize,
    error_value: i32,
}

impl CodePointTrie {
    /// `CodePointTrie.fromBinary(type, valueWidth, bytes)`: `None` for
    /// either requirement accepts what the header says.
    pub fn from_binary(
        required_type: Option<TrieType>,
        required_width: Option<ValueWidth>,
        r: &mut ByteReader<'_>,
    ) -> Result<CodePointTrie, IcuError> {
        let outer = r.is_big_endian();
        let result = Self::read(required_type, required_width, r);
        r.set_big_endian(outer);
        result
    }

    fn read(
        required_type: Option<TrieType>,
        required_width: Option<ValueWidth>,
        r: &mut ByteReader<'_>,
    ) -> Result<CodePointTrie, IcuError> {
        if r.remaining() < 16 {
            return Err(IcuError::new("Buffer too short for a CodePointTrie header"));
        }
        match r.i32()? {
            0x5472_6933 => {}
            0x3369_7254 => {
                let be = r.is_big_endian();
                r.set_big_endian(!be);
            }
            _ => {
                return Err(IcuError::new(
                    "Buffer does not contain a serialized CodePointTrie",
                ))
            }
        }
        let options = r.u16()?;
        let index_length = usize::from(r.u16()?);
        let mut data_length = u32::from(r.u16()?);
        let _index3_null_offset = r.u16()?;
        let mut data_null_offset = u32::from(r.u16()?);
        let shifted_high_start = u32::from(r.u16()?);
        let actual_type = match (options >> 6) & 3 {
            0 => TrieType::Fast,
            1 => TrieType::Small,
            _ => {
                return Err(IcuError::new(
                    "CodePointTrie data header has an unsupported type",
                ))
            }
        };
        let actual_width = match options & OPTIONS_VALUE_BITS_MASK {
            0 => ValueWidth::Bits16,
            1 => ValueWidth::Bits32,
            2 => ValueWidth::Bits8,
            _ => {
                return Err(IcuError::new(
                    "CodePointTrie data header has an unsupported value width",
                ))
            }
        };
        if options & OPTIONS_RESERVED_MASK != 0 {
            return Err(IcuError::new(
                "CodePointTrie data header has unsupported options",
            ));
        }
        if required_type.is_some_and(|t| t != actual_type)
            || required_width.is_some_and(|w| w != actual_width)
        {
            return Err(IcuError::new(
                "CodePointTrie data header has a different type or value width than required",
            ));
        }
        data_length |= u32::from(options & OPTIONS_DATA_LENGTH_MASK) << 4;
        data_null_offset |= u32::from(options & OPTIONS_DATA_NULL_OFFSET_MASK) << 8;
        let _ = data_null_offset;
        let high_start = shifted_high_start << SHIFT_2;
        let data_length = data_length as usize;
        let index = r.u16s(index_length)?;
        let data: Vec<i32> = match actual_width {
            ValueWidth::Bits16 => r.u16s(data_length)?.into_iter().map(i32::from).collect(),
            ValueWidth::Bits32 => r.i32s(data_length)?,
            ValueWidth::Bits8 => r.take(data_length)?.iter().map(|&b| i32::from(b)).collect(),
        };
        let error_index = data.len().saturating_sub(ERROR_VALUE_NEG_DATA_OFFSET);
        let high_index = data.len().saturating_sub(HIGH_VALUE_NEG_DATA_OFFSET);
        let error_value = data.get(error_index).copied().unwrap_or(0);
        Ok(CodePointTrie {
            trie_type: actual_type,
            index,
            data,
            high_start,
            error_index,
            high_index,
            error_value,
        })
    }

    /// `getType()`.
    pub fn trie_type(&self) -> TrieType {
        self.trie_type
    }

    #[inline]
    fn idx(&self, i: u32) -> u32 {
        self.index.get(i as usize).map_or(0, |&v| u32::from(v))
    }

    /// `fastIndex(c)`.
    #[inline]
    fn fast_index(&self, c: u32) -> usize {
        // ARITH: a u16 index entry plus a 6-bit offset fits in u32.
        #[allow(clippy::arithmetic_side_effects)]
        let i = self.idx(c >> FAST_SHIFT) + (c & FAST_DATA_MASK);
        i as usize
    }

    /// `smallIndex(type, c)`.
    fn small_index(&self, c: u32) -> usize {
        if c >= self.high_start {
            return self.high_index;
        }
        self.internal_small_index(c)
    }

    /// `internalSmallIndex(type, c)`.
    // ARITH: c < 0x110000, so `c >> SHIFT_1` < 0x44; index entries are u16
    // and the two-bit group extension below adds at most 0x30000: every
    // sum stays far below u32::MAX.
    #[allow(clippy::arithmetic_side_effects)]
    fn internal_small_index(&self, c: u32) -> usize {
        let mut i1 = c >> SHIFT_1;
        if self.trie_type == TrieType::Fast {
            i1 += BMP_INDEX_LENGTH - OMITTED_BMP_INDEX_1_LENGTH;
        } else {
            i1 += SMALL_INDEX_LENGTH;
        }
        let i3_block = self.idx(self.idx(i1) + ((c >> SHIFT_2) & INDEX_2_MASK));
        let mut i3 = (c >> SHIFT_3) & INDEX_3_MASK;
        let data_block = if i3_block & 0x8000 == 0 {
            self.idx(i3_block + i3)
        } else {
            let mut b = (i3_block & 0x7fff) + (i3 & !7) + (i3 >> 3);
            i3 &= 7;
            let mut d = (self.idx(b) << (2 + (2 * i3))) & 0x30000;
            b += 1;
            d |= self.idx(b + i3);
            d
        };
        (data_block + (c & SMALL_DATA_MASK)) as usize
    }

    /// `cpIndex(c)`.
    #[inline]
    fn cp_index(&self, c: i32) -> usize {
        if c >= 0 {
            let u = c as u32;
            let fast_max = match self.trie_type {
                TrieType::Fast => 0xffff,
                TrieType::Small => SMALL_MAX,
            };
            if u <= fast_max {
                return self.fast_index(u);
            } else if u <= 0x10ffff {
                return self.small_index(u);
            }
        }
        self.error_index
    }

    #[inline]
    fn value(&self, i: usize) -> i32 {
        self.data.get(i).copied().unwrap_or(self.error_value)
    }

    /// `get(c)`: the value of code point `c` (the error value for a
    /// negative or out-of-range `c`).
    #[inline]
    pub fn get(&self, c: i32) -> i32 {
        self.value(self.cp_index(c))
    }

    /// `Fast.bmpGet(c)`, for `0 <= c <= 0xffff` on a fast trie.
    #[inline(always)]
    pub fn bmp_get(&self, c: u32) -> i32 {
        if self.trie_type == TrieType::Fast && c <= 0xffff {
            self.value(self.fast_index(c))
        } else {
            self.get(c as i32)
        }
    }

    /// `Fast.suppGet(c)`, for a supplementary `c`.
    #[inline]
    pub fn supp_get(&self, c: u32) -> i32 {
        self.value(self.small_index(c))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk
    use super::*;

    /// A serialized trie: `values` per BMP 64-block via a fast index (every
    /// block its own data block), supplementary via one shared block.
    fn build(width: u16, small: bool, sig_swapped: bool) -> Vec<u8> {
        // A tiny SMALL/FAST trie whose highStart is 0x10000 (so every
        // supplementary code point reads the high value) and whose BMP
        // index points every block at data block 0 except block 1.
        let blocks = if small {
            SMALL_INDEX_LENGTH
        } else {
            BMP_INDEX_LENGTH
        } as usize;
        let mut index = vec![0u16; blocks];
        index[1] = 64;
        let mut data: Vec<u32> = (0..128).map(|i| i as u32).collect();
        data.push(0xEE); // high value
        data.push(0xFF); // error value
        let mut out = Vec::new();
        let sig: u32 = if sig_swapped {
            0x3369_7254
        } else {
            0x5472_6933
        };
        out.extend_from_slice(&sig.to_be_bytes());
        let ty = if small { 1u16 << 6 } else { 0 };
        let w = |out: &mut Vec<u8>, v: u16| {
            if sig_swapped {
                out.extend_from_slice(&v.to_le_bytes())
            } else {
                out.extend_from_slice(&v.to_be_bytes())
            }
        };
        w(&mut out, ty | width);
        w(&mut out, index.len() as u16);
        w(&mut out, data.len() as u16);
        w(&mut out, 0x7fff);
        w(&mut out, 0);
        w(&mut out, (0x10000u32 >> SHIFT_2) as u16);
        for &i in &index {
            w(&mut out, i);
        }
        for &d in &data {
            match width {
                0 => w(&mut out, d as u16),
                1 => {
                    if sig_swapped {
                        out.extend_from_slice(&d.to_le_bytes())
                    } else {
                        out.extend_from_slice(&d.to_be_bytes())
                    }
                }
                _ => out.push(d as u8),
            }
        }
        out
    }

    #[test]
    fn lookups_every_width_and_type() {
        for width in 0..3u16 {
            for small in [false, true] {
                for swapped in [false, true] {
                    let bytes = build(width, small, swapped);
                    let mut r = ByteReader::new(&bytes);
                    let t = CodePointTrie::from_binary(None, None, &mut r).unwrap();
                    assert!(r.is_big_endian());
                    assert_eq!(t.trie_type() == TrieType::Small, small);
                    assert_eq!(t.get(5), 5);
                    assert_eq!(t.get(64 + 7), 64 + 7);
                    assert_eq!(t.bmp_get(64 + 7), 64 + 7);
                    assert_eq!(t.get(-1), 0xFF);
                    assert_eq!(t.get(0x110000), 0xFF);
                    assert_eq!(t.get(0x10000), 0xEE);
                    assert_eq!(t.supp_get(0x10400), 0xEE);
                    if small {
                        // above SMALL_MAX and below highStart: the 3-level
                        // index, which here points at entry 0.
                        assert_eq!(t.get(0x5000), 0);
                        assert_eq!(t.bmp_get(0x5000), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn header_errors() {
        let good = build(0, false, false);
        let mut r = ByteReader::new(&good[..10]);
        assert!(CodePointTrie::from_binary(None, None, &mut r).is_err());
        let mut b = good.clone();
        b[0] = 0;
        assert!(CodePointTrie::from_binary(None, None, &mut ByteReader::new(&b)).is_err());
        let mut b = good.clone();
        b[5] = 3 << 6;
        assert!(CodePointTrie::from_binary(None, None, &mut ByteReader::new(&b)).is_err());
        let mut b = good.clone();
        b[5] = 3;
        assert!(CodePointTrie::from_binary(None, None, &mut ByteReader::new(&b)).is_err());
        let mut b = good.clone();
        b[5] = 0x08;
        assert!(CodePointTrie::from_binary(None, None, &mut ByteReader::new(&b)).is_err());
        let e =
            CodePointTrie::from_binary(Some(TrieType::Small), None, &mut ByteReader::new(&good));
        assert!(e.is_err());
        let e =
            CodePointTrie::from_binary(None, Some(ValueWidth::Bits32), &mut ByteReader::new(&good));
        assert!(e.is_err());
        assert!(CodePointTrie::from_binary(
            None,
            None,
            &mut ByteReader::new(&good[..good.len() - 1])
        )
        .is_err());
    }

    #[test]
    fn corrupt_index_reads_error_value() {
        let mut bytes = build(0, false, false);
        // Point BMP block 2 far past the data.
        let at = 16 + 2 * 2;
        bytes[at] = 0xff;
        bytes[at + 1] = 0xff;
        let t = CodePointTrie::from_binary(None, None, &mut ByteReader::new(&bytes)).unwrap();
        assert_eq!(t.get(128), 0xFF);
    }
}
