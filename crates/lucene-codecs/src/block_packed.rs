//! Port of `org.apache.lucene.util.packed.BlockPackedReaderIterator` (decode
//! side of `BlockPackedWriter`) — read-only, **decode-once, not lazy**.
//!
//! A sequence of values is split into fixed-size blocks (64 values each,
//! for term vectors). Each block is self-describing: a one-byte token
//! (`bitsPerValue << 1 | minValueEquals0`), an optional zigzag-encoded
//! `minValue` (omitted when it's exactly 0), then `bitsPerValue`-wide
//! deltas-from-`minValue` for every value in the block, packed via
//! [`crate::packed_ints`]'s MSB-first bitstream (a `bitsPerValue` of 0 means
//! every value in the block equals `minValue`, with no packed data at all).
//!
//! Ported as a single `decode_all` that materializes the whole sequence,
//! not Java's seekable iterator (`next`/`skip`) -- this port doesn't need
//! partial reads, matching the decode-fully choice already made for
//! `IndexedDISI`, stored fields, and the terms dictionary.

use lucene_store::data_input::DataInput;
use lucene_store::Result;

use crate::packed_ints;

const BLOCK_SIZE: i64 = 64;
const MIN_VALUE_EQUALS_0: u32 = 1;

/// Decodes `total_value_count` values written by `BlockPackedWriter`.
/// Reads nothing at all if `total_value_count == 0` (matches the writer,
/// which emits zero blocks for an empty stream).
pub(crate) fn decode_all(input: &mut impl DataInput, total_value_count: i64) -> Result<Vec<i64>> {
    let mut out = Vec::with_capacity(total_value_count.max(0) as usize);
    let mut produced: i64 = 0;
    while produced < total_value_count {
        let token = input.read_byte()? as u32;
        let min_value_equals_0 = token & MIN_VALUE_EQUALS_0 != 0;
        let bits_per_value = token >> 1;
        if bits_per_value > 64 {
            return Err(lucene_store::Error::Corrupted(format!(
                "block-packed bitsPerValue out of range: {bits_per_value}"
            )));
        }
        let min_value = if min_value_equals_0 {
            0i64
        } else {
            lucene_util::zigzag::decode(1u64.wrapping_add(input.read_vlong()? as u64))
        };

        let block_value_count = (total_value_count - produced).min(BLOCK_SIZE) as usize;
        if bits_per_value == 0 {
            out.extend(std::iter::repeat_n(min_value, block_value_count));
        } else {
            let byte_len = packed_ints::byte_count(block_value_count as i64, bits_per_value);
            let mut block_bytes = vec![0u8; byte_len];
            input.read_bytes(&mut block_bytes)?;
            for i in 0..block_value_count {
                let raw = packed_ints::get(&block_bytes, bits_per_value, i as i64)?;
                out.push(raw.wrapping_add(min_value));
            }
        }
        produced += block_value_count as i64;
    }
    Ok(out)
}

/// Encode side of [`decode_all`] (`BlockPackedWriter`'s write path) --
/// **always writes `minValue = 0`** rather than Java's per-block min-value
/// optimization: correct (a decoder never needs the min-value shortcut),
/// just not minimal, matching this port's "worst-case width over minimal
/// width" stance for the write path generally. Each 64-value block's
/// `bitsPerValue` is the exact width needed for that block's max value (not
/// rounded to a fixed set of widths -- `packed_ints`, unlike
/// `direct_reader`, allows any width 0..=64). Writes nothing for an empty
/// slice, matching the decoder reading nothing for `total_value_count == 0`.
pub(crate) fn encode_all(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < values.len() {
        let end = (i + BLOCK_SIZE as usize).min(values.len());
        let block = &values[i..end];
        let max = *block.iter().max().unwrap();
        let bits: u32 = if max <= 0 {
            0
        } else {
            64 - (max as u64).leading_zeros()
        };
        let token = (bits << 1) | MIN_VALUE_EQUALS_0;
        out.push(token as u8);
        if bits > 0 {
            out.extend_from_slice(&packed_ints::encode(block, bits));
        }
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::data_input::SliceInput;

    fn write_vlong(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    #[test]
    fn empty_stream_reads_nothing() {
        let mut input = SliceInput::new(&[]);
        assert_eq!(decode_all(&mut input, 0).unwrap(), Vec::<i64>::new());
    }

    #[test]
    fn single_block_all_same_value() {
        // bitsPerValue=0, minValueEquals0=1 (bit0 set -> no minValue byte,
        // implied value 0) -> token=1.
        let bytes = [1u8];
        let mut input = SliceInput::new(&bytes);
        assert_eq!(decode_all(&mut input, 3).unwrap(), vec![0, 0, 0]);
    }

    #[test]
    fn single_block_constant_nonzero_value() {
        // bitsPerValue=0, minValueEquals0=0 -> token=0, then zigzag(1+vlong)=minValue.
        // want minValue=42: 1+vlong == zigzag_encode(42) => vlong = zigzag_encode(42)-1
        let mut bytes = vec![0u8]; // token: bits=0, min_equals_0=false
        let target = lucene_util::zigzag::encode(42) - 1;
        write_vlong(&mut bytes, target);
        let mut input = SliceInput::new(&bytes);
        assert_eq!(decode_all(&mut input, 4).unwrap(), vec![42, 42, 42, 42]);
    }

    #[test]
    fn single_block_bit_packed_deltas_from_min() {
        // 5 values: [10, 12, 11, 10, 13] -> min=10, deltas=[0,2,1,0,3], bpv=2.
        let min_value = 10i64;
        let deltas = [0u8, 2, 1, 0, 3];
        let bits_per_value = 2u32;
        let mut bytes = Vec::new();
        let token = bits_per_value << 1; // min_equals_0=false (min=10 != 0)
        bytes.push(token as u8);
        let target = lucene_util::zigzag::encode(min_value) - 1;
        write_vlong(&mut bytes, target);
        // pack deltas MSB-first, 2 bits each: values 0,2,1,0,3 ->
        // byte0 = 00_10_01_00 (first 4 values), byte1 = 11_000000 (5th value)
        let mut packed_bits: u16 = 0;
        let mut nbits: usize = 0;
        for &d in &deltas {
            packed_bits = (packed_bits << 2) | d as u16;
            nbits += 2;
        }
        // left-align into full bytes
        let total_bytes = nbits.div_ceil(8);
        let packed_bits = packed_bits << (total_bytes * 8 - nbits);
        for i in (0..total_bytes).rev() {
            bytes.push(((packed_bits >> (i * 8)) & 0xFF) as u8);
        }
        let mut input = SliceInput::new(&bytes);
        assert_eq!(decode_all(&mut input, 5).unwrap(), vec![10, 12, 11, 10, 13]);
    }

    #[test]
    fn multiple_blocks_across_65_values() {
        // 65 values forces a second block (block size 64): first block all
        // zero (bpv=0, min=0), second block a single constant value 7.
        let mut bytes = vec![1u8]; // block 0: bpv=0, min_equals_0=true (min=0)
        bytes.push(0u8); // block 1: bpv=0, min_equals_0=false
        let target = lucene_util::zigzag::encode(7) - 1;
        write_vlong(&mut bytes, target);
        let mut input = SliceInput::new(&bytes);
        let values = decode_all(&mut input, 65).unwrap();
        assert_eq!(values.len(), 65);
        assert!(values[..64].iter().all(|&v| v == 0));
        assert_eq!(values[64], 7);
    }

    #[test]
    fn invalid_bits_per_value_rejected() {
        // token >> 1 must be <= 64; use a byte where bits_per_value = 127.
        let bytes = [0xFEu8]; // 0xFE >> 1 = 127
        let mut input = SliceInput::new(&bytes);
        assert!(decode_all(&mut input, 1).is_err());
    }

    #[test]
    fn encode_all_empty_writes_nothing() {
        assert_eq!(encode_all(&[]), Vec::<u8>::new());
    }

    #[test]
    fn encode_all_round_trips_through_decode_all_single_block() {
        let values = vec![0i64, 2, 1, 0, 3, 100];
        let encoded = encode_all(&values);
        let mut input = SliceInput::new(&encoded);
        assert_eq!(decode_all(&mut input, values.len() as i64).unwrap(), values);
    }

    #[test]
    fn encode_all_round_trips_across_multiple_blocks() {
        let values: Vec<i64> = (0..130).map(|i| (i * 37) % 1000).collect();
        let encoded = encode_all(&values);
        let mut input = SliceInput::new(&encoded);
        assert_eq!(decode_all(&mut input, values.len() as i64).unwrap(), values);
    }

    #[test]
    fn encode_all_all_zero_block_uses_zero_bits() {
        let values = vec![0i64; 64];
        let encoded = encode_all(&values);
        assert_eq!(encoded, vec![1u8]); // token: bits=0, min_equals_0=1
    }
}
