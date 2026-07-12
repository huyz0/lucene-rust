//! Port of `org.apache.lucene.util.packed.PackedInts.Format.PACKED`'s bulk
//! bit-packing (`BulkOperationPacked`) — a *different* convention from
//! [`crate::direct_reader`] (which ports `DirectReader`/`DirectWriter`):
//! values are packed **MSB-first as one contiguous bitstream** across the
//! whole byte array, with no per-value byte alignment, versus
//! `direct_reader`'s LSB-first-within-byte, whitelisted-width scheme.
//! Term vectors uses both conventions for different arrays in the same
//! file, so both need to exist side by side.
//!
//! This is the "headerless flat array" case (a fixed `bits_per_value` for
//! every value, no min-value, no block splitting) — used directly for
//! term vectors' distinct-field-numbers array, and as the per-block body
//! decoder inside [`crate::block_packed`].

use lucene_store::Result;

/// Reads the `index`-th `bits_per_value`-wide value from `data`, where
/// values are packed MSB-first as one contiguous bitstream (bit 7 of byte 0
/// is the first bit of value 0).
pub(crate) fn get(data: &[u8], bits_per_value: u32, index: i64) -> Result<i64> {
    let bit_pos = (index as u128) * (bits_per_value as u128);
    let byte_pos =
        usize::try_from(bit_pos >> 3).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
    let bit_offset = (bit_pos & 7) as u32;
    let total_bits = bit_offset + bits_per_value;
    let n_bytes = total_bits.div_ceil(8) as usize;

    let bytes = data
        .get(byte_pos..byte_pos + n_bytes)
        .ok_or(lucene_store::Error::Eof { offset: byte_pos })?;
    let mut acc: u128 = 0;
    for &b in bytes {
        acc = (acc << 8) | b as u128;
    }
    let shift = (n_bytes as u32) * 8 - bit_offset - bits_per_value;
    let mask: u128 = if bits_per_value >= 128 {
        u128::MAX
    } else {
        (1u128 << bits_per_value) - 1
    };
    Ok(((acc >> shift) & mask) as i64)
}

/// Number of bytes needed to pack `count` values of `bits_per_value` width
/// (`PackedInts.Format.PACKED.byteCount`): `ceil(count * bits_per_value / 8)`.
pub(crate) fn byte_count(count: i64, bits_per_value: u32) -> usize {
    ((count as i128 * bits_per_value as i128 + 7) / 8) as usize
}

/// Encode side of [`get`]: packs `values` MSB-first as one contiguous
/// bitstream, the exact inverse of `get`'s formula. `bits_per_value` may be
/// any width `0..=64` (unlike [`crate::direct_reader`], this convention has
/// no whitelist of supported widths) -- `bits_per_value == 0` writes nothing
/// (every value is assumed to be 0, matching `get`'s masked-to-zero read).
pub(crate) fn encode(values: &[i64], bits_per_value: u32) -> Vec<u8> {
    let n_bytes = byte_count(values.len() as i64, bits_per_value);
    let mut out = vec![0u8; n_bytes];
    let mut bit_pos: u64 = 0;
    for &v in values {
        let mut remaining = bits_per_value;
        while remaining > 0 {
            let byte_idx = (bit_pos >> 3) as usize;
            let bit_off = (bit_pos & 7) as u32;
            let free = 8 - bit_off;
            let take = remaining.min(free);
            let shift_in_value = remaining - take;
            let mask: u64 = if take == 64 {
                u64::MAX
            } else {
                (1u64 << take) - 1
            };
            let bits_val = ((v as u64) >> shift_in_value) & mask;
            out[byte_idx] |= (bits_val as u8) << (free - take);
            bit_pos += take as u64;
            remaining -= take;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_direct_reader_style_widths_but_different_bit_order() {
        // bits=8: byte-aligned, so MSB-first vs LSB-first within a byte
        // doesn't matter -- same as a plain byte read.
        let data = [0x12u8, 0x34, 0x56];
        assert_eq!(get(&data, 8, 0).unwrap(), 0x12);
        assert_eq!(get(&data, 8, 1).unwrap(), 0x34);
        assert_eq!(get(&data, 8, 2).unwrap(), 0x56);
    }

    #[test]
    fn sub_byte_width_is_msb_first_not_lsb_first() {
        // bits=4: MSB-first means byte 0xAB packs value 0=0xA (high nibble),
        // value 1=0xB (low nibble) -- opposite of direct_reader's LSB-first.
        let data = [0xABu8];
        assert_eq!(get(&data, 4, 0).unwrap(), 0xA);
        assert_eq!(get(&data, 4, 1).unwrap(), 0xB);
    }

    #[test]
    fn arbitrary_width_five_bits_spans_byte_boundary() {
        // 5-bit values packed MSB-first: 0b10101_01010_101... etc.
        // value0=0b10101=21, value1=0b01010=10, packed into bits:
        // byte0=10101010=0xAA, byte1=1......=0x80 (only top bit of value1's
        // remainder used, rest zero-padded for this 2-value test).
        let data = [0b1010_1010u8, 0b1000_0000u8];
        assert_eq!(get(&data, 5, 0).unwrap(), 0b10101);
        assert_eq!(get(&data, 5, 1).unwrap(), 0b01010);
    }

    #[test]
    fn byte_count_matches_java_format_packed() {
        assert_eq!(byte_count(0, 5), 0);
        assert_eq!(byte_count(1, 5), 1);
        assert_eq!(byte_count(8, 5), 5); // 40 bits = 5 bytes exactly
        assert_eq!(byte_count(3, 5), 2); // 15 bits -> 2 bytes
    }

    #[test]
    fn out_of_range_is_error() {
        let data = [0u8; 1];
        assert!(get(&data, 16, 5).is_err());
    }

    #[test]
    fn encode_round_trips_through_get_for_various_widths() {
        for bits in [1u32, 3, 4, 5, 8, 12, 16, 20, 31] {
            let max = if bits >= 63 {
                i64::MAX
            } else {
                (1i64 << bits) - 1
            };
            let values: Vec<i64> = (0..17).map(|i| (i as i64 * 7) % (max.max(1) + 1)).collect();
            let encoded = encode(&values, bits);
            assert_eq!(encoded.len(), byte_count(values.len() as i64, bits));
            for (i, &v) in values.iter().enumerate() {
                assert_eq!(
                    get(&encoded, bits, i as i64).unwrap(),
                    v,
                    "bits={bits} i={i}"
                );
            }
        }
        // bits=0: every value is assumed/decoded as 0, regardless of input.
        let encoded = encode(&[5, 9, 0], 0);
        assert_eq!(encoded, Vec::<u8>::new());
        assert_eq!(get(&encoded, 0, 0).unwrap(), 0);
    }

    #[test]
    fn encode_empty_values_produces_empty_output() {
        assert_eq!(encode(&[], 5), Vec::<u8>::new());
    }
}
