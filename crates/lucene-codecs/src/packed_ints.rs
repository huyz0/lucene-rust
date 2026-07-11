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
}
