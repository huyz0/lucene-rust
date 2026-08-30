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
    // `bits_per_value` is a token field off a `.tvd` chunk header on both call
    // paths, and every shift below is only in range once it is bounded. Both
    // current callers do bound it before calling -- `block_packed::decode_all`
    // rejects a width above 64 outright, and term vectors masks the token to
    // five bits -- but this is where the shifts actually live, so this is
    // where the invariant belongs: a `pub(crate)` primitive should not depend
    // on every future caller remembering. `PackedInts.Format.PACKED` cannot
    // represent a width above 64 (`BlockPackedReaderIterator` throws on
    // exactly this), so anything wider is corruption.
    if bits_per_value > 64 {
        return Err(lucene_store::Error::Corrupted(format!(
            "packed-ints bitsPerValue out of range: {bits_per_value}"
        )));
    }
    // `index as u128` sign-extends, so a negative index becomes ~2^128 and
    // the multiply below overflows -- a debug-build panic before any bounds
    // check runs. Both current callers pass a non-negative index, but so does
    // every caller of the `bits_per_value` check above, and the same argument
    // applies: this is where the arithmetic lives.
    let Ok(index) = u64::try_from(index) else {
        return Err(lucene_store::Error::Corrupted(format!(
            "packed-ints index must be non-negative, got {index}"
        )));
    };
    // ARITH: the product is computed in `u128` from a `u64` and a value now
    // known to be `<= 64`, so it cannot overflow. `bit_offset` is masked to
    // `0..=7`, so `total_bits <= 71` and `n_bytes <= 9`.
    #[allow(clippy::arithmetic_side_effects)]
    let (byte_pos, bit_offset, n_bytes) = {
        let bit_pos = (index as u128) * (bits_per_value as u128);
        let byte_pos =
            usize::try_from(bit_pos >> 3).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
        let bit_offset = (bit_pos & 7) as u32;
        let total_bits = bit_offset + bits_per_value;
        (byte_pos, bit_offset, total_bits.div_ceil(8) as usize)
    };

    // `index` is a value ordinal off disk, so `byte_pos` is unbounded on a
    // 32-bit target (the `try_from` above admits up to `u32::MAX`, where
    // `byte_pos + n_bytes` would wrap). One comparison against the slice
    // length makes the range below provably in bounds on every target.
    if byte_pos > data.len() {
        return Err(lucene_store::Error::Eof { offset: byte_pos });
    }
    // ARITH: `byte_pos <= data.len() <= isize::MAX` and `n_bytes <= 9`.
    #[allow(clippy::arithmetic_side_effects)]
    let range = byte_pos..byte_pos + n_bytes;
    let bytes = data
        .get(range)
        .ok_or(lucene_store::Error::Eof { offset: byte_pos })?;
    let mut acc: u128 = 0;
    // ARITH: `bytes.len() == n_bytes <= 9`, so `acc` holds at most 72 bits of
    // a 128-bit accumulator.
    #[allow(clippy::arithmetic_side_effects)]
    for &b in bytes {
        acc = (acc << 8) | b as u128;
    }
    // ARITH: `n_bytes * 8 = ceil(total_bits / 8) * 8 >= total_bits =
    // bit_offset + bits_per_value`, so the two subtractions cannot underflow,
    // and `n_bytes * 8 <= 72`. `bits_per_value <= 64 < 128`, so the mask's
    // shift is in range and its result is at least 1.
    #[allow(clippy::arithmetic_side_effects)]
    let value = {
        let shift = (n_bytes as u32) * 8 - bit_offset - bits_per_value;
        let mask: u128 = (1u128 << bits_per_value) - 1;
        ((acc >> shift) & mask) as i64
    };
    Ok(value)
}

/// Number of bytes needed to pack `count` values of `bits_per_value` width
/// (`PackedInts.Format.PACKED.byteCount`): `ceil(count * bits_per_value / 8)`.
// ARITH: the product is computed in `u128` from a `u64` and a `u32`, so it
// cannot overflow. `count` is unsigned rather than Java's `long` on purpose:
// as an `i64` a negative count produced a negative quotient, and `as usize`
// turned that into a gigantic length that callers then handed to
// `vec![0u8; n]`.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn byte_count(count: u64, bits_per_value: u32) -> usize {
    ((count as u128 * bits_per_value as u128).div_ceil(8)) as usize
}

/// Encode side of [`get`]: packs `values` MSB-first as one contiguous
/// bitstream, the exact inverse of `get`'s formula. `bits_per_value` may be
/// any width `0..=64` (unlike [`crate::direct_reader`], this convention has
/// no whitelist of supported widths) -- `bits_per_value == 0` writes nothing
/// (every value is assumed to be 0, matching `get`'s masked-to-zero read).
// ARITH: this is the encode side and `bits_per_value` is chosen by the caller
// from its own data, never read off disk. `bit_off` is masked to `0..=7`, so
// `free = 8 - bit_off` is in `1..=8`; `take = min(remaining, free)` is in
// `0..=8`, so `shift_in_value = remaining - take` cannot underflow, `1u64 <<
// take` is in range, `free - take` is in `0..=8`, `remaining -= take`
// terminates at 0, and `bit_pos` advances by at most `values.len() *
// bits_per_value` bits, which `byte_count` already sized `out` for.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn encode(values: &[i64], bits_per_value: u32) -> Vec<u8> {
    let n_bytes = byte_count(values.len() as u64, bits_per_value);
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
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

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

    /// `bits_per_value` reaches `get` from a `.tvd` chunk token. Above 64 the
    /// shift `n_bytes * 8 - bit_offset - bits_per_value` underflows -- a panic
    /// in a debug build -- before any bounds check runs.
    /// `index as u128` sign-extends, so a negative index used to become
    /// ~2^128 and overflow the `index * bits_per_value` multiply -- a
    /// debug-build panic before any bounds check ran.
    #[test]
    fn negative_index_is_a_decode_error_not_a_multiply_overflow() {
        let data = [0u8; 64];
        for index in [-1i64, i64::MIN, -12345] {
            assert!(get(&data, 8, index).is_err(), "index={index}");
        }
    }

    #[test]
    fn bits_per_value_above_64_is_a_decode_error_not_an_underflow() {
        let data = [0u8; 64];
        for bits in [65u32, 100, 128, 200, u32::MAX] {
            assert!(get(&data, bits, 0).is_err(), "bits={bits}");
        }
        // 64 is the widest `PackedInts.Format.PACKED` can represent and must
        // still decode.
        assert_eq!(get(&[0xFFu8; 8], 64, 0).unwrap(), -1);
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
            assert_eq!(encoded.len(), byte_count(values.len() as u64, bits));
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
