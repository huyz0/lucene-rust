//! Port of `org.apache.lucene.util.packed.DirectReader.getInstance(...).get(index)`.
//!
//! Generalized into a single bit-position formula instead of Java's thirteen
//! width-specialized `DirectPackedReaderN` classes: those exist to give the
//! JIT a monomorphic call site per width, a concern this port doesn't have
//! yet (no hot per-doc-value loop). Shared by [`crate::doc_values`] (plain
//! bit-packed value arrays) and [`crate::direct_monotonic`] (each block's
//! deltas-from-expected-average array).

use lucene_store::Result;

/// `bits_per_value` must be one of the widths `DirectWriter` supports (the
/// caller validates this at parse time). `index` addresses the `index`-th
/// `bits_per_value`-wide value packed little-endian (LSB-first within each
/// byte) starting at byte 0 of `slice`.
pub(crate) fn get(slice: &[u8], bits_per_value: u8, index: i64) -> Result<i64> {
    // `index * bits_per_value` fits comfortably in u64: `index` addresses an
    // element of an in-memory-decoded array, itself bounded by `slice.len() *
    // 8` bits (a real allocated buffer, far under u64::MAX). A wide u128
    // multiply here is unnecessary overhead on a hot per-value decode path
    // (called once per doc-values lookup / monotonic-sequence element).
    let bit_pos = (index as u64).wrapping_mul(bits_per_value as u64);
    let byte_pos =
        usize::try_from(bit_pos >> 3).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
    let shift = (bit_pos & 7) as u32;
    let bytes_needed = (shift as usize + bits_per_value as usize).div_ceil(8);

    let bytes = slice
        .get(byte_pos..byte_pos + bytes_needed)
        .ok_or(lucene_store::Error::Eof { offset: byte_pos })?;
    let mut acc: u64 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        acc |= (b as u64) << (8 * i);
    }
    acc >>= shift;
    let mask: u64 = if bits_per_value == 64 {
        u64::MAX
    } else {
        (1u64 << bits_per_value) - 1
    };
    Ok((acc & mask) as i64)
}

/// Port of `DirectWriter.add`/`flush`'s bit-packing (encode side of [`get`]):
/// packs `values` (each assumed to fit in `bits_per_value` unsigned bits) as
/// one little-endian, LSB-first-within-byte bitstream -- the exact inverse
/// of `get`'s formula, so this port doesn't need Java's thirteen
/// width-specialized encoders either.
pub(crate) fn encode(values: &[i64], bits_per_value: u8) -> Vec<u8> {
    let total_bits = values.len() as u128 * bits_per_value as u128;
    let n_bytes = total_bits.div_ceil(8) as usize;
    let mut out = vec![0u8; n_bytes];
    for (i, &v) in values.iter().enumerate() {
        let mut bit_pos = i as u128 * bits_per_value as u128;
        let mut remaining = bits_per_value as u32;
        let mut val = v as u64;
        while remaining > 0 {
            let byte_idx = (bit_pos >> 3) as usize;
            let bit_off = (bit_pos & 7) as u32;
            let can_write = 8 - bit_off;
            let take = remaining.min(can_write);
            let mask = if take == 64 {
                u64::MAX
            } else {
                (1u64 << take) - 1
            };
            out[byte_idx] |= (((val & mask) << bit_off) & 0xFF) as u8;
            val >>= take;
            bit_pos += take as u128;
            remaining -= take;
        }
    }
    out
}

/// `DirectWriter`'s supported bit widths -- `bitsRequired`/`unsignedBitsRequired`
/// always round up to one of these (`DirectWriter.roundBits`).
const SUPPORTED_BITS: [u32; 14] = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64];

/// Port of `DirectWriter.unsignedBitsRequired`: the minimum bit width (among
/// [`SUPPORTED_BITS`]) that can hold `max_value` interpreted as unsigned.
pub(crate) fn unsigned_bits_required(max_value: i64) -> u8 {
    let bits = if max_value == 0 {
        1
    } else {
        64 - (max_value as u64).leading_zeros()
    };
    SUPPORTED_BITS
        .into_iter()
        .find(|&w| w >= bits)
        .unwrap_or(64) as u8
}

/// Port of `DirectWriter.paddingBytesNeeded`: extra zero bytes appended after
/// a block's packed data so a reader could always do one fixed-width
/// (u8/u16/u32/u64) read without touching the next block's bytes. This
/// port's own [`get`] is bounds-checked and never needs this, but the
/// padding is part of the on-disk byte layout (it shifts every subsequent
/// block's offset), so a writer must still emit it for wire compatibility.
pub(crate) fn padding_bytes_needed(bits_per_value: u8) -> usize {
    let padding_bits = if bits_per_value > 32 {
        64 - bits_per_value as u32
    } else if bits_per_value > 16 {
        32 - bits_per_value as u32
    } else if bits_per_value > 8 {
        16 - bits_per_value as u32
    } else {
        0
    };
    (padding_bits as usize).div_ceil(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_byte_aligned_width_round_trips() {
        let payload = [0x34, 0x12, 0xCD, 0xAB];
        assert_eq!(get(&payload, 16, 0).unwrap(), 0x1234);
        assert_eq!(get(&payload, 16, 1).unwrap(), 0xABCD);

        let payload = [0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(get(&payload, 32, 0).unwrap(), 1);
        assert_eq!(get(&payload, 32, 1).unwrap(), 0xFFFFFFFF);

        let payload = (-1i64).to_le_bytes();
        assert_eq!(get(&payload, 64, 0).unwrap(), -1);
    }

    #[test]
    fn sub_byte_widths_pack_multiple_values_per_byte() {
        let payload = [0xBA];
        assert_eq!(get(&payload, 4, 0).unwrap(), 0xA);
        assert_eq!(get(&payload, 4, 1).unwrap(), 0xB);

        let payload = [0b0000_1101u8];
        assert_eq!(get(&payload, 1, 0).unwrap(), 1);
        assert_eq!(get(&payload, 1, 1).unwrap(), 0);
        assert_eq!(get(&payload, 1, 2).unwrap(), 1);
        assert_eq!(get(&payload, 1, 3).unwrap(), 1);
    }

    #[test]
    fn non_byte_aligned_width_12_matches_two_values_per_three_bytes() {
        // index 0 -> 0xABC, index 1 -> 0xDEF, packed as Java's DirectPackedReader12:
        // byte0=0xBC, byte1=0xFA (low nibble 0xA is high nibble of value0,
        // high nibble 0xF is low nibble of value1), byte2=0xDE
        let payload = [0xBC, 0xFA, 0xDE];
        assert_eq!(get(&payload, 12, 0).unwrap(), 0xABC);
        assert_eq!(get(&payload, 12, 1).unwrap(), 0xDEF);
    }

    #[test]
    fn out_of_range_is_error() {
        let payload = [0u8; 1];
        assert!(get(&payload, 16, 5).is_err());
    }

    #[test]
    fn encode_round_trips_through_get_for_every_supported_width() {
        for &bits in &[1u8, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64] {
            let values: Vec<i64> = (0..17)
                .map(|i| {
                    let raw = i as u64 * 7;
                    if bits == 64 {
                        raw as i64
                    } else {
                        (raw % (1u64 << bits)) as i64
                    }
                })
                .collect();
            let packed = encode(&values, bits);
            for (i, &want) in values.iter().enumerate() {
                assert_eq!(
                    get(&packed, bits, i as i64).unwrap(),
                    want,
                    "bits={bits} i={i}"
                );
            }
        }
    }

    #[test]
    fn encode_sub_byte_width_matches_hand_derived_bytes() {
        // Same case as `sub_byte_widths_pack_multiple_values_per_byte`, in reverse.
        let packed = encode(&[0xA, 0xB], 4);
        assert_eq!(packed, vec![0xBA]);
    }

    #[test]
    fn padding_bytes_needed_matches_java_thresholds() {
        assert_eq!(padding_bytes_needed(1), 0);
        assert_eq!(padding_bytes_needed(8), 0);
        assert_eq!(padding_bytes_needed(12), 1); // 16-12=4 bits -> 1 byte
        assert_eq!(padding_bytes_needed(16), 0);
        assert_eq!(padding_bytes_needed(20), 2); // 32-20=12 bits -> 2 bytes
        assert_eq!(padding_bytes_needed(32), 0);
        assert_eq!(padding_bytes_needed(40), 3); // 64-40=24 bits -> 3 bytes
        assert_eq!(padding_bytes_needed(64), 0);
    }
}
