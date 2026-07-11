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
    let bit_pos = (index as u128) * (bits_per_value as u128);
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
}
