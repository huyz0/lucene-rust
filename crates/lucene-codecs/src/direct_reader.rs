//! Port of `org.apache.lucene.util.packed.DirectReader.getInstance(...).get(index)`.
//!
//! Generalized into a single bit-position formula instead of Java's thirteen
//! width-specialized `DirectPackedReaderN` classes: those exist to give the
//! JIT a monomorphic call site per width, a concern this port doesn't have
//! yet (no hot per-doc-value loop). Shared by [`crate::doc_values`] (plain
//! bit-packed value arrays) and [`crate::direct_monotonic`] (each block's
//! deltas-from-expected-average array).

use lucene_store::Result;

/// `bits_per_value` must be one of the widths `DirectWriter` supports; anything
/// else is rejected here, exactly where `DirectReader.getInstance`'s `switch`
/// falls through to `IllegalArgumentException`. Java validates once per reader
/// and this validates once per read, but the alternative is worse: several
/// callers take `bits_per_value` straight off disk mid-lookup rather than at
/// parse time (`doc_values`' varying-bits-per-value block header,
/// `direct_monotonic`'s per-block `bpvs`), and an unsupported width reaches the
/// mask below as `1u64 << bits_per_value` -- a debug-build panic and, worse, a
/// silently masked shift returning a plausible wrong value in release.
///
/// `index` addresses the `index`-th `bits_per_value`-wide value packed
/// little-endian (LSB-first within each byte) starting at byte 0 of `slice`.
pub fn get(slice: &[u8], bits_per_value: u8, index: i64) -> Result<i64> {
    if !is_supported_bits(bits_per_value) {
        return Err(lucene_store::Error::Corrupted(format!(
            "unsupported DirectReader bitsPerValue: {bits_per_value}"
        )));
    }
    // `index * bits_per_value` fits comfortably in u64: `index` addresses an
    // element of an in-memory-decoded array, itself bounded by `slice.len() *
    // 8` bits (a real allocated buffer, far under u64::MAX). A wide u128
    // multiply here is unnecessary overhead on a hot per-value decode path
    // (called once per doc-values lookup / monotonic-sequence element).
    let bit_pos = (index as u64).wrapping_mul(bits_per_value as u64);
    let byte_pos =
        usize::try_from(bit_pos >> 3).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
    let shift = (bit_pos & 7) as u32;
    // ARITH: `shift` is masked to `0..=7`, and `is_supported_bits` above caps
    // `bits_per_value` at 64. Those two alone would allow `bytes_needed == 9`,
    // which is *not* enough for the shift loop below -- `8 * 8 == 64` is
    // already a shift overflow on a `u64`. The real bound is tighter and comes
    // from `SUPPORTED_BITS`: `shift` is non-zero only when `bit_pos` is not a
    // multiple of 8, which for a supported width means the width itself is not
    // a multiple of 8, and those stop at 28. So either `shift == 0` and
    // `bytes_needed <= 8`, or `bits_per_value <= 28` and `bytes_needed <=
    // ceil(35 / 8) == 5`. Either way `bytes_needed <= 8` and `8 * i <= 56`.
    #[allow(clippy::arithmetic_side_effects)]
    let bytes_needed = (shift as usize + bits_per_value as usize).div_ceil(8);
    debug_assert!(
        bytes_needed <= 8,
        "bytes_needed={bytes_needed} would overflow the shift below"
    );

    // `index` reaches here from a doc-values ordinal or a monotonic-block
    // index, both of which can come off disk. On a 64-bit target `bit_pos` is
    // a `u64` and `bit_pos >> 3` therefore never exceeds `2^61 - 1`, so
    // `byte_pos + 8` could not overflow there anyway -- but on a 32-bit target
    // the `try_from` above admits anything up to `u32::MAX`, where it would.
    // One comparison against the slice length, hoisted out of both reads
    // below, makes every `byte_pos + n` here provably in range on every
    // target, since a slice length is at most `isize::MAX`. It also turns a
    // wildly out-of-range ordinal into an `Eof` before the two `slice.get`
    // calls rather than after.
    if byte_pos > slice.len() {
        return Err(lucene_store::Error::Eof { offset: byte_pos });
    }

    // One wide load whenever eight bytes are in range, which is what
    // `DirectWriter` pads its output for -- `padding_bytes_needed` below is
    // this port's copy of the same rule, and `Lucene90DocValuesConsumer` writes
    // that padding for exactly this reason. Every width `DirectWriter` supports
    // needs at most eight bytes: the non-byte-aligned ones stop at 28 bits
    // (five bytes at worst), and the byte-aligned ones never carry a shift.
    //
    // The byte-at-a-time loop below is the tail fallback, and it used to be the
    // only path. It cost one load, one shift and one OR per byte, so the read
    // got linearly slower with the width: 1.96 ns at one bit rising to 5.25 ns
    // at 64, against Lucene's flat 2.3 ns for all fourteen widths -- Lucene's
    // `DirectPackedReaderNN` classes each do a single `readInt`/`readLong`.
    // ARITH: `byte_pos <= slice.len() <= isize::MAX` was just checked, and
    // `bytes_needed <= 9`, so neither sum can overflow `usize`.
    #[allow(clippy::arithmetic_side_effects)]
    let acc = if let Some(window) = slice.get(byte_pos..byte_pos + 8) {
        u64::from_le_bytes(window.try_into().expect("exactly 8 bytes"))
    } else {
        let Some(bytes) = slice.get(byte_pos..byte_pos + bytes_needed) else {
            return Err(lucene_store::Error::Eof { offset: byte_pos });
        };
        let mut acc: u64 = 0;
        // ARITH: `bytes.len() == bytes_needed <= 8` (see the bound above),
        // so `i <= 7` and `8 * i <= 56`, a legal `u64` shift.
        #[allow(clippy::arithmetic_side_effects)]
        for (i, &b) in bytes.iter().enumerate() {
            acc |= (b as u64) << (8 * i);
        }
        acc
    };
    let acc = acc >> shift;
    // ARITH: the `else` arm has `bits_per_value < 64` (the `== 64` case is the
    // arm above and `is_supported_bits` rejects everything over 64), so the
    // shift is in range and its result is at least 1.
    #[allow(clippy::arithmetic_side_effects)]
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
// ARITH: every product is computed in `u128` from a slice length and a `u8`,
// so it cannot overflow. Inside the loop `bit_off` is masked to `0..=7`,
// `take = min(remaining, 8 - bit_off)` is in `0..=8`, `remaining` only ever
// decreases by `take` (and the loop stops at 0), and `bit_pos` advances by
// `take` a bounded number of times. This is the encode side: no operand here
// comes off disk.
#[allow(clippy::arithmetic_side_effects)]
pub fn encode(values: &[i64], bits_per_value: u8) -> Vec<u8> {
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

/// [`SUPPORTED_BITS`] below 64, as a bit set, so the check in [`get`] is a
/// shift and a test rather than a scan of the table. 64 is handled separately
/// rather than widening this to `u128`: a 128-bit variable shift costs several
/// instructions, and this sits on a per-value decode path.
// ARITH: `i` indexes a 14-element array and the loop stops at its length.
// Evaluated at compile time, so an overflow here would be a build error, not
// a runtime panic.
#[allow(clippy::arithmetic_side_effects)]
const SUPPORTED_BITS_MASK: u64 = {
    let mut mask = 0u64;
    let mut i = 0;
    while i < SUPPORTED_BITS.len() {
        if SUPPORTED_BITS[i] < 64 {
            mask |= 1u64 << SUPPORTED_BITS[i];
        }
        i += 1;
    }
    mask
};

/// Port of `DirectWriter.checkBitsPerValue`'s membership test: is
/// `bits_per_value` one of the fourteen widths `DirectWriter` can emit?
#[inline]
pub(crate) fn is_supported_bits(bits_per_value: u8) -> bool {
    bits_per_value == 64
        || (bits_per_value < 64 && (SUPPORTED_BITS_MASK >> bits_per_value) & 1 != 0)
}

/// Port of `DirectWriter.unsignedBitsRequired`: the minimum bit width (among
/// [`SUPPORTED_BITS`]) that can hold `max_value` interpreted as unsigned.
// ARITH: `u64::leading_zeros()` returns `0..=64`, so `64 - lz` is in `0..=64`.
#[allow(clippy::arithmetic_side_effects)]
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
// ARITH: each subtraction sits under the branch that establishes it --
// `64 - bits` only for `bits > 32`, and so on -- but that alone only bounds
// the *lower* end. The upper end comes from the call sites: both of them
// (`direct_monotonic::write`, `doc_values`' numeric writer) pass a
// width that `unsigned_bits_required` returned, which is one of the fourteen
// `SUPPORTED_BITS` and so never exceeds 64. A width read off disk must not be
// passed here; `get` is the entry point that validates one.
#[allow(clippy::arithmetic_side_effects)]
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
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

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
    fn unsupported_bit_width_is_rejected_not_a_shift_overflow() {
        // Java's `DirectReader.getInstance` throws for anything outside the
        // fourteen supported widths; `bits_per_value` reaches here straight off
        // disk in the varying-bpv and monotonic-block paths.
        let payload = [0xFFu8; 32];
        for bits in [0u8, 3, 5, 7, 9, 33, 63, 65, 100, 255] {
            assert!(
                get(&payload, bits, 0).is_err(),
                "bits_per_value={bits} must be rejected"
            );
        }
        for &bits in &SUPPORTED_BITS {
            assert!(is_supported_bits(bits as u8), "bits_per_value={bits}");
            assert!(
                get(&payload, bits as u8, 0).is_ok(),
                "bits_per_value={bits}"
            );
        }
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
