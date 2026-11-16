#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]

use super::*;

/// The bit-at-a-time encoder `encode` replaced, kept as its specification.
#[allow(clippy::arithmetic_side_effects)]
fn encode_bitwise(values: &[i64], bits_per_value: u8) -> Vec<u8> {
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

#[test]
fn encode_matches_the_bitwise_specification() {
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    for bpv in SUPPORTED_BITS.iter().map(|&b| b as u8).chain([0, 3, 63]) {
        for n in [0usize, 1, 2, 7, 8, 9, 63, 64, 65, 300] {
            let values: Vec<i64> = (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x as i64
                })
                .collect();
            assert_eq!(
                encode(&values, bpv),
                encode_bitwise(&values, bpv),
                "bpv {bpv} n {n}"
            );
        }
    }
}

// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.

/// The reader agrees with [`get`] on every width, including the values
/// in the last eight bytes of an *unpadded* slice (where it has to defer)
/// and indices past the end (where both must report the same error).
#[test]
fn direct_reader_agrees_with_get_at_every_width_and_at_the_end() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for bits in (0..=64u8).filter(|&b| is_supported_bits(b)) {
        let mask = u64::MAX.checked_shr(64 - u32::from(bits)).unwrap_or(0);
        let values: Vec<i64> = (0..77)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & mask) as i64
            })
            .collect();
        let packed = encode(&values, bits);
        let reader = DirectReader::new(&packed, bits).unwrap();
        for index in (0..90i64).chain([-1, -9, i64::MIN, i64::MAX, i64::MAX / 8]) {
            let want = get(&packed, bits, index);
            let got = reader.get(index);
            assert_eq!(
                format!("{got:?}"),
                format!("{want:?}"),
                "bits {bits} index {index}"
            );
            if let Some(&v) = values.get(index as usize) {
                assert_eq!(got.unwrap(), v, "bits {bits} index {index}");
            }
        }
    }
    // The same through `with_width`, whose visitor sees the width as a
    // constant: every arm of its dispatch, and every width's fast body.
    struct Compare<'v> {
        packed: &'v [u8],
        bits: u8,
    }
    impl WidthVisitor for Compare<'_> {
        type Output = usize;
        fn visit<const B: u32>(self, reader: FixedWidthReader<'_, B>) -> usize {
            assert_eq!(B, u32::from(self.bits));
            for index in (0..90i64).chain([-1, i64::MIN, i64::MAX]) {
                assert_eq!(
                    format!("{:?}", reader.get(index)),
                    format!("{:?}", get(self.packed, self.bits, index)),
                    "bits {B} index {index}"
                );
            }
            B as usize
        }
    }
    for bits in (0..=64u8).filter(|&b| is_supported_bits(b)) {
        let values: Vec<i64> = (0..77).map(|i| i * 7 % 3).collect();
        let packed = encode(&values, bits);
        let reader = DirectReader::new(&packed, bits).unwrap();
        let seen = reader.with_width(Compare {
            packed: &packed,
            bits,
        });
        assert_eq!(seen, usize::from(bits));
    }
    assert!(DirectReader::new(&[0u8; 16], 3).is_err());
    assert!(DirectReader::new(&[0u8; 16], 65).is_err());
}

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
