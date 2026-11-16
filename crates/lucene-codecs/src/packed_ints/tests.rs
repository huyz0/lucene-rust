#![cfg(test)]
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
/// `index as u128` sign-extends, so a negative index becomes ~2^128 and
/// would overflow the `index * bits_per_value` multiply -- a debug-build
/// panic before any bounds check ran.
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

/// The writer's and iterator's bookkeeping as Java's: `ord()` from -1,
/// `getBitsPerValue`/`getFormat`/`size`; a value past a known count and a
/// read past the end are `EOFException`s; a width the format cannot hold,
/// or a count the input cannot fill, is refused before anything is read.
#[test]
fn writer_and_iterator_bookkeeping_and_refusals() {
    use lucene_store::data_input::SliceInput;
    for format in [Format::Packed, Format::PackedSingleBlock] {
        let mut buf = Vec::new();
        {
            let mut w = PackedWriter::new(&mut buf, format, Some(5), 7, 0);
            assert_eq!((w.bits_per_value(), w.format(), w.ord()), (7, format, -1));
            for v in [1, 2, 3] {
                w.add(v).unwrap();
            }
            assert_eq!(w.ord(), 2);
            w.finish().unwrap();
            assert_eq!(w.ord(), 4, "finish pads the known count");
        }
        {
            let mut full = Vec::new();
            let mut w = PackedWriter::new(&mut full, format, Some(1), 7, 0);
            w.add(1).unwrap();
            assert!(matches!(w.add(2), Err(lucene_store::Error::Eof { .. })));
        }
        let mut input = SliceInput::new(&buf);
        let mut it =
            PackedReaderIterator::new(&mut input, format, VERSION_CURRENT, 5, 7, 0).unwrap();
        assert_eq!((it.bits_per_value(), it.size(), it.ord()), (7, 5, -1));
        let got: Vec<i64> = (0..5).map(|_| it.next_value().unwrap()).collect();
        assert_eq!(got, [1, 2, 3, 0, 0]);
        assert_eq!(it.ord(), 4);
        assert!(matches!(
            it.next_value(),
            Err(lucene_store::Error::Eof { .. })
        ));
        // More values than the bytes hold.
        let mut short = SliceInput::new(&buf);
        assert!(matches!(
            PackedReaderIterator::new(&mut short, format, VERSION_CURRENT, 5000, 7, 0),
            Err(lucene_store::Error::Eof { .. })
        ));
    }
    // `PACKED_SINGLE_BLOCK` holds no 64-bit (or 11-bit) values.
    for bits in [11u32, 64] {
        let mut input = SliceInput::new(&[0u8; 64]);
        assert!(PackedReaderIterator::new(
            &mut input,
            Format::PackedSingleBlock,
            VERSION_CURRENT,
            1,
            bits,
            0
        )
        .is_err());
        assert!(DirectPacked64SingleBlockReader::new(bits, 1, &[0u8; 64]).is_err());
    }
}
