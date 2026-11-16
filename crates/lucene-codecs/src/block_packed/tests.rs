#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

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

/// `total_value_count` is a chunk header field off a `.tvd`. Reserving it
/// verbatim turned a corrupt header into a multi-gigabyte allocation --
/// an abort, which `catch_unwind` cannot intercept -- before the decoder
/// had read a single block. The loop itself is bounded by the input, so
/// a truncated stream must come back as a decode error.
#[test]
fn absurd_total_value_count_errors_instead_of_reserving_it() {
    let bytes = [1u8]; // one bitsPerValue=0, min=0 block: 64 values
    let mut input = SliceInput::new(&bytes);
    let got = decode_all(&mut input, i64::MAX);
    assert!(got.is_err(), "{got:?}");
}

#[test]
fn empty_stream_reads_nothing() {
    let mut input = SliceInput::new(&[]);
    assert_eq!(decode_all(&mut input, 0).unwrap(), Vec::<i64>::new());
}

/// The one-pass unpacker reads every width exactly as the per-value
/// `packed_ints::get` does, at every count a block can hold.
#[test]
fn unpack_block_matches_packed_ints_get_at_every_width() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let bytes: Vec<u8> = (0..BLOCK_BYTES_MAX)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect();
    for bits in 1..=64u32 {
        for count in [1usize, 5, 63, 64] {
            let len = packed_ints::byte_count(count as u64, bits);
            let mut out = Vec::new();
            unpack_block(&bytes[..len], bits, count, -3, &mut out);
            let want: Vec<i64> = (0..count)
                .map(|i| {
                    packed_ints::get(&bytes[..len], bits, i as i64)
                        .unwrap()
                        .wrapping_add(-3)
                })
                .collect();
            assert_eq!(out, want, "bits={bits} count={count}");
        }
    }
    // a slice short of its promise loads zero bits, it does not panic
    let mut out = Vec::new();
    unpack_block(&[0xFF], 12, 2, 0, &mut out);
    assert_eq!(out, vec![0xFF0, 0]);
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
fn encode_all_round_trips_negative_and_mixed_sign_values() {
    // A block whose max is <= 0 must not pick bitsPerValue = 0 (it would
    // decode as all zeros), and a mixed-sign block must not truncate the
    // negatives.
    for values in [
        vec![-5i64, -3, -100, -1],
        vec![-5i64, 10, -3, 0, 7],
        vec![i64::MIN, 0, i64::MAX],
        vec![i64::MIN; 3],
        (0..130).map(|i| i - 65).collect::<Vec<i64>>(),
    ] {
        let encoded = encode_all(&values);
        let mut input = SliceInput::new(&encoded);
        assert_eq!(
            decode_all(&mut input, values.len() as i64).unwrap(),
            values,
            "values={values:?}"
        );
    }
}

#[test]
fn encode_all_matches_javas_flush_min_value_choice() {
    // Java: delta = 300 - 300 == 0 -> bitsPerValue 0, min stays 300, so the
    // token has minValueEquals0 clear and the minValue varint follows.
    let encoded = encode_all(&[300, 300, 300]);
    assert_eq!(encoded[0], 0, "bitsPerValue=0, minValueEquals0=false");
    let mut input = SliceInput::new(&encoded[1..]);
    let min = lucene_util::zigzag::decode(
        1u64.wrapping_add(read_min_value_vlong(&mut input).unwrap() as u64),
    );
    assert_eq!(min, 300);

    // delta = 260 - 256 = 4 -> bitsPerValue 3, maxValue(3) = 7, so Java
    // lowers min from 256 to max(0, 260 - 7) = 253 rather than keeping the
    // real block minimum.
    let encoded = encode_all(&[256, 258, 260]);
    assert_eq!(encoded[0], 3 << 1, "bitsPerValue=3, minValueEquals0=false");
    let mut input = SliceInput::new(&encoded[1..]);
    let min = lucene_util::zigzag::decode(
        1u64.wrapping_add(read_min_value_vlong(&mut input).unwrap() as u64),
    );
    assert_eq!(min, 253);
}

#[test]
fn min_value_vlong_round_trips_the_nine_byte_negative_capable_form() {
    // The generic `DataInput::read_vlong` would treat the ninth byte's top
    // bit as a continuation marker and read into the next block.
    for v in [
        0u64,
        1,
        0x7f,
        0x80,
        (1 << 56) - 1,
        1 << 56,
        u64::MAX,
        u64::MAX - 1,
        0xFFFF_FFFF_FFFF_FFF0,
    ] {
        let mut bytes = Vec::new();
        write_min_value_vlong(&mut bytes, v);
        assert!(bytes.len() <= 9, "v={v:#x} took {} bytes", bytes.len());
        let mut input = SliceInput::new(&bytes);
        assert_eq!(
            read_min_value_vlong(&mut input).unwrap() as u64,
            v,
            "v={v:#x}"
        );
    }
}

#[test]
fn encode_all_all_zero_block_uses_zero_bits() {
    let values = vec![0i64; 64];
    let encoded = encode_all(&values);
    assert_eq!(encoded, vec![1u8]); // token: bits=0, min_equals_0=1
}
#[test]
fn streaming_iterator_next_skip_and_bulk_agree_with_decode_all() {
    let values: Vec<i64> = (0..1000i64)
        .map(|i| (i * 7919) % 1013 - 300 + if i % 97 == 0 { i64::MIN / 3 } else { 0 })
        .collect();
    for block_size in [64usize, 128, 256] {
        let bytes = encode_all_with_block_size(&values, block_size);
        let mut input = SliceInput::new(&bytes);
        let mut it =
            BlockPackedReaderIterator::new(&mut input, 2, block_size, values.len() as u64).unwrap();
        for &v in &values {
            assert_eq!(it.next_value().unwrap(), v);
        }
        assert!(it.next_value().is_err());
        assert!(it.next_values(3).is_err());
        for (skip, take) in [
            (3u64, 5usize),
            (10, 300),
            (64, 1),
            (700, 1000),
            (999, 1),
            (1000, 0),
        ] {
            let mut input = SliceInput::new(&bytes);
            let mut it =
                BlockPackedReaderIterator::new(&mut input, 2, block_size, values.len() as u64)
                    .unwrap();
            it.skip(3).unwrap();
            it.skip(skip - 3).unwrap();
            let start = skip as usize;
            assert_eq!(it.ord(), skip);
            let mut got = Vec::new();
            while got.len() < take && it.ord() < values.len() as u64 {
                let chunk = it.next_values(take - got.len()).unwrap();
                assert!(!chunk.is_empty() && chunk.len() <= block_size);
                got.extend_from_slice(chunk);
            }
            let end = (start + take).min(values.len());
            assert_eq!(
                got,
                &values[start..end],
                "block_size={block_size} skip={skip}"
            );
        }
        let mut input = SliceInput::new(&bytes);
        let mut it =
            BlockPackedReaderIterator::new(&mut input, 2, block_size, values.len() as u64).unwrap();
        assert!(it.skip(1001).is_err());
        it.skip(1000).unwrap();
        it.reset(0);
        assert_eq!(it.ord(), 0);
    }
}

#[test]
fn streaming_iterator_rejects_bad_parameters_and_widths() {
    let bytes = [0xffu8, 0, 0];
    let mut input = SliceInput::new(&bytes);
    assert!(BlockPackedReaderIterator::new(&mut input, 1, 64, 1).is_err());
    let mut input = SliceInput::new(&bytes);
    assert!(BlockPackedReaderIterator::new(&mut input, 2, 100, 1).is_err());
    let mut input = SliceInput::new(&bytes);
    let mut it = BlockPackedReaderIterator::new(&mut input, 2, 64, 200).unwrap();
    assert!(it.next_value().is_err()); // token 0xff: width 127
    let mut input = SliceInput::new(&bytes);
    let mut it = BlockPackedReaderIterator::new(&mut input, 2, 64, 200).unwrap();
    assert!(it.skip(100).is_err());
}
