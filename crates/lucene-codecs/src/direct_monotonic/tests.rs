#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use lucene_store::data_input::SliceInput;

fn build_meta_bytes(blocks: &[(i64, f32, i64, u8)]) -> Vec<u8> {
    let mut out = Vec::new();
    for &(min, avg, offset, bpv) in blocks {
        out.extend_from_slice(&min.to_le_bytes());
        out.extend_from_slice(&(avg.to_bits() as i32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        out.push(bpv);
    }
    out
}

#[test]
fn single_block_all_zero_bpv_is_constant_min_plus_avg_slope() {
    // avg=1.0, min=10, bpv=0 -> value(i) = 10 + i (no stored deltas at all)
    let meta_bytes = build_meta_bytes(&[(10, 1.0, 0, 0)]);
    let mut input = SliceInput::new(&meta_bytes);
    let meta = load_meta(&mut input, 5, 3).unwrap(); // blockShift=3 -> 1 block covers up to 8 values
    assert_eq!(get(&[], &meta, 0).unwrap(), 10);
    assert_eq!(get(&[], &meta, 4).unwrap(), 14);
}

#[test]
fn multi_block_splits_at_block_shift() {
    // blockShift=2 (the minimum `DirectMonotonicWriter` will emit) -> 4
    // values per block. 9 values -> 3 blocks.
    let meta_bytes = build_meta_bytes(&[(0, 0.0, 0, 0), (100, 0.0, 0, 0), (200, 0.0, 0, 0)]);
    let mut input = SliceInput::new(&meta_bytes);
    let meta = load_meta(&mut input, 9, 2).unwrap();
    assert_eq!(get(&[], &meta, 0).unwrap(), 0);
    assert_eq!(get(&[], &meta, 3).unwrap(), 0);
    assert_eq!(get(&[], &meta, 4).unwrap(), 100);
    assert_eq!(get(&[], &meta, 8).unwrap(), 200);
}

/// `blockShift` is a raw `i32` in the `.fdm`/`.tvm` header and a vint in
/// the `.dvm`, and neither Java's reader nor this one bounded it: a shift
/// of 64 panicked on `num_values >> block_shift` in a debug build. A
/// negative `numValues` was worse -- `num_blocks` came out as `-1`, and
/// `-1 as usize` reserved `usize::MAX` blocks.
#[test]
fn out_of_range_block_shift_and_num_values_are_decode_errors() {
    // Shifts below `MIN_BLOCK_SHIFT` are accepted on the read side; only
    // the ceiling bounds the shift arithmetic.
    let meta_bytes = build_meta_bytes(&[(0, 0.0, 0, 0)]);
    for shift in [23u32, 64, 255, u32::MAX] {
        let mut input = SliceInput::new(&meta_bytes);
        assert!(
            load_meta(&mut input, 4, shift).is_err(),
            "blockShift={shift}"
        );
    }
    let mut input = SliceInput::new(&meta_bytes);
    assert!(load_meta(&mut input, -1, 2).is_err());
}

/// A corrupt `index_num_chunks` must not be reserved for verbatim: for the
/// `i32` the `.fdm` carries that is a 51 GB `Vec` -- an abort, which
/// `catch_unwind` cannot intercept. The read loop must run out of input
/// instead.
#[test]
fn absurd_num_values_errors_instead_of_reserving_for_it() {
    let meta_bytes = build_meta_bytes(&[(0, 0.0, 0, 0)]);
    let mut input = SliceInput::new(&meta_bytes);
    let got = load_meta(&mut input, i64::MAX, 2);
    assert!(got.is_err(), "{got:?}");
}

#[test]
fn floor_index_rejects_a_negative_lower_bound() {
    let meta_bytes = build_meta_bytes(&[(0, 0.0, 0, 0)]);
    let mut input = SliceInput::new(&meta_bytes);
    let meta = load_meta(&mut input, 4, 2).unwrap();
    assert!(floor_index(&[], &meta, -1, 3, 0).is_err());
    // `to <= from` is an empty range, not an error: `to - 1` on an
    // `i64::MIN` chunk count off disk would otherwise underflow.
    assert_eq!(floor_index(&[], &meta, 0, i64::MIN, 0).unwrap(), 0);
    assert_eq!(floor_index(&[], &meta, 0, 0, 0).unwrap(), 0);
}

#[test]
#[should_panic(expected = "blockShift must be in [2, 22]")]
fn write_rejects_a_block_shift_java_would_throw_on() {
    write(&[0, 1, 2], 63);
}

#[test]
fn nonzero_bpv_adds_bit_packed_delta_on_top_of_linear_estimate() {
    // avg=2.0, min=0, deltas [0, 1, -1+... ] -- use bpv=2 unsigned deltas 0..3
    // stored raw as bit-packed unsigned ints (Java stores delta - actual is
    // always >=0 by construction of the writer; here just checking decode math).
    let deltas = [0u8, 1, 2, 3]; // packed 2 bits each -> one byte 0b11_10_01_00
    let mut packed = 0u8;
    for (i, &d) in deltas.iter().enumerate() {
        packed |= d << (i * 2);
    }
    let data = [packed];
    let meta_bytes = build_meta_bytes(&[(0, 2.0, 0, 2)]);
    let mut input = SliceInput::new(&meta_bytes);
    let meta = load_meta(&mut input, 4, 2).unwrap(); // 1 block of up to 4 values
    assert_eq!(get(&data, &meta, 0).unwrap(), 0); // 0 + 2*0 + 0
    assert_eq!(get(&data, &meta, 1).unwrap(), 3); // 0 + 2*1 + 1
    assert_eq!(get(&data, &meta, 2).unwrap(), 6); // 0 + 2*2 + 2
    assert_eq!(get(&data, &meta, 3).unwrap(), 9); // 0 + 2*3 + 3
}

#[test]
fn out_of_range_offset_is_error() {
    let meta_bytes = build_meta_bytes(&[(0, 1.0, 100, 4)]);
    let mut input = SliceInput::new(&meta_bytes);
    let meta = load_meta(&mut input, 4, 2).unwrap();
    assert!(get(&[], &meta, 0).is_err());
}

#[test]
fn unsupported_block_bits_per_value_is_rejected_at_load() {
    // Java's `DirectMonotonicReader.getInstance` builds a `DirectReader`
    // per block eagerly, so an unsupported width fails at load, not at the
    // first lookup. 3 is not one of DirectWriter's fourteen widths.
    for bad in [3u8, 5, 63, 65, 200] {
        let meta_bytes = build_meta_bytes(&[(0, 1.0, 0, bad)]);
        let mut input = SliceInput::new(&meta_bytes);
        assert!(
            load_meta(&mut input, 4, 2).is_err(),
            "bpv={bad} must be rejected"
        );
    }
    // 0 (constant block, no packed data at all) stays legal.
    let meta_bytes = build_meta_bytes(&[(0, 1.0, 0, 0)]);
    let mut input = SliceInput::new(&meta_bytes);
    assert!(load_meta(&mut input, 4, 2).is_ok());
}

#[test]
fn get_out_of_range_index_is_an_error_not_a_panic() {
    let meta_bytes = build_meta_bytes(&[(0, 1.0, 0, 0)]);
    let mut input = SliceInput::new(&meta_bytes);
    let meta = load_meta(&mut input, 4, 2).unwrap();
    assert!(get(&[], &meta, 4).is_err());
    assert!(get(&[], &meta, i64::MAX).is_err());
}

#[test]
fn get_wraps_like_java_instead_of_panicking_on_overflow() {
    // Corrupt metadata: a huge min plus a huge linear estimate. Java's
    // `long` arithmetic wraps; this must not be a debug-build panic.
    let meta_bytes = build_meta_bytes(&[(i64::MAX, 1e18, 0, 0)]);
    let mut input = SliceInput::new(&meta_bytes);
    let meta = load_meta(&mut input, 8, 3).unwrap();
    assert!(get(&[], &meta, 5).is_ok());
}

#[test]
fn floor_index_finds_rightmost_le_key() {
    // 3 blocks of 4 constant values each (blockShift=2, the minimum
    // `DirectMonotonicWriter` emits): [0,0,0,0, 5,5,5,5, 12,12,12,12].
    let meta_bytes = build_meta_bytes(&[(0, 0.0, 0, 0), (5, 0.0, 0, 0), (12, 0.0, 0, 0)]);
    let mut input = SliceInput::new(&meta_bytes);
    let meta = load_meta(&mut input, 12, 2).unwrap();
    assert_eq!(floor_index(&[], &meta, 0, 12, 0).unwrap(), 3);
    assert_eq!(floor_index(&[], &meta, 0, 12, 4).unwrap(), 3);
    assert_eq!(floor_index(&[], &meta, 0, 12, 5).unwrap(), 7);
    assert_eq!(floor_index(&[], &meta, 0, 12, 11).unwrap(), 7);
    assert_eq!(floor_index(&[], &meta, 0, 12, 12).unwrap(), 11);
    assert_eq!(floor_index(&[], &meta, 0, 12, 999).unwrap(), 11);
}

#[test]
fn floor_index_single_chunk_covers_whole_range() {
    let meta_bytes = build_meta_bytes(&[(0, 0.0, 0, 0)]);
    let mut input = SliceInput::new(&meta_bytes);
    let meta = load_meta(&mut input, 1, 10).unwrap();
    assert_eq!(floor_index(&[], &meta, 0, 1, 0).unwrap(), 0);
    assert_eq!(floor_index(&[], &meta, 0, 1, 500).unwrap(), 0);
}

fn round_trip(values: &[i64], block_shift: u32) {
    let (meta_bytes, data) = write(values, block_shift);
    let mut input = SliceInput::new(&meta_bytes);
    let meta = load_meta(&mut input, values.len() as i64, block_shift).unwrap();
    for (i, &want) in values.iter().enumerate() {
        assert_eq!(
            get(&data, &meta, i as i64).unwrap(),
            want,
            "index {i} (values={values:?}, block_shift={block_shift})"
        );
    }
}

#[test]
fn write_round_trips_linear_sequence_across_multiple_blocks() {
    // Perfectly linear -> every block's maxDelta is 0 (bpv=0 path).
    let values: Vec<i64> = (0..37).map(|i| i * 3).collect();
    round_trip(&values, 2); // block size 4
}

#[test]
fn write_round_trips_a_full_block_whose_avg_inc_is_not_exact_in_f32() {
    // The block-size-1024 case real formats use (`.fdx` chunk offsets, doc
    // values), with an average increment that has no exact f32
    // representation. `avgInc * i` then diverges between f32 and f64 well
    // before the end of the block, so a writer computing the linear
    // estimate at f64 disagrees with the f32 reader Lucene specifies --
    // silently, and only for indices far enough into a block.
    let values: Vec<i64> = (0..1024).map(|i| (i as i64 * 170) + i as i64 / 3).collect();
    round_trip(&values, 10);
}

#[test]
fn write_round_trips_irregular_sequence_needing_bit_packed_deltas() {
    let values = vec![0i64, 1, 5, 6, 6, 100, 1000, 1000, 1001, 5000, 5000, 5001];
    round_trip(&values, 2);
}

#[test]
fn write_round_trips_single_value_block() {
    round_trip(&[42], 4);
}

#[test]
fn write_round_trips_empty_sequence() {
    round_trip(&[], 4);
}

#[test]
fn write_round_trips_single_full_block_no_remainder() {
    // Exactly one block (block size 4), so there's no partial final chunk.
    round_trip(&[0, 2, 9, 1000], 2);
}

#[test]
fn write_matches_java_flush_algorithm_by_hand() {
    // 4 values [0, 3, 6, 9] in one block of size 4: perfectly linear,
    // avgInc=(9-0)/3=3.0, deltas all 0 -> min=0, maxDelta=0, bpv=0.
    let (meta, data) = write(&[0, 3, 6, 9], 2);
    assert!(data.is_empty());
    let mut input = SliceInput::new(&meta);
    assert_eq!(input.read_i64().unwrap(), 0); // min
    assert_eq!(f32::from_bits(input.read_i32().unwrap() as u32), 3.0); // avgInc
    assert_eq!(input.read_i64().unwrap(), 0); // offset
    assert_eq!(input.read_byte().unwrap(), 0); // bpv
}

/// `Reader` answers what the free `get` answers, value or error: over
/// several blocks (constant and packed), past the last block, for a
/// negative index, and over data cut short (no resolved packed reader).
#[test]
fn reader_agrees_with_get() {
    let values: Vec<i64> = (0..37i64).map(|i| i * 7 + (i * i) % 5).collect();
    let mut with_constant = values.clone();
    with_constant.splice(0..4, [0, 3, 6, 9]); // a bpv-0 block
    for vals in [values, with_constant] {
        let (meta_bytes, data) = write(&vals, 2);
        let meta = load_meta(&mut SliceInput::new(&meta_bytes), vals.len() as i64, 2).unwrap();
        for data in [&data[..], &data[..data.len() / 2], &[][..]] {
            let reader = Reader::new(data, &meta);
            for i in [-1i64, 0, 1, 3, 4, 17, 35, 36, 40, i64::MAX] {
                let (want, got) = (get(data, &meta, i), reader.get(i));
                assert_eq!(want.is_ok(), got.is_ok(), "index {i}");
                if let (Ok(w), Ok(g)) = (want, got) {
                    assert_eq!(w, g, "index {i}");
                }
            }
        }
        let reader = Reader::new(&data, &meta);
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(reader.get(i as i64).unwrap(), v);
        }
    }
}
