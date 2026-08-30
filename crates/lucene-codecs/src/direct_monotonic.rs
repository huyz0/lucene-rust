//! Port of `org.apache.lucene.util.packed.DirectMonotonicReader` (read-only).
//!
//! Stores a monotonically non-decreasing `i64` sequence (e.g. the end
//! offsets of variable-length binary doc-values entries) as fixed-size
//! blocks. Each block records a `min` and an average per-index slope
//! (`avg`); within a block, only the *delta* from that linear estimate is
//! bit-packed (via [`crate::direct_reader`]), which is small and cheap to
//! pack when the sequence is close to linear — the common case for
//! monotonically increasing offsets.
//!
//! [`floor_index`] finds the rightmost index whose value is `<=` a key
//! (used by stored fields to find which chunk contains a given doc id) —
//! a direct binary search via repeated [`get`] calls, not a port of Java's
//! generic `DirectMonotonicReader.binarySearch` (which pre-checks cheap
//! per-block bounds before touching the bit-packed reader to dodge page
//! faults; not a concern for an in-memory decode).

use lucene_store::data_input::DataInput;
use lucene_store::data_output::DataOutput;
use lucene_store::Result;

use crate::direct_reader;

/// One block's header. Java keeps four parallel arrays (`mins`, `avgs`,
/// `bpvs`, `offsets`) and indexes all four on every `get`; every `get` here
/// reads all four fields of exactly one block, so one array of 24-byte structs
/// is one bounds check and one cache line instead of four of each. (See the
/// `rust-performance` skill on not transliterating Java's parallel arrays.)
#[derive(Debug, Clone, Copy)]
struct Block {
    min: i64,
    offset: i64,
    avg: f32,
    bpv: u8,
}

/// `DirectMonotonicWriter.MIN_BLOCK_SHIFT`/`MAX_BLOCK_SHIFT`. Java enforces
/// both in the writer's constructor only; [`write()`] does the same, and
/// [`load_meta`] enforces the ceiling on the read side because that is what
/// bounds every shift below.
const MIN_BLOCK_SHIFT: u32 = 2;
const MAX_BLOCK_SHIFT: u32 = 22;

/// On-disk size of one block's metadata tuple: `min` (i64), `avg` (i32 float
/// bits), `offset` (i64), `bpv` (u8). Used only to bound [`load_meta`]'s
/// reservation by what the metadata stream could actually contain.
const BLOCK_META_BYTES: usize = 8 + 4 + 8 + 1;

#[derive(Debug, Clone)]
pub struct Meta {
    block_shift: u32,
    blocks: Vec<Block>,
}

/// Reads `Meta` from the metadata stream (e.g. the `.dvm`/`.fnm` file), one
/// `(min: i64, avg: f32-as-i32-bits, offset: i64, bpv: u8)` tuple per block.
pub fn load_meta(input: &mut impl DataInput, num_values: i64, block_shift: u32) -> Result<Meta> {
    // `block_shift` and `num_values` both arrive straight off disk -- the
    // `.fdm`/`.tvm` header reads them as raw `i32`s, and `.dvm` as a vint --
    // and neither Java nor this port validated them on the read side. Three
    // separate failures came out of that:
    //
    //   * `num_values >> block_shift` with `block_shift >= 64` is a panic in a
    //     debug build (Java's `>>>` merely masks the shift to 6 bits);
    //   * a negative `num_values` gives `num_blocks == -1`, and `-1 as usize`
    //     is `usize::MAX`;
    //   * `Vec::with_capacity(num_blocks)` for a `num_blocks` a corrupt header
    //     chose is an *allocation failure*, which aborts -- `catch_unwind`
    //     cannot intercept it, so it takes the JVM down through the FFI.
    //     `index_num_chunks` is an `i32` off the `.fdm`, so two flipped bytes
    //     buy a 51 GB reservation.
    //
    // `DirectMonotonicWriter`'s constructor rejects a `blockShift` outside
    // `[MIN_BLOCK_SHIFT, MAX_BLOCK_SHIFT]`, so no file Lucene wrote can carry
    // one. Checking the writer's ceiling on the read side is the same move
    // this function already makes for `bitsPerValue` below, and it is what
    // makes `1i64 << block_shift` in `get`'s hot path provably safe. Only the
    // ceiling is enforced: the floor buys nothing here (a small shift only
    // means more blocks, which the reservation cap below already handles) and
    // this module's own synthetic metadata uses shifts under it.
    if block_shift > MAX_BLOCK_SHIFT {
        return Err(lucene_store::Error::Corrupted(format!(
            "DirectMonotonic blockShift must be at most {MAX_BLOCK_SHIFT}, got {block_shift}"
        )));
    }
    if num_values < 0 {
        return Err(lucene_store::Error::Corrupted(format!(
            "DirectMonotonic numValues must be non-negative, got {num_values}"
        )));
    }
    // ARITH: `block_shift <= 22` and `num_values >= 0`, so the right shift is
    // in range and `num_blocks` is in `0..=num_values`; shifting it back left
    // recovers a value at or below `num_values`, so it cannot overflow. The
    // `+= 1` cannot either: at `block_shift == 0` the shift is exact and the
    // `<` is false, so the increment never runs, and at `block_shift >= 1`
    // `num_blocks <= num_values / 2 <= i64::MAX / 2`.
    #[allow(clippy::arithmetic_side_effects)]
    let num_blocks = {
        let mut num_blocks = num_values >> block_shift;
        if (num_blocks << block_shift) < num_values {
            num_blocks += 1;
        }
        num_blocks as usize
    };

    // Each block costs exactly 21 bytes on disk (i64 min + i32 avg + i64
    // offset + u8 bpv), so the metadata stream itself bounds how many there
    // can really be. Clamping to that costs a well-formed `.dvm`/`.fdm`
    // nothing -- its bytes are all there -- while a corrupt `numValues` can no
    // longer reserve more than the stream could hold.
    let ceiling = input.remaining() / BLOCK_META_BYTES;
    let mut blocks = Vec::with_capacity(num_blocks.min(ceiling));
    for _ in 0..num_blocks {
        let min = input.read_i64()?;
        let avg = f32::from_bits(input.read_i32()? as u32);
        let offset = input.read_i64()?;
        let bpv = input.read_byte()?;
        // Java validates here too, though implicitly and one layer down:
        // `DirectMonotonicReader.getInstance` eagerly builds a `DirectReader`
        // per non-zero-width block, and `DirectReader.getInstance`'s `switch`
        // throws `IllegalArgumentException` on an unsupported width. Checking
        // at load time rather than at first `get` keeps a corrupt `.dvm` from
        // surfacing as a lookup failure hundreds of calls later.
        if bpv != 0 && !direct_reader::is_supported_bits(bpv) {
            return Err(lucene_store::Error::Corrupted(format!(
                "DirectMonotonic block has unsupported bitsPerValue: {bpv}"
            )));
        }
        blocks.push(Block {
            min,
            offset,
            avg,
            bpv,
        });
    }

    Ok(Meta {
        block_shift,
        blocks,
    })
}

/// Reads the monotonic sequence's value at `index`. `data` is the slice this
/// meta's offsets are relative to (the whole `.dvd` file, for doc-values
/// addresses).
pub fn get(data: &[u8], meta: &Meta, index: i64) -> Result<i64> {
    // ARITH: `Meta`'s fields are private and `load_meta` is its only
    // constructor, which rejects a `block_shift` above 22. So the shift is in
    // range and the mask is at least 0.
    #[allow(clippy::arithmetic_side_effects)]
    let block_index = index & ((1i64 << meta.block_shift) - 1);
    // `let ... else`, not `ok_or(Error::Eof { .. })`: `lucene_store::Error` is
    // 32 bytes wide (it carries a `String`), and `ok_or` builds its argument
    // eagerly -- a real store on the happy path of a per-value lookup, worth
    // ~20% of this function in `direct_monotonic/get_block`.
    let Some(&block) = meta.blocks.get((index >> meta.block_shift) as usize) else {
        return Err(lucene_store::Error::Eof { offset: 0 });
    };
    let delta = if block.bpv == 0 {
        0
    } else {
        let Some(slice) = data.get(block.offset as usize..) else {
            return Err(lucene_store::Error::Eof { offset: 0 });
        };
        direct_reader::get(slice, block.bpv, block_index)?
    };
    // Java computes `mins[block] + (long) (avgs[block] * blockIndex) + delta` in
    // `long` arithmetic, which wraps. Corrupt metadata can make that overflow,
    // and a debug-build panic in a decoder that otherwise reports corruption
    // through `Result` is the wrong failure mode.
    Ok(block
        .min
        .wrapping_add((block.avg * block_index as f32) as i64)
        .wrapping_add(delta))
}

/// Returns the largest `i` in `[from, to)` with `get(data, meta, i) <= key`.
/// Callers must ensure `get(data, meta, from) <= key` (true whenever `key` is
/// a valid doc id and index 0's value is the first chunk's doc base, 0).
pub fn floor_index(data: &[u8], meta: &Meta, from: i64, to: i64, key: i64) -> Result<i64> {
    // `to` is a chunk count off disk (`.fdm`/`.tvm`), so it is neither
    // guaranteed positive nor guaranteed above `from`. Establish both here
    // rather than inside the loop: the search below runs `log2(to - from)`
    // times per stored-field and per term-vector lookup, and these are two
    // comparisons hoisted out of it.
    if from < 0 {
        return Err(lucene_store::Error::Corrupted(format!(
            "DirectMonotonic floor_index needs a non-negative lower bound, got {from}"
        )));
    }
    if to <= from {
        return Ok(from);
    }
    // ARITH: `to > from >= 0`, so `to - 1` cannot underflow and `hi` stays in
    // `0..=i64::MAX - 1`. Inside the loop `lo <= hi` and both are
    // non-negative, so `hi - lo` cannot overflow, `mid` is in `lo..=hi`,
    // `mid + 1` is at most `hi + 1 <= to <= i64::MAX`, and `mid - 1` is at
    // least `-1`.
    #[allow(clippy::arithmetic_side_effects)]
    {
        let (mut lo, mut hi) = (from, to - 1);
        let mut result = from;
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            if get(data, meta, mid)? <= key {
                result = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        Ok(result)
    }
}

/// Port of `DirectMonotonicWriter`: encodes a monotonically non-decreasing
/// `i64` sequence into `(meta_bytes, data_bytes)` -- the write-side
/// counterpart of [`load_meta`]/[`get`]. `values` must already be sorted
/// non-decreasing (mirrors Java's `add`, which throws on out-of-order
/// input); this port just asserts it via `debug_assert`, since every caller
/// so far builds `values` from an already-sorted source (chunk boundaries,
/// offsets).
///
/// Unlike the real `DirectMonotonicWriter`, this returns two full buffers
/// rather than streaming to an `IndexOutput` -- there's no incremental
/// `IndexOutput` in this port yet (see `lucene_store::data_output`'s module
/// doc), and every value set built so far comfortably fits in memory.
pub fn write(values: &[i64], block_shift: u32) -> (Vec<u8>, Vec<u8>) {
    // Java's `DirectMonotonicWriter` constructor throws
    // `IllegalArgumentException` here, and every caller in this port passes a
    // format constant, so a panic is the faithful port of an unreachable
    // programming error rather than a decode failure. It is also what makes
    // `1usize << block_shift` below safe.
    assert!(
        (MIN_BLOCK_SHIFT..=MAX_BLOCK_SHIFT).contains(&block_shift),
        "blockShift must be in [{MIN_BLOCK_SHIFT}, {MAX_BLOCK_SHIFT}], got {block_shift}"
    );
    // ARITH: `block_shift <= 22` by the assertion above.
    #[allow(clippy::arithmetic_side_effects)]
    let block_size = 1usize << block_shift;
    let mut meta = Vec::new();
    let mut data = Vec::new();

    for chunk in values.chunks(block_size) {
        debug_assert!(chunk.windows(2).all(|w| w[0] <= w[1]));

        // ARITH: the `else` arm has `chunk.len() >= 2`, so `chunk.len() - 1`
        // is a valid index and a non-zero divisor. The value subtraction is
        // `wrapping_sub` for the same reason the two below are: Java does all
        // of this in `long`, which wraps, and `get` undoes it with
        // `wrapping_add`, so wrapping is what round-trips.
        #[allow(clippy::arithmetic_side_effects)]
        let avg_inc = if chunk.len() <= 1 {
            0.0f64
        } else {
            chunk[chunk.len() - 1].wrapping_sub(chunk[0]) as f64 / (chunk.len() - 1) as f64
        } as f32;

        let mut deltas: Vec<i64> = chunk
            .iter()
            .enumerate()
            // `avg_inc * i` in **f32**, exactly as Java's
            // `(long) (avgInc * (long) i)` does -- `avgInc` is a float there,
            // so the product is computed at float precision before truncating.
            // Widening this to f64 makes the writer disagree with both Lucene's
            // reader and this port's own (which correctly uses f32) once a
            // block holds enough values for the precision gap to move the
            // truncated result by one.
            .map(|(i, &v)| v.wrapping_sub((avg_inc * i as f32) as i64))
            .collect();
        let min = *deltas.iter().min().unwrap();
        for d in &mut deltas {
            *d = d.wrapping_sub(min);
        }
        // Matches Java's `maxDelta |= buffer[i]` -- an OR-based upper bound
        // rather than a real max, but equivalent for bit-width purposes and
        // robust to the (unreachable here, since deltas are all >= 0 after
        // the subtraction above) negative-overflow case Java's comment
        // mentions.
        let max_delta = deltas.iter().fold(0i64, |acc, &d| acc | d);

        meta.write_i64(min);
        meta.write_i32(avg_inc.to_bits() as i32);
        meta.write_i64(data.len() as i64);
        if max_delta == 0 {
            meta.push(0);
        } else {
            let bits_per_value = direct_reader::unsigned_bits_required(max_delta);
            data.extend_from_slice(&direct_reader::encode(&deltas, bits_per_value));
            data.extend(std::iter::repeat_n(
                0u8,
                direct_reader::padding_bytes_needed(bits_per_value),
            ));
            meta.push(bits_per_value);
        }
    }

    (meta, data)
}

#[cfg(test)]
mod tests {
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

    /// A corrupt `index_num_chunks` used to be reserved for verbatim, which
    /// for the `i32` the `.fdm` carries is a 51 GB `Vec` -- an abort, which
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
}
