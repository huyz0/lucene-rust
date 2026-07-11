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
use lucene_store::Result;

use crate::direct_reader;

#[derive(Debug, Clone)]
pub struct Meta {
    block_shift: u32,
    mins: Vec<i64>,
    avgs: Vec<f32>,
    bpvs: Vec<u8>,
    offsets: Vec<i64>,
}

/// Reads `Meta` from the metadata stream (e.g. the `.dvm`/`.fnm` file), one
/// `(min: i64, avg: f32-as-i32-bits, offset: i64, bpv: u8)` tuple per block.
pub fn load_meta(input: &mut impl DataInput, num_values: i64, block_shift: u32) -> Result<Meta> {
    let mut num_blocks = num_values >> block_shift;
    if (num_blocks << block_shift) < num_values {
        num_blocks += 1;
    }
    let num_blocks = num_blocks as usize;

    let mut mins = Vec::with_capacity(num_blocks);
    let mut avgs = Vec::with_capacity(num_blocks);
    let mut bpvs = Vec::with_capacity(num_blocks);
    let mut offsets = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        mins.push(input.read_i64()?);
        avgs.push(f32::from_bits(input.read_i32()? as u32));
        offsets.push(input.read_i64()?);
        bpvs.push(input.read_byte()?);
    }

    Ok(Meta {
        block_shift,
        mins,
        avgs,
        bpvs,
        offsets,
    })
}

/// Reads the monotonic sequence's value at `index`. `data` is the slice this
/// meta's offsets are relative to (the whole `.dvd` file, for doc-values
/// addresses).
pub fn get(data: &[u8], meta: &Meta, index: i64) -> Result<i64> {
    let block = (index >> meta.block_shift) as usize;
    let block_index = index & ((1i64 << meta.block_shift) - 1);
    let delta = if meta.bpvs[block] == 0 {
        0
    } else {
        let slice = data
            .get(meta.offsets[block] as usize..)
            .ok_or(lucene_store::Error::Eof { offset: 0 })?;
        direct_reader::get(slice, meta.bpvs[block], block_index)?
    };
    Ok(meta.mins[block] + (meta.avgs[block] * block_index as f32) as i64 + delta)
}

/// Returns the largest `i` in `[from, to)` with `get(data, meta, i) <= key`.
/// Callers must ensure `get(data, meta, from) <= key` (true whenever `key` is
/// a valid doc id and index 0's value is the first chunk's doc base, 0).
pub fn floor_index(data: &[u8], meta: &Meta, from: i64, to: i64, key: i64) -> Result<i64> {
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

#[cfg(test)]
mod tests {
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
        // blockShift=1 -> 2 values per block. 5 values -> 3 blocks.
        let meta_bytes = build_meta_bytes(&[(0, 0.0, 0, 0), (100, 0.0, 0, 0), (200, 0.0, 0, 0)]);
        let mut input = SliceInput::new(&meta_bytes);
        let meta = load_meta(&mut input, 5, 1).unwrap();
        assert_eq!(get(&[], &meta, 0).unwrap(), 0);
        assert_eq!(get(&[], &meta, 1).unwrap(), 0);
        assert_eq!(get(&[], &meta, 2).unwrap(), 100);
        assert_eq!(get(&[], &meta, 4).unwrap(), 200);
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
    fn floor_index_finds_rightmost_le_key() {
        // 3 chunks with doc bases 0, 5, 12 (blockShift=0 -> 1 value/block, 3 blocks)
        let meta_bytes = build_meta_bytes(&[(0, 0.0, 0, 0), (5, 0.0, 0, 0), (12, 0.0, 0, 0)]);
        let mut input = SliceInput::new(&meta_bytes);
        let meta = load_meta(&mut input, 3, 0).unwrap();
        assert_eq!(floor_index(&[], &meta, 0, 3, 0).unwrap(), 0);
        assert_eq!(floor_index(&[], &meta, 0, 3, 4).unwrap(), 0);
        assert_eq!(floor_index(&[], &meta, 0, 3, 5).unwrap(), 1);
        assert_eq!(floor_index(&[], &meta, 0, 3, 11).unwrap(), 1);
        assert_eq!(floor_index(&[], &meta, 0, 3, 12).unwrap(), 2);
        assert_eq!(floor_index(&[], &meta, 0, 3, 999).unwrap(), 2);
    }

    #[test]
    fn floor_index_single_chunk_covers_whole_range() {
        let meta_bytes = build_meta_bytes(&[(0, 0.0, 0, 0)]);
        let mut input = SliceInput::new(&meta_bytes);
        let meta = load_meta(&mut input, 1, 10).unwrap();
        assert_eq!(floor_index(&[], &meta, 0, 1, 0).unwrap(), 0);
        assert_eq!(floor_index(&[], &meta, 0, 1, 500).unwrap(), 0);
    }
}
