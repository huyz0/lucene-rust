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

impl Meta {
    /// The meta of no values.
    pub(crate) fn empty() -> Self {
        Self {
            block_shift: 0,
            blocks: Vec::new(),
        }
    }
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
    } else if let Some(v) = fast_packed_get(data, block.offset, block.bpv, block_index) {
        v
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

/// `DirectMonotonicReader.getInstance(meta, data)`: [`get`] over one
/// `data` slice with each block's packed reader resolved once, as Java
/// builds a `DirectReader` per block up front -- a lookup is then the
/// block's header, one comparison and one 8-byte load. For a reader of many
/// values of one sequence (a doc-values iterator's addresses). Allocates one
/// entry per block.
#[derive(Debug, Clone)]
pub struct Reader<'a> {
    data: &'a [u8],
    meta: &'a Meta,
    blocks: Vec<ResolvedBlock<'a>>,
}

/// One block of a [`Reader`].
#[derive(Debug, Clone, Copy)]
struct ResolvedBlock<'a> {
    min: i64,
    avg: f32,
    /// `bpv == 0`: every delta is 0.
    zero: bool,
    /// The block's packed deltas; `None` when they cannot be read with one
    /// load (a block whose bytes are not inside `data`), and [`get`] answers.
    packed: Option<lucene_util::packed_longs::PackedLongs<'a>>,
}

impl<'a> Reader<'a> {
    /// Resolves `meta`'s blocks against `data` (the slice [`get`] would be
    /// handed).
    pub fn new(data: &'a [u8], meta: &'a Meta) -> Self {
        let blocks = meta
            .blocks
            .iter()
            .map(|b| ResolvedBlock {
                min: b.min,
                avg: b.avg,
                zero: b.bpv == 0,
                packed: usize::try_from(b.offset)
                    .ok()
                    .and_then(|o| data.get(o..))
                    .and_then(|slice| {
                        lucene_util::packed_longs::PackedLongs::new(slice, u32::from(b.bpv))
                    }),
            })
            .collect();
        Reader { data, meta, blocks }
    }

    /// `get(index)`: what [`get`] answers for `index`, errors included.
    #[inline]
    pub fn get(&self, index: i64) -> Result<i64> {
        let shift = self.meta.block_shift;
        // A negative index lands on no block and [`get`] reports it.
        let Some(block) = usize::try_from(index >> shift)
            .ok()
            .and_then(|b| self.blocks.get(b))
        else {
            return get(self.data, self.meta, index);
        };
        // ARITH: `block_shift <= 22` (see `get`), so the mask is in range.
        #[allow(clippy::arithmetic_side_effects)]
        let block_index = index & ((1i64 << shift) - 1);
        let delta = if block.zero {
            0
        } else {
            match block.packed.and_then(|p| p.get(block_index as u64)) {
                Some(v) => v as i64,
                None => return get(self.data, self.meta, index),
            }
        };
        Ok(block
            .min
            .wrapping_add((block.avg * block_index as f32) as i64)
            .wrapping_add(delta))
    }
}

/// [`get`] for the indexes `from..from + out.len()`, in order: each block's
/// parameters looked up once and its packed deltas decoded a run at a time
/// (`PackedLongs::decode_range`), each value the same expression [`get`]
/// computes. For a reader of consecutive addresses (a window of documents'
/// value ranges).
///
/// # Errors
/// What [`get`] reports for the first index it would fail on.
pub fn fill(data: &[u8], meta: &Meta, from: i64, out: &mut [i64]) -> Result<()> {
    // ARITH: `block_shift <= 22` (see `get`), so the mask and a block's length
    // fit; `i < out.len()` and `n <= out.len() - i`; `from + i` is checked.
    #[allow(clippy::arithmetic_side_effects)]
    {
        let mask = (1i64 << meta.block_shift) - 1;
        let mut i = 0usize;
        while i < out.len() {
            let Some(index) = i64::try_from(i).ok().and_then(|k| from.checked_add(k)) else {
                return Err(lucene_store::Error::Eof { offset: 0 });
            };
            if index < 0 {
                return Err(lucene_store::Error::Eof { offset: 0 });
            }
            let Some(&block) = meta.blocks.get((index >> meta.block_shift) as usize) else {
                return Err(lucene_store::Error::Eof { offset: 0 });
            };
            let block_index = index & mask;
            let in_block = usize::try_from(mask - block_index + 1).unwrap_or(usize::MAX);
            let n = (out.len() - i).min(in_block);
            let run = &mut out[i..i + n];
            let mut decoded = 0;
            if block.bpv == 0 {
                run.fill(0);
                decoded = n;
            } else if direct_reader::is_supported_bits(block.bpv) {
                let packed = usize::try_from(block.offset)
                    .ok()
                    .and_then(|o| data.get(o..))
                    .and_then(|slice| {
                        lucene_util::packed_longs::PackedLongs::new(slice, u32::from(block.bpv))
                    });
                if let Some(packed) = packed {
                    decoded = packed.decode_range(block_index as u64, run);
                }
            }
            // What the packed run could not reach (the end of the data), as
            // `get` reads it.
            for (k, v) in run.iter_mut().enumerate().skip(decoded) {
                let Some(slice) = usize::try_from(block.offset)
                    .ok()
                    .and_then(|o| data.get(o..))
                else {
                    return Err(lucene_store::Error::Eof { offset: 0 });
                };
                *v = direct_reader::get(slice, block.bpv, block_index + k as i64)?;
            }
            for (k, v) in run.iter_mut().enumerate() {
                *v = block
                    .min
                    .wrapping_add((block.avg * (block_index + k as i64) as f32) as i64)
                    .wrapping_add(*v);
            }
            i += n;
        }
    }
    Ok(())
}

/// The packed value at `index` of a block starting at byte `offset`, read with
/// one 8-byte load when that window is inside `data` -- `DirectReader`'s
/// per-width `get`, which the JIT specializes. `None` for anything the checked
/// [`direct_reader::get`] must answer instead: an unsupported width (its
/// corruption error), a negative offset or index, or a window running past
/// the end of `data`.
#[inline]
fn fast_packed_get(data: &[u8], offset: i64, bpv: u8, index: i64) -> Option<i64> {
    if !direct_reader::is_supported_bits(bpv) {
        return None;
    }
    let offset = usize::try_from(offset).ok()?;
    let bit = u64::try_from(index).ok()?.checked_mul(u64::from(bpv))?;
    let byte = offset.checked_add(usize::try_from(bit >> 3).ok()?)?;
    let window = data.get(byte..byte.checked_add(8)?)?;
    let word = u64::from_le_bytes(window.try_into().ok()?);
    // A supported width not divisible by 8 is at most 28 bits, so `shift + bpv
    // <= 35` fits the word; a width divisible by 8 has `shift == 0`. `bpv` is
    // 1..=64 here, so the mask shift is in range.
    let shift = (bit & 7) as u32;
    let mask = u64::MAX.checked_shr(64u32.saturating_sub(u32::from(bpv)))?;
    Some(((word >> shift) & mask) as i64)
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
mod tests;
