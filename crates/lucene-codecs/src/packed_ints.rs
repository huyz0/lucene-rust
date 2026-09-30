//! Port of `org.apache.lucene.util.packed.PackedInts.Format.PACKED`'s bulk
//! bit-packing (`BulkOperationPacked`) — a *different* convention from
//! [`crate::direct_reader`] (which ports `DirectReader`/`DirectWriter`):
//! values are packed **MSB-first as one contiguous bitstream** across the
//! whole byte array, with no per-value byte alignment, versus
//! `direct_reader`'s LSB-first-within-byte, whitelisted-width scheme.
//! Term vectors uses both conventions for different arrays in the same
//! file, so both need to exist side by side.
//!
//! This is the "headerless flat array" case (a fixed `bits_per_value` for
//! every value, no min-value, no block splitting) — used directly for
//! term vectors' distinct-field-numbers array, and as the per-block body
//! decoder inside [`crate::block_packed`].

use lucene_store::Result;

/// Reads the `index`-th `bits_per_value`-wide value from `data`, where
/// values are packed MSB-first as one contiguous bitstream (bit 7 of byte 0
/// is the first bit of value 0).
pub(crate) fn get(data: &[u8], bits_per_value: u32, index: i64) -> Result<i64> {
    // `bits_per_value` is a token field off a `.tvd` chunk header on both call
    // paths, and every shift below is only in range once it is bounded. Both
    // current callers do bound it before calling -- `block_packed::decode_all`
    // rejects a width above 64 outright, and term vectors masks the token to
    // five bits -- but this is where the shifts actually live, so this is
    // where the invariant belongs: a `pub(crate)` primitive should not depend
    // on every future caller remembering. `PackedInts.Format.PACKED` cannot
    // represent a width above 64 (`BlockPackedReaderIterator` throws on
    // exactly this), so anything wider is corruption.
    if bits_per_value > 64 {
        return Err(lucene_store::Error::Corrupted(format!(
            "packed-ints bitsPerValue out of range: {bits_per_value}"
        )));
    }
    // `index as u128` sign-extends, so a negative index becomes ~2^128 and
    // the multiply below overflows -- a debug-build panic before any bounds
    // check runs. Both current callers pass a non-negative index, but so does
    // every caller of the `bits_per_value` check above, and the same argument
    // applies: this is where the arithmetic lives.
    let Ok(index) = u64::try_from(index) else {
        return Err(lucene_store::Error::Corrupted(format!(
            "packed-ints index must be non-negative, got {index}"
        )));
    };
    // ARITH: the product is computed in `u128` from a `u64` and a value now
    // known to be `<= 64`, so it cannot overflow. `bit_offset` is masked to
    // `0..=7`, so `total_bits <= 71` and `n_bytes <= 9`.
    #[allow(clippy::arithmetic_side_effects)]
    let (byte_pos, bit_offset, n_bytes) = {
        let bit_pos = (index as u128) * (bits_per_value as u128);
        let byte_pos =
            usize::try_from(bit_pos >> 3).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
        let bit_offset = (bit_pos & 7) as u32;
        let total_bits = bit_offset + bits_per_value;
        (byte_pos, bit_offset, total_bits.div_ceil(8) as usize)
    };

    // `index` is a value ordinal off disk, so `byte_pos` is unbounded on a
    // 32-bit target (the `try_from` above admits up to `u32::MAX`, where
    // `byte_pos + n_bytes` would wrap). One comparison against the slice
    // length makes the range below provably in bounds on every target.
    if byte_pos > data.len() {
        return Err(lucene_store::Error::Eof { offset: byte_pos });
    }
    // ARITH: `byte_pos <= data.len() <= isize::MAX` and `n_bytes <= 9`.
    #[allow(clippy::arithmetic_side_effects)]
    let range = byte_pos..byte_pos + n_bytes;
    let bytes = data
        .get(range)
        .ok_or(lucene_store::Error::Eof { offset: byte_pos })?;
    let mut acc: u128 = 0;
    // ARITH: `bytes.len() == n_bytes <= 9`, so `acc` holds at most 72 bits of
    // a 128-bit accumulator.
    #[allow(clippy::arithmetic_side_effects)]
    for &b in bytes {
        acc = (acc << 8) | b as u128;
    }
    // ARITH: `n_bytes * 8 = ceil(total_bits / 8) * 8 >= total_bits =
    // bit_offset + bits_per_value`, so the two subtractions cannot underflow,
    // and `n_bytes * 8 <= 72`. `bits_per_value <= 64 < 128`, so the mask's
    // shift is in range and its result is at least 1.
    #[allow(clippy::arithmetic_side_effects)]
    let value = {
        let shift = (n_bytes as u32) * 8 - bit_offset - bits_per_value;
        let mask: u128 = (1u128 << bits_per_value) - 1;
        ((acc >> shift) & mask) as i64
    };
    Ok(value)
}

/// Number of bytes needed to pack `count` values of `bits_per_value` width
/// (`PackedInts.Format.PACKED.byteCount`): `ceil(count * bits_per_value / 8)`.
// ARITH: the product is computed in `u128` from a `u64` and a `u32`, so it
// cannot overflow. `count` is unsigned rather than Java's `long` on purpose:
// as an `i64` a negative count produced a negative quotient, and `as usize`
// turned that into a gigantic length that callers then handed to
// `vec![0u8; n]`.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn byte_count(count: u64, bits_per_value: u32) -> usize {
    ((count as u128 * bits_per_value as u128).div_ceil(8)) as usize
}

/// Encode side of [`get`]: packs `values` MSB-first as one contiguous
/// bitstream, the exact inverse of `get`'s formula. `bits_per_value` may be
/// any width `0..=64` (unlike [`crate::direct_reader`], this convention has
/// no whitelist of supported widths) -- `bits_per_value == 0` writes nothing
/// (every value is assumed to be 0, matching `get`'s masked-to-zero read).
// ARITH: this is the encode side and `bits_per_value` is chosen by the caller
// from its own data, never read off disk. `bit_off` is masked to `0..=7`, so
// `free = 8 - bit_off` is in `1..=8`; `take = min(remaining, free)` is in
// `0..=8`, so `shift_in_value = remaining - take` cannot underflow, `1u64 <<
// take` is in range, `free - take` is in `0..=8`, `remaining -= take`
// terminates at 0, and `bit_pos` advances by at most `values.len() *
// bits_per_value` bits, which `byte_count` already sized `out` for.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn encode(values: &[i64], bits_per_value: u32) -> Vec<u8> {
    let n_bytes = byte_count(values.len() as u64, bits_per_value);
    let mut out = vec![0u8; n_bytes];
    let mut bit_pos: u64 = 0;
    for &v in values {
        let mut remaining = bits_per_value;
        while remaining > 0 {
            let byte_idx = (bit_pos >> 3) as usize;
            let bit_off = (bit_pos & 7) as u32;
            let free = 8 - bit_off;
            let take = remaining.min(free);
            let shift_in_value = remaining - take;
            let mask: u64 = if take == 64 {
                u64::MAX
            } else {
                (1u64 << take) - 1
            };
            let bits_val = ((v as u64) >> shift_in_value) & mask;
            out[byte_idx] |= (bits_val as u8) << (free - take);
            bit_pos += take as u64;
            remaining -= take;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The streaming serialized forms: `PackedWriter`, `PackedReaderIterator` and
// `DirectPacked64SingleBlockReader`, for both `PackedInts.Format`s. The bit
// layouts are `lucene_util::packed::BulkOperation`'s; this half moves bytes.
// ---------------------------------------------------------------------------

use lucene_store::data_input::DataInput;
use lucene_store::data_output::DataOutput;
use lucene_util::packed::{BulkOperation, Format, VERSION_CURRENT};

fn corrupt<T>(msg: impl Into<String>) -> Result<T> {
    Err(lucene_store::Error::Corrupted(msg.into()))
}

/// Port of `PackedInts.Writer`/`PackedWriter` (`PackedInts.getWriterNoHeader`):
/// buffers values and flushes them through the format's byte encoder. The
/// stream carries no header; the reader must be told the format, count and
/// width.
#[derive(Debug)]
pub struct PackedWriter<'o, O: DataOutput> {
    out: &'o mut O,
    format: Format,
    /// `None` is Java's `-1`: an unknown number of values.
    value_count: Option<usize>,
    bits_per_value: u32,
    encoder: BulkOperation,
    next_blocks: Vec<u8>,
    next_values: Vec<i64>,
    iterations: usize,
    off: usize,
    written: usize,
    finished: bool,
}

impl<'o, O: DataOutput> PackedWriter<'o, O> {
    /// `PackedInts.getWriterNoHeader(out, format, valueCount, bitsPerValue, mem)`.
    /// `mem` bounds the buffer; it does not change the bytes written.
    pub fn new(
        out: &'o mut O,
        format: Format,
        value_count: Option<usize>,
        bits_per_value: u32,
        mem: usize,
    ) -> Self {
        let encoder = BulkOperation::of(format, bits_per_value);
        let iterations = encoder.compute_iterations(value_count.unwrap_or(i32::MAX as usize), mem);
        // ARITH: `iterations` is at most `mem / (block + 8 * value)` or
        // `ceil(value_count / byte_value_count)`, and the block and value
        // counts are at most 64 -- a writer-side geometry, not a disk value.
        #[allow(clippy::arithmetic_side_effects)]
        let (blocks_len, values_len) = (
            iterations * encoder.byte_block_count(),
            iterations * encoder.byte_value_count(),
        );
        PackedWriter {
            out,
            format,
            value_count,
            bits_per_value,
            encoder,
            next_blocks: vec![0u8; blocks_len],
            next_values: vec![0i64; values_len],
            iterations,
            off: 0,
            written: 0,
            finished: false,
        }
    }

    /// `Writer.bitsPerValue()`.
    pub fn bits_per_value(&self) -> u32 {
        self.bits_per_value
    }

    /// `Writer.getFormat()`.
    pub fn format(&self) -> Format {
        self.format
    }

    /// `add(long)`. Writing past a known value count is Java's `EOFException`.
    pub fn add(&mut self, v: i64) -> Result<()> {
        debug_assert!(lucene_util::packed::unsigned_bits_required(v as u64) <= self.bits_per_value);
        debug_assert!(!self.finished);
        if let Some(count) = self.value_count {
            if self.written >= count {
                return Err(lucene_store::Error::Eof {
                    offset: self.written,
                });
            }
        }
        self.next_values[self.off] = v;
        // ARITH: `off < next_values.len()` (it is reset to 0 on reaching it),
        // and `written` counts calls, bounded by `value_count` or by memory.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.off += 1;
            if self.off == self.next_values.len() {
                self.flush();
            }
            self.written += 1;
        }
        Ok(())
    }

    /// `finish()`: pads a known count with zeros, then flushes.
    pub fn finish(&mut self) -> Result<()> {
        debug_assert!(!self.finished);
        if let Some(count) = self.value_count {
            while self.written < count {
                self.add(0)?;
            }
        }
        self.flush();
        self.finished = true;
        Ok(())
    }

    fn flush(&mut self) {
        self.encoder
            .encode_bytes(&self.next_values, &mut self.next_blocks, self.iterations);
        let block_count =
            self.format
                .byte_count(VERSION_CURRENT, self.off, self.bits_per_value) as usize;
        self.out.write_bytes(&self.next_blocks[..block_count]);
        self.next_values.fill(0);
        self.off = 0;
    }

    /// `ord()`: values written so far, minus one (`-1` before the first).
    pub fn ord(&self) -> i64 {
        // ARITH: `written` fits an `i64` (it is bounded by memory), and
        // subtracting 1 from a non-negative `i64` cannot overflow.
        #[allow(clippy::arithmetic_side_effects)]
        let ord = self.written as i64 - 1;
        ord
    }
}

/// Port of `PackedReaderIterator` (`PackedInts.getReaderIteratorNoHeader`):
/// decodes a headerless stream a buffer at a time.
#[derive(Debug)]
pub struct PackedReaderIterator<'i, I: DataInput> {
    input: &'i mut I,
    format: Format,
    value_count: usize,
    bits_per_value: u32,
    bulk_operation: BulkOperation,
    next_blocks: Vec<u8>,
    next_values: Vec<i64>,
    /// `nextValues.offset` / `.length`.
    values_offset: usize,
    values_length: usize,
    iterations: usize,
    /// `position`: the last value returned, `-1` before the first.
    position: i64,
}

impl<'i, I: DataInput> PackedReaderIterator<'i, I> {
    /// `PackedInts.getReaderIteratorNoHeader(in, format, version, valueCount,
    /// bitsPerValue, mem)`. The format, width and count come from the
    /// caller's metadata, so they are validated here.
    pub fn new(
        input: &'i mut I,
        format: Format,
        version: i32,
        value_count: usize,
        bits_per_value: u32,
        mem: usize,
    ) -> Result<Self> {
        lucene_util::packed::check_version(version)
            .map_err(|e| lucene_store::Error::Corrupted(e.to_string()))?;
        if !format.is_supported(bits_per_value) {
            return corrupt(format!(
                "unsupported bitsPerValue {bits_per_value} for {format:?}"
            ));
        }
        // Not in Java, which trusts its caller: a count off disk must not size
        // a buffer the input cannot fill.
        let needed = format.byte_count(version, value_count, bits_per_value);
        if needed > input.remaining() as u64 {
            return Err(lucene_store::Error::Eof {
                offset: input.remaining(),
            });
        }
        let bulk_operation = BulkOperation::of(format, bits_per_value);
        let iterations = bulk_operation.compute_iterations(value_count, mem);
        // ARITH: `compute_iterations` returns at most `ceil(value_count /
        // byte_value_count)` or `mem / (block + 8 * value)`, so the products
        // are at most about `value_count * 8 + mem`; a count off disk that
        // large is refused by the input length before anything is read.
        #[allow(clippy::arithmetic_side_effects)]
        let (blocks_len, values_len) = (
            iterations * bulk_operation.byte_block_count(),
            iterations * bulk_operation.byte_value_count(),
        );
        Ok(PackedReaderIterator {
            input,
            format,
            value_count,
            bits_per_value,
            bulk_operation,
            next_blocks: vec![0u8; blocks_len],
            next_values: vec![0i64; values_len],
            values_offset: values_len,
            values_length: 0,
            iterations,
            position: -1,
        })
    }

    /// `getBitsPerValue()`.
    pub fn bits_per_value(&self) -> u32 {
        self.bits_per_value
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.value_count
    }

    /// `ord()`: the index of the last value returned (`-1` before the first).
    pub fn ord(&self) -> i64 {
        self.position
    }

    /// `next(int count)`: at least one and at most `count` next values;
    /// reading past the end is Java's `EOFException`.
    pub fn next_values(&mut self, count: usize) -> Result<&[i64]> {
        debug_assert!(count > 0);
        // ARITH: `values_offset + values_length <= next_values.len()` holds by
        // construction below; `position` is in `-1..value_count`, so
        // `value_count - position - 1` is in `0..=value_count`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.values_offset += self.values_length;
            let remaining = self.value_count as i64 - self.position - 1;
            if remaining <= 0 {
                return Err(lucene_store::Error::Eof {
                    offset: self.value_count,
                });
            }
            let remaining = remaining as usize;
            let count = remaining.min(count);
            if self.values_offset == self.next_values.len() {
                let remaining_blocks =
                    self.format
                        .byte_count(VERSION_CURRENT, remaining, self.bits_per_value);
                let blocks_to_read = (remaining_blocks as usize).min(self.next_blocks.len());
                self.input
                    .read_bytes(&mut self.next_blocks[..blocks_to_read])?;
                self.next_blocks[blocks_to_read..].fill(0);
                self.bulk_operation.decode_bytes(
                    &self.next_blocks,
                    &mut self.next_values,
                    self.iterations,
                );
                self.values_offset = 0;
            }
            self.values_length = (self.next_values.len() - self.values_offset).min(count);
            self.position += self.values_length as i64;
            Ok(&self.next_values[self.values_offset..self.values_offset + self.values_length])
        }
    }

    /// `next()`: the next value.
    pub fn next_value(&mut self) -> Result<i64> {
        let v = self.next_values(1)?[0];
        // Java's `ReaderIteratorImpl.next` consumes the one value from the ref.
        // ARITH: `next_values(1)` just returned one value, so `values_length`
        // is 1 and `values_offset < next_values.len()`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.values_offset += 1;
            self.values_length -= 1;
        }
        Ok(v)
    }
}

/// Port of `DirectPacked64SingleBlockReader`: random access to a
/// `PACKED_SINGLE_BLOCK` stream left on disk, one `readLong` (little-endian,
/// as every Lucene 9+ `DataInput.readLong` is) per lookup.
#[derive(Debug, Clone, Copy)]
pub struct DirectPacked64SingleBlockReader<'a> {
    data: &'a [u8],
    value_count: usize,
    bits_per_value: u32,
    values_per_block: usize,
    mask: u64,
}

impl<'a> DirectPacked64SingleBlockReader<'a> {
    /// `new DirectPacked64SingleBlockReader(bitsPerValue, valueCount, in)`,
    /// with `data` starting at the stream's first block.
    pub fn new(bits_per_value: u32, value_count: usize, data: &'a [u8]) -> Result<Self> {
        if !Format::PackedSingleBlock.is_supported(bits_per_value) {
            return corrupt(format!(
                "unsupported PACKED_SINGLE_BLOCK bitsPerValue {bits_per_value}"
            ));
        }
        // ARITH: `bits_per_value` is one of the supported widths (1..=32).
        #[allow(clippy::arithmetic_side_effects)]
        let values_per_block = 64 / bits_per_value as usize;
        Ok(DirectPacked64SingleBlockReader {
            data,
            value_count,
            bits_per_value,
            values_per_block,
            mask: !(u64::MAX << bits_per_value),
        })
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.value_count
    }

    /// `get(int)`.
    pub fn get(&self, index: usize) -> Result<i64> {
        // ARITH: `values_per_block >= 2`; `block_offset * 8` is checked;
        // `offset_in_block < values_per_block`, so the shift is below 64.
        #[allow(clippy::arithmetic_side_effects)]
        let (skip, shift) = {
            let block_offset = index / self.values_per_block;
            let skip = block_offset
                .checked_mul(8)
                .ok_or(lucene_store::Error::Eof { offset: usize::MAX })?;
            (
                skip,
                (index % self.values_per_block) as u32 * self.bits_per_value,
            )
        };
        let end = skip
            .checked_add(8)
            .ok_or(lucene_store::Error::Eof { offset: skip })?;
        let bytes = self
            .data
            .get(skip..end)
            .ok_or(lucene_store::Error::Eof { offset: skip })?;
        let block = u64::from_le_bytes(bytes.try_into().expect("8 bytes"));
        Ok(((block >> shift) & self.mask) as i64)
    }
}

#[cfg(test)]
mod tests {
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
    /// `index as u128` sign-extends, so a negative index used to become
    /// ~2^128 and overflow the `index * bits_per_value` multiply -- a
    /// debug-build panic before any bounds check ran.
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
}
