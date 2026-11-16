//! Port of `org.apache.lucene.util.packed.BlockPackedReaderIterator` (decode
//! side of `BlockPackedWriter`) — read-only, **decode-once, not lazy**.
//!
//! A sequence of values is split into fixed-size blocks (64 values each,
//! for term vectors). Each block is self-describing: a one-byte token
//! (`bitsPerValue << 1 | minValueEquals0`), an optional zigzag-encoded
//! `minValue` (omitted when it's exactly 0), then `bitsPerValue`-wide
//! deltas-from-`minValue` for every value in the block, packed via
//! [`crate::packed_ints`]'s MSB-first bitstream (a `bitsPerValue` of 0 means
//! every value in the block equals `minValue`, with no packed data at all).
//!
//! Ported as a single `decode_all` that materializes the whole sequence,
//! not Java's seekable iterator (`next`/`skip`) -- this port doesn't need
//! partial reads, matching the decode-fully choice already made for
//! `IndexedDISI`, stored fields, and the terms dictionary.

use lucene_store::data_input::DataInput;
use lucene_store::Result;

use crate::packed_ints;

const BLOCK_SIZE: i64 = 64;
const MIN_VALUE_EQUALS_0: u32 = 1;

/// `BlockPackedReaderIterator.readVLong` / `AbstractBlockPackedWriter.writeVLong`
/// -- **not** the same varint as `DataInput.readVLong`.
///
/// Java's generic `readVLong` rejects a 10-byte encoding; this one is
/// deliberately negative-capable: at most 8 continuation groups (bits 0..55),
/// after which a ninth byte contributes all 8 of its bits at shift 56, high bit
/// and all. Reading a block's zigzag-encoded `minValue` with the generic varint
/// instead mis-parses exactly the values whose ninth byte has its top bit set
/// (`|minValue|` around 2^62 and up): the generic reader treats that bit as a
/// continuation marker and swallows a byte of the *next* block.
// ARITH: `shift` starts at 0 and only advances by 7 while `shift < 56`, so
// it never exceeds 56 and `1u64 << shift` stays in range.
#[allow(clippy::arithmetic_side_effects)]
fn read_min_value_vlong(input: &mut impl DataInput) -> Result<i64> {
    let mut l: u64 = 0;
    let mut shift = 0u32;
    while shift < 56 {
        let b = input.read_byte()?;
        l |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(l as i64);
        }
        shift += 7;
    }
    let b = input.read_byte()?;
    Ok((l | ((b as u64) << 56)) as i64)
}

/// Encode side of [`read_min_value_vlong`].
// ARITH: `i` is a `u64` shifted right by a constant 7, and `k` is compared
// against 8 before every increment, so it never leaves `0..=8`.
#[allow(clippy::arithmetic_side_effects)]
fn write_min_value_vlong(out: &mut Vec<u8>, mut i: u64) {
    let mut k = 0;
    while i & !0x7f != 0 && k < 8 {
        out.push(((i & 0x7f) | 0x80) as u8);
        i >>= 7;
        k += 1;
    }
    out.push(i as u8);
}

/// `PackedInts.unsignedBitsRequired`: at least 1, never 0.
// ARITH: `u64::leading_zeros()` returns `0..=64`, so `64 - lz` is in `0..=64`.
#[allow(clippy::arithmetic_side_effects)]
fn unsigned_bits_required(v: i64) -> u32 {
    (64 - (v as u64).leading_zeros()).max(1)
}

/// `PackedInts.maxValue(bitsPerValue)` for `bits_per_value < 64`.
///
/// ARITH: the shift and the subtraction are safe because the sole call site
/// (in [`encode_all`]) reaches this only on the `bits_required != 64` branch,
/// and `bits_required` is itself `unsigned_bits_required`'s output, which is
/// `0..=64`. So `bits_per_value` is in `0..=63` here and `1u64 <<
/// bits_per_value` is at least 1. This is a write-path helper: no caller
/// passes it a width read off disk.
#[allow(clippy::arithmetic_side_effects)]
fn max_value(bits_per_value: u32) -> i64 {
    debug_assert!(bits_per_value < 64);
    ((1u64 << bits_per_value) - 1) as i64
}

/// The packed body of one block at its widest: 64 values of 64 bits.
const BLOCK_BYTES_MAX: usize = BLOCK_SIZE as usize * 8;

/// `count` values of `bits_per_value` (`1..=64`) bits each, packed MSB-first
/// in `bytes` (which [`packed_ints::byte_count`] sized for them), each plus
/// `min_value`, onto `out` -- what `PackedInts.ReaderIterator.next` yields
/// one at a time, read through a bit accumulator in one pass instead of one
/// [`packed_ints::get`] per value.
///
/// ARITH: `bits_per_value` is in `1..=64` (the caller rejected a width above
/// 64 and handles 0 itself), so `1 << bits_per_value` fits a `u128` and the
/// mask is at least 1. Before each value `acc_bits < bits_per_value` triggers
/// a refill of whole bytes, so `acc_bits <= bits_per_value + 7 <= 71 < 128`
/// and `acc_bits - bits_per_value` cannot underflow once the refill stops.
/// `bytes` holds at least `count * bits_per_value` bits (`byte_count` sized
/// it), so before the `k`-th value at least `(k + 1) * bits_per_value` bits
/// are there to load and the refill never runs short (were it to, it would
/// load zero bits rather than index past the slice).
#[allow(clippy::arithmetic_side_effects)]
fn unpack_block(
    bytes: &[u8],
    bits_per_value: u32,
    count: usize,
    min_value: i64,
    out: &mut Vec<i64>,
) {
    let mask: u128 = (1u128 << bits_per_value) - 1;
    let mut acc: u128 = 0;
    let mut acc_bits = 0u32;
    let mut at = 0usize;
    for _ in 0..count {
        while acc_bits < bits_per_value {
            acc = (acc << 8) | u128::from(bytes.get(at).copied().unwrap_or(0));
            acc_bits += 8;
            at += 1;
        }
        let raw = ((acc >> (acc_bits - bits_per_value)) & mask) as i64;
        acc_bits -= bits_per_value;
        out.push(raw.wrapping_add(min_value));
    }
}

/// Decodes `total_value_count` values written by `BlockPackedWriter`.
/// Reads nothing at all if `total_value_count == 0` (matches the writer,
/// which emits zero blocks for an empty stream).
pub(crate) fn decode_all(input: &mut impl DataInput, total_value_count: i64) -> Result<Vec<i64>> {
    // Every caller derives `total_value_count` from a chunk header in a `.tvd`
    // (`chunk_docs`, `total_terms`, `total_positions`, `total_offsets`,
    // `total_payloads`), so it is a value off disk. Reserving it up front lets
    // a corrupt header ask for a multi-terabyte allocation, which is an
    // *abort* -- `catch_unwind` cannot intercept an allocation failure, so it
    // takes the JVM down through the FFI.
    //
    // The input itself bounds how many values can really be there: the
    // cheapest possible block is a single token byte carrying 64 constant
    // values, so `remaining * BLOCK_SIZE` is a hard ceiling. Clamping to it
    // costs a well-formed stream nothing (its bytes are all there, so the
    // ceiling is above `total_value_count`) while a corrupt count can no
    // longer reserve more than the file could hold.
    let ceiling = input.remaining().saturating_mul(BLOCK_SIZE as usize);
    let reserve = (total_value_count.max(0) as usize).min(ceiling);
    let mut out = Vec::with_capacity(reserve);
    let mut produced: i64 = 0;
    while produced < total_value_count {
        let token = input.read_byte()? as u32;
        let min_value_equals_0 = token & MIN_VALUE_EQUALS_0 != 0;
        let bits_per_value = token >> 1;
        if bits_per_value > 64 {
            return Err(lucene_store::Error::Corrupted(format!(
                "block-packed bitsPerValue out of range: {bits_per_value}"
            )));
        }
        let min_value = if min_value_equals_0 {
            0i64
        } else {
            lucene_util::zigzag::decode(1u64.wrapping_add(read_min_value_vlong(input)? as u64))
        };

        // ARITH: `produced < total_value_count` is the loop condition and
        // `produced` starts at 0 and only ever grows, so the difference is in
        // `1..=total_value_count` and cannot overflow.
        #[allow(clippy::arithmetic_side_effects)]
        let block_value_count = (total_value_count - produced).min(BLOCK_SIZE) as usize;
        if bits_per_value == 0 {
            out.extend(std::iter::repeat_n(min_value, block_value_count));
        } else {
            // At most `BLOCK_SIZE * 64` bits: the block's packed body always
            // fits the stack buffer, whatever the token said.
            let byte_len = packed_ints::byte_count(block_value_count as u64, bits_per_value);
            let mut block_bytes = [0u8; BLOCK_BYTES_MAX];
            let block_bytes = &mut block_bytes[..byte_len];
            input.read_bytes(block_bytes)?;
            unpack_block(
                block_bytes,
                bits_per_value,
                block_value_count,
                min_value,
                &mut out,
            );
        }
        // ARITH: `block_value_count <= total_value_count - produced` by its
        // definition above, so `produced` lands in `..=total_value_count`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            produced += block_value_count as i64;
        }
    }
    Ok(out)
}

/// Encode side of [`decode_all`]: a full port of `BlockPackedWriter.flush`,
/// including the per-block minimum-value delta encoding.
///
/// Per 64-value block: `bitsPerValue` covers `max - min` rather than `max`,
/// `min` is then pulled down as far as that width still allows (Java's
/// `min = max(0, max - maxValue(bitsPerValue))`, which shortens the
/// `minValue` varint without widening anything), and `min == 0` is signalled
/// by the token's low bit instead of being written out.
///
/// Negative values matter: sizing `bitsPerValue` from the block max alone
/// would encode an all-negative block as the constant `0` and truncate a mixed
/// block's negative values. The term-vectors writer stores a length as `span - prefixLength -
/// suffixLength`, where `span` is `endOffset - startOffset`: the
/// occurrence's span minus the term's length *in UTF-8 bytes* -- exactly
/// what Java's
/// `Lucene90CompressingTermVectorsWriter.flushOffsets` writes and what its
/// reader adds back. The span is measured in UTF-16 code units
/// (`OffsetAttribute`'s unit), so **every
/// multi-byte term produces a negative length**: `caf\u{e9}` spans 4 `char`s
/// and occupies 5 bytes, giving `-1`. Start-offset deltas can be negative
/// too, since `charsPerTerm * position` may overshoot. `BlockPackedWriter` is
/// a general signed-`long` sequence writer and its decoder here already
/// handled negatives correctly on the read side.
///
/// Writes nothing for an empty slice, matching the decoder reading nothing for
/// `total_value_count == 0`.
pub(crate) fn encode_all(values: &[i64]) -> Vec<u8> {
    encode_all_with_block_size(values, BLOCK_SIZE as usize)
}

/// [`encode_all`] for any `BlockPackedWriter` block size (a power of two in
/// `64..=2^27`; the caller's choice, never a disk value).
pub fn encode_all_with_block_size(values: &[i64], block_size: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for block in values.chunks(block_size) {
        let mut min = *block.iter().min().unwrap();
        let max = *block.iter().max().unwrap();

        // Java lets `max - min` overflow into a negative `delta` and relies on
        // `unsignedBitsRequired` reading it as unsigned -- which lands on 64,
        // the only width that can hold the pair anyway.
        let delta = max.wrapping_sub(min);
        let bits_required = if delta == 0 {
            0
        } else {
            unsigned_bits_required(delta)
        };
        if bits_required == 64 {
            // No need to delta-encode: every value fits raw.
            min = 0;
        } else if min > 0 {
            // Make min as small as possible so that its varint is shorter.
            min = 0.max(max.wrapping_sub(max_value(bits_required)));
        }

        let token = (bits_required << 1) | if min == 0 { MIN_VALUE_EQUALS_0 } else { 0 };
        out.push(token as u8);
        if min != 0 {
            write_min_value_vlong(&mut out, lucene_util::zigzag::encode(min).wrapping_sub(1));
        }
        if bits_required > 0 {
            let deltas: Vec<i64> = block.iter().map(|&v| v.wrapping_sub(min)).collect();
            out.extend_from_slice(&packed_ints::encode(&deltas, bits_required));
        }
    }
    out
}

/// Port of `BlockPackedReaderIterator`: the streaming reader of a
/// `BlockPackedWriter` stream of any block size, with `next`, `next(count)`
/// and `skip`. [`decode_all`] is the decode-everything shortcut term vectors
/// uses; this is the general form.
#[derive(Debug)]
pub struct BlockPackedReaderIterator<'i, I: DataInput> {
    input: &'i mut I,
    value_count: u64,
    block_size: usize,
    values: Vec<i64>,
    blocks: Vec<u8>,
    off: usize,
    ord: u64,
}

impl<'i, I: DataInput> BlockPackedReaderIterator<'i, I> {
    /// `new BlockPackedReaderIterator(in, packedIntsVersion, blockSize, valueCount)`.
    pub fn new(
        input: &'i mut I,
        packed_ints_version: i32,
        block_size: usize,
        value_count: u64,
    ) -> Result<Self> {
        lucene_util::packed::check_version(packed_ints_version)
            .map_err(|e| lucene_store::Error::Corrupted(e.to_string()))?;
        lucene_util::packed::check_block_size(block_size, 64, 1 << 27)
            .map_err(|e| lucene_store::Error::Corrupted(e.to_string()))?;
        Ok(BlockPackedReaderIterator {
            input,
            value_count,
            block_size,
            values: vec![0i64; block_size],
            blocks: Vec::new(),
            off: block_size,
            ord: 0,
        })
    }

    /// `ord()`: values consumed so far.
    pub fn ord(&self) -> u64 {
        self.ord
    }

    /// `reset(in, valueCount)` without swapping the input: start over on a
    /// new stream of `value_count` values at the input's position.
    pub fn reset(&mut self, value_count: u64) {
        self.value_count = value_count;
        self.off = self.block_size;
        self.ord = 0;
    }

    /// `skip(count)`.
    pub fn skip(&mut self, mut count: u64) -> Result<()> {
        let target = self.ord.checked_add(count);
        if target.is_none_or(|t| t > self.value_count) {
            return Err(lucene_store::Error::Eof {
                offset: self.input.remaining(),
            });
        }
        // 1. skip buffered values
        // ARITH: `off <= block_size`; `skip_buffer <= count` and `ord + count
        // <= value_count` was checked above, so nothing overflows.
        #[allow(clippy::arithmetic_side_effects)]
        {
            let skip_buffer = count.min((self.block_size - self.off) as u64) as usize;
            self.off += skip_buffer;
            self.ord += skip_buffer as u64;
            count -= skip_buffer as u64;
        }
        if count == 0 {
            return Ok(());
        }
        // 2. skip as many blocks as necessary
        while count >= self.block_size as u64 {
            let token = self.input.read_byte()? as u32;
            let bits_per_value = token >> 1;
            if bits_per_value > 64 {
                return Err(lucene_store::Error::Corrupted("Corrupted".into()));
            }
            if token & MIN_VALUE_EQUALS_0 == 0 {
                read_min_value_vlong(self.input)?;
            }
            let block_bytes = packed_ints::byte_count(self.block_size as u64, bits_per_value);
            self.input.skip(block_bytes)?;
            // ARITH: `count >= block_size` is the loop condition, and `ord +
            // count <= value_count` still holds.
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.ord += self.block_size as u64;
                count -= self.block_size as u64;
            }
        }
        if count == 0 {
            return Ok(());
        }
        // 3. skip the last values
        self.refill()?;
        // ARITH: `count < block_size` and `off == 0` after `refill`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.ord += count;
            self.off += count as usize;
        }
        Ok(())
    }

    /// `next()`.
    pub fn next_value(&mut self) -> Result<i64> {
        if self.ord == self.value_count {
            return Err(lucene_store::Error::Eof {
                offset: self.input.remaining(),
            });
        }
        if self.off == self.block_size {
            self.refill()?;
        }
        let value = self.values[self.off];
        // ARITH: `off < block_size` after the refill; `ord < value_count`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.off += 1;
            self.ord += 1;
        }
        Ok(value)
    }

    /// `next(count)`: at least one and at most `count` values, never crossing
    /// a block.
    pub fn next_values(&mut self, count: usize) -> Result<&[i64]> {
        debug_assert!(count > 0);
        if self.ord == self.value_count {
            return Err(lucene_store::Error::Eof {
                offset: self.input.remaining(),
            });
        }
        if self.off == self.block_size {
            self.refill()?;
        }
        // ARITH: `off < block_size` and `ord < value_count` here.
        #[allow(clippy::arithmetic_side_effects)]
        {
            let count = count.min(self.block_size - self.off);
            let count = (count as u64).min(self.value_count - self.ord) as usize;
            let start = self.off;
            self.off += count;
            self.ord += count as u64;
            Ok(&self.values[start..start + count])
        }
    }

    /// `refill()`: decode the next block into `values`.
    fn refill(&mut self) -> Result<()> {
        let token = self.input.read_byte()? as u32;
        let min_equals_0 = token & MIN_VALUE_EQUALS_0 != 0;
        let bits_per_value = token >> 1;
        if bits_per_value > 64 {
            return Err(lucene_store::Error::Corrupted("Corrupted".into()));
        }
        let min_value = if min_equals_0 {
            0
        } else {
            lucene_util::zigzag::decode(1u64.wrapping_add(read_min_value_vlong(self.input)? as u64))
        };
        if bits_per_value == 0 {
            self.values.fill(min_value);
        } else {
            let decoder = lucene_util::packed::BulkOperation::of(
                lucene_util::packed::Format::Packed,
                bits_per_value,
            );
            // ARITH: the block size is at most 2^27 and the decoder's counts
            // at most 64; `ord < value_count` whenever a block is refilled.
            #[allow(clippy::arithmetic_side_effects)]
            let (iterations, blocks_size, value_count) = {
                let iterations = self.block_size / decoder.byte_value_count();
                (
                    iterations,
                    iterations * decoder.byte_block_count(),
                    (self.value_count - self.ord).min(self.block_size as u64) as usize,
                )
            };
            if self.blocks.len() < blocks_size {
                self.blocks.resize(blocks_size, 0);
            }
            let blocks_count = packed_ints::byte_count(value_count as u64, bits_per_value);
            self.input.read_bytes(&mut self.blocks[..blocks_count])?;
            decoder.decode_bytes(&self.blocks, &mut self.values, iterations);
            if min_value != 0 {
                for v in &mut self.values[..value_count] {
                    *v = v.wrapping_add(min_value);
                }
            }
        }
        self.off = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
