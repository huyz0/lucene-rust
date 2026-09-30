//! Port of `org.apache.lucene.util.packed.MonotonicBlockPackedWriter` and
//! `MonotonicBlockPackedReader`: a (mostly) increasing sequence of `long`s in
//! fixed-size blocks, each block storing its values as deltas from a line.
//!
//! Per block of `blockSize` values (the last may be short):
//!
//! ```text
//! min      zlong      the line's origin, lowered until every delta is >= 0
//! average  int32      Float.floatToIntBits((last - first) / (n - 1)), 0 for n == 1
//! bpv      vint       width of the deltas, 0 when every delta is 0
//! deltas   PACKED     n values of bpv bits: value[i] - expected(min, average, i)
//! ```
//!
//! with `expected(min, avg, i) = min + (long) (avg * i)` in `float` arithmetic
//! ([`lucene_util::packed::packed_long_values::expected`]).

use lucene_store::data_input::DataInput;
use lucene_store::data_output::DataOutput;
use lucene_store::Result;
use lucene_util::packed::packed_long_values::expected;
use lucene_util::packed::{BulkOperation, Format, VERSION_CURRENT};

/// `AbstractBlockPackedWriter.MIN_BLOCK_SIZE`.
pub const MIN_BLOCK_SIZE: usize = 64;
/// `AbstractBlockPackedWriter.MAX_BLOCK_SIZE`.
pub const MAX_BLOCK_SIZE: usize = 1 << (30 - 3);

fn corrupt<T>(msg: impl Into<String>) -> Result<T> {
    Err(lucene_store::Error::Corrupted(msg.into()))
}

fn check_block_size(block_size: usize) -> Result<u32> {
    lucene_util::packed::check_block_size(block_size, MIN_BLOCK_SIZE, MAX_BLOCK_SIZE)
        .map_err(|e| lucene_store::Error::Corrupted(e.to_string()))
}

/// `MonotonicBlockPackedWriter`.
#[derive(Debug)]
pub struct MonotonicBlockPackedWriter<'o, O: DataOutput> {
    out: &'o mut O,
    values: Vec<i64>,
    blocks: Vec<u8>,
    off: usize,
    ord: u64,
    finished: bool,
}

impl<'o, O: DataOutput> MonotonicBlockPackedWriter<'o, O> {
    /// `new MonotonicBlockPackedWriter(out, blockSize)`. Panics on a block
    /// size that is not a power of two in `64..=2^27` (Java's
    /// `IllegalArgumentException`; a writer-side bug).
    pub fn new(out: &'o mut O, block_size: usize) -> Self {
        check_block_size(block_size).expect("valid block size");
        MonotonicBlockPackedWriter {
            out,
            values: vec![0i64; block_size],
            blocks: Vec::new(),
            off: 0,
            ord: 0,
            finished: false,
        }
    }

    /// `add(long)`. Values must be non-negative (a Java `assert`). Adding
    /// after `finish` is Java's `IllegalStateException`, a panic here.
    pub fn add(&mut self, l: i64) {
        debug_assert!(l >= 0);
        assert!(!self.finished, "Already finished");
        if self.off == self.values.len() {
            self.flush();
        }
        self.values[self.off] = l;
        // ARITH: `off < values.len()` after the flush above; `ord` counts
        // calls.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.off += 1;
            self.ord += 1;
        }
    }

    /// `finish()`: flush the last (possibly short) block.
    pub fn finish(&mut self) {
        assert!(!self.finished, "Already finished");
        if self.off > 0 {
            self.flush();
        }
        self.finished = true;
    }

    /// `ord()`: values added so far.
    pub fn ord(&self) -> u64 {
        self.ord
    }

    /// `MonotonicBlockPackedWriter.flush`.
    fn flush(&mut self) {
        debug_assert!(self.off > 0);
        let off = self.off;
        // ARITH: `off >= 1`, so `off - 1` cannot underflow; value arithmetic
        // wraps as Java's `long` does.
        #[allow(clippy::arithmetic_side_effects)]
        let avg = if off == 1 {
            0f32
        } else {
            self.values[off - 1].wrapping_sub(self.values[0]) as f32 / (off - 1) as f32
        };
        let mut min = self.values[0];
        for i in 1..off {
            let actual = self.values[i];
            let exp = expected(min, avg, i as i64);
            if exp > actual {
                min = min.wrapping_sub(exp.wrapping_sub(actual));
            }
        }
        let mut max_delta = 0i64;
        for i in 0..off {
            self.values[i] = self.values[i].wrapping_sub(expected(min, avg, i as i64));
            max_delta = max_delta.max(self.values[i]);
        }
        self.out.write_zlong(min);
        self.out.write_i32(avg.to_bits() as i32);
        if max_delta == 0 {
            self.out.write_vint(0);
        } else {
            let bits_required = lucene_util::packed::unsigned_bits_required(max_delta as u64);
            self.out.write_vint(bits_required as i32);
            self.write_values(bits_required);
        }
        self.off = 0;
    }

    /// `AbstractBlockPackedWriter.writeValues`.
    fn write_values(&mut self, bits_required: u32) {
        let encoder = BulkOperation::of(Format::Packed, bits_required);
        // ARITH: `values.len()` is a block size (at most 2^27) and the
        // encoder's counts are at most 64.
        #[allow(clippy::arithmetic_side_effects)]
        let (iterations, block_size) = {
            let iterations = self.values.len() / encoder.byte_value_count();
            (iterations, encoder.byte_block_count() * iterations)
        };
        if self.blocks.len() < block_size {
            self.blocks.resize(block_size, 0);
        }
        let off = self.off;
        self.values[off..].fill(0);
        encoder.encode_bytes(&self.values, &mut self.blocks, iterations);
        let block_count = Format::Packed.byte_count(VERSION_CURRENT, off, bits_required) as usize;
        self.out.write_bytes(&self.blocks[..block_count]);
    }
}

/// `MonotonicBlockPackedReader`: every block's header decoded up front, the
/// packed deltas kept as bytes and read per `get`.
#[derive(Debug, Clone)]
pub struct MonotonicBlockPackedReader {
    block_shift: u32,
    block_mask: u64,
    value_count: u64,
    min_values: Vec<i64>,
    averages: Vec<f32>,
    /// Per block: `(bits_per_value, packed bytes)`; width 0 means all zeros.
    sub_readers: Vec<(u32, Vec<u8>)>,
}

impl MonotonicBlockPackedReader {
    /// `MonotonicBlockPackedReader.of(in, packedIntsVersion, blockSize, valueCount)`.
    pub fn of(
        input: &mut impl DataInput,
        packed_ints_version: i32,
        block_size: usize,
        value_count: u64,
    ) -> Result<Self> {
        lucene_util::packed::check_version(packed_ints_version)
            .map_err(|e| lucene_store::Error::Corrupted(e.to_string()))?;
        let block_shift = check_block_size(block_size)?;
        let num_blocks = lucene_util::packed::num_blocks(value_count, block_size)
            .map_err(|e| lucene_store::Error::Corrupted(e.to_string()))?;
        // Every block costs at least 6 bytes (a 1-byte zlong, the 4-byte
        // average and a 1-byte width), so a count the input cannot hold is
        // refused before anything is reserved for it.
        if num_blocks > input.remaining() / 6 {
            return Err(lucene_store::Error::Eof {
                offset: input.remaining(),
            });
        }
        let mut min_values = Vec::with_capacity(num_blocks);
        let mut averages = Vec::with_capacity(num_blocks);
        let mut sub_readers = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            min_values.push(input.read_zlong()?);
            averages.push(f32::from_bits(input.read_i32()? as u32));
            let bits_per_value = input.read_vint()?;
            if !(0..=64).contains(&bits_per_value) {
                return corrupt("Corrupted");
            }
            let bits_per_value = bits_per_value as u32;
            if bits_per_value == 0 {
                sub_readers.push((0, Vec::new()));
            } else {
                // ARITH: `i < num_blocks = ceil(value_count / block_size)`, so
                // `i * block_size < value_count` and the subtraction is
                // positive; the product fits `u64` for the same reason.
                #[allow(clippy::arithmetic_side_effects)]
                let size =
                    (block_size as u64).min(value_count - i as u64 * block_size as u64) as usize;
                let byte_count =
                    Format::Packed.byte_count(packed_ints_version, size, bits_per_value) as usize;
                if byte_count > input.remaining() {
                    return Err(lucene_store::Error::Eof {
                        offset: input.remaining(),
                    });
                }
                let mut blocks = vec![0u8; byte_count];
                input.read_bytes(&mut blocks)?;
                sub_readers.push((bits_per_value, blocks));
            }
        }
        Ok(MonotonicBlockPackedReader {
            block_shift,
            // ARITH: `block_size` is a power of two >= 64.
            #[allow(clippy::arithmetic_side_effects)]
            block_mask: block_size as u64 - 1,
            value_count,
            min_values,
            averages,
            sub_readers,
        })
    }

    /// `size()`.
    pub fn size(&self) -> u64 {
        self.value_count
    }

    /// Per-block widths (for the differential test and `toString`'s avgBPV).
    pub fn block_bits_per_value(&self) -> impl Iterator<Item = u32> + '_ {
        self.sub_readers.iter().map(|(b, _)| *b)
    }

    /// `get(long)`. Panics past `size()` (a Java `assert` on the caller).
    pub fn get(&self, index: u64) -> i64 {
        assert!(
            index < self.value_count,
            "index {index} >= size {}",
            self.value_count
        );
        let block = (index >> self.block_shift) as usize;
        let idx = index & self.block_mask;
        let (bpv, bytes) = &self.sub_readers[block];
        let delta = if *bpv == 0 {
            0
        } else {
            // The anonymous `LongValues` in Java's constructor; the same
            // MSB-first bitstream `packed_ints::get` reads, whose bounds were
            // established when `bytes` was sized from the block's length.
            crate::packed_ints::get(bytes, *bpv, idx as i64).expect("in-bounds by construction")
        };
        expected(self.min_values[block], self.averages[block], idx as i64).wrapping_add(delta)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use lucene_store::data_input::SliceInput;

    fn round_trip(values: &[i64], block_size: usize) {
        let mut buf = Vec::new();
        {
            let mut w = MonotonicBlockPackedWriter::new(&mut buf, block_size);
            for &v in values {
                w.add(v);
            }
            assert_eq!(w.ord(), values.len() as u64);
            w.finish();
        }
        let mut input = SliceInput::new(&buf);
        let r = MonotonicBlockPackedReader::of(
            &mut input,
            VERSION_CURRENT,
            block_size,
            values.len() as u64,
        )
        .unwrap();
        assert_eq!(input.remaining(), 0);
        assert_eq!(r.size(), values.len() as u64);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(r.get(i as u64), v, "i={i} block_size={block_size}");
        }
    }

    #[test]
    fn round_trips_monotonic_constant_and_noisy_sequences() {
        for block_size in [64usize, 128, 1024] {
            round_trip(&[], block_size);
            round_trip(&[5], block_size);
            round_trip(&(0..1000).map(|i| i * 3).collect::<Vec<_>>(), block_size);
            round_trip(&vec![42; 300], block_size);
            round_trip(
                &(0..777)
                    .map(|i: i64| i * 1000 + (i * 7919 % 13))
                    .collect::<Vec<_>>(),
                block_size,
            );
            round_trip(
                &(0..200)
                    .map(|i: i64| (i * 104729) % 100_000)
                    .collect::<Vec<_>>(),
                block_size,
            );
            round_trip(&[0, i64::MAX / 2, 1, i64::MAX, 3], block_size);
        }
    }

    #[test]
    fn exact_line_needs_zero_bits() {
        let values: Vec<i64> = (0..64).map(|i| 10 + 4 * i).collect();
        let mut buf = Vec::new();
        let mut w = MonotonicBlockPackedWriter::new(&mut buf, 64);
        for &v in &values {
            w.add(v);
        }
        w.finish();
        // zlong(10) = 1 byte, float 4.0 bits, vint 0.
        assert_eq!(buf.len(), 6);
        let r = MonotonicBlockPackedReader::of(&mut SliceInput::new(&buf), VERSION_CURRENT, 64, 64)
            .unwrap();
        assert_eq!(r.block_bits_per_value().collect::<Vec<_>>(), vec![0]);
        assert_eq!(r.get(63), 10 + 4 * 63);
    }

    #[test]
    fn corrupt_inputs_are_errors() {
        // Width 65.
        let mut buf = Vec::new();
        buf.write_zlong(0);
        buf.write_i32(0);
        buf.write_vint(65);
        assert!(MonotonicBlockPackedReader::of(
            &mut SliceInput::new(&buf),
            VERSION_CURRENT,
            64,
            10
        )
        .is_err());
        // A count the input cannot hold.
        assert!(MonotonicBlockPackedReader::of(
            &mut SliceInput::new(&buf),
            VERSION_CURRENT,
            64,
            1 << 40
        )
        .is_err());
        // Packed bytes missing.
        let mut buf = Vec::new();
        buf.write_zlong(0);
        buf.write_i32(0);
        buf.write_vint(8);
        buf.extend_from_slice(&[0u8; 5]);
        assert!(MonotonicBlockPackedReader::of(
            &mut SliceInput::new(&buf),
            VERSION_CURRENT,
            64,
            10
        )
        .is_err());
        // Bad version and block size.
        assert!(MonotonicBlockPackedReader::of(&mut SliceInput::new(&buf), 1, 64, 10).is_err());
        assert!(MonotonicBlockPackedReader::of(
            &mut SliceInput::new(&buf),
            VERSION_CURRENT,
            100,
            10
        )
        .is_err());
    }

    #[test]
    #[should_panic(expected = "Already finished")]
    fn add_after_finish_panics() {
        let mut buf = Vec::new();
        let mut w = MonotonicBlockPackedWriter::new(&mut buf, 64);
        w.finish();
        w.add(1);
    }
}
