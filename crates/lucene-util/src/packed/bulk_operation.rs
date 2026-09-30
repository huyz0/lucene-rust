//! Port of `org.apache.lucene.util.packed.BulkOperation`,
//! `BulkOperationPacked` (with its 24 unrolled `BulkOperationPacked1..24`
//! specialisations, which compute the same blocks as the generic loop) and
//! `BulkOperationPackedSingleBlock`: the `PackedInts.Decoder`/`Encoder` pair
//! that moves values in and out of whole blocks, as `long[]` words or as the
//! big-endian `byte[]` those words serialise to.
//!
//! `PACKED` stores values most-significant-bit first as one bitstream;
//! `PACKED_SINGLE_BLOCK` stores `64 / bpv` values per word, least significant
//! first. Byte forms of both are the words written big-endian
//! (`BulkOperation.writeLong`).

use super::Format;

/// One `BulkOperation`: a (format, bits-per-value) pair and its block
/// geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BulkOperation {
    format: Format,
    bits_per_value: u32,
    long_block_count: usize,
    long_value_count: usize,
    byte_block_count: usize,
    byte_value_count: usize,
    mask: u64,
}

impl BulkOperation {
    /// `BulkOperation.of`. Panics on an unsupported width, where Java's table
    /// lookup asserts/throws.
    pub fn of(format: Format, bits_per_value: u32) -> BulkOperation {
        assert!(
            format.is_supported(bits_per_value),
            "unsupported bitsPerValue {bits_per_value} for {format:?}"
        );
        let mask = if bits_per_value == 64 {
            u64::MAX
        } else {
            (1u64 << bits_per_value) - 1
        };
        match format {
            Format::Packed => {
                // Java: BulkOperationPacked's constructor.
                let mut blocks = bits_per_value as usize;
                while blocks & 1 == 0 {
                    blocks >>= 1;
                }
                let long_block_count = blocks;
                let long_value_count = 64 * long_block_count / bits_per_value as usize;
                let mut byte_block_count = 8 * long_block_count;
                let mut byte_value_count = long_value_count;
                while byte_block_count & 1 == 0 && byte_value_count & 1 == 0 {
                    byte_block_count >>= 1;
                    byte_value_count >>= 1;
                }
                BulkOperation {
                    format,
                    bits_per_value,
                    long_block_count,
                    long_value_count,
                    byte_block_count,
                    byte_value_count,
                    mask,
                }
            }
            Format::PackedSingleBlock => {
                let value_count = 64 / bits_per_value as usize;
                BulkOperation {
                    format,
                    bits_per_value,
                    long_block_count: 1,
                    long_value_count: value_count,
                    byte_block_count: 8,
                    byte_value_count: value_count,
                    mask,
                }
            }
        }
    }

    /// The format this operation encodes.
    pub fn format(&self) -> Format {
        self.format
    }

    /// Bits per value.
    pub fn bits_per_value(&self) -> u32 {
        self.bits_per_value
    }

    /// `longBlockCount()`: longs consumed per iteration.
    pub fn long_block_count(&self) -> usize {
        self.long_block_count
    }

    /// `longValueCount()`: values produced per long iteration.
    pub fn long_value_count(&self) -> usize {
        self.long_value_count
    }

    /// `byteBlockCount()`: bytes consumed per iteration.
    pub fn byte_block_count(&self) -> usize {
        self.byte_block_count
    }

    /// `byteValueCount()`: values produced per byte iteration.
    pub fn byte_value_count(&self) -> usize {
        self.byte_value_count
    }

    /// `computeIterations`: how many byte iterations fit `ram_budget` bytes
    /// (at least one, and no more than `value_count` needs).
    pub fn compute_iterations(&self, value_count: usize, ram_budget: usize) -> usize {
        let iterations = ram_budget / (self.byte_block_count + 8 * self.byte_value_count);
        if iterations == 0 {
            1
        } else if (iterations - 1) * self.byte_value_count >= value_count {
            // Java: (int) Math.ceil((double) valueCount / byteValueCount()).
            value_count.div_ceil(self.byte_value_count)
        } else {
            iterations
        }
    }

    /// `decode(long[], int, long[], int, int)`.
    pub fn decode_longs(&self, blocks: &[u64], values: &mut [i64], iterations: usize) {
        match self.format {
            Format::Packed => {
                let bpv = self.bits_per_value as i32;
                let mut bits_left: i32 = 64;
                let mut bo = 0usize;
                for v in values[..self.long_value_count * iterations].iter_mut() {
                    bits_left -= bpv;
                    if bits_left < 0 {
                        let high = blocks[bo] & ((1u64 << (bpv + bits_left)) - 1);
                        bo += 1;
                        // Java shifts mod 64: at 64 bits `high` is 0 and the shift by 64 is a
                        // shift by 0, which `wrapping_shl` reproduces.
                        *v = (high.wrapping_shl((-bits_left) as u32)
                            | (blocks[bo] >> (64 + bits_left))) as i64;
                        bits_left += 64;
                    } else {
                        *v = ((blocks[bo] >> bits_left) & self.mask) as i64;
                    }
                }
            }
            Format::PackedSingleBlock => {
                let mut vo = 0usize;
                for &block in &blocks[..iterations] {
                    vo = self.decode_single(block, values, vo);
                }
            }
        }
    }

    /// `decode(byte[], int, long[], int, int)`.
    pub fn decode_bytes(&self, blocks: &[u8], values: &mut [i64], iterations: usize) {
        match self.format {
            Format::Packed => {
                let bpv = self.bits_per_value;
                let mut next_value: u64 = 0;
                let mut bits_left = bpv;
                let mut vo = 0usize;
                for &b in &blocks[..iterations * self.byte_block_count] {
                    let bytes = b as u64;
                    if bits_left > 8 {
                        bits_left -= 8;
                        next_value |= bytes << bits_left;
                    } else {
                        let mut bits = 8 - bits_left;
                        values[vo] = (next_value | (bytes >> bits)) as i64;
                        vo += 1;
                        while bits >= bpv {
                            bits -= bpv;
                            values[vo] = ((bytes >> bits) & self.mask) as i64;
                            vo += 1;
                        }
                        bits_left = bpv - bits;
                        next_value = (bytes & ((1u64 << bits) - 1)).wrapping_shl(bits_left);
                    }
                }
                debug_assert_eq!(bits_left, bpv);
            }
            Format::PackedSingleBlock => {
                let mut vo = 0usize;
                for chunk in blocks[..iterations * 8].chunks_exact(8) {
                    let block = u64::from_be_bytes(chunk.try_into().expect("8 bytes"));
                    vo = self.decode_single(block, values, vo);
                }
            }
        }
    }

    /// `decode(long[], int, int[], int, int)`. `None` for a width above 32,
    /// where Java throws `UnsupportedOperationException`.
    pub fn decode_longs_to_ints(
        &self,
        blocks: &[u64],
        values: &mut [i32],
        iterations: usize,
    ) -> Option<()> {
        if self.bits_per_value > 32 {
            return None;
        }
        let mut tmp = vec![0i64; self.long_value_count * iterations];
        self.decode_longs(blocks, &mut tmp, iterations);
        for (d, s) in values.iter_mut().zip(tmp) {
            *d = s as i32;
        }
        Some(())
    }

    /// `decode(byte[], int, int[], int, int)`. `None` above 32 bits.
    pub fn decode_bytes_to_ints(
        &self,
        blocks: &[u8],
        values: &mut [i32],
        iterations: usize,
    ) -> Option<()> {
        if self.bits_per_value > 32 {
            return None;
        }
        let mut tmp = vec![0i64; self.byte_value_count * iterations];
        self.decode_bytes(blocks, &mut tmp, iterations);
        for (d, s) in values.iter_mut().zip(tmp) {
            *d = s as i32;
        }
        Some(())
    }

    /// `encode(long[], int, long[], int, int)`.
    pub fn encode_longs(&self, values: &[i64], blocks: &mut [u64], iterations: usize) {
        match self.format {
            Format::Packed => {
                let bpv = self.bits_per_value as i32;
                let mut next_block: u64 = 0;
                let mut bits_left: i32 = 64;
                let mut bo = 0usize;
                for &v in &values[..self.long_value_count * iterations] {
                    let v = v as u64;
                    bits_left -= bpv;
                    if bits_left > 0 {
                        next_block |= v << bits_left;
                    } else if bits_left == 0 {
                        next_block |= v;
                        blocks[bo] = next_block;
                        bo += 1;
                        next_block = 0;
                        bits_left = 64;
                    } else {
                        next_block |= v >> -bits_left;
                        blocks[bo] = next_block;
                        bo += 1;
                        next_block = (v & ((1u64 << -bits_left) - 1)) << (64 + bits_left);
                        bits_left += 64;
                    }
                }
            }
            Format::PackedSingleBlock => {
                for (i, block) in blocks[..iterations].iter_mut().enumerate() {
                    *block = self.encode_single(&values[i * self.long_value_count..]);
                }
            }
        }
    }

    /// `encode(long[], int, byte[], int, int)`.
    pub fn encode_bytes(&self, values: &[i64], blocks: &mut [u8], iterations: usize) {
        match self.format {
            Format::Packed => {
                let bpv = self.bits_per_value;
                let mut next_block: u32 = 0;
                let mut bits_left: u32 = 8;
                let mut bo = 0usize;
                for &v in &values[..self.byte_value_count * iterations] {
                    let v = v as u64;
                    if bpv < bits_left {
                        next_block |= (v << (bits_left - bpv)) as u32;
                        bits_left -= bpv;
                    } else {
                        let mut bits = bpv - bits_left;
                        blocks[bo] = (next_block as u64 | (v >> bits)) as u8;
                        bo += 1;
                        while bits >= 8 {
                            bits -= 8;
                            blocks[bo] = (v >> bits) as u8;
                            bo += 1;
                        }
                        bits_left = 8 - bits;
                        next_block = ((v & ((1u64 << bits) - 1)) << bits_left) as u32;
                    }
                }
                debug_assert_eq!(bits_left, 8);
            }
            Format::PackedSingleBlock => {
                for i in 0..iterations {
                    let block = self.encode_single(&values[i * self.long_value_count..]);
                    blocks[i * 8..i * 8 + 8].copy_from_slice(&block.to_be_bytes());
                }
            }
        }
    }

    /// `encode(int[], int, long[], int, int)`: `int` values are read unsigned,
    /// as Java masks them with `0xFFFFFFFFL`.
    pub fn encode_ints_to_longs(&self, values: &[i32], blocks: &mut [u64], iterations: usize) {
        let wide: Vec<i64> = values.iter().map(|&v| v as u32 as i64).collect();
        self.encode_longs(&wide, blocks, iterations);
    }

    /// `encode(int[], int, byte[], int, int)`.
    pub fn encode_ints_to_bytes(&self, values: &[i32], blocks: &mut [u8], iterations: usize) {
        let wide: Vec<i64> = values.iter().map(|&v| v as u32 as i64).collect();
        self.encode_bytes(&wide, blocks, iterations);
    }

    /// `BulkOperationPackedSingleBlock.decode(long, long[], int)`.
    #[inline]
    fn decode_single(&self, mut block: u64, values: &mut [i64], mut vo: usize) -> usize {
        values[vo] = (block & self.mask) as i64;
        vo += 1;
        for _ in 1..self.long_value_count {
            block >>= self.bits_per_value;
            values[vo] = (block & self.mask) as i64;
            vo += 1;
        }
        vo
    }

    /// `BulkOperationPackedSingleBlock.encode(long[], int)`.
    #[inline]
    fn encode_single(&self, values: &[i64]) -> u64 {
        let mut block = values[0] as u64;
        for (j, &v) in values[1..self.long_value_count].iter().enumerate() {
            block |= (v as u64) << ((j + 1) as u32 * self.bits_per_value);
        }
        block
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::packed64_single_block;

    fn mask(bpv: u32) -> u64 {
        if bpv == 64 {
            u64::MAX
        } else {
            (1u64 << bpv) - 1
        }
    }

    fn values(bpv: u32, n: usize, seed: u64) -> Vec<i64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s ^ (s >> 29)) & mask(bpv)) as i64
            })
            .collect()
    }

    #[test]
    fn packed_geometry_matches_java() {
        let op = BulkOperation::of(Format::Packed, 1);
        assert_eq!(
            (
                op.long_block_count(),
                op.long_value_count(),
                op.byte_block_count(),
                op.byte_value_count()
            ),
            (1, 64, 1, 8)
        );
        let op = BulkOperation::of(Format::Packed, 12);
        assert_eq!(
            (
                op.long_block_count(),
                op.long_value_count(),
                op.byte_block_count(),
                op.byte_value_count()
            ),
            (3, 16, 3, 2)
        );
        let op = BulkOperation::of(Format::Packed, 64);
        assert_eq!(
            (
                op.long_block_count(),
                op.long_value_count(),
                op.byte_block_count(),
                op.byte_value_count()
            ),
            (1, 1, 8, 1)
        );
        let op = BulkOperation::of(Format::PackedSingleBlock, 21);
        assert_eq!(
            (
                op.long_block_count(),
                op.long_value_count(),
                op.byte_block_count(),
                op.byte_value_count()
            ),
            (1, 3, 8, 3)
        );
        assert_eq!(op.format(), Format::PackedSingleBlock);
        assert_eq!(op.bits_per_value(), 21);
    }

    #[test]
    fn compute_iterations_matches_java() {
        let op = BulkOperation::of(Format::Packed, 12); // 3 bytes / 2 values
        assert_eq!(op.compute_iterations(100, 0), 1);
        // 1024 / (3 + 16) = 53 iterations; 52 * 2 >= 100 -> ceil(100/2) = 50.
        assert_eq!(op.compute_iterations(100, 1024), 50);
        assert_eq!(op.compute_iterations(10_000, 1024), 53);
    }

    #[test]
    #[should_panic(expected = "unsupported")]
    fn unsupported_single_block_width_panics() {
        BulkOperation::of(Format::PackedSingleBlock, 11);
    }

    #[test]
    fn long_and_byte_forms_agree_and_round_trip_for_every_width() {
        for format in [Format::Packed, Format::PackedSingleBlock] {
            for bpv in 1..=64u32 {
                if !format.is_supported(bpv) {
                    continue;
                }
                let op = BulkOperation::of(format, bpv);
                let iters = 3;
                let n = op.long_value_count() * iters;
                let vals = values(bpv, n, bpv as u64);
                let mut longs = vec![0u64; op.long_block_count() * iters];
                op.encode_longs(&vals, &mut longs, iters);
                let mut back = vec![0i64; n];
                op.decode_longs(&longs, &mut back, iters);
                assert_eq!(back, vals, "{format:?} bpv={bpv} longs");

                // The byte form is the long form written big-endian.
                let biters = n / op.byte_value_count();
                let mut bytes = vec![0u8; op.byte_block_count() * biters];
                op.encode_bytes(&vals, &mut bytes, biters);
                let be: Vec<u8> = longs.iter().flat_map(|w| w.to_be_bytes()).collect();
                assert_eq!(bytes, be, "{format:?} bpv={bpv} bytes");
                let mut back = vec![0i64; n];
                op.decode_bytes(&bytes, &mut back, biters);
                assert_eq!(back, vals, "{format:?} bpv={bpv} byte decode");

                if bpv <= 32 {
                    let ints: Vec<i32> = vals.iter().map(|&v| v as i32).collect();
                    let mut l2 = vec![0u64; longs.len()];
                    op.encode_ints_to_longs(&ints, &mut l2, iters);
                    assert_eq!(l2, longs);
                    let mut b2 = vec![0u8; bytes.len()];
                    op.encode_ints_to_bytes(&ints, &mut b2, biters);
                    assert_eq!(b2, bytes);
                    let mut i2 = vec![0i32; n];
                    op.decode_longs_to_ints(&longs, &mut i2, iters).unwrap();
                    assert_eq!(i2, ints);
                    let mut i3 = vec![0i32; n];
                    op.decode_bytes_to_ints(&bytes, &mut i3, biters).unwrap();
                    assert_eq!(i3, ints);
                } else {
                    let mut i2 = vec![0i32; n];
                    assert!(op.decode_longs_to_ints(&longs, &mut i2, iters).is_none());
                    assert!(op.decode_bytes_to_ints(&bytes, &mut i2, biters).is_none());
                }
            }
        }
        assert!(packed64_single_block::is_supported(32));
    }

    #[test]
    fn packed_is_msb_first_and_single_block_lsb_first() {
        let op = BulkOperation::of(Format::Packed, 4);
        let mut bytes = [0u8; 1];
        op.encode_bytes(&[0xA, 0xB], &mut bytes, 1);
        assert_eq!(bytes, [0xAB]);
        let op = BulkOperation::of(Format::PackedSingleBlock, 32);
        let mut longs = [0u64; 1];
        op.encode_longs(&[1, 2], &mut longs, 1);
        assert_eq!(longs, [(2u64 << 32) | 1]);
    }
}
