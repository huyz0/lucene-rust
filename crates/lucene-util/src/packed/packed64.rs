//! Port of `org.apache.lucene.util.packed.Packed64`: `PACKED` values in a
//! `long[]`, most significant bit first, a value spanning at most two words.

use super::bulk_operation::BulkOperation;
use super::{Format, Mutable, Reader, VERSION_CURRENT};

const BLOCK_SIZE: i64 = 64;
const BLOCK_BITS: u32 = 6;
const MOD_MASK: u64 = 63;

/// `Packed64`: a space-optimal packed array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packed64 {
    blocks: Vec<u64>,
    value_count: usize,
    bits_per_value: u32,
    /// Java `maskRight`: the low `bits_per_value` bits.
    mask_right: u64,
    /// Java `bpvMinusBlockSize`.
    bpv_minus_block_size: i64,
}

impl Packed64 {
    /// `new Packed64(valueCount, bitsPerValue)`, all zeros.
    pub fn new(value_count: usize, bits_per_value: u32) -> Self {
        assert!(
            (1..=64).contains(&bits_per_value),
            "bitsPerValue={bits_per_value}"
        );
        let long_count = Format::Packed.long_count(VERSION_CURRENT, value_count, bits_per_value);
        let shift = 64 - bits_per_value;
        Packed64 {
            blocks: vec![0u64; long_count],
            value_count,
            bits_per_value,
            mask_right: (u64::MAX << shift) >> shift,
            bpv_minus_block_size: bits_per_value as i64 - BLOCK_SIZE,
        }
    }

    /// The backing words (Java's `blocks`).
    pub fn blocks(&self) -> &[u64] {
        &self.blocks
    }

    fn position(&self, index: usize) -> (usize, i64) {
        let major_bit_pos = index as u64 * self.bits_per_value as u64;
        let element_pos = (major_bit_pos >> BLOCK_BITS) as usize;
        let end_bits = (major_bit_pos & MOD_MASK) as i64 + self.bpv_minus_block_size;
        (element_pos, end_bits)
    }
}

impl Reader for Packed64 {
    #[inline]
    fn get(&self, index: usize) -> i64 {
        debug_assert!(index < self.value_count);
        let (e, end_bits) = self.position(index);
        if end_bits <= 0 {
            return ((self.blocks[e] >> (-end_bits) as u32) & self.mask_right) as i64;
        }
        (((self.blocks[e] << end_bits as u32)
            | (self.blocks[e + 1] >> (BLOCK_SIZE - end_bits) as u32))
            & self.mask_right) as i64
    }

    fn size(&self) -> usize {
        self.value_count
    }

    fn get_bulk(&self, mut index: usize, arr: &mut [i64]) -> usize {
        debug_assert!(!arr.is_empty());
        debug_assert!(index < self.value_count);
        let mut len = arr.len().min(self.value_count - index);
        let mut off = 0usize;
        let original_index = index;
        let decoder = BulkOperation::of(Format::Packed, self.bits_per_value);

        // go to the next block where the value does not span across two blocks
        let offset_in_blocks = index % decoder.long_value_count();
        if offset_in_blocks != 0 {
            let mut i = offset_in_blocks;
            while i < decoder.long_value_count() && len > 0 {
                arr[off] = self.get(index);
                off += 1;
                index += 1;
                len -= 1;
                i += 1;
            }
            if len == 0 {
                return index - original_index;
            }
        }

        // bulk get
        debug_assert_eq!(index % decoder.long_value_count(), 0);
        let block_index = ((index as u64 * self.bits_per_value as u64) >> BLOCK_BITS) as usize;
        let iterations = len / decoder.long_value_count();
        decoder.decode_longs(&self.blocks[block_index..], &mut arr[off..], iterations);
        let got_values = iterations * decoder.long_value_count();
        index += got_values;
        len -= got_values;

        if index > original_index {
            // stay at the block boundary
            index - original_index
        } else {
            // no progress so far => already at a block boundary but no full block to get
            let gets = len.min(self.value_count - index);
            for (o, slot) in arr[off..off + gets].iter_mut().enumerate() {
                *slot = self.get(index + o);
            }
            gets
        }
    }
}

impl Mutable for Packed64 {
    fn bits_per_value(&self) -> u32 {
        self.bits_per_value
    }

    #[inline]
    fn set(&mut self, index: usize, value: i64) {
        debug_assert!(index < self.value_count);
        let value = value as u64;
        let (e, end_bits) = self.position(index);
        if end_bits <= 0 {
            let shift = (-end_bits) as u32;
            self.blocks[e] = (self.blocks[e] & !(self.mask_right << shift)) | (value << shift);
            return;
        }
        let end = end_bits as u32;
        self.blocks[e] = (self.blocks[e] & !(self.mask_right >> end)) | (value >> end);
        self.blocks[e + 1] =
            (self.blocks[e + 1] & (u64::MAX >> end)) | (value << (BLOCK_SIZE as u32 - end));
    }

    fn set_bulk(&mut self, mut index: usize, arr: &[i64]) -> usize {
        debug_assert!(!arr.is_empty());
        debug_assert!(index < self.value_count);
        let mut len = arr.len().min(self.value_count - index);
        let mut off = 0usize;
        let original_index = index;
        let encoder = BulkOperation::of(Format::Packed, self.bits_per_value);

        // go to the next block where the value does not span across two blocks
        let offset_in_blocks = index % encoder.long_value_count();
        if offset_in_blocks != 0 {
            let mut i = offset_in_blocks;
            while i < encoder.long_value_count() && len > 0 {
                self.set(index, arr[off]);
                index += 1;
                off += 1;
                len -= 1;
                i += 1;
            }
            if len == 0 {
                return index - original_index;
            }
        }

        // bulk set
        let block_index = ((index as u64 * self.bits_per_value as u64) >> BLOCK_BITS) as usize;
        let iterations = len / encoder.long_value_count();
        encoder.encode_longs(&arr[off..], &mut self.blocks[block_index..], iterations);
        let set_values = iterations * encoder.long_value_count();
        index += set_values;
        len -= set_values;

        if index > original_index {
            index - original_index
        } else {
            super::default_set_bulk(self, index, &arr[off..off + len])
        }
    }

    fn fill(&mut self, mut from: usize, to: usize, val: i64) {
        debug_assert!(super::unsigned_bits_required(val as u64) <= self.bits_per_value);
        debug_assert!(from <= to);
        let bpv = self.bits_per_value as usize;

        // minimum number of values that use an exact number of full blocks
        let n_aligned_values = 64 / gcd(64, bpv);
        let span = to - from;
        if span <= 3 * n_aligned_values {
            // there needs be at least 2 * nAlignedValues aligned values for the
            // block approach to be worth trying
            super::default_fill(self, from, to, val);
            return;
        }

        // fill the first values naively until the next block start
        let from_mod = from % n_aligned_values;
        if from_mod != 0 {
            for _ in from_mod..n_aligned_values {
                self.set(from, val);
                from += 1;
            }
        }
        debug_assert_eq!(from % n_aligned_values, 0);

        // compute the long[] blocks for nAlignedValues consecutive values and
        // use them to set as many values as possible without applying any mask
        // or shift
        let n_aligned_blocks = (n_aligned_values * bpv) >> 6;
        let aligned_blocks = {
            let mut values = Packed64::new(n_aligned_values, self.bits_per_value);
            for i in 0..n_aligned_values {
                values.set(i, val);
            }
            values.blocks
        };
        let start_block = ((from as u64 * bpv as u64) >> 6) as usize;
        let end_block = ((to as u64 * bpv as u64) >> 6) as usize;
        for block in start_block..end_block {
            self.blocks[block] = aligned_blocks[block % n_aligned_blocks];
        }

        // fill the gap
        let gap_start = ((end_block as u64) << 6) / bpv as u64;
        for i in gap_start as usize..to {
            self.set(i, val);
        }
    }

    fn clear(&mut self) {
        self.blocks.fill(0);
    }

    fn ram_bytes_used(&self) -> usize {
        std::mem::size_of::<Self>() + self.blocks.capacity() * 8
    }
}

/// `Packed64.gcd`.
fn gcd(a: usize, b: usize) -> usize {
    if a < b {
        gcd(b, a)
    } else if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop, clippy::identity_op)]

    use super::*;
    use crate::packed::max_value;

    fn rng(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed ^ (*seed >> 31)
    }

    #[test]
    fn set_get_every_width_against_a_model() {
        let mut seed = 42u64;
        for bpv in 1..=64u32 {
            let n = 211;
            let mut p = Packed64::new(n, bpv);
            let mask = max_value(bpv) as u64 | if bpv == 64 { u64::MAX } else { 0 };
            let mut model = vec![0i64; n];
            for _ in 0..600 {
                let i = (rng(&mut seed) % n as u64) as usize;
                let v = (rng(&mut seed) & mask) as i64;
                p.set(i, v);
                model[i] = v;
            }
            for (i, &m) in model.iter().enumerate() {
                assert_eq!(p.get(i), m, "bpv={bpv} i={i}");
            }
            // bulk get from every offset agrees with get
            for start in [0usize, 1, 63, 64, 100] {
                let mut out = vec![0i64; n];
                let mut idx = start;
                while idx < n {
                    let got = p.get_bulk(idx, &mut out[idx..]);
                    assert!(got > 0);
                    idx += got;
                }
                assert_eq!(
                    &out[start..],
                    &model[start..],
                    "bulk bpv={bpv} start={start}"
                );
            }
            // bulk set from every offset agrees with set
            let mut q = Packed64::new(n, bpv);
            let mut idx = 5;
            while idx < n {
                let got = q.set_bulk(idx, &model[idx..]);
                assert!(got > 0);
                idx += got;
            }
            for i in 5..n {
                assert_eq!(q.get(i), model[i]);
            }
            assert_eq!(q.get(0), 0);
            assert!(q.ram_bytes_used() > 0);
            assert_eq!(q.bits_per_value(), bpv);
        }
    }

    #[test]
    fn fill_matches_set_for_long_and_short_spans() {
        for bpv in [1u32, 3, 7, 8, 12, 21, 33, 63, 64] {
            let val = (max_value(bpv) / 3).max(1);
            for (from, to) in [(0usize, 5usize), (3, 500), (64, 1000), (17, 17), (1, 999)] {
                let mut p = Packed64::new(1000, bpv);
                p.fill(from, to, val);
                for i in 0..1000 {
                    let want = if (from..to).contains(&i) { val } else { 0 };
                    assert_eq!(p.get(i), want, "bpv={bpv} {from}..{to} i={i}");
                }
            }
        }
        let mut p = Packed64::new(10, 5);
        p.fill(0, 10, 3);
        Mutable::clear(&mut p);
        assert!(p.blocks().iter().all(|&b| b == 0));
        assert_eq!(p.size(), 10);
    }

    #[test]
    fn value_spanning_two_words_lands_in_both() {
        let mut p = Packed64::new(3, 40);
        p.set(1, (1i64 << 40) - 1);
        // value 1 occupies bits 40..80: low 24 bits of word 0, high 16 of word 1.
        assert_eq!(p.blocks()[0], (1u64 << 24) - 1);
        assert_eq!(p.blocks()[1], 0xFFFF_u64 << 48);
        assert_eq!(p.get(1), (1i64 << 40) - 1);
        assert_eq!(gcd(12, 64), 4);
    }
}
