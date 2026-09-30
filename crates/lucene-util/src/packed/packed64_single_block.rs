//! Port of `org.apache.lucene.util.packed.Packed64SingleBlock`: values that
//! never span two words. Each 64-bit word holds `64 / bpv` values, least
//! significant first; the high `64 % bpv` bits are wasted.
//!
//! Java generates one subclass per supported width (`Packed64SingleBlock1`
//! ... `Packed64SingleBlock32`) so each `get`/`set` shifts by constants; here
//! one type precomputes the per-block value count, and the arithmetic is the
//! same.

use super::bulk_operation::BulkOperation;
use super::{Format, Mutable, Reader};

/// `Packed64SingleBlock.MAX_SUPPORTED_BITS_PER_VALUE`.
pub const MAX_SUPPORTED_BITS_PER_VALUE: u32 = 32;
const SUPPORTED_BITS_PER_VALUE: [u32; 14] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 21, 32];

/// `Packed64SingleBlock.isSupported`.
pub fn is_supported(bits_per_value: u32) -> bool {
    SUPPORTED_BITS_PER_VALUE
        .binary_search(&bits_per_value)
        .is_ok()
}

/// `Packed64SingleBlock`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packed64SingleBlock {
    blocks: Vec<u64>,
    value_count: usize,
    bits_per_value: u32,
    values_per_block: usize,
    mask: u64,
}

impl Packed64SingleBlock {
    /// `Packed64SingleBlock.create`. Panics on an unsupported width, as Java
    /// throws `IllegalArgumentException`.
    pub fn new(value_count: usize, bits_per_value: u32) -> Self {
        assert!(
            is_supported(bits_per_value),
            "Unsupported number of bits per value: {bits_per_value}"
        );
        let values_per_block = 64 / bits_per_value as usize;
        Packed64SingleBlock {
            blocks: vec![0u64; value_count.div_ceil(values_per_block)],
            value_count,
            bits_per_value,
            values_per_block,
            mask: (1u64 << bits_per_value) - 1,
        }
    }

    /// The backing words (Java's `blocks`).
    pub fn blocks(&self) -> &[u64] {
        &self.blocks
    }

    #[inline]
    fn position(&self, index: usize) -> (usize, u32) {
        let o = index / self.values_per_block;
        let b = index % self.values_per_block;
        (o, b as u32 * self.bits_per_value)
    }
}

impl Reader for Packed64SingleBlock {
    #[inline]
    fn get(&self, index: usize) -> i64 {
        debug_assert!(index < self.value_count);
        let (o, shift) = self.position(index);
        ((self.blocks[o] >> shift) & self.mask) as i64
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
        let vpb = self.values_per_block;

        // go to the next block boundary
        let offset_in_block = index % vpb;
        if offset_in_block != 0 {
            let mut i = offset_in_block;
            while i < vpb && len > 0 {
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
        let decoder = BulkOperation::of(Format::PackedSingleBlock, self.bits_per_value);
        let block_index = index / vpb;
        let nblocks = (index + len) / vpb - block_index;
        decoder.decode_longs(&self.blocks[block_index..], &mut arr[off..], nblocks);
        let diff = nblocks * vpb;
        index += diff;
        len -= diff;

        if index > original_index {
            index - original_index
        } else {
            let gets = len.min(self.value_count - index);
            for (o, slot) in arr[off..off + gets].iter_mut().enumerate() {
                *slot = self.get(index + o);
            }
            gets
        }
    }
}

impl Mutable for Packed64SingleBlock {
    fn bits_per_value(&self) -> u32 {
        self.bits_per_value
    }

    #[inline]
    fn set(&mut self, index: usize, value: i64) {
        debug_assert!(index < self.value_count);
        let (o, shift) = self.position(index);
        self.blocks[o] = (self.blocks[o] & !(self.mask << shift)) | ((value as u64) << shift);
    }

    fn set_bulk(&mut self, mut index: usize, arr: &[i64]) -> usize {
        debug_assert!(!arr.is_empty());
        debug_assert!(index < self.value_count);
        let mut len = arr.len().min(self.value_count - index);
        let mut off = 0usize;
        let original_index = index;
        let vpb = self.values_per_block;

        let offset_in_block = index % vpb;
        if offset_in_block != 0 {
            let mut i = offset_in_block;
            while i < vpb && len > 0 {
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

        let op = BulkOperation::of(Format::PackedSingleBlock, self.bits_per_value);
        let block_index = index / vpb;
        let nblocks = (index + len) / vpb - block_index;
        op.encode_longs(&arr[off..], &mut self.blocks[block_index..], nblocks);
        let diff = nblocks * vpb;
        index += diff;
        len -= diff;

        if index > original_index {
            index - original_index
        } else {
            super::default_set_bulk(self, index, &arr[off..off + len])
        }
    }

    fn fill(&mut self, mut from: usize, to: usize, val: i64) {
        debug_assert!(from <= to);
        debug_assert!(super::unsigned_bits_required(val as u64) <= self.bits_per_value);
        let vpb = self.values_per_block;
        if to - from <= vpb << 1 {
            // there needs to be at least one full block to set for the block
            // approach to be worth trying
            super::default_fill(self, from, to, val);
            return;
        }

        // set values naively until the next block start
        let from_offset_in_block = from % vpb;
        if from_offset_in_block != 0 {
            for _ in from_offset_in_block..vpb {
                self.set(from, val);
                from += 1;
            }
        }

        // bulk set of the inner blocks
        let from_block = from / vpb;
        let to_block = to / vpb;
        let mut block_value = 0u64;
        for i in 0..vpb {
            block_value |= (val as u64) << (i as u32 * self.bits_per_value);
        }
        self.blocks[from_block..to_block].fill(block_value);

        // fill the gap
        for i in vpb * to_block..to {
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

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop, clippy::identity_op)]

    use super::*;

    #[test]
    fn supported_widths() {
        let supported: Vec<u32> = (0..=64).filter(|&b| is_supported(b)).collect();
        assert_eq!(supported, SUPPORTED_BITS_PER_VALUE.to_vec());
        assert_eq!(MAX_SUPPORTED_BITS_PER_VALUE, 32);
    }

    #[test]
    #[should_panic(expected = "Unsupported number of bits per value")]
    fn unsupported_width_panics() {
        Packed64SingleBlock::new(10, 11);
    }

    #[test]
    fn set_get_bulk_fill_every_supported_width() {
        for &bpv in &SUPPORTED_BITS_PER_VALUE {
            let n = 333;
            let mask = (1i64 << bpv) - 1;
            let model: Vec<i64> = (0..n).map(|i| (i as i64 * 2654435761) & mask).collect();
            let mut p = Packed64SingleBlock::new(n, bpv);
            for (i, &v) in model.iter().enumerate() {
                p.set(i, v);
            }
            for start in [0usize, 1, 7, 64] {
                let mut out = vec![0i64; n];
                let mut idx = start;
                while idx < n {
                    idx += p.get_bulk(idx, &mut out[idx..]);
                }
                assert_eq!(&out[start..], &model[start..], "bpv={bpv}");
            }
            let mut q = Packed64SingleBlock::new(n, bpv);
            let mut idx = 3;
            while idx < n {
                idx += q.set_bulk(idx, &model[idx..]);
            }
            assert_eq!(q.get(2), 0);
            for i in 3..n {
                assert_eq!(q.get(i), model[i]);
            }
            for (from, to) in [(0usize, 3usize), (5, 300), (0, 333), (64, 65)] {
                let mut f = Packed64SingleBlock::new(n, bpv);
                f.fill(from, to, mask);
                for i in 0..n {
                    assert_eq!(f.get(i), if (from..to).contains(&i) { mask } else { 0 });
                }
            }
            Mutable::clear(&mut q);
            assert!(q.blocks().iter().all(|&b| b == 0));
            assert!(q.ram_bytes_used() > 0);
            assert_eq!(q.size(), n);
            assert_eq!(q.bits_per_value(), bpv);
        }
    }

    #[test]
    fn layout_is_lsb_first_within_a_word() {
        let mut p = Packed64SingleBlock::new(4, 21);
        p.set(0, 1);
        p.set(1, 2);
        p.set(2, 3);
        p.set(3, 4);
        assert_eq!(p.blocks()[0], 1 | (2 << 21) | (3 << 42));
        assert_eq!(p.blocks()[1], 4);
    }
}
