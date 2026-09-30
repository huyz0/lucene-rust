//! Port of `org.apache.lucene.util.packed.GrowableWriter`: a packed array
//! whose width grows on demand -- setting a value that does not fit copies the
//! array into a wider one.

use super::{
    copy, get_mutable, unsigned_bits_required, Mutable, PackedMutable, Reader, DEFAULT_BUFFER_SIZE,
};

/// `GrowableWriter`.
#[derive(Debug, Clone, PartialEq)]
pub struct GrowableWriter {
    current_mask: u64,
    current: PackedMutable,
    acceptable_overhead_ratio: f32,
}

/// `GrowableWriter.mask`.
fn mask(bits_per_value: u32) -> u64 {
    if bits_per_value == 64 {
        u64::MAX
    } else {
        super::max_value(bits_per_value) as u64
    }
}

impl GrowableWriter {
    /// `new GrowableWriter(startBitsPerValue, valueCount, acceptableOverheadRatio)`.
    pub fn new(
        start_bits_per_value: u32,
        value_count: usize,
        acceptable_overhead_ratio: f32,
    ) -> Self {
        let current = get_mutable(value_count, start_bits_per_value, acceptable_overhead_ratio);
        GrowableWriter {
            current_mask: mask(current.bits_per_value()),
            current,
            acceptable_overhead_ratio,
        }
    }

    /// `getMutable()`: the array currently backing this writer.
    pub fn mutable(&self) -> &PackedMutable {
        &self.current
    }

    /// `ensureCapacity`: widen so `value` fits.
    fn ensure_capacity(&mut self, value: i64) {
        let value = value as u64;
        if value & self.current_mask == value {
            return;
        }
        let bits_required = unsigned_bits_required(value);
        debug_assert!(bits_required > self.current.bits_per_value());
        let value_count = self.size();
        let mut next = get_mutable(value_count, bits_required, self.acceptable_overhead_ratio);
        copy(
            &self.current,
            0,
            &mut next,
            0,
            value_count,
            DEFAULT_BUFFER_SIZE,
        );
        self.current = next;
        self.current_mask = mask(self.current.bits_per_value());
    }

    /// `resize(newSize)`: a copy holding the first `min(size, new_size)`
    /// values, at the current width.
    pub fn resize(&self, new_size: usize) -> GrowableWriter {
        let mut next = GrowableWriter::new(
            self.bits_per_value(),
            new_size,
            self.acceptable_overhead_ratio,
        );
        let limit = self.size().min(new_size);
        copy(&self.current, 0, &mut next, 0, limit, DEFAULT_BUFFER_SIZE);
        next
    }
}

impl Reader for GrowableWriter {
    #[inline]
    fn get(&self, index: usize) -> i64 {
        self.current.get(index)
    }
    fn size(&self) -> usize {
        self.current.size()
    }
    fn get_bulk(&self, index: usize, arr: &mut [i64]) -> usize {
        self.current.get_bulk(index, arr)
    }
}

impl Mutable for GrowableWriter {
    fn bits_per_value(&self) -> u32 {
        self.current.bits_per_value()
    }

    fn set(&mut self, index: usize, value: i64) {
        self.ensure_capacity(value);
        self.current.set(index, value);
    }

    fn set_bulk(&mut self, index: usize, arr: &[i64]) -> usize {
        let max = arr.iter().fold(0i64, |m, &v| m | v);
        self.ensure_capacity(max);
        self.current.set_bulk(index, arr)
    }

    fn fill(&mut self, from: usize, to: usize, val: i64) {
        self.ensure_capacity(val);
        self.current.fill(from, to, val);
    }

    fn clear(&mut self) {
        Mutable::clear(&mut self.current);
    }

    fn ram_bytes_used(&self) -> usize {
        std::mem::size_of::<Self>() - std::mem::size_of::<PackedMutable>()
            + self.current.ram_bytes_used()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::{COMPACT, FASTEST};

    #[test]
    fn grows_only_when_a_value_does_not_fit() {
        let mut w = GrowableWriter::new(1, 100, COMPACT);
        assert_eq!(w.bits_per_value(), 1);
        w.set(0, 1);
        assert_eq!(w.bits_per_value(), 1);
        w.set(1, 5);
        assert_eq!(w.bits_per_value(), 3);
        w.set(2, 1 << 20);
        assert_eq!(w.bits_per_value(), 21);
        assert_eq!((w.get(0), w.get(1), w.get(2)), (1, 5, 1 << 20));
        w.set(3, -1);
        assert_eq!(w.bits_per_value(), 64);
        assert_eq!(w.get(3), -1);
        assert_eq!(w.get(2), 1 << 20);
        assert_eq!(w.mutable().bits_per_value(), 64);
    }

    #[test]
    fn fastest_rounds_to_byte_widths() {
        let mut w = GrowableWriter::new(1, 10, FASTEST);
        assert_eq!(w.bits_per_value(), 8);
        w.set(0, 300);
        assert_eq!(w.bits_per_value(), 16);
    }

    #[test]
    fn bulk_fill_resize_clear() {
        let mut w = GrowableWriter::new(2, 50, COMPACT);
        assert_eq!(w.set_bulk(10, &[1, 2, 1000]), 3);
        assert_eq!(w.bits_per_value(), 10);
        let mut out = [0i64; 3];
        assert_eq!(w.get_bulk(10, &mut out), 3);
        assert_eq!(out, [1, 2, 1000]);
        w.fill(20, 30, 4000);
        assert_eq!(w.bits_per_value(), 12);
        assert_eq!(w.get(25), 4000);
        let smaller = w.resize(15);
        assert_eq!(smaller.size(), 15);
        assert_eq!(smaller.get(12), 1000);
        assert_eq!(smaller.bits_per_value(), 12);
        let bigger = w.resize(80);
        assert_eq!(bigger.get(29), 4000);
        assert_eq!(bigger.get(79), 0);
        w.clear();
        assert_eq!(w.get(25), 0);
        assert!(w.ram_bytes_used() > 0);
    }
}
