//! Port of `org.apache.lucene.util.packed.PackedLongValues`,
//! `DeltaPackedLongValues` and `MonotonicLongValues` (and their `Builder`s):
//! an append-only buffer of `long`s compressed page by page.
//!
//! - *packed*: each page stores its values at the width of its maximum
//!   (64 bits if any is negative);
//! - *delta-packed*: each page stores `value - min(page)`;
//! - *monotonic*: each page stores `value - expected(i)` with
//!   `expected(i) = (long)(average * i)` for the page's average slope, then
//!   delta-packs that.
//!
//! Java's three-class hierarchy becomes one type and a [`Kind`]; each `get`
//! and `decodeBlock` override is a branch on it.

use super::{
    bits_required, check_block_size, get_mutable, Mutable, NullReader, PackedMutable, Reader,
    Result,
};

/// `PackedLongValues.DEFAULT_PAGE_SIZE`.
pub const DEFAULT_PAGE_SIZE: usize = 256;
/// `PackedLongValues.MIN_PAGE_SIZE`.
pub const MIN_PAGE_SIZE: usize = 64;
/// `PackedLongValues.MAX_PAGE_SIZE`.
pub const MAX_PAGE_SIZE: usize = 1 << 20;
/// `PackedLongValues.Builder.INITIAL_PAGE_COUNT`.
const INITIAL_PAGE_COUNT: usize = 16;

/// Which of the three Java classes a buffer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `PackedLongValues`.
    Packed,
    /// `DeltaPackedLongValues`.
    DeltaPacked,
    /// `MonotonicLongValues`.
    Monotonic,
}

/// `MonotonicBlockPackedReader.expected`: `origin + (long) (average * (long) index)`,
/// shared with the monotonic block-packed format.
#[inline]
pub fn expected(origin: i64, average: f32, index: i64) -> i64 {
    // Java's float -> long cast saturates and maps NaN to 0, as `as` does.
    origin.wrapping_add((average * index as f32) as i64)
}

/// One compressed page: `PackedInts.NullReader` when every value is 0, else
/// the `Mutable` `getMutable` picked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Page {
    Null(NullReader),
    Packed(PackedMutable),
}

impl Page {
    /// The page's width (0 for a `NullReader`).
    pub fn bits_per_value(&self) -> u32 {
        match self {
            Page::Null(_) => 0,
            Page::Packed(m) => m.bits_per_value(),
        }
    }

    fn ram_bytes_used(&self) -> usize {
        match self {
            Page::Null(_) => 0,
            Page::Packed(m) => m.ram_bytes_used(),
        }
    }
}

impl Reader for Page {
    #[inline]
    fn get(&self, index: usize) -> i64 {
        match self {
            Page::Null(n) => n.get(index),
            Page::Packed(m) => m.get(index),
        }
    }
    fn size(&self) -> usize {
        match self {
            Page::Null(n) => n.size(),
            Page::Packed(m) => m.size(),
        }
    }
    fn get_bulk(&self, index: usize, arr: &mut [i64]) -> usize {
        match self {
            Page::Null(n) => n.get_bulk(index, arr),
            Page::Packed(m) => m.get_bulk(index, arr),
        }
    }
}

/// `PackedLongValues` (and its delta/monotonic subclasses): the built,
/// read-only buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedLongValues {
    kind: Kind,
    values: Vec<Page>,
    /// Per-page minimum (delta and monotonic only; empty for packed).
    mins: Vec<i64>,
    /// Per-page average slope (monotonic only).
    averages: Vec<f32>,
    page_shift: u32,
    page_mask: usize,
    size: u64,
}

impl PackedLongValues {
    /// `PackedLongValues.packedBuilder(pageSize, acceptableOverheadRatio)`.
    pub fn packed_builder(
        page_size: usize,
        acceptable_overhead_ratio: f32,
    ) -> Result<PackedLongValuesBuilder> {
        PackedLongValuesBuilder::new(Kind::Packed, page_size, acceptable_overhead_ratio)
    }

    /// `PackedLongValues.deltaPackedBuilder(pageSize, acceptableOverheadRatio)`.
    pub fn delta_packed_builder(
        page_size: usize,
        acceptable_overhead_ratio: f32,
    ) -> Result<PackedLongValuesBuilder> {
        PackedLongValuesBuilder::new(Kind::DeltaPacked, page_size, acceptable_overhead_ratio)
    }

    /// `PackedLongValues.monotonicBuilder(pageSize, acceptableOverheadRatio)`.
    pub fn monotonic_builder(
        page_size: usize,
        acceptable_overhead_ratio: f32,
    ) -> Result<PackedLongValuesBuilder> {
        PackedLongValuesBuilder::new(Kind::Monotonic, page_size, acceptable_overhead_ratio)
    }

    /// Which of the three buffers this is.
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// `size()`.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The compressed pages (Java's `values`).
    pub fn pages(&self) -> &[Page] {
        &self.values
    }

    /// Per-page minimums (Java's `mins`; empty for a packed buffer).
    pub fn mins(&self) -> &[i64] {
        &self.mins
    }

    /// Per-page averages (Java's `averages`; empty unless monotonic).
    pub fn averages(&self) -> &[f32] {
        &self.averages
    }

    /// `get(int block, int element)`.
    #[inline]
    fn get_in(&self, block: usize, element: usize) -> i64 {
        let raw = self.values[block].get(element);
        match self.kind {
            Kind::Packed => raw,
            Kind::DeltaPacked => self.mins[block].wrapping_add(raw),
            Kind::Monotonic => {
                expected(self.mins[block], self.averages[block], element as i64).wrapping_add(raw)
            }
        }
    }

    /// `get(long)`.
    #[inline]
    pub fn get(&self, index: u64) -> i64 {
        debug_assert!(index < self.size);
        let block = (index >> self.page_shift) as usize;
        let element = index as usize & self.page_mask;
        self.get_in(block, element)
    }

    /// `decodeBlock(int, long[])`: every value of page `block` into `dest`.
    fn decode_block(&self, block: usize, dest: &mut [i64]) -> usize {
        let vals = &self.values[block];
        let size = vals.size();
        let mut k = 0;
        while k < size {
            k += vals.get_bulk(k, &mut dest[k..size]);
        }
        match self.kind {
            Kind::Packed => {}
            Kind::DeltaPacked | Kind::Monotonic => {
                let min = self.mins[block];
                for d in &mut dest[..size] {
                    *d = d.wrapping_add(min);
                }
                if self.kind == Kind::Monotonic {
                    let average = self.averages[block];
                    for (i, d) in dest[..size].iter_mut().enumerate() {
                        *d = d.wrapping_add(expected(0, average, i as i64));
                    }
                }
            }
        }
        size
    }

    /// `iterator()`.
    pub fn iter(&self) -> Iter<'_> {
        let cap = (self.size.min(self.page_mask as u64 + 1)) as usize;
        let mut it = Iter {
            owner: self,
            current_values: vec![0i64; cap],
            v_off: 0,
            p_off: 0,
            current_count: 0,
        };
        it.fill_block();
        it
    }

    /// Heap bytes (Java's `ramBytesUsed`, measured for Rust).
    pub fn ram_bytes_used(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.values.capacity() * std::mem::size_of::<Page>()
            + self.values.iter().map(Page::ram_bytes_used).sum::<usize>()
            + self.mins.capacity() * 8
            + self.averages.capacity() * 4
    }
}

/// `PackedLongValues.Iterator`: decodes a page at a time.
#[derive(Debug)]
pub struct Iter<'a> {
    owner: &'a PackedLongValues,
    current_values: Vec<i64>,
    v_off: usize,
    p_off: usize,
    current_count: usize,
}

impl Iter<'_> {
    fn fill_block(&mut self) {
        if self.v_off == self.owner.values.len() {
            self.current_count = 0;
        } else {
            self.current_count = self
                .owner
                .decode_block(self.v_off, &mut self.current_values);
            debug_assert!(self.current_count > 0);
        }
    }
}

impl Iterator for Iter<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<i64> {
        if self.p_off >= self.current_count {
            return None;
        }
        let result = self.current_values[self.p_off];
        self.p_off += 1;
        if self.p_off == self.current_count {
            self.v_off += 1;
            self.p_off = 0;
            self.fill_block();
        }
        Some(result)
    }
}

/// `PackedLongValues.Builder` (and the delta/monotonic builders).
#[derive(Debug, Clone)]
pub struct PackedLongValuesBuilder {
    kind: Kind,
    page_shift: u32,
    page_mask: usize,
    acceptable_overhead_ratio: f32,
    /// `None` once built (Java nulls `pending`).
    pending: Option<Vec<i64>>,
    size: u64,
    values: Vec<Page>,
    mins: Vec<i64>,
    averages: Vec<f32>,
    pending_off: usize,
}

impl PackedLongValuesBuilder {
    fn new(kind: Kind, page_size: usize, acceptable_overhead_ratio: f32) -> Result<Self> {
        let page_shift = check_block_size(page_size, MIN_PAGE_SIZE, MAX_PAGE_SIZE)?;
        Ok(PackedLongValuesBuilder {
            kind,
            page_shift,
            page_mask: page_size - 1,
            acceptable_overhead_ratio,
            pending: Some(vec![0i64; page_size]),
            size: 0,
            values: Vec::with_capacity(INITIAL_PAGE_COUNT),
            mins: Vec::new(),
            averages: Vec::new(),
            pending_off: 0,
        })
    }

    /// `size()`: values added so far.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// `build()`. Consumes the builder, so Java's "cannot be reused after
    /// build()" is a compile error.
    pub fn build(mut self) -> PackedLongValues {
        self.finish();
        PackedLongValues {
            kind: self.kind,
            values: self.values,
            mins: self.mins,
            averages: self.averages,
            page_shift: self.page_shift,
            page_mask: self.page_mask,
            size: self.size,
        }
    }

    /// `add(long)`.
    pub fn add(&mut self, l: i64) -> &mut Self {
        self.pack_if_full();
        let pending = self.pending.as_mut().expect("not built");
        pending[self.pending_off] = l;
        self.pending_off += 1;
        self.size += 1;
        self
    }

    /// `add(long value, int count)`: `count` copies of `value`.
    pub fn add_repeated(&mut self, value: i64, mut count: usize) -> &mut Self {
        while count > 0 {
            self.pack_if_full();
            let pending = self.pending.as_mut().expect("not built");
            let to_fill = count.min(pending.len() - self.pending_off);
            pending[self.pending_off..self.pending_off + to_fill].fill(value);
            self.pending_off += to_fill;
            count -= to_fill;
            self.size += to_fill as u64;
        }
        self
    }

    /// `add(LongValuesCursor)`'s role: append every value of `values`.
    pub fn add_all(&mut self, values: &[i64]) -> &mut Self {
        let mut rest = values;
        while !rest.is_empty() {
            self.pack_if_full();
            let pending = self.pending.as_mut().expect("not built");
            let to_fill = rest.len().min(pending.len() - self.pending_off);
            pending[self.pending_off..self.pending_off + to_fill].copy_from_slice(&rest[..to_fill]);
            self.pending_off += to_fill;
            self.size += to_fill as u64;
            rest = &rest[to_fill..];
        }
        self
    }

    fn pack_if_full(&mut self) {
        let full = self.pending.as_ref().map(Vec::len) == Some(self.pending_off);
        if full {
            self.pack();
        }
    }

    /// `finish()`.
    fn finish(&mut self) {
        if self.pending_off > 0 {
            self.pack();
        }
    }

    fn pack(&mut self) {
        let mut pending = self.pending.take().expect("not built");
        let n = self.pending_off;
        self.pack_values(&mut pending[..n]);
        self.pending = Some(pending);
        self.pending_off = 0;
    }

    /// The three `pack(long[], int, int, float)` overrides, outermost first.
    fn pack_values(&mut self, values: &mut [i64]) {
        debug_assert!(!values.is_empty());
        if self.kind == Kind::Monotonic {
            // MonotonicLongValues.Builder.pack
            let n = values.len();
            let average = if n == 1 {
                0.0
            } else {
                values[n - 1].wrapping_sub(values[0]) as f32 / (n - 1) as f32
            };
            for (i, v) in values.iter_mut().enumerate() {
                *v = v.wrapping_sub(expected(0, average, i as i64));
            }
            self.averages.push(average);
        }
        if self.kind != Kind::Packed {
            // DeltaPackedLongValues.Builder.pack
            let min = values.iter().copied().min().expect("non-empty");
            for v in values.iter_mut() {
                *v = v.wrapping_sub(min);
            }
            self.mins.push(min);
        }
        // PackedLongValues.Builder.pack
        let (mut min_value, mut max_value) = (values[0], values[0]);
        for &v in &values[1..] {
            min_value = min_value.min(v);
            max_value = max_value.max(v);
        }
        let page = if min_value == 0 && max_value == 0 {
            Page::Null(NullReader::for_count(values.len()))
        } else {
            let bits = if min_value < 0 {
                64
            } else {
                bits_required(max_value).expect("max_value >= min_value >= 0")
            };
            let mut mutable = get_mutable(values.len(), bits, self.acceptable_overhead_ratio);
            let mut i = 0;
            while i < values.len() {
                i += mutable.set_bulk(i, &values[i..]);
            }
            Page::Packed(mutable)
        };
        self.values.push(page);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::{COMPACT, DEFAULT, FASTEST};

    fn check(values: &[i64], kind: Kind, page_size: usize, ratio: f32) -> PackedLongValues {
        let mut b = PackedLongValuesBuilder::new(kind, page_size, ratio).unwrap();
        for &v in values {
            b.add(v);
        }
        assert_eq!(b.size(), values.len() as u64);
        let built = b.build();
        assert_eq!(built.size(), values.len() as u64);
        assert_eq!(built.kind(), kind);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(built.get(i as u64), v, "{kind:?} i={i}");
        }
        assert_eq!(built.iter().collect::<Vec<_>>(), values);
        built
    }

    #[test]
    fn all_kinds_round_trip_awkward_inputs() {
        let monotonic: Vec<i64> = (0..1000).map(|i| i * 37 + (i % 5)).collect();
        let random: Vec<i64> = (0..1000)
            .map(|i: i64| i.wrapping_mul(0x9E3779B97F4A7C15u64 as i64) >> (i % 60))
            .collect();
        let extremes = vec![i64::MIN, i64::MAX, 0, -1, 1, i64::MIN, i64::MAX];
        for kind in [Kind::Packed, Kind::DeltaPacked, Kind::Monotonic] {
            for page in [64usize, 256, 1024] {
                for ratio in [COMPACT, DEFAULT, FASTEST] {
                    check(&monotonic, kind, page, ratio);
                    check(&random, kind, page, ratio);
                    check(&extremes, kind, page, ratio);
                    check(&[], kind, page, ratio);
                    check(&[5], kind, page, ratio);
                }
            }
        }
    }

    #[test]
    fn zero_pages_are_null_readers_and_widths_follow_the_kind() {
        let mut values = vec![0i64; 64];
        values.extend((0..64).map(|i| 1000 + i));
        let p = check(&values, Kind::Packed, 64, COMPACT);
        assert_eq!(p.pages()[0], Page::Null(NullReader::for_count(64)));
        assert_eq!(p.pages()[1].bits_per_value(), 11); // 1063 < 2048
        assert!(p.mins().is_empty());
        let d = check(&values, Kind::DeltaPacked, 64, COMPACT);
        assert_eq!(d.pages()[1].bits_per_value(), 6); // 0..=63
        assert_eq!(d.mins(), &[0, 1000]);
        let m = check(&values, Kind::Monotonic, 64, COMPACT);
        assert_eq!(m.pages()[1], Page::Null(NullReader::for_count(64)));
        assert_eq!(m.averages(), &[0.0, 1.0]);
        assert!(m.ram_bytes_used() > 0);
        let negative = check(&[-5, 3], Kind::Packed, 64, COMPACT);
        assert_eq!(negative.pages()[0].bits_per_value(), 64);
    }

    #[test]
    fn repeated_and_slice_adds_span_pages() {
        let mut b = PackedLongValues::delta_packed_builder(64, DEFAULT).unwrap();
        b.add_repeated(7, 150)
            .add_all(&(0..100).collect::<Vec<_>>())
            .add(-3);
        let built = b.build();
        assert_eq!(built.size(), 251);
        assert_eq!(built.get(149), 7);
        assert_eq!(built.get(150), 0);
        assert_eq!(built.get(249), 99);
        assert_eq!(built.get(250), -3);
        assert_eq!(built.pages().len(), 4);
        assert!(PackedLongValues::packed_builder(32, DEFAULT).is_err());
        assert!(PackedLongValues::monotonic_builder(MAX_PAGE_SIZE * 2, DEFAULT).is_err());
        assert!(PackedLongValues::monotonic_builder(DEFAULT_PAGE_SIZE, DEFAULT).is_ok());
    }

    #[test]
    fn expected_matches_java_float_semantics() {
        assert_eq!(expected(10, 2.5, 3), 17);
        assert_eq!(expected(0, f32::NAN, 3), 0);
        assert_eq!(expected(0, f32::INFINITY, 3), i64::MAX);
        assert_eq!(expected(0, -0.4, 1), 0);
    }
}
