//! Port of `org.apache.lucene.util.packed.AbstractPagedMutable`,
//! `PagedMutable` and `PagedGrowableWriter`: a `long`-indexed packed array
//! split into power-of-two pages, each page its own `PackedInts.Mutable`.
//!
//! Java's `AbstractPagedMutable<T>` is generic over its subclass and asks it
//! for `newMutable`/`newUnfilledCopy`; here [`AbstractPagedMutable`] is generic
//! over a [`PageFactory`] that plays that role.

use super::{
    check_block_size, copy_with_buffer, fastest_format_and_bits, get_mutable_with_format,
    num_blocks, Format, GrowableWriter, Mutable, PackedMutable, Reader, Result,
};

/// `AbstractPagedMutable.MIN_BLOCK_SIZE`.
pub const MIN_BLOCK_SIZE: usize = 1 << 6;
/// `AbstractPagedMutable.MAX_BLOCK_SIZE`.
pub const MAX_BLOCK_SIZE: usize = 1 << 30;

/// What a subclass of `AbstractPagedMutable` supplies: how to make a page.
pub trait PageFactory: Clone {
    /// The page type.
    type Page: Mutable;
    /// `newMutable(valueCount, bitsPerValue)`. `array_bits_per_value` is the
    /// paged array's own width (Java's `this.bitsPerValue`), which
    /// `PagedMutable` uses in place of the requested one.
    fn new_page(
        &self,
        value_count: usize,
        bits_per_value: u32,
        array_bits_per_value: u32,
    ) -> Self::Page;
}

/// `PagedMutable`'s page factory: a fixed format and width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedPages {
    format: Format,
}

impl PageFactory for FixedPages {
    type Page = PackedMutable;
    fn new_page(
        &self,
        value_count: usize,
        bits_per_value: u32,
        array_bits_per_value: u32,
    ) -> PackedMutable {
        debug_assert!(array_bits_per_value >= bits_per_value);
        get_mutable_with_format(value_count, array_bits_per_value, self.format)
    }
}

/// `PagedGrowableWriter`'s page factory: a [`GrowableWriter`] per page.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrowablePages {
    acceptable_overhead_ratio: f32,
}

impl PageFactory for GrowablePages {
    type Page = GrowableWriter;
    fn new_page(
        &self,
        value_count: usize,
        bits_per_value: u32,
        _array_bits_per_value: u32,
    ) -> GrowableWriter {
        GrowableWriter::new(bits_per_value, value_count, self.acceptable_overhead_ratio)
    }
}

/// `AbstractPagedMutable`.
#[derive(Debug, Clone)]
pub struct AbstractPagedMutable<F: PageFactory> {
    size: u64,
    page_shift: u32,
    page_mask: usize,
    sub_mutables: Vec<F::Page>,
    bits_per_value: u32,
    factory: F,
}

/// `PagedMutable`: pages of one fixed width.
pub type PagedMutable = AbstractPagedMutable<FixedPages>;
/// `PagedGrowableWriter`: pages that each grow independently.
pub type PagedGrowableWriter = AbstractPagedMutable<GrowablePages>;

impl PagedMutable {
    /// `new PagedMutable(size, pageSize, bitsPerValue, acceptableOverheadRatio)`.
    pub fn new(
        size: u64,
        page_size: usize,
        bits_per_value: u32,
        acceptable_overhead_ratio: f32,
    ) -> Result<Self> {
        let fb =
            fastest_format_and_bits(Some(page_size), bits_per_value, acceptable_overhead_ratio);
        AbstractPagedMutable::with_factory(
            fb.bits_per_value,
            size,
            page_size,
            FixedPages { format: fb.format },
            true,
        )
    }

    /// The page format (Java's `format` field).
    pub fn format(&self) -> Format {
        self.factory.format
    }
}

impl PagedGrowableWriter {
    /// `new PagedGrowableWriter(size, pageSize, startBitsPerValue, acceptableOverheadRatio)`.
    pub fn new(
        size: u64,
        page_size: usize,
        start_bits_per_value: u32,
        acceptable_overhead_ratio: f32,
    ) -> Result<Self> {
        AbstractPagedMutable::with_factory(
            start_bits_per_value,
            size,
            page_size,
            GrowablePages {
                acceptable_overhead_ratio,
            },
            true,
        )
    }
}

impl<F: PageFactory> AbstractPagedMutable<F> {
    fn with_factory(
        bits_per_value: u32,
        size: u64,
        page_size: usize,
        factory: F,
        fill: bool,
    ) -> Result<Self> {
        let page_shift = check_block_size(page_size, MIN_BLOCK_SIZE, MAX_BLOCK_SIZE)?;
        let num_pages = num_blocks(size, page_size)?;
        let mut this = AbstractPagedMutable {
            size,
            page_shift,
            page_mask: page_size - 1,
            sub_mutables: Vec::with_capacity(num_pages),
            bits_per_value,
            factory,
        };
        if fill {
            this.fill_pages(num_pages);
        }
        Ok(this)
    }

    /// `fillPages()`.
    fn fill_pages(&mut self, num_pages: usize) {
        for i in 0..num_pages {
            let value_count = if i == num_pages - 1 {
                self.last_page_size(self.size)
            } else {
                self.page_size()
            };
            let page = self
                .factory
                .new_page(value_count, self.bits_per_value, self.bits_per_value);
            self.sub_mutables.push(page);
        }
    }

    /// `lastPageSize(size)`.
    fn last_page_size(&self, size: u64) -> usize {
        let sz = self.index_in_page(size);
        if sz == 0 {
            self.page_size()
        } else {
            sz
        }
    }

    /// `pageSize()`.
    pub fn page_size(&self) -> usize {
        self.page_mask + 1
    }

    /// `size()`.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The array's starting width (Java's `bitsPerValue`).
    pub fn bits_per_value(&self) -> u32 {
        self.bits_per_value
    }

    /// The pages (Java's `subMutables`).
    pub fn pages(&self) -> &[F::Page] {
        &self.sub_mutables
    }

    #[inline]
    fn page_index(&self, index: u64) -> usize {
        (index >> self.page_shift) as usize
    }

    #[inline]
    fn index_in_page(&self, index: u64) -> usize {
        index as usize & self.page_mask
    }

    /// `get(long)`.
    #[inline]
    pub fn get(&self, index: u64) -> i64 {
        debug_assert!(index < self.size, "index={index} size={}", self.size);
        self.sub_mutables[self.page_index(index)].get(self.index_in_page(index))
    }

    /// `set(long, long)`.
    #[inline]
    pub fn set(&mut self, index: u64, value: i64) {
        debug_assert!(index < self.size);
        let p = self.page_index(index);
        let i = self.index_in_page(index);
        self.sub_mutables[p].set(i, value);
    }

    /// Heap bytes (Java's `ramBytesUsed`, measured for Rust).
    pub fn ram_bytes_used(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.sub_mutables.capacity() * std::mem::size_of::<F::Page>()
            + self
                .sub_mutables
                .iter()
                .map(|m| m.ram_bytes_used())
                .sum::<usize>()
            - self.sub_mutables.len() * std::mem::size_of::<F::Page>()
    }

    /// `resize(newSize)`: a copy of the first `min(size, new_size)` values;
    /// a page shared with this array keeps its (possibly grown) width.
    pub fn resize(&self, new_size: u64) -> Result<Self> {
        let mut copy = AbstractPagedMutable::with_factory(
            self.bits_per_value,
            new_size,
            self.page_size(),
            self.factory.clone(),
            false,
        )?;
        let copy_pages = num_blocks(new_size, self.page_size())?;
        let num_common_pages = copy_pages.min(self.sub_mutables.len());
        let mut copy_buffer = vec![0i64; 1024];
        for i in 0..copy_pages {
            let value_count = if i == copy_pages - 1 {
                self.last_page_size(new_size)
            } else {
                self.page_size()
            };
            let bpv = if i < num_common_pages {
                self.sub_mutables[i].bits_per_value()
            } else {
                self.bits_per_value
            };
            let mut page = self.factory.new_page(value_count, bpv, self.bits_per_value);
            if i < num_common_pages {
                let copy_length = value_count.min(self.sub_mutables[i].size());
                copy_with_buffer(
                    &self.sub_mutables[i],
                    0,
                    &mut page,
                    0,
                    copy_length,
                    &mut copy_buffer,
                );
            }
            copy.sub_mutables.push(page);
        }
        Ok(copy)
    }

    /// `grow(minSize)`: this array if it already holds `min_size` values,
    /// else a copy about 1/8 larger than asked.
    pub fn grow(self, min_size: u64) -> Result<Self> {
        if min_size <= self.size {
            return Ok(self);
        }
        let extra = (min_size >> 3).max(3);
        self.resize(min_size + extra)
    }

    /// `grow()`: room for one more value.
    pub fn grow_by_one(self) -> Result<Self> {
        let min = self.size + 1;
        self.grow(min)
    }
}

impl<F: PageFactory> std::fmt::Display for AbstractPagedMutable<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(size={},pageSize={})", self.size, self.page_size())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop, clippy::identity_op)]

    use super::*;
    use crate::packed::{COMPACT, DEFAULT, FASTEST};

    #[test]
    fn paged_mutable_set_get_across_pages() {
        let mut p = PagedMutable::new(1000, 64, 10, COMPACT).unwrap();
        assert_eq!(p.pages().len(), 16);
        assert_eq!(p.pages()[15].size(), 1000 - 15 * 64);
        for i in 0..1000u64 {
            p.set(i, (i * 7 % 1024) as i64);
        }
        for i in 0..1000u64 {
            assert_eq!(p.get(i), (i * 7 % 1024) as i64);
        }
        assert_eq!(p.format(), Format::Packed);
        assert_eq!(p.bits_per_value(), 10);
        assert!(p.ram_bytes_used() > 0);
        assert_eq!(p.to_string(), "(size=1000,pageSize=64)");

        let r = p.resize(100).unwrap();
        assert_eq!(r.size(), 100);
        assert_eq!(r.get(99), (99 * 7 % 1024) as i64);
        let g = r.grow(101).unwrap();
        assert_eq!(g.size(), 101 + 12);
        assert_eq!(g.get(99), (99 * 7 % 1024) as i64);
        assert_eq!(g.get(112), 0);
        let same = g.grow(5).unwrap();
        assert_eq!(same.size(), 113);
        let one = same.grow_by_one().unwrap();
        assert_eq!(one.size(), 114 + 14);
        let small = PagedMutable::new(10, 64, 3, COMPACT)
            .unwrap()
            .grow(11)
            .unwrap();
        assert_eq!(small.size(), 14);
    }

    #[test]
    fn fastest_rounds_page_width() {
        let p = PagedMutable::new(10, 64, 3, FASTEST).unwrap();
        assert_eq!(p.bits_per_value(), 8);
        assert_eq!(p.pages()[0].bits_per_value(), 8);
    }

    #[test]
    fn exact_multiple_of_page_size_has_a_full_last_page() {
        let p = PagedMutable::new(128, 64, 3, COMPACT).unwrap();
        assert_eq!(p.pages().len(), 2);
        assert_eq!(p.pages()[1].size(), 64);
        let empty = PagedMutable::new(0, 64, 3, COMPACT).unwrap();
        assert!(empty.pages().is_empty());
    }

    #[test]
    fn bad_page_sizes_are_rejected() {
        assert!(PagedMutable::new(10, 32, 3, COMPACT).is_err());
        assert!(PagedMutable::new(10, 100, 3, COMPACT).is_err());
        assert!(PagedGrowableWriter::new(10, 1 << 31, 3, COMPACT).is_err());
    }

    #[test]
    fn growable_pages_grow_independently_and_keep_width_on_resize() {
        let mut p = PagedGrowableWriter::new(300, 128, 1, DEFAULT).unwrap();
        p.set(5, 1);
        p.set(200, 1 << 30);
        assert_eq!(p.pages()[0].bits_per_value(), 1);
        assert_eq!(p.pages()[1].bits_per_value(), 32);
        let r = p.resize(1000).unwrap();
        assert_eq!(r.pages()[1].bits_per_value(), 32);
        assert_eq!(r.pages()[5].bits_per_value(), 1);
        assert_eq!(r.get(200), 1 << 30);
        assert_eq!(r.get(5), 1);
        assert_eq!(r.get(999), 0);
    }
}
