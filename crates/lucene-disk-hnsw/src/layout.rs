//! Where each level-0 node's record sits in the `.dhn` file.
//!
//! A record is, little-endian:
//!
//! | bytes | holds |
//! |---|---|
//! | 4 | the node's document (`i32`) |
//! | 2 | its neighbour count (`u16`) |
//! | 2 | zero |
//! | `vector_bytes` | its vector in the field's `VectorEncoding` |
//! | `4 * max_neighbors` | its neighbours (`u32` node ids; slots past the count are zero) |
//! | `code_len * max_neighbors` | with inline codes, each neighbour's quantized code in neighbour order |
//!
//! padded to a multiple of four bytes. Records are packed whole into pages
//! of `page_len` bytes (the rest of each page is zero) so that none crosses
//! a page boundary; a record longer than a page starts on a page boundary
//! and takes whole pages. The field's region starts on a page boundary of
//! the file, so one record is always one aligned read of at most
//! [`NodeLayout::pages_per_record`] pages.

/// Bytes before a record's vector: document, neighbour count, padding.
pub const RECORD_HEADER: usize = 8;

/// The default page: the common block size of local disks and the unit a
/// page cache fetches and keeps.
pub const DEFAULT_PAGE_LEN: usize = 4096;

/// The geometry of a field's records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeLayout {
    page_len: usize,
    record_len: usize,
    vector_bytes: usize,
    max_neighbors: usize,
    inline_code_len: usize,
    records_per_page: usize,
    pages_per_record: usize,
}

impl NodeLayout {
    /// The layout of records holding `vector_bytes` of vector, up to
    /// `max_neighbors` neighbours and, when `inline_code_len > 0`, that many
    /// code bytes per neighbour, in pages of `page_len` bytes. `None` when
    /// `page_len` is zero or not a multiple of four, or a size overflows.
    pub fn new(
        page_len: usize,
        vector_bytes: usize,
        max_neighbors: usize,
        inline_code_len: usize,
    ) -> Option<Self> {
        if page_len == 0 || !page_len.is_multiple_of(4) {
            return None;
        }
        let record_len = max_neighbors
            .checked_mul(4)?
            .checked_add(max_neighbors.checked_mul(inline_code_len)?)?
            .checked_add(vector_bytes)?
            .checked_add(RECORD_HEADER)?
            .checked_next_multiple_of(4)?;
        let (records_per_page, pages_per_record) = if record_len <= page_len {
            (page_len.checked_div(record_len)?, 1)
        } else {
            (1, record_len.div_ceil(page_len))
        };
        page_len.checked_mul(pages_per_record)?;
        Some(NodeLayout {
            page_len,
            record_len,
            vector_bytes,
            max_neighbors,
            inline_code_len,
            records_per_page,
            pages_per_record,
        })
    }

    /// Bytes of a page.
    pub fn page_len(&self) -> usize {
        self.page_len
    }

    /// Bytes of a record.
    pub fn record_len(&self) -> usize {
        self.record_len
    }

    /// Bytes of a stored vector.
    pub fn vector_bytes(&self) -> usize {
        self.vector_bytes
    }

    /// The most neighbours a record holds.
    pub fn max_neighbors(&self) -> usize {
        self.max_neighbors
    }

    /// Bytes of each inline neighbour code, zero without inline codes.
    pub fn inline_code_len(&self) -> usize {
        self.inline_code_len
    }

    /// Records in a page (1 when a record takes several pages).
    pub fn records_per_page(&self) -> usize {
        self.records_per_page
    }

    /// Pages a record takes (1 unless a record is longer than a page).
    pub fn pages_per_record(&self) -> usize {
        self.pages_per_record
    }

    /// Bytes from the region's start to `node`'s record, `None` on overflow.
    pub fn record_offset(&self, node: usize) -> Option<usize> {
        let stride = self.page_len.checked_mul(self.pages_per_record)?;
        let page = node.checked_div(self.records_per_page)?;
        let slot = node.checked_rem(self.records_per_page)?;
        page.checked_mul(stride)?
            .checked_add(slot.checked_mul(self.record_len)?)
    }

    /// Bytes of the region holding `count` records, whole pages, `None` on
    /// overflow.
    pub fn region_len(&self, count: usize) -> Option<usize> {
        count
            .div_ceil(self.records_per_page)
            .checked_mul(self.page_len)?
            .checked_mul(self.pages_per_record)
    }

    /// Where the vector starts in a record.
    pub fn vector_start(&self) -> usize {
        RECORD_HEADER
    }

    /// Where the neighbour list starts in a record.
    // ARITH: `new` checked the whole record length, which this is a prefix of.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn neighbors_start(&self) -> usize {
        RECORD_HEADER + self.vector_bytes
    }

    /// Where the inline codes start in a record.
    // ARITH: as for `neighbors_start`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn codes_start(&self) -> usize {
        self.neighbors_start() + 4 * self.max_neighbors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_records_pack_into_pages() {
        // 100 f32 dims, 32 neighbours: 8 + 400 + 128 = 536 bytes.
        let l = NodeLayout::new(4096, 400, 32, 0).unwrap();
        assert_eq!(l.record_len(), 536);
        assert_eq!(l.records_per_page(), 7);
        assert_eq!(l.pages_per_record(), 1);
        assert_eq!(l.record_offset(0), Some(0));
        assert_eq!(l.record_offset(6), Some(6 * 536));
        assert_eq!(l.record_offset(7), Some(4096));
        assert_eq!(l.region_len(0), Some(0));
        assert_eq!(l.region_len(7), Some(4096));
        assert_eq!(l.region_len(8), Some(8192));
        assert_eq!(l.vector_start(), 8);
        assert_eq!(l.neighbors_start(), 408);
        assert_eq!(l.codes_start(), 536);
        for node in 0..100 {
            let at = l.record_offset(node).unwrap();
            assert_eq!(at / 4096, (at + l.record_len() - 1) / 4096, "node {node}");
        }
    }

    #[test]
    fn large_records_take_whole_pages() {
        // 300 f32 dims and 32 inline codes of 54 bytes.
        let l = NodeLayout::new(4096, 1200, 32, 54).unwrap();
        assert_eq!(l.record_len(), 8 + 1200 + 128 + 32 * 54);
        assert_eq!(l.records_per_page(), 1);
        assert_eq!(l.pages_per_record(), 1);
        let big = NodeLayout::new(4096, 3072 * 4, 64, 0).unwrap();
        assert_eq!(big.pages_per_record(), 4);
        assert_eq!(big.record_offset(3), Some(3 * 4 * 4096));
        assert_eq!(big.region_len(2), Some(8 * 4096));
        assert_eq!(l.inline_code_len(), 54);
        assert_eq!(l.max_neighbors(), 32);
        assert_eq!(l.vector_bytes(), 1200);
        assert_eq!(l.page_len(), 4096);
    }

    #[test]
    fn records_are_padded_to_four_bytes() {
        let l = NodeLayout::new(64, 3, 1, 1).unwrap();
        assert_eq!(l.record_len(), 16);
        assert_eq!(l.records_per_page(), 4);
    }

    #[test]
    fn bad_geometry_is_refused() {
        assert_eq!(NodeLayout::new(0, 4, 1, 0), None);
        assert_eq!(NodeLayout::new(4095, 4, 1, 0), None);
        assert_eq!(NodeLayout::new(4096, usize::MAX, 1, 0), None);
        assert_eq!(NodeLayout::new(4096, 4, usize::MAX, 0), None);
        let l = NodeLayout::new(4096, 4, 1, 0).unwrap();
        assert_eq!(l.record_offset(usize::MAX), None);
        assert_eq!(l.region_len(usize::MAX), None);
    }
}
