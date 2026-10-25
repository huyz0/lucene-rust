//! Lucene's asserting live-docs adapter. Checks are always enabled in Rust.
//!
//! The concrete codec returns an owned dense live-docs view. Java's nullable
//! Bits return and negative lengths are unrepresentable; signed access helpers
//! retain its negative-index/offset assertions. Sparse storage selection,
//! IOContext and Codec/IndexWriter routing remain with the shared delegate.
use std::io;

use lucene_search::reader::DocIdSetIterator;
use lucene_store::directory::Directory;
use lucene_util::bits::{Bits, NO_MORE_DOCS};
use lucene_util::fixed_bit_set::FixedBitSet;

use super::cranky_live_docs::{LiveDocsFormat, LiveDocsInfo};

/// Random access wrapper; mask operations go directly to the raw delegate.
pub struct AssertingBits<B> {
    delegate: B,
}

impl<B: Bits> AssertingBits<B> {
    pub fn new(delegate: B) -> Self {
        Self { delegate }
    }

    pub fn get_signed(&self, index: i32) -> bool {
        assert!(index >= 0);
        self.get(index as usize)
    }

    pub fn apply_mask_signed(&self, mask: &mut FixedBitSet, offset: i32) {
        assert!(offset >= 0);
        self.apply_mask(mask, offset as usize);
    }
}

impl<B: Bits> Bits for AssertingBits<B> {
    fn get(&self, index: usize) -> bool {
        assert!(
            index < self.delegate.length(),
            "index={index} vs in.length()={}",
            self.delegate.length()
        );
        self.delegate.get(index)
    }

    fn length(&self) -> usize {
        self.delegate.length()
    }

    fn apply_mask(&self, mask: &mut FixedBitSet, offset: usize) {
        self.delegate.apply_mask(mask, offset);
    }
}

/// The LiveDocs surface available through this port's iterator interface.
/// Java's docIDRunEnd/intoBitSet optimizations have no equivalent here.
pub trait LiveDocs: Bits {
    fn deleted_count(&self) -> i32;
    fn live_docs_iterator(&self) -> Box<dyn DocIdSetIterator + '_>;
    fn deleted_docs_iterator(&self) -> Box<dyn DocIdSetIterator + '_>;
}

/// Preserves both delegate iterator objects and validates its deletion count.
pub struct AssertingLiveDocs<B> {
    bits: AssertingBits<B>,
}

impl<B: LiveDocs> AssertingLiveDocs<B> {
    pub fn new(delegate: B) -> Self {
        Self {
            bits: AssertingBits::new(delegate),
        }
    }
}

impl<B: LiveDocs> Bits for AssertingLiveDocs<B> {
    fn get(&self, index: usize) -> bool {
        // FBS: raw Bits get requires index < the delegate bitset length; the asserting view checks it.
        self.bits.get(index)
    }
    fn length(&self) -> usize {
        self.bits.length()
    }
    fn apply_mask(&self, mask: &mut FixedBitSet, offset: usize) {
        self.bits.apply_mask(mask, offset);
    }
}

impl<B: LiveDocs> LiveDocs for AssertingLiveDocs<B> {
    fn deleted_count(&self) -> i32 {
        let count = self.bits.delegate.deleted_count();
        assert!(count >= 0);
        assert!((count as usize) <= self.length());
        count
    }
    fn live_docs_iterator(&self) -> Box<dyn DocIdSetIterator + '_> {
        self.bits.delegate.live_docs_iterator()
    }
    fn deleted_docs_iterator(&self) -> Box<dyn DocIdSetIterator + '_> {
        self.bits.delegate.deleted_docs_iterator()
    }
}

/// Dense LiveDocs view of the shared codec's owned bitset.
pub struct DenseLiveDocs {
    bits: FixedBitSet,
    deleted: i32,
}

impl DenseLiveDocs {
    pub fn new(bits: FixedBitSet) -> Self {
        let deleted =
            i32::try_from(bits.len() - bits.cardinality()).expect("too many deleted docs");
        assert!(bits.len() < NO_MORE_DOCS as usize);
        Self { bits, deleted }
    }
}

impl Bits for DenseLiveDocs {
    fn get(&self, index: usize) -> bool {
        // FBS: raw Bits get requires index < the delegate bitset length; the asserting view checks it.
        self.bits.get(index)
    }
    fn length(&self) -> usize {
        self.bits.len()
    }
    fn apply_mask(&self, mask: &mut FixedBitSet, offset: usize) {
        // Java Bits.applyMask calls nextSetBit(0), which asserts for an empty
        // mask. Preserve that observed behavior for the concrete Java view.
        assert!(!mask.is_empty());
        // Use the Bits default: unlike FixedBitSet's inherent bulk helper,
        // Java DenseLiveDocs checks each set mask bit and asserts at the
        // first out-of-range access after preserving earlier mutations.
        Bits::apply_mask(&self.bits, mask, offset);
    }
}

struct DenseIterator<'a> {
    bits: &'a FixedBitSet,
    live: bool,
    doc: i32,
    cost: i64,
}

impl DocIdSetIterator for DenseIterator<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> lucene_search::Result<i32> {
        self.advance(self.doc.saturating_add(1))
    }
    fn advance(&mut self, target: i32) -> lucene_search::Result<i32> {
        assert!(target >= 0);
        let target = target as usize;
        let next = if target >= self.bits.len() {
            None
        } else if self.live {
            self.bits.next_set_bit(target)
        } else {
            self.bits.next_clear_bit(target)
        };
        self.doc = next.map_or(NO_MORE_DOCS, |doc| doc as i32);
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.cost
    }
}

impl LiveDocs for DenseLiveDocs {
    fn deleted_count(&self) -> i32 {
        self.deleted
    }
    fn live_docs_iterator(&self) -> Box<dyn DocIdSetIterator + '_> {
        Box::new(DenseIterator {
            bits: &self.bits,
            live: true,
            doc: -1,
            cost: (self.bits.len() - self.deleted as usize) as i64,
        })
    }
    fn deleted_docs_iterator(&self) -> Box<dyn DocIdSetIterator + '_> {
        Box::new(DenseIterator {
            bits: &self.bits,
            live: false,
            doc: -1,
            cost: self.deleted as i64,
        })
    }
}

/// Asserts length/count before writing and after a successful delegate read.
/// Its inherent read method retains the checked view instead of flattening it.
pub struct AssertingLiveDocsFormat<D> {
    delegate: D,
}

impl<D: LiveDocsFormat> AssertingLiveDocsFormat<D> {
    pub fn new(delegate: D) -> Self {
        Self { delegate }
    }

    // Java: AssertingLiveDocsFormat.check.
    fn check(bits: &FixedBitSet, max_doc: usize, expected: usize) {
        assert_eq!(bits.len(), max_doc);
        // Stage 2: the 4096-bit scalar guard measured 4089.012 ns in Rust
        // versus 2433.006 ns in Java (0.60x). Format I/O uses concrete,
        // immutable FixedBitSet: get has no callbacks. Range popcounts mask
        // padding, so even dirty from_words input in release counts only the
        // logical bits that Java scans. Generic wrappers still delegate.
        let deleted = bits.len() - bits.cardinality_range(0, bits.len());
        assert_eq!(
            deleted, expected,
            "deleted: {deleted} != expected: {expected}"
        );
    }

    /// Java: AssertingLiveDocsFormat.readLiveDocs.
    pub fn read_live_docs(
        &self,
        dir: &dyn Directory,
        info: &LiveDocsInfo<'_>,
    ) -> io::Result<AssertingLiveDocs<DenseLiveDocs>> {
        let bits = self.delegate.read_live_docs(dir, info)?;
        Self::check(&bits, info.max_doc, info.del_count);
        Ok(AssertingLiveDocs::new(DenseLiveDocs::new(bits)))
    }

    /// Java: AssertingLiveDocsFormat.writeLiveDocs.
    pub fn write_live_docs(
        &self,
        bits: &FixedBitSet,
        dir: &dyn Directory,
        info: &LiveDocsInfo<'_>,
        new_del_count: usize,
    ) -> io::Result<()> {
        let expected = info
            .del_count
            .checked_add(new_del_count)
            .expect("deleted count overflow");
        Self::check(bits, info.max_doc, expected);
        self.delegate
            .write_live_docs(bits, dir, info, new_del_count)
    }

    /// Java: AssertingLiveDocsFormat.files.
    pub fn files(&self, info: &LiveDocsInfo<'_>, files: &mut Vec<String>) -> io::Result<()> {
        self.delegate.files(info, files)
    }
}

impl<B: std::fmt::Display> std::fmt::Display for AssertingBits<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Asserting({})", self.delegate)
    }
}

impl<B: std::fmt::Display> std::fmt::Display for AssertingLiveDocs<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.bits.fmt(f)
    }
}

impl<D: std::fmt::Display> std::fmt::Display for AssertingLiveDocsFormat<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Asserting({})", self.delegate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Named;
    impl std::fmt::Display for Named {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("delegate")
        }
    }

    // from_words checks ghost bits only with debug assertions. Exercise its
    // representable release-only dirty padding against the original scan.
    #[cfg(not(debug_assertions))]
    #[test]
    fn format_check_ignores_dirty_padding_like_java_bit_scan() {
        use super::super::cranky_live_docs::Lucene90LiveDocsFormat;
        for length in [1usize, 63, 65, 130] {
            let bits = FixedBitSet::from_words(
                vec![u64::MAX; lucene_util::fixed_bit_set::bits2words(length)],
                length,
            );
            let scalar_deleted = (0..bits.len()).filter(|&index| !bits.get(index)).count();
            assert_eq!(scalar_deleted, 0);
            AssertingLiveDocsFormat::<Lucene90LiveDocsFormat>::check(&bits, length, scalar_deleted);
        }
    }
    #[test]
    fn display_preserves_delegate_description() {
        assert_eq!(
            AssertingBits { delegate: Named }.to_string(),
            "Asserting(delegate)"
        );
        assert_eq!(
            AssertingLiveDocs {
                bits: AssertingBits { delegate: Named }
            }
            .to_string(),
            "Asserting(delegate)"
        );
        let format = AssertingLiveDocsFormat { delegate: Named };
        assert_eq!(format.to_string(), "Asserting(delegate)");
    }
}
