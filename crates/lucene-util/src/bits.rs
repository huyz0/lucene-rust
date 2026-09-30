//! Ports of `org.apache.lucene.util.Bits` (with `MatchAllBits`/`MatchNoBits`
//! and `applyMask`), the abstract `org.apache.lucene.util.BitSet` over the
//! dense [`FixedBitSet`] and the sparse [`SparseFixedBitSet`] (with
//! `BitSet.of`'s density choice), and `org.apache.lucene.util.NotDocIdSet`.
//!
//! A `DocIdSetIterator` is a Rust `Iterator<Item = i32>` in this port (see
//! `lucene_search::docid_set`), so `BitSet.or(DocIdSetIterator)` takes one and
//! `NotDocIdSet.iterator()` is an iterator adapter with Java's `advance`.

use crate::fixed_bit_set::FixedBitSet;
use crate::sparse_fixed_bit_set::SparseFixedBitSet;

/// `DocIdSetIterator.NO_MORE_DOCS`.
pub const NO_MORE_DOCS: i32 = i32::MAX;

/// `org.apache.lucene.util.Bits`: random access to a set of bits.
pub trait Bits {
    /// `get(index)`.
    fn get(&self, index: usize) -> bool;
    /// `length()`.
    fn length(&self) -> usize;
    /// `Bits.applyMask(bitSet, offset)`: clears every bit `i` of `bit_set`
    /// for which `get(offset + i)` is false.
    fn apply_mask(&self, bit_set: &mut FixedBitSet, offset: usize) {
        let mut i = bit_set.next_set_bit(0);
        while let Some(b) = i {
            if !self.get(offset + b) {
                // FBS: `b` came from `bit_set.next_set_bit`, so `b < bit_set.len()`.
                bit_set.clear(b);
            }
            i = if b + 1 >= bit_set.len() {
                None
            } else {
                bit_set.next_set_bit(b + 1)
            };
        }
    }
}

/// `Bits.MatchAllBits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchAllBits(pub usize);

impl Bits for MatchAllBits {
    fn get(&self, _index: usize) -> bool {
        true
    }
    fn length(&self) -> usize {
        self.0
    }
}

/// `Bits.MatchNoBits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchNoBits(pub usize);

impl Bits for MatchNoBits {
    fn get(&self, _index: usize) -> bool {
        false
    }
    fn length(&self) -> usize {
        self.0
    }
}

impl Bits for FixedBitSet {
    fn get(&self, index: usize) -> bool {
        // FBS: the trait's contract is Java's -- `index < length()` -- and
        // `FixedBitSet::get` enforces it (`self.len()` is `length()`).
        FixedBitSet::get(self, index)
    }
    fn length(&self) -> usize {
        self.len()
    }
}

impl Bits for SparseFixedBitSet {
    fn get(&self, index: usize) -> bool {
        SparseFixedBitSet::get(self, index)
    }
    fn length(&self) -> usize {
        self.len()
    }
}

/// The abstract `org.apache.lucene.util.BitSet`. `None` stands for Java's
/// `-1`/`NO_MORE_DOCS` sentinels.
pub trait BitSet: Bits {
    /// `set(i)`.
    fn set(&mut self, i: usize);
    /// `getAndSet(i)`.
    fn get_and_set(&mut self, i: usize) -> bool;
    /// `clear(i)`.
    fn clear(&mut self, i: usize);
    /// `clear(startIndex, endIndex)`.
    fn clear_range(&mut self, start: usize, end: usize);
    /// `clear()`: every bit.
    fn clear_all(&mut self) {
        let n = self.length();
        self.clear_range(0, n);
    }
    /// `cardinality()`.
    fn cardinality(&self) -> usize;
    /// `approximateCardinality()`.
    fn approximate_cardinality(&self) -> usize;
    /// `prevSetBit(index)`.
    fn prev_set_bit(&self, index: usize) -> Option<usize>;
    /// `nextSetBit(start, end)`.
    fn next_set_bit_in_range(&self, start: usize, end: usize) -> Option<usize>;
    /// `nextSetBit(index)`.
    fn next_set_bit(&self, index: usize) -> Option<usize> {
        self.next_set_bit_in_range(index, self.length())
    }
    /// `nextClearBit(start, upperBound)`.
    fn next_clear_bit_in_range(&self, start: usize, upper_bound: usize) -> Option<usize>;
    /// `nextClearBit(index)`.
    fn next_clear_bit(&self, index: usize) -> Option<usize> {
        self.next_clear_bit_in_range(index, self.length())
    }
    /// `or(DocIdSetIterator)`: sets every doc the (unpositioned) iterator
    /// returns.
    fn or_iter(&mut self, docs: &mut dyn Iterator<Item = i32>) {
        for d in docs {
            if d == NO_MORE_DOCS {
                break;
            }
            self.set(d as usize);
        }
    }
}

impl BitSet for FixedBitSet {
    // FBS (every method below): delegations to `FixedBitSet`'s own methods,
    // which check `index < self.len()` themselves.
    fn set(&mut self, i: usize) {
        FixedBitSet::set(self, i);
    }
    fn get_and_set(&mut self, i: usize) -> bool {
        FixedBitSet::get_and_set(self, i)
    }
    fn clear(&mut self, i: usize) {
        FixedBitSet::clear(self, i);
    }
    fn clear_range(&mut self, start: usize, end: usize) {
        FixedBitSet::clear_range(self, start, end);
    }
    fn cardinality(&self) -> usize {
        FixedBitSet::cardinality(self)
    }
    fn approximate_cardinality(&self) -> usize {
        FixedBitSet::approximate_cardinality(self)
    }
    fn prev_set_bit(&self, index: usize) -> Option<usize> {
        FixedBitSet::prev_set_bit(self, index)
    }
    fn next_set_bit_in_range(&self, start: usize, end: usize) -> Option<usize> {
        FixedBitSet::next_set_bit_in_range(self, start, end)
    }
    fn next_clear_bit_in_range(&self, start: usize, upper_bound: usize) -> Option<usize> {
        FixedBitSet::next_clear_bit_in_range(self, start, upper_bound)
    }
}

impl BitSet for SparseFixedBitSet {
    fn set(&mut self, i: usize) {
        SparseFixedBitSet::set(self, i);
    }
    fn get_and_set(&mut self, i: usize) -> bool {
        SparseFixedBitSet::get_and_set(self, i)
    }
    fn clear(&mut self, i: usize) {
        SparseFixedBitSet::clear(self, i);
    }
    fn clear_range(&mut self, start: usize, end: usize) {
        SparseFixedBitSet::clear_range(self, start, end);
    }
    fn cardinality(&self) -> usize {
        SparseFixedBitSet::cardinality(self)
    }
    fn approximate_cardinality(&self) -> usize {
        SparseFixedBitSet::approximate_cardinality(self)
    }
    fn prev_set_bit(&self, index: usize) -> Option<usize> {
        SparseFixedBitSet::prev_set_bit(self, index)
    }
    fn next_set_bit_in_range(&self, start: usize, end: usize) -> Option<usize> {
        SparseFixedBitSet::next_set_bit_in_range(self, start, end)
    }
    fn next_clear_bit_in_range(&self, start: usize, upper_bound: usize) -> Option<usize> {
        SparseFixedBitSet::next_clear_bit_in_range(self, start, upper_bound)
    }
}

/// What `BitSet.of` returns: a dense or a sparse bit set.
#[derive(Debug, Clone)]
pub enum AnyBitSet {
    /// `FixedBitSet`.
    Dense(FixedBitSet),
    /// `SparseFixedBitSet`.
    Sparse(SparseFixedBitSet),
}

impl AnyBitSet {
    /// `BitSet.of(it, maxDoc)`: sparse when the iterator's `cost` is below
    /// `maxDoc >>> 7` (one doc in 128), else dense; then `or(it)`.
    pub fn of(docs: &mut dyn Iterator<Item = i32>, cost: u64, max_doc: usize) -> AnyBitSet {
        let threshold = (max_doc >> 7) as u64;
        let mut set = if cost < threshold {
            AnyBitSet::Sparse(SparseFixedBitSet::new(max_doc))
        } else {
            AnyBitSet::Dense(FixedBitSet::new(max_doc))
        };
        set.as_bit_set_mut().or_iter(docs);
        set
    }

    /// The set behind the abstract interface.
    pub fn as_bit_set(&self) -> &dyn BitSet {
        match self {
            AnyBitSet::Dense(b) => b,
            AnyBitSet::Sparse(b) => b,
        }
    }

    /// The set behind the abstract interface, mutably.
    pub fn as_bit_set_mut(&mut self) -> &mut dyn BitSet {
        match self {
            AnyBitSet::Dense(b) => b,
            AnyBitSet::Sparse(b) => b,
        }
    }
}

/// `NotDocIdSet.bits()`: the complement of `inner`.
#[derive(Debug, Clone, Copy)]
pub struct NotBits<B>(pub B);

impl<B: Bits> Bits for NotBits<B> {
    fn get(&self, index: usize) -> bool {
        !self.0.get(index)
    }
    fn length(&self) -> usize {
        self.0.length()
    }
}

/// `NotDocIdSet.iterator()`: every doc in `0..max_doc` that `inner` (an
/// ascending doc iterator) does not return, with Java's `advance` and
/// `docIDRunEnd`.
#[derive(Debug)]
pub struct NotDocIdSetIter<I> {
    inner: I,
    max_doc: i32,
    doc: i32,
    next_skipped_doc: i32,
}

impl<I: Iterator<Item = i32>> NotDocIdSetIter<I> {
    /// `new NotDocIdSet(maxDoc, in).iterator()`.
    pub fn new(max_doc: i32, inner: I) -> Self {
        NotDocIdSetIter {
            inner,
            max_doc,
            doc: -1,
            next_skipped_doc: -1,
        }
    }

    fn inner_advance(&mut self, target: i32) -> i32 {
        // `inIterator.advance(target)` over a plain ascending iterator.
        loop {
            match self.inner.next() {
                Some(d) if d >= target => return d,
                Some(_) => continue,
                None => return NO_MORE_DOCS,
            }
        }
    }

    /// `docID()`.
    pub fn doc_id(&self) -> i32 {
        self.doc
    }

    /// `advance(target)`.
    pub fn advance(&mut self, target: i32) -> i32 {
        self.doc = target;
        if self.doc > self.next_skipped_doc {
            self.next_skipped_doc = self.inner_advance(self.doc);
        }
        loop {
            if self.doc >= self.max_doc {
                self.doc = NO_MORE_DOCS;
                return self.doc;
            }
            if self.doc != self.next_skipped_doc {
                return self.doc;
            }
            self.doc += 1;
            self.next_skipped_doc = self.inner.next().unwrap_or(NO_MORE_DOCS);
        }
    }

    /// `nextDoc()`.
    pub fn next_doc(&mut self) -> i32 {
        let target = self.doc.saturating_add(1);
        self.advance(target)
    }

    /// `docIDRunEnd()`: the end of the run of matching docs the iterator is
    /// on.
    pub fn doc_id_run_end(&self) -> i32 {
        self.next_skipped_doc.min(self.max_doc)
    }

    /// `cost()`.
    pub fn cost(&self) -> i64 {
        i64::from(self.max_doc)
    }
}

impl<I: Iterator<Item = i32>> Iterator for NotDocIdSetIter<I> {
    type Item = i32;
    fn next(&mut self) -> Option<i32> {
        if self.doc == NO_MORE_DOCS {
            return None;
        }
        match self.next_doc() {
            NO_MORE_DOCS => None,
            d => Some(d),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_all_none_and_apply_mask() {
        let all = MatchAllBits(10);
        let none = MatchNoBits(10);
        assert!(all.get(3) && !none.get(3));
        assert_eq!((all.length(), none.length()), (10, 10));
        let mut bs = FixedBitSet::new(64);
        for i in [0, 5, 63] {
            bs.set(i);
        }
        all.apply_mask(&mut bs, 0);
        assert_eq!(bs.cardinality(), 3);
        let mut mask = FixedBitSet::new(80);
        mask.set(10 + 5);
        mask.set(10 + 63);
        Bits::apply_mask(&mask, &mut bs, 10);
        assert!(Bits::get(&bs, 5));
        assert!(!Bits::get(&bs, 0));
        assert!(Bits::get(&bs, 63));
        none.apply_mask(&mut bs, 0);
        assert_eq!(bs.cardinality(), 0);
        let not = NotBits(MatchNoBits(4));
        assert!(not.get(2));
        assert_eq!(not.length(), 4);
    }

    fn exercise(set: &mut dyn BitSet) {
        set.set(3);
        assert!(!set.get_and_set(700));
        assert!(set.get_and_set(700));
        set.set(701);
        assert_eq!(set.cardinality(), 3);
        assert!(set.approximate_cardinality() >= 1);
        assert_eq!(set.next_set_bit(4), Some(700));
        assert_eq!(set.next_set_bit_in_range(4, 700), None);
        assert_eq!(set.prev_set_bit(699), Some(3));
        assert_eq!(set.next_clear_bit(700), Some(702));
        assert_eq!(set.next_clear_bit_in_range(700, 702), None);
        set.clear(3);
        set.clear_range(700, 701);
        assert_eq!(set.next_set_bit(0), Some(701));
        set.or_iter(&mut [9, 10, NO_MORE_DOCS, 11].into_iter());
        assert_eq!(set.cardinality(), 3);
        set.clear_all();
        assert_eq!(set.cardinality(), 0);
        assert_eq!(set.length(), 1000);
    }

    #[test]
    fn bit_set_trait_over_both_implementations() {
        exercise(&mut FixedBitSet::new(1000));
        exercise(&mut SparseFixedBitSet::new(1000));
        assert!(!Bits::get(&SparseFixedBitSet::new(5), 1));
    }

    #[test]
    fn of_picks_density_like_java() {
        let sparse = AnyBitSet::of(&mut [1, 500].into_iter(), 2, 1280);
        assert!(matches!(sparse, AnyBitSet::Sparse(_)));
        assert_eq!(sparse.as_bit_set().cardinality(), 2);
        let dense = AnyBitSet::of(&mut [1, 500].into_iter(), 10, 1280);
        assert!(matches!(dense, AnyBitSet::Dense(_)));
        assert_eq!(dense.as_bit_set().next_set_bit(2), Some(500));
    }

    #[test]
    fn not_doc_id_set_iterates_the_complement() {
        let got: Vec<i32> = NotDocIdSetIter::new(10, [0, 1, 4, 9].into_iter()).collect();
        assert_eq!(got, vec![2, 3, 5, 6, 7, 8]);
        let mut it = NotDocIdSetIter::new(10, [3, 4, 8].into_iter());
        assert_eq!(it.cost(), 10);
        assert_eq!(it.advance(3), 5);
        assert_eq!(it.doc_id(), 5);
        assert_eq!(it.doc_id_run_end(), 8);
        assert_eq!(it.next_doc(), 6);
        assert_eq!(it.advance(8), 9);
        assert_eq!(it.doc_id_run_end(), 10);
        assert_eq!(it.next_doc(), NO_MORE_DOCS);
        assert_eq!(it.next(), None);
        let empty: Vec<i32> = NotDocIdSetIter::new(3, [0, 1, 2].into_iter()).collect();
        assert!(empty.is_empty());
        let all: Vec<i32> = NotDocIdSetIter::new(3, std::iter::empty()).collect();
        assert_eq!(all, vec![0, 1, 2]);
    }
}
