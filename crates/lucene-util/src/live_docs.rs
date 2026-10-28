//! Ports of `org.apache.lucene.util.LiveDocs`, `DenseLiveDocs` and
//! `SparseLiveDocs`: a segment's live documents together with the cached
//! deleted count, and iterators over the live and the deleted docs.
//!
//! Dense keeps a bitset of *live* docs (the `.liv` file's content); sparse
//! keeps a [`SparseFixedBitSet`] of *deleted* docs, which is what makes it
//! cheap when deletions are rare. The iterators are Rust iterators (this
//! port's `DocIdSetIterator`), with Java's `cost()` exposed alongside.

use crate::bits::Bits;
use crate::fixed_bit_set::FixedBitSet;
use crate::sparse_fixed_bit_set::SparseFixedBitSet;

/// `org.apache.lucene.util.LiveDocs`.
pub trait LiveDocs: Bits {
    /// `liveDocsIterator()`, with its `cost()`.
    fn live_docs_iter(&self) -> (Box<dyn Iterator<Item = i32> + '_>, i64);
    /// `deletedDocsIterator()`, with its `cost()`.
    fn deleted_docs_iter(&self) -> (Box<dyn Iterator<Item = i32> + '_>, i64);
    /// `deletedCount()`.
    fn deleted_count(&self) -> i32;
}

/// `DenseLiveDocs.Builder.build`'s / `SparseLiveDocs.Builder.build`'s
/// rejection: the deleted count is outside `0..=maxDoc`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("deletedCount={deleted_count} is outside valid range [0, {max_doc}]")]
pub struct DeletedCountOutOfRange {
    pub deleted_count: i64,
    pub max_doc: i32,
}

fn check_count(count: i64, max_doc: i32) -> Result<i32, DeletedCountOutOfRange> {
    if count < 0 || count > i64::from(max_doc) {
        return Err(DeletedCountOutOfRange {
            deleted_count: count,
            max_doc,
        });
    }
    Ok(count as i32)
}

/// The set bits below `n` of the words `word(0), word(1), ...`, ascending:
/// each word is fetched once and its bits peeled with `trailing_zeros`.
struct WordBitsIter<F> {
    word: F,
    n: usize,
    /// Index of the word `cur` came from.
    w: usize,
    cur: u64,
}

impl<F: Fn(usize) -> u64> WordBitsIter<F> {
    fn new(n: usize, word: F) -> Self {
        let mut it = WordBitsIter {
            word,
            n,
            w: 0,
            cur: 0,
        };
        if n > 0 {
            it.cur = it.load(0);
        }
        it
    }

    /// Word `w`, with the bits at or past `n` masked off.
    fn load(&self, w: usize) -> u64 {
        let v = (self.word)(w);
        let end = self.n - (w << 6);
        if end < 64 {
            v & ((1u64 << end) - 1)
        } else {
            v
        }
    }
}

impl<F: Fn(usize) -> u64> Iterator for WordBitsIter<F> {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        let num_words = self.n.div_ceil(64);
        while self.cur == 0 {
            self.w += 1;
            if self.w >= num_words {
                return None;
            }
            self.cur = self.load(self.w);
        }
        let d = (self.w << 6) + self.cur.trailing_zeros() as usize;
        self.cur &= self.cur - 1;
        Some(d as i32)
    }
}

/// `org.apache.lucene.util.DenseLiveDocs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLiveDocs {
    live_docs: FixedBitSet,
    max_doc: i32,
    deleted_count: i32,
}

impl DenseLiveDocs {
    /// `DenseLiveDocs.builder(liveDocs, maxDoc)[.withDeletedCount(n)].build()`:
    /// the deleted count is `maxDoc - cardinality` unless given.
    pub fn build(
        live_docs: FixedBitSet,
        max_doc: i32,
        deleted_count: Option<i32>,
    ) -> Result<Self, DeletedCountOutOfRange> {
        let count = match deleted_count {
            Some(c) => i64::from(c),
            None => i64::from(max_doc) - live_docs.cardinality() as i64,
        };
        let deleted_count = check_count(count, max_doc)?;
        debug_assert!(live_docs.len() as i64 >= i64::from(max_doc));
        Ok(DenseLiveDocs {
            live_docs,
            max_doc,
            deleted_count,
        })
    }

    /// The live-docs bitset.
    pub fn bits(&self) -> &FixedBitSet {
        &self.live_docs
    }

    /// `ramBytesUsed()` of the bitset (8 bytes per word).
    pub fn ram_bytes_used(&self) -> usize {
        self.live_docs.words().len() * 8
    }
}

impl Bits for DenseLiveDocs {
    fn get(&self, index: usize) -> bool {
        // FBS: Java's contract is `index < length() <= liveDocs.length()`;
        // `FixedBitSet::get` checks against `self.live_docs.len()`.
        self.live_docs.get(index)
    }
    fn length(&self) -> usize {
        self.max_doc.max(0) as usize
    }
}

impl LiveDocs for DenseLiveDocs {
    /// `new BitSetIterator(liveDocs, maxDoc - deletedCount)`.
    fn live_docs_iter(&self) -> (Box<dyn Iterator<Item = i32> + '_>, i64) {
        let bits = &self.live_docs;
        let mut next = bits.next_set_bit(0);
        let it = std::iter::from_fn(move || {
            let b = next?;
            next = bits.next_set_bit(b + 1);
            Some(b as i32)
        });
        (
            Box::new(it),
            i64::from(self.max_doc) - i64::from(self.deleted_count),
        )
    }

    /// `new FilteredDocIdSetIterator(maxDoc, deletedCount, doc -> !live)`.
    fn deleted_docs_iter(&self) -> (Box<dyn Iterator<Item = i32> + '_>, i64) {
        let words = self.live_docs.words();
        // Word by word over the complement, rather than a `get` per document.
        // `n = max_doc <= live_docs.len()`, so every word read exists.
        let it = WordBitsIter::new(self.length(), move |w| !words[w]);
        (Box::new(it), i64::from(self.deleted_count))
    }

    fn deleted_count(&self) -> i32 {
        self.deleted_count
    }
}

impl std::fmt::Display for DenseLiveDocs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DenseLiveDocs(maxDoc={}, deleted={}, deletionRate={:.2}%)",
            self.max_doc,
            self.deleted_count,
            100.0 * f64::from(self.deleted_count) / f64::from(self.max_doc)
        )
    }
}

/// `org.apache.lucene.util.SparseLiveDocs`.
#[derive(Debug, Clone)]
pub struct SparseLiveDocs {
    deleted_docs: SparseFixedBitSet,
    max_doc: i32,
    deleted_count: i32,
}

impl SparseLiveDocs {
    /// `SparseLiveDocs.builder(deletedDocs, maxDoc)[.withDeletedCount(n)].build()`:
    /// the deleted count is the set's cardinality unless given.
    pub fn build(
        deleted_docs: SparseFixedBitSet,
        max_doc: i32,
        deleted_count: Option<i32>,
    ) -> Result<Self, DeletedCountOutOfRange> {
        let count = match deleted_count {
            Some(c) => i64::from(c),
            None => deleted_docs.cardinality() as i64,
        };
        let deleted_count = check_count(count, max_doc)?;
        Ok(SparseLiveDocs {
            deleted_docs,
            max_doc,
            deleted_count,
        })
    }

    /// The deleted-docs set.
    pub fn deleted_docs(&self) -> &SparseFixedBitSet {
        &self.deleted_docs
    }

    /// `ramBytesUsed()` of the sparse set.
    pub fn ram_bytes_used(&self) -> usize {
        self.deleted_docs.ram_bytes_used()
    }
}

impl Bits for SparseLiveDocs {
    fn get(&self, index: usize) -> bool {
        // FBS: a `SparseFixedBitSet`, whose `get` checks `index` against its
        // own length; Java's contract is `index < length()`.
        !self.deleted_docs.get(index)
    }
    fn length(&self) -> usize {
        self.max_doc.max(0) as usize
    }
}

impl LiveDocs for SparseLiveDocs {
    /// `new FilteredDocIdSetIterator(maxDoc, maxDoc - deletedCount, doc -> !deleted)`.
    fn live_docs_iter(&self) -> (Box<dyn Iterator<Item = i32> + '_>, i64) {
        let deleted = &self.deleted_docs;
        // Word by word: one sparse lookup per 64 documents, not one each.
        let it = WordBitsIter::new(self.length(), move |w| !deleted.word(w));
        (
            Box::new(it),
            i64::from(self.max_doc) - i64::from(self.deleted_count),
        )
    }

    /// `new BitSetIterator(deletedDocs, deletedCount)`.
    fn deleted_docs_iter(&self) -> (Box<dyn Iterator<Item = i32> + '_>, i64) {
        let deleted = &self.deleted_docs;
        let mut next = deleted.next_set_bit(0);
        let it = std::iter::from_fn(move || {
            let b = next?;
            next = deleted.next_set_bit(b + 1);
            Some(b as i32)
        });
        (Box::new(it), i64::from(self.deleted_count))
    }

    fn deleted_count(&self) -> i32 {
        self.deleted_count
    }
}

impl std::fmt::Display for SparseLiveDocs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SparseLiveDocs(maxDoc={}, deleted={}, deletionRate={:.2}%)",
            self.max_doc,
            self.deleted_count,
            100.0 * f64::from(self.deleted_count) / f64::from(self.max_doc)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_counts_and_iterates() {
        let mut live = FixedBitSet::new(10);
        for i in [0, 2, 3, 9] {
            live.set(i);
        }
        let d = DenseLiveDocs::build(live.clone(), 10, None).unwrap();
        assert_eq!(d.deleted_count(), 6);
        assert!(d.get(2) && !d.get(1));
        assert_eq!(d.length(), 10);
        let (it, cost) = d.live_docs_iter();
        assert_eq!(it.collect::<Vec<_>>(), vec![0, 2, 3, 9]);
        assert_eq!(cost, 4);
        let (it, cost) = d.deleted_docs_iter();
        assert_eq!(it.collect::<Vec<_>>(), vec![1, 4, 5, 6, 7, 8]);
        // A bitset longer than `max_doc`: its clear tail is not reported.
        let mut wide = FixedBitSet::new(200);
        wide.set_range(0, 200);
        wide.clear(63);
        wide.clear(64);
        wide.clear(150);
        let w = DenseLiveDocs::build(wide, 130, Some(2)).unwrap();
        assert_eq!(w.deleted_docs_iter().0.collect::<Vec<_>>(), vec![63, 64]);
        assert_eq!(cost, 6);
        assert_eq!(
            d.to_string(),
            "DenseLiveDocs(maxDoc=10, deleted=6, deletionRate=60.00%)"
        );
        assert_eq!(d.bits().len(), 10);
        assert_eq!(d.ram_bytes_used(), 8);
        let given = DenseLiveDocs::build(live.clone(), 10, Some(6)).unwrap();
        assert_eq!(given, d);
        let err = DenseLiveDocs::build(live, 10, Some(11)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "deletedCount=11 is outside valid range [0, 10]"
        );
    }

    #[test]
    fn sparse_counts_and_iterates() {
        let mut del = SparseFixedBitSet::new(100_000);
        for i in [5, 70_000] {
            del.set(i);
        }
        let s = SparseLiveDocs::build(del.clone(), 100_000, None).unwrap();
        assert_eq!(s.deleted_count(), 2);
        assert!(!s.get(5) && s.get(6));
        let (it, cost) = s.deleted_docs_iter();
        assert_eq!(it.collect::<Vec<_>>(), vec![5, 70_000]);
        assert_eq!(cost, 2);
        let (it, cost) = s.live_docs_iter();
        assert_eq!(it.take(6).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 6]);
        assert_eq!(cost, 99_998);
        assert_eq!(s.length(), 100_000);
        assert!(s
            .to_string()
            .starts_with("SparseLiveDocs(maxDoc=100000, deleted=2,"));
        assert!(s.ram_bytes_used() > 0);
        assert_eq!(s.deleted_docs().cardinality(), 2);
        let all: Vec<i32> = s.live_docs_iter().0.collect();
        assert_eq!(all.len(), 99_998);
        assert!(!all.contains(&5) && !all.contains(&70_000));
        assert_eq!(all.last(), Some(&99_999));
        let empty = SparseLiveDocs::build(SparseFixedBitSet::new(1), 0, None).unwrap();
        assert_eq!(empty.live_docs_iter().0.count(), 0);
        assert!(SparseLiveDocs::build(del, 100_000, Some(-1)).is_err());
    }
}
