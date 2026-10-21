//! `BaseDocIdSetTestCase`: a doc-id set's iterator (and its `bits()`, when
//! it has one) checked against the `java.util.BitSet` it was copied from.
//!
//! The set under test is a [`TestDocIdSet`]: its `iterator()` (`None` is
//! Java's `null`, "no documents") and optional `bits()`.
//!
//! Not ported: `testRamBytesUsed` (`RamUsageTester`), and
//! `testIntoBitSet`/`testIntoBitSetBoundChecks` -- the Rust iterators have
//! no `intoBitSet`.

use lucene_search::docid_set::RoaringDocIdSet;
use lucene_search::facets::DocIdSet;
use lucene_search::reader::{DocIdSetIterator, NO_MORE_DOCS};
use lucene_util::bits::{BitSet, Bits};

use super::base_bit_set::{random_set_load, JavaUtilBitSet};
use super::lucene_test_case::LuceneTestCase;
use super::test_util::next_int;

/// A `DocIdSet` under test.
pub trait TestDocIdSet {
    /// `iterator()`; `None` for Java's `null`.
    fn iterator(&self) -> Option<Box<dyn DocIdSetIterator + '_>>;
    /// `bits()`; `None` for Java's `null`.
    fn bits(&self) -> Option<&dyn Bits> {
        None
    }
}

impl TestDocIdSet for DocIdSet {
    fn iterator(&self) -> Option<Box<dyn DocIdSetIterator + '_>> {
        Some(Box::new(DocIdSet::iterator(self)))
    }

    /// `BitDocIdSet.bits()` is its bit set; `IntArrayDocIdSet` has none.
    fn bits(&self) -> Option<&dyn Bits> {
        match self {
            DocIdSet::Bits { bits, .. } => Some(bits),
            _ => None,
        }
    }
}

impl TestDocIdSet for RoaringDocIdSet {
    /// `RoaringDocIdSet.iterator()`: `null` when empty. The Rust iterator
    /// is a plain `Iterator`, so `advance` is [`SlowAdvance`]'s.
    fn iterator(&self) -> Option<Box<dyn DocIdSetIterator + '_>> {
        if self.is_empty() {
            None
        } else {
            Some(Box::new(SlowAdvance::new(
                self.iter(),
                self.cardinality() as i64,
            )))
        }
    }
}

/// A `DocIdSetIterator` over an ascending `Iterator`, `advance` by
/// `slowAdvance` (`nextDoc` until the target is reached).
pub struct SlowAdvance<I: Iterator<Item = i32>> {
    it: I,
    doc: i32,
    cost: i64,
}

impl<I: Iterator<Item = i32>> SlowAdvance<I> {
    /// Unpositioned (`docID() == -1`).
    pub fn new(it: I, cost: i64) -> Self {
        Self { it, doc: -1, cost }
    }
}

impl<I: Iterator<Item = i32>> DocIdSetIterator for SlowAdvance<I> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> lucene_search::Result<i32> {
        self.doc = self.it.next().unwrap_or(NO_MORE_DOCS);
        Ok(self.doc)
    }

    fn advance(&mut self, target: i32) -> lucene_search::Result<i32> {
        while self.doc < target {
            self.next_doc()?;
        }
        Ok(self.doc)
    }

    fn cost(&self) -> i64 {
        self.cost
    }
}

/// `nextSetBit` as `java.util.BitSet` answers it: `-1` when none.
fn next_set(bs: &JavaUtilBitSet, from: i32) -> i32 {
    bs.next_set_bit_any(from as usize).map_or(-1, |d| d as i32)
}

/// The suite, over the sets a `copy_of` builds.
pub struct BaseDocIdSetTestCase<T: TestDocIdSet, F: FnMut(&JavaUtilBitSet, usize) -> T> {
    copy_of: F,
    /// A subclass's extra `assertEquals` check.
    extra_check: Option<fn(&JavaUtilBitSet, &T)>,
}

impl<T: TestDocIdSet, F: FnMut(&JavaUtilBitSet, usize) -> T> BaseDocIdSetTestCase<T, F> {
    /// The suite for the sets `copy_of(bs, length)` builds.
    pub fn new(copy_of: F) -> Self {
        Self {
            copy_of,
            extra_check: None,
        }
    }

    /// Java's overridable `assertEquals(numBits, ds1, ds2)`: `check` runs
    /// after the iterator comparison (`TestRoaringDocIdSet` compares
    /// cardinalities).
    pub fn with_extra_check(mut self, check: fn(&JavaUtilBitSet, &T)) -> Self {
        self.extra_check = Some(check);
        self
    }

    /// Every test of the suite, in Java's declaration order (bar the
    /// unported ones in the module docs).
    pub fn run_all(&mut self, t: &mut LuceneTestCase) {
        self.test_no_bit(t);
        self.test_1_bit(t);
        self.test_2_bits(t);
        self.test_against_bit_set(t);
    }

    fn check(&mut self, t: &mut LuceneTestCase, num_bits: usize, bs: &JavaUtilBitSet) -> T {
        let copy = (self.copy_of)(bs, num_bits);
        assert_equals(t, num_bits, bs, &copy);
        if let Some(check) = self.extra_check {
            check(bs, &copy);
        }
        copy
    }

    /// `testNoBit`.
    pub fn test_no_bit(&mut self, t: &mut LuceneTestCase) {
        self.check(t, 1, &JavaUtilBitSet::new(1));
    }

    /// `test1Bit`.
    pub fn test_1_bit(&mut self, t: &mut LuceneTestCase) {
        let mut bs = JavaUtilBitSet::new(1);
        if t.random().next_boolean() {
            BitSet::set(&mut bs, 0);
        }
        self.check(t, 1, &bs);
    }

    /// `test2Bits`.
    pub fn test_2_bits(&mut self, t: &mut LuceneTestCase) {
        let mut bs = JavaUtilBitSet::new(2);
        for i in 0..2 {
            if t.random().next_boolean() {
                BitSet::set(&mut bs, i);
            }
        }
        self.check(t, 2, &bs);
    }

    /// `testAgainstBitSet`: random sets at several loads, one document,
    /// then regular strides.
    pub fn test_against_bit_set(&mut self, t: &mut LuceneTestCase) {
        let num_bits = next_int(t.random(), 100, 1 << 20) as usize;
        let random_load = t.random().next_float();
        for p in [0.0, 0.0001, random_load, 0.9, 1.0] {
            let set = random_set_load(t.random(), num_bits, p);
            self.check(t, num_bits, &set);
        }
        let mut set = JavaUtilBitSet::new(num_bits);
        BitSet::set(&mut set, 0);
        self.check(t, num_bits, &set);
        BitSet::clear(&mut set, 0);
        let d = t.random().next_int_bounded(num_bits as i32) as usize;
        BitSet::set(&mut set, d);
        self.check(t, num_bits, &set);
        let max_iterations = if t.nightly() { i32::MAX } else { 10 };
        let mut iterations = 0;
        let mut inc = 2;
        while inc < 1000 {
            if iterations >= max_iterations {
                break;
            }
            iterations += 1;
            let mut set = JavaUtilBitSet::new(num_bits);
            let mut d = t.random().next_int_bounded(10) as usize;
            while d < num_bits {
                BitSet::set(&mut set, d);
                d += inc;
            }
            self.check(t, num_bits, &set);
            inc += next_int(t.random(), 1, 100) as usize;
        }
    }
}

/// `assertEquals(numBits, ds1, ds2)`: `nextDoc` walks, a random
/// `nextDoc`/`advance` walk, and `bits()` agreeing with the iterator.
///
/// # Panics
///
/// On the first disagreement.
pub fn assert_equals(
    t: &mut LuceneTestCase,
    num_bits: usize,
    ds1: &JavaUtilBitSet,
    ds2: &dyn TestDocIdSet,
) {
    let io = |r: lucene_search::Result<i32>| r.expect("an in-memory iterator does not fail");
    match ds2.iterator() {
        None => assert_eq!(-1, next_set(ds1, 0)),
        Some(mut it2) => {
            assert_eq!(-1, it2.doc_id());
            let mut doc = next_set(ds1, 0);
            while doc != -1 {
                assert_eq!(doc, io(it2.next_doc()));
                assert_eq!(doc, it2.doc_id());
                doc = next_set(ds1, doc + 1);
            }
            assert_eq!(NO_MORE_DOCS, io(it2.next_doc()));
            assert_eq!(NO_MORE_DOCS, it2.doc_id());
        }
    }
    match ds2.iterator() {
        None => assert_eq!(-1, next_set(ds1, 0)),
        Some(mut it2) => {
            let mut doc = -1;
            while doc != NO_MORE_DOCS {
                if t.random().next_boolean() {
                    doc = next_set(ds1, doc + 1);
                    if doc == -1 {
                        doc = NO_MORE_DOCS;
                    }
                    assert_eq!(doc, io(it2.next_doc()));
                } else {
                    let bound = if t.random().next_boolean() {
                        64
                    } else {
                        (num_bits as i32 / 8).max(1)
                    };
                    let target = doc + 1 + t.random().next_int_bounded(bound);
                    doc = next_set(ds1, target);
                    if doc == -1 {
                        doc = NO_MORE_DOCS;
                    }
                    assert_eq!(doc, io(it2.advance(target)), "advance({target})");
                }
                assert_eq!(doc, it2.doc_id());
            }
        }
    }
    if let Some(bits) = ds2.bits() {
        let mut it2 = ds2.iterator().expect("a set with bits() has an iterator");
        let mut previous = -1;
        loop {
            let doc = io(it2.next_doc());
            let max = if doc == NO_MORE_DOCS {
                bits.length() as i32
            } else {
                doc
            };
            // FBS: `max` is at most `bits.length()`: the iterator's doc, which
            // the assertion below checks against `bits`, or the length itself.
            for i in previous + 1..max {
                assert!(
                    !bits.get(i as usize),
                    "bits().get({i}) but the iterator skipped it"
                );
            }
            if doc == NO_MORE_DOCS {
                break;
            }
            // FBS: a doc past `bits.length()` panics here, failing the test as
            // Java's `get` would: the iterator and `bits()` disagree.
            assert!(
                bits.get(doc as usize),
                "the iterator returned {doc} but bits() is clear"
            );
            previous = doc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `IntArrayDocIdSet`'s copy: the set bits as a sorted array.
    fn array(bs: &JavaUtilBitSet, _len: usize) -> DocIdSet {
        DocIdSet::Array(super::super::base_bit_set::set_bits(bs))
    }

    #[test]
    fn the_suite_passes_an_array_set() {
        let mut t = LuceneTestCase::with_config("array", 5, false, 1);
        BaseDocIdSetTestCase::new(array).run_all(&mut t);
    }

    #[test]
    #[should_panic(expected = "extra check fired")]
    fn the_extra_check_runs_after_each_copy() {
        let mut t = LuceneTestCase::with_config("extra", 5, false, 1);
        BaseDocIdSetTestCase::new(array)
            .with_extra_check(|_, _| panic!("extra check fired"))
            .test_no_bit(&mut t);
    }

    #[test]
    fn a_bits_set_is_checked_through_bits_too() {
        let mut t = LuceneTestCase::with_config("bits", 5, false, 1);
        let mut bits = lucene_util::FixedBitSet::new(100);
        bits.set(3);
        bits.set(99);
        let set = DocIdSet::Bits { bits, cost: 2 };
        let mut model = JavaUtilBitSet::new(100);
        BitSet::set(&mut model, 3);
        BitSet::set(&mut model, 99);
        assert_equals(&mut t, 100, &model, &set);
        assert!(TestDocIdSet::bits(&DocIdSet::Empty).is_none());
        let mut it = SlowAdvance::new([1, 5].into_iter(), 2);
        assert_eq!(it.cost(), 2);
        assert_eq!(it.advance(4).unwrap(), 5);
        assert_eq!(it.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    #[should_panic(expected = "but bits() is clear")]
    fn bits_that_disagree_with_the_iterator_fail() {
        struct Lying(DocIdSet, lucene_util::FixedBitSet);
        impl TestDocIdSet for Lying {
            fn iterator(&self) -> Option<Box<dyn DocIdSetIterator + '_>> {
                TestDocIdSet::iterator(&self.0)
            }
            fn bits(&self) -> Option<&dyn Bits> {
                Some(&self.1)
            }
        }
        let mut t = LuceneTestCase::with_config("lying", 5, false, 1);
        let mut model = JavaUtilBitSet::new(10);
        BitSet::set(&mut model, 4);
        let set = Lying(DocIdSet::Array(vec![4]), lucene_util::FixedBitSet::new(10));
        assert_equals(&mut t, 10, &model, &set);
    }

    #[test]
    #[should_panic(expected = "but the iterator skipped it")]
    fn bits_with_an_extra_doc_fail() {
        struct Extra(DocIdSet, lucene_util::FixedBitSet);
        impl TestDocIdSet for Extra {
            fn iterator(&self) -> Option<Box<dyn DocIdSetIterator + '_>> {
                TestDocIdSet::iterator(&self.0)
            }
            fn bits(&self) -> Option<&dyn Bits> {
                Some(&self.1)
            }
        }
        let mut t = LuceneTestCase::with_config("extra", 5, false, 1);
        let mut bits = lucene_util::FixedBitSet::new(10);
        bits.set(2);
        assert_equals(
            &mut t,
            10,
            &JavaUtilBitSet::new(10),
            &Extra(DocIdSet::Empty, bits),
        );
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn an_iterator_that_drops_a_doc_fails() {
        let mut t = LuceneTestCase::with_config("drop", 5, false, 1);
        BaseDocIdSetTestCase::new(|bs: &JavaUtilBitSet, len| {
            let mut docs = super::super::base_bit_set::set_bits(bs);
            if len > 2 {
                docs.pop();
            }
            DocIdSet::Array(docs)
        })
        .test_against_bit_set(&mut t);
    }

    #[test]
    fn null_iterators_mean_empty() {
        let mut t = LuceneTestCase::with_config("null", 5, false, 1);
        let empty = lucene_search::docid_set::RoaringBuilder::new(10).build();
        assert!(TestDocIdSet::iterator(&empty).is_none());
        assert_equals(&mut t, 10, &JavaUtilBitSet::new(10), &empty);
    }
}
