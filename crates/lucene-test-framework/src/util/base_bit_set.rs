//! `BaseBitSetTestCase`: a [`BitSet`] implementation checked operation for
//! operation against a model, `java.util.BitSet` ([`JavaUtilBitSet`]).
//!
//! A concrete test supplies `copyOf` -- how to build its set from another
//! [`BitSet`] -- and calls the suite's tests (or [`BaseBitSetTestCase::run_all`]).
//! Every test draws from the [`LuceneTestCase`]'s stream in Java's order.

use lucene_search::docid_set::RoaringBuilder;
use lucene_util::bits::{BitSet, Bits};
use lucene_util::java_random::JavaRandom;
use lucene_util::sparse_fixed_bit_set::SparseFixedBitSet;
use lucene_util::FixedBitSet;

use super::lucene_test_case::LuceneTestCase;

/// The load factors most suite tests sweep.
pub const PERCENT_SET: [f32; 7] = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0];

/// `java.util.BitSet` -- growable, word-at-a-time -- behind Lucene's
/// `BitSet` interface with a fixed `length()` (the suite's
/// `JavaUtilBitSet`). It is the model the suites compare against, so it is
/// written for obviousness, independently of [`FixedBitSet`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JavaUtilBitSet {
    words: Vec<u64>,
    num_bits: usize,
}

impl JavaUtilBitSet {
    /// `new java.util.BitSet(numBits)`, of `length()` `num_bits`.
    pub fn new(num_bits: usize) -> Self {
        Self {
            words: vec![0; num_bits.div_ceil(64)],
            num_bits,
        }
    }

    /// `java.util.BitSet.size()`: the bits allocated, a multiple of 64.
    pub fn size(&self) -> usize {
        self.words.len() * 64
    }

    fn grow(&mut self, bit: usize) {
        let need = bit / 64 + 1;
        if self.words.len() < need {
            self.words.resize(need, 0);
        }
    }

    /// `java.util.BitSet.set(fromIndex, toIndex)`.
    pub fn set_range(&mut self, from: usize, to: usize) {
        for i in from..to {
            BitSet::set(self, i);
        }
    }

    /// `java.util.BitSet.flip(i)`.
    pub fn flip(&mut self, i: usize) {
        self.grow(i);
        self.words[i / 64] ^= 1 << (i % 64);
    }

    /// `java.util.BitSet.flip(fromIndex, toIndex)`.
    pub fn flip_range(&mut self, from: usize, to: usize) {
        for i in from..to {
            self.flip(i);
        }
    }

    fn zip(&mut self, other: &Self, f: impl Fn(u64, u64) -> u64) {
        let n = self.words.len().max(other.words.len());
        self.words.resize(n, 0);
        for (i, w) in self.words.iter_mut().enumerate() {
            *w = f(*w, other.words.get(i).copied().unwrap_or(0));
        }
    }

    /// `java.util.BitSet.and(set)`.
    pub fn and(&mut self, other: &Self) {
        self.zip(other, |a, b| a & b);
    }

    /// `java.util.BitSet.or(set)`.
    pub fn or(&mut self, other: &Self) {
        self.zip(other, |a, b| a | b);
    }

    /// `java.util.BitSet.xor(set)`.
    pub fn xor(&mut self, other: &Self) {
        self.zip(other, |a, b| a ^ b);
    }

    /// `java.util.BitSet.andNot(set)`.
    pub fn and_not(&mut self, other: &Self) {
        self.zip(other, |a, b| a & !b);
    }

    /// `java.util.BitSet.nextSetBit(from)`, unbounded by `length()`.
    pub fn next_set_bit_any(&self, from: usize) -> Option<usize> {
        let mut w = from / 64;
        let mut word = self.words.get(w)? & (u64::MAX << (from % 64));
        loop {
            if word != 0 {
                return Some(w * 64 + word.trailing_zeros() as usize);
            }
            w += 1;
            word = *self.words.get(w)?;
        }
    }

    /// `java.util.BitSet.nextClearBit(from)`, unbounded by `length()`.
    pub fn next_clear_bit_any(&self, from: usize) -> usize {
        let mut i = from;
        while Bits::get(self, i) {
            i += 1;
        }
        i
    }
}

impl Bits for JavaUtilBitSet {
    fn get(&self, index: usize) -> bool {
        self.words
            .get(index / 64)
            .is_some_and(|w| w >> (index % 64) & 1 == 1)
    }

    fn length(&self) -> usize {
        self.num_bits
    }
}

impl BitSet for JavaUtilBitSet {
    fn set(&mut self, i: usize) {
        self.grow(i);
        self.words[i / 64] |= 1 << (i % 64);
    }

    fn get_and_set(&mut self, i: usize) -> bool {
        let v = Bits::get(self, i);
        BitSet::set(self, i);
        v
    }

    fn clear(&mut self, i: usize) {
        if let Some(w) = self.words.get_mut(i / 64) {
            *w &= !(1 << (i % 64));
        }
    }

    fn clear_range(&mut self, start: usize, end: usize) {
        for i in start..end {
            BitSet::clear(self, i);
        }
    }

    fn clear_all(&mut self) {
        self.words.fill(0);
    }

    fn cardinality(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    fn approximate_cardinality(&self) -> usize {
        BitSet::cardinality(self)
    }

    fn prev_set_bit(&self, index: usize) -> Option<usize> {
        // `java.util.BitSet.previousSetBit`: word at a time, downwards.
        let mut w = index / 64;
        let mut word = self.words.get(w).copied().unwrap_or(0) & (u64::MAX >> (63 - index % 64));
        if w >= self.words.len() {
            w = self.words.len();
            word = 0;
        }
        loop {
            if word != 0 {
                return Some(w * 64 + 63 - word.leading_zeros() as usize);
            }
            if w == 0 {
                return None;
            }
            w -= 1;
            word = self.words[w];
        }
    }

    fn next_set_bit_in_range(&self, start: usize, end: usize) -> Option<usize> {
        self.next_set_bit_any(start).filter(|&n| n < end)
    }

    fn next_clear_bit_in_range(&self, start: usize, upper_bound: usize) -> Option<usize> {
        Some(self.next_clear_bit_any(start)).filter(|&n| n < upper_bound)
    }
}

/// `BaseBitSetTestCase.randomSet(numBits, numBitsSet)`: `num_bits_set`
/// distinct random bits (all of them when equal), by rejection.
pub fn random_set(r: &mut JavaRandom, num_bits: usize, num_bits_set: usize) -> JavaUtilBitSet {
    assert!(num_bits_set <= num_bits);
    let mut set = JavaUtilBitSet::new(num_bits);
    if num_bits_set == num_bits {
        set.set_range(0, num_bits);
    } else {
        let bound = i32::try_from(num_bits).expect("a test bit set fits an int");
        for _ in 0..num_bits_set {
            loop {
                let o = r.next_int_bounded(bound) as usize;
                if !Bits::get(&set, o) {
                    BitSet::set(&mut set, o);
                    break;
                }
            }
        }
    }
    set
}

/// `randomSet(numBits, percentSet)`: `(int) (percentSet * numBits)` bits,
/// the product in `float` as Java computes it.
pub fn random_set_load(r: &mut JavaRandom, num_bits: usize, percent_set: f32) -> JavaUtilBitSet {
    random_set(r, num_bits, (percent_set * num_bits as f32) as usize)
}

/// Every set bit of `bs`, ascending.
pub fn set_bits(bs: &dyn BitSet) -> Vec<i32> {
    let mut out = Vec::new();
    let mut next = if bs.length() == 0 {
        None
    } else {
        bs.next_set_bit(0)
    };
    while let Some(d) = next {
        out.push(d as i32);
        next = if d + 1 >= bs.length() {
            None
        } else {
            bs.next_set_bit(d + 1)
        };
    }
    out
}

/// `assertEquals(set1, set2, maxDoc)`: `get` agrees on every index.
///
/// # Panics
///
/// At the first index where they differ.
pub fn assert_bits_equal(set1: &dyn Bits, set2: &dyn Bits, max_doc: usize) {
    for i in 0..max_doc {
        // FBS: `max_doc` is the length both sets were built with (Java's
        // `assertEquals` contract); a set shorter than that panics, which
        // fails the test as Java's `IndexOutOfBoundsException` would.
        assert_eq!(set1.get(i), set2.get(i), "Different at {i}");
    }
}

/// What `randomCopy` hands `or(DocIdSetIterator)`: plain ids, or a
/// `BitSetIterator` over a `FixedBitSet` or `SparseFixedBitSet`, whose
/// presence Java's `or` checks for to take a specialised union.
#[derive(Debug)]
pub enum OrSource {
    /// Ids from any other iterator (the set itself, `copyOf`, Roaring).
    Docs(Vec<i32>),
    /// A `BitSetIterator` over a `FixedBitSet`.
    Fixed(FixedBitSet),
    /// A `BitSetIterator` over a `SparseFixedBitSet`.
    Sparse(SparseFixedBitSet),
}

impl OrSource {
    /// The ids the source's iterator yields.
    pub fn docs(&self) -> Vec<i32> {
        match self {
            Self::Docs(d) => d.clone(),
            Self::Fixed(f) => set_bits(f),
            Self::Sparse(s) => set_bits(s),
        }
    }
}

/// The suite, over the sets a `copy_of` builds.
pub struct BaseBitSetTestCase<T: BitSet, F: FnMut(&dyn BitSet, usize) -> T> {
    copy_of: F,
    /// A subclass's extra `assertEquals` check on the set under test.
    extra_check: Option<fn(&T)>,
    /// The set's own `or(DocIdSetIterator)` dispatch: `true` when it took a
    /// specialised union for the source, `false` to fall back to `or_iter`.
    or_dispatch: Option<fn(&mut T, &OrSource) -> bool>,
}

impl<T: BitSet, F: FnMut(&dyn BitSet, usize) -> T> BaseBitSetTestCase<T, F> {
    /// The suite for the sets `copy_of(bs, length)` builds.
    pub fn new(copy_of: F) -> Self {
        Self {
            copy_of,
            extra_check: None,
            or_dispatch: None,
        }
    }

    /// The set's `or(DocIdSetIterator)` type checks (`FixedBitSet.or` and
    /// `SparseFixedBitSet.or` union a same-type `BitSetIterator`'s set
    /// directly): `dispatch` returns whether it handled the source.
    pub fn with_or_dispatch(mut self, dispatch: fn(&mut T, &OrSource) -> bool) -> Self {
        self.or_dispatch = Some(dispatch);
        self
    }

    /// Java's overridable `assertEquals(set1, set2, maxDoc)`: `check` runs on
    /// the set under test after every bit-by-bit comparison
    /// (`TestSparseFixedBitSet` checks its blocks' structure).
    pub fn with_extra_check(mut self, check: fn(&T)) -> Self {
        self.extra_check = Some(check);
        self
    }

    fn assert_equals(&self, set1: &dyn Bits, set2: &T, max_doc: usize) {
        assert_bits_equal(set1, set2, max_doc);
        if let Some(check) = self.extra_check {
            check(set2);
        }
    }

    fn copy(&mut self, bs: &dyn BitSet, length: usize) -> T {
        (self.copy_of)(bs, length)
    }

    fn num_bits(t: &mut LuceneTestCase) -> usize {
        1 + t.random().next_int_bounded(100_000) as usize
    }

    /// Every test of the suite, in Java's declaration order.
    pub fn run_all(&mut self, t: &mut LuceneTestCase) {
        self.test_cardinality(t);
        self.test_prev_set_bit(t);
        self.test_next_set_bit(t);
        self.test_next_set_bit_in_range(t);
        self.test_set(t);
        self.test_get_and_set(t);
        self.test_clear(t);
        self.test_clear_range(t);
        self.test_clear_all(t);
        self.test_or_sparse(t);
        self.test_or_dense(t);
        self.test_or_random(t);
    }

    /// `testCardinality`.
    pub fn test_cardinality(&mut self, t: &mut LuceneTestCase) {
        let num_bits = Self::num_bits(t);
        for p in PERCENT_SET {
            let set1 = random_set_load(t.random(), num_bits, p);
            let set2 = self.copy(&set1, num_bits);
            assert_eq!(BitSet::cardinality(&set1), set2.cardinality());
        }
    }

    /// `testPrevSetBit`.
    pub fn test_prev_set_bit(&mut self, t: &mut LuceneTestCase) {
        let num_bits = Self::num_bits(t);
        for p in PERCENT_SET {
            let set1 = random_set_load(t.random(), num_bits, p);
            let set2 = self.copy(&set1, num_bits);
            let mut want = None;
            for i in 0..num_bits {
                // The model's answer, incrementally: the scan is O(n) per call.
                if Bits::get(&set1, i) {
                    want = Some(i);
                }
                assert_eq!(want, set2.prev_set_bit(i), "{i}");
            }
        }
    }

    /// `testNextSetBit`.
    pub fn test_next_set_bit(&mut self, t: &mut LuceneTestCase) {
        let num_bits = Self::num_bits(t);
        for p in PERCENT_SET {
            let set1 = random_set_load(t.random(), num_bits, p);
            let set2 = self.copy(&set1, num_bits);
            for i in 0..num_bits {
                assert_eq!(BitSet::next_set_bit(&set1, i), set2.next_set_bit(i), "{i}");
            }
        }
    }

    /// `testNextSetBitInRange`.
    pub fn test_next_set_bit_in_range(&mut self, t: &mut LuceneTestCase) {
        let num_bits = Self::num_bits(t);
        for p in PERCENT_SET {
            let set1 = random_set_load(t.random(), num_bits, p);
            let set2 = self.copy(&set1, num_bits);
            for start in 0..num_bits {
                let end = super::random_numbers::random_int_between(
                    t.random(),
                    start as i32 + 1,
                    num_bits as i32,
                ) as usize;
                assert_eq!(
                    set1.next_set_bit_in_range(start, end),
                    set2.next_set_bit_in_range(start, end),
                    "start={start}, end={end}, numBits={num_bits}"
                );
            }
        }
    }

    /// `testSet`.
    pub fn test_set(&mut self, t: &mut LuceneTestCase) {
        let num_bits = Self::num_bits(t);
        let mut set1 = random_set(t.random(), num_bits, 0);
        let mut set2 = self.copy(&set1, num_bits);
        let iters = 10_000 + t.random().next_int_bounded(10_000);
        for _ in 0..iters {
            // FBS: `index < num_bits`, the length `copy` built `set2` with.
            let index = t.random().next_int_bounded(num_bits as i32) as usize;
            BitSet::set(&mut set1, index);
            set2.set(index);
        }
        self.assert_equals(&set1, &set2, num_bits);
    }

    /// `testGetAndSet`.
    pub fn test_get_and_set(&mut self, t: &mut LuceneTestCase) {
        let num_bits = Self::num_bits(t);
        let mut set1 = random_set(t.random(), num_bits, 0);
        let mut set2 = self.copy(&set1, num_bits);
        let iters = 10_000 + t.random().next_int_bounded(10_000);
        for _ in 0..iters {
            let index = t.random().next_int_bounded(num_bits as i32) as usize;
            assert_eq!(
                BitSet::get_and_set(&mut set1, index),
                set2.get_and_set(index)
            );
        }
        self.assert_equals(&set1, &set2, num_bits);
    }

    /// `testClear`.
    pub fn test_clear(&mut self, t: &mut LuceneTestCase) {
        let num_bits = Self::num_bits(t);
        for p in PERCENT_SET {
            let mut set1 = random_set_load(t.random(), num_bits, p);
            let mut set2 = self.copy(&set1, num_bits);
            let iters = 1 + t.random().next_int_bounded(num_bits as i32 * 2);
            for _ in 0..iters {
                // FBS: `index < num_bits`, the length `copy` built `set2` with.
                let index = t.random().next_int_bounded(num_bits as i32) as usize;
                BitSet::clear(&mut set1, index);
                set2.clear(index);
            }
            self.assert_equals(&set1, &set2, num_bits);
        }
    }

    /// `testClearRange`.
    pub fn test_clear_range(&mut self, t: &mut LuceneTestCase) {
        let num_bits = Self::num_bits(t);
        for p in PERCENT_SET {
            let mut set1 = random_set_load(t.random(), num_bits, p);
            let mut set2 = self.copy(&set1, num_bits);
            let iters = t.at_least(10);
            for _ in 0..iters {
                let from = t.random().next_int_bounded(num_bits as i32) as usize;
                let to = t.random().next_int_bounded(num_bits as i32 + 1) as usize;
                // `from >= to` included: the model treats it as a no-op, and
                // the implementation must too (Java calls it unguarded).
                set1.clear_range(from, to);
                set2.clear_range(from, to);
                self.assert_equals(&set1, &set2, num_bits);
            }
        }
    }

    /// `testClearAll`.
    pub fn test_clear_all(&mut self, t: &mut LuceneTestCase) {
        let num_bits = Self::num_bits(t);
        for p in PERCENT_SET {
            let mut set1 = random_set_load(t.random(), num_bits, p);
            let mut set2 = self.copy(&set1, num_bits);
            let iters = t.at_least(10);
            for _ in 0..iters {
                set1.clear_all();
                set2.clear_all();
                self.assert_equals(&set1, &set2, num_bits);
            }
        }
    }

    /// `randomCopy`: the same documents through one of five `DocIdSet`
    /// shapes (the set itself, `copyOf`, `RoaringDocIdSet`, `FixedBitSet`,
    /// `SparseFixedBitSet`).
    fn random_copy(
        &mut self,
        t: &mut LuceneTestCase,
        set: &JavaUtilBitSet,
        num_bits: usize,
    ) -> OrSource {
        let docs = set_bits(set);
        match t.random().next_int_bounded(5) {
            0 => OrSource::Docs(docs),
            1 => OrSource::Docs(set_bits(&self.copy(set, num_bits))),
            2 => {
                let mut builder = RoaringBuilder::new(num_bits);
                for &d in &docs {
                    builder.add(d);
                }
                OrSource::Docs(builder.build().iter().collect())
            }
            3 => {
                let mut fbs = FixedBitSet::new(num_bits);
                BitSet::or_iter(&mut fbs, &mut docs.iter().copied());
                OrSource::Fixed(fbs)
            }
            _ => {
                let mut sfbs = SparseFixedBitSet::new(num_bits);
                BitSet::or_iter(&mut sfbs, &mut docs.iter().copied());
                OrSource::Sparse(sfbs)
            }
        }
    }

    fn test_or(&mut self, t: &mut LuceneTestCase, load: f32) {
        let num_bits = Self::num_bits(t);
        let mut set1 = random_set(t.random(), num_bits, 0);
        let mut set2 = self.copy(&set1, num_bits);
        let iterations = t.at_least(10);
        for _ in 0..iterations {
            let other = random_set_load(t.random(), num_bits, load);
            let source = self.random_copy(t, &other, num_bits);
            let docs = source.docs();
            set1.or_iter(&mut docs.iter().copied());
            if !self
                .or_dispatch
                .is_some_and(|dispatch| dispatch(&mut set2, &source))
            {
                set2.or_iter(&mut docs.iter().copied());
            }
            self.assert_equals(&set1, &set2, num_bits);
        }
    }

    /// `testOrSparse`.
    pub fn test_or_sparse(&mut self, t: &mut LuceneTestCase) {
        self.test_or(t, 0.001);
    }

    /// `testOrDense`.
    pub fn test_or_dense(&mut self, t: &mut LuceneTestCase) {
        self.test_or(t, 0.5);
    }

    /// `testOrRandom`.
    pub fn test_or_random(&mut self, t: &mut LuceneTestCase) {
        let load = t.random().next_float();
        self.test_or(t, load);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_copy(bs: &dyn BitSet, len: usize) -> JavaUtilBitSet {
        let mut c = JavaUtilBitSet::new(len);
        for d in set_bits(bs) {
            BitSet::set(&mut c, d as usize);
        }
        c
    }

    impl<F: FnMut(&dyn BitSet, usize) -> Broken> BaseBitSetTestCase<Broken, F> {
        /// Every test the broken `prev_set_bit` does not touch passes (so
        /// the wrapper is a faithful set), then `testPrevSetBit` fails.
        fn run_until_prev_set_bit(&mut self, t: &mut LuceneTestCase) {
            self.test_cardinality(t);
            self.test_next_set_bit(t);
            self.test_next_set_bit_in_range(t);
            self.test_set(t);
            self.test_get_and_set(t);
            self.test_clear(t);
            self.test_clear_range(t);
            self.test_clear_all(t);
            self.test_or_random(t);
            self.test_prev_set_bit(t);
        }
    }

    #[test]
    fn the_model_is_java_util_bit_set() {
        let mut a = JavaUtilBitSet::new(10);
        assert_eq!(a.size(), 64);
        BitSet::set(&mut a, 3);
        a.flip(70);
        assert_eq!(a.size(), 128);
        assert_eq!(a.next_set_bit_any(4), Some(70));
        assert_eq!(a.next_set_bit_any(71), None);
        assert_eq!(a.next_set_bit_any(500), None);
        assert_eq!(BitSet::next_set_bit(&a, 4), None, "bounded by length()");
        assert_eq!(a.next_clear_bit_any(3), 4);
        assert_eq!(a.next_clear_bit_in_range(3, 4), None);
        assert_eq!(a.prev_set_bit(9), Some(3));
        assert_eq!(a.prev_set_bit(2), None);
        assert_eq!(a.prev_set_bit(500), Some(70));
        assert_eq!(a.prev_set_bit(69), Some(3));
        let mut b = JavaUtilBitSet::new(10);
        b.set_range(2, 5);
        let mut x = a.clone();
        x.and(&b);
        assert_eq!(set_bits(&x), [3]);
        let mut x = a.clone();
        x.or(&b);
        assert_eq!(BitSet::cardinality(&x), 4);
        let mut x = a.clone();
        x.xor(&b);
        assert_eq!(BitSet::cardinality(&x), 3);
        let mut x = a.clone();
        x.and_not(&b);
        assert!(!Bits::get(&x, 3) && Bits::get(&x, 70));
        x.flip_range(0, 2);
        assert!(Bits::get(&x, 1));
        assert!(!BitSet::get_and_set(&mut x, 5) && Bits::get(&x, 5));
        x.clear_range(0, 6);
        BitSet::clear(&mut x, 1000);
        assert_eq!(x.approximate_cardinality(), 1);
        x.clear_all();
        assert_eq!(BitSet::cardinality(&x), 0);
        assert!(set_bits(&JavaUtilBitSet::new(0)).is_empty());
    }

    #[test]
    fn random_sets_have_the_asked_cardinality() {
        let mut r = JavaRandom::new(1);
        assert_eq!(BitSet::cardinality(&random_set(&mut r, 100, 37)), 37);
        assert_eq!(BitSet::cardinality(&random_set(&mut r, 100, 100)), 100);
        assert_eq!(
            BitSet::cardinality(&random_set_load(&mut r, 1000, 0.5)),
            500
        );
    }

    /// A deliberately broken set: `prev_set_bit` misses bit 0.
    #[derive(Debug)]
    struct Broken(JavaUtilBitSet);

    impl Bits for Broken {
        fn get(&self, i: usize) -> bool {
            Bits::get(&self.0, i)
        }
        fn length(&self) -> usize {
            self.0.length()
        }
    }

    impl BitSet for Broken {
        fn set(&mut self, i: usize) {
            BitSet::set(&mut self.0, i);
        }
        fn get_and_set(&mut self, i: usize) -> bool {
            BitSet::get_and_set(&mut self.0, i)
        }
        fn clear(&mut self, i: usize) {
            BitSet::clear(&mut self.0, i);
        }
        fn clear_range(&mut self, s: usize, e: usize) {
            self.0.clear_range(s, e);
        }
        fn cardinality(&self) -> usize {
            BitSet::cardinality(&self.0)
        }
        fn approximate_cardinality(&self) -> usize {
            self.0.approximate_cardinality()
        }
        fn prev_set_bit(&self, i: usize) -> Option<usize> {
            self.0.prev_set_bit(i).filter(|&b| b != 0)
        }
        fn next_set_bit_in_range(&self, s: usize, e: usize) -> Option<usize> {
            self.0.next_set_bit_in_range(s, e)
        }
        fn next_clear_bit_in_range(&self, s: usize, e: usize) -> Option<usize> {
            self.0.next_clear_bit_in_range(s, e)
        }
    }

    #[test]
    fn the_suite_passes_the_model_itself() {
        let mut t = LuceneTestCase::with_config("model", 3, false, 1);
        BaseBitSetTestCase::new(model_copy).run_all(&mut t);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn the_suite_catches_a_broken_prev_set_bit() {
        let mut t = LuceneTestCase::with_config("broken", 3, false, 1);
        BaseBitSetTestCase::new(|bs: &dyn BitSet, len| Broken(model_copy(bs, len)))
            .run_until_prev_set_bit(&mut t);
    }

    #[test]
    #[should_panic(expected = "Different at")]
    fn bits_equal_reports_the_index() {
        let a = JavaUtilBitSet::new(4);
        let mut b = JavaUtilBitSet::new(4);
        BitSet::set(&mut b, 2);
        assert_bits_equal(&a, &b, 4);
    }

    #[test]
    #[should_panic(expected = "extra check fired")]
    fn the_extra_check_runs_on_every_comparison() {
        let mut t = LuceneTestCase::with_config("extra", 3, false, 1);
        BaseBitSetTestCase::new(model_copy)
            .with_extra_check(|_| panic!("extra check fired"))
            .test_set(&mut t);
    }

    #[test]
    fn the_or_dispatch_sees_every_shape_and_its_union_is_checked() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static FIXED: AtomicUsize = AtomicUsize::new(0);
        static SPARSE: AtomicUsize = AtomicUsize::new(0);
        let model = model_copy;
        let mut suite = BaseBitSetTestCase::new(model).with_or_dispatch(|_, source| {
            match source {
                OrSource::Fixed(_) => FIXED.fetch_add(1, Ordering::Relaxed),
                OrSource::Sparse(_) => SPARSE.fetch_add(1, Ordering::Relaxed),
                OrSource::Docs(_) => return false,
            };
            false
        });
        for seed in 0..8 {
            suite.test_or_random(&mut LuceneTestCase::with_config("dispatch", seed, false, 1));
        }
        assert!(FIXED.load(Ordering::Relaxed) > 0 && SPARSE.load(Ordering::Relaxed) > 0);
    }

    #[test]
    #[should_panic(expected = "Different at")]
    fn a_dispatch_that_drops_the_union_fails_the_suite() {
        let mut t = LuceneTestCase::with_config("dispatch-drop", 3, false, 1);
        BaseBitSetTestCase::new(model_copy)
            .with_or_dispatch(|_, source| !matches!(source, OrSource::Docs(_)))
            .test_or_dense(&mut t);
    }
}
