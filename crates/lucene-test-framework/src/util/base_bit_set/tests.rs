#![cfg(test)]
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
