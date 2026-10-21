//! `TestSparseFixedBitSet`: `BaseBitSetTestCase` over `SparseFixedBitSet`,
//! with Java's `assertEquals` override (block structure and
//! `nonZeroLongCount`) run after every comparison, plus the class's own
//! tests.
//!
//! `testRamBytesUsed` compares Rust's measured heap (`Vec` capacities), not
//! `RamUsageEstimator`'s figure; its relations are Java's, and its unions go
//! through `or(&SparseFixedBitSet)` as Java's `BitSetIterator` ones do.

use lucene_test_framework::util::base_bit_set::{
    assert_bits_equal, BaseBitSetTestCase, OrSource, PERCENT_SET,
};
use lucene_test_framework::util::lucene_test_case::{at_least, LuceneTestCase};
use lucene_test_framework::util::test_util::next_int;
use lucene_util::bits::BitSet;
use lucene_util::sparse_fixed_bit_set::SparseFixedBitSet;
use lucene_util::FixedBitSet;

/// `copyOf(bs, length)`.
fn copy_of(bs: &dyn BitSet, length: usize) -> SparseFixedBitSet {
    let mut set = SparseFixedBitSet::new(length);
    let mut doc = bs.next_set_bit(0);
    while let Some(d) = doc {
        set.set(d);
        doc = if d + 1 >= length {
            None
        } else {
            bs.next_set_bit(d + 1)
        };
    }
    set
}

/// Java's `assertEquals` override, after the bit-by-bit half.
fn check_structure(set: &SparseFixedBitSet) {
    if let Err(e) = set.check_structure() {
        panic!("{e}");
    }
}

type Copy = fn(&dyn BitSet, usize) -> SparseFixedBitSet;

/// `SparseFixedBitSet.or(DocIdSetIterator)`: a `BitSetIterator` over a
/// `SparseFixedBitSet` takes the block-wise `or(SparseFixedBitSet)`.
fn or_dispatch(set: &mut SparseFixedBitSet, source: &OrSource) -> bool {
    match source {
        OrSource::Sparse(other) => {
            set.or(other);
            true
        }
        _ => false,
    }
}

fn suite() -> BaseBitSetTestCase<SparseFixedBitSet, Copy> {
    BaseBitSetTestCase::new(copy_of as Copy)
        .with_extra_check(check_structure)
        .with_or_dispatch(or_dispatch)
}

macro_rules! base_tests {
    ($($name:ident),*) => {$(
        #[test]
        fn $name() {
            suite().$name(&mut LuceneTestCase::new(stringify!($name)));
        }
    )*};
}

base_tests!(
    test_cardinality,
    test_prev_set_bit,
    test_next_set_bit,
    test_next_set_bit_in_range,
    test_set,
    test_get_and_set,
    test_clear,
    test_clear_range,
    test_clear_all,
    test_or_sparse,
    test_or_dense,
    test_or_random
);

#[test]
fn test_approximate_cardinality() {
    let mut t = LuceneTestCase::new("test_approximate_cardinality");
    let r = t.random();
    let mut set = SparseFixedBitSet::new(10_000);
    let first = r.next_int_bounded(1000) as usize;
    let interval = 200 + r.next_int_bounded(1000) as usize;
    for i in (first..set.len()).step_by(interval) {
        set.set(i);
    }
    assert!((set.cardinality() as i64 - set.approximate_cardinality() as i64).abs() <= 20);
}

#[test]
fn test_approximate_cardinality_on_dense_set() {
    let mut t = LuceneTestCase::new("test_approximate_cardinality_on_dense_set");
    let num_docs = next_int(t.random(), 1, 10_000) as usize;
    let mut set = SparseFixedBitSet::new(num_docs);
    for i in 0..num_docs {
        set.set(i);
    }
    assert_eq!(num_docs, set.approximate_cardinality());
}

#[test]
fn test_ram_bytes_used() {
    let mut t = LuceneTestCase::new("test_ram_bytes_used");
    let r = t.random();
    let size = 1000 + r.next_int_bounded(10_000) as usize;
    let mut original = SparseFixedBitSet::new(size);
    for _ in 0..3 {
        original.set(r.next_int_bounded(size as i32) as usize);
    }
    assert!(original.ram_bytes_used() > 0);

    let mut copy = copy_of(&original, size);
    let mut other = SparseFixedBitSet::new(size);
    let interval = 10 + r.next_int_bounded(100) as usize;
    for i in (0..size).step_by(interval) {
        other.set(i);
    }
    copy.or(&other);
    assert!(copy.ram_bytes_used() > original.ram_bytes_used());

    let mut copy = copy_of(&original, size);
    copy.or_iter(0..size);
    assert!(copy.ram_bytes_used() > original.ram_bytes_used());
    assert!(copy.ram_bytes_used() > size / 8);

    let set_copy = copy_of(&original, size);
    assert_eq!(set_copy.ram_bytes_used(), original.ram_bytes_used());

    let mut or_copy = SparseFixedBitSet::new(size);
    or_copy.or(&original);
    assert!((original.ram_bytes_used() as i64 - or_copy.ram_bytes_used() as i64).abs() <= 64);
}

fn brute_next_unset(set: &SparseFixedBitSet, from: usize, upper_bound: usize) -> Option<usize> {
    (from..upper_bound).find(|&i| !set.get(i))
}

#[test]
fn test_set_range() {
    let mut t = LuceneTestCase::new("test_set_range");
    let multiplier = t.multiplier();
    let r = t.random();
    let num_bits = 1 + r.next_int_bounded(100_000) as usize;
    for p in PERCENT_SET {
        let mut expected = FixedBitSet::new(num_bits);
        let num_initial = (p * num_bits as f32) as usize;
        for _ in 0..num_initial {
            expected.set(r.next_int_bounded(num_bits as i32) as usize);
        }
        let mut set = copy_of(&expected, num_bits);
        let iters = at_least(r, multiplier, 10);
        for _ in 0..iters {
            let from = r.next_int_bounded(num_bits as i32) as usize;
            let to = r.next_int_bounded(num_bits as i32 + 1) as usize;
            expected.set_range(from, to);
            set.set_range(from, to);
            assert_bits_equal(&expected, &set, num_bits);
            check_structure(&set);
        }
    }
}

#[test]
fn test_next_clear_bit() {
    let mut t = LuceneTestCase::new("test_next_clear_bit");
    let outer = if t.nightly() { 300 } else { 60 };
    let r = t.random();
    for _ in 0..outer {
        let n = next_int(r, 1, 40_000);
        let mut set = SparseFixedBitSet::new(n as usize);
        let num_sets = next_int(r, 0, n.min(8_000));
        for _ in 0..num_sets {
            set.set(r.next_int_bounded(n) as usize);
        }
        for _ in 0..300 {
            let from = r.next_int_bounded(n) as usize;
            assert_eq!(
                brute_next_unset(&set, from, n as usize),
                set.next_clear_bit(from)
            );
            let ub = from + 1 + r.next_int_bounded(n - from as i32) as usize;
            assert_eq!(
                brute_next_unset(&set, from, ub),
                set.next_clear_bit_in_range(from, ub)
            );
        }
    }
}
