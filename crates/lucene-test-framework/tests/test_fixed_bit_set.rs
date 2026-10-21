//! `TestFixedBitSet`: `BaseBitSetTestCase` over `FixedBitSet`, plus the
//! class's own tests against a `java.util.BitSet` model.
//!
//! Adapted, with Java's random draws kept:
//! - `makeFixedBitSet` builds over a word array longer than needed half the
//!   time; `FixedBitSet::from_words` takes an exact length, so the draw is
//!   made and the exact array used.
//! - `testAndNot` clears through `andNot(DocIdSetIterator)`, which the Rust
//!   set lacks; all three iterator shapes reduce to `and_not(&FixedBitSet)`.
//! - `ensureCapacity` consumes its argument, so the "returns a copy" checks
//!   clone first.
//! - `testAsBits` (`asReadOnlyBits`) is not ported: the Rust set has no
//!   read-only view; `copyOf` over an anonymous `Bits` is.

use lucene_test_framework::util::base_bit_set::{BaseBitSetTestCase, JavaUtilBitSet, OrSource};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_test_framework::util::test_util::next_int;
use lucene_util::bits::{BitSet, Bits};
use lucene_util::fixed_bit_set::bits2words;
use lucene_util::java_random::JavaRandom;
use lucene_util::FixedBitSet;

const NO_MORE_DOCS: i64 = i32::MAX as i64;

/// `copyOf(bs, length)`.
fn copy_of(bs: &dyn BitSet, length: usize) -> FixedBitSet {
    let mut set = FixedBitSet::new(length);
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

type Copy = fn(&dyn BitSet, usize) -> FixedBitSet;

/// `FixedBitSet.or(DocIdSetIterator)`: a `BitSetIterator` over a
/// `FixedBitSet` is unioned word by word.
fn or_dispatch(set: &mut FixedBitSet, source: &OrSource) -> bool {
    match source {
        OrSource::Fixed(other) => {
            set.or(other);
            true
        }
        _ => false,
    }
}

fn suite() -> BaseBitSetTestCase<FixedBitSet, Copy> {
    BaseBitSetTestCase::new(copy_of as Copy).with_or_dispatch(or_dispatch)
}

macro_rules! base_tests {
    ($($name:ident),*) => {$(
        #[test]
        fn $name() {
            suite().$name(&mut LuceneTestCase::new(stringify!($name)));
        }
    )*};
}

// `testPrevSetBit` is overridden below, as in Java.
base_tests!(
    test_cardinality,
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

fn as_doc(o: Option<usize>) -> i64 {
    o.map_or(NO_MORE_DOCS, |v| v as i64)
}

fn as_prev(o: Option<usize>) -> i64 {
    o.map_or(-1, |v| v as i64)
}

fn do_get(a: &JavaUtilBitSet, b: &FixedBitSet) {
    assert_eq!(BitSet::cardinality(a), b.cardinality());
    for i in 0..b.len() {
        assert_eq!(Bits::get(a, i), b.get(i), "mismatch: BitSet=[{i}]");
    }
}

fn do_next_set_bit(a: &JavaUtilBitSet, b: &FixedBitSet) {
    assert_eq!(BitSet::cardinality(a), b.cardinality());
    let (mut aa, mut bb): (i64, i64) = (-1, -1);
    loop {
        aa = as_doc(a.next_set_bit_any((aa + 1) as usize));
        bb = if bb < b.len() as i64 - 1 {
            as_doc(b.next_set_bit((bb + 1) as usize))
        } else {
            NO_MORE_DOCS
        };
        assert_eq!(aa, bb);
        if aa == NO_MORE_DOCS {
            break;
        }
    }
}

fn do_next_clear_bit(a: &JavaUtilBitSet, b: &FixedBitSet) {
    assert_eq!(BitSet::cardinality(a), b.cardinality());
    let len = b.len() as i64;
    let (mut aa, mut bb): (i64, i64) = (-1, -1);
    loop {
        let from = aa + 1;
        aa = if from >= len {
            NO_MORE_DOCS
        } else {
            let nc = a.next_clear_bit_any(from as usize) as i64;
            if nc < len {
                nc
            } else {
                NO_MORE_DOCS
            }
        };
        bb = if bb < len - 1 {
            as_doc(b.next_clear_bit((bb + 1) as usize))
        } else {
            NO_MORE_DOCS
        };
        assert_eq!(aa, bb);
        if aa == NO_MORE_DOCS {
            break;
        }
    }
}

fn do_prev_set_bit(r: &mut JavaRandom, a: &JavaUtilBitSet, b: &FixedBitSet) {
    assert_eq!(BitSet::cardinality(a), b.cardinality());
    let mut aa = a.size() as i64 + i64::from(r.next_int_bounded(100));
    let mut bb = aa;
    let blen = b.len() as i64;
    loop {
        aa -= 1;
        while aa >= 0 && !Bits::get(a, aa as usize) {
            aa -= 1;
        }
        bb = if blen == 0 {
            -1
        } else if bb > blen - 1 {
            as_prev(b.prev_set_bit((blen - 1) as usize))
        } else if bb < 1 {
            -1
        } else {
            as_prev(b.prev_set_bit((bb - 1) as usize))
        };
        assert_eq!(aa, bb);
        if aa < 0 {
            break;
        }
    }
}

/// `doIterate`: `nextDoc` or `advance(bb + 1)`, which on a bit set both
/// land on the next set bit.
fn do_iterate(r: &mut JavaRandom, a: &JavaUtilBitSet, b: &FixedBitSet) {
    assert_eq!(BitSet::cardinality(a), b.cardinality());
    let (mut aa, mut bb): (i64, i64) = (-1, -1);
    loop {
        aa = a
            .next_set_bit_any((aa + 1) as usize)
            .map_or(-1, |v| v as i64);
        let _next_doc = r.next_boolean();
        let from = (bb + 1) as usize;
        bb = if from >= b.len() {
            NO_MORE_DOCS
        } else {
            as_doc(b.next_set_bit(from))
        };
        assert_eq!(if aa == -1 { NO_MORE_DOCS } else { aa }, bb);
        if aa < 0 {
            break;
        }
    }
}

fn do_random_sets(t: &mut LuceneTestCase, max_size: i32, iter: i32) {
    let mut prev: Option<(JavaUtilBitSet, FixedBitSet)> = None;
    for _ in 0..iter {
        let r = t.random();
        let sz = next_int(r, 2, max_size) as usize;
        let mut a = JavaUtilBitSet::new(sz);
        let mut b = FixedBitSet::new(sz);
        let n_oper = r.next_int_bounded(sz as i32);
        for _ in 0..n_oper {
            let idx = r.next_int_bounded(sz as i32) as usize;
            BitSet::set(&mut a, idx);
            b.set(idx);
            let idx = r.next_int_bounded(sz as i32) as usize;
            BitSet::clear(&mut a, idx);
            b.clear(idx);
            let idx = r.next_int_bounded(sz as i32) as usize;
            a.flip_range(idx, idx + 1);
            b.flip_range(idx, idx + 1);
            let idx = r.next_int_bounded(sz as i32) as usize;
            a.flip(idx);
            b.flip(idx);
            let val2 = b.get(idx);
            let val = b.get_and_set(idx);
            assert_eq!(val2, val);
            assert!(b.get(idx));
            if !val {
                b.clear(idx);
            }
            assert_eq!(b.get(idx), val);
        }
        do_get(&a, &b);

        let range = |r: &mut JavaRandom| {
            let from = r.next_int_bounded(sz as i32 / 2) as usize;
            (from, from + r.next_int_bounded((sz - from) as i32) as usize)
        };
        let (from, to) = range(r);
        let (mut aa, mut bb) = (a.clone(), b.clone());
        aa.flip_range(from, to);
        bb.flip_range(from, to);
        do_iterate(r, &aa, &bb);

        let (from, to) = range(r);
        let (mut aa, mut bb) = (a.clone(), b.clone());
        BitSet::clear_range(&mut aa, from, to);
        bb.clear_range(from, to);
        do_next_set_bit(&aa, &bb);
        do_next_clear_bit(&aa, &bb);
        do_prev_set_bit(r, &aa, &bb);

        let (from, to) = range(r);
        let (mut aa, mut bb) = (a.clone(), b.clone());
        aa.set_range(from, to);
        bb.set_range(from, to);
        do_next_set_bit(&aa, &bb);
        do_next_clear_bit(&aa, &bb);
        do_prev_set_bit(r, &aa, &bb);

        if let Some((a0, b0)) = prev.as_ref().filter(|(_, b0)| b0.len() <= b.len()) {
            assert_eq!(BitSet::cardinality(&a), b.cardinality());
            let mut a_and = a.clone();
            a_and.and(a0);
            let mut a_or = a.clone();
            a_or.or(a0);
            let mut a_xor = a.clone();
            a_xor.xor(a0);
            let mut a_andn = a.clone();
            a_andn.and_not(a0);
            let mut b_and = b.clone();
            assert_eq!(b, b_and);
            b_and.and(b0);
            let mut b_or = b.clone();
            b_or.or(b0);
            let mut b_xor = b.clone();
            b_xor.xor(b0);
            let mut b_andn = b.clone();
            b_andn.and_not(b0);
            assert_eq!(BitSet::cardinality(a0), b0.cardinality());
            assert_eq!(BitSet::cardinality(&a_or), b_or.cardinality());
            do_iterate(r, &a_and, &b_and);
            do_iterate(r, &a_or, &b_or);
            do_iterate(r, &a_andn, &b_andn);
            do_iterate(r, &a_xor, &b_xor);
            assert_eq!(BitSet::cardinality(&a_and), b_and.cardinality());
            assert_eq!(BitSet::cardinality(&a_xor), b_xor.cardinality());
            assert_eq!(BitSet::cardinality(&a_andn), b_andn.cardinality());
        }
        prev = Some((a, b));
    }
}

#[test]
fn test_approximate_cardinality() {
    let mut t = LuceneTestCase::new("test_approximate_cardinality");
    let r = t.random();
    let mut set = FixedBitSet::new(next_int(r, 100_000, 200_000) as usize);
    let first = r.next_int_bounded(10) as usize;
    let interval = next_int(r, 10, 20) as usize;
    for i in (first..set.len()).step_by(interval) {
        set.set(i);
    }
    let c = set.cardinality() as i64;
    assert!((c - set.approximate_cardinality() as i64).abs() <= c / 20);
}

#[test]
fn test_small() {
    let mut t = LuceneTestCase::new("test_small");
    let iters = if t.nightly() { t.at_least(1000) } else { 100 };
    let max = t.at_least(1200);
    do_random_sets(&mut t, max, iters);
}

#[test]
fn test_equals() {
    let mut t = LuceneTestCase::new("test_equals");
    let num_bits = t.random().next_int_bounded(2000) as usize + 1;
    let mut b1 = FixedBitSet::new(num_bits);
    let mut b2 = FixedBitSet::new(num_bits);
    assert!(b1 == b2 && b2 == b1);
    for _ in 0..10 * t.multiplier() {
        let idx = t.random().next_int_bounded(num_bits as i32) as usize;
        if !b1.get(idx) {
            b1.set(idx);
            assert!(b1 != b2 && b2 != b1);
            b2.set(idx);
            assert!(b1 == b2 && b2 == b1);
        }
    }
}

#[test]
fn test_hash_code_equals() {
    let mut t = LuceneTestCase::new("test_hash_code_equals");
    let num_bits = t.random().next_int_bounded(2000) as usize + 1;
    let mut b1 = FixedBitSet::new(num_bits);
    let mut b2 = FixedBitSet::new(num_bits);
    assert_eq!(b1, b2);
    for _ in 0..10 * t.multiplier() {
        let idx = t.random().next_int_bounded(num_bits as i32) as usize;
        if !b1.get(idx) {
            b1.set(idx);
            assert_ne!(b1, b2);
            assert_ne!(b1.java_hash_code(), b2.java_hash_code());
            b2.set(idx);
            assert_eq!(b1, b2);
            assert_eq!(b1.java_hash_code(), b2.java_hash_code());
        }
    }
}

#[test]
fn test_small_bit_sets() {
    for num_bits in 0..10 {
        let mut b1 = FixedBitSet::new(num_bits);
        let b2 = FixedBitSet::new(num_bits);
        assert_eq!(b1, b2);
        assert_eq!(b1.java_hash_code(), b2.java_hash_code());
        assert_eq!(0, b1.cardinality());
        if num_bits > 0 {
            b1.set_range(0, num_bits);
            assert_eq!(num_bits, b1.cardinality());
            b1.flip_range(0, num_bits);
            assert_eq!(0, b1.cardinality());
        }
    }
}

/// `makeFixedBitSet(a, numBits)`.
fn make_fixed_bit_set(r: &mut JavaRandom, a: &[i32], num_bits: usize) -> FixedBitSet {
    let mut bs = if r.next_boolean() {
        // Java over-allocates the word array by `nextInt(100)` words.
        let _extra_words = r.next_int_bounded(100);
        FixedBitSet::from_words(vec![0; bits2words(num_bits)], num_bits)
    } else {
        FixedBitSet::new(num_bits)
    };
    for &e in a {
        bs.set(e as usize);
    }
    bs
}

/// `makeBitSet(a)`.
fn make_bit_set(a: &[i32], num_bits: usize) -> JavaUtilBitSet {
    let mut bs = JavaUtilBitSet::new(num_bits);
    for &e in a {
        BitSet::set(&mut bs, e as usize);
    }
    bs
}

#[test]
fn test_prev_set_bit() {
    let mut t = LuceneTestCase::new("test_prev_set_bit");
    for (a, n) in [(&[][..], 0), (&[0][..], 1), (&[0, 2][..], 3)] {
        let r = t.random();
        let obs = make_fixed_bit_set(r, a, n);
        do_prev_set_bit(r, &make_bit_set(a, n), &obs);
    }
}

#[test]
fn test_next_bit_set() {
    let mut t = LuceneTestCase::new("test_next_bit_set");
    let r = t.random();
    let len = r.next_int_bounded(1000);
    let set_bits: Vec<i32> = (0..len).map(|_| r.next_int_bounded(len)).collect();
    let n = (len + r.next_int_bounded(10)) as usize;
    do_next_set_bit(
        &make_bit_set(&set_bits, n),
        &make_fixed_bit_set(r, &set_bits, n),
    );
    let n = (len + r.next_int_bounded(10)) as usize;
    do_next_set_bit(&make_bit_set(&[], n), &make_fixed_bit_set(r, &[], n));
}

#[test]
fn test_ensure_capacity() {
    let mut bits = FixedBitSet::new(5);
    bits.set(1);
    bits.set(4);
    let mut new_bits = FixedBitSet::ensure_capacity(bits.clone(), 8); // grow within the word
    assert!(new_bits.get(1) && new_bits.get(4));
    new_bits.clear(1);
    assert!(bits.get(1));
    assert!(!new_bits.get(1));

    new_bits.set(1);
    let len = new_bits.len();
    let new_bits = FixedBitSet::ensure_capacity(new_bits, len - 2); // reuse
    assert!(new_bits.get(1));

    bits.set(1);
    let mut new_bits = FixedBitSet::ensure_capacity(bits.clone(), 72); // beyond one word
    assert!(new_bits.get(1) && new_bits.get(4));
    new_bits.clear(1);
    assert!(bits.get(1));
    assert!(!new_bits.get(1));
}

#[test]
fn test_bits2words() {
    for (bits, words) in [
        (0, 0),
        (1, 1),
        (64, 1),
        (65, 2),
        (128, 2),
        (129, 3),
        (65536, 1024),
        (65537, 1025),
        (i32::MAX as usize, 1 << (31 - 6)),
    ] {
        assert_eq!(words, bits2words(bits), "bits2words({bits})");
    }
}

fn make_int_array(r: &mut JavaRandom, count: i32, min: i32, max: i32) -> Vec<i32> {
    (0..count).map(|_| next_int(r, min, max)).collect()
}

/// The draws `testIntersectionCount`, `testUnionCount` and
/// `testAndNotCount` share: two sets and their models.
fn two_sets(r: &mut JavaRandom) -> (FixedBitSet, FixedBitSet, JavaUtilBitSet, JavaUtilBitSet) {
    let n1 = next_int(r, 1000, 2000);
    let n2 = next_int(r, 1000, 2000);
    let c1 = next_int(r, 0, n1 - 1);
    let c2 = next_int(r, 0, n2 - 1);
    let bits1 = make_int_array(r, c1, 0, n1 - 1);
    let bits2 = make_int_array(r, c2, 0, n2 - 1);
    let f1 = make_fixed_bit_set(r, &bits1, n1 as usize);
    let f2 = make_fixed_bit_set(r, &bits2, n2 as usize);
    let width = n1.max(n2) as usize;
    (
        f1,
        f2,
        make_bit_set(&bits1, width),
        make_bit_set(&bits2, width),
    )
}

#[test]
fn test_intersection_count() {
    let mut t = LuceneTestCase::new("test_intersection_count");
    let (f1, f2, mut s1, s2) = two_sets(t.random());
    let count = FixedBitSet::intersection_count(&f1, &f2);
    s1.and(&s2);
    assert_eq!(BitSet::cardinality(&s1), count);
}

#[test]
fn test_union_count() {
    let mut t = LuceneTestCase::new("test_union_count");
    let (f1, f2, mut s1, s2) = two_sets(t.random());
    let count = FixedBitSet::union_count(&f1, &f2);
    s1.or(&s2);
    assert_eq!(BitSet::cardinality(&s1), count);
}

#[test]
fn test_and_not_count() {
    let mut t = LuceneTestCase::new("test_and_not_count");
    let (f1, f2, mut s1, s2) = two_sets(t.random());
    let count = FixedBitSet::and_not_count(&f1, &f2);
    s1.and_not(&s2);
    assert_eq!(BitSet::cardinality(&s1), count);
}

#[test]
fn test_and_not() {
    let mut t = LuceneTestCase::new("test_and_not");
    let r = t.random();
    let n2 = next_int(r, 1000, 2000);
    let n1 = next_int(r, 1000, n2);
    let c1 = next_int(r, 0, n1 - 1);
    let c2 = next_int(r, 0, n2 - 1);
    let min = next_int(r, 0, n1 - 1);
    let bits1 = make_int_array(r, c1, min, n1 - 1);
    let bits2 = make_int_array(r, c2, 0, n2 - 1);
    let s1 = make_bit_set(&bits1, n2 as usize);
    let mut s2 = make_bit_set(&bits2, n2 as usize);
    s2.and_not(&s1);
    // Java's three iterator shapes (`BitSetIterator`, `DocBaseBitSetIterator`
    // at a word offset, `IntArrayDocIdSetIterator`) each make one
    // `makeFixedBitSet` for `bits2` and, for the first two, one for `bits1`.
    for shape in 0..3 {
        let mut f2 = make_fixed_bit_set(r, &bits2, n2 as usize);
        if shape < 2 {
            let offset = if shape == 1 { (min >> 6) << 6 } else { 0 };
            let shifted: Vec<i32> = bits1.iter().map(|&b| b - offset).collect();
            let f1 = make_fixed_bit_set(r, &shifted, (n1 - offset) as usize);
            let mut unshifted = FixedBitSet::new(n1 as usize);
            f1.for_each_in_range(0, f1.len(), offset, |d| unshifted.set(d as usize));
            f2.and_not(&unshifted);
        } else {
            f2.and_not(&make_bit_set_fbs(&bits1, n1 as usize));
        }
        do_get(&s2, &f2);
    }
}

fn make_bit_set_fbs(a: &[i32], num_bits: usize) -> FixedBitSet {
    let mut bs = FixedBitSet::new(num_bits);
    a.iter().for_each(|&e| bs.set(e as usize));
    bs
}

#[test]
fn test_copy_of() {
    let mut t = LuceneTestCase::new("test_copy_of");
    let r = t.random();
    let num_bits = next_int(r, 1000, 2000);
    let count = next_int(r, 0, num_bits - 1);
    let bits = make_int_array(r, count, 0, num_bits - 1);
    let fixed = make_bit_set_fbs(&bits, num_bits as usize);
    let copy = FixedBitSet::copy_of(fixed.len(), |i| fixed.get(i));
    assert_eq!(copy, fixed);
    // Through an anonymous `Bits`.
    let as_bits: &dyn Bits = &fixed;
    let copy = FixedBitSet::copy_of(as_bits.length(), |i| as_bits.get(i));
    assert_eq!(copy, fixed);
}

#[test]
fn test_scan_is_empty() {
    let mut set = FixedBitSet::new(0);
    assert!(set.scan_is_empty());
    for (n, bits) in [
        (13, &[10][..]),
        (1024, &[3, 1020][..]),
        (1030, &[3, 1028][..]),
    ] {
        set = FixedBitSet::new(n);
        assert!(set.scan_is_empty());
        let mut last = None;
        for &b in bits {
            if let Some(l) = last {
                set.clear(l);
            }
            set.set(b);
            assert!(!set.scan_is_empty());
            last = Some(b);
        }
    }
}

fn range_test(t: &mut LuceneTestCase, every: usize, and: bool) {
    let mut dest = FixedBitSet::new(1_000);
    let mut source = FixedBitSet::new(10_000);
    for i in (0..source.len()).step_by(3) {
        source.set(i);
    }
    for source_from in 64..128 {
        for dest_from in 256..320 {
            let lengths = [
                0,
                next_int(t.random(), 1, 63),
                next_int(t.random(), 64, 512),
            ];
            for length in lengths.map(|l| l as usize) {
                dest.clear_all();
                for i in (0..dest.len()).step_by(every) {
                    dest.set(i);
                }
                if and {
                    FixedBitSet::and_range(&source, source_from, &mut dest, dest_from, length);
                } else {
                    FixedBitSet::or_range(&source, source_from, &mut dest, dest_from, length);
                }
                for i in 0..dest.len() {
                    let dest_set = i % every == 0;
                    let want = if i < dest_from || i >= dest_from + length {
                        dest_set
                    } else {
                        let source_set = source.get(source_from + (i - dest_from));
                        if and {
                            source_set && dest_set
                        } else {
                            source_set || dest_set
                        }
                    };
                    assert_eq!(want, dest.get(i), "{i}");
                }
            }
        }
    }
}

#[test]
fn test_or_range() {
    range_test(&mut LuceneTestCase::new("test_or_range"), 10, false);
}

#[test]
fn test_and_range() {
    range_test(&mut LuceneTestCase::new("test_and_range"), 2, true);
}

fn slow_cardinality(set: &FixedBitSet, from: usize, to: usize) -> usize {
    (from..to).filter(|&i| set.get(i)).count()
}

/// A set of `nextInt(64, 4096)` bits with every `nextInt(1, 16)`th set.
fn stepped_set(r: &mut JavaRandom) -> FixedBitSet {
    let mut set = FixedBitSet::new(next_int(r, 64, 4096) as usize);
    let mut i = 0;
    while i < set.len() {
        set.set(i);
        i += next_int(r, 1, 16) as usize;
    }
    set
}

#[test]
fn test_range_cardinality() {
    let mut t = LuceneTestCase::new("test_range_cardinality");
    assert_eq!(0, FixedBitSet::new(0).cardinality_range(0, 0));
    let set = stepped_set(t.random());
    let n = set.len();
    assert_eq!(0, set.cardinality_range(0, 0));
    assert_eq!(0, set.cardinality_range(n, n));
    for (from, to) in [(0, 1), (n - 1, n), (0, 63), (n - 63, n), (1, n - 1)] {
        assert_eq!(
            slow_cardinality(&set, from, to),
            set.cardinality_range(from, to)
        );
    }
}

#[test]
fn test_for_each() {
    let mut t = LuceneTestCase::new("test_for_each");
    let r = t.random();
    let set = stepped_set(r);
    for _ in 0..100 {
        let from = next_int(r, 0, set.len() as i32) as usize;
        let to = next_int(r, from as i32, set.len() as i32) as usize;
        let base = next_int(r, 0, 100);
        let expected: Vec<i32> = (from..to)
            .filter(|&i| set.get(i))
            .map(|i| base + i as i32)
            .collect();
        let mut actual = Vec::new();
        set.for_each_in_range(from, to, base, |d| actual.push(d));
        assert_eq!(expected, actual);
    }
}

#[test]
fn test_into_array() {
    let mut t = LuceneTestCase::new("test_into_array");
    let r = t.random();
    for _ in 0..100 {
        let num_bits = next_int(r, 10, 1_000);
        let num_set = next_int(r, 0, num_bits) as usize;
        let mut set = FixedBitSet::new(num_bits as usize);
        while set.cardinality() < num_set {
            set.set(r.next_int_bounded(num_bits) as usize);
        }
        let mut array = vec![0; next_int(r, 0, num_set as i32 + 10) as usize];
        let (from, to) = if r.next_boolean() {
            (0, set.len())
        } else {
            let from = r.next_int_bounded(num_bits) as usize;
            (
                from,
                from + r.next_int_bounded(num_bits - from as i32) as usize,
            )
        };
        let base = r.next_int_bounded(1000);
        let size = set.into_array(from, to, base, &mut array);
        if size < array.len() {
            assert_eq!(set.cardinality_range(from, to), size);
        }
        let mut index = 0;
        set.for_each_in_range(from, to, base, |bit| {
            if index < array.len() {
                assert_eq!(bit, array[index]);
                index += 1;
            }
        });
        assert_eq!(size, index);
    }
}

#[test]
fn test_or_mask() {
    let mut t = LuceneTestCase::new("test_or_mask");
    let r = t.random();
    for mask_len in [2usize, 4, 8] {
        for start_bit in 0..128 {
            let mask = (r.next_long() as u64) & ((1u64 << mask_len) - 1);
            let mut fast = FixedBitSet::new(256);
            let mut slow = FixedBitSet::new(256);
            for i in (0..256).step_by(7) {
                fast.set(i);
                slow.set(i);
            }
            fast.or_mask(start_bit, mask, mask_len);
            let mut m = mask;
            while m != 0 {
                slow.set(start_bit + m.trailing_zeros() as usize);
                m &= m - 1;
            }
            assert_eq!(
                slow, fast,
                "orMask mismatch: startBit={start_bit}, mask={mask:#b}, maskLen={mask_len}"
            );
        }
    }
}

#[test]
fn test_or_mask_straddling() {
    for mask_len in [4usize, 8] {
        for bit_offset in 64 - mask_len + 1..64 {
            let mut fast = FixedBitSet::new(128);
            fast.or_mask(bit_offset, (1u64 << mask_len) - 1, mask_len);
            let mut slow = FixedBitSet::new(128);
            for i in 0..mask_len {
                slow.set(bit_offset + i);
            }
            assert_eq!(
                slow, fast,
                "Straddling mismatch: startBit={bit_offset}, maskLen={mask_len}"
            );
        }
    }
}
