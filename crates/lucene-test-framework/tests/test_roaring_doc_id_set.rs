//! `TestRoaringDocIdSet`: `BaseDocIdSetTestCase` over `RoaringDocIdSet`,
//! with Java's `assertEquals` override (cardinalities equal).
//!
//! Not ported: the `docIDRunEnd` tests and the `Builder.add(min, max)`
//! range tests -- the Rust iterator has no `docIDRunEnd` and the builder no
//! range add.

use lucene_search::docid_set::{RoaringBuilder, RoaringDocIdSet};
use lucene_test_framework::util::base_bit_set::JavaUtilBitSet;
use lucene_test_framework::util::base_doc_id_set::BaseDocIdSetTestCase;
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_util::bits::BitSet;

/// `copyOf(bs, length)`.
fn copy_of(bs: &JavaUtilBitSet, length: usize) -> RoaringDocIdSet {
    let mut builder = RoaringBuilder::new(length);
    let mut i = bs.next_set_bit_any(0);
    while let Some(d) = i {
        builder.add(d as i32);
        i = bs.next_set_bit_any(d + 1);
    }
    builder.build()
}

fn same_cardinality(ds1: &JavaUtilBitSet, ds2: &RoaringDocIdSet) {
    assert_eq!(BitSet::cardinality(ds1), ds2.cardinality());
}

type Copy = fn(&JavaUtilBitSet, usize) -> RoaringDocIdSet;

macro_rules! base_tests {
    ($($name:ident),*) => {$(
        #[test]
        fn $name() {
            BaseDocIdSetTestCase::new(copy_of as Copy)
                .with_extra_check(same_cardinality)
                .$name(&mut LuceneTestCase::new(stringify!($name)));
        }
    )*};
}

base_tests!(
    test_no_bit,
    test_1_bit,
    test_2_bits,
    test_against_bit_set,
    test_ram_bytes_used,
    test_into_bit_set,
    test_into_bit_set_bound_checks
);
