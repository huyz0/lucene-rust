//! `TestIntArrayDocIdSet`: `BaseDocIdSetTestCase` over `IntArrayDocIdSet`
//! (`DocIdSet::Array`; Java's trailing `NO_MORE_DOCS` sentinel is implicit
//! in the Rust iterator).

use lucene_search::facets::DocIdSet;
use lucene_test_framework::util::base_bit_set::{set_bits, JavaUtilBitSet};
use lucene_test_framework::util::base_doc_id_set::BaseDocIdSetTestCase;
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

/// `copyOf(bs, length)`.
fn copy_of(bs: &JavaUtilBitSet, _length: usize) -> DocIdSet {
    DocIdSet::Array(set_bits(bs))
}

type Copy = fn(&JavaUtilBitSet, usize) -> DocIdSet;

macro_rules! base_tests {
    ($($name:ident),*) => {$(
        #[test]
        fn $name() {
            BaseDocIdSetTestCase::new(copy_of as Copy).$name(&mut LuceneTestCase::new(stringify!($name)));
        }
    )*};
}

base_tests!(test_no_bit, test_1_bit, test_2_bits, test_against_bit_set);
