//! `TestCheckIndex`'s `BaseTestCheckIndex` half, over a heap directory and
//! a `MockDirectoryWrapper`.

use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::index::base_test_check_index::{
    test_checksums_only, test_checksums_only_verbose, test_deleted_docs, test_obtains_lock,
};
use lucene_test_framework::store::mock_directory_wrapper::MockDirectoryWrapper;
use lucene_util::java_random::JavaRandom;

fn mock(seed: i64) -> MockDirectoryWrapper {
    MockDirectoryWrapper::new(&mut JavaRandom::new(seed), ByteBuffersDirectory::new())
}

#[test]
fn deleted_docs() {
    for seed in 0..3 {
        test_deleted_docs(&mut JavaRandom::new(seed), &ByteBuffersDirectory::new());
        test_deleted_docs(&mut JavaRandom::new(seed), &mock(seed));
    }
}

#[test]
fn checksums_only() {
    for seed in 0..3 {
        test_checksums_only(&mut JavaRandom::new(seed), &ByteBuffersDirectory::new());
        test_checksums_only_verbose(&mut JavaRandom::new(seed), &mock(seed));
    }
}

#[test]
fn obtains_lock() {
    test_obtains_lock(&ByteBuffersDirectory::new());
    test_obtains_lock(&mock(7));
}
