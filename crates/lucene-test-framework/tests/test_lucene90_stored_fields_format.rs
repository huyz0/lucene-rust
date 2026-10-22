//! `TestLucene90StoredFieldsFormat`: `BaseStoredFieldsFormatTestCase` and
//! the `BaseIndexFileFormatTestCase` tests it inherits, over the default
//! codec, a few seeds each.

use lucene_test_framework::index::base_index_file_format::SuiteTest;
use lucene_test_framework::index::base_index_file_format::{
    test_check_integrity_reads_all_bytes, test_merge_stability, test_random_exceptions,
};
use lucene_test_framework::index::base_stored_fields_format::{self as sf, StoredFieldsSuite};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

const SEEDS: [i64; 2] = [0x5EED, -0x1234_5678_9ABC];

fn run(name: &str, f: fn(&mut LuceneTestCase)) {
    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config(name, seed, false, 1);
        f(&mut t);
    }
}

macro_rules! suite_tests {
    ($($test:ident => $name:literal),* $(,)?) => {
        $(
            #[test]
            fn $test() {
                run($name, sf::$test);
            }
        )*
    };
}

suite_tests! {
    test_random_stored_fields => "testRandomStoredFields",
    test_stored_fields_order => "testStoredFieldsOrder",
    test_binary_field_offset_length => "testBinaryFieldOffsetLength",
    test_numeric_field => "testNumericField",
    test_indexed_bit => "testIndexedBit",
    test_read_skip => "testReadSkip",
    test_empty_docs => "testEmptyDocs",
    test_concurrent_reads => "testConcurrentReads",
    test_write_read_merge => "testWriteReadMerge",
    test_merge_filter_reader => "testMergeFilterReader",
    test_big_documents => "testBigDocuments",
    test_bulk_merge_with_deletes => "testBulkMergeWithDeletes",
    test_mismatched_fields => "testMismatchedFields",
    test_random_stored_fields_with_index_sort => "testRandomStoredFieldsWithIndexSort",
    test_line_file_docs => "testLineFileDocs",
}

/// The suite names this file's `suite_tests!` invocations run, from its own
/// source.
fn run_names() -> String {
    let source = include_str!("test_lucene90_stored_fields_format.rs");
    source
        .split("suite_tests! {")
        .skip(1)
        .filter_map(|part| {
            part.split(
                "
}",
            )
            .next()
        })
        .collect()
}

/// `tests` names that no `suite_tests!` entry runs.
fn unrun(tests: &[SuiteTest]) -> Vec<&'static str> {
    let runs = run_names();
    tests
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| !runs.contains(&format!("\"{name}\"")))
        .collect()
}

#[test]
fn every_suite_test_has_a_runner() {
    assert_eq!(unrun(sf::TESTS), [""; 0]);
}

/// `@Nightly` sizes are out of reach here; the same code at a size still
/// many times a chunk.
#[test]
fn big_documents_at_a_smaller_size() {
    let mut t = LuceneTestCase::with_config("testBigDocuments", 3, false, 1);
    sf::big_documents(&mut t, (5_000, 10_000), (200_000, 400_000));
}

#[test]
fn merge_stability() {
    run("testMergeStability", |t| {
        test_merge_stability(&StoredFieldsSuite, t)
    });
}

#[test]
fn random_exceptions() {
    // Gate rule #10: the injected errors must actually fire.
    let mut injected = 0;
    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config("testRandomExceptions", seed, false, 1);
        injected += test_random_exceptions(&StoredFieldsSuite, &mut t);
    }
    assert!(injected > 0, "no error was injected");
}

#[test]
fn check_integrity_reads_all_bytes() {
    run("testCheckIntegrityReadsAllBytes", |t| {
        test_check_integrity_reads_all_bytes(&StoredFieldsSuite, t);
    });
}
