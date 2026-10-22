//! `TestLucene90TermVectorsFormat`: `BaseTermVectorsFormatTestCase` and
//! the `BaseIndexFileFormatTestCase` tests it inherits, over the default
//! codec, a few seeds each.

use lucene_test_framework::index::base_index_file_format::SuiteTest;
use lucene_test_framework::index::base_index_file_format::{
    test_check_integrity_reads_all_bytes, test_merge_stability, test_random_exceptions,
};
use lucene_test_framework::index::base_term_vectors_format::{self as tv, TermVectorsSuite};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

const SEEDS: [i64; 2] = [0x7EC7, -0x0FED_CBA9_8765];

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
                run($name, tv::$test);
            }
        )*
    };
}

suite_tests! {
    test_rare_vectors => "testRareVectors",
    test_high_freqs => "testHighFreqs",
    test_lots_of_fields => "testLotsOfFields",
    test_mixed_options => "testMixedOptions",
    test_random => "testRandom",
    test_merge => "testMerge",
    test_merge_with_deletes => "testMergeWithDeletes",
    test_merge_with_index_sort => "testMergeWithIndexSort",
    test_merge_with_index_sort_and_deletes => "testMergeWithIndexSortAndDeletes",
    test_clone => "testClone",
    test_postings_enum_freqs => "testPostingsEnumFreqs",
    test_postings_enum_positions => "testPostingsEnumPositions",
    test_postings_enum_offsets => "testPostingsEnumOffsets",
    test_postings_enum_offsets_without_positions => "testPostingsEnumOffsetsWithoutPositions",
    test_postings_enum_payloads => "testPostingsEnumPayloads",
    test_postings_enum_all => "testPostingsEnumAll",
}

/// The suite names this file's `suite_tests!` invocations run, from its own
/// source.
fn run_names() -> String {
    let source = include_str!("test_lucene90_term_vectors_format.rs");
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
    assert_eq!(unrun(tv::TESTS), [""; 0]);
}

#[test]
fn nightly_sizes_run_too() {
    let mut t = LuceneTestCase::with_config("testLotsOfFields", 5, true, 1);
    tv::test_lots_of_fields(&mut t);
}

#[test]
fn merge_stability() {
    run("testMergeStability", |t| {
        test_merge_stability(&TermVectorsSuite, t)
    });
}

#[test]
fn random_exceptions() {
    let mut injected = 0;
    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config("testRandomExceptions", seed, false, 1);
        injected += test_random_exceptions(&TermVectorsSuite, &mut t);
    }
    assert!(injected > 0, "no error was injected");
}

#[test]
fn check_integrity_reads_all_bytes() {
    run("testCheckIntegrityReadsAllBytes", |t| {
        test_check_integrity_reads_all_bytes(&TermVectorsSuite, t);
    });
}
