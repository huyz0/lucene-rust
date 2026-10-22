//! `TestLucene90PointsFormat`: `BasePointsFormatTestCase` and the
//! `BaseIndexFileFormatTestCase` tests it inherits, a few seeds each.

use lucene_test_framework::index::base_index_file_format::SuiteTest;
use lucene_test_framework::index::base_index_file_format::{
    test_check_integrity_reads_all_bytes, test_random_exceptions,
};
use lucene_test_framework::index::base_points_format::{self as pf, PointsSuite};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

const SEEDS: [i64; 2] = [0xB4D7, -0x0B4D_7EE0_5150];

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
                run($name, pf::$test);
            }
        )*
    };
}

suite_tests! {
    test_basic => "testBasic",
    test_merge => "testMerge",
    test_all_point_docs_deleted_in_segment => "testAllPointDocsDeletedInSegment",
    test_multi_valued => "testMultiValued",
    test_all_equal => "testAllEqual",
    test_one_dim_equal => "testOneDimEqual",
    test_one_dim_two_values => "testOneDimTwoValues",
    test_big_int_n_dims => "testBigIntNDims",
    test_random_binary_tiny => "testRandomBinaryTiny",
    test_random_binary_medium => "testRandomBinaryMedium",
    test_add_indexes => "testAddIndexes",
    test_merge_missing => "testMergeMissing",
    test_doc_count_edge_cases => "testDocCountEdgeCases",
    test_random_doc_count => "testRandomDocCount",
    test_mismatched_fields => "testMismatchedFields",
}

/// The suite names this file's `suite_tests!` invocations run, from its own
/// source.
fn run_names() -> String {
    let source = include_str!("test_lucene90_points_format.rs");
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
    assert_eq!(unrun(pf::TESTS), [""; 0]);
}

#[test]
fn random_exceptions() {
    // Gate rule #10: the injected errors must actually fire.
    let mut injected = 0;
    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config("testRandomExceptions", seed, false, 1);
        injected += test_random_exceptions(&PointsSuite, &mut t);
    }
    assert!(injected > 0, "no error was injected");
}

#[test]
fn check_integrity_reads_all_bytes() {
    run("testCheckIntegrityReadsAllBytes", |t| {
        test_check_integrity_reads_all_bytes(&PointsSuite, t);
    });
}

#[test]
fn with_exceptions() {
    // Gate rule #10: each seed's loop ends only on an injected error, and
    // panics when 100 runs saw none.
    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config("testWithExceptions", seed, false, 1);
        pf::test_with_exceptions(&mut t);
    }
}

/// `@Nightly testRandomBinaryBig` at a size a test run affords.
#[test]
fn random_binary_big() {
    let mut t = LuceneTestCase::with_config("testRandomBinaryBig", 3, false, 1);
    pf::random_binary(&mut t, 10_001);
}
