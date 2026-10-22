//! `TestLucene90NormsFormat`: `BaseNormsFormatTestCase` and the
//! `BaseIndexFileFormatTestCase` tests it inherits, a few seeds each.

use lucene_test_framework::index::base_index_file_format::SuiteTest;
use lucene_test_framework::index::base_index_file_format::{
    test_check_integrity_reads_all_bytes, test_random_exceptions,
};
use lucene_test_framework::index::base_norms_format::{self as nf, NormsSuite};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

const SEEDS: [i64; 2] = [0x7055, -0x0A0B_0C0D_0E0F];

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
                run($name, nf::$test);
            }
        )*
    };
}

suite_tests! {
    test_byte_range => "testByteRange",
    test_sparse_byte_range => "testSparseByteRange",
    test_short_range => "testShortRange",
    test_sparse_short_range => "testSparseShortRange",
    test_long_range => "testLongRange",
    test_sparse_long_range => "testSparseLongRange",
    test_full_long_range => "testFullLongRange",
    test_sparse_full_long_range => "testSparseFullLongRange",
    test_few_values => "testFewValues",
    test_few_sparse_values => "testFewSparseValues",
    test_few_large_values => "testFewLargeValues",
    test_few_sparse_large_values => "testFewSparseLargeValues",
    test_all_zeros => "testAllZeros",
    test_sparse_all_zeros => "testSparseAllZeros",
    test_most_zeros => "testMostZeros",
    test_outliers => "testOutliers",
    test_sparse_outliers => "testSparseOutliers",
    test_outliers2 => "testOutliers2",
    test_sparse_outliers2 => "testSparseOutliers2",
    test_n_common => "testNCommon",
    test_sparse_n_common => "testSparseNCommon",
    test_undead_norms => "testUndeadNorms",
    test_threads => "testThreads",
    test_independant_iterators => "testIndependantIterators",
    test_independant_sparse_iterators => "testIndependantSparseIterators",
}

/// The suite names this file's `suite_tests!` invocations run, from its own
/// source.
fn run_names() -> String {
    let source = include_str!("test_lucene90_norms_format.rs");
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
    assert_eq!(unrun(nf::TESTS), [""; 0]);
}

#[test]
fn random_exceptions() {
    // Gate rule #10: the injected errors must actually fire.
    let mut injected = 0;
    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config("testRandomExceptions", seed, false, 1);
        injected += test_random_exceptions(&NormsSuite, &mut t);
    }
    assert!(injected > 0, "no error was injected");
}

#[test]
fn check_integrity_reads_all_bytes() {
    run("testCheckIntegrityReadsAllBytes", |t| {
        test_check_integrity_reads_all_bytes(&NormsSuite, t);
    });
}

/// `@Nightly testNCommonBig` and `testSparseNCommonBig`, one seed each.
#[test]
fn n_common_big() {
    for sparse in [false, true] {
        let mut t = LuceneTestCase::with_config("testNCommonBig", 5, false, 1);
        nf::n_common_big(&mut t, sparse);
    }
}

#[test]
fn norms_are_compared() {
    // Gate rule #10: a check over leaves with no norms passes vacuously.
    let mut t = LuceneTestCase::with_config("testByteRange", 11, false, 1);
    let compared = nf::norms_versus_doc_values(&mut t, 1.0, &mut nf::ranged(-128, 127));
    assert!(compared > 0, "no norm was compared");
}
