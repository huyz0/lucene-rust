//! `TestLucene94FieldInfosFormat`: `BaseFieldInfoFormatTestCase` and the
//! `BaseIndexFileFormatTestCase` tests it inherits, a few seeds each.

use lucene_test_framework::index::base_field_info_format::{self as fi, FieldInfoSuite};
use lucene_test_framework::index::base_index_file_format::SuiteTest;
use lucene_test_framework::index::base_index_file_format::{
    test_check_integrity_reads_all_bytes, test_merge_stability, test_random_exceptions,
};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

const SEEDS: [i64; 2] = [0xF1E1, -0x0F1E_1D00_4242];

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
                run($name, fi::$test);
            }
        )*
    };
}

suite_tests! {
    test_one_field => "testOneField",
    test_immutable_attributes => "testImmutableAttributes",
    test_exception_on_create_output => "testExceptionOnCreateOutput",
    test_exception_on_close_output => "testExceptionOnCloseOutput",
    test_exception_on_open_input => "testExceptionOnOpenInput",
    test_exception_on_close_input => "testExceptionOnCloseInput",
    test_random => "testRandom",
}

/// The suite names this file's `suite_tests!` invocations run, from its own
/// source.
fn run_names() -> String {
    let source = include_str!("test_lucene94_field_infos_format.rs");
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
    assert_eq!(unrun(fi::TESTS), [""; 0]);
}

#[test]
fn merge_stability() {
    run("testMergeStability", |t| {
        test_merge_stability(&FieldInfoSuite, t)
    });
}

#[test]
fn random_exceptions() {
    // Gate rule #10: the injected errors must actually fire.
    let mut injected = 0;
    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config("testRandomExceptions", seed, false, 1);
        injected += test_random_exceptions(&FieldInfoSuite, &mut t);
    }
    assert!(injected > 0, "no error was injected");
}

#[test]
fn check_integrity_reads_all_bytes() {
    run("testCheckIntegrityReadsAllBytes", |t| {
        test_check_integrity_reads_all_bytes(&FieldInfoSuite, t);
    });
}
