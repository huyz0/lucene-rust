//! `TestLucene90LiveDocsFormat`: `BaseLiveDocsFormatTestCase`, a few seeds
//! each.

use lucene_test_framework::index::base_index_file_format::SuiteTest;
use lucene_test_framework::index::base_live_docs_format::{self as ld};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

const SEEDS: [i64; 2] = [0x11FE, -0x0D0C_5A11_0042];

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
                run($name, ld::$test);
            }
        )*
    };
}

suite_tests! {
    test_dense_live_docs => "testDenseLiveDocs",
    test_empty_live_docs => "testEmptyLiveDocs",
    test_sparse_live_docs => "testSparseLiveDocs",
}

/// The suite names this file's `suite_tests!` invocations run, from its own
/// source.
fn run_names() -> String {
    let source = include_str!("test_lucene90_live_docs_format.rs");
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
    assert_eq!(unrun(ld::TESTS), [""; 0]);
}

/// `@Monster testOverflow` at a size a test run affords: the same code,
/// past several 64-bit words and an odd tail.
#[test]
fn overflow_at_a_smaller_size() {
    let mut t = LuceneTestCase::with_config("testOverflow", 7, false, 1);
    ld::overflow(&mut t, 1 << 20);
}
