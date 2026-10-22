//! `TestLucene104PostingsFormat`'s inherited half: `BasePostingsFormatTestCase`
//! and the `BaseIndexFileFormatTestCase` tests it inherits, over the default
//! codec, a few seeds each.

use lucene_test_framework::index::base_index_file_format::SuiteTest;
use lucene_test_framework::index::base_index_file_format::{
    test_check_integrity_reads_all_bytes, test_merge_stability, test_random_exceptions,
};
use lucene_test_framework::index::base_postings_format::{self as pf, PostingsSuite};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

const SEEDS: [i64; 2] = [0x9057, -0x0DDB_A115_EED0];

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
    test_docs_only => "testDocsOnly",
    test_docs_and_freqs => "testDocsAndFreqs",
    test_docs_and_freqs_and_positions => "testDocsAndFreqsAndPositions",
    test_docs_and_freqs_and_positions_and_payloads => "testDocsAndFreqsAndPositionsAndPayloads",
    test_docs_and_freqs_and_positions_and_offsets => "testDocsAndFreqsAndPositionsAndOffsets",
    test_docs_and_freqs_and_positions_and_offsets_and_payloads =>
        "testDocsAndFreqsAndPositionsAndOffsetsAndPayloads",
    test_random => "testRandom",
    test_postings_enum_reuse => "testPostingsEnumReuse",
    test_just_empty_field => "testJustEmptyField",
    test_empty_field_and_empty_term => "testEmptyFieldAndEmptyTerm",
    test_didnt_want_freqs_but_asked_anyway => "testDidntWantFreqsButAskedAnyway",
    test_ask_for_positions_when_not_there => "testAskForPositionsWhenNotThere",
    test_ghosts => "testGhosts",
    test_disorder => "testDisorder",
    test_binary_search_term_leaf => "testBinarySearchTermLeaf",
    test_level2_ghosts => "testLevel2Ghosts",
    test_postings_enum_docs_only => "testPostingsEnumDocsOnly",
    test_postings_enum_freqs => "testPostingsEnumFreqs",
    test_postings_enum_positions => "testPostingsEnumPositions",
    test_postings_enum_offsets => "testPostingsEnumOffsets",
    test_postings_enum_payloads => "testPostingsEnumPayloads",
    test_postings_enum_all => "testPostingsEnumAll",
    test_line_file_docs => "testLineFileDocs",
    test_mismatched_fields => "testMismatchedFields",
}

/// The suite names this file's `suite_tests!` invocations run, from its own
/// source.
fn run_names() -> String {
    let source = include_str!("test_lucene104_postings_format.rs");
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
    // `testDocIDRunEnd` runs below, at a smaller size.
    assert_eq!(unrun(pf::TESTS), ["testDocIDRunEnd"]);
}

/// `testDocIDRunEnd`'s 100 indexes of up to 25 000 documents, cut to a few.
#[test]
fn doc_id_run_end_at_fewer_iterations() {
    let mut t = LuceneTestCase::with_config("testDocIDRunEnd", 11, false, 1);
    pf::doc_id_run_end(&mut t, 3);
}

/// `@Nightly` `testLineFileDocs`, at a size a default run affords.
#[test]
fn line_file_docs_at_a_smaller_size() {
    let mut t = LuceneTestCase::with_config("testLineFileDocs", 7, false, 1);
    pf::line_file_docs(&mut t, 500);
}

#[test]
fn merge_stability() {
    run("testMergeStability", |t| {
        test_merge_stability(&PostingsSuite, t)
    });
}

#[test]
fn random_exceptions() {
    // Gate rule #10: the injected errors must actually fire.
    let mut injected = 0;
    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config("testRandomExceptions", seed, false, 1);
        injected += test_random_exceptions(&PostingsSuite, &mut t);
    }
    assert!(injected > 0, "no error was injected");
}

#[test]
fn check_integrity_reads_all_bytes() {
    run("testCheckIntegrityReadsAllBytes", |t| {
        test_check_integrity_reads_all_bytes(&PostingsSuite, t);
    });
}
