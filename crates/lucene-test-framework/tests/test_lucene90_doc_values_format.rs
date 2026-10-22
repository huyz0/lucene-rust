//! `TestLucene90DocValuesFormat`'s inherited half:
//! `BaseCompressingDocValuesFormatTestCase`, `BaseDocValuesFormatTestCase`,
//! `LegacyBaseDocValuesFormatTestCase` and the `BaseIndexFileFormatTestCase`
//! tests they inherit, over the default codec, a few seeds each. The
//! `@Nightly` tests run at smaller sizes, at the end.

use lucene_test_framework::index::base_doc_values_format as bdv;
use lucene_test_framework::index::base_index_file_format::SuiteTest;
use lucene_test_framework::index::base_index_file_format::{
    test_check_integrity_reads_all_bytes, test_merge_stability, test_random_exceptions,
};
use lucene_test_framework::index::legacy_base_doc_values_format::{self as dv, DocValuesSuite};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

const SEEDS: [i64; 2] = [0xD0C5, -0x07AB_1E5E_ED00];

fn run(name: &str, f: fn(&mut LuceneTestCase)) {
    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config(name, seed, false, 1);
        f(&mut t);
    }
}

macro_rules! suite_tests {
    ($module:ident: $($test:ident => $name:literal),* $(,)?) => {
        $(
            #[test]
            fn $test() {
                run($name, $module::$test);
            }
        )*
    };
}
suite_tests! {
    dv:
    test_one_number => "testOneNumber",
    test_one_float => "testOneFloat",
    test_two_numbers => "testTwoNumbers",
    test_two_binary_values => "testTwoBinaryValues",
    test_variously_compressible_binary_values => "testVariouslyCompressibleBinaryValues",
    test_two_fields_mixed => "testTwoFieldsMixed",
    test_three_fields_mixed => "testThreeFieldsMixed",
    test_three_fields_mixed2 => "testThreeFieldsMixed2",
    test_two_documents_numeric => "testTwoDocumentsNumeric",
    test_two_documents_merged => "testTwoDocumentsMerged",
    test_big_numeric_range => "testBigNumericRange",
    test_big_numeric_range2 => "testBigNumericRange2",
    test_bytes => "testBytes",
    test_bytes_two_documents_merged => "testBytesTwoDocumentsMerged",
    test_bytes_merge_away_all_values => "testBytesMergeAwayAllValues",
    test_sorted_bytes => "testSortedBytes",
    test_sorted_bytes_two_documents => "testSortedBytesTwoDocuments",
    test_sorted_bytes_three_documents => "testSortedBytesThreeDocuments",
    test_sorted_bytes_two_documents_merged => "testSortedBytesTwoDocumentsMerged",
    test_sorted_merge_away_all_values => "testSortedMergeAwayAllValues",
    test_bytes_with_newline => "testBytesWithNewline",
    test_missing_sorted_bytes => "testMissingSortedBytes",
    test_sorted_terms_enum => "testSortedTermsEnum",
    test_empty_sorted_bytes => "testEmptySortedBytes",
    test_empty_bytes => "testEmptyBytes",
    test_very_large_but_legal_bytes => "testVeryLargeButLegalBytes",
    test_very_large_but_legal_sorted_bytes => "testVeryLargeButLegalSortedBytes",
    test_codec_uses_own_bytes => "testCodecUsesOwnBytes",
    test_codec_uses_own_sorted_bytes => "testCodecUsesOwnSortedBytes",
    test_doc_values_simple => "testDocValuesSimple",
    test_random_sorted_bytes => "testRandomSortedBytes",
    test_boolean_numerics_vs_stored_fields => "testBooleanNumericsVsStoredFields",
    test_sparse_boolean_numerics_vs_stored_fields => "testSparseBooleanNumericsVsStoredFields",
    test_byte_numerics_vs_stored_fields => "testByteNumericsVsStoredFields",
    test_sparse_byte_numerics_vs_stored_fields => "testSparseByteNumericsVsStoredFields",
    test_short_numerics_vs_stored_fields => "testShortNumericsVsStoredFields",
    test_sparse_short_numerics_vs_stored_fields => "testSparseShortNumericsVsStoredFields",
    test_int_numerics_vs_stored_fields => "testIntNumericsVsStoredFields",
    test_sparse_int_numerics_vs_stored_fields => "testSparseIntNumericsVsStoredFields",
    test_long_numerics_vs_stored_fields => "testLongNumericsVsStoredFields",
    test_sparse_long_numerics_vs_stored_fields => "testSparseLongNumericsVsStoredFields",
    test_binary_fixed_length_vs_stored_fields => "testBinaryFixedLengthVsStoredFields",
    test_sparse_binary_fixed_length_vs_stored_fields => "testSparseBinaryFixedLengthVsStoredFields",
    test_binary_variable_length_vs_stored_fields => "testBinaryVariableLengthVsStoredFields",
    test_sparse_binary_variable_length_vs_stored_fields => "testSparseBinaryVariableLengthVsStoredFields",
    test_sorted_fixed_length_vs_stored_fields => "testSortedFixedLengthVsStoredFields",
    test_sparse_sorted_fixed_length_vs_stored_fields => "testSparseSortedFixedLengthVsStoredFields",
    test_sorted_variable_length_vs_stored_fields => "testSortedVariableLengthVsStoredFields",
    test_sparse_sorted_variable_length_vs_stored_fields => "testSparseSortedVariableLengthVsStoredFields",
    test_sorted_set_one_value => "testSortedSetOneValue",
    test_sorted_set_two_fields => "testSortedSetTwoFields",
    test_sorted_set_two_documents_merged => "testSortedSetTwoDocumentsMerged",
    test_sorted_set_two_values => "testSortedSetTwoValues",
    test_sorted_set_two_values_unordered => "testSortedSetTwoValuesUnordered",
    test_sorted_set_three_values_two_docs => "testSortedSetThreeValuesTwoDocs",
    test_sorted_set_two_documents_last_missing => "testSortedSetTwoDocumentsLastMissing",
    test_sorted_set_two_documents_last_missing_merge => "testSortedSetTwoDocumentsLastMissingMerge",
    test_sorted_set_two_documents_first_missing => "testSortedSetTwoDocumentsFirstMissing",
    test_sorted_set_two_documents_first_missing_merge => "testSortedSetTwoDocumentsFirstMissingMerge",
    test_sorted_set_merge_away_all_values => "testSortedSetMergeAwayAllValues",
    test_sorted_set_terms_enum => "testSortedSetTermsEnum",
    test_sorted_set_fixed_length_vs_stored_fields => "testSortedSetFixedLengthVsStoredFields",
    test_sorted_numerics_single_valued_vs_stored_fields => "testSortedNumericsSingleValuedVsStoredFields",
    test_sorted_numerics_single_valued_missing_vs_stored_fields => "testSortedNumericsSingleValuedMissingVsStoredFields",
    test_sorted_numerics_multiple_values_vs_stored_fields => "testSortedNumericsMultipleValuesVsStoredFields",
    test_sorted_numerics_few_unique_sets_vs_stored_fields => "testSortedNumericsFewUniqueSetsVsStoredFields",
    test_sorted_set_variable_length_vs_stored_fields => "testSortedSetVariableLengthVsStoredFields",
    test_sorted_set_fixed_length_single_valued_vs_stored_fields => "testSortedSetFixedLengthSingleValuedVsStoredFields",
    test_sorted_set_variable_length_single_valued_vs_stored_fields => "testSortedSetVariableLengthSingleValuedVsStoredFields",
    test_sorted_set_fixed_length_few_unique_sets_vs_stored_fields => "testSortedSetFixedLengthFewUniqueSetsVsStoredFields",
    test_sorted_set_variable_length_few_unique_sets_vs_stored_fields => "testSortedSetVariableLengthFewUniqueSetsVsStoredFields",
    test_sorted_set_variable_length_many_values_per_doc_vs_stored_fields => "testSortedSetVariableLengthManyValuesPerDocVsStoredFields",
    test_sorted_set_fixed_length_many_values_per_doc_vs_stored_fields => "testSortedSetFixedLengthManyValuesPerDocVsStoredFields",
    test_gcd_compression => "testGCDCompression",
    test_sparse_gcd_compression => "testSparseGCDCompression",
    test_zeros => "testZeros",
    test_sparse_zeros => "testSparseZeros",
    test_zero_or_min => "testZeroOrMin",
    test_two_numbers_one_missing => "testTwoNumbersOneMissing",
    test_two_numbers_one_missing_with_merging => "testTwoNumbersOneMissingWithMerging",
    test_three_numbers_one_missing_with_merging => "testThreeNumbersOneMissingWithMerging",
    test_two_bytes_one_missing => "testTwoBytesOneMissing",
    test_two_bytes_one_missing_with_merging => "testTwoBytesOneMissingWithMerging",
    test_three_bytes_one_missing_with_merging => "testThreeBytesOneMissingWithMerging",
    test_threads => "testThreads",
    test_one_sorted_number => "testOneSortedNumber",
    test_one_sorted_number_one_missing => "testOneSortedNumberOneMissing",
    test_number_merge_away_all_values => "testNumberMergeAwayAllValues",
    test_two_sorted_number => "testTwoSortedNumber",
    test_two_sorted_number_same_value => "testTwoSortedNumberSameValue",
    test_two_sorted_number_one_missing => "testTwoSortedNumberOneMissing",
    test_sorted_number_merge => "testSortedNumberMerge",
    test_sorted_number_merge_away_all_values => "testSortedNumberMergeAwayAllValues",
    test_sorted_enum_advance_independently => "testSortedEnumAdvanceIndependently",
    test_sorted_set_enum_advance_independently => "testSortedSetEnumAdvanceIndependently",
    test_sorted_merge_away_all_values_large_segment => "testSortedMergeAwayAllValuesLargeSegment",
    test_sorted_set_merge_away_all_values_large_segment => "testSortedSetMergeAwayAllValuesLargeSegment",
    test_numeric_merge_away_all_values_large_segment => "testNumericMergeAwayAllValuesLargeSegment",
    test_sorted_numeric_merge_away_all_values_large_segment => "testSortedNumericMergeAwayAllValuesLargeSegment",
    test_binary_merge_away_all_values_large_segment => "testBinaryMergeAwayAllValuesLargeSegment",
    test_random_advance_numeric => "testRandomAdvanceNumeric",
    test_random_advance_binary => "testRandomAdvanceBinary",
}

suite_tests! {
    bdv:
    test_sorted_merge_away_all_values_with_skipper => "testSortedMergeAwayAllValuesWithSkipper",
    test_sorted_set_merge_away_all_values_with_skipper => "testSortedSetMergeAwayAllValuesWithSkipper",
    test_number_merge_away_all_values_with_skipper => "testNumberMergeAwayAllValuesWithSkipper",
    test_sorted_number_merge_away_all_values_with_skipper => "testSortedNumberMergeAwayAllValuesWithSkipper",
    test_sorted_merge_away_all_values_large_segment_with_skipper => "testSortedMergeAwayAllValuesLargeSegmentWithSkipper",
    test_sorted_set_merge_away_all_values_large_segment_with_skipper => "testSortedSetMergeAwayAllValuesLargeSegmentWithSkipper",
    test_numeric_merge_away_all_values_large_segment_with_skipper => "testNumericMergeAwayAllValuesLargeSegmentWithSkipper",
    test_sorted_numeric_merge_away_all_values_large_segment_with_skipper => "testSortedNumericMergeAwayAllValuesLargeSegmentWithSkipper",
    test_max_value_count_with_skipper => "testMaxValueCountWithSkipper",
    test_numeric_doc_values_with_skipper_small => "testNumericDocValuesWithSkipperSmall",
    test_numeric_doc_values_with_skipper_medium => "testNumericDocValuesWithSkipperMedium",
    test_sorted_numeric_doc_values_with_skipper_small => "testSortedNumericDocValuesWithSkipperSmall",
    test_sorted_numeric_doc_values_with_skipper_medium => "testSortedNumericDocValuesWithSkipperMedium",
    test_sorted_doc_values_with_skipper_small => "testSortedDocValuesWithSkipperSmall",
    test_sorted_doc_values_with_skipper_medium => "testSortedDocValuesWithSkipperMedium",
    test_sorted_set_doc_values_with_skipper_small => "testSortedSetDocValuesWithSkipperSmall",
    test_sorted_set_doc_values_with_skipper_medium => "testSortedSetDocValuesWithSkipperMedium",
    test_mismatched_fields => "testMismatchedFields",
    test_random_dense_numeric_into_bit_set => "testRandomDenseNumericIntoBitSet",
    test_random_sparse_numeric_into_bit_set => "testRandomSparseNumericIntoBitSet",
    test_random_dense_sorted_numeric_into_bit_set => "testRandomDenseSortedNumericIntoBitSet",
    test_random_sparse_sorted_numeric_into_bit_set => "testRandomSparseSortedNumericIntoBitSet",
    test_random_dense_sorted_into_bit_set => "testRandomDenseSortedIntoBitSet",
    test_random_sparse_sorted_into_bit_set => "testRandomSparseSortedIntoBitSet",
    test_random_dense_sorted_set_into_bit_set => "testRandomDenseSortedSetIntoBitSet",
    test_random_sparse_sorted_set_into_bit_set => "testRandomSparseSortedSetIntoBitSet",
    test_random_dense_binary_into_bit_set => "testRandomDenseBinaryIntoBitSet",
    test_random_sparse_binary_into_bit_set => "testRandomSparseBinaryIntoBitSet",
    test_unique_values_compression => "testUniqueValuesCompression",
    test_date_compression => "testDateCompression",
    test_single_big_value_compression => "testSingleBigValueCompression",
}

/// The suite names this file's `suite_tests!` invocations run, from its own
/// source.
fn run_names() -> String {
    let source = include_str!("test_lucene90_doc_values_format.rs");
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
    // The `@Nightly` eight run below, at smaller sizes.
    assert_eq!(
        unrun(dv::TESTS),
        [
            "testThreads2",
            "testThreads3",
            "testEmptyBinaryValueOnPageSizes",
            "testHighOrdsSortedSetDV"
        ]
    );
    assert_eq!(
        unrun(bdv::TESTS),
        [
            "testNumericDocValuesWithSkipperBig",
            "testSortedNumericDocValuesWithSkipperBig",
            "testSortedDocValuesWithSkipperBig",
            "testSortedSetDocValuesWithSkipperBig"
        ]
    );
    assert_eq!(unrun(bdv::COMPRESSING_TESTS), [""; 0]);
}

/// `@Nightly testThreads2`, at a few hundred documents.
#[test]
fn threads2_at_a_smaller_size() {
    let mut t = LuceneTestCase::with_config("testThreads2", 21, false, 1);
    dv::threads2(&mut t, 300);
}

/// `@Nightly testThreads3`, at a few hundred documents and two rounds.
#[test]
fn threads3_at_a_smaller_size() {
    let mut t = LuceneTestCase::with_config("testThreads3", 22, false, 1);
    dv::threads3(&mut t, 300, 2);
}

/// `@Nightly testEmptyBinaryValueOnPageSizes`, up to 64 KiB values.
#[test]
fn empty_binary_value_on_page_sizes_up_to_64k() {
    let mut t = LuceneTestCase::with_config("testEmptyBinaryValueOnPageSizes", 23, false, 1);
    dv::empty_binary_value_on_page_sizes(&mut t, 17);
}

/// `@Nightly testHighOrdsSortedSetDV`, at 2 000 documents.
#[test]
fn high_ords_sorted_set_dv_at_a_smaller_size() {
    let mut t = LuceneTestCase::with_config("testHighOrdsSortedSetDV", 24, false, 1);
    dv::high_ords_sorted_set_dv(&mut t, 2_000);
}

/// The `@Nightly` `*WithSkipperBig` tests, at 40 000 documents: ten
/// intervals of 4 096 when the density draw is dense, past the eight a
/// second skip level needs.
#[test]
fn skipper_tests_at_a_bigger_size() {
    let mut t = LuceneTestCase::with_config("testWithSkipperBig", 25, false, 1);
    bdv::numeric_doc_values_with_skipper(&mut t, 40_000);
    bdv::sorted_numeric_doc_values_with_skipper(&mut t, 40_000);
    bdv::sorted_doc_values_with_skipper(&mut t, 40_000);
    bdv::sorted_set_doc_values_with_skipper(&mut t, 40_000);
}

#[test]
fn merge_stability() {
    run("testMergeStability", |t| {
        test_merge_stability(&DocValuesSuite, t)
    });
}

#[test]
fn random_exceptions() {
    // Gate rule #10: the injected errors must actually fire.
    let mut injected = 0;
    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config("testRandomExceptions", seed, false, 1);
        injected += test_random_exceptions(&DocValuesSuite, &mut t);
    }
    assert!(injected > 0, "no error was injected");
}

#[test]
fn check_integrity_reads_all_bytes() {
    run("testCheckIntegrityReadsAllBytes", |t| {
        test_check_integrity_reads_all_bytes(&DocValuesSuite, t);
    });
}
