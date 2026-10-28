//! `TestLucene99HnswVectorsFormat`: `BaseKnnVectorsFormatTestCase` and the
//! `BaseIndexFileFormatTestCase` tests it inherits, a few seeds each.

use lucene_test_framework::index::base_index_file_format::SuiteTest;
use lucene_test_framework::index::base_index_file_format::{
    test_check_integrity_reads_all_bytes, test_random_exceptions,
};
use lucene_test_framework::index::base_knn_vectors_format::{self as kf, KnnVectorsSuite};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

const SEEDS: [i64; 2] = [0x4E5F, -0x4E5F_0C70_1234];

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
                run($name, kf::$test);
            }
        )*
    };
}

suite_tests! {
    test_field_constructor => "testFieldConstructor",
    test_field_constructor_exceptions => "testFieldConstructorExceptions",
    test_different_codecs1 => "testDifferentCodecs1",
    test_different_codecs2 => "testDifferentCodecs2",
    test_illegal_dim_change_two_docs => "testIllegalDimChangeTwoDocs",
    test_illegal_similarity_function_change => "testIllegalSimilarityFunctionChange",
    test_illegal_dim_change_two_writers => "testIllegalDimChangeTwoWriters",
    test_merging_with_different_knn_fields => "testMergingWithDifferentKnnFields",
    test_merging_with_different_byte_knn_fields => "testMergingWithDifferentByteKnnFields",
    test_writer_ram_estimate => "testWriterRamEstimate",
    test_writer_byte_vector_ram_estimate => "testWriterByteVectorRamEstimate",
    test_illegal_similarity_function_change_two_writers => "testIllegalSimilarityFunctionChangeTwoWriters",
    test_add_indexes_directory0 => "testAddIndexesDirectory0",
    test_add_indexes_directory1 => "testAddIndexesDirectory1",
    test_add_indexes_directory01 => "testAddIndexesDirectory01",
    test_illegal_dim_change_via_add_indexes => "testIllegalDimChangeViaAddIndexesDirectory",
    test_illegal_similarity_function_change_via_add_indexes => "testIllegalSimilarityFunctionChangeViaAddIndexesDirectory",
    test_illegal_multiple_values => "testIllegalMultipleValues",
    test_illegal_dimension_too_large => "testIllegalDimensionTooLarge",
    test_illegal_empty_vector => "testIllegalEmptyVector",
    test_invalid_knn_vector_field_usage => "testInvalidKnnVectorFieldUsage",
    test_delete_all_vector_docs => "testDeleteAllVectorDocs",
    test_knn_vector_field_missing_from_one_segment => "testKnnVectorFieldMissingFromOneSegment",
    test_sparse_vectors => "testSparseVectors",
    test_float_vector_scorer_iteration => "testFloatVectorScorerIteration",
    test_byte_vector_scorer_iteration => "testByteVectorScorerIteration",
    test_empty_float_vector_data => "testEmptyFloatVectorData",
    test_empty_byte_vector_data => "testEmptyByteVectorData",
    test_indexed_value_not_aliased => "testIndexedValueNotAliased",
    test_sorted_index => "testSortedIndex",
    test_sorted_index_bytes => "testSortedIndexBytes",
    test_index_multiple_knn_vector_fields => "testIndexMultipleKnnVectorFields",
    test_random => "testRandom",
    test_random_bytes => "testRandomBytes",
    test_search_with_visited_limit => "testSearchWithVisitedLimit",
    test_random_with_updates_and_graph => "testRandomWithUpdatesAndGraph",
    test_check_index_includes_vectors => "testCheckIndexIncludesVectors",
    test_similarity_function_identifiers => "testSimilarityFunctionIdentifiers",
    test_vector_encoding_ordinals => "testVectorEncodingOrdinals",
    test_advance => "testAdvance",
    test_vector_values_report_correct_docs => "testVectorValuesReportCorrectDocs",
    test_mismatched_fields => "testMismatchedFields",
}

/// The suite names this file's `suite_tests!` invocations run, from its own
/// source.
fn run_names() -> String {
    let source = include_str!("test_lucene99_hnsw_vectors_format.rs");
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
    assert_eq!(unrun(kf::TESTS), [""; 0]);
}

/// The suite `@Before init()` draws for `t`.
fn suite(t: &mut LuceneTestCase) -> KnnVectorsSuite {
    KnnVectorsSuite::new(t)
}

#[test]
fn random_exceptions() {
    // Gate rule #10: the injected errors must actually fire.
    let mut injected = 0;
    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config("testRandomExceptions", seed, false, 1);
        let suite = suite(&mut t);
        injected += test_random_exceptions(&suite, &mut t);
    }
    assert!(injected > 0, "no error was injected");
}

#[test]
fn check_integrity_reads_all_bytes() {
    run("testCheckIntegrityReadsAllBytes", |t| {
        let suite = suite(t);
        test_check_integrity_reads_all_bytes(&suite, t);
    });
}

#[test]
fn random_draws_cover_every_encoding_and_similarity() {
    // The per-test draw reaches both encodings and all four similarities.
    let (mut encodings, mut similarities) = (Vec::new(), Vec::new());
    for i in 0..64i64 {
        // Consecutive seeds start Java's LCG on correlated high bits.
        let seed = i.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let mut t = LuceneTestCase::with_config("init", seed, false, 1);
        let s = suite(&mut t);
        encodings.push(s.encoding as i32);
        similarities.push(s.similarity as i32);
    }
    encodings.sort_unstable();
    encodings.dedup();
    similarities.sort_unstable();
    similarities.dedup();
    assert_eq!(vec![0, 1], encodings);
    assert_eq!(vec![0, 1, 2, 3], similarities);
}
