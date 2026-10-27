//! `TestLucene90CompoundFormat`: `BaseCompoundFormatTestCase`, its own
//! `testFileLengthOrdering`, and the `BaseIndexFileFormatTestCase` tests it
//! inherits, a few seeds each.

use lucene_test_framework::index::base_compound_format::{self as cf, CompoundSuite};
use lucene_test_framework::index::base_index_file_format::SuiteTest;
use lucene_test_framework::index::base_index_file_format::{
    test_check_integrity_reads_all_bytes, test_random_exceptions,
};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

const SEEDS: [i64; 2] = [0xCF5E, -0x0C0F_FEE0_1234];

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
                run($name, cf::$test);
            }
        )*
    };
}

suite_tests! {
    test_empty => "testEmpty",
    test_single_file => "testSingleFile",
    test_two_files => "testTwoFiles",
    test_large_cfs => "testLargeCFS",
    test_list_all => "testListAll",
    test_create_output_disabled => "testCreateOutputDisabled",
    test_delete_file_disabled => "testDeleteFileDisabled",
    test_rename_file_disabled => "testRenameFileDisabled",
    test_sync_disabled => "testSyncDisabled",
    test_make_lock_disabled => "testMakeLockDisabled",
    test_random_files => "testRandomFiles",
    test_many_sub_files => "testManySubFiles",
    test_cloned_streams_closing => "testClonedStreamsClosing",
    test_random_access => "testRandomAccess",
    test_random_access_clones => "testRandomAccessClones",
    test_double_close => "testDoubleClose",
    test_pass_io_context => "testPassIOContext",
    test_resource_name_inside_compound_file => "testResourceNameInsideCompoundFile",
    test_file_not_found => "testFileNotFound",
    test_read_past_eof => "testReadPastEOF",
    test_missing_codec_headers_are_caught => "testMissingCodecHeadersAreCaught",
    test_corrupt_files_are_caught => "testCorruptFilesAreCaught",
    test_check_integrity => "testCheckIntegrity",
}

/// The suite names this file's `suite_tests!` invocations run, from its own
/// source.
fn run_names() -> String {
    let source = include_str!("test_lucene90_compound_format.rs");
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
    assert_eq!(unrun(cf::TESTS), [""; 0]);
}

#[test]
fn random_exceptions() {
    // Gate rule #10: the injected errors must actually fire.
    let mut injected = 0;
    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config("testRandomExceptions", seed, false, 1);
        injected += test_random_exceptions(&CompoundSuite, &mut t);
    }
    assert!(injected > 0, "no error was injected");
}

#[test]
fn check_integrity_reads_all_bytes() {
    run("testCheckIntegrityReadsAllBytes", |t| {
        test_check_integrity_reads_all_bytes(&CompoundSuite, t);
    });
}

#[test]
fn list_all_saw_a_compound_segment() {
    // Gate rule #10: `testListAll` passes vacuously over an index with no
    // compound segment.
    let mut compound = 0;
    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config("testListAll", seed, false, 1);
        compound += cf::list_all(&mut t);
    }
    assert!(compound > 0, "no compound segment was written");
}

/// `testFileLengthOrdering`: `.cfe` lists the files smallest first, at
/// ascending offsets, whatever order the segment names them in.
#[test]
fn file_length_ordering() {
    use lucene_codecs::compound_format;
    use lucene_test_framework::index::merge_policies::shuffle;
    use lucene_test_framework::util::test_util;

    for seed in SEEDS {
        let mut t = LuceneTestCase::with_config("testFileLengthOrdering", seed, false, 1);
        let dir = t.new_directory();
        let segment = "_123";
        let chunk = 1024;
        let mut si = cf::new_segment_info(&mut t, segment);
        let mut ordered = Vec::new();
        let mut size = test_util::next_int_in(t.random(), 0, chunk);
        for i in 0..10 {
            let name = format!("{segment}.{i}");
            cf::create_random_file(&mut t, &*dir, &name, size as usize, &si.id);
            size += test_util::next_int_in(t.random(), 1, 100);
            ordered.push(name);
        }
        let mut shuffled = ordered.clone();
        shuffle(t.random(), &mut shuffled);
        si.files = shuffled;
        cf::write_compound(&*dir, segment, &si).expect("compoundFormat().write");

        let cfe = dir.open(&format!("{segment}.cfe")).expect("open .cfe");
        let entries = compound_format::parse_entries(&cfe, &si.id).expect("entries");
        let (mut last_offset, mut last_length) = (0, 0);
        for (i, id) in entries.names().enumerate() {
            assert_eq!(ordered[i], format!("{segment}{id}"));
            let entry = entries.get(id).expect("entry");
            assert!(entry.offset > last_offset);
            last_offset = entry.offset;
            assert!(entry.length >= last_length);
            last_length = entry.length;
        }
        drop(cfe);
        dir.close().expect("close");
    }
}
