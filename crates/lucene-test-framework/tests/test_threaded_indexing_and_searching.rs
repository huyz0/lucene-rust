//! `ThreadedIndexingAndSearchingTestCase` as `TestNRTThreads` runs it (one
//! reader per pass, reopened or opened after a commit, searched by every
//! search thread), and with a shared reader reopened before each search, a
//! few seeds each (`scripts/test-framework-stress.sh` runs 200).

use lucene_test_framework::index::threaded_indexing_and_searching::{
    run_test, ReaderMode, ThreadedRunStats,
};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

const SEEDS: [i64; 3] = [0x7E57, -0x0BAD_5EED_1234, 42];

fn runs(name: &str, mode: ReaderMode, multiplier: i32) -> Vec<ThreadedRunStats> {
    SEEDS
        .iter()
        .map(|&seed| {
            let mut t = LuceneTestCase::with_config(name, seed, false, multiplier);
            run_test(&mut t, name, mode)
        })
        .collect()
}

/// Gate rule #10: what the checks verify must have happened.
fn assert_exercised(stats: &[ThreadedRunStats]) {
    let sum = |f: fn(&ThreadedRunStats) -> u64| stats.iter().map(f).sum::<u64>();
    assert!(sum(|s| s.adds as u64) > 0, "{stats:?}");
    assert!(
        sum(|s| s.deletes as u64) > 0,
        "nothing was deleted: {stats:?}"
    );
    assert!(
        sum(|s| s.packs as u64) > 0,
        "no pack was indexed: {stats:?}"
    );
    assert!(sum(|s| s.searches) > 0, "nothing was searched: {stats:?}");
    assert!(
        sum(|s| s.term_queries) > 0,
        "no term was searched: {stats:?}"
    );
    for s in stats {
        assert_eq!(s.adds - s.deletes, s.num_docs);
    }
}

#[test]
fn nrt_threads() {
    assert_exercised(&runs("testNRTThreads", ReaderMode::Nrt, 1));
}

#[test]
fn reopened_reader_threads() {
    assert_exercised(&runs("testSearcherManager", ReaderMode::Reopen, 3));
}
