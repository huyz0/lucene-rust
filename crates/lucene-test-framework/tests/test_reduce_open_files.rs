//! `TestUtil.reduceOpenFiles` and `syncConcurrentMerges`: the live policy
//! and scheduler a writer shares are changed as Java changes them.

use lucene_index::merge_policy::api::{MergePolicy, NoMergePolicy, TieredMergePolicy};
use lucene_index::merge_policy::log::LogDocMergePolicy;
use lucene_index::merge_scheduler::ConcurrentMergeScheduler;
use lucene_test_framework::index::index_writer_config::{LiveMergePolicy, PolicyKind};
use lucene_test_framework::util::test_util::{reduce_open_files, sync_concurrent_merges};

fn no_cfs_ratio(p: &LiveMergePolicy) -> f64 {
    p.kind().policy().compound_file_settings().no_cfs_ratio
}

fn merge_factor(p: &LiveMergePolicy) -> i32 {
    let PolicyKind::Log(lmp) = &*p.kind() else {
        unreachable!()
    };
    lmp.merge_factor()
}

#[test]
fn log_policy() {
    let mut lmp = (*LogDocMergePolicy::new()).clone();
    lmp.set_merge_factor(10).unwrap();
    lmp.compound_file_settings_mut()
        .set_no_cfs_ratio(0.1)
        .unwrap();
    let live = LiveMergePolicy::new(PolicyKind::Log(lmp));
    let cms = ConcurrentMergeScheduler::new();
    reduce_open_files(&live, Some(&cms));
    assert_eq!(no_cfs_ratio(&live), 1.0);
    assert_eq!(merge_factor(&live), 5);
    assert_eq!((cms.max_merge_count(), cms.max_thread_count()), (3, 2));
    sync_concurrent_merges(&cms).unwrap();

    // A factor already below 5 stays.
    live.change(|k| {
        let PolicyKind::Log(lmp) = k else {
            unreachable!()
        };
        lmp.set_merge_factor(3).unwrap();
    });
    reduce_open_files(&live, None);
    assert_eq!(merge_factor(&live), 3);
}

#[test]
fn tiered_policy() {
    let mut tmp = TieredMergePolicy::default();
    tmp.config.max_merge_at_once = 30;
    tmp.config.segments_per_tier = 4;
    let live = LiveMergePolicy::new(PolicyKind::Tiered(tmp));
    reduce_open_files(&live, None);
    assert_eq!(no_cfs_ratio(&live), 1.0);
    let PolicyKind::Tiered(tmp) = &*live.kind() else {
        unreachable!()
    };
    assert_eq!(
        (tmp.config.max_merge_at_once, tmp.config.segments_per_tier),
        (5, 4)
    );
}

#[test]
fn other_policies_only_go_compound() {
    let live = LiveMergePolicy::new(PolicyKind::NoMerge(NoMergePolicy::default()));
    reduce_open_files(&live, None);
    assert_eq!(no_cfs_ratio(&live), 1.0);
}
