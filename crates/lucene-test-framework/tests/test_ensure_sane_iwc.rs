//! `LuceneTestCase.ensureSaneIWCOnNightly`: each merge policy kind's caps,
//! the compound-file switch and the `noCFSRatio` floor, nightly only.

use std::sync::Arc;

use lucene_index::merge_policy::api::{NoMergePolicy, TieredMergePolicy};
use lucene_index::merge_policy::log::{LogByteSizeMergePolicy, LogDocMergePolicy};
use lucene_test_framework::analysis::mock_analyzer::MockAnalyzer;
use lucene_test_framework::index::index_writer_config::{
    ensure_sane_iwc_on_nightly, new_index_writer_config, Draws, LiveMergePolicy, PolicyKind,
    RandomIndexWriterConfig,
};
use lucene_util::java_random::JavaRandom;

const MB: i64 = 1024 * 1024;

fn config(kind: PolicyKind) -> RandomIndexWriterConfig {
    let mut r = JavaRandom::new(3);
    let analyzer = Arc::new(MockAnalyzer::new(&mut JavaRandom::new(3)).build());
    let mut c = new_index_writer_config(&mut Draws::new(&mut r), analyzer, 0);
    c.merge_policy = LiveMergePolicy::new(kind);
    c.use_compound_file = false;
    c.merge_policy.change(|k| {
        k.policy_mut()
            .compound_file_settings_mut()
            .set_no_cfs_ratio(0.1)
            .unwrap()
    });
    c
}

fn tiered_max(c: &RandomIndexWriterConfig) -> u64 {
    let kind = c.merge_policy.kind();
    match &*kind {
        PolicyKind::Tiered(t) => t.config.max_merged_segment_size,
        _ => unreachable!(),
    }
}

fn ratio(c: &RandomIndexWriterConfig) -> f64 {
    c.merge_policy
        .change(|k| k.policy_mut().compound_file_settings_mut().no_cfs_ratio)
}

#[test]
fn a_nightly_run_caps_each_policy() {
    let mut tiered = config(PolicyKind::Tiered(TieredMergePolicy::default()));
    ensure_sane_iwc_on_nightly(true, &mut tiered);
    assert!(tiered.use_compound_file);
    assert_eq!(ratio(&tiered), 0.25);
    assert_eq!(tiered_max(&tiered), 5000 * MB as u64);

    let mut bytes = config(PolicyKind::Log((*LogByteSizeMergePolicy::new()).clone()));
    ensure_sane_iwc_on_nightly(true, &mut bytes);
    match &*bytes.merge_policy.kind() {
        PolicyKind::Log(l) => assert_eq!(l.max_merge_size, 1000 * MB),
        _ => unreachable!(),
    }

    let mut docs = config(PolicyKind::Log((*LogDocMergePolicy::new()).clone()));
    ensure_sane_iwc_on_nightly(true, &mut docs);
    match &*docs.merge_policy.kind() {
        PolicyKind::Log(l) => assert_eq!(l.max_merge_docs, 100_000),
        _ => unreachable!(),
    }

    // Any other policy: only the compound-file settings.
    let mut none = config(PolicyKind::NoMerge(NoMergePolicy::default()));
    ensure_sane_iwc_on_nightly(true, &mut none);
    assert!(none.use_compound_file);
    assert_eq!(ratio(&none), 0.25);
}

#[test]
fn a_higher_ratio_stays_and_a_regular_run_is_untouched() {
    let mut c = config(PolicyKind::Tiered(TieredMergePolicy::default()));
    c.merge_policy.change(|k| {
        k.policy_mut()
            .compound_file_settings_mut()
            .set_no_cfs_ratio(0.8)
            .unwrap()
    });
    ensure_sane_iwc_on_nightly(true, &mut c);
    assert_eq!(ratio(&c), 0.8);

    let mut regular = config(PolicyKind::Tiered(TieredMergePolicy::default()));
    let before = tiered_max(&regular);
    ensure_sane_iwc_on_nightly(false, &mut regular);
    assert!(!regular.use_compound_file);
    assert_eq!(ratio(&regular), 0.1);
    assert_eq!(tiered_max(&regular), before);
}
