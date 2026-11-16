#![cfg(test)]
use super::*;

fn seg(name: &str, max_doc: i32, del: i32, bytes: i64) -> MergeSegment {
    MergeSegment::new(name, max_doc, del, bytes)
}

#[test]
fn default_size_prorates_by_deletes() {
    let ctx = BasicMergeContext::default();
    assert_eq!(default_size(&seg("_a", 100, 25, 1000), &ctx).unwrap(), 750);
    assert_eq!(default_size(&seg("_a", 0, 0, 1000), &ctx).unwrap(), 1000);
}

#[test]
fn compound_settings_validate_like_java() {
    let mut s = CompoundFileSettings::default();
    assert!(s.set_no_cfs_ratio(1.5).is_err());
    assert!(s.set_no_cfs_ratio(-0.1).is_err());
    s.set_no_cfs_ratio(0.5).unwrap();
    assert_eq!(s.no_cfs_ratio, 0.5);
    assert!(s.set_max_cfs_segment_size_mb(-1.0).is_err());
    s.set_max_cfs_segment_size_mb(2.0).unwrap();
    assert_eq!(s.max_cfs_segment_size, 2 * 1024 * 1024);
    assert_eq!(s.max_cfs_segment_size_mb(), 2.0);
    s.set_max_cfs_segment_size_mb(f64::INFINITY).unwrap();
    assert_eq!(s.max_cfs_segment_size, i64::MAX);
}

#[test]
fn use_compound_file_follows_the_ratio() {
    let ctx = BasicMergeContext::default();
    let infos = vec![seg("_a", 10, 0, 900), seg("_b", 10, 0, 100)];
    let mut p = TieredMergePolicy::default();
    // 10% of 1000 is 100: `_b` qualifies, `_a` does not.
    assert!(p.use_compound_file(&infos, &infos[1], &ctx).unwrap());
    assert!(!p.use_compound_file(&infos, &infos[0], &ctx).unwrap());
    p.compound_file_settings_mut()
        .set_no_cfs_ratio(0.0)
        .unwrap();
    assert!(!p.use_compound_file(&infos, &infos[1], &ctx).unwrap());
    p.compound_file_settings_mut()
        .set_no_cfs_ratio(1.0)
        .unwrap();
    assert!(p.use_compound_file(&infos, &infos[0], &ctx).unwrap());
    p.compound_file_settings_mut()
        .set_max_cfs_segment_size_mb(0.0)
        .unwrap();
    assert!(!p.use_compound_file(&infos, &infos[0], &ctx).unwrap());
}

#[test]
fn no_merge_policy_never_merges() {
    let ctx = BasicMergeContext::default();
    let infos = vec![seg("_a", 10, 5, 100), seg("_b", 10, 5, 100)];
    let mut p = NoMergePolicy::default();
    let t = MergeTrigger::Explicit;
    assert!(p.find_merges(t, &infos, &ctx).unwrap().is_none());
    assert!(p
        .find_forced_merges(&infos, 1, &HashMap::new(), &ctx)
        .unwrap()
        .is_none());
    assert!(p
        .find_forced_deletes_merges(&infos, &ctx)
        .unwrap()
        .is_none());
    assert!(p.find_full_flush_merges(t, &infos, &ctx).unwrap().is_none());
    let cfs = infos[0].clone().with_compound_file(true);
    assert!(p.use_compound_file(&infos, &cfs, &ctx).unwrap());
    assert!(!p.use_compound_file(&infos, &infos[0], &ctx).unwrap());
    assert_eq!(p.size(&infos[0], &ctx).unwrap(), i64::MAX);
    p.compound_file_settings_mut()
        .set_no_cfs_ratio(0.3)
        .unwrap();
    assert_eq!(p.compound_file_settings().no_cfs_ratio, 0.3);
    assert!(!p.keep_fully_deleted_segment());
    assert_eq!(p.num_deletes_to_merge(&infos[0], 3), 3);
    let spec = p.find_merges_for_readers(&[5, 7]).unwrap();
    assert_eq!(spec.merges.len(), 1);
    assert_eq!(spec.merges[0].reader_count, 2);
    assert_eq!(spec.merges[0].total_max_doc, 12);
}

#[test]
fn full_flush_merges_keep_only_small_inputs() {
    let ctx = BasicMergeContext::default();
    let mut config = MergePolicyConfig {
        segments_per_tier: 2.0,
        max_merge_at_once: 2,
        floor_segment_size: 1000,
        ..MergePolicyConfig::default()
    };
    let small: Vec<MergeSegment> = (0..4).map(|i| seg(&format!("_{i}"), 10, 0, 10)).collect();
    let p = TieredMergePolicy::new(config.clone());
    let t = MergeTrigger::FullFlush;
    let natural = p.find_merges(t, &small, &ctx).unwrap().unwrap();
    let full = p.find_full_flush_merges(t, &small, &ctx).unwrap().unwrap();
    assert_eq!(natural.groups(), full.groups());
    config.floor_segment_size = 5;
    let p = TieredMergePolicy::new(config);
    assert!(p.find_full_flush_merges(t, &small, &ctx).unwrap().is_none());
    assert_eq!(p.max_full_flush_merge_size(), 5);
}

#[test]
fn merge_segment_from_stat_and_spec_groups() {
    let stat = SegmentStat {
        name: "_x".into(),
        doc_count: 3,
        del_count: 1,
        size_bytes: u64::MAX,
    };
    let s = MergeSegment::from(&stat);
    assert_eq!(s.size_in_bytes, i64::MAX);
    let merge = OneMerge::new(vec![s.clone(), seg("_y", 4, 0, 1)]);
    assert_eq!(merge.total_max_doc, 7);
    let mut spec = MergeSpecification::new();
    spec.add(merge);
    assert_eq!(
        spec.groups(),
        vec![vec!["_x".to_string(), "_y".to_string()]]
    );
}

/// `TieredMergePolicy.findForcedMerges` asks `isMerged` only for the
/// single-segment branch Java's short-circuit reaches, not once per
/// segment up front: the context sees exactly the one
/// `numDeletesToMerge` per segment that the size stats need.
#[test]
fn tiered_forced_merge_asks_is_merged_lazily() {
    struct Counting {
        calls: std::cell::Cell<usize>,
        merging: HashSet<String>,
    }
    impl MergeContext for Counting {
        fn num_deletes_to_merge(&self, info: &MergeSegment) -> Result<i32> {
            self.calls.set(self.calls.get() + 1);
            Ok(info.del_count)
        }
        fn num_deleted_docs(&self, info: &MergeSegment) -> i32 {
            info.del_count
        }
        fn merging_segments(&self) -> &HashSet<String> {
            &self.merging
        }
    }
    let ctx = Counting {
        calls: std::cell::Cell::new(0),
        merging: HashSet::new(),
    };
    let infos: Vec<MergeSegment> = ["_a", "_b", "_c", "_d"]
        .iter()
        .map(|n| seg(n, 10, 0, 100))
        .collect();
    let to_merge: HashMap<String, bool> = infos.iter().map(|s| (s.name.clone(), true)).collect();
    let p = TieredMergePolicy::default();
    let spec = p
        .find_forced_merges(&infos, 1, &to_merge, &ctx)
        .unwrap()
        .unwrap();
    assert_eq!(spec.groups().len(), 1);
    assert_eq!(ctx.calls.get(), infos.len());
}
