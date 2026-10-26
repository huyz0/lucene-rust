//! `BaseMergePolicyTestCase`'s suites (`TestLogMergePolicy`,
//! `TestTieredMergePolicy`, `TestNoMergePolicy`,
//! `TestUpgradeIndexMergePolicy`, `TestMergeOnFlushMergePolicy`), each with
//! its `mergePolicy`/`assertSegmentInfos`/`assertMerge`, and the test
//! framework's own `TestForceMergePolicy` and `TestMockRandomMergePolicy`.
//! The suites' policy-specific tests belong to the policies' own crates;
//! only `testFindFullFlushMerges`, which exercises `MockMergeContext`'s
//! merging set, is here.

use std::collections::HashSet;
use std::sync::Arc;

use lucene_codecs::stored_fields::Document;
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::log::LogMergePolicy;
use lucene_index::merge_policy::upgrade::UpgradeIndexMergePolicy;
use lucene_index::merge_policy::{
    BasicMergeContext, MergeContext, MergeOnFlushMergePolicy, MergePolicy, MergeSegment,
    MergeSpecification, MergeTrigger, NoMergePolicy, TieredMergePolicy,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::base_merge_policy_tests;
use lucene_test_framework::index::base_merge_policy::{
    del_count, make_segment_commit_info, BaseMergePolicyTestCase, MergePolicySuite,
    MockMergeContext,
};
use lucene_test_framework::index::index_writer_config::{
    new_log_merge_policy, new_merge_policy, new_tiered_merge_policy, Draws, PolicyKind,
};
use lucene_test_framework::index::merge_policies::{
    reverse, ForceMergePolicy, MockRandomMergePolicy,
};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_test_framework::util::test_util;
use lucene_util::java_random::JavaRandom;

/// `(long) (mb * 1024 * 1024)`.
fn mb(mb: i32) -> u64 {
    u64::try_from(mb).unwrap() * 1024 * 1024
}

mod log {
    use super::*;

    /// `TestLogMergePolicy`.
    pub struct Suite;

    impl MergePolicySuite for Suite {
        type Policy = LogMergePolicy;

        fn merge_policy(&self, random: &mut JavaRandom) -> LogMergePolicy {
            new_log_merge_policy(&mut Draws::new(random))
        }

        fn assert_segment_infos(&self, mp: &LogMergePolicy, infos: &[MergeSegment]) {
            let ctx = MockMergeContext::new(del_count);
            for info in infos {
                assert!(
                    mp.size(info, &ctx).unwrap() / i64::from(mp.merge_factor()) < mp.max_merge_size
                );
            }
        }

        fn assert_merge(&self, lmp: &LogMergePolicy, merge: &MergeSpecification) {
            let ctx = MockMergeContext::new(del_count);
            for one in &merge.merges {
                let merge_size: i64 = one
                    .segments
                    .iter()
                    .map(|s| lmp.size(s, &ctx).unwrap())
                    .sum();
                assert!(
                    merge_size <= lmp.min_merge_size
                        || one.segments.len() <= lmp.merge_factor() as usize,
                    "mergeSize: {merge_size} minMergeSize: {} segmentsCount: {} mergeFactor: {}",
                    lmp.min_merge_size,
                    one.segments.len(),
                    lmp.merge_factor()
                );
            }
        }
    }

    base_merge_policy_tests!(Suite);

    #[test]
    fn a_merge_of_too_many_segments_fails_assert_merge() {
        // Gate rule #10: the hook fires on what it guards against.
        let mut lmp =
            LogMergePolicy::clone(&lucene_index::merge_policy::log::LogDocMergePolicy::new());
        lmp.set_merge_factor(2).unwrap();
        let mut spec = MergeSpecification::new();
        spec.add(lucene_index::merge_policy::OneMerge::new(
            (0..3)
                .map(|i| MergeSegment::new(format!("_{i}"), 5000, 0, 1))
                .collect(),
        ));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Suite.assert_merge(&lmp, &spec)
        }));
        assert!(caught.is_err());
    }
}

mod tiered {
    use super::*;

    /// `TestTieredMergePolicy`.
    pub struct Suite;

    impl MergePolicySuite for Suite {
        type Policy = TieredMergePolicy;

        fn merge_policy(&self, random: &mut JavaRandom) -> TieredMergePolicy {
            new_tiered_merge_policy(&mut Draws::new(random))
        }

        fn prepare_simulation(&self, mp: &mut TieredMergePolicy, random: &mut JavaRandom) {
            // Avoid low values of the max merged segment size which prevent
            // this merge policy from scaling well.
            mp.config.max_merged_segment_size = mb(test_util::next_int(random, 1024, 10 * 1024));
        }

        fn assert_segment_infos(&self, tmp: &TieredMergePolicy, infos: &[MergeSegment]) {
            let c = &tmp.config;
            let max_merged_segment_bytes = c.max_merged_segment_size as i64;
            let mut min_segment_bytes = i64::MAX;
            let mut total_del_count = 0i32;
            let mut total_max_doc = 0i32;
            let mut total_bytes = 0i64;
            // (docCount, sizeInBytes)
            let mut segment_sizes: Vec<(i32, i64)> = Vec::new();
            for sci in infos {
                total_del_count += sci.del_count;
                total_max_doc += sci.max_doc;
                let live_ratio = 1.0 - f64::from(sci.del_count) / f64::from(sci.max_doc);
                let weighted = (live_ratio * sci.size_in_bytes as f64) as i64;
                total_bytes += weighted;
                segment_sizes.push((sci.max_doc - sci.del_count, weighted));
                min_segment_bytes = min_segment_bytes.min(weighted);
            }
            segment_sizes.sort_by_key(|s| s.1);

            let del_percentage = 100.0 * f64::from(total_del_count) / f64::from(total_max_doc);
            assert!(
                del_percentage <= c.deletes_pct_allowed,
                "Percentage of deleted docs {del_percentage} is larger than the target: {}",
                c.deletes_pct_allowed
            );

            let segs_per_tier = c.segments_per_tier as f64;
            let tsc = c.target_search_concurrency;
            let mut level_size_bytes = min_segment_bytes.max(c.floor_segment_size as i64);
            let mut bytes_left = total_bytes;
            let mut allowed_seg_count = 0f64;
            let biggest = if segment_sizes.len() > tsc - 1 {
                &segment_sizes[segment_sizes.len() - tsc + 1..]
            } else {
                &segment_sizes[..]
            };
            // Allow whole segments for the targetSearchConcurrency-1 biggest
            // segments.
            for size in biggest {
                bytes_left -= size.1;
                allowed_seg_count += 1.0;
            }
            let too_big_count = segment_sizes
                .iter()
                .filter(|s| s.1 >= max_merged_segment_bytes / 2)
                .count();
            let merge_factor = segs_per_tier.min(c.max_merge_at_once as f64) as i64;
            loop {
                let seg_count_level = bytes_left as f64 / level_size_bytes as f64;
                if seg_count_level <= segs_per_tier
                    || level_size_bytes >= max_merged_segment_bytes / 2
                {
                    allowed_seg_count += seg_count_level.ceil();
                    break;
                }
                allowed_seg_count += segs_per_tier;
                bytes_left -= (segs_per_tier * level_size_bytes as f64) as i64;
                level_size_bytes =
                    (level_size_bytes * merge_factor).min(max_merged_segment_bytes / 2);
            }
            // Allow at least a full tier in addition of the too big segments.
            allowed_seg_count = allowed_seg_count.max(too_big_count as f64 + segs_per_tier);
            // Allow at least `targetSearchConcurrency` segments.
            allowed_seg_count = allowed_seg_count.max(tsc as f64);

            // `getMaxAllowedDocs`: ceilDiv(live docs, targetSearchConcurrency).
            let live = i64::from(total_max_doc - total_del_count);
            let max_docs_per_segment = (live + tsc as i64 - 1).div_euclid(tsc as i64);
            let has_legal_merges = segment_sizes.windows(2).any(|w| {
                let merged_bytes = w[0].1 + w[1].1;
                let merged_docs = i64::from(w[0].0 + w[1].0);
                merged_bytes <= max_merged_segment_bytes
                    && w[1].1 as f64 * 1.5 <= merged_bytes as f64
                    && merged_docs <= max_docs_per_segment
            });
            let num_segments = infos.len();
            assert!(
                num_segments as f64 <= allowed_seg_count || !has_legal_merges,
                "mergeFactor={merge_factor} minSegmentBytes={min_segment_bytes} \
                 maxMergedSegmentBytes={max_merged_segment_bytes} segmentsPerTier={segs_per_tier} \
                 maxMergeAtOnce={} numSegments={num_segments} allowed={allowed_seg_count} \
                 totalBytes={total_bytes} delPercentage={del_percentage} deletesPctAllowed={} \
                 targetNumSegments={tsc}",
                c.max_merge_at_once,
                c.deletes_pct_allowed
            );
        }

        fn assert_merge(&self, tmp: &TieredMergePolicy, merges: &MergeSpecification) {
            for merge in &merges.merges {
                assert!(merge.segments.len() <= tmp.config.max_merge_at_once);
            }
        }
    }

    base_merge_policy_tests!(Suite);

    #[test]
    fn too_many_segments_with_a_legal_merge_fail_assert_segment_infos() {
        // Gate rule #10: forty equal segments, each pair mergeable, under a
        // policy allowing a handful.
        let mut tmp = TieredMergePolicy::default();
        tmp.config.segments_per_tier = 2;
        tmp.config.target_search_concurrency = 1;
        let infos: Vec<MergeSegment> = (0..40)
            .map(|i| MergeSegment::new(format!("_{i}"), 100, 0, 1 << 20))
            .collect();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Suite.assert_segment_infos(&tmp, &infos)
        }));
        assert!(caught.is_err());
        let mut deleted = infos.clone();
        deleted[0].del_count = 99;
        tmp.config.segments_per_tier = 50;
        tmp.config.deletes_pct_allowed = 0.0;
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Suite.assert_segment_infos(&tmp, &deleted)
        }));
        assert!(caught.is_err(), "deletes above deletesPctAllowed");
        let mut spec = MergeSpecification::new();
        spec.add(lucene_index::merge_policy::OneMerge::new(
            infos[..11].to_vec(),
        ));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Suite.assert_merge(&tmp, &spec)
        }));
        assert!(caught.is_err(), "more than maxMergeAtOnce");
    }
}

mod no_merge {
    use super::*;

    /// `TestNoMergePolicy`. Its `assertSegmentInfos` checks every segment's
    /// `source` attribute is a flush; this port's segments carry no
    /// attributes, and a merge would already have failed `assert_merge`.
    pub struct Suite;

    impl MergePolicySuite for Suite {
        type Policy = NoMergePolicy;

        fn merge_policy(&self, _random: &mut JavaRandom) -> NoMergePolicy {
            NoMergePolicy::default()
        }

        fn assert_segment_infos(&self, _policy: &NoMergePolicy, _infos: &[MergeSegment]) {}

        fn assert_merge(&self, _policy: &NoMergePolicy, _merge: &MergeSpecification) {
            panic!("should never happen");
        }

        // Reduce numbers as this merge policy doesn't work well with lots of
        // data.
        fn append_only_sizes(&self) -> (i32, i32) {
            (1_000_000, 10_000)
        }

        fn updates_sizes(&self, _tc: &mut LuceneTestCase) -> (i32, i32) {
            (100_000, 1000)
        }
    }

    base_merge_policy_tests!(Suite);

    #[test]
    #[should_panic(expected = "should never happen")]
    fn assert_merge_fails_on_any_merge() {
        Suite.assert_merge(&NoMergePolicy::default(), &MergeSpecification::new());
    }
}

mod upgrade {
    use super::*;

    /// `TestUpgradeIndexMergePolicy`.
    pub struct Suite;

    impl MergePolicySuite for Suite {
        type Policy = UpgradeIndexMergePolicy;

        fn merge_policy(&self, random: &mut JavaRandom) -> UpgradeIndexMergePolicy {
            let mut inner = new_tiered_merge_policy(&mut Draws::new(random));
            // Avoid low values of the max merged segment size which prevent
            // this merge policy from scaling well.
            inner.config.max_merged_segment_size = mb(test_util::next_int(random, 1024, 10 * 1024));
            UpgradeIndexMergePolicy::new(Box::new(inner))
        }

        fn assert_segment_infos(&self, _: &UpgradeIndexMergePolicy, _: &[MergeSegment]) {}

        fn assert_merge(&self, _: &UpgradeIndexMergePolicy, _: &MergeSpecification) {}
    }

    base_merge_policy_tests!(Suite);
}

mod merge_on_flush {
    use super::*;

    /// `TestMergeOnFlushMergePolicy`.
    pub struct Suite;

    impl MergePolicySuite for Suite {
        type Policy = MergeOnFlushMergePolicy;

        fn merge_policy(&self, random: &mut JavaRandom) -> MergeOnFlushMergePolicy {
            let mut inner = new_merge_policy(&mut Draws::new(random), true, 0);
            let settings = inner.policy().compound_file_settings();
            let threshold = test_util::next_int(random, 1, 100);
            if let PolicyKind::Tiered(tmp) = &mut inner {
                tmp.config.max_merged_segment_size =
                    mb(test_util::next_int(random, 1024, 10 * 1024));
            }
            let mut mp = MergeOnFlushMergePolicy::new(Box::new(inner));
            *mp.compound_file_settings_mut() = settings;
            mp.set_small_segment_threshold_mb(f64::from(threshold));
            mp
        }

        fn wraps_mock_random(&self, mp: &MergeOnFlushMergePolicy) -> bool {
            format!("{:?}", mp.unwrap()).contains("MockRandomMergePolicy")
        }

        fn assert_segment_infos(&self, _: &MergeOnFlushMergePolicy, _: &[MergeSegment]) {}

        fn assert_merge(&self, _: &MergeOnFlushMergePolicy, _: &MergeSpecification) {}
    }

    // Java: TestMergeOnFlushMergePolicy.testNoPathologicalMerges is a no-op:
    // this policy makes no effort to avoid O(n^2) merges. Keep the other
    // inherited scenarios; the generic amplification bound still tests the
    // policies whose Java suites retain it.
    base_merge_policy_tests!(@each Suite;
        test_force_merge_not_needed, test_find_forced_deletes_merges,
        test_simulate_append_only, test_simulate_updates);

    #[test]
    fn test_find_full_flush_merges() {
        let mut tc = LuceneTestCase::new("test_find_full_flush_merges");
        let merge_policy = Suite.merge_policy(tc.random());
        let small_segment_threshold_mb = merge_policy.small_segment_threshold_mb();
        let threshold_bytes = (small_segment_threshold_mb * 1024.0 * 1024.0) as i64;
        let mut merged_some = false;
        for _ in 0..10_000 {
            let r = tc.random();
            let num_segs = r.next_int_bounded(50);
            let mut infos = Vec::new();
            let mut merging = HashSet::new();
            let mut small = HashSet::new();
            for i in 0..num_segs {
                let max_doc = test_util::next_int(r, 10, 100);
                let del = r.next_int_bounded(10);
                let size_mb = r.next_double() * 2.0 * small_segment_threshold_mb;
                let sci = make_segment_commit_info(r, &format!("_{i}"), max_doc, del, size_mb);
                if sci.size_in_bytes < threshold_bytes {
                    small.insert(sci.name.clone());
                }
                if r.next_boolean() {
                    merging.insert(sci.name.clone());
                }
                infos.push(sci);
            }
            let mut ctx = MockMergeContext::new(del_count);
            ctx.set_merging_segments(merging.clone());
            match merge_policy
                .find_full_flush_merges(MergeTrigger::Commit, &infos, &ctx)
                .unwrap()
            {
                None => {
                    // If we didn't compute a merge, then we have at most one
                    // small segment not already participating in a merge.
                    let non_merging_small = small.iter().filter(|s| !merging.contains(*s)).count();
                    assert!(
                        non_merging_small <= 1,
                        "If no merges, then at most one small segment is not already merging"
                    );
                }
                Some(spec) => {
                    merged_some = true;
                    for one in &spec.merges {
                        for sci in &one.segments {
                            assert!(
                                small.contains(&sci.name),
                                "Merges only contain small segments"
                            );
                            assert!(
                                !merging.contains(&sci.name),
                                "findFullFlushMerges must not return already merging segments"
                            );
                        }
                    }
                }
            }
        }
        assert!(merged_some);
    }
}

/// The scenarios' skip for `MockRandomMergePolicy`, which cannot tolerate
/// the forced-merge checks, and its simulations otherwise.
mod mock_random {
    use super::*;

    pub struct Suite;

    impl MergePolicySuite for Suite {
        type Policy = MockRandomMergePolicy;

        fn merge_policy(&self, random: &mut JavaRandom) -> MockRandomMergePolicy {
            MockRandomMergePolicy::new(random)
        }

        fn assert_segment_infos(&self, _: &MockRandomMergePolicy, _: &[MergeSegment]) {}

        fn assert_merge(&self, _: &MockRandomMergePolicy, merge: &MergeSpecification) {
            assert!(!merge.merges.is_empty());
        }

        fn append_only_sizes(&self) -> (i32, i32) {
            (200_000, 1000)
        }

        fn updates_sizes(&self, _tc: &mut LuceneTestCase) -> (i32, i32) {
            (50_000, 500)
        }
    }

    #[test]
    fn force_merge_not_needed_skips_it() {
        let base = BaseMergePolicyTestCase::new(Suite);
        let mut tc = LuceneTestCase::new("skip");
        let mut after_policy = tc.random().clone();
        let _ = Suite.merge_policy(&mut after_policy);
        base.test_force_merge_not_needed(&mut tc);
        // Only the policy was drawn: the scenario stopped there.
        assert_eq!(tc.random().next_long(), after_policy.next_long());
        assert!(Suite.is_mock_random(&Suite.merge_policy(tc.random())));
    }

    #[test]
    fn find_forced_deletes_merges_skips_a_wrapper_of_it() {
        let suite = merge_on_flush::Suite;
        let mut r = JavaRandom::new(1);
        let wrapped = MergeOnFlushMergePolicy::new(Box::new(MockRandomMergePolicy::new(&mut r)));
        assert!(suite.wraps_mock_random(&wrapped));
        let plain = MergeOnFlushMergePolicy::new(Box::new(NoMergePolicy::default()));
        assert!(!suite.wraps_mock_random(&plain));
    }

    #[test]
    fn simulations_run_over_random_merges() {
        let base = BaseMergePolicyTestCase::new(Suite);
        let mut tc = LuceneTestCase::new("simulate");
        base.test_simulate_append_only(&mut tc);
        base.test_simulate_updates(&mut tc);
    }

    #[test]
    fn forced_merges_pack_every_eligible_segment() {
        let mut r = JavaRandom::new(7);
        let mp = MockRandomMergePolicy::new(&mut r);
        let infos: Vec<MergeSegment> = (0..25)
            .map(|i| MergeSegment::new(format!("_{i}"), 10, 0, 100))
            .collect();
        let ctx = BasicMergeContext::default();
        let mut to_merge =
            lucene_test_framework::index::base_merge_policy::all_segments_to_merge(&infos);
        to_merge.remove("_3");
        let spec = mp
            .find_forced_merges(&infos, 1, &to_merge, &ctx)
            .unwrap()
            .unwrap();
        let mut seen: Vec<String> = spec
            .merges
            .iter()
            .flat_map(|m| {
                assert!((1..=10).contains(&m.segments.len()));
                m.segments.iter().map(|s| s.name.clone())
            })
            .collect();
        seen.sort();
        let mut want: Vec<String> = to_merge.keys().cloned().collect();
        want.sort();
        assert_eq!(seen, want);
        // One eligible segment already merged: nothing to do.
        let one = [MergeSegment::new("_0", 10, 0, 100).with_compound_file(true)];
        let mut no_cfs = MockRandomMergePolicy::new(&mut r);
        no_cfs.compound_file_settings_mut().no_cfs_ratio = 1.0;
        let _ = no_cfs.find_forced_merges(
            &one,
            1,
            &lucene_test_framework::index::base_merge_policy::all_segments_to_merge(&one),
            &ctx,
        );
    }

    #[test]
    fn full_flush_merges_drop_merging_segments() {
        let mut r = JavaRandom::new(3);
        let mp = MockRandomMergePolicy::new(&mut r);
        // More than 30 segments: findMerges always merges.
        let infos: Vec<MergeSegment> = (0..40)
            .map(|i| MergeSegment::new(format!("_{i}"), 10, 0, 100))
            .collect();
        let merging: HashSet<String> = (0..20).map(|i| format!("_{i}")).collect();
        let mut ctx = MockMergeContext::new(del_count);
        ctx.set_merging_segments(merging.clone());
        for _ in 0..50 {
            if let Some(spec) = mp
                .find_full_flush_merges(MergeTrigger::FullFlush, &infos, &ctx)
                .unwrap()
            {
                for one in &spec.merges {
                    assert!(one.segments.iter().all(|s| !merging.contains(&s.name)));
                }
            }
            // Every segment free: more than 30, so findMerges always merges.
            assert!(mp
                .find_forced_deletes_merges(&infos, &BasicMergeContext::default())
                .unwrap()
                .is_some());
        }
        let cfs = (0..100)
            .filter(|_| mp.use_compound_file(&infos, &infos[0], &ctx).unwrap())
            .count();
        assert!((60..=95).contains(&cfs), "about 80% compound: {cfs}");
    }
}

#[test]
fn test_force_merge_policy() {
    let mp = ForceMergePolicy::new(Box::new(
        lucene_index::merge_policy::log::LogDocMergePolicy::new(),
    ));
    let ctx = BasicMergeContext::default();
    let infos: Vec<MergeSegment> = (0..20)
        .map(|i| MergeSegment::new(format!("_{i}"), 10, 0, 100))
        .collect();
    assert!(mp
        .find_merges(MergeTrigger::SegmentFlush, &infos, &ctx)
        .unwrap()
        .is_none());
    // Forced merges are the wrapped policy's.
    let all = lucene_test_framework::index::base_merge_policy::all_segments_to_merge(&infos);
    assert!(mp
        .find_forced_merges(&infos, 1, &all, &ctx)
        .unwrap()
        .is_some());
    assert_eq!(
        mp.compound_file_settings(),
        mp.unwrap().compound_file_settings()
    );
    // Everything but `findMerges` is `FilterMergePolicy`'s delegation.
    let readers = mp.find_merges_for_readers(&[10, 20]).unwrap();
    let inner = mp.unwrap().find_merges_for_readers(&[10, 20]).unwrap();
    assert_eq!(readers.merges.len(), inner.merges.len());
    assert_eq!(
        mp.keep_fully_deleted_segment(),
        mp.unwrap().keep_fully_deleted_segment()
    );
    assert_eq!(mp.num_deletes_to_merge(&infos[0], 3), 3);
}

#[test]
fn test_reverse_with_parents() {
    let dir = ByteBuffersDirectory::new();
    {
        let mut w = IndexWriter::open(
            &dir,
            Vec::new(),
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        w.set_parent_field(Some("_parent")).unwrap();
        let empty = || Document { fields: Vec::new() };
        w.add_documents((0..2).map(|_| empty()).collect()).unwrap();
        w.add_documents((0..4).map(|_| empty()).collect()).unwrap();
        w.add_documents((0..3).map(|_| empty()).collect()).unwrap();
        w.force_merge(1).unwrap();
        w.close().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 1);
    let doc_map = reverse(&reader.segment_readers()[0]).unwrap();
    let old_to_new: Vec<i32> = (0..9).map(|d| doc_map.old_to_new(d)).collect();
    let new_to_old: Vec<i32> = (0..9).map(|d| doc_map.new_to_old(d)).collect();
    assert_eq!(old_to_new, [7, 8, 3, 4, 5, 6, 0, 1, 2]);
    assert_eq!(new_to_old, [6, 7, 8, 2, 3, 4, 5, 0, 1]);
}

#[test]
fn reverse_without_parents_reverses_every_document() {
    let dir = ByteBuffersDirectory::new();
    {
        let mut w = IndexWriter::open(
            &dir,
            Vec::new(),
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        for _ in 0..4 {
            w.add_document(Document { fields: Vec::new() }).unwrap();
        }
        w.close().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let doc_map = reverse(&reader.segment_readers()[0]).unwrap();
    let new_to_old: Vec<i32> = (0..4).map(|d| doc_map.new_to_old(d)).collect();
    assert_eq!(new_to_old, [3, 2, 1, 0]);
}

#[test]
fn a_shared_live_policy_reaches_the_writer() {
    use lucene_test_framework::index::index_writer_config::LiveMergePolicy;
    let live = LiveMergePolicy::new(PolicyKind::NoMerge(NoMergePolicy::default()));
    let shared: Arc<dyn MergePolicy> = Arc::new(live.clone());
    let ctx = BasicMergeContext::default();
    let infos: Vec<MergeSegment> = (0..20)
        .map(|i| MergeSegment::new(format!("_{i}"), 10, 0, 100))
        .collect();
    assert!(shared
        .find_merges(MergeTrigger::SegmentFlush, &infos, &ctx)
        .unwrap()
        .is_none());
    live.change(|k| {
        *k = PolicyKind::Log(LogMergePolicy::clone(
            &lucene_index::merge_policy::log::LogDocMergePolicy::new(),
        ))
    });
    let _: &dyn MergeContext = &ctx;
    assert!(shared
        .find_merges(MergeTrigger::SegmentFlush, &infos, &ctx)
        .unwrap()
        .is_some());
}

/// Calls every `MergePolicy` method of `p`, comparing the deterministic ones
/// with `inner`'s: what a delegating wrapper must forward.
fn exercise_delegation(p: &mut dyn MergePolicy, inner: Option<&dyn MergePolicy>) {
    let ctx = BasicMergeContext::default();
    let infos: Vec<MergeSegment> = (0..12)
        .map(|i| MergeSegment::new(format!("_{i}"), 10 + i, i % 3, 1000 * i64::from(i + 1)))
        .collect();
    let all = lucene_test_framework::index::base_merge_policy::all_segments_to_merge(&infos);
    let _ = p
        .find_merges(MergeTrigger::SegmentFlush, &infos, &ctx)
        .unwrap();
    let _ = p.find_forced_merges(&infos, 1, &all, &ctx).unwrap();
    let _ = p.find_forced_deletes_merges(&infos, &ctx).unwrap();
    let _ = p
        .find_full_flush_merges(MergeTrigger::FullFlush, &infos, &ctx)
        .unwrap();
    let _ = p.use_compound_file(&infos, &infos[0], &ctx).unwrap();
    let _ = p.is_merged(&infos, &infos[0], &ctx).unwrap();
    let readers = p.find_merges_for_readers(&[3, 4]).map(|s| s.merges.len());
    let settings = p.compound_file_settings();
    assert_eq!(
        p.compound_file_settings_mut().no_cfs_ratio,
        settings.no_cfs_ratio
    );
    if let Some(inner) = inner {
        assert_eq!(
            p.size(&infos[2], &ctx).unwrap(),
            inner.size(&infos[2], &ctx).unwrap()
        );
        assert_eq!(
            p.max_full_flush_merge_size(),
            inner.max_full_flush_merge_size()
        );
        assert_eq!(
            p.keep_fully_deleted_segment(),
            inner.keep_fully_deleted_segment()
        );
        assert_eq!(
            p.num_deletes_to_merge(&infos[1], 5),
            inner.num_deletes_to_merge(&infos[1], 5)
        );
        assert_eq!(
            readers,
            inner
                .find_merges_for_readers(&[3, 4])
                .map(|s| s.merges.len())
        );
        assert_eq!(settings, inner.compound_file_settings());
    }
}

#[test]
fn wrappers_delegate_every_merge_policy_method() {
    use lucene_test_framework::index::index_writer_config::LiveMergePolicy;
    let mut r = JavaRandom::new(11);
    let kinds = vec![
        PolicyKind::Log(LogMergePolicy::clone(
            &lucene_index::merge_policy::log::LogDocMergePolicy::new(),
        )),
        PolicyKind::Tiered(TieredMergePolicy::default()),
        PolicyKind::MockRandom(MockRandomMergePolicy::new(&mut r)),
        PolicyKind::NoMerge(NoMergePolicy::default()),
        PolicyKind::Other(Box::new(TieredMergePolicy::default())),
    ];
    for mut kind in kinds {
        exercise_delegation(&mut kind, None);
        let mut live = LiveMergePolicy::new(kind);
        exercise_delegation(&mut live, None);
    }
    // The deterministic policies forward exactly.
    let mut tiered = PolicyKind::Tiered(TieredMergePolicy::default());
    exercise_delegation(&mut tiered, Some(&TieredMergePolicy::default()));
    let mut live = LiveMergePolicy::new(PolicyKind::NoMerge(NoMergePolicy::default()));
    exercise_delegation(&mut live, Some(&NoMergePolicy::default()));
    let mut force = ForceMergePolicy::new(Box::new(TieredMergePolicy::default()));
    exercise_delegation(&mut force, Some(&TieredMergePolicy::default()));
}

#[test]
fn every_merge_scheduler_choice_builds() {
    use lucene_test_framework::index::index_writer_config::MergeSchedulerChoice;
    for choice in [
        MergeSchedulerChoice::Serial,
        MergeSchedulerChoice::Concurrent {
            always_stall: false,
            max_merge_count: 4,
            max_thread_count: 2,
            auto_io_throttle: false,
            force_merge_mb_per_sec: 50.0,
        },
        MergeSchedulerChoice::Concurrent {
            always_stall: true,
            max_merge_count: 2,
            max_thread_count: 1,
            auto_io_throttle: true,
            force_merge_mb_per_sec: f64::INFINITY,
        },
        MergeSchedulerChoice::Default { test: true },
    ] {
        let _scheduler = choice.build();
    }
}

#[test]
fn alcoholic_policy_drinks_at_night() {
    use lucene_test_framework::index::merge_policies::{hour_of_day, AlcoholicMergePolicy};
    let ctx = BasicMergeContext::default();
    let info = MergeSegment::new("_0", 10, 0, 1000);
    let (mut drunk, mut sober) = (0, 0);
    for seed in 0..40 {
        // The policy's own first draw is its clock.
        let millis = test_util::next_long(&mut JavaRandom::new(seed), 0, i64::MAX);
        let night = !(6..=20).contains(&hour_of_day(millis, 0));
        let p = AlcoholicMergePolicy::new(0, JavaRandom::new(seed));
        for _ in 0..5 {
            let size = p.size(&info, &ctx).unwrap();
            if size == 1000 {
                sober += 1;
                assert!(!night, "seed {seed}: sober at night");
            } else {
                drunk += 1;
                assert!([15, 17, 21, 22, 30].contains(&(size / 1000)), "{size}");
            }
        }
    }
    assert!(drunk > 0 && sober > 0, "drunk {drunk}, sober {sober}");
}

#[test]
fn mock_random_one_merge_wraps_and_reorders_at_random() {
    use lucene_search::reader::merge_readers::MergeReaderHooks;
    use lucene_search::reader::CodecReader;
    use lucene_test_framework::index::merge_policies::MockRandomOneMerge;
    let dir = ByteBuffersDirectory::new();
    {
        let mut w = IndexWriter::open(
            &dir,
            vec![lucene_codecs::field_infos::FieldInfo::new("a", 0)],
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        for i in 0..4 {
            w.add_document(Document {
                fields: vec![lucene_codecs::stored_fields::StoredField {
                    field_number: 0,
                    value: lucene_codecs::stored_fields::FieldValue::Int(i),
                }],
            })
            .unwrap();
        }
        w.close().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let codec: Arc<dyn CodecReader> = Arc::new(reader.segment_readers()[0].clone());
    let (mut wrapped, mut reversed, mut kept) = (0, 0, 0);
    for seed in 0..40 {
        let hooks = MockRandomOneMerge::new(seed);
        let r = hooks.wrap_for_merge(Arc::clone(&codec)).unwrap();
        assert_eq!(r.max_doc(), 4);
        if !Arc::ptr_eq(&r, &codec) {
            wrapped += 1;
        }
        match hooks.reorder(codec.as_ref()).unwrap() {
            Some(map) => {
                reversed += 1;
                let order: Vec<i32> = (0..4).map(|d| map.new_to_old(d)).collect();
                assert_eq!(order, [3, 2, 1, 0]);
            }
            None => kept += 1,
        }
    }
    assert!(
        wrapped > 0 && reversed > 0 && kept > 0,
        "{wrapped} {reversed} {kept}"
    );
}

#[test]
fn mock_random_forced_merges_of_two_and_non_bulk_merges() {
    let mut r = JavaRandom::new(5);
    let mut mp = MockRandomMergePolicy::new(&mut r);
    mp.set_do_non_bulk_merges(false);
    let infos: Vec<MergeSegment> = (0..2)
        .map(|i| MergeSegment::new(format!("_{i}"), 10, 0, 100))
        .collect();
    let all = lucene_test_framework::index::base_merge_policy::all_segments_to_merge(&infos);
    let spec = mp
        .find_forced_merges(&infos, 1, &all, &BasicMergeContext::default())
        .unwrap()
        .unwrap();
    // Two eligible: one merge of both, no draw for its size.
    assert_eq!(spec.merges.len(), 1);
    assert_eq!(spec.merges[0].segments.len(), 2);
}

#[test]
fn reverse_keeps_a_one_document_first_block() {
    let dir = ByteBuffersDirectory::new();
    {
        let mut w = IndexWriter::open(
            &dir,
            Vec::new(),
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        w.set_parent_field(Some("_parent")).unwrap();
        let empty = || Document { fields: Vec::new() };
        w.add_documents(vec![empty()]).unwrap();
        w.add_documents((0..3).map(|_| empty()).collect()).unwrap();
        w.close().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let doc_map = reverse(&reader.segment_readers()[0]).unwrap();
    let new_to_old: Vec<i32> = (0..4).map(|d| doc_map.new_to_old(d)).collect();
    assert_eq!(new_to_old, [1, 2, 3, 0]);
}
