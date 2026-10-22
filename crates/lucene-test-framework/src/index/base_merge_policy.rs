//! `BaseMergePolicyTestCase`: the checks every merge policy must pass, over
//! simulated segment lists (no index on disk) and one real writer.
//!
//! A suite is a [`MergePolicySuite`]: Java's three abstract methods
//! (`mergePolicy`, `assertSegmentInfos`, `assertMerge`).
//! [`BaseMergePolicyTestCase`] runs Java's scenarios over it, and
//! [`base_merge_policy_tests!`](crate::base_merge_policy_tests) declares one
//! `#[test]` per scenario.
//!
//! Differs:
//!
//! - A segment is a [`MergeSegment`] (name, `maxDoc`, deletes, size), what
//!   this port's policies read, not a `SegmentCommitInfo` over a fake
//!   directory; [`make_segment_commit_info`] keeps Java's draws (the 16-byte
//!   id) so a seed walks the same stream. The `source` attribute
//!   (`IndexWriter.SOURCE`) is not carried: no policy here reads it.
//! - `testForceMergeNotNeeded` cannot intercept the writer's merge
//!   scheduler, so it checks the outcome instead: a `forceMerge` asked for
//!   no fewer segments than the index has leaves every segment as it was.
//!   Its segments are made by commits (Java's NRT reopen flushes them),
//!   through a default writer rather than `newIndexWriterConfig`'s random
//!   one: only the merge policy under test matters to the outcome.
//! - `IOStats`' fields are Java's `long`s accumulated with `+=` of a
//!   `double`, which truncates at every step; that is kept.

use std::collections::{HashMap, HashSet};

use lucene_codecs::stored_fields::Document;
use lucene_index::merge_policy::{
    MergeContext, MergePolicy, MergeSegment, MergeSpecification, MergeTrigger, OneMerge,
};
use lucene_store::ByteBuffersDirectory;
use lucene_util::java_random::JavaRandom;

use crate::store::base_lock_factory::open_writer;
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// The policy-specific half of a merge-policy test: Java's abstract
/// methods.
pub trait MergePolicySuite {
    /// The policy under test: the hooks read its own settings, as Java's
    /// downcast in them does.
    type Policy: MergePolicy + 'static;
    /// `mergePolicy()`: a new policy.
    fn merge_policy(&self, random: &mut JavaRandom) -> Self::Policy;
    /// `assertSegmentInfos(policy, infos)`: the segment list matches what
    /// the policy promises, for segments only flushed or merged by it.
    fn assert_segment_infos(&self, policy: &Self::Policy, infos: &[MergeSegment]);
    /// `assertMerge(policy, merge)`: a merge the policy returned is one it
    /// may return.
    fn assert_merge(&self, policy: &Self::Policy, merge: &MergeSpecification);
    /// `mp.toString().contains("MockRandomMergePolicy")` (here its `Debug`):
    /// `testForceMergeNotNeeded`, which cannot tolerate random forced
    /// merges, is skipped for it.
    fn is_mock_random(&self, policy: &Self::Policy) -> bool {
        format!("{policy:?}").contains("MockRandomMergePolicy")
    }
    /// `((FilterMergePolicy) mp).unwrap() instanceof MockRandomMergePolicy`:
    /// `testFindForcedDeletesMerges` is skipped for a wrapper of one (and
    /// only for a wrapper, as in Java).
    fn wraps_mock_random(&self, _policy: &Self::Policy) -> bool {
        false
    }
    /// What a suite's `testSimulateAppendOnly`/`testSimulateUpdates`
    /// override changes on the policy before simulating (none by default).
    fn prepare_simulation(&self, _policy: &mut Self::Policy, _random: &mut JavaRandom) {}
    /// `testSimulateAppendOnly`'s `(totalDocs, maxDocsPerFlush)`.
    fn append_only_sizes(&self) -> (i32, i32) {
        (100_000_000, 10_000)
    }
    /// `testSimulateUpdates`' `(totalDocs, maxDocsPerFlush)`: by default
    /// `atLeast(1_000_000)` documents, 2500 at most per flush.
    fn updates_sizes(&self, tc: &mut LuceneTestCase) -> (i32, i32) {
        (tc.at_least(1_000_000), 2500)
    }
}

/// `MockMergeContext`: deletes as a function of the segment, and a merging
/// set the test controls.
pub struct MockMergeContext<F> {
    num_deletes: F,
    merging: HashSet<String>,
}

impl<F: Fn(&MergeSegment) -> i32> MockMergeContext<F> {
    /// `new MockMergeContext(numDeletesFunc)`.
    pub fn new(num_deletes: F) -> Self {
        Self {
            num_deletes,
            merging: HashSet::new(),
        }
    }

    /// `setMergingSegments`.
    pub fn set_merging_segments(&mut self, merging: HashSet<String>) {
        self.merging = merging;
    }
}

impl<F: Fn(&MergeSegment) -> i32> MergeContext for MockMergeContext<F> {
    fn num_deletes_to_merge(
        &self,
        info: &MergeSegment,
    ) -> lucene_index::merge_policy::api::Result<i32> {
        Ok((self.num_deletes)(info))
    }
    fn num_deleted_docs(&self, info: &MergeSegment) -> i32 {
        (self.num_deletes)(info)
    }
    fn merging_segments(&self) -> &HashSet<String> {
        &self.merging
    }
}

/// `SegmentCommitInfo::getDelCount`, the deletes function the simulations
/// use.
pub fn del_count(info: &MergeSegment) -> i32 {
    info.del_count
}

/// `makeSegmentCommitInfo(name, maxDoc, numDeletedDocs, sizeMB, source)`:
/// a segment whose `sizeInBytes` is `(long) (sizeMB * 1024 * 1024)` (Java's
/// fake directory reads it off the file name). Draws Java's 16-byte id.
///
/// # Panics
/// When `name` does not start with `_`, as Java's
/// `IllegalArgumentException`.
pub fn make_segment_commit_info(
    random: &mut JavaRandom,
    name: &str,
    max_doc: i32,
    num_deleted_docs: i32,
    size_mb: f64,
) -> MergeSegment {
    assert!(
        name.starts_with('_'),
        "name must start with an _, got {name}"
    );
    let mut id = [0u8; 16];
    random.next_bytes(&mut id);
    MergeSegment::new(
        name,
        max_doc,
        num_deleted_docs,
        (size_mb * 1024.0 * 1024.0) as i64,
    )
}

/// `IOStats`: bytes written by flushes and by merges.
#[derive(Debug, Default, Clone, Copy)]
pub struct IoStats {
    /// `flushBytesWritten`.
    pub flush_bytes_written: i64,
    /// `mergeBytesWritten`.
    pub merge_bytes_written: i64,
}

/// Java's `long += double`: the sum converted back to `long`.
fn add_bytes(total: &mut i64, bytes: f64) {
    *total = (*total as f64 + bytes) as i64;
}

/// `applyMerge(infos, merge, mergedSegmentName, stats)`: `infos` with the
/// merge's segments replaced, where the first of them was, by one merged
/// segment of their live documents.
pub fn apply_merge(
    random: &mut JavaRandom,
    infos: &[MergeSegment],
    merge: &OneMerge,
    merged_segment_name: &str,
    stats: &mut IoStats,
) -> Vec<MergeSegment> {
    let mut new_max_doc = 0i32;
    let mut new_size = 0f64;
    for sci in &merge.segments {
        let num_live_docs = sci.max_doc - sci.del_count;
        new_size += sci.size_in_bytes as f64 * f64::from(num_live_docs)
            / f64::from(sci.max_doc)
            / 1024.0
            / 1024.0;
        new_max_doc += num_live_docs;
    }
    let merged = make_segment_commit_info(random, merged_segment_name, new_max_doc, 0, new_size);
    let merged_away: HashSet<&str> = merge.segments.iter().map(|s| s.name.as_str()).collect();
    let mut merged_segment_added = false;
    let mut new_infos = Vec::with_capacity(infos.len());
    for info in infos {
        if merged_away.contains(info.name.as_str()) {
            if !merged_segment_added {
                new_infos.push(merged.clone());
                merged_segment_added = true;
            }
        } else {
            new_infos.push(info.clone());
        }
    }
    add_bytes(&mut stats.merge_bytes_written, new_size * 1024.0 * 1024.0);
    new_infos
}

/// `applyDeletes(infos, numDeletes)`: `num_deletes` spread over the
/// segments in proportion to their live documents, dropping a segment left
/// with none.
///
/// # Panics
/// With more deletes than live documents, as Java's
/// `IllegalArgumentException`.
pub fn apply_deletes(infos: &[MergeSegment], mut num_deletes: i32) -> Vec<MergeSegment> {
    let total_num_docs: i32 = infos.iter().map(|s| s.max_doc - s.del_count).sum();
    assert!(num_deletes <= total_num_docs, "More deletes than documents");
    let w = f64::from(num_deletes) / f64::from(total_num_docs);
    let mut new_infos = Vec::with_capacity(infos.len());
    for (i, sci) in infos.iter().enumerate() {
        assert!(num_deletes >= 0);
        let seg_deletes = if i == infos.len() - 1 {
            num_deletes
        } else {
            num_deletes.min((w * f64::from(sci.max_doc - sci.del_count)).ceil() as i32)
        };
        let new_del_count = sci.del_count + seg_deletes;
        assert!(new_del_count <= sci.max_doc);
        if new_del_count < sci.max_doc {
            let mut info = sci.clone();
            info.del_count = new_del_count;
            new_infos.push(info);
        }
        num_deletes -= seg_deletes;
    }
    assert_eq!(num_deletes, 0);
    new_infos
}

/// `BaseMergePolicyTestCase` over one [`MergePolicySuite`].
pub struct BaseMergePolicyTestCase<S> {
    suite: S,
}

impl<S: MergePolicySuite> BaseMergePolicyTestCase<S> {
    /// The scenarios over `suite`.
    pub fn new(suite: S) -> Self {
        Self { suite }
    }

    /// The suite the scenarios run over.
    pub fn suite(&self) -> &S {
        &self.suite
    }

    /// `testForceMergeNotNeeded`: a `forceMerge` to no fewer segments than
    /// the index has merges nothing.
    pub fn test_force_merge_not_needed(&self, tc: &mut LuceneTestCase) {
        let mut mp = self.suite.merge_policy(tc.random());
        if self.suite.is_mock_random(&mp) {
            // assumeFalse: "this test cannot tolerate random forceMerges".
            return;
        }
        mp.compound_file_settings_mut().no_cfs_ratio =
            if tc.random().next_boolean() { 0.0 } else { 1.0 };
        let dir = ByteBuffersDirectory::new();
        let mut writer = open_writer(&dir).unwrap();
        writer.set_pluggable_merge_policy(Some(std::sync::Arc::new(mp)));
        let num_segments = test_util::next_int(tc.random(), 2, 20);
        for _ in 0..num_segments {
            let num_docs = test_util::next_int(tc.random(), 1, 5);
            for _ in 0..num_docs {
                writer
                    .add_document(Document { fields: Vec::new() })
                    .unwrap();
            }
            writer.commit().unwrap();
        }
        for i in (0..=5).rev() {
            let before: Vec<String> = writer
                .segment_infos()
                .segments
                .iter()
                .map(|s| s.segment_name.clone())
                .collect();
            let max_num_segments = if i == 0 {
                1
            } else {
                test_util::next_int(tc.random(), 1, 10)
            };
            let may_merge = before.len() > max_num_segments as usize;
            writer.force_merge(max_num_segments as usize).unwrap();
            let after: Vec<String> = writer
                .segment_infos()
                .segments
                .iter()
                .map(|s| s.segment_name.clone())
                .collect();
            if !may_merge {
                assert_eq!(
                    before, after,
                    "we should not need any merging (maxNumSegments={max_num_segments}), yet the policy merged"
                );
            }
        }
    }

    /// `testFindForcedDeletesMerges`: with no deletes anywhere, a
    /// forced-deletes merge finds nothing to do.
    pub fn test_find_forced_deletes_merges(&self, tc: &mut LuceneTestCase) {
        let mp = self.suite.merge_policy(tc.random());
        if self.suite.wraps_mock_random(&mp) {
            // assumeFalse: "test doesn't work with MockRandomMP".
            return;
        }
        let ctx = MockMergeContext::new(|_: &MergeSegment| 0);
        let num_segs = tc.random().next_int_bounded(10);
        let mut infos = Vec::new();
        for _ in 0..num_segs {
            let r = tc.random();
            let name = test_util::random_simple_string(r);
            let max_doc = r.next_int_bounded(i32::MAX);
            let use_compound_file = r.next_boolean();
            let _id = test_util::random_simple_string_between(r, 16, 16);
            let del_count = r.next_int_bounded(1);
            let mut info = MergeSegment::new(name, max_doc, del_count, 0);
            info.use_compound_file = use_compound_file;
            infos.push(info);
        }
        if let Some(spec) = mp.find_forced_deletes_merges(&infos, &ctx).unwrap() {
            assert_eq!(spec.merges.len(), 0);
        }
    }

    /// `testSimulateAppendOnly`.
    pub fn test_simulate_append_only(&self, tc: &mut LuceneTestCase) {
        let mut mp = self.suite.merge_policy(tc.random());
        self.suite.prepare_simulation(&mut mp, tc.random());
        let (total_docs, max_docs_per_flush) = self.suite.append_only_sizes();
        self.do_test_simulate_append_only(tc, &mp, total_docs, max_docs_per_flush);
    }

    /// `doTestSimulateAppendOnly(mergePolicy, totalDocs, maxDocsPerFlush)`:
    /// `total_docs` documents flushed at most `max_docs_per_flush` at a
    /// time, no deletes; returns the write statistics.
    pub fn do_test_simulate_append_only(
        &self,
        tc: &mut LuceneTestCase,
        mp: &S::Policy,
        total_docs: i32,
        max_docs_per_flush: i32,
    ) -> IoStats {
        let mut stats = IoStats::default();
        let mut seg_name = 0u64;
        let ctx = MockMergeContext::new(del_count);
        let mut infos: Vec<MergeSegment> = Vec::new();
        let avg_doc_size_mb = 5.0 / 1024.0;
        let mut num_docs = 0;
        while num_docs < total_docs {
            let r = tc.random();
            let flush_doc_count = test_util::next_int(r, 1, max_docs_per_flush);
            num_docs += flush_doc_count;
            let flush_size_mb = f64::from(flush_doc_count) * avg_doc_size_mb;
            add_bytes(
                &mut stats.flush_bytes_written,
                flush_size_mb * 1024.0 * 1024.0,
            );
            infos.push(make_segment_commit_info(
                r,
                &format!("_{seg_name}"),
                flush_doc_count,
                0,
                flush_size_mb,
            ));
            seg_name += 1;
            let mut merges = mp
                .find_full_flush_merges(MergeTrigger::SegmentFlush, &infos, &ctx)
                .unwrap();
            if merges.is_none() {
                merges = mp
                    .find_merges(MergeTrigger::SegmentFlush, &infos, &ctx)
                    .unwrap();
            }
            while let Some(spec) = merges {
                assert!(!spec.merges.is_empty());
                self.suite.assert_merge(mp, &spec);
                for one in &spec.merges {
                    infos = apply_merge(
                        tc.random(),
                        &infos,
                        one,
                        &format!("_{seg_name}"),
                        &mut stats,
                    );
                    seg_name += 1;
                }
                merges = mp
                    .find_merges(MergeTrigger::MergeFinished, &infos, &ctx)
                    .unwrap();
            }
            self.suite.assert_segment_infos(mp, &infos);
        }
        stats
    }

    /// `testSimulateUpdates`.
    pub fn test_simulate_updates(&self, tc: &mut LuceneTestCase) {
        let mut mp = self.suite.merge_policy(tc.random());
        self.suite.prepare_simulation(&mut mp, tc.random());
        let (total_docs, max_docs_per_flush) = self.suite.updates_sizes(tc);
        self.do_test_simulate_updates(tc, &mp, total_docs, max_docs_per_flush);
    }

    /// `doTestSimulateUpdates(mergePolicy, totalDocs, maxDocsPerFlush)`:
    /// flushes whose documents are, more and more, updates of documents
    /// spread uniformly over the segments.
    pub fn do_test_simulate_updates(
        &self,
        tc: &mut LuceneTestCase,
        mp: &S::Policy,
        total_docs: i32,
        max_docs_per_flush: i32,
    ) -> IoStats {
        let mut stats = IoStats::default();
        let mut seg_name = 0u64;
        let ctx = MockMergeContext::new(del_count);
        let mut infos: Vec<MergeSegment> = Vec::new();
        let avg_doc_size_mb = 5.0 / 1024.0;
        let mut num_docs = 0;
        while num_docs < total_docs {
            let flush_doc_count = if tc.usually() {
                // reasonable value
                test_util::next_int(tc.random(), max_docs_per_flush / 2, max_docs_per_flush)
            } else {
                // crazy value
                test_util::next_int(tc.random(), 1, max_docs_per_flush)
            };
            // how many of these documents are actually updates
            let del_count = (f64::from(flush_doc_count) * 0.9 * f64::from(num_docs)
                / f64::from(total_docs)) as i32;
            num_docs += flush_doc_count - del_count;
            infos = apply_deletes(&infos, del_count);
            let flush_size = f64::from(flush_doc_count) * avg_doc_size_mb;
            add_bytes(&mut stats.flush_bytes_written, flush_size * 1024.0 * 1024.0);
            infos.push(make_segment_commit_info(
                tc.random(),
                &format!("_{seg_name}"),
                flush_doc_count,
                0,
                flush_size,
            ));
            seg_name += 1;
            let mut merges = mp
                .find_full_flush_merges(MergeTrigger::SegmentFlush, &infos, &ctx)
                .unwrap();
            if merges.is_none() {
                merges = mp
                    .find_merges(MergeTrigger::SegmentFlush, &infos, &ctx)
                    .unwrap();
            }
            while let Some(spec) = merges {
                self.suite.assert_merge(mp, &spec);
                for one in &spec.merges {
                    infos = apply_merge(
                        tc.random(),
                        &infos,
                        one,
                        &format!("_{seg_name}"),
                        &mut stats,
                    );
                    seg_name += 1;
                }
                merges = mp
                    .find_merges(MergeTrigger::MergeFinished, &infos, &ctx)
                    .unwrap();
            }
            self.suite.assert_segment_infos(mp, &infos);
        }
        stats
    }

    /// `testNoPathologicalMerges`: tiny flushes of tiny documents must not
    /// merge quadratically -- write amplification stays under
    /// `log(numFlushes) / log(1.5)`.
    pub fn test_no_pathological_merges(&self, tc: &mut LuceneTestCase) {
        let mp = self.suite.merge_policy(tc.random());
        let mp = &mp;
        let mut stats = IoStats::default();
        let mut seg_name = 0u64;
        let ctx = MockMergeContext::new(del_count);
        let mut infos: Vec<MergeSegment> = Vec::new();
        let avg_doc_size_mb = 10.0 / 1024.0 / 1024.0;
        let max_docs_per_flush = 3;
        let total_docs = 10_000;
        let mut num_flushes = 0i32;
        let mut num_docs = 0;
        while num_docs < total_docs {
            let r = tc.random();
            let flush_doc_count = test_util::next_int(r, 1, max_docs_per_flush);
            num_docs += flush_doc_count;
            let flush_size_mb = f64::from(flush_doc_count) * avg_doc_size_mb;
            add_bytes(
                &mut stats.flush_bytes_written,
                flush_size_mb * 1024.0 * 1024.0,
            );
            infos.push(make_segment_commit_info(
                r,
                &format!("_{seg_name}"),
                flush_doc_count,
                0,
                flush_size_mb,
            ));
            seg_name += 1;
            num_flushes += 1;
            let mut merges = mp
                .find_merges(MergeTrigger::SegmentFlush, &infos, &ctx)
                .unwrap();
            while let Some(spec) = merges {
                assert!(!spec.merges.is_empty());
                self.suite.assert_merge(mp, &spec);
                for one in &spec.merges {
                    infos = apply_merge(
                        tc.random(),
                        &infos,
                        one,
                        &format!("_{seg_name}"),
                        &mut stats,
                    );
                    seg_name += 1;
                }
                merges = mp
                    .find_merges(MergeTrigger::MergeFinished, &infos, &ctx)
                    .unwrap();
            }
            self.suite.assert_segment_infos(mp, &infos);
        }
        let write_amplification = (stats.flush_bytes_written + stats.merge_bytes_written) as f64
            / stats.flush_bytes_written as f64;
        let max_allowed = f64::from(num_flushes).ln() / 1.5f64.ln();
        assert!(
            write_amplification < max_allowed,
            "write amplification {write_amplification} >= {max_allowed}"
        );
    }
}

/// The segments-to-merge map `findForcedMerges` takes, every segment of
/// `infos` original (`true`): what `forceMerge` passes.
pub fn all_segments_to_merge(infos: &[MergeSegment]) -> HashMap<String, bool> {
    infos.iter().map(|s| (s.name.clone(), true)).collect()
}

/// Declares one `#[test]` per [`BaseMergePolicyTestCase`] scenario over
/// `$suite` (an expression building the [`MergePolicySuite`]).
#[macro_export]
macro_rules! base_merge_policy_tests {
    ($suite:expr) => {
        $crate::base_merge_policy_tests!(@each $suite;
            test_force_merge_not_needed, test_find_forced_deletes_merges,
            test_simulate_append_only, test_simulate_updates, test_no_pathological_merges);
    };
    (@each $suite:expr; $($name:ident),*) => {$(
        #[test]
        fn $name() {
            $crate::index::base_merge_policy::BaseMergePolicyTestCase::new($suite)
                .$name(&mut $crate::util::lucene_test_case::LuceneTestCase::new(stringify!($name)));
        }
    )*};
}
