//! Port of lucene-sandbox's `org.apache.lucene.sandbox.index.MergeOnFlushMergePolicy`:
//! a [`FilterMergePolicy`] whose `findFullFlushMerges` merges, on a commit
//! or an NRT reopen, every segment smaller than a threshold (100 MB by
//! default, `SegmentCommitInfo.sizeInBytes()` against it) that no running
//! merge holds -- one merge of all of them, when there are at least two.
//! Every other decision is the wrapped policy's.
//!
//! The threshold is kept in bytes, as Java's `(long) (mb * 1024 * 1024)`, and
//! read back as `bytes / 1024. / 1024.`; the writer only asks for full-flush
//! merges when `IndexWriterConfig.maxFullFlushMergeWaitMillis` is positive.

use std::collections::HashMap;

use super::api::{
    CompoundFileSettings, MergeContext, MergePolicy, MergeSegment, MergeSpecification,
    MergeTrigger, OneMerge, Result,
};
use super::filter::FilterMergePolicy;

/// `Units.mbToBytes(megabytes)`: Java's `(long)` cast saturates and maps
/// `NaN` to `0`, as `f64 as i64` does.
fn mb_to_bytes(megabytes: f64) -> i64 {
    (megabytes * 1024.0 * 1024.0) as i64
}

/// `Units.bytesToMB(bytes)`.
fn bytes_to_mb(bytes: i64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

/// `MergeOnFlushMergePolicy(mergePolicy)`.
#[derive(Debug)]
pub struct MergeOnFlushMergePolicy {
    filter: FilterMergePolicy,
    small_segment_threshold_bytes: i64,
}

impl MergeOnFlushMergePolicy {
    /// `new MergeOnFlushMergePolicy(mergePolicy)`: a 100 MB threshold.
    pub fn new(inner: Box<dyn MergePolicy>) -> Self {
        MergeOnFlushMergePolicy {
            filter: FilterMergePolicy::new(inner),
            small_segment_threshold_bytes: mb_to_bytes(100.0),
        }
    }

    /// `getSmallSegmentThresholdMB()`.
    pub fn small_segment_threshold_mb(&self) -> f64 {
        bytes_to_mb(self.small_segment_threshold_bytes)
    }

    /// `setSmallSegmentThresholdMB(mb)`.
    pub fn set_small_segment_threshold_mb(&mut self, mb: f64) {
        self.small_segment_threshold_bytes = mb_to_bytes(mb);
    }

    /// `Unwrappable.unwrap`.
    pub fn unwrap(&self) -> &dyn MergePolicy {
        self.filter.unwrap()
    }
}

impl MergePolicy for MergeOnFlushMergePolicy {
    fn find_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        self.filter.find_merges(trigger, infos, ctx)
    }

    fn find_forced_merges(
        &self,
        infos: &[MergeSegment],
        max_segment_count: i32,
        segments_to_merge: &HashMap<String, bool>,
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        self.filter
            .find_forced_merges(infos, max_segment_count, segments_to_merge, ctx)
    }

    fn find_forced_deletes_merges(
        &self,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        self.filter.find_forced_deletes_merges(infos, ctx)
    }

    fn find_merges_for_readers(&self, max_docs: &[i64]) -> Option<MergeSpecification> {
        self.filter.find_merges_for_readers(max_docs)
    }

    /// `findFullFlushMerges`: one merge of the small segments not already
    /// merging, in the commit's order, when there are two or more.
    fn find_full_flush_merges(
        &self,
        _trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        let merging = ctx.merging_segments();
        let small: Vec<MergeSegment> = infos
            .iter()
            .filter(|sci| sci.size_in_bytes < self.small_segment_threshold_bytes)
            .filter(|sci| !merging.contains(&sci.name))
            .cloned()
            .collect();
        if small.len() > 1 {
            let mut spec = MergeSpecification::new();
            spec.add(OneMerge::new(small));
            return Ok(Some(spec));
        }
        Ok(None)
    }

    fn use_compound_file(
        &self,
        infos: &[MergeSegment],
        merged: &MergeSegment,
        ctx: &dyn MergeContext,
    ) -> Result<bool> {
        self.filter.use_compound_file(infos, merged, ctx)
    }

    fn size(&self, info: &MergeSegment, ctx: &dyn MergeContext) -> Result<i64> {
        self.filter.size(info, ctx)
    }

    fn max_full_flush_merge_size(&self) -> i64 {
        self.filter.max_full_flush_merge_size()
    }

    fn keep_fully_deleted_segment(&self) -> bool {
        self.filter.keep_fully_deleted_segment()
    }

    fn num_deletes_to_merge(&self, info: &MergeSegment, del_count: i32) -> i32 {
        self.filter.num_deletes_to_merge(info, del_count)
    }

    fn compound_file_settings(&self) -> CompoundFileSettings {
        self.filter.compound_file_settings()
    }

    fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
        self.filter.compound_file_settings_mut()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::merge_policy::api::{BasicMergeContext, NoMergePolicy, TieredMergePolicy};

    fn seg(name: &str, bytes: i64) -> MergeSegment {
        MergeSegment::new(name, 10, 0, bytes)
    }

    fn names(spec: &MergeSpecification) -> Vec<Vec<String>> {
        spec.merges
            .iter()
            .map(|m| m.segments.iter().map(|s| s.name.clone()).collect())
            .collect()
    }

    #[test]
    fn merges_the_small_segments_not_already_merging() {
        let mut p = MergeOnFlushMergePolicy::new(Box::new(NoMergePolicy::default()));
        assert_eq!(p.small_segment_threshold_mb(), 100.0);
        p.set_small_segment_threshold_mb(1.0);
        assert_eq!(p.small_segment_threshold_mb(), 1.0);
        let infos = [
            seg("_0", 5 << 20),
            seg("_1", 100),
            seg("_2", (1 << 20) - 1),
            seg("_3", 1 << 20),
            seg("_4", 7),
        ];
        let ctx = BasicMergeContext::new(HashSet::from(["_4".to_string()]));
        let spec = p
            .find_full_flush_merges(MergeTrigger::FullFlush, &infos, &ctx)
            .unwrap()
            .unwrap();
        assert_eq!(names(&spec), vec![vec!["_1", "_2"]]);
        // One small segment left: nothing to merge.
        let ctx = BasicMergeContext::new(HashSet::from(["_2".to_string(), "_4".to_string()]));
        assert!(p
            .find_full_flush_merges(MergeTrigger::FullFlush, &infos, &ctx)
            .unwrap()
            .is_none());
    }

    #[test]
    fn the_threshold_round_trips_as_javas() {
        let mut p = MergeOnFlushMergePolicy::new(Box::new(NoMergePolicy::default()));
        p.set_small_segment_threshold_mb(0.3);
        // `(long) (0.3 * 1024 * 1024)` truncates.
        assert_eq!(p.small_segment_threshold_bytes, 314_572);
        assert_eq!(p.small_segment_threshold_mb(), 314_572.0 / 1024.0 / 1024.0);
        p.set_small_segment_threshold_mb(f64::NAN);
        assert_eq!(p.small_segment_threshold_bytes, 0);
        p.set_small_segment_threshold_mb(f64::INFINITY);
        assert_eq!(p.small_segment_threshold_bytes, i64::MAX);
    }

    #[test]
    fn everything_else_is_the_wrapped_policys() {
        let mut p = MergeOnFlushMergePolicy::new(Box::new(TieredMergePolicy::default()));
        let tiered = TieredMergePolicy::default();
        let ctx = BasicMergeContext::new(HashSet::new());
        let infos: Vec<MergeSegment> = (0..12).map(|i| seg(&format!("_{i}"), 1000)).collect();
        assert_eq!(
            p.find_merges(MergeTrigger::SegmentFlush, &infos, &ctx)
                .unwrap()
                .map(|s| names(&s)),
            tiered
                .find_merges(MergeTrigger::SegmentFlush, &infos, &ctx)
                .unwrap()
                .map(|s| names(&s))
        );
        assert_eq!(
            p.find_forced_merges(&infos, 1, &HashMap::new(), &ctx)
                .unwrap()
                .map(|s| names(&s)),
            tiered
                .find_forced_merges(&infos, 1, &HashMap::new(), &ctx)
                .unwrap()
                .map(|s| names(&s))
        );
        assert_eq!(
            p.find_forced_deletes_merges(&infos, &ctx)
                .unwrap()
                .is_some(),
            tiered
                .find_forced_deletes_merges(&infos, &ctx)
                .unwrap()
                .is_some()
        );
        assert_eq!(
            p.find_merges_for_readers(&[1, 2]).map(|s| s.merges.len()),
            Some(1)
        );
        assert_eq!(
            p.use_compound_file(&infos, &infos[0], &ctx).unwrap(),
            tiered.use_compound_file(&infos, &infos[0], &ctx).unwrap()
        );
        assert_eq!(p.size(&infos[0], &ctx).unwrap(), 1000);
        assert_eq!(
            p.max_full_flush_merge_size(),
            tiered.max_full_flush_merge_size()
        );
        assert_eq!(
            p.keep_fully_deleted_segment(),
            tiered.keep_fully_deleted_segment()
        );
        assert_eq!(p.num_deletes_to_merge(&infos[0], 3), 3);
        assert_eq!(
            p.compound_file_settings().no_cfs_ratio,
            tiered.compound_file_settings().no_cfs_ratio
        );
        p.compound_file_settings_mut().no_cfs_ratio = 0.5;
        assert_eq!(p.unwrap().compound_file_settings().no_cfs_ratio, 0.5);
        assert!(format!("{p:?}").contains("MergeOnFlushMergePolicy"));
    }
}
