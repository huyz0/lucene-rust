//! Port of `org.apache.lucene.index.UpgradeIndexMergePolicy`: a
//! `FilterMergePolicy` whose forced merges only rewrite segments an older
//! Lucene wrote, so `forceMerge` "upgrades" an index (M8) instead of merging
//! it down. Every other decision is the wrapped policy's.
//!
//! Rust-only differences: Java's `shouldUpgradeSegment` is an overridable
//! method; here it is [`UpgradeIndexMergePolicy::should_upgrade_segment`]
//! over the segment's [`MergeSegment::version`], and a segment whose
//! version the caller did not supply counts as old. Java calls the wrapped
//! policy's `findMerges` with a `null` trigger; this port has no null
//! trigger and passes the one it was given (`TieredMergePolicy`, the default
//! delegate, ignores it either way).

use std::collections::HashMap;

use super::api::{
    CompoundFileSettings, MergeContext, MergePolicy, MergeSegment, MergeSpecification,
    MergeTrigger, OneMerge, Result,
};
use crate::segment_info::LuceneVersion;

/// `Version.LATEST` of the Lucene this port pins.
pub const LATEST: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// `UpgradeIndexMergePolicy`.
#[derive(Debug)]
pub struct UpgradeIndexMergePolicy {
    inner: Box<dyn MergePolicy>,
}

impl UpgradeIndexMergePolicy {
    /// `new UpgradeIndexMergePolicy(in)`.
    pub fn new(inner: Box<dyn MergePolicy>) -> Self {
        UpgradeIndexMergePolicy { inner }
    }

    /// `Unwrappable.unwrap`.
    pub fn unwrap(&self) -> &dyn MergePolicy {
        self.inner.as_ref()
    }

    /// `shouldUpgradeSegment(si)`: `!Version.LATEST.equals(si.info.getVersion())`.
    pub fn should_upgrade_segment(&self, si: &MergeSegment) -> bool {
        si.version != Some(LATEST)
    }
}

impl MergePolicy for UpgradeIndexMergePolicy {
    fn find_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        self.inner.find_merges(trigger, infos, ctx)
    }

    /// Only the old segments of `segments_to_merge` go to the wrapped
    /// policy; whichever of them its specification leaves out are merged
    /// together into one more segment.
    fn find_forced_merges(
        &self,
        infos: &[MergeSegment],
        max_segment_count: i32,
        segments_to_merge: &HashMap<String, bool>,
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        // "first find all old segments"
        let mut old_segments: HashMap<String, bool> = HashMap::new();
        for si in infos {
            if let Some(&v) = segments_to_merge.get(&si.name) {
                if self.should_upgrade_segment(si) {
                    old_segments.insert(si.name.clone(), v);
                }
            }
        }
        if old_segments.is_empty() {
            return Ok(None);
        }
        let mut spec =
            self.inner
                .find_forced_merges(infos, max_segment_count, &old_segments, ctx)?;
        if let Some(spec) = &spec {
            // Whatever the wrapped policy merges is upgraded by that merge.
            for merge in &spec.merges {
                for s in &merge.segments {
                    old_segments.remove(&s.name);
                }
            }
        }
        if !old_segments.is_empty() {
            // "does not want to merge all old segments, merge remaining ones
            // into new segment", in `segmentInfos` order.
            let new_infos: Vec<MergeSegment> = infos
                .iter()
                .filter(|si| old_segments.contains_key(&si.name))
                .cloned()
                .collect();
            spec.get_or_insert_with(MergeSpecification::new)
                .add(OneMerge::new(new_infos));
        }
        Ok(spec)
    }

    fn find_forced_deletes_merges(
        &self,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        self.inner.find_forced_deletes_merges(infos, ctx)
    }

    fn find_merges_for_readers(&self, max_docs: &[i64]) -> Option<MergeSpecification> {
        self.inner.find_merges_for_readers(max_docs)
    }

    fn find_full_flush_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        self.inner.find_full_flush_merges(trigger, infos, ctx)
    }

    fn use_compound_file(
        &self,
        infos: &[MergeSegment],
        merged: &MergeSegment,
        ctx: &dyn MergeContext,
    ) -> Result<bool> {
        self.inner.use_compound_file(infos, merged, ctx)
    }

    fn size(&self, info: &MergeSegment, ctx: &dyn MergeContext) -> Result<i64> {
        self.inner.size(info, ctx)
    }

    fn max_full_flush_merge_size(&self) -> i64 {
        self.inner.max_full_flush_merge_size()
    }

    fn keep_fully_deleted_segment(&self) -> bool {
        self.inner.keep_fully_deleted_segment()
    }

    fn num_deletes_to_merge(&self, info: &MergeSegment, del_count: i32) -> i32 {
        self.inner.num_deletes_to_merge(info, del_count)
    }

    fn compound_file_settings(&self) -> CompoundFileSettings {
        self.inner.compound_file_settings()
    }

    fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
        self.inner.compound_file_settings_mut()
    }

    fn is_merged(
        &self,
        infos: &[MergeSegment],
        info: &MergeSegment,
        ctx: &dyn MergeContext,
    ) -> Result<bool> {
        self.inner.is_merged(infos, info, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::merge_policy::api::{NoMergePolicy, TieredMergePolicy};

    struct Ctx(HashSet<String>);

    impl MergeContext for Ctx {
        fn num_deletes_to_merge(&self, info: &MergeSegment) -> Result<i32> {
            Ok(info.del_count)
        }
        fn num_deleted_docs(&self, info: &MergeSegment) -> i32 {
            info.del_count
        }
        fn merging_segments(&self) -> &HashSet<String> {
            &self.0
        }
    }

    fn seg(name: &str, version: Option<LuceneVersion>) -> MergeSegment {
        let mut s = MergeSegment::new(name, 100, 0, 10_000);
        s.version = version;
        s
    }

    const V9: LuceneVersion = LuceneVersion {
        major: 9,
        minor: 0,
        bugfix: 0,
    };

    fn all(infos: &[MergeSegment]) -> HashMap<String, bool> {
        infos.iter().map(|s| (s.name.clone(), true)).collect()
    }

    fn names(spec: &MergeSpecification) -> Vec<Vec<String>> {
        spec.merges
            .iter()
            .map(|m| m.segments.iter().map(|s| s.name.clone()).collect())
            .collect()
    }

    #[test]
    fn only_old_segments_are_force_merged() {
        let infos = vec![
            seg("_0", Some(V9)),
            seg("_1", Some(LATEST)),
            seg("_2", None),
            seg("_3", Some(V9)),
        ];
        let ctx = Ctx(HashSet::new());
        // `NoMergePolicy` wants nothing, so every old segment ends up in the
        // one extra merge, in `segmentInfos` order.
        let p = UpgradeIndexMergePolicy::new(Box::new(NoMergePolicy::default()));
        assert!(p.should_upgrade_segment(&infos[0]));
        assert!(!p.should_upgrade_segment(&infos[1]));
        assert!(p.should_upgrade_segment(&infos[2]));
        let spec = p
            .find_forced_merges(&infos, 1, &all(&infos), &ctx)
            .unwrap()
            .unwrap();
        assert_eq!(names(&spec), vec![vec!["_0", "_2", "_3"]]);
        // A segment the caller did not ask to merge stays out.
        let mut some = all(&infos);
        some.remove("_3");
        let spec = p
            .find_forced_merges(&infos, 1, &some, &ctx)
            .unwrap()
            .unwrap();
        assert_eq!(names(&spec), vec![vec!["_0", "_2"]]);
        // Nothing old: nothing to do.
        let current = vec![seg("_1", Some(LATEST))];
        assert!(p
            .find_forced_merges(&current, 1, &all(&current), &ctx)
            .unwrap()
            .is_none());
    }

    #[test]
    fn what_the_wrapped_policy_merges_is_not_merged_again() {
        let infos = vec![
            seg("_0", Some(V9)),
            seg("_1", Some(V9)),
            seg("_2", Some(LATEST)),
        ];
        let ctx = Ctx(HashSet::new());
        let p = UpgradeIndexMergePolicy::new(Box::new(TieredMergePolicy::default()));
        let spec = p
            .find_forced_merges(&infos, 1, &all(&infos), &ctx)
            .unwrap()
            .unwrap();
        // Tiered merges the two old segments (the only ones it was offered);
        // no leftover merge follows, and the current segment is untouched.
        let got = names(&spec);
        assert_eq!(got.len(), 1, "{got:?}");
        let mut merged = got[0].clone();
        merged.sort();
        assert_eq!(merged, vec!["_0", "_1"]);
    }

    #[test]
    fn everything_else_is_the_wrapped_policys() {
        let infos = vec![seg("_0", Some(V9))];
        let ctx = Ctx(HashSet::new());
        let mut p = UpgradeIndexMergePolicy::new(Box::new(TieredMergePolicy::default()));
        let tiered = TieredMergePolicy::default();
        assert_eq!(
            p.find_merges(MergeTrigger::Explicit, &infos, &ctx)
                .unwrap()
                .is_some(),
            tiered
                .find_merges(MergeTrigger::Explicit, &infos, &ctx)
                .unwrap()
                .is_some()
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
            p.find_full_flush_merges(MergeTrigger::FullFlush, &infos, &ctx)
                .unwrap()
                .is_some(),
            tiered
                .find_full_flush_merges(MergeTrigger::FullFlush, &infos, &ctx)
                .unwrap()
                .is_some()
        );
        assert_eq!(
            p.find_merges_for_readers(&[3, 4]).unwrap().merges[0].total_max_doc,
            7
        );
        assert_eq!(
            p.use_compound_file(&infos, &infos[0], &ctx).unwrap(),
            tiered.use_compound_file(&infos, &infos[0], &ctx).unwrap()
        );
        assert_eq!(p.size(&infos[0], &ctx).unwrap(), 10_000);
        assert_eq!(
            p.max_full_flush_merge_size(),
            tiered.max_full_flush_merge_size()
        );
        assert!(!p.keep_fully_deleted_segment());
        assert_eq!(p.num_deletes_to_merge(&infos[0], 3), 3);
        assert_eq!(
            p.is_merged(&infos, &infos[0], &ctx).unwrap(),
            tiered.is_merged(&infos, &infos[0], &ctx).unwrap()
        );
        p.compound_file_settings_mut().no_cfs_ratio = 0.0;
        assert_eq!(p.compound_file_settings().no_cfs_ratio, 0.0);
        assert_eq!(p.unwrap().compound_file_settings().no_cfs_ratio, 0.0);
    }
}
