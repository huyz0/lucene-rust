//! Port of `org.apache.lucene.index.UpgradeIndexMergePolicy`: a
//! `FilterMergePolicy` whose forced merges only rewrite segments an older
//! Lucene wrote, so `forceMerge` "upgrades" an index instead of merging
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
    bugfix: 2,
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
mod tests;
