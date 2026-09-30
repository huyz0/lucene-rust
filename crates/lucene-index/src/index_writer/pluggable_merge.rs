//! `IndexWriterConfig.setMergePolicy` with any [`MergePolicy`]: the writer's
//! own merging -- after a commit, and [`IndexWriter::force_merge`] -- asks a
//! pluggable policy ([`crate::merge_policy::log`], [`crate::merge_policy::temporal`],
//! [`crate::merge_policy::filter`], [`crate::merge_policy::TieredMergePolicy`],
//! or a caller's own) instead of the built-in [`super::MergePolicyConfig`].
//!
//! The writer is the policy's `MergeContext`: `numDeletesToMerge` is
//! [`IndexWriter::num_deletes_to_merge`] (hard deletes, plus the soft deletes a
//! retention policy no longer keeps), and nothing is ever merging while the
//! policy is asked, because this writer runs its merges one at a time on the
//! calling thread.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::{Error, IndexWriter, Result};
use crate::merge_policy::{self, MergeContext, MergePolicy, MergeSegment, MergeTrigger};
use crate::segment_info;

/// The writer's answers to a policy's `MergeContext` questions, computed once
/// per consultation.
struct WriterContext {
    deletes_to_merge: HashMap<String, i32>,
    merging: HashSet<String>,
}

impl MergeContext for WriterContext {
    fn num_deletes_to_merge(&self, info: &MergeSegment) -> merge_policy::api::Result<i32> {
        Ok(self
            .deletes_to_merge
            .get(&info.name)
            .copied()
            .unwrap_or(info.del_count))
    }

    fn num_deleted_docs(&self, info: &MergeSegment) -> i32 {
        info.del_count
    }

    fn merging_segments(&self) -> &HashSet<String> {
        &self.merging
    }
}

fn policy_error(e: merge_policy::api::Error) -> Error {
    Error::Explicit(format!("merge policy: {e}"))
}

impl IndexWriter<'_> {
    /// `IndexWriterConfig.setMergePolicy(policy)`: from here on the writer's
    /// merges -- after every commit, and in [`IndexWriter::force_merge`] --
    /// are the ones `policy` specifies. `None` returns to the
    /// [`super::MergePolicyConfig`] set by [`IndexWriter::set_merge_policy`].
    pub fn set_pluggable_merge_policy(&mut self, policy: Option<Arc<dyn MergePolicy>>) {
        self.pluggable_merge_policy = policy;
    }

    /// The policy [`IndexWriter::set_pluggable_merge_policy`] installed.
    pub fn pluggable_merge_policy(&self) -> Option<&Arc<dyn MergePolicy>> {
        self.pluggable_merge_policy.as_ref()
    }

    /// The committed segments as a policy sees them, and the context that
    /// answers for them.
    fn merge_inputs(&self) -> Result<(Vec<MergeSegment>, WriterContext)> {
        let mut infos = Vec::with_capacity(self.segment_infos.segments.len());
        let mut deletes_to_merge = HashMap::new();
        for sci in &self.segment_infos.segments {
            let si_bytes = self.dir.open(&format!("{}.si", sci.segment_name))?.to_vec();
            let si = segment_info::parse(&si_bytes, &sci.segment_id)?;
            let size = merge_policy::segment_byte_size(self.dir, &si);
            let stat = merge_policy::SegmentStat {
                name: sci.segment_name.clone(),
                doc_count: si.doc_count,
                del_count: sci.del_count,
                size_bytes: size,
            };
            deletes_to_merge.insert(stat.name.clone(), self.num_deletes_to_merge(&stat)?);
            infos.push(MergeSegment::from(&stat).with_compound_file(si.is_compound_file));
        }
        Ok((
            infos,
            WriterContext {
                deletes_to_merge,
                merging: HashSet::new(),
            },
        ))
    }

    /// The merge policy's `useCompoundFile`, closed over the current segments
    /// and context, for a merge about to start: the pluggable policy's when
    /// one is set; otherwise `TieredMergePolicy`'s (the built-in
    /// [`super::MergePolicyConfig`]) when compound files are on; otherwise
    /// none -- the layout this writer's merges had before compound files.
    pub(super) fn merge_compound_rule(&self) -> Result<Option<super::CompoundRule>> {
        let policy: Arc<dyn MergePolicy> = match &self.pluggable_merge_policy {
            Some(policy) => Arc::clone(policy),
            None if self.use_compound_file() => Arc::new(merge_policy::TieredMergePolicy::new(
                self.merge_policy.clone().unwrap_or_default(),
            )),
            None => return Ok(None),
        };
        let (infos, ctx) = self.merge_inputs()?;
        Ok(Some(Arc::new(move |merged: &MergeSegment| {
            policy
                .use_compound_file(&infos, merged, &ctx)
                .map_err(policy_error)
        })))
    }

    /// `IndexWriter.maybeMerge(FULL_FLUSH)` under a pluggable policy: runs
    /// every merge it specifies, then asks again, until it specifies none.
    /// Stops as well when a round merged nothing new (a policy proposing a
    /// single-segment merge of a delete-free segment would otherwise repeat
    /// it forever -- Java's writer ends that loop by marking it merging).
    pub(super) fn auto_merge_pluggable(&mut self, policy: &Arc<dyn MergePolicy>) -> Result<()> {
        let mut last: Option<Vec<Vec<String>>> = None;
        loop {
            let (infos, ctx) = self.merge_inputs()?;
            let Some(spec) = policy
                .find_merges(MergeTrigger::FullFlush, &infos, &ctx)
                .map_err(policy_error)?
            else {
                return Ok(());
            };
            let groups = spec.groups();
            if groups.is_empty() || last.as_ref() == Some(&groups) {
                return Ok(());
            }
            for group in &groups {
                self.execute_merge(group)?;
            }
            last = Some(groups);
        }
    }

    /// `IndexWriter.forceMergeDeletes()` under a pluggable policy: one round
    /// of `findForcedDeletesMerges`, every merge it specifies run.
    pub(super) fn force_merge_deletes_pluggable(
        &mut self,
        policy: &Arc<dyn MergePolicy>,
    ) -> Result<()> {
        let (infos, ctx) = self.merge_inputs()?;
        let Some(spec) = policy
            .find_forced_deletes_merges(&infos, &ctx)
            .map_err(policy_error)?
        else {
            return Ok(());
        };
        for group in spec.groups() {
            self.execute_merge(&group)?;
        }
        Ok(())
    }

    /// `IndexWriter.forceMerge(maxNumSegments)` under a pluggable policy:
    /// `findForcedMerges` with every current segment "original", merged
    /// round after round until the policy specifies nothing more.
    pub(super) fn force_merge_pluggable(
        &mut self,
        policy: &Arc<dyn MergePolicy>,
        max_num_segments: usize,
    ) -> Result<()> {
        let mut to_merge: HashMap<String, bool> = self
            .segment_infos
            .segments
            .iter()
            .map(|s| (s.segment_name.clone(), true))
            .collect();
        let max_segment_count = i32::try_from(max_num_segments).unwrap_or(i32::MAX);
        let mut last: Option<Vec<Vec<String>>> = None;
        loop {
            let (infos, ctx) = self.merge_inputs()?;
            let Some(spec) = policy
                .find_forced_merges(&infos, max_segment_count, &to_merge, &ctx)
                .map_err(policy_error)?
            else {
                return Ok(());
            };
            let groups = spec.groups();
            if groups.is_empty() || last.as_ref() == Some(&groups) {
                return Ok(());
            }
            let before: HashSet<String> = infos.into_iter().map(|i| i.name).collect();
            for group in &groups {
                self.execute_merge(group)?;
            }
            // `IndexWriter.updatePendingMerges`: a merged segment replaces its
            // sources in `segmentsToMerge`, marked "not original".
            for sci in &self.segment_infos.segments {
                if !before.contains(&sci.segment_name) {
                    to_merge.insert(sci.segment_name.clone(), false);
                }
            }
            last = Some(groups);
        }
    }
}
