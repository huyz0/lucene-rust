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

/// A merge policy's `useCompoundFile` with the segments it measures a merged
/// segment against, taken when the merge started, so the merge thread can
/// decide before it warms the merged segment, as `mergeMiddle` does.
pub(crate) struct CompoundFileDecider {
    policy: Arc<dyn MergePolicy>,
    infos: Vec<MergeSegment>,
    ctx: WriterContext,
}

impl CompoundFileDecider {
    pub(crate) fn decide(
        &self,
        dir: &dyn lucene_store::directory::Directory,
        merged: &crate::merge::MergedSegment,
    ) -> Result<bool> {
        let name = &merged.info.segment_name;
        let si_bytes = dir.open(&format!("{name}.si"))?.to_vec();
        let si = segment_info::parse(&si_bytes, &merged.info.segment_id)?;
        let size = merge_policy::segment_byte_size(dir, &si);
        let segment = MergeSegment::new(
            name.clone(),
            si.doc_count,
            0,
            i64::try_from(size).unwrap_or(i64::MAX),
        );
        self.policy
            .use_compound_file(&self.infos, &segment, &self.ctx)
            .map_err(policy_error)
    }
}

/// The writer's answers to a policy's `MergeContext` questions, computed once
/// per consultation.
pub(super) struct WriterContext {
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

/// The merges a full flush runs, each with its estimated merge bytes, and how
/// long the flush waits for them.
pub(crate) type FullFlushMerges = (
    Vec<(merge_policy::api::RunnableMerge, u64)>,
    std::time::Duration,
);

impl IndexWriter<'_> {
    /// `IndexWriterConfig.setMergePolicy(policy)`: from here on the writer's
    /// merges -- after every commit, and in [`IndexWriter::force_merge`] --
    /// are the ones `policy` specifies. `None` returns to the
    /// [`super::MergePolicyConfig`] set by [`IndexWriter::set_merge_policy`].
    pub fn set_pluggable_merge_policy(&mut self, policy: Option<Arc<dyn MergePolicy>>) {
        self.pluggable_merge_policy = policy;
    }

    /// The policy whose `findFullFlushMerges` picks the merges a commit or
    /// near-real-time reader runs ([`IndexWriter::set_max_full_flush_merge_wait_millis`]),
    /// in place of the writer's own policy. Every other merge decision stays
    /// the writer's, which is how a filter that only overrides
    /// `findFullFlushMerges` around the writer's policy behaves -- such as
    /// [`merge_policy::MergeOnFlushMergePolicy`]. `None` asks the writer's own.
    pub fn set_full_flush_merge_policy(&mut self, policy: Option<Arc<dyn MergePolicy>>) {
        self.full_flush_merge_policy = policy;
    }

    /// The policy [`IndexWriter::set_pluggable_merge_policy`] installed.
    pub fn pluggable_merge_policy(&self) -> Option<&Arc<dyn MergePolicy>> {
        self.pluggable_merge_policy.as_ref()
    }

    /// `IndexWriter.updatePendingMerges(trigger)`'s selection for a writer
    /// whose merges run on merge threads: every merge the installed policy
    /// specifies that takes no segment in `merging`, with its hooks and its
    /// `OneMerge.estimatedMergeBytes` (the sources' sizes pro-rated by their
    /// live documents). The pluggable policy decides when one is installed,
    /// else `TieredMergePolicy` over the [`super::MergePolicyConfig`]; with
    /// neither there is nothing to merge.
    pub(crate) fn find_scheduled_merges(
        &self,
        trigger: MergeTrigger,
        merging: &HashSet<String>,
    ) -> Result<Vec<(merge_policy::api::RunnableMerge, u64)>> {
        let Some(policy) = self.pluggable_merge_policy.clone() else {
            let Some(config) = self.merge_policy.clone() else {
                return Ok(Vec::new());
            };
            return Ok(self
                .find_merges_sized(&config, merging)?
                .into_iter()
                .map(|(names, bytes)| ((names, None), bytes))
                .collect());
        };
        let (infos, mut ctx) = self.merge_inputs()?;
        ctx.merging = merging.clone();
        let Some(spec) = self
            .per_partition_spec(&infos, |part| policy.find_merges(trigger, part, &ctx))?
            .map_err(policy_error)?
        else {
            return Ok(Vec::new());
        };
        // `registerMerge`: a merge taking a segment already merging, or one an
        // earlier merge of this spec takes, is refused.
        let mut claimed: HashSet<String> = HashSet::new();
        Ok(spec
            .runnable()
            .into_iter()
            .filter(|(group, _)| {
                let free = !group.is_empty()
                    && !group
                        .iter()
                        .any(|n| merging.contains(n) || claimed.contains(n));
                if free {
                    claimed.extend(group.iter().cloned());
                }
                free
            })
            .map(|merge| {
                let bytes = infos
                    .iter()
                    .filter(|i| merge.0.contains(&i.name))
                    .map(|i| {
                        super::live_pro_rated_bytes(
                            u64::try_from(i.size_in_bytes).unwrap_or(0),
                            i.max_doc,
                            i.del_count,
                        )
                    })
                    .fold(0u64, u64::saturating_add);
                (merge, bytes)
            })
            .collect())
    }

    /// The committed segments as a policy sees them, and the context that
    /// answers for them.
    pub(super) fn merge_inputs(&self) -> Result<(Vec<MergeSegment>, WriterContext)> {
        let mut infos = Vec::with_capacity(self.segment_infos.segments.len());
        let mut deletes_to_merge = HashMap::new();
        for sci in &self.segment_infos.segments {
            let si = self.cfg.reader_pool.segment_info(self.dir, sci)?;
            // `SegmentCommitInfo.sizeInBytes`: the live-docs and generational
            // files count too.
            let size = merge_policy::commit_byte_size(self.dir, sci, &si);
            let stat = merge_policy::SegmentStat {
                name: sci.segment_name.clone(),
                doc_count: si.doc_count,
                del_count: sci.del_count,
                size_bytes: size,
            };
            deletes_to_merge.insert(stat.name.clone(), self.num_deletes_to_merge(&stat)?);
            infos.push(
                MergeSegment::from(&stat)
                    .with_compound_file(si.is_compound_file)
                    .with_version(si.version),
            );
        }
        Ok((
            infos,
            WriterContext {
                deletes_to_merge,
                merging: HashSet::new(),
            },
        ))
    }

    /// `MergePolicy.useCompoundFile(segmentInfos, mergedInfo, writer)` for a
    /// finished merge: the installed pluggable policy's answer, or -- with
    /// none installed -- `TieredMergePolicy`'s defaults (`noCFSRatio` 0.1,
    /// no size cap), Java's default policy, when the writer uses compound
    /// files at all (see [`IndexWriter::set_use_compound_file`] for this
    /// port's default). The merged segment is measured by its files on
    /// disk, with no deletions yet.
    pub(super) fn merged_segment_uses_compound_file(
        &self,
        merged: &crate::merge::MergedSegment,
    ) -> Result<bool> {
        match self.compound_file_decider()? {
            Some(decider) => decider.decide(self.dir, merged),
            None => Ok(false),
        }
    }

    /// What [`Self::merged_segment_uses_compound_file`] decides with, taken
    /// now: the policy and the segments it measures against. `None` when no
    /// merged segment can be compound.
    pub(super) fn compound_file_decider(&self) -> Result<Option<CompoundFileDecider>> {
        if self.pluggable_merge_policy.is_none()
            && self.cfg.merge_compound_file.is_none()
            && !self.cfg.use_compound_file
        {
            return Ok(None);
        }
        let policy = match &self.pluggable_merge_policy {
            Some(policy) => Arc::clone(policy),
            None => {
                let mut tiered = merge_policy::TieredMergePolicy::default();
                if let Some(settings) = self.cfg.merge_compound_file {
                    *tiered.compound_file_settings_mut() = settings;
                }
                Arc::new(tiered) as Arc<dyn MergePolicy>
            }
        };
        let (infos, ctx) = self.merge_inputs()?;
        Ok(Some(CompoundFileDecider { policy, infos, ctx }))
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
            let Some(spec) = self
                .per_partition_spec(&infos, |part| {
                    policy.find_merges(MergeTrigger::FullFlush, part, &ctx)
                })?
                .map_err(policy_error)?
            else {
                return Ok(());
            };
            let groups = spec.groups();
            if groups.is_empty() || last.as_ref() == Some(&groups) {
                return Ok(());
            }
            for (group, hooks) in spec.runnable() {
                self.execute_merge_hooked(&group, hooks)?;
            }
            last = Some(groups);
        }
    }

    /// `IndexWriterConfig.setMaxFullFlushMergeWaitMillis`: how long a commit
    /// ([`IndexWriter::commit`], [`IndexWriter::prepare_commit`]) or a
    /// near-real-time reader (`getReader`) waits for the merges
    /// `MergePolicy.findFullFlushMerges` asks for on the segments it is about
    /// to publish. `0` (or less) turns merge-on-commit/refresh off.
    ///
    /// The default is Java's, [`super::DEFAULT_MAX_FULL_FLUSH_MERGE_WAIT_MILLIS`]
    /// (500 ms): with a merge policy installed, a commit's point-in-time
    /// segments are the merged ones whenever `TieredMergePolicy` (below its
    /// floor size) or the installed policy asks for a merge.
    pub fn set_max_full_flush_merge_wait_millis(&mut self, millis: i64) {
        self.max_full_flush_merge_wait_millis = millis;
    }

    /// `LiveIndexWriterConfig.getMaxFullFlushMergeWaitMillis`.
    pub fn max_full_flush_merge_wait_millis(&self) -> i64 {
        self.max_full_flush_merge_wait_millis
    }

    /// `IndexWriter.preparePointInTimeMerge` for a `COMMIT` or `GET_READER`
    /// trigger, run to completion: the policy's `findFullFlushMerges` over
    /// every segment the commit or reader is about to see -- those this call's
    /// flush just wrote included -- then each merge it asks for, published in
    /// memory (`commitMerge`), so the `segments_N` or reader built next holds
    /// the merged segment in place of its sources. Returns how many merges
    /// ran.
    ///
    /// The policy is [`IndexWriter::set_full_flush_merge_policy`]'s, else the
    /// installed pluggable one, or -- with only a
    /// [`super::MergePolicyConfig`] -- `TieredMergePolicy` over that
    /// configuration, whose `maxFullFlushMergeSize` is its floor segment
    /// size. With neither, or a wait of `0`, nothing happens.
    ///
    /// A merge including a segment in `merging` (merges another thread is
    /// running, [`crate::concurrent_writer::ConcurrentIndexWriter`]) is
    /// skipped, as `registerMerge` rejects it.
    ///
    /// **The wait**: Java starts every merge on the merge scheduler and waits
    /// up to `maxFullFlushMergeWaitMillis` for them; one finishing later still
    /// lands in the writer, only not in this commit. This writer merges on
    /// the calling thread, so a merge it starts always finishes inside the
    /// commit; once the wait has elapsed it starts no further one, and those
    /// are left to the merges after the commit (`maybeMerge`).
    pub(crate) fn merge_on_full_flush(
        &mut self,
        trigger: MergeTrigger,
        merging: &HashSet<String>,
    ) -> Result<usize> {
        let Some((merges, deadline)) = self.full_flush_merges(trigger, merging)? else {
            return Ok(0);
        };
        let start = std::time::Instant::now();
        let by_caller = self.merges_by_caller;
        self.merges_by_caller = true;
        let mut ran = 0usize;
        let mut result = Ok(());
        for ((group, hooks), _) in merges {
            if start.elapsed() >= deadline {
                break;
            }
            result = self.execute_merge_hooked(&group, hooks);
            if result.is_err() {
                break;
            }
            ran = ran.saturating_add(1);
        }
        self.merges_by_caller = by_caller;
        result.map(|()| ran)
    }

    /// Moves the segments flushed since the last commit into `segment_infos`,
    /// in memory: Java's `segmentInfos` holds every flushed segment, so its
    /// merge policy sees them before any commit names them.
    pub(crate) fn fold_flushed_segments(&mut self) {
        if !self.flushed_segments.is_empty() {
            let flushed = std::mem::take(&mut self.flushed_segments);
            self.segment_infos.segments.extend(flushed);
        }
    }

    /// `preparePointInTimeMerge`'s selection: the merges
    /// `findFullFlushMerges` specifies over every flushed segment, each with
    /// its estimated merge bytes, and how long merge-on-commit/refresh waits
    /// for them (`maxFullFlushMergeWaitMillis`). `None` with a wait of `0` or
    /// no policy. A merge including a segment in `merging` is left out, as
    /// `registerMerge` rejects it.
    pub(crate) fn full_flush_merges(
        &mut self,
        trigger: MergeTrigger,
        merging: &HashSet<String>,
    ) -> Result<Option<FullFlushMerges>> {
        self.write_pending_try_changes()?;
        let wait = self.max_full_flush_merge_wait_millis;
        if wait <= 0 {
            return Ok(None);
        }
        let policy: Arc<dyn MergePolicy> = match (
            &self.full_flush_merge_policy,
            &self.pluggable_merge_policy,
            &self.merge_policy,
        ) {
            (Some(policy), _, _) | (None, Some(policy), _) => Arc::clone(policy),
            (None, None, Some(config)) => {
                Arc::new(merge_policy::TieredMergePolicy::new(config.clone()))
            }
            (None, None, None) => return Ok(None),
        };
        let deadline = std::time::Duration::from_millis(u64::try_from(wait).unwrap_or(0));
        self.fold_flushed_segments();
        let (infos, mut ctx) = self.merge_inputs()?;
        ctx.merging = merging.clone();
        let Some(spec) = self
            .per_partition_spec(&infos, |part| {
                policy.find_full_flush_merges(trigger, part, &ctx)
            })?
            .map_err(policy_error)?
        else {
            return Ok(Some((Vec::new(), deadline)));
        };
        let merges = spec
            .runnable()
            .into_iter()
            .filter(|(group, _)| !group.iter().any(|name| merging.contains(name)))
            .map(|merge| {
                let bytes = infos
                    .iter()
                    .filter(|i| merge.0.contains(&i.name))
                    .map(|i| u64::try_from(i.size_in_bytes).unwrap_or(0))
                    .fold(0u64, u64::saturating_add);
                (merge, bytes)
            })
            .collect();
        Ok(Some((merges, deadline)))
    }

    /// `IndexWriter.forceMergeDeletes()` under a pluggable policy: one round
    /// of `findForcedDeletesMerges`, every merge it specifies run.
    pub(super) fn force_merge_deletes_pluggable(
        &mut self,
        policy: &Arc<dyn MergePolicy>,
    ) -> Result<()> {
        let (infos, ctx) = self.merge_inputs()?;
        let Some(spec) = self
            .per_partition_spec(&infos, |part| policy.find_forced_deletes_merges(part, &ctx))?
            .map_err(policy_error)?
        else {
            return Ok(());
        };
        for (group, hooks) in spec.runnable() {
            self.execute_merge_hooked(&group, hooks)?;
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
            let Some(spec) = self
                .per_partition_spec(&infos, |part| {
                    policy.find_forced_merges(part, max_segment_count, &to_merge, &ctx)
                })?
                .map_err(policy_error)?
            else {
                return Ok(());
            };
            let groups = spec.groups();
            if groups.is_empty() || last.as_ref() == Some(&groups) {
                return Ok(());
            }
            let before: HashSet<String> = infos.into_iter().map(|i| i.name).collect();
            // `ShuffleForcedMergePolicy` wraps the merges a policy forces.
            let stats = if self.cfg.shuffle_forced_merges {
                self.segment_stats()?
            } else {
                Vec::new()
            };
            for (group, hooks) in spec.runnable() {
                match hooks {
                    None => self.execute_forced_merge(group, &stats)?,
                    Some(_) => self.execute_merge_hooked(&group, hooks)?,
                }
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

#[cfg(test)]
mod tests;
