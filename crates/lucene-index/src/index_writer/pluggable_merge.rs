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
    pub(super) fn merge_inputs(&self) -> Result<(Vec<MergeSegment>, WriterContext)> {
        let mut infos = Vec::with_capacity(self.segment_infos.segments.len());
        let mut deletes_to_merge = HashMap::new();
        for sci in &self.segment_infos.segments {
            let si_bytes = self.dir.open(&format!("{}.si", sci.segment_name))?.to_vec();
            let si = segment_info::parse_for_codec(&si_bytes, &sci.segment_id, &sci.codec_name)?;
            let size = merge_policy::segment_byte_size(self.dir, &si);
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
        if self.pluggable_merge_policy.is_none() && !self.cfg.use_compound_file {
            return Ok(false);
        }
        let name = &merged.info.segment_name;
        let si_bytes = self.dir.open(&format!("{name}.si"))?.to_vec();
        let si = segment_info::parse(&si_bytes, &merged.info.segment_id)?;
        let size = merge_policy::segment_byte_size(self.dir, &si);
        let segment = MergeSegment::new(
            name.clone(),
            si.doc_count,
            0,
            i64::try_from(size).unwrap_or(i64::MAX),
        );
        let (infos, ctx) = self.merge_inputs()?;
        let decision = match &self.pluggable_merge_policy {
            Some(policy) => policy.use_compound_file(&infos, &segment, &ctx),
            None => {
                merge_policy::TieredMergePolicy::default().use_compound_file(&infos, &segment, &ctx)
            }
        };
        decision.map_err(policy_error)
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
    /// The policy is the installed pluggable one, or -- with only a
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
        let wait = self.max_full_flush_merge_wait_millis;
        if wait <= 0 {
            return Ok(0);
        }
        let policy: Arc<dyn MergePolicy> = match (&self.pluggable_merge_policy, &self.merge_policy)
        {
            (Some(policy), _) => Arc::clone(policy),
            (None, Some(config)) => Arc::new(merge_policy::TieredMergePolicy::new(config.clone())),
            (None, None) => return Ok(0),
        };
        let start = std::time::Instant::now();
        let deadline = std::time::Duration::from_millis(u64::try_from(wait).unwrap_or(0));
        // Java's `segmentInfos` already holds every flushed segment; this
        // writer keeps them apart until a commit folds them in, so fold them
        // in now -- in memory, as the merges below are published.
        if !self.flushed_segments.is_empty() {
            let flushed = std::mem::take(&mut self.flushed_segments);
            self.segment_infos.segments.extend(flushed);
        }
        let (infos, mut ctx) = self.merge_inputs()?;
        ctx.merging = merging.clone();
        let Some(spec) = policy
            .find_full_flush_merges(trigger, &infos, &ctx)
            .map_err(policy_error)?
        else {
            return Ok(0);
        };
        let by_caller = self.merges_by_caller;
        self.merges_by_caller = true;
        let mut ran = 0usize;
        let mut result = Ok(());
        for group in spec.groups() {
            if group.iter().any(|name| merging.contains(name)) {
                continue;
            }
            if start.elapsed() >= deadline {
                break;
            }
            result = self.execute_merge(&group);
            if result.is_err() {
                break;
            }
            ran = ran.saturating_add(1);
        }
        self.merges_by_caller = by_caller;
        result.map(|()| ran)
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

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::compound_format;
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
    use lucene_store::directory::{Directory, FsDirectory};
    use lucene_util::test_support::TempDir;

    use crate::segment_info::LuceneVersion;
    use crate::segment_infos;

    fn field() -> FieldInfo {
        FieldInfo {
            name: "id".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        }
    }

    fn doc(id: &str) -> Document {
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            }],
        }
    }

    fn version() -> LuceneVersion {
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        }
    }

    /// Every stored document of every committed segment, read through the
    /// compound archive when the segment is one.
    fn stored_ids(dir: &FsDirectory) -> (Vec<String>, Vec<bool>) {
        let sis = segment_infos::read_latest(dir).unwrap();
        let mut ids = Vec::new();
        let mut compound = Vec::new();
        for sci in &sis.segments {
            let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
            let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
            compound.push(si.is_compound_file);
            let read = |ext: &str| -> Vec<u8> {
                if si.is_compound_file {
                    let cfs = dir.open(&format!("{}.cfs", sci.segment_name)).unwrap();
                    let cfe = dir.open(&format!("{}.cfe", sci.segment_name)).unwrap();
                    let entries = compound_format::parse_entries(&cfe, &sci.segment_id).unwrap();
                    compound_format::check_data_header_footer(&cfs, &sci.segment_id, &entries)
                        .unwrap();
                    compound_format::open_input(&cfs, &entries, ext)
                        .unwrap()
                        .as_slice()
                        .to_vec()
                } else {
                    dir.open(&format!("{}{ext}", sci.segment_name))
                        .unwrap()
                        .to_vec()
                }
            };
            let (fdt, fdx, fdm) = (read(".fdt"), read(".fdx"), read(".fdm"));
            let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
            for d in 0..reader.max_doc() {
                if let FieldValue::String(s) = &reader.document(d).unwrap().fields[0].value {
                    ids.push(s.clone());
                }
            }
        }
        (ids, compound)
    }

    /// `useCompoundFile`: a flushed segment is packed, `.si` outside the
    /// archive and listing exactly the archive and itself; the loose files
    /// are gone. A merge then follows the merge policy: `TieredMergePolicy`'s
    /// default `noCFSRatio` (0.1) keeps a merge of the whole index loose, a
    /// ratio of 1.0 packs it.
    #[test]
    fn flushes_and_merges_write_compound_segments_as_java_decides() {
        let tmp = TempDir::new("pluggable-merge-compound");
        let dir = FsDirectory::open(&tmp);
        let mut w = IndexWriter::open(&dir, vec![field()], "Lucene104", version()).unwrap();
        assert!(!w.use_compound_file());
        w.set_use_compound_file(true);
        assert!(w.use_compound_file());
        for (i, id) in ["a", "b", "c", "d"].iter().enumerate() {
            w.add_document(doc(id)).unwrap();
            if i % 2 == 1 {
                w.commit().unwrap();
            }
        }
        let (ids, compound) = stored_ids(&dir);
        assert_eq!(ids, ["a", "b", "c", "d"]);
        assert_eq!(compound, [true, true]);
        let names = dir.list_all().unwrap();
        assert!(names.iter().any(|n| n == "_0.cfs") && names.iter().any(|n| n == "_0.cfe"));
        assert!(!names.iter().any(|n| n == "_0.fdt" || n == "_0.fnm"));
        let si = segment_info::parse(
            &dir.open("_0.si").unwrap(),
            &segment_infos::read_latest(&dir).unwrap().segments[0].segment_id,
        )
        .unwrap();
        assert_eq!(si.files, ["_0.cfs", "_0.cfe", "_0.si"]);

        // The whole index merged is more than 10% of the index: loose.
        w.force_merge(1).unwrap();
        let (ids, compound) = stored_ids(&dir);
        assert_eq!(ids, ["a", "b", "c", "d"]);
        assert_eq!(compound, [false]);

        // A policy with `noCFSRatio` 1.0 packs the merged segment.
        let mut policy = merge_policy::TieredMergePolicy::default();
        policy
            .compound_file_settings_mut()
            .set_no_cfs_ratio(1.0)
            .unwrap();
        w.set_pluggable_merge_policy(Some(Arc::new(policy)));
        w.set_use_compound_file(false);
        w.add_document(doc("e")).unwrap();
        w.commit().unwrap();
        let (_, compound) = stored_ids(&dir);
        assert_eq!(compound.last(), Some(&false), "the flag is off for flushes");
        w.force_merge(1).unwrap();
        let (ids, compound) = stored_ids(&dir);
        assert_eq!(ids, ["a", "b", "c", "d", "e"]);
        assert_eq!(compound, [true]);
    }

    /// With the flag off and no pluggable policy, a merge stays loose
    /// whatever `TieredMergePolicy` would have said.
    #[test]
    fn a_writer_without_compound_files_merges_loose() {
        let tmp = TempDir::new("pluggable-merge-loose");
        let dir = FsDirectory::open(&tmp);
        let mut w = IndexWriter::open(&dir, vec![field()], "Lucene104", version()).unwrap();
        for id in ["a", "b"] {
            w.add_document(doc(id)).unwrap();
            w.commit().unwrap();
        }
        w.force_merge(1).unwrap();
        let (ids, compound) = stored_ids(&dir);
        assert_eq!(ids, ["a", "b"]);
        assert_eq!(compound, [false]);
    }
}
