//! `IndexWriter.addIndexes(Directory...)` and `addIndexes(CodecReader...)`:
//! bring the segments of other indexes into this one.
//!
//! - [`IndexWriter::add_indexes`] (`addIndexes(Directory...)`) copies every
//!   segment of each source's latest commit as it is (`copySegmentAsIs`):
//!   each file under a fresh segment name of this index, the codec files
//!   byte for byte (their headers carry the segment's id and suffix, never
//!   its name, and a compound file's entries are stored without the name),
//!   and a new `.si` naming the renamed files. Deletions, field-infos and
//!   doc-values generations come along.
//! - [`IndexWriter::add_indexes_merged`] (`addIndexes(CodecReader...)` over
//!   each source's segment readers, `DirectoryReader.open(dir).leaves()`)
//!   merges every live document of every source into **one** new segment
//!   through this writer's merge (`SegmentMerger`), so deleted documents are
//!   dropped and the segment is written by this writer's codec.
//!
//! Both first flush this writer (Java's `flush(false, true)`), refuse a
//! source whose segments are not sorted by a
//! sort congruent with this writer's index sort, and refuse a field whose
//! schema conflicts with this writer's (`FieldInfos.verifyFieldInfos`). The
//! added segments are published in this writer's view; the next commit makes
//! them durable, and a rollback drops them. Only [`IndexWriter::add_indexes`]
//! holds each source's `write.lock` while it copies (Java's
//! `acquireWriteLocks`), so it refuses a source a writer has open; Java's `addIndexes(CodecReader...)` reads open
//! readers and locks nothing, so neither does [`IndexWriter::add_indexes_merged`].
//!
//! # What differs from Java
//!
//! - `addIndexes(CodecReader...)` takes directories (their latest commits)
//!   rather than arbitrary `CodecReader`s -- readers live above this crate --
//!   and is carried out as a copy of the sources' segments followed by one
//!   merge of those copies. The merged segment is the same, its diagnostics
//!   saying `addIndexes(CodecReader...)` as Java's do.
//! - `addIndexes(CodecReader...)` reads each source's latest commit unlocked,
//!   as a `DirectoryReader` would, but without a reader's open files: a
//!   writer that commits to the source during the copy and deletes a file the
//!   copy still needs makes the call fail, where Java's reader keeps it.
//! - One directory named twice to `addIndexes(Directory...)` is refused by
//!   its second `write.lock` (`LockObtainFailed`); Java's `noDupDirs` refuses
//!   it first, with an `IllegalArgumentException`. Nothing is added either way.
//! - An unsorted source cannot be added to an index-sorted writer by
//!   either method: Java sorts such a reader during the merge
//!   (`MergeState.maybeSortReaders`), which this port's merge does not do.

use lucene_codecs::field_infos::FieldInfo;
use lucene_store::directory::Directory;

use super::{Error, IndexWriter, Result, SeqNo, WRITE_LOCK_NAME};
use crate::segment_info;
use crate::segment_infos::{self, SegmentCommitInfo};

/// `IndexFileNames.stripSegmentName`, re-prefixed: `_3.fdt` of segment `_3`
/// becomes `_9.fdt` of segment `_9`, `_3_1.liv` becomes `_9_1.liv`.
fn rename(file: &str, from: &str, to: &str) -> String {
    match file.strip_prefix(from) {
        Some(rest) if rest.starts_with('.') || rest.starts_with('_') => format!("{to}{rest}"),
        _ => file.to_string(),
    }
}

impl IndexWriter<'_> {
    /// `IndexWriter.addIndexes(Directory...)`: see the module documentation.
    /// Returns the operation's sequence number.
    pub fn add_indexes(&mut self, sources: &[&dyn Directory]) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        let added = self.copy_segments_in(sources, None, true)?;
        self.segment_infos.segments.extend(added);
        self.prune_segment_versions();
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
        // `addIndexes` ends in `checkpoint()`, which counts as a change even
        // when the sources held no segment.
        self.changed();
        Ok(self.delete_queue.next_sequence_number())
    }

    /// `IndexWriter.addIndexes(CodecReader...)` over every segment of each
    /// source's latest commit: one new segment holding all of their live
    /// documents. See the module documentation.
    pub fn add_indexes_merged(&mut self, sources: &[&dyn Directory]) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        self.add_merged(sources, None, None)
    }

    /// [`Self::add_indexes_merged`] over sources whose SORTED_NUMERIC
    /// values, in `remap`'s fields, read through `remap` -- lucene-facet's
    /// `addIndexes(SlowCodecReaderWrapper.wrap(new
    /// OrdinalMappingLeafReader(leaf, ordinalMap, srcConfig)))`, which
    /// `TaxonomyMergeUtils.merge` runs.
    pub fn add_indexes_merged_remapping(
        &mut self,
        sources: &[&dyn Directory],
        remap: crate::merge::SortedNumericRemap,
    ) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        // The merge orders documents by the index sort's keys before the
        // remap rewrites any value, where Java sorts the wrapped (remapped)
        // reader: refused rather than sorted by the unmapped values.
        if let Some(field) = self
            .cfg
            .index_sort
            .iter()
            .flatten()
            .find_map(|f| remap.fields.contains(&f.field).then(|| f.field.clone()))
        {
            return Err(Error::AddIndexes(format!(
                "cannot remap {field:?}, a field of the index sort"
            )));
        }
        self.add_merged(sources, None, Some(std::sync::Arc::new(remap)))
    }

    /// `IndexWriter.addIndexes(CodecReader...)` over readers that hide some
    /// documents (Java's `FilterCodecReader` overriding `getLiveDocs` /
    /// `numDocs`, as `PKIndexSplitter` and `MultiPassIndexSplitter` do):
    /// [`Self::add_indexes_merged`] keeping only the documents `keep` accepts.
    /// `keep(source, segment, doc)` names the source (its place in `sources`),
    /// the segment (its place in that source's latest commit) and the
    /// segment-local document. `keep` is the reader's whole live set, as a
    /// `FilterCodecReader.getLiveDocs` is: a document the source had deleted
    /// is live again if `keep` accepts it (callers that must respect the
    /// source's deletes say so in `keep`). The rest are deleted in the copy
    /// before the merge, so the merged segment holds exactly the kept ones.
    pub fn add_indexes_filtered(
        &mut self,
        sources: &[&dyn Directory],
        keep: &dyn Fn(usize, usize, i32) -> bool,
    ) -> Result<SeqNo> {
        self.write_pending_try_changes()?;
        self.add_merged(sources, Some(keep), None)
    }

    fn add_merged(
        &mut self,
        sources: &[&dyn Directory],
        keep: Option<&dyn Fn(usize, usize, i32) -> bool>,
        remap: Option<std::sync::Arc<crate::merge::SortedNumericRemap>>,
    ) -> Result<SeqNo> {
        let added = self.copy_segments_in(sources, keep, false)?;
        if added.is_empty() {
            return Ok(self.delete_queue.next_sequence_number());
        }
        let names: Vec<String> = added.iter().map(|s| s.segment_name.clone()).collect();
        self.segment_infos.segments.extend(added);
        self.prune_segment_versions();
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
        // Published in this writer's view, as `addIndexes` publishes: not a
        // commit of its own.
        let by_caller = std::mem::replace(&mut self.merges_by_caller, true);
        let merged = self.execute_merge_planned(&names, None, remap, true);
        self.merges_by_caller = by_caller;
        match merged {
            Ok(true) => {
                self.changed();
                Ok(self.delete_queue.next_sequence_number())
            }
            // Abandoned: Java's `addIndexes(CodecReader...)` returns without
            // adding anything when its merge is aborted.
            Ok(false) => {
                self.withdraw_copies(&names)?;
                Ok(self.delete_queue.next_sequence_number())
            }
            Err(e) => {
                // Java's `addIndexes(CodecReader...)` publishes only the
                // merged segment, so a failed merge adds nothing.
                let _ = self.withdraw_copies(&names);
                Err(e)
            }
        }
    }

    /// Takes the copies `names` back out of this writer's view, their files
    /// left to the deleter.
    fn withdraw_copies(&mut self, names: &[String]) -> Result<()> {
        self.segment_infos
            .segments
            .retain(|s| !names.contains(&s.segment_name));
        self.prune_segment_versions();
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
        Ok(())
    }

    /// Flushes, checks every source and copies its segments in under fresh
    /// names, returning their new `SegmentCommitInfo`s (not yet published).
    /// `lock_sources`: hold each source's `write.lock` throughout, as
    /// `addIndexes(Directory...)` does.
    fn copy_segments_in(
        &mut self,
        sources: &[&dyn Directory],
        keep: Option<&dyn Fn(usize, usize, i32) -> bool>,
        lock_sources: bool,
    ) -> Result<Vec<SegmentCommitInfo>> {
        if self.prepared_commit.is_some() {
            return Err(Error::PreparedCommitPending("add_indexes"));
        }
        self.flush()?;
        // `acquireWriteLocks`: held until every copy is done.
        let _locks = sources
            .iter()
            .filter(|_| lock_sources)
            .map(|d| d.obtain_lock(WRITE_LOCK_NAME))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut plans = Vec::new();
        let mut total = self.live_doc_total()?;
        // `globalFieldNumberMap.addOrGet` registers each incoming field as
        // it goes, so a later source is checked against an earlier one.
        let mut incoming_fields = std::collections::HashMap::new();
        for (source, src) in sources.iter().enumerate() {
            let infos = segment_infos::read_latest(*src)?;
            for (segment, sci) in infos.segments.into_iter().enumerate() {
                let si = segment_info::parse_for_codec(
                    &src.open(&format!("{}.si", sci.segment_name))?,
                    &sci.segment_id,
                    &sci.codec_name,
                )?;
                self.check_addable(*src, &sci, &si, &mut incoming_fields)?;
                total = total.saturating_add(i64::from(si.doc_count));
                plans.push((*src, (source, segment), sci, si));
            }
        }
        let limit = i64::try_from(self.max_docs).unwrap_or(i64::MAX);
        if total > limit {
            return Err(Error::TooManyDocs(self.max_docs));
        }
        let mut added = Vec::with_capacity(plans.len());
        for (src, (source, segment), sci, si) in plans {
            let max_doc = usize::try_from(si.doc_count).unwrap_or(0);
            let mut copy = self.copy_segment_as_is(src, &sci, si)?;
            if let Some(keep) = keep {
                copy = self.hide_documents(copy, max_doc, |doc| keep(source, segment, doc))?;
            }
            added.push(copy);
        }
        Ok(added)
    }

    /// Makes the freshly copied segment's live documents exactly those `keep`
    /// accepts: a `FilterCodecReader` whose `getLiveDocs` is the filter, so a
    /// document the source had deleted is live again when `keep` accepts it.
    fn hide_documents(
        &self,
        mut sci: SegmentCommitInfo,
        max_doc: usize,
        keep: impl Fn(i32) -> bool,
    ) -> Result<SegmentCommitInfo> {
        let hidden: Vec<i32> = (0..max_doc)
            .filter_map(|d| i32::try_from(d).ok())
            .filter(|&d| !keep(d))
            .collect();
        if sci.del_gen >= 0 {
            // Start from "everything live"; the next deletes generation is
            // past the copied one, so its file is never overwritten.
            let next = sci.next_write_del_gen();
            self.dir.delete_file(&crate::deletes::liv_file_name(
                &sci.segment_name,
                sci.del_gen,
            ))?;
            sci.del_gen = -1;
            sci.del_count = 0;
            sci.set_next_write_del_gen(next);
        }
        if hidden.is_empty() {
            return Ok(sci);
        }
        Ok(crate::deletes::apply_deletes_with_codec(
            self.dir,
            &sci,
            None,
            max_doc,
            hidden,
            self.cfg.codec.as_deref(),
        )?)
    }

    /// Every document (deleted or not) already in this writer's view.
    fn live_doc_total(&self) -> Result<i64> {
        let mut total = 0i64;
        for sci in self.live_infos().segments {
            let si = segment_info::parse_for_codec(
                &self.dir.open(&format!("{}.si", sci.segment_name))?,
                &sci.segment_id,
                &sci.codec_name,
            )?;
            total = total.saturating_add(i64::from(si.doc_count));
        }
        Ok(total)
    }

    /// The index sort and the fields of one incoming segment against this
    /// writer's and the earlier incoming segments' (`incoming_fields`, which
    /// this call extends).
    fn check_addable(
        &mut self,
        src: &dyn Directory,
        sci: &SegmentCommitInfo,
        si: &segment_info::SegmentInfo,
        incoming_fields: &mut std::collections::HashMap<String, FieldInfo>,
    ) -> Result<()> {
        if let Some(sort) = &self.cfg.index_sort {
            let congruent = si.index_sort.as_ref().is_some_and(|existing| {
                sort.len() <= existing.len() && sort[..] == existing[..sort.len()]
            });
            if !congruent {
                return Err(Error::IncongruentIndexSort {
                    segment: sci.segment_name.clone(),
                    existing: segment_info::describe_index_sort(si.index_sort.as_deref()),
                    incoming: segment_info::describe_index_sort(Some(sort)),
                });
            }
        }
        // Through the segment's compound file when it has one.
        let mut incoming = super::segment_field_infos(src, sci)?.fields;
        // Java's FieldInfos iterator visits ascending field numbers.
        incoming.sort_by_key(|field| field.number);
        for theirs in &incoming {
            // Configured fields retain their declared capabilities, while
            // optional writers determine the effective persisted schema. A
            // matching declaration must not hide a conflicting live segment.
            if let Some(persisted) = self.segments_schema(&theirs.name)? {
                if let Some(message) = super::explicit::schema_change(&persisted, theirs) {
                    return Err(Error::AddIndexes(message));
                }
            }
            let ours = match self.cfg.field_by_name(&theirs.name) {
                Some(ours) => Some(ours.clone()),
                None => self
                    .cfg
                    .explicit_field_numbers
                    .schema(&theirs.name)
                    .cloned()
                    .or(self.segments_schema(&theirs.name)?)
                    .or_else(|| incoming_fields.get(&theirs.name).cloned()),
            };
            // `FieldNumbers.verifyFieldInfo`, for every field of every
            // incoming segment, followed by addOrGet's persistent reservation.
            if let Some(ours) = ours {
                if let Some(message) = super::explicit::schema_change(&ours, theirs) {
                    return Err(Error::AddIndexes(message));
                }
            } else {
                incoming_fields.insert(theirs.name.clone(), theirs.clone());
            }
            if self.cfg.explicit {
                self.cfg_mut()
                    .explicit_field_numbers
                    .register(theirs, Some(theirs.number))?;
            }
        }
        Ok(())
    }

    /// `IndexWriter.copySegmentAsIs`.
    fn copy_segment_as_is(
        &mut self,
        src: &dyn Directory,
        sci: &SegmentCommitInfo,
        mut si: segment_info::SegmentInfo,
    ) -> Result<SegmentCommitInfo> {
        let from = sci.segment_name.clone();
        let to = self.new_segment_name();
        let si_name = format!("{from}.si");
        for file in sci.files(&si.files) {
            if file == si_name {
                continue;
            }
            let dest = rename(&file, &from, &to);
            self.dir.copy_from(src, &file, &dest)?;
        }
        si.files = si.files.iter().map(|f| rename(f, &from, &to)).collect();
        let new_si = format!("{to}.si");
        super::write_file(self.dir, &new_si, &segment_info::write(&si, ""))?;
        // Not synced: the next commit syncs every file it names that this
        // writer wrote (`IndexWriter.startCommit`).
        let mut added = sci.clone();
        added.segment_name = to.clone();
        added.field_infos_files = sci
            .field_infos_files
            .iter()
            .map(|f| rename(f, &from, &to))
            .collect();
        added.dv_update_files = sci
            .dv_update_files
            .iter()
            .map(|(field, files)| {
                (
                    *field,
                    files.iter().map(|f| rename(f, &from, &to)).collect(),
                )
            })
            .collect();
        // A merged segment's convention: open to every delete buffered from
        // here on (the flush above left none pending).
        added.set_buffered_deletes_gen(-1);
        Ok(added)
    }
}

#[cfg(test)]
mod tests;
