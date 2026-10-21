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
//! Both first flush this writer (Java's `flush(false, true)`), hold each
//! source's `write.lock` so no writer changes it underneath (Java's
//! `acquireWriteLocks`), refuse a source whose segments are not sorted by a
//! sort congruent with this writer's index sort, and refuse a field whose
//! schema conflicts with this writer's (`FieldInfos.verifyFieldInfos`). The
//! added segments are published in this writer's view; the next commit makes
//! them durable, and a rollback drops them.
//!
//! # What differs from Java
//!
//! - `addIndexes(CodecReader...)` takes directories (their latest commits)
//!   rather than arbitrary `CodecReader`s -- readers live above this crate --
//!   and is carried out as a copy of the sources' segments followed by one
//!   merge of those copies. The merged segment is the same, its diagnostics
//!   saying `addIndexes(CodecReader...)` as Java's do.
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
        let added = self.copy_segments_in(sources, None)?;
        self.segment_infos.segments.extend(added);
        self.prune_segment_versions();
        let live = self.live_infos();
        self.deleter.checkpoint(&live, false)?;
        Ok(self.delete_queue.next_sequence_number())
    }

    /// `IndexWriter.addIndexes(CodecReader...)` over every segment of each
    /// source's latest commit: one new segment holding all of their live
    /// documents. See the module documentation.
    pub fn add_indexes_merged(&mut self, sources: &[&dyn Directory]) -> Result<SeqNo> {
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
        self.add_merged(sources, Some(keep), None)
    }

    fn add_merged(
        &mut self,
        sources: &[&dyn Directory],
        keep: Option<&dyn Fn(usize, usize, i32) -> bool>,
        remap: Option<std::sync::Arc<crate::merge::SortedNumericRemap>>,
    ) -> Result<SeqNo> {
        let added = self.copy_segments_in(sources, keep)?;
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
            Ok(true) => Ok(self.delete_queue.next_sequence_number()),
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
    fn copy_segments_in(
        &mut self,
        sources: &[&dyn Directory],
        keep: Option<&dyn Fn(usize, usize, i32) -> bool>,
    ) -> Result<Vec<SegmentCommitInfo>> {
        if self.prepared_commit.is_some() {
            return Err(Error::PreparedCommitPending("add_indexes"));
        }
        self.flush()?;
        // `acquireWriteLocks`: held until every copy is done.
        let _locks = sources
            .iter()
            .map(|d| d.obtain_lock(WRITE_LOCK_NAME))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut plans = Vec::new();
        let mut total = self.live_doc_total()?;
        for (source, src) in sources.iter().enumerate() {
            let infos = segment_infos::read_latest(*src)?;
            for (segment, sci) in infos.segments.into_iter().enumerate() {
                let si = segment_info::parse_for_codec(
                    &src.open(&format!("{}.si", sci.segment_name))?,
                    &sci.segment_id,
                    &sci.codec_name,
                )?;
                self.check_addable(*src, &sci, &si)?;
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
        Ok(crate::deletes::apply_deletes(
            self.dir, &sci, None, max_doc, hidden,
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
    /// writer's.
    fn check_addable(
        &self,
        src: &dyn Directory,
        sci: &SegmentCommitInfo,
        si: &segment_info::SegmentInfo,
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
        let incoming = super::segment_field_infos(src, sci)?;
        for theirs in &incoming.fields {
            let Some(ours) = self.cfg.fields.iter().find(|f| f.name == theirs.name) else {
                continue;
            };
            if let Some(what) = Self::schema_conflict(ours, theirs) {
                return Err(Error::AddIndexes(format!(
                    "cannot change field {:?} from this index's {what} to segment {}'s",
                    theirs.name, sci.segment_name
                )));
            }
        }
        Ok(())
    }

    /// What an incoming segment's field would change in this writer's field of
    /// the same name -- `FieldInfos.FieldNumbers.verifyFieldInfo` /
    /// `verifySameSchema`: a doc-values type or index options both sides set
    /// differently, point or vector shape, the soft-deletes role.
    fn schema_conflict(ours: &FieldInfo, theirs: &FieldInfo) -> Option<&'static str> {
        use lucene_codecs::field_infos::{DocValuesType, IndexOptions};
        [
            (ours.doc_values_type != theirs.doc_values_type
                && ours.doc_values_type != DocValuesType::None
                && theirs.doc_values_type != DocValuesType::None)
                .then_some("doc values type"),
            (ours.index_options != theirs.index_options
                && ours.index_options != IndexOptions::None
                && theirs.index_options != IndexOptions::None)
                .then_some("index options"),
            (theirs.point_dimension_count != 0
                && (ours.point_dimension_count, ours.point_num_bytes)
                    != (theirs.point_dimension_count, theirs.point_num_bytes))
                .then_some("point dimensions"),
            (theirs.vector_dimension != 0
                && (
                    ours.vector_dimension,
                    ours.vector_encoding,
                    ours.vector_similarity_function,
                ) != (
                    theirs.vector_dimension,
                    theirs.vector_encoding,
                    theirs.vector_similarity_function,
                ))
                .then_some("vector"),
            (ours.soft_deletes_field != theirs.soft_deletes_field).then_some("soft-deletes"),
        ]
        .into_iter()
        .flatten()
        .next()
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
        let mut copied = Vec::new();
        for file in sci.files(&si.files) {
            if file == si_name {
                continue;
            }
            let dest = rename(&file, &from, &to);
            self.dir.copy_from(src, &file, &dest)?;
            copied.push(dest);
        }
        si.files = si.files.iter().map(|f| rename(f, &from, &to)).collect();
        let new_si = format!("{to}.si");
        super::write_file(self.dir, &new_si, &segment_info::write(&si, ""))?;
        copied.push(new_si);
        self.dir.sync(&copied)?;
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
