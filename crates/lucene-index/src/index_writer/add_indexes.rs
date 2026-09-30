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
//!   merge of those copies. The merged segment is the same; its diagnostics
//!   say `merge` where Java's say `addIndexes(CodecReader...)`.
//! - An unsorted source cannot be added to an index-sorted writer by
//!   either method: Java sorts such a reader during the merge
//!   (`MergeState.maybeSortReaders`), which this port's merge does not do.

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
        let added = self.copy_segments_in(sources)?;
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
        let added = self.copy_segments_in(sources)?;
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
        let merged = self.execute_merge(&names);
        self.merges_by_caller = by_caller;
        merged?;
        Ok(self.delete_queue.next_sequence_number())
    }

    /// Flushes, checks every source and copies its segments in under fresh
    /// names, returning their new `SegmentCommitInfo`s (not yet published).
    fn copy_segments_in(&mut self, sources: &[&dyn Directory]) -> Result<Vec<SegmentCommitInfo>> {
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
        for src in sources {
            let infos = segment_infos::read_latest(*src)?;
            for sci in infos.segments {
                let si = segment_info::parse_for_codec(
                    &src.open(&format!("{}.si", sci.segment_name))?,
                    &sci.segment_id,
                    &sci.codec_name,
                )?;
                self.check_addable(*src, &sci, &si)?;
                total = total.saturating_add(i64::from(si.doc_count));
                plans.push((*src, sci, si));
            }
        }
        let limit = i64::try_from(self.max_docs).unwrap_or(i64::MAX);
        if total > limit {
            return Err(Error::TooManyDocs(self.max_docs));
        }
        let mut added = Vec::with_capacity(plans.len());
        for (src, sci, si) in plans {
            added.push(self.copy_segment_as_is(src, &sci, si)?);
        }
        Ok(added)
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
        let incoming = crate::field_updates::read_current_field_infos(src, sci, &si.files)?;
        for theirs in &incoming.fields {
            let Some(ours) = self.cfg.fields.iter().find(|f| f.name == theirs.name) else {
                continue;
            };
            let conflict = [
                (ours.doc_values_type != theirs.doc_values_type
                    && ours.doc_values_type != lucene_codecs::field_infos::DocValuesType::None
                    && theirs.doc_values_type != lucene_codecs::field_infos::DocValuesType::None)
                    .then_some("doc values type"),
                (ours.index_options != theirs.index_options
                    && ours.index_options != lucene_codecs::field_infos::IndexOptions::None
                    && theirs.index_options != lucene_codecs::field_infos::IndexOptions::None)
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
            ];
            if let Some(what) = conflict.into_iter().flatten().next() {
                return Err(Error::AddIndexes(format!(
                    "cannot change field {:?} from this index's {what} to segment {}'s",
                    theirs.name, sci.segment_name
                )));
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
