//! `IndexWriter.tryDeleteDocument` and `IndexWriter.tryUpdateDocValue`:
//! delete, or update the doc values of, one document by its doc id in a
//! near-real-time reader's view, as long as the reader's segment is still in
//! the writer.
//!
//! Java resolves the reader's leaf to its `SegmentReader.getOriginalSegmentInfo`
//! and checks that exact object is still in the writer's `segmentInfos`; the
//! reader here is `lucene-search`'s, above this crate, so it is identified by
//! the segment list it was opened over ([`SegmentInfos`], `DirectoryReader::
//! segment_infos`) and a segment counts as "still in the writer" when one
//! with the same name and id is. Java answers `-1` when it is not; this
//! returns `None`.
//!
//! Both are buffered in memory, as Java keeps them in the pooled
//! `ReadersAndUpdates`: consecutive deletes of a segment become one new
//! `.liv` generation, and consecutive doc-values updates one doc-values
//! generation, instead of one each. The writer writes them before any other
//! operation touches its segments ([`IndexWriter::write_pending_try_changes`],
//! called at the top of every writer operation), so a flush, commit, merge,
//! near-real-time reader, delete or update by term sees them exactly as if
//! each had been written at once. A delete that would leave its segment with
//! no live document is written at once, so the segment is dropped as Java
//! drops it. A failed write stays buffered for the next operation to retry;
//! a rollback or `delete_all` discards them like any other uncommitted
//! change.

use super::{DocValuesUpdate, Error, IndexWriter, Result, SegmentInfos, SeqNo};
use crate::deletes;
use lucene_util::fixed_bit_set::FixedBitSet;

/// One segment's buffered [`IndexWriter::try_update_doc_value`]s, applied in
/// order (a later update of a document's field wins).
pub(crate) struct PendingTryUpdates {
    segment_name: String,
    segment_id: [u8; lucene_store::codec_util::ID_LENGTH],
    numeric: Vec<super::PerFieldNumericUpdates>,
    binary: Vec<super::PerFieldBinaryUpdates>,
}

/// One segment's buffered [`IndexWriter::try_delete_document`]s.
pub(crate) struct PendingTryDeletes {
    segment_name: String,
    segment_id: [u8; lucene_store::codec_util::ID_LENGTH],
    /// The segment's live docs as of its current `.liv`; `None` when it has
    /// no deletions.
    base_live: Option<FixedBitSet>,
    /// The documents buffered for deletion, `max_doc` bits long.
    deleted: FixedBitSet,
    count: usize,
}

/// Where a doc id of a reader landed in the writer.
struct Target {
    /// `true`: `segment_infos.segments[index]`; `false`: `flushed_segments[index]`.
    committed: bool,
    index: usize,
    leaf_doc: i32,
    max_doc: usize,
}

impl IndexWriter<'_> {
    /// `IndexWriter.tryDeleteDocument(reader, docID)`: deletes the document
    /// `doc_id` of a reader opened over `reader_infos` (a near-real-time
    /// reader of this writer), returning the operation's sequence number --
    /// or `None` when the document's segment is no longer in the writer
    /// (merged away), so the caller must delete by term or query instead.
    ///
    /// A segment the delete leaves with no live document is dropped, as
    /// Java's does (unless the merge policy keeps fully deleted segments).
    pub fn try_delete_document(
        &mut self,
        reader_infos: &SegmentInfos,
        doc_id: i32,
    ) -> Result<Option<SeqNo>> {
        Ok(self
            .try_delete_resolved(reader_infos, doc_id)?
            .then(|| self.delete_queue.next_sequence_number()))
    }

    /// [`Self::try_delete_document`] without the sequence number: whether
    /// the document's segment was still in the writer. A concurrent writer
    /// numbers the operation from its own log.
    pub(crate) fn try_delete_resolved(
        &mut self,
        reader_infos: &SegmentInfos,
        doc_id: i32,
    ) -> Result<bool> {
        let Some(target) = self.locate(reader_infos, doc_id)? else {
            return Ok(false);
        };
        let sci = self.target_mut(&target).clone();
        let leaf_doc = usize::try_from(target.leaf_doc).unwrap_or(0);
        let position = match self
            .pending_try_deletes
            .iter()
            .position(|p| p.segment_name == sci.segment_name && p.segment_id == sci.segment_id)
        {
            Some(position) => position,
            None => {
                let base_live = self.read_live_docs(&sci, target.max_doc)?;
                self.pending_try_deletes.push(PendingTryDeletes {
                    segment_name: sci.segment_name.clone(),
                    segment_id: sci.segment_id,
                    base_live,
                    deleted: FixedBitSet::new(target.max_doc),
                    count: 0,
                });
                self.pending_try_deletes.len().saturating_sub(1)
            }
        };
        let pending = &mut self.pending_try_deletes[position];
        // FBS: `leaf_doc < max_doc`, which `locate` read off the `.si` that
        // sized both bitsets (`parse` checks the `.liv`'s length against it).
        let already = leaf_doc >= pending.deleted.len()
            || pending.deleted.get(leaf_doc)
            || pending.base_live.as_ref().is_some_and(|l| !l.get(leaf_doc));
        if already {
            if pending.count == 0 {
                // Nothing to buffer: forget the entry this call made.
                self.pending_try_deletes.swap_remove(position);
            }
            return Ok(true);
        }
        pending.deleted.set(leaf_doc);
        pending.count = pending.count.saturating_add(1);
        let deleted = usize::try_from(sci.del_count)
            .unwrap_or(0)
            .saturating_add(pending.count);
        if deleted >= target.max_doc {
            // The segment is left with no live document: write now, so it is
            // dropped as Java drops it.
            let pending = self.pending_try_deletes.swap_remove(position);
            if let Err(e) = self.write_try_deletes(&pending) {
                self.pending_try_deletes.push(pending);
                return Err(e);
            }
            let live_infos = self.live_infos();
            self.deleter.checkpoint(&live_infos, false)?;
        }
        Ok(true)
    }

    /// Writes every buffered [`Self::try_delete_document`], one `.liv`
    /// generation per segment. Every writer operation calls this first, so
    /// nothing else ever sees a segment with deletes still in memory.
    pub(crate) fn write_pending_try_changes(&mut self) -> Result<()> {
        if self.pending_try_deletes.is_empty() && self.pending_try_updates.is_empty() {
            return Ok(());
        }
        let mut pending = std::mem::take(&mut self.pending_try_deletes);
        let mut result = Ok(());
        while let Some(p) = pending.pop() {
            if let Err(e) = self.write_try_deletes(&p) {
                // Kept for the next operation to retry, as Java keeps a
                // failed `writeLiveDocs`' deletes pending.
                pending.push(p);
                self.pending_try_deletes = pending;
                result = Err(e);
                break;
            }
        }
        if result.is_ok() {
            let mut pending = std::mem::take(&mut self.pending_try_updates);
            // Oldest segment first; each segment's updates in call order.
            pending.reverse();
            while let Some(p) = pending.pop() {
                if let Err(e) = self.write_try_updates(&p) {
                    pending.push(p);
                    pending.reverse();
                    self.pending_try_updates = pending;
                    result = Err(e);
                    break;
                }
            }
        }
        let live_infos = self.live_infos();
        self.deleter.checkpoint(&live_infos, false)?;
        result
    }

    /// Drops the buffered deletes: a rollback or `deleteAll` discards them.
    pub(crate) fn discard_pending_try_changes(&mut self) {
        self.pending_try_deletes.clear();
        self.pending_try_updates.clear();
    }

    fn read_live_docs(
        &self,
        sci: &super::SegmentCommitInfo,
        max_doc: usize,
    ) -> Result<Option<FixedBitSet>> {
        if sci.del_gen < 0 {
            return Ok(None);
        }
        let liv = self
            .dir
            .open(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))?;
        Ok(Some(lucene_codecs::live_docs::parse(
            &liv,
            &sci.segment_id,
            sci.del_gen,
            max_doc,
            usize::try_from(sci.del_count).unwrap_or(0),
        )?))
    }

    /// One segment's buffered deletes as a new `.liv` generation; a segment
    /// merged away or dropped since has nothing to write.
    fn write_try_deletes(&mut self, pending: &PendingTryDeletes) -> Result<()> {
        if pending.count == 0 {
            return Ok(());
        }
        let same = |c: &super::SegmentCommitInfo| {
            c.segment_name == pending.segment_name && c.segment_id == pending.segment_id
        };
        let found = self
            .segment_infos
            .segments
            .iter()
            .position(same)
            .map(|index| (true, index))
            .or_else(|| {
                self.flushed_segments
                    .iter()
                    .position(same)
                    .map(|index| (false, index))
            });
        let Some((committed, index)) = found else {
            return Ok(());
        };
        let target = Target {
            committed,
            index,
            leaf_doc: 0,
            max_doc: pending.deleted.len(),
        };
        let sci = self.target_mut(&target).clone();
        // Nothing but a doc-values update can have touched the segment since
        // the deletes were buffered, and that leaves its `.liv` alone; read
        // it again rather than trust that.
        let live = if sci.del_gen >= 0 {
            self.read_live_docs(&sci, target.max_doc)?
        } else {
            None
        };
        let mut docs: Vec<i32> = Vec::with_capacity(pending.count);
        pending.deleted.for_each_set_bit(|doc| {
            if let Ok(doc) = i32::try_from(doc) {
                docs.push(doc);
            }
        });
        let updated = deletes::apply_deletes_with_codec(
            self.dir,
            &sci,
            live.as_ref(),
            target.max_doc,
            docs,
            self.cfg.codec.as_deref(),
        );
        if updated.is_err() {
            // `PendingDeletes.writeLiveDocs`' failure path:
            // `advanceNextWriteDelGen()` on the live segment.
            let live_sci = self.target_mut(&target);
            let next = live_sci.next_write_del_gen();
            live_sci.set_next_write_del_gen(next.saturating_add(1));
        }
        let updated = updated?;
        let fully_deleted = usize::try_from(updated.del_count).is_ok_and(|n| n == target.max_doc);
        *self.target_mut(&target) = updated;
        if fully_deleted {
            let flags = |committed: bool, len: usize| -> Vec<bool> {
                (0..len)
                    .map(|i| target.committed == committed && i == target.index)
                    .collect()
            };
            let committed = flags(true, self.segment_infos.segments.len());
            let flushed = flags(false, self.flushed_segments.len());
            self.drop_fully_deleted_segments(&committed, &flushed);
        }
        Ok(())
    }

    /// `IndexWriter.tryUpdateDocValue(reader, docID, fields...)`: sets the
    /// NUMERIC/BINARY doc values of document `doc_id` of a reader opened over
    /// `reader_infos`, deleted or not (which is how a soft-deleted document is
    /// revived); a `None` value removes the field's value. Each update's
    /// `term` is ignored -- Java builds these updates with a `null` term.
    /// `None` when the document's segment is no longer in the writer.
    pub fn try_update_doc_value(
        &mut self,
        reader_infos: &SegmentInfos,
        doc_id: i32,
        updates: &[DocValuesUpdate],
    ) -> Result<Option<SeqNo>> {
        Ok(self
            .try_update_resolved(reader_infos, doc_id, updates)?
            .then(|| self.delete_queue.next_sequence_number()))
    }

    /// [`Self::try_update_doc_value`] without the sequence number.
    pub(crate) fn try_update_resolved(
        &mut self,
        reader_infos: &SegmentInfos,
        doc_id: i32,
        updates: &[DocValuesUpdate],
    ) -> Result<bool> {
        if updates.is_empty() {
            return Err(Error::NoDocValuesUpdatesSupplied);
        }
        self.create_doc_values_only_fields(updates)?;
        let mut numeric: Vec<super::PerFieldNumericUpdates> = Vec::new();
        let mut binary: Vec<super::PerFieldBinaryUpdates> = Vec::new();
        for update in updates {
            self.cfg.verify_doc_values_update_field(update)?;
            let number = self
                .cfg
                .fields
                .iter()
                .find(|f| f.name == update.field())
                .map(|f| f.number)
                .ok_or_else(|| Error::UnknownDocValuesUpdateField(update.field().to_string()))?;
            match update {
                DocValuesUpdate::Numeric { value, .. } => numeric.push((number, vec![(0, *value)])),
                DocValuesUpdate::Binary { value, .. } => {
                    binary.push((number, vec![(0, value.clone())]))
                }
            }
        }
        let Some(target) = self.locate(reader_infos, doc_id)? else {
            return Ok(false);
        };
        let sci = self.target_mut(&target);
        let (name, id) = (sci.segment_name.clone(), sci.segment_id);
        let position = match self
            .pending_try_updates
            .iter()
            .position(|p| p.segment_name == name && p.segment_id == id)
        {
            Some(position) => position,
            None => {
                self.pending_try_updates.push(PendingTryUpdates {
                    segment_name: name,
                    segment_id: id,
                    numeric: Vec::new(),
                    binary: Vec::new(),
                });
                self.pending_try_updates.len().saturating_sub(1)
            }
        };
        let pending = &mut self.pending_try_updates[position];
        for (number, mut docs) in numeric {
            docs[0].0 = target.leaf_doc;
            match pending.numeric.iter_mut().find(|(n, _)| *n == number) {
                Some((_, buffered)) => buffered.append(&mut docs),
                None => pending.numeric.push((number, docs)),
            }
        }
        for (number, mut docs) in binary {
            docs[0].0 = target.leaf_doc;
            match pending.binary.iter_mut().find(|(n, _)| *n == number) {
                Some((_, buffered)) => buffered.append(&mut docs),
                None => pending.binary.push((number, docs)),
            }
        }
        Ok(true)
    }

    /// One segment's buffered doc-values updates as one generation; a
    /// segment merged away or dropped since has nothing to write.
    fn write_try_updates(&mut self, pending: &PendingTryUpdates) -> Result<()> {
        let same = |c: &super::SegmentCommitInfo| {
            c.segment_name == pending.segment_name && c.segment_id == pending.segment_id
        };
        let target = if let Some(index) = self.segment_infos.segments.iter().position(same) {
            &mut self.segment_infos.segments[index]
        } else if let Some(index) = self.flushed_segments.iter().position(same) {
            &mut self.flushed_segments[index]
        } else {
            return Ok(());
        };
        let mut sci = target.clone();
        let written = super::IndexingConfig::write_doc_values_update_generation(
            self.dir,
            &mut sci,
            &pending.numeric,
            &pending.binary,
            &self.cfg.fields,
            self.cfg.codec.as_deref(),
        );
        // On failure `sci` describes only files that exist, with its
        // next-write generations advanced past the failed attempt's.
        *target = sci;
        written
    }

    /// `ReaderUtil.subIndex` over the reader's segments, then the writer's
    /// own entry for that segment, if it still has one.
    fn locate(&self, reader_infos: &SegmentInfos, doc_id: i32) -> Result<Option<Target>> {
        if doc_id < 0 {
            return Err(Error::Explicit(format!("docID must be >= 0, got {doc_id}")));
        }
        let mut base = 0i32;
        for s in &reader_infos.segments {
            let si = self.cfg.reader_pool.segment_info(self.dir, s)?;
            let end = base.saturating_add(si.doc_count);
            if doc_id < end {
                let leaf_doc = doc_id.saturating_sub(base);
                let max_doc = usize::try_from(si.doc_count).unwrap_or(0);
                let same = |c: &super::SegmentCommitInfo| {
                    c.segment_name == s.segment_name && c.segment_id == s.segment_id
                };
                let found = self
                    .segment_infos
                    .segments
                    .iter()
                    .position(same)
                    .map(|index| (true, index))
                    .or_else(|| {
                        self.flushed_segments
                            .iter()
                            .position(same)
                            .map(|index| (false, index))
                    });
                return Ok(found.map(|(committed, index)| Target {
                    committed,
                    index,
                    leaf_doc,
                    max_doc,
                }));
            }
            base = end;
        }
        Err(Error::Explicit(format!(
            "docID {doc_id} is out of bounds for a reader of maxDoc {base}"
        )))
    }

    fn target_mut(&mut self, target: &Target) -> &mut super::SegmentCommitInfo {
        if target.committed {
            &mut self.segment_infos.segments[target.index]
        } else {
            &mut self.flushed_segments[target.index]
        }
    }
}

#[cfg(test)]
mod tests;
