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
//! A delete is buffered in memory, as Java keeps it in the pooled
//! `ReadersAndUpdates`: consecutive deletes of a segment become one new
//! `.liv` generation instead of one each. The writer writes them before any
//! other operation touches its segments ([`IndexWriter::write_pending_try_deletes`],
//! called at the top of every writer operation), so a flush, commit, merge,
//! near-real-time reader or delete by term sees them exactly as if each had
//! been written at once. A delete that would leave its segment with no live
//! document is written at once, so the segment is dropped as Java drops it.
//! A rollback discards them like any other uncommitted change.
//!
//! A doc-values update is written at once, as a new doc-values generation
//! referenced by the writer's in-memory view and made durable by the next
//! commit.

use super::{DocValuesUpdate, Error, IndexWriter, Result, SegmentInfos, SeqNo};
use crate::deletes;
use lucene_util::fixed_bit_set::FixedBitSet;

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
    pub(crate) fn write_pending_try_deletes(&mut self) -> Result<()> {
        if self.pending_try_deletes.is_empty() {
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
        let live_infos = self.live_infos();
        self.deleter.checkpoint(&live_infos, false)?;
        result
    }

    /// Drops the buffered deletes: a rollback or `deleteAll` discards them.
    pub(crate) fn discard_pending_try_deletes(&mut self) {
        self.pending_try_deletes.clear();
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
        self.write_pending_try_deletes()?;
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
        self.write_pending_try_deletes()?;
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
        for (_, docs) in &mut numeric {
            docs[0].0 = target.leaf_doc;
        }
        for (_, docs) in &mut binary {
            docs[0].0 = target.leaf_doc;
        }
        let dir = self.dir;
        let schema = self.cfg.fields.clone();
        let mut sci = self.target_mut(&target).clone();
        super::IndexingConfig::write_doc_values_update_generation(
            dir,
            &mut sci,
            &numeric,
            &binary,
            &schema,
            self.cfg.codec.as_deref(),
        )?;
        *self.target_mut(&target) = sci;
        let live_infos = self.live_infos();
        self.deleter.checkpoint(&live_infos, false)?;
        Ok(true)
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
mod tests {
    use super::*;
    use crate::buffered_updates::Term;
    use crate::index_writer::DISABLE_AUTO_FLUSH_MB;
    use crate::segment_info::{self, LuceneVersion};
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions,
    };
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
    use lucene_store::directory::Directory;
    use lucene_store::FsDirectory;
    use lucene_util::test_support::TempDir;

    fn writer(dir: &FsDirectory) -> IndexWriter<'_> {
        let fields = vec![
            FieldInfo::new("id", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
            FieldInfo::new("n", 1)
                .with_omit_norms(true)
                .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1),
            FieldInfo::new("b", 2)
                .with_omit_norms(true)
                .with_doc_values(DocValuesType::Binary, DocValuesSkipIndexType::None, -1),
        ];
        let mut w = IndexWriter::open(
            dir,
            fields,
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        w.set_postings_field(Some("id")).unwrap();
        w.set_doc_values_field(Some("n")).unwrap();
        w.add_doc_values_field("b").unwrap();
        w.set_max_buffered_docs(1000).unwrap();
        w.set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB).unwrap();
        w
    }

    fn doc(i: i64) -> Document {
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(format!("d{i}")),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(i),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::Binary(vec![i as u8]),
                },
            ],
        }
    }

    /// Every document's current NUMERIC value of `field` in `sci`, through
    /// the newest doc-values generation.
    fn numeric_values(
        dir: &FsDirectory,
        sci: &crate::segment_infos::SegmentCommitInfo,
        field: &str,
    ) -> Vec<Option<i64>> {
        let si = segment_info::parse(
            &dir.open(&format!("{}.si", sci.segment_name)).unwrap(),
            &sci.segment_id,
        )
        .unwrap();
        let infos = crate::field_updates::read_current_field_infos(dir, sci, &si.files).unwrap();
        let index = infos.fields.iter().position(|f| f.name == field).unwrap();
        let per_field = crate::field_updates::per_field_component(
            &infos.fields[index],
            &crate::index_writer::per_field_codec_suffix("Lucene90"),
        );
        let (meta, data) = crate::field_updates::read_current_column(
            dir, sci, &si.files, &infos, index, &per_field,
        )
        .unwrap()
        .unwrap();
        let entry = meta.numeric_entry(infos.fields[index].number).unwrap();
        (0..si.doc_count)
            .map(|d| lucene_codecs::doc_values::numeric_value(&data, entry, d).unwrap())
            .collect()
    }

    /// Each segment's deletion count, buffered deletes written first.
    fn live(w: &mut IndexWriter<'_>) -> Vec<i32> {
        w.write_pending_try_deletes().unwrap();
        w.live_infos()
            .segments
            .iter()
            .map(|s| s.del_count)
            .collect()
    }

    #[test]
    fn deletes_by_doc_id_of_a_reader_snapshot() {
        let tmp = TempDir::new("try-delete");
        let dir = FsDirectory::open(&tmp);
        let mut w = writer(&dir);
        for i in 0..3 {
            w.add_document(doc(i)).unwrap();
        }
        w.commit().unwrap();
        for i in 3..5 {
            w.add_document(doc(i)).unwrap();
        }
        let snap = w.nrt_snapshot(true, false).unwrap();
        // doc 4 is the second of the flushed (uncommitted) segment.
        let seq = w.try_delete_document(&snap.segment_infos, 4).unwrap();
        assert!(seq.is_some());
        assert_eq!(live(&mut w), [0, 1]);
        // Deleting it again is a no-op that still succeeds.
        assert!(w
            .try_delete_document(&snap.segment_infos, 4)
            .unwrap()
            .is_some());
        assert_eq!(live(&mut w), [0, 1]);
        w.try_delete_document(&snap.segment_infos, 1)
            .unwrap()
            .unwrap();
        assert_eq!(live(&mut w), [1, 1]);
        // The flushed segment emptied is dropped.
        w.try_delete_document(&snap.segment_infos, 3)
            .unwrap()
            .unwrap();
        assert_eq!(live(&mut w), [1]);
        w.commit().unwrap();
        for r in crate::check_index::check_directory(&dir).unwrap() {
            assert!(r.all_passed(), "{:?}", r.failures());
        }
        // Out of bounds, and a segment merged away.
        assert!(w.try_delete_document(&snap.segment_infos, 9).is_err());
        assert!(w.try_delete_document(&snap.segment_infos, -1).is_err());
        w.add_document(doc(9)).unwrap();
        w.commit().unwrap();
        w.force_merge(1).unwrap();
        assert_eq!(w.try_delete_document(&snap.segment_infos, 0).unwrap(), None);
    }

    #[test]
    fn buffered_deletes_become_one_generation_and_reach_the_next_reader() {
        let tmp = TempDir::new("try-delete-buffered");
        let dir = FsDirectory::open(&tmp);
        let mut w = writer(&dir);
        for i in 0..10 {
            w.add_document(doc(i)).unwrap();
        }
        w.commit().unwrap();
        let snap = w.nrt_snapshot(true, false).unwrap();
        let before = w.live_infos().segments[0].del_gen;
        for d in [1, 3, 5, 3] {
            w.try_delete_document(&snap.segment_infos, d)
                .unwrap()
                .unwrap();
        }
        // Nothing written yet, but the writer knows it has changes.
        assert_eq!(w.live_infos().segments[0].del_gen, before);
        assert!(w.has_uncommitted_changes());
        assert!(!w.nrt_is_current(&snap.segment_infos));
        // The next reader sees all three, from one new `.liv` generation.
        let next = w.nrt_snapshot(true, false).unwrap();
        let sci = &next.segment_infos.segments[0];
        assert_eq!(sci.del_count, 3);
        assert_eq!(sci.del_gen, before.max(0) + 1);
        // A delete by term on top of them keeps them.
        w.delete_documents_by_term(&[Term::new("id", "d7")])
            .unwrap();
        w.try_delete_document(&next.segment_infos, 8)
            .unwrap()
            .unwrap();
        w.commit().unwrap();
        assert_eq!(w.segment_infos().segments[0].del_count, 5);
        // A rollback drops a buffered delete with everything else uncommitted.
        let snap = w.nrt_snapshot(true, false).unwrap();
        w.try_delete_document(&snap.segment_infos, 0)
            .unwrap()
            .unwrap();
        w.rollback();
        w.commit().unwrap();
        assert_eq!(w.segment_infos().segments[0].del_count, 5);
        for r in crate::check_index::check_directory(&dir).unwrap() {
            assert!(r.all_passed(), "{:?}", r.failures());
        }
    }

    #[test]
    fn updates_doc_values_by_doc_id() {
        let tmp = TempDir::new("try-update");
        let dir = FsDirectory::open(&tmp);
        let mut w = writer(&dir);
        for i in 0..4 {
            w.add_document(doc(i)).unwrap();
        }
        w.commit().unwrap();
        let snap = w.nrt_snapshot(true, false).unwrap();
        let any = Term::new("", "");
        let seq = w
            .try_update_doc_value(
                &snap.segment_infos,
                2,
                &[
                    DocValuesUpdate::Numeric {
                        term: any.clone(),
                        field: "n".into(),
                        value: Some(42),
                    },
                    DocValuesUpdate::Binary {
                        term: any.clone(),
                        field: "b".into(),
                        value: None,
                    },
                ],
            )
            .unwrap();
        assert!(seq.is_some());
        assert!(w.live_infos().segments[0].doc_values_gen > 0);
        w.commit().unwrap();
        for r in crate::check_index::check_directory(&dir).unwrap() {
            assert!(r.all_passed(), "{:?}", r.failures());
        }
        let sci = w.segment_infos().segments[0].clone();
        assert_eq!(
            numeric_values(&dir, &sci, "n"),
            [Some(0), Some(1), Some(42), Some(3)]
        );
        // An update by term on another document lands on top of it.
        w.update_numeric_doc_value(Term::new("id", "d1"), "n", 7)
            .unwrap();
        w.commit().unwrap();
        let sci = w.segment_infos().segments[0].clone();
        assert_eq!(
            numeric_values(&dir, &sci, "n"),
            [Some(0), Some(7), Some(42), Some(3)]
        );
        assert!(matches!(
            w.try_update_doc_value(&snap.segment_infos, 0, &[]),
            Err(Error::NoDocValuesUpdatesSupplied)
        ));
        assert!(w
            .try_update_doc_value(
                &snap.segment_infos,
                0,
                &[DocValuesUpdate::Numeric {
                    term: any,
                    field: "id".into(),
                    value: Some(1)
                }]
            )
            .is_err());
    }
}
