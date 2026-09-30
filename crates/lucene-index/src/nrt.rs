//! The writer's half of near-real-time readers --
//! `IndexWriter.getReader(applyAllDeletes, writeAllDeletes)` and
//! `IndexWriter.nrtIsCurrent`, as far as they concern the writer: flush what
//! is buffered, apply the buffered deletes, and hand out the **live** segment
//! list (the last commit plus every segment flushed since) with its files
//! pinned against deletion for as long as a reader uses them.
//! `lucene_search::directory_reader::DirectoryReader::open_nrt` opens a
//! reader over it.
//!
//! # What differs from Java
//!
//! - **The pin outlives nothing it should not.** Java's
//!   `StandardDirectoryReader` holds a reference on the writer's
//!   `IndexFileDeleter` and decrements it in `doClose`, under the writer's
//!   monitor. Here the reader owns an [`NrtFileHold`]; dropping it returns
//!   the pinned files to a queue the writer drains at its next flush, commit,
//!   snapshot or `delete_unused_files` -- so a reader outliving its writer is
//!   harmless, and no reader ever takes the writer's lock.
//! - **Deletes are always applied, and written.** This port resolves buffered
//!   deletes at flush and writes them as `.liv` generations, so
//!   `applyAllDeletes = false` (which lets Java return deleted documents) and
//!   `writeAllDeletes = false` (which lets Java keep them only in memory)
//!   both behave as `true` -- which each flag's contract allows.
//! - **The version.** Java's `SegmentInfos.version` counts every change; this
//!   port counts commits. An NRT snapshot of a view that differs from the
//!   last commit gets a version above every one handed out before, and the
//!   next commit is written with a version above that, so versions stay
//!   monotonic across snapshots and commits (`DirectoryReader.getVersion`).
//! - **Readers are not pooled with the writer's delete resolution** (Java's
//!   `ReaderPool` shares one `SegmentReader` per segment between both): the
//!   reader lives in `lucene-search`, above this crate; successive NRT
//!   readers share unchanged segments with each other instead
//!   (`openIfChanged`), which is where the reuse pays.

use std::sync::{Arc, Mutex};

use lucene_store::directory::Directory;

use crate::index_writer::{IndexWriter, Result};
use crate::segment_infos::{SegmentCommitInfo, SegmentInfos};

/// Files an NRT reader pins against deletion. Dropping it hands them back
/// to the writer that pinned them, which releases them at its next
/// operation.
pub struct NrtFileHold {
    files: Vec<String>,
    returned: Arc<Mutex<Vec<Vec<String>>>>,
}

impl std::fmt::Debug for NrtFileHold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NrtFileHold")
            .field("files", &self.files.len())
            .finish()
    }
}

impl NrtFileHold {
    /// The pinned file names.
    pub fn files(&self) -> &[String] {
        &self.files
    }
}

impl Drop for NrtFileHold {
    fn drop(&mut self) {
        let files = std::mem::take(&mut self.files);
        if !files.is_empty() {
            self.returned
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(files);
        }
    }
}

/// What `IndexWriter.getReader` hands `StandardDirectoryReader`: the live
/// segment list and the pin on its files.
#[derive(Debug)]
pub struct NrtSnapshot {
    pub segment_infos: SegmentInfos,
    pub hold: Arc<NrtFileHold>,
}

/// A writer a near-real-time reader can be opened from and refreshed
/// against -- `DirectoryReader.open(IndexWriter)`'s argument, abstracted over
/// this port's two writers (a single-threaded [`IndexWriter`] behind a
/// `Mutex`, and a [`crate::concurrent_writer::ConcurrentIndexWriter`]).
pub trait NrtSource: Send + Sync {
    /// The directory the writer writes, which the reader reads.
    fn directory(&self) -> &dyn Directory;
    /// `IndexWriter.getReader(applyAllDeletes, writeAllDeletes)`, the
    /// writer's half: flush, apply deletes, snapshot and pin.
    fn nrt_snapshot(&self, apply_all_deletes: bool, write_all_deletes: bool)
        -> Result<NrtSnapshot>;
    /// `IndexWriter.nrtIsCurrent(infos)`: whether a reader over `infos`
    /// already sees everything the writer has -- no buffered documents, no
    /// buffered deletes, and the same segments at the same generations.
    fn nrt_is_current(&self, infos: &SegmentInfos) -> Result<bool>;
}

/// The writer-side state of NRT snapshots.
#[derive(Debug, Default)]
pub(crate) struct NrtState {
    /// Pins dropped by readers, waiting to be released.
    returned: Arc<Mutex<Vec<Vec<String>>>>,
    /// The highest version handed to a snapshot; `-1` before the first.
    pub(crate) max_version: i64,
    /// The last snapshot's segment identities and the version it got, so an
    /// unchanged view keeps its version.
    last: Option<(Vec<SegmentKey>, i64)>,
}

impl NrtState {
    pub(crate) fn new() -> Self {
        NrtState {
            returned: Arc::default(),
            max_version: -1,
            last: None,
        }
    }
}

/// What identifies one segment's state for a reader: which segment, and
/// which generation of its deletes, field infos and doc values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SegmentKey {
    name: String,
    id: [u8; 16],
    del_gen: i64,
    del_count: i32,
    field_infos_gen: i64,
    doc_values_gen: i64,
}

pub(crate) fn segment_keys(segments: &[SegmentCommitInfo]) -> Vec<SegmentKey> {
    segments
        .iter()
        .map(|s| SegmentKey {
            name: s.segment_name.clone(),
            id: s.segment_id,
            del_gen: s.del_gen,
            del_count: s.del_count,
            field_infos_gen: s.field_infos_gen,
            doc_values_gen: s.doc_values_gen,
        })
        .collect()
}

impl IndexWriter<'_> {
    /// `IndexWriter.getReader(applyAllDeletes, writeAllDeletes)`, the
    /// writer's half: flushes every buffered document into a segment
    /// (without committing), applies every buffered delete, and returns the
    /// live segment list with its files pinned. See the module
    /// documentation for the two flags.
    pub fn nrt_snapshot(
        &mut self,
        _apply_all_deletes: bool,
        _write_all_deletes: bool,
    ) -> Result<NrtSnapshot> {
        self.release_nrt_holds()?;
        self.flush()?;
        self.nrt_snapshot_of_live_view()
    }

    /// The snapshot of what is already flushed and applied -- the part a
    /// concurrent writer takes under its control lock after its own flush.
    pub(crate) fn nrt_snapshot_of_live_view(&mut self) -> Result<NrtSnapshot> {
        self.release_nrt_holds()?;
        let mut infos = self.live_infos();
        let keys = segment_keys(&infos.segments);
        let committed = self.segment_infos().version;
        let version = match &self.nrt.last {
            Some((last, v)) if *last == keys => *v,
            _ if keys == segment_keys(&self.segment_infos().segments)
                && self.nrt.max_version <= committed =>
            {
                committed
            }
            _ => self.nrt.max_version.max(committed).saturating_add(1),
        };
        self.nrt.max_version = self.nrt.max_version.max(version);
        self.nrt.last = Some((keys, version));
        infos.version = version;
        let files = self.pin_segment_files(&infos.segments)?;
        Ok(NrtSnapshot {
            segment_infos: infos,
            hold: Arc::new(NrtFileHold {
                files,
                returned: Arc::clone(&self.nrt.returned),
            }),
        })
    }

    /// `IndexWriter.nrtIsCurrent(infos)`.
    pub fn nrt_is_current(&self, infos: &SegmentInfos) -> bool {
        !self.has_buffered_changes()
            && segment_keys(&self.live_infos().segments) == segment_keys(&infos.segments)
    }

    /// Releases every pin a dropped NRT reader handed back
    /// (`StandardDirectoryReader.doClose`'s `decRefDeleter`).
    pub(crate) fn release_nrt_holds(&mut self) -> Result<()> {
        let returned: Vec<Vec<String>> =
            std::mem::take(&mut *self.nrt.returned.lock().unwrap_or_else(|p| p.into_inner()));
        for files in returned {
            self.release_files(&files)?;
        }
        Ok(())
    }
}

impl<'d> NrtSource for Mutex<IndexWriter<'d>> {
    fn directory(&self) -> &dyn Directory {
        self.lock().unwrap_or_else(|p| p.into_inner()).dir()
    }

    fn nrt_snapshot(
        &self,
        apply_all_deletes: bool,
        write_all_deletes: bool,
    ) -> Result<NrtSnapshot> {
        self.lock()
            .unwrap_or_else(|p| p.into_inner())
            .nrt_snapshot(apply_all_deletes, write_all_deletes)
    }

    fn nrt_is_current(&self, infos: &SegmentInfos) -> Result<bool> {
        Ok(self
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .nrt_is_current(infos))
    }
}

#[cfg(test)]
mod tests;
