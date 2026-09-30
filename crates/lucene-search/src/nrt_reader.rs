//! Near-real-time readers: `DirectoryReader.open(IndexWriter)`,
//! `DirectoryReader.openIfChanged(reader, writer)` and `isCurrent` for a
//! reader opened from a writer -- the reading half of
//! `StandardDirectoryReader`'s writer path (`open(writer, readerFunction,
//! infos, applyAllDeletes, writeAllDeletes)`).
//!
//! The writer's half -- flushing buffered documents into segments without
//! committing, applying buffered deletes, and pinning the live segment list's
//! files -- is [`lucene_index::nrt`]; any [`NrtSource`] (a single-threaded
//! `IndexWriter` behind a `Mutex`, or a `ConcurrentIndexWriter`) can be read
//! this way. The reader opens every segment of that list, flushed-but-
//! uncommitted ones included, and keeps the pin until it (and every reader
//! sharing it) is dropped.
//!
//! A refresh reuses every unchanged segment of the previous reader
//! ([`DirectoryReader::reopen_at`]), as Java's does through its
//! `ReaderPool`-shared readers; a segment whose deletes changed gets new
//! live docs, and only new segments are read from disk.

use std::sync::Arc;

use lucene_index::nrt::NrtSource;

use crate::directory_reader::{DirectoryReader, Result, SegmentReader};

impl DirectoryReader {
    /// `DirectoryReader.open(IndexWriter)`: `open_nrt(writer, true, false)`.
    pub fn open_from_writer(writer: &dyn NrtSource) -> Result<Self> {
        Self::open_nrt(writer, true, false)
    }

    /// `DirectoryReader.open(IndexWriter, applyAllDeletes, writeAllDeletes)`:
    /// a reader over everything `writer` has indexed, committed or not. See
    /// [`lucene_index::nrt`] for what the two flags do in this port.
    pub fn open_nrt(
        writer: &dyn NrtSource,
        apply_all_deletes: bool,
        write_all_deletes: bool,
    ) -> Result<Self> {
        let snapshot = writer.nrt_snapshot(apply_all_deletes, write_all_deletes)?;
        let mut reader = Self::open_at(writer.directory(), snapshot.segment_infos)?;
        reader.nrt_hold = Some(snapshot.hold);
        Ok(reader)
    }

    /// `DirectoryReader.openIfChanged(reader, writer)`: `None` when `writer`
    /// has nothing this reader does not already show
    /// (`IndexWriter.nrtIsCurrent`), otherwise a new near-real-time reader
    /// sharing every segment of this one that did not change.
    pub fn open_if_changed_nrt(&self, writer: &dyn NrtSource) -> Result<Option<Self>> {
        self.open_if_changed_nrt_with(writer, true, false)
    }

    /// [`Self::open_if_changed_nrt`] with `applyAllDeletes`/`writeAllDeletes`.
    pub fn open_if_changed_nrt_with(
        &self,
        writer: &dyn NrtSource,
        apply_all_deletes: bool,
        write_all_deletes: bool,
    ) -> Result<Option<Self>> {
        if writer.nrt_is_current(&self.segment_infos)? {
            return Ok(None);
        }
        let snapshot = writer.nrt_snapshot(apply_all_deletes, write_all_deletes)?;
        let mut reader = self.reopen_at(writer.directory(), snapshot.segment_infos)?;
        reader.nrt_hold = Some(snapshot.hold);
        Ok(Some(reader))
    }

    /// `DirectoryReader.isCurrent()` for a reader opened from `writer`:
    /// whether it already shows everything the writer has.
    pub fn is_current_nrt(&self, writer: &dyn NrtSource) -> Result<bool> {
        Ok(writer.nrt_is_current(&self.segment_infos)?)
    }

    /// `IndexWriterConfig.setLeafSorter(comparator)` /
    /// `DirectoryReader.open(directory, leafSorter)`: this reader with its
    /// segments in `leaf_sorter`'s order (stable, as Java's `Arrays.sort`
    /// over the leaves), doc bases recomputed; segments are shared, not
    /// reopened, and a near-real-time reader keeps its file pin. The writer's
    /// configuration cannot carry a comparator over this crate's
    /// [`SegmentReader`], so the sorter is applied to the reader a writer
    /// hands out -- which is where Java applies it too.
    pub fn with_leaf_sorter(
        &self,
        leaf_sorter: impl Fn(&SegmentReader, &SegmentReader) -> std::cmp::Ordering,
    ) -> Self {
        let segments = self.segment_readers();
        let mut order: Vec<usize> = (0..segments.len()).collect();
        order.sort_by(|&a, &b| leaf_sorter(&segments[a], &segments[b]));
        self.with_segment_order(&order)
    }

    /// Whether this reader was opened from a writer (and pins that writer's
    /// files) rather than from a commit.
    pub fn is_nrt(&self) -> bool {
        self.nrt_hold.is_some()
    }

    /// The file pin this reader holds, shared with every reader built from
    /// it (`Arc` count included), for tests and diagnostics.
    pub fn nrt_hold(&self) -> Option<&Arc<lucene_index::nrt::NrtFileHold>> {
        self.nrt_hold.as_ref()
    }
}

#[cfg(test)]
mod tests;
