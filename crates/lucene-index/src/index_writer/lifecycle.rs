//! `IndexWriter.maybeMerge()`, `flushNextBuffer()`, `close()` under
//! `IndexWriterConfig.setCommitOnClose`, and `LiveIndexWriterConfig
//! .setMergedSegmentWarmer`.

use std::sync::Arc;

use super::{Error, IndexWriter, Result};
use crate::segment_infos::SegmentCommitInfo;
use lucene_store::directory::Directory;

/// `IndexWriter.IndexReaderWarmer`: called with every newly merged segment
/// after its files are written and before the merge is published, so a
/// near-real-time reader opened next finds it warm. Java hands it the merged
/// segment's `LeafReader`; the reader lives above this crate, so a warmer
/// here gets the directory and the segment, and opens what it wants to warm.
pub trait MergedSegmentWarmer: Send + Sync {
    /// `warm(reader)`. A failure fails the merge, as in Java.
    fn warm(&self, dir: &dyn Directory, segment: &SegmentCommitInfo) -> Result<()>;
}

impl IndexWriter<'_> {
    /// `IndexWriter.maybeMerge()`: runs the merges the merge policy wants now,
    /// one after another on this thread (this writer's scheduler is serial;
    /// a [`crate::concurrent_writer::ConcurrentIndexWriter`] takes any
    /// [`crate::merge_scheduler::MergeScheduler`]). Merged segments are
    /// published in this writer's view only, as `commitMerge` publishes
    /// them, and become durable with the next commit. A no-op without a
    /// merge policy.
    pub fn maybe_merge(&mut self) -> Result<()> {
        self.write_pending_try_changes()?;
        if self.merge_policy.is_none() && self.pluggable_merge_policy.is_none() {
            return Ok(());
        }
        if self.prepared_commit.is_some() {
            return Err(Error::PreparedCommitPending("maybe_merge"));
        }
        let by_caller = std::mem::replace(&mut self.merges_by_caller, true);
        let result = self.auto_merge();
        self.merges_by_caller = by_caller;
        result
    }

    /// `IndexWriter.flushNextBuffer()`: flushes the buffered documents into a
    /// segment (without committing), returning whether there were any. This
    /// writer has one buffer; a [`crate::concurrent_writer::ConcurrentIndexWriter`]
    /// flushes its largest.
    pub fn flush_next_buffer(&mut self) -> Result<bool> {
        self.write_pending_try_changes()?;
        if self.pending_docs.is_empty() {
            return Ok(false);
        }
        self.flush()?;
        Ok(true)
    }

    /// `IndexWriterConfig.setCommitOnClose(on)`: whether [`Self::close`]
    /// commits (`true`, Java's default) or rolls back.
    pub fn set_commit_on_close(&mut self, on: bool) {
        self.commit_on_close = on;
    }

    /// `getCommitOnClose()`.
    pub fn commit_on_close(&self) -> bool {
        self.commit_on_close
    }

    /// `IndexWriter.close()`: commits everything (running the merges a
    /// commit runs) when [`Self::commit_on_close`], else discards everything
    /// since the last commit (`rollback`); then releases the write lock.
    /// Dropping a writer without `close` is a rollback that keeps no files
    /// it wrote (the next writer's deleter reclaims them), as a crashed
    /// Java writer's are.
    pub fn close(mut self) -> Result<()> {
        if self.commit_on_close {
            if self.prepared_commit.is_some() {
                self.finish_commit()?;
            }
            self.commit()?;
        } else {
            self.try_rollback()?;
        }
        Ok(())
    }

    /// `LiveIndexWriterConfig.setMergedSegmentWarmer(warmer)`: see
    /// [`MergedSegmentWarmer`]. `None` (the default) warms nothing.
    pub fn set_merged_segment_warmer(&mut self, warmer: Option<Arc<dyn MergedSegmentWarmer>>) {
        self.cfg_mut().merged_segment_warmer = warmer;
    }
}

#[cfg(test)]
mod tests;
