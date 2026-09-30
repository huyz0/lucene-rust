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
            self.rollback();
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
mod tests {
    use super::*;
    use crate::index_writer::DISABLE_AUTO_FLUSH_MB;
    use crate::merge_policy::MergePolicyConfig;
    use crate::segment_info::LuceneVersion;
    use lucene_codecs::field_infos::FieldInfo;
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
    use lucene_store::FsDirectory;
    use lucene_util::test_support::TempDir;
    use std::sync::Mutex;

    fn writer(dir: &FsDirectory) -> IndexWriter<'_> {
        let mut w = IndexWriter::open(
            dir,
            vec![FieldInfo::new("id", 0)],
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        w.set_max_buffered_docs(1000).unwrap();
        w.set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB).unwrap();
        w
    }

    fn doc(i: usize) -> Document {
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(format!("d{i}")),
            }],
        }
    }

    fn committed_segments(dir: &FsDirectory) -> usize {
        crate::segment_infos::read_latest(dir)
            .map(|s| s.segments.len())
            .unwrap_or(0)
    }

    #[derive(Default)]
    struct Recorder(Mutex<Vec<String>>);

    impl MergedSegmentWarmer for Recorder {
        fn warm(&self, dir: &dyn Directory, segment: &SegmentCommitInfo) -> Result<()> {
            // The merged segment's files are already there to warm.
            dir.open(&format!("{}.si", segment.segment_name))?;
            self.0.lock().unwrap().push(segment.segment_name.clone());
            Ok(())
        }
    }

    /// `maybeMerge` merges without committing; the merge becomes durable with
    /// the next commit. The warmer sees the merged segment first.
    #[test]
    fn maybe_merge_publishes_in_memory_and_warms() {
        let tmp = TempDir::new("maybe-merge");
        let dir = FsDirectory::open(&tmp);
        let mut w = writer(&dir);
        w.maybe_merge().unwrap(); // no policy: nothing to do
        for i in 0..6 {
            w.add_document(doc(i)).unwrap();
            w.commit().unwrap();
        }
        assert_eq!(committed_segments(&dir), 6);
        let warmer = Arc::new(Recorder::default());
        w.set_merged_segment_warmer(Some(warmer.clone()));
        w.set_merge_policy(Some(MergePolicyConfig {
            max_merge_at_once: 10,
            segments_per_tier: 2,
            floor_segment_size: 1 << 30,
            ..MergePolicyConfig::default()
        }));
        w.maybe_merge().unwrap();
        assert!(w.segment_infos().segments.len() < 6);
        assert_eq!(committed_segments(&dir), 6, "not durable yet");
        assert!(!warmer.0.lock().unwrap().is_empty());
        assert!(w.has_uncommitted_changes());
        w.commit().unwrap();
        assert!(committed_segments(&dir) < 6);

        w.prepare_commit().unwrap();
        assert!(matches!(
            w.maybe_merge(),
            Err(Error::PreparedCommitPending("maybe_merge"))
        ));
        w.rollback();
    }

    #[test]
    fn flush_next_buffer_flushes_only_when_something_is_buffered() {
        let tmp = TempDir::new("flush-next");
        let dir = FsDirectory::open(&tmp);
        let mut w = writer(&dir);
        assert!(!w.flush_next_buffer().unwrap());
        w.add_document(doc(0)).unwrap();
        assert!(w.flush_next_buffer().unwrap());
        assert_eq!(w.pending_doc_count(), 0);
        assert!(!w.flush_next_buffer().unwrap());
        assert_eq!(committed_segments(&dir), 0);
    }

    /// `close()` commits by default, and rolls back with `commitOnClose`
    /// off -- including a prepared commit.
    #[test]
    fn close_commits_or_rolls_back() {
        let tmp = TempDir::new("close-commit");
        let dir = FsDirectory::open(&tmp);
        let mut w = writer(&dir);
        assert!(w.commit_on_close());
        w.add_document(doc(0)).unwrap();
        w.close().unwrap();
        assert_eq!(committed_segments(&dir), 1);

        let mut w = writer(&dir);
        w.add_document(doc(1)).unwrap();
        w.prepare_commit().unwrap();
        w.close().unwrap();
        assert_eq!(committed_segments(&dir), 2);

        let mut w = writer(&dir);
        w.set_commit_on_close(false);
        assert!(!w.commit_on_close());
        w.add_document(doc(2)).unwrap();
        w.close().unwrap();
        assert_eq!(committed_segments(&dir), 2);
    }
}
