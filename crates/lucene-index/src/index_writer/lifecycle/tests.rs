#![cfg(test)]
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

#[test]
fn segment_count_includes_flushed_segments() {
    let tmp = TempDir::new("segment-count");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir);
    assert_eq!(w.segment_count(), 0);
    w.add_document(doc(0)).unwrap();
    assert!(w.flush_next_buffer().unwrap());
    w.add_document(doc(1)).unwrap();
    w.flush().unwrap();
    assert_eq!(w.segment_count(), 2);
    w.commit().unwrap();
    assert_eq!(w.segment_count(), 2);
    w.force_merge(1).unwrap();
    assert_eq!(w.segment_count(), 1);
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
        segments_per_tier: 2.0,
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

    // A change, so the prepare has a commit to hold open (one with
    // nothing to commit prepares nothing).
    w.set_live_commit_data(vec![("k".into(), "v".into())]);
    w.prepare_commit().unwrap();
    assert!(matches!(
        w.maybe_merge(),
        Err(Error::PreparedCommitPending("maybe_merge"))
    ));
    w.rollback();
}

/// Records whether each merged segment was compound when warmed.
#[derive(Default)]
struct CompoundRecorder(Mutex<Vec<bool>>);

impl MergedSegmentWarmer for CompoundRecorder {
    fn warm(&self, dir: &dyn Directory, segment: &SegmentCommitInfo) -> Result<()> {
        let cfs = dir.open(&format!("{}.cfs", segment.segment_name)).is_ok();
        self.0.lock().unwrap().push(cfs);
        Ok(())
    }
}

/// `mergeMiddle` builds the compound file before it warms the merged
/// segment, so the warmer sees the files the segment is published with.
#[test]
fn the_warmer_sees_the_merged_segment_compound() {
    let tmp = TempDir::new("warm-compound");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir);
    for i in 0..4 {
        w.add_document(doc(i)).unwrap();
        w.commit().unwrap();
    }
    let mut compound = crate::merge_policy::CompoundFileSettings::default();
    compound.set_no_cfs_ratio(1.0).unwrap();
    w.set_merge_compound_file_settings(Some(compound));
    let warmer = Arc::new(CompoundRecorder::default());
    w.set_merged_segment_warmer(Some(warmer.clone()));
    w.force_merge(1).unwrap();
    assert_eq!(*warmer.0.lock().unwrap(), vec![true]);
    let sci = &w.segment_infos().segments[0];
    assert!(dir.open(&format!("{}.cfs", sci.segment_name)).is_ok());
    w.commit().unwrap();
}

/// The files a warmer saw, by name and bytes.
#[derive(Default)]
struct FileRecorder(Mutex<Vec<(String, Vec<u8>)>>);

impl MergedSegmentWarmer for FileRecorder {
    fn warm(&self, dir: &dyn Directory, segment: &SegmentCommitInfo) -> Result<()> {
        let dot = format!("{}.", segment.segment_name);
        let gen = format!("{}_", segment.segment_name);
        let mut seen = self.0.lock().unwrap();
        for name in dir.list_all()? {
            if name.starts_with(&dot) || name.starts_with(&gen) {
                let bytes = dir.open(&name)?.to_vec();
                seen.push((name, bytes));
            }
        }
        Ok(())
    }
}

/// A warmed segment's files are the ones it is published and committed
/// with: segment replication pre-copies them, and a replica fails on a
/// file whose bytes change under the same name.
#[test]
fn the_warmed_files_are_never_rewritten() {
    for ratio in [0.0, 1.0] {
        let tmp = TempDir::new("warm-files");
        let dir = FsDirectory::open(&tmp);
        let mut w = writer(&dir);
        for i in 0..4 {
            w.add_document(doc(i)).unwrap();
            w.commit().unwrap();
        }
        let mut compound = crate::merge_policy::CompoundFileSettings::default();
        compound.set_no_cfs_ratio(ratio).unwrap();
        w.set_merge_compound_file_settings(Some(compound));
        let warmer = Arc::new(FileRecorder::default());
        w.set_merged_segment_warmer(Some(warmer.clone()));
        w.force_merge(1).unwrap();
        w.commit().unwrap();
        let seen = warmer.0.lock().unwrap();
        assert!(!seen.is_empty());
        for (name, bytes) in seen.iter() {
            assert_eq!(
                &dir.open(name).unwrap().to_vec(),
                bytes,
                "{name} (no_cfs_ratio {ratio})"
            );
        }
    }
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
