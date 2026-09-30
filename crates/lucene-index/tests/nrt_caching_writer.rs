//! `IndexWriter` over an `NRTCachingDirectory`: the writer passes its flush
//! and merge size estimates (Java's `IOContext` `FlushInfo`/`MergeInfo`), so a
//! small flushed segment is written to the in-memory cache and moves to disk
//! when it is synced.
//!
//! Caching has no on-disk trace once a file is synced, so there is no Java
//! fixture: what is checked is how each file reached the disk, and which
//! estimate every output of a flush and a merge carries.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Mutex;

use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::directory::{Directory, Input};
use lucene_store::{FsDirectory, FsIndexOutput, Lock, NrtCachingDirectory};
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn doc(i: usize) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(format!("document number {i}")),
        }],
    }
}

fn open<'d>(dir: &'d dyn Directory) -> IndexWriter<'d> {
    IndexWriter::open(dir, vec![FieldInfo::new("body", 0)], "Lucene104", VERSION).unwrap()
}

/// A small flushed segment is written to the cache and reaches the disk by
/// the cache's `unCache` copy when the segment is synced (this port's flush
/// fsyncs its segment, where Java's waits for the commit); over the cache's
/// `maxMergeSizeMB` it is written to the disk directly.
#[test]
fn a_small_flush_is_written_to_the_cache() {
    for (max_merge_mb, cached) in [(5.0, true), (0.000_001, false)] {
        let tmp = TempDir::new("nrt-writer-flush");
        let fs = FsDirectory::open(tmp.path());
        let disk = Recording::new(&fs);
        let nrt = NrtCachingDirectory::new(&disk, max_merge_mb, 60.0);
        let mut w = open(&nrt);
        for i in 0..10 {
            w.add_document(doc(i)).unwrap();
        }
        w.flush().unwrap();
        let fdt = "_0.fdt".to_string();
        assert!(tmp.path().join(&fdt).exists(), "synced to disk");
        assert!(nrt.list_cached_files().is_empty());
        let copied = disk.copied.lock().unwrap().contains(&fdt);
        let created = disk.seen.lock().unwrap().iter().any(|(f, _)| *f == fdt);
        assert_eq!((copied, created), (cached, !cached));
        w.commit().unwrap();
    }
}

/// Records the estimate every output is created with.
struct Recording<'a> {
    inner: &'a dyn Directory,
    seen: Mutex<Vec<(String, Option<u64>)>>,
    /// Files that arrived by `copy_from` (an `NRTCachingDirectory` moving a
    /// cached file out).
    copied: Mutex<Vec<String>>,
}

impl<'a> Recording<'a> {
    fn new(inner: &'a dyn Directory) -> Self {
        Recording {
            inner,
            seen: Mutex::new(Vec::new()),
            copied: Mutex::new(Vec::new()),
        }
    }
}

impl Directory for Recording<'_> {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<Input> {
        self.inner.open(name)
    }
    fn file_length(&self, name: &str) -> lucene_store::Result<u64> {
        self.inner.file_length(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<FsIndexOutput> {
        self.create_output_with_estimate(name, None)
    }
    fn create_output_with_estimate(
        &self,
        name: &str,
        estimated_bytes: Option<u64>,
    ) -> lucene_store::Result<FsIndexOutput> {
        self.seen
            .lock()
            .unwrap()
            .push((name.to_string(), estimated_bytes));
        self.inner.create_output(name)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn Lock>> {
        self.inner.obtain_lock(name)
    }
    fn copy_from(&self, from: &dyn Directory, src: &str, dest: &str) -> lucene_store::Result<()> {
        self.copied.lock().unwrap().push(dest.to_string());
        self.inner.copy_from(from, src, dest)
    }
}

/// A merge's outputs carry `estimatedMergeBytes`: the sources' sizes
/// (`SegmentCommitInfo.sizeInBytes()`, no deletions here), throttled or not;
/// a flush's carry a positive estimate; a commit's own files none.
#[test]
fn flush_and_merge_outputs_carry_their_estimates() {
    for throttle in [None, Some(1024.0)] {
        let tmp = TempDir::new("nrt-writer-merge");
        let fs = FsDirectory::open(tmp.path());
        let dir = Recording::new(&fs);
        let mut w = open(&dir);
        for round in 0..2 {
            for i in 0..10 {
                w.add_document(doc(round * 10 + i)).unwrap();
            }
            w.commit().unwrap();
        }
        let flushed = std::mem::take(&mut *dir.seen.lock().unwrap());
        assert!(flushed
            .iter()
            .any(|(f, e)| f.ends_with(".fdt") && e.is_some_and(|e| e > 0)));
        assert!(flushed
            .iter()
            .filter(|(f, _)| f.contains("segments_"))
            .all(|(_, e)| e.is_none()));
        let sources: u64 = fs
            .list_all()
            .unwrap()
            .iter()
            .filter(|f| f.starts_with("_0") || f.starts_with("_1"))
            .map(|f| fs.file_length(f).unwrap())
            .sum();
        w.set_merge_mb_per_sec(throttle).unwrap();
        w.force_merge(1).unwrap();
        let merged = std::mem::take(&mut *dir.seen.lock().unwrap());
        let fdt: Vec<&(String, Option<u64>)> =
            merged.iter().filter(|(f, _)| f.ends_with(".fdt")).collect();
        assert_eq!(fdt.len(), 1, "{merged:?}");
        assert_eq!(fdt[0].1, Some(sources), "estimatedMergeBytes");
    }
}
