//! Port of `org.apache.lucene.store.NRTCachingDirectory`: a wrapper that
//! writes small new segments (small flushes and merges) into a
//! [`ByteBuffersDirectory`] instead of the delegate, so near-real-time
//! reopens do not pay for I/O on segments that will soon be merged away.
//! A cached file moves to the delegate when it is synced (a commit) or
//! when the directory is closed.
//!
//! Whether a write is cached depends on the size estimate Java reads off the
//! `IOContext` (`mergeInfo.estimatedMergeBytes` or
//! `flushInfo.estimatedSegmentSize`). This port's [`Directory::create_output`]
//! carries no context, which Java treats as "don't cache"; callers that know
//! the size use [`NrtCachingDirectory::create_output_with_estimate`].

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::byte_buffers_directory::ByteBuffersDirectory;
use crate::directory::{Directory, Input};
use crate::error::{Error, Result};
use crate::index_output::{FsIndexOutput, IndexOutput};
use crate::lock::{lock_ignoring_poison, Lock};

/// Port of `NRTCachingDirectory`.
pub struct NrtCachingDirectory<D: Directory> {
    inner: D,
    cache: ByteBuffersDirectory,
    /// `cacheSize`: bytes of every closed, still-cached file.
    cache_size: Arc<AtomicU64>,
    max_merge_size_bytes: u64,
    max_cached_bytes: u64,
    /// Java's `synchronized` methods: serialises the "is it cached?" check
    /// with the move out of the cache.
    guard: Mutex<()>,
    closed: AtomicBool,
}

/// `(long) (mb * 1024 * 1024)`: Java's saturating double-to-long cast (and 0
/// for NaN or a negative size, which disables caching).
fn mb_to_bytes(mb: f64) -> u64 {
    // `as` from f64 saturates and maps NaN to 0, exactly the cast Java does
    // (clamped at 0, since a negative budget caches nothing either way).
    (mb * 1024.0 * 1024.0) as u64
}

impl<D: Directory> NrtCachingDirectory<D> {
    /// `new NRTCachingDirectory(delegate, maxMergeSizeMB, maxCachedMB)`.
    pub fn new(delegate: D, max_merge_size_mb: f64, max_cached_mb: f64) -> Self {
        let cache_size = Arc::new(AtomicU64::new(0));
        let published = Arc::clone(&cache_size);
        Self {
            inner: delegate,
            cache: ByteBuffersDirectory::with_publish_hook(Arc::new(move |len| {
                published.fetch_add(len, Ordering::AcqRel);
            })),
            cache_size,
            max_merge_size_bytes: mb_to_bytes(max_merge_size_mb),
            max_cached_bytes: mb_to_bytes(max_cached_mb),
            guard: Mutex::new(()),
            closed: AtomicBool::new(false),
        }
    }

    /// `FilterDirectory.getDelegate()`.
    pub fn delegate(&self) -> &D {
        &self.inner
    }

    /// `ramBytesUsed()`: bytes currently held in the cache.
    pub fn ram_bytes_used(&self) -> u64 {
        self.cache_size.load(Ordering::Acquire)
    }

    /// `listCachedFiles()`.
    pub fn list_cached_files(&self) -> Vec<String> {
        // The in-memory directory's listing cannot fail.
        self.cache.list_all().unwrap_or_default()
    }

    /// `doCacheWrite(name, context)`: cache a file whose segment is expected
    /// to be at most `maxMergeSizeMB` and still fit in `maxCachedMB`. `None`
    /// (no flush or merge info) never caches.
    pub fn do_cache_write(&self, _name: &str, estimated_bytes: Option<u64>) -> bool {
        let Some(bytes) = estimated_bytes else {
            return false;
        };
        bytes <= self.max_merge_size_bytes
            && bytes.saturating_add(self.ram_bytes_used()) <= self.max_cached_bytes
    }

    /// `createOutput(name, context)` with the context's size estimate:
    /// cached when [`Self::do_cache_write`] says so, else the delegate's.
    pub fn create_output_with_estimate(
        &self,
        name: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        if self.do_cache_write(name, estimated_bytes) {
            self.cache.create_output(name)
        } else {
            self.inner.create_output(name)
        }
    }

    /// `createTempOutput(prefix, suffix, context)` with the context's size
    /// estimate: creates in the preferred directory, retrying until the name
    /// is free in the other one too, and removes the rejected attempts.
    pub fn create_temp_output_with_estimate(
        &self,
        prefix: &str,
        suffix: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        let (first, second): (&dyn Directory, &dyn Directory) =
            if self.do_cache_write(prefix, estimated_bytes) {
                (&self.cache, &self.inner)
            } else {
                (&self.inner, &self.cache)
            };
        let mut to_delete = Vec::new();
        let result = loop {
            let out = match first.create_temp_output(prefix, suffix) {
                Ok(out) => out,
                Err(e) => break Err(e),
            };
            let name = out.name().to_string();
            match slow_file_exists(second, &name) {
                Ok(true) => {
                    to_delete.push(name);
                    if let Err(e) = out.close() {
                        break Err(e);
                    }
                }
                Ok(false) => break Ok(out),
                Err(e) => {
                    to_delete.push(name);
                    drop(out);
                    break Err(e);
                }
            }
        };
        match result {
            // `IOUtils.deleteFiles(first, toDelete)`.
            Ok(out) => {
                for name in &to_delete {
                    first.delete_file(name)?;
                }
                Ok(out)
            }
            // `IOUtils.deleteFilesIgnoringExceptions(first, toDelete)`.
            Err(e) => {
                for name in &to_delete {
                    let _ = first.delete_file(name);
                }
                Err(e)
            }
        }
    }

    /// `unCache(fileName)`: moves a cached file to the delegate.
    fn un_cache(&self, file_name: &str) -> Result<()> {
        let _guard = lock_ignoring_poison(&self.guard);
        if !self.cache.file_exists(file_name) {
            // Another thread beat us.
            return Ok(());
        }
        debug_assert!(
            !slow_file_exists(&self.inner, file_name).unwrap_or(false),
            "fileName={file_name} exists both in cache and in delegate"
        );
        self.inner.copy_from(&self.cache, file_name, file_name)?;
        if let Some(len) = self.cache.remove_entry(file_name) {
            self.cache_size.fetch_sub(len, Ordering::AcqRel);
        }
        Ok(())
    }

    /// `close()`: moves every cached file to the delegate. Java's `close`
    /// also closes the delegate; here the delegate is dropped with `self`.
    /// Runs once; also run (best effort) on drop.
    pub fn close(&self) -> Result<()> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        for file_name in self.list_cached_files() {
            self.un_cache(&file_name)?;
        }
        Ok(())
    }
}

impl<D: Directory> Drop for NrtCachingDirectory<D> {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// `NRTCachingDirectory.slowFileExists(dir, fileName)`.
fn slow_file_exists(dir: &dyn Directory, file_name: &str) -> Result<bool> {
    match dir.file_length(file_name) {
        Ok(_) => Ok(true),
        Err(e) if e.is_no_such_file() => Ok(false),
        Err(e) => Err(e),
    }
}

impl<D: Directory> Directory for NrtCachingDirectory<D> {
    fn list_all(&self) -> Result<Vec<String>> {
        let _guard = lock_ignoring_poison(&self.guard);
        let mut files: BTreeSet<String> = self.cache.list_all()?.into_iter().collect();
        files.extend(self.inner.list_all()?);
        Ok(files.into_iter().collect())
    }

    fn open(&self, name: &str) -> Result<Input> {
        let _guard = lock_ignoring_poison(&self.guard);
        if self.cache.file_exists(name) {
            self.cache.open(name)
        } else {
            self.inner.open(name)
        }
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        let _guard = lock_ignoring_poison(&self.guard);
        if self.cache.file_exists(name) {
            self.cache.file_length(name)
        } else {
            self.inner.file_length(name)
        }
    }

    /// No size estimate: never cached (Java's `IOContext.DEFAULT`).
    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.create_output_with_estimate(name, None)
    }

    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        self.create_temp_output_with_estimate(prefix, suffix, None)
    }

    /// Moves every named file out of the cache, then syncs the delegate.
    fn sync(&self, names: &[String]) -> Result<()> {
        for name in names {
            self.un_cache(name)?;
        }
        self.inner.sync(names)
    }

    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        self.un_cache(source)?;
        if self.cache.file_exists(dest) {
            return Err(Error::IllegalArgument(format!(
                "target file {dest} already exists"
            )));
        }
        self.inner.rename(source, dest)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        let _guard = lock_ignoring_poison(&self.guard);
        match self.cache.remove_entry(name) {
            Some(len) => {
                self.cache_size.fetch_sub(len, Ordering::AcqRel);
                Ok(())
            }
            None => self.inner.delete_file(name),
        }
    }

    fn sync_meta_data(&self) -> Result<()> {
        self.inner.sync_meta_data()
    }

    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.inner.obtain_lock(name)
    }

    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        self.inner.pending_deletions()
    }

    fn fs_directory_path(&self) -> Option<&Path> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_output::DataOutput;
    use crate::FsDirectory;
    use lucene_util::test_support::TempDir;

    fn write_sized(
        dir: &NrtCachingDirectory<impl Directory>,
        name: &str,
        n: usize,
        est: Option<u64>,
    ) {
        let mut out = dir.create_output_with_estimate(name, est).unwrap();
        out.write_bytes(&vec![7u8; n]);
        out.close().unwrap();
    }

    /// Mirrors `TestNRTCachingDirectory.testNRTAndCommit`'s intent: small
    /// flushed files are cached (invisible on disk) until a sync moves them,
    /// and the cache size tracks exactly what is cached.
    #[test]
    fn small_writes_are_cached_until_synced() {
        let root = TempDir::new("nrt-cache");
        let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
        write_sized(&dir, "_0.fdt", 100, Some(1000));
        write_sized(&dir, "_0.fdx", 50, Some(1000));
        assert_eq!(dir.list_cached_files(), vec!["_0.fdt", "_0.fdx"]);
        assert_eq!(dir.ram_bytes_used(), 150);
        assert!(!root.join("_0.fdt").exists());
        assert_eq!(dir.list_all().unwrap(), vec!["_0.fdt", "_0.fdx"]);
        assert_eq!(dir.file_length("_0.fdt").unwrap(), 100);
        assert_eq!(dir.open("_0.fdx").unwrap().len(), 50);

        dir.sync(&["_0.fdt".to_string()]).unwrap();
        assert_eq!(dir.list_cached_files(), vec!["_0.fdx"]);
        assert_eq!(dir.ram_bytes_used(), 50);
        assert_eq!(std::fs::read(root.join("_0.fdt")).unwrap(), vec![7u8; 100]);
        assert_eq!(dir.file_length("_0.fdt").unwrap(), 100);
        assert_eq!(dir.open("_0.fdt").unwrap().len(), 100);
        // Listing still shows each file once.
        assert_eq!(dir.list_all().unwrap(), vec!["_0.fdt", "_0.fdx"]);
    }

    #[test]
    fn large_or_unestimated_writes_go_to_the_delegate() {
        let root = TempDir::new("nrt-bypass");
        let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
        // No context: never cached.
        let mut out = dir.create_output("_1.fdt").unwrap();
        out.write_bytes(b"x");
        out.close().unwrap();
        assert!(root.join("_1.fdt").exists());
        // A segment too large for maxMergeSizeMB.
        write_sized(&dir, "_2.fdt", 1, Some(2 * 1024 * 1024));
        assert!(root.join("_2.fdt").exists());
        // Would overflow maxCachedMB.
        write_sized(&dir, "_3.fdt", 1024 * 1024, Some(1024 * 1024));
        assert_eq!(dir.ram_bytes_used(), 1024 * 1024);
        assert!(!dir.do_cache_write("_4", Some(1024 * 1024 + 1)));
        write_sized(&dir, "_4.fdt", 1, Some(1024 * 1024 + 1));
        assert!(root.join("_4.fdt").exists());
        assert_eq!(dir.list_cached_files(), vec!["_3.fdt"]);
        assert!(!dir.do_cache_write("x", None));
    }

    #[test]
    fn delete_rename_and_close_move_or_drop_cached_files() {
        let root = TempDir::new("nrt-lifecycle");
        let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
        write_sized(&dir, "a", 10, Some(10));
        write_sized(&dir, "b", 20, Some(10));
        write_sized(&dir, "c", 30, Some(10));
        dir.delete_file("a").unwrap();
        assert_eq!(dir.ram_bytes_used(), 50);
        // Rename moves the source to the delegate first.
        dir.rename("b", "b2").unwrap();
        assert_eq!(std::fs::read(root.join("b2")).unwrap().len(), 20);
        assert_eq!(dir.ram_bytes_used(), 30);
        // A rename onto a cached name is refused.
        write_sized(&dir, "d", 1, None);
        let err = dir.rename("d", "c").unwrap_err();
        assert!(matches!(err, Error::IllegalArgument(_)), "{err}");
        // Deleting a file that is not cached reaches the delegate.
        dir.delete_file("d").unwrap();
        assert!(!root.join("d").exists());
        dir.sync_meta_data().unwrap();
        assert!(dir.pending_deletions().unwrap().is_empty());
        assert!(dir.fs_directory_path().is_none());
        assert_eq!(dir.delegate().directory(), &*root);

        dir.close().unwrap();
        dir.close().unwrap();
        assert!(dir.list_cached_files().is_empty());
        assert_eq!(dir.ram_bytes_used(), 0);
        assert_eq!(std::fs::read(root.join("c")).unwrap().len(), 30);
    }

    #[test]
    fn drop_flushes_the_cache_to_the_delegate() {
        let root = TempDir::new("nrt-drop");
        {
            let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
            write_sized(&dir, "_0.si", 5, Some(5));
            assert!(!root.join("_0.si").exists());
        }
        assert_eq!(std::fs::read(root.join("_0.si")).unwrap().len(), 5);
    }

    #[test]
    fn a_deleted_open_output_does_not_count_against_the_cache() {
        let root = TempDir::new("nrt-deleted-open");
        let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
        let mut out = dir.create_output_with_estimate("x", Some(1)).unwrap();
        out.write_bytes(b"12345");
        dir.delete_file("x").unwrap();
        out.close().unwrap();
        assert_eq!(dir.ram_bytes_used(), 0);
    }

    #[test]
    fn temp_outputs_avoid_names_taken_in_the_other_directory() {
        let root = TempDir::new("nrt-temp");
        let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
        // The cache holds the name the delegate will try first.
        write_sized(&dir, "_0_sort_0.tmp", 1, Some(1));
        let out = dir.create_temp_output("_0", "sort").unwrap();
        assert_eq!(out.name(), "_0_sort_1.tmp");
        out.close().unwrap();
        // The rejected attempt was removed from the delegate.
        assert!(!root.join("_0_sort_0.tmp").exists());
        assert!(root.join("_0_sort_1.tmp").exists());

        // Cached temp output: the delegate holds the cache's first name.
        std::fs::write(root.join("_1_s_0.tmp"), b"").unwrap();
        let out = dir
            .create_temp_output_with_estimate("_1", "s", Some(1))
            .unwrap();
        assert_eq!(out.name(), "_1_s_1.tmp");
        out.close().unwrap();
        assert!(dir.list_cached_files().contains(&"_1_s_1.tmp".to_string()));
        assert!(!dir.list_cached_files().contains(&"_1_s_0.tmp".to_string()));
    }

    #[test]
    fn temp_output_errors_propagate() {
        let root = TempDir::new("nrt-temp-err");
        let missing = root.join("missing");
        let dir = NrtCachingDirectory::new(FsDirectory::open(&missing), 1.0, 2.0);
        assert!(dir.create_temp_output("_0", "x").is_err());
    }

    #[test]
    fn locks_come_from_the_delegate() {
        let root = TempDir::new("nrt-lock");
        let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
        let lock = dir.obtain_lock("write.lock").unwrap();
        assert!(root.join("write.lock").exists());
        assert!(matches!(
            FsDirectory::open(&root).obtain_lock("write.lock"),
            Err(Error::LockObtainFailed(_))
        ));
        lock.close().unwrap();
    }

    #[test]
    fn mb_to_bytes_is_javas_cast() {
        assert_eq!(mb_to_bytes(1.0), 1024 * 1024);
        assert_eq!(mb_to_bytes(-1.0), 0);
        assert_eq!(mb_to_bytes(f64::NAN), 0);
        assert_eq!(mb_to_bytes(f64::INFINITY), u64::MAX);
    }
}
