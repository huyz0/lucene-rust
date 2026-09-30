//! Port of `org.apache.lucene.store.ByteBuffersDirectory`: a [`Directory`]
//! held entirely in memory -- Lucene's replacement for `RAMDirectory`, and
//! the cache behind [`crate::NrtCachingDirectory`].
//!
//! A file is listed from the moment its output is created, reads as empty
//! (`fileLength` 0) and cannot be opened until that output is closed, and is
//! written exactly once, as in Java. Opened files share the stored bytes
//! ([`Input::Shared`]): Java's `IndexInput.clone()`.
//!
//! Java's `outputToInput` choice between one buffer and many is a
//! read-performance knob; a closed file here is always one contiguous
//! `Arc<[u8]>`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::directory::{temp_file_name, BaseDirectory, Directory, Input};
use crate::error::{Error, Result};
use crate::index_output::FsIndexOutput;
use crate::lock::{lock_ignoring_poison, Lock, LockFactory, SingleInstanceLockFactory};

/// Called with a file's length when its output closes while the file is
/// still listed -- how [`crate::NrtCachingDirectory`] accounts its cache
/// size (its `outputToInput` lambda's `cacheSize.addAndGet`).
pub(crate) type OnPublish = Arc<dyn Fn(u64) + Send + Sync>;

/// `ByteBuffersDirectory.FileEntry`.
#[derive(Debug, Default)]
struct FileEntry {
    /// `null` until the output is closed.
    content: Mutex<Option<Arc<[u8]>>>,
}

impl FileEntry {
    fn content(&self) -> MutexGuard<'_, Option<Arc<[u8]>>> {
        lock_ignoring_poison(&self.content)
    }

    /// `FileEntry.length()`: 0 until the output is closed.
    fn length(&self) -> u64 {
        self.content().as_ref().map_or(0, |c| c.len() as u64)
    }
}

type Files = Arc<Mutex<BTreeMap<String, Arc<FileEntry>>>>;

/// Port of `ByteBuffersDirectory`.
pub struct ByteBuffersDirectory {
    base: BaseDirectory,
    files: Files,
    /// The `tempFileName` function's counter.
    temp_counter: AtomicU64,
    on_publish: Option<OnPublish>,
}

impl std::fmt::Debug for ByteBuffersDirectory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ByteBuffersDirectory lockFactory={:?}",
            self.base.lock_factory()
        )
    }
}

impl Default for ByteBuffersDirectory {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteBuffersDirectory {
    /// `new ByteBuffersDirectory()`: locks with a fresh
    /// [`SingleInstanceLockFactory`].
    pub fn new() -> Self {
        Self::with_lock_factory(Arc::new(SingleInstanceLockFactory::new()))
    }

    /// `new ByteBuffersDirectory(lockFactory)`.
    pub fn with_lock_factory(lock_factory: Arc<dyn LockFactory>) -> Self {
        Self {
            base: BaseDirectory::new(lock_factory),
            files: Arc::default(),
            temp_counter: AtomicU64::new(0),
            on_publish: None,
        }
    }

    /// The cache [`crate::NrtCachingDirectory`] builds: publishes report
    /// their size to `on_publish`.
    pub(crate) fn with_publish_hook(on_publish: OnPublish) -> Self {
        let mut dir = Self::new();
        dir.on_publish = Some(on_publish);
        dir
    }

    fn files(&self) -> MutexGuard<'_, BTreeMap<String, Arc<FileEntry>>> {
        lock_ignoring_poison(&self.files)
    }

    /// `ByteBuffersDirectory.fileExists(name)`.
    pub fn file_exists(&self, name: &str) -> bool {
        self.files().contains_key(name)
    }

    /// Removes `name` and returns the length it had, in one step -- so a
    /// cache's size accounting cannot race a concurrent publish between
    /// `fileLength` and `deleteFile` (the window Java's
    /// `NRTCachingDirectory.deleteFile` leaves open).
    pub(crate) fn remove_entry(&self, name: &str) -> Option<u64> {
        self.files().remove(name).map(|e| e.length())
    }

    /// `FileEntry.createOutput`, registering `entry` under `name` first.
    fn output_for(&self, name: &str, entry: Arc<FileEntry>) -> FsIndexOutput {
        let files = Arc::clone(&self.files);
        let on_publish = self.on_publish.clone();
        let file_name = name.to_string();
        FsIndexOutput::in_memory(
            name,
            Box::new(move |bytes: Vec<u8>| {
                let len = bytes.len() as u64;
                // Under the directory lock, so "still listed" and the size
                // report are one atomic step with respect to deletes.
                let files = lock_ignoring_poison(&files);
                *entry.content() = Some(Arc::from(bytes));
                let listed = files
                    .get(&file_name)
                    .is_some_and(|e| Arc::ptr_eq(e, &entry));
                if listed {
                    if let Some(hook) = &on_publish {
                        hook(len);
                    }
                }
            }),
        )
    }
}

impl Directory for ByteBuffersDirectory {
    fn list_all(&self) -> Result<Vec<String>> {
        Ok(self.files().keys().cloned().collect())
    }

    fn open(&self, name: &str) -> Result<Input> {
        let entry = self
            .files()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::no_such_file(name))?;
        let content = entry.content().clone();
        match content {
            Some(bytes) => Ok(Input::Shared(bytes)),
            None => Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("Can't open a file still open for writing: {name}"),
            ))),
        }
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        self.files()
            .get(name)
            .map(|e| e.length())
            .ok_or_else(|| Error::no_such_file(name))
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        let entry = Arc::new(FileEntry::default());
        {
            let mut files = self.files();
            if files.contains_key(name) {
                return Err(Error::file_already_exists(format!(
                    "File already exists: {name}"
                )));
            }
            files.insert(name.to_string(), Arc::clone(&entry));
        }
        Ok(self.output_for(name, entry))
    }

    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        loop {
            let counter = self.temp_counter.fetch_add(1, Ordering::Relaxed);
            let name = temp_file_name(prefix, suffix, counter);
            let entry = Arc::new(FileEntry::default());
            let inserted = {
                let mut files = self.files();
                if files.contains_key(&name) {
                    false
                } else {
                    files.insert(name.clone(), Arc::clone(&entry));
                    true
                }
            };
            if inserted {
                return Ok(self.output_for(&name, entry));
            }
        }
    }

    /// A no-op: there is nothing to make durable.
    fn sync(&self, _names: &[String]) -> Result<()> {
        Ok(())
    }

    /// Moves the entry -- including one whose output is still open, which
    /// then publishes under the new name. Refuses to replace an existing
    /// `dest` (`FileAlreadyExistsException`), unlike a filesystem rename.
    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        let mut files = self.files();
        if !files.contains_key(source) {
            return Err(Error::no_such_file(source));
        }
        if files.contains_key(dest) {
            return Err(Error::file_already_exists(dest));
        }
        if let Some(entry) = files.remove(source) {
            files.insert(dest.to_string(), entry);
        }
        Ok(())
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        match self.files().remove(name) {
            Some(_) => Ok(()),
            None => Err(Error::no_such_file(name)),
        }
    }

    fn sync_meta_data(&self) -> Result<()> {
        Ok(())
    }

    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.base.obtain_lock(self, name)
    }

    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        Ok(BTreeSet::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_output::DataOutput;
    use crate::index_output::IndexOutput;
    use crate::lock::NoLockFactory;

    fn write(dir: &dyn Directory, name: &str, bytes: &[u8]) -> u64 {
        let mut out = dir.create_output(name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap()
    }

    #[test]
    fn written_files_read_back_and_list_sorted() {
        let dir = ByteBuffersDirectory::new();
        let checksum = write(&dir, "b.dat", b"hello");
        assert_eq!(checksum, crc32fast::hash(b"hello") as u64);
        write(&dir, "a.dat", b"");
        assert_eq!(dir.list_all().unwrap(), vec!["a.dat", "b.dat"]);
        let input = dir.open("b.dat").unwrap();
        assert_eq!(&*input, b"hello");
        assert_eq!(format!("{input:?}"), "Input::Shared(5 bytes)");
        assert_eq!(dir.file_length("b.dat").unwrap(), 5);
        assert_eq!(dir.file_length("a.dat").unwrap(), 0);
        assert!(dir.file_exists("a.dat"));
        assert!(!dir.file_exists("c.dat"));
        assert!(dir.pending_deletions().unwrap().is_empty());
        dir.sync(&["b.dat".to_string()]).unwrap();
        dir.sync_meta_data().unwrap();
        assert!(dir.fs_directory_path().is_none());
        assert!(format!("{dir:?}").contains("SingleInstanceLockFactory"));
    }

    #[test]
    fn an_open_output_is_listed_empty_and_unreadable() {
        let dir = ByteBuffersDirectory::default();
        let mut out = dir.create_output("_0.fdt").unwrap();
        out.write_bytes(b"abc");
        assert_eq!(out.file_pointer(), 3);
        assert!(out.path().is_none());
        assert_eq!(dir.list_all().unwrap(), vec!["_0.fdt"]);
        assert_eq!(dir.file_length("_0.fdt").unwrap(), 0);
        let err = dir.open("_0.fdt").unwrap_err();
        assert!(
            matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::PermissionDenied),
            "{err}"
        );
        out.close().unwrap();
        assert_eq!(&*dir.open("_0.fdt").unwrap(), b"abc");
    }

    #[test]
    fn a_file_is_written_once() {
        let dir = ByteBuffersDirectory::new();
        write(&dir, "x", b"1");
        let err = dir.create_output("x").unwrap_err();
        assert!(
            matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::AlreadyExists),
            "{err}"
        );
    }

    #[test]
    fn missing_files_are_no_such_file() {
        let dir = ByteBuffersDirectory::new();
        assert!(dir.open("nope").unwrap_err().is_no_such_file());
        assert!(dir.file_length("nope").unwrap_err().is_no_such_file());
        assert!(dir.delete_file("nope").unwrap_err().is_no_such_file());
        assert!(dir.rename("nope", "x").unwrap_err().is_no_such_file());
    }

    #[test]
    fn rename_moves_and_refuses_to_overwrite() {
        let dir = ByteBuffersDirectory::new();
        write(&dir, "pending_segments_1", b"commit");
        write(&dir, "other", b"o");
        dir.rename("pending_segments_1", "segments_1").unwrap();
        assert_eq!(dir.list_all().unwrap(), vec!["other", "segments_1"]);
        assert_eq!(&*dir.open("segments_1").unwrap(), b"commit");
        let err = dir.rename("other", "segments_1").unwrap_err();
        assert!(
            matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::AlreadyExists),
            "{err}"
        );
        // An output renamed while open publishes under its new name.
        let mut out = dir.create_output("tmp").unwrap();
        dir.rename("tmp", "final").unwrap();
        out.write_bytes(b"late");
        out.close().unwrap();
        assert_eq!(&*dir.open("final").unwrap(), b"late");
    }

    #[test]
    fn delete_removes_and_a_later_close_is_harmless() {
        let dir = ByteBuffersDirectory::new();
        write(&dir, "a", b"1");
        dir.delete_file("a").unwrap();
        assert!(dir.list_all().unwrap().is_empty());
        let out = dir.create_output("b").unwrap();
        dir.delete_file("b").unwrap();
        out.close().unwrap();
        assert!(!dir.file_exists("b"));
        assert_eq!(dir.remove_entry("b"), None);
    }

    #[test]
    fn temp_outputs_get_fresh_names() {
        let dir = ByteBuffersDirectory::new();
        // Occupy the first name the counter will produce.
        write(&dir, "_0_sort_0.tmp", b"taken");
        let a = dir.create_temp_output("_0", "sort").unwrap();
        let b = dir.create_temp_output("_0", "sort").unwrap();
        assert_eq!(a.name(), "_0_sort_1.tmp");
        assert_eq!(b.name(), "_0_sort_2.tmp");
        a.close().unwrap();
        b.close().unwrap();
        assert_eq!(dir.list_all().unwrap().len(), 3);
    }

    #[test]
    fn copy_from_copies_between_directories() {
        let src = ByteBuffersDirectory::new();
        let dst = ByteBuffersDirectory::new();
        write(&src, "a", b"payload");
        dst.copy_from(&src, "a", "b").unwrap();
        assert_eq!(&*dst.open("b").unwrap(), b"payload");
        // A missing source fails and leaves no partial destination.
        assert!(dst.copy_from(&src, "missing", "c").is_err());
        assert!(!dst.file_exists("c"));
    }

    #[test]
    fn default_lock_factory_is_per_directory() {
        let a = ByteBuffersDirectory::new();
        let b = ByteBuffersDirectory::new();
        let held = a.obtain_lock("write.lock").unwrap();
        assert!(matches!(
            a.obtain_lock("write.lock"),
            Err(Error::LockObtainFailed(_))
        ));
        let _other = b.obtain_lock("write.lock").unwrap();
        held.close().unwrap();
        a.obtain_lock("write.lock").unwrap();

        let none = ByteBuffersDirectory::with_lock_factory(Arc::new(NoLockFactory));
        let _x = none.obtain_lock("write.lock").unwrap();
        let _y = none.obtain_lock("write.lock").unwrap();
    }

    #[test]
    fn publish_hook_sees_only_files_still_listed() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let hook_seen = Arc::clone(&seen);
        let dir = ByteBuffersDirectory::with_publish_hook(Arc::new(move |len| {
            hook_seen.lock().unwrap().push(len)
        }));
        write(&dir, "kept", b"1234");
        let out = dir.create_output("dropped").unwrap();
        dir.delete_file("dropped").unwrap();
        out.close().unwrap();
        assert_eq!(*seen.lock().unwrap(), vec![4]);
        assert_eq!(dir.remove_entry("kept"), Some(4));
    }
}
