//! A [`Directory`] over an object ("blob") store: the adapter that lets an
//! index live in S3, GCS, Azure Blob, a key-value database or any other
//! store that keeps whole named objects, by implementing the handful of
//! operations in [`BlobStore`].
//!
//! Lucene's own answer is to subclass `Directory` (or `FilterDirectory`),
//! which this port supports too: [`Directory`] is an open trait,
//! [`FsIndexOutput::from_sink`] lets any directory hand out its own output,
//! and [`Input::external`] its own input. [`BlobDirectory`] packages the
//! parts every object store shares, so an implementation only maps names to
//! objects:
//!
//! - A file is one object, written once. [`BlobStore::writer`] streams it
//!   (a multipart upload); the default buffers it and calls
//!   [`BlobStore::put`] when the output closes, so the object only appears
//!   once it is complete -- the "no half-written file under a final name"
//!   property `FSDirectory` gets from the OS.
//! - [`Directory::rename`] is how a commit is published
//!   (`pending_segments_N` to `segments_N`). Object stores have no atomic
//!   rename; the default copies then deletes. That is safe for Lucene's use:
//!   a reader that lists `segments_N` before the copy completes cannot see
//!   it (an object is visible only once fully written), and a crash between
//!   copy and delete leaves an extra `pending_segments_N`, which the next
//!   writer deletes as an unreferenced file.
//! - [`Directory::sync`] defaults to a no-op: an object is durable once
//!   `put` returns.
//! - Locking defaults to [`SingleInstanceLockFactory`], which excludes
//!   writers in this process only. A store shared by several writer
//!   processes must pass its own [`LockFactory`] (a conditional put, a
//!   lease) through [`BlobDirectory::with_lock_factory`].
//!
//! [`BlobDirectory`] reads whole objects ([`BlobStore::get`]). For large
//! shards, [`crate::CachedBlobDirectory`] reads the same store by range
//! ([`BlobStore::get_range`]) through a page cache on local disk and in
//! memory, fetching only the pages a reader touches.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::directory::{temp_file_name, BaseDirectory, Directory, Input};
use crate::error::{Error, Result};
use crate::index_output::{FsIndexOutput, OutputSink};
use crate::lock::{lock_ignoring_poison, Lock, LockFactory, SingleInstanceLockFactory};

/// The operations [`BlobDirectory`] needs from an object store. Names are
/// Lucene file names (`_0.cfs`, `segments_3`); mapping them under a bucket
/// prefix is the implementation's business.
///
/// A missing object must be reported as `Error::Io` of kind
/// `ErrorKind::NotFound` (see [`not_found`]): Lucene's readers and its
/// file deleter tell "no such file" apart from a real failure.
pub trait BlobStore: Send + Sync {
    /// Every object name, in any order.
    fn list(&self) -> Result<Vec<String>>;

    /// A whole object's bytes.
    fn get(&self, name: &str) -> Result<Input>;

    /// `len` bytes of object `name` from `offset`, fewer where the object
    /// ends first (a ranged `GET`). [`crate::CachedBlobDirectory`] reads
    /// through this alone. The default downloads the whole object; a real
    /// store overrides it.
    fn get_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let bytes = self.get(name)?;
        let start = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(bytes.len());
        let end = start.saturating_add(len).min(bytes.len());
        Ok(bytes[start..end].to_vec())
    }

    /// Every object name with its length, when the store's listing carries
    /// lengths (S3's `ListObjectsV2`, Azure's and GCS's lists do): a reader
    /// then opens files without asking for each length. `None`, the
    /// default, when it does not.
    fn list_with_sizes(&self) -> Result<Option<Vec<(String, u64)>>> {
        Ok(None)
    }

    /// An object's length. The default downloads it; a real store answers
    /// from metadata (`HEAD`).
    fn size(&self, name: &str) -> Result<u64> {
        Ok(self.get(name)?.len() as u64)
    }

    /// Whether an object exists, as [`BlobStore::size`] answers.
    fn exists(&self, name: &str) -> Result<bool> {
        match self.size(name) {
            Ok(_) => Ok(true),
            Err(e) if e.is_no_such_file() => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Stores a whole object, visible to readers only once complete.
    fn put(&self, name: &str, bytes: Vec<u8>) -> Result<()>;

    /// A streaming writer for a new object. The default buffers the bytes
    /// and [`BlobStore::put`]s them when the output closes; a store with
    /// multipart uploads overrides it so a large segment file is not held
    /// in memory.
    fn writer(self: Arc<Self>, name: &str) -> Result<Box<dyn OutputSink>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(BufferedPut {
            store: self,
            name: name.to_string(),
            bytes: Vec::new(),
        }))
    }

    /// Deletes an object. A missing one is reported as not found.
    fn delete(&self, name: &str) -> Result<()>;

    /// Copies an object (server side, where the store can). The default
    /// downloads and re-uploads it.
    fn copy(&self, source: &str, dest: &str) -> Result<()> {
        let bytes = self.get(source)?.to_vec();
        self.put(dest, bytes)
    }

    /// Makes objects durable. The default does nothing: most stores are
    /// durable once a put returns.
    fn sync(&self, names: &[String]) -> Result<()> {
        let _ = names;
        Ok(())
    }
}

/// `NoSuchFileException(name)` in the shape [`BlobStore`] implementations
/// report a missing object.
pub fn not_found(name: &str) -> Error {
    Error::no_such_file(name)
}

/// [`BlobStore::writer`]'s default: buffer, then one `put` at close.
struct BufferedPut<S: BlobStore> {
    store: Arc<S>,
    name: String,
    bytes: Vec<u8>,
}

impl<S: BlobStore> OutputSink for BufferedPut<S> {
    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn finish(self: Box<Self>) -> std::io::Result<()> {
        let this = *self;
        this.store.put(&this.name, this.bytes).map_err(into_io)
    }
}

/// An [`Error`] as the `std::io::Error` an [`OutputSink`] returns, keeping
/// an I/O error's kind.
pub fn into_io(e: Error) -> std::io::Error {
    match e {
        Error::Io(io) => io,
        other => std::io::Error::other(other.to_string()),
    }
}

/// A [`Directory`] over a [`BlobStore`]. See the module docs for the
/// semantics each Lucene operation gets.
pub struct BlobDirectory<S: BlobStore + 'static> {
    store: Arc<S>,
    base: BaseDirectory,
    temp_counter: AtomicU64,
}

impl<S: BlobStore + 'static> std::fmt::Debug for BlobDirectory<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BlobDirectory(lockFactory={:?})",
            self.base.lock_factory()
        )
    }
}

impl<S: BlobStore + 'static> BlobDirectory<S> {
    /// A directory over `store`, locking with a [`SingleInstanceLockFactory`].
    pub fn new(store: S) -> Self {
        Self::with_lock_factory(store, Arc::new(SingleInstanceLockFactory::new()))
    }

    /// A directory over `store` whose `write.lock` is `lock_factory`'s.
    pub fn with_lock_factory(store: S, lock_factory: Arc<dyn LockFactory>) -> Self {
        Self {
            store: Arc::new(store),
            base: BaseDirectory::new(lock_factory),
            temp_counter: AtomicU64::new(0),
        }
    }

    /// A directory over a store already shared with another owner
    /// ([`crate::CachedBlobDirectory`]'s page fetches).
    pub(crate) fn from_shared(store: Arc<S>, lock_factory: Arc<dyn LockFactory>) -> Self {
        Self {
            store,
            base: BaseDirectory::new(lock_factory),
            temp_counter: AtomicU64::new(0),
        }
    }

    /// The store underneath.
    pub fn store(&self) -> &S {
        &self.store
    }
}

impl<S: BlobStore + 'static> Directory for BlobDirectory<S> {
    fn list_all(&self) -> Result<Vec<String>> {
        let mut names = self.store.list()?;
        names.sort_unstable();
        names.dedup();
        Ok(names)
    }

    fn open(&self, name: &str) -> Result<Input> {
        self.store.get(name)
    }

    fn read_at(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.store.get_range(name, offset, len)
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        self.store.size(name)
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        if self.store.exists(name)? {
            return Err(Error::file_already_exists(format!(
                "File already exists: {name}"
            )));
        }
        let sink = Arc::clone(&self.store).writer(name)?;
        Ok(FsIndexOutput::from_sink(name, sink))
    }

    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        loop {
            let counter = self.temp_counter.fetch_add(1, Ordering::Relaxed);
            let name = temp_file_name(prefix, suffix, counter);
            if !self.store.exists(&name)? {
                let sink = Arc::clone(&self.store).writer(&name)?;
                return Ok(FsIndexOutput::from_sink(&name, sink));
            }
        }
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        self.store.sync(names)
    }

    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        self.store.copy(source, dest)?;
        self.store.delete(source)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.store.delete(name)
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

/// A [`BlobStore`] held in a map: a test double for a real object store,
/// and the reference for what one must do. Objects are shared on read
/// ([`Input::Shared`]).
#[derive(Debug, Default, Clone)]
pub struct InMemoryBlobStore {
    objects: Arc<Mutex<BTreeMap<String, Arc<[u8]>>>>,
}

impl InMemoryBlobStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl BlobStore for InMemoryBlobStore {
    fn list(&self) -> Result<Vec<String>> {
        Ok(lock_ignoring_poison(&self.objects)
            .keys()
            .cloned()
            .collect())
    }

    fn get(&self, name: &str) -> Result<Input> {
        lock_ignoring_poison(&self.objects)
            .get(name)
            .map(|b| Input::Shared(Arc::clone(b)))
            .ok_or_else(|| not_found(name))
    }

    fn get_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let objects = lock_ignoring_poison(&self.objects);
        let bytes = objects.get(name).ok_or_else(|| not_found(name))?;
        let start = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(bytes.len());
        let end = start.saturating_add(len).min(bytes.len());
        Ok(bytes[start..end].to_vec())
    }

    fn list_with_sizes(&self) -> Result<Option<Vec<(String, u64)>>> {
        Ok(Some(
            lock_ignoring_poison(&self.objects)
                .iter()
                .map(|(name, bytes)| (name.clone(), bytes.len() as u64))
                .collect(),
        ))
    }

    fn size(&self, name: &str) -> Result<u64> {
        lock_ignoring_poison(&self.objects)
            .get(name)
            .map(|b| b.len() as u64)
            .ok_or_else(|| not_found(name))
    }

    fn put(&self, name: &str, bytes: Vec<u8>) -> Result<()> {
        lock_ignoring_poison(&self.objects).insert(name.to_string(), Arc::from(bytes));
        Ok(())
    }

    fn delete(&self, name: &str) -> Result<()> {
        lock_ignoring_poison(&self.objects)
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| not_found(name))
    }

    fn copy(&self, source: &str, dest: &str) -> Result<()> {
        let mut objects = lock_ignoring_poison(&self.objects);
        let bytes = objects
            .get(source)
            .cloned()
            .ok_or_else(|| not_found(source))?;
        objects.insert(dest.to_string(), bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_output::DataOutput;
    use crate::index_output::IndexOutput;

    #[test]
    fn a_file_appears_only_when_its_output_closes() {
        let dir = BlobDirectory::new(InMemoryBlobStore::new());
        let mut out = dir.create_output("_0.cfs").unwrap();
        out.write_bytes(b"hello ");
        out.write_bytes(&[b'x'; 20_000]);
        assert!(dir.list_all().unwrap().is_empty());
        assert!(out.path().is_none());
        assert!(format!("{out:?}").contains("SinkIndexOutput(file=_0.cfs)"));
        let crc = out.checksum();
        assert_eq!(out.close().unwrap(), crc);
        assert_eq!(dir.list_all().unwrap(), vec!["_0.cfs".to_string()]);
        assert_eq!(dir.file_length("_0.cfs").unwrap(), 20_006);
        let bytes = dir.open("_0.cfs").unwrap();
        assert_eq!(&bytes[..6], b"hello ");
        assert_eq!(crc32fast::hash(&bytes) as u64, crc);
    }

    #[test]
    fn create_output_refuses_an_existing_name_and_temp_outputs_skip_taken_ones() {
        let dir = BlobDirectory::new(InMemoryBlobStore::new());
        dir.store().put("a", b"1".to_vec()).unwrap();
        let err = dir.create_output("a").unwrap_err();
        assert!(matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::AlreadyExists));
        dir.store()
            .put(&temp_file_name("p", "s", 0), Vec::new())
            .unwrap();
        let out = dir.create_temp_output("p", "s").unwrap();
        assert_eq!(out.name(), temp_file_name("p", "s", 1));
        out.close().unwrap();
        assert!(dir.store().exists(&temp_file_name("p", "s", 1)).unwrap());
    }

    #[test]
    fn rename_copies_then_deletes_and_missing_files_are_not_found() {
        let dir = BlobDirectory::new(InMemoryBlobStore::new());
        let mut out = dir.create_output("pending_segments_1").unwrap();
        out.write_bytes(b"commit");
        out.close().unwrap();
        dir.sync(&["pending_segments_1".to_string()]).unwrap();
        dir.rename("pending_segments_1", "segments_1").unwrap();
        dir.sync_meta_data().unwrap();
        assert_eq!(dir.list_all().unwrap(), vec!["segments_1".to_string()]);
        assert_eq!(&*dir.open("segments_1").unwrap(), b"commit");
        assert!(dir.open("nope").unwrap_err().is_no_such_file());
        assert!(dir.file_length("nope").unwrap_err().is_no_such_file());
        assert!(dir.delete_file("nope").unwrap_err().is_no_such_file());
        assert!(dir.rename("nope", "x").unwrap_err().is_no_such_file());
        dir.delete_file("segments_1").unwrap();
        assert!(dir.list_all().unwrap().is_empty());
        assert!(dir.pending_deletions().unwrap().is_empty());
    }

    #[test]
    fn the_lock_excludes_a_second_writer_in_this_process() {
        let dir = BlobDirectory::new(InMemoryBlobStore::new());
        assert!(format!("{dir:?}").starts_with("BlobDirectory("));
        let lock = dir.obtain_lock("write.lock").unwrap();
        assert!(matches!(
            dir.obtain_lock("write.lock"),
            Err(Error::LockObtainFailed(_))
        ));
        drop(lock);
        dir.obtain_lock("write.lock").unwrap();
    }

    /// A store with only the required methods, so the trait's defaults run.
    struct Minimal(InMemoryBlobStore);

    impl BlobStore for Minimal {
        fn list(&self) -> Result<Vec<String>> {
            let mut names = self.0.list()?;
            names.extend(names.clone());
            Ok(names)
        }
        fn get(&self, name: &str) -> Result<Input> {
            if name == "broken" {
                return Err(Error::Corrupted("store failure".into()));
            }
            self.0.get(name)
        }
        fn put(&self, name: &str, bytes: Vec<u8>) -> Result<()> {
            if name == "readonly" {
                return Err(Error::IllegalArgument("read-only object".into()));
            }
            self.0.put(name, bytes)
        }
        fn delete(&self, name: &str) -> Result<()> {
            self.0.delete(name)
        }
    }

    #[test]
    fn the_trait_defaults_download_copy_and_dedupe() {
        let dir = BlobDirectory::new(Minimal(InMemoryBlobStore::new()));
        let mut out = dir.create_output("a").unwrap();
        out.write_bytes(b"abc");
        out.close().unwrap();
        assert_eq!(dir.list_all().unwrap(), vec!["a".to_string()]);
        assert_eq!(dir.file_length("a").unwrap(), 3);
        dir.rename("a", "b").unwrap();
        assert_eq!(&*dir.open("b").unwrap(), b"abc");
        assert!(dir.store().exists("broken").is_err());
        assert!(!dir.store().exists("a").unwrap());
        // A failed put fails the close, with a kind the caller can read.
        let mut out = dir.create_output("readonly").unwrap();
        out.write_byte(1);
        let err = out.close().unwrap_err();
        assert!(err.to_string().contains("read-only object"), "{err}");
    }

    #[test]
    fn into_io_keeps_an_io_errors_kind() {
        let e = into_io(not_found("x"));
        assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
        let e = into_io(Error::Corrupted("bad".into()));
        assert_eq!(e.kind(), std::io::ErrorKind::Other);
    }
}
