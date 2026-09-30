//! Port of `org.apache.lucene.store.Directory` / `FSDirectory` / `MMapDirectory`,
//! plus the generation-lookup logic from `org.apache.lucene.index.SegmentInfos`
//! (`getLastCommitGeneration`, `generationFromSegmentsFileName`) that depends only
//! on a file listing.
//!
//! Two backends, one trait:
//! - [`FsDirectory`]: `std::fs::read` — safe, no `unsafe`, always correct. Default.
//! - [`MmapDirectory`]: `memmap2` — zero-copy reads matching Lucene's own default
//!   (`MMapDirectory`) for real workloads. Contains one of this crate's two
//!   `unsafe` sites (the other is `fs_lock_factory`'s `fcntl`), documented on
//!   the call site: mapping a file is only sound if nothing else
//!   truncates/mutates it concurrently, same caveat Lucene's own Javadoc carries.
//!
//! Both return an [`Input`] — an owned-or-mapped byte buffer that `Deref`s to
//! `&[u8]`, so callers (codec_util, segment_info, segment_infos) are unchanged
//! regardless of backend.

use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::data_output::DataOutput;
use crate::error::{Error, Result};
use crate::fs_lock_factory::default_fs_lock_factory;
#[cfg(doc)]
use crate::fs_lock_factory::NativeFsLockFactory;
use crate::index_output::{self, FsIndexOutput};
use crate::lock::{Lock, LockFactory};

/// The `segments` file-name prefix (`IndexFileNames.SEGMENTS`). Excludes the
/// pre-4.0 `segments.gen` pointer file, which is not a valid commit file name.
const SEGMENTS_PREFIX: &str = "segments";
const OLD_SEGMENTS_GEN: &str = "segments.gen";
/// `IndexFileNames.PENDING_SEGMENTS`: the name a `segments_N` is written
/// under before it is renamed into place. Deliberately *not* prefixed with
/// [`SEGMENTS_PREFIX`], so a half-written commit file can never be picked up
/// by [`last_commit_generation`] -- that invisibility is the whole point of
/// Java's two-phase `prepareCommit`/`finishCommit` protocol.
const PENDING_SEGMENTS_PREFIX: &str = "pending_segments";

/// A file's bytes, however the backend obtained them.
pub enum Input {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
    /// Shared with the directory that holds them in memory
    /// ([`crate::ByteBuffersDirectory`]): opening costs a reference count,
    /// not a copy -- Java's `IndexInput.clone()` of the stored content.
    Shared(Arc<[u8]>),
}

impl std::fmt::Debug for Input {
    /// Length and provenance only. The alternative is dumping a
    /// half-gigabyte mapping into a panic message.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            Input::Owned(_) => "Owned",
            Input::Mapped(_) => "Mapped",
            Input::Shared(_) => "Shared",
        };
        write!(f, "Input::{kind}({} bytes)", self.len())
    }
}

impl Deref for Input {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Input::Owned(v) => v,
            Input::Mapped(m) => m,
            Input::Shared(s) => s,
        }
    }
}

/// The same bytes as [`Deref`], as an `AsRef` — the bound a type-erased
/// shared buffer needs.
///
/// `Deref` alone cannot be used behind `dyn`: `Arc<dyn Deref<Target = [u8]>>`
/// is legal but every consumer would have to name the associated type, and
/// `Arc<[u8]>` (the obvious alternative) cannot alias a mapping — it owns its
/// allocation, so handing an `Input` to one always costs a full copy. With
/// this impl an `Arc<Input>` coerces straight to
/// `Arc<dyn AsRef<[u8]> + Send + Sync>`, which is how
/// `lucene_codecs::blocktree::open_shared` takes a `.tim`/`.tip` mapping
/// without copying it (c12, ~199 µs on a 4.7 MB `.tim`).
impl AsRef<[u8]> for Input {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

/// Directory abstraction covering both Lucene's read path (`listAll`, `open`
/// a whole file's bytes) and the write path: `createOutput`/`createTempOutput`
/// (an [`FsIndexOutput`]), `sync` (the fsync-before-durable contract),
/// `rename`/`deleteFile`/`syncMetaData` (what `SegmentInfos.prepareCommit`/
/// `finishCommit`/`rollbackCommit` need to publish a commit atomically),
/// `obtainLock` (the `write.lock` an `IndexWriter` holds) and
/// `getPendingDeletions`.
///
/// `Send + Sync`, as Java's `Directory` is by contract: a writer's indexing
/// threads flush segments into it concurrently while its merge thread reads
/// sources out of it (M4's T4.3). Every implementation here is either
/// stateless over the filesystem or guards its own state.
///
/// Java's `close()` is `Drop`: [`FsDirectory`] retries its pending deletes
/// when dropped, as `FSDirectory.close` does.
pub trait Directory: Send + Sync {
    /// Port of `Directory.listAll()`: every file name in the directory, sorted.
    fn list_all(&self) -> Result<Vec<String>>;

    /// Reads a whole file's bytes (`Directory.openInput`).
    fn open(&self, name: &str) -> Result<Input>;

    /// Port of `Directory.fileLength(name)`. The default reads the file,
    /// which is correct for any implementation; the real ones override it.
    fn file_length(&self, name: &str) -> Result<u64> {
        Ok(self.open(name)?.len() as u64)
    }

    /// Port of `Directory.createOutput(name, context)`: creates (truncating
    /// any existing file of the same name) a new file for sequential
    /// writing.
    fn create_output(&self, name: &str) -> Result<FsIndexOutput>;

    /// Port of `Directory.createOutput(name, context)` for a context that
    /// carries a flush or merge size estimate (`FlushInfo`'s
    /// `estimatedSegmentSize`, `MergeInfo`'s `estimatedMergeBytes`), the only
    /// part of `IOContext` a directory's *results* can depend on
    /// ([`crate::NrtCachingDirectory`] caches small segments in memory). The
    /// default ignores the estimate; a wrapper forwards it.
    fn create_output_with_estimate(
        &self,
        name: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        let _ = estimated_bytes;
        self.create_output(name)
    }

    /// Port of `Directory.createTempOutput(prefix, suffix, context)`: a new
    /// output under a fresh name built by [`temp_file_name`], never
    /// clobbering an existing file.
    ///
    /// Java declares it abstract; the default here refuses (as a read-only
    /// directory such as a compound file does), so a test double need not
    /// implement it. Every real directory in this crate does.
    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        let _ = (prefix, suffix);
        Err(unsupported("createTempOutput"))
    }

    /// [`Directory::create_temp_output`] with a context's size estimate, as
    /// [`Directory::create_output_with_estimate`].
    fn create_temp_output_with_estimate(
        &self,
        prefix: &str,
        suffix: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        let _ = estimated_bytes;
        self.create_temp_output(prefix, suffix)
    }

    /// Port of `Directory.sync(Collection<String>)`: fsyncs every named
    /// file's contents (and, best-effort, the directory entry) to disk.
    /// Callers must sync a new segment's files before referencing them from
    /// a commit file — that's Lucene's actual durability contract.
    fn sync(&self, names: &[String]) -> Result<()>;

    /// Port of `Directory.rename(source, dest)`: atomically makes `source`'s
    /// contents visible under `dest`. This is the operation Lucene's
    /// `SegmentInfos.finishCommit` relies on to publish a commit — the
    /// `pending_segments_N` file is fully written and fsynced first, then a
    /// single rename makes it the new `segments_N`, so no crash can ever
    /// expose a half-written commit file under a name a reader scans for.
    fn rename(&self, source: &str, dest: &str) -> Result<()>;

    /// Port of `Directory.deleteFile(name)`.
    fn delete_file(&self, name: &str) -> Result<()>;

    /// Port of `Directory.syncMetaData()`: fsyncs the directory itself, so a
    /// rename/create of a *name* (not just a file's contents) survives a
    /// crash. Lucene calls this on both sides of the commit rename.
    fn sync_meta_data(&self) -> Result<()>;

    /// Port of `Directory.obtainLock(name)`: acquires the named lock
    /// (`write.lock`) through this directory's [`LockFactory`], failing with
    /// [`Error::LockObtainFailed`] when it is held elsewhere.
    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>>;

    /// Port of `Directory.getPendingDeletions()`: files a delete was
    /// requested for but could not yet be removed. Only a filesystem
    /// directory ever has any.
    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        Ok(BTreeSet::new())
    }

    /// `dir instanceof FSDirectory ? dir.getDirectory() : null`: the
    /// filesystem path a [`crate::fs_lock_factory::FsLockFactory`] puts its
    /// lock file in. `None` for anything that is not a filesystem directory.
    fn fs_directory_path(&self) -> Option<&Path> {
        None
    }

    /// Port of `Directory.copyFrom(from, src, dest, context)`: copies a file
    /// from another directory, deleting the partial `dest` on failure.
    fn copy_from(&self, from: &dyn Directory, src: &str, dest: &str) -> Result<()> {
        let copied = from.open(src).and_then(|bytes| {
            let mut out = self.create_output(dest)?;
            out.write_bytes(&bytes);
            out.close()
        });
        match copied {
            Ok(_) => Ok(()),
            Err(e) => {
                // `IOUtils.deleteFilesIgnoringExceptions(this, dest)`.
                let _ = self.delete_file(dest);
                Err(e)
            }
        }
    }
}

/// `UnsupportedOperationException` from a directory that cannot perform
/// `what`.
pub(crate) fn unsupported(what: &str) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!("{what} is not supported by this directory"),
    ))
}

/// Forwards every [`Directory`] method through a pointer type, so a
/// reference or an `Arc` to a directory is a directory too -- which is what
/// lets the wrappers ([`crate::NrtCachingDirectory`],
/// [`crate::FileSwitchDirectory`], ...) own either.
macro_rules! forward_directory {
    ($($ty:ty),*) => {$(
        impl<T: Directory + ?Sized> Directory for $ty {
            fn list_all(&self) -> Result<Vec<String>> {
                (**self).list_all()
            }
            fn open(&self, name: &str) -> Result<Input> {
                (**self).open(name)
            }
            fn file_length(&self, name: &str) -> Result<u64> {
                (**self).file_length(name)
            }
            fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
                (**self).create_output(name)
            }
            fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
                (**self).create_temp_output(prefix, suffix)
            }
            fn create_output_with_estimate(
                &self,
                name: &str,
                estimated_bytes: Option<u64>,
            ) -> Result<FsIndexOutput> {
                (**self).create_output_with_estimate(name, estimated_bytes)
            }
            fn create_temp_output_with_estimate(
                &self,
                prefix: &str,
                suffix: &str,
                estimated_bytes: Option<u64>,
            ) -> Result<FsIndexOutput> {
                (**self).create_temp_output_with_estimate(prefix, suffix, estimated_bytes)
            }
            fn sync(&self, names: &[String]) -> Result<()> {
                (**self).sync(names)
            }
            fn rename(&self, source: &str, dest: &str) -> Result<()> {
                (**self).rename(source, dest)
            }
            fn delete_file(&self, name: &str) -> Result<()> {
                (**self).delete_file(name)
            }
            fn sync_meta_data(&self) -> Result<()> {
                (**self).sync_meta_data()
            }
            fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
                (**self).obtain_lock(name)
            }
            fn pending_deletions(&self) -> Result<BTreeSet<String>> {
                (**self).pending_deletions()
            }
            fn fs_directory_path(&self) -> Option<&Path> {
                (**self).fs_directory_path()
            }
            fn copy_from(&self, from: &dyn Directory, src: &str, dest: &str) -> Result<()> {
                (**self).copy_from(from, src, dest)
            }
        }
    )*};
}

forward_directory!(&T, Arc<T>);

/// A directory every output of which is created with one size estimate:
/// the Rust shape of the single `IOContext` (`new IOContext(flushInfo)` or
/// `new IOContext(mergeInfo)`) Java's `IndexWriter` passes to every
/// `createOutput` of one flush or one merge. Everything else forwards.
pub struct EstimatedWrites<'a> {
    inner: &'a dyn Directory,
    estimated_bytes: u64,
}

impl<'a> EstimatedWrites<'a> {
    /// `inner`, with `estimated_bytes` (`FlushInfo.estimatedSegmentSize` or
    /// `MergeInfo.estimatedMergeBytes`) on every output it creates.
    pub fn new(inner: &'a dyn Directory, estimated_bytes: u64) -> Self {
        EstimatedWrites {
            inner,
            estimated_bytes,
        }
    }

    /// The estimate every output carries.
    pub fn estimated_bytes(&self) -> u64 {
        self.estimated_bytes
    }
}

impl Directory for EstimatedWrites<'_> {
    fn list_all(&self) -> Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> Result<Input> {
        self.inner.open(name)
    }
    fn file_length(&self, name: &str) -> Result<u64> {
        self.inner.file_length(name)
    }
    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.inner
            .create_output_with_estimate(name, Some(self.estimated_bytes))
    }
    fn create_output_with_estimate(
        &self,
        name: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        self.inner
            .create_output_with_estimate(name, estimated_bytes)
    }
    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        self.inner
            .create_temp_output_with_estimate(prefix, suffix, Some(self.estimated_bytes))
    }
    fn create_temp_output_with_estimate(
        &self,
        prefix: &str,
        suffix: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        self.inner
            .create_temp_output_with_estimate(prefix, suffix, estimated_bytes)
    }
    fn sync(&self, names: &[String]) -> Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> Result<()> {
        self.inner.delete_file(name)
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
        self.inner.fs_directory_path()
    }
}

/// Port of `Directory.getTempFileName(prefix, suffix, counter)`:
/// `IndexFileNames.segmentFileName(prefix, suffix + "_" + base36(counter),
/// "tmp")`, i.e. `{prefix}_{suffix}_{counter}.tmp`.
pub fn temp_file_name(prefix: &str, suffix: &str, counter: u64) -> String {
    let counter = lucene_util::base36::to_base36(i64::try_from(counter).unwrap_or(i64::MAX));
    format!("{prefix}_{suffix}_{counter}.tmp")
}

/// Port of `BaseDirectory`: the part of a directory that holds its
/// [`LockFactory`] and implements `obtainLock` through it. Java's abstract
/// base class becomes a field the concrete directories
/// ([`FsDirectory`], [`MmapDirectory`], [`crate::ByteBuffersDirectory`])
/// embed.
#[derive(Debug, Clone)]
pub struct BaseDirectory {
    lock_factory: Arc<dyn LockFactory>,
}

impl BaseDirectory {
    /// `BaseDirectory(lockFactory)`.
    pub fn new(lock_factory: Arc<dyn LockFactory>) -> Self {
        Self { lock_factory }
    }

    /// `BaseDirectory.lockFactory`.
    pub fn lock_factory(&self) -> &Arc<dyn LockFactory> {
        &self.lock_factory
    }

    /// `BaseDirectory.obtainLock(name)`: `lockFactory.obtainLock(this,
    /// name)`, with `this` passed explicitly.
    pub fn obtain_lock(&self, dir: &dyn Directory, name: &str) -> Result<Box<dyn Lock>> {
        self.lock_factory.obtain_lock(dir, name)
    }
}

/// The state `FSDirectory` keeps over a filesystem path, shared by both
/// backends: the lock factory, the pending deletes, and the temp-file
/// counter. Only reads differ between [`FsDirectory`] and [`MmapDirectory`].
struct FsCore {
    root: PathBuf,
    base: BaseDirectory,
    /// `FSDirectory.pendingDeletes`: files whose delete failed with an error
    /// other than "no such file" (on Windows, a file still open elsewhere),
    /// hidden from listings and retried later.
    pending_deletes: Mutex<BTreeSet<String>>,
    /// `FSDirectory.opsSinceLastDelete`.
    ops_since_last_delete: AtomicUsize,
    /// `FSDirectory.nextTempFileCounter`.
    next_temp_file_counter: AtomicU64,
}

impl FsCore {
    fn new(root: PathBuf, lock_factory: Arc<dyn LockFactory>) -> Self {
        Self {
            root,
            base: BaseDirectory::new(lock_factory),
            pending_deletes: Mutex::new(BTreeSet::new()),
            ops_since_last_delete: AtomicUsize::new(0),
            next_temp_file_counter: AtomicU64::new(0),
        }
    }

    fn pending(&self) -> std::sync::MutexGuard<'_, BTreeSet<String>> {
        crate::lock::lock_ignoring_poison(&self.pending_deletes)
    }

    fn is_pending(&self, name: &str) -> bool {
        self.pending().contains(name)
    }

    /// `FSDirectory.listAll()`: the listing, minus pending deletes.
    fn list_all(&self) -> Result<Vec<String>> {
        let mut names = list_all(&self.root)?;
        let pending = self.pending();
        if !pending.is_empty() {
            names.retain(|n| !pending.contains(n));
        }
        Ok(names)
    }

    /// `FSDirectory.fileLength(name)`.
    fn file_length(&self, name: &str) -> Result<u64> {
        if self.is_pending(name) {
            return Err(Error::no_such_file(format!(
                "file \"{name}\" is pending delete"
            )));
        }
        Ok(fs::metadata(self.root.join(name))?.len())
    }

    /// `FSDirectory.ensureCanRead(name)`.
    fn ensure_can_read(&self, name: &str) -> Result<()> {
        if self.is_pending(name) {
            return Err(Error::no_such_file(format!(
                "file \"{name}\" is pending delete and cannot be opened for read"
            )));
        }
        Ok(())
    }

    /// `FSDirectory.createOutput(name, context)`.
    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.maybe_delete_pending_files()?;
        // If this file was pending delete, we are now bringing it back to
        // life.
        if self.pending().remove(name) {
            // Try again to delete it -- this is the best effort.
            self.private_delete_file(name, true)?;
            // If the delete failed it went back in; take it out again.
            self.pending().remove(name);
        }
        index_output::create_output(&self.root, name)
    }

    /// `FSDirectory.createTempOutput(prefix, suffix, context)`.
    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        self.maybe_delete_pending_files()?;
        loop {
            let counter = self.next_temp_file_counter.fetch_add(1, Ordering::Relaxed);
            let name = temp_file_name(prefix, suffix, counter);
            if self.is_pending(&name) {
                continue;
            }
            match FsIndexOutput::create_new(&self.root, &name) {
                Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                other => return other,
            }
        }
    }

    /// `FSDirectory.sync(names)`.
    fn sync(&self, names: &[String]) -> Result<()> {
        index_output::sync(&self.root, names)?;
        self.maybe_delete_pending_files()
    }

    /// `FSDirectory.rename(source, dest)`.
    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        if self.is_pending(source) {
            return Err(Error::no_such_file(format!(
                "file \"{source}\" is pending delete and cannot be moved"
            )));
        }
        self.maybe_delete_pending_files()?;
        if self.pending().remove(dest) {
            self.private_delete_file(dest, true)?;
            self.pending().remove(dest);
        }
        index_output::rename(&self.root, source, dest)
    }

    /// `FSDirectory.syncMetaData()`.
    fn sync_meta_data(&self) -> Result<()> {
        index_output::sync_meta_data(&self.root)?;
        self.maybe_delete_pending_files()
    }

    /// `FSDirectory.deleteFile(name)`.
    fn delete_file(&self, name: &str) -> Result<()> {
        if self.is_pending(name) {
            return Err(Error::no_such_file(format!(
                "file \"{name}\" is already pending delete"
            )));
        }
        self.private_delete_file(name, false)?;
        self.maybe_delete_pending_files()
    }

    /// `FSDirectory.deletePendingFiles()`: retries every pending delete.
    fn delete_pending_files(&self) -> Result<()> {
        // Clone the set, since `private_delete_file` mutates it.
        let pending: Vec<String> = self.pending().iter().cloned().collect();
        for name in pending {
            self.private_delete_file(&name, true)?;
        }
        Ok(())
    }

    /// `FSDirectory.maybeDeletePendingFiles()`: retries the pending deletes
    /// once every `pendingDeletes.size()` operations -- "a silly heuristic to
    /// try to avoid O(N^2) behaviour on Windows".
    fn maybe_delete_pending_files(&self) -> Result<()> {
        let pending = self.pending().len();
        if pending == 0 {
            return Ok(());
        }
        let count = self
            .ops_since_last_delete
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        if count >= pending {
            self.ops_since_last_delete
                .fetch_sub(count, Ordering::AcqRel);
            self.delete_pending_files()?;
        }
        Ok(())
    }

    /// `FSDirectory.privateDeleteFile(name, isPendingDelete)`.
    fn private_delete_file(&self, name: &str, is_pending_delete: bool) -> Result<()> {
        match fs::remove_file(self.root.join(name)) {
            Ok(()) => {
                self.pending().remove(name);
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // We were asked to delete a non-existent file.
                self.pending().remove(name);
                if is_pending_delete && cfg!(windows) {
                    // LUCENE-6684: a file can sit in a "pending delete" state
                    // on Windows, failing the first attempt with access
                    // denied and then this one with no-such-file.
                    Ok(())
                } else {
                    Err(Error::Io(e))
                }
            }
            Err(_) => {
                // On Windows a delete can fail while a handle is still open
                // against the file: record it and try again later. (Java
                // does this for every other error, on every platform -- a
                // CIFS mount on Linux can behave the same way.)
                self.pending().insert(name.to_string());
                Ok(())
            }
        }
    }

    /// `FSDirectory.getPendingDeletions()`: retries first, then reports what
    /// is still pending.
    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        self.delete_pending_files()?;
        Ok(self.pending().clone())
    }
}

impl Drop for FsCore {
    /// `FSDirectory.close()`: a last attempt at the pending deletes.
    fn drop(&mut self) {
        let _ = self.delete_pending_files();
    }
}

/// Safe, copying backend (`std::fs::read`) -- Java's `NIOFSDirectory` in
/// role. No `unsafe` on its read path.
pub struct FsDirectory {
    core: FsCore,
}

impl FsDirectory {
    /// `FSDirectory.open(path)`: locks with [`NativeFsLockFactory`]
    /// (`FSLockFactory.getDefault()`). Neither creates nor canonicalises the
    /// path up front, unlike Java's constructor -- the lock factory creates
    /// the directory when a writer first locks it.
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self::with_lock_factory(root, default_fs_lock_factory())
    }

    /// `FSDirectory.open(path, lockFactory)`.
    pub fn with_lock_factory(root: impl Into<PathBuf>, lock_factory: Arc<dyn LockFactory>) -> Self {
        Self {
            core: FsCore::new(root.into(), lock_factory),
        }
    }

    /// `FSDirectory.getDirectory()`.
    pub fn directory(&self) -> &Path {
        &self.core.root
    }

    /// `BaseDirectory.lockFactory`.
    pub fn lock_factory(&self) -> &Arc<dyn LockFactory> {
        self.core.base.lock_factory()
    }

    /// `FSDirectory.deletePendingFiles()`.
    pub fn delete_pending_files(&self) -> Result<()> {
        self.core.delete_pending_files()
    }
}

impl Directory for FsDirectory {
    fn list_all(&self) -> Result<Vec<String>> {
        self.core.list_all()
    }

    fn open(&self, name: &str) -> Result<Input> {
        self.core.ensure_can_read(name)?;
        Ok(Input::Owned(fs::read(self.core.root.join(name))?))
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        self.core.file_length(name)
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.core.create_output(name)
    }

    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        self.core.create_temp_output(prefix, suffix)
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        self.core.sync(names)
    }

    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        self.core.rename(source, dest)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.core.delete_file(name)
    }

    fn sync_meta_data(&self) -> Result<()> {
        self.core.sync_meta_data()
    }

    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.core.base.obtain_lock(self, name)
    }

    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        self.core.pending_deletions()
    }

    fn fs_directory_path(&self) -> Option<&Path> {
        Some(&self.core.root)
    }
}

/// The size at or below which [`MmapDirectory`] reads a file instead of
/// mapping it.
///
/// Mapping is only cheaper than reading once the mapping is large enough to
/// amortise its own syscalls. Measured on this project's 5 M-document
/// benchmark corpus (`crates/lucene-search/examples/reader_open_profile.rs`),
/// a few-hundred-byte file costs **1.86 µs** to `open`+`mmap`+`munmap` and
/// **1.19 µs** to `open`+`read`, before the mapping's first page fault --
/// and every file this small in a segment (`.si`, `.fnm`, `.tmd`, `.dvm`,
/// `.nvm`, `.kdm`, `segments_N`) is parsed whole at open, so it takes that
/// fault immediately.
///
/// 16 KiB, not larger, because the files a reader *holds* and then accesses
/// randomly -- `.tip`, `.kdi` -- are the ones a copy would pessimise, and
/// they are above this on any index big enough for it to matter. The worst
/// case the threshold can cost is one 16 KiB `memcpy`.
///
/// Real Lucene has no equivalent: `MMapDirectory.openInput` maps
/// unconditionally. This is a Rust-side win, not a port divergence -- the
/// bytes a caller sees are identical, and [`Input`] already had both
/// representations because [`FsDirectory`] produces the owned one.
pub const SMALL_FILE_READ_THRESHOLD: u64 = 16 * 1024;

/// Zero-copy backend (`memmap2`), matching Lucene's default `MMapDirectory`.
pub struct MmapDirectory {
    core: FsCore,
    /// Files of at most this many bytes are read rather than mapped -- see
    /// [`SMALL_FILE_READ_THRESHOLD`]. `0` maps everything, which is what
    /// this backend did before and what
    /// [`MmapDirectory::with_read_threshold`] exists to reproduce.
    read_threshold: u64,
}

impl MmapDirectory {
    /// `new MMapDirectory(path)`, locking with [`NativeFsLockFactory`].
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self::with_read_threshold(root, SMALL_FILE_READ_THRESHOLD)
    }

    /// [`Self::open`] with an explicit small-file threshold: files of at most
    /// `read_threshold` bytes are read into memory instead of mapped. `0`
    /// maps every file.
    ///
    /// Exposed because it is the only way to measure the two arms in **one
    /// process**, which is how this project measures anything (see
    /// `docs/sweep/m2/c24-arith-codecs.md` on why criterion is not trusted
    /// here) -- and because a caller with an unusual access pattern (a
    /// long-lived reader over many small files it seeks in repeatedly) can
    /// turn it off.
    pub fn with_read_threshold(root: impl Into<PathBuf>, read_threshold: u64) -> Self {
        Self {
            core: FsCore::new(root.into(), default_fs_lock_factory()),
            read_threshold,
        }
    }

    /// `new MMapDirectory(path, lockFactory)`.
    pub fn with_lock_factory(root: impl Into<PathBuf>, lock_factory: Arc<dyn LockFactory>) -> Self {
        Self {
            core: FsCore::new(root.into(), lock_factory),
            read_threshold: SMALL_FILE_READ_THRESHOLD,
        }
    }

    /// `FSDirectory.getDirectory()`.
    pub fn directory(&self) -> &Path {
        &self.core.root
    }

    /// `FSDirectory.deletePendingFiles()`.
    pub fn delete_pending_files(&self) -> Result<()> {
        self.core.delete_pending_files()
    }
}

impl Directory for MmapDirectory {
    fn list_all(&self) -> Result<Vec<String>> {
        self.core.list_all()
    }

    fn open(&self, name: &str) -> Result<Input> {
        self.core.ensure_can_read(name)?;
        let mut file = fs::File::open(self.core.root.join(name))?;
        // A small file is read, not mapped: see `SMALL_FILE_READ_THRESHOLD`.
        // The `metadata` call is one `fstat` on an already-open descriptor,
        // which is cheaper than the `munmap` it avoids.
        let len = file.metadata()?.len();
        if self.read_threshold > 0 && len <= self.read_threshold {
            // Read from the descriptor already open, not `fs::read(path)`:
            // that re-opens by name and re-`fstat`s for its own size hint, so
            // the "cheaper than a mapping" arm would have paid *two*
            // `open`+`fstat` pairs -- and would have reopened a path that can
            // have changed underneath it in between. `len` is at most
            // `read_threshold`, so the `usize` cast cannot truncate on any
            // target this crate builds for.
            let mut buf = Vec::with_capacity(len as usize);
            file.read_to_end(&mut buf)?;
            return Ok(Input::Owned(buf));
        }
        // SAFETY: mapping is only unsound if another process truncates or
        // mutates this file while it's mapped, which we do not do ourselves and
        // which Lucene's own `MMapDirectory` accepts the same risk for (see its
        // Javadoc). Lucene never rewrites a file once written, so a mapped
        // index file is immutable for as long as anyone reads it.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Input::Mapped(mmap))
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        self.core.file_length(name)
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.core.create_output(name)
    }

    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        self.core.create_temp_output(prefix, suffix)
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        self.core.sync(names)
    }

    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        self.core.rename(source, dest)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.core.delete_file(name)
    }

    fn sync_meta_data(&self) -> Result<()> {
        self.core.sync_meta_data()
    }

    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.core.base.obtain_lock(self, name)
    }

    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        self.core.pending_deletions()
    }

    fn fs_directory_path(&self) -> Option<&Path> {
        Some(&self.core.root)
    }
}

fn list_all(root: &Path) -> Result<Vec<String>> {
    let mut names: Vec<String> = fs::read_dir(root)?
        .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<_>>()?;
    names.sort();
    Ok(names)
}

/// Port of `SegmentInfos.generationFromSegmentsFileName`.
pub fn generation_from_segments_file_name(file_name: &str) -> Result<i64> {
    if file_name == OLD_SEGMENTS_GEN {
        return Err(Error::Corrupted(format!(
            "\"{OLD_SEGMENTS_GEN}\" is not a valid segment file name since 4.0"
        )));
    }
    if file_name == SEGMENTS_PREFIX {
        return Ok(0);
    }
    if let Some(suffix) = file_name.strip_prefix(&format!("{SEGMENTS_PREFIX}_")) {
        return lucene_util::base36::from_base36(suffix).ok_or_else(|| {
            Error::Corrupted(format!("fileName \"{file_name}\" is not a segments file"))
        });
    }
    Err(Error::Corrupted(format!(
        "fileName \"{file_name}\" is not a segments file"
    )))
}

/// Port of `SegmentInfos.getLastCommitGeneration(String[])`: the highest
/// generation among `segments`/`segments_N` file names (excluding the legacy
/// `segments.gen` pointer), or -1 if none exist.
///
/// Strict, like Java: an unparsable `segments*` name aborts the scan rather
/// than being skipped. Skipping it is not a safe simplification in either
/// direction — a reader would silently open the *previous* commit and hide
/// every document committed since, and a writer would read -1 from a
/// directory that does have a commit and create a fresh index over it.
pub fn last_commit_generation(files: &[String]) -> Result<i64> {
    let mut generation = -1i64;
    for file in files {
        if is_segments_candidate(file) {
            generation = generation.max(generation_from_segments_file_name(file)?);
        }
    }
    Ok(generation)
}

/// The `startsWith(SEGMENTS) && startsWith(OLD_SEGMENTS_GEN) == false` guard
/// Java applies before calling `generationFromSegmentsFileName`. Note the
/// second test is a prefix test in Java, not equality: `segments.gen_1` is
/// skipped too, not treated as a corrupt generation.
fn is_segments_candidate(file_name: &str) -> bool {
    file_name.starts_with(SEGMENTS_PREFIX) && !file_name.starts_with(OLD_SEGMENTS_GEN)
}

/// Port of `IndexFileNames.fileNameFromGeneration("segments", "", gen)`.
pub fn segments_file_name(generation: i64) -> Option<String> {
    match generation {
        -1 => None,
        0 => Some(SEGMENTS_PREFIX.to_string()),
        gen => Some(format!(
            "{SEGMENTS_PREFIX}_{}",
            lucene_util::base36::to_base36(gen)
        )),
    }
}

/// Port of `IndexFileNames.fileNameFromGeneration("pending_segments", "",
/// gen)`: the name a commit file is written under before
/// `SegmentInfos.finishCommit` renames it to [`segments_file_name`]'s name.
///
/// Same `gen == -1 -> null`, `gen == 0 -> bare base name` shape
/// [`segments_file_name`] has, since both go through the same Java helper.
/// Generation 0 is unreachable in practice (`getNextPendingGeneration()`
/// returns `1` for a never-committed index and `generation + 1` otherwise),
/// but the mapping is kept total and exact rather than special-cased away.
pub fn pending_segments_file_name(generation: i64) -> Option<String> {
    match generation {
        g if g < 0 => None,
        0 => Some(PENDING_SEGMENTS_PREFIX.to_string()),
        gen => Some(format!(
            "{PENDING_SEGMENTS_PREFIX}_{}",
            lucene_util::base36::to_base36(gen)
        )),
    }
}

/// Finds and reads the most recent `segments_N` commit file in `dir`.
/// Returns `(generation, bytes)`; callers pass both to `segment_infos::parse`.
pub fn read_latest_commit(dir: &(impl Directory + ?Sized)) -> Result<(i64, Input)> {
    let files = dir.list_all()?;
    let generation = last_commit_generation(&files)?;
    let name = segments_file_name(generation)
        .ok_or_else(|| Error::Corrupted("no segments_N commit file found".to_string()))?;
    let bytes = dir.open(&name)?;
    Ok((generation, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_output::DataOutput;

    #[test]
    fn generation_from_segments_file_name_valid_cases() {
        assert_eq!(generation_from_segments_file_name("segments").unwrap(), 0);
        assert_eq!(generation_from_segments_file_name("segments_1").unwrap(), 1);
        assert_eq!(generation_from_segments_file_name("segments_2").unwrap(), 2);
        // base-36: "segments_a" -> 10
        assert_eq!(
            generation_from_segments_file_name("segments_a").unwrap(),
            10
        );
    }

    #[test]
    fn generation_from_segments_file_name_rejects_old_pointer_file() {
        assert!(matches!(
            generation_from_segments_file_name("segments.gen"),
            Err(Error::Corrupted(_))
        ));
    }

    #[test]
    fn generation_from_segments_file_name_rejects_garbage() {
        assert!(matches!(
            generation_from_segments_file_name("not-a-segments-file"),
            Err(Error::Corrupted(_))
        ));
        // Has the prefix but a non-base-36 suffix.
        assert!(matches!(
            generation_from_segments_file_name("segments_!!!"),
            Err(Error::Corrupted(_))
        ));
    }

    #[test]
    fn last_commit_generation_ignores_old_pointer_and_non_segments_files() {
        let files = vec![
            "segments.gen".to_string(),
            "_0.si".to_string(),
            "segments_1".to_string(),
            "segments_3".to_string(),
            "segments_2".to_string(),
        ];
        assert_eq!(last_commit_generation(&files).unwrap(), 3);
    }

    #[test]
    fn segments_file_name_all_branches() {
        assert_eq!(segments_file_name(-1), None);
        assert_eq!(segments_file_name(0), Some("segments".to_string()));
        assert_eq!(segments_file_name(1), Some("segments_1".to_string()));
        assert_eq!(segments_file_name(10), Some("segments_a".to_string()));
    }

    /// A `Directory` that can only be listed: every test using it asserts
    /// that `read_latest_commit` fails during the *scan*, before any file is
    /// opened, so the remaining methods must never be called.
    struct ListingOnlyDir(Vec<String>);

    impl Directory for ListingOnlyDir {
        fn list_all(&self) -> Result<Vec<String>> {
            Ok(self.0.clone())
        }
        fn open(&self, name: &str) -> Result<Input> {
            panic!("open({name}) must not be reached: the generation scan should have failed")
        }
        fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
            panic!("create_output({name}) is not part of the read path under test")
        }
        fn sync(&self, _names: &[String]) -> Result<()> {
            panic!("sync() is not part of the read path under test")
        }
        fn rename(&self, source: &str, dest: &str) -> Result<()> {
            panic!("rename({source}, {dest}) is not part of the read path under test")
        }
        fn delete_file(&self, name: &str) -> Result<()> {
            panic!("delete_file({name}) is not part of the read path under test")
        }
        fn sync_meta_data(&self) -> Result<()> {
            panic!("sync_meta_data() is not part of the read path under test")
        }
        fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
            panic!("obtain_lock({name}) is not part of the read path under test")
        }
    }

    #[test]
    fn input_debug_reports_provenance_and_length_not_contents() {
        // The Debug impl exists so a panic message can't dump a mapped
        // half-gigabyte file; assert it stays that way.
        let owned = Input::Owned(vec![7u8; 42]);
        assert_eq!(format!("{owned:?}"), "Input::Owned(42 bytes)");

        let root = tempdir();
        // `with_read_threshold(_, 0)`: a ten-byte file is below
        // `SMALL_FILE_READ_THRESHOLD`, so the default backend would read it.
        let dir = MmapDirectory::with_read_threshold(&root, 0);
        index_output::write_all_bytes(&root, "_0.si", b"0123456789").unwrap();
        let mapped = dir.open("_0.si").unwrap();
        assert_eq!(format!("{mapped:?}"), "Input::Mapped(10 bytes)");
        fs::remove_dir_all(&root).ok();
    }

    /// [`SMALL_FILE_READ_THRESHOLD`] decides *how* a file is obtained and
    /// nothing else: the bytes a caller sees are identical either way.
    ///
    /// Both sides are asserted, because the interesting failure is silent --
    /// a threshold that never fires costs the syscalls it exists to remove
    /// and nothing reports it, and one that fires on a large file copies a
    /// mapping the reader meant to hold.
    #[test]
    fn small_files_are_read_and_large_ones_mapped() {
        let root = tempdir();
        let small = vec![3u8; 8];
        let large = vec![4u8; (SMALL_FILE_READ_THRESHOLD as usize) + 1];
        index_output::write_all_bytes(&root, "_0.nvm", &small).unwrap();
        index_output::write_all_bytes(&root, "_0.doc", &large).unwrap();

        let dir = MmapDirectory::open(&root);
        let got_small = dir.open("_0.nvm").unwrap();
        let got_large = dir.open("_0.doc").unwrap();
        assert!(matches!(got_small, Input::Owned(_)), "{got_small:?}");
        assert!(matches!(got_large, Input::Mapped(_)), "{got_large:?}");
        assert_eq!(&*got_small, &small[..]);
        assert_eq!(&*got_large, &large[..]);

        // Exactly at the threshold is still read (`<=`).
        let exact = vec![5u8; SMALL_FILE_READ_THRESHOLD as usize];
        index_output::write_all_bytes(&root, "_0.dvm", &exact).unwrap();
        assert!(matches!(dir.open("_0.dvm").unwrap(), Input::Owned(_)));

        // And `0` turns the whole thing off, which is the A/B arm
        // `reader_open_profile` measures against.
        let mapping = MmapDirectory::with_read_threshold(&root, 0);
        assert!(matches!(mapping.open("_0.nvm").unwrap(), Input::Mapped(_)));
        assert_eq!(&*mapping.open("_0.nvm").unwrap(), &small[..]);
        fs::remove_dir_all(&root).ok();
    }

    /// A **zero-length** file: the edge `SMALL_FILE_READ_THRESHOLD` moves
    /// across a code path, checked to move nothing observable.
    ///
    /// `0 <= threshold` is always true, so an empty file now takes the read
    /// arm where it used to be mapped. It is worth pinning because the
    /// obvious guess about the old behaviour is wrong: `mmap(2)` rejects a
    /// zero length, but `memmap2::Mmap::map` special-cases it and hands back
    /// an empty mapping rather than an error -- so the two arms *already*
    /// agreed and this change kept them agreeing. Asserted rather than
    /// assumed, because a reviewer of this batch predicted the opposite and
    /// only running it settled which.
    #[test]
    fn a_zero_length_file_reads_as_empty_on_every_backend() {
        let root = tempdir();
        index_output::write_all_bytes(&root, "_0.nvm", b"").unwrap();

        // The read arm, which an empty file now takes.
        let read = MmapDirectory::open(&root).open("_0.nvm").unwrap();
        assert!(matches!(read, Input::Owned(_)), "{read:?}");
        assert!(read.is_empty());

        // The mapping arm, which it used to take -- still `Ok`, still empty.
        let mapped = MmapDirectory::with_read_threshold(&root, 0)
            .open("_0.nvm")
            .expect("memmap2 maps a zero-length file rather than refusing it");
        assert!(matches!(mapped, Input::Mapped(_)), "{mapped:?}");
        assert!(mapped.is_empty());

        // And the copying backend, unchanged throughout.
        assert!(FsDirectory::open(&root).open("_0.nvm").unwrap().is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn pending_segments_file_name_matches_file_name_from_generation() {
        assert_eq!(pending_segments_file_name(-1), None);
        assert_eq!(
            pending_segments_file_name(0),
            Some("pending_segments".to_string())
        );
        assert_eq!(
            pending_segments_file_name(1),
            Some("pending_segments_1".to_string())
        );
        // base-36, same radix as `segments_N`.
        assert_eq!(
            pending_segments_file_name(10),
            Some("pending_segments_a".to_string())
        );
    }

    /// The whole point of the pending name: `getLastCommitGeneration` must not
    /// see it, so a half-written commit can never become the current one.
    #[test]
    fn a_pending_segments_file_is_invisible_to_the_commit_generation_scan() {
        let files = vec![
            "segments_1".to_string(),
            "pending_segments_2".to_string(),
            "_0.si".to_string(),
        ];
        assert_eq!(last_commit_generation(&files).unwrap(), 1);
    }

    #[test]
    fn rename_publishes_a_file_under_a_new_name_and_delete_file_removes_it() {
        let root = tempdir();
        let dir = FsDirectory::open(&root);

        index_output::write_all_bytes(&root, "pending_segments_1", b"commit").unwrap();
        assert!(dir
            .list_all()
            .unwrap()
            .contains(&"pending_segments_1".to_string()));

        dir.rename("pending_segments_1", "segments_1").unwrap();
        dir.sync_meta_data().unwrap();
        let listed = dir.list_all().unwrap();
        assert!(!listed.contains(&"pending_segments_1".to_string()));
        assert_eq!(&*dir.open("segments_1").unwrap(), b"commit");

        dir.delete_file("segments_1").unwrap();
        assert!(dir.list_all().unwrap().is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rename_and_delete_file_surface_io_errors_for_a_missing_source() {
        let root = tempdir();
        let dir = MmapDirectory::open(&root);
        assert!(matches!(
            dir.rename("nope", "segments_1"),
            Err(Error::Io(_))
        ));
        assert!(matches!(dir.delete_file("nope"), Err(Error::Io(_))));
        // `sync_meta_data` is best-effort by design (not every platform lets a
        // directory be fsynced) -- it reports success either way.
        dir.sync_meta_data().unwrap();
        fs::remove_dir_all(&root).ok();
    }

    use lucene_util::test_support::TempDir;

    /// A scratch directory that removes itself when the test ends -- unless
    /// the test is panicking, in which case its bytes stay for inspection.
    fn tempdir() -> TempDir {
        TempDir::new("directory-write")
    }

    #[test]
    fn fs_directory_create_output_round_trips_through_open_and_list_all() {
        let root = tempdir();
        let dir = FsDirectory::open(&root);

        let mut out = dir.create_output("_0.si").unwrap();
        out.write_bytes(b"hello lucene-rust");
        let checksum = out.close().unwrap();
        assert_eq!(checksum, crc32fast::hash(b"hello lucene-rust") as u64);

        dir.sync(&["_0.si".to_string()]).unwrap();

        assert_eq!(dir.list_all().unwrap(), vec!["_0.si".to_string()]);
        let bytes = dir.open("_0.si").unwrap();
        assert_eq!(&*bytes, b"hello lucene-rust");

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn mmap_directory_create_output_round_trips_through_open() {
        let root = tempdir();
        let dir = MmapDirectory::open(&root);

        let mut out = dir.create_output("_0.si").unwrap();
        out.write_bytes(b"mmap round trip");
        out.close().unwrap();

        let bytes = dir.open("_0.si").unwrap();
        assert_eq!(&*bytes, b"mmap round trip");

        assert_eq!(dir.list_all().unwrap(), vec!["_0.si".to_string()]);
        dir.sync(&["_0.si".to_string()]).unwrap();

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn slice_input_over_a_real_file_has_independent_file_pointers() {
        // Writes a real file to a temp dir via FsDirectory/FsIndexOutput, then
        // slices it via SliceInput::slice_input the way a merge would slice a
        // sub-range of a real on-disk `.cfs`. Guards against any real
        // OS-file-handle-sharing bug that an in-memory-only test can't catch
        // (e.g. accidentally sharing one file's read position across slices).
        use crate::data_input::{DataInput, SliceInput};

        let root = tempdir();
        let dir = FsDirectory::open(&root);
        let mut out = dir.create_output("_0.cfs").unwrap();
        out.write_bytes(b"HEADER|firstpart|secondpart|FOOTER");
        out.close().unwrap();

        let bytes = dir.open("_0.cfs").unwrap();
        let root_input = SliceInput::new(&bytes);

        // "firstpart" starts at offset 7, "secondpart" at offset 17.
        let mut first = root_input.slice_input("first", 7, 9).unwrap();
        let mut second = root_input.slice_input("second", 17, 10).unwrap();

        // Interleave reads through both real-file-backed slices.
        let mut buf1 = [0u8; 4];
        let mut buf2 = [0u8; 4];
        first.read_bytes(&mut buf1).unwrap();
        second.read_bytes(&mut buf2).unwrap();
        assert_eq!(&buf1, b"firs");
        assert_eq!(&buf2, b"seco");

        let mut buf1b = [0u8; 5];
        let mut buf2b = [0u8; 6];
        first.read_bytes(&mut buf1b).unwrap();
        second.read_bytes(&mut buf2b).unwrap();
        assert_eq!(&buf1b, b"tpart");
        assert_eq!(&buf2b, b"ndpart");

        // Both slices are now fully consumed; further reads are Eof, not a
        // leak into the other slice's or the footer's bytes.
        assert!(first.read_byte().is_err());
        assert!(second.read_byte().is_err());

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn fs_directory_open_nonexistent_file_is_io_error() {
        let dir = FsDirectory::open("/nonexistent-lucene-rust-test-path");
        assert!(matches!(dir.open("whatever"), Err(Error::Io(_))));
    }

    #[test]
    fn fs_directory_list_all_nonexistent_dir_is_io_error() {
        let dir = FsDirectory::open("/nonexistent-lucene-rust-test-path");
        assert!(matches!(dir.list_all(), Err(Error::Io(_))));
    }

    #[test]
    fn mmap_directory_open_nonexistent_file_is_io_error() {
        let dir = MmapDirectory::open("/nonexistent-lucene-rust-test-path");
        assert!(matches!(dir.open("whatever"), Err(Error::Io(_))));
    }

    #[test]
    fn read_latest_commit_finds_highest_generation_segments_file() {
        let root = tempdir();
        let dir = FsDirectory::open(&root);
        index_output::write_all_bytes(&root, "segments_1", b"old").unwrap();
        index_output::write_all_bytes(&root, "segments_2", b"newest").unwrap();

        let (generation, bytes) = read_latest_commit(&dir).unwrap();
        assert_eq!(generation, 2);
        assert_eq!(&*bytes, b"newest");

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn read_latest_commit_rejects_an_unparsable_segments_file_name() {
        // Java's getLastCommitGeneration lets generationFromSegmentsFileName's
        // exception escape here. Skipping the bad name instead would open
        // segments_1 and silently drop whatever segments_2 committed.
        let files = vec![
            "segments_1".to_string(),
            "segments_zzzzzzzzzzzzz".to_string(),
        ];
        assert!(matches!(
            read_latest_commit(&ListingOnlyDir(files.clone())),
            Err(Error::Corrupted(_))
        ));
        // ... and the scan the write path shares with it is equally strict:
        // reporting generation 1 here would let `IndexWriter` create a fresh
        // index over a directory that does hold a commit.
        assert!(matches!(
            last_commit_generation(&files),
            Err(Error::Corrupted(_))
        ));
    }

    #[test]
    fn read_latest_commit_ignores_the_legacy_pointer_file() {
        let root = tempdir();
        let dir = FsDirectory::open(&root);
        index_output::write_all_bytes(&root, "segments.gen", b"legacy").unwrap();
        index_output::write_all_bytes(&root, "segments_1", b"real").unwrap();
        let (generation, bytes) = read_latest_commit(&dir).unwrap();
        assert_eq!(generation, 1);
        assert_eq!(&*bytes, b"real");
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn read_latest_commit_no_segments_file_is_corrupted_error() {
        assert!(matches!(
            read_latest_commit(&ListingOnlyDir(vec!["_0.si".to_string()])),
            Err(Error::Corrupted(_))
        ));
    }

    /// Deletes that fail with anything but "no such file" become pending, as
    /// on Windows: hidden from listings and reads, retried later, and
    /// resurrected by a create or a rename onto the name. Here the failure
    /// is `EISDIR` -- `remove_file` on a directory -- which is portable to
    /// every Unix and needs no permissions trickery (tests run as root).
    #[test]
    fn a_failed_delete_becomes_pending_until_it_can_be_retried() {
        for backend in 0..2 {
            let root = tempdir();
            let dir: Box<dyn Directory> = if backend == 0 {
                Box::new(FsDirectory::open(&root))
            } else {
                Box::new(MmapDirectory::open(&root))
            };
            fs::create_dir(root.join("_0.cfs")).unwrap();
            index_output::write_all_bytes(&root, "_1.si", b"x").unwrap();
            dir.delete_file("_0.cfs").unwrap();
            assert_eq!(dir.list_all().unwrap(), vec!["_1.si".to_string()]);
            assert!(dir.open("_0.cfs").unwrap_err().is_no_such_file());
            assert!(dir.file_length("_0.cfs").unwrap_err().is_no_such_file());
            assert!(dir.delete_file("_0.cfs").unwrap_err().is_no_such_file());
            assert!(dir.rename("_0.cfs", "x").unwrap_err().is_no_such_file());
            // Still undeletable: still pending.
            assert_eq!(
                dir.pending_deletions().unwrap(),
                BTreeSet::from(["_0.cfs".to_string()])
            );
            // Once deletable, the next retry takes it.
            replace_dir_with_file(&root, "_0.cfs");
            assert!(dir.pending_deletions().unwrap().is_empty());
            assert!(!root.join("_0.cfs").exists());
            assert_eq!(dir.file_length("_1.si").unwrap(), 1);
        }
    }

    /// Makes a directory that failed to delete deletable.
    fn replace_dir_with_file(root: &Path, name: &str) {
        fs::remove_dir(root.join(name)).unwrap();
        fs::write(root.join(name), b"old").unwrap();
    }

    /// Leaves `name` pending in `dir`, deletable at the next retry.
    fn make_pending(dir: &FsDirectory, root: &Path, name: &str) {
        fs::create_dir(root.join(name)).unwrap();
        dir.delete_file(name).unwrap();
        assert!(dir.core.is_pending(name));
        replace_dir_with_file(root, name);
    }

    #[test]
    fn pending_deletes_are_retried_by_later_operations_and_on_drop() {
        let root = tempdir();
        let dir = FsDirectory::open(&root);
        // One pending file: the very next mutating op retries it.
        make_pending(&dir, &root, "a");
        dir.sync_meta_data().unwrap();
        assert!(!root.join("a").exists());

        make_pending(&dir, &root, "e");
        index_output::write_all_bytes(&root, "s", b"s").unwrap();
        dir.sync(&["s".to_string()]).unwrap();
        assert!(!root.join("e").exists());

        // Creating over a pending name brings it back to life.
        make_pending(&dir, &root, "b");
        let mut out = dir.create_output("b").unwrap();
        out.write_bytes(b"new");
        out.close().unwrap();
        assert_eq!(&*dir.open("b").unwrap(), b"new");
        assert!(dir.pending_deletions().unwrap().is_empty());

        // So does a rename onto it.
        make_pending(&dir, &root, "c");
        dir.rename("b", "c").unwrap();
        assert_eq!(&*dir.open("c").unwrap(), b"new");

        // A resurrection whose retry still fails leaves the name to the
        // create, which then fails on it -- and it is no longer pending.
        fs::create_dir(root.join("d")).unwrap();
        dir.delete_file("d").unwrap();
        assert!(dir.create_output("d").is_err());
        assert!(dir.pending_deletions().unwrap().is_empty());
        fs::remove_dir(root.join("d")).unwrap();

        // A pending file that vanished on its own is an error on retry
        // outside Windows (Java's `NoSuchFileException`), and is forgotten.
        fs::create_dir(root.join("g")).unwrap();
        dir.delete_file("g").unwrap();
        fs::remove_dir(root.join("g")).unwrap();
        assert!(dir.delete_pending_files().unwrap_err().is_no_such_file());
        assert!(dir.pending_deletions().unwrap().is_empty());

        // Drop is `close()`: a last retry.
        make_pending(&dir, &root, "f");
        drop(dir);
        assert!(!root.join("f").exists());
    }

    #[test]
    fn create_temp_output_never_clobbers() {
        let root = tempdir();
        let dir = MmapDirectory::open(&root);
        index_output::write_all_bytes(&root, "_0_sort_0.tmp", b"taken").unwrap();
        let out = dir.create_temp_output("_0", "sort").unwrap();
        assert_eq!(crate::IndexOutput::name(&out), "_0_sort_1.tmp");
        out.close().unwrap();
        assert_eq!(&*dir.open("_0_sort_0.tmp").unwrap(), b"taken");
        let fs_dir = FsDirectory::open(&root);
        let out = fs_dir.create_temp_output("_0", "sort").unwrap();
        assert_eq!(crate::IndexOutput::name(&out), "_0_sort_2.tmp");
        // A missing directory is an error, not an endless retry.
        assert!(FsDirectory::open(root.join("missing"))
            .create_temp_output("_0", "x")
            .is_err());
        assert_eq!(temp_file_name("_5", "fdt", 35), "_5_fdt_z.tmp");
        assert_eq!(
            temp_file_name("p", "s", u64::MAX),
            temp_file_name("p", "s", i64::MAX as u64)
        );
    }

    #[test]
    fn fs_directories_expose_their_path_and_lock_factory() {
        let root = tempdir();
        let dir = FsDirectory::with_lock_factory(&root, Arc::new(crate::NoLockFactory));
        assert_eq!(dir.directory(), root.path());
        assert_eq!(dir.fs_directory_path(), Some(root.path()));
        assert!(format!("{:?}", dir.lock_factory()).contains("NoLockFactory"));
        let _a = dir.obtain_lock("write.lock").unwrap();
        let _b = dir.obtain_lock("write.lock").unwrap();

        let mmap = MmapDirectory::with_lock_factory(&root, Arc::new(crate::NoLockFactory));
        assert_eq!(mmap.directory(), root.path());
        assert_eq!(mmap.fs_directory_path(), Some(root.path()));
        mmap.delete_pending_files().unwrap();
        let native = MmapDirectory::open(&root);
        let held = native.obtain_lock("write.lock").unwrap();
        assert!(matches!(
            FsDirectory::open(&root).obtain_lock("write.lock"),
            Err(Error::LockObtainFailed(_))
        ));
        drop(held);
        let base = BaseDirectory::new(Arc::new(crate::NoLockFactory));
        assert!(format!("{base:?}").contains("NoLockFactory"));
        base.obtain_lock(&native, "x").unwrap();
    }

    #[test]
    fn references_and_arcs_are_directories() {
        let root = tempdir();
        let dir = Arc::new(FsDirectory::open(&root));
        let by_ref = &*dir;
        let as_ref: &dyn Directory = &by_ref;
        let shared: &dyn Directory = &dir;
        for d in [as_ref, shared] {
            let mut out = d.create_output("a").unwrap();
            out.write_bytes(b"1");
            out.close().unwrap();
            assert_eq!(d.file_length("a").unwrap(), 1);
            assert_eq!(&*d.open("a").unwrap(), b"1");
            d.sync(&["a".to_string()]).unwrap();
            d.rename("a", "b").unwrap();
            d.copy_from(&*dir, "b", "c").unwrap();
            d.delete_file("b").unwrap();
            d.delete_file("c").unwrap();
            d.sync_meta_data().unwrap();
            d.create_temp_output("t", "s").unwrap().close().unwrap();
            assert!(d.pending_deletions().unwrap().is_empty());
            assert_eq!(d.fs_directory_path(), Some(root.path()));
            let lock = d.obtain_lock("write.lock").unwrap();
            lock.close().unwrap();
            assert!(!d.list_all().unwrap().is_empty());
        }
    }

    /// A directory that implements only the required methods, over an
    /// in-memory one: exercises the trait's defaults.
    struct RequiredOnly<'a>(&'a crate::ByteBuffersDirectory);

    impl Directory for RequiredOnly<'_> {
        fn list_all(&self) -> Result<Vec<String>> {
            self.0.list_all()
        }
        fn open(&self, name: &str) -> Result<Input> {
            self.0.open(name)
        }
        fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
            self.0.create_output(name)
        }
        fn sync(&self, names: &[String]) -> Result<()> {
            self.0.sync(names)
        }
        fn rename(&self, source: &str, dest: &str) -> Result<()> {
            self.0.rename(source, dest)
        }
        fn delete_file(&self, name: &str) -> Result<()> {
            self.0.delete_file(name)
        }
        fn sync_meta_data(&self) -> Result<()> {
            self.0.sync_meta_data()
        }
        fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
            self.0.obtain_lock(name)
        }
    }

    #[test]
    fn trait_defaults_read_the_file_and_refuse_temp_outputs() {
        let bb = crate::ByteBuffersDirectory::new();
        let only = RequiredOnly(&bb);
        let err = only.create_temp_output("a", "b").unwrap_err();
        assert!(matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::Unsupported));
        assert!(only.pending_deletions().unwrap().is_empty());
        assert!(only.fs_directory_path().is_none());
        let mut out = only.create_output("f").unwrap();
        out.write_bytes(b"four");
        out.close().unwrap();
        assert_eq!(only.file_length("f").unwrap(), 4);
        assert_eq!(only.list_all().unwrap(), vec!["f"]);
        only.sync(&[]).unwrap();
        only.sync_meta_data().unwrap();
        only.copy_from(&bb, "f", "g").unwrap();
        only.rename("g", "h").unwrap();
        only.delete_file("h").unwrap();
        only.obtain_lock("l").unwrap();
    }
}
