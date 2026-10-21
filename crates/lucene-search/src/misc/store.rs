//! `org.apache.lucene.misc.store`: directory wrappers and filesystem
//! directories that Lucene keeps outside `core`.
//!
//! Java's `IOContext` decides what a wrapper does with a new output; the Rust
//! [`Directory`] trait carries no context, so the wrappers that depend on one
//! take it as an argument ([`IoContext`]) and their [`Directory`] impl is the
//! `DEFAULT` context.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use lucene_store::directory::Input;
use lucene_store::lock::Lock;
use lucene_store::{DataOutput, Directory, Error, FsDirectory, FsIndexOutput, IndexOutput, Result};

/// `IOContext.Context`: why an output is being written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoContext {
    /// `DEFAULT`
    Default,
    /// `FLUSH`: a new segment from buffered documents.
    Flush,
    /// `MERGE`: a segment being merged.
    Merge,
}

/// `ByteTrackingIndexOutput`: an output that adds the bytes written through
/// it to a shared counter when it is closed.
pub struct ByteTrackingIndexOutput {
    out: FsIndexOutput,
    byte_tracker: Arc<AtomicU64>,
    name: String,
}

impl ByteTrackingIndexOutput {
    /// `new ByteTrackingIndexOutput(out, byteTracker)`.
    pub fn new(out: FsIndexOutput, byte_tracker: Arc<AtomicU64>) -> Self {
        let name = format!("Byte tracking wrapper for: {}", out.name());
        Self {
            out,
            byte_tracker,
            name,
        }
    }

    /// `close()`: counts the file pointer, then closes the wrapped output.
    /// Returns the checksum, as [`FsIndexOutput::close`] does.
    ///
    /// # Errors
    /// Whatever closing the wrapped output fails with.
    pub fn close(self) -> Result<u64> {
        self.byte_tracker
            .fetch_add(self.out.file_pointer(), Ordering::SeqCst);
        self.out.close()
    }
}

impl DataOutput for ByteTrackingIndexOutput {
    fn write_byte(&mut self, b: u8) {
        self.out.write_byte(b);
    }

    fn write_bytes(&mut self, b: &[u8]) {
        self.out.write_bytes(b);
    }
}

impl IndexOutput for ByteTrackingIndexOutput {
    fn name(&self) -> &str {
        &self.name
    }

    fn file_pointer(&self) -> u64 {
        self.out.file_pointer()
    }

    fn checksum(&self) -> u64 {
        self.out.checksum()
    }
}

/// Implements [`Directory`] for a wrapper by forwarding every method to the
/// field `inner`, except the ones listed after the semicolon, which the
/// wrapper writes itself.
macro_rules! forward_directory {
    () => {
        fn list_all(&self) -> Result<Vec<String>> {
            self.inner.list_all()
        }
        fn open(&self, name: &str) -> Result<Input> {
            self.inner.open(name)
        }
        fn file_length(&self, name: &str) -> Result<u64> {
            self.inner.file_length(name)
        }
        fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
            self.inner.create_temp_output(prefix, suffix)
        }
        fn create_output_with_estimate(
            &self,
            name: &str,
            estimated_bytes: Option<u64>,
        ) -> Result<FsIndexOutput> {
            self.inner
                .create_output_with_estimate(name, estimated_bytes)
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
    };
}

/// `ByteWritesTrackingDirectoryWrapper`: counts the bytes written for
/// flushes and for merges, to measure the write amplification of an index.
pub struct ByteWritesTrackingDirectory<D> {
    inner: D,
    flushed_bytes: Arc<AtomicU64>,
    merged_bytes: Arc<AtomicU64>,
    /// `trackTempOutput`
    pub track_temp_output: bool,
}

impl<D: Directory> ByteWritesTrackingDirectory<D> {
    /// `new ByteWritesTrackingDirectoryWrapper(in)`.
    pub fn new(inner: D) -> Self {
        Self::with_temp_tracking(inner, false)
    }

    /// `new ByteWritesTrackingDirectoryWrapper(in, trackTempOutput)`.
    pub fn with_temp_tracking(inner: D, track_temp_output: bool) -> Self {
        Self {
            inner,
            flushed_bytes: Arc::new(AtomicU64::new(0)),
            merged_bytes: Arc::new(AtomicU64::new(0)),
            track_temp_output,
        }
    }

    /// `createOutput(name, ioContext)`: tracked for `FLUSH` and `MERGE`,
    /// returned as the wrapped directory made it for `DEFAULT`.
    ///
    /// # Errors
    /// Whatever the wrapped directory fails with.
    pub fn create_output_for(&self, name: &str, context: IoContext) -> Result<TrackedOutput> {
        Ok(self.track(self.inner.create_output(name)?, context))
    }

    /// `createTempOutput(prefix, suffix, ioContext)`: tracked only when
    /// `track_temp_output` is set.
    ///
    /// # Errors
    /// Whatever the wrapped directory fails with.
    pub fn create_temp_output_for(
        &self,
        prefix: &str,
        suffix: &str,
        context: IoContext,
    ) -> Result<TrackedOutput> {
        let out = self.inner.create_temp_output(prefix, suffix)?;
        Ok(if self.track_temp_output {
            self.track(out, context)
        } else {
            TrackedOutput::Plain(out)
        })
    }

    fn track(&self, out: FsIndexOutput, context: IoContext) -> TrackedOutput {
        match context {
            IoContext::Flush => TrackedOutput::Tracked(ByteTrackingIndexOutput::new(
                out,
                Arc::clone(&self.flushed_bytes),
            )),
            IoContext::Merge => TrackedOutput::Tracked(ByteTrackingIndexOutput::new(
                out,
                Arc::clone(&self.merged_bytes),
            )),
            IoContext::Default => TrackedOutput::Plain(out),
        }
    }

    /// `getFlushedBytes()`
    pub fn flushed_bytes(&self) -> u64 {
        self.flushed_bytes.load(Ordering::SeqCst)
    }

    /// `getMergedBytes()`
    pub fn merged_bytes(&self) -> u64 {
        self.merged_bytes.load(Ordering::SeqCst)
    }
}

/// An output [`ByteWritesTrackingDirectory`] made: counted or not.
pub enum TrackedOutput {
    /// A `FLUSH` or `MERGE` output.
    Tracked(ByteTrackingIndexOutput),
    /// A `DEFAULT` output, or an untracked temporary one.
    Plain(FsIndexOutput),
}

impl TrackedOutput {
    /// Closes the output, counting its bytes if it is tracked.
    ///
    /// # Errors
    /// Whatever closing the output fails with.
    pub fn close(self) -> Result<u64> {
        match self {
            Self::Tracked(o) => o.close(),
            Self::Plain(o) => o.close(),
        }
    }
}

impl DataOutput for TrackedOutput {
    fn write_byte(&mut self, b: u8) {
        match self {
            Self::Tracked(o) => o.write_byte(b),
            Self::Plain(o) => o.write_byte(b),
        }
    }

    fn write_bytes(&mut self, b: &[u8]) {
        match self {
            Self::Tracked(o) => o.write_bytes(b),
            Self::Plain(o) => o.write_bytes(b),
        }
    }
}

impl<D: Directory> Directory for ByteWritesTrackingDirectory<D> {
    forward_directory!();

    /// The `DEFAULT` context: not tracked. Use
    /// [`ByteWritesTrackingDirectory::create_output_for`] for the others.
    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.inner.create_output(name)
    }
}

/// `HardlinkCopyDirectoryWrapper`: `copyFrom` hard-links the file when both
/// directories are filesystem directories, and copies its bytes when it
/// cannot (another filesystem, or one without hard links).
pub struct HardlinkCopyDirectory<D> {
    inner: D,
}

impl<D: Directory> HardlinkCopyDirectory<D> {
    /// `new HardlinkCopyDirectoryWrapper(in)`.
    pub fn new(inner: D) -> Self {
        Self { inner }
    }
}

impl<D: Directory> Directory for HardlinkCopyDirectory<D> {
    forward_directory!();

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.inner.create_output(name)
    }

    fn copy_from(&self, from: &dyn Directory, src: &str, dest: &str) -> Result<()> {
        let mut link_failure = None;
        if let (Some(from_path), Some(to_path)) =
            (from.fs_directory_path(), self.inner.fs_directory_path())
        {
            let source = from_path.join(src);
            if std::fs::File::open(&source).is_ok() {
                // Whatever the failure (no hard links, another filesystem, a
                // missing or existing file), the byte copy is tried and
                // reports a true error condition itself.
                match std::fs::hard_link(&source, to_path.join(dest)) {
                    Ok(()) => return Ok(()),
                    Err(e) => link_failure = Some(e),
                }
            }
        }
        self.inner.copy_from(from, src, dest).map_err(|e| {
            if let Some(link) = link_failure {
                Error::Io(std::io::Error::other(format!(
                    "{e} (hard link failed first: {link})"
                )))
            } else {
                e
            }
        })
    }
}

/// `RAFDirectory`: Java's `FSDirectory` over `java.io.RandomAccessFile`, kept
/// for its poor concurrency. This port has one filesystem directory, so this
/// is that directory under the other's name.
pub struct RafDirectory {
    inner: FsDirectory,
}

impl RafDirectory {
    /// `new RAFDirectory(path)`.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            inner: FsDirectory::open(path.as_ref()),
        }
    }
}

impl Directory for RafDirectory {
    forward_directory!();

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.inner.create_output(name)
    }
}

/// `DirectIODirectory.DEFAULT_MERGE_BUFFER_SIZE`
pub const DEFAULT_MERGE_BUFFER_SIZE: usize = 256 * 1024;
/// `DirectIODirectory.DEFAULT_MIN_BYTES_DIRECT`
pub const DEFAULT_MIN_BYTES_DIRECT: u64 = 10 * 1024 * 1024;

/// `DirectIODirectory`: merges bypass the OS cache for files of at least
/// `min_bytes_direct` bytes. Reading and writing here go through the wrapped
/// filesystem directory (no `O_DIRECT`): the decision [`Self::use_direct_io`]
/// is ported, the unbuffered I/O it selects is not.
pub struct DirectIoDirectory {
    inner: FsDirectory,
    merge_buffer_size: usize,
    min_bytes_direct: u64,
}

impl DirectIoDirectory {
    /// `new DirectIODirectory(delegate)`.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self::with(path, DEFAULT_MERGE_BUFFER_SIZE, DEFAULT_MIN_BYTES_DIRECT)
    }

    /// `new DirectIODirectory(delegate, mergeBufferSize, minBytesDirect)`.
    pub fn with(path: impl AsRef<Path>, merge_buffer_size: usize, min_bytes_direct: u64) -> Self {
        Self {
            inner: FsDirectory::open(path.as_ref()),
            merge_buffer_size,
            min_bytes_direct,
        }
    }

    /// `mergeBufferSize`
    pub fn merge_buffer_size(&self) -> usize {
        self.merge_buffer_size
    }

    /// `useDirectIO(name, context, fileLength)`: only a merge, and only of a
    /// file of at least `min_bytes_direct` bytes.
    pub fn use_direct_io(&self, context: IoContext, file_length: Option<u64>) -> bool {
        context == IoContext::Merge && file_length.is_some_and(|len| len >= self.min_bytes_direct)
    }
}

impl Directory for DirectIoDirectory {
    forward_directory!();

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.inner.create_output(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_util::test_support::TempDir;

    #[test]
    fn tracking_counts_flush_and_merge_but_not_default() {
        let tmp = TempDir::new("track");
        let dir = ByteWritesTrackingDirectory::new(FsDirectory::open(tmp.path()));
        for (name, ctx, n) in [
            ("a", IoContext::Flush, 10usize),
            ("b", IoContext::Merge, 7),
            ("c", IoContext::Default, 5),
            ("d", IoContext::Flush, 3),
        ] {
            let mut o = dir.create_output_for(name, ctx).unwrap();
            o.write_bytes(&vec![1u8; n]);
            o.close().unwrap();
        }
        assert_eq!(dir.flushed_bytes(), 13);
        assert_eq!(dir.merged_bytes(), 7);
    }

    #[test]
    fn temp_outputs_are_tracked_only_on_request() {
        let tmp = TempDir::new("track-temp");
        let off = ByteWritesTrackingDirectory::new(FsDirectory::open(tmp.path()));
        let mut o = off
            .create_temp_output_for("p", "s", IoContext::Merge)
            .unwrap();
        o.write_bytes(&[1, 2, 3]);
        o.close().unwrap();
        assert_eq!(off.merged_bytes(), 0);
        let on =
            ByteWritesTrackingDirectory::with_temp_tracking(FsDirectory::open(tmp.path()), true);
        let mut o = on
            .create_temp_output_for("p", "s", IoContext::Merge)
            .unwrap();
        o.write_bytes(&[1, 2, 3]);
        o.close().unwrap();
        assert_eq!(on.merged_bytes(), 3);
    }

    #[test]
    fn tracking_output_reports_its_wrapper_name_and_position() {
        let tmp = TempDir::new("track-name");
        let tracker = Arc::new(AtomicU64::new(0));
        let inner = FsDirectory::open(tmp.path()).create_output("x").unwrap();
        let mut o = ByteTrackingIndexOutput::new(inner, Arc::clone(&tracker));
        o.write_byte(9);
        assert_eq!(o.name(), "Byte tracking wrapper for: x");
        assert_eq!(o.file_pointer(), 1);
        let _ = o.checksum();
        o.close().unwrap();
        assert_eq!(tracker.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn hardlink_copy_links_between_filesystem_directories() {
        let a = TempDir::new("hl-a");
        let b = TempDir::new("hl-b");
        let from = FsDirectory::open(a.path());
        let mut o = from.create_output("f").unwrap();
        o.write_bytes(b"hello");
        o.close().unwrap();
        let to = HardlinkCopyDirectory::new(FsDirectory::open(b.path()));
        to.copy_from(&from, "f", "g").unwrap();
        assert_eq!(&*to.open("g").unwrap(), b"hello");
        // The destination exists now: Java's wrapper returns the link's
        // FileAlreadyExistsException as suppressed and the byte copy, which
        // is CREATE_NEW, fails with it too. Directory.copyFrom deletes the
        // destination on any failure, a failed createOutput included, so the
        // earlier copy is gone.
        let err = to.copy_from(&from, "f", "g").unwrap_err();
        assert!(format!("{err}").contains("File exists"), "{err}");
        assert!(to.open("g").is_err());
        to.copy_from(&from, "f", "g").unwrap();
        assert_eq!(&*to.open("g").unwrap(), b"hello");
        assert!(to.copy_from(&from, "missing", "h").is_err());
    }

    #[test]
    fn hardlink_copy_copies_bytes_between_other_directories() {
        let from = lucene_store::ByteBuffersDirectory::new();
        let mut o = from.create_output("f").unwrap();
        o.write_bytes(b"xyz");
        o.close().unwrap();
        let b = TempDir::new("hl-c");
        let to = HardlinkCopyDirectory::new(FsDirectory::open(b.path()));
        to.copy_from(&from, "f", "g").unwrap();
        assert_eq!(&*to.open("g").unwrap(), b"xyz");
        assert_eq!(to.list_all().unwrap(), vec!["g".to_string()]);
    }

    #[test]
    fn direct_io_is_for_big_merged_files() {
        let tmp = TempDir::new("dio");
        let d = DirectIoDirectory::with(tmp.path(), 1024, 100);
        assert_eq!(d.merge_buffer_size(), 1024);
        assert!(d.use_direct_io(IoContext::Merge, Some(100)));
        assert!(!d.use_direct_io(IoContext::Merge, Some(99)));
        assert!(!d.use_direct_io(IoContext::Merge, None));
        assert!(!d.use_direct_io(IoContext::Flush, Some(1000)));
        let mut o = d.create_output("f").unwrap();
        o.write_bytes(b"abc");
        o.close().unwrap();
        assert_eq!(d.file_length("f").unwrap(), 3);
        assert_eq!(
            DirectIoDirectory::new(tmp.path()).merge_buffer_size(),
            262144
        );
    }

    #[test]
    fn raf_directory_is_a_filesystem_directory() {
        let tmp = TempDir::new("raf");
        let d = RafDirectory::new(tmp.path());
        let mut o = d.create_output("f").unwrap();
        o.write_bytes(b"abc");
        o.close().unwrap();
        assert_eq!(&*d.open("f").unwrap(), b"abc");
        assert_eq!(d.fs_directory_path(), Some(tmp.path()));
    }
}
