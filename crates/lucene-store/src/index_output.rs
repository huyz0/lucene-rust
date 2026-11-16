//! Port of `org.apache.lucene.store.IndexOutput` / `OutputStreamIndexOutput` —
//! the write-side counterpart of [`crate::directory::Input`]/`IndexInput`,
//! backed by a real `std::fs::File` so a Rust program can write files that
//! Java's `FSDirectory.open(path)` opens directly -- or, for
//! `ByteBuffersDirectory`, by memory (`ByteBuffersIndexOutput`).
//!
//! Scope of this slice (see PLAN.md Phase 5 / docs/parity.md): a single-file
//! sequential output plus `Directory::sync` for the fsync-before-durable
//! contract Lucene itself uses (`Directory.sync(Collection<String>)`,
//! `IndexWriter` calls it on every new segment's files before referencing
//! them from a `segments_N` commit). Writing `segments_N` itself — the
//! generation/checksum-of-checksums commit lifecycle — is explicitly out of
//! scope here; see the parity matrix.
//!
//! [`DataOutput`]'s methods are infallible by signature (matching the
//! existing in-memory `VecDataOutput`, which can never fail, so changing the
//! trait to return `Result` would ripple through every already-committed
//! codec writer for no benefit there). Real file I/O for FS-backed
//! output can fail, so [`FsIndexOutput`] uses a **sticky error**: the first
//! I/O error from a `write_byte`/`write_bytes` call is latched and every
//! subsequent write becomes a no-op; the latched error surfaces from
//! [`FsIndexOutput::close`], mirroring how a `BufWriter` drop swallowing a
//! flush error would otherwise silently lose data.
//!
//! Test seam (feature `test-util` only): `FsIndexOutput::set_faults`
//! installs an `OutputFaults` hook, consulted before every write and at
//! close -- what the test framework's `MockIndexOutputWrapper` needs to throw
//! random and deterministic I/O errors and to fake a full disk. A fault
//! latches like a real write error, so it surfaces from `close`. The hook is
//! dropped with the output, closed or not, which is how a mock directory
//! counts the output's handle as released.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::data_output::DataOutput;
use crate::error::{Error, Result};
use crate::rate_limiter::RateLimiter;

/// Write-side counterpart of Lucene's `IndexOutput`: `getName()`,
/// `getFilePointer()`, `getChecksum()`.
pub trait IndexOutput: DataOutput {
    /// Port of `IndexOutput.getName()`.
    fn name(&self) -> &str;

    /// Port of `IndexOutput.getFilePointer()`: bytes written so far.
    fn file_pointer(&self) -> u64;

    /// Port of `IndexOutput.getChecksum()`: running CRC32 over every byte
    /// written so far (not yet flushed/synced to disk).
    fn checksum(&self) -> u64;
}

/// `FSDirectory.FSIndexOutput.CHUNK_SIZE`, which is also the buffer size
/// `OutputStreamIndexOutput` is built with: bytes reach the file in writes of
/// at most this size, and a write at least this large bypasses the buffer.
pub const CHUNK_SIZE: usize = 8192;

/// A callback that receives an in-memory output's bytes when it is closed:
/// `ByteBuffersIndexOutput`'s `onClose`.
pub(crate) type PublishFn = Box<dyn FnOnce(Vec<u8>) + Send>;

/// Where a third-party [`crate::Directory`]'s output bytes go: the write
/// half of a pluggable store (a blob store's upload, an encrypting
/// wrapper's cipher stream). [`FsIndexOutput::from_sink`] wraps one in the
/// output type every codec writer writes through, with the same buffering
/// (writes reach the sink in pieces of at most [`CHUNK_SIZE`] bytes, or one
/// larger piece for a large write), CRC32 and rate limiting a file gets.
///
/// Lucene's `IndexOutput` is an open class a directory subclasses; here the
/// output type is one concrete struct (codec writers are monomorphic over
/// it, which is what keeps `write_byte` an inlined push), and this trait is
/// the seam instead.
pub trait OutputSink: Send {
    /// Appends `bytes`. An error is latched by the output, later writes are
    /// dropped, and it is returned by [`FsIndexOutput::close`].
    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()>;

    /// Called once by [`FsIndexOutput::close`] after the last write, when no
    /// write failed: flush, finalize (a cipher's last block, an upload's
    /// commit) and publish the file. An error fails the close. It is not
    /// called when a write failed, and a sink dropped without it must leave
    /// no visible file (or a file the directory can still delete).
    fn finish(self: Box<Self>) -> std::io::Result<()>;
}

/// Where an [`FsIndexOutput`]'s bytes go.
enum Sink {
    /// A real file: `FSDirectory.FSIndexOutput`.
    File { file: File, path: PathBuf },
    /// A pluggable directory's [`OutputSink`], buffered like a file.
    Stream(Box<dyn OutputSink>),
    /// Kept in memory and handed over on close: `ByteBuffersIndexOutput`,
    /// which is what `ByteBuffersDirectory` (and through it
    /// `NRTCachingDirectory`'s cache) hands out.
    Memory { publish: Option<PublishFn> },
}

/// `RateLimitedIndexOutput`'s state, when an output is throttled: see
/// [`FsIndexOutput::set_rate_limiter`].
struct RateLimit {
    limiter: Arc<dyn RateLimiter>,
    bytes_since_last_pause: u64,
    current_min_pause_check_bytes: u64,
}

impl RateLimit {
    /// `RateLimitedIndexOutput.checkRate()`, run before each write as Java's
    /// `writeByte`/`writeBytes` do.
    fn check_rate(&mut self, bytes: usize) {
        self.bytes_since_last_pause = self.bytes_since_last_pause.saturating_add(bytes as u64);
        if self.bytes_since_last_pause > self.current_min_pause_check_bytes {
            self.limiter.pause(self.bytes_since_last_pause);
            self.bytes_since_last_pause = 0;
            self.current_min_pause_check_bytes = self.limiter.min_pause_check_bytes();
        }
    }
}

/// The output every [`crate::Directory`] hands out.
///
/// Despite the name, not always a file: [`crate::ByteBuffersDirectory`]'s
/// outputs are memory-backed. The name is kept because it is the type every
/// codec writer in the workspace already names; what differs between the two
/// is only where [`FsIndexOutput::close`] delivers the bytes.
///
/// File-backed outputs are buffered in [`CHUNK_SIZE`] pieces so small
/// `write_byte` calls don't each incur a syscall.
pub struct FsIndexOutput {
    name: String,
    sink: Sink,
    buf: Vec<u8>,
    /// How much of `buf` `crc` already covers. The CRC is computed lazily, a
    /// buffer at a time (as Java's `CheckedOutputStream` under a
    /// `BufferedOutputStream` does), not per write call: a file sink hashes
    /// `buf` just before flushing it (so this stays 0), an in-memory sink
    /// whenever `CHUNK_SIZE` unhashed bytes have piled up.
    crc_pos: usize,
    bytes_written: u64,
    crc: crc32fast::Hasher,
    pending_err: Option<std::io::Error>,
    /// A write fault the hook injected (feature `test-util`): later writes
    /// are dropped, as Java's caller stops at the exception, and `close`
    /// returns it after flushing what was written before it -- the bytes a
    /// Java `writeBytes` wrote before throwing are in the file.
    #[cfg(any(test, feature = "test-util"))]
    fault_err: Option<std::io::Error>,
    /// `Some` once [`FsIndexOutput::set_rate_limiter`] throttled this output.
    rate_limit: Option<Box<RateLimit>>,
    /// `Some` once [`FsIndexOutput::set_faults`] installed a test hook.
    #[cfg(any(test, feature = "test-util"))]
    faults: Option<Box<dyn OutputFaults>>,
}

/// What an [`OutputFaults`] hook decides about one write.
#[cfg(any(test, feature = "test-util"))]
#[derive(Debug)]
pub enum WriteVerdict {
    /// Write the bytes.
    Write,
    /// Write the first `keep` bytes (all of them when `keep` is the write's
    /// length), then fail with `error`: latched, and returned by `close`.
    /// `MockIndexOutputWrapper`'s fake disk full writes the space left; its
    /// random exception fires after the first write went through.
    Fail { keep: usize, error: std::io::Error },
}

/// A test hook on an [`FsIndexOutput`] (feature `test-util`): the seam the
/// test framework's `MockIndexOutputWrapper` wraps an output through, since
/// every directory hands out this one concrete type.
#[cfg(any(test, feature = "test-util"))]
pub trait OutputFaults: Send {
    /// Called before each write of `len` bytes to the output named `name`.
    fn before_write(&mut self, name: &str, len: usize) -> WriteVerdict;
    /// Called after a write the hook let through, with the bytes written
    /// (`ThrottledIndexOutput` times the write between the two calls).
    fn after_write(&mut self, _name: &str, _len: usize) {}
    /// Called as the output closes, before its bytes are flushed.
    fn on_close(&mut self, name: &str) -> CloseVerdict;
}

/// What an [`OutputFaults`] hook decides about a close.
#[cfg(any(test, feature = "test-util"))]
#[derive(Debug)]
pub enum CloseVerdict {
    /// Flush and close.
    Close,
    /// Flush, then fail: Java's try-with-resources in
    /// `MockIndexOutputWrapper.close` closes the delegate before the
    /// failure's exception propagates.
    FailAfterFlush(std::io::Error),
    /// Fail without flushing the buffered bytes: the output was already
    /// force-closed, as `MockDirectoryWrapper.crash()` closes every open
    /// handle before it damages the unsynced files.
    FailWithoutFlush(std::io::Error),
}

impl FsIndexOutput {
    /// Port of `Directory.createOutput(name, context)`: creates (truncating
    /// any existing file of the same name) a new file at `root/name` for
    /// writing.
    ///
    /// Java's `FSDirectory.createOutput` opens with `CREATE_NEW` and so
    /// refuses an existing name: the directories create through
    /// [`FsIndexOutput::create_new`]. This truncating form is for a caller
    /// outside a directory that means to overwrite.
    pub fn create(root: &Path, name: &str) -> Result<Self> {
        let path = root.join(name);
        let file = File::create(&path)?;
        Ok(Self::over_file(name, file, path))
    }

    /// `FSIndexOutput(name, WRITE, CREATE_NEW)`: fails with
    /// `ErrorKind::AlreadyExists` when `root/name` exists.
    pub fn create_new(root: &Path, name: &str) -> Result<Self> {
        let path = root.join(name);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self::over_file(name, file, path))
    }

    /// An output named `name` whose bytes go to `sink`: what a third-party
    /// [`crate::Directory`]'s `create_output` returns. See [`OutputSink`].
    pub fn from_sink(name: &str, sink: Box<dyn OutputSink>) -> Self {
        Self::over_sink(name, Sink::Stream(sink))
    }

    fn over_file(name: &str, file: File, path: PathBuf) -> Self {
        Self::over_sink(name, Sink::File { file, path })
    }

    fn over_sink(name: &str, sink: Sink) -> Self {
        Self {
            name: name.to_string(),
            sink,
            buf: Vec::with_capacity(CHUNK_SIZE),
            crc_pos: 0,
            bytes_written: 0,
            crc: crc32fast::Hasher::new(),
            pending_err: None,
            #[cfg(any(test, feature = "test-util"))]
            fault_err: None,
            rate_limit: None,
            #[cfg(any(test, feature = "test-util"))]
            faults: None,
        }
    }

    /// `ByteBuffersIndexOutput`: every byte stays in memory until
    /// [`FsIndexOutput::close`] hands the whole content to `publish`.
    pub(crate) fn in_memory(name: &str, publish: PublishFn) -> Self {
        Self {
            name: name.to_string(),
            sink: Sink::Memory {
                publish: Some(publish),
            },
            buf: Vec::new(),
            crc_pos: 0,
            bytes_written: 0,
            crc: crc32fast::Hasher::new(),
            pending_err: None,
            #[cfg(any(test, feature = "test-util"))]
            fault_err: None,
            rate_limit: None,
            #[cfg(any(test, feature = "test-util"))]
            faults: None,
        }
    }

    /// Installs a test hook consulted before every write and at close
    /// (feature `test-util`; see [`OutputFaults`]). Replaces any earlier one.
    #[cfg(any(test, feature = "test-util"))]
    pub fn set_faults(&mut self, faults: Box<dyn OutputFaults>) {
        self.faults = Some(faults);
    }

    /// Port of `IndexOutput.close()`: flushes buffered bytes to the OS (or,
    /// for an in-memory output, publishes them to its directory) and returns
    /// the final CRC32 checksum. Does **not** fsync -- Lucene's own
    /// durability contract is that `IndexOutput.close()` merely hands bytes
    /// to the OS; a segment is only durable once its writer calls
    /// `Directory.sync(names)` on every file, which this crate exposes as
    /// [`crate::directory::Directory::sync`]. Returns the first latched I/O
    /// error, if any write since `create` failed.
    pub fn close(mut self) -> Result<u64> {
        #[cfg(any(test, feature = "test-util"))]
        let close_fault = match self.faults.as_deref_mut() {
            Some(faults) => match faults.on_close(&self.name) {
                CloseVerdict::Close => None,
                CloseVerdict::FailAfterFlush(e) => Some(e),
                CloseVerdict::FailWithoutFlush(e) => return Err(Error::Io(e)),
            },
            None => None,
        };
        if let Some(e) = self.pending_err.take() {
            return Err(Error::Io(e));
        }
        let checksum = self.checksum();
        match self.sink {
            Sink::File { mut file, .. } => {
                file.write_all(&self.buf)?;
                file.flush()?;
            }
            Sink::Stream(mut sink) => {
                if !self.buf.is_empty() {
                    sink.write_all(&self.buf)?;
                }
                sink.finish()?;
            }
            Sink::Memory { publish } => {
                if let Some(publish) = publish {
                    publish(std::mem::take(&mut self.buf));
                }
            }
        }
        #[cfg(any(test, feature = "test-util"))]
        if let Some(e) = self.fault_err.take().or(close_fault) {
            return Err(Error::Io(e));
        }
        Ok(checksum)
    }

    /// `new RateLimitedIndexOutput(limiter, this)`, in place: from here on
    /// every write first lets `limiter` pause the caller, so the bytes written
    /// stay within its rate. `IndexWriter` throttles a merge's outputs this
    /// way (its merge directory's `createOutput`).
    ///
    /// In place rather than a wrapper type because every
    /// [`crate::Directory`] hands out this one concrete type, which is what
    /// lets a directory wrapper throttle the outputs a codec writer creates.
    pub fn set_rate_limiter(&mut self, limiter: Arc<dyn RateLimiter>) {
        self.rate_limit = Some(Box::new(RateLimit {
            current_min_pause_check_bytes: limiter.min_pause_check_bytes(),
            limiter,
            bytes_since_last_pause: 0,
        }));
    }

    /// The on-disk path this output writes to, or `None` for an in-memory
    /// output.
    pub fn path(&self) -> Option<&Path> {
        match &self.sink {
            Sink::File { path, .. } => Some(path),
            Sink::Stream(_) | Sink::Memory { .. } => None,
        }
    }

    /// Hands `b` to the sink, latching the first error. `false` when the
    /// write failed. Every byte handed over is either hashed into `crc` here
    /// or left in `buf` past `crc_pos` for [`Self::checksum`] to cover.
    fn write_through(&mut self, b: &[u8]) -> bool {
        if let Sink::Memory { .. } = self.sink {
            self.buf.extend_from_slice(b);
            if self.buf.len().saturating_sub(self.crc_pos) >= CHUNK_SIZE {
                self.crc.update(&self.buf[self.crc_pos..]);
                self.crc_pos = self.buf.len();
            }
            return true;
        }
        let file = &mut self.sink;
        if self.buf.len().saturating_add(b.len()) > CHUNK_SIZE && !self.buf.is_empty() {
            self.crc.update(&self.buf);
            if let Err(e) = file.write_all(&self.buf) {
                self.pending_err = Some(e);
                return false;
            }
            self.buf.clear();
        }
        if b.len() >= CHUNK_SIZE {
            // Straight through in one `write_all`: Java's `FSIndexOutput`
            // caps each underlying write at `CHUNK_SIZE` only to bound a
            // direct `ByteBuffer`'s size, which a slice write here never
            // allocates. The bytes on disk are the same; a whole-file
            // `write_bytes` is one syscall instead of one per 8 KiB.
            self.crc.update(b);
            if let Err(e) = file.write_all(b) {
                self.pending_err = Some(e);
                return false;
            }
        } else {
            self.buf.extend_from_slice(b);
        }
        true
    }

    /// No rate limiter, no latched error and (in test builds) no fault hook:
    /// a write may go straight into `buf`.
    #[inline(always)]
    fn is_plain(&self) -> bool {
        #[cfg(any(test, feature = "test-util"))]
        if self.faults.is_some() || self.fault_err.is_some() {
            return false;
        }
        self.rate_limit.is_none() && self.pending_err.is_none()
    }

    /// Whether `n` more bytes fit in `buf` without a flush (file sink) or a
    /// CRC catch-up (in-memory sink): `buf` past `crc_pos` stays within
    /// `CHUNK_SIZE`, the invariant `write_through` keeps.
    #[inline(always)]
    fn has_room(&self, n: usize) -> bool {
        match CHUNK_SIZE.checked_sub(n) {
            Some(limit) => self.buf.len().saturating_sub(self.crc_pos) <= limit,
            None => false,
        }
    }
}

impl DataOutput for FsIndexOutput {
    #[inline]
    fn write_byte(&mut self, b: u8) {
        if self.is_plain() && self.has_room(1) {
            self.buf.push(b);
            self.bytes_written = self.bytes_written.saturating_add(1);
            return;
        }
        self.write_bytes(std::slice::from_ref(&b));
    }

    #[inline]
    fn write_vint(&mut self, v: i32) {
        self.write_vlong_raw_u64(v as u32 as u64);
    }

    #[inline]
    fn write_vlong_raw_u64(&mut self, v: u64) {
        if self.is_plain() && self.has_room(10) {
            let before = self.buf.len();
            crate::data_output::push_varint(&mut self.buf, v);
            // ARITH: `push_varint` only appends, so `len >= before`.
            #[allow(clippy::arithmetic_side_effects)]
            let n = self.buf.len() - before;
            self.bytes_written = self.bytes_written.saturating_add(n as u64);
            return;
        }
        let mut buf = [0u8; 10];
        let n = crate::data_output::encode_varint(v, &mut buf);
        self.write_bytes(&buf[..n]);
    }

    #[inline]
    fn write_bytes(&mut self, b: &[u8]) {
        if self.is_plain() && self.has_room(b.len()) {
            self.buf.extend_from_slice(b);
            self.bytes_written = self.bytes_written.saturating_add(b.len() as u64);
            return;
        }
        self.write_bytes_slow(b);
    }
}

impl FsIndexOutput {
    #[inline(never)]
    fn write_bytes_slow(&mut self, b: &[u8]) {
        if self.rate_limit.is_none() {
            self.write_unlimited(b);
            return;
        }
        // Java's writers stream through small buffers, so a rate check per
        // `writeBytes` call is a check every few KiB. Several writers here
        // hand over a whole file in one call instead (`merge.rs`'s
        // `write_file`); one check for it would let the first such file
        // through unthrottled (`lastNS` starts in the past) and throttle the
        // rest in lumps. A check per `CHUNK_SIZE` keeps Java's granularity.
        for chunk in b.chunks(CHUNK_SIZE) {
            if let Some(rate_limit) = self.rate_limit.as_deref_mut() {
                if rate_limit.limiter.is_aborted() {
                    if self.pending_err.is_none() {
                        self.pending_err = Some(std::io::Error::other("write aborted"));
                    }
                    return;
                }
                rate_limit.check_rate(chunk.len());
            }
            self.write_unlimited(chunk);
        }
    }

    fn write_unlimited(&mut self, b: &[u8]) {
        if self.pending_err.is_some() {
            return;
        }
        #[cfg(any(test, feature = "test-util"))]
        if self.fault_err.is_some() {
            return;
        }
        #[cfg(any(test, feature = "test-util"))]
        if let Some(faults) = self.faults.as_deref_mut() {
            if let WriteVerdict::Fail { keep, error } = faults.before_write(&self.name, b.len()) {
                self.write_counted(&b[..keep.min(b.len())]);
                self.fault_err = Some(error);
                return;
            }
            self.write_counted(b);
            if self.pending_err.is_none() {
                if let Some(faults) = self.faults.as_deref_mut() {
                    faults.after_write(&self.name, b.len());
                }
            }
            return;
        }
        self.write_counted(b);
    }

    /// Writes `b` through, then counts it into the checksum and length.
    fn write_counted(&mut self, b: &[u8]) {
        if !self.write_through(b) {
            return;
        }
        // A file longer than 2^64 bytes is not representable by the OS, let
        // alone writable; saturating keeps the counter monotone rather than
        // panicking if one somehow were.
        self.bytes_written = self.bytes_written.saturating_add(b.len() as u64);
    }
}

impl std::fmt::Debug for FsIndexOutput {
    /// `OutputStreamIndexOutput.toString()`'s resource description.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.sink {
            Sink::File { path, .. } => write!(f, "FSIndexOutput(path=\"{}\")", path.display()),
            Sink::Stream(_) => write!(f, "SinkIndexOutput(file={})", self.name),
            Sink::Memory { .. } => write!(f, "ByteBuffersIndexOutput(file={})", self.name),
        }
    }
}

impl Sink {
    /// Writes `bytes` to a file or a stream sink; a memory sink never gets
    /// here ([`FsIndexOutput::write_through`] buffers it).
    #[inline]
    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        match self {
            Sink::File { file, .. } => write_all_bounced(file, bytes),
            Sink::Stream(sink) => sink.write_all(bytes),
            Sink::Memory { .. } => unreachable!("a memory sink is buffered, not written through"),
        }
    }
}

/// `write_all`, except that bytes the kernel cannot read in place (`EFAULT`)
/// are copied through a user-space buffer and written from there. A
/// demand-paged input (`lucene-store-crypto`'s lazy decryption under a
/// user-mode-only `userfaultfd`) faults pages in only on user-space access,
/// so a `write(2)` straight from one of its pages that is not resident fails
/// with `EFAULT`; the copy faults it in.
fn write_all_bounced(file: &mut File, mut bytes: &[u8]) -> std::io::Result<()> {
    const BOUNCE: usize = 64 * 1024;
    while !bytes.is_empty() {
        match file.write(bytes) {
            Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
            Ok(n) => bytes = bytes.get(n..).unwrap_or_default(),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) if is_efault(&e) => {
                let n = bytes.len().min(BOUNCE);
                let bounce = bytes[..n].to_vec();
                file.write_all(&bounce)?;
                bytes = &bytes[n..];
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn is_efault(e: &std::io::Error) -> bool {
    e.raw_os_error() == Some(libc::EFAULT)
}

#[cfg(not(unix))]
fn is_efault(_: &std::io::Error) -> bool {
    false
}

impl IndexOutput for FsIndexOutput {
    fn name(&self) -> &str {
        &self.name
    }

    fn file_pointer(&self) -> u64 {
        self.bytes_written
    }

    fn checksum(&self) -> u64 {
        let mut crc = self.crc.clone();
        crc.update(&self.buf[self.crc_pos..]);
        crc.finalize() as u64
    }
}

/// Port of `FSDirectory.sync(Collection<String>)`: fsyncs every named
/// file's contents (Lucene's durability contract: a new segment's files
/// must be fsynced before a commit file references them). Like Java's, it
/// leaves the directory itself alone: the names become durable at
/// [`sync_meta_data`], which a commit runs once, after the rename that
/// publishes `segments_N`, and which persists every entry created before
/// it. Fsyncing the directory here as well cost each commit two to three
/// extra device flushes.
pub(crate) fn sync(root: &Path, names: &[String]) -> Result<()> {
    for name in names {
        // Write access (never truncating): Windows refuses `FlushFileBuffers`
        // on a read-only handle, as Java's `FileChannel.open(path, WRITE)` implies.
        let file = OpenOptions::new().write(true).open(root.join(name))?;
        file.sync_all()?;
    }
    Ok(())
}

/// Port of `FSDirectory.rename(source, dest)`: `std::fs::rename` is the same
/// `rename(2)` Java's `Files.move(..., ATOMIC_MOVE)` performs, so a reader
/// scanning the directory sees either the old `dest` or the complete new one,
/// never a partial file. The directory entry itself is only durable once
/// [`sync_meta_data`] runs, which is why Lucene calls that immediately after.
pub(crate) fn rename(root: &Path, source: &str, dest: &str) -> Result<()> {
    fs::rename(root.join(source), root.join(dest))?;
    Ok(())
}

/// Port of `FSDirectory.syncMetaData()`: fsync the directory so a name
/// created or renamed inside it survives a crash. Best-effort in the same
/// sense [`sync`]'s trailing directory fsync is -- not every platform lets a
/// directory be opened as a file.
pub(crate) fn sync_meta_data(root: &Path) -> Result<()> {
    if let Ok(dir) = File::open(root) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Convenience used by tests/examples: writes `bytes` to `root/name` via a
/// fresh [`FsIndexOutput`] and closes it, returning the checksum.
#[cfg(test)]
pub(crate) fn write_all_bytes(root: &Path, name: &str, bytes: &[u8]) -> Result<u64> {
    let mut out = FsIndexOutput::create(root, name)?;
    out.write_bytes(bytes);
    out.close()
}

#[cfg(test)]
mod tests;
