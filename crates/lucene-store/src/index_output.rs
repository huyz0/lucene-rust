//! Port of `org.apache.lucene.store.IndexOutput` / `OutputStreamIndexOutput` —
//! the write-side counterpart of [`crate::directory::Input`]/`IndexInput`,
//! backed by a real `std::fs::File` so a Rust program can write files that
//! Java's `FSDirectory.open(path)` opens directly.
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

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::data_output::DataOutput;
use crate::error::{Error, Result};

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

/// A single output file backed by a real `std::fs::File`, buffered
/// (`BufWriter`) so small `write_byte` calls don't each incur a syscall.
pub struct FsIndexOutput {
    name: String,
    path: PathBuf,
    writer: BufWriter<File>,
    bytes_written: u64,
    crc: crc32fast::Hasher,
    pending_err: Option<std::io::Error>,
}

impl FsIndexOutput {
    /// Port of `Directory.createOutput(name, context)`: creates (truncating
    /// any existing file of the same name, matching Java's semantics) a new
    /// file at `root/name` for writing.
    pub fn create(root: &Path, name: &str) -> Result<Self> {
        let path = root.join(name);
        let file = File::create(&path)?;
        Ok(Self {
            name: name.to_string(),
            path,
            writer: BufWriter::new(file),
            bytes_written: 0,
            crc: crc32fast::Hasher::new(),
            pending_err: None,
        })
    }

    /// Port of `IndexOutput.close()`: flushes buffered bytes to the OS and
    /// returns the final CRC32 checksum. Does **not** fsync — Lucene's own
    /// durability contract is that `IndexOutput.close()` merely hands bytes
    /// to the OS; a segment is only durable once its writer calls
    /// `Directory.sync(names)` on every file, which this crate exposes as
    /// [`crate::directory::Directory::sync`]. Returns the first latched I/O
    /// error, if any write since `create` failed.
    pub fn close(mut self) -> Result<u64> {
        if let Some(e) = self.pending_err.take() {
            return Err(Error::Io(e));
        }
        self.writer.flush()?;
        Ok(self.checksum())
    }

    /// The on-disk path this output writes to (used by `Directory::sync`).
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl DataOutput for FsIndexOutput {
    #[inline]
    fn write_byte(&mut self, b: u8) {
        self.write_bytes(std::slice::from_ref(&b));
    }

    fn write_bytes(&mut self, b: &[u8]) {
        if self.pending_err.is_some() {
            return;
        }
        if let Err(e) = self.writer.write_all(b) {
            self.pending_err = Some(e);
            return;
        }
        self.crc.update(b);
        self.bytes_written += b.len() as u64;
    }
}

impl IndexOutput for FsIndexOutput {
    fn name(&self) -> &str {
        &self.name
    }

    fn file_pointer(&self) -> u64 {
        self.bytes_written
    }

    fn checksum(&self) -> u64 {
        self.crc.clone().finalize() as u64
    }
}

/// Port of `Directory.createOutput`, shared by [`crate::directory::
/// FsDirectory`] and [`crate::directory::MmapDirectory`] (both back onto a
/// plain directory of real files; only the *read* path differs between
/// them).
pub(crate) fn create_output(root: &Path, name: &str) -> Result<FsIndexOutput> {
    FsIndexOutput::create(root, name)
}

/// Port of `Directory.sync(Collection<String>)`: fsyncs every named file's
/// contents (Lucene's actual durability contract — a new segment's files
/// must be fsynced before a commit file references them), then fsyncs the
/// containing directory itself so the new directory entries survive a crash
/// too (the same belt-and-suspenders `FSDirectory.sync` does on platforms
/// that support directory fsync).
pub(crate) fn sync(root: &Path, names: &[String]) -> Result<()> {
    for name in names {
        let file = OpenOptions::new().read(true).open(root.join(name))?;
        file.sync_all()?;
    }
    // Best-effort directory fsync so the new file names themselves survive a
    // crash, not just their contents. Not all platforms support opening a
    // directory this way; ignore failure the same way Java's FSDirectory
    // treats this as an optional hardening step, not a correctness gate the
    // file-content fsync above already provides.
    if let Ok(dir) = File::open(root) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Convenience used by tests/examples: writes `bytes` to `root/name` via a
/// fresh [`FsIndexOutput`] and closes it, returning the checksum.
#[cfg(test)]
pub(crate) fn write_all_bytes(root: &Path, name: &str, bytes: &[u8]) -> Result<u64> {
    let mut out = create_output(root, name)?;
    out.write_bytes(bytes);
    out.close()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_input::{DataInput, SliceInput};

    fn tempdir() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lucene-rust-index-output-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn write_bytes_round_trips_and_tracks_file_pointer_and_checksum() {
        let dir = tempdir();
        let mut out = FsIndexOutput::create(&dir, "_0.test").unwrap();
        assert_eq!(out.name(), "_0.test");
        assert_eq!(out.file_pointer(), 0);

        out.write_vint(300);
        out.write_string("hello");
        assert_eq!(out.file_pointer(), 2 + 1 + 5);
        let checksum_before_close = out.checksum();
        let expected_crc = {
            let mut buf = Vec::new();
            buf.write_vint(300);
            buf.write_string("hello");
            crc32fast::hash(&buf) as u64
        };
        assert_eq!(checksum_before_close, expected_crc);

        let final_checksum = out.close().unwrap();
        assert_eq!(final_checksum, expected_crc);

        let bytes = std::fs::read(dir.join("_0.test")).unwrap();
        let mut input = SliceInput::new(&bytes);
        assert_eq!(input.read_vint().unwrap(), 300);
        assert_eq!(input.read_string().unwrap(), "hello");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn create_truncates_existing_file() {
        let dir = tempdir();
        write_all_bytes(&dir, "_0.test", b"0123456789").unwrap();
        write_all_bytes(&dir, "_0.test", b"ab").unwrap();
        assert_eq!(std::fs::read(dir.join("_0.test")).unwrap(), b"ab");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn create_output_nonexistent_dir_is_io_error() {
        let result = FsIndexOutput::create(Path::new("/nonexistent-lucene-rust-test-dir"), "x");
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn write_after_pending_error_is_noop_and_close_surfaces_it() {
        // A closed underlying file descriptor is the simplest way to force a
        // write error deterministically without relying on disk-full/ENOSPC.
        let dir = tempdir();
        let path = dir.join("_0.test");
        let file = File::create(&path).unwrap();
        drop(file);
        // Re-open read-only, then wrap it directly bypassing `create` so the
        // first `write_all` fails with a real OS error (bad file descriptor
        // for writing).
        let ro_file = OpenOptions::new().read(true).open(&path).unwrap();
        let mut out = FsIndexOutput {
            name: "_0.test".to_string(),
            path: path.clone(),
            writer: BufWriter::new(ro_file),
            bytes_written: 0,
            crc: crc32fast::Hasher::new(),
            pending_err: None,
        };
        // `BufWriter`'s default capacity is 8KB; writing past it forces an
        // internal flush (a real `write(2)` syscall), which fails on a
        // read-only fd -- a single `write_byte` would just sit in the
        // buffer and never touch the OS.
        out.write_bytes(&vec![0u8; 9000]);
        assert!(out.pending_err.is_some());
        let fp_before = out.file_pointer();
        out.write_byte(2);
        assert_eq!(out.file_pointer(), fp_before, "no-op after pending error");
        assert!(matches!(out.close(), Err(Error::Io(_))));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sync_fsyncs_named_files_without_corrupting_them() {
        let dir = tempdir();
        write_all_bytes(&dir, "_0.test", b"payload-a").unwrap();
        write_all_bytes(&dir, "_1.test", b"payload-b").unwrap();
        sync(&dir, &["_0.test".to_string(), "_1.test".to_string()]).unwrap();
        assert_eq!(std::fs::read(dir.join("_0.test")).unwrap(), b"payload-a");
        assert_eq!(std::fs::read(dir.join("_1.test")).unwrap(), b"payload-b");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sync_missing_file_is_io_error() {
        let dir = tempdir();
        let result = sync(&dir, &["does-not-exist".to_string()]);
        assert!(matches!(result, Err(Error::Io(_))));
        std::fs::remove_dir_all(&dir).ok();
    }
}
