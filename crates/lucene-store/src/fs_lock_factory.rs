//! Port of `FSLockFactory`, `NativeFSLockFactory` and `SimpleFSLockFactory`:
//! the lock factories that hold a lock file (`write.lock`) inside a
//! filesystem directory.
//!
//! [`NativeFsLockFactory`] is the default, as in Java, and takes **the same
//! OS lock Java's does**, so a Rust `IndexWriter` and a Java one exclude each
//! other on the same directory. Java's `FileChannel.tryLock()` is, on every
//! Unix, an exclusive `fcntl` record lock over the whole file (`F_SETLK`,
//! `l_start = 0`, `l_len = 0`). This port takes an exclusive whole-file
//! `fcntl` lock too:
//!
//! - On Linux/Android it is an **open file description** lock
//!   (`F_OFD_SETLK`). OFD and classic record locks conflict with each other
//!   even inside one process, so the lock excludes a Java writer in *another*
//!   JVM and in the JVM this library is loaded into (the OpenSearch plugin's
//!   case) alike. It also removes the classic lock's trap -- closing *any*
//!   descriptor of the file in the process silently drops a classic lock.
//! - Elsewhere it is the classic `F_SETLK` Java itself takes.
//!
//! `flock(2)` -- what `std::fs::File::try_lock` uses on Linux -- would have
//! been simpler and wrong: `flock` and `fcntl` locks are independent on
//! Linux, so it would exclude only other Rust writers.
//!
//! Like Java, a process-wide set of held lock paths (`LOCK_HELD`) turns a
//! second attempt from the same process into [`Error::LockObtainFailed`]
//! before any descriptor is opened.

use std::collections::HashSet;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::directory::Directory;
use crate::error::{Error, Result};
use crate::lock::{lock_ignoring_poison, Lock, LockFactory};

/// Port of `FSLockFactory`: a [`LockFactory`] that only works on a directory
/// backed by a filesystem path.
pub trait FsLockFactory: Send + Sync + fmt::Debug {
    /// Port of `FSLockFactory.obtainFSLock(FSDirectory dir, lockName)`, given
    /// the directory's path (`FSDirectory.getDirectory()`).
    fn obtain_fs_lock(&self, lock_dir: &Path, lock_name: &str) -> Result<Box<dyn Lock>>;
}

/// Port of `FSLockFactory.obtainLock`: refuses a directory that is not a
/// filesystem directory (`UnsupportedOperationException` in Java).
fn obtain_through_fs(
    factory: &dyn FsLockFactory,
    dir: &dyn Directory,
    lock_name: &str,
) -> Result<Box<dyn Lock>> {
    let Some(path) = dir.fs_directory_path() else {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("{factory:?} can only be used with FSDirectory subclasses"),
        )));
    };
    factory.obtain_fs_lock(path, lock_name)
}

/// Port of `FSLockFactory.getDefault()`: [`NativeFsLockFactory`].
pub fn default_fs_lock_factory() -> Arc<dyn LockFactory> {
    Arc::new(NativeFsLockFactory::INSTANCE)
}

/// `Files.readAttributes(path, BasicFileAttributes.class).creationTime()`:
/// the birth time where the filesystem records one, and the modification
/// time where it does not -- the same fallback the JDK makes on Linux.
/// A lock file is never written, so either identifies one file's lifetime.
fn creation_time(path: &Path) -> std::io::Result<SystemTime> {
    let meta = fs::metadata(path)?;
    meta.created().or_else(|_| meta.modified())
}

/// Port of `NativeFSLockFactory`: an OS file lock on `write.lock`. See the
/// module documentation for which lock and why.
///
/// The lock file is created and left in place: its existence means nothing,
/// only the OS lock does, so a crashed holder never leaves the directory
/// locked.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeFsLockFactory;

impl NativeFsLockFactory {
    /// `NativeFSLockFactory.INSTANCE`.
    pub const INSTANCE: NativeFsLockFactory = NativeFsLockFactory;
}

/// `NativeFSLockFactory.LOCK_HELD`: canonical paths of every lock this
/// process holds through this factory.
static LOCK_HELD: Mutex<Option<HashSet<PathBuf>>> = Mutex::new(None);

fn mark_lock_held(path: &Path) -> bool {
    lock_ignoring_poison(&LOCK_HELD)
        .get_or_insert_with(HashSet::new)
        .insert(path.to_path_buf())
}

fn is_lock_held(path: &Path) -> bool {
    lock_ignoring_poison(&LOCK_HELD)
        .as_ref()
        .is_some_and(|held| held.contains(path))
}

/// `NativeFSLockFactory.clearLockHeld`.
fn clear_lock_held(path: &Path) -> Result<()> {
    let removed = lock_ignoring_poison(&LOCK_HELD)
        .as_mut()
        .is_some_and(|held| held.remove(path));
    if removed {
        Ok(())
    } else {
        Err(Error::AlreadyClosed(format!(
            "Lock path was cleared but never marked as held: {}",
            path.display()
        )))
    }
}

impl LockFactory for NativeFsLockFactory {
    fn obtain_lock(&self, dir: &dyn Directory, lock_name: &str) -> Result<Box<dyn Lock>> {
        obtain_through_fs(self, dir, lock_name)
    }
}

impl FsLockFactory for NativeFsLockFactory {
    fn obtain_fs_lock(&self, lock_dir: &Path, lock_name: &str) -> Result<Box<dyn Lock>> {
        // Ensure that lockDir exists and is a directory.
        fs::create_dir_all(lock_dir)?;
        let lock_file = lock_dir.join(lock_name);

        // "we must create the file to have a truly canonical path. if it's
        // already created, we don't care. if it cant be created, it will
        // fail below."
        let creation_error = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_file)
            .err();

        let real_path = fs::canonicalize(&lock_file).map_err(|e| match creation_error {
            Some(created) => Error::Io(std::io::Error::new(
                e.kind(),
                format!("{e} (creating the lock file failed too: {created})"),
            )),
            None => Error::Io(e),
        })?;

        // A best-effort check, to see if the underlying file has changed.
        let creation_time = creation_time(&real_path)?;

        if !mark_lock_held(&real_path) {
            return Err(Error::LockObtainFailed(format!(
                "Lock held by this process: {}",
                real_path.display()
            )));
        }
        let attempt = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&real_path)
            .and_then(|file| os_lock::try_lock_exclusive(&file).map(|held| (file, held)));
        match attempt {
            Ok((file, true)) => Ok(Box::new(NativeFsLock {
                file: Mutex::new(Some(file)),
                path: real_path,
                creation_time,
                closed: AtomicBool::new(false),
            })),
            Ok((_file, false)) => {
                clear_lock_held(&real_path)?;
                Err(Error::LockObtainFailed(format!(
                    "Lock held by another program: {}",
                    real_path.display()
                )))
            }
            Err(e) => {
                clear_lock_held(&real_path)?;
                Err(Error::Io(e))
            }
        }
    }
}

/// `NativeFSLockFactory.NativeFSLock`. The OS lock lives as long as `file`
/// is open: closing the descriptor releases it.
struct NativeFsLock {
    file: Mutex<Option<File>>,
    path: PathBuf,
    creation_time: SystemTime,
    closed: AtomicBool,
}

impl fmt::Debug for NativeFsLock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NativeFSLock(path={},creationTime={:?})",
            self.path.display(),
            self.creation_time
        )
    }
}

impl Lock for NativeFsLock {
    fn ensure_valid(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::AlreadyClosed(format!(
                "Lock instance already released: {self:?}"
            )));
        }
        // Check we are still in the locks map.
        if !is_lock_held(&self.path) {
            return Err(Error::AlreadyClosed(format!(
                "Lock path unexpectedly cleared from map: {self:?}"
            )));
        }
        // Validate the underlying descriptor (`channel.size()`).
        let size = {
            let file = lock_ignoring_poison(&self.file);
            match file.as_ref() {
                Some(file) => file.metadata()?.len(),
                None => {
                    return Err(Error::AlreadyClosed(format!(
                        "Lock instance already released: {self:?}"
                    )))
                }
            }
        };
        if size != 0 {
            return Err(Error::AlreadyClosed(format!(
                "Unexpected lock file size: {size}, (lock={self:?})"
            )));
        }
        // The backing file name still exists, and is the same file: if it
        // differs, someone deleted our lock file (and we are ineffective).
        let ctime = creation_time(&self.path)?;
        if ctime != self.creation_time {
            return Err(Error::AlreadyClosed(format!(
                "Underlying file changed by an external force at {ctime:?}, (lock={self:?})"
            )));
        }
        Ok(())
    }

    /// Releases the OS lock (by closing its descriptor), then clears the
    /// path from `LOCK_HELD`. Deliberately does not validate first: unlike
    /// [`SimpleFsLockFactory`], releasing can never break someone else's
    /// lock.
    fn close(&self) -> Result<()> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        drop(lock_ignoring_poison(&self.file).take());
        clear_lock_held(&self.path)
    }
}

impl Drop for NativeFsLock {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// The `fcntl` lock itself. The only `unsafe` in this file.
#[cfg(unix)]
mod os_lock {
    use std::fs::File;
    use std::os::fd::AsRawFd;

    /// Linux's open-file-description lock: conflicts with classic record
    /// locks (Java's) even within one process. See the module doc.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const SET_LOCK: libc::c_int = libc::F_OFD_SETLK;
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    const SET_LOCK: libc::c_int = libc::F_SETLK;

    /// `FileChannel.tryLock()`: an exclusive, non-blocking lock over the
    /// whole file. `Ok(false)` when another holder has it.
    pub(super) fn try_lock_exclusive(file: &File) -> std::io::Result<bool> {
        // SAFETY: `flock` is a plain C struct of integers for which all-zero
        // is a valid value (and the value OFD locks require of `l_pid`).
        let mut fl: libc::flock = unsafe { std::mem::zeroed() };
        // The `F_WRLCK`/`SEEK_SET` constants are `c_int` while the fields
        // are `c_short` on every target; both values are tiny (1 and 0).
        fl.l_type = libc::F_WRLCK as libc::c_short;
        fl.l_whence = libc::SEEK_SET as libc::c_short;
        fl.l_start = 0;
        // Zero length: to end of file and beyond, which is how the JDK
        // encodes `tryLock()`'s `Long.MAX_VALUE` size as well.
        fl.l_len = 0;
        // SAFETY: `fd` is an open descriptor owned by `file`, which outlives
        // the call; `fl` is a valid, initialised `flock` the kernel only
        // reads for a set-lock command.
        let rc = unsafe { libc::fcntl(file.as_raw_fd(), SET_LOCK, &mut fl) };
        if rc == 0 {
            return Ok(true);
        }
        let err = std::io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::EAGAIN) | Some(libc::EACCES) => Ok(false),
            _ => Err(err),
        }
    }
}

/// Non-Unix targets: the standard library's advisory lock (`LockFileEx` on
/// Windows, which is also what the JDK uses there).
#[cfg(not(unix))]
mod os_lock {
    use std::fs::File;

    pub(super) fn try_lock_exclusive(file: &File) -> std::io::Result<bool> {
        match file.try_lock() {
            Ok(()) => Ok(true),
            Err(std::fs::TryLockError::WouldBlock) => Ok(false),
            Err(std::fs::TryLockError::Error(e)) => Err(e),
        }
    }
}

/// Port of `SimpleFSLockFactory`: the lock *is* the lock file's existence.
/// Works on filesystems without OS locks, but a crashed holder leaves
/// `write.lock` behind and the directory locked until someone removes it.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleFsLockFactory;

impl SimpleFsLockFactory {
    /// `SimpleFSLockFactory.INSTANCE`.
    pub const INSTANCE: SimpleFsLockFactory = SimpleFsLockFactory;
}

impl LockFactory for SimpleFsLockFactory {
    fn obtain_lock(&self, dir: &dyn Directory, lock_name: &str) -> Result<Box<dyn Lock>> {
        obtain_through_fs(self, dir, lock_name)
    }
}

impl FsLockFactory for SimpleFsLockFactory {
    fn obtain_fs_lock(&self, lock_dir: &Path, lock_name: &str) -> Result<Box<dyn Lock>> {
        fs::create_dir_all(lock_dir)?;
        let lock_file = lock_dir.join(lock_name);
        // Create the file: this fails if it already exists.
        if let Err(e) = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_file)
        {
            return Err(match e.kind() {
                std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::PermissionDenied => {
                    Error::LockObtainFailed(format!(
                        "Lock held elsewhere: {} ({e})",
                        lock_file.display()
                    ))
                }
                _ => Error::Io(e),
            });
        }
        let creation_time = creation_time(&lock_file)?;
        Ok(Box::new(SimpleFsLock {
            path: lock_file,
            creation_time,
            closed: AtomicBool::new(false),
        }))
    }
}

/// `SimpleFSLockFactory.SimpleFSLock`.
struct SimpleFsLock {
    path: PathBuf,
    creation_time: SystemTime,
    closed: AtomicBool,
}

impl fmt::Debug for SimpleFsLock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SimpleFSLock(path={},creationTime={:?})",
            self.path.display(),
            self.creation_time
        )
    }
}

impl Lock for SimpleFsLock {
    fn ensure_valid(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::AlreadyClosed(format!(
                "Lock instance already released: {self:?}"
            )));
        }
        let ctime = creation_time(&self.path)?;
        if ctime != self.creation_time {
            return Err(Error::AlreadyClosed(format!(
                "Underlying file changed by an external force at {ctime:?}, (lock={self:?})"
            )));
        }
        Ok(())
    }

    /// Unlike the native lock, removing the file can delete someone else's
    /// lock if things have gone wrong, so it is validated first and any
    /// failure is a [`Error::LockReleaseFailed`].
    fn close(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        let result = match self.ensure_valid() {
            Err(e) => Err(Error::LockReleaseFailed(format!(
                "Lock file cannot be safely removed. Manual intervention is recommended. ({e})"
            ))),
            Ok(()) => fs::remove_file(&self.path).map_err(|e| {
                Error::LockReleaseFailed(format!(
                    "Unable to remove lock file. Manual intervention is recommended ({e})"
                ))
            }),
        };
        self.closed.store(true, Ordering::Release);
        result
    }
}

impl Drop for SimpleFsLock {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ByteBuffersDirectory, FsDirectory};
    use lucene_util::test_support::TempDir;

    #[test]
    fn native_lock_excludes_a_second_holder_in_this_process() {
        let root = TempDir::new("native-lock");
        let dir = FsDirectory::open(&root);
        let lock = NativeFsLockFactory.obtain_lock(&dir, "write.lock").unwrap();
        lock.ensure_valid().unwrap();
        // The lock file exists, empty.
        assert_eq!(fs::metadata(root.join("write.lock")).unwrap().len(), 0);

        // A second directory instance over the same path: still excluded.
        let other = FsDirectory::open(&root);
        let err = NativeFsLockFactory
            .obtain_lock(&other, "write.lock")
            .unwrap_err();
        assert!(matches!(err, Error::LockObtainFailed(ref m) if m.contains("this process")));

        lock.close().unwrap();
        lock.close().unwrap();
        assert!(matches!(lock.ensure_valid(), Err(Error::AlreadyClosed(_))));
        // The file stays; the lock is free.
        assert!(root.join("write.lock").exists());
        let again = NativeFsLockFactory
            .obtain_lock(&other, "write.lock")
            .unwrap();
        drop(again);
        let _third = NativeFsLockFactory.obtain_lock(&dir, "write.lock").unwrap();
    }

    #[test]
    fn native_lock_creates_a_missing_directory() {
        let root = TempDir::new("native-lock-mkdir");
        let nested = root.join("a").join("b");
        let lock = NativeFsLockFactory
            .obtain_fs_lock(&nested, "write.lock")
            .unwrap();
        assert!(nested.join("write.lock").is_file());
        assert!(format!("{lock:?}").starts_with("NativeFSLock(path="));
    }

    #[test]
    fn native_lock_detects_external_tampering() {
        let root = TempDir::new("native-lock-tamper");
        let lock = NativeFsLockFactory
            .obtain_fs_lock(&root, "write.lock")
            .unwrap();
        // Someone writes into the lock file.
        fs::write(root.join("write.lock"), b"x").unwrap();
        let err = lock.ensure_valid().unwrap_err();
        assert!(
            err.to_string().contains("Unexpected lock file size"),
            "{err}"
        );
        lock.close().unwrap();

        // The lock itself is free again: the size only matters to a holder's
        // validity check, which the next holder fails straight away.
        let again = NativeFsLockFactory
            .obtain_fs_lock(&root, "write.lock")
            .unwrap();
        assert!(again.ensure_valid().is_err());
    }

    #[test]
    fn native_lock_detects_a_deleted_lock_file() {
        let root = TempDir::new("native-lock-deleted");
        let lock = NativeFsLockFactory
            .obtain_fs_lock(&root, "write.lock")
            .unwrap();
        fs::remove_file(root.join("write.lock")).unwrap();
        assert!(matches!(lock.ensure_valid(), Err(Error::Io(_))));
        // Cleared from the map behind its back.
        let path = fs::canonicalize(&root).unwrap().join("write.lock");
        clear_lock_held(&path).unwrap();
        let err = lock.ensure_valid().unwrap_err();
        assert!(err.to_string().contains("cleared from map"), "{err}");
        // Closing reports the missing map entry, once.
        assert!(matches!(lock.close(), Err(Error::AlreadyClosed(_))));
        lock.close().unwrap();
    }

    #[test]
    fn fs_factories_refuse_a_non_filesystem_directory() {
        let dir = ByteBuffersDirectory::new();
        for lf in [
            &NativeFsLockFactory as &dyn LockFactory,
            &SimpleFsLockFactory,
        ] {
            let err = lf.obtain_lock(&dir, "write.lock").unwrap_err();
            assert!(
                matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::Unsupported),
                "{err}"
            );
        }
        assert!(format!("{:?}", default_fs_lock_factory()).contains("NativeFsLockFactory"));
    }

    #[test]
    fn native_lock_reports_an_unusable_lock_directory() {
        let root = TempDir::new("native-lock-file");
        let file = root.join("not-a-dir");
        fs::write(&file, b"").unwrap();
        assert!(matches!(
            NativeFsLockFactory.obtain_fs_lock(&file, "write.lock"),
            Err(Error::Io(_))
        ));
        // The lock name is itself a directory: it can be canonicalised, but
        // not opened for writing.
        fs::create_dir(root.join("dir.lock")).unwrap();
        assert!(matches!(
            NativeFsLockFactory.obtain_fs_lock(&root, "dir.lock"),
            Err(Error::Io(_))
        ));
        // ... and the failed attempt did not leave it marked as held.
        let path = fs::canonicalize(root.join("dir.lock")).unwrap();
        assert!(!is_lock_held(&path));
    }

    #[test]
    fn simple_lock_is_the_lock_file() {
        let root = TempDir::new("simple-lock");
        let dir = FsDirectory::open(&root);
        let lock = SimpleFsLockFactory::INSTANCE
            .obtain_lock(&dir, "write.lock")
            .unwrap();
        lock.ensure_valid().unwrap();
        let err = SimpleFsLockFactory
            .obtain_lock(&dir, "write.lock")
            .unwrap_err();
        assert!(matches!(err, Error::LockObtainFailed(_)), "{err}");
        assert!(format!("{lock:?}").starts_with("SimpleFSLock(path="));
        lock.close().unwrap();
        assert!(!root.join("write.lock").exists());
        lock.close().unwrap();
        assert!(matches!(lock.ensure_valid(), Err(Error::AlreadyClosed(_))));

        // A stale file from a crashed holder keeps the directory locked.
        fs::write(root.join("write.lock"), b"").unwrap();
        assert!(matches!(
            SimpleFsLockFactory.obtain_lock(&dir, "write.lock"),
            Err(Error::LockObtainFailed(_))
        ));
    }

    #[test]
    fn simple_lock_refuses_to_remove_a_file_it_no_longer_owns() {
        let root = TempDir::new("simple-lock-stolen");
        let lock = SimpleFsLockFactory
            .obtain_fs_lock(&root, "write.lock")
            .unwrap();
        fs::remove_file(root.join("write.lock")).unwrap();
        let err = lock.close().unwrap_err();
        assert!(matches!(err, Error::LockReleaseFailed(ref m) if m.contains("safely removed")));
        // Closed regardless.
        lock.close().unwrap();

        // Validation passes but removal fails: a "lock file" that is really
        // a directory cannot be removed as a file.
        let dir_lock = SimpleFsLock {
            path: root.to_path_buf(),
            creation_time: creation_time(&root).unwrap(),
            closed: AtomicBool::new(false),
        };
        let err = dir_lock.close().unwrap_err();
        assert!(matches!(err, Error::LockReleaseFailed(ref m) if m.contains("Unable to remove")));
        assert!(root.exists());

        // Creating the lock in a path that is a file is a plain I/O error.
        let file = root.join("f");
        fs::write(&file, b"").unwrap();
        assert!(matches!(
            SimpleFsLockFactory.obtain_fs_lock(&file, "write.lock"),
            Err(Error::Io(_))
        ));
    }

    /// A lock file deleted and created again behind a holder's back is a
    /// different file: both factories' locks report the change. (Twenty
    /// milliseconds apart, so the new file's time differs at any filesystem
    /// timestamp granularity in use here.)
    #[test]
    fn a_lock_file_recreated_behind_the_holder_is_detected() {
        let root = TempDir::new("lock-recreated");
        let native = NativeFsLockFactory
            .obtain_fs_lock(&root, "native.lock")
            .unwrap();
        let simple = SimpleFsLockFactory
            .obtain_fs_lock(&root, "simple.lock")
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        for name in ["native.lock", "simple.lock"] {
            fs::remove_file(root.join(name)).unwrap();
            fs::write(root.join(name), b"").unwrap();
        }
        for lock in [&native, &simple] {
            let err = lock.ensure_valid().unwrap_err();
            assert!(
                err.to_string().contains("changed by an external force"),
                "{err}"
            );
        }
        native.close().unwrap();
        // The recreated file is not the one this lock created: removing it
        // is refused.
        assert!(simple.close().is_err());
    }

    /// A lock name under a directory that does not exist: neither factory
    /// can create the file, and the native one reports both failures.
    #[test]
    fn a_lock_under_a_missing_directory_is_an_io_error() {
        let root = TempDir::new("lock-missing-subdir");
        let err = NativeFsLockFactory
            .obtain_fs_lock(&root, "nosuch/write.lock")
            .unwrap_err();
        assert!(
            matches!(&err, Error::Io(e) if e.to_string().contains("creating the lock file failed too")),
            "{err}"
        );
        assert!(matches!(
            SimpleFsLockFactory.obtain_fs_lock(&root, "nosuch/write.lock"),
            Err(Error::Io(_))
        ));
    }
}
