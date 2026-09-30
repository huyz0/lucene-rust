//! Port of `org.apache.lucene.store.Lock`, `LockFactory`, and the
//! filesystem-independent factories: `NoLockFactory`,
//! `SingleInstanceLockFactory` and `VerifyingLockFactory`. The filesystem
//! ones (`FSLockFactory`, `NativeFSLockFactory`, `SimpleFSLockFactory`) are in
//! [`crate::fs_lock_factory`].
//!
//! A [`Lock`] is released by [`Lock::close`] or, failing that, by dropping it
//! -- Java's `Closeable` contract with RAII behind it, so a writer that
//! unwinds cannot leave `write.lock` held for the life of the process.

use std::collections::HashSet;
use std::fmt;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::directory::Directory;
use crate::error::{Error, Result};

/// Port of `Lock`: an exclusive hold on a named lock, in practice
/// `IndexWriter.WRITE_LOCK_NAME` (`write.lock`).
///
/// `close` takes `&self` (with interior mutability, as Java's `volatile
/// boolean closed` is) so a holder can release explicitly and still drop the
/// handle afterwards; a second `close` is a no-op, as in every Java
/// implementation.
pub trait Lock: Send + Sync + fmt::Debug {
    /// Port of `Lock.close()`: releases exclusive access. Idempotent.
    fn close(&self) -> Result<()>;

    /// Port of `Lock.ensureValid()`: a best-effort check that the lock is
    /// still held -- that nobody released it, deleted its file, or replaced
    /// the file with another. Fails with [`Error::AlreadyClosed`] when it is
    /// not.
    fn ensure_valid(&self) -> Result<()>;
}

/// Port of `LockFactory`: hands out [`Lock`]s for a [`Directory`].
pub trait LockFactory: Send + Sync + fmt::Debug {
    /// Port of `LockFactory.obtainLock(dir, lockName)`. Fails with
    /// [`Error::LockObtainFailed`] when the lock is held elsewhere.
    fn obtain_lock(&self, dir: &dyn Directory, lock_name: &str) -> Result<Box<dyn Lock>>;
}

/// Locks a mutex whose data (a set of names) stays consistent even if a
/// holder panicked: every critical section is a single insert/remove.
pub(crate) fn lock_ignoring_poison<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Port of `NoLockFactory`: every `obtain_lock` succeeds and nothing is
/// excluded. Only safe when the application guarantees a single writer.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoLockFactory;

impl NoLockFactory {
    /// `NoLockFactory.INSTANCE`.
    pub const INSTANCE: NoLockFactory = NoLockFactory;
}

/// `NoLockFactory.NoLock`.
#[derive(Debug)]
struct NoLock;

impl Lock for NoLock {
    fn close(&self) -> Result<()> {
        Ok(())
    }

    fn ensure_valid(&self) -> Result<()> {
        Ok(())
    }
}

impl LockFactory for NoLockFactory {
    fn obtain_lock(&self, _dir: &dyn Directory, _lock_name: &str) -> Result<Box<dyn Lock>> {
        Ok(Box::new(NoLock))
    }
}

/// Port of `SingleInstanceLockFactory`: locks are names in an in-memory set
/// private to this factory instance, so it excludes only other users of the
/// *same* factory -- the default for [`crate::ByteBuffersDirectory`].
#[derive(Debug, Clone, Default)]
pub struct SingleInstanceLockFactory {
    locks: Arc<Mutex<HashSet<String>>>,
}

impl SingleInstanceLockFactory {
    pub fn new() -> Self {
        Self::default()
    }
}

impl LockFactory for SingleInstanceLockFactory {
    fn obtain_lock(&self, _dir: &dyn Directory, lock_name: &str) -> Result<Box<dyn Lock>> {
        if lock_ignoring_poison(&self.locks).insert(lock_name.to_string()) {
            Ok(Box::new(SingleInstanceLock {
                locks: Arc::clone(&self.locks),
                lock_name: lock_name.to_string(),
                closed: AtomicBool::new(false),
            }))
        } else {
            Err(Error::LockObtainFailed(format!(
                "lock instance already obtained: (lockName={lock_name})"
            )))
        }
    }
}

/// `SingleInstanceLockFactory.SingleInstanceLock`.
struct SingleInstanceLock {
    locks: Arc<Mutex<HashSet<String>>>,
    lock_name: String,
    closed: AtomicBool,
}

impl fmt::Debug for SingleInstanceLock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SingleInstanceLock: {}", self.lock_name)
    }
}

impl Lock for SingleInstanceLock {
    fn ensure_valid(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::AlreadyClosed(format!(
                "Lock instance already released: {self:?}"
            )));
        }
        if !lock_ignoring_poison(&self.locks).contains(&self.lock_name) {
            return Err(Error::AlreadyClosed(format!(
                "Lock instance was invalidated from map: {self:?}"
            )));
        }
        Ok(())
    }

    fn close(&self) -> Result<()> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        if !lock_ignoring_poison(&self.locks).remove(&self.lock_name) {
            return Err(Error::AlreadyClosed(format!(
                "Lock was already released: {self:?}"
            )));
        }
        Ok(())
    }
}

impl Drop for SingleInstanceLock {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// `VerifyingLockFactory.MSG_LOCK_RELEASED`.
pub const MSG_LOCK_RELEASED: u8 = 0;
/// `VerifyingLockFactory.MSG_LOCK_ACQUIRED`.
pub const MSG_LOCK_ACQUIRED: u8 = 1;

/// The socket a [`VerifyingLockFactory`] reports to (`LockVerifyServer`).
struct Channel {
    input: Box<dyn Read + Send>,
    output: Box<dyn Write + Send>,
}

/// Port of `VerifyingLockFactory`: wraps another factory and reports every
/// acquire and release to a verifier (Java's `LockVerifyServer`), which
/// echoes the message back only if no other client holds the lock -- the
/// harness `LockStressTest` uses to prove a factory really excludes.
pub struct VerifyingLockFactory {
    lf: Arc<dyn LockFactory>,
    channel: Arc<Mutex<Channel>>,
}

impl fmt::Debug for VerifyingLockFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VerifyingLockFactory({:?})", self.lf)
    }
}

impl VerifyingLockFactory {
    /// `new VerifyingLockFactory(lf, in, out)`.
    pub fn new(
        lf: Arc<dyn LockFactory>,
        input: Box<dyn Read + Send>,
        output: Box<dyn Write + Send>,
    ) -> Self {
        Self {
            lf,
            channel: Arc::new(Mutex::new(Channel { input, output })),
        }
    }
}

impl LockFactory for VerifyingLockFactory {
    fn obtain_lock(&self, dir: &dyn Directory, lock_name: &str) -> Result<Box<dyn Lock>> {
        let lock = self.lf.obtain_lock(dir, lock_name)?;
        let checked = CheckedLock {
            lock,
            channel: Arc::clone(&self.channel),
        };
        // Java's constructor sends the acquire message; if the verifier
        // rejects it the lock is dropped here -- released -- where Java's
        // would leak until collected.
        verify(&checked.channel, MSG_LOCK_ACQUIRED)?;
        Ok(Box::new(checked))
    }
}

/// `VerifyingLockFactory.CheckedLock.verify`.
fn verify(channel: &Mutex<Channel>, message: u8) -> Result<()> {
    let mut channel = lock_ignoring_poison(channel);
    channel.output.write_all(&[message])?;
    channel.output.flush()?;
    let mut ret = [0u8; 1];
    if channel.input.read(&mut ret)? == 0 {
        return Err(Error::AlreadyClosed(
            "Lock server died because of locking error.".to_string(),
        ));
    }
    if ret[0] != message {
        return Err(Error::Io(std::io::Error::other("Protocol violation.")));
    }
    Ok(())
}

/// `VerifyingLockFactory.CheckedLock`.
struct CheckedLock {
    lock: Box<dyn Lock>,
    channel: Arc<Mutex<Channel>>,
}

impl fmt::Debug for CheckedLock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CheckedLock({:?})", self.lock)
    }
}

impl Lock for CheckedLock {
    fn ensure_valid(&self) -> Result<()> {
        self.lock.ensure_valid()
    }

    /// `try (Lock l = lock) { l.ensureValid(); verify(RELEASED); }`: the
    /// wrapped lock is closed whatever the check or the verifier says, and
    /// the first failure is the one reported.
    fn close(&self) -> Result<()> {
        let checked = self
            .lock
            .ensure_valid()
            .and_then(|()| verify(&self.channel, MSG_LOCK_RELEASED));
        let closed = self.lock.close();
        checked.and(closed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ByteBuffersDirectory;
    use std::io::Cursor;

    #[test]
    fn no_lock_factory_hands_out_any_number_of_locks() {
        let dir = ByteBuffersDirectory::new();
        let a = NoLockFactory::INSTANCE
            .obtain_lock(&dir, "write.lock")
            .unwrap();
        let b = NoLockFactory.obtain_lock(&dir, "write.lock").unwrap();
        a.ensure_valid().unwrap();
        a.close().unwrap();
        a.close().unwrap();
        a.ensure_valid().unwrap();
        b.close().unwrap();
        assert_eq!(format!("{a:?}"), "NoLock");
    }

    #[test]
    fn single_instance_lock_factory_excludes_until_released() {
        let dir = ByteBuffersDirectory::new();
        let lf = SingleInstanceLockFactory::new();
        let l = lf.obtain_lock(&dir, "write.lock").unwrap();
        assert!(matches!(
            lf.obtain_lock(&dir, "write.lock"),
            Err(Error::LockObtainFailed(_))
        ));
        // A different name is a different lock.
        let other = lf.obtain_lock(&dir, "other.lock").unwrap();
        l.ensure_valid().unwrap();
        l.close().unwrap();
        // Close is idempotent, and a closed lock is no longer valid.
        l.close().unwrap();
        assert!(matches!(l.ensure_valid(), Err(Error::AlreadyClosed(_))));
        let again = lf.obtain_lock(&dir, "write.lock").unwrap();
        drop(again);
        // Dropping releases too.
        lf.obtain_lock(&dir, "write.lock").unwrap();
        other.close().unwrap();
        // A separate factory is a separate namespace.
        let separate = SingleInstanceLockFactory::new();
        let _held = lf.obtain_lock(&dir, "x").unwrap();
        separate.obtain_lock(&dir, "x").unwrap();
    }

    #[test]
    fn single_instance_lock_detects_invalidation_from_the_map() {
        let dir = ByteBuffersDirectory::new();
        let lf = SingleInstanceLockFactory::new();
        let l = lf.obtain_lock(&dir, "write.lock").unwrap();
        // "some debugger or something crazy" removes it.
        lf.locks.lock().unwrap().remove("write.lock");
        let err = l.ensure_valid().unwrap_err();
        assert!(err.to_string().contains("invalidated from map"), "{err}");
        let err = l.close().unwrap_err();
        assert!(err.to_string().contains("already released"), "{err}");
        assert!(format!("{l:?}").contains("write.lock"));
    }

    /// A verifier that echoes whatever it is told, recording it.
    #[derive(Clone, Default)]
    struct Recorder(Arc<Mutex<Vec<u8>>>);

    impl Write for Recorder {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn verifying(replies: &[u8]) -> (VerifyingLockFactory, Recorder) {
        let sent = Recorder::default();
        let lf = VerifyingLockFactory::new(
            Arc::new(SingleInstanceLockFactory::new()),
            Box::new(Cursor::new(replies.to_vec())),
            Box::new(sent.clone()),
        );
        (lf, sent)
    }

    #[test]
    fn verifying_lock_factory_reports_acquire_and_release() {
        let dir = ByteBuffersDirectory::new();
        let (lf, sent) = verifying(&[MSG_LOCK_ACQUIRED, MSG_LOCK_RELEASED]);
        let l = lf.obtain_lock(&dir, "write.lock").unwrap();
        l.ensure_valid().unwrap();
        l.close().unwrap();
        assert_eq!(*sent.0.lock().unwrap(), vec![1, 0]);
        assert!(format!("{lf:?}").starts_with("VerifyingLockFactory("));
        assert!(format!("{l:?}").starts_with("CheckedLock("));
    }

    #[test]
    fn verifying_lock_factory_protocol_errors() {
        let dir = ByteBuffersDirectory::new();
        // The server replies with the wrong message.
        let (lf, _) = verifying(&[MSG_LOCK_RELEASED]);
        let err = lf.obtain_lock(&dir, "write.lock").unwrap_err();
        assert!(err.to_string().contains("Protocol violation"), "{err}");
        // ... and the inner lock was released, not leaked: a second attempt
        // gets past the inner factory (a leak would fail it with
        // LockObtainFailed) and on to the now-silent server.
        let err = lf.obtain_lock(&dir, "write.lock").unwrap_err();
        assert!(matches!(err, Error::AlreadyClosed(_)), "{err}");

        // Release is reported to a dead server: the error surfaces, and the
        // wrapped lock is closed anyway.
        let (lf, _) = verifying(&[MSG_LOCK_ACQUIRED]);
        let l = lf.obtain_lock(&dir, "write.lock").unwrap();
        assert!(l.close().is_err());
        assert!(l.ensure_valid().is_err());
    }
}
