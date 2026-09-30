//! Port of the two lock-related `FilterDirectory`s:
//! `SleepingLockWrapper` (retry `obtainLock` until a timeout) and
//! `LockValidatingDirectoryWrapper` (check the write lock before every
//! mutation).

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::directory::{Directory, Input};
use crate::error::{Error, Result};
use crate::index_output::FsIndexOutput;
use crate::lock::Lock;

/// Port of `SleepingLockWrapper`: a directory whose `obtain_lock` polls the
/// delegate until the lock is free or a timeout passes, instead of failing
/// at once.
pub struct SleepingLockWrapper<D> {
    inner: D,
    lock_wait_timeout: i64,
    poll_interval: i64,
}

impl<D: Directory> SleepingLockWrapper<D> {
    /// `SleepingLockWrapper.LOCK_OBTAIN_WAIT_FOREVER`: retry forever.
    pub const LOCK_OBTAIN_WAIT_FOREVER: i64 = -1;
    /// `SleepingLockWrapper.DEFAULT_POLL_INTERVAL`, in milliseconds.
    pub const DEFAULT_POLL_INTERVAL: i64 = 1000;

    /// `new SleepingLockWrapper(delegate, lockWaitTimeout)`.
    pub fn new(delegate: D, lock_wait_timeout: i64) -> Result<Self> {
        Self::with_poll_interval(delegate, lock_wait_timeout, Self::DEFAULT_POLL_INTERVAL)
    }

    /// `new SleepingLockWrapper(delegate, lockWaitTimeout, pollInterval)`;
    /// both in milliseconds.
    pub fn with_poll_interval(
        delegate: D,
        lock_wait_timeout: i64,
        poll_interval: i64,
    ) -> Result<Self> {
        if lock_wait_timeout < 0 && lock_wait_timeout != Self::LOCK_OBTAIN_WAIT_FOREVER {
            return Err(Error::IllegalArgument(format!(
                "lockWaitTimeout should be LOCK_OBTAIN_WAIT_FOREVER or a non-negative number (got {lock_wait_timeout})"
            )));
        }
        if poll_interval < 0 {
            return Err(Error::IllegalArgument(format!(
                "pollInterval must be a non-negative number (got {poll_interval})"
            )));
        }
        Ok(Self {
            inner: delegate,
            lock_wait_timeout,
            poll_interval,
        })
    }

    /// `FilterDirectory.getDelegate()`.
    pub fn delegate(&self) -> &D {
        &self.inner
    }
}

/// Every method but the one a wrapper overrides, forwarded to `self.inner`.
macro_rules! forward_to_inner {
    ($($method:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => {$(
        fn $method(&self, $($arg: $ty),*) -> $ret {
            self.inner.$method($($arg),*)
        }
    )*};
}

impl<D: Directory> Directory for SleepingLockWrapper<D> {
    forward_to_inner! {
        list_all() -> Result<Vec<String>>;
        open(name: &str) -> Result<Input>;
        file_length(name: &str) -> Result<u64>;
        create_output(name: &str) -> Result<FsIndexOutput>;
        create_temp_output(prefix: &str, suffix: &str) -> Result<FsIndexOutput>;
        sync(names: &[String]) -> Result<()>;
        rename(source: &str, dest: &str) -> Result<()>;
        delete_file(name: &str) -> Result<()>;
        sync_meta_data() -> Result<()>;
        pending_deletions() -> Result<BTreeSet<String>>;
    }

    /// Tries once, then every `poll_interval` ms until `lock_wait_timeout`
    /// ms have passed; the timeout error wraps the first failure.
    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        let Some(max_sleep_count) = self.lock_wait_timeout.checked_div(self.poll_interval) else {
            // Java divides unconditionally, so a zero poll interval is an
            // `ArithmeticException` on the first call.
            return Err(Error::IllegalArgument(
                "pollInterval of 0: / by zero".to_string(),
            ));
        };
        let mut failure_reason: Option<String> = None;
        let mut sleep_count: i64 = 0;
        loop {
            match self.inner.obtain_lock(name) {
                Ok(lock) => return Ok(lock),
                Err(Error::LockObtainFailed(reason)) => {
                    failure_reason.get_or_insert(reason);
                }
                Err(e) => return Err(e),
            }
            std::thread::sleep(Duration::from_millis(
                u64::try_from(self.poll_interval).unwrap_or(0),
            ));
            let keep_going = sleep_count < max_sleep_count
                || self.lock_wait_timeout == Self::LOCK_OBTAIN_WAIT_FOREVER;
            sleep_count = sleep_count.saturating_add(1);
            if !keep_going {
                break;
            }
        }
        Err(Error::LockObtainFailed(format!(
            "Lock obtain timed out: SleepingLockWrapper: {}",
            failure_reason.unwrap_or_default()
        )))
    }

    fn fs_directory_path(&self) -> Option<&Path> {
        None
    }
}

/// Port of `LockValidatingDirectoryWrapper`: checks
/// [`Lock::ensure_valid`] before every operation that changes the
/// directory, so a writer whose `write.lock` was lost stops writing instead
/// of corrupting an index another writer now owns.
pub struct LockValidatingDirectoryWrapper<D> {
    inner: D,
    write_lock: Arc<dyn Lock>,
}

impl<D: Directory> LockValidatingDirectoryWrapper<D> {
    /// `new LockValidatingDirectoryWrapper(in, writeLock)`.
    pub fn new(inner: D, write_lock: Arc<dyn Lock>) -> Self {
        Self { inner, write_lock }
    }

    /// `FilterDirectory.getDelegate()`.
    pub fn delegate(&self) -> &D {
        &self.inner
    }
}

impl<D: Directory> Directory for LockValidatingDirectoryWrapper<D> {
    forward_to_inner! {
        list_all() -> Result<Vec<String>>;
        open(name: &str) -> Result<Input>;
        file_length(name: &str) -> Result<u64>;
        create_temp_output(prefix: &str, suffix: &str) -> Result<FsIndexOutput>;
        obtain_lock(name: &str) -> Result<Box<dyn Lock>>;
        pending_deletions() -> Result<BTreeSet<String>>;
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.write_lock.ensure_valid()?;
        self.inner.delete_file(name)
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.write_lock.ensure_valid()?;
        self.inner.create_output(name)
    }

    fn copy_from(&self, from: &dyn Directory, src: &str, dest: &str) -> Result<()> {
        self.write_lock.ensure_valid()?;
        self.inner.copy_from(from, src, dest)
    }

    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        self.write_lock.ensure_valid()?;
        self.inner.rename(source, dest)
    }

    fn sync_meta_data(&self) -> Result<()> {
        self.write_lock.ensure_valid()?;
        self.inner.sync_meta_data()
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        self.write_lock.ensure_valid()?;
        self.inner.sync(names)
    }

    fn fs_directory_path(&self) -> Option<&Path> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_output::DataOutput;
    use crate::{ByteBuffersDirectory, FsDirectory};
    use lucene_util::test_support::TempDir;
    use std::time::Instant;

    #[test]
    fn sleeping_wrapper_rejects_bad_arguments() {
        let dir = ByteBuffersDirectory::new();
        assert!(matches!(
            SleepingLockWrapper::new(&dir, -2),
            Err(Error::IllegalArgument(_))
        ));
        assert!(matches!(
            SleepingLockWrapper::with_poll_interval(&dir, 10, -1),
            Err(Error::IllegalArgument(_))
        ));
        let zero = SleepingLockWrapper::with_poll_interval(&dir, 10, 0).unwrap();
        assert!(matches!(
            zero.obtain_lock("write.lock"),
            Err(Error::IllegalArgument(_))
        ));
        SleepingLockWrapper::new(
            &dir,
            SleepingLockWrapper::<&ByteBuffersDirectory>::LOCK_OBTAIN_WAIT_FOREVER,
        )
        .unwrap();
    }

    /// Mirrors `TestSleepingLockWrapper`: a held lock times out after the
    /// wait, and a lock released during the wait is obtained.
    #[test]
    fn sleeping_wrapper_retries_until_timeout() {
        let dir = ByteBuffersDirectory::new();
        let held = dir.obtain_lock("write.lock").unwrap();
        let sleeping = SleepingLockWrapper::with_poll_interval(&dir, 30, 10).unwrap();
        let started = Instant::now();
        let err = sleeping.obtain_lock("write.lock").unwrap_err();
        assert!(started.elapsed() >= Duration::from_millis(30));
        assert!(
            matches!(&err, Error::LockObtainFailed(m) if m.starts_with("Lock obtain timed out") && m.contains("already obtained")),
            "{err}"
        );

        std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(Duration::from_millis(30));
                held.close().unwrap();
            });
            let lock = SleepingLockWrapper::with_poll_interval(&dir, 5_000, 5)
                .unwrap()
                .obtain_lock("write.lock")
                .unwrap();
            lock.ensure_valid().unwrap();
        });
    }

    #[test]
    fn sleeping_wrapper_forwards_everything_else() {
        let root = TempDir::new("sleeping-wrapper");
        let dir = SleepingLockWrapper::new(FsDirectory::open(&root), 0).unwrap();
        let mut out = dir.create_output("a").unwrap();
        out.write_bytes(b"xyz");
        out.close().unwrap();
        dir.create_temp_output("_0", "t").unwrap().close().unwrap();
        dir.sync(&["a".to_string()]).unwrap();
        dir.rename("a", "b").unwrap();
        dir.sync_meta_data().unwrap();
        assert_eq!(&*dir.open("b").unwrap(), b"xyz");
        assert_eq!(dir.file_length("b").unwrap(), 3);
        assert_eq!(dir.list_all().unwrap(), vec!["_0_t_0.tmp", "b"]);
        dir.delete_file("b").unwrap();
        assert!(dir.pending_deletions().unwrap().is_empty());
        assert!(dir.fs_directory_path().is_none());
        assert_eq!(dir.delegate().directory(), root.path());
        // A timeout of 0 tries exactly once more after one poll.
        let held = dir.obtain_lock("write.lock").unwrap();
        assert!(matches!(
            SleepingLockWrapper::with_poll_interval(FsDirectory::open(&root), 0, 1)
                .unwrap()
                .obtain_lock("write.lock"),
            Err(Error::LockObtainFailed(_))
        ));
        drop(held);
    }

    const LOCK_WAIT_FOREVER_FOR_TEST: i64 = -1;

    #[test]
    fn sleeping_wrapper_does_not_retry_other_errors() {
        let root = TempDir::new("sleeping-wrapper-err");
        let dir = FsDirectory::with_lock_factory(
            root.join("f").join("g"),
            crate::fs_lock_factory::default_fs_lock_factory(),
        );
        // The lock directory cannot be created under a regular file.
        std::fs::write(root.join("f"), b"").unwrap();
        let sleeping = SleepingLockWrapper::new(dir, LOCK_WAIT_FOREVER_FOR_TEST).unwrap();
        assert!(matches!(
            sleeping.obtain_lock("write.lock"),
            Err(Error::Io(_))
        ));
    }

    #[test]
    fn lock_validating_wrapper_checks_before_each_mutation() {
        let root = TempDir::new("lock-validating");
        let fs_dir = FsDirectory::open(&root);
        let lock: Arc<dyn Lock> = Arc::from(fs_dir.obtain_lock("write.lock").unwrap());
        let dir = LockValidatingDirectoryWrapper::new(&fs_dir, Arc::clone(&lock));
        let mut out = dir.create_output("a").unwrap();
        out.write_bytes(b"1");
        out.close().unwrap();
        dir.sync(&["a".to_string()]).unwrap();
        dir.rename("a", "b").unwrap();
        dir.sync_meta_data().unwrap();
        let src = ByteBuffersDirectory::new();
        let mut o = src.create_output("s").unwrap();
        o.write_bytes(b"src");
        o.close().unwrap();
        dir.copy_from(&src, "s", "c").unwrap();
        assert_eq!(&*dir.open("c").unwrap(), b"src");
        assert_eq!(dir.file_length("b").unwrap(), 1);
        dir.delete_file("b").unwrap();
        dir.create_temp_output("_0", "t").unwrap().close().unwrap();
        assert!(dir.pending_deletions().unwrap().is_empty());
        assert!(dir.fs_directory_path().is_none());
        assert!(dir.delegate().fs_directory_path().is_some());
        assert!(matches!(
            dir.obtain_lock("write.lock"),
            Err(Error::LockObtainFailed(_))
        ));
        assert!(dir.list_all().unwrap().contains(&"c".to_string()));

        // Once the lock is gone, every mutation refuses; reads still work.
        lock.close().unwrap();
        let closed = |r: Result<()>| assert!(matches!(r, Err(Error::AlreadyClosed(_))));
        closed(dir.create_output("x").map(|_| ()));
        closed(dir.delete_file("c"));
        closed(dir.rename("c", "d"));
        closed(dir.sync(&["c".to_string()]));
        closed(dir.sync_meta_data());
        closed(dir.copy_from(&src, "s", "e"));
        assert_eq!(&*dir.open("c").unwrap(), b"src");
    }
}
