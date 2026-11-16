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
        create_output_with_estimate(name: &str, estimated_bytes: Option<u64>) -> Result<FsIndexOutput>;
        create_temp_output_with_estimate(
            prefix: &str,
            suffix: &str,
            estimated_bytes: Option<u64>
        ) -> Result<FsIndexOutput>;
        sync(names: &[String]) -> Result<()>;
        rename(source: &str, dest: &str) -> Result<()>;
        delete_file(name: &str) -> Result<()>;
        sync_meta_data() -> Result<()>;
        pending_deletions() -> Result<BTreeSet<String>>;
    }

    #[cfg(any(test, feature = "test-util"))]
    fn close_input(&self, name: &str, input: Input) -> Result<()> {
        self.inner.close_input(name, input)
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

    #[cfg(any(test, feature = "test-util"))]
    fn disable_virus_checker(&self) -> bool {
        self.inner.disable_virus_checker()
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
        create_temp_output_with_estimate(
            prefix: &str,
            suffix: &str,
            estimated_bytes: Option<u64>
        ) -> Result<FsIndexOutput>;
        obtain_lock(name: &str) -> Result<Box<dyn Lock>>;
        pending_deletions() -> Result<BTreeSet<String>>;
    }

    #[cfg(any(test, feature = "test-util"))]
    fn close_input(&self, name: &str, input: Input) -> Result<()> {
        self.inner.close_input(name, input)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.write_lock.ensure_valid()?;
        self.inner.delete_file(name)
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.write_lock.ensure_valid()?;
        self.inner.create_output(name)
    }

    fn create_output_with_estimate(
        &self,
        name: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        self.write_lock.ensure_valid()?;
        self.inner
            .create_output_with_estimate(name, estimated_bytes)
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

    #[cfg(any(test, feature = "test-util"))]
    fn disable_virus_checker(&self) -> bool {
        self.inner.disable_virus_checker()
    }

    fn fs_directory_path(&self) -> Option<&Path> {
        None
    }
}

#[cfg(test)]
mod tests;
