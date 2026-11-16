//! The mock filesystem behaviours of `org.apache.lucene.tests.mockfile`, as
//! one [`MockFs`]:
//!
//! - `WindowsFS`: a file with an open handle cannot be deleted, and cannot
//!   be the source of a move ("access denied").
//! - `VirusCheckingFS`: a delete is refused for a fifth of the calls, by a
//!   deterministic hash of a counter and the file name (never `write.lock`,
//!   never a file that does not exist).
//! - `ExtrasFS`: a new directory sometimes starts with an `extra0` file or
//!   directory in it, which an index must ignore.
//! - `HandleLimitFS`: opening more than [`MAX_OPEN_HANDLES`] handles fails
//!   with "Too many open files".
//! - `LeakFS`: handles still open when the filesystem closes are a failure
//!   ([`MockFs::check_leaks`]).
//! - `DisableFsyncFS`: `fsync` is a no-op ([`MockFs::fsync_disabled`]).
//! - `ShuffleFS`: the seed is drawn, but nothing is shuffled -- every
//!   directory listing in this port is sorted (`FSDirectory.listAll` sorts
//!   in Java too, so no `Directory` caller ever saw the shuffle).
//!
//! The deletes and moves reach this through lucene-store's `FsFaults` seam,
//! which an `FsDirectory`/`MmapDirectory` consults before the real call, so
//! a refusal lands in the directory's own pending-delete handling as it does
//! in Java. Handles are counted by
//! [`crate::store::mock_directory_wrapper::MockDirectoryWrapper`], the only
//! place this port sees a file opened and closed; a `MockFs` under a
//! directory nothing tracks sees no handles.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use lucene_store::directory::FsFaults;
use lucene_util::java_random::JavaRandom;

use crate::util::lucene_test_case::java_string_hash;

/// `ExtrasFS.EXTRA_FILE_NAME`.
pub const EXTRA_FILE_NAME: &str = "extra0";

/// `HandleLimitFS.MaxOpenHandles.MAX_OPEN_FILES`.
pub const MAX_OPEN_HANDLES: usize = 2048;

/// `IndexWriter.WRITE_LOCK_NAME`, which the virus checker never holds.
const WRITE_LOCK_NAME: &str = "write.lock";

/// Which behaviours a [`MockFs`] has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MockFsConfig {
    /// `DisableFsyncFS`.
    pub disable_fsync: bool,
    /// `ShuffleFS`'s seed (drawn, never used: see the module docs).
    pub shuffle_seed: Option<i64>,
    /// `LeakFS`.
    pub leak: bool,
    /// `HandleLimitFS`'s limit.
    pub handle_limit: Option<usize>,
    /// `WindowsFS`.
    pub windows: bool,
    /// An active `ExtrasFS`: `Some(create_directory)`.
    pub extras: Option<bool>,
    /// `VirusCheckingFS`'s salt.
    pub virus_checker: Option<i64>,
}

impl MockFsConfig {
    /// `TestRuleTemporaryFilesCleanup.initializeFileSystem()`'s draws, in
    /// its order, for a test class with no `@SuppressFileSystems`: fsync
    /// disabled 99% of the time, shuffled 99%, and 90% of the time the
    /// leak, handle-limit, (one in ten) Windows and (one in four) extras
    /// behaviours. The virus checker is not part of it, as in Java
    /// (`newMaybeVirusCheckingDirectory` adds it: [`Self::with_virus_checker`]).
    pub fn random(r: &mut JavaRandom) -> Self {
        let mut config = MockFsConfig {
            disable_fsync: r.next_int_bounded(100) > 0,
            ..MockFsConfig::default()
        };
        if r.next_int_bounded(100) > 0 {
            config.shuffle_seed = Some(r.next_long());
        }
        if r.next_int_bounded(10) > 0 {
            config.leak = true;
            config.handle_limit = Some(MAX_OPEN_HANDLES);
            config.windows = r.next_int_bounded(10) == 0;
            let active = r.next_int_bounded(4) == 0;
            let create_directory = r.next_boolean();
            config.extras = active.then_some(create_directory);
        }
        config
    }

    /// `LuceneTestCase.addVirusChecker(path)`: a `VirusCheckingFS` salted
    /// with `random.nextLong()`.
    pub fn with_virus_checker(mut self, r: &mut JavaRandom) -> Self {
        self.virus_checker = Some(r.next_long());
        self
    }
}

#[derive(Debug, Default)]
struct Handles {
    /// Open handles per file name (`WindowsFS.openFiles`, by name: one
    /// `MockFs` serves one directory, so the name is the key).
    by_name: BTreeMap<String, usize>,
    /// `HandleLimitFS.count`.
    count: usize,
}

/// A mock filesystem under one directory: see the module docs.
#[derive(Debug)]
pub struct MockFs {
    config: MockFsConfig,
    /// `VirusCheckingFS.enabled`.
    virus_enabled: AtomicBool,
    /// `VirusCheckingFS.state`.
    virus_state: AtomicI64,
    handles: Mutex<Handles>,
}

impl MockFs {
    /// A filesystem with `config`'s behaviours.
    pub fn new(config: MockFsConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            virus_enabled: AtomicBool::new(true),
            virus_state: AtomicI64::new(config.virus_checker.unwrap_or(0)),
            handles: Mutex::new(Handles::default()),
        })
    }

    /// The behaviours this filesystem has.
    pub fn config(&self) -> MockFsConfig {
        self.config
    }

    fn handles(&self) -> MutexGuard<'_, Handles> {
        self.handles.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `ExtrasFS.createDirectory(dir)`: creates `dir`, then, when active,
    /// an [`EXTRA_FILE_NAME`] file (or directory) in it -- best effort, as
    /// in Java.
    pub fn create_directory(&self, dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(dir)?;
        if let Some(create_directory) = self.config.extras {
            let target = dir.join(EXTRA_FILE_NAME);
            let _ = if create_directory {
                std::fs::create_dir(&target)
            } else {
                std::fs::File::create(&target).map(drop)
            };
        }
        Ok(())
    }

    /// `ExtrasFS.isExtra(fileName)`.
    pub fn is_extra(file_name: &str) -> bool {
        file_name == EXTRA_FILE_NAME
    }

    /// `HandleTrackingFS.onOpen`: `HandleLimitFS` refuses a handle past its
    /// limit; `WindowsFS` and `LeakFS` count it.
    pub fn on_open(&self, name: &str) -> io::Result<()> {
        let mut handles = self.handles();
        if self
            .config
            .handle_limit
            .is_some_and(|limit| handles.count >= limit)
        {
            return Err(io::Error::other(format!("{name}: Too many open files")));
        }
        handles.count += 1;
        *handles.by_name.entry(name.to_string()).or_insert(0) += 1;
        Ok(())
    }

    /// `HandleTrackingFS.onClose`.
    pub fn on_close(&self, name: &str) {
        let mut handles = self.handles();
        handles.count = handles.count.saturating_sub(1);
        if let Some(n) = handles.by_name.get_mut(name) {
            *n -= 1;
            if *n == 0 {
                handles.by_name.remove(name);
            }
        }
    }

    /// The files with open handles, and how many each has.
    pub fn open_handles(&self) -> BTreeMap<String, usize> {
        self.handles().by_name.clone()
    }

    /// `LeakFS.onClose()`: an error naming the files still open, when this
    /// filesystem tracks leaks.
    pub fn check_leaks(&self) -> Result<(), String> {
        let handles = self.handles();
        if self.config.leak && !handles.by_name.is_empty() {
            return Err(format!("file handle leaks: {:?}", handles.by_name));
        }
        Ok(())
    }

    /// `DisableFsyncFS`: `fsync` does nothing.
    pub fn fsync_disabled(&self) -> bool {
        self.config.disable_fsync
    }

    /// `TestUtil.hasVirusChecker`.
    pub fn has_virus_checker(&self) -> bool {
        self.config.virus_checker.is_some()
    }

    /// `TestUtil.disableVirusChecker`: `true` when a checker was enabled
    /// (and now is not).
    pub fn disable_virus_checker(&self) -> bool {
        self.has_virus_checker() && self.virus_enabled.swap(false, Ordering::SeqCst)
    }

    /// `TestUtil.enableVirusChecker`.
    pub fn enable_virus_checker(&self) {
        self.virus_enabled.store(true, Ordering::SeqCst);
    }

    /// `WindowsFS.checkDeleteAccess`: a file that exists and has a handle
    /// open cannot be deleted or moved.
    fn check_delete_access(&self, path: &Path) -> io::Result<()> {
        if self.config.windows && path.exists() {
            let name = file_name(path);
            if self.handles().by_name.contains_key(&name) {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!("access denied: {}", path.display()),
                ));
            }
        }
        Ok(())
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

impl FsFaults for MockFs {
    fn disable_virus_checker(&self) -> bool {
        MockFs::disable_virus_checker(self)
    }

    /// `VirusCheckingFS.delete` (outermost, as `addVirusChecker` wraps the
    /// test filesystem), then `WindowsFS.delete`.
    fn check_delete(&self, path: &Path) -> io::Result<()> {
        if self.config.virus_checker.is_some() {
            let name = file_name(path);
            // "Fake but deterministic and hopefully portable like-randomness":
            // the counter moves on every delete, enabled or not.
            let hash = self
                .virus_state
                .fetch_add(1, Ordering::SeqCst)
                .wrapping_add(1)
                .wrapping_mul(i64::from(java_string_hash(&name)));
            if self.virus_enabled.load(Ordering::SeqCst)
                && path.exists()
                && name != WRITE_LOCK_NAME
                && hash % 5 == 1
            {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!(
                        "VirusCheckingFS is randomly refusing to delete file \"{}\"",
                        path.display()
                    ),
                ));
            }
        }
        self.check_delete_access(path)
    }

    /// `WindowsFS.move`: the source must be deletable.
    fn check_rename(&self, source: &Path, _dest: &Path) -> io::Result<()> {
        self.check_delete_access(source)
    }
}

#[cfg(test)]
mod tests;
