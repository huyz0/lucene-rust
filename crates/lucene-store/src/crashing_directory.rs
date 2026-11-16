//! A [`Directory`] that can **crash** -- the durability half of Lucene's
//! test-framework `MockDirectoryWrapper` (`crash()`, `unSyncedFiles`), for
//! the crash fuzzing.
//!
//! A process kill leaves the page cache intact, so it cannot test what
//! `fsync` ordering is for. This models a **power loss** instead. It wraps a
//! real [`FsDirectory`] and keeps two ledgers:
//!
//! - **Unsynced contents.** A file written but never passed to
//!   [`Directory::sync`] may come back whole, truncated, zeroed, or not at
//!   all. (Java's `crash()` does the same to its `unSyncedFiles`.)
//! - **Unsynced names.** Every create, rename and delete since the last
//!   [`Directory::sync_meta_data`] is logged in order, and a crash keeps only
//!   a random **prefix** of that log: the POSIX contract (a name is durable
//!   once its directory is fsynced, not before) on a filesystem whose
//!   metadata operations reach disk in order. That is stricter than ext4 in
//!   one respect -- there any fsync commits the whole journal, so a later
//!   file `sync` would make an earlier create durable too -- which is the
//!   contract Lucene is written against, and so the one worth testing. The
//!   model is stronger than Java's, which never un-renames or un-deletes a
//!   file. An operation that replaced or removed a file keeps its old bytes
//!   and durability, so undoing it restores both.
//!
//! [`CrashingDirectory::crash_after`] arms a crash at the n-th mutating call
//! (create, sync, rename, delete, syncMetaData). That call and every call
//! after it fail, as they would in a machine that has lost power. Once the
//! writer has given up, [`CrashingDirectory::power_loss`] rewrites the
//! directory on disk into one state the crash could have left. A fresh
//! directory over the same path then sees what a restarted process would.
//!
//! [`Directory::create_output`] refuses a name that already exists, as
//! `FSDirectory`'s `CREATE_NEW` does, rather than truncating a file that may
//! be durable. [`CrashingDirectory::published`] counts renames onto a
//! `segments_N` name, which is how a harness tells a crash before a commit's
//! publish (the commit must not be visible) from one after it (it may be).
//!
//! **Blind spots.** Contents are only damaged whole-file, never torn inside
//! an already-synced region, since Lucene never rewrites a file. Only
//! ordered-metadata filesystems are modelled: on one that reorders metadata
//! (XFS, btrfs), a rename could survive a crash that loses an earlier create,
//! and a writer that deleted an old commit's files before publishing the new
//! one would lose data there and still pass here.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::directory::{Directory, FsDirectory, Input};
use crate::index_output::FsIndexOutput;
use crate::lock::Lock;
use crate::{Error, Result};

/// A file's bytes as they were before an operation replaced or removed it,
/// and whether those bytes had been synced -- so undoing the operation can
/// put back the file *and* its durability.
#[derive(Debug, Clone)]
struct Saved {
    bytes: Vec<u8>,
    unsynced: bool,
}

/// One namespace change not yet made durable by `sync_meta_data`.
#[derive(Debug, Clone)]
enum MetaEvent {
    Create(String),
    /// `overwritten` is the destination as it was, when the rename replaced
    /// one.
    Rename {
        from: String,
        to: String,
        overwritten: Option<Saved>,
    },
    Delete(String, Saved),
}

#[derive(Debug, Default)]
struct State {
    /// Mutating calls made so far.
    ops: u64,
    /// The call count at which the crash happens.
    crash_at: Option<u64>,
    crashed: bool,
    /// The call the crash hit, for a failure report.
    crash_point: Option<String>,
    /// Successful renames onto a `segments_N` name: commits published.
    published: u64,
    /// Files whose contents were never synced.
    unsynced: BTreeSet<String>,
    /// Namespace changes since the last `sync_meta_data`, oldest first.
    events: Vec<MetaEvent>,
    rng: u64,
}

impl State {
    fn next(&mut self) -> u64 {
        // xorshift64: deterministic for a seed, which is what makes a crash
        // replayable.
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    /// Counts one mutating call and reports whether the machine is down.
    fn step(&mut self, what: impl FnOnce() -> String) -> Result<()> {
        if !self.crashed {
            self.ops = self.ops.saturating_add(1);
            if self.crash_at.is_some_and(|at| self.ops >= at) {
                self.crashed = true;
                self.crash_point = Some(what());
            }
        }
        if self.crashed {
            return Err(crashed());
        }
        Ok(())
    }

    /// Reads `name`'s current bytes and durability, if it exists.
    fn save(&self, root: &Path, name: &str) -> Option<Saved> {
        fs::read(root.join(name)).ok().map(|bytes| Saved {
            bytes,
            unsynced: self.unsynced.contains(name),
        })
    }

    /// Puts `saved` back under `name`, durability included.
    fn restore(&mut self, root: &Path, name: &str, saved: &Saved) -> Result<()> {
        fs::write(root.join(name), &saved.bytes)?;
        if saved.unsynced {
            self.unsynced.insert(name.to_string());
        } else {
            self.unsynced.remove(name);
        }
        Ok(())
    }
}

/// SplitMix64's finaliser: spreads a small seed over all 64 bits. Raw small
/// seeds would start xorshift in a corner where its first outputs share their
/// low bits (every draw `% 2` the same), and never yields zero, xorshift's
/// one fixed point.
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (z ^ (z >> 31)) | 1
}

fn crashed() -> Error {
    Error::Io(std::io::Error::other("simulated power loss"))
}

/// What [`CrashingDirectory::power_loss`] did, for a failure report.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PowerLoss {
    /// Namespace changes that survived, of those pending.
    pub kept_events: usize,
    pub pending_events: usize,
    /// Unsynced files, and what happened to each.
    pub damaged: Vec<(String, &'static str)>,
}

/// See the module documentation.
pub struct CrashingDirectory {
    root: PathBuf,
    inner: FsDirectory,
    state: Mutex<State>,
}

impl CrashingDirectory {
    /// Wraps the directory at `root`. Whatever is already there counts as
    /// durable. `seed` drives every random choice [`Self::power_loss`] makes.
    pub fn new(root: impl Into<PathBuf>, seed: u64) -> Self {
        let root = root.into();
        CrashingDirectory {
            inner: FsDirectory::open(&root),
            root,
            state: Mutex::new(State {
                rng: splitmix64(seed),
                ..State::default()
            }),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        // A poisoned lock means a panic mid-update, and the ledgers may be
        // half-written; the crash model is test infrastructure, so carrying on
        // with them is better than a second panic hiding the first.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Arms a crash at the `n`-th mutating call from now (`n >= 1`).
    pub fn crash_after(&self, n: u64) {
        let mut state = self.state();
        state.crash_at = Some(state.ops.saturating_add(n.max(1)));
    }

    /// Mutating calls so far -- for choosing a crash point inside a run whose
    /// length a first, crash-free run measured.
    pub fn ops(&self) -> u64 {
        self.state().ops
    }

    /// Whether the armed crash has happened.
    pub fn crashed(&self) -> bool {
        self.state().crashed
    }

    /// The call the crash hit (`"rename pending_segments_3 -> segments_3"`),
    /// once it has.
    pub fn crash_point(&self) -> Option<String> {
        self.state().crash_point.clone()
    }

    /// How many renames onto a `segments_N` name have succeeded: the commits
    /// this directory has seen *published*. A crash that leaves this count
    /// unchanged across a commit hit it before its publish, so that commit
    /// must not be visible.
    pub fn published(&self) -> u64 {
        self.state().published
    }

    /// Rewrites the directory into one state the power loss could have left,
    /// then counts everything as durable again, disarmed. Call it once the
    /// writer is dropped; its open outputs may otherwise still write to disk.
    pub fn power_loss(&self) -> Result<PowerLoss> {
        let mut state = self.state();
        let pending = state.events.len();
        let keep = if pending == 0 {
            0
        } else {
            // In `0..=pending`: `% (pending + 1)` is below `pending + 1`.
            let draw = state.next();
            usize::try_from(
                draw.checked_rem((pending as u64).saturating_add(1))
                    .unwrap_or(0),
            )
            .unwrap_or(pending)
        };
        let lost: Vec<MetaEvent> = state.events.split_off(keep);
        for event in lost.iter().rev() {
            match event {
                MetaEvent::Create(name) => {
                    remove_if_present(&self.root.join(name))?;
                    state.unsynced.remove(name);
                }
                MetaEvent::Rename {
                    from,
                    to,
                    overwritten,
                } => {
                    let (from_path, to_path) = (self.root.join(from), self.root.join(to));
                    if to_path.exists() && !from_path.exists() {
                        fs::rename(&to_path, &from_path)?;
                    }
                    if state.unsynced.remove(to) {
                        state.unsynced.insert(from.clone());
                    }
                    if let Some(saved) = overwritten {
                        state.restore(&self.root, to, saved)?;
                    }
                }
                MetaEvent::Delete(name, saved) => {
                    if !self.root.join(name).exists() {
                        state.restore(&self.root, name, saved)?;
                    }
                }
            }
        }

        let mut damaged = Vec::new();
        let unsynced = std::mem::take(&mut state.unsynced);
        for name in unsynced {
            let path = self.root.join(&name);
            let Ok(bytes) = fs::read(&path) else {
                continue;
            };
            let what = match state.next().checked_rem(4).unwrap_or(0) {
                0 => "kept",
                1 => {
                    let len = usize::try_from(
                        state
                            .next()
                            .checked_rem((bytes.len() as u64).saturating_add(1))
                            .unwrap_or(0),
                    )
                    .unwrap_or(0);
                    fs::write(&path, &bytes[..len.min(bytes.len())])?;
                    "truncated"
                }
                2 => {
                    fs::write(&path, vec![0u8; bytes.len()])?;
                    "zeroed"
                }
                _ => {
                    fs::remove_file(&path)?;
                    "lost"
                }
            };
            damaged.push((name, what));
        }

        state.events.clear();
        state.crash_at = None;
        state.crashed = false;
        Ok(PowerLoss {
            kept_events: keep,
            pending_events: pending,
            damaged,
        })
    }
}

fn remove_if_present(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

impl Directory for CrashingDirectory {
    fn list_all(&self) -> Result<Vec<String>> {
        if self.state().crashed {
            return Err(crashed());
        }
        self.inner.list_all()
    }

    fn open(&self, name: &str) -> Result<Input> {
        if self.state().crashed {
            return Err(crashed());
        }
        self.inner.open(name)
    }

    /// `FSDirectory.createOutput` opens with `CREATE_NEW`: Lucene writes every
    /// file once, so a name that already exists is a writer bug, reported as
    /// `FileAlreadyExistsException` in Java and `AlreadyExists` here -- never a
    /// silent truncation of a file that may be durable.
    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        let mut state = self.state();
        state.step(|| format!("create {name}"))?;
        if self.root.join(name).exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("{name} already exists"),
            )));
        }
        let output = self.inner.create_output(name)?;
        state.unsynced.insert(name.to_string());
        state.events.push(MetaEvent::Create(name.to_string()));
        Ok(output)
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        let mut state = self.state();
        state.step(|| format!("sync {}", names.join(" ")))?;
        self.inner.sync(names)?;
        for name in names {
            state.unsynced.remove(name);
        }
        Ok(())
    }

    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        let mut state = self.state();
        state.step(|| format!("rename {source} -> {dest}"))?;
        let overwritten = state.save(&self.root, dest);
        self.inner.rename(source, dest)?;
        if state.unsynced.remove(source) {
            state.unsynced.insert(dest.to_string());
        } else {
            state.unsynced.remove(dest);
        }
        state.events.push(MetaEvent::Rename {
            from: source.to_string(),
            to: dest.to_string(),
            overwritten,
        });
        if dest.starts_with("segments_") {
            state.published = state.published.saturating_add(1);
        }
        Ok(())
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        let mut state = self.state();
        state.step(|| format!("delete {name}"))?;
        // Kept whether or not its contents were durable: a lost delete brings
        // the name back either way, and an unsynced file comes back unsynced
        // -- damaged by the same pass as any other.
        let saved = state.save(&self.root, name);
        self.inner.delete_file(name)?;
        state.unsynced.remove(name);
        if let Some(saved) = saved {
            state
                .events
                .push(MetaEvent::Delete(name.to_string(), saved));
        }
        Ok(())
    }

    fn sync_meta_data(&self) -> Result<()> {
        let mut state = self.state();
        state.step(|| "syncMetaData".to_string())?;
        self.inner.sync_meta_data()?;
        state.events.clear();
        Ok(())
    }

    /// Outside the crash model: a power loss releases an OS lock with
    /// everything else, and the lock file holds no data to damage.
    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.inner.obtain_lock(name)
    }
}

#[cfg(test)]
mod tests;
