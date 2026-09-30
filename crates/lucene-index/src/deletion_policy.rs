//! The pluggable half of `org.apache.lucene.index.IndexDeletionPolicy`: the
//! [`IndexDeletionPolicy`] trait a writer's [`crate::index_file_deleter`]
//! consults on open (`onInit`) and after every commit (`onCommit`), the
//! [`IndexCommit`] it hands over, and Lucene's policies behind it --
//! [`KeepOnlyLastCommitDeletionPolicy`], [`NoDeletionPolicy`],
//! [`KeepLastNCommitsDeletionPolicy`], [`SnapshotDeletionPolicy`] and
//! [`PersistentSnapshotDeletionPolicy`].
//!
//! # Shape
//!
//! Java hands a policy a `List<IndexCommit>` and the policy calls
//! `commit.delete()` on the ones it no longer wants; the deleter then drops
//! every commit so marked. Here the list is `&mut [IndexCommit]` and
//! [`IndexCommit::delete`] sets the same flag. A policy with state
//! (snapshots) keeps it behind a `Mutex`, Java's `synchronized`, because the
//! writer holds the policy as an `Arc` shared with the caller who takes and
//! releases snapshots.
//!
//! `SnapshotDeletionPolicy` wraps each commit so that `delete()` on a
//! snapshotted one is ignored (`SnapshotCommitPoint.delete`); here the primary
//! policy decides over a copy, and a deletion is carried back only for a
//! commit no snapshot holds -- the same outcome.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use lucene_store::codec_util;
use lucene_store::data_input::DataInput;
use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;
use lucene_store::SliceInput;

/// A deletion-policy failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    /// Java's `IllegalArgumentException`.
    #[error("illegal argument: {0}")]
    IllegalArgument(String),
    /// Java's `IllegalStateException`.
    #[error("illegal state: {0}")]
    IllegalState(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// `IndexCommit`: one commit point -- a `segments_N` and the files it names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexCommit {
    generation: i64,
    segments_file_name: String,
    file_names: Vec<String>,
    segment_count: usize,
    user_data: Vec<(String, String)>,
    deleted: bool,
}

impl IndexCommit {
    pub fn new(
        generation: i64,
        segments_file_name: impl Into<String>,
        file_names: Vec<String>,
        segment_count: usize,
        user_data: Vec<(String, String)>,
    ) -> Self {
        IndexCommit {
            generation,
            segments_file_name: segments_file_name.into(),
            file_names,
            segment_count,
            user_data,
            deleted: false,
        }
    }

    /// `getGeneration`.
    pub fn generation(&self) -> i64 {
        self.generation
    }

    /// `getSegmentsFileName`.
    pub fn segments_file_name(&self) -> &str {
        &self.segments_file_name
    }

    /// `getFileNames`: every file the commit references, `segments_N`
    /// included.
    pub fn file_names(&self) -> &[String] {
        &self.file_names
    }

    /// `getSegmentCount`.
    pub fn segment_count(&self) -> usize {
        self.segment_count
    }

    /// `getUserData`.
    pub fn user_data(&self) -> &[(String, String)] {
        &self.user_data
    }

    /// `delete()`: asks the deleter to drop this commit once the policy
    /// returns.
    pub fn delete(&mut self) {
        self.deleted = true;
    }

    /// `isDeleted`.
    pub fn is_deleted(&self) -> bool {
        self.deleted
    }
}

/// `IndexDeletionPolicy`.
pub trait IndexDeletionPolicy: Send + Sync + fmt::Debug {
    /// `onInit`: the commits found when the writer opened, oldest first.
    fn on_init(&self, commits: &mut [IndexCommit]) -> Result<()>;
    /// `onCommit`: every live commit after a new one, oldest first.
    fn on_commit(&self, commits: &mut [IndexCommit]) -> Result<()>;
}

/// `KeepOnlyLastCommitDeletionPolicy`, Lucene's default.
#[derive(Debug, Clone, Copy, Default)]
pub struct KeepOnlyLastCommitDeletionPolicy;

impl IndexDeletionPolicy for KeepOnlyLastCommitDeletionPolicy {
    fn on_init(&self, commits: &mut [IndexCommit]) -> Result<()> {
        self.on_commit(commits)
    }
    fn on_commit(&self, commits: &mut [IndexCommit]) -> Result<()> {
        let keep = commits.len().saturating_sub(1);
        for commit in &mut commits[..keep] {
            commit.delete();
        }
        Ok(())
    }
}

/// `NoDeletionPolicy.INSTANCE`: keeps every commit.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoDeletionPolicy;

impl IndexDeletionPolicy for NoDeletionPolicy {
    fn on_init(&self, _commits: &mut [IndexCommit]) -> Result<()> {
        Ok(())
    }
    fn on_commit(&self, _commits: &mut [IndexCommit]) -> Result<()> {
        Ok(())
    }
}

/// `KeepLastNCommitsDeletionPolicy`: keeps the newest `n` commits.
#[derive(Debug, Clone, Copy)]
pub struct KeepLastNCommitsDeletionPolicy {
    num_commits_to_keep: usize,
}

impl KeepLastNCommitsDeletionPolicy {
    pub fn new(num_commits_to_keep: i32) -> Result<Self> {
        if num_commits_to_keep <= 0 {
            return Err(Error::IllegalArgument(
                "number of recent commits to keep must be positive".into(),
            ));
        }
        Ok(KeepLastNCommitsDeletionPolicy {
            num_commits_to_keep: usize::try_from(num_commits_to_keep).unwrap_or(usize::MAX),
        })
    }
}

impl IndexDeletionPolicy for KeepLastNCommitsDeletionPolicy {
    fn on_init(&self, commits: &mut [IndexCommit]) -> Result<()> {
        self.on_commit(commits)
    }
    fn on_commit(&self, commits: &mut [IndexCommit]) -> Result<()> {
        let doomed = commits.len().saturating_sub(self.num_commits_to_keep);
        for commit in &mut commits[..doomed] {
            commit.delete();
        }
        Ok(())
    }
}

/// `SnapshotDeletionPolicy`'s synchronized state.
#[derive(Debug, Default)]
struct SnapshotState {
    /// `refCounts`: generation -> how many snapshots hold it.
    ref_counts: HashMap<i64, i32>,
    /// `indexCommits`: the snapshotted commits.
    index_commits: HashMap<i64, IndexCommit>,
    /// `lastCommit`.
    last_commit: Option<IndexCommit>,
    /// `initCalled`: set once a writer has handed this policy its commits.
    init_called: bool,
}

const NOT_USED_BY_WRITER: &str = "this instance is not being used by IndexWriter; be sure to use \
     the instance returned from writer.getConfig().getIndexDeletionPolicy()";

/// `SnapshotDeletionPolicy`: wraps a primary policy, and keeps every commit
/// a caller has [`Self::snapshot`]ted until [`Self::release`]d -- how a
/// backup copies a consistent commit while the writer keeps committing.
///
/// A released commit is not deleted until the policy next runs: the next
/// commit, or `IndexWriter::delete_unused_files`.
#[derive(Debug)]
pub struct SnapshotDeletionPolicy {
    primary: Box<dyn IndexDeletionPolicy>,
    state: Mutex<SnapshotState>,
}

impl SnapshotDeletionPolicy {
    pub fn new(primary: Box<dyn IndexDeletionPolicy>) -> Self {
        SnapshotDeletionPolicy {
            primary,
            state: Mutex::new(SnapshotState::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, SnapshotState> {
        // A panic while holding the lock leaves plain data behind; keep going.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs the primary policy over `commits`, carrying its deletions back
    /// only for commits no snapshot holds (`SnapshotCommitPoint.delete`).
    fn run_primary(
        &self,
        state: &SnapshotState,
        commits: &mut [IndexCommit],
        init: bool,
    ) -> Result<()> {
        let mut wrapped: Vec<IndexCommit> = commits.to_vec();
        if init {
            self.primary.on_init(&mut wrapped)?;
        } else {
            self.primary.on_commit(&mut wrapped)?;
        }
        for (commit, decided) in commits.iter_mut().zip(&wrapped) {
            if decided.is_deleted() && !state.ref_counts.contains_key(&commit.generation) {
                commit.delete();
            }
        }
        Ok(())
    }

    /// `snapshot()`: pins the most recent commit.
    pub fn snapshot(&self) -> Result<IndexCommit> {
        let mut state = self.lock();
        if !state.init_called {
            return Err(Error::IllegalState(NOT_USED_BY_WRITER.into()));
        }
        let Some(last) = state.last_commit.clone() else {
            return Err(Error::IllegalState("No index commit to snapshot".into()));
        };
        Self::inc_ref(&mut state, &last);
        Ok(last)
    }

    /// `incRef`.
    fn inc_ref(state: &mut SnapshotState, commit: &IndexCommit) {
        let gen = commit.generation;
        let count = match state.ref_counts.get(&gen) {
            None => {
                // Java stores `lastCommit` here, which `snapshot` just passed.
                state.index_commits.insert(gen, commit.clone());
                0
            }
            Some(c) => *c,
        };
        state.ref_counts.insert(gen, count.saturating_add(1));
    }

    /// `release(IndexCommit)`.
    pub fn release(&self, commit: &IndexCommit) -> Result<()> {
        self.release_gen(commit.generation)
    }

    /// `releaseGen(long)`.
    pub fn release_gen(&self, gen: i64) -> Result<()> {
        let mut state = self.lock();
        Self::release_locked(&mut state, gen)
    }

    fn release_locked(state: &mut SnapshotState, gen: i64) -> Result<()> {
        if !state.init_called {
            return Err(Error::IllegalState(NOT_USED_BY_WRITER.into()));
        }
        let Some(count) = state.ref_counts.get(&gen).copied() else {
            return Err(Error::IllegalArgument(format!(
                "commit gen={gen} is not currently snapshotted"
            )));
        };
        let count = count.saturating_sub(1);
        if count == 0 {
            state.ref_counts.remove(&gen);
            state.index_commits.remove(&gen);
        } else {
            state.ref_counts.insert(gen, count);
        }
        Ok(())
    }

    /// `getSnapshots`, oldest generation first.
    pub fn snapshots(&self) -> Vec<IndexCommit> {
        let state = self.lock();
        let mut out: Vec<IndexCommit> = state.index_commits.values().cloned().collect();
        out.sort_by_key(|c| c.generation);
        out
    }

    /// `getSnapshotCount`: snapshots taken and not released, counting repeats.
    pub fn snapshot_count(&self) -> i32 {
        self.lock()
            .ref_counts
            .values()
            .fold(0i32, |a, c| a.saturating_add(*c))
    }

    /// `getIndexCommit(gen)`.
    pub fn index_commit(&self, gen: i64) -> Option<IndexCommit> {
        self.lock().index_commits.get(&gen).cloned()
    }

    /// The snapshot reference counts, by generation, ascending.
    fn ref_counts_sorted(state: &SnapshotState) -> Vec<(i64, i32)> {
        let mut v: Vec<(i64, i32)> = state.ref_counts.iter().map(|(g, c)| (*g, *c)).collect();
        v.sort_unstable();
        v
    }
}

impl IndexDeletionPolicy for SnapshotDeletionPolicy {
    fn on_init(&self, commits: &mut [IndexCommit]) -> Result<()> {
        let mut state = self.lock();
        state.init_called = true;
        self.run_primary(&state, commits, true)?;
        for commit in commits.iter() {
            if state.ref_counts.contains_key(&commit.generation) {
                state
                    .index_commits
                    .insert(commit.generation, commit.clone());
            }
        }
        if let Some(last) = commits.last() {
            state.last_commit = Some(last.clone());
        }
        Ok(())
    }

    fn on_commit(&self, commits: &mut [IndexCommit]) -> Result<()> {
        let mut state = self.lock();
        self.run_primary(&state, commits, false)?;
        state.last_commit = commits.last().cloned();
        Ok(())
    }
}

/// `IndexWriterConfig.OpenMode`, as `PersistentSnapshotDeletionPolicy` reads
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenMode {
    /// Discard any snapshots stored in the directory.
    Create,
    /// Require stored snapshots.
    Append,
    /// Load stored snapshots if there are any.
    CreateOrAppend,
}

/// `PersistentSnapshotDeletionPolicy`: a [`SnapshotDeletionPolicy`] whose
/// snapshot reference counts are written to `snapshots_N` in the directory
/// on every change, so they survive a restart.
///
/// `snapshots_N` is `CodecUtil.writeHeader("snapshots", 0)`, a vint count,
/// then per snapshotted generation a vlong generation and a vint count.
pub struct PersistentSnapshotDeletionPolicy {
    inner: SnapshotDeletionPolicy,
    dir: Arc<dyn Directory>,
    next_write_gen: Mutex<i64>,
}

impl fmt::Debug for PersistentSnapshotDeletionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PersistentSnapshotDeletionPolicy")
            .field("inner", &self.inner)
            .field("last_save_file", &self.last_save_file())
            .finish()
    }
}

impl PersistentSnapshotDeletionPolicy {
    /// `SNAPSHOTS_PREFIX`.
    pub const SNAPSHOTS_PREFIX: &'static str = "snapshots_";
    const CODEC_NAME: &'static str = "snapshots";
    const VERSION_START: i32 = 0;
    const VERSION_CURRENT: i32 = Self::VERSION_START;

    /// `PersistentSnapshotDeletionPolicy(primary, dir, mode)`.
    pub fn new(
        primary: Box<dyn IndexDeletionPolicy>,
        dir: Arc<dyn Directory>,
        mode: OpenMode,
    ) -> Result<Self> {
        let policy = PersistentSnapshotDeletionPolicy {
            inner: SnapshotDeletionPolicy::new(primary),
            dir,
            next_write_gen: Mutex::new(0),
        };
        if mode == OpenMode::Create {
            policy.clear_prior_snapshots()?;
        }
        policy.load_prior_snapshots()?;
        if mode == OpenMode::Append && *policy.gen_lock() == 0 {
            return Err(Error::IllegalState(
                "no snapshots stored in this directory".into(),
            ));
        }
        Ok(policy)
    }

    fn gen_lock(&self) -> MutexGuard<'_, i64> {
        self.next_write_gen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// The wrapped [`SnapshotDeletionPolicy`] (`getSnapshots`, `getIndexCommit`,
    /// `getSnapshotCount`).
    pub fn snapshots_policy(&self) -> &SnapshotDeletionPolicy {
        &self.inner
    }

    /// `snapshot()`, persisted; a failed write undoes the snapshot.
    pub fn snapshot(&self) -> Result<IndexCommit> {
        let commit = self.inner.snapshot()?;
        if let Err(e) = self.persist() {
            let _ = self.inner.release(&commit);
            return Err(e);
        }
        Ok(commit)
    }

    /// `release(IndexCommit)`, persisted; a failed write restores the
    /// snapshot.
    pub fn release(&self, commit: &IndexCommit) -> Result<()> {
        self.inner.release(commit)?;
        if let Err(e) = self.persist() {
            SnapshotDeletionPolicy::inc_ref(&mut self.inner.lock(), commit);
            return Err(e);
        }
        Ok(())
    }

    /// `release(long gen)`, persisted.
    pub fn release_gen(&self, gen: i64) -> Result<()> {
        self.inner.release_gen(gen)?;
        self.persist()
    }

    /// `getLastSaveFile`.
    pub fn last_save_file(&self) -> Option<String> {
        let gen = *self.gen_lock();
        (gen != 0).then(|| format!("{}{}", Self::SNAPSHOTS_PREFIX, gen.saturating_sub(1)))
    }

    /// `persist`: writes `snapshots_<nextWriteGen>`, syncs it, deletes the
    /// previous one.
    fn persist(&self) -> Result<()> {
        let mut gen = self.gen_lock();
        let file_name = format!("{}{}", Self::SNAPSHOTS_PREFIX, *gen);
        let mut bytes: Vec<u8> = Vec::new();
        codec_util::write_header(&mut bytes, Self::CODEC_NAME, Self::VERSION_CURRENT);
        let entries = SnapshotDeletionPolicy::ref_counts_sorted(&self.inner.lock());
        bytes.write_vint(i32::try_from(entries.len()).unwrap_or(i32::MAX));
        for (commit_gen, count) in entries {
            bytes.write_vlong(commit_gen);
            bytes.write_vint(count);
        }
        let written = (|| -> lucene_store::Result<()> {
            let mut out = self.dir.create_output(&file_name)?;
            out.write_bytes(&bytes);
            out.close()?;
            Ok(())
        })();
        if let Err(e) = written {
            let _ = self.dir.delete_file(&file_name);
            return Err(e.into());
        }
        self.dir.sync(std::slice::from_ref(&file_name))?;
        if *gen > 0 {
            let last = format!("{}{}", Self::SNAPSHOTS_PREFIX, gen.saturating_sub(1));
            let _ = self.dir.delete_file(&last);
        }
        *gen = gen.saturating_add(1);
        Ok(())
    }

    /// `clearPriorSnapshots`.
    fn clear_prior_snapshots(&self) -> Result<()> {
        for file in self.dir.list_all()? {
            if file.starts_with(Self::SNAPSHOTS_PREFIX) {
                self.dir.delete_file(&file)?;
            }
        }
        Ok(())
    }

    /// `loadPriorSnapshots`: reads the newest `snapshots_N` (in listing
    /// order, as Java does) into the reference counts and deletes the others.
    fn load_prior_snapshots(&self) -> Result<()> {
        let mut gen_loaded: i64 = -1;
        let mut first_error: Option<Error> = None;
        let mut snapshot_files: Vec<String> = Vec::new();
        for file in self.dir.list_all()? {
            let Some(suffix) = file.strip_prefix(Self::SNAPSHOTS_PREFIX) else {
                continue;
            };
            let gen: i64 = suffix.parse().map_err(|_| {
                Error::IllegalArgument(format!("unparsable snapshots file name {file:?}"))
            })?;
            if gen_loaded == -1 || gen > gen_loaded {
                snapshot_files.push(file.clone());
                let mut m: HashMap<i64, i32> = HashMap::new();
                match self.read_snapshots_file(&file, &mut m) {
                    Ok(()) => {}
                    Err(e) => {
                        if first_error.is_none() {
                            first_error = Some(e);
                        }
                    }
                }
                gen_loaded = gen;
                let mut state = self.inner.lock();
                state.ref_counts = m;
            }
        }
        if gen_loaded == -1 {
            if let Some(e) = first_error {
                return Err(e);
            }
        } else {
            if snapshot_files.len() > 1 {
                let current = format!("{}{gen_loaded}", Self::SNAPSHOTS_PREFIX);
                for file in &snapshot_files {
                    if file != &current {
                        let _ = self.dir.delete_file(file);
                    }
                }
            }
            *self.gen_lock() = gen_loaded.saturating_add(1);
        }
        Ok(())
    }

    fn read_snapshots_file(&self, file: &str, m: &mut HashMap<i64, i32>) -> Result<()> {
        let bytes = self.dir.open(file)?;
        let mut input = SliceInput::new(&bytes);
        codec_util::check_header(
            &mut input,
            Self::CODEC_NAME,
            Self::VERSION_START,
            Self::VERSION_START,
        )?;
        let count = input.read_vint()?;
        for _ in 0..count.max(0) {
            let commit_gen = input.read_vlong()?;
            let ref_count = input.read_vint()?;
            m.insert(commit_gen, ref_count);
        }
        Ok(())
    }
}

impl IndexDeletionPolicy for PersistentSnapshotDeletionPolicy {
    fn on_init(&self, commits: &mut [IndexCommit]) -> Result<()> {
        self.inner.on_init(commits)
    }
    fn on_commit(&self, commits: &mut [IndexCommit]) -> Result<()> {
        self.inner.on_commit(commits)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use lucene_util::test_support::TempDir;

    fn commits(n: i64) -> Vec<IndexCommit> {
        (1..=n)
            .map(|g| {
                IndexCommit::new(
                    g,
                    format!("segments_{g}"),
                    vec![format!("segments_{g}")],
                    1,
                    vec![("k".into(), g.to_string())],
                )
            })
            .collect()
    }

    fn deleted(c: &[IndexCommit]) -> Vec<i64> {
        c.iter()
            .filter(|c| c.is_deleted())
            .map(|c| c.generation())
            .collect()
    }

    #[test]
    fn stateless_policies() {
        let mut c = commits(3);
        KeepOnlyLastCommitDeletionPolicy.on_init(&mut c).unwrap();
        assert_eq!(deleted(&c), vec![1, 2]);
        let mut c = commits(3);
        NoDeletionPolicy.on_init(&mut c).unwrap();
        NoDeletionPolicy.on_commit(&mut c).unwrap();
        assert!(deleted(&c).is_empty());
        assert!(KeepLastNCommitsDeletionPolicy::new(0).is_err());
        let p = KeepLastNCommitsDeletionPolicy::new(2).unwrap();
        let mut c = commits(5);
        p.on_init(&mut c).unwrap();
        assert_eq!(deleted(&c), vec![1, 2, 3]);
        let mut c = commits(1);
        p.on_commit(&mut c).unwrap();
        assert!(deleted(&c).is_empty());
        let c = &commits(1)[0];
        assert_eq!(c.segments_file_name(), "segments_1");
        assert_eq!(c.file_names(), ["segments_1"]);
        assert_eq!(c.segment_count(), 1);
        assert_eq!(c.user_data()[0].1, "1");
    }

    #[test]
    fn snapshots_pin_commits_until_released() {
        let p = SnapshotDeletionPolicy::new(Box::new(KeepOnlyLastCommitDeletionPolicy));
        assert!(matches!(p.snapshot(), Err(Error::IllegalState(_))));
        assert!(matches!(p.release_gen(1), Err(Error::IllegalState(_))));
        let mut empty: Vec<IndexCommit> = Vec::new();
        p.on_init(&mut empty).unwrap();
        assert!(matches!(p.snapshot(), Err(Error::IllegalState(_))));
        let mut c = commits(1);
        p.on_commit(&mut c).unwrap();
        let s1 = p.snapshot().unwrap();
        let s1b = p.snapshot().unwrap();
        assert_eq!(s1, s1b);
        assert_eq!(p.snapshot_count(), 2);
        let mut c = commits(2);
        p.on_commit(&mut c).unwrap();
        // gen 1 is snapshotted: the primary's delete is ignored.
        assert!(deleted(&c).is_empty());
        p.release(&s1).unwrap();
        let mut c = commits(2);
        p.on_commit(&mut c).unwrap();
        assert!(deleted(&c).is_empty());
        assert_eq!(p.snapshots().len(), 1);
        assert_eq!(p.index_commit(1).unwrap().generation(), 1);
        p.release(&s1b).unwrap();
        assert!(p.index_commit(1).is_none());
        let mut c = commits(2);
        p.on_commit(&mut c).unwrap();
        assert_eq!(deleted(&c), vec![1]);
        assert!(matches!(p.release_gen(1), Err(Error::IllegalArgument(_))));
    }

    #[test]
    fn on_init_rebinds_known_snapshots() {
        let p = SnapshotDeletionPolicy::new(Box::new(KeepOnlyLastCommitDeletionPolicy));
        p.lock().ref_counts.insert(1, 1);
        let mut c = commits(3);
        p.on_init(&mut c).unwrap();
        assert_eq!(deleted(&c), vec![2]);
        assert_eq!(p.index_commit(1).unwrap().generation(), 1);
    }

    #[test]
    fn persistent_snapshots_survive_reopen() {
        let tmp = TempDir::new("persistent-snapshots");
        let dir: Arc<dyn Directory> = Arc::new(lucene_store::FsDirectory::open(tmp.path()));
        let p = PersistentSnapshotDeletionPolicy::new(
            Box::new(KeepOnlyLastCommitDeletionPolicy),
            Arc::clone(&dir),
            OpenMode::CreateOrAppend,
        )
        .unwrap();
        assert!(p.last_save_file().is_none());
        assert!(matches!(
            PersistentSnapshotDeletionPolicy::new(
                Box::new(NoDeletionPolicy),
                Arc::clone(&dir),
                OpenMode::Append
            ),
            Err(Error::IllegalState(_))
        ));
        let mut c = commits(3);
        p.on_init(&mut c).unwrap();
        let s = p.snapshot().unwrap();
        assert_eq!(s.generation(), 3);
        p.snapshot().unwrap();
        assert_eq!(p.last_save_file().as_deref(), Some("snapshots_1"));
        assert_eq!(dir.list_all().unwrap(), vec!["snapshots_1".to_string()]);

        // A second policy over the same directory sees the stored counts.
        let q = PersistentSnapshotDeletionPolicy::new(
            Box::new(KeepOnlyLastCommitDeletionPolicy),
            Arc::clone(&dir),
            OpenMode::Append,
        )
        .unwrap();
        assert_eq!(q.snapshots_policy().snapshot_count(), 2);
        let mut c = commits(4);
        q.on_init(&mut c).unwrap();
        assert_eq!(deleted(&c), vec![1, 2]);
        q.release_gen(3).unwrap();
        q.release(&s).unwrap();
        assert!(q.release_gen(3).is_err());
        assert_eq!(q.snapshots_policy().snapshot_count(), 0);
        let mut c = commits(4);
        q.on_commit(&mut c).unwrap();
        assert_eq!(deleted(&c), vec![1, 2, 3]);

        // CREATE wipes them.
        let r = PersistentSnapshotDeletionPolicy::new(
            Box::new(NoDeletionPolicy),
            Arc::clone(&dir),
            OpenMode::Create,
        )
        .unwrap();
        assert!(r.last_save_file().is_none());
        assert!(dir.list_all().unwrap().is_empty());
    }

    #[test]
    fn corrupt_snapshots_file_is_an_error() {
        let tmp = TempDir::new("persistent-snapshots-corrupt");
        let dir: Arc<dyn Directory> = Arc::new(lucene_store::FsDirectory::open(tmp.path()));
        let mut out = dir.create_output("snapshots_0").unwrap();
        out.write_bytes(b"garbage");
        out.close().unwrap();
        // The newest file failed to read: its counts are empty, and it is
        // still the one loaded (Java records the error only when no file
        // loaded at all).
        let p = PersistentSnapshotDeletionPolicy::new(
            Box::new(NoDeletionPolicy),
            Arc::clone(&dir),
            OpenMode::CreateOrAppend,
        )
        .unwrap();
        assert_eq!(p.last_save_file().as_deref(), Some("snapshots_0"));
        let out = dir.create_output("snapshots_x").unwrap();
        out.close().unwrap();
        assert!(PersistentSnapshotDeletionPolicy::new(
            Box::new(NoDeletionPolicy),
            dir,
            OpenMode::CreateOrAppend,
        )
        .is_err());
    }
}
