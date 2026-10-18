//! `ReplicaNode`: an index that follows a primary by copying the files of
//! each of its NRT points.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Instant;

use lucene_index::index_file_deleter::files_of_infos;
use lucene_index::segment_infos::{self, SegmentInfos};
use lucene_store::directory::Directory;
use lucene_store::lock::Lock;

use super::copy::{CopyJob, OnceDone, ReplicaCore};
use super::node::{bytes_to_string, NodeCore, Printer, PRIMARY_GEN_KEY, VERSION_KEY};
use super::replica_file_deleter::ReplicaFileDeleter;
use super::searcher_manager::{describe, segments_file_name, SegmentInfosSearcherManager};
use super::types::{FileMetaData, NodeCommunicationError};
use crate::directory_reader::DirectoryReader;
use crate::{Error, Result};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn illegal_state(message: impl Into<String>) -> Error {
    Error::IllegalState(message.into())
}

/// The three abstract methods of `ReplicaNode`: how a copy job is made, how
/// it is run, and how the primary is told this replica exists.
pub trait ReplicaHooks: Send + Sync {
    /// `newCopyJob(reason, files, prevFiles, highPriority, onceDone)`.
    ///
    /// `files` is `None` for an NRT point, whose files come with the
    /// primary's copy state; `prev_files` are the files of the last point,
    /// which the primary may use to send less.
    ///
    /// # Errors
    /// [`Error::NodeCommunication`] when the primary cannot be reached; any
    /// other error is a failure of the job itself.
    fn new_copy_job(
        &self,
        core: &Arc<ReplicaCore>,
        reason: &str,
        files: Option<Arc<BTreeMap<String, FileMetaData>>>,
        prev_files: Option<Arc<BTreeMap<String, FileMetaData>>>,
        high_priority: bool,
        once_done: Option<OnceDone>,
    ) -> Result<Arc<dyn CopyJob>>;

    /// `launch(job)`: runs the job, then its `onceDone`.
    ///
    /// # Errors
    /// The job could not be queued.
    fn launch(&self, job: Arc<dyn CopyJob>) -> Result<()>;

    /// `sendNewReplica()`.
    ///
    /// # Errors
    /// The primary could not be told.
    fn send_new_replica(&self) -> Result<()>;
}

#[derive(Default)]
struct ReplicaState {
    /// `lastCommitFiles`.
    last_commit_files: BTreeSet<String>,
    /// `lastNRTFiles`.
    last_nrt_files: BTreeSet<String>,
    /// `curNRTCopy`.
    cur_nrt_copy: Option<Arc<dyn CopyJob>>,
    /// `lastPrimaryGen`.
    last_primary_gen: i64,
}

/// `ReplicaNode`.
pub struct ReplicaNode {
    core: Arc<ReplicaCore>,
    hooks: Arc<dyn ReplicaHooks>,
    state: Mutex<ReplicaState>,
    /// `commitLock`.
    commit_lock: Mutex<()>,
    /// `pendingMergeFiles`.
    pending_merge_files: Mutex<BTreeSet<String>>,
    /// `mergeCopyJobs`.
    merge_copy_jobs: Mutex<Vec<Arc<dyn CopyJob>>>,
    /// `writeFileLock`.
    write_file_lock: Mutex<Option<Box<dyn Lock>>>,
    /// `mgr`.
    mgr: Mutex<Option<Arc<SegmentInfosSearcherManager>>>,
}

impl ReplicaNode {
    /// `new ReplicaNode(id, dir, searcherFactory, printStream)`.
    ///
    /// # Errors
    /// The directory still has pending deletions, or the write lock is held.
    pub fn new(
        id: i32,
        dir: Arc<dyn Directory>,
        printer: Option<Printer>,
        hooks: Arc<dyn ReplicaHooks>,
    ) -> Result<Arc<Self>> {
        if !dir.pending_deletions()?.is_empty() {
            return Err(Error::IllegalArgument(
                "Directory still has pending deleted files; cannot initialize IndexWriter"
                    .to_string(),
            ));
        }
        let node = Arc::new(NodeCore::new(id, false, Arc::clone(&dir), printer));
        node.message("top: init replica");
        let write_lock = dir.obtain_lock("write.lock")?;
        node.set_state("init");
        let deleter = ReplicaFileDeleter::new(Arc::clone(&node), Arc::clone(&dir));
        Ok(Arc::new(Self {
            core: Arc::new(ReplicaCore::new(node, deleter)),
            hooks,
            state: Mutex::new(ReplicaState::default()),
            commit_lock: Mutex::new(()),
            pending_merge_files: Mutex::new(BTreeSet::new()),
            merge_copy_jobs: Mutex::new(Vec::new()),
            write_file_lock: Mutex::new(Some(write_lock)),
            mgr: Mutex::new(None),
        }))
    }

    /// The node's shared half: log, state, deleter.
    pub fn core(&self) -> &Arc<ReplicaCore> {
        &self.core
    }

    /// `name()`.
    pub fn name(&self) -> String {
        self.core.node.name()
    }

    fn message(&self, text: &str) {
        self.core.node.message(text);
    }

    fn manager(&self) -> Result<Arc<SegmentInfosSearcherManager>> {
        lock(&self.mgr)
            .clone()
            .ok_or_else(|| illegal_state("replica was not started"))
    }

    /// `getSearcherManager()`'s acquire: the current reader.
    ///
    /// # Errors
    /// The node is not started or is closed.
    pub fn acquire(&self) -> Result<Arc<DirectoryReader>> {
        self.manager()?.acquire()
    }

    /// `getCurrentSearchingVersion()`.
    ///
    /// # Errors
    /// The node is not started or is closed.
    pub fn current_searching_version(&self) -> Result<i64> {
        Ok(self.acquire()?.segment_infos.version)
    }

    /// `isClosed()`.
    pub fn is_closed(&self) -> bool {
        matches!(
            self.core.node.state().as_str(),
            "closed" | "closing" | "crashing" | "crashed"
        )
    }

    /// `isCopying()`.
    pub fn is_copying(&self) -> bool {
        lock(&self.state).cur_nrt_copy.is_some()
    }

    /// The node's state string.
    pub fn state(&self) -> String {
        self.core.node.state()
    }

    /// `start(curPrimaryGen)`: reads the last commit, deletes what it does
    /// not name, syncs from the primary when that moved on while this
    /// replica was down, and opens the first reader.
    ///
    /// # Errors
    /// The directory cannot be read, or the sync failed ("replica cannot
    /// start" when the stale commit cannot be removed).
    pub fn start(self: &Arc<Self>, cur_primary_gen: i64) -> Result<()> {
        if self.core.node.state() != "init" {
            return Err(illegal_state("already started"));
        }
        self.message("top: now start");
        let result = self.start_inner(cur_primary_gen);
        if let Err(e) = &result {
            if e.to_string().starts_with("replica cannot start") {
                // Java closes the directory here, which frees the write lock.
                if let Some(write_lock) = lock(&self.write_file_lock).take() {
                    // The start error is the one reported.
                    let _ = write_lock.close();
                }
            } else {
                self.message(&format!("exc on start: {e}"));
            }
        }
        result
    }

    fn start_inner(self: &Arc<Self>, cur_primary_gen: i64) -> Result<()> {
        let dir = Arc::clone(self.core.dir());
        let names = dir.list_all()?;
        let generation = lucene_store::directory::last_commit_generation(&names)?;
        let segments_file = lucene_store::directory::segments_file_name(generation);
        let mut max_pending_gen = -1i64;
        for name in &names {
            if let Some(suffix) = name.strip_prefix("pending_segments_") {
                let gen = lucene_util::base36::from_base36(suffix).ok_or_else(|| {
                    Error::Store(lucene_store::Error::Corrupted(format!(
                        "fileName \"{name}\" is not a pending segments file"
                    )))
                })?;
                max_pending_gen = max_pending_gen.max(gen);
            }
        }
        let mut st = lock(&self.state);
        let mut infos: SegmentInfos;
        match &segments_file {
            None => {
                infos = empty_infos();
                self.message("top: init: no segments in index");
            }
            Some(name) => {
                self.message(&format!("top: init: read existing segments commit {name}"));
                let bytes = dir.open(name)?.to_vec();
                infos = segment_infos::parse(&bytes, generation)?;
                self.message(&format!(
                    "top: init: segments: {} version={}",
                    describe(&infos),
                    infos.version
                ));
                let index_files = files_of_infos(dir.as_ref(), &infos, false)?;
                st.last_commit_files.insert(name.clone());
                st.last_commit_files.extend(index_files.iter().cloned());
                self.core
                    .deleter
                    .inc_ref(st.last_commit_files.iter().map(String::as_str));
                st.last_nrt_files.extend(index_files);
                self.core
                    .deleter
                    .inc_ref(st.last_nrt_files.iter().map(String::as_str));
                self.message(&format!("top: commitFiles={:?}", st.last_commit_files));
                self.message(&format!("top: nrtFiles={:?}", st.last_nrt_files));
            }
        }
        self.message(&format!(
            "top: delete unknown files on init: all files={:?}",
            dir.list_all()?
        ));
        self.core
            .deleter
            .delete_unknown_files(segments_file.as_deref())?;
        self.message(&format!(
            "top: done delete unknown files on init: all files={:?}",
            dir.list_all()?
        ));

        let my_primary_gen = match user_data(&infos, PRIMARY_GEN_KEY) {
            None => -1,
            Some(s) => s.parse::<i64>().map_err(|e| {
                Error::Store(lucene_store::Error::Corrupted(format!(
                    "{PRIMARY_GEN_KEY}={s}: {e}"
                )))
            })?,
        };
        self.message(&format!("top: myPrimaryGen={my_primary_gen}"));

        let do_commit;
        if !infos.segments.is_empty() && my_primary_gen != -1 && my_primary_gen != cur_primary_gen {
            let init_sync_start = Instant::now();
            self.message(&format!(
                "top: init: primary changed while we were down myPrimaryGen={my_primary_gen} vs curPrimaryGen={cur_primary_gen}; sync now before mgr init"
            ));
            let segments_name = segments_file.clone().unwrap_or_default();
            self.message(&format!(
                "top: now delete starting commit point {segments_name}"
            ));
            debug_assert_eq!(self.core.deleter.ref_count(&segments_name), 1);
            self.core.deleter.dec_ref([segments_name.as_str()])?;
            if !dir.pending_deletions()?.is_empty() {
                return Err(illegal_state(format!(
                    "replica cannot start: existing segments file={segments_name} must be removed in order to start, but the file delete failed"
                )));
            }
            st.last_commit_files.remove(&segments_name);
            let job = loop {
                let job = self.hooks.new_copy_job(
                    &self.core,
                    &format!(
                        "sync on startup replica={} myVersion={}",
                        self.name(),
                        infos.version
                    ),
                    None,
                    None,
                    true,
                    None,
                )?;
                job.start()?;
                let version = job.copy_state().map_or(-1, |c| c.version);
                self.message(&format!("top: init: sync sis.version={version}"));
                let outcome = job.run_blocking().and_then(|()| job.finish());
                match outcome {
                    Ok(()) => break job,
                    Err(e) => {
                        job.cancel("startup failed", Some(&e.to_string()))?;
                        if e.to_string().contains("checksum mismatch after file copy") {
                            self.message(&format!("top: failed to copy: {e}; retrying"));
                        } else {
                            return Err(e);
                        }
                    }
                }
            };
            let copy_state = job
                .copy_state()
                .ok_or_else(|| illegal_state("sync job without a copy state"))?
                .clone();
            st.last_primary_gen = copy_state.primary_gen;
            let mut sync_infos = segment_infos::parse(&copy_state.infos_bytes, copy_state.gen)?;
            sync_infos.generation = infos.generation;
            infos = sync_infos;
            debug_assert_eq!(infos.version, copy_state.version);
            self.message(&format!(
                "  version={} segments={}",
                infos.version,
                describe(&infos)
            ));
            let job_files = job.file_names();
            self.message(&format!("top: init: incRef nrtFiles={job_files:?}"));
            self.core
                .deleter
                .inc_ref(job_files.iter().map(String::as_str));
            self.message(&format!(
                "top: init: decRef lastNRTFiles={:?}",
                st.last_nrt_files
            ));
            self.core
                .deleter
                .dec_ref(st.last_nrt_files.iter().map(String::as_str))?;
            st.last_nrt_files = job_files;
            self.message(&format!(
                "top: init: set lastNRTFiles={:?}",
                st.last_nrt_files
            ));
            self.core
                .node
                .set_last_file_meta_data(Some(Arc::clone(&copy_state.files)));
            self.message(&format!(
                "top: {}: start: done sync: took {:.3}s for {}, opened NRT reader version={}",
                self.core.node.id,
                init_sync_start.elapsed().as_secs_f64(),
                bytes_to_string(job.total_bytes_copied()),
                copy_state.version
            ));
            do_commit = true;
        } else {
            do_commit = false;
            st.last_primary_gen = cur_primary_gen;
            self.message("top: same primary as before");
        }

        if infos.generation < max_pending_gen {
            self.message(&format!(
                "top: move infos generation from {} to {max_pending_gen}",
                infos.generation
            ));
            infos.generation = max_pending_gen;
        }
        drop(st);

        self.hooks.send_new_replica()?;
        *lock(&self.mgr) = Some(Arc::new(SegmentInfosSearcherManager::new(
            dir,
            Arc::clone(&self.core.node),
            infos,
        )?));
        if do_commit {
            self.commit()?;
        }
        self.message("top: done start");
        self.core.node.set_state("idle");
        Ok(())
    }

    /// `commit()`: writes the current infos as a `segments_N` carrying the
    /// primary generation and version, syncing the files it names first.
    ///
    /// # Errors
    /// Syncing or writing failed.
    pub fn commit(&self) -> Result<()> {
        let _commit = lock(&self.commit_lock);
        let mgr = self.manager()?;
        let dir = Arc::clone(self.core.dir());
        let (mut infos, index_files) = {
            let _node = lock(&self.state);
            let infos = mgr.current_infos();
            let index_files = files_of_infos(dir.as_ref(), &infos, false)?;
            self.core
                .deleter
                .inc_ref(index_files.iter().map(String::as_str));
            (infos, index_files)
        };
        let last_primary_gen = lock(&self.state).last_primary_gen;
        self.message(&format!(
            "top: commit primaryGen={last_primary_gen} infos={} files={index_files:?}",
            describe(&infos)
        ));
        dir.sync(&index_files)?;
        let commit_data = vec![
            (PRIMARY_GEN_KEY.to_string(), last_primary_gen.to_string()),
            (
                VERSION_KEY.to_string(),
                self.current_searching_version()?.to_string(),
            ),
        ];
        infos.user_data = commit_data.clone();
        infos.generation = if infos.generation < 0 {
            1
        } else {
            infos.generation.saturating_add(1)
        };
        let segments_file = segment_infos::write(&infos, dir.as_ref())?;
        mgr.committed(infos.version, infos.generation, commit_data.clone());
        self.message(&format!(
            "top: commit wrote segments file {segments_file} version={} sis={} commitData={commit_data:?}",
            infos.version,
            describe(&infos)
        ));
        self.core.deleter.inc_ref([segments_file.as_str()]);
        let mut st = lock(&self.state);
        self.message(&format!(
            "top: commit decRef lastCommitFiles={:?}",
            st.last_commit_files
        ));
        self.core
            .deleter
            .dec_ref(st.last_commit_files.iter().map(String::as_str))?;
        st.last_commit_files = index_files.into_iter().collect();
        st.last_commit_files.insert(segments_file);
        self.message(&format!(
            "top: commit version={} files now {:?}",
            infos.version, st.last_commit_files
        ));
        Ok(())
    }

    /// `finishNRTCopy(job, startNS)`: the job's files are in place; switch
    /// to the point they belong to.
    ///
    /// # Errors
    /// Reading the new infos or opening the reader failed.
    pub fn finish_nrt_copy(&self, job: &dyn CopyJob, start: Instant) -> Result<()> {
        let copy_state = job
            .copy_state()
            .ok_or_else(|| illegal_state("NRT copy job without a copy state"))?
            .clone();
        self.message(&format!(
            "top: finishNRTCopy: version={}{} job={}",
            copy_state.version,
            if job.failed() { " FAILED" } else { "" },
            job.core().reason
        ));
        {
            let mut st = lock(&self.state);
            if self.core.node.state() == "syncing" {
                self.core.node.set_state("idle");
            }
            let is_current = st
                .cur_nrt_copy
                .as_ref()
                .is_some_and(|c| std::ptr::eq(c.core(), job.core()));
            if is_current {
                self.message("top: now clear curNRTCopy");
                st.cur_nrt_copy = None;
            } else {
                debug_assert!(job.failed());
                self.message("top: skip clear curNRTCopy: we were cancelled");
            }
            if job.failed() {
                return Ok(());
            }
            job.finish()?;
            let infos = segment_infos::parse(&copy_state.infos_bytes, copy_state.gen)?;
            debug_assert_eq!(infos.version, copy_state.version);
            self.message(&format!(
                "  version={} segments={}",
                infos.version,
                describe(&infos)
            ));
            self.manager()?.set_current_infos(infos)?;
            let new_files: BTreeSet<String> = copy_state.files.keys().cloned().collect();
            self.message(&format!("top: incRef newNRTFiles={new_files:?}"));
            self.core
                .deleter
                .inc_ref(new_files.iter().map(String::as_str));
            {
                let mut pending = lock(&self.pending_merge_files);
                for f in &new_files {
                    pending.remove(f);
                }
                self.message(&format!(
                    "top: after remove from pending merges pendingMergeFiles={pending:?}"
                ));
            }
            self.message(&format!("top: decRef lastNRTFiles={:?}", st.last_nrt_files));
            self.core
                .deleter
                .dec_ref(st.last_nrt_files.iter().map(String::as_str))?;
            st.last_nrt_files = new_files;
            self.message(&format!("top: set lastNRTFiles={:?}", st.last_nrt_files));
            if !copy_state.completed_merge_files.is_empty() {
                self.message(&format!(
                    "now remove-if-not-ref'd completed merge files: {:?}",
                    copy_state.completed_merge_files
                ));
                for f in copy_state.completed_merge_files.iter() {
                    if lock(&self.pending_merge_files).remove(f) {
                        self.core.deleter.delete_if_no_ref(f)?;
                    }
                }
            }
            self.core
                .node
                .set_last_file_meta_data(Some(Arc::clone(&copy_state.files)));
        }
        self.message(&format!(
            "top: done sync: took {:.3}s for {}, opened NRT reader version={}",
            start.elapsed().as_secs_f64(),
            bytes_to_string(job.total_bytes_copied()),
            copy_state.version
        ));
        Ok(())
    }

    /// `newNRTPoint(newPrimaryGen, version)`: starts copying the primary's
    /// point `version`; `None` when this replica is already at or past it,
    /// or the primary could not be reached.
    ///
    /// # Errors
    /// The node is closed or the job failed to start.
    pub fn new_nrt_point(
        self: &Arc<Self>,
        new_primary_gen: i64,
        version: i64,
    ) -> Result<Option<Arc<dyn CopyJob>>> {
        if self.is_closed() {
            return Err(Error::AlreadyClosed(format!(
                "this replica is closed: state={}",
                self.state()
            )));
        }
        let mut st = lock(&self.state);
        self.maybe_new_primary(&mut st, new_primary_gen)?;
        if self.core.node.state() == "idle" {
            self.core.node.set_state("syncing");
        }
        let cur_version = self.current_searching_version()?;
        self.message(&format!("top: start sync sis.version={version}"));
        if version == cur_version {
            self.message("top: new NRT point has same version as current; skipping");
            return Ok(None);
        }
        if version < cur_version {
            // SENTINEL-OK: `current (` below is Java's log text, not a call.
            self.message(&format!(
                "top: new NRT point (version={version}) is older than current (version={cur_version}); skipping"
            ));
            return Ok(None);
        }
        let start = Instant::now();
        self.message("top: newNRTPoint");
        let weak: Weak<Self> = Arc::downgrade(self);
        let once_done: OnceDone = Arc::new(move |job: &dyn CopyJob| match weak.upgrade() {
            Some(node) => node.finish_nrt_copy(job, start),
            None => Ok(()),
        });
        let job = match self.hooks.new_copy_job(
            &self.core,
            &format!("NRT point sync version={version}"),
            None,
            self.core.node.last_file_meta_data(),
            true,
            Some(once_done),
        ) {
            Ok(job) => job,
            Err(Error::NodeCommunication(e)) => {
                self.message(&format!(
                    "top: ignoring communication exception creating CopyJob: {e}"
                ));
                if self.core.node.state() == "syncing" {
                    self.core.node.set_state("idle");
                }
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        debug_assert_eq!(
            job.copy_state().map(|c| c.primary_gen),
            Some(new_primary_gen)
        );
        self.message(&format!(
            "top: newNRTPoint: job files={:?}",
            job.file_names()
        ));
        if let Some(prev) = st.cur_nrt_copy.take() {
            job.transfer_and_cancel(prev.as_ref())?;
            debug_assert!(prev.failed());
        }
        st.cur_nrt_copy = Some(Arc::clone(&job));
        for file_name in job.file_names_to_copy() {
            debug_assert!(
                !st.last_commit_files.contains(&file_name),
                "fileName={file_name} is in lastCommitFiles and is being copied?"
            );
            for merge_job in lock(&self.merge_copy_jobs).iter() {
                if merge_job.file_names().contains(&file_name) {
                    self.message(&format!(
                        "top: now cancel merge copy job={}: file {file_name} is now being copied via NRT point",
                        merge_job.core().reason
                    ));
                    merge_job.cancel("newNRTPoint is copying over the same file", None)?;
                }
            }
        }
        match job.start() {
            Ok(()) => {}
            Err(Error::NodeCommunication(e)) => {
                self.message(&format!("top: ignoring exception starting CopyJob: {e}"));
                if self.core.node.state() == "syncing" {
                    self.core.node.set_state("idle");
                }
                return Ok(None);
            }
            Err(e) => return Err(e),
        }
        self.hooks.launch(Arc::clone(&job))?;
        Ok(Some(job))
    }

    fn maybe_new_primary(
        &self,
        st: &mut MutexGuard<'_, ReplicaState>,
        new_primary_gen: i64,
    ) -> Result<()> {
        if new_primary_gen != st.last_primary_gen {
            let pending: Vec<String> = lock(&self.pending_merge_files).iter().cloned().collect();
            self.message(&format!(
                "top: now change lastPrimaryGen from {} to {new_primary_gen} pendingMergeFiles={pending:?}",
                st.last_primary_gen
            ));
            self.message(&format!(
                "top: delete if no ref pendingMergeFiles={pending:?}"
            ));
            for f in &pending {
                self.core.deleter.delete_if_no_ref(f)?;
            }
            debug_assert!(new_primary_gen > st.last_primary_gen);
            st.last_primary_gen = new_primary_gen;
            lock(&self.pending_merge_files).clear();
        } else {
            self.message(&format!(
                "top: keep current lastPrimaryGen={}",
                st.last_primary_gen
            ));
        }
        Ok(())
    }

    /// `launchPreCopyMerge(finished, newPrimaryGen, files)`: copies the files
    /// of a merged segment before the NRT point that shows it arrives.
    ///
    /// # Errors
    /// The job could not be made, started or launched.
    pub fn launch_pre_copy_merge(
        self: &Arc<Self>,
        finished: Arc<AtomicBool>,
        new_primary_gen: i64,
        files: BTreeMap<String, FileMetaData>,
    ) -> Result<Arc<dyn CopyJob>> {
        let mut st = lock(&self.state);
        self.maybe_new_primary(&mut st, new_primary_gen)?;
        let primary_gen_start = st.last_primary_gen;
        let file_names: BTreeSet<String> = files.keys().cloned().collect();
        self.message(&format!(
            "now pre-copy warm merge files={file_names:?} primaryGen={new_primary_gen}"
        ));
        for f in &file_names {
            debug_assert!(
                !lock(&self.pending_merge_files).contains(f),
                "file \"{f}\" is already being warmed!"
            );
            debug_assert!(
                !st.last_nrt_files.contains(f),
                "file \"{f}\" is already NRT visible!"
            );
        }
        let weak: Weak<Self> = Arc::downgrade(self);
        let names = file_names.clone();
        let once_done: OnceDone = Arc::new(move |job: &dyn CopyJob| {
            let Some(node) = weak.upgrade() else {
                finished.store(true, Ordering::SeqCst);
                return Ok(());
            };
            node.merge_copy_jobs_remove(job);
            node.message(&format!(
                "done warming merge {names:?} failed?={}",
                job.failed()
            ));
            {
                let st = lock(&node.state);
                if !job.failed() {
                    if st.last_primary_gen != primary_gen_start {
                        node.message(&format!(
                            "merge pre copy finished but primary has changed; cancelling job files={names:?}"
                        ));
                        job.cancel("primary changed during merge copy", None)?;
                    } else {
                        let mut abort = false;
                        for f in &names {
                            if st.last_nrt_files.contains(f) {
                                node.message(&format!(
                                    "abort merge finish: file {f} is referenced by last NRT point"
                                ));
                                abort = true;
                            }
                            if st.last_commit_files.contains(f) {
                                node.message(&format!(
                                    "abort merge finish: file {f} is referenced by last commit point"
                                ));
                                abort = true;
                            }
                        }
                        if abort {
                            job.cancel("merged segment was separately copied via NRT point", None)?;
                        } else {
                            job.finish()?;
                            node.message(&format!("merge pre copy finished files={names:?}"));
                            for f in &names {
                                node.message(&format!("add file {f} to pendingMergeFiles"));
                                lock(&node.pending_merge_files).insert(f.clone());
                            }
                        }
                    }
                } else {
                    node.message("merge copy finished with failure");
                }
            }
            finished.store(true, Ordering::SeqCst);
            Ok(())
        });
        let job = self.hooks.new_copy_job(
            &self.core,
            &format!("warm merge on {} filesNames={file_names:?}", self.name()),
            Some(Arc::new(files)),
            None,
            false,
            Some(once_done),
        )?;
        job.start()?;
        debug_assert_eq!(job.file_names_to_copy().len(), job.file_names().len());
        lock(&self.merge_copy_jobs).push(Arc::clone(&job));
        self.hooks.launch(Arc::clone(&job))?;
        Ok(job)
    }

    fn merge_copy_jobs_remove(&self, job: &dyn CopyJob) {
        lock(&self.merge_copy_jobs).retain(|j| !std::ptr::eq(j.core(), job.core()));
    }

    /// `createTempOutput(prefix, suffix, ioContext)`.
    ///
    /// # Errors
    /// The directory could not create the file.
    pub fn create_temp_output(
        &self,
        prefix: &str,
        suffix: &str,
    ) -> Result<lucene_store::FsIndexOutput> {
        self.core.create_temp_output(prefix, suffix)
    }

    /// `getFilesToCopy(files)`.
    ///
    /// # Errors
    /// Reading a local file failed.
    pub fn files_to_copy(
        &self,
        files: &BTreeMap<String, FileMetaData>,
    ) -> Result<Vec<(String, FileMetaData)>> {
        self.core.files_to_copy(files)
    }

    /// `close()`: cancels the copy in flight, drops the references of the
    /// NRT point and releases the directory's write lock.
    ///
    /// # Errors
    /// Deleting files or closing the manager failed.
    pub fn close(&self) -> Result<()> {
        self.message("top: now close");
        {
            let st = lock(&self.state);
            self.core.node.set_state("closing");
            if let Some(job) = &st.cur_nrt_copy {
                job.cancel("closing", None)?;
            }
        }
        {
            let mut st = lock(&self.state);
            self.message("top: close mgr");
            if let Some(mgr) = lock(&self.mgr).as_ref() {
                mgr.close()?;
            }
            self.message(&format!("top: decRef lastNRTFiles={:?}", st.last_nrt_files));
            self.core
                .deleter
                .dec_ref(st.last_nrt_files.iter().map(String::as_str))?;
            st.last_nrt_files.clear();
            st.last_commit_files.clear();
            let pending: Vec<String> = std::mem::take(&mut *lock(&self.pending_merge_files))
                .into_iter()
                .collect();
            self.message(&format!(
                "top: delete if no ref pendingMergeFiles={pending:?}"
            ));
            for f in &pending {
                self.core.deleter.delete_if_no_ref(f)?;
            }
            self.message("top: close dir");
            if let Some(write_lock) = lock(&self.write_file_lock).take() {
                write_lock.close()?;
            }
        }
        self.message("top: done close");
        self.core.node.set_state("closed");
        Ok(())
    }

    /// The error a hook returns when the primary cannot be reached.
    pub fn communication_error(when: &str, cause: &str) -> Error {
        Error::NodeCommunication(NodeCommunicationError {
            when: when.to_string(),
            cause: cause.to_string(),
        })
    }

    /// `startCopyFile(name)`.
    ///
    /// # Errors
    /// The file is already being copied.
    pub fn start_copy_file(&self, name: &str) -> Result<()> {
        self.core.start_copy_file(name)
    }

    /// `finishCopyFile(name)`.
    ///
    /// # Errors
    /// The file was not being copied.
    pub fn finish_copy_file(&self, name: &str) -> Result<()> {
        self.core.finish_copy_file(name)
    }

    /// `deleter.getRefCount(fileName)`, for tests and tools.
    pub fn ref_count(&self, file_name: &str) -> i32 {
        self.core.deleter.ref_count(file_name)
    }

    /// The commit the infos currently name, e.g. `segments_3`.
    ///
    /// # Errors
    /// The node is not started.
    pub fn current_segments_file_name(&self) -> Result<String> {
        Ok(segments_file_name(&self.manager()?.current_infos()))
    }
}

fn user_data<'a>(infos: &'a SegmentInfos, key: &str) -> Option<&'a str> {
    infos
        .user_data
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// `new SegmentInfos(Version.LATEST.major)`: an index with no commit.
pub(super) fn empty_infos() -> SegmentInfos {
    SegmentInfos {
        id: [0u8; lucene_store::codec_util::ID_LENGTH],
        generation: -1,
        format_version: segment_infos::VERSION_86,
        lucene_version: segment_infos::LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
        index_created_version_major: 10,
        version: 0,
        counter: 0,
        min_segment_lucene_version: None,
        segments: Vec::new(),
        user_data: Vec::new(),
    }
}
