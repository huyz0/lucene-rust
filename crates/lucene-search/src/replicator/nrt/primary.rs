//! `PrimaryNode` and `PreCopyMergedSegmentWarmer`: the node that owns the
//! `IndexWriter` and publishes its NRT points.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicI32, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use lucene_index::index_file_deleter::{files_of_infos, segment_files};
use lucene_index::index_writer::{self, IndexWriter, MergedSegmentWarmer};
use lucene_index::nrt::{NrtFileHold, NrtSource};
use lucene_index::segment_infos::{self, SegmentCommitInfo, SegmentInfos};
use lucene_store::directory::Directory;

use super::copy::CopyStream;
use super::node::{bytes_to_string, NodeCore, Printer, PRIMARY_GEN_KEY, VERSION_KEY};
use super::types::{CopyState, FileMetaData};
use crate::directory_reader::DirectoryReader;
use crate::{Error, Result};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// `PrimaryNode.preCopyMergedSegmentFiles`: the abstract method a transport
/// implements to push a merged segment's files to the replicas.
pub trait PrimaryHooks: Send + Sync {
    /// Copies the files of `info`, a merge that just finished, to the
    /// replicas before the NRT point that shows it.
    ///
    /// # Errors
    /// A replica could not be reached.
    fn pre_copy_merged_segment_files(
        &self,
        info: &SegmentCommitInfo,
        files: &BTreeMap<String, FileMetaData>,
    ) -> Result<()>;
}

/// What the writer's merged-segment warmer needs of the primary, kept apart
/// from the writer's lifetime so the warmer can be handed to the writer.
pub struct PrimaryShared {
    core: Arc<NodeCore>,
    hooks: Arc<dyn PrimaryHooks>,
    /// `finishedMergedFiles`.
    finished_merged_files: Mutex<BTreeSet<String>>,
    state: Mutex<PrimaryState>,
    /// `copyingCount`.
    copying_count: AtomicI32,
}

impl PrimaryShared {
    fn closed(&self, allow_closing: bool) -> bool {
        let state = self.core.state();
        state == "closed" || (!allow_closing && state == "closing")
    }

    fn ensure_open(&self, allow_closing: bool) -> Result<()> {
        if self.closed(allow_closing) {
            return Err(Error::AlreadyClosed(self.core.state()));
        }
        Ok(())
    }

    /// `getCopyState()`: the current point, pinned until it is released.
    ///
    /// # Errors
    /// The node is closed.
    pub fn copy_state(&self) -> Result<CopyState> {
        self.ensure_open(false)?;
        let st = lock(&self.state);
        let copy_state = st
            .copy_state
            .clone()
            .ok_or_else(|| Error::IllegalState("no copy state".to_string()))?;
        let count = self.copying_count.fetch_add(1, Ordering::SeqCst);
        debug_assert!(count >= 0);
        Ok(copy_state)
    }

    /// `releaseCopyState(copyState)`.
    pub fn release_copy_state(&self, copy_state: CopyState) {
        drop(copy_state);
        let count = self.copying_count.fetch_sub(1, Ordering::SeqCst);
        debug_assert!(count > 0);
    }

    /// The bytes of the files named in `requests` -- each from its offset to
    /// the end, preceded by its length -- as the stream a replica copies
    /// from.
    ///
    /// # Errors
    /// A file is missing.
    pub fn open_copy_stream(&self, requests: &[(String, u64)]) -> Result<Box<dyn CopyStream>> {
        Ok(Box::new(super::local::FileStream::open(
            self.core.dir.as_ref(),
            requests,
        )?))
    }
}

/// `PreCopyMergedSegmentWarmer`: before a merge is published, ships the
/// merged segment to the replicas, so the NRT point that shows it finds the
/// files already there.
struct PreCopyMergedSegmentWarmer {
    primary: Arc<PrimaryShared>,
}

impl MergedSegmentWarmer for PreCopyMergedSegmentWarmer {
    fn warm(&self, dir: &dyn Directory, segment: &SegmentCommitInfo) -> index_writer::Result<()> {
        let started = Instant::now();
        let warm = || -> Result<()> {
            let mut files_meta_data = BTreeMap::new();
            for file_name in segment_files(dir, segment).map_err(|e| {
                Error::IllegalState(format!("segment {}: {e}", segment.segment_name))
            })? {
                let meta = self
                    .primary
                    .core
                    .read_local_file_metadata(&file_name)?
                    .ok_or_else(|| {
                        Error::IllegalState(format!("file \"{file_name}\" is missing metadata"))
                    })?;
                files_meta_data.insert(file_name, meta);
            }
            self.primary
                .hooks
                .pre_copy_merged_segment_files(segment, &files_meta_data)?;
            let size: u64 = files_meta_data.values().map(|m| m.length).sum();
            self.primary.core.message(&format!(
                "top: done warm merge {}: took {:.3} sec, {:.1} MB",
                segment.segment_name,
                started.elapsed().as_secs_f64(),
                size as f64 / 1024.0 / 1024.0
            ));
            lock(&self.primary.finished_merged_files).extend(files_meta_data.into_keys());
            Ok(())
        };
        warm().map_err(|e| index_writer::Error::MergeHook(e.to_string()))
    }
}

struct PrimaryState {
    /// `curInfos` and the pin on its files (`incRefDeleter(curInfos)`).
    cur: Option<(SegmentInfos, Arc<NrtFileHold>)>,
    /// `copyState`.
    copy_state: Option<CopyState>,
}

/// `PrimaryNode`.
pub struct PrimaryNode<'w, 'd> {
    shared: Arc<PrimaryShared>,
    writer: &'w Mutex<IndexWriter<'d>>,
    primary_gen: i64,
    /// `remoteCloseTimeoutMs`.
    remote_close_timeout_ms: AtomicI64,
}

impl<'w, 'd> PrimaryNode<'w, 'd> {
    /// `new PrimaryNode(writer, id, primaryGen, forcePrimaryVersion, ...)`:
    /// stamps the writer's commit data with this primary's generation, makes
    /// merged segments pre-copy through `hooks`, and takes the first NRT
    /// point. `dir` is the directory `writer` writes. A
    /// `force_primary_version` of `-1` forces nothing.
    ///
    /// # Errors
    /// The first NRT point could not be taken.
    pub fn new(
        writer: &'w Mutex<IndexWriter<'d>>,
        dir: Arc<dyn Directory>,
        id: i32,
        primary_gen: i64,
        force_primary_version: i64,
        printer: Option<Printer>,
        hooks: Arc<dyn PrimaryHooks>,
    ) -> Result<Self> {
        let core = Arc::new(NodeCore::new(id, true, dir, printer));
        core.message("top: now init primary");
        let shared = Arc::new(PrimaryShared {
            core,
            hooks,
            finished_merged_files: Mutex::new(BTreeSet::new()),
            state: Mutex::new(PrimaryState {
                cur: None,
                copy_state: None,
            }),
            copying_count: AtomicI32::new(0),
        });
        {
            let mut w = lock_writer(writer);
            w.set_merged_segment_warmer(Some(Arc::new(PreCopyMergedSegmentWarmer {
                primary: Arc::clone(&shared),
            })));
            shared
                .core
                .message(&format!("commitData: {:?}", w.live_commit_data()));
            let mut commit_data: Vec<(String, String)> = w.live_commit_data().to_vec();
            set_entry(&mut commit_data, PRIMARY_GEN_KEY, primary_gen.to_string());
            if commit_data.iter().any(|(k, _)| k == VERSION_KEY) {
                shared.core.message(&format!(
                    "keep current commitData version={}",
                    commit_data
                        .iter()
                        .find(|(k, _)| k == VERSION_KEY)
                        .map_or("", |(_, v)| v.as_str())
                ));
            } else {
                commit_data.push((VERSION_KEY.to_string(), "0".to_string()));
                shared.core.message("add initial commitData version=0");
            }
            w.set_live_commit_data(commit_data);
            if force_primary_version != -1 {
                shared.core.message(&format!(
                    "now forcePrimaryVersion to version={force_primary_version}"
                ));
                w.advance_segment_infos_version(force_primary_version);
            }
        }
        let node = Self {
            shared,
            writer,
            primary_gen,
            remote_close_timeout_ms: AtomicI64::new(-1),
        };
        node.set_current_infos(BTreeSet::new())?;
        let version = lock(&node.shared.state)
            .cur
            .as_ref()
            .map_or(-1, |(infos, _)| infos.version);
        node.shared
            .core
            .message(&format!("init: infos version={version}"));
        Ok(node)
    }

    /// The node's shared half: log, state, directory.
    pub fn core(&self) -> &Arc<NodeCore> {
        &self.shared.core
    }

    /// `getPrimaryGen()`.
    pub fn primary_gen(&self) -> i64 {
        self.primary_gen
    }

    /// `flushAndRefresh()`: flushes the writer and, when that made a new
    /// point, publishes it. `true` when there is a new point.
    ///
    /// # Errors
    /// Flushing or reading the files of the point failed.
    pub fn flush_and_refresh(&self) -> Result<bool> {
        self.shared.core.message("top: now flushAndRefresh");
        let completed: BTreeSet<String> = lock(&self.shared.finished_merged_files).clone();
        let result = self.set_current_infos(completed.clone())?;
        let version = lock(&self.shared.state)
            .cur
            .as_ref()
            .map_or(-1, |(infos, _)| infos.version);
        if result {
            self.shared
                .core
                .message(&format!("top: opened NRT reader version={version}"));
            let mut finished = lock(&self.shared.finished_merged_files);
            for f in &completed {
                finished.remove(f);
            }
            self.shared.core.message(&format!(
                "flushAndRefresh: version={version} completedMergeFiles={completed:?} finishedMergedFiles={finished:?}"
            ));
        } else {
            self.shared.core.message(&format!(
                "top: no changes in flushAndRefresh; still version={version}"
            ));
        }
        Ok(result)
    }

    /// `getCopyStateVersion()`.
    pub fn copy_state_version(&self) -> i64 {
        lock(&self.shared.state)
            .copy_state
            .as_ref()
            .map_or(-1, |c| c.version)
    }

    /// `getLastCommitVersion()`.
    ///
    /// # Errors
    /// The writer's commit data has no version.
    pub fn last_commit_version(&self) -> Result<i64> {
        let w = lock_writer(self.writer);
        let value = w
            .live_commit_data()
            .iter()
            .find(|(k, _)| k == VERSION_KEY)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| Error::IllegalState("missing VERSION_KEY".to_string()))?;
        value
            .parse()
            .map_err(|e| Error::IllegalState(format!("{VERSION_KEY}={value}: {e}")))
    }

    /// `getRemoteCloseTimeoutMs()`.
    pub fn remote_close_timeout_ms(&self) -> i64 {
        self.remote_close_timeout_ms.load(Ordering::Relaxed)
    }

    /// `setRemoteCloseTimeoutMs(ms)`: how long [`Self::close`] waits for
    /// replicas to finish copying (`-1` forever, `0` not at all).
    ///
    /// # Errors
    /// `ms` is below `-1`.
    pub fn set_remote_close_timeout_ms(&self, ms: i64) -> Result<()> {
        if ms < -1 {
            return Err(Error::IllegalArgument(format!(
                "bad timeout + + remoteCloseTimeoutMs{ms}"
            )));
        }
        self.remote_close_timeout_ms.store(ms, Ordering::Relaxed);
        Ok(())
    }

    /// `commit()`: stamps the commit data with this primary's generation and
    /// the current point's version, then commits the writer.
    ///
    /// # Errors
    /// Committing failed.
    pub fn commit(&self) -> Result<()> {
        let version = self.copy_state_version();
        let commit_data = vec![
            (PRIMARY_GEN_KEY.to_string(), self.primary_gen.to_string()),
            (VERSION_KEY.to_string(), version.to_string()),
        ];
        self.shared
            .core
            .message(&format!("top: commit commitData={commit_data:?}"));
        let mut w = lock_writer(self.writer);
        w.set_live_commit_data(commit_data);
        w.commit().map_err(index_error)?;
        Ok(())
    }

    /// The node's side of the copy protocol, which replicas copy from.
    pub fn source(&self) -> Arc<PrimaryShared> {
        Arc::clone(&self.shared)
    }

    /// `getCopyState()`.
    ///
    /// # Errors
    /// The node is closed.
    pub fn copy_state(&self) -> Result<CopyState> {
        self.shared.copy_state()
    }

    /// `releaseCopyState(copyState)`.
    pub fn release_copy_state(&self, copy_state: CopyState) {
        self.shared.release_copy_state(copy_state);
    }

    /// `isClosed()`.
    pub fn is_closed(&self) -> bool {
        self.shared.closed(false)
    }

    /// Opens a reader over the current point, as the primary's searcher
    /// manager would hand out.
    ///
    /// # Errors
    /// Opening the segments failed.
    pub fn open_reader(&self) -> Result<DirectoryReader> {
        let infos = lock(&self.shared.state)
            .cur
            .as_ref()
            .map(|(infos, _)| infos.clone())
            .ok_or_else(|| Error::IllegalState("no current infos".to_string()))?;
        Ok(DirectoryReader::open_at(
            self.shared.core.dir.as_ref(),
            infos,
        )?)
    }

    /// `getCurrentSearchingVersion()`.
    pub fn current_searching_version(&self) -> i64 {
        lock(&self.shared.state)
            .cur
            .as_ref()
            .map_or(-1, |(infos, _)| infos.version)
    }

    /// `setCurrentInfos(completedMergeFiles)`: takes the writer's NRT
    /// snapshot and, when its version is new, makes it the current point.
    fn set_current_infos(&self, completed_merge_files: BTreeSet<String>) -> Result<bool> {
        let snapshot = NrtSource::nrt_snapshot(self.writer, true, false).map_err(index_error)?;
        let infos = snapshot.segment_infos;
        let mut st = lock(&self.shared.state);
        // Java skips an equal version only: it reads the newest reader under
        // its monitor. Here the snapshot is taken before the state lock (the
        // merged-segment warmer may wait on replicas that read the state), so
        // a concurrent refresh may have installed a newer point already;
        // versions only grow, so `<=` is Java's `==` without the race.
        if let Some((cur, _)) = &st.cur {
            if infos.version <= cur.version {
                self.shared.core.message(&format!(
                    "top: skip switch to infos: version={} is unchanged",
                    infos.version
                ));
                return Ok(false);
            }
        }
        self.shared.core.message(&format!(
            "top: switch to infos version={} segments={}",
            infos.version,
            infos.segments.len()
        ));
        let infos_bytes = segment_infos::to_bytes(&infos);
        let mut files_meta_data = BTreeMap::new();
        for file_name in files_of_infos(self.shared.core.dir.as_ref(), &infos, false)
            .map_err(|e| Error::IllegalState(e.to_string()))?
        {
            let meta = self
                .shared
                .core
                .read_local_file_metadata(&file_name)?
                .ok_or_else(|| {
                    Error::IllegalState(format!("file \"{file_name}\" is missing metadata"))
                })?;
            files_meta_data.insert(file_name, meta);
        }
        let files = Arc::new(files_meta_data);
        // Only now, with every file's metadata read, do `cur` and the copy
        // state move together: an error above leaves both on the old point.
        // The old point is released when its pin drops, below.
        let old = st.cur.replace((infos.clone(), Arc::clone(&snapshot.hold)));
        self.shared
            .core
            .set_last_file_meta_data(Some(Arc::clone(&files)));
        self.shared.core.message(&format!(
            "top: set copyState primaryGen={} version={} files={:?}",
            self.primary_gen,
            infos.version,
            files.keys().collect::<Vec<_>>()
        ));
        st.copy_state = Some(CopyState {
            files,
            version: infos.version,
            gen: infos.generation,
            infos_bytes: Arc::new(infos_bytes),
            completed_merge_files: Arc::new(completed_merge_files),
            primary_gen: self.primary_gen,
            infos: Arc::new(infos),
            hold: Some(snapshot.hold),
        });
        drop(old);
        Ok(true)
    }

    fn wait_for_all_remotes_to_close(&self) {
        let timeout = self.remote_close_timeout_ms();
        if timeout == 0 {
            return;
        }
        let started = Instant::now();
        while timeout < 0 || started.elapsed() < Duration::from_millis(timeout as u64) {
            let count = self.shared.copying_count.load(Ordering::SeqCst);
            if count == 0 {
                return;
            }
            self.shared.core.message(&format!("pendingCopies: {count}"));
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// `close()`: waits for the replicas to finish copying (up to the remote
    /// close timeout), drops the current point and rolls the writer back.
    /// The merged-segment warmer is removed with it: Java's rollback closes
    /// the writer, this port's leaves it usable, and later merges must not
    /// push to replicas through a closed primary.
    ///
    /// # Errors
    /// None today; the `Result` mirrors Java's `throws IOException`.
    pub fn close(&self) -> Result<()> {
        self.shared.core.set_state("closing");
        self.shared.core.message("top: close primary");
        self.wait_for_all_remotes_to_close();
        {
            let mut st = lock(&self.shared.state);
            st.cur = None;
            st.copy_state = None;
        }
        {
            let mut w = lock_writer(self.writer);
            w.rollback();
            w.set_merged_segment_warmer(None);
        }
        self.shared.core.set_state("closed");
        Ok(())
    }

    /// `bytesToString` of the files in the current point, for logs.
    pub fn point_size(&self) -> String {
        bytes_to_string(
            lock(&self.shared.state)
                .copy_state
                .as_ref()
                .map_or(0, |c| c.files.values().map(|m| m.length).sum()),
        )
    }
}

fn lock_writer<'w, 'd>(writer: &'w Mutex<IndexWriter<'d>>) -> MutexGuard<'w, IndexWriter<'d>> {
    writer.lock().unwrap_or_else(|p| p.into_inner())
}

fn index_error(e: index_writer::Error) -> Error {
    Error::DirectoryReader(crate::directory_reader::Error::IndexWriter(e))
}

fn set_entry(data: &mut Vec<(String, String)>, key: &str, value: String) {
    match data.iter_mut().find(|(k, _)| k == key) {
        Some(entry) => entry.1 = value,
        None => data.push((key.to_string(), value)),
    }
}
