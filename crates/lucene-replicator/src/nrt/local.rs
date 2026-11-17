//! An in-process transport for the NRT replication nodes: the counterpart of
//! the test framework's `SimpleCopyJob`, `SimpleReplicaNode`, `Jobs` and the
//! file-serving half of `SimplePrimaryNode`, with the socket replaced by a
//! [`FileStream`] over the primary's directory.
//!
//! [`super::PrimaryNode`] and [`super::ReplicaNode`] are abstract about how a
//! replica reaches the primary; this is the transport the tests, the
//! differential fixtures and a single-process deployment use.

use lucene_util::vint::write_vlong_u64 as put_vlong;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use lucene_store::directory::{Directory, Input};

use super::copy::{
    compare_jobs, CopyJob, CopyJobCore, CopyOneFile, CopyStream, OnceDone, ReplicaCore,
};
use super::node::NodeCore;
use super::primary::{PrimaryHooks, PrimaryShared};
use super::replica::{ReplicaHooks, ReplicaNode};
use super::types::{CopyState, FileMetaData};
use lucene_index::segment_infos::SegmentCommitInfo;
use lucene_search::{Error, Result};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn io_error(message: impl Into<String>) -> Error {
    Error::Store(lucene_store::Error::Io(std::io::Error::other(
        message.into(),
    )))
}

enum Piece {
    /// Bytes built in memory (a file's length), read from `pos`.
    Bytes(Vec<u8>, usize),
    /// A file, read from `pos`.
    File(Input, usize),
}

/// What the primary sends for a fetch: for each requested file, its length
/// as a `VLong`, then the file from the requested offset.
pub struct FileStream {
    pieces: VecDeque<Piece>,
    closed: bool,
}

impl FileStream {
    /// Opens the files named in `requests`, `(name, offset)` each.
    ///
    /// # Errors
    /// A file is missing or shorter than its offset.
    pub fn open(dir: &dyn Directory, requests: &[(String, u64)]) -> Result<Self> {
        let mut pieces = VecDeque::new();
        for (name, offset) in requests {
            let input = dir.open(name)?;
            let length = input.len() as u64;
            let start = usize::try_from(*offset)
                .ok()
                .filter(|&o| o <= input.len())
                .ok_or_else(|| {
                    io_error(format!(
                        "file {name}: offset {offset} is past length {length}"
                    ))
                })?;
            let mut header = Vec::new();
            put_vlong(&mut header, length);
            pieces.push_back(Piece::Bytes(header, 0));
            pieces.push_back(Piece::File(input, start));
        }
        Ok(Self {
            pieces,
            closed: false,
        })
    }
}

impl CopyStream for FileStream {
    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()> {
        if self.closed {
            return Err(io_error("stream is closed"));
        }
        let mut filled = 0;
        while filled < buf.len() {
            let Some(front) = self.pieces.front_mut() else {
                return Err(io_error("unexpected end of stream"));
            };
            let (data, pos): (&[u8], &mut usize) = match front {
                Piece::Bytes(b, pos) => (b.as_slice(), pos),
                Piece::File(input, pos) => (&**input, pos),
            };
            let available = data.len().saturating_sub(*pos);
            if available == 0 {
                self.pieces.pop_front();
                continue;
            }
            let n = available.min(buf.len().saturating_sub(filled));
            buf[filled..filled + n].copy_from_slice(&data[*pos..*pos + n]);
            *pos += n;
            filled += n;
        }
        Ok(())
    }

    fn close(&mut self) {
        self.closed = true;
        self.pieces.clear();
    }
}

/// What a copy job needs of the primary.
pub trait CopySource: Send + Sync {
    /// The primary's current point, pinned until released.
    ///
    /// # Errors
    /// The primary is closed.
    fn copy_state(&self) -> Result<CopyState>;
    /// Gives the point back.
    fn release_copy_state(&self, copy_state: CopyState);
    /// `CMD_FETCH_FILES`.
    ///
    /// # Errors
    /// A file is missing.
    fn open_files(&self, requests: &[(String, u64)]) -> Result<Box<dyn CopyStream>>;
}

impl CopySource for PrimaryShared {
    fn copy_state(&self) -> Result<CopyState> {
        PrimaryShared::copy_state(self)
    }
    fn release_copy_state(&self, copy_state: CopyState) {
        PrimaryShared::release_copy_state(self, copy_state);
    }
    fn open_files(&self, requests: &[(String, u64)]) -> Result<Box<dyn CopyStream>> {
        self.open_copy_stream(requests)
    }
}

struct Connection {
    stream: Option<Box<dyn CopyStream>>,
    /// `iter`: the index of the next entry of `toCopy`; `None` until started.
    next: Option<usize>,
}

/// `SimpleCopyJob`.
pub struct LocalCopyJob {
    core: CopyJobCore,
    source: Arc<dyn CopySource>,
    copy_state: Option<CopyState>,
    conn: Mutex<Connection>,
    released: AtomicBool,
}

const COPY_BUFFER_SIZE: usize = 65536;

impl LocalCopyJob {
    /// `new SimpleCopyJob(reason, c, copyState, dest, files, highPriority,
    /// onceDone)`.
    ///
    /// # Errors
    /// Reading the local files failed.
    pub fn new(
        reason: &str,
        source: Arc<dyn CopySource>,
        copy_state: Option<CopyState>,
        dest: Arc<ReplicaCore>,
        files: Option<Arc<BTreeMap<String, FileMetaData>>>,
        high_priority: bool,
        once_done: Option<OnceDone>,
    ) -> Result<Self> {
        let core = CopyJobCore::new(reason, files, Arc::clone(&dest), high_priority, once_done)?;
        dest.message(&format!("create SimpleCopyJob o{}", core.ord));
        Ok(Self {
            core,
            source,
            copy_state,
            conn: Mutex::new(Connection {
                stream: None,
                next: None,
            }),
            released: AtomicBool::new(false),
        })
    }

    /// Ends the connection and gives the primary's point back, once.
    fn close_connection(&self) {
        if let Some(stream) = lock(&self.conn).stream.as_mut() {
            stream.close();
        }
        if let Some(cs) = &self.copy_state {
            if !self.released.swap(true, Ordering::SeqCst) {
                self.source.release_copy_state(cs.clone());
            }
        }
    }

    fn files_count(&self) -> usize {
        self.core.files.as_ref().map_or_else(
            || self.copy_state.as_ref().map_or(0, |c| c.files.len()),
            |f| f.len(),
        )
    }
}

impl Drop for LocalCopyJob {
    fn drop(&mut self) {
        if let Some(cs) = &self.copy_state {
            if !self.released.swap(true, Ordering::SeqCst) {
                self.source.release_copy_state(cs.clone());
            }
        }
    }
}

impl CopyJob for LocalCopyJob {
    fn core(&self) -> &CopyJobCore {
        &self.core
    }

    fn start(&self) -> Result<()> {
        let mut conn = lock(&self.conn);
        if conn.next.is_some() {
            return Err(Error::IllegalState("already started".to_string()));
        }
        let result = (|| -> Result<()> {
            let mut st = self.core.state();
            let mut requests: Vec<(String, u64)> = Vec::new();
            if let Some(current) = &st.current {
                requests.push((current.name.clone(), current.bytes_copied()));
                st.tot_bytes = st.tot_bytes.saturating_add(current.meta_data.length);
            }
            let mut added = 0u64;
            for (name, meta) in &st.to_copy {
                added = added.saturating_add(meta.length);
                requests.push((name.clone(), 0));
            }
            st.tot_bytes = st.tot_bytes.saturating_add(added);
            let mut stream = self.source.open_files(&requests)?;
            if let Some(current) = &st.current {
                let len = stream.read_vlong()?;
                if len != current.meta_data.length {
                    return Err(Error::IllegalState(format!(
                        "file {}: meta data says length={} but c.in says {len}",
                        current.name, current.meta_data.length
                    )));
                }
            }
            self.core.dest.message(&format!(
                "SimpleCopyJob.init: done start files count={} totBytes={}",
                st.to_copy.len(),
                st.tot_bytes
            ));
            conn.stream = Some(stream);
            conn.next = Some(0);
            Ok(())
        })();
        drop(conn);
        if let Err(e) = result {
            let _ = self.cancel("exc during start", Some(&e.to_string()));
            return Err(Error::NodeCommunication(
                super::types::NodeCommunicationError {
                    when: "exc during start".to_string(),
                    cause: e.to_string(),
                },
            ));
        }
        Ok(())
    }

    fn visit(&self) -> Result<bool> {
        let mut st = self.core.state();
        if st.exc.is_some() {
            return Ok(true);
        }
        let mut conn = lock(&self.conn);
        if st.current.is_none() {
            let idx = conn
                .next
                .ok_or_else(|| Error::IllegalState("job was not started".to_string()))?;
            let Some((name, meta)) = st.to_copy.get(idx).cloned() else {
                drop(conn);
                drop(st);
                self.close_connection();
                return Ok(true);
            };
            conn.next = Some(idx.saturating_add(1));
            let stream = conn
                .stream
                .as_mut()
                .ok_or_else(|| io_error("connection is closed"))?;
            let len = stream.read_vlong()?;
            if len != meta.length {
                return Err(Error::IllegalState(format!(
                    "file {name}: meta data says length={} but c.in says {len}",
                    meta.length
                )));
            }
            st.current = Some(CopyOneFile::new(
                Arc::clone(&self.core.dest),
                &name,
                meta,
                COPY_BUFFER_SIZE,
            )?);
        }
        let stream = conn
            .stream
            .as_mut()
            .ok_or_else(|| io_error("connection is closed"))?;
        let current = st.current.as_mut().expect("set above");
        if current.visit(stream.as_mut())? {
            let done = st.current.take().expect("just visited");
            st.tot_bytes_copied = st.tot_bytes_copied.saturating_add(done.bytes_copied());
            debug_assert!(st.tot_bytes_copied <= st.tot_bytes);
            st.copied_files
                .insert(done.name.clone(), done.tmp_name.clone());
        }
        Ok(false)
    }

    fn run_blocking(&self) -> Result<()> {
        while !self.visit()? {}
        if self.failed() {
            let st = self.core.state();
            return Err(io_error(format!(
                "copy failed: {}",
                st.cancel_reason.as_deref().unwrap_or("")
            )));
        }
        Ok(())
    }

    fn finish(&self) -> Result<()> {
        let mut st = self.core.state();
        self.core.dest.message(&format!(
            "top: file copy done; took {:.1} msec to copy {} bytes; now rename {} tmp files",
            self.core.start.elapsed().as_secs_f64() * 1000.0,
            st.tot_bytes_copied,
            st.copied_files.len()
        ));
        for (file_name, tmp_name) in std::mem::take(&mut st.copied_files) {
            if self.core.dest.is_verbose_files() {
                self.core
                    .dest
                    .message(&format!("rename file {tmp_name} to {file_name}"));
            }
            self.core.dest.dir().rename(&tmp_name, &file_name)?;
        }
        Ok(())
    }

    fn copy_state(&self) -> Option<&CopyState> {
        self.copy_state.as_ref()
    }

    fn cancel(&self, reason: &str, exc: Option<&str>) -> Result<()> {
        let result = self.core.cancel(reason, exc);
        self.close_connection();
        result
    }

    fn transfer_and_cancel(&self, prev: &dyn CopyJob) -> Result<()> {
        let result = self.core.transfer_and_cancel(prev.core());
        // The previous job's connection goes whatever happened.
        prev.close();
        result
    }

    fn close(&self) {
        self.close_connection();
    }

    fn description(&self) -> String {
        let st = self.core.state();
        format!(
            "SimpleCopyJob(ord={} {} highPriority={} files count={} bytesCopied={} (of {}) filesCopied={})",
            self.core.ord,
            self.core.reason,
            self.core.high_priority,
            self.files_count(),
            st.tot_bytes_copied,
            st.tot_bytes,
            st.copied_files.len()
        )
    }
}

struct JobsQueue {
    jobs: Vec<Arc<dyn CopyJob>>,
    finish: bool,
}

struct JobsInner {
    queue: Mutex<JobsQueue>,
    wake: Condvar,
    node: Arc<NodeCore>,
}

/// `Jobs`: a thread running the launched copy jobs, high priority and oldest
/// first, one step at a time.
pub struct Jobs {
    inner: Arc<JobsInner>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Jobs {
    /// Starts the thread.
    pub fn start(node: Arc<NodeCore>) -> Self {
        let inner = Arc::new(JobsInner {
            queue: Mutex::new(JobsQueue {
                jobs: Vec::new(),
                finish: false,
            }),
            wake: Condvar::new(),
            node,
        });
        let runner = Arc::clone(&inner);
        let thread = std::thread::Builder::new()
            .name("jobs".to_string())
            .spawn(move || runner.run())
            .expect("spawn jobs thread");
        Self {
            inner,
            thread: Mutex::new(Some(thread)),
        }
    }

    /// `launch(job)`.
    ///
    /// # Errors
    /// The thread was closed.
    pub fn launch(&self, job: Arc<dyn CopyJob>) -> Result<()> {
        let mut q = lock(&self.inner.queue);
        if q.finish {
            return Err(Error::AlreadyClosed("closed".to_string()));
        }
        q.jobs.push(job);
        self.inner.wake.notify_one();
        Ok(())
    }

    /// `cancelConflictingJobs(newJob)`.
    ///
    /// # Errors
    /// Cancelling a job failed.
    pub fn cancel_conflicting_jobs(&self, new_job: &dyn CopyJob) -> Result<()> {
        let q = lock(&self.inner.queue);
        for job in &q.jobs {
            if job.conflicts(new_job) {
                self.inner.node.message(&format!(
                    "top: now cancel existing conflicting job={} due to newJob={}",
                    job.description(),
                    new_job.description()
                ));
                job.cancel("conflicts with new job", None)?;
            }
        }
        Ok(())
    }

    /// `close()`: stops the thread after the job it is running.
    pub fn close(&self) {
        {
            let mut q = lock(&self.inner.queue);
            q.finish = true;
            self.inner.wake.notify_all();
        }
        if let Some(thread) = lock(&self.thread).take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Jobs {
    fn drop(&mut self) {
        self.close();
    }
}

impl JobsInner {
    fn next_job(&self) -> Option<Arc<dyn CopyJob>> {
        let mut q = lock(&self.queue);
        loop {
            if q.finish {
                return None;
            }
            let best =
                (0..q.jobs.len()).min_by(|&a, &b| compare_jobs(q.jobs[a].core(), q.jobs[b].core()));
            match best {
                Some(i) => return Some(q.jobs.swap_remove(i)),
                None => q = self.wake.wait(q).unwrap_or_else(|p| p.into_inner()),
            }
        }
    }

    fn once_done(&self, job: &Arc<dyn CopyJob>) {
        if let Some(done) = &job.core().once_done {
            if let Err(e) = done(job.as_ref()) {
                self.node
                    .message(&format!("ignore exception calling OnceDone: {e}"));
            }
        }
    }

    fn run(&self) {
        while let Some(job) = self.next_job() {
            match job.visit() {
                Err(e) => {
                    let closed = matches!(e, Error::AlreadyClosed(_));
                    self.node.message(&format!(
                        "{} during job.visit job={}; now cancel",
                        if closed {
                            "AlreadyClosedException".to_string()
                        } else {
                            format!("exception {e}")
                        },
                        job.description()
                    ));
                    if let Err(e2) =
                        job.cancel("unexpected exception in visit", Some(&e.to_string()))
                    {
                        self.node
                            .message(&format!("ignore exception calling cancel: {e2}"));
                    }
                    self.once_done(&job);
                }
                Ok(false) => {
                    lock(&self.queue).jobs.push(job);
                }
                Ok(true) => self.once_done(&job),
            }
        }
        self.node.message("top: jobs now exit run thread");
        let rest: Vec<Arc<dyn CopyJob>> = std::mem::take(&mut lock(&self.queue).jobs);
        for job in rest {
            self.node
                .message(&format!("top: Jobs: now cancel job={}", job.description()));
            if let Err(e) = job.cancel("jobs closing", None) {
                self.node
                    .message(&format!("ignore exception calling cancel: {e}"));
            }
            self.once_done(&job);
        }
    }
}

/// `SimpleReplicaNode`'s side of [`ReplicaHooks`]: jobs copy from `source`
/// on a [`Jobs`] thread.
pub struct LocalReplicaHooks {
    source: Arc<dyn CopySource>,
    jobs: Mutex<Option<Arc<Jobs>>>,
    new_replica: Option<Box<dyn Fn() + Send + Sync>>,
}

impl LocalReplicaHooks {
    /// A transport copying from `source`; `new_replica` is called when the
    /// replica announces itself (`CMD_NEW_REPLICA`).
    pub fn new(
        source: Arc<dyn CopySource>,
        new_replica: Option<Box<dyn Fn() + Send + Sync>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            source,
            jobs: Mutex::new(None),
            new_replica,
        })
    }

    /// Stops the jobs thread; jobs still queued are cancelled.
    pub fn close(&self) {
        if let Some(jobs) = lock(&self.jobs).take() {
            jobs.close();
        }
    }

    fn jobs_for(&self, node: &Arc<NodeCore>) -> Arc<Jobs> {
        let mut jobs = lock(&self.jobs);
        Arc::clone(jobs.get_or_insert_with(|| Arc::new(Jobs::start(Arc::clone(node)))))
    }
}

impl ReplicaHooks for LocalReplicaHooks {
    fn new_copy_job(
        &self,
        core: &Arc<ReplicaCore>,
        reason: &str,
        files: Option<Arc<BTreeMap<String, FileMetaData>>>,
        _prev_files: Option<Arc<BTreeMap<String, FileMetaData>>>,
        high_priority: bool,
        once_done: Option<OnceDone>,
    ) -> Result<Arc<dyn CopyJob>> {
        let (copy_state, files) = match files {
            Some(files) => (None, files),
            None => {
                let cs = self.source.copy_state().map_err(|e| {
                    ReplicaNode::communication_error("exc while creating copy job", &e.to_string())
                })?;
                let files = Arc::clone(&cs.files);
                (Some(cs), files)
            }
        };
        // A job that fails to build never drops through `LocalCopyJob`'s
        // `Drop`, so the copy state it took is released here: left pinned, the
        // primary's copying count never returns to zero and its `close` waits
        // forever (Java ties the release to the connection).
        let taken = copy_state.clone();
        match LocalCopyJob::new(
            reason,
            Arc::clone(&self.source),
            copy_state,
            Arc::clone(core),
            Some(files),
            high_priority,
            once_done,
        ) {
            Ok(job) => Ok(Arc::new(job)),
            Err(e) => {
                if let Some(cs) = taken {
                    self.source.release_copy_state(cs);
                }
                Err(e)
            }
        }
    }

    fn launch(&self, job: Arc<dyn CopyJob>) -> Result<()> {
        let jobs = self.jobs_for(&job.core().dest.node);
        jobs.launch(job)
    }

    fn send_new_replica(&self) -> Result<()> {
        if let Some(f) = &self.new_replica {
            f();
        }
        Ok(())
    }
}

/// The primary's side of a group of in-process replicas: pushes NRT points
/// to them and pre-copies merged segments.
#[derive(Default)]
pub struct LocalPrimaryHooks {
    replicas: Mutex<Vec<Weak<ReplicaNode>>>,
    primary_gen: Mutex<i64>,
}

impl LocalPrimaryHooks {
    /// A group with no replicas, for primary generation `primary_gen`.
    pub fn new(primary_gen: i64) -> Arc<Self> {
        Arc::new(Self {
            replicas: Mutex::new(Vec::new()),
            primary_gen: Mutex::new(primary_gen),
        })
    }

    /// Adds a replica to the group.
    pub fn add_replica(&self, replica: &Arc<ReplicaNode>) {
        lock(&self.replicas).push(Arc::downgrade(replica));
    }

    fn live_replicas(&self) -> Vec<Arc<ReplicaNode>> {
        lock(&self.replicas)
            .iter()
            .filter_map(Weak::upgrade)
            .filter(|r| !r.is_closed())
            .collect()
    }

    /// `CMD_NEW_NRT_POINT` to every replica; returns the jobs started.
    ///
    /// # Errors
    /// A replica failed to start a job.
    pub fn push_nrt_point(&self, version: i64) -> Result<Vec<Arc<dyn CopyJob>>> {
        let primary_gen = *lock(&self.primary_gen);
        let mut jobs = Vec::new();
        for replica in self.live_replicas() {
            if let Some(job) = replica.new_nrt_point(primary_gen, version)? {
                jobs.push(job);
            }
        }
        Ok(jobs)
    }
}

impl PrimaryHooks for LocalPrimaryHooks {
    fn pre_copy_merged_segment_files(
        &self,
        _info: &SegmentCommitInfo,
        files: &BTreeMap<String, FileMetaData>,
    ) -> Result<()> {
        let primary_gen = *lock(&self.primary_gen);
        let mut waits = Vec::new();
        for replica in self.live_replicas() {
            let finished = Arc::new(AtomicBool::new(false));
            replica.launch_pre_copy_merge(Arc::clone(&finished), primary_gen, files.clone())?;
            waits.push(finished);
        }
        let deadline = Instant::now() + Duration::from_secs(60);
        for finished in waits {
            while !finished.load(Ordering::SeqCst) {
                if Instant::now() > deadline {
                    return Err(io_error("timed out waiting for the merge pre-copy"));
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
