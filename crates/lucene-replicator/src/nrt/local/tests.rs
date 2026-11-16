#![cfg(test)]
use super::*;
use crate::nrt::replica::empty_infos;
use crate::nrt::replica_file_deleter::ReplicaFileDeleter;
use lucene_store::{ByteBuffersDirectory, DataOutput};
use std::sync::atomic::AtomicUsize;

fn put(dir: &ByteBuffersDirectory, name: &str, bytes: &[u8]) {
    let mut out = dir.create_output(name).unwrap();
    out.write_bytes(bytes);
    out.close().unwrap();
}

fn replica() -> Arc<ReplicaCore> {
    let dir = Arc::new(ByteBuffersDirectory::new());
    let node = Arc::new(NodeCore::new(1, false, dir.clone(), None));
    let deleter = ReplicaFileDeleter::new(Arc::clone(&node), dir);
    Arc::new(ReplicaCore::new(node, deleter))
}

fn meta(length: u64) -> FileMetaData {
    FileMetaData {
        header: Vec::new(),
        footer: Vec::new(),
        length,
        checksum: 0,
    }
}

fn point(files: &[(&str, u64)]) -> CopyState {
    CopyState {
        files: Arc::new(
            files
                .iter()
                .map(|(n, l)| (n.to_string(), meta(*l)))
                .collect(),
        ),
        version: 4,
        gen: 1,
        infos_bytes: Arc::new(Vec::new()),
        completed_merge_files: Arc::new(Default::default()),
        primary_gen: 1,
        infos: Arc::new(empty_infos()),
        hold: None,
    }
}

/// A source over an in-memory directory that counts released points.
struct Source {
    dir: ByteBuffersDirectory,
    released: AtomicUsize,
    fail_open: bool,
    fail_state: bool,
}

impl CopySource for Source {
    fn copy_state(&self) -> Result<CopyState> {
        if self.fail_state {
            return Err(io_error("primary is closed"));
        }
        Ok(point(&[("f", 10)]))
    }
    fn release_copy_state(&self, _copy_state: CopyState) {
        self.released.fetch_add(1, Ordering::SeqCst);
    }
    fn open_files(&self, requests: &[(String, u64)]) -> Result<Box<dyn CopyStream>> {
        if self.fail_open {
            return Err(io_error("connection refused"));
        }
        Ok(Box::new(FileStream::open(&self.dir, requests)?))
    }
}

fn source(fail_open: bool) -> Arc<Source> {
    let dir = ByteBuffersDirectory::new();
    put(&dir, "f", b"0123456789");
    Arc::new(Source {
        dir,
        released: AtomicUsize::new(0),
        fail_open,
        fail_state: false,
    })
}

fn one_file() -> Option<Arc<BTreeMap<String, FileMetaData>>> {
    Some(Arc::new(BTreeMap::from([("f".to_string(), meta(10))])))
}

#[test]
fn a_stream_sends_each_length_then_the_file_from_the_offset() {
    let src = source(false);
    let mut s = FileStream::open(&src.dir, &[("f".to_string(), 3)]).unwrap();
    assert_eq!(s.read_vlong().unwrap(), 10);
    let mut seven = [0u8; 7];
    s.read_bytes(&mut seven).unwrap();
    assert_eq!(&seven, b"3456789");
    assert!(s.read_bytes(&mut [0u8; 1]).is_err(), "end of stream");

    let mut s = FileStream::open(&src.dir, &[("f".to_string(), 10)]).unwrap();
    assert_eq!(s.read_vlong().unwrap(), 10);
    s.close();
    assert!(s.read_vlong().is_err(), "closed");

    assert!(FileStream::open(&src.dir, &[("f".to_string(), 11)]).is_err());
    assert!(FileStream::open(&src.dir, &[("nope".to_string(), 0)]).is_err());
    let mut v = Vec::new();
    put_vlong(&mut v, 300);
    assert_eq!(v, [0xac, 0x02]);
}

#[test]
fn a_job_that_cannot_start_reports_a_communication_error() {
    let dest = replica();
    let src = source(true);
    let job = LocalCopyJob::new(
        "t",
        src.clone(),
        Some(point(&[("f", 10)])),
        dest,
        Some(Arc::new(BTreeMap::from([("f".to_string(), meta(10))]))),
        false,
        None,
    )
    .unwrap();
    assert!(
        matches!(job.visit(), Err(Error::IllegalState(_))),
        "not started"
    );
    assert!(matches!(job.start(), Err(Error::NodeCommunication(_))));
    assert!(job.failed());
    assert!(job.run_blocking().is_err());
    assert!(job.description().contains("SimpleCopyJob"));
    assert!(job.copy_state().is_some());
    drop(job);
    assert_eq!(
        src.released.load(Ordering::SeqCst),
        1,
        "the point is given back"
    );
}

#[test]
fn a_length_the_primary_disagrees_on_fails_the_copy() {
    let dest = replica();
    let src = source(false);
    let files = Arc::new(BTreeMap::from([("f".to_string(), meta(99))]));
    let job = LocalCopyJob::new("t", src.clone(), None, dest, Some(files), false, None).unwrap();
    job.start().unwrap();
    assert!(matches!(job.start(), Err(Error::IllegalState(_))), "twice");
    let err = job.visit().unwrap_err();
    assert!(
        err.to_string().contains("meta data says length=99"),
        "{err}"
    );
    job.close();
    assert_eq!(src.released.load(Ordering::SeqCst), 0, "no point was held");
}

/// A job whose steps are scripted.
struct Scripted {
    core: CopyJobCore,
    steps: Mutex<Vec<Result<bool>>>,
}

impl CopyJob for Scripted {
    fn core(&self) -> &CopyJobCore {
        &self.core
    }
    fn start(&self) -> Result<()> {
        Ok(())
    }
    fn run_blocking(&self) -> Result<()> {
        Ok(())
    }
    fn visit(&self) -> Result<bool> {
        lock(&self.steps).pop().unwrap_or(Ok(true))
    }
    fn finish(&self) -> Result<()> {
        Ok(())
    }
    fn copy_state(&self) -> Option<&CopyState> {
        None
    }
}

fn scripted(
    dest: &Arc<ReplicaCore>,
    steps: Vec<Result<bool>>,
    done: &Arc<AtomicUsize>,
) -> Arc<dyn CopyJob> {
    let done = Arc::clone(done);
    let once_done: OnceDone = Arc::new(move |job: &dyn CopyJob| {
        done.fetch_add(1, Ordering::SeqCst);
        if job.failed() {
            Err(io_error("once done sees a failure"))
        } else {
            Ok(())
        }
    });
    Arc::new(Scripted {
        core: CopyJobCore::new("s", None, Arc::clone(dest), false, Some(once_done)).unwrap(),
        steps: Mutex::new(steps),
    })
}

fn wait_for(n: &AtomicUsize, at_least: usize) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while n.load(Ordering::SeqCst) < at_least {
        assert!(Instant::now() < deadline, "timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn the_jobs_thread_runs_steps_until_done_and_cancels_what_fails() {
    let dest = replica();
    let jobs = Jobs::start(Arc::clone(&dest.node));
    let done = Arc::new(AtomicUsize::new(0));
    // Two steps, then done.
    jobs.launch(scripted(&dest, vec![Ok(true), Ok(false), Ok(false)], &done))
        .unwrap();
    // A step that fails: the job is cancelled and its onceDone runs, whose
    // own failure is only logged.
    jobs.launch(scripted(&dest, vec![Err(io_error("boom"))], &done))
        .unwrap();
    jobs.launch(scripted(
        &dest,
        vec![Err(Error::AlreadyClosed("gone".to_string()))],
        &done,
    ))
    .unwrap();
    wait_for(&done, 3);
    let first = scripted(&dest, vec![], &done);
    let second = scripted(&dest, vec![], &done);
    jobs.cancel_conflicting_jobs(first.as_ref()).unwrap();
    jobs.cancel_conflicting_jobs(second.as_ref()).unwrap();
    jobs.close();
    assert!(matches!(jobs.launch(first), Err(Error::AlreadyClosed(_))));
    jobs.close();
}

#[test]
fn closing_the_jobs_thread_cancels_the_jobs_still_queued() {
    let dest = replica();
    let jobs = Jobs::start(Arc::clone(&dest.node));
    let done = Arc::new(AtomicUsize::new(0));
    // Never finishes by itself.
    let endless: Vec<Result<bool>> = (0..100_000).map(|_| Ok(false)).collect();
    jobs.launch(scripted(&dest, endless, &done)).unwrap();
    std::thread::sleep(Duration::from_millis(20));
    jobs.close();
    assert_eq!(
        done.load(Ordering::SeqCst),
        1,
        "onceDone ran after the cancel"
    );
}

/// A job whose step blocks until released, then never finishes.
struct Gated {
    core: CopyJobCore,
    entered: AtomicBool,
    release: AtomicBool,
}

impl CopyJob for Gated {
    fn core(&self) -> &CopyJobCore {
        &self.core
    }
    fn start(&self) -> Result<()> {
        Ok(())
    }
    fn run_blocking(&self) -> Result<()> {
        Ok(())
    }
    fn visit(&self) -> Result<bool> {
        self.entered.store(true, Ordering::SeqCst);
        while !self.release.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(false)
    }
    fn finish(&self) -> Result<()> {
        Ok(())
    }
    fn copy_state(&self) -> Option<&CopyState> {
        None
    }
}

#[test]
fn a_new_job_cancels_queued_conflicts_and_close_cancels_the_rest() {
    let dest = replica();
    let jobs = Jobs::start(Arc::clone(&dest.node));
    let gated = Arc::new(Gated {
        core: CopyJobCore::new("g", None, Arc::clone(&dest), true, None).unwrap(),
        entered: AtomicBool::new(false),
        release: AtomicBool::new(false),
    });
    jobs.launch(gated.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    while !gated.entered.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
    // The thread is busy with `gated`, so this one waits in the queue.
    let src = source(false);
    let queued: Arc<dyn CopyJob> = Arc::new(
        LocalCopyJob::new(
            "q",
            src.clone(),
            Some(point(&[("f", 10)])),
            Arc::clone(&dest),
            one_file(),
            false,
            None,
        )
        .unwrap(),
    );
    jobs.launch(Arc::clone(&queued)).unwrap();
    let newer = LocalCopyJob::new(
        "n",
        src.clone(),
        None,
        Arc::clone(&dest),
        one_file(),
        false,
        None,
    )
    .unwrap();
    jobs.cancel_conflicting_jobs(&newer).unwrap();
    assert!(queued.failed(), "the queued job copies the same file");
    assert_eq!(
        src.released.load(Ordering::SeqCst),
        1,
        "cancelling gives its point back"
    );
    // `gated` never finishes: closing the thread cancels it.
    gated.release.store(true, Ordering::SeqCst);
    jobs.close();
    assert!(gated.failed());
}

#[test]
fn a_job_that_cannot_be_built_gives_the_point_back() {
    // A directory where the replica's file should be: reading it fails
    // with something other than "not found".
    let tmp = lucene_util::test_support::TempDir::new("local-unreadable");
    std::fs::create_dir(tmp.path().join("f")).unwrap();
    let dir: Arc<dyn Directory> = Arc::new(lucene_store::FsDirectory::open(tmp.path()));
    let node = Arc::new(NodeCore::new(1, false, Arc::clone(&dir), None));
    let deleter = ReplicaFileDeleter::new(Arc::clone(&node), dir);
    let dest = Arc::new(ReplicaCore::new(node, deleter));
    let src = source(false);
    let hooks = LocalReplicaHooks::new(src.clone(), None);
    assert!(hooks
        .new_copy_job(&dest, "r", None, None, false, None)
        .is_err());
    assert_eq!(
        src.released.load(Ordering::SeqCst),
        1,
        "the point taken for the job is given back"
    );
    hooks.close();

    // A primary that cannot hand out its point: a communication error.
    let closed = Arc::new(Source {
        dir: ByteBuffersDirectory::new(),
        released: AtomicUsize::new(0),
        fail_open: false,
        fail_state: true,
    });
    let hooks = LocalReplicaHooks::new(closed, None);
    assert!(matches!(
        hooks.new_copy_job(&replica(), "r", None, None, false, None),
        Err(Error::NodeCommunication(_))
    ));
}

#[test]
fn a_job_dropped_unstarted_gives_its_point_back_and_counts_its_files() {
    let src = source(false);
    let job = LocalCopyJob::new(
        "d",
        src.clone(),
        Some(point(&[("f", 10), ("g", 20)])),
        replica(),
        None,
        false,
        None,
    )
    .unwrap();
    assert!(
        job.description().contains("files count=2"),
        "{}",
        job.description()
    );
    drop(job);
    assert_eq!(src.released.load(Ordering::SeqCst), 1);

    let announced = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&announced);
    let hooks = LocalReplicaHooks::new(
        source(false),
        Some(Box::new(move || {
            seen.fetch_add(1, Ordering::SeqCst);
        })),
    );
    hooks.send_new_replica().unwrap();
    assert_eq!(announced.load(Ordering::SeqCst), 1);
}
