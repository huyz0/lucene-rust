//! Merge pre-copy jobs that lose: the primary changes under them, or an NRT
//! point needs the same files. The transport holds `launch` back so the test
//! decides when each job runs.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::replicator::nrt::local::{LocalPrimaryHooks, LocalReplicaHooks};
use lucene_search::replicator::nrt::{
    CopyJob, FileMetaData, OnceDone, PrimaryNode, Printer, ReplicaCore, ReplicaHooks, ReplicaNode,
};
use lucene_search::Result;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn doc(id: &str) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(id.to_string()),
        }],
    }
}

fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

struct Held {
    inner: Arc<LocalReplicaHooks>,
}

impl ReplicaHooks for Held {
    fn new_copy_job(
        &self,
        core: &Arc<ReplicaCore>,
        reason: &str,
        files: Option<Arc<BTreeMap<String, FileMetaData>>>,
        prev_files: Option<Arc<BTreeMap<String, FileMetaData>>>,
        high_priority: bool,
        once_done: Option<OnceDone>,
    ) -> Result<Arc<dyn CopyJob>> {
        self.inner
            .new_copy_job(core, reason, files, prev_files, high_priority, once_done)
    }

    fn launch(&self, _job: Arc<dyn CopyJob>) -> Result<()> {
        Ok(())
    }

    fn send_new_replica(&self) -> Result<()> {
        self.inner.send_new_replica()
    }
}

struct Rig {
    _tmps: Vec<TempDir>,
    log: Arc<Mutex<Vec<String>>>,
    inner: Arc<LocalReplicaHooks>,
    replica: Arc<ReplicaNode>,
    primary: PrimaryNode<'static, 'static>,
    files: Arc<BTreeMap<String, FileMetaData>>,
    version: i64,
}

/// A primary of generation `primary_gen` with one document in its point,
/// and a replica started at `replica_gen` that has nothing yet.
fn rig(primary_gen: i64, replica_gen: i64) -> Rig {
    let ptmp = TempDir::new("replicator-mergeedge-primary");
    let pdir: &'static FsDirectory = Box::leak(Box::new(FsDirectory::open(ptmp.path())));
    let fields = vec![FieldInfo {
        index_options: IndexOptions::Docs,
        omit_norms: true,
        ..FieldInfo::new("id", 0)
    }];
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(pdir, fields, "Lucene104", version).unwrap();
    w.set_postings_field(Some("id")).unwrap();
    let writer: &'static Mutex<IndexWriter<'static>> = Box::leak(Box::new(Mutex::new(w)));
    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = Arc::clone(&log);
    let printer: Printer = Arc::new(move |s: &str| sink.lock().unwrap().push(s.to_string()));
    let primary = PrimaryNode::new(
        writer,
        Arc::new(FsDirectory::open(ptmp.path())),
        0,
        primary_gen,
        -1,
        Some(Arc::clone(&printer)),
        LocalPrimaryHooks::new(primary_gen),
    )
    .unwrap();
    writer.lock().unwrap().add_document(doc("a")).unwrap();
    assert!(primary.flush_and_refresh().unwrap());
    let point = primary.copy_state().unwrap();
    let files = Arc::clone(&point.files);
    let version = point.version;
    primary.release_copy_state(point);

    let rtmp = TempDir::new("replicator-mergeedge-replica");
    let inner = LocalReplicaHooks::new(primary.source(), None);
    let replica = ReplicaNode::new(
        1,
        Arc::new(FsDirectory::open(rtmp.path())),
        Some(printer),
        Arc::new(Held {
            inner: Arc::clone(&inner),
        }),
    )
    .unwrap();
    replica.start(replica_gen).unwrap();
    Rig {
        _tmps: vec![ptmp, rtmp],
        log,
        inner,
        replica,
        primary,
        files,
        version,
    }
}

impl Rig {
    fn logged(&self, needle: &str) -> bool {
        self.log.lock().unwrap().iter().any(|l| l.contains(needle))
    }

    fn close(self) {
        self.replica.close().unwrap();
        self.inner.close();
        self.primary.close().unwrap();
    }
}

#[test]
fn a_merge_copy_is_dropped_when_the_primary_changes_under_it() {
    let rig = rig(2, 1);
    let first = Arc::new(AtomicBool::new(false));
    let second = Arc::new(AtomicBool::new(false));
    let job1 = rig
        .replica
        .launch_pre_copy_merge(Arc::clone(&first), 2, (*rig.files).clone())
        .unwrap();
    let job2 = rig
        .replica
        .launch_pre_copy_merge(Arc::clone(&second), 3, (*rig.files).clone())
        .unwrap();
    assert!(rig.logged("now change lastPrimaryGen from 1 to 2"));
    assert!(rig.logged("now change lastPrimaryGen from 2 to 3"));

    // The first finishes after the group moved on to generation 3.
    rig.inner.launch(Arc::clone(&job1)).unwrap();
    wait_for("the first merge copy", || first.load(Ordering::SeqCst));
    assert!(job1.failed());
    assert!(rig.logged("primary has changed; cancelling job"));

    // The second is current: it finishes and its files wait for a point.
    rig.inner.launch(job2).unwrap();
    wait_for("the second merge copy", || second.load(Ordering::SeqCst));
    assert!(rig.logged("add file"));
    rig.close();
}

#[test]
fn an_nrt_point_that_needs_the_same_files_cancels_the_merge_copy() {
    let rig = rig(2, 2);
    let finished = Arc::new(AtomicBool::new(false));
    let merge = rig
        .replica
        .launch_pre_copy_merge(Arc::clone(&finished), 2, (*rig.files).clone())
        .unwrap();
    assert!(rig.logged("keep current lastPrimaryGen=2"));
    let point = rig
        .replica
        .new_nrt_point(2, rig.version)
        .unwrap()
        .expect("a job");
    assert!(merge.failed());
    assert!(rig.logged("now cancel merge copy job"));

    rig.inner.launch(merge).unwrap();
    wait_for("the cancelled merge copy", || {
        finished.load(Ordering::SeqCst)
    });
    assert!(rig.logged("merge copy finished with failure"));
    rig.inner.launch(point).unwrap();
    wait_for("the point", || {
        !rig.replica.is_copying() && rig.replica.current_searching_version().unwrap() == rig.version
    });
    assert_eq!(rig.replica.acquire().unwrap().num_docs(), 1);
    rig.close();
}
