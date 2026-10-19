//! `replicator.nrt` when a newer point arrives while the previous one is
//! still being copied: the new job takes over the files already copied, and
//! a primary that cannot be reached leaves the replica where it was.
//!
//! The transport is the local one with `launch` held back, so the test
//! decides how far each job gets before the next point shows up.

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
    CopyJob, FileMetaData, NodeCommunicationError, OnceDone, PrimaryNode, Printer, ReplicaCore,
    ReplicaHooks, ReplicaNode,
};
use lucene_search::{Error, Result};
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

/// A document whose stored field is too big for ten copy buffers, and does
/// not compress.
fn big_doc(id: &str) -> Document {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let blob: String = (0..2_000_000)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            char::from(b'a' + (state % 26) as u8)
        })
        .collect();
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(blob),
            },
        ],
    }
}

fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// The local transport, with jobs held until the test launches them and a
/// switch that makes the primary unreachable.
struct Held {
    inner: Arc<LocalReplicaHooks>,
    held: Mutex<Vec<Arc<dyn CopyJob>>>,
    unreachable: AtomicBool,
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
        if self.unreachable.load(Ordering::SeqCst) {
            return Err(Error::NodeCommunication(NodeCommunicationError {
                when: "connect".to_string(),
                cause: "refused".to_string(),
            }));
        }
        self.inner
            .new_copy_job(core, reason, files, prev_files, high_priority, once_done)
    }

    fn launch(&self, job: Arc<dyn CopyJob>) -> Result<()> {
        self.held.lock().unwrap().push(job);
        Ok(())
    }

    fn send_new_replica(&self) -> Result<()> {
        self.inner.send_new_replica()
    }
}

#[test]
fn a_newer_point_takes_over_the_files_already_copied() {
    let ptmp = TempDir::new("replicator-transfer-primary");
    let pdir: &'static FsDirectory = Box::leak(Box::new(FsDirectory::open(ptmp.path())));
    let fields = vec![
        FieldInfo {
            index_options: IndexOptions::Docs,
            omit_norms: true,
            ..FieldInfo::new("id", 0)
        },
        FieldInfo::new("blob", 1),
    ];
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
    let group = LocalPrimaryHooks::new(1);
    let primary = PrimaryNode::new(
        writer,
        Arc::new(FsDirectory::open(ptmp.path())),
        0,
        1,
        -1,
        Some(Arc::clone(&printer)),
        group,
    )
    .unwrap();

    let rtmp = TempDir::new("replicator-transfer-replica");
    let inner = LocalReplicaHooks::new(primary.source(), None);
    let hooks = Arc::new(Held {
        inner: Arc::clone(&inner),
        held: Mutex::new(Vec::new()),
        unreachable: AtomicBool::new(false),
    });
    let replica = ReplicaNode::new(
        1,
        Arc::new(FsDirectory::open(rtmp.path())),
        Some(Arc::clone(&printer)),
        hooks.clone(),
    )
    .unwrap();
    replica.start(1).unwrap();

    let publish_docs = |docs: Vec<Document>| {
        let mut w = writer.lock().unwrap();
        for d in docs {
            w.add_document(d).unwrap();
        }
        drop(w);
        assert!(primary.flush_and_refresh().unwrap());
        primary.copy_state_version()
    };
    let publish = |ids: &[&str]| publish_docs(ids.iter().map(|id| doc(id)).collect());

    // The first job copies a file and stops in the middle of the big one.
    let v1 = publish_docs(vec![big_doc("a"), doc("b")]);
    let job1 = replica.new_nrt_point(1, v1).unwrap().expect("a job");
    assert!(replica.is_copying());
    while job1.core().state().current.is_none() {
        assert!(!job1.visit().unwrap(), "the big file is still to come");
    }
    assert!(job1.total_bytes_copied() > 0);

    // The second point arrives: its job takes over what job1 has, and job1
    // is cancelled.
    let v2 = publish(&["c"]);
    let job2 = replica.new_nrt_point(1, v2).unwrap().expect("a job");
    assert!(job1.failed());
    assert!(!job2.failed());
    assert!(log
        .lock()
        .unwrap()
        .iter()
        .any(|l| l.contains("xfer: carry over already-copied file")));

    // Run job2 to the end the way the transport would.
    // job1 was cancelled; run through the transport it finishes without effect.
    inner.launch(Arc::clone(&job1)).unwrap();
    inner.launch(Arc::clone(&job2)).unwrap();
    wait_for("the second point", || {
        !replica.is_copying() && replica.current_searching_version().unwrap() == v2
    });
    assert_eq!(replica.acquire().unwrap().num_docs(), 3);
    assert!(log
        .lock()
        .unwrap()
        .iter()
        .any(|l| l.contains("xfer: carry over in-progress file")));

    // A primary that cannot be reached leaves the replica as it was.
    let v3 = publish(&["d"]);
    hooks.unreachable.store(true, Ordering::SeqCst);
    assert!(replica.new_nrt_point(1, v3).unwrap().is_none());
    assert_eq!(replica.state(), "idle");
    assert_eq!(replica.current_searching_version().unwrap(), v2);
    hooks.unreachable.store(false, Ordering::SeqCst);

    // And once it can, the replica catches up.
    let job3 = replica.new_nrt_point(1, v3).unwrap().expect("a job");
    inner.launch(job3).unwrap();
    wait_for("the third point", || {
        !replica.is_copying() && replica.current_searching_version().unwrap() == v3
    });
    assert_eq!(replica.acquire().unwrap().num_docs(), 4);
    assert_eq!(hooks.held.lock().unwrap().len(), 3);

    // The pass-throughs the transports use.
    assert!(replica.start(1).is_err(), "already started");
    assert_eq!(
        replica.current_segments_file_name().unwrap(),
        "",
        "nothing committed yet"
    );
    replica.commit().unwrap();
    assert!(replica
        .current_segments_file_name()
        .unwrap()
        .starts_with("segments_"));
    assert_eq!(replica.ref_count("no-such-file"), 0);
    let point = primary.copy_state().unwrap();
    assert!(replica.files_to_copy(&point.files).unwrap().is_empty());
    primary.release_copy_state(point);
    let tmp = replica.create_temp_output("probe", "tmp").unwrap();
    let tmp_name = lucene_store::IndexOutput::name(&tmp).to_string();
    replica.start_copy_file("probe").unwrap();
    assert!(replica.start_copy_file("probe").is_err());
    replica.finish_copy_file("probe").unwrap();
    tmp.close().unwrap();
    lucene_store::Directory::delete_file(&**replica.core().dir(), &tmp_name).unwrap();
    assert!(matches!(
        ReplicaNode::communication_error("x", "y"),
        Error::NodeCommunication(_)
    ));

    replica.close().unwrap();
    inner.close();
    primary.close().unwrap();
}
