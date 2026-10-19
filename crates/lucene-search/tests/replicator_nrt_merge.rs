//! `replicator.nrt` with merges: the primary's merged-segment warmer copies
//! the merged files to the replicas before the point that shows them.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::replicator::nrt::local::{LocalPrimaryHooks, LocalReplicaHooks};
use lucene_search::replicator::nrt::{PrimaryNode, Printer, ReplicaNode};
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

struct Rig {
    _tmps: Vec<TempDir>,
    writer: &'static Mutex<IndexWriter<'static>>,
    log: Arc<Mutex<Vec<String>>>,
    group: Arc<LocalPrimaryHooks>,
    replicas: Vec<Arc<ReplicaNode>>,
    transports: Vec<Arc<LocalReplicaHooks>>,
    primary: PrimaryNode<'static, 'static>,
}

fn rig(primary_gen: i64, n_replicas: usize) -> Rig {
    let ptmp = TempDir::new("replicator-merge-primary");
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
    let group = LocalPrimaryHooks::new(primary_gen);
    let primary = PrimaryNode::new(
        writer,
        Arc::new(FsDirectory::open(ptmp.path())),
        0,
        primary_gen,
        -1,
        Some(Arc::clone(&printer)),
        group.clone(),
    )
    .unwrap();
    let mut tmps = vec![ptmp];
    let mut replicas = Vec::new();
    let mut transports = Vec::new();
    for i in 0..n_replicas {
        let tmp = TempDir::new(&format!("replicator-merge-replica-{i}"));
        let transport = LocalReplicaHooks::new(primary.source(), None);
        let replica = ReplicaNode::new(
            i as i32 + 1,
            Arc::new(FsDirectory::open(tmp.path())),
            Some(Arc::clone(&printer)),
            transport.clone(),
        )
        .unwrap();
        replica.start(primary_gen).unwrap();
        group.add_replica(&replica);
        tmps.push(tmp);
        replicas.push(replica);
        transports.push(transport);
    }
    Rig {
        _tmps: tmps,
        writer,
        log,
        group,
        replicas,
        transports,
        primary,
    }
}

impl Rig {
    fn add(&self, ids: &[&str]) {
        let mut w = self.writer.lock().unwrap();
        for id in ids {
            w.add_document(doc(id)).unwrap();
        }
    }

    /// Publishes a point and waits until every replica shows it.
    fn publish(&self) -> i64 {
        assert!(self.primary.flush_and_refresh().unwrap());
        let v = self.primary.copy_state_version();
        self.group.push_nrt_point(v).unwrap();
        for r in &self.replicas {
            wait_for("a point", || {
                !r.is_copying() && r.current_searching_version().unwrap() == v
            });
        }
        v
    }

    fn logged(&self, needle: &str) -> bool {
        self.log.lock().unwrap().iter().any(|l| l.contains(needle))
    }

    fn close(self) {
        for r in &self.replicas {
            r.close().unwrap();
        }
        for t in &self.transports {
            t.close();
        }
        self.primary.close().unwrap();
    }
}

#[test]
fn a_merged_segment_is_pre_copied_before_its_point() {
    let rig = rig(1, 2);
    rig.add(&["a", "b"]);
    rig.publish();
    rig.add(&["c", "d"]);
    rig.publish();
    rig.add(&["e"]);
    let v = rig.publish();

    // The merge's warmer copies the merged segment to every replica, and
    // only then does the writer finish the merge.
    {
        let mut w = rig.writer.lock().unwrap();
        w.commit().unwrap();
        w.force_merge(1).unwrap();
    }
    assert!(rig.logged("now pre-copy warm merge"));
    assert!(rig.logged("merge pre copy finished"));
    let v2 = rig.publish();
    assert!(v2 > v);
    for r in &rig.replicas {
        assert_eq!(r.acquire().unwrap().num_docs(), 5);
        r.commit().unwrap();
        // Once the commit and the newest point are all that is left, so are
        // their files.
        let dir = &**r.core().dir();
        let infos = lucene_index::segment_infos::read_latest(dir).unwrap();
        let named: BTreeSet<String> =
            lucene_index::index_file_deleter::files_of_infos(dir, &infos, true)
                .unwrap()
                .into_iter()
                .collect();
        let on_disk: BTreeSet<String> = dir
            .list_all()
            .unwrap()
            .into_iter()
            .filter(|f| f != "write.lock")
            .collect();
        assert_eq!(on_disk, named);
    }
    rig.close();
}
