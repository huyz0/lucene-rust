//! `replicator.nrt` end to end over the in-process transport: a primary
//! publishes NRT points of a real `IndexWriter`, replicas copy and open them,
//! keep exactly the files the points need, commit, and restart.
//!
//! The wire format of a point is Lucene's own (`FileMetaData` from the
//! files' headers and footers, `segments_N` bytes from `SegmentInfos.write`);
//! `replicator_nrt_fixtures.rs` checks those against Lucene's.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::replicator::nrt::local::{LocalPrimaryHooks, LocalReplicaHooks};
use lucene_search::replicator::nrt::{PrimaryNode, ReplicaNode, PRIMARY_GEN_KEY, VERSION_KEY};
use lucene_store::{Directory, FsDirectory};
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

fn sorted_files(dir: &dyn Directory) -> BTreeSet<String> {
    dir.list_all()
        .unwrap()
        .into_iter()
        .filter(|f| f != "write.lock")
        .collect()
}

struct Group {
    _primary_dir: TempDir,
    _replica_dirs: Vec<TempDir>,
}

#[test]
fn replicas_follow_the_primary_point_by_point() {
    let primary_tmp = TempDir::new("replicator-primary");
    let primary_dir: &'static FsDirectory =
        Box::leak(Box::new(FsDirectory::open(primary_tmp.path())));
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
    let mut w = IndexWriter::open(primary_dir, fields, "Lucene104", version).unwrap();
    w.set_postings_field(Some("id")).unwrap();
    let writer = Mutex::new(w);

    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = Arc::clone(&log);
    let printer: lucene_search::replicator::nrt::Printer =
        Arc::new(move |s: &str| sink.lock().unwrap().push(s.to_string()));

    let group_hooks = LocalPrimaryHooks::new(1);
    let primary = PrimaryNode::new(
        &writer,
        Arc::new(FsDirectory::open(primary_tmp.path())),
        0,
        1,
        -1,
        Some(Arc::clone(&printer)),
        group_hooks.clone(),
    )
    .unwrap();
    assert_eq!(primary.core().name(), "P0");
    assert_eq!(primary.last_commit_version().unwrap(), 0);

    let replica_tmps: Vec<TempDir> = (0..2)
        .map(|i| TempDir::new(&format!("replicator-replica-{i}")))
        .collect();
    let mut replicas = Vec::new();
    let mut transports = Vec::new();
    for (i, tmp) in replica_tmps.iter().enumerate() {
        let transport = LocalReplicaHooks::new(primary.source(), None);
        let replica = ReplicaNode::new(
            i as i32 + 1,
            Arc::new(FsDirectory::open(tmp.path())),
            Some(Arc::clone(&printer)),
            transport.clone(),
        )
        .unwrap();
        replica.start(1).unwrap();
        group_hooks.add_replica(&replica);
        assert_eq!(replica.state(), "idle");
        replicas.push(replica);
        transports.push(transport);
    }

    // Point 1: three documents.
    {
        let mut w = writer.lock().unwrap();
        for id in ["a", "b", "c"] {
            w.add_document(doc(id)).unwrap();
        }
    }
    assert!(primary.flush_and_refresh().unwrap());
    let v1 = primary.copy_state_version();
    assert!(!primary.flush_and_refresh().unwrap(), "nothing changed");
    group_hooks.push_nrt_point(v1).unwrap();
    for r in &replicas {
        wait_for("point 1", || {
            !r.is_copying() && r.current_searching_version().unwrap() == v1
        });
        let reader = r.acquire().unwrap();
        assert_eq!(reader.num_docs(), 3);
        assert_eq!(reader.doc_freq("id", b"b").unwrap(), 1);
    }

    // Point 2: a delete and an add; the replica keeps point 1's segment and
    // takes the new files only.
    {
        let mut w = writer.lock().unwrap();
        w.add_document(doc("d")).unwrap();
        w.delete_documents_by_term(&[Term::new("id", "a")]).unwrap();
    }
    assert!(primary.flush_and_refresh().unwrap());
    let v2 = primary.copy_state_version();
    assert!(v2 > v1);
    group_hooks.push_nrt_point(v2).unwrap();
    for r in &replicas {
        wait_for("point 2", || {
            !r.is_copying() && r.current_searching_version().unwrap() == v2
        });
        let reader = r.acquire().unwrap();
        assert_eq!(reader.num_docs(), 3, "d added, a deleted");
        // docFreq counts deleted documents until a merge drops them.
        assert_eq!(reader.doc_freq("id", b"a").unwrap(), 1);
        assert_eq!(reader.doc_freq("id", b"d").unwrap(), 1);
    }

    // The same point again is skipped; an older one too.
    for r in &replicas {
        assert!(r.new_nrt_point(1, v2).unwrap().is_none());
        assert!(r.new_nrt_point(1, v1).unwrap().is_none());
    }

    // A commit on the replica writes a segments_N with the primary's
    // generation and the point's version; only files of the point and of the
    // commit remain.
    let replica = &replicas[0];
    replica.commit().unwrap();
    let on_disk = lucene_index::segment_infos::read_latest(&**replica.core().dir()).unwrap();
    let data: std::collections::HashMap<_, _> = on_disk.user_data.iter().cloned().collect();
    assert_eq!(data[PRIMARY_GEN_KEY], "1");
    assert_eq!(data[VERSION_KEY], v2.to_string());
    let named: BTreeSet<String> =
        lucene_index::index_file_deleter::files_of_infos(&**replica.core().dir(), &on_disk, true)
            .unwrap()
            .into_iter()
            .collect();
    let on_dir = sorted_files(&**replica.core().dir());
    assert_eq!(
        on_dir, named,
        "the replica holds exactly its commit's files"
    );

    // Restart: the replica reopens its commit, same primary.
    replica.close().unwrap();
    assert!(replica.is_closed());
    transports[0].close();
    let transport = LocalReplicaHooks::new(primary.source(), None);
    let reopened = ReplicaNode::new(
        9,
        Arc::new(FsDirectory::open(replica_tmps[0].path())),
        Some(Arc::clone(&printer)),
        transport.clone(),
    )
    .unwrap();
    reopened.start(1).unwrap();
    assert_eq!(reopened.current_searching_version().unwrap(), v2);
    assert_eq!(reopened.acquire().unwrap().num_docs(), 3);

    // Restart against a new primary generation: the stale commit is replaced
    // by a sync from the primary.
    reopened.close().unwrap();
    transport.close();
    let transport = LocalReplicaHooks::new(primary.source(), None);
    let synced = ReplicaNode::new(
        10,
        Arc::new(FsDirectory::open(replica_tmps[0].path())),
        Some(Arc::clone(&printer)),
        transport.clone(),
    )
    .unwrap();
    synced.start(2).unwrap();
    assert_eq!(synced.current_searching_version().unwrap(), v2);
    assert_eq!(synced.acquire().unwrap().num_docs(), 3);
    let committed = lucene_index::segment_infos::read_latest(&**synced.core().dir()).unwrap();
    let data: std::collections::HashMap<_, _> = committed.user_data.iter().cloned().collect();
    assert_eq!(
        data[PRIMARY_GEN_KEY], "1",
        "the replica's own commit at start, stamped with the copied point's primaryGen"
    );
    synced.close().unwrap();
    transport.close();

    for r in &replicas[1..] {
        r.close().unwrap();
    }
    for t in &transports[1..] {
        t.close();
    }
    primary.set_remote_close_timeout_ms(5000).unwrap();
    primary.commit().unwrap();
    assert_eq!(primary.last_commit_version().unwrap(), v2);
    primary.close().unwrap();
    assert!(primary.is_closed());

    let lines = log.lock().unwrap();
    assert!(lines
        .iter()
        .any(|l| l.contains("top: set copyState primaryGen=1")));
    assert!(lines.iter().any(|l| l.contains("top: newNRTPoint")));
    assert!(lines
        .iter()
        .any(|l| l.contains("top: commit wrote segments file")));
    let _ = Group {
        _primary_dir: primary_tmp,
        _replica_dirs: replica_tmps,
    };
}
