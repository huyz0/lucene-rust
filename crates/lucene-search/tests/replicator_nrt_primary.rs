//! `PrimaryNode` on its own: the commit data it stamps, a forced starting
//! version, the point it serves, and what closing does.

use std::sync::{Arc, Mutex};

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::replicator::nrt::local::LocalPrimaryHooks;
use lucene_search::replicator::nrt::{PrimaryNode, Printer, PRIMARY_GEN_KEY, VERSION_KEY};
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

fn writer(dir: &'static FsDirectory) -> Mutex<IndexWriter<'static>> {
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
    let mut w = IndexWriter::open(dir, fields, "Lucene104", version).unwrap();
    w.set_postings_field(Some("id")).unwrap();
    Mutex::new(w)
}

fn commit_data(writer: &Mutex<IndexWriter<'_>>, key: &str) -> Option<String> {
    writer
        .lock()
        .unwrap()
        .live_commit_data()
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.clone())
}

#[test]
fn a_primary_stamps_its_generation_and_serves_its_point() {
    let tmp = TempDir::new("replicator-primary-only");
    let dir: &'static FsDirectory = Box::leak(Box::new(FsDirectory::open(tmp.path())));
    let w = &writer(dir);
    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = Arc::clone(&log);
    let printer: Printer = Arc::new(move |s: &str| sink.lock().unwrap().push(s.to_string()));
    let primary = PrimaryNode::new(
        w,
        Arc::new(FsDirectory::open(tmp.path())),
        7,
        3,
        100,
        Some(Arc::clone(&printer)),
        LocalPrimaryHooks::new(3),
    )
    .unwrap();
    assert_eq!(primary.core().name(), "P7");
    assert_eq!(primary.primary_gen(), 3);
    assert_eq!(commit_data(w, PRIMARY_GEN_KEY).as_deref(), Some("3"));
    assert_eq!(commit_data(w, VERSION_KEY).as_deref(), Some("0"));
    assert!(
        primary.current_searching_version() >= 100,
        "the forced version is the floor"
    );
    assert!(log
        .lock()
        .unwrap()
        .iter()
        .any(|l| l.contains("now forcePrimaryVersion to version=100")));

    w.lock().unwrap().add_document(doc("a")).unwrap();
    assert!(primary.flush_and_refresh().unwrap());
    assert_eq!(primary.open_reader().unwrap().num_docs(), 1);
    let point = primary.copy_state().unwrap();
    assert_eq!(point.primary_gen, 3);
    assert_eq!(point.version, primary.copy_state_version());
    assert_ne!(primary.point_size(), "0 b");
    primary.release_copy_state(point);

    assert_eq!(primary.remote_close_timeout_ms(), -1);
    assert!(primary.set_remote_close_timeout_ms(-2).is_err());
    primary.set_remote_close_timeout_ms(0).unwrap();
    assert_eq!(primary.remote_close_timeout_ms(), 0);

    primary.commit().unwrap();
    let committed = primary.last_commit_version().unwrap();
    assert_eq!(committed, primary.copy_state_version());
    primary.close().unwrap();
    assert!(primary.is_closed());
    assert!(
        primary.copy_state().is_err(),
        "a closed primary serves nothing"
    );
}

#[test]
fn a_primary_over_a_committed_index_keeps_its_version() {
    let tmp = TempDir::new("replicator-primary-reopen");
    let dir: &'static FsDirectory = Box::leak(Box::new(FsDirectory::open(tmp.path())));
    let first = writer(dir);
    let primary = PrimaryNode::new(
        &first,
        Arc::new(FsDirectory::open(tmp.path())),
        1,
        1,
        -1,
        None,
        LocalPrimaryHooks::new(1),
    )
    .unwrap();
    first.lock().unwrap().add_document(doc("a")).unwrap();
    primary.flush_and_refresh().unwrap();
    primary.commit().unwrap();
    let version = commit_data(&first, VERSION_KEY).unwrap();
    primary.close().unwrap();
    drop(primary);
    drop(first);
    // The first writer held the directory's lock until it is dropped.
    let second = &writer(dir);
    let primary = PrimaryNode::new(
        second,
        Arc::new(FsDirectory::open(tmp.path())),
        2,
        2,
        -1,
        None,
        LocalPrimaryHooks::new(2),
    )
    .unwrap();
    assert_eq!(commit_data(second, PRIMARY_GEN_KEY).as_deref(), Some("2"));
    assert_eq!(commit_data(second, VERSION_KEY), Some(version));
    assert_eq!(primary.open_reader().unwrap().num_docs(), 1);
    primary.close().unwrap();
}
