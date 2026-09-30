// Test code: see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use std::sync::Mutex;

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{IndexWriter, DISABLE_AUTO_FLUSH_MB};
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn writer(dir: &FsDirectory) -> IndexWriter<'_> {
    let fields = vec![FieldInfo {
        index_options: IndexOptions::Docs,
        omit_norms: true,
        ..FieldInfo::new("id", 0)
    }];
    let mut w = IndexWriter::open(
        dir,
        fields,
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    w.set_postings_field(Some("id")).unwrap();
    w.set_max_buffered_docs(1000).unwrap();
    w.set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB).unwrap();
    w
}

fn doc(id: &str) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(id.to_string()),
        }],
    }
}

/// Every live document's id, in doc-id order.
fn ids(reader: &DirectoryReader) -> Vec<String> {
    let mut out = Vec::new();
    for seg in reader.segment_readers() {
        for d in 0..seg.max_doc {
            if seg.live_docs().is_some_and(|l| !l.get(d as usize)) {
                continue;
            }
            let stored = seg.stored_document(d).unwrap().expect("stored fields");
            if let FieldValue::String(s) = &stored.fields[0].value {
                out.push(s.clone());
            }
        }
    }
    out
}

/// `DirectoryReader.open(writer)` sees what is indexed and not committed;
/// `openIfChanged(reader, writer)` is `None` until the writer changes, then
/// shows the change -- deletes included -- while the directory's own
/// commit still shows none of it.
#[test]
fn an_nrt_reader_sees_uncommitted_documents_and_deletes() {
    let tmp = TempDir::new("nrt-reader");
    let dir = FsDirectory::open(&tmp);
    let w = Mutex::new(writer(&dir));
    for i in 0..5 {
        w.lock()
            .unwrap()
            .add_document(doc(&format!("c{i}")))
            .unwrap();
    }
    w.lock().unwrap().commit().unwrap();
    for i in 0..3 {
        w.lock()
            .unwrap()
            .add_document(doc(&format!("u{i}")))
            .unwrap();
    }

    let r1 = DirectoryReader::open_from_writer(&w).unwrap();
    assert!(r1.is_nrt());
    assert_eq!(r1.num_docs(), 8);
    assert_eq!(r1.doc_freq("id", b"u1").unwrap(), 1);
    assert!(r1.is_current_nrt(&w).unwrap());
    assert!(r1.open_if_changed_nrt(&w).unwrap().is_none());
    assert_eq!(DirectoryReader::open(&dir).unwrap().num_docs(), 5);

    w.lock()
        .unwrap()
        .delete_documents_by_term(&[Term::new("id", "c2"), Term::new("id", "u0")])
        .unwrap();
    assert!(!r1.is_current_nrt(&w).unwrap());
    let r2 = r1.open_if_changed_nrt(&w).unwrap().expect("changed");
    assert_eq!(r2.num_docs(), 6);
    assert!(!ids(&r2).contains(&"c2".to_string()));
    assert!(!ids(&r2).contains(&"u0".to_string()));
    // The old reader is a point-in-time view and keeps its documents.
    assert_eq!(r1.num_docs(), 8);
    assert!(r2.segment_infos.version > r1.segment_infos.version);

    w.lock().unwrap().add_document(doc("n")).unwrap();
    let r3 = DirectoryReader::open_nrt(&w, false, true).unwrap();
    assert_eq!(r3.num_docs(), 7);
    assert!(r3.nrt_hold().is_some());
    let plain = DirectoryReader::open(&dir).unwrap();
    assert!(!plain.is_nrt() && plain.nrt_hold().is_none());
}

/// A near-real-time reader of a concurrent writer keeps reading its segments
/// after a merge replaced them and a commit superseded them, and lets the
/// writer reclaim them once dropped.
#[test]
fn an_nrt_reader_of_a_concurrent_writer_survives_merges() {
    let tmp = TempDir::new("nrt-concurrent-reader");
    let dir = FsDirectory::open(&tmp);
    let mut single = writer(&dir);
    single.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 10,
        segments_per_tier: 2,
        floor_segment_size: 1 << 30,
        ..MergePolicyConfig::default()
    }));
    let w = ConcurrentIndexWriter::new(single, 2).unwrap();
    for i in 0..4 {
        w.add_document(doc(&format!("a{i}"))).unwrap();
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open_from_writer(&w).unwrap();
    let before = ids(&reader);
    assert_eq!(before.len(), 4);
    let segments = reader.segment_readers().len();
    assert!(segments >= 2);

    assert!(w.maybe_merge().unwrap() > 0);
    w.commit().unwrap();
    // Still readable, the same documents.
    assert_eq!(ids(&reader), before);
    let pinned = reader.nrt_hold().unwrap().files().to_vec();
    assert!(pinned.iter().all(|f| tmp.path().join(f).exists()));

    let fresh = reader.open_if_changed_nrt(&w).unwrap().expect("merged");
    assert!(fresh.segment_readers().len() < segments);
    let mut a = ids(&fresh);
    a.sort();
    let mut b = before.clone();
    b.sort();
    assert_eq!(a, b);
    drop(reader);
    drop(fresh);
    let _ = DirectoryReader::open_from_writer(&w).unwrap();
    // The superseded segments' files are gone once nothing pins them.
    let _ = w.commit().unwrap();
    let remaining = pinned
        .iter()
        .filter(|f| tmp.path().join(f).exists())
        .count();
    assert!(
        remaining < pinned.len(),
        "released files were not reclaimed"
    );
}
