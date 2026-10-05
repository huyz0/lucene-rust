// Test code: see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::buffered_updates::Term;
use crate::concurrent_writer::ConcurrentIndexWriter;
use crate::index_writer::DISABLE_AUTO_FLUSH_MB;
use crate::segment_info::LuceneVersion;
use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
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

fn live_docs(infos: &SegmentInfos, dir: &FsDirectory) -> i32 {
    infos
        .segments
        .iter()
        .map(|s| {
            let si = crate::segment_info::parse(
                &dir.open(&format!("{}.si", s.segment_name)).unwrap(),
                &s.segment_id,
            )
            .unwrap();
            si.doc_count - s.del_count
        })
        .sum()
}

/// A snapshot holds what is flushed but not committed, with buffered deletes
/// applied; `nrt_is_current` turns false on any change and true again after
/// a snapshot; versions only grow, across snapshots and commits.
#[test]
fn a_snapshot_sees_uncommitted_segments_and_deletes() {
    let tmp = TempDir::new("nrt-snapshot");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir);
    for i in 0..10 {
        w.add_document(doc(&format!("d{i}"))).unwrap();
    }
    w.commit().unwrap();
    let committed = w.segment_infos().version;

    // Unchanged since the commit: the commit's own version.
    let s0 = w.nrt_snapshot(true, false).unwrap();
    assert_eq!(s0.segment_infos.version, committed);
    assert!(w.nrt_is_current(&s0.segment_infos));

    w.add_document(doc("x")).unwrap();
    assert!(!w.nrt_is_current(&s0.segment_infos));
    w.delete_documents_by_term(&[Term::new("id", "d3")])
        .unwrap();
    let s1 = w.nrt_snapshot(true, false).unwrap();
    assert_eq!(s1.segment_infos.segments.len(), 2, "the flushed segment");
    assert_eq!(live_docs(&s1.segment_infos, &dir), 10);
    assert!(s1.segment_infos.version > s0.segment_infos.version);
    assert!(w.nrt_is_current(&s1.segment_infos));
    assert!(!w.nrt_is_current(&s0.segment_infos));
    // Nothing committed: a reader of the directory still sees 10 documents.
    let latest = crate::segment_infos::read_latest(&dir).unwrap();
    assert_eq!(latest.segments.len(), 1);

    // An unchanged view keeps its version.
    let s2 = w.nrt_snapshot(true, false).unwrap();
    assert_eq!(s2.segment_infos.version, s1.segment_infos.version);

    // A commit is written above every snapshot's version.
    w.commit().unwrap();
    assert!(w.segment_infos().version > s2.segment_infos.version);
    let s3 = w.nrt_snapshot(true, false).unwrap();
    assert!(s3.segment_infos.version >= s2.segment_infos.version);
    assert!(!s3.hold.files().is_empty());
    assert!(format!("{:?}", s3.hold).contains("files"));
}

/// The pin is the point: a snapshot's segments are merged away and their
/// commit superseded, and the files stay until the reader lets go -- then
/// the writer's next operation reclaims them.
///
/// Seen to fail with pinning disabled (`pin_segment_files` returning no
/// files): the merge's commit deletes them at once.
#[test]
fn a_snapshots_files_outlive_a_merge_until_it_is_dropped() {
    let tmp = TempDir::new("nrt-hold");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir);
    for i in 0..3 {
        w.add_document(doc(&format!("a{i}"))).unwrap();
        w.commit().unwrap();
    }
    let snap = w.nrt_snapshot(true, false).unwrap();
    // The three segments' files, as the directory lists them -- not as the
    // hold says, so the check below does not trust what it is checking.
    let pinned: Vec<String> = std::fs::read_dir(tmp.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|f| {
            ["_0.", "_0_", "_1.", "_1_", "_2.", "_2_"]
                .iter()
                .any(|p| f.starts_with(p))
        })
        .collect();
    assert!(pinned.iter().any(|f| f.ends_with(".si")), "{pinned:?}");
    assert!(snap.hold.files().iter().all(|f| pinned.contains(f)));
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    assert_eq!(w.segment_infos().segments.len(), 1);
    let on_disk = |f: &String| tmp.path().join(f).exists();
    assert!(
        pinned.iter().all(on_disk),
        "a pinned file was deleted while a reader held it"
    );

    drop(snap);
    // Released at the writer's next operation.
    w.delete_unused_files().unwrap();
    let merged = &w.segment_infos().segments[0].segment_name;
    assert!(
        pinned
            .iter()
            .filter(|f| !f.starts_with(&format!("{merged}.")))
            .all(|f| !on_disk(f)),
        "released files must be reclaimed"
    );
}

/// The two `NrtSource`s: a single-threaded writer behind a mutex, and the
/// concurrent writer, whose slots' buffers and delete log also count as
/// changes.
#[test]
fn both_writers_are_nrt_sources() {
    let tmp = TempDir::new("nrt-mutex");
    let dir = FsDirectory::open(&tmp);
    let w = Mutex::new(writer(&dir));
    w.lock().unwrap().add_document(doc("a")).unwrap();
    let src: &dyn NrtSource = &w;
    assert!(std::ptr::eq(
        src.directory() as *const dyn Directory as *const u8,
        &dir as *const FsDirectory as *const u8
    ));
    let snap = src.nrt_snapshot(true, true).unwrap();
    assert_eq!(snap.segment_infos.segments.len(), 1);
    assert!(src.nrt_is_current(&snap.segment_infos).unwrap());
    w.lock().unwrap().add_document(doc("b")).unwrap();
    assert!(!src.nrt_is_current(&snap.segment_infos).unwrap());

    let tmp2 = TempDir::new("nrt-concurrent");
    let dir2 = FsDirectory::open(&tmp2);
    let cw = ConcurrentIndexWriter::new(writer(&dir2), 2).unwrap();
    cw.add_document(doc("a")).unwrap();
    cw.add_document(doc("b")).unwrap();
    let src: &dyn NrtSource = &cw;
    let snap = src.nrt_snapshot(true, false).unwrap();
    assert_eq!(live_docs(&snap.segment_infos, &dir2), 2);
    assert!(src.nrt_is_current(&snap.segment_infos).unwrap());
    cw.delete_documents_by_term(&[Term::new("id", "a")])
        .unwrap();
    assert!(!src.nrt_is_current(&snap.segment_infos).unwrap());
    let snap2 = src.nrt_snapshot(true, false).unwrap();
    assert_eq!(live_docs(&snap2.segment_infos, &dir2), 1);
    cw.add_document(doc("c")).unwrap();
    assert!(!src.nrt_is_current(&snap2.segment_infos).unwrap());
    assert!(std::ptr::eq(
        src.directory() as *const dyn Directory as *const u8,
        &dir2 as *const FsDirectory as *const u8
    ));
}

/// A snapshot of a writer with no segments pins no files: dropping its
/// hold returns nothing to release, and the next snapshot (and commit) go
/// on as usual.
#[test]
fn a_snapshot_of_an_empty_writer_pins_nothing() {
    let tmp = TempDir::new("nrt-empty");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir);
    let snap = w.nrt_snapshot(true, false).unwrap();
    assert!(snap.segment_infos.segments.is_empty());
    assert!(snap.hold.files().is_empty());
    drop(snap);
    w.add_document(doc("a")).unwrap();
    let snap = w.nrt_snapshot(true, false).unwrap();
    assert_eq!(live_docs(&snap.segment_infos, &dir), 1);
    assert!(!snap.hold.files().is_empty());
    drop(snap);
    w.commit().unwrap();
}
