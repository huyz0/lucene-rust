//! Differential test for `lucene_index::deletion_policy` driven through a
//! real writer, against `fixtures/data/merge_policies/snapshots.properties`
//! (`GenMergePolicies.snapshotIndex`).
//!
//! Java wrote three commits under a `PersistentSnapshotDeletionPolicy` over
//! `KeepOnlyLastCommitDeletionPolicy`, holding generation 1 once and
//! generation 2 twice. This test:
//!
//! - replays the same operations with the Rust writer on an empty directory,
//!   and requires the same surviving commits and a `snapshots_N` of the same
//!   name and the **same bytes**;
//! - opens a Rust writer over a copy of Java's index with the snapshots Java
//!   persisted (`onInit`), releases generation 1, runs `deleteUnusedFiles`
//!   and commits once more, requiring Java's listing and bytes at each step.
//!
//! Regenerate with `scripts/gen-fixtures.sh --only GenMergePolicies`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::sync::Arc;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::deletion_policy::{
    IndexDeletionPolicy, KeepLastNCommitsDeletionPolicy, KeepOnlyLastCommitDeletionPolicy,
    OpenMode, PersistentSnapshotDeletionPolicy, SnapshotDeletionPolicy,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::directory::Directory;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn data(rel: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(format!(
        "{}/../../fixtures/data/merge_policies/{rel}",
        env!("CARGO_MANIFEST_DIR")
    ))
}

fn expected() -> HashMap<String, String> {
    std::fs::read_to_string(data("snapshots.properties"))
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn id_field() -> FieldInfo {
    FieldInfo {
        name: "id".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: true,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Byte,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

fn doc(id: &str) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(id.to_string()),
        }],
    }
}

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

/// `GenMergePolicies.listing`: commit points and snapshot files only.
fn listing(dir: &dyn Directory) -> String {
    let mut names: Vec<String> = dir
        .list_all()
        .unwrap()
        .into_iter()
        .filter(|f| f.starts_with("segments_") || f.starts_with("snapshots_"))
        .collect();
    names.sort();
    names.join(",")
}

fn hex(dir: &dyn Directory, name: &str) -> String {
    dir.open(name)
        .unwrap()
        .to_vec()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn open_writer<'d>(dir: &'d FsDirectory, policy: Arc<dyn IndexDeletionPolicy>) -> IndexWriter<'d> {
    IndexWriter::open_with_deletion_policy(dir, vec![id_field()], "Lucene104", version(), policy)
        .unwrap()
}

#[test]
fn rust_writer_persists_the_snapshots_java_persists() {
    let want = expected();
    let tmp = TempDir::new("deletion-policies-written");
    let dir = Arc::new(FsDirectory::open(tmp.path()));
    let policy = Arc::new(
        PersistentSnapshotDeletionPolicy::new(
            Box::new(KeepOnlyLastCommitDeletionPolicy),
            dir.clone(),
            OpenMode::Create,
        )
        .unwrap(),
    );
    let mut w = open_writer(&dir, policy.clone());
    w.add_document(doc("a")).unwrap();
    w.commit().unwrap();
    policy.snapshot().unwrap();
    w.add_document(doc("b")).unwrap();
    w.commit().unwrap();
    policy.snapshot().unwrap();
    policy.snapshot().unwrap();
    w.add_document(doc("c")).unwrap();
    w.commit().unwrap();
    drop(w);

    assert_eq!(listing(&*dir), want["written.files"]);
    let file = policy.last_save_file().unwrap();
    assert_eq!(file, want["written.snapshots_file"]);
    assert_eq!(hex(&*dir, &file), want["written.snapshots_bytes"]);
    assert_eq!(
        policy.snapshots_policy().snapshot_count().to_string(),
        want["written.snapshot_count"]
    );
}

#[test]
fn rust_writer_honours_snapshots_java_persisted() {
    let want = expected();
    let tmp = TempDir::new("deletion-policies-replay");
    for entry in std::fs::read_dir(data("snapshots_index")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), tmp.path().join(entry.file_name())).unwrap();
    }
    let dir = Arc::new(FsDirectory::open(tmp.path()));
    let policy = Arc::new(
        PersistentSnapshotDeletionPolicy::new(
            Box::new(KeepOnlyLastCommitDeletionPolicy),
            dir.clone(),
            OpenMode::Append,
        )
        .unwrap(),
    );
    let mut w = open_writer(&dir, policy.clone());
    assert_eq!(listing(&*dir), want["open.files"]);
    let gens: Vec<String> = policy
        .snapshots_policy()
        .snapshots()
        .iter()
        .map(|c| c.generation().to_string())
        .collect();
    assert_eq!(gens.join(","), want["open.snapshot_gens"]);
    assert_eq!(
        policy.snapshots_policy().snapshot_count().to_string(),
        want["open.snapshot_count"]
    );
    // The snapshotted commit carries what Java's `IndexCommit` does.
    let held = policy.snapshots_policy().index_commit(1).unwrap();
    assert_eq!(held.segments_file_name(), "segments_1");
    assert_eq!(held.segment_count(), 1);
    assert!(held.file_names().iter().any(|f| f == "_0.si"));

    policy.release_gen(1).unwrap();
    let file = policy.last_save_file().unwrap();
    assert_eq!(file, want["release.snapshots_file"]);
    assert_eq!(hex(&*dir, &file), want["release.snapshots_bytes"]);
    w.delete_unused_files().unwrap();
    assert_eq!(listing(&*dir), want["release.files"]);

    w.add_document(doc("d")).unwrap();
    w.commit().unwrap();
    assert_eq!(listing(&*dir), want["commit.files"]);
    let gens: Vec<i64> = w.index_commits().iter().map(|c| c.generation()).collect();
    assert_eq!(gens, vec![2, 4]);
}

/// `KeepLastNCommitsDeletionPolicy` and a plain `SnapshotDeletionPolicy`
/// through the writer: `onInit` prunes what was already on disk, `onCommit`
/// keeps the window, and a snapshot outlives the window until released.
#[test]
fn keep_last_n_and_snapshot_through_the_writer() {
    let tmp = TempDir::new("deletion-policies-keep-n");
    let dir = FsDirectory::open(tmp.path());
    {
        let mut w = open_writer(
            &dir,
            Arc::new(lucene_index::deletion_policy::NoDeletionPolicy),
        );
        for id in ["a", "b", "c", "d"] {
            w.add_document(doc(id)).unwrap();
            w.commit().unwrap();
        }
        assert_eq!(listing(&dir), "segments_1,segments_2,segments_3,segments_4");
    }
    let snap = Arc::new(SnapshotDeletionPolicy::new(Box::new(
        KeepLastNCommitsDeletionPolicy::new(2).unwrap(),
    )));
    let mut w = open_writer(&dir, snap.clone());
    // onInit: only the newest two survive.
    assert_eq!(listing(&dir), "segments_3,segments_4");
    let held = snap.snapshot().unwrap();
    assert_eq!(held.generation(), 4);
    for id in ["e", "f", "g"] {
        w.add_document(doc(id)).unwrap();
        w.commit().unwrap();
    }
    assert_eq!(listing(&dir), "segments_4,segments_6,segments_7");
    snap.release(&held).unwrap();
    w.delete_unused_files().unwrap();
    assert_eq!(listing(&dir), "segments_6,segments_7");

    // Switching the policy re-runs it at once (`revisitPolicy`).
    w.set_index_deletion_policy(Arc::new(KeepOnlyLastCommitDeletionPolicy))
        .unwrap();
    assert_eq!(listing(&dir), "segments_7");
    assert!(KeepLastNCommitsDeletionPolicy::new(0).is_err());
}
