//! Differential test for `DirectoryReader::list_commits` and
//! `DirectoryReader::open_commit` against
//! `fixtures/data/merge_policies/commits.properties`
//! (`GenMergePolicies.commitsIndex`).
//!
//! Java kept three commits under `NoDeletionPolicy` -- two documents; a third
//! and a delete, with user data; a fourth with other user data -- and
//! recorded what `DirectoryReader.listCommits` says of each, and
//! `maxDoc`/`numDocs` of `DirectoryReader.open(commit)`. This test requires
//! the same answers from the Rust reader over Java's index, and over an index
//! the Rust writer built by the same operations.
//!
//! Regenerate with `scripts/gen-fixtures.sh --only GenMergePolicies`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_file_deleter::DeletionPolicy;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::directory::Directory;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn fixtures() -> String {
    format!(
        "{}/../../fixtures/data/merge_policies",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn expected() -> HashMap<String, String> {
    std::fs::read_to_string(format!("{}/commits.properties", fixtures()))
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// Every key Java recorded for commit `i`, as the Rust reader sees it.
fn check(dir: &dyn Directory, want: &HashMap<String, String>, files_too: bool, what: &str) {
    let commits = DirectoryReader::list_commits(dir).unwrap();
    assert_eq!(commits.len().to_string(), want["count"], "{what}: count");
    for (i, c) in commits.iter().enumerate() {
        let key = |k: &str| want[&format!("commit.{i}.{k}")].clone();
        assert_eq!(c.generation().to_string(), key("generation"), "{what} {i}");
        assert_eq!(c.segments_file_name(), key("segments_file"), "{what} {i}");
        assert_eq!(
            c.segment_count().to_string(),
            key("segment_count"),
            "{what} {i}"
        );
        if files_too {
            let mut files = c.file_names().to_vec();
            files.sort();
            assert_eq!(files.join(","), key("files"), "{what} {i} files");
        }
        let mut ud: Vec<String> = c
            .user_data()
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect();
        ud.sort();
        assert_eq!(ud.join(";"), key("user_data"), "{what} {i} user data");
        let r = DirectoryReader::open_commit(dir, c).unwrap();
        assert_eq!(r.max_doc().to_string(), key("max_doc"), "{what} {i} maxDoc");
        assert_eq!(
            r.num_docs().to_string(),
            key("num_docs"),
            "{what} {i} numDocs"
        );
    }
}

#[test]
fn list_commits_over_javas_index_matches_lucene() {
    let dir = FsDirectory::open(format!("{}/commits_index", fixtures()));
    check(&dir, &expected(), true, "java index");
}

fn doc(id: &str) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(id.to_string()),
        }],
    }
}

#[test]
fn list_commits_over_a_rust_written_index_matches_lucene() {
    let tmp = TempDir::new("list-commits");
    let dir = FsDirectory::open(tmp.path());
    let field = FieldInfo {
        index_options: IndexOptions::Docs,
        omit_norms: true,
        ..FieldInfo::new("id", 0)
    };
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, vec![field], "Lucene104", version).unwrap();
    w.set_postings_field(Some("id")).unwrap();
    w.set_deletion_policy(DeletionPolicy::KeepAll).unwrap();
    w.add_document(doc("a")).unwrap();
    w.add_document(doc("b")).unwrap();
    w.commit().unwrap();
    w.add_document(doc("c")).unwrap();
    w.delete_documents_by_term(&[Term::new("id", b"a".to_vec())])
        .unwrap();
    w.set_live_commit_data(vec![
        ("step".into(), "two".into()),
        ("note".into(), "delete".into()),
    ]);
    w.commit().unwrap();
    w.add_document(doc("d")).unwrap();
    w.set_live_commit_data(vec![("step".into(), "three".into())]);
    w.commit().unwrap();
    drop(w);
    // The file sets are Java's too: the same codec names the same files.
    check(&dir, &expected(), true, "rust index");
}

#[test]
fn list_commits_refuses_an_empty_directory_and_a_bad_segments_name() {
    let tmp = TempDir::new("list-commits-empty");
    let dir = FsDirectory::open(tmp.path());
    assert!(DirectoryReader::list_commits(&dir).is_err());

    // Java's `listCommits` tests `startsWith("segments")` alone, so a stray
    // unparsable name is an error rather than skipped.
    let src = format!("{}/commits_index", fixtures());
    for entry in std::fs::read_dir(&src).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), tmp.path().join(entry.file_name())).unwrap();
    }
    std::fs::write(tmp.path().join("segments_zz!"), b"").unwrap();
    assert!(DirectoryReader::list_commits(&dir).is_err());
}
