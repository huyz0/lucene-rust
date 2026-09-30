//! `ReaderManager` over a real index: a refresh after a commit swaps in a
//! reader that sees it, one with nothing new keeps the current reader, and a
//! reader acquired before a refresh keeps answering for its own commit.
//!
//! `ReaderManager` writes nothing to disk, so there is no Java fixture: what
//! it serves is `DirectoryReader::open_if_changed`, which has its own.

use std::sync::Arc;

use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::reader_manager::{Error, ReaderManager};
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

#[test]
fn reader_manager_follows_commits() {
    let tmp = TempDir::new("reader-manager");
    let dir = FsDirectory::open(tmp.path());
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w =
        IndexWriter::open(&dir, vec![FieldInfo::new("id", 0)], "Lucene104", version).unwrap();
    w.add_document(doc("a")).unwrap();
    w.commit().unwrap();

    let manager = ReaderManager::open(&dir).unwrap();
    let first = manager.acquire().unwrap();
    assert_eq!(first.num_docs(), 1);
    assert!(manager.maybe_refresh().unwrap());
    assert!(
        Arc::ptr_eq(&first, &manager.acquire().unwrap()),
        "no new commit"
    );

    w.add_document(doc("b")).unwrap();
    w.add_document(doc("c")).unwrap();
    w.commit().unwrap();
    manager.maybe_refresh_blocking().unwrap();
    let second = manager.acquire().unwrap();
    assert_eq!(second.num_docs(), 3);
    assert_eq!(second.segment_readers().len(), 2);
    // The reader acquired before still answers for its own commit.
    assert_eq!(first.num_docs(), 1);
    assert_eq!(
        first.segment_readers()[0]
            .stored_document(0)
            .unwrap()
            .unwrap()
            .fields[0]
            .value,
        FieldValue::String("a".into())
    );
    manager.release(first);
    manager.release(second);

    manager.close();
    assert!(matches!(manager.acquire(), Err(Error::AlreadyClosed)));
}
