//! `ReaderManager` over a real index: a refresh after a commit swaps in a
//! reader that sees it, one with nothing new keeps the current reader, and a
//! reader acquired before a refresh keeps answering for its own commit.
//!
//! `ReaderManager` writes nothing to disk, so there is no Java fixture: what
//! it serves is `DirectoryReader::open_if_changed`, which has its own.

use std::sync::{Arc, Mutex};

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::reader_manager::ReaderManager;
use lucene_search::Error;
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
    let dir = Arc::new(FsDirectory::open(tmp.path()));
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w =
        IndexWriter::open(&*dir, vec![FieldInfo::new("id", 0)], "Lucene104", version).unwrap();
    w.add_document(doc("a")).unwrap();
    w.commit().unwrap();

    let manager = ReaderManager::open(dir.clone()).unwrap();
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

    manager.close().unwrap();
    assert!(matches!(manager.acquire(), Err(Error::AlreadyClosed(_))));

    // `ReaderManager(DirectoryReader)`: starts from the given reader.
    let from = ReaderManager::from_reader(
        dir.clone(),
        lucene_search::directory_reader::DirectoryReader::open(&*dir).unwrap(),
    );
    assert_eq!(from.acquire().unwrap().num_docs(), 3);
    assert!(from.maybe_refresh().unwrap());
}

/// `new ReaderManager(writer)`: a refresh shows what the writer indexed and
/// deleted without any commit, and a reader acquired before it keeps its
/// own point-in-time view.
#[test]
fn reader_manager_over_a_writer_is_near_real_time() {
    let tmp = TempDir::new("reader-manager-nrt");
    let dir: &'static FsDirectory = Box::leak(Box::new(FsDirectory::open(tmp.path())));
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
    w.add_document(doc("a")).unwrap();
    w.add_document(doc("b")).unwrap();
    let writer = Arc::new(Mutex::new(w));

    let manager = ReaderManager::open_from_writer(Arc::clone(&writer) as _).unwrap();
    let first = manager.acquire().unwrap();
    assert!(first.is_nrt());
    assert_eq!(first.num_docs(), 2);
    assert!(manager.maybe_refresh().unwrap());
    assert!(
        Arc::ptr_eq(&first, &manager.acquire().unwrap()),
        "the writer did not change"
    );

    {
        let mut w = writer.lock().unwrap();
        w.add_document(doc("c")).unwrap();
        w.delete_documents_by_term(&[Term::new("id", "a")]).unwrap();
    }
    manager.maybe_refresh_blocking().unwrap();
    let second = manager.acquire().unwrap();
    assert_eq!(second.num_docs(), 2, "c added, a deleted");
    assert_eq!(second.doc_freq("id", b"c").unwrap(), 1);
    assert_eq!(first.num_docs(), 2);
    assert_eq!(first.doc_freq("id", b"c").unwrap(), 0);
    // Nothing was committed.
    assert!(lucene_index::segment_infos::read_latest(dir).is_err());
    manager.release(first);
    manager.release(second);
    manager.close().unwrap();

    // `new ReaderManager(writer, false, true)`.
    let other = ReaderManager::from_writer(Arc::clone(&writer) as _, false, true).unwrap();
    assert!(other.acquire().unwrap().is_nrt());
    other.close().unwrap();
}
