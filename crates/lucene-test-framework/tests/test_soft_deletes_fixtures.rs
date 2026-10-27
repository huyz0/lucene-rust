//! `SoftDeletesDirectoryReaderWrapper`, soft updates and update by query:
//! differential against
//! `fixtures/src/testframework/GenTestFrameworkSoftDeletes.java`'s
//! `soft_deletes.txt` -- the same operations, from the same seeds, through
//! this port's writer, and every reader taken on the way described the same
//! way (its leaves' `maxDoc/numDocs` and its live ids).
//!
//! A `dir` reader is [`DirectoryReader::soft_deletes_wrapper`] over the
//! latest commit; an `nrt` reader is a near-real-time reader, which Java's
//! writer hands out with soft deletes resolved and this port wraps.
//!
//! Regenerate with `scripts/gen-test-framework-fixtures.sh --only TestFrameworkSoftDeletes`.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo};
use lucene_codecs::stored_fields::FieldValue;
use lucene_index::buffered_updates::{DeleteQuery, DocValuesUpdate, Term};
use lucene_index::document::{Document, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::ByteBuffersDirectory;
use lucene_util::java_random::JavaRandom;

const SOFT: &str = "___soft_deletes";

fn fixture() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data/soft_deletes.txt");
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn doc(id: i32) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", id.to_string(), Store::Yes));
    d
}

fn id(id: i32) -> Term {
    Term::new("id", id.to_string())
}

fn soft(term: &Term) -> DocValuesUpdate {
    DocValuesUpdate::Numeric {
        term: term.clone(),
        field: SOFT.to_string(),
        value: Some(1),
    }
}

fn describe(seed: i64, op: i32, kind: &str, reader: &DirectoryReader) -> String {
    let mut line = format!("{seed} {op} {kind}");
    let mut live = BTreeSet::new();
    for seg in reader.segment_readers() {
        line.push_str(&format!(" {}/{}", seg.max_doc, seg.num_docs()));
        for doc in 0..seg.max_doc {
            if seg.live_docs().is_none_or(|l| l.get(doc as usize)) {
                let stored = seg.stored_document(doc).unwrap().unwrap();
                let id = stored
                    .fields
                    .iter()
                    .find_map(|f| match &f.value {
                        FieldValue::String(s) => Some(s.parse::<i32>().unwrap()),
                        _ => None,
                    })
                    .unwrap();
                live.insert(id);
            }
        }
    }
    line.push_str(" |");
    for id in live {
        line.push_str(&format!(" {id}"));
    }
    line
}

fn run(seed: i64, out: &mut Vec<String>) {
    let dir = ByteBuffersDirectory::new();
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    w.enable_explicit_documents().unwrap();
    w.register_field(
        FieldInfo::new(SOFT, 0)
            .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1)
            .with_soft_deletes_field(true),
    )
    .unwrap();
    let w = Mutex::new(w);
    let dir_reader = |op: i32, out: &mut Vec<String>| {
        let reader = DirectoryReader::open(&dir).unwrap();
        let wrapped = reader.soft_deletes_wrapper(SOFT).unwrap();
        out.push(describe(seed, op, "dir", &wrapped));
    };
    let mut ops = JavaRandom::new(seed);
    for i in 0..150 {
        let k = ops.next_int_bounded(30);
        let mut writer = w.lock().unwrap();
        match ops.next_int_bounded(12) {
            0..=3 => {
                let t = id(k);
                let s = soft(&t);
                writer
                    .soft_update_fields_documents(t, &[doc(k)], &[s])
                    .unwrap();
            }
            4 => {
                writer.delete_documents_by_term(&[id(k)]).unwrap();
            }
            5 => {
                writer.update_fields_documents(id(k), &[doc(k)]).unwrap();
            }
            6 => {
                writer.commit().unwrap();
                drop(writer);
                dir_reader(i, out);
            }
            7 => {
                drop(writer);
                let reader = DirectoryReader::open_nrt(&w, true, false).unwrap();
                out.push(describe(
                    seed,
                    i,
                    "nrt",
                    &reader.soft_deletes_wrapper(SOFT).unwrap(),
                ));
            }
            8 => {
                writer
                    .update_fields_documents_by_query(DeleteQuery::Term(id(k)), &[doc(k)])
                    .unwrap();
            }
            9 => writer.flush().unwrap(),
            _ => {
                writer.add_fields_document(&doc(k)).unwrap();
            }
        }
    }
    w.lock().unwrap().commit().unwrap();
    dir_reader(150, out);
    w.into_inner().unwrap().close().unwrap();
}

#[test]
fn soft_deletes_match_lucene() {
    let expected = fixture();
    let mut actual = Vec::new();
    for seed in 0..40 {
        run(seed, &mut actual);
    }
    let expected: Vec<&str> = expected.lines().collect();
    let mut diffs = 0;
    for (i, (want, got)) in expected.iter().zip(&actual).enumerate() {
        if *want != got {
            diffs += 1;
            if diffs <= 10 {
                eprintln!("line {}:\n  want {want}\n  got  {got}", i + 1);
            }
        }
    }
    assert_eq!(expected.len(), actual.len());
    assert_eq!(diffs, 0, "{diffs} of {} lines differ", expected.len());
}

fn soft_writer(dir: &ByteBuffersDirectory) -> IndexWriter<'_> {
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version).unwrap();
    w.enable_explicit_documents().unwrap();
    w.register_field(
        FieldInfo::new(SOFT, 0)
            .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1)
            .with_soft_deletes_field(true),
    )
    .unwrap();
    w
}

fn soft_update(w: &mut IndexWriter<'_>, k: i32) {
    let t = id(k);
    let s = soft(&t);
    w.soft_update_fields_documents(t, &[doc(k)], &[s]).unwrap();
}

/// Every reopen of a wrapper wraps again (Java's `doWrapDirectoryReader`),
/// a segment with no soft deletes is shared as it is, and segments left
/// without a live document -- soft or hard -- are dropped.
#[test]
fn a_wrapper_wraps_again_on_reopen() {
    let dir = ByteBuffersDirectory::new();
    let mut w = soft_writer(&dir);
    for k in 0..4 {
        w.add_fields_document(&doc(k)).unwrap();
    }
    w.commit().unwrap();
    // A second segment, entirely soft-deleted below, and a third whose
    // documents are hard-deleted.
    w.add_fields_document(&doc(10)).unwrap();
    w.commit().unwrap();
    w.add_fields_document(&doc(20)).unwrap();
    w.commit().unwrap();
    w.delete_documents_by_term(&[id(20)]).unwrap();
    w.commit().unwrap();
    let plain = DirectoryReader::open(&dir).unwrap();
    let first = plain.soft_deletes_wrapper(SOFT).unwrap();
    assert_eq!(first.num_docs(), 5);
    // Nothing soft-deleted: the segments are the plain reader's own.
    let leaf_key =
        |r: &DirectoryReader, i: usize| r.segment_readers()[i].reader_cache_helper().key_id();
    assert_eq!(leaf_key(&first, 0), leaf_key(&plain, 0));
    assert_ne!(
        first.reader_cache_helper().key_id(),
        plain.reader_cache_helper().key_id()
    );

    soft_update(&mut w, 1);
    soft_update(&mut w, 10);
    w.commit().unwrap();
    let second = first.open_if_changed(&dir).unwrap().expect("a new commit");
    // The new version of 10 is live; its old segment is dropped.
    assert_eq!(second.num_docs(), 5);
    assert_eq!(
        describe(0, 0, "dir", &second),
        "0 0 dir 4/3 2/2 | 0 1 2 3 10"
    );
    let unwrapped = DirectoryReader::open(&dir).unwrap();
    assert_eq!(unwrapped.num_docs(), 7);
    // The soft-deleted segment shares its core under a new reader key.
    let rewrapped = unwrapped.soft_deletes_wrapper(SOFT).unwrap();
    let soft_leaf = &rewrapped.segment_readers()[0];
    let plain_leaf = &unwrapped.segment_readers()[0];
    assert_eq!(
        soft_leaf.core_cache_helper().key_id(),
        plain_leaf.core_cache_helper().key_id()
    );
    assert_ne!(
        soft_leaf.reader_cache_helper().key_id(),
        plain_leaf.reader_cache_helper().key_id()
    );

    // Reopening to a commit, and through a leaf sorter, keeps the wrapper.
    soft_update(&mut w, 2);
    w.commit().unwrap();
    let commit = DirectoryReader::list_commits(&dir).unwrap().pop().unwrap();
    let third = second
        .open_if_changed_to_commit(&dir, &commit)
        .unwrap()
        .expect("a new commit");
    assert_eq!(
        describe(0, 0, "dir", &third),
        "0 0 dir 4/2 2/2 1/1 | 0 1 2 3 10"
    );
    let sorted = third.with_leaf_sorter(|a, b| b.max_doc.cmp(&a.max_doc).reverse());
    soft_update(&mut w, 3);
    w.commit().unwrap();
    let fourth = sorted.open_if_changed(&dir).unwrap().expect("a new commit");
    assert_eq!(fourth.num_docs(), 5);
    assert!(sorted.open_if_changed(&dir).unwrap().is_some());
    assert!(fourth.open_if_changed(&dir).unwrap().is_none());
    // A field no segment has leaves every segment as it is.
    let none = unwrapped.soft_deletes_wrapper("absent").unwrap();
    assert_eq!(none.num_docs(), unwrapped.num_docs());
    w.close().unwrap();
}

/// A near-real-time wrapper reopened from the writer wraps again.
#[test]
fn a_near_real_time_wrapper_wraps_again() {
    let dir = ByteBuffersDirectory::new();
    let w = Mutex::new(soft_writer(&dir));
    for k in 0..3 {
        w.lock().unwrap().add_fields_document(&doc(k)).unwrap();
    }
    let first = DirectoryReader::open_nrt(&w, true, false)
        .unwrap()
        .soft_deletes_wrapper(SOFT)
        .unwrap();
    assert_eq!(first.num_docs(), 3);
    soft_update(&mut w.lock().unwrap(), 0);
    let second = first
        .open_if_changed_nrt(&w)
        .unwrap()
        .expect("the writer changed");
    assert_eq!(second.num_docs(), 3);
    assert_eq!(describe(0, 0, "nrt", &second), "0 0 nrt 3/2 1/1 | 0 1 2");
    assert!(second.open_if_changed_nrt(&w).unwrap().is_none());
    drop((first, second));
    w.into_inner().unwrap().close().unwrap();
}
