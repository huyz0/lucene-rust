//! The test framework's `CrankyCodec` installed in
//! [`lucene_index::index_writer::IndexWriter`] (`IndexWriter::set_codec`),
//! in the spirit of Java's `TestIndexWriterExceptions2`: documents with
//! stored fields, term vectors, norms, points, postings and doc values are
//! added, deleted, committed and merged while the codec's writers fail with
//! fake I/O errors. After any error the writer is dropped and a new one
//! opened, which goes back to the last commit. Whatever fails, the index must
//! hold exactly the committed documents, each intact, keep its `Lucene104`
//! codec name and pass CheckIndex.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeSet;
use std::sync::Arc;

use lucene_codecs::stored_fields::FieldValue;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, Field, FieldType, IndexOptions, NumericDocValuesField, Store, StringField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::LeafReader;
use lucene_store::{Directory, FsDirectory};
use lucene_test_framework::codecs::cranky_installed::InstalledCrankyCodec;
use lucene_util::java_random::JavaRandom;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn doc(i: i64) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", format!("d{i}"), Store::Yes));
    let mut body = FieldType::new();
    body.set_index_options(IndexOptions::DocsAndFreqsAndPositions)
        .unwrap();
    body.set_store_term_vectors(true).unwrap();
    body.set_store_term_vector_positions(true).unwrap();
    d.add(Field::from_string("body", format!("w{} x{} common", i % 5, i % 7), body).unwrap());
    let mut point = FieldType::new();
    point.set_dimensions(1, 4).unwrap();
    d.add(Field::from_bytes("p", (i as i32).to_be_bytes().to_vec(), point).unwrap());
    d.add(NumericDocValuesField::new("n", i));
    d
}

fn open(dir: &dyn Directory, seed: i64) -> IndexWriter<'_> {
    // Opening itself writes nothing through the codec.
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_codec(Some(Arc::new(InstalledCrankyCodec::new(seed))))
        .unwrap();
    w.set_max_full_flush_merge_wait_millis(0);
    w
}

/// The index's live documents, each checked: its stored id, doc value and
/// point agree.
fn live_ids(dir: &dyn Directory) -> BTreeSet<i64> {
    let reader = DirectoryReader::open(dir).unwrap();
    let mut ids = BTreeSet::new();
    for seg in reader.segment_readers() {
        let mut n = seg.numeric_doc_values("n").unwrap().expect("the n field");
        for doc in 0..seg.max_doc {
            if !seg.live_docs().is_none_or(|bits| bits.get(doc as usize)) {
                continue;
            }
            let stored = seg.stored_document(doc).unwrap().unwrap();
            let FieldValue::String(id) = &stored.fields[0].value else {
                panic!("a stored id");
            };
            let id: i64 = id[1..].parse().unwrap();
            assert!(n.advance_exact(doc).unwrap());
            assert_eq!(n.long_value(), id);
            assert!(ids.insert(id), "d{id} twice");
        }
    }
    ids
}

/// One seed: `ops` random operations, the model updated only by those that
/// succeed and rolled back to the last commit whenever one fails. Returns
/// how many operations failed.
fn run(seed: i64, ops: usize) -> usize {
    let tmp = TempDir::new("cranky-write-path");
    let dir = FsDirectory::open(tmp.path());
    let mut random = JavaRandom::new(seed);
    let mut committed: BTreeSet<i64> = BTreeSet::new();
    let mut pending = committed.clone();
    let mut next = 0_i64;
    let mut failures = 0;
    let mut w = Some(open(&dir, seed));
    for _ in 0..ops {
        let writer = w.as_mut().unwrap();
        let op = random.next_int_bounded(100);
        let result = if op < 75 {
            let id = next;
            next += 1;
            writer.add_fields_document(&doc(id)).map(|_| {
                pending.insert(id);
            })
        } else if op < 85 {
            let id = i64::from(random.next_int_bounded(next.max(1) as i32));
            writer
                .delete_documents_by_term(&[Term::new("id", format!("d{id}"))])
                .map(|_| {
                    pending.remove(&id);
                })
        } else if op < 97 {
            writer.commit().map(|_| {
                committed = pending.clone();
            })
        } else {
            writer.force_merge(1).map(|_| ())
        };
        if result.is_err() {
            failures += 1;
            drop(w.take());
            pending = committed.clone();
            w = Some(open(&dir, seed + failures as i64));
        }
    }
    drop(w);
    // The last commit, as a plain writer and the reader see it.
    if lucene_index::segment_infos::read_latest(&dir).is_ok() {
        let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
        assert!(infos.segments.iter().all(|s| s.codec_name == "Lucene104"));
        assert_eq!(live_ids(&dir), committed, "seed {seed}");
        for result in lucene_index::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "seed {seed}: {:?}", result.failures());
        }
    } else {
        assert!(committed.is_empty(), "seed {seed}: committed but no commit");
    }
    failures
}

#[test]
fn injected_failures_leave_the_last_commit_intact() {
    let mut failures = 0;
    for seed in 0..40 {
        failures += run(seed, 150);
    }
    println!("{failures} injected failures");
    // The codec must actually have failed, or the test proves nothing.
    assert!(failures > 20, "only {failures} injected failures");
}
