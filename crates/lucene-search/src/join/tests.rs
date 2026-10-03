// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::directory_reader::DirectoryReader;
use crate::query::{Clause, MatchAllDocsQuery, TermQuery};
use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn doc(id: &str, kind: &str, block: usize) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", id, Store::Yes));
    d.add(StringField::new("type", kind, Store::No));
    d.add(StringField::new("block", block.to_string(), Store::No));
    d
}

fn block(b: usize, children: usize) -> Vec<Document> {
    let mut docs: Vec<Document> = (0..children)
        .map(|j| doc(&format!("c{b}_{j}"), "child", b))
        .collect();
    docs.push(doc(&format!("p{b}"), "parent", b));
    docs
}

fn term(field: &str, value: &str) -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::Term(TermQuery::new(
            field,
            value.as_bytes().to_vec(),
        ))],
        ..Default::default()
    }
}

fn parents() -> QueryBitSetProducer {
    QueryBitSetProducer::new(term("type", "parent"))
}

/// Writes `ops` into a fresh index and runs `check_join_index` over it.
fn check(name: &str, ops: impl FnOnce(&mut IndexWriter<'_>)) -> Result<()> {
    let tmp = TempDir::new(name);
    let dir = FsDirectory::open(tmp.path());
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        ops(&mut w);
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    check_join_index(&segments, &parents())
}

#[test]
fn a_well_formed_block_index_passes() {
    check("cji-ok", |w| {
        for b in 0..20 {
            w.add_fields_documents(&block(b, b % 4)).unwrap();
            if b % 7 == 6 {
                w.commit().unwrap();
            }
        }
        // Whole blocks deleted, by a term every document of the block has.
        w.delete_documents_by_term(&[Term::new("block", "3"), Term::new("block", "12")])
            .unwrap();
    })
    .unwrap();
}

#[test]
fn every_segment_needs_a_parent() {
    let err = check("cji-no-parent", |w| {
        w.add_fields_document(&doc("c0", "child", 0)).unwrap();
    })
    .unwrap_err();
    assert!(
        err.to_string()
            .starts_with("Every segment should have at least one parent, but "),
        "{err}"
    );
}

#[test]
fn a_segment_must_end_in_a_parent() {
    let err = check("cji-child-last", |w| {
        w.add_fields_documents(&block(0, 2)).unwrap();
        w.add_fields_document(&doc("c1_0", "child", 1)).unwrap();
    })
    .unwrap_err();
    assert!(
        err.to_string().ends_with("has a child as a last doc"),
        "{err}"
    );
}

#[test]
fn blocks_are_deleted_whole() {
    let err = check("cji-deleted-child", |w| {
        w.add_fields_documents(&block(0, 2)).unwrap();
        w.add_fields_documents(&block(1, 2)).unwrap();
        w.delete_documents_by_term(&[Term::new("id", "c1_1")])
            .unwrap();
    })
    .unwrap_err();
    assert_eq!(
        err.to_string()
            .split(" of segment ")
            .next()
            .map(str::to_string),
        Some("Parent doc 5".to_string()),
        "{err}"
    );
    assert!(err
        .to_string()
        .ends_with("is live but has a deleted child document 4"));

    let err = check("cji-deleted-parent", |w| {
        w.add_fields_documents(&block(0, 2)).unwrap();
        w.delete_documents_by_term(&[Term::new("id", "p0")])
            .unwrap();
    })
    .unwrap_err();
    assert!(
        err.to_string()
            .ends_with("is deleted but has a live child document 0"),
        "{err}"
    );
}

/// `QueryBitSetProducer`: the query's matches with deleted documents, one
/// set per segment core, cached; `None` without a scorer; a match-all is
/// every document.
#[test]
fn the_producer_caches_each_segment_and_keeps_deleted_parents() {
    let tmp = TempDir::new("qbsp");
    let dir = FsDirectory::open(tmp.path());
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        for b in 0..3 {
            w.add_fields_documents(&block(b, 1)).unwrap();
        }
        w.delete_documents_by_term(&[Term::new("block", "1")])
            .unwrap();
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let seg = &segments[0];
    assert!(seg.live_docs.is_some());

    let producer = parents();
    let bits = producer.bit_set(seg).unwrap().unwrap();
    let set: Vec<usize> = (0..bits.len()).filter(|&i| bits.get(i)).collect();
    assert_eq!(set, [1, 3, 5], "the deleted parent 3 included");
    let again = producer.bit_set(seg).unwrap().unwrap();
    assert!(Arc::ptr_eq(&bits, &again), "cached");
    producer.clear();
    let fresh = producer.bit_set(seg).unwrap().unwrap();
    assert!(!Arc::ptr_eq(&bits, &fresh));
    assert_eq!(*fresh, *bits);
    assert_eq!(producer.query(), &term("type", "parent"));
    assert!(producer.key().starts_with("QueryBitSetProducer("));

    assert!(QueryBitSetProducer::new(term("type", "nothing"))
        .bit_set(seg)
        .unwrap()
        .is_none());
    let all = QueryBitSetProducer::new(BooleanQuery {
        must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX))],
        ..Default::default()
    })
    .bit_set(seg)
    .unwrap()
    .unwrap();
    assert_eq!(all.cardinality(), 6);
}
