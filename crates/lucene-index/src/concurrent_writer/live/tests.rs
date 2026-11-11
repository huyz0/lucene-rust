// Test arithmetic on small constants; see docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::atomic::Ordering;

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

use super::*;
use crate::buffered_updates::{DeleteQuery, Term};
use crate::index_writer::{
    ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use crate::segment_info::LuceneVersion;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

struct Fields {
    id: i32,
    body: i32,
    soft: i32,
}

fn open(dir: &FsDirectory, slots: usize, live: bool) -> (ConcurrentIndexWriter<'_>, Fields) {
    open_with(dir, slots, live, 16.0)
}

fn open_with(
    dir: &FsDirectory,
    slots: usize,
    live: bool,
    ram_mb: f64,
) -> (ConcurrentIndexWriter<'_>, Fields) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_ram_buffer_size_mb(ram_mb).unwrap();
    w.enable_explicit_documents().unwrap();
    let id = w
        .register_field(
            FieldInfo::new("_id", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
    let body = w
        .register_field(
            FieldInfo::new("body", 0).with_index_options(IndexOptions::DocsAndFreqsAndPositions),
        )
        .unwrap();
    let soft = w
        .register_field(
            FieldInfo::new("__soft_deletes", 0)
                .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1)
                .with_soft_deletes_field(true),
        )
        .unwrap();
    let w = ConcurrentIndexWriter::new(w, slots).unwrap();
    if live {
        w.enable_live_search().unwrap();
    }
    (w, Fields { id, body, soft })
}

fn term(t: &str, positions: &[i32]) -> InvertedTerm {
    InvertedTerm {
        term: t.as_bytes().to_vec(),
        freq: i32::try_from(positions.len().max(1)).unwrap(),
        positions: positions.to_vec(),
        offsets: Vec::new(),
        payloads: Vec::new(),
    }
}

/// Document `i`: `_id` "d{i}", `body` "common" at 0 and, in every third
/// document, "rare" at 1 and 2. Its norm is its length.
fn doc(f: &Fields, i: i64) -> ExplicitDocument {
    let mut body = vec![term("common", &[0])];
    if i % 3 == 0 {
        body.push(term("rare", &[1, 2]));
    }
    let length = body.iter().map(|t| i64::from(t.freq)).sum();
    ExplicitDocument {
        stored: vec![
            StoredField {
                field_number: f.id,
                value: FieldValue::String(format!("d{i}")),
            },
            StoredField {
                field_number: f.body,
                value: FieldValue::Long(i),
            },
        ],
        fields: ExplicitFields {
            inverted: vec![
                InvertedField {
                    field_number: f.id,
                    terms: vec![term(&format!("d{i}"), &[])].into(),
                    norm: None,
                },
                InvertedField {
                    field_number: f.body,
                    terms: body.into(),
                    norm: Some(length),
                },
            ],
            ..ExplicitFields::default()
        },
    }
}

fn id(i: i64) -> Term {
    Term::new("_id", format!("d{i}").into_bytes())
}

fn soft_delete(i: i64) -> DocValuesUpdate {
    DocValuesUpdate::Numeric {
        term: id(i),
        field: "__soft_deletes".to_string(),
        value: Some(1),
    }
}

fn body(positions: bool, terms: &[&str]) -> LiveRequest {
    LiveRequest {
        fields: vec![LiveFieldRequest {
            field: "body".to_string(),
            terms: terms.iter().map(|t| t.as_bytes().to_vec()).collect(),
            positions,
            patterns: Vec::new(),
        }],
    }
}

/// Every buffered document's id, by buffer.
fn deleted_ids(snap: &LiveSnapshot) -> Vec<Vec<i32>> {
    snap.buffers.iter().map(|b| b.deleted.clone()).collect()
}

/// Buffered documents are read with the whole buffer's statistics, their
/// norms, and their occurrences only when asked for.
#[test]
fn buffered_documents_are_read_with_their_statistics() {
    let tmp = TempDir::new("live-read");
    let dir = FsDirectory::open(tmp.path());
    let (w, f) = open(&dir, 1, true);
    for i in 0..30 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
    }
    let snap = w
        .live_snapshot(&body(false, &["rare", "absent", "rare"]), None)
        .unwrap();
    assert_eq!(snap.published, 0);
    assert_eq!(snap.buffers.len(), 1);
    let b = &snap.buffers[0];
    assert_eq!(b.max_doc, 30);
    assert!(b.deleted.is_empty());
    let field = &b.fields[0];
    assert_eq!(field.info.name, "body");
    assert_eq!(field.info.index_options, IndexOptions::DocsAndFreqs);
    assert_eq!(
        field.terms.len(),
        1,
        "a repeated term once, an absent one not"
    );
    let rare = &field.terms[0];
    let want: Vec<(i32, i32)> = (0..30).step_by(3).map(|d| (d, 2)).collect();
    assert_eq!(rare.docs, want);
    assert!(rare.positions.is_empty());
    assert_eq!(
        (
            field.num_terms,
            field.sum_doc_freq,
            field.sum_total_term_freq,
            field.doc_count
        ),
        (2, 40, 50, 30)
    );
    let norms = field.norms.as_ref().unwrap();
    assert_eq!(norms.len(), 30);
    assert!(norms
        .iter()
        .enumerate()
        .all(|(d, &n)| n == if d % 3 == 0 { 3 } else { 1 }));

    let snap = w.live_snapshot(&body(true, &["rare"]), None).unwrap();
    let field = &snap.buffers[0].fields[0];
    assert_eq!(
        field.info.index_options,
        IndexOptions::DocsAndFreqsAndPositions
    );
    assert_eq!(field.terms[0].positions, vec![vec![1, 2]; 10]);

    let ids = LiveRequest {
        fields: vec![LiveFieldRequest {
            field: "_id".to_string(),
            terms: vec![b"d4".to_vec()],
            positions: true,
            patterns: Vec::new(),
        }],
    };
    let snap = w.live_snapshot(&ids, None).unwrap();
    let field = &snap.buffers[0].fields[0];
    assert_eq!(field.terms[0].docs, [(4, 1)]);
    assert!(field.norms.is_none(), "the field omits norms");
    assert_eq!(field.sum_total_term_freq, field.sum_doc_freq);
}

/// A soft update hides the copy it replaces, in its own buffer or in
/// another; a document added soft-deleted (a tombstone) and a delete by
/// term hide theirs. Each document is in the cut once.
#[test]
fn updates_tombstones_and_deletes_hide_buffered_documents() {
    let tmp = TempDir::new("live-deletes");
    let dir = FsDirectory::open(tmp.path());
    let (w, f) = open(&dir, 2, true);
    w.rotate_slots.store(true, Ordering::Relaxed);
    // Slot 0 takes the even adds, slot 1 the odd ones.
    for i in 0..10 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
    }
    // d2 is in slot 0 (doc 1); its update lands in slot 0 (add 10).
    w.soft_update_explicit_documents(id(2), vec![doc(&f, 2)], &[soft_delete(2)])
        .unwrap();
    // d4 is in slot 0 (doc 2); its update lands in slot 1 (add 11), and
    // slot 0 has not applied it yet.
    w.soft_update_explicit_documents(id(4), vec![doc(&f, 4)], &[soft_delete(4)])
        .unwrap();
    // d5 is in slot 1 (doc 2).
    w.delete_documents_by_term(&[id(5)]).unwrap();
    // A tombstone, in slot 0 (add 12).
    let mut tombstone = doc(&f, 100);
    tombstone.fields.doc_values.push(StoredField {
        field_number: f.soft,
        value: FieldValue::Long(1),
    });
    w.add_explicit_documents(vec![tombstone]).unwrap();

    let snap = w.live_snapshot(&body(false, &["common"]), None).unwrap();
    let max_docs: Vec<i32> = snap.buffers.iter().map(|b| b.max_doc).collect();
    assert_eq!(max_docs, [7, 6]);
    // Slot 0: d0 d2 d4 d6 d8 d2' tombstone; slot 1: d1 d3 d5 d7 d9 d4'.
    assert_eq!(deleted_ids(&snap), [vec![1, 2, 6], vec![2]]);

    // Once each slot adds again it has applied every delete itself.
    w.add_explicit_documents(vec![doc(&f, 200)]).unwrap();
    w.add_explicit_documents(vec![doc(&f, 201)]).unwrap();
    let snap = w.live_snapshot(&body(false, &["common"]), None).unwrap();
    assert_eq!(deleted_ids(&snap), [vec![1, 2, 6], vec![2]]);
}

/// A buffer taken for a flush is read until its segment is published,
/// and a delete issued meanwhile reaches it; then it is gone from the cut
/// unless the reader's commit lacks it, until a commit holding it is
/// released.
#[test]
fn a_buffer_being_flushed_stays_readable_until_published() {
    let tmp = TempDir::new("live-flushing");
    let dir = FsDirectory::open(tmp.path());
    let (w, f) = open(&dir, 1, true);
    let start = w.with_writer(|iw| iw.segment_infos().generation);
    for i in 0..5 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
    }
    let batch = {
        let mut slot = super::write_slot(&w.slots[0]);
        w.begin_flush(0, &mut slot).unwrap()
    };
    w.add_explicit_documents(vec![doc(&f, 5)]).unwrap();
    w.delete_documents_by_term(&[id(1)]).unwrap();
    let snap = w.live_snapshot(&body(false, &["common"]), None).unwrap();
    let max_docs: Vec<i32> = snap.buffers.iter().map(|b| b.max_doc).collect();
    assert_eq!(max_docs, [1, 5], "the slot, then the buffer being flushed");
    assert_eq!(deleted_ids(&snap), [vec![], vec![1]]);
    assert_eq!(snap.buffers[1].fields[0].terms[0].docs.len(), 5);

    // Another add leaves the cut's deletes as they were.
    w.add_explicit_documents(vec![doc(&f, 6)]).unwrap();
    let snap = w.live_snapshot(&body(false, &["common"]), None).unwrap();
    assert_eq!(deleted_ids(&snap), [vec![], vec![1]]);

    w.complete_flush(batch).unwrap();
    let snap = w.live_snapshot(&body(false, &["common"]), None).unwrap();
    assert_eq!(snap.published, 1);
    let max_docs: Vec<i32> = snap.buffers.iter().map(|b| b.max_doc).collect();
    assert_eq!(max_docs, [2]);

    // A reader of the commit before it still lacks the published buffer,
    // and reads it, with the deletes that reach it after publishing.
    w.delete_documents_by_term(&[id(3)]).unwrap();
    let snap = w
        .live_snapshot(&body(false, &["common"]), Some(start))
        .unwrap();
    let max_docs: Vec<i32> = snap.buffers.iter().map(|b| b.max_doc).collect();
    assert_eq!(max_docs, [2, 5], "the slot, then the published buffer");
    assert_eq!(deleted_ids(&snap), [vec![], vec![1, 3]]);

    // A commit holds every buffer; once released, an older reader is
    // refused rather than answered without them.
    w.commit().unwrap();
    let generation = w.with_writer(|iw| iw.segment_infos().generation);
    let snap = w
        .live_snapshot(&body(false, &["common"]), Some(generation))
        .unwrap();
    assert!(snap.buffers.is_empty());
    assert!(w
        .live_snapshot(&body(false, &["common"]), Some(start))
        .is_ok());
    w.release_live(generation).unwrap();
    assert!(matches!(
        w.live_snapshot(&body(false, &["common"]), Some(start)),
        Err(Error::LiveSearch(_))
    ));
    assert!(matches!(
        w.live_snapshot(&body(false, &["common"]), Some(generation + 7)),
        Err(Error::LiveSearch(_))
    ));
}

/// Flushes slot 0 and publishes it.
fn flush(w: &ConcurrentIndexWriter<'_>) {
    let batch = {
        let mut slot = super::write_slot(&w.slots[0]);
        w.begin_flush(0, &mut slot).unwrap()
    };
    w.complete_flush(batch).unwrap();
}

/// A release keeps a buffer only while a view reads it, a reader whose
/// buffers are not all kept is refused, and a rollback drops every buffer.
#[test]
fn released_buffers_go_when_their_last_view_closes() {
    let tmp = TempDir::new("live-release");
    let dir = FsDirectory::open(tmp.path());
    let (w, f) = open(&dir, 1, true);
    let generation = || w.with_writer(|iw| iw.segment_infos().generation);
    let req = body(false, &["common"]);
    let mut gens = Vec::new();
    for i in 0..2 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
        flush(&w);
        w.commit().unwrap();
        gens.push(generation());
    }
    // Reads the second publish only.
    let view = w.open_live_view(Some(gens[0])).unwrap();
    w.add_explicit_documents(vec![doc(&f, 2)]).unwrap();
    flush(&w);
    w.commit().unwrap();
    gens.push(generation());
    w.release_live(gens[2]).unwrap();
    // The second publish is kept for the view, the third is not: a reader
    // of the first commit would miss documents.
    assert!(matches!(
        w.live_snapshot(&req, Some(gens[0])),
        Err(Error::LiveSearch(_))
    ));
    assert_eq!(w.live_view_snapshot(&view, &req).unwrap().buffers.len(), 1);
    assert_eq!(w.live_flushes().retained.len(), 1);
    w.close_live_view(&view);
    assert!(w.live_flushes().retained.is_empty());

    w.add_explicit_documents(vec![doc(&f, 3)]).unwrap();
    flush(&w);
    let view = w.open_live_view(Some(gens[2])).unwrap();
    assert_eq!(view.buffers.len(), 1);
    w.rollback();
    assert!(w.live_flushes().retained.is_empty());
    // Back at that commit: its reader holds every document.
    assert!(w
        .live_snapshot(&req, Some(gens[2]))
        .unwrap()
        .buffers
        .is_empty());
    assert_eq!(
        deleted_ids(&w.live_view_snapshot(&view, &req).unwrap()),
        [vec![0]]
    );
    w.close_live_view(&view);
}

/// A view of a commit names the deletes its reader lacks, those issued
/// after the commit whether handed to the segments yet or not; a delete by
/// query refuses it, and a release drops the deletes no reader needs.
#[test]
fn a_view_of_a_commit_carries_the_deletes_issued_since() {
    let tmp = TempDir::new("live-commit-deletes");
    let dir = FsDirectory::open(tmp.path());
    let (w, f) = open(&dir, 1, true);
    let generation = || w.with_writer(|iw| iw.segment_infos().generation);
    let deletes = |g: i64| {
        let view = w.open_live_view(Some(g)).unwrap();
        w.close_live_view(&view);
        let ids: Vec<String> = view
            .commit_deletes
            .iter()
            .map(|t| {
                assert_eq!(t.field, "_id");
                String::from_utf8(t.bytes.clone()).unwrap()
            })
            .collect();
        ids
    };
    for i in 0..3 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
    }
    // Before the commit: its segments have it applied.
    w.delete_documents_by_term(&[id(0)]).unwrap();
    w.commit().unwrap();
    let g0 = generation();
    assert!(deletes(g0).is_empty());

    w.soft_update_explicit_documents(id(1), vec![doc(&f, 1)], &[soft_delete(1)])
        .unwrap();
    w.delete_documents_by_term(&[id(2)]).unwrap();
    assert_eq!(deletes(g0), ["d1", "d2"]);
    // Handed to the published segments, they are still the reader's.
    flush(&w);
    assert_eq!(deletes(g0), ["d1", "d2"]);
    assert!(deletes_none(&w));

    w.commit().unwrap();
    let g1 = generation();
    w.soft_update_explicit_documents(id(1), vec![doc(&f, 1)], &[soft_delete(1)])
        .unwrap();
    assert_eq!(deletes(g1), ["d1"]);
    assert_eq!(deletes(g0), ["d1", "d2", "d1"]);

    // No reader of the first commit is left.
    w.release_live(g1).unwrap();
    assert!(w.open_live_view(Some(g0)).is_err());
    assert_eq!(deletes(g1), ["d1"]);
    let history = w.live_flushes().history.len();
    assert_eq!(history, 0, "the deletes before the second commit went");
    assert_eq!(w.live_flushes().history_bytes, 0);

    w.delete_documents_by_query(&[DeleteQuery::Term(id(1))])
        .unwrap();
    assert!(matches!(
        w.open_live_view(Some(g1)),
        Err(Error::LiveSearch(_))
    ));
    assert!(
        w.live_flushes().pins.is_empty(),
        "a refused view keeps nothing"
    );
    // A rollback goes back to the last commit, which can be viewed again.
    w.rollback();
    assert!(deletes(g1).is_empty());
}

/// A view of the buffers alone names no deletes: no reader needs them.
fn deletes_none(w: &ConcurrentIndexWriter<'_>) -> bool {
    let view = w.open_live_view(None).unwrap();
    w.close_live_view(&view);
    view.commit_deletes.is_empty()
}

/// A big document's stored fields are read from the compressed chunk the
/// buffer wrote, through its flush and publish, not from a second copy.
#[test]
fn a_big_document_is_read_from_its_compressed_chunk() {
    let tmp = TempDir::new("live-big");
    let dir = FsDirectory::open(tmp.path());
    let (w, f) = open(&dir, 1, true);
    // Compressible, as `_source` is: a small part of it stays in memory.
    let source: String = (0..400_000)
        .map(|i| format!("{{\"k{}\":{i}}}", i % 50))
        .collect();
    for i in 0..5 {
        let mut d = doc(&f, i);
        if i == 2 {
            d.stored[0].value = FieldValue::String(source.clone());
        }
        w.add_explicit_documents(vec![d]).unwrap();
    }
    let view = w.open_live_view(None).unwrap();
    let read = |doc| w.live_document(&view, 0, doc).unwrap();
    assert_eq!(read(2)[0].value, FieldValue::String(source.clone()));
    assert_eq!(read(1)[0].value, FieldValue::String("d1".into()));
    assert_eq!(read(4)[1].value, FieldValue::Long(4));
    assert!(
        w.live_bytes() < source.len() / 2,
        "{} live bytes for a {} byte document",
        w.live_bytes(),
        source.len()
    );
    // Flushed and published, the view still reads it.
    w.commit().unwrap();
    assert_eq!(read(2)[0].value, FieldValue::String(source));
    assert_eq!(read(3)[0].value, FieldValue::String("d3".into()));
    w.close_live_view(&view);
}

/// A view holds its buffers' documents and deletes as of its opening, and
/// its stored fields, through a flush, a publish and a release, until it is
/// closed.
#[test]
fn a_view_reads_its_buffers_as_of_its_opening() {
    let tmp = TempDir::new("live-view");
    let dir = FsDirectory::open(tmp.path());
    let (w, f) = open(&dir, 1, true);
    for i in 0..9 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
    }
    w.delete_documents_by_term(&[id(4)]).unwrap();
    let view = w.open_live_view(None).unwrap();
    assert_eq!(view.buffers.len(), 1);
    assert_eq!(view.buffers[0].deleted, [4]);

    // Later documents and deletes are not in it.
    for i in 9..20 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
    }
    w.delete_documents_by_term(&[id(5)]).unwrap();
    let snap = w
        .live_view_snapshot(&view, &body(true, &["rare", "common"]))
        .unwrap();
    let b = &snap.buffers[0];
    assert_eq!((b.max_doc, b.deleted.clone()), (9, vec![4]));
    let field = &b.fields[0];
    assert_eq!(field.terms[0].docs.len(), 9, "common");
    assert_eq!(field.terms[1].docs, [(0, 2), (3, 2), (6, 2)], "rare");
    assert_eq!(field.terms[1].positions.len(), 3);
    assert_eq!(field.norms.as_ref().unwrap().len(), 9);
    assert_eq!(field.doc_count, 9, "statistics as of the view");

    let stored = w.live_document(&view, 0, 7).unwrap();
    assert_eq!(stored[0].value, FieldValue::String("d7".into()));
    assert_eq!(stored[1].value, FieldValue::Long(7));
    assert!(w.live_document(&view, 0, 9).is_err(), "past the view");
    assert!(w.live_document(&view, 1, 0).is_err());

    // Committed and released, the buffer stays while the view is open.
    let start = w.with_writer(|iw| iw.segment_infos().generation);
    let old = w.open_live_view(Some(start)).unwrap();
    w.commit().unwrap();
    let generation = w.with_writer(|iw| iw.segment_infos().generation);
    w.release_live(generation).unwrap();
    let snap = w
        .live_view_snapshot(&view, &body(false, &["common"]))
        .unwrap();
    assert_eq!(snap.buffers[0].fields[0].terms[0].docs.len(), 9);
    assert_eq!(
        w.live_document(&old, 0, 19).unwrap()[0].value,
        FieldValue::String("d19".into())
    );
    w.close_live_view(&view);
    w.close_live_view(&old);
    w.release_live(generation).unwrap();
    assert!(w.live_document(&view, 0, 7).is_err(), "released");
    let snap = w
        .live_view_snapshot(&view, &body(false, &["common"]))
        .unwrap();
    assert_eq!(
        snap.buffers[0].deleted.len(),
        9,
        "a dropped buffer reads as deleted"
    );
    assert!(w.live_bytes() < 1024);
}

/// Truncating a term's postings at a view's last document keeps its
/// occurrences and payloads in step.
#[test]
fn truncated_postings_keep_occurrences_and_payloads_in_step() {
    let mut p = TermPostings {
        term: b"t".to_vec(),
        docs: vec![(0, 2), (3, 1), (5, 1)],
        positions: vec![vec![1, 2], vec![4], vec![6]],
        offsets: Vec::new(),
        payload_bytes: b"abcdef".to_vec(),
        payload_lengths: vec![1, 2, 0, 3],
    };
    truncate(&mut p, 4);
    assert_eq!(p.docs, [(0, 2), (3, 1)]);
    assert_eq!(p.positions, [vec![1, 2], vec![4]]);
    assert_eq!(p.payload_lengths, [1, 2, 0]);
    assert_eq!(p.payload_bytes, b"abc");
}

/// A writer that keeps no live-search columns, and a buffer a delete by
/// query reached, are refused rather than answered wrongly.
#[test]
fn what_the_buffers_cannot_answer_is_refused() {
    let tmp = TempDir::new("live-refused");
    let dir = FsDirectory::open(tmp.path());
    let (w, f) = open(&dir, 1, false);
    w.add_explicit_documents(vec![doc(&f, 0)]).unwrap();
    assert!(!w.live_search());
    assert!(
        matches!(w.enable_live_search(), Err(Error::LiveSearch(_))),
        "too late once a document is buffered"
    );
    assert!(matches!(
        w.live_snapshot(&body(false, &["common"]), None),
        Err(Error::LiveSearch(_))
    ));

    let tmp = TempDir::new("live-by-query");
    let dir = FsDirectory::open(tmp.path());
    let (w, f) = open(&dir, 1, true);
    w.add_explicit_documents(vec![doc(&f, 0)]).unwrap();
    assert!(w.live_bytes() > 0);
    w.delete_documents_by_query(&[DeleteQuery::Term(id(0))])
        .unwrap();
    assert!(matches!(
        w.live_snapshot(&body(false, &["common"]), None),
        Err(Error::LiveSearch(_))
    ));
}

/// What live search costs (`--ignored --nocapture`, release): adds with
/// and without its columns, and a snapshot of a frequent and a rare term.
#[test]
#[ignore]
fn live_search_cost() {
    const N: i64 = 200_000;
    for live in [false, true] {
        let tmp = TempDir::new("live-cost");
        let dir = FsDirectory::open(tmp.path());
        let (w, f) = open_with(&dir, 1, live, 4096.0);
        let docs: Vec<ExplicitDocument> = (0..N).map(|i| doc(&f, i)).collect();
        let t0 = std::time::Instant::now();
        for d in docs {
            w.add_explicit_documents(vec![d]).unwrap();
        }
        let add_ns = t0.elapsed().as_nanos() as f64 / N as f64;
        println!(
            "live={live} add_ns_per_doc={add_ns:.0} live_bytes={}",
            w.live_bytes()
        );
        if !live {
            continue;
        }
        for (t, positions) in [("common", false), ("rare", false), ("rare", true)] {
            let req = body(positions, &[t]);
            let rounds = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..rounds {
                std::hint::black_box(w.live_snapshot(&req, None).unwrap());
            }
            let us = t0.elapsed().as_micros() as f64 / f64::from(rounds);
            println!("snapshot term={t} positions={positions} us={us:.0}");
        }
    }
}

/// A live writer of one slot holding 400 documents with norms, doc values
/// and points, all kept as columns.
fn columns_writer(dir: &FsDirectory) -> ConcurrentIndexWriter<'_> {
    columns_writer_with(dir, true)
}

/// [`columns_writer`]'s documents, live search on or off.
fn columns_writer_with(dir: &FsDirectory, live: bool) -> ConcurrentIndexWriter<'_> {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    let body = w
        .register_field(FieldInfo::new("body", 0).with_index_options(IndexOptions::DocsAndFreqs))
        .unwrap();
    let kw = w
        .register_field(
            FieldInfo::new("kw", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true)
                .with_doc_values(DocValuesType::SortedSet, DocValuesSkipIndexType::None, -1),
        )
        .unwrap();
    let price = w
        .register_field(
            FieldInfo::new("price", 0)
                .with_doc_values(
                    DocValuesType::SortedNumeric,
                    DocValuesSkipIndexType::None,
                    -1,
                )
                .with_points(1, 1, 8),
        )
        .unwrap();
    let loc = w
        .register_field(FieldInfo::new("loc", 0).with_points(2, 2, 4))
        .unwrap();
    let w = ConcurrentIndexWriter::new(w, 1).unwrap();
    if live {
        w.enable_live_search().unwrap();
    }
    for i in 0..400i64 {
        let mut fields = ExplicitFields {
            inverted: vec![InvertedField {
                field_number: body,
                terms: vec![term("w", &[])].into(),
                norm: Some(1 + i % 7),
            }],
            ..ExplicitFields::default()
        };
        if i % 4 != 0 {
            let tag = format!("t{}", i % 9).into_bytes();
            fields.inverted.push(InvertedField {
                field_number: kw,
                terms: vec![InvertedTerm {
                    term: tag.clone(),
                    ..term("", &[])
                }]
                .into(),
                norm: None,
            });
            fields.doc_values.push(StoredField {
                field_number: kw,
                value: FieldValue::Binary(tag),
            });
        }
        if i % 3 != 2 {
            fields.doc_values.push(StoredField {
                field_number: price,
                value: FieldValue::Long(1000 - i),
            });
            fields.points.push(StoredField {
                field_number: price,
                value: FieldValue::Binary(
                    (((1000 - i) as u64) ^ 0x8000_0000_0000_0000)
                        .to_be_bytes()
                        .to_vec(),
                ),
            });
        }
        if i % 5 == 0 {
            let mut packed = ((i as u32) ^ 0x8000_0000).to_be_bytes().to_vec();
            packed.extend_from_slice(&((7 * i as u32) ^ 0x8000_0000).to_be_bytes());
            fields.points.push(StoredField {
                field_number: loc,
                value: FieldValue::Binary(packed),
            });
        }
        w.add_explicit_documents(vec![ExplicitDocument {
            stored: Vec::new(),
            fields,
        }])
        .unwrap();
    }
    w
}

/// The files of segment `name` in `dir`, by extension, their segment ids
/// zeroed and their footers cut.
fn segment_files(
    dir: &std::path::Path,
    sci: &crate::segment_infos::SegmentCommitInfo,
    name: &str,
) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|f| f.starts_with(&format!("{name}.")) || f.starts_with(&format!("{name}_")))
        .filter(|f| !f.ends_with(".si"))
        .map(|f| {
            let mut bytes = std::fs::read(dir.join(&f)).unwrap();
            // Headers carry the segment's random id, footers a checksum.
            let n = bytes.len().saturating_sub(8);
            bytes.truncate(n);
            for at in 0..bytes.len().saturating_sub(15) {
                if bytes[at..at + 16] == sci.segment_id {
                    bytes[at..at + 16].fill(0);
                }
            }
            (f[name.len()..].to_string(), bytes)
        })
        .collect();
    files.sort();
    files
}

/// A flushed segment is byte for byte the same with live search on as off:
/// what live search keeps beside a buffer (sealed postings blocks, columns,
/// bounds) changes nothing the flush writes.
#[test]
fn live_search_leaves_flushed_segments_unchanged() {
    let flushed = |live: bool, with_positions: bool| {
        let tmp = TempDir::new("live-flush-same");
        let dir = FsDirectory::open(tmp.path());
        {
            if with_positions {
                let (w, f) = open(&dir, 1, live);
                for i in 0..700 {
                    w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
                }
                w.commit().unwrap();
            } else {
                let w = columns_writer_with(&dir, live);
                w.commit().unwrap();
            }
        }
        let infos = crate::segment_infos::read_latest(&dir).unwrap();
        assert_eq!(infos.segments.len(), 1);
        let sci = &infos.segments[0];
        let files = segment_files(tmp.path(), sci, &sci.segment_name);
        assert!(files.iter().any(|(n, _)| n.ends_with(".doc")), "{files:?}");
        files
    };
    for with_positions in [false, true] {
        let (on, off) = (
            flushed(true, with_positions),
            flushed(false, with_positions),
        );
        let names: Vec<&str> = on.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            off.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
        );
        for ((name, a), (_, b)) in on.iter().zip(&off) {
            assert!(a == b, "{name} differs with live search on");
        }
    }
}

/// A chunk written from the buffer's columns holds the files the decoded
/// documents give, for norms, doc values, points and field infos alike.
#[test]
fn chunks_from_columns_match_chunks_from_documents() {
    let tmp = TempDir::new("live-chunk-columns");
    let dir = FsDirectory::open(tmp.path());
    let w = columns_writer(&dir);
    let id = w.live_slot_buffers()[0].id;
    assert!(
        read_slot(&w.slots[0]).live.chunk_columns(37, 301).is_some(),
        "the run is kept as columns"
    );
    let chunk = |name: &str, decode: bool| {
        DECODE_CHUNKS.with(|c| c.set(decode));
        let sci = w
            .live_buffer_segment(id, 37..301, HashMap::new(), &[], false, &dir, name)
            .unwrap();
        DECODE_CHUNKS.with(|c| c.set(false));
        sci
    };
    let read = |sci: &crate::segment_infos::SegmentCommitInfo, name: &str| {
        segment_files(tmp.path(), sci, name)
    };
    let from_columns = chunk("_c1", false);
    let decoded = chunk("_c2", true);
    let (a, b) = (read(&from_columns, "_c1"), read(&decoded, "_c2"));
    let names: Vec<&str> = a.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.iter().any(|n| n.ends_with(".kdd")), "{names:?}");
    assert!(names.iter().any(|n| n.ends_with(".dvd")), "{names:?}");
    assert!(names.iter().any(|n| n.ends_with(".nvd")), "{names:?}");
    assert_eq!(a, b);
}

/// A buffer taken for a flush is written as chunks without postings as it
/// was in its slot, until it is published and the views that read it as
/// chunks are closed.
#[test]
fn a_buffer_holding_blocks_is_written_not_built_in_memory() {
    let tmp = TempDir::new("live-lean-blocks");
    let dir = FsDirectory::open(tmp.path());
    let w = columns_writer(&dir);
    let block = |n: usize| {
        (0..n)
            .map(|_| ExplicitDocument {
                stored: Vec::new(),
                fields: ExplicitFields::default(),
            })
            .collect::<Vec<_>>()
    };
    w.add_explicit_documents(block(3)).unwrap();
    let id = w.live_slot_buffers()[0].id;
    let lean = w
        .live_lean_segment(id, 37..301, HashMap::new(), &[], true, &dir, "_b")
        .unwrap();
    assert!(
        matches!(lean, super::LeanSegment::Written(..)),
        "a segment's block flag comes from its segment info"
    );
}

#[test]
fn a_flushing_buffer_is_written_as_lean_chunks_until_published() {
    let tmp = TempDir::new("live-flushing-lean");
    let dir = FsDirectory::open(tmp.path());
    let w = columns_writer(&dir);
    let id = w.live_slot_buffers()[0].id;
    let lean = |name: &str| w.live_lean_segment(id, 37..301, HashMap::new(), &[], true, &dir, name);
    let parts = |s: super::LeanSegment| match s {
        super::LeanSegment::Parts(p, dv) => (*p, dv),
        super::LeanSegment::Written(..) => panic!("a run kept as columns is built in memory"),
    };
    let (in_slot, slot_dv) = parts(lean("_s").unwrap());
    let slot_view = w.open_live_view(None).unwrap();
    let batch = {
        let mut slot = super::write_slot(&w.slots[0]);
        w.begin_flush(0, &mut slot).unwrap()
    };
    let view = w.open_live_view(None).unwrap();
    assert!(view
        .buffers
        .iter()
        .any(|b| b.id == id && !b.in_slot && b.chunks));
    let (flushing, flushing_dv) = parts(lean("_f").unwrap());
    let named = |p: &crate::index_writer::LeanParts| {
        p.field_infos
            .iter()
            .map(|f| (f.number, f.name.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(named(&in_slot), named(&flushing));
    assert_eq!((in_slot.max_doc, flushing.max_doc), (264, 264));
    let unframed = |b: &Option<(Vec<u8>, Vec<u8>)>| b.as_ref().map(|(m, d)| (m.len(), d.len()));
    assert!(in_slot.norms.is_some() && in_slot.points.is_some());
    assert_eq!(unframed(&in_slot.norms), unframed(&flushing.norms));
    assert_eq!(
        in_slot.points.as_ref().map(|p| p.2.len()),
        flushing.points.as_ref().map(|p| p.2.len())
    );
    let fields: Vec<i32> = slot_dv.fields().collect();
    assert!(!fields.is_empty());
    assert_eq!(fields, flushing_dv.fields().collect::<Vec<_>>());
    for f in fields {
        let (a, b) = (
            slot_dv.encode(f, &[0; 16]).unwrap().unwrap(),
            flushing_dv.encode(f, &[0; 16]).unwrap().unwrap(),
        );
        assert_eq!((a.dvm, a.dvd), (b.dvm, b.dvd), "field {f}");
    }

    // Published, it is still written so while a view reading it as chunks
    // is open: one that saw it flushing, or in its slot.
    w.complete_flush(batch).unwrap();
    assert!(w.live_published_segment(id).is_some());
    lean("_p").unwrap();
    let later = w.open_live_view(None).unwrap();
    assert!(later.buffers.iter().all(|b| b.id != id || !b.chunks));
    w.close_live_view(&view);
    lean("_q").unwrap();
    w.close_live_view(&slot_view);
    assert!(
        lean("_r").is_err(),
        "read as its published segment from now on"
    );
    w.close_live_view(&later);
}

/// A multi-valued field's values come sorted (and, for SORTED_SET,
/// without repeats); a single-valued field gives its first value.
#[test]
fn a_documents_doc_values_take_the_segments_shape() {
    use super::LiveDocValue::{self, Bytes, Numbers};
    let longs = [
        FieldValue::Long(3),
        FieldValue::Long(-1),
        FieldValue::Long(3),
    ];
    let bytes = [
        FieldValue::Binary(b"b".to_vec()),
        FieldValue::Binary(b"a".to_vec()),
        FieldValue::Binary(b"b".to_vec()),
    ];
    let of = |kind, values: &[FieldValue]| LiveDocValue::of(kind, values.iter());
    assert_eq!(
        of(DocValuesType::SortedNumeric, &longs),
        Some(Numbers(vec![-1, 3, 3]))
    );
    assert_eq!(of(DocValuesType::Numeric, &longs), Some(Numbers(vec![3])));
    assert_eq!(
        of(DocValuesType::SortedSet, &bytes),
        Some(Bytes(vec![b"a".to_vec(), b"b".to_vec()]))
    );
    assert_eq!(
        of(DocValuesType::Sorted, &bytes),
        Some(Bytes(vec![b"b".to_vec()]))
    );
    assert_eq!(of(DocValuesType::SortedSet, &longs), None);
    assert_eq!(of(DocValuesType::None, &bytes), None);
}

/// One buffered document's doc values read the same in its slot, while
/// flushing and once published, as long as a view holds the buffer.
#[test]
fn a_buffered_documents_doc_values_read_in_every_state() {
    use super::LiveDocValue::{Bytes, Numbers};
    let tmp = TempDir::new("live-doc-values");
    let dir = FsDirectory::open(tmp.path());
    let w = columns_writer(&dir);
    let cfg = w.cfg();
    let number = |name: &str| cfg.field_by_name(name).unwrap().number;
    let (kw, price, body) = (number("kw"), number("price"), number("body"));
    let fields = [kw, price, body];
    let check = |view: &LiveView| {
        let b = view.buffers.iter().position(|b| b.max_doc == 400).unwrap();
        let read = |doc| w.live_doc_values(view, b, doc, &fields).unwrap();
        assert_eq!(
            read(7),
            vec![
                (kw, Bytes(vec![b"t7".to_vec()])),
                (price, Numbers(vec![993]))
            ]
        );
        assert_eq!(read(4), vec![(price, Numbers(vec![996]))]);
        assert_eq!(read(11), vec![(kw, Bytes(vec![b"t2".to_vec()]))]);
        assert!(w.live_doc_values(view, b, 400, &fields).is_err());
    };
    let slot_view = w.open_live_view(None).unwrap();
    check(&slot_view);
    let batch = {
        let mut slot = super::write_slot(&w.slots[0]);
        w.begin_flush(0, &mut slot).unwrap()
    };
    let view = w.open_live_view(None).unwrap();
    check(&view);
    check(&slot_view);
    w.complete_flush(batch).unwrap();
    check(&view);
    w.close_live_view(&view);
    w.close_live_view(&slot_view);
}

/// A buffer's points' bounds take in each of its documents' points, per
/// indexed dimension, whether it is in its slot, flushing or published.
#[test]
fn a_buffers_point_bounds_read_in_every_state() {
    let tmp = TempDir::new("live-point-bounds");
    let dir = FsDirectory::open(tmp.path());
    let w = columns_writer(&dir);
    let cfg = w.cfg();
    let number = |name: &str| cfg.field_by_name(name).unwrap().number;
    let (price, loc, body) = (number("price"), number("loc"), number("body"));
    let long = |v: u64| (v ^ 0x8000_0000_0000_0000).to_be_bytes().to_vec();
    let int = |v: u32| (v ^ 0x8000_0000).to_be_bytes().to_vec();
    let check = |view: &LiveView| {
        let b = view.buffers.iter().position(|b| b.max_doc == 400).unwrap();
        let p = w.live_point_bounds(view, b, price).unwrap().unwrap();
        assert_eq!((p.min, p.max, p.count), (long(601), long(1000), 267));
        let l = w.live_point_bounds(view, b, loc).unwrap().unwrap();
        assert_eq!(l.min, [int(0), int(0)].concat());
        assert_eq!(l.max, [int(395), int(7 * 395)].concat());
        assert_eq!(l.count, 80);
        assert_eq!(w.live_point_bounds(view, b, body).unwrap(), None);
        assert!(w
            .live_point_bounds(view, view.buffers.len(), price)
            .is_err());
    };
    let slot_view = w.open_live_view(None).unwrap();
    check(&slot_view);
    let batch = {
        let mut slot = super::write_slot(&w.slots[0]);
        w.begin_flush(0, &mut slot).unwrap()
    };
    let view = w.open_live_view(None).unwrap();
    check(&view);
    w.complete_flush(batch).unwrap();
    check(&view);
    check(&slot_view);
    w.close_live_view(&view);
    w.close_live_view(&slot_view);
}

/// A published buffer a view reads keeps its segment's files after a
/// merge replaced the segment and a commit dropped it, until the view is
/// closed and a release lets the buffer go.
#[test]
fn a_published_buffer_keeps_its_files_while_a_view_reads_it() {
    let tmp = TempDir::new("live-published-files");
    let dir = FsDirectory::open(tmp.path());
    let (w, f) = open(&dir, 1, true);
    let start = w.with_writer(|iw| iw.segment_infos().generation);
    let mut ids = Vec::new();
    for i in 0..2 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
        ids.push(w.live_slot_buffers()[0].id);
        w.commit().unwrap();
    }
    let view = w.open_live_view(Some(start)).unwrap();
    assert_eq!(view.buffers.len(), 2);
    assert!(view.buffers.iter().all(|b| !b.chunks));
    let files = |id| {
        let name = w.live_published_segment(id).unwrap().segment_name;
        std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|f| f.starts_with(&format!("{name}.")) || f.starts_with(&format!("{name}_")))
            .collect::<Vec<_>>()
    };
    let held: Vec<String> = ids.iter().flat_map(|&id| files(id)).collect();
    assert!(!held.is_empty());
    w.with_writer(|iw| iw.force_merge(1)).unwrap();
    assert_eq!(w.with_writer(|iw| iw.segment_infos().segments.len()), 1);
    w.commit().unwrap();
    let generation = w.with_writer(|iw| iw.segment_infos().generation);
    w.release_live(generation).unwrap();
    for name in &held {
        assert!(
            tmp.path().join(name).exists(),
            "{name} is pinned by the view"
        );
    }
    w.close_live_view(&view);
    assert!(w.live_flushes().retained.is_empty());
    for name in &held {
        assert!(!tmp.path().join(name).exists(), "{name} is unpinned");
    }
}

/// A writer of explicit documents with an `_id` and a 16-d dot-product
/// vector field `v`.
fn vector_writer(dir: &FsDirectory, live: bool) -> (ConcurrentIndexWriter<'_>, i32) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_ram_buffer_size_mb(64.0).unwrap();
    w.enable_explicit_documents().unwrap();
    let id = w
        .register_field(
            FieldInfo::new("_id", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
    w.register_field(FieldInfo::new("v", 0).with_vectors(
        16,
        lucene_codecs::field_infos::VectorEncoding::Float32,
        lucene_codecs::field_infos::VectorSimilarityFunction::DotProduct,
    ))
    .unwrap();
    let w = ConcurrentIndexWriter::new(w, 1).unwrap();
    if live {
        w.enable_live_search().unwrap();
    }
    (w, id)
}

/// Unit vector `i` of 16 dimensions.
fn unit(i: i64) -> Vec<f32> {
    let mut v: Vec<f32> = (0..16)
        .map(|d| ((i * 31 + d * 7) % 23) as f32 - 11.0)
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

/// Document `i`, with vector `i` unless `i` is a multiple of 5.
fn add_vector_doc(w: &ConcurrentIndexWriter<'_>, id: i32, i: i64) {
    let doc = ExplicitDocument {
        stored: Vec::new(),
        fields: ExplicitFields {
            inverted: vec![InvertedField {
                field_number: id,
                terms: vec![term(&format!("d{i}"), &[])].into(),
                norm: None,
            }],
            ..ExplicitFields::default()
        },
    };
    let vectors = if i % 5 == 0 {
        Vec::new()
    } else {
        vec![crate::index_writer::DocumentVector::float32("v", unit(i))]
    };
    w.add_explicit_documents_with_vectors_in(None, vec![doc], vec![vectors])
        .unwrap();
}

/// With live search on, the flush writes the graph kept over the buffer's
/// vectors, and the segment is byte for byte the one it writes with live
/// search off, which builds the graph at the flush.
#[test]
fn live_search_flushes_the_vector_graph_it_kept() {
    let flushed = |live: bool| {
        let tmp = TempDir::new("live-flush-graph");
        let dir = FsDirectory::open(tmp.path());
        {
            let (w, id) = vector_writer(&dir, live);
            for i in 0..2500 {
                add_vector_doc(&w, id, i);
            }
            w.commit().unwrap();
        }
        let infos = crate::segment_infos::read_latest(&dir).unwrap();
        assert_eq!(infos.segments.len(), 1);
        let sci = &infos.segments[0];
        let files = segment_files(tmp.path(), sci, &sci.segment_name);
        assert!(files.iter().any(|(n, _)| n.ends_with(".vex")), "{files:?}");
        files
    };
    let kept = || {
        crate::index_writer::KEPT_GRAPHS
            .lock()
            .unwrap()
            .iter()
            .filter(|&&n| n == 2000)
            .count()
    };
    let before = kept();
    let on = flushed(true);
    // Its 2000 vectors (no other test keeps a graph of that many).
    assert!(kept() > before, "the flush wrote the kept graph");
    let off = flushed(false);
    let names: Vec<&str> = on.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        off.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
    );
    for ((name, a), (_, b)) in on.iter().zip(&off) {
        assert!(a == b, "{name} differs with live search on");
    }
}

/// A view holds each buffer's vector graph with the vectors it saw, and a
/// search of it finds a buffered vector as its own best match.
#[test]
fn a_view_holds_the_buffer_vector_graph() {
    let tmp = TempDir::new("live-view-graph");
    let dir = FsDirectory::open(tmp.path());
    let (w, id) = vector_writer(&dir, true);
    for i in 0..1000 {
        add_vector_doc(&w, id, i);
    }
    let view = w.open_live_view(None).unwrap();
    for i in 1000..1200 {
        add_vector_doc(&w, id, i);
    }
    let b = &view.buffers[0];
    let g = &b.vector_graphs[0];
    assert_eq!(g.count, 800, "every document but each fifth has a vector");
    assert_eq!(g.graph.len(), 960);
    let query = lucene_codecs::buffer_vectors::BufferQuery::Float32(unit(7));
    let hits = g.graph.search(&query, 3, g.count, None).unwrap();
    assert_eq!(hits[0].0, 7);
    // The documents after the view are not in it.
    let late = lucene_codecs::buffer_vectors::BufferQuery::Float32(unit(1101));
    let hits = g.graph.search(&late, 1, g.count, None).unwrap();
    assert!(hits[0].0 < 1000, "{hits:?}");
    let graph = std::sync::Arc::clone(&g.graph);
    w.close_live_view(&view);
    drop(view);
    w.flush().unwrap();
    // The flushed buffer, kept for views, reads its vectors from the
    // segment: its graph, and the vectors in it, are gone.
    // (An inserter task may still be finishing with it.)
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::sync::Arc::strong_count(&graph) > 1 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(
        std::sync::Arc::strong_count(&graph),
        1,
        "the flush, a view or an inserter still holds the graph"
    );
    let view = w.open_live_view(None).unwrap();
    assert!(view.buffers.iter().all(|b| b.vector_graphs.is_empty()));
    w.close_live_view(&view);
    w.commit().unwrap();
}

/// A writer whose `v` field is 64-d floats in the IVF format.
fn ivf_writer(dir: &FsDirectory) -> (ConcurrentIndexWriter<'_>, i32) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_ram_buffer_size_mb(64.0).unwrap();
    w.enable_explicit_documents().unwrap();
    let id = w
        .register_field(
            FieldInfo::new("_id", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
    w.register_field(FieldInfo::new("v", 0).with_vectors(
        64,
        lucene_codecs::field_infos::VectorEncoding::Float32,
        lucene_codecs::field_infos::VectorSimilarityFunction::DotProduct,
    ))
    .unwrap();
    w.set_knn_vectors_format_for_field(
        "v",
        lucene_codecs::per_field_knn_vectors::KnnVectorsFormat::ivf_quantized(64, 0, 0).unwrap(),
    );
    let w = ConcurrentIndexWriter::new(w, 1).unwrap();
    w.enable_live_search().unwrap();
    (w, id)
}

/// Unit vector `i` of 64 dimensions, near one of 50 cluster means.
fn clustered(i: i64) -> Vec<f32> {
    let mut s = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let c = i.rem_euclid(50) as u64;
    let mut v: Vec<f32> = (0..64u64)
        .map(|d| {
            let mean = ((c * 131 + d * 17) % 23) as f32 / 11.0 - 1.0;
            mean * 2.0 + next()
        })
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn add_clustered_doc(w: &ConcurrentIndexWriter<'_>, id: i32, i: i64) {
    let doc = ExplicitDocument {
        stored: Vec::new(),
        fields: ExplicitFields {
            inverted: vec![InvertedField {
                field_number: id,
                terms: vec![term(&format!("d{i}"), &[])].into(),
                norm: None,
            }],
            ..ExplicitFields::default()
        },
    };
    let vectors = vec![crate::index_writer::DocumentVector::float32(
        "v",
        clustered(i),
    )];
    w.add_explicit_documents_with_vectors_in(None, vec![doc], vec![vectors])
        .unwrap();
}

/// After a flush wrote an IVF field's clusters, the next buffer routes its
/// vectors to them, and a view's search of those lists finds the exact
/// neighbours. The buffer before that flush had none to route to.
#[test]
fn a_view_holds_the_buffer_ivf_lists() {
    let tmp = TempDir::new("live-view-ivf");
    let dir = FsDirectory::open(tmp.path());
    let (w, id) = ivf_writer(&dir);
    for i in 0..3000 {
        add_clustered_doc(&w, id, i);
    }
    let view = w.open_live_view(None).unwrap();
    assert!(view.buffers.iter().all(|b| b.vector_lists.is_empty()));
    w.close_live_view(&view);
    drop(view);
    w.flush().unwrap();
    for i in 3000..11_000 {
        add_clustered_doc(&w, id, i);
    }
    let view = w.open_live_view(None).unwrap();
    let b = view.buffers.iter().find(|b| b.in_slot).unwrap();
    let l = &b.vector_lists[0];
    assert_eq!(l.count, 8000);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while l.lists.assigned() < l.count && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(l.lists.assigned(), l.count, "the router finished");
    let score = |a: &[f32], b: &[f32]| {
        lucene_codecs::field_infos::VectorSimilarityFunction::DotProduct.score(a, b)
    };
    let mut found = 0usize;
    for qi in 0..20i64 {
        let q = clustered(100_000 + qi * 7);
        let mut want: Vec<(i32, f32)> = (0..8000)
            .map(|d| (d, score(&q, &clustered(i64::from(d) + 3000))))
            .collect();
        want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let want: Vec<i32> = want.iter().take(10).map(|w| w.0).collect();
        let hits = l
            .lists
            .search(
                &lucene_codecs::buffer_vectors::BufferQuery::Float32(q),
                10,
                l.count,
                None,
                0,
            )
            .unwrap();
        found += hits.iter().filter(|h| want.contains(&h.0)).count();
    }
    assert!(found >= 180, "recall {found}/200");
    let lists = std::sync::Arc::clone(&l.lists);
    w.close_live_view(&view);
    drop(view);
    w.flush().unwrap();
    // The flushed buffer's lists went with it (a routing task may still be
    // returning).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::sync::Arc::strong_count(&lists) > 1 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(std::sync::Arc::strong_count(&lists), 1);
    assert!(!lists.schedule(), "retired");
    w.commit().unwrap();
}

/// A buffer with vectors stays readable as chunks after it leaves its slot
/// for a flush, and after its publish while a view reading it so is open:
/// the chunks hold its vectors (each fifth document has none).
#[test]
fn a_flushing_buffer_with_vectors_is_written_as_chunks() {
    let tmp = TempDir::new("live-flushing-vectors");
    let dir = FsDirectory::open(tmp.path());
    let (w, id) = vector_writer(&dir, true);
    for i in 0..1000 {
        add_vector_doc(&w, id, i);
    }
    let buffer = w.live_slot_buffers()[0].id;
    let slot_view = w.open_live_view(None).unwrap();
    let out = TempDir::new("live-flushing-vectors-out");
    let out_dir = FsDirectory::open(out.path());
    let vectors_bytes = |name: &str| {
        let s = w
            .live_lean_segment(buffer, 100..600, HashMap::new(), &[], false, &out_dir, name)
            .unwrap();
        assert!(matches!(s, super::LeanSegment::Written(..)));
        std::fs::read_dir(out.path())
            .unwrap()
            .map(|e| e.unwrap())
            .filter(|e| {
                let n = e.file_name().into_string().unwrap();
                n.starts_with(&format!("{name}_")) && n.ends_with(".vec")
            })
            .map(|e| e.metadata().unwrap().len())
            .sum::<u64>()
    };
    let in_slot = vectors_bytes("_s");
    // 400 of documents 100..600 have a 16-dimension vector.
    assert!(in_slot >= 400 * 16 * 4, "{in_slot}");
    let batch = {
        let mut slot = super::write_slot(&w.slots[0]);
        w.begin_flush(0, &mut slot).unwrap()
    };
    let view = w.open_live_view(None).unwrap();
    assert!(view
        .buffers
        .iter()
        .any(|b| b.id == buffer && !b.in_slot && b.chunks));
    assert_eq!(vectors_bytes("_f"), in_slot);
    // With postings too: the flush only reads them.
    let with_postings = |name: &str| {
        w.live_buffer_segment(
            buffer,
            100..600,
            HashMap::from([(id, 500)]),
            &[],
            true,
            &out_dir,
            name,
        )
        .unwrap();
        with_postings_kinds(out.path(), name)
    };
    let slot_files = {
        // Written while in the slot, as reference: the same file kinds.
        let tmp2 = TempDir::new("live-flushing-vectors-ref");
        let d2 = FsDirectory::open(tmp2.path());
        let (w2, id2) = vector_writer(&d2, true);
        for i in 0..1000 {
            add_vector_doc(&w2, id2, i);
        }
        let b2 = w2.live_slot_buffers()[0].id;
        w2.live_buffer_segment(
            b2,
            100..600,
            HashMap::from([(id2, 500)]),
            &[],
            true,
            &out_dir,
            "_r",
        )
        .unwrap();
        with_postings_kinds(out.path(), "_r")
    };
    assert!(
        slot_files.contains("tim") && slot_files.contains("vec"),
        "{slot_files:?}"
    );
    assert_eq!(with_postings("_g"), slot_files);
    w.complete_flush(batch).unwrap();
    assert_eq!(vectors_bytes("_p"), in_slot);
    assert_eq!(with_postings("_q"), slot_files);
    w.close_live_view(&view);
    w.close_live_view(&slot_view);
}

fn with_postings_kinds(dir: &std::path::Path, name: &str) -> std::collections::BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.starts_with(&format!("{name}_")) || n.starts_with(&format!("{name}.")))
        .map(|n| n.rsplit('.').next().unwrap().to_owned())
        .collect()
}
