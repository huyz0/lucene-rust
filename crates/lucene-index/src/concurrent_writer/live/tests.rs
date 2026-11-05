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
        let mut slot = lock(&w.slots[0]);
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
