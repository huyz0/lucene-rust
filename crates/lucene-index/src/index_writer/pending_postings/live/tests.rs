#![cfg(test)]
// Test arithmetic on small constants; see docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::IndexOptions;
use lucene_codecs::postings_writer::TermSource;

use super::super::super::explicit::{InvertedField, InvertedTerm};
use super::super::{Checkpoint, SEALED_DOCS};
use super::*;

const FULL: IndexOptions = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;

fn term(t: &str, positions: &[i32], payload: &[u8]) -> InvertedTerm {
    InvertedTerm {
        term: t.as_bytes().to_vec(),
        freq: i32::try_from(positions.len()).unwrap(),
        positions: positions.to_vec(),
        offsets: positions.iter().map(|&p| (p * 4, p * 4 + 3)).collect(),
        payloads: positions.iter().map(|_| payload.to_vec()).collect(),
    }
}

fn add(p: &mut PendingPostings, doc: i32, terms: Vec<InvertedTerm>) {
    let field = InvertedField {
        field_number: 0,
        terms: terms.into(),
        norm: None,
    };
    p.add_field(doc, &field, FULL).unwrap();
}

/// Documents `0..n`: "a" in each (doc % 3 + 1 times), "b" in the even
/// ones, a payload on "b" in every fourth.
fn buffer(n: i32) -> PendingPostings {
    let mut p = PendingPostings::default();
    p.seal_blocks();
    for doc in 0..n {
        let a: Vec<i32> = (0..=doc % 3).map(|k| k * 2).collect();
        let mut terms = vec![term("a", &a, b"")];
        if doc % 2 == 0 {
            let payload: &[u8] = if doc % 4 == 0 { b"pp" } else { b"" };
            terms.push(term("b", &[7], payload));
        }
        add(&mut p, doc, terms);
    }
    p
}

/// A buffered term reads back as the flush lays it out, with and
/// without its occurrences; a term the buffer lacks is `None`.
#[test]
fn a_buffered_term_reads_as_the_flush_lays_it_out() {
    let p = buffer(300);
    let flushed = p.flushed(None).unwrap();
    let lists = flushed.term_lists();
    let runs = p.live_field(0).unwrap();
    for t in ["a", "b"] {
        let want = lists[0]
            .2
            .iter()
            .find(|v| v.term() == t.as_bytes())
            .unwrap();
        let full = runs.live_term(t.as_bytes(), true, 0).unwrap().unwrap();
        assert_eq!(full.docs, want.docs(), "{t}");
        for (i, positions) in full.positions.iter().enumerate() {
            assert_eq!(positions, want.positions(i), "{t} doc {i}");
            assert_eq!(full.offsets[i], want.offsets(i), "{t} doc {i}");
        }
        assert_eq!(full.payload_lengths, want.payload_lengths(), "{t}");
        assert_eq!(full.payload_bytes, want.payload_bytes(), "{t}");
        let docs = runs.live_term(t.as_bytes(), false, 0).unwrap().unwrap();
        assert_eq!(docs.docs, want.docs(), "{t}");
        assert!(docs.positions.is_empty() && docs.payload_bytes.is_empty());
    }
    assert!(runs.live_term(b"c", true, 0).unwrap().is_none());
    assert!(runs.has_payloads());
}

/// The statistics count every buffered document, and a rollback takes
/// back what the undone documents added.
#[test]
fn statistics_follow_adds_and_rollbacks() {
    let mut p = buffer(300);
    let want = LiveFieldStats {
        num_terms: 2,
        // "a" in all 300, "b" in 150.
        sum_doc_freq: 450,
        // "a" 1+2+3 per three documents, "b" once each.
        sum_total_term_freq: 600 + 150,
        doc_count: 300,
    };
    assert_eq!(p.live_field(0).unwrap().live_stats(), want);
    let cp = p.checkpoint(300);
    add(
        &mut p,
        300,
        vec![term("a", &[0], b""), term("c", &[1, 2], b"")],
    );
    let grown = p.live_field(0).unwrap().live_stats();
    assert_eq!(
        (
            grown.num_terms,
            grown.sum_doc_freq,
            grown.sum_total_term_freq,
            grown.doc_count
        ),
        (3, 452, 753, 301)
    );
    p.rollback(cp);
    assert_eq!(p.live_field(0).unwrap().live_stats(), want);
}

/// A term with sealed blocks reads the same documents from any
/// document on, and after a rollback cut its last block.
#[test]
fn sealed_blocks_read_as_the_streams() {
    let n = 3 * SEALED_DOCS as i32 + 17;
    let mut p = buffer(n);
    let flushed = p.flushed(None).unwrap();
    let lists = flushed.term_lists();
    let want = |t: &str| {
        lists[0]
            .2
            .iter()
            .find(|v| v.term() == t.as_bytes())
            .unwrap()
            .docs()
            .to_vec()
    };
    let (a, b) = (want("a"), want("b"));
    drop(flushed);
    let runs = p.live_field(0).unwrap();
    let id = runs.terms.find(b"a").unwrap();
    assert_eq!(runs.sealed_blocks(id).len(), 3);
    for from in [0, 1, 255, 256, 257, 600, n - 1, n] {
        let mut got = Vec::new();
        runs.live_docs_of(b"a", from, &mut got);
        let expect: Vec<_> = a.iter().copied().filter(|d| d.0 >= from).collect();
        assert_eq!(got, expect, "a from {from}");
        got.clear();
        runs.live_docs_of(b"b", from, &mut got);
        let expect: Vec<_> = b.iter().copied().filter(|d| d.0 >= from).collect();
        assert_eq!(got, expect, "b from {from}");
    }
    // Cut back into the third block: it goes, the first two stay.
    let cut = 2 * SEALED_DOCS as i32 + 5;
    let cp = p.checkpoint(n);
    p.rollback(Checkpoint { doc: cut, ..cp });
    let runs = p.live_field(0).unwrap();
    assert_eq!(runs.sealed_blocks(id).len(), 2);
    let mut got = Vec::new();
    runs.live_docs_of(b"a", 0, &mut got);
    let expect: Vec<_> = a.iter().copied().filter(|d| d.0 < cut).collect();
    assert_eq!(got, expect);
}
