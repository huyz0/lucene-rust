#![cfg(test)]
use super::super::explicit::InvertedTerm;
use super::*;

const FULL: IndexOptions = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;

/// A term at `positions`, each at offsets `(2p, 2p + 1)`.
fn term(t: &str, positions: &[i32], payloads: &[&[u8]]) -> InvertedTerm {
    InvertedTerm {
        term: t.as_bytes().to_vec(),
        freq: positions.len().max(1) as i32,
        positions: positions.to_vec(),
        offsets: positions
            .iter()
            .map(|&p| (p.saturating_mul(2), p.saturating_mul(2).saturating_add(1)))
            .collect(),
        payloads: payloads.iter().map(|p| p.to_vec()).collect(),
    }
}

fn add(
    p: &mut PendingPostings,
    field: i32,
    doc: i32,
    opts: IndexOptions,
    terms: Vec<InvertedTerm>,
) {
    let field = InvertedField {
        field_number: field,
        terms: terms.into(),
        norm: None,
    };
    p.add_field(doc, &field, opts).unwrap();
}

fn get(p: &PendingPostings, field: i32, doc: i32, t: &str) -> Option<InvertedTerm> {
    p.doc_terms(field, doc)
        .into_iter()
        .find(|x| x.term == t.as_bytes())
}

fn view<'a>(lists: &[(i32, i32, Vec<TermView<'a>>)], field: i32, t: &str) -> TermView<'a> {
    let (_, _, terms) = lists.iter().find(|(n, _, _)| *n == field).unwrap();
    *terms.iter().find(|v| v.term() == t.as_bytes()).unwrap()
}

/// Streams interleave in one pool across many slice levels and blocks,
/// read back in order, and a cut at any length resumes appending from
/// there.
/// Every posting of `f`: field, term, document (plus `base`), then its
/// positions, offsets and payloads.
#[allow(clippy::type_complexity)]
fn flat(
    f: &FlushedPostings,
    base: i32,
) -> Vec<(i32, Vec<u8>, i32, Vec<i32>, Vec<(i32, i32)>, Vec<Vec<u8>>)> {
    let mut out = Vec::new();
    for (field, _, terms) in f.term_lists() {
        for t in terms {
            let mut at = 0usize;
            for (i, &(doc, freq)) in t.docs.iter().enumerate() {
                let span = t.span(i);
                let mut payloads = Vec::new();
                for j in span.clone() {
                    let len = t.payload_lengths.get(j).copied().unwrap_or(0) as usize;
                    payloads.push(t.payload_bytes[at..at.saturating_add(len)].to_vec());
                    at = at.saturating_add(len);
                }
                assert_eq!(
                    span.len(),
                    if t.ends.is_empty() { 0 } else { freq as usize }
                );
                out.push((
                    field,
                    t.term.to_vec(),
                    doc.saturating_add(base),
                    t.positions[span.clone()].to_vec(),
                    t.offsets.get(span).map(<[_]>::to_vec).unwrap_or_default(),
                    payloads,
                ));
            }
        }
    }
    out.sort();
    out
}

/// Ranges laid out one after the other, each from where the last
/// stopped (and some again from 0), hold what the whole buffer's
/// layout holds of their documents, while documents keep arriving.
#[test]
fn ranges_lay_out_their_documents_alone() {
    let mut p = PendingPostings::default();
    let words = ["a", "b", "c", "d"];
    let mut x = 11u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let (mut doc, mut from) = (0i32, 0i32);
    for round in 0..40 {
        for _ in 0..1 + next() % 40 {
            let mut terms = Vec::new();
            for w in words {
                if next() % 3 != 0 {
                    let n = 1 + (next() % 3) as i32;
                    let positions: Vec<i32> = (0..n).map(|k| k * 3).collect();
                    let payloads: Vec<Vec<u8>> = (0..n)
                        .map(|k| {
                            if (doc + k) % 5 == 0 {
                                vec![k as u8; 2]
                            } else {
                                Vec::new()
                            }
                        })
                        .collect();
                    let refs: Vec<&[u8]> = payloads.iter().map(Vec::as_slice).collect();
                    terms.push(term(w, &positions, &refs));
                }
            }
            add(&mut p, 0, doc, FULL, terms.clone());
            // a field without positions too
            let docs_only: Vec<InvertedTerm> = terms
                .iter()
                .map(|t| InvertedTerm {
                    freq: 1,
                    positions: Vec::new(),
                    offsets: Vec::new(),
                    payloads: Vec::new(),
                    ..t.clone()
                })
                .collect();
            add(&mut p, 1, doc, IndexOptions::DocsAndFreqs, docs_only);
            doc += 1;
        }
        let start = if round % 7 == 6 { 0 } else { from };
        let whole = flat(&p.flushed(None).unwrap(), 0);
        let want: Vec<_> = whole
            .into_iter()
            .filter(|e| (start..doc).contains(&e.2))
            .collect();
        let got = flat(&p.flushed_below(None, Some((start, doc))).unwrap(), start);
        assert_eq!(got, want, "round {round}: {start}..{doc}");
        let mut streamed = Vec::new();
        for field in [0, 1] {
            p.stream_field_below(field, None, Some((start, doc)), &mut |terms| {
                for t in terms {
                    for &(d, _) in t.docs {
                        streamed.push((field, t.term.to_vec(), d + start));
                    }
                }
                Ok(())
            })
            .unwrap();
        }
        streamed.sort();
        let want: Vec<_> = want.iter().map(|e| (e.0, e.1.clone(), e.2)).collect();
        assert_eq!(streamed, want, "round {round}: streamed");
        from = doc;
    }
}

#[test]
fn pool_streams_chain_slices_and_cut_anywhere() {
    let mut pool = BytePool::default();
    let (sa, sb) = (pool.new_slice(), pool.new_slice());
    let (mut a, mut b) = (sa, sb);
    let value = |i: u64| i.wrapping_mul(0x9e37_79b9) % (1 << (i % 40));
    for i in 0..20_000u64 {
        pool.write_vlong(&mut a, i);
        pool.write_vlong(&mut b, value(i));
    }
    assert!(pool.blocks.len() > 2);
    let mut r = pool.reader(sa, a);
    assert!((0..20_000u64).all(|i| r.read_vlong() == i));
    assert!(r.eof());
    let mut r = pool.reader(sb, b);
    assert!((0..20_000u64).all(|i| r.read_vlong() == value(i)));
    assert!(r.eof());
    // longest first: a cut overwrites what followed it
    for keep in [20_000u64, 7_777, 300, 129, 128, 127, 5, 4, 3, 1, 0] {
        let mut r = pool.reader(sa, a);
        for _ in 0..keep {
            r.read_vlong();
        }
        let pos = r.pos();
        let mut c = pool.cut(pos);
        for v in [9_999u64, 1, 300] {
            pool.write_vlong(&mut c, v);
        }
        let mut r = pool.reader(sa, c);
        assert!((0..keep).all(|i| r.read_vlong() == i), "{keep}");
        assert_eq!(
            [r.read_vlong(), r.read_vlong(), r.read_vlong()],
            [9_999, 1, 300]
        );
        assert!(r.eof(), "{keep}");
        a = c;
    }
    let empty = pool.new_slice();
    assert!(pool.reader(empty, empty).eof());
    assert!(pool.reader(NONE, NONE).eof());
}

/// Every address maps into the block that holds it, small blocks first.
#[test]
fn addresses_locate_their_blocks() {
    let mut at = 0usize;
    for b in 0..12 {
        assert_eq!(block_start(b), at);
        assert_eq!(locate(at as u32), (b, 0));
        let end = at + block_size(b) - 1;
        assert_eq!(locate(end as u32), (b, block_size(b) - 1));
        at += block_size(b);
    }
    let mut p = PendingPostings::default();
    add(&mut p, 0, 0, IndexOptions::Docs, vec![term("a", &[], &[])]);
    assert!(
        p.bytes() < 16 * 1024,
        "a one-term field holds {}",
        p.bytes()
    );
}

/// A term's documents and occurrences cost bytes, not structs: a frequent
/// term's document is a byte or two, its positions a byte each.
#[test]
fn buffered_postings_cost_bytes_per_occurrence() {
    let mut p = PendingPostings::default();
    let opts = IndexOptions::DocsAndFreqsAndPositions;
    for doc in 0..200_000 {
        add(
            &mut p,
            0,
            doc,
            opts,
            vec![term("a", &[0, 2, 3], &[]), term("b", &[1], &[])],
        );
    }
    let per_doc = p.bytes() / 200_000;
    assert!(per_doc <= 8, "{per_doc} bytes per document");
    let flushed = p.flushed(None).unwrap();
    let lists = flushed.term_lists();
    let a = view(&lists, 0, "a");
    assert_eq!(a.docs().len(), 200_000);
    assert_eq!(a.docs()[199_999], (199_999, 3));
    assert_eq!(a.positions(123_456), [0, 2, 3]);
    assert_eq!(view(&lists, 0, "b").positions(7), [1]);
}

/// A rollback across slice boundaries leaves every stream ready to
/// take the next document.
#[test]
fn rollback_past_many_documents_keeps_appending() {
    let mut p = PendingPostings::default();
    for doc in 0..50 {
        add(&mut p, 0, doc, FULL, vec![term("a", &[doc, doc + 1], &[])]);
    }
    let cp = p.checkpoint(50);
    for doc in 50..80 {
        add(&mut p, 0, doc, FULL, vec![term("a", &[1, 2, 3], &[b"x"])]);
    }
    p.rollback(cp);
    add(&mut p, 0, 50, FULL, vec![term("a", &[7], &[])]);
    let flushed = p.flushed(None).unwrap();
    let lists = flushed.term_lists();
    let a = view(&lists, 0, "a");
    assert_eq!(a.docs().len(), 51);
    assert_eq!(a.docs()[49], (49, 2));
    assert_eq!(a.docs()[50], (50, 1));
    assert_eq!(a.positions(49), [49, 50]);
    assert_eq!(a.positions(50), [7]);
    assert_eq!(a.offsets(50), [(14, 15)]);
    assert!(a.payload_lengths().is_empty());
}

/// A rollback takes back a document's new terms, its occurrences of old
/// ones, and the payloads it was the first to bring -- the zero lengths
/// they backfilled into earlier documents included.
#[test]
fn rollback_undoes_a_document_down_to_its_payloads() {
    let mut p = PendingPostings::default();
    add(
        &mut p,
        0,
        0,
        FULL,
        vec![term("a", &[0, 1], &[]), term("b", &[2], &[])],
    );
    let (cp, bytes) = (p.checkpoint(1), p.bytes());
    add(
        &mut p,
        0,
        1,
        FULL,
        vec![term("a", &[0], &[b"xy"]), term("c", &[1], &[])],
    );
    // the first payload backfills a zero length for every occurrence
    assert_eq!(get(&p, 0, 0, "a").unwrap().payloads, [b"", b""]);
    assert_eq!(get(&p, 0, 1, "a").unwrap().payloads, [b"xy"]);
    assert_eq!(p.payload_fields().collect::<Vec<_>>(), [0]);
    p.rollback(cp);
    assert!(p.doc_terms(0, 1).is_empty());
    let a = get(&p, 0, 0, "a").unwrap();
    assert_eq!((a.positions, a.payloads), (vec![0, 1], vec![]));
    assert_eq!(p.payload_fields().count(), 0);
    // the pool keeps what the undone document took
    assert!(p.bytes() >= bytes);
    let flushed = p.flushed(None).unwrap();
    let lists = flushed.term_lists();
    assert_eq!(lists.len(), 1);
    assert_eq!(lists[0].1, 1, "docCount");
    let terms: Vec<&[u8]> = lists[0].2.iter().map(TermSource::term).collect();
    assert_eq!(terms, [b"a", b"b"]);
    assert!(view(&lists, 0, "a").payload_bytes().is_empty());
}

/// Payloads the checkpoint already had survive, cut to its documents.
#[test]
fn rollback_keeps_the_payloads_before_the_checkpoint() {
    let mut p = PendingPostings::default();
    add(&mut p, 0, 0, FULL, vec![term("a", &[0], &[b"p"])]);
    let cp = p.checkpoint(1);
    add(&mut p, 0, 1, FULL, vec![term("a", &[0, 1], &[b"qq", b""])]);
    p.rollback(cp);
    let flushed = p.flushed(None).unwrap();
    let lists = flushed.term_lists();
    let a = view(&lists, 0, "a");
    assert_eq!(
        (a.payload_bytes(), a.payload_lengths()),
        (&b"p"[..], &[1u32][..])
    );
    assert_eq!(a.docs(), [(0, 1)]);
    // a field the checkpoint never saw goes altogether
    let cp = p.checkpoint(1);
    add(&mut p, 5, 1, IndexOptions::Docs, vec![term("k", &[], &[])]);
    p.rollback(cp);
    assert!(p.doc_terms(5, 1).is_empty());
    assert_eq!(p.flushed(None).unwrap().term_lists().len(), 1);
}

/// An index sort's remap moves each document's positions, offsets and
/// payloads with it.
#[test]
fn remap_carries_each_documents_occurrences_into_sort_order() {
    let mut p = PendingPostings::default();
    add(&mut p, 0, 0, FULL, vec![term("a", &[0], &[b"a0"])]);
    add(&mut p, 1, 0, IndexOptions::Docs, vec![term("k", &[], &[])]);
    add(&mut p, 0, 1, FULL, vec![term("a", &[0, 3], &[b"", b"b"])]);
    add(&mut p, 0, 2, FULL, vec![term("a", &[5], &[b"cc"])]);
    add(&mut p, 1, 2, IndexOptions::Docs, vec![term("k", &[], &[])]);
    let remapped = p.flushed(Some(&[2, 0, 1])).unwrap();
    let lists = remapped.term_lists();
    let a = view(&lists, 0, "a");
    assert_eq!(a.docs(), [(0, 2), (1, 1), (2, 1)]);
    assert_eq!(a.position_lists(), 3);
    assert_eq!(a.positions(0), [0, 3]);
    assert_eq!(a.positions(1), [5]);
    assert_eq!(a.positions(2), [0]);
    assert_eq!(a.offset_lists(), 3);
    assert_eq!(a.offsets(0), [(0, 1), (6, 7)]);
    assert_eq!(a.offsets(2), [(0, 1)]);
    assert_eq!(a.payload_lengths(), [0, 1, 2, 2]);
    assert_eq!(a.payload_bytes(), b"bcca0");
    // a docs-only field has no occurrences to move
    let k = view(&lists, 1, "k");
    assert_eq!(k.docs(), [(1, 1), (2, 1)]);
    assert_eq!((k.position_lists(), k.offset_lists()), (0, 0));
    assert!(k.positions(0).is_empty() && k.offsets(0).is_empty());
    assert!(k.payload_bytes().is_empty() && k.payload_lengths().is_empty());
    // the buffered order is untouched
    let flushed = p.flushed(None).unwrap();
    assert_eq!(view(&flushed.term_lists(), 0, "a").docs()[0], (0, 1));
    let e = p.flushed(Some(&[0])).unwrap_err().to_string();
    assert!(e.contains("outside the sort map"), "{e}");
}

/// The document API's side of a run: frequencies counted per document,
/// occurrences kept only where the field indexes them.
#[test]
fn occurrences_are_kept_as_the_field_indexes_them() {
    let mut p = PendingPostings::default();
    add(&mut p, 0, 0, FULL, Vec::new());
    assert!(
        p.flushed(None).unwrap().term_lists().is_empty(),
        "an empty field adds nothing"
    );
    let f = p.field(2, IndexOptions::DocsAndFreqsAndPositions);
    let slot = f.occurrence(0, b"x");
    assert!(slot.new_to_segment && slot.freq.is_none());
    f.first_in_doc(slot, 0, 1);
    f.prox(slot, 4, (9, 9), b"");
    let slot = f.occurrence(0, b"x");
    assert_eq!((slot.new_to_segment, slot.freq), (false, Some(1)));
    f.set_freq(slot, 2);
    f.prox(slot, 6, (9, 9), b"");
    let docs_only = p.field(3, IndexOptions::Docs);
    let slot = docs_only.occurrence(0, b"y");
    docs_only.first_in_doc(slot, 0, 1);
    docs_only.prox(slot, 1, (0, 1), b"ignored");
    let flushed = p.flushed(None).unwrap();
    let lists = flushed.term_lists();
    let x = view(&lists, 2, "x");
    assert_eq!((x.docs(), x.positions(0)), (&[(0, 2)][..], &[4, 6][..]));
    assert_eq!(x.offset_lists(), 0, "no offsets indexed");
    assert!(x.offsets(0).is_empty());
    assert!(get(&p, 3, 0, "y").unwrap().positions.is_empty());
    assert_eq!(
        p.payload_fields().count(),
        0,
        "a docs-only field stores no payloads"
    );
    assert!(p.bytes() > 0);
    p.clear();
    assert_eq!(p.bytes(), 0);
}
