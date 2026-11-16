#![cfg(test)]
use super::*;
use lucene_codecs::completion::{CompletionFieldWriter, TermDoc};
use lucene_store::DataOutput;
use lucene_util::automaton::automata::make_binary;

fn doc(id: i32, surface: &str, weight: i32) -> TermDoc {
    let mut p = Vec::new();
    p.write_vint(surface.len() as i32);
    p.extend_from_slice(surface.as_bytes());
    p.write_vint(weight + 1);
    p.push(0);
    TermDoc {
        doc_id: id,
        payloads: vec![p],
    }
}

fn suggester(rows: &[(&str, i32, i32)]) -> NrtSuggester {
    let mut w = CompletionFieldWriter::new();
    let mut by_term: Vec<(&str, Vec<TermDoc>)> = Vec::new();
    for &(t, id, weight) in rows {
        match by_term.iter_mut().find(|(k, _)| *k == t) {
            Some((_, v)) => v.push(doc(id, t, weight)),
            None => by_term.push((t, vec![doc(id, t, weight)])),
        }
    }
    by_term.sort_by(|a, b| a.0.cmp(b.0));
    for (t, docs) in by_term {
        w.write_term(t.as_bytes(), &docs).unwrap();
    }
    let mut out = Vec::new();
    assert!(w.finish(&mut out).unwrap());
    NrtSuggester::load(&out).unwrap()
}

fn run(
    s: &NrtSuggester,
    prefix: &str,
    n: usize,
    dedup: bool,
    view: &SegmentView<'_>,
) -> Vec<(String, i32, i64)> {
    let mut c = TopSuggestDocsCollector::new(n, dedup).unwrap();
    lookup(s, &make_binary(prefix.as_bytes()), view, &mut c).unwrap();
    c.finish();
    c.get()
        .into_iter()
        .map(|h| (h.key, h.doc, h.score as i64))
        .collect()
}

fn view(num_docs: i32, max_doc: i32, accept: &dyn Fn(i32) -> bool) -> SegmentView<'_> {
    SegmentView {
        num_docs,
        max_doc,
        filtered: false,
        accept,
    }
}

#[test]
fn hits_come_best_first_and_stop_at_num() {
    let s = suggester(&[("ab", 0, 5), ("ac", 1, 9), ("ad", 2, 7)]);
    let accept = |_: i32| true;
    let v = view(3, 3, &accept);
    assert_eq!(
        run(&s, "a", 2, false, &v),
        vec![("ac".into(), 1, 9), ("ad".into(), 2, 7)]
    );
    assert!(run(&s, "zz", 2, false, &v).is_empty());
}

#[test]
fn a_rejected_document_is_skipped_and_a_filter_widens_the_queue() {
    let s = suggester(&[("ab", 0, 5), ("ac", 1, 9)]);
    let accept = |d: i32| d != 1;
    let mut v = view(2, 4, &accept);
    v.filtered = true;
    assert_eq!(run(&s, "a", 5, false, &v), vec![("ab".into(), 0, 5)]);
    let none = view(0, 4, &accept);
    assert!(run(&s, "a", 5, false, &none).is_empty());
}

#[test]
fn duplicates_collapse_to_the_best_document() {
    let s = suggester(&[("ab", 0, 5), ("ab", 1, 8), ("ab", 2, 6), ("ac", 3, 1)]);
    let accept = |_: i32| true;
    let v = view(4, 4, &accept);
    assert_eq!(
        run(&s, "a", 3, true, &v),
        vec![("ab".into(), 1, 8), ("ac".into(), 3, 1)]
    );
    assert_eq!(run(&s, "a", 3, false, &v).len(), 3);
}

#[test]
fn a_full_collector_ends_the_segment() {
    let s = suggester(&[("a1", 0, 9), ("a2", 1, 8), ("a3", 2, 7), ("a4", 3, 6)]);
    let accept = |_: i32| true;
    let v = view(4, 4, &accept);
    assert_eq!(run(&s, "a", 1, false, &v), vec![("a1".into(), 0, 9)]);
}

/// `PriorityQueue.insertWithOverflow` replaces the top only with a
/// strictly greater element: a hit equal to the last one that fit (the
/// same document and key under another context) overflows and ends the
/// segment, so the first of equals stays.
#[test]
fn an_equal_hit_does_not_replace_the_last_that_fit() {
    let mut c = TopSuggestDocsCollector::new(1, false).unwrap();
    c.set_next_reader(0);
    c.collect(7, "x", Some("first"), 2.0);
    assert!(!c.is_terminated());
    c.collect(7, "x", Some("second"), 2.0);
    assert!(c.is_terminated());
    assert_eq!(c.get()[0].context.as_deref(), Some("first"));
}

#[test]
fn collector_validates_and_merges_segments() {
    assert!(TopSuggestDocsCollector::new(0, false).is_err());
    let mut c = TopSuggestDocsCollector::new(2, true).unwrap();
    assert!(c.do_skip_duplicates());
    c.set_next_reader(0);
    c.collect(1, "x", None, 3.0);
    c.collect(2, "y", Some("ctx"), 1.0);
    c.finish();
    c.set_next_reader(100);
    assert!(!c.is_terminated());
    c.collect(1, "x", None, 3.0);
    c.finish();
    let hits = c.get();
    assert_eq!(hits.iter().map(|h| h.doc).collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(hits[1].context.as_deref(), Some("ctx"));
    let mut plain = TopSuggestDocsCollector::new(1, false).unwrap();
    plain.collect(0, "a", None, 1.0);
    plain.collect(1, "b", None, 2.0);
    plain.collect(2, "c", None, 0.5);
    assert!(plain.is_terminated());
    plain.set_next_reader(0);
    assert!(!plain.is_terminated());
    assert_eq!(plain.get()[0].key, "b");
}

#[test]
fn scoring_follows_the_scorer() {
    assert_eq!(score(5.0, 0.0), 5.0);
    assert_eq!(score(0.0, 3.0), 3.0);
    assert_eq!(score(2.0, 3.0), 6.0);
    assert_eq!(max_top_n_searcher_queue_size(10, 100, 0.5, true, 3), 110);
    assert_eq!(
        max_top_n_searcher_queue_size(10_000, 100, 1.0, false, 3),
        5000
    );
}

#[test]
fn ties_leave_the_queue_by_key_then_document() {
    let s = suggester(&[("a1", 5, 4), ("a2", 3, 4), ("a1", 2, 4)]);
    let accept = |_: i32| true;
    let v = view(3, 3, &accept);
    let got = run(&s, "a", 3, false, &v);
    assert_eq!(got.len(), 3);
    assert!(got[0].0 <= got[1].0 && got[1].0 <= got[2].0);
}
