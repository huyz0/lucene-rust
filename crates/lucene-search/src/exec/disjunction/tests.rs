#![cfg(test)]
use super::*;
use crate::exec::leaf::DocList;

/// `topList()` from the heap (more members than a linear queue keeps)
/// names exactly the members on the top's document -- the ones a scan
/// finds -- as the top moves through the members' documents.
#[test]
fn a_heaped_queue_lists_every_member_on_the_top_document() {
    // Twelve members, document lists overlapping in places.
    let lists: Vec<Vec<i32>> = (0..12)
        .map(|m| (0..40).filter(|d| (d + m) % (2 + m % 5) == 0).collect())
        .collect();
    let disis = || -> Vec<Disi<'static, DocList>> {
        lists
            .iter()
            .map(|l| Disi::new(Box::new(DocList::new(l.clone(), Vec::new()))))
            .collect()
    };
    let (mut heaped, mut scanned) = (disis(), disis());
    let mut heap = DisiQueue::for_disjunction(lists.len());
    assert!(!heap.linear);
    for i in 0..heaped.len() {
        heaped[i].doc = heaped[i].scorer.next_doc().unwrap();
        scanned[i].doc = heaped[i].doc;
        heap.push(&heaped, i);
    }
    let (mut got, mut steps) = (Vec::new(), 0);
    while let Some(top) = heap.top_of(&heaped) {
        let doc = heaped[top].doc;
        if doc == NO_MORE_DOCS {
            break;
        }
        heap.top_list(&heaped, &mut got);
        got.sort_unstable();
        let want: Vec<usize> = (0..scanned.len())
            .filter(|&i| scanned[i].doc == doc)
            .collect();
        assert_eq!(got, want, "doc {doc}");
        // Move every member on `doc` on, the top at a time, as
        // `nextDoc` does.
        let mut top = top;
        while heaped[top].doc == doc {
            heaped[top].doc = heaped[top].scorer.next_doc().unwrap();
            scanned[top].doc = heaped[top].doc;
            top = heap.update_top(&heaped);
        }
        steps += 1;
    }
    assert!(steps > 20, "{steps}");
    // An empty queue lists nothing.
    let empty = DisiQueue::for_disjunction(12);
    empty.top_list(&heaped, &mut got);
    assert!(got.is_empty());
}

fn lists() -> Vec<Vec<i32>> {
    (0..3)
        .map(|m: i32| {
            (0..20_000)
                .filter(|d| (d * 7 + m * 3) % (3 + m * 5) == 0 || d % 997 == m)
                .collect()
        })
        .collect()
}

fn disjunction(lists: &[Vec<i32>]) -> DisjunctionScorer<'static, DocList> {
    let subs = lists
        .iter()
        .map(|l| Box::new(DocList::new(l.clone(), Vec::new())))
        .collect();
    let mut d = DisjunctionScorer::new(subs, Combine::Sum, false);
    d.next_doc().unwrap();
    d
}

/// The window read ahead by `next_docs_unscored` lists exactly the documents a
/// document-at-a-time walk finds, whatever mixes of batches, `next_doc`, `advance` and
/// deleted documents come between.
#[test]
fn a_window_lists_the_documents_one_at_a_time_walk_finds() {
    use lucene_util::fixed_bit_set::FixedBitSet;
    let lists = lists();
    let mut want: Vec<i32> = lists.iter().flatten().copied().collect();
    want.sort_unstable();
    want.dedup();
    // Batches to the end.
    let mut d = disjunction(&lists);
    let mut got = Vec::new();
    let mut out = crate::bulk_scorer::DocScores::default();
    loop {
        d.next_docs_unscored(i32::MAX, None, &mut out).unwrap();
        if out.docs.is_empty() {
            break;
        }
        assert!(out.docs.len() <= super::super::NEXT_DOCS_BATCH);
        got.extend_from_slice(&out.docs);
    }
    assert_eq!(got, want);
    assert_eq!(d.doc_id(), NO_MORE_DOCS);
    // A bound, a step, a jump, then the rest; deleted documents skipped.
    let mut live = FixedBitSet::new(20_001);
    for i in 0..20_001 {
        if i % 5 != 0 {
            live.set(i);
        }
    }
    let mut d = disjunction(&lists);
    let mut got = Vec::new();
    d.next_docs_unscored(9000, Some(&live), &mut out).unwrap();
    got.extend_from_slice(&out.docs);
    let at = d.doc_id();
    assert_eq!(Some(&at), want.iter().find(|&&x| x > *got.last().unwrap()));
    let next = d.next_doc().unwrap();
    let jump = d.advance(next + 3000).unwrap();
    assert_eq!(Some(&jump), want.iter().find(|&&x| x >= next + 3000));
    d.next_docs_unscored(i32::MAX, Some(&live), &mut out)
        .unwrap();
    assert!(out.docs.iter().all(|&x| x % 5 != 0 && want.contains(&x)));
    let first = d.doc_id();
    let mut rest = vec![first];
    loop {
        let n = d.next_doc().unwrap();
        if n == NO_MORE_DOCS {
            break;
        }
        rest.push(n);
    }
    let last = *out.docs.last().unwrap();
    assert_eq!(
        rest,
        want.iter()
            .copied()
            .filter(|&x| x > last)
            .collect::<Vec<_>>()
    );
}

/// A scored window's documents and scores are those a one-document-at-a-time walk finds.
#[test]
fn a_scored_window_scores_as_the_walk_does() {
    let lists = lists();
    let scored = |on: bool| {
        let subs = lists
            .iter()
            .enumerate()
            .map(|(m, l)| {
                let scores = l
                    .iter()
                    .map(|d| 0.25 + ((d * (m as i32 + 3)) % 17) as f32 * 0.37)
                    .collect();
                Box::new(DocList::new(l.clone(), scores))
            })
            .collect();
        let mut d = DisjunctionScorer::new(subs, Combine::Sum, on);
        d.next_doc().unwrap();
        d
    };
    let mut walk = scored(true);
    let mut want = Vec::new();
    while walk.doc_id() != NO_MORE_DOCS {
        want.push((walk.doc_id(), walk.score().unwrap()));
        walk.next_doc().unwrap();
    }
    let mut batched = scored(true);
    let mut got = Vec::new();
    let mut out = crate::bulk_scorer::DocScores::default();
    loop {
        batched
            .next_docs_and_scores(i32::MAX, None, &mut out)
            .unwrap();
        if out.docs.is_empty() {
            break;
        }
        got.extend(out.docs.iter().copied().zip(out.scores.iter().copied()));
    }
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g.0, w.0);
        assert_eq!(g.1.to_bits(), w.1.to_bits(), "doc {}", g.0);
    }
    // Stepping with `next_doc` and `score` in the middle of a window reads the same scores.
    let mut mixed = scored(true);
    mixed
        .next_docs_and_scores(i32::MAX, None, &mut out)
        .unwrap();
    let at = mixed.doc_id();
    let k = want.iter().position(|w| w.0 == at).unwrap();
    assert_eq!(mixed.score().unwrap().to_bits(), want[k].1.to_bits());
    assert_eq!(mixed.next_doc().unwrap(), want[k + 1].0);
    assert_eq!(mixed.score().unwrap().to_bits(), want[k + 1].1.to_bits());
    let target = want[k + 900].0;
    assert_eq!(mixed.advance(target).unwrap(), target);
    assert_eq!(mixed.score().unwrap().to_bits(), want[k + 900].1.to_bits());
}
