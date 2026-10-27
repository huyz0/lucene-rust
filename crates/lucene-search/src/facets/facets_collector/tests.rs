//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::test_index::{basic_index, collect, match_all, term};

fn docs_of(it: &mut dyn DocIdSetIterator) -> Vec<i32> {
    let mut out = Vec::new();
    loop {
        let d = it.next_doc().unwrap();
        if d == NO_MORE_DOCS {
            return out;
        }
        out.push(d);
    }
}

#[test]
fn doc_id_sets_iterate_and_advance() {
    let empty = DocIdSet::Empty;
    assert!(empty.is_empty());
    assert_eq!(docs_of(&mut empty.iterator()), Vec::<i32>::new());
    assert_eq!(empty.iterator().cost(), 0);

    let array = DocIdSet::Array(vec![1, 4, 9]);
    assert_eq!(array.len(), 3);
    assert_eq!(docs_of(&mut array.iterator()), vec![1, 4, 9]);
    let mut it = array.iterator();
    assert_eq!(it.doc_id(), -1);
    assert_eq!(it.advance(2).unwrap(), 4);
    assert_eq!(
        it.advance(4).unwrap(),
        9,
        "advance moves past the current doc's target"
    );
    assert_eq!(it.advance(10).unwrap(), NO_MORE_DOCS);
    assert_eq!(
        it.next_doc().unwrap(),
        NO_MORE_DOCS,
        "exhausted stays exhausted"
    );
    assert_eq!(it.cost(), 3);

    let mut bits = FixedBitSet::new(10);
    // FBS: constant indexes below the bitset's length of 10.
    bits.set(0);
    // FBS: as above.
    bits.set(7);
    let set = DocIdSet::Bits { bits, cost: 2 };
    assert_eq!(set.len(), 2);
    let mut owned = set.clone().into_iterator();
    assert_eq!(docs_of(&mut owned), vec![0, 7]);
    let mut it = set.iterator();
    assert_eq!(it.advance(-5).unwrap(), 0);
    assert_eq!(it.advance(8).unwrap(), NO_MORE_DOCS);
    assert_eq!(set.iterator().cost(), 2);
}

#[test]
fn the_builder_switches_to_bits_past_its_threshold() {
    // 256 documents: buffered up to 256 >>> 7 = 2 ids.
    let mut b = DocIdSetBuilder::new(256);
    b.add(5);
    b.add(3);
    b.add(3);
    match DocIdSetBuilder::new(256).build() {
        DocIdSet::Array(d) => assert!(d.is_empty()),
        other => panic!("{other:?}"),
    }
    let mut small = DocIdSetBuilder::new(256);
    small.add(9);
    small.add(2);
    small.add(9);
    // Three ids is past the threshold: bits.
    assert!(matches!(small.build(), DocIdSet::Bits { .. }));
    b.add(300);
    b.add(-1);
    match b.build() {
        DocIdSet::Bits { bits, cost } => {
            assert_eq!(cost, 5, "every add counts, as Java's estimate does");
            assert_eq!(bits.cardinality(), 2);
        }
        other => panic!("{other:?}"),
    }
    let mut big = DocIdSetBuilder::new(100_000);
    big.add(7);
    big.add_all(&mut DocIdSet::Array(vec![1, 7]).into_iterator())
        .unwrap();
    match big.build() {
        DocIdSet::Array(d) => assert_eq!(d, vec![1, 7], "sorted, deduplicated"),
        other => panic!("{other:?}"),
    }
    // A negative maxDoc is an empty builder.
    assert!(DocIdSetBuilder::new(-3).build().is_empty());
}

#[test]
fn java_max_keeps_nan_and_positive_zero() {
    assert!(java_max(f32::NAN, 1.0).is_nan());
    assert!(java_max(1.0, f32::NAN).is_nan());
    assert!(java_max(-0.0, 0.0).is_sign_positive());
    assert!(java_max(0.0, -0.0).is_sign_positive());
    assert_eq!(java_max(2.0, 1.0), 2.0);
    assert_eq!(java_max(1.0, 2.0), 2.0);
}

#[test]
fn collecting_keeps_matching_docs_and_scores_per_segment() {
    let index = basic_index();
    index.with(|searcher, reader, _| {
        let fc = collect(searcher, &match_all(reader), true);
        assert!(fc.keep_scores());
        // Three segments (flushed after the second and fourth documents).
        let docs = fc.matching_docs();
        assert_eq!(docs.len(), 3);
        assert_eq!(
            docs.iter().map(|m| m.total_hits).collect::<Vec<_>>(),
            vec![2, 2, 1]
        );
        assert_eq!(docs[1].scores.as_deref(), Some(&[1.0f32, 1.0][..]));
        assert!(format!("{fc:?}").contains("total_hits: 2"));
        assert!(format!("{:?}", docs[0]).contains("doc_base: 0"));

        let fc = collect(searcher, &term("id", "d3"), false);
        let docs = fc.matching_docs();
        assert_eq!(
            docs.iter().map(|m| m.total_hits).collect::<Vec<_>>(),
            vec![0, 1, 0],
            "every segment is reported, matching or not"
        );
        assert!(docs[1].scores.is_none());
        assert_eq!(docs_of(&mut docs[1].bits.iterator()), vec![1]);
    });
}

#[test]
fn reduce_merges_a_segment_seen_twice() {
    let index = basic_index();
    index.with(|searcher, reader, _| {
        let fcm = FacetsCollectorManager::new(true);
        let a = collect(searcher, &term("id", "d0"), true);
        let b = collect(searcher, &term("id", "d1"), true);
        let c = collect(searcher, &term("id", "d4"), false);
        let merged = fcm.reduce(vec![a, b, c]).unwrap();
        assert!(merged.keep_scores());
        let docs = merged.matching_docs();
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].total_hits, 2);
        assert_eq!(docs_of(&mut docs[0].bits.iterator()), vec![0, 1]);
        let scores = docs[0].scores.clone().unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|s| *s > 0.0));
        // Kept on one side only: that side's scores; on neither: empty.
        assert_eq!(docs[2].scores.as_deref().map(<[f32]>::len), Some(0));
        // None is a fresh collector, one is itself.
        assert!(!fcm.reduce(Vec::new()).unwrap().keep_scores());
        let one = fcm
            .reduce(vec![collect(searcher, &match_all(reader), false)])
            .unwrap();
        assert!(!one.keep_scores());
        let mut by_hand = fcm.new_collector();
        by_hand.push_matching_docs(one.matching_docs()[0].clone());
        assert_eq!(by_hand.matching_docs().len(), 1);
    });
}

#[test]
fn search_returns_top_docs_and_facets_together() {
    let index = basic_index();
    index.with(|searcher, reader, _| {
        let fcm = FacetsCollectorManager::default();
        let q = match_all(reader);
        let r = FacetsCollectorManager::search(searcher, &q, 2, &fcm).unwrap();
        assert_eq!(r.top_docs.total_hits.value, 5);
        assert_eq!(
            r.top_docs
                .score_docs
                .iter()
                .map(|d| d.doc)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(r.facets_collector.matching_docs().len(), 3);
        // `n == 0`: the hit count only.
        let r = FacetsCollectorManager::search(searcher, &q, 0, &fcm).unwrap();
        assert!(r.top_docs.score_docs.is_empty());
        assert_eq!(r.top_docs.total_hits.value, 5);
        // More than maxDoc is maxDoc.
        let r = FacetsCollectorManager::search(searcher, &q, 99, &fcm).unwrap();
        assert_eq!(r.top_docs.score_docs.len(), 5);
        let after = ScoreDoc {
            doc_id: 1,
            score: 1.0,
        };
        let r = FacetsCollectorManager::search_after(searcher, after, &q, 2, &fcm).unwrap();
        assert_eq!(
            r.top_docs
                .score_docs
                .iter()
                .map(|d| d.doc)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        let past = ScoreDoc {
            doc_id: 5,
            score: 1.0,
        };
        match FacetsCollectorManager::search_after(searcher, past, &q, 2, &fcm) {
            Err(Error::IllegalArgument(m)) => assert_eq!(
                m,
                "after.doc exceeds the number of documents in the reader: after.doc=5 limit=5"
            ),
            other => panic!("{other:?}"),
        }
        match FacetsCollectorManager::search(searcher, &q, -1, &fcm) {
            Err(Error::IllegalArgument(m)) => assert!(m.starts_with("numHits must be > 0")),
            other => panic!("{other:?}"),
        }
    });
}

#[test]
fn finish_without_a_segment_is_an_error() {
    let mut fc = FacetsCollector::new(false);
    assert!(matches!(fc.finish(), Err(Error::IllegalState(_))));
    // Collecting before a segment keeps nothing.
    fc.collect(3, 1.0).unwrap();
    assert!(fc.matching_docs().is_empty());
    assert_eq!(fc.score_mode(), ScoreMode::CompleteNoScores);
    assert_eq!(FacetsCollector::new(true).score_mode(), ScoreMode::Complete);
}

/// `add_range` leaves the builder as an `add` per document would: buffered
/// under the threshold, upgraded on the run that crosses it, set in the
/// bits after; an empty, inverted or negative range adds nothing.
#[test]
fn a_range_builds_what_its_documents_one_by_one_build() {
    // 1024 documents: buffered up to 1024 >>> 7 = 8 ids.
    let runs: [&[(i32, i32)]; 4] = [
        &[(1, 4), (6, 9)],
        &[(0, 3), (3, 20), (30, 31)],
        &[(2, 2), (5, 1), (-3, 2), (100, 1000)],
        &[(1000, 1030)],
    ];
    for ranges in runs {
        let (mut by_doc, mut by_range) = (DocIdSetBuilder::new(1024), DocIdSetBuilder::new(1024));
        for &(min, max) in ranges {
            by_range.add_range(min, max);
            for d in min.max(0)..max {
                by_doc.add(d);
            }
        }
        let (want, got) = (by_doc.build(), by_range.build());
        assert_eq!(
            matches!(want, DocIdSet::Bits { .. }),
            matches!(got, DocIdSet::Bits { .. }),
            "{ranges:?}"
        );
        assert_eq!(want.len(), got.len(), "{ranges:?}");
        assert_eq!(docs_of(&mut want.iterator()), docs_of(&mut got.iterator()));
    }
}

/// A [`FacetsCollector`] without its own `collect_range`.
struct ByDoc<'a>(FacetsCollector<'a>);

impl<'a> SegmentCollector<'a> for ByDoc<'a> {
    fn score_mode(&self) -> ScoreMode {
        self.0.score_mode()
    }
    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.0.set_next_reader(ord, leaf)
    }
    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        self.0.collect(doc, score)
    }
    fn finish(&mut self) -> Result<()> {
        self.0.finish()
    }
}

/// A global range handed to the per-segment driver is split at the segment
/// boundaries and collects what a `collect` per document does: hits,
/// counts and scores.
#[test]
fn a_range_collects_like_its_documents_one_by_one() {
    let index = basic_index();
    index.with(|searcher, _, _| {
        for keep in [false, true] {
            // Collecting through `SegmentCollector::collect_range`'s
            // default: a `collect` per document.
            let mut by_doc = ByDoc(FacetsCollector::new(keep));
            let mut by_range = FacetsCollector::new(keep);
            let mut driver = PerSegment::new(searcher.segments(), &mut by_doc);
            driver.collect(0, 1.5);
            driver.collect_range(1, 5, 1.5);
            driver.close().unwrap();
            let by_doc = by_doc.0;
            let mut driver = PerSegment::new(searcher.segments(), &mut by_range);
            driver.collect_range(0, 1, 1.5);
            driver.collect_range(1, 5, 1.5);
            driver.collect_range(5, 5, 1.5);
            driver.close().unwrap();
            let (want, got) = (by_doc.matching_docs(), by_range.matching_docs());
            assert_eq!(want.len(), got.len());
            for (w, g) in want.iter().zip(got) {
                assert_eq!(w.total_hits, g.total_hits);
                assert_eq!(w.scores, g.scores);
                assert_eq!(
                    docs_of(&mut w.bits.iterator()),
                    docs_of(&mut g.bits.iterator())
                );
            }
        }
    });
}

#[test]
fn into_bit_set_fills_up_to_the_bound_and_stops_there() {
    let array = DocIdSet::Array(vec![3, 5, 8, 12]);
    let mut it = array.iterator();
    assert_eq!(it.advance(4).unwrap(), 5);
    let mut bits = FixedBitSet::new(10);
    // Nothing below the current document: no bit, no move.
    it.fill_bit_set(5, &mut bits, 2).unwrap();
    assert_eq!(bits.cardinality(), 0);
    assert_eq!(it.doc_id(), 5);
    it.fill_bit_set(10, &mut bits, 2).unwrap();
    // FBS: 3 and 6 are below the 10 bits `bits` was built with.
    assert!(bits.get(3) && bits.get(6));
    assert_eq!(bits.cardinality(), 2);
    assert_eq!(it.doc_id(), 12);
    // To the end: the iterator is exhausted.
    let mut rest = FixedBitSet::new(1);
    it.fill_bit_set(NO_MORE_DOCS, &mut rest, 12).unwrap();
    // FBS: `rest` holds one bit.
    assert!(rest.get(0));
    assert_eq!(it.doc_id(), NO_MORE_DOCS);
}

#[test]
#[should_panic(expected = "offset=6 docID()=5")]
fn into_bit_set_refuses_an_offset_past_the_current_document() {
    let array = DocIdSet::Array(vec![5]);
    let mut it = array.iterator();
    it.next_doc().unwrap();
    let _ = it.fill_bit_set(10, &mut FixedBitSet::new(10), 6);
}

#[test]
fn ram_bytes_used_counts_the_owned_heap() {
    let base = std::mem::size_of::<DocIdSet>();
    assert_eq!(DocIdSet::Empty.ram_bytes_used(), base);
    let mut docs = Vec::with_capacity(16);
    docs.extend([1, 2, 3]);
    assert_eq!(DocIdSet::Array(docs).ram_bytes_used(), base + 64);
    let bits = FixedBitSet::new(130);
    assert_eq!(DocIdSet::Bits { bits, cost: 0 }.ram_bytes_used(), base + 24);
}
